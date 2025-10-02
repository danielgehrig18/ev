import torch
from mingpt.model import GPT
from torch import nn

def stack_features(samples, use_frame=False):
    f = samples["f"]
    features = [f[...,:3,0], f[...,:3,1], f[...,:3,3]]

    if use_frame:
        dfr = samples["rotation_frames_orth"]
        features.extend([dfr[...,:3,0], dfr[...,:3,1]])
        dft = samples["rotation_frames_orth"]
        features.extend([dft[...,:3,0], dft[...,:3,1]])

    return torch.cat(features, dim=-1)


def orthogonalize(x, eps=1e-18, out=None):
    r1 = x[...,0:3]
    r2 = x[...,3:6]
    t = x[...,6:9]

    norm_1 = r1.pow(2).sum(-1, keepdim=True)
    r1 = r1 / torch.sqrt(eps + norm_1)
    r2 = r2 - (r1 * r2).sum(-1, keepdim=True) * r1

    norm_2 = r2.pow(2).sum(-1, keepdim=True)
    r2 = r2 / torch.sqrt(eps + norm_2)
    r3 = torch.linalg.cross(r1, r2)

    R = torch.stack([r1, r2, r3], dim=-1)

    if out is None:
        out = torch.zeros(size=x.shape[:-1] + (4, 4), device=x.device)

    out[...,:3,:3] = R
    out[...,:3, 3] = t
    out[...,-1,-1] = 1

    return out


class ModelSE3GPT(GPT):
    def __init__(self, *args, **kwargs):
        use_frame = kwargs.pop('use_frame')
        GPT.__init__(self, *args, **kwargs)
        #self.use_frame = use_frame
        self.pos_emb = None
        self.input_projection = nn.Linear(in_features=9 + (12 if use_frame else 0), out_features=768, bias=True)
        self.out = None

    def forward(self, samples):
        # samples: B x T x C
        #samples = stack_features(samples, use_frame=self.use_frame)
        device = samples.device
        b, t, C = samples.size()

        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        if self.pos_emb is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
            self.pos_emb = self.transformer.wpe(pos)

        # forward the GPT model itself
        tok_emb = self.input_projection(samples)  # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb + self.pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        output = self.lm_head(x)

        if self.out is None:
            self.out = torch.zeros(size=output.shape[:-2] + (4, 4), device=x.device)

        output = orthogonalize(output[:,-1,:9], out=self.out[:len(output)].clone())

        return output


class ModelTransformerSE3(torch.nn.Module):
    def __init__(self, f=4, use_frame=False):
        torch.nn.Module.__init__(self)

        self.use_frame = use_frame

        out_dimensions = 9
        dimensions = 9 + 6 + 6 if use_frame else 9
        self.input_linear = torch.nn.Linear(dimensions, int(f * 32), bias=False)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=int(f * 32),
            nhead=1,  # todo theres a bug with more than 1 head
            dim_feedforward=int(f * 64),
            batch_first=True,
            dropout=0,
            layer_norm_eps=1e-12
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=4,
                                                   enable_nested_tensor=False)

        self.output_mlp = torch.nn.Sequential(
            *[
                torch.nn.Linear(int(f * 32), int(f * 64), bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(int(f * 64), out_dimensions, bias=True),
            ]
        )

        self.register_buffer("arange", 10**(2 * torch.arange(int(f*32)//2) / int(f*32)))
        self.position = None

    def forward(self, x):
        # x:
        #     f: N x 4 x 4
        #    df: N x 6
        #   ddf: N x 6
        #  dddf: N x 6

        #x = stack_features(samples, self.use_frame)
        if self.position is None:
            self.position_encoding = self.positional_encoding(torch.arange(x.shape[-2], device=x.device))

        x = self.input_linear(x) + self.position_encoding[None]#self.positional_encoding(self.position)

        x = self.encoder(x)

        output = self.output_mlp(x[...,-1,:])
        output = orthogonalize(output)

        return output

    def positional_encoding(self, t):
        # B X S x 1 -> B X S x P, t in [0, 1]
        sin = torch.sin(t[:,None] / self.arange[None, :])
        cos = torch.cos(t[:,None] / self.arange[None, :])
        return torch.cat([sin, cos], dim=-1)