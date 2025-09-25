import torch
class ModelTransformerSE3(torch.nn.Module):
    def __init__(self, f=4):
        torch.nn.Module.__init__(self)

        dimensions = 9
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
                torch.nn.Linear(int(f * 64), int(f * 32), bias=True),
            ]
        )


        self.register_buffer("arange", 10**(2 * torch.arange(int(f*32)//2) / int(f*32)))


    def stack_se3(self, f):
        return torch.cat([f[...,:3,0], f[...,:3,1], f[...,:3,3]], dim=-1)

    def forward(self, timestamps, samples):
        # timestamps B x N
        # samples:
        #     f: N x 4 x 4
        #    df: N x 6
        #   ddf: N x 6
        #  dddf: N x 6

        x = self.stack_se3(samples["f"])
        x = self.input_linear(x) + self.positional_encoding(samples["position"])

        x = self.encoder(x)

        output = self.output_mlp(x[...,-1,:])
        output = self.orthogonalize(output)

        return output

    def orthogonalize(self, x, eps=1e-18):
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
        se3 = torch.zeros(size=x.shape[:-1] + (4, 4), device=x.device)
        se3[...,:3,:3] = R
        se3[...,:3, 3] = t
        se3[...,-1,-1] = 1

        assert not torch.isnan(se3).any()

        return se3

    def positional_encoding(self, t):
        # B X S x 1 -> B X S x P, t in [0, 1]
        sin = torch.sin(t[:,:,None] / self.arange[None, None, :])
        cos = torch.cos(t[:,:,None] / self.arange[None, None, :])
        return torch.cat([sin, cos], dim=-1)