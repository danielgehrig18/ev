import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch_scatter


class TransformerEncoderJagged(torch.nn.Module):
    def __init__(self, d_model, dim_feedforward, nhead, dropout=0, layer_norm_eps=1e-12):
        torch.nn.Module.__init__(self)

        self.attention = SelfAttentionJagged(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._ff_block = torch.nn.Sequential(
            torch.nn.Linear(d_model, dim_feedforward),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self._ff_block(x))
        return x

class SelfAttentionJagged(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0):
        super().__init__()
        self.nhead = nhead
        self.dropout_p = dropout
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        assert d_model % nhead == 0, "Embedding dim is not divisible by nheads"
        self.d_head = d_model // nhead

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): query of shape (N, L_t, E_q)

        Returns:
            attn_output (torch.Tensor): output of shape (N, L_t, E_q)
        """
        # Step 1. Apply input projection
        # TODO: demonstrate packed projection
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)

        # Step 2. Split heads and prepare for SDPA
        # reshape query, key, value to separate by head
        # (N, L_t, E_total) -> (N, L_t, nheads, E_head) -> (N, nheads, L_t, E_head)
        query = query.unflatten(-1, [self.nhead, self.d_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        key = key.unflatten(-1, [self.nhead, self.d_head]).transpose(1, 2)
        # (N, L_s, E_total) -> (N, L_s, nheads, E_head) -> (N, nheads, L_s, E_head)
        value = value.unflatten(-1, [self.nhead, self.d_head]).transpose(1, 2)

        # Step 3. Run SDPA
        # (N, nheads, L_t, E_head)
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout_p)
        # (N, nheads, L_t, E_head) -> (N, L_t, nheads, E_head) -> (N, L_t, E_total)
        attn_output = attn_output.transpose(1, 2).flatten(-2)

        # Step 4. Apply output projection
        # (N, L_t, E_total) -> (N, L_t, E_out)
        attn_output = self.out_proj(attn_output)

        return attn_output

def coerce_offsets(src, tgt):
    assert torch.eq(src.offsets(), tgt.offsets()).all().item()
    assert src._ragged_idx == tgt._ragged_idx

    def mb_get_size(t):
        return t.shape[0] if t is not None else None

    return torch.nested.nested_tensor_from_jagged(
        src.values(),
        tgt.offsets(),
        None,
        src._ragged_idx,
        mb_get_size(src._max_seqlen_tensor) if tgt._max_seqlen_tensor is None else mb_get_size(src._max_seqlen_tensor),
        mb_get_size(src._min_seqlen_tensor) if tgt._min_seqlen_tensor is None else mb_get_size(src._min_seqlen_tensor),
    )


def transformer_block(f):
    return TransformerEncoderJagged(
            d_model=int(f*32),
            nhead=1, # todo theres a bug with more than 1 head
            dim_feedforward=int(f*64),
            dropout=0,
            layer_norm_eps=1e-12
        )

class ModelTransformerJagged(torch.nn.Module):
    def __init__(self, f=4):
        torch.nn.Module.__init__(self)
        self.input_linear = torch.nn.Linear(1, int(f*32), bias=False)
        self.output_linear = torch.nn.Linear(int(f*32), 1, bias=False)

        self.encoder = torch.nn.Sequential(
            transformer_block(f),
            transformer_block(f),
            transformer_block(f),
            transformer_block(f)
        )

    def forward(self, c_jagged, v_jagged):
        x = self.input_linear(v_jagged)
        p = self.positional_encoding(c_jagged)

        #p = coerce_offsets(p, x)
        #x = x + p

        x = self.encoder(x)
        x = self.output_linear(x)
        output = torch.stack([x_[0,0] for x_ in x])

        return output

    def positional_encoding(self, t):
        # B X S x P
        sin = torch.sin(t)
        cos = torch.cos(t)
        return torch.cat([sin, cos], dim=-1)


        
def to_nested(batch, x):
    x = [x[batch == b] for b in batch.unique()]
    x = torch.nested.nested_tensor(x, dtype=x.dtype, device=x.device)
    return x




class ScatterTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0, layer_norm_eps=1e-12, bias=True, norm_first=False, **factory_kwargs):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps

        self.self_attn = ScatterMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def scaled_dot_product(self, batch, q, k, v):
        # N x H x C
        q_nested = nt = torch.nested.nested_tensor([torch.arange(12).reshape(
    2, 6), torch.arange(18).reshape(3, 6)], dtype=torch.float, device=device)

        return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    def self_attn(self, x):
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(-1, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1) # N x H x C

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

    def _sa_block(self, batch, x):
        x = self.self_attn(x)
        return self.dropout1(x)

    def forward(self, batch, src):
        x = src
        if self.norm_first:
            x = x + self._sa_block(batch, self.norm1(x))
            x = x + self._ff_block(batch, self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(batch, x))
            x = self.norm2(x + self._ff_block(batch, x))
        return x

class ScatterTransformer(torch.nn.Module):
    def __init__(self, f=4, batch_size=1, sequence_length=100):
        torch.nn.Module.__init__(self)
        self.sequence_length = sequence_length

        self.input_linear = torch.nn.Linear(1, int(f*32), bias=False)
        self.output_linear = torch.nn.Linear(int(f*32), 1, bias=False)

        self.encoder_layer = ScatterTransformerEncoderLayer(
            d_model=int(f*32),
            nhead=1, # todo theres a bug with more than 1 head
            dim_feedforward=int(f*64),
            batch_first=True,
            dropout=0,
            layer_norm_eps=1e-12
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=4,
                                             enable_nested_tensor=False)

        self.register_buffer("arange", 10**(2 * torch.arange(int(f*32)//2) / int(f*32)))

    def forward(self, batch_size, max_num_events, c, b, t, v):
        if max_num_events > self.input.shape[-1]:
            self.input = torch.zeros(size=(batch_size, 2, max_num_events), device=c.device)
            self.mask = torch.zeros(size=(batch_size, max_num_events), device=c.device) > 0

        self.input = self.input.to(c.device)
        self.mask = self.mask.to(c.device)

        x = self.input[:batch_size, :, :max_num_events].clone()

        # normalize c to have range 0,...,N-1 for fixed N
        cmax, _ = torch_scatter.scatter_max(c, b, dim_size=batch_size)
        eps = 1e-9

        x[b,0,c] = v.float()
        x[b,1,c] = c.float() / (cmax[b] + eps) * (self.sequence_length - 1)
        #x[b,1,c] = t.float()
        #x[b,2,c] = c.float()

        mask = self.mask[:batch_size, :max_num_events].clone()
        mask[b,c] = True

        x = x.permute(0, 2, 1)
        x = self.input_linear(x[...,:1]) + self.positional_encoding(x[...,1:])
        attn_mask = ~(mask[:, None, :] & mask[:, :, None]).repeat_interleave(self.encoder_layer.self_attn.num_heads, 0)
        x = self.encoder(x, mask=attn_mask)
        x = self.output_linear(x)
        output = x[:,0,0]

        return output

    def positional_encoding(self, t):
        # B X S x 1 -> B X S x P, t in [0, 1]
        sin = torch.sin(t / self.arange[None, None, :])
        cos = torch.cos(t / self.arange[None, None, :])
        return torch.cat([sin, cos], dim=-1)


class ReconstructionModel(torch.nn.Module):
    def __init__(self, batch_size, sequence_length, dimensions, bandwidth, resolution, time_window_s,
                 threshold, reconstruction_type="pocs"):
        torch.nn.Module.__init__(self)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dimensions = dimensions
        self.bandwidth = bandwidth
        self.resolution = resolution
        self.time_window_s = time_window_s
        self.threshold = threshold
        self.register_buffer("timestamps", torch.linspace(start=0, end=self.time_window_s, steps=self.resolution))

        self.reconstruction_type = reconstruction_type

    def compute_w(self, t):
        w = torch.zeros_like(t)
        w[1:-1] = (t[2:] - t[:-2]) / 2
        w[0] = t[1] - t[0]
        w[-1] = t[-1] - t[-2]
        return w

    def forward(self, batch_size, max_num_events, c, b, t, v, gt_x, gt_timestamps):
        if self.reconstruction_type == "whittaker":
            return self.forward_whittaker(batch_size, max_num_events, c, b, t, v)
        elif self.reconstruction_type == "pocs":
            return self.forward_pocs(batch_size, b, t, v, gt_x, gt_timestamps)

    def plot(self, timestamps, output, t, v, b, title="bla"):
        timestamps, output, t, v, b = timestamps.cpu().numpy(), output.cpu().numpy(), t.cpu().numpy(), v.cpu().numpy(), b.cpu().numpy()

        fig, ax = plt.subplots()
        ax.plot(timestamps, output[0,:])
        ax.scatter(t[b==0], v[b==0])
        fig.suptitle(title)
        plt.show()


    def forward_pocs(self, batch_size, b, t, v, gt_v, gt_timestamps):

        rmse = lambda gt, est: (gt - est).pow(2).mean().pow(.5)

        v0 = self.init_pocs(batch_size, b, t, v, order=0)
        output = self.init_pocs(batch_size, b, t, v, order=1)

        print("RMSE: ", rmse(gt_v, output))

        self.plot(self.timestamps, output, t, v, b, title="Init")

        for i in range(100):
            output = self.project_bandlimited_through_points(self.timestamps, output, t, v, b)
            print("RMSE: ", rmse(gt_v, output))

            self.plot(self.timestamps, output, t, v, b, title="After band")

            output = self.project_range(v0, output, self.threshold)
            print("RMSE: ", rmse(gt_v, output))

            self.plot(self.timestamps, output, t, v, b, title="After range proj")

        return output

    def interpolate(self, t, y, t_q):
        idx = torch.searchsorted(t, t_q)
        r = (t_q - t[idx-1]) / (t[idx] - t[idx-1])
        y_out = y[idx] * r + y[idx-1] * (1 - r)
        return y_out

    def compute_c(self, y_n, t_n, damping, omega):
        wdt = torch.sinc(omega/torch.pi * (t_n[:,None] - t_n[None,:]))
        S_inv = torch.linalg.pinv(wdt + torch.eye(len(wdt), device=wdt.device) * damping)
        c_n = S_inv @ y_n
        return c_n

    def fit_residuals(self, t, g_w, t_n, x_n, omega):
        # compute g_w_n at time stamps t_n via interpolation
        idx = torch.searchsorted(t, t_n)
        r = (t_n - t[idx-1]) / (t[idx] - t[idx-1])
        r[idx<0] = 1
        g_w_n = g_w[idx] * r + g_w[idx-1] * (1 - r)

        # find the residuals y_n to measurements x_n
        y_n = x_n - g_w_n

        # compute the coefficients c_n
        damping = 1e-1
        S = torch.sinc(omega/torch.pi * (t_n[:,None] - t_n[None,:]))
        I = torch.eye(len(S), device=S.device)
        S_inv = torch.linalg.pinv(S + I * damping)
        c_n = S_inv @ y_n

        # compute the signal at original timestamps t
        S = torch.sinc(omega/torch.pi * (t[:,None] - t_n[None,:]))
        y_hat = S @ c_n

        return y_hat

    def project_bandlimited_through_points(self, t, g, t_n, v_n, b):
        n_max = 10
        fft = torch.fft.rfft(g, dim=1, norm="ortho")
        fft[:,n_max:] = 0
        g_w = torch.fft.irfft(fft, dim=1, norm="ortho")

        omega = n_max * 2 * torch.pi
        y_hat = torch.cat(
            [
                self.fit_residuals(t=t, g_w=g_w[i], t_n=t_n[b==i], x_n=v_n[b==i], omega=omega)
                for i in range(len(g))
            ]
        )

        output = g_w + y_hat

        return output

    def project_range(self, v_0, output, threshold):
        n = output - v_0
        if len(n.shape) > 2:
            norm = torch.norm(n, dim=-1)
        else:
            norm = torch.abs(n)
        mask = norm > threshold
        output[mask] = v_0[mask] + n[mask] / norm[mask] * threshold
        return output

    def init_pocs(self, batch_size, b, t, v, order=0):
        index = torch.stack([torch.searchsorted(t[bidx == b], self.timestamps, right=True) for bidx in range(batch_size)])
        index = torch.clamp(index-1, 0, max=torch.inf).long()
        if order == 0:
            v_output = v[index]
        else:
            index1 = torch.clamp(index+1, 0, max=len(v)-1)
            v0 = v[index]
            v1 = v[index1]
            eps = 1e-6
            r = (self.timestamps - t[index]) / (t[index1] - t[index] + eps)
            r[index1==index] = 0
            v_output = v0 * (1-r) + v1 * r

        return v_output

    def forward_whittaker(self, batch_size, max_num_events, c, b, t, v):
        # x(t) = \sum_j x(t_j) w_j  sinc(w(t-t_j))
        cmax, _ = torch_scatter.scatter_max(c, b, dim_size=batch_size)
        output = []
        for bidx in range(batch_size):
            t_b = t[b==bidx]
            x_j = v[b==bidx]
            dt = (self.time_window_s / len(t_b))
            w_j = self.compute_w(t_b) / dt
            sinc_ij = torch.sinc((self.timestamps[:, None] - t_b[None, :]) / dt)
            w_j_sinc_ij = w_j[None, :] * sinc_ij
            x_t = w_j_sinc_ij @ x_j
            output.append(x_t)
        output = torch.stack(output)
        return output, self.timestamps





class ModelTransformer(torch.nn.Module):
    def __init__(self, f=4, batch_size=1, sequence_length=100, dimensions=1):
        torch.nn.Module.__init__(self)
        self.sequence_length = sequence_length
        self.input = torch.zeros(size=(batch_size, 1+dimensions, sequence_length))
        self.mask= torch.zeros(size=(batch_size, sequence_length)) > 0

        self.dimensions = dimensions
        self.input_linear = torch.nn.Linear(dimensions, int(f*32), bias=False)
        self.output_linear = torch.nn.Linear(int(f*32), dimensions, bias=False)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=int(f*32),
            nhead=1, # todo theres a bug with more than 1 head
            dim_feedforward=int(f*64),
            batch_first=True,
            dropout=0,
            layer_norm_eps=1e-12
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=4,
                                             enable_nested_tensor=False)

        self.register_buffer("arange", 10**(2 * torch.arange(int(f*32)//2) / int(f*32)))



    def forward(self, batch_size, max_num_events, c, b, t, v):
        if max_num_events > self.input.shape[-1]:
            self.input = torch.zeros(size=(batch_size, 1+self.dimensions, max_num_events), device=c.device)
            self.mask = torch.zeros(size=(batch_size, max_num_events), device=c.device) > 0

        self.input = self.input.to(c.device)
        self.mask = self.mask.to(c.device)

        x = self.input[:batch_size, :, :max_num_events].clone()

        # normalize c to have range 0,...,N-1 for fixed N
        cmax, _ = torch_scatter.scatter_max(c, b, dim_size=batch_size)
        eps = 1e-9

        x[b,0,c] = c.float() / (cmax[b] + eps) * (self.sequence_length - 1)
        x[b,1:,c] = v.float()
        #x[b,1,c] = t.float()
        #x[b,2,c] = c.float()

        mask = self.mask[:batch_size, :max_num_events].clone()
        mask[b,c] = True

        x = x.permute(0, 2, 1)
        x = self.input_linear(x[...,1:]) + self.positional_encoding(x[...,:1])
        attn_mask = ~(mask[:, None, :] & mask[:, :, None]).repeat_interleave(self.encoder_layer.self_attn.num_heads, 0)
        x = self.encoder(x, mask=attn_mask)
        x = self.output_linear(x)
        output = x[:,0]

        return output

    def positional_encoding(self, t):
        # B X S x 1 -> B X S x P, t in [0, 1]
        sin = torch.sin(t / self.arange[None, None, :])
        cos = torch.cos(t / self.arange[None, None, :])
        return torch.cat([sin, cos], dim=-1)


class ModelCNN(torch.nn.Module):
    def __init__(self, f=1, batch_for_loop=False, batch_size=1, sequence_length=100):
        torch.nn.Module.__init__(self)

        self.model = ResNet18(num_classes=1)
        self.input = torch.zeros(size=(batch_size, 1, sequence_length))
        self.mask= torch.zeros(size=(batch_size, sequence_length)) > 0
        self.batch_for_loop = batch_for_loop

    def forward_single(self, c, t, v):
        S = int((c[-1]+1).cpu())
        x = self.input[0,:,:S].clone()
        x[0,c] = v.float()
        #x[1,c] = t.float()
        #x[2,c] = c.float()
        return self.model(x[None], None)[0]

    def forward(self, batch_size, max_num_events, c, b, t, v):
        if self.batch_for_loop:
            outputs = []
            for bi in range(batch_size):
                output = self.forward_single(c[b==bi], t[b==bi], v[b==bi])
                outputs.append(output)
            outputs = torch.stack(outputs)
            return outputs[:,0]
        else:
            return self.forward_batched(batch_size, max_num_events, c, b, t, v)

    def forward_batched(self, batch_size, max_num_events, c, b, t, v):
        if max_num_events > self.input.shape[-1]:
            self.input = torch.zeros(size=(batch_size, 1, max_num_events), device=c.device)
            self.mask = torch.zeros(size=(batch_size, max_num_events), device=c.device) > 0

        self.input = self.input.to(c.device)
        self.mask = self.mask.to(c.device)

        x = self.input[:batch_size, :, :max_num_events].clone()
        x[b,0,c] = v.float()
        #x[b,1,c] = t.float()
        #x[b,2,c] = c.float()

        mask = self.mask[:batch_size, :max_num_events].clone()
        mask[b,c] = True

        return self.model(x, mask)[:,0]




class MaskedReLU(nn.ReLU):
    def forward(self, x):
        x, mask = x
        x = nn.ReLU.forward(self, x)
        return x, mask


class MaskedBatchNorm1D(nn.BatchNorm1d):
    def forward(self, input):
        input, mask = input
        # input: B x C x L
        # mask: B x L

        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            if mask is not None:
                n = mask.sum([0,1])+1e-9
            else:
                n = input.size(0) * input.size(2)

            mean = input.sum([0, 2]) / (1e-9 + n)
            sq_mean = (input**2).sum([0, 2]) / (1e-9 + n)
            var = sq_mean - mean**2


            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean \
                                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1) \
                                   + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None] + self.bias[None, :, None]

        return input, mask


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.branch = nn.Sequential(
            MaskedConv1D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            MaskedBatchNorm1D(out_channels),
            MaskedReLU(),
            MaskedConv1D(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            MaskedBatchNorm1D(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                MaskedConv1D(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                MaskedBatchNorm1D(out_channels)
            )

    def forward(self, x):
        x, mask = x
        out, new_mask = self.branch((x, mask))
        out += self.shortcut((x, mask) )[0]
        out = nn.functional.relu(out)
        return out, new_mask

def update_mask(mask, stride):
    if stride < 2:
        return mask

    if mask.shape[1] % 2 == 1:
        mask = torch.cat([mask, torch.zeros(size=(mask.shape[0], 1), device=mask.device) > 0], dim=1)

    num_events = mask.sum(1).long()
    non_border = num_events < mask.shape[1]
    arange = torch.arange(len(mask),device=mask.device)
    mask[arange[non_border], num_events[non_border]] = (num_events[non_border] % 2 == 1)

    mask = mask[:, ::stride]

    return mask


class MaskedConv1D(nn.Conv1d):
    def forward(self, x):
        x, mask = x
        # x: B X C X L
        # mask: B X L
        x = super(MaskedConv1D, self).forward(x)
        if mask is None:
            return x, mask

        # update mask
        mask = update_mask(mask, self.stride[0])

        # apply mask
        x = (x * mask[:, None, :].float())

        return x, mask

class MaskedMaxPool1D(nn.MaxPool1d):
    def forward(self, x):
        x, mask = x

        # x: B X C X L
        # mask: B X L

        if mask is not None:
            x = x.permute(0, 2, 1)
            x[~mask, :] = -torch.inf
            x = x.permute(0, 2, 1)

            mask = update_mask(mask, self.stride)

        x = super(MaskedMaxPool1D, self).forward(x)

        if mask is not None:
            x = x.permute(0, 2, 1)
            x[~mask, :] = 0
            x = x.permute(0, 2, 1)

        return x, mask

class MaskedAdaptiveAvgPool1d(nn.Module):
    def forward(self, x):
        x, mask = x
        # x: B x C x L
        # mask: B x L
        if mask is None:
            return x.mean(-1)
        else:
            return x.sum(2) / mask[:,None,:].sum(2)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = MaskedConv1D(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = MaskedBatchNorm1D(64)
        self.relu = nn.ReLU()
        self.maxpool = MaskedMaxPool1D(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        self.avgpool = MaskedAdaptiveAvgPool1d()
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        out, mask = self.conv1((x, mask))
        out, mask = self.bn1((out, mask))
        #out = torch.nn.functional.relu(out)
        out[out<0] = 0
        out, mask = self.maxpool((out, mask))

        out, mask = self.layer1((out, mask))
        out, mask = self.layer2((out, mask))
        out, mask = self.layer3((out, mask))
        out, mask = self.layer4((out, mask))

        out = self.avgpool((out, mask))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    import torch

    nt = torch.nested.nested_tensor(
        [torch.arange(12).reshape(12, 1) / 12,
                  torch.arange(14).reshape(14, 1) / 14,
                  torch.arange(15).reshape(15, 1) / 15,
                  torch.arange(10).reshape(10, 1) / 10], dtype=torch.float, device="cuda", layout=torch.jagged)

    layer = ModelTransformerJagged(f=1).cuda()
    output = layer.forward(nt, nt)