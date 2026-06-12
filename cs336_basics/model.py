import torch
from torch import nn
import math
from einops import einsum,rearrange
class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device":device,"dtype":dtype}
        
        self.weight=nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
        ) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(
            x,
            self.weight,
            "... d_in, d_out d_in -> ... d_out",
        )
        

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    
class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
            )
    
    def forward(self, x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype
        x=x.to(torch.float32)
        rms=torch.sqrt(torch.mean(x**2,dim=-1,keepdim=True)+self.eps)
        result=x/rms*self.weight
        return result.to(in_dtype)
    
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = 64 * math.ceil(d_ff / 64)

        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.w1(x)
        b = self.w3(x)

        silu_a = a * torch.sigmoid(a)

        return self.w2(silu_a * b)
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        assert d_k % 2 == 0

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 第几对维度: 0, 1, 2, ..., d_k//2 - 1
        dim_pair_idx = torch.arange(d_k // 2, device=device, dtype=torch.float32)

        # inv_freq[j] = 1 / theta^(2j / d_k)
        inv_freq = 1.0 / (theta ** (2 * dim_pair_idx / d_k))

        # 位置: 0, 1, ..., max_seq_len - 1
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        # angles.shape = (max_seq_len, d_k // 2)
        angles = positions[:, None] * inv_freq[None, :]

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x.shape = (..., seq_len, d_k)
        # token_positions.shape = (..., seq_len)

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # 让 cos/sin 能 broadcast 到 x_even 的 shape
        # 比如 x 是 (batch, heads, seq, d_k)，token_positions 是 (batch, seq)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 按照作业公式:
        # [ cos   sin ]
        # [-sin   cos ]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd

        return out
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    exp_x = torch.exp(x_stable)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp
    