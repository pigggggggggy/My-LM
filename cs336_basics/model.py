import torch
from torch import nn
import math
from einops import einsum
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