import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float
from einops import einsum
import math
from cs336_basics.linear import Linear

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    prod = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(Q.shape[-1])
    if mask is not None:
        prod = prod.masked_fill(mask == 0, -torch.inf)
    weights = torch.softmax(prod, dim=-1)
    return einsum(weights, V, "... q k, ... k d_v -> ... q d_v")

class TransformerHead(nn.Module):
    def __init__(self, d_v: int, d_k: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.W_q = Linear(d_v, d_k, device, dtype)
        self.W_k = Linear(d_v, d_k, device, dtype)
        self.W_v = Linear(d_v, d_v, device, dtype)

    
    def forward(self, x: Tensor) -> Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        mask = torch.triu(
            torch.ones(Q.size(-2), K.size(-2),
                       dtype=torch.bool, device=Q.device),
            diagonal=1
        )
        return scaled_dot_product_attention(Q, K, V, mask)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.heads = nn.ModuleList([TransformerHead(d_model // num_heads, d_model // num_heads) for _ in range(num_heads)])
        self.W_o = Linear(d_model, d_model, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.W_o(torch.cat([head(x) for head in self.heads], dim=-1))