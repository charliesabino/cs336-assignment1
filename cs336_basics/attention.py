import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum
import math

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