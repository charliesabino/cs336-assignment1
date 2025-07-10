import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float, Int
from einops import einsum, rearrange
import math
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

# scaled_dot_product_attention and Linear classes remain the same.

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    prod = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(Q.shape[-1])
    if mask is not None:
        # Fills elements with -inf where mask is 0
        prod = prod.masked_fill(mask == 0, -torch.inf)
    weights = torch.softmax(prod, dim=-1)
    return einsum(weights, V, "... queries keys, ... keys d_v -> ... queries d_v")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_v = d_model // num_heads
        self.W_q = Linear(d_model, d_k * num_heads, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_k * num_heads, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_v * num_heads, device=device, dtype=dtype)
        self.W_o = Linear(d_v * num_heads, d_model, device=device, dtype=dtype)

    def forward(
        self, 
        x: Float[Tensor, " ... sequence_length d_in"],
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)
        
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        output_reshaped = rearrange(attention_output, "... h s d -> ... s (h d)")

        return self.W_o(output_reshaped)


class MultiHeadAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_v = d_model // num_heads
        self.W_q = Linear(d_model, d_k * num_heads, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_k * num_heads, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_v * num_heads, device=device, dtype=dtype)
        self.W_o = Linear(d_v * num_heads, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len, device=device)

    def forward(
        self, 
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, " ... sequence_length d_out"]:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(Q, "... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K, "... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V, "... s (h d) -> ... h s d", h=self.num_heads)

        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))

        attention_output = scaled_dot_product_attention(Q, K, V, causal_mask)

        output_reshaped = rearrange(attention_output, "... h s d -> ... s (h d)")

        return self.W_o(output_reshaped)

        