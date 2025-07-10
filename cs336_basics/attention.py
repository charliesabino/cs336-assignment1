import torch
from torch import Tensor
from torch import nn
from jaxtyping import Float, Int
from einops import einsum, rearrange
import math
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

    logits = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(Q.shape[-1])
    if mask is not None:
        logits.masked_fill_(mask == 0, -torch.inf)
    weights = torch.softmax(logits, dim=-1)

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

        
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:

        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        factory_kwargs = {"device": device, "dtype": dtype}

        self.q_proj = Linear(d_model, d_model, **factory_kwargs)
        self.k_proj = Linear(d_model, d_model, **factory_kwargs)
        self.v_proj = Linear(d_model, d_model, **factory_kwargs)
        self.output_proj = Linear(d_model, d_model, **factory_kwargs)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:
        *batch_dims, seq_len, d_model = x.shape
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_v)

        Q = Q.transpose(-3, -2)
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)

        if rope is not None and token_positions is not None:
            original_q_shape = Q.shape
            original_k_shape = K.shape

            Q_flat = Q.reshape(-1, seq_len, self.d_k)
            K_flat = K.reshape(-1, seq_len, self.d_k)

            pos_expanded = token_positions.unsqueeze(-2)
            pos_expanded = pos_expanded.expand(*batch_dims, self.num_heads, seq_len)
            pos_flat = pos_expanded.reshape(-1, seq_len)

            Q_flat = rope(Q_flat, pos_flat)
            K_flat = rope(K_flat, pos_flat)

            Q = Q_flat.reshape(original_q_shape)
            K = K_flat.reshape(original_k_shape)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        causal_mask = ~causal_mask

        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
        attn_output = attn_output.transpose(-3, -2)
        attn_output = attn_output.reshape(*batch_dims, seq_len, self.d_model)

        output = self.output_proj(attn_output)
        return output