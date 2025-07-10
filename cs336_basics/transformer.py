import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.attention import MultiHeadAttentionWithRope
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU
from cs336_basics.RMSNorm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.device = device
        self.dtype = dtype

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = MultiHeadAttentionWithRope(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "batch sequence_length d_model"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, "batch sequence_length d_model"] :
        x += self.mha(self.ln1(x), token_positions)
        x += self.ffn(self.ln2(x))
        return x