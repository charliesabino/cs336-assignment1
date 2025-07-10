import torch
from torch import nn
from jaxtyping import Float, Int
from torch import Tensor
from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.linear import Linear
from cs336_basics.swiglu import SwiGLU
from cs336_basics.RMSNorm import RMSNorm
from cs336_basics.softmax import softmax
from cs336_basics.rope import RotaryPositionalEmbedding

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        self.attn = MultiHeadSelfAttention(d_model, num_heads, **factory_kwargs)
        self.ln1 = RMSNorm(d_model, eps, **factory_kwargs)

        self.ffn = SwiGLU(d_model, d_ff, **factory_kwargs)
        self.ln2 = RMSNorm(d_model, eps, **factory_kwargs)

    def forward(
        self,
        x: Float[torch.Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding | None = None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None = None,
    ) -> Float[torch.Tensor, "... seq_len d_model"]:

        normalized_x = self.ln1(x)
        attn_output = self.attn(normalized_x, rope=rope, token_positions=token_positions)
        z = x + attn_output

        normalized_z = self.ln2(z)
        ffn_output = self.ffn(normalized_z)
        y = z + ffn_output

        return y


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.token_embeddings = nn.Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, device=device, dtype=dtype) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Int[Tensor, "batch sequence_length"], token_positions: Int[Tensor, " ... sequence_length"] | None = None) -> Float[Tensor, "batch sequence_length vocab_size"]:
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x, token_positions)
        return softmax(self.lm_head(self.ln_final(x)), i=-1)