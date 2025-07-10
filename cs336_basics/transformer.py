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
from cs336_basics.embedding import Embedding

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        eps: float = 1e-6,
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
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.device | None = None,
    ) -> None:
        super().__init__()

        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        assert vocab_size > 0, f"vocab_size must be positive, got {vocab_size}"
        assert context_length > 0, f"context_length must be positive, got {context_length}"
        assert num_layers > 0, f"num_layers must be positive, got {num_layers}"

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        factory_kwargs = {"device": device, "dtype": dtype}

        self.token_embeddings = Embedding(vocab_size, d_model, **factory_kwargs)

        d_k = d_model // num_heads
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_k, max_seq_len=context_length, device=device)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, eps=eps, **factory_kwargs)
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(d_model, eps, **factory_kwargs)

        self.lm_head = Linear(d_model, vocab_size, **factory_kwargs)

    def forward(
        self, input_ids: Int[torch.Tensor, "... batch_size seq_len"]
    ) -> Float[torch.Tensor, "batch_size seq_len vocab_size"]:
        batch_size, seq_len = input_ids.shape

        assert seq_len <= self.context_length, (
            f"Input sequence length ({seq_len}) exceeds context length ({self.context_length})"
        )

        token_positions = torch.arange(seq_len, device=input_ids.device)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x, rope=self.rope, token_positions=token_positions)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits