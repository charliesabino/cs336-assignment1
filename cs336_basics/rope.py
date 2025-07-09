from torch import nn
import torch
from jaxtyping import Float, Int
from torch import Tensor

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2, device=self.device).float() / self.d_k))
        self.register_buffer("inv_freq", inv_freq)


    def forward(self, x: Float[Tensor, " ... sequence_length d_k"], token_positions: Int[Tensor, " ... sequence_length"]) -> Float[Tensor, " ... sequence_length d_k"]:
        freqs = token_positions.unsqueeze(-1).float() * self.inv_freq

        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_out_even = x_even * cos_vals - x_odd * sin_vals
        x_out_odd = x_even * sin_vals + x_odd * cos_vals

        output = torch.empty_like(x)
        output[..., ::2] = x_out_even
        output[..., 1::2] = x_out_odd

        return output