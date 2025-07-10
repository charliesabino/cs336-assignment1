from torch import nn, Tensor
from jaxtyping import Float
import torch
from cs336_basics.linear import Linear

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.d_ff = d_ff
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.w1 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.w3 = Linear(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.w2 = Linear(d_ff, d_model, device=self.device, dtype=self.dtype)


    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        return self.w2(swish(self.w1(x)) * self.w3(x))