from torch import nn
import torch
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=self.device, dtype=self.dtype))
        var = (2 / (in_features + out_features)) ** 0.5
        sigma = var ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")