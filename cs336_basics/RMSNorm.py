import torch
from torch import nn
from einops import reduce

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.gain = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(self.dtype)
        RMS = torch.sqrt(reduce(x ** 2, '... d -> ... 1', 'mean') + self.eps)
        x = (x / RMS) * self.gain
        return x.to(in_dtype)