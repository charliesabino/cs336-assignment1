from typing import Iterable
import torch

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    total_norm = torch.sqrt(
        torch.sum(
            torch.stack([torch.norm(param.grad.data, p=2)**2 for param in parameters if param.grad is not None])
        )
    )
    
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        
        for param in parameters:
            if param.grad is not None:
                param.grad.data *= scale