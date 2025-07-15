import torch


def softmax(x: torch.Tensor, i: int, temp: float = 1.0) -> torch.Tensor:
    max_val = torch.max(x, dim=i, keepdim=True).values
    stab = x - max_val
    exp = torch.exp(stab / temp)
    return exp / torch.sum(exp, dim=i, keepdim=True)
