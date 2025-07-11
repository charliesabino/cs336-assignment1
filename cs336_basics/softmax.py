import torch


def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    max_val = torch.max(x, dim=i, keepdim=True).values
    stab = x - max_val
    exp = torch.exp(stab)
    return exp / torch.sum(exp, dim=i, keepdim=True)
