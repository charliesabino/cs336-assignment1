import torch
from cs336_basics.softmax import softmax

def sample_top_p(dist: torch.Tensor, p: float = 0.9):
    sorted_indices = torch.argsort(dist, dim=-1, descending=True)
    sorted_dist = dist[sorted_indices]
    cumsum = torch.cumsum(sorted_dist, dim=-1)
    mask = cumsum > p
    mask[:, 1:] = mask[:, :-1].clone()
    mask[:, 0] = False
    sorted_dist[mask] = 0.0
    sorted_dist = sorted_dist / sorted_dist.sum(dim=-1, keepdim=True)
    unsorted_dist = torch.zeros_like(dist)
    unsorted_dist.scatter_(dim=-1, index=sorted_indices, src=sorted_dist)
    return torch.multinomial(unsorted_dist, num_samples=1).squeeze(-1)

def decode(model, input: torch.Tensor, temperature: float = 1.0):
    logits = model(input)
    dist = softmax(logits, i=-1, temp=temperature)
    sample = sample_top_p(dist, p=0.9)
    return sample