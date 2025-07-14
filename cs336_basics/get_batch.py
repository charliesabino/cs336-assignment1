import numpy as np
import numpy.typing as npt
import torch

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = torch.tensor(dataset, dtype=torch.long, device=device)
    dataset_len = len(dataset)
    max_start = dataset_len - context_length
    
    start_indices = torch.randint(0, max_start, (batch_size,), device=device)
    
    samples = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    labels = torch.empty((batch_size, context_length), dtype=torch.long, device=device)
    
    for i in range(batch_size):
        start = start_indices[i]
        samples[i] = dataset[start : start + context_length]
        labels[i] = dataset[start + 1 : start + context_length + 1]
    
    return samples, labels
