import torch
from cs336_basics.softmax import softmax
from jaxtyping import Float, Int
from torch import Tensor


def cross_entropy(
    logits: Float[Tensor, "batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    tot = torch.sum(torch.exp(logits), dim=-1, keepdim=True)
    log_probs = logits - torch.log(tot)
    chosen_log_probs = log_probs.gather(
        dim=1, index=targets.unsqueeze(1)).squeeze(1)
    return -chosen_log_probs.mean()
