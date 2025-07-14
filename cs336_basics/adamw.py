from collections.abc import Callable
from typing import Optional
import torch
import math


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-3, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta_1, beta_2 = betas
        defaults = {"lr": lr, "beta_1": beta_1, "beta_2": beta_2,
                    "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            beta_1 = group["beta_1"]
            beta_2 = group["beta_2"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 1)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data

                m = state.get("m", torch.zeros_like(p))
                m = beta_1 * m + (1 - beta_1) * grad
                state["m"] = m

                v = state.get("v", torch.zeros_like(p))
                v = beta_2 * v + (1 - beta_2) * grad**2
                state["v"] = v

                lr_t = lr * math.sqrt(1 - beta_2**t) / (1 - beta_1**t)

                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= p.data * lr * weight_decay

                state["t"] = t + 1  # Increment iteration number.

        return loss
