from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                # Get iteration number from the state, or initial value.
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # Increment iteration number.
        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    for lr in [1, 1e1, 1e2, 1e3]:
        print(f"Learning rate: {lr}")
        opt = SGD([weights], lr=lr)
        for t in range(100):
            # Reset the gradients for all learnable parameters.
            opt.zero_grad()
            loss = (weights**2).mean()  # Compute a scalar loss value.
            print(f"Loss: {loss.cpu().item()}")
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
