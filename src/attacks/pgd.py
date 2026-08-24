"""Shared L-inf PGD loop (Adam on the perturbation), used by every attack variant."""
import torch


def run_pgd(x: torch.Tensor, loss_fn, eps: float, steps: int, lr: float, device: str) -> torch.Tensor:
    x = x.to(device)
    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn((x + delta).clamp(-1, 1))
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
    return (x + delta.detach()).clamp(-1, 1)
