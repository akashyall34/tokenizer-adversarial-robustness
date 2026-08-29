"""BPDA (Backward Pass Differentiable Approximation, Athalye/Carlini/Wagner
2018): substitutes identity for the backward pass through a non-differentiable
forward operation (e.g. SQS's mode/argmin voting, or a plain quantizer's
argmin), so gradient can still flow through an otherwise-real, non-relaxed
forward pass. Needed for every "does this defense survive an attacker who
knows it's there" check in this repo.

`batched_pgd_attack` factors out the PGD-with-chunking loop shared by every
attack variant here (BCE-through-SQS, BCE-through-plain-quantize, geometric
margin, and the blended geo+BCE attack) -- they differ only in what loss
they optimize per step, not in the optimization loop itself. Chunking is
purely a memory concern: each image's delta/Adam state is independent of
every other image in the batch (no cross-image interaction anywhere in the
loss or model), so splitting into chunks changes nothing about the per-image
result, only peak memory.
"""
import torch


class _StraightThrough(torch.autograd.Function):
    """Forward: the real, non-differentiable quantize_fn(h) (fresh randomness
    on every call, if the underlying mechanism uses any). Backward: identity
    -- gradient passes straight through h unchanged."""

    @staticmethod
    def forward(ctx, h, quantize_fn):
        with torch.no_grad():
            return quantize_fn(h)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def straight_through_quantize(h: torch.Tensor, quantize_fn) -> torch.Tensor:
    """quantize_fn: h -> z_q (not a tuple -- unwrap (idx, z_q) callables before passing in)."""
    return _StraightThrough.apply(h, quantize_fn)


def batched_pgd_attack(x: torch.Tensor, eps: float, steps: int, lr: float, bs: int, device: str,
                        make_loss_fn, y: torch.Tensor = None):
    """make_loss_fn(xb, yb) -> loss_fn(xa) -> scalar loss. make_loss_fn is
    called once per chunk (batch) to do any per-chunk setup under no_grad
    (e.g. computing the clean target code) before the PGD loop starts;
    loss_fn is then called once per PGD step with the current perturbed
    input. yb is None if y is None (attacks that don't need labels, e.g. the
    pure geometric margin attack).
    """
    x = x.to(device)
    y = y.float().to(device) if y is not None else None
    outs = []
    for i in range(0, x.shape[0], bs):
        xb = x[i:i + bs]
        yb = y[i:i + bs] if y is not None else None
        loss_fn = make_loss_fn(xb, yb)
        delta = torch.zeros_like(xb, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            xa = (xb + delta).clamp(-1, 1)
            loss = loss_fn(xa)
            loss.backward()
            opt.step()
            with torch.no_grad():
                delta.clamp_(-eps, eps)
        outs.append((xb + delta.detach()).clamp(-1, 1))
        del delta, opt
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)
