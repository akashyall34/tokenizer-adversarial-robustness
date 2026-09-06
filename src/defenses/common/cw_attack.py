"""Carlini and Wagner (2017)-style margin/confidence attack under an L2 norm
ball, used as the held-out generalization test. Every other attack in this repo
reduces to one of two loss primitives, the margin-flip loss or raw BCE against
the classifier, under an L-infinity budget, and those are the same primitives
the defense training data was generated from. This attack differs on both axes:
a confidence-margin loss that keeps applying pressure after the prediction has
already flipped, rather than stopping the instant BCE's sign flips, and L2-ball
projection instead of per-pixel clamping. No defense in this repo was ever
exposed to this loss shape or this norm, cached or live, so it is a real
generalization test rather than another remix of the same two objectives.

Not tokenizer- or classifier-specific. `reconstruct_fn` is whichever BPDA
reconstruct wrapper the tokenizer under test needs (for example
`src.defenses.vq16.bpda_attacks.plain_reconstruct_bpda`, or the multicodebook or
fsq equivalent), and `logit_fn` is any callable mapping an image to a [B]
classifier logit.
"""
import torch


def cw_margin_attack(model, reconstruct_fn, logit_fn, x, y, eps_l2: float, kappa: float,
                      steps: int, lr: float, bs: int, device: str):
    """CW's f6-style loss, adapted to a single binary logit: sign = +1 if the
    true label is positive (push logit DOWN past -kappa), -1 if negative
    (push logit UP past +kappa). f = max(sign*logit + kappa, 0). Unlike BCE,
    this keeps applying gradient pressure after the prediction has flipped,
    driving toward a confident wrong answer rather than a bare boundary
    crossing. The perturbation is L2-ball-projected every step instead of
    L-infinity-clamped."""
    x, y = x.to(device), y.float().to(device)
    sign = torch.where(y > 0.5, torch.ones_like(y), -torch.ones_like(y))
    outs = []
    for i in range(0, x.shape[0], bs):
        xb, sb = x[i:i + bs], sign[i:i + bs]
        delta = torch.zeros_like(xb, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            xa = (xb + delta).clamp(-1, 1)
            rec = reconstruct_fn(model, xa)
            lg = logit_fn(rec)
            loss = (sb * lg + kappa).clamp(min=0.0).mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                flat = delta.view(delta.shape[0], -1)
                norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
                factor = (eps_l2 / norms).clamp(max=1.0)
                delta.mul_(factor.view(-1, 1, 1, 1))
        outs.append((xb + delta.detach()).clamp(-1, 1))
        del delta, opt
        torch.cuda.empty_cache()
    return torch.cat(outs, dim=0)
