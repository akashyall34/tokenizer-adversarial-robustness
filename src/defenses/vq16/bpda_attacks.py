"""Adaptive-attack constructors for VQ-16: an attacker who knows the defense is
there and optimizes against it directly. Every recovery number elsewhere in this
repo measures recovery against attacks generated on the undefended tokenizer, an
attacker with no knowledge the defense exists; a real robustness claim needs the
adaptive version.

Three attack families, in increasing order of what they can see:

- BCE-through-SQS / BCE-through-plain-quantize: routes through decode() and the
  classifier, so it can exert pressure on decoder-side defenses. Needs BPDA
  (src/defenses/common/bpda.py) because the quantize step is non-differentiable.
- geo (pure margin): maximizes the correct (normalized) margin against the
  model's current boundary. Fully differentiable end to end, and never touches
  decode() or the classifier. This is a real blind spot, not a simplification: a
  defense that only changed the decoder (decoder-realism, or the combined
  checkpoint) is invisible to this attack, because the attack's gradient has no
  path through the thing that changed. A high geo-adaptive score for a
  decoder-side defense is not evidence of robustness, only evidence that this
  attack never saw what changed. See docs/known_limitations.md.
- geo_bpda (blended): geo margin plus BCE-through-decode, the same 30/70
  geometric/BCE blend as the M3_alpha03 attack, but generated live against each
  defended checkpoint rather than cached against the undefended tokenizer. Its
  BCE term routes through decode(), so this is the one that can see decoder-side
  defenses.
"""
import torch.nn.functional as F

from src.defenses.common.bpda import batched_pgd_attack, straight_through_quantize
from src.defenses.vq16.margin import margin, plain_quantize, clean_top1
from src.defenses.vq16.sqs import dispatch_quantize, quant_prequant
from src.models.densenet_classifier import logit as densenet_logit


def sqs_reconstruct_bpda(model, x, sqs_name: str, sqs_params: dict):
    h = quant_prequant(model, x)
    z_q = straight_through_quantize(h, lambda hh: dispatch_quantize(sqs_name, model, hh, sqs_params))
    return model.decode(z_q)


def plain_reconstruct_bpda(model, x):
    h = quant_prequant(model, x)
    z_q = straight_through_quantize(h, lambda hh: plain_quantize(model, hh))
    return model.decode(z_q)


def bpda_sqs_attack(model, densenet, x, y, eps, sqs_name, sqs_params, steps, lr, bs, device):
    def make_loss_fn(xb, yb):
        def loss_fn(xa):
            rec = sqs_reconstruct_bpda(model, xa, sqs_name, sqs_params)
            return -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)


def bpda_plain_attack(model, densenet, x, y, eps, steps, lr, bs, device):
    """BCE through plain (undefended) quantize, for the "alone, no SQS" checks:
    whether advft, decoder-realism, or combined survive adaptive attack on their
    own, with no SQS in the pipeline."""
    def make_loss_fn(xb, yb):
        def loss_fn(xa):
            rec = plain_reconstruct_bpda(model, xa)
            return -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)


def geo_adaptive_attack(model, x, eps, steps, lr, bs, device):
    """Maximizes the correct margin against the model's current boundary. No BPDA
    needed: the margin loss lives entirely on the continuous pre-quantization
    latent and never touches the discrete quantize step."""
    def make_loss_fn(xb, yb):
        k1 = clean_top1(model, xb)  # clean_top1 is already @torch.no_grad() internally

        def loss_fn(xa):
            return -margin(model, xa, k1, saturate=True)
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=None)


def geo_bpda_attack(model, densenet, x, y, eps, alpha, steps, lr, bs, device):
    """Blends the geo margin term with BCE-through-decode (needs BPDA, since
    decode requires a real discrete code). The live-adaptive counterpart to
    M3_alpha03's cached alpha-blend, generated fresh against each defended
    checkpoint."""
    def make_loss_fn(xb, yb):
        k1 = clean_top1(model, xb)

        def loss_fn(xa):
            geo_loss = -margin(model, xa, k1, saturate=True)
            rec = plain_reconstruct_bpda(model, xa)
            bce_loss = -F.binary_cross_entropy_with_logits(densenet_logit(densenet, rec), yb)
            return alpha * geo_loss + (1 - alpha) * bce_loss
        return loss_fn

    return batched_pgd_attack(x, eps, steps, lr, bs, device, make_loss_fn, y=y)
