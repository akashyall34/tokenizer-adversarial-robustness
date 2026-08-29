"""APTOS downstream classifier: EfficientNet-B5, 5-grade softmax head collapsed
to one binary logit (referable-DR vs not) via logsumexp over grade groups.

Checkpoint is a bare state_dict (torchvision efficientnet_b5, 5-class head),
fine-tuned at 456px with standard ImageNet normalization -- the checkpoint
itself doesn't record input size/normalization, so both are asserted here
via `load_densenet`-style convention rather than read from the checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

EFF_SIZE = 456  # torchvision B5 native resolution; the checkpoint was fine-tuned at this size
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_efficientnet(ckpt_path: str, device: str, num_grades: int = 5) -> nn.Module:
    net = torchvision.models.efficientnet_b5(weights=None)
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_grades)
    net.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def make_to_effnet_space(device: str, eff_size: int = EFF_SIZE):
    """Returns to_effnet_space(x): [B,3,256,256] in [-1,1] (tokenizer space) ->
    B5 input space (resized, ImageNet-normed). Shared by the logit and
    perceptual-feature closures below so both preprocess identically."""
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def to_effnet_space(x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1.0) / 2.0
        if eff_size != x.shape[-1]:
            x01 = F.interpolate(x01, size=eff_size, mode="bilinear", align_corners=False)
        return (x01 - mean) / std

    return to_effnet_space


def make_binary_logit_fn(net: nn.Module, device: str, pos_grades=(2, 3, 4), num_grades: int = 5, eff_size: int = EFF_SIZE):
    """Returns logit_fn(x): [B,3,256,256] in [-1,1] (tokenizer space) -> [B] single
    logit for the positive (referable-DR) group. Differentiable end-to-end."""
    to_effnet_space = make_to_effnet_space(device, eff_size)
    pos = sorted(pos_grades)
    neg = [g for g in range(num_grades) if g not in pos_grades]
    pos_idx = torch.tensor(pos, device=device)
    neg_idx = torch.tensor(neg, device=device)

    def logit_fn(x: torch.Tensor) -> torch.Tensor:
        logits = net(to_effnet_space(x))  # [B, num_grades]
        pos_lse = torch.logsumexp(logits.index_select(1, pos_idx), dim=1)
        neg_lse = torch.logsumexp(logits.index_select(1, neg_idx), dim=1)
        return pos_lse - neg_lse

    return logit_fn


def make_features_fn(net: nn.Module, device: str, eff_size: int = EFF_SIZE):
    """Returns features_fn(x): [B,3,256,256] in [-1,1] (tokenizer space) ->
    B5's conv-feature map (pre-classifier-head), for perceptual-loss use in
    decoder-realism training -- the EfficientNet-B5 analogue of DenseNet's
    own `.features(x)`, but with the required resize+ImageNet-normalize
    preprocessing folded in (DenseNet's `.features` needs no such step,
    since its checkpoint's own input convention matches tokenizer space
    directly at 256px)."""
    to_effnet_space = make_to_effnet_space(device, eff_size)

    def features_fn(x: torch.Tensor) -> torch.Tensor:
        return net.features(to_effnet_space(x))

    return features_fn


class EfficientNetClassifier:
    """Wraps EfficientNet-B5 into the same two-method interface DenseNet121's
    raw nn.Module exposes for free (`net(x)` -> logit-shaped output,
    `net.features(x)` -> perceptual features), so the shared
    margin / decoder-realism / certify code written against DenseNet's
    calling convention runs unchanged against APTOS. Callers alias it as
    `densenet` / `densenet_logit` for exactly this reason."""

    def __init__(self, net: nn.Module, device: str, pos_grades=(2, 3, 4), num_grades: int = 5, eff_size: int = EFF_SIZE):
        self.net = net
        self._logit_fn = make_binary_logit_fn(net, device, pos_grades, num_grades, eff_size)
        self._features_fn = make_features_fn(net, device, eff_size)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._logit_fn(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._features_fn(x)
