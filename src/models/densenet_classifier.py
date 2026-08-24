"""Downstream binary classifier (DenseNet121) that sits after the tokenizer.

Expects images in [-1, 1], 256x256. Checkpoint format matches the
`ckpts/classifiers/*.pt` files: a dict with `model_state_dict` plus metadata
(`image_size`, `input_range`, `positive_class`, `seed`, `best_validation_auc`).
"""
import torch
import torch.nn as nn
import torchvision


def load_densenet(ckpt_path: str, device: str) -> nn.Module:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    net = torchvision.models.densenet121(weights=None)
    net.classifier = nn.Linear(net.classifier.in_features, 1)  # binary head
    net.load_state_dict(blob["model_state_dict"])
    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def logit(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """x: [B,3,256,256] in [-1,1]. Returns [B] logit for the positive class."""
    return net(x).squeeze(-1)
