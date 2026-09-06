"""ISIC downstream classifier: EfficientNet-B4 from timm, not torchvision, so a
different checkpoint and library than APTOS's EfficientNet-B5. The 8-class
softmax head is collapsed to one binary logit (malignant vs benign) by a
logsumexp over the two grade groups.
"""
import timm
import torch
import torch.nn.functional as F

IMG_SIZE_CLF = 380
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
MAL_IDX = [0, 2, 3, 7]  # MEL, BCC, AK, SCC
BEN_IDX = [1, 4, 5, 6]  # NV, BKL, DF, VASC


def load_efficientnet_b4(ckpt_path: str, device: str, num_classes: int = 8) -> torch.nn.Module:
    net = timm.create_model("efficientnet_b4", pretrained=False, num_classes=num_classes).to(device)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd)) if isinstance(sd, dict) else sd
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print("loaded effnet-b4 | missing:", len(missing), "unexpected:", len(unexpected))
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def make_to_effnet_space(device: str, img_size_clf: int = IMG_SIZE_CLF):
    """Returns to_effnet_space(x): [B,3,256,256] in [-1,1] (tokenizer space)
    -> B4 input space (resized, ImageNet-normed). Shared by the logit and
    perceptual-feature closures below so both preprocess identically."""
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def to_effnet_space(x: torch.Tensor) -> torch.Tensor:
        x01 = (x.clamp(-1, 1) + 1) / 2
        xn = (x01 - mean) / std
        return F.interpolate(xn, size=img_size_clf, mode="bilinear", align_corners=False)

    return to_effnet_space


def make_binary_logit_fn(net: torch.nn.Module, device: str, img_size_clf: int = IMG_SIZE_CLF):
    """Returns logit_fn(x): [B,3,256,256] in [-1,1] (tokenizer space) -> [B] single
    logit for the malignant group. Differentiable end-to-end."""
    to_effnet_space = make_to_effnet_space(device, img_size_clf)
    mal_idx = torch.tensor(MAL_IDX, device=device)
    ben_idx = torch.tensor(BEN_IDX, device=device)

    def logit_fn(x: torch.Tensor) -> torch.Tensor:
        logits8 = net(to_effnet_space(x))
        return torch.logsumexp(logits8[:, mal_idx], dim=1) - torch.logsumexp(logits8[:, ben_idx], dim=1)

    return logit_fn


def make_features_fn(net: torch.nn.Module, device: str, img_size_clf: int = IMG_SIZE_CLF):
    """Returns features_fn(x): [B,3,256,256] in [-1,1] (tokenizer space) ->
    B4's conv-feature map (pre-classifier-head), for perceptual-loss use in
    decoder-realism training. timm's `forward_features` is the library's own
    name for this (torchvision's EfficientNet exposes the equivalent
    submodule as `.features`; timm's B4 doesn't have a `.features`
    attribute, so this calls the model's own accessor instead)."""
    to_effnet_space = make_to_effnet_space(device, img_size_clf)

    def features_fn(x: torch.Tensor) -> torch.Tensor:
        return net.forward_features(to_effnet_space(x))

    return features_fn


class EfficientNetB4Classifier:
    """Wraps timm's EfficientNet-B4 into the same two-method interface
    DenseNet121's raw nn.Module exposes for free (`net(x)` -> logit-shaped
    output, `net.features(x)` -> perceptual features), so the shared
    margin / decoder-realism / certify code written against DenseNet's
    calling convention runs unchanged against ISIC."""

    def __init__(self, net: torch.nn.Module, device: str, img_size_clf: int = IMG_SIZE_CLF):
        self.net = net
        self._logit_fn = make_binary_logit_fn(net, device, img_size_clf)
        self._features_fn = make_features_fn(net, device, img_size_clf)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._logit_fn(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self._features_fn(x)
