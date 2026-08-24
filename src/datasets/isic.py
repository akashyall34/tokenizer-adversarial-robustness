"""ISIC 2019 (skin lesion) dataset: 8-class ground truth collapsed to
malignant vs benign, single random 70/10/20 train/val/test shuffle (no
provided split, unlike APTOS -- and no separate stratified val carve-out
from a fixed train dir, unlike Kermany).

Image loading/normalization is identical to Kermany's, reused from there.
"""
import os

import numpy as np
import pandas as pd

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
MALIGNANT = {0, 2, 3, 7}  # MEL, BCC, AK, SCC -> 1 ; NV, BKL, DF, VASC -> 0


def _resolve_image_dir(data_root: str) -> str:
    """Handles the Kaggle archive's possible double-nesting."""
    d = os.path.join(data_root, "ISIC_2019_Training_Input")
    nested = os.path.join(d, "ISIC_2019_Training_Input")
    return nested if os.path.isdir(nested) else d


def build_splits(data_root: str, seed: int, val_frac: float, test_frac: float):
    img_dir = _resolve_image_dir(data_root)
    gt = pd.read_csv(os.path.join(data_root, "ISIC_2019_Training_GroundTruth.csv"))
    gt["cls"] = gt[CLASSES].values.argmax(1)
    gt["label"] = gt["cls"].apply(lambda c: 1 if c in MALIGNANT else 0)
    gt["path"] = gt["image"].apply(lambda x: f"{img_dir}/{x}.jpg")
    gt = gt[gt["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"usable: {len(gt)} | malignant {int(gt.label.sum())} / benign {int((1 - gt.label).sum())}")

    items = list(zip(gt["path"], gt["label"]))
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    n = len(items)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test_items = items[:n_test]
    val_items = items[n_test:n_test + n_val]
    train_items = items[n_test + n_val:]
    return train_items, val_items, test_items
