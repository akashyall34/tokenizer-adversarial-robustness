"""Kermany chest X-ray dataset: deterministic splits + loaders.

Images are RGB, resized to a square, normalized to [-1, 1] (the range
every tokenizer in this repo expects).
"""
import collections
import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

CLASSES = ["NORMAL", "PNEUMONIA"]  # label 0, 1


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _scan(root: str, split: str) -> list[tuple[str, int]]:
    items = []
    for lbl, c in enumerate(CLASSES):
        d = os.path.join(root, split, c)
        for fn in sorted(os.listdir(d)):  # sorted() -> deterministic order
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                items.append((os.path.join(d, fn), lbl))
    return items


def build_splits(root: str, out_dir: str, seed: int, val_frac: float):
    """Stratified train/val split from train/, plus the untouched test/ split.

    Split indices are written to CSVs under `out_dir` so the exact split used
    for a run can be checked without re-deriving it.
    """
    rng = np.random.default_rng(seed)
    train_all, test_items = _scan(root, "train"), _scan(root, "test")
    by = collections.defaultdict(list)
    for it in train_all:
        by[it[1]].append(it)
    train_items, val_items = [], []
    for _, grp in by.items():
        idx = rng.permutation(len(grp))
        nval = max(1, int(round(len(grp) * val_frac)))
        vset = set(idx[:nval].tolist())
        for j, it in enumerate(grp):
            (val_items if j in vset else train_items).append(it)

    os.makedirs(out_dir, exist_ok=True)
    for name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        with open(os.path.join(out_dir, f"kermany_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "class"])
            for p, l in items:
                w.writerow([p, l, CLASSES[l]])

    return train_items, val_items, test_items


def stratified_subset(items, n: int, seed: int):
    """A balanced n-item subset, used as the fixed subset attacks run against."""
    rng = np.random.default_rng(seed + 777)
    by = collections.defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    per, out = n // len(by), []
    for _, grp in by.items():
        for i in rng.permutation(len(grp))[:per]:
            out.append(grp[i])
    rng.shuffle(out)
    return out


class KermanyDS(Dataset):
    def __init__(self, items, img_size: int):
        self.items = items
        self.img_size = img_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, lbl = self.items[i]
        img = Image.open(path).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        x = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        return x, lbl


def load_xy(items, img_size: int, device: str):
    """Loads a fixed item list directly into one [N,3,H,W] tensor pair, rather
    than a DataLoader -- used where the whole set is small and needs to stay
    resident (e.g. a fixed 128-image SEL/REPORT slice reused across many
    defense-training epochs)."""
    xs, ys = [], []
    for path, lbl in items:
        img = Image.open(path).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.LANCZOS)
        xs.append(torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0 * 2.0 - 1.0)
        ys.append(lbl)
    return torch.stack(xs).to(device), np.array(ys)


def make_loader(items, img_size: int, batch_size: int, num_workers: int, device: str, shuffle: bool = False):
    return DataLoader(
        KermanyDS(items, img_size),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )
