"""Saving/loading adversarial images to/from disk, one .pt per image plus a meta.json."""
import json
import os

import numpy as np
import torch


def save_dir(adv_root: str, tokenizer: str, attack: str, eps) -> str:
    d = f"{adv_root}/{tokenizer}/{attack}/eps{eps}"
    os.makedirs(d, exist_ok=True)
    return d


def already_done(adv_root: str, tokenizer: str, attack: str, eps, n_expected: int) -> bool:
    d = save_dir(adv_root, tokenizer, attack, eps)
    return os.path.isfile(f"{d}/meta.json") and len([f for f in os.listdir(d) if f.endswith(".pt")]) == n_expected


def save_batch(d: str, x_adv: torch.Tensor, items, start_idx: int) -> int:
    """Saves each image in x_adv as img_<idx>_y<label>.pt, matching `items[start_idx:]`."""
    n = x_adv.shape[0]
    for i in range(n):
        path, lbl = items[start_idx + i]
        torch.save({"adv": x_adv[i].clone(), "label": int(lbl), "src": path},
                   f"{d}/img_{start_idx + i:05d}_y{int(lbl)}.pt")
    return n


def write_meta(d: str, **fields) -> None:
    json.dump(fields, open(f"{d}/meta.json", "w"), indent=2)


def load_folder(adv_root: str, tokenizer: str, attack: str, eps, idx_subset=None):
    """idx_subset: optional list of positions into the full saved set (files are
    saved in a fixed deterministic order) -- used to pull only a SEL or REPORT
    half without needing separate cached files."""
    d = save_dir(adv_root, tokenizer, attack, eps)
    files = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    if idx_subset is not None:
        files = [files[i] for i in idx_subset]
    advs, ys = [], []
    for fn in files:
        r = torch.load(f"{d}/{fn}", map_location="cpu")
        advs.append(r["adv"])
        ys.append(r["label"])
    return torch.stack(advs), np.array(ys)
