"""APTOS 2019 (diabetic retinopathy) dataset: pre-split Kaggle mirror
(mariaherrerot/aptos2019), binarized to referable-DR vs not.

Image loading/normalization is identical to Kermany's (see
src/datasets/kermany.py, which this module reuses for that part), so only the
APTOS-specific split-file parsing lives here.
"""
import collections
import csv
import glob
import os

import pandas as pd

CLASSES = ["no_ref_DR", "referable_DR"]  # label 0, 1
POS_GRADES = {2, 3, 4}  # referable DR = positive; {1,2,3,4} would be "any DR"

SPLITS = {
    "train": ("train_1.csv", "train_images"),
    "val": ("valid.csv", "val_images"),
    "test": ("test.csv", "test_images"),
}

_ID_CANDS = ["id_code", "id", "image", "image_id", "img", "filename", "file", "name", "path"]
_LBL_CANDS = ["diagnosis", "label", "level", "grade", "class", "target"]


def to_label(grade: int) -> int:
    return int(grade in POS_GRADES)


def _pick(cols, cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _img_dir(root: str, base: str) -> str:
    """Handles the mirror's double-nesting: prefers <base>/<base> if it holds the pngs."""
    nested = os.path.join(root, base, base)
    flat = os.path.join(root, base)
    if glob.glob(f"{nested}/*.png"):
        return nested
    if glob.glob(f"{flat}/*.png"):
        return flat
    hit = glob.glob(f"{root}/{base}/**/*.png", recursive=True)
    return os.path.dirname(hit[0]) if hit else flat


def _resolve(img_dir: str, raw) -> str | None:
    stem = str(raw).strip()
    for cand in (stem, f"{stem}.png", os.path.splitext(stem)[0] + ".png"):
        p = cand if os.path.isabs(cand) else os.path.join(img_dir, os.path.basename(cand))
        if os.path.isfile(p):
            return p
    return None


def _load_split(root: str, name: str):
    csv_name, base = SPLITS[name]
    df = pd.read_csv(os.path.join(root, csv_name))
    idc, lblc = _pick(df.columns, _ID_CANDS), _pick(df.columns, _LBL_CANDS)
    assert idc and lblc, f"{csv_name}: couldn't find id/label cols in {list(df.columns)}"
    img_dir = _img_dir(root, base)
    items, miss = [], 0
    for _, r in df.iterrows():
        p = _resolve(img_dir, r[idc])
        if p is None:
            miss += 1
            continue
        items.append((p, to_label(int(r[lblc])), int(r[lblc])))
    print(f"  {name:5s}: csv={csv_name} cols=({idc},{lblc}) dir=.../{os.path.relpath(img_dir, root)} "
          f"-> {len(items)} imgs, {miss} missing")
    return items


def build_splits(root: str, out_dir: str):
    """Uses the mirror's provided train/val/test split as-is (no re-splitting)."""
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for name in ["train", "val", "test"]:
        items = _load_split(root, name)
        with open(os.path.join(out_dir, f"aptos_{name}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label", "grade", "class"])
            for p, l, g in items:
                w.writerow([p, l, g, CLASSES[l]])
        cc = collections.Counter(l for _, l, _ in items)
        gg = collections.Counter(g for _, _, g in items)
        print(f"         label={dict(cc)}  grade={dict(sorted(gg.items()))}")
        out[name] = [(p, l) for p, l, _ in items]
    return out["train"], out["val"], out["test"]
