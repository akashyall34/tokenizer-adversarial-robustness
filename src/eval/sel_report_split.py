"""Splits the fixed attack subset (e.g. the 256-image set attacks run
against) into two halves: SEL for choosing any hyperparameter (fine-tuning
itself, sigma, spatial_weight, which SQS variant wins, ...), REPORT for the
final numbers. Nothing that ends up in a reported table may have also been
used to pick its own hyperparameters or train the model being reported on.

Stratified by label so both halves keep the same class balance as the full set.
"""
import collections

import numpy as np


def stratified_half_split(items, seed: int):
    rng = np.random.default_rng(seed)
    by_label = collections.defaultdict(list)
    for i, (_, lbl) in enumerate(items):
        by_label[lbl].append(i)
    sel_idx, report_idx = [], []
    for lbl, idxs in by_label.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        half = len(idxs) // 2
        sel_idx.extend(idxs[:half])
        report_idx.extend(idxs[half:])
    return sorted(sel_idx), sorted(report_idx)
