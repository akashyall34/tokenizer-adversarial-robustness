"""Merges two checkpoints that were fine-tuned on disjoint parameter subsets
of the same base model (e.g. one run only unfroze the encoder, another only
unfroze the decoder) into a single combined state dict.

This is a state-dict merge, not a joint training run -- valid only because
the two source runs never touched overlapping parameters and neither moved
whatever stayed frozen in both (verified below, not assumed).
"""
import torch


def merge_state_dicts(sd_a: dict, sd_b: dict, a_prefixes: tuple, b_prefixes: tuple,
                       diagnostic_only_suffixes: tuple = ()) -> dict:
    """Keys starting with a_prefixes come from sd_a, b_prefixes from sd_b,
    everything else must be identical in both (frozen in both source runs) --
    raises if that assumption doesn't actually hold, rather than silently
    picking one side.

    diagnostic_only_suffixes: keys ending in one of these are skipped from
    the identity check and taken from sd_a without comparison -- for buffers
    that legitimately diverge between any two checkpoints even from the same
    starting weights (e.g. an EMA/usage-counter stat updated on every forward
    pass, including plain eval-mode inference, that never feeds back into the
    model's actual output). Only use this for buffers you've confirmed are
    genuinely inert, not for anything that could silently mask a real
    merge-assumption violation.
    """
    merged = {}
    n_from_a, n_from_b, n_shared, n_diagnostic_skipped = 0, 0, 0, 0
    for key in sd_a.keys():
        if key.startswith(a_prefixes):
            merged[key] = sd_a[key]
            n_from_a += 1
        elif key.startswith(b_prefixes):
            merged[key] = sd_b[key]
            n_from_b += 1
        elif key.endswith(diagnostic_only_suffixes):
            merged[key] = sd_a[key]
            n_diagnostic_skipped += 1
        else:
            if not torch.equal(sd_a[key], sd_b[key]):
                raise RuntimeError(
                    f"key '{key}' expected frozen/identical in both checkpoints but differs -- "
                    f"merge assumption violated, do not trust this combined checkpoint")
            merged[key] = sd_a[key]
            n_shared += 1
    print(f"merge: {n_from_a} keys <- a, {n_from_b} keys <- b, {n_shared} shared/frozen keys verified identical, "
          f"{n_diagnostic_skipped} diagnostic-only keys skipped from the identity check and kept from a")
    return merged
