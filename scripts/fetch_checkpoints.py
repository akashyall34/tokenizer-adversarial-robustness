#!/usr/bin/env python3
"""Reassemble the one checkpoint that ships split.

Every checkpoint in `ckpts/` is committed to the repo directly, except
`ckpts/classifiers/effnet_b5_aptos.pt` (~114 MB), which exceeds GitHub's 100 MB
single-file limit and is therefore stored as `.part00` / `.part01`. Run this
once after cloning to concatenate them back:

    python scripts/fetch_checkpoints.py

Idempotent: does nothing if the assembled file already exists and matches.
"""
import hashlib
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(ROOT, "ckpts", "classifiers", "effnet_b5_aptos.pt")
PARTS = [TARGET + ".part00", TARGET + ".part01"]
EXPECT_BYTES = 114410547
EXPECT_SHA256 = "d55c4007859276a4948643c35434a86ae6bbfca0e51405d4567cc60ea2866263"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if os.path.isfile(TARGET) and os.path.getsize(TARGET) == EXPECT_BYTES and _sha256(TARGET) == EXPECT_SHA256:
        print(f"{os.path.relpath(TARGET, ROOT)} already assembled and verified")
        return

    missing = [p for p in PARTS if not os.path.isfile(p)]
    if missing:
        raise SystemExit("missing part file(s): " + ", ".join(os.path.relpath(p, ROOT) for p in missing))

    with open(TARGET, "wb") as out:
        for p in PARTS:
            with open(p, "rb") as f:
                out.write(f.read())

    n, digest = os.path.getsize(TARGET), _sha256(TARGET)
    if n != EXPECT_BYTES or digest != EXPECT_SHA256:
        os.remove(TARGET)
        raise SystemExit(f"reassembled file failed verification (got {n} bytes, sha256 {digest})")
    print(f"wrote {os.path.relpath(TARGET, ROOT)} ({n} bytes, sha256 ok)")


if __name__ == "__main__":
    main()
