#!/usr/bin/env bash
# Clones the tokenizer source repos and fetches their pretrained checkpoints
# into third_party/. These are upstream code + weights, not part of this
# repo's own contribution, so they're not vendored in git.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
CKPT="${TP}/ckpt"
mkdir -p "$TP" "$CKPT"

clone_if_missing() {
  local url="$1" dir="$2"
  if [[ ! -d "$dir" ]]; then
    echo "cloning $url -> $dir"
    git clone -q "$url" "$dir"
  fi
}

clone_if_missing "https://github.com/FoundationVision/LlamaGen.git" "${TP}/LlamaGen"
clone_if_missing "https://github.com/Masaaki-75/meditok.git" "${TP}/meditok"
clone_if_missing "https://github.com/FoundationVision/UniTok.git" "${TP}/unitok"
clone_if_missing "https://github.com/duchenzhuang/FSQ-pytorch.git" "${TP}/FSQ-pytorch"

python3 - "$CKPT" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download

ckpt_dir = sys.argv[1]
hf_hub_download("FoundationVision/LlamaGen", "vq_ds16_c2i.pt", local_dir=ckpt_dir)
print("VQ-16 checkpoint ready.")
PYEOF

FSQ_CKPT="${CKPT}/fsq_ckpt_16k.pt"
FSQ_URL="https://cloud.tsinghua.edu.cn/f/e362caa7ca5142d49847/?dl=1"
if [[ ! -f "$FSQ_CKPT" ]]; then
  echo "downloading FSQ 16k checkpoint (can be slow)..."
  curl -L --fail -o "$FSQ_CKPT" "$FSQ_URL"
fi

echo "Third-party repos + checkpoints ready under $TP"
echo "MediTok/UniTok checkpoints are fetched lazily on first load (see src/tokenizers/meditok_unitok.py)."
