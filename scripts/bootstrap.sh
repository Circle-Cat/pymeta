#!/usr/bin/env bash
# bootstrap.sh — prepare a fresh workspace to run PyMETA prompting experiments.
#
# Idempotent: re-running skips whatever is already in place. Downloads the data
# splits from Hugging Face (they are not in git — see data/README.md) and builds
# the aligned multi-error subset.
#
#   bash scripts/bootstrap.sh              # test split only (what the experiments need)
#   bash scripts/bootstrap.sh --with-dev   # also fetch dev.csv
#   bash scripts/bootstrap.sh --with-train # also fetch train.csv (105 MB)
#
# Then export the key for the provider you are running and launch, e.g.
#   export DEEPSEEK_API_KEY=...
#   .venv/bin/python scripts/single_error_prompting.py --data data/test.csv \
#       --output ./results/single_full --model deepseek-v4-pro \
#       --temperature 0 --max-output-tokens 32768 --concurrency 8

set -euo pipefail
cd "$(dirname "$0")/.."
HF=https://huggingface.co/datasets/CircleCat/pymeta/resolve/main

fetch() {  # fetch <name> <expected-min-bytes>
  local name=$1 min=$2
  if [ -s "data/$name" ] && [ "$(stat -c%s "data/$name")" -ge "$min" ]; then
    echo "  data/$name already present ($(du -h "data/$name" | cut -f1))"
    return
  fi
  echo "  downloading data/$name ..."
  curl -fsSL -o "data/$name" "$HF/$name"
  echo "  got data/$name ($(du -h "data/$name" | cut -f1))"
}

echo "[1/4] Python environment"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
fi
.venv/bin/pip install -q -r requirements.txt
echo "  $(.venv/bin/python -V), deps installed"

echo "[2/4] Dataset splits"
mkdir -p data
fetch test.csv 10000000
for arg in "$@"; do
  case "$arg" in
    --with-dev)   fetch dev.csv 10000000 ;;
    --with-train) fetch train.csv 90000000 ;;
  esac
done

echo "[3/4] Multi-error subset (97 samples) + aligned index table"
.venv/bin/python scripts/make_multi_error_subset.py \
    --data ./data/test.csv --index ./annotation/multi_error_subset_97.csv --outdir ./data

echo "[4/4] Sanity check"
.venv/bin/python - <<'PY'
import pandas as pd
t = pd.read_csv("data/test.csv")
s = pd.read_csv("data/multi_error_subset_97.csv")
i = pd.read_csv("data/multi_error_subset_97_index.csv")
assert len(t) == 4865, f"test split has {len(t)} rows, expected 4865"
assert len(s) == 97 and len(i) == 97, "subset/index row count mismatch"
assert list(i["Sample_Index"]) == list(range(97)), "index table is not remapped to 0..96"
print(f"  test split {len(t)} rows, subset {len(s)} rows, index remapped 0..{len(i)-1}  OK")
PY

echo
echo "Ready. Set the provider's API key and run an experiment:"
echo "  OPENAI_API_KEY | ANTHROPIC_API_KEY | GEMINI_API_KEY | DEEPSEEK_API_KEY"
