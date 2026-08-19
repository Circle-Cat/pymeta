"""
make_multi_error_subset.py
==========================

Build a standalone CSV for the expert-annotated multi-error subset, plus a
matching index table, so the subset can be prompted on its own instead of
running a full 4,865-sample pass first.

Why this exists
---------------
``annotation/multi_error_subset_97.csv`` stores ``Sample_Index`` as the *positional
index into the full test split* (420 .. 4733). ``multi_error_metrics.py`` aligns a
run log to the annotations by that index, and ``multi_error_prompting.py`` logs
``Sample <n>`` using the position of the row inside the CSV it was given. So
slicing the 97 rows into a new CSV silently breaks alignment: the log would carry
0..96, the annotation table 420..4733, every row would fail to join, and the
metrics would come out as zeros *without raising an error*.

This script emits both halves of a consistent pair:

    <prefix>.csv         the 97 submissions, in annotation-file order
                         -> feed to multi_error_prompting.py --data
    <prefix>_index.csv   Sample_Index 0..96 + the same "True Label" column
                         -> feed to multi_error_metrics.py --excel
    <prefix>_map.csv     new index <-> original Sample_Index <-> labels
                         -> keep for traceability back to the paper's numbering

Example
-------
    python make_multi_error_subset.py \
        --data ./data/test.csv \
        --index ./annotation/multi_error_subset_97.csv \
        --outdir ./data

    python multi_error_prompting.py --data ./data/multi_error_subset_97.csv \
        --output ./results --model deepseek-v4-pro --temperature 0

    python multi_error_metrics.py \
        --excel ./data/multi_error_subset_97_index.csv \
        --logs ./results/outcome-*/deepseek*_classification_log_*.txt \
        --outdir ./results/metrics
"""

import argparse
import os
import re
import sys

import pandas as pd

# The 14 taxonomy names the prompting drivers map to ids.
TAXONOMY_NAMES = {
    "No error", "LogicError", "SyntaxError", "NameError", "TypeError",
    "IndentationError", "UnboundLocalError", "KeyError", "IndexError",
    "EOFError", "RecursionError", "ValueError", "TabError", "Other errors",
}

GOLD_COLUMN = "all_errortype2"

# Class 13 "Other errors" is the taxonomy's catch-all, so any exception-shaped
# name outside the 13 named classes belongs there.
EXCEPTION_SUFFIXES = ("error", "errors", "warning", "exception")


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Build the multi-error subset CSV and its aligned index table."
    )
    p.add_argument("--data", default="./data/test.csv",
                   help="Full split the annotation indices refer to (default: %(default)s)")
    p.add_argument("--index", default="./annotation/multi_error_subset_97.csv",
                   help="Annotation table with Sample_Index + True Label (default: %(default)s)")
    p.add_argument("--outdir", default="./data",
                   help="Where to write the outputs (default: %(default)s)")
    p.add_argument("--prefix", default=None,
                   help="Output basename (default: the --index basename)")
    return p.parse_args()


def annotated_names(label_field: str):
    """Split 'LogicError (#1), KeyError (#7)' into ['LogicError', 'KeyError']."""
    parts = re.split(r"[,;、]+", str(label_field))
    return [re.sub(r"\(#\s*-?\d+\s*\)", "", part).strip() for part in parts if part.strip()]


def main():
    args = parse_arguments()

    for path in (args.data, args.index):
        if not os.path.isfile(path):
            sys.exit(f"[ERROR] not found: {path}")

    full = pd.read_csv(args.data)
    index = pd.read_csv(args.index)

    if "Sample_Index" not in index.columns or "True Label" not in index.columns:
        sys.exit(f"[ERROR] {args.index} must contain 'Sample_Index' and 'True Label' columns; "
                 f"found {list(index.columns)}")

    positions = index["Sample_Index"].astype(int)
    out_of_range = positions[(positions < 0) | (positions >= len(full))]
    if len(out_of_range):
        sys.exit(f"[ERROR] {len(out_of_range)} annotation indices fall outside {args.data} "
                 f"(0..{len(full) - 1}), e.g. {out_of_range.head().tolist()}. "
                 f"Wrong split? The indices are positions in the *test* split.")

    print(f"[INFO] {args.data}: {len(full)} rows")
    print(f"[INFO] {args.index}: {len(index)} annotated samples "
          f"(Sample_Index {positions.min()}..{positions.max()})")

    # Rows in annotation-file order, so the two outputs share one row order.
    subset = full.iloc[positions.tolist()].copy()
    subset.insert(0, "orig_sample_index", positions.tolist())
    subset.reset_index(drop=True, inplace=True)

    # Integrity signal: how often the single gold label appears in the annotated
    # multi-label set. It is deliberately not required to be 100% -- gold records
    # only the first sequential error, and annotators legitimately disagree -- but
    # a very low rate means --data is the wrong file.
    if GOLD_COLUMN in subset.columns:
        norm = lambda s: str(s).lower().replace(" ", "")
        hits = sum(
            norm(gold) in {norm(n) for n in annotated_names(lab)}
            for gold, lab in zip(subset[GOLD_COLUMN], index["True Label"])
        )
        rate = hits / len(subset)
        print(f"[INFO] gold label contained in the annotated set: {hits}/{len(subset)} ({rate:.0%})")
        if rate < 0.6:
            print("[WARN] that rate is suspiciously low -- is --data the split the "
                  "annotation indices were taken from?")

        outside = sorted({g for g in subset[GOLD_COLUMN].astype(str)
                          if g not in TAXONOMY_NAMES})
        # Exception-shaped names belong to class 13 ("Other errors"), which the
        # drivers now fold them into. Anything else is genuinely unscoreable.
        folded = [g for g in outside
                  if g.lower().replace(" ", "").endswith(EXCEPTION_SUFFIXES)]
        unscoreable = [g for g in outside if g not in folded]
        if folded:
            print(f"[INFO] gold labels folded into 'Other errors' (#13): {folded}")
        if unscoreable:
            print(f"[WARN] gold labels that map to no taxonomy id and will be scored "
                  f"as -1 (never correct): {unscoreable}")
    else:
        print(f"[WARN] {args.data} has no '{GOLD_COLUMN}' column; skipping the label checks")

    prefix = args.prefix or os.path.splitext(os.path.basename(args.index))[0]
    os.makedirs(args.outdir, exist_ok=True)
    subset_path = os.path.join(args.outdir, f"{prefix}.csv")
    index_path = os.path.join(args.outdir, f"{prefix}_index.csv")
    map_path = os.path.join(args.outdir, f"{prefix}_map.csv")

    if os.path.abspath(subset_path) == os.path.abspath(args.index):
        sys.exit(f"[ERROR] refusing to overwrite the annotation file at {subset_path}; "
                 f"pass a different --outdir or --prefix")

    subset.to_csv(subset_path, index=False)

    # Re-indexed 0..N-1 so it aligns with the "Sample <n>" lines the driver writes
    # when prompted on subset_path. Row order matches subset_path exactly.
    remapped = pd.DataFrame({
        "Sample_Index": range(len(subset)),
        "True Label": index["True Label"].tolist(),
    })
    remapped.to_csv(index_path, index=False)

    pd.DataFrame({
        "Sample_Index": range(len(subset)),
        "orig_sample_index": positions.tolist(),
        "True Label": index["True Label"].tolist(),
        "gold_single_label": subset[GOLD_COLUMN].tolist() if GOLD_COLUMN in subset.columns else "",
    }).to_csv(map_path, index=False)

    print(f"\n[OK] wrote {subset_path}   ({len(subset)} rows)  -> multi_error_prompting.py --data")
    print(f"[OK] wrote {index_path}   (Sample_Index 0..{len(subset) - 1})  -> multi_error_metrics.py --excel")
    print(f"[OK] wrote {map_path}   (traceability back to the original indices)")
    print("\nNext:")
    print(f"  python multi_error_prompting.py --data {subset_path} \\")
    print(f"      --output ./results --model deepseek-v4-pro --temperature 0")
    print(f"  python multi_error_metrics.py --excel {index_path} \\")
    print(f"      --logs ./results/outcome-*/*_classification_log_*.txt --outdir ./results/metrics")


if __name__ == "__main__":
    main()
