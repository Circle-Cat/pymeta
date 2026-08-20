"""
export_run_summary.py
=====================

Distil one experiment run into a handful of small files that are safe to commit,
so several workspaces can each run a different model and the results can be
collected in one repository.

Why not commit the raw outputs
------------------------------
``sample_index_mapping_*.csv`` re-serialises ``question`` / ``exceptedAnswer`` /
``studentAnswer`` for every sample, which is ~13 MB per full run and duplicates
the Hugging Face dataset. The classification log carries the full reasoning
traces. Both stay out of git (see .gitignore); this script writes only what the
paper's tables and the cross-model analysis actually read.

Output layout — one directory per run, keyed by provider and model, so runs
committed from different workspaces never touch the same paths:

    results_summary/<run_tag>/metrics.json         run config, token usage, metrics
    results_summary/<run_tag>/predictions.csv      per-sample gold vs predicted (slim)
    results_summary/<run_tag>/predicted_sets.csv   multi-error runs only: label sets

Usage
-----
    python export_run_summary.py --run ./results/single_full
    python export_run_summary.py --run ./results/multi_97 --outdir ./results_summary
"""

import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

SLIM_COLUMNS = ["Sample_Index", "true_label", "true_label_num",
                "predicted_label_num", "predicted_label_name"]


def parse_arguments():
    p = argparse.ArgumentParser(description="Export a committable summary of one run.")
    p.add_argument("--run", required=True,
                   help="Run directory: either an outcome-* folder or the --output dir "
                        "that contains them (the newest is used)")
    p.add_argument("--outdir", default="./results_summary",
                   help="Where to write the summary (default: %(default)s)")
    return p.parse_args()


def resolve_outcome(path):
    """Accept either an outcome-* folder or the parent --output directory."""
    if os.path.basename(path.rstrip("/")).startswith("outcome-"):
        return path
    folders = glob.glob(os.path.join(path, "outcome-*"))
    if not folders:
        sys.exit(f"[ERROR] no outcome-* folder under {path}")
    return max(folders, key=os.path.getctime)


def only(pattern, what):
    hits = sorted(glob.glob(pattern))
    if not hits:
        sys.exit(f"[ERROR] no {what} matching {pattern}")
    return hits[-1]


def predicted_label_lines(log_path):
    """Sample index -> the model's own 'Predicted Label:' text, for multi-label runs.

    The driver writes its parsed line first and the raw reply after it, so the
    last match in a sample block is what the model actually emitted.
    """
    text = open(log_path, encoding="utf-8").read()
    out = {}
    for m in re.finditer(r"^\[.*?\] Sample (\d+):(.*?)(?=^\[|\Z)", text, flags=re.M | re.S):
        preds = re.findall(r"^\s*Predicted Label:\s*(.+?)\s*$", m.group(2), flags=re.M)
        if preds:
            out[int(m.group(1))] = preds[-1]
    return out


def main():
    args = parse_arguments()
    outcome = resolve_outcome(args.run)
    print(f"[INFO] run: {outcome}")

    metrics_path = only(os.path.join(outcome, "*_test_results_*.json"), "results JSON")
    mapping_path = only(os.path.join(outcome, "sample_index_mapping_*.csv"), "sample mapping")
    log_path = only(os.path.join(outcome, "*_classification_log_*.txt"), "classification log")

    metrics = json.load(open(metrics_path))
    provider = metrics.get("provider", "unknown")
    model = metrics.get("model", "unknown")
    run_tag = "".join(c if c.isalnum() else "_" for c in f"{provider}-{model}").strip("_")

    # The task has to be part of the path: one model produces both a single-error
    # and a multi-error run, and keying on the model alone made the second export
    # silently overwrite the first. Runs recorded before the drivers wrote a
    # "task" field are inferred from which macro-F1 variant they carry.
    task = metrics.get("task")
    if not task:
        task = "multi_error" if "macro_f1_top1" in metrics else "single_error"
        print(f"[INFO] run predates the 'task' field; inferred task={task}")

    if not metrics.get("is_complete"):
        print(f"[WARN] this run is not marked complete "
              f"({metrics.get('total_processed_samples')}/"
              f"{metrics.get('total_samples_in_dataset')} samples) — exporting anyway")

    dest = os.path.join(args.outdir, task, run_tag)
    os.makedirs(dest, exist_ok=True)

    with open(os.path.join(dest, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    df = pd.read_csv(mapping_path)
    missing = [c for c in SLIM_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"[ERROR] {mapping_path} is missing {missing}")
    slim = df[SLIM_COLUMNS]
    slim.to_csv(os.path.join(dest, "predictions.csv"), index=False)

    sets = predicted_label_lines(log_path)
    multi = any("," in v for v in sets.values())
    if multi:
        pd.DataFrame({
            "Sample_Index": sorted(sets),
            "predicted_label_raw": [sets[k] for k in sorted(sets)],
            "predicted_ids": ["|".join(re.findall(r"\(#(\d+)\)", sets[k])) for k in sorted(sets)],
        }).to_csv(os.path.join(dest, "predicted_sets.csv"), index=False)

    print(f"[OK] {dest}")
    for name in sorted(os.listdir(dest)):
        size = os.path.getsize(os.path.join(dest, name))
        print(f"       {name:<22} {size / 1024:>8.1f} KB")
    saved = os.path.getsize(mapping_path) / 1024
    kept = sum(os.path.getsize(os.path.join(dest, n)) for n in os.listdir(dest)) / 1024
    print(f"[INFO] raw mapping CSV was {saved / 1024:.1f} MB; summary is {kept / 1024:.2f} MB")


if __name__ == "__main__":
    main()
