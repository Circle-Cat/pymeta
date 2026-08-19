#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_error_metrics.py  (no-MISS, Policy-A)

Computes multi-label evaluation metrics for the error-classification logs
produced by the prompting drivers (e.g. multi_error_prompting.py). For each
model log it parses the "Predicted Label:" lines, aligns them against the
ground-truth labels in an index table (CSV/Excel), and reports two modes per
model:
  - contains (any-hit): a prediction counts as correct if the gold label is
    anywhere in the predicted set.
  - top1: only the first predicted label is used.
Macro precision/recall/F1 are computed over only the true (gold) classes; a
"__MISS__" placeholder is used for misses and is never treated as a class.

Outputs (written to --outdir):
  - metrics_summary.csv          (two rows per model: contains / top1)
  - <model>_aligned_full.csv     (per-sample alignment detail for each log)

API key / environment
----------------------
    None required. This script does not call any LLM API; it only reads local
    log files and an index table.

Dependencies
------------
    pip install pandas openpyxl scikit-learn

Example usage
-------------
    python multi_error_metrics.py \
        --excel ./data/index_table.csv \
        --logs ./results/deepseek_classification_log_20250504.txt \
        --outdir ./metrics_out

    # --excel accepts .csv or .xlsx (use --sheet to select an Excel sheet).
    # --logs accepts one or more log files.

Original design notes (kept for reference, in Chinese):
- contains = any-hit：预测集合里只要包含 gold（任一）即记命中
- contains 的 macro P/R/F1：命中 → 预测记为 gold；未命中 → 预测记为 "__MISS__"
  再用 sklearn 在“仅真实类（gold 出现过的类）”上做宏平均（不把 MISS 当类）
- top1 的 macro：预测类 = top1（无则 "__MISS__"），同样仅在真实类上做宏平均
- 输出：metrics_summary.csv（每模型两行：contains / top1），以及对齐明细 *_aligned_full.csv
"""

import os
import re
import argparse
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------- 正则与映射 ----------
SAMPLE_SPLIT_PATTERN = re.compile(r"Sample\s+(\d+):", re.IGNORECASE)
PRED_LINE_PATTERN    = re.compile(r"^\s*Predicted\s*Label:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
LABEL_TOKEN_PATTERN  = re.compile(r"\s*,\s*")
LABEL_ID_PATTERN     = re.compile(r"^(?P<name>.*?)(?:\s*\(#(?P<id>\d+)\))?$")

STD_LABELS = {
    "No error": "0",
    "LogicError": "1",
    "SyntaxError": "2",
    "NameError": "3",
    "TypeError": "4",
    "IndentationError": "5",
    "UnboundLocalError": "6",
    "KeyError": "7",
    "IndexError": "8",
    "EOFError": "9",
    "RecursionError": "10",
    "ValueError": "11",
    "TabError": "12",
    "Other errors": "13",
}
STD_LABELS_LOWER = {k.lower(): v for k, v in STD_LABELS.items()}
# Space-insensitive variant: the prompt asks the model for "Syntax Error (#2)",
# so a reply that omits the id still has to resolve to an id here. Without this,
# a correct name-only answer is silently counted as a miss.
STD_LABELS_LOOSE = {k.lower().replace(" ", ""): v for k, v in STD_LABELS.items()}


# ---------- 解析工具 ----------
def normalize_to_id(token: str) -> str:
    if token is None: return ""
    t = str(token).strip()
    if not t: return ""
    if t.isdigit(): return t
    if t.lower() in STD_LABELS_LOWER: return STD_LABELS_LOWER[t.lower()]
    loose = t.lower().replace(" ", "")
    if loose in STD_LABELS_LOOSE: return STD_LABELS_LOOSE[loose]
    # Class 13 is the catch-all ("Other errors": AttributeError, RuntimeError,
    # ZeroDivisionError, ModuleNotFoundError, ... "etc."). Any other
    # exception-shaped name therefore belongs to 13 rather than to nothing --
    # otherwise a correct "AttributeError" prediction is scored as a miss.
    if loose.endswith(("error", "errors", "warning", "exception")): return "13"
    return t

def parse_label_list(label_str: str) -> Tuple[str, List[str], List[str]]:
    raw = (label_str or "").strip()
    if not raw: return raw, [], []
    parts = LABEL_TOKEN_PATTERN.split(raw)
    names, ids = [], []
    for p in parts:
        p = p.strip()
        if not p: continue
        m = LABEL_ID_PATTERN.match(p)
        if not m:
            names.append(p); continue
        name = (m.group("name") or "").strip()
        id_  = m.group("id")
        if id_:
            ids.append(id_.strip())
        if name:
            names.append(name)
        if (not name) and (not id_) and p.isdigit():
            ids.append(p)
    return raw, names, ids

def parse_log_file(filepath: str) -> Dict[int, str]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    samples = list(SAMPLE_SPLIT_PATTERN.finditer(content))
    bounds = [(m.start(), m.end(), int(m.group(1))) for m in samples]
    bounds.append((len(content), len(content), -1))
    result = {}
    for i in range(len(bounds) - 1):
        start = bounds[i][1]; end = bounds[i+1][0]; sid = bounds[i][2]
        block = content[start:end]
        preds = list(PRED_LINE_PATTERN.finditer(block))
        if preds:
            result[sid] = preds[-1].group(1).strip()
    return result

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip() for c in df.columns})
    col_map = {}
    for c in df.columns:
        lc = c.lower().replace(" ", "").replace("_", "")
        if lc in ("sampleindex","sample_idx","sampleid","sample"):
            col_map[c] = "Sample_Index"
        elif lc in ("truelabel","goldlabel","label"):
            col_map[c] = "True Label"
    df = df.rename(columns=col_map)
    if "Sample_Index" not in df.columns or "True Label" not in df.columns:
        raise ValueError("索引表需包含 'Sample_Index' 与 'True Label'")
    return df[["Sample_Index","True Label"]].copy()

def read_index(path: str, sheet: str|int|None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=(sheet if sheet is not None else 0))
    return ensure_cols(df)


# ---------- 指标（no-MISS, Policy-A） ----------
def macro_prf_only_true_classes(y_true: List[str], y_pred: List[str]) -> Tuple[float,float,float]:
    """在仅 '真实类集合' 上做宏平均；y_pred 可含 '__MISS__' 占位，但不会纳入类别集合。"""
    classes = sorted(set(y_true))                  # 真实出现过的类
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average="macro", zero_division=0
    )
    return float(p), float(r), float(f1)

def compute_contains_metrics(y_true_ids: List[str], y_pred_id_sets: List[List[str]]) -> Dict[str,float]:
    # any-hit accuracy
    acc = accuracy_score(
        y_true_ids,
        [t if (t and preds and t in preds) else "__MISS__" for t, preds in zip(y_true_ids, y_pred_id_sets)]
    )
    # Policy-A 的宏平均：命中→预测当 gold；未命中→"__MISS__"（不会被计入类别集合）
    y_pred_for_macro = [
        t if (t and preds and t in preds) else "__MISS__"
        for t, preds in zip(y_true_ids, y_pred_id_sets)
    ]
    p, r, f1 = macro_prf_only_true_classes(y_true_ids, y_pred_for_macro)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def compute_top1_metrics(y_true_ids: List[str], y_pred_top1_ids: List[str]) -> Dict[str,float]:
    acc = accuracy_score(y_true_ids, [p if p else "__MISS__" for p in y_pred_top1_ids])
    p, r, f1 = macro_prf_only_true_classes(y_true_ids, [p if p else "__MISS__" for p in y_pred_top1_ids])
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="no-MISS, Policy-A contains/top1 metrics")
    ap.add_argument("--excel",  required=True)
    ap.add_argument("--logs",   nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--sheet",  default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 索引与真实ID（单标签：取第一个）
    df = read_index(args.excel, args.sheet)
    true_ids, true_names, true_raws = [], [], []
    for s in df["True Label"].astype(str).tolist():
        raw, names, ids = parse_label_list(s)
        if ids:
            tid = ids[0]
        elif names:
            tid = normalize_to_id(names[0])
        else:
            tid = normalize_to_id(raw)
        true_raws.append(raw)
        true_names.append(names[0] if names else "")
        true_ids.append(normalize_to_id(tid))

    metrics_rows = []
    for log_path in args.logs:
        if not os.path.isfile(log_path):
            print(f"[WARN] not found: {log_path}"); continue
        model = os.path.splitext(os.path.basename(log_path))[0]
        print(f"[INFO] parsing: {model}")

        s2pred = parse_log_file(log_path)

        out = df.copy()
        out["Predicted Label"] = out["Sample_Index"].map(s2pred).fillna("")

        # A Sample_Index that matches nothing in the log yields an empty prediction
        # and is scored as a miss, so a bad join looks exactly like a bad model.
        # The usual cause is an index table whose indices are positions in the full
        # split while the log came from a run over an extracted subset (or the
        # reverse) -- see make_multi_error_subset.py.
        joined = int((out["Predicted Label"] != "").sum())
        if joined == 0:
            print(f"[WARN] {model}: none of the {len(out)} index rows matched a "
                  f"'Sample <n>' block in the log (log has "
                  f"{len(s2pred)} samples, indices "
                  f"{min(s2pred, default='-')}..{max(s2pred, default='-')}; index table "
                  f"{out['Sample_Index'].min()}..{out['Sample_Index'].max()}). "
                  f"All metrics below will be 0 for join reasons, not model reasons.")
        elif joined < len(out):
            print(f"[WARN] {model}: only {joined}/{len(out)} index rows matched a log "
                  f"sample; the rest are scored as misses.")

        pred_ids_list, pred_top1_ids, pred_names_list = [], [], []
        for s in out["Predicted Label"]:
            raw, names, ids = parse_label_list(s)
            if not ids and names:
                ids = [normalize_to_id(n) for n in names if n]
            ids_std = [normalize_to_id(x) for x in ids if x]
            pred_ids_list.append(ids_std)
            pred_top1_ids.append(ids_std[0] if ids_std else "")
            pred_names_list.append("|".join(names) if names else "")

        # 明细
        out["True_ID_norm"]  = true_ids
        out["True_Name_norm"] = true_names
        out["True_Raw"]       = true_raws
        out["Pred_IDs"]       = ["|".join(x) for x in pred_ids_list]
        out["Pred_Top1_ID"]   = pred_top1_ids
        out["Pred_Names"]     = pred_names_list

        # contains（any-hit，Policy-A 宏平均）
        m_contains = compute_contains_metrics(true_ids, pred_ids_list)
        metrics_rows.append({"model": model, "mode": "contains", **m_contains})

        # top1（宏平均仅在真实类）
        m_top1 = compute_top1_metrics(true_ids, pred_top1_ids)
        metrics_rows.append({"model": model, "mode": "top1", **m_top1})

        out.to_csv(os.path.join(args.outdir, f"{model}_aligned_full.csv"), index=False)

    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(os.path.join(args.outdir, "metrics_summary.csv"), index=False)
        print(f"[OK] saved metrics_summary.csv at {args.outdir}")
    else:
        print("[WARN] no metrics produced; check paths and log format")

if __name__ == "__main__":
    main()
