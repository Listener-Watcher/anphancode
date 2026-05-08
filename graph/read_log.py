import os
import re
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED_RE = re.compile(r"Seed random\s*=\s*(\d+)")
FOLD_RE = re.compile(r"=+\s*Fold:\s*(\d+)\s*=+")
ACC_RE = re.compile(r"^Accuracy:\s*([0-9.]+)\s*$")
BAL_RE = re.compile(r"^Balanced Accuracy:\s*([0-9.]+)\s*$")
F1_RE = re.compile(r"^Macro-F1:\s*([0-9.]+)\s*$")


def parse_confusion_matrix(lines, start_idx):
    """
    Parse confusion matrix starting right after the line 'Confusion Matrix:'.
    Expected format like:
    [[4 0 0]
     [1 2 1]
     [0 1 2]]
    """
    mat_lines = []
    i = start_idx

    while i < len(lines):
        s = lines[i].strip()
        if s == "":
            break

        # stop if next block starts
        if s.startswith("Final validation metrics:") or s.startswith("Final test metrics:") \
           or s.startswith("TRAIN fingerprint stats:") or s.startswith("TEST  fingerprint stats:") \
           or s.startswith("========== Fold:") or s.startswith("======================================"):
            break

        if s.startswith("[") or mat_lines:
            mat_lines.append(s)
            if s.endswith("]]"):
                break
        else:
            break

        i += 1

    if not mat_lines:
        return None, start_idx

    # convert lines like "[[4 0 0]" -> rows of ints
    rows = []
    for row in mat_lines:
        row_clean = row.replace("[", " ").replace("]", " ").strip()
        if row_clean:
            vals = [int(x) for x in row_clean.split()]
            rows.append(vals)

    return rows, i


def parse_metrics_block(lines, start_idx):
    """
    Parse one metrics block after:
      Final validation metrics:
      or
      Final test metrics:
    """
    metrics = {
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "conf_matrix": None,
    }

    i = start_idx + 1
    while i < len(lines):
        s = lines[i].strip()

        if s.startswith("Final validation metrics:") or s.startswith("Final test metrics:") \
           or s.startswith("TRAIN fingerprint stats:") or s.startswith("TEST  fingerprint stats:") \
           or s.startswith("========== Fold:") or s.startswith("======================================"):
            break

        m = BAL_RE.search(s)
        if m:
            metrics["balanced_accuracy"] = float(m.group(1))
            i += 1
            continue

        m = ACC_RE.search(s)
        if m:
            metrics["accuracy"] = float(m.group(1))
            i += 1
            continue

        m = F1_RE.search(s)
        if m:
            metrics["macro_f1"] = float(m.group(1))
            i += 1
            continue

        if s.startswith("Confusion Matrix:"):
            cm, new_i = parse_confusion_matrix(lines, i + 1)
            metrics["conf_matrix"] = cm
            i = new_i + 1
            continue

        i += 1

    return metrics, i


def parse_fingerprint_line(line):
    """
    Parse:
    TRAIN fingerprint stats: {...}
    TEST  fingerprint stats: {...}
    """
    if ":" not in line:
        return None
    payload = line.split(":", 1)[1].strip()
    try:
        return ast.literal_eval(payload)
    except Exception:
        return None


def parse_log_file(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    rows = []
    current_seed = None
    current_fold = None
    current_row = None

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        m = SEED_RE.search(s)
        if m:
            current_seed = int(m.group(1))
            i += 1
            continue

        m = FOLD_RE.search(s)
        if m:
            if current_row is not None:
                rows.append(current_row)

            current_fold = int(m.group(1))
            current_row = {
                "split_seed": current_seed,
                "fold": current_fold,
                "val_accuracy": None,
                "val_balanced_accuracy": None,
                "val_macro_f1": None,
                "val_conf_matrix": None,
                "test_accuracy": None,
                "test_balanced_accuracy": None,
                "test_macro_f1": None,
                "test_conf_matrix": None,
                "train_fingerprint_stats": None,
                "test_fingerprint_stats": None,
            }
            i += 1
            continue

        if s.startswith("Final validation metrics:") and current_row is not None:
            metrics, i = parse_metrics_block(lines, i)
            current_row["val_accuracy"] = metrics["accuracy"]
            current_row["val_balanced_accuracy"] = metrics["balanced_accuracy"]
            current_row["val_macro_f1"] = metrics["macro_f1"]
            current_row["val_conf_matrix"] = metrics["conf_matrix"]
            continue

        if s.startswith("Final test metrics:") and current_row is not None:
            metrics, i = parse_metrics_block(lines, i)
            current_row["test_accuracy"] = metrics["accuracy"]
            current_row["test_balanced_accuracy"] = metrics["balanced_accuracy"]
            current_row["test_macro_f1"] = metrics["macro_f1"]
            current_row["test_conf_matrix"] = metrics["conf_matrix"]
            continue

        if s.startswith("TRAIN fingerprint stats:") and current_row is not None:
            current_row["train_fingerprint_stats"] = parse_fingerprint_line(s)
            i += 1
            continue

        if s.startswith("TEST  fingerprint stats:") and current_row is not None:
            current_row["test_fingerprint_stats"] = parse_fingerprint_line(s)
            i += 1
            continue

        i += 1

    if current_row is not None:
        rows.append(current_row)

    return rows


def save_folder_outputs(folder_path):
    folder_path = Path(folder_path)
    log_path = folder_path / "log.txt"
    if not log_path.exists():
        return None

    rows = parse_log_file(log_path)
    if len(rows) == 0:
        return None

    df = pd.DataFrame(rows)

    # save parsed fold-level table
    df_out = df.copy()
    df_out["val_conf_matrix"] = df_out["val_conf_matrix"].apply(json.dumps)
    df_out["test_conf_matrix"] = df_out["test_conf_matrix"].apply(json.dumps)
    df_out["train_fingerprint_stats"] = df_out["train_fingerprint_stats"].apply(json.dumps)
    df_out["test_fingerprint_stats"] = df_out["test_fingerprint_stats"].apply(json.dumps)

    fold_csv = folder_path / "fold_metrics_from_log.csv"
    df_out.to_csv(fold_csv, index=False)

    # save long-format fold metrics like original fold_metrics_all_seeds.csv
    long_rows = []
    for _, r in df.iterrows():
        long_rows.append({
            "split_seed": r["split_seed"],
            "fold": r["fold"],
            "split": "val",
            "accuracy": r["val_accuracy"],
            "balanced_accuracy": r["val_balanced_accuracy"],
            "macro_f1": r["val_macro_f1"],
            "conf_matrix": json.dumps(r["val_conf_matrix"]),
        })
        long_rows.append({
            "split_seed": r["split_seed"],
            "fold": r["fold"],
            "split": "test",
            "accuracy": r["test_accuracy"],
            "balanced_accuracy": r["test_balanced_accuracy"],
            "macro_f1": r["test_macro_f1"],
            "conf_matrix": json.dumps(r["test_conf_matrix"]),
        })

    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(folder_path / "fold_metrics_all_seeds.csv", index=False)

    # test summary by split seed
    test_summary_by_split = (
        long_df[long_df["split"] == "test"]
        .groupby("split_seed")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )
    test_summary_by_split.to_csv(folder_path / "test_summary_by_split_seed.csv", index=False)

    # overall summary test like original
    overall_summary = (
        long_df[long_df["split"] == "test"][["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    overall_summary.to_csv(folder_path / "overall_summary_test.csv")

    return {
        "folder": str(folder_path),
        "n_folds": len(df),
        "test_acc_mean": float(df["test_accuracy"].mean()),
        "test_bal_acc_mean": float(df["test_balanced_accuracy"].mean()),
        "test_macro_f1_mean": float(df["test_macro_f1"].mean()),
    }


def rebuild_all_from_logs(base_dir):
    base_dir = Path(base_dir)
    summaries = []

    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir():
            continue
        log_path = folder / "log.txt"
        if not log_path.exists():
            continue
        print(f"Writing: {folder / 'overall_summary_test.csv'}")
        result = save_folder_outputs(folder)
        if result is not None:
            summaries.append(result)

    if len(summaries) > 0:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(base_dir / "all_experiment_summary_from_logs.csv", index=False)
        return summary_df

    return pd.DataFrame()


if __name__ == "__main__":
    base_dir = "/home/anphan/Documents/EEG_Project/AHEAP_data/result_Arp12_nodeinput"
    summary_df = rebuild_all_from_logs(base_dir)
    print(summary_df)