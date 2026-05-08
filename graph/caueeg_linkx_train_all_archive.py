"""
caueeg_linkx_train_all.py

Unified CAUEEG graph training entry point for three approaches:

    --training_approach mil
    --training_approach segment_k
    --training_approach segment_all

Design goal
-----------
Keep data loading, official split, H5 reuse/building, graph construction,
feature families, connectivity/topology, seeds, and subject-level evaluation
aligned across MIL and segment-training baselines.

Main difference after graph construction:
    mil         : segment graphs -> subject bag -> MIL pooling -> subject loss
    segment_k   : sample k segment graphs per subject per epoch -> segment loss,
                  validate/test by subject-level probability aggregation
    segment_all : use all selected training segment graphs -> segment loss,
                  validate/test by subject-level probability aggregation

This file intentionally depends on your current project modules:
    caueeg_loader_min.py
    master_builder.py
    mil_full_std.py
    mil_utils.py
    config.py  (optional, for MONOFIXEDGES)

Put this file in the same folder as those project modules, then run it.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from caueeg_loader_min import load_caueeg_task_datasets
from master_builder import build_master_eeg_dataset
from mil_full_std import SubjectMILClassifier, fit_mil_baseline, load_h5_payload_for_subjects

try:
    from mil_full_std import EarlyStopping as TopKEarlyStopping
except Exception:  # pragma: no cover
    TopKEarlyStopping = None

from mil_utils import (
    LabelAwareSubjectBagDataset,
    SubjectBagGraphDataset,
    build_graphs_from_payload,
    build_graphs_from_payload_multiband,
    collate_subject_bags,
    collate_subject_bags_multiband,
    collect_subject_embeddings,
    evaluate,
)

try:
    from utils_all import set_global_seed
except Exception:  # fallback for standalone debugging
    def set_global_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# CAUEEG constants
# ---------------------------------------------------------
CAUEEG_EEG19 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4",
    "T6", "FZ", "CZ", "PZ",
]

SFREQ = 200.0
CROP_LEN = 2000      # 10 sec at 200 Hz
LATENCY = 2000       # skip first 10 sec like CEEDNet
OVERLAP = 0.5
STEP = int(CROP_LEN * (1.0 - OVERLAP))

EdgeEndpoint = Union[int, str]
EdgeSpec = Sequence[Tuple[EdgeEndpoint, EdgeEndpoint]]


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------
def _call_with_supported_kwargs(fn, /, *args, **kwargs):
    """Call fn while dropping keyword args not supported by its current signature."""
    sig = inspect.signature(fn)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*args, **kwargs)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return fn(*args, **filtered)


def _filter_kwargs_for_class(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in allowed}


def parse_optional_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"", "none", "null", "-1"}:
        return None
    return int(s)


def parse_bool(x: Union[str, bool, int]) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def make_jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [make_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x


def normalize_summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    row = make_jsonable(row)
    if isinstance(row.get("feature_families"), list):
        row["feature_families"] = ",".join(map(str, row["feature_families"]))
    if "confusion_matrix" in row:
        row["confusion_matrix_json"] = json.dumps(row["confusion_matrix"])
        del row["confusion_matrix"]
    return row


def save_summary_metrics_csv(summary_rows: List[Dict[str, Any]], csv_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame([normalize_summary_row(r) for r in summary_rows])
    df.to_csv(csv_path, index=False)
    return df


def save_history_csv(history: Sequence[Dict[str, Any]], csv_path: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df = pd.DataFrame(list(history))
    df.to_csv(csv_path, index=False)
    return df


def save_seed_aggregation(summary_rows: List[Dict[str, Any]], output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    rows = [normalize_summary_row(r) for r in summary_rows]
    df = pd.DataFrame(rows)
    raw_path = os.path.join(output_dir, "all_seed_results.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "macro_f1"] if c in df.columns]
    variant_cols = [
        "encoder_type", "training_approach", "mil_pool_type", "feature_families",
        "topology", "connectivity_metric", "connectivity_band", "edge_mode",
        "segment_selection_strategy", "level", "base_k", "batch_size", "epochs",
        "patience", "start_epoch", "lr", "dropout", "weight_decay", "graph_emb_dim",
        "attn_dim", "seed",
    ]
    variant_cols = [c for c in variant_cols if c in df.columns and c != "seed"]

    if len(metric_cols) > 0 and len(variant_cols) > 0:
        agg = df.groupby(variant_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
        agg.columns = [c[0] if c[1] == "" else f"{c[0]}_{c[1]}" for c in agg.columns]
        for m in metric_cols:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in agg.columns and std_col in agg.columns:
                agg[f"{m}_mean_std"] = agg.apply(
                    lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}" if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
                    axis=1,
                )
    else:
        agg = pd.DataFrame()

    agg_path = os.path.join(output_dir, "aggregate_seed_results.csv")
    agg.to_csv(agg_path, index=False)

    if "confusion_matrix_json" in df.columns:
        cms = []
        for s in df["confusion_matrix_json"].dropna():
            cms.append(np.asarray(json.loads(s), dtype=float))
        if len(cms) > 0:
            cm_stack = np.stack(cms, axis=0)
            cm_info = {
                "num_seeds": int(len(cms)),
                "confusion_matrix_sum": cm_stack.sum(axis=0).astype(int).tolist(),
                "confusion_matrix_mean": cm_stack.mean(axis=0).tolist(),
                "confusion_matrix_std": cm_stack.std(axis=0, ddof=1).tolist() if len(cms) > 1 else np.zeros_like(cm_stack[0]).tolist(),
            }
            with open(os.path.join(output_dir, "aggregate_confusion_matrix.json"), "w") as f:
                json.dump(cm_info, f, indent=2)

    print(f"Saved per-seed results: {raw_path}")
    print(f"Saved aggregate results: {agg_path}")
    return df, agg


# ---------------------------------------------------------
# Fixed edge / split helpers
# ---------------------------------------------------------
def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set[Tuple[int, int]]:
    if fixed_edges is None:
        return set()

    fixed_pairs: set[Tuple[int, int]] = set()
    name_to_idx = None

    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(f"channel_names has length {len(channel_names)} but n_channels={n_channels}")
        name_to_idx = {str(name): i for i, name in enumerate(channel_names)}

    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            if name_to_idx is None:
                raise ValueError("fixed_edges contains channel names, but channel_names was not provided.")
            if str(u) not in name_to_idx or str(v) not in name_to_idx:
                continue
            i, j = name_to_idx[str(u)], name_to_idx[str(v)]
        else:
            i, j = int(u), int(v)

        if i == j:
            continue
        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(u, v)} is out of range for {n_channels} nodes.")

        fixed_pairs.add(tuple(sorted((i, j))))

    return fixed_pairs


def make_split_subject_id(serial: str, split_name: str, use_split_prefix: bool = True) -> str:
    serial = str(serial)
    if not use_split_prefix:
        return serial
    if serial.startswith(("train_", "val_", "test_")):
        return serial
    return f"{split_name}_{serial}"


def _bare_serial(sid: str) -> str:
    sid = str(sid)
    for p in ("train_", "val_", "test_"):
        if sid.startswith(p):
            return sid[len(p):]
    return sid


def filter_bad_ids(records: List[Dict[str, Any]], ids: List[str], bad_ids: Optional[Iterable[str]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if bad_ids is None:
        return records, ids
    bad = {str(x) for x in bad_ids}

    def keep(sid: str) -> bool:
        sid = str(sid)
        return sid not in bad and _bare_serial(sid) not in bad

    records_f = [r for r in records if keep(r["subject_id"])]
    ids_f = [sid for sid in ids if keep(sid)]
    return records_f, ids_f


# ---------------------------------------------------------
# CAUEEG -> H5 subject records
# ---------------------------------------------------------
def segment_recording(
    signal: np.ndarray,
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
) -> Tuple[List[np.ndarray], List[int]]:
    x = np.asarray(signal, dtype=np.float32)
    x = x[:19]  # drop EKG + photic
    total_len = x.shape[-1]
    starts = list(range(int(latency), total_len - int(crop_len) + 1, int(step)))
    windows = [x[:, s:s + int(crop_len)].astype(np.float32, copy=False) for s in starts]
    return windows, starts


def dataset_to_subject_records(
    dataset,
    split_name: str,
    use_split_prefix: bool = True,
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
    limit: Optional[int] = None,
    bad_ids: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    records: List[Dict[str, Any]] = []
    subject_ids: List[str] = []
    bad = set(str(x) for x in bad_ids) if bad_ids is not None else set()

    for sample in dataset:
        serial = str(sample["serial"])
        subject_id = make_split_subject_id(serial, split_name, use_split_prefix=use_split_prefix)

        if serial in bad or subject_id in bad:
            continue

        signal = sample["signal"]
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal, crop_len=crop_len, step=step, latency=latency)
        if len(windows) == 0:
            continue

        rec = {
            "subject_id": subject_id,
            "label": label,
            "class_id": label,
            "sampling_rate": SFREQ,
            "channel_names": CAUEEG_EEG19,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": list(range(len(windows))),
            "recording_info": {
                "serial": serial,
                "age": age,
                "split_name": split_name,
            },
        }
        records.append(rec)
        subject_ids.append(subject_id)

        if limit is not None and len(subject_ids) >= int(limit):
            break

    return records, subject_ids


# ---------------------------------------------------------
# Optional manifest-based segment selection
# ---------------------------------------------------------
def graph_key(g: Data) -> Tuple[str, int]:
    return str(g.subject_id), int(getattr(g, "segment_id", -1))


def load_cleancluster_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"subject_id", "segment_id", "keep_clean"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CleanCluster manifest missing columns: {missing}")
    if df["keep_clean"].dtype != bool:
        df["keep_clean"] = df["keep_clean"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    return df


def load_global_cluster_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"subject_id", "segment_id", "global_cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Global cluster manifest missing columns: {missing}")
    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    if "split" not in df.columns:
        df["split"] = df["subject_id"].str.extract(r"^(train|val|test)_", expand=False).fillna("unknown")
    if "keep_clean" in df.columns and df["keep_clean"].dtype != bool:
        df["keep_clean"] = df["keep_clean"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return df


def filter_graphs_by_manifest_keep_clean(graphs: Sequence[Data], manifest_df: pd.DataFrame) -> List[Data]:
    clean_keys = set(
        manifest_df.loc[manifest_df["keep_clean"], ["subject_id", "segment_id"]]
        .itertuples(index=False, name=None)
    )
    out = [g for g in graphs if graph_key(g) in clean_keys]
    if len(out) == 0:
        raise RuntimeError("No graphs remain after CleanCluster filtering.")
    return out


def _sample_rows(
    df: pd.DataFrame,
    *,
    n: int,
    rng: np.random.Generator,
    replace: bool,
    weight_col: Optional[str] = None,
) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    p = None
    if weight_col is not None and weight_col in df.columns:
        w = df[weight_col].to_numpy(dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 1e-8, None)
        p = w / w.sum()
    n_eff = n if replace else min(n, len(df))
    idx = rng.choice(np.arange(len(df)), size=n_eff, replace=replace, p=p)
    return df.iloc[idx].copy()


def select_global_cluster_proportional_random_graphs_from_manifest(
    graphs: Sequence[Data],
    manifest_df: pd.DataFrame,
    *,
    k: int,
    seed: int,
    split_name: Optional[str] = "train",
    cluster_col: str = "global_cluster_id",
    cluster_weight_alpha: float = 0.5,
    uniform_cluster_mix: float = 0.10,
    max_cluster_fraction: Optional[float] = 0.60,
    use_keep_clean_if_available: bool = False,
    fill_with_replacement: bool = True,
    save_selection_path: Optional[str] = None,
) -> Tuple[List[Data], pd.DataFrame]:
    if k is None or int(k) <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = int(k)
    rng = np.random.default_rng(seed)
    graph_lookup = {(str(g.subject_id), int(getattr(g, "segment_id", -1))): g for g in graphs}
    df = manifest_df.copy()

    if split_name is not None and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == str(split_name).lower()].copy()
    if use_keep_clean_if_available and "keep_clean" in df.columns:
        df = df[df["keep_clean"]].copy()

    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df = df[df.apply(lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup, axis=1)].copy()
    if len(df) == 0:
        raise RuntimeError("No global-cluster manifest rows match the provided graphs.")

    selected_graphs: List[Data] = []
    selected_rows: List[Dict[str, Any]] = []

    for sid, sdf in df.groupby("subject_id"):
        sdf = sdf.drop_duplicates(subset=["subject_id", "segment_id"]).copy()
        full_cluster_sizes = sdf[cluster_col].value_counts().sort_index()
        cluster_ids = full_cluster_sizes.index.to_numpy(dtype=int)
        counts = full_cluster_sizes.to_numpy(dtype=np.float64)
        weights = np.power(counts, float(cluster_weight_alpha))
        weights = np.clip(weights, 1e-8, None)
        weights = weights / weights.sum()
        if uniform_cluster_mix > 0:
            mix = max(0.0, min(1.0, float(uniform_cluster_mix)))
            weights = (1.0 - mix) * weights + mix * (np.ones_like(weights) / len(weights))
            weights = weights / weights.sum()

        allocation = {int(c): 0 for c in cluster_ids}
        max_per_cluster = None if max_cluster_fraction is None else max(1, int(math.ceil(float(max_cluster_fraction) * k)))

        for _ in range(k):
            available = []
            available_w = []
            for c, w in zip(cluster_ids, weights):
                c = int(c)
                unique_ok = allocation[c] < int(full_cluster_sizes.loc[c])
                cap_ok = True if max_per_cluster is None else allocation[c] < max_per_cluster
                if unique_ok and cap_ok:
                    available.append(c)
                    available_w.append(float(w))
            if len(available) == 0:
                for c, w in zip(cluster_ids, weights):
                    c = int(c)
                    if allocation[c] < int(full_cluster_sizes.loc[c]):
                        available.append(c)
                        available_w.append(float(w))
            if len(available) == 0:
                break
            available_w = np.asarray(available_w, dtype=np.float64)
            available_w = available_w / available_w.sum()
            chosen_c = int(rng.choice(np.asarray(available), p=available_w))
            allocation[chosen_c] += 1

        chosen_parts = []
        for c, n_pick in allocation.items():
            if n_pick <= 0:
                continue
            cdf = sdf[sdf[cluster_col] == int(c)].copy()
            chosen_parts.append(_sample_rows(cdf, n=n_pick, rng=rng, replace=False))
        chosen_df = pd.concat(chosen_parts, ignore_index=True) if chosen_parts else sdf.iloc[0:0].copy()
        chosen_df = chosen_df.drop_duplicates(subset=["subject_id", "segment_id"]).copy()

        if len(chosen_df) < k:
            chosen_keys = set(zip(chosen_df["subject_id"].astype(str), chosen_df["segment_id"].astype(int)))
            remaining = sdf[~sdf.apply(lambda r: (str(r["subject_id"]), int(r["segment_id"])) in chosen_keys, axis=1)].copy()
            if len(remaining) > 0:
                fill_df = _sample_rows(remaining, n=min(k - len(chosen_df), len(remaining)), rng=rng, replace=False)
                chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        if len(chosen_df) < k and fill_with_replacement:
            fill_df = _sample_rows(sdf, n=k - len(chosen_df), rng=rng, replace=True)
            chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        chosen_df = chosen_df.head(k).copy()
        chosen_df["selected_rank"] = np.arange(1, len(chosen_df) + 1)
        chosen_df["selection_strategy"] = "global_cluster_proportional_random_k"
        selected_counts = chosen_df[cluster_col].value_counts().to_dict()

        for _, row in chosen_df.iterrows():
            key = (str(row["subject_id"]), int(row["segment_id"]))
            if key not in graph_lookup:
                continue
            selected_graphs.append(copy.copy(graph_lookup[key]))
            row_dict = dict(row)
            row_dict["selected_cluster_counts"] = json.dumps({str(k_): int(v_) for k_, v_ in selected_counts.items()})
            selected_rows.append(row_dict)

    selection_df = pd.DataFrame(selected_rows)
    if save_selection_path is not None:
        os.makedirs(os.path.dirname(save_selection_path), exist_ok=True)
        selection_df.to_csv(save_selection_path, index=False)
    if len(selected_graphs) == 0:
        raise RuntimeError("No graphs selected by global cluster proportional selector.")
    return selected_graphs, selection_df


def select_train_graphs_by_strategy(
    train_graphs: List[Data],
    *,
    strategy: str,
    base_k: int,
    seed: int,
    run_dir: str,
    cleancluster_manifest_path: Optional[str] = None,
) -> Tuple[List[Data], str]:
    """
    Returns selected train graphs and a dataset mode hint:
        label_aware_random : sample k per subject inside dataset each epoch
        fixed_all_selected : use the selected graph list as-is
    """
    strategy = str(strategy).lower()

    if strategy in {"original_random_k", "raw_random_k"}:
        return train_graphs, "label_aware_random"

    if strategy in {"all_raw", "raw_all"}:
        return train_graphs, "fixed_all_selected"

    if cleancluster_manifest_path is None:
        raise ValueError(f"cleancluster_manifest_path is required for segment_selection_strategy={strategy!r}")

    if strategy.startswith("global_cluster"):
        manifest_df = load_global_cluster_manifest(cleancluster_manifest_path)
    else:
        manifest_df = load_cleancluster_manifest(cleancluster_manifest_path)

    if strategy == "clean_random_k":
        return filter_graphs_by_manifest_keep_clean(train_graphs, manifest_df), "label_aware_random"

    if strategy == "all_clean":
        return filter_graphs_by_manifest_keep_clean(train_graphs, manifest_df), "fixed_all_selected"

    if strategy == "global_cluster_proportional_random_k":
        selected, _ = select_global_cluster_proportional_random_graphs_from_manifest(
            train_graphs,
            manifest_df,
            k=base_k,
            seed=seed,
            split_name="train",
            save_selection_path=os.path.join(run_dir, "selected_train_segments_global_cluster_proportional_random_k.csv"),
        )
        return selected, "fixed_all_selected"

    raise ValueError(
        f"Unknown segment_selection_strategy={strategy!r}. "
        "Supported: original_random_k, all_raw, clean_random_k, all_clean, global_cluster_proportional_random_k."
    )


# ---------------------------------------------------------
# Optional segment -> macro/subject level conversion
# ---------------------------------------------------------
def ensure_graph_dense_attrs(g: Data) -> Data:
    if not hasattr(g, "adj") or g.adj is None:
        n = int(g.x.shape[0])
        adj = torch.zeros((n, n), dtype=torch.float32)
        if hasattr(g, "edge_index") and g.edge_index is not None:
            ew = getattr(g, "edge_weight", None)
            if ew is None:
                ew = getattr(g, "edge_attr", None)
            if ew is None:
                ew = torch.ones(g.edge_index.shape[1], dtype=torch.float32)
            if ew.dim() > 1:
                ew = ew[:, 0]
            adj[g.edge_index[0].detach().cpu(), g.edge_index[1].detach().cpu()] = ew.detach().cpu().float()
        g.adj = adj

    adj = g.adj.detach().cpu().float() if torch.is_tensor(g.adj) else torch.tensor(g.adj, dtype=torch.float32)
    adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)
    edge_index, edge_weight = dense_to_sparse(adj)
    g.edge_index = edge_index.long()
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()
    g.adj = adj.float()
    return g


def reduce_graph_tensor_stack(xs: Sequence[torch.Tensor], how: str = "mean") -> torch.Tensor:
    stack = torch.stack([x.detach().cpu().float() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32) for x in xs], dim=0)
    how = str(how).lower()
    if how == "mean":
        return stack.mean(dim=0)
    if how == "median":
        return stack.median(dim=0).values
    if how == "max":
        return stack.max(dim=0).values
    if how == "min":
        return stack.min(dim=0).values
    if how == "sum":
        return stack.sum(dim=0)
    raise ValueError(f"Unsupported level_reduce={how!r}.")


def make_graph_from_dense_level(
    *,
    x: torch.Tensor,
    adj: torch.Tensor,
    y: int,
    subject_id: str,
    level: str,
    instance_id: str,
    segment_id: Optional[int] = None,
    start_sample: Optional[int] = None,
    adj_bank: Optional[torch.Tensor] = None,
    topology_bank: Optional[torch.Tensor] = None,
    topology_names: Optional[Sequence[str]] = None,
    conn_stack: Optional[torch.Tensor] = None,
) -> Data:
    adj = torch.nan_to_num(adj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)
    edge_index, edge_weight = dense_to_sparse(adj)
    g = Data(x=x.float(), edge_index=edge_index.long(), y=torch.tensor([int(y)], dtype=torch.long))
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()
    g.adj = adj.float()
    g.subject_id = str(subject_id)
    g.level = str(level)
    g.instance_id = str(instance_id)
    if segment_id is not None:
        g.segment_id = int(segment_id)
    if start_sample is not None:
        g.start_sample = int(start_sample)
    if adj_bank is not None:
        g.adj_bank = adj_bank.float()
        g.topology_bank = (adj_bank.abs() > 0).float() if topology_bank is None else topology_bank.float()
        g.topology_names = list(topology_names or [f"cand_{i}" for i in range(adj_bank.shape[0])])
    if conn_stack is not None:
        g.conn_stack = conn_stack.float()
    return g


def convert_segment_graphs_to_level(
    graphs: Sequence[Data],
    *,
    level: str = "segment",
    macro_duration_sec: float = 60.0,
    sfreq: float = SFREQ,
    reduce: str = "mean",
) -> List[Data]:
    level = str(level).lower()
    if level not in {"segment", "macro", "subject"}:
        raise ValueError("level must be one of: segment, macro, subject")
    graphs = [ensure_graph_dense_attrs(copy.copy(g)) for g in graphs]
    if level == "segment":
        for g in graphs:
            g.level = "segment"
            g.instance_id = f"{g.subject_id}_seg{int(getattr(g, 'segment_id', 0))}"
        return list(graphs)

    macro_len_samples = max(int(round(float(macro_duration_sec) * float(sfreq))), 1)
    grouped: Dict[Tuple[str, int], List[Data]] = defaultdict(list)
    for g in graphs:
        sid = str(g.subject_id)
        group_id = 0 if level == "subject" else int(getattr(g, "start_sample", 0)) // macro_len_samples
        grouped[(sid, group_id)].append(g)

    out: List[Data] = []
    for (sid, group_id), gs in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        y_values = [int(g.y.view(-1)[0].item()) for g in gs]
        if len(set(y_values)) != 1:
            raise ValueError(f"Mixed labels inside {level} group {(sid, group_id)}: {set(y_values)}")
        x = reduce_graph_tensor_stack([g.x for g in gs], how=reduce)
        adj = reduce_graph_tensor_stack([g.adj for g in gs], how=reduce)

        adj_bank = None
        topology_bank = None
        topology_names = None
        if hasattr(gs[0], "adj_bank") and getattr(gs[0], "adj_bank") is not None:
            adj_bank = reduce_graph_tensor_stack([g.adj_bank for g in gs], how=reduce)
            topology_bank = (adj_bank.abs() > 0).float()
            topology_names = list(getattr(gs[0], "topology_names", [f"cand_{i}" for i in range(adj_bank.shape[0])]))

        conn_stack = None
        if hasattr(gs[0], "conn_stack") and getattr(gs[0], "conn_stack") is not None:
            conn_stack = reduce_graph_tensor_stack([g.conn_stack for g in gs], how=reduce)

        starts = [int(getattr(g, "start_sample", 0)) for g in gs]
        segs = [int(getattr(g, "segment_id", -1)) for g in gs]
        new_g = make_graph_from_dense_level(
            x=x,
            adj=adj,
            y=y_values[0],
            subject_id=sid,
            level=level,
            instance_id=f"{sid}_{level}{group_id}",
            segment_id=min(segs) if segs else None,
            start_sample=min(starts) if starts else None,
            adj_bank=adj_bank,
            topology_bank=topology_bank,
            topology_names=topology_names,
            conn_stack=conn_stack,
        )
        new_g.num_source_segments = len(gs)
        new_g.source_segment_ids = segs
        out.append(new_g)
    return out


# ---------------------------------------------------------
# Graph preparation
# ---------------------------------------------------------
def summarize_graph_pool(graphs: Sequence[Data], name: str) -> Dict[str, Any]:
    subject_to_count: Dict[str, int] = defaultdict(int)
    label_to_subjects: Dict[int, set] = defaultdict(set)
    for g in graphs:
        sid = str(g.subject_id)
        y = int(g.y.view(-1)[0].item())
        subject_to_count[sid] += 1
        label_to_subjects[y].add(sid)
    counts = np.asarray(list(subject_to_count.values()), dtype=np.int64)
    info = {
        "num_graphs": int(len(graphs)),
        "num_subjects": int(len(subject_to_count)),
        "segments_min": int(counts.min()) if len(counts) else 0,
        "segments_mean": float(counts.mean()) if len(counts) else 0.0,
        "segments_max": int(counts.max()) if len(counts) else 0,
        "subjects_per_label": {int(k): int(len(v)) for k, v in label_to_subjects.items()},
    }
    print(f"\n[{name}] {info}")
    return info


def build_split_graphs(
    *,
    dataset_path: str,
    task: str,
    file_format: str,
    out_h5: str,
    feature_families: Sequence[str],
    connectivity_metric: str,
    connectivity_band: Optional[int],
    encoder_type: str,
    filter_method: str,
    fixed_edges,
    channel_names: Sequence[str],
    rebuild_h5: bool,
    use_split_prefix: bool,
    bad_ids: Optional[Iterable[str]],
    crop_len: int,
    step: int,
    latency: int,
    test_code: bool = False,
    test_n_subjects: int = 30,
) -> Tuple[Dict[str, Any], List[Data], List[Data], List[Data], int]:
    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    limit = int(test_n_subjects) if test_code else None
    train_records, train_ids = dataset_to_subject_records(train_set, "train", use_split_prefix, crop_len, step, latency, limit, bad_ids)
    val_records, val_ids = dataset_to_subject_records(val_set, "val", use_split_prefix, crop_len, step, latency, limit, bad_ids)
    test_records, test_ids = dataset_to_subject_records(test_set, "test", use_split_prefix, crop_len, step, latency, limit, bad_ids)

    train_records, train_ids = filter_bad_ids(train_records, train_ids, bad_ids)
    val_records, val_ids = filter_bad_ids(val_records, val_ids, bad_ids)
    test_records, test_ids = filter_bad_ids(test_records, test_ids, bad_ids)

    all_records = train_records + val_records + test_records
    if len(all_records) == 0:
        raise RuntimeError("No CAUEEG records available after filtering.")
    num_classes = len(sorted({int(r["label"]) for r in all_records}))

    need_build = bool(rebuild_h5) or (not os.path.isfile(out_h5))
    if need_build:
        print(f"[H5] Building master file: {out_h5}")
        build_master_eeg_dataset(
            subject_records=all_records,
            output_h5_path=out_h5,
            feature_families=list(feature_families),
            connectivity_metrics=[connectivity_metric],
            overwrite=True,
            skip_bad_segments=False,
            target_sampling_rate=None,
            qc_input_unit="auto",
        )
    else:
        print(f"[H5] Reusing existing master file: {out_h5}")

    encoder_l = str(encoder_type).lower()
    payload_connectivity_band = None if encoder_l in {"linkx_cnn5", "cnn5"} else connectivity_band
    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=train_ids + val_ids + test_ids,
        feature_families=list(feature_families),
        connectivity_metrics=[connectivity_metric],
        connectivity_band=payload_connectivity_band,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    if encoder_l in {"linkx_cnn5", "cnn5"}:
        train_graphs = build_graphs_from_payload_multiband(payload, train_ids, feature_families, connectivity_metric)
        val_graphs = build_graphs_from_payload_multiband(payload, val_ids, feature_families, connectivity_metric)
        test_graphs = build_graphs_from_payload_multiband(payload, test_ids, feature_families, connectivity_metric)
    else:
        train_graphs = build_graphs_from_payload(
            payload,
            train_ids,
            feature_families=list(feature_families),
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=True,
        )
        val_graphs = build_graphs_from_payload(
            payload,
            val_ids,
            feature_families=list(feature_families),
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=True,
        )
        test_graphs = build_graphs_from_payload(
            payload,
            test_ids,
            feature_families=list(feature_families),
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=True,
        )

    summarize_graph_pool(train_graphs, "train_graphs_original")
    summarize_graph_pool(val_graphs, "val_graphs_original")
    summarize_graph_pool(test_graphs, "test_graphs_original")
    return config, train_graphs, val_graphs, test_graphs, num_classes


# ---------------------------------------------------------
# Segment-training datasets / collate
# ---------------------------------------------------------
def _stable_int_from_string(x: str) -> int:
    import hashlib
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)


class GraphSegmentDataset(Dataset):
    def __init__(self, graphs: Sequence[Data]):
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("GraphSegmentDataset received an empty graph list.")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


class SubjectBalancedSegmentKDataset(Dataset):
    """
    Flat segment dataset that resamples k segments per subject per epoch.

    This is the budget-matched segment baseline for MIL:
        same candidate graph pool, same base_k, but segment-level CE loss.
    """
    def __init__(
        self,
        graphs: Sequence[Data],
        k: int,
        seed: int = 42,
        fill_with_replacement: bool = True,
        sort_graphs_by: str = "segment_id",
    ):
        if k is None or int(k) <= 0:
            raise ValueError(f"k must be positive for SubjectBalancedSegmentKDataset, got {k}")
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        self.fill_with_replacement = bool(fill_with_replacement)
        self.subject_to_graphs: Dict[str, List[Data]] = defaultdict(list)
        self.subject_to_label: Dict[str, int] = {}

        for g in graphs:
            sid = str(g.subject_id)
            y = int(g.y.view(-1)[0].item())
            self.subject_to_graphs[sid].append(g)
            if sid in self.subject_to_label and self.subject_to_label[sid] != y:
                raise ValueError(f"Subject {sid} has inconsistent labels.")
            self.subject_to_label[sid] = y

        self.subject_ids = sorted(self.subject_to_graphs.keys())
        self.subject_labels = [self.subject_to_label[sid] for sid in self.subject_ids]
        if len(self.subject_ids) == 0:
            raise ValueError("No subjects in SubjectBalancedSegmentKDataset.")

        for sid in self.subject_ids:
            if sort_graphs_by == "segment_id":
                self.subject_to_graphs[sid] = sorted(
                    self.subject_to_graphs[sid],
                    key=lambda g: (int(getattr(g, "segment_id", 0)), int(getattr(g, "start_sample", 0))),
                )
            elif sort_graphs_by == "start_sample":
                self.subject_to_graphs[sid] = sorted(
                    self.subject_to_graphs[sid],
                    key=lambda g: (int(getattr(g, "start_sample", 0)), int(getattr(g, "segment_id", 0))),
                )
            else:
                raise ValueError(f"Unsupported sort_graphs_by={sort_graphs_by!r}")

        first_graph = self.subject_to_graphs[self.subject_ids[0]][0]
        self.num_node_features = int(first_graph.x.shape[-1])
        self.num_nodes = int(first_graph.x.shape[0])
        self._indices: List[Tuple[str, int]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        indices: List[Tuple[str, int]] = []
        for sid in self.subject_ids:
            graphs = self.subject_to_graphs[sid]
            n = len(graphs)
            rng = random.Random(self.seed + 1000003 * self.epoch + _stable_int_from_string(sid))
            if n >= self.k:
                chosen = rng.sample(range(n), self.k)
            else:
                chosen = list(range(n))
                if self.fill_with_replacement:
                    chosen += [rng.randrange(n) for _ in range(self.k - n)]
            for j in chosen:
                indices.append((sid, int(j)))
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Data:
        sid, j = self._indices[idx]
        return self.subject_to_graphs[sid][j]


def collate_graph_segments(batch: Sequence[Data]) -> Dict[str, Any]:
    graphs = list(batch)
    pyg_batch = Batch.from_data_list(graphs)
    labels = torch.tensor([int(g.y.view(-1)[0].item()) for g in graphs], dtype=torch.long)
    subject_ids = [str(getattr(g, "subject_id", "")) for g in graphs]
    segment_ids = [int(getattr(g, "segment_id", -1)) for g in graphs]
    start_samples = [int(getattr(g, "start_sample", -1)) for g in graphs]

    full_adj = []
    conn_stack = []
    adj_bank = []
    topology_bank = []
    topology_names = None

    for g in graphs:
        if hasattr(g, "adj") and g.adj is not None:
            a = g.adj.detach().cpu().float() if torch.is_tensor(g.adj) else torch.tensor(g.adj, dtype=torch.float32)
            full_adj.append(a)
        if hasattr(g, "conn_stack") and g.conn_stack is not None:
            cs = g.conn_stack.detach().cpu().float() if torch.is_tensor(g.conn_stack) else torch.tensor(g.conn_stack, dtype=torch.float32)
            conn_stack.append(cs)
        if hasattr(g, "adj_bank") and g.adj_bank is not None:
            ab = g.adj_bank.detach().cpu().float() if torch.is_tensor(g.adj_bank) else torch.tensor(g.adj_bank, dtype=torch.float32)
            adj_bank.append(ab)
        if hasattr(g, "topology_bank") and g.topology_bank is not None:
            tb = g.topology_bank.detach().cpu().float() if torch.is_tensor(g.topology_bank) else torch.tensor(g.topology_bank, dtype=torch.float32)
            topology_bank.append(tb)
        if topology_names is None and hasattr(g, "topology_names"):
            topology_names = list(g.topology_names)

    out: Dict[str, Any] = {
        "pyg_batch": pyg_batch,
        "labels": labels,
        "subject_ids": subject_ids,
        "segment_ids": segment_ids,
        "start_samples": start_samples,
    }
    if len(full_adj) == len(graphs):
        out["full_adj"] = torch.stack(full_adj, dim=0)
    if len(conn_stack) == len(graphs):
        out["conn_stack"] = torch.stack(conn_stack, dim=0)
    if len(adj_bank) == len(graphs):
        out["adj_bank"] = torch.stack(adj_bank, dim=0)
    if len(topology_bank) == len(graphs):
        out["topology_bank"] = torch.stack(topology_bank, dim=0)
    if topology_names is not None:
        out["topology_names"] = topology_names
    return out


def move_batch_to_device(batch: Dict[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to"):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------
# Segment model uses the exact graph encoder/classifier family from SubjectMILClassifier
# ---------------------------------------------------------
class SegmentGraphClassifierFromMIL(nn.Module):
    def __init__(self, **mil_kwargs):
        super().__init__()
        mil_kwargs = dict(mil_kwargs)
        mil_kwargs.setdefault("mil_pool_type", "mean")
        base = SubjectMILClassifier(**_filter_kwargs_for_class(SubjectMILClassifier, mil_kwargs))
        self.encoder_type = str(getattr(base, "encoder_type", mil_kwargs.get("encoder_type", "gnn"))).lower()
        self.graph_encoder = base.graph_encoder
        self.classifier = base.classifier

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if self.encoder_type == "linkx_cnn":
            if "full_adj" not in batch_dict:
                raise KeyError("batch_dict missing full_adj for encoder_type='linkx_cnn'.")
            graph_emb = self.graph_encoder(batch_dict["pyg_batch"], batch_dict["full_adj"])
        elif self.encoder_type in {"cnn5", "linkx_cnn5"}:
            if "conn_stack" not in batch_dict:
                raise KeyError("batch_dict missing conn_stack for encoder_type in {'cnn5','linkx_cnn5'}.")
            graph_emb = self.graph_encoder(batch_dict["pyg_batch"], batch_dict["conn_stack"])
        else:
            graph_emb = self.graph_encoder(batch_dict["pyg_batch"])

        # Some newer encoders may return (embedding, aux).
        if isinstance(graph_emb, tuple):
            graph_emb, aux = graph_emb
        else:
            aux = {}
        logits = self.classifier(graph_emb)
        return {"logits": logits, "graph_emb": graph_emb, **aux}


def build_model_kwargs(
    *,
    num_node_features: int,
    num_classes: int,
    num_nodes: int,
    encoder_type: str,
    edge_mode: str,
    graph_emb_dim: int,
    dropout: float,
    mil_pool_type: str,
    attn_dim: int,
    gnn_hidden_dim: int,
    node_hidden_dims: Sequence[int],
    edge_hidden_dims: Sequence[int],
    branch_emb_dim: int,
    cnn_num_bands: Optional[int] = None,
    use_gcn_norm: bool = False,
) -> Dict[str, Any]:
    return {
        "num_node_features": int(num_node_features),
        "num_classes": int(num_classes),
        "num_nodes": int(num_nodes),
        "encoder_type": encoder_type,
        "edge_mode": edge_mode,
        "graph_emb_dim": int(graph_emb_dim),
        "dropout": float(dropout),
        "mil_pool_type": mil_pool_type,
        "attn_dim": int(attn_dim),
        "gnn_hidden_dim": int(gnn_hidden_dim),
        "node_hidden_dims": tuple(node_hidden_dims),
        "edge_hidden_dims": tuple(edge_hidden_dims),
        "branch_emb_dim": int(branch_emb_dim),
        "cnn_num_bands": cnn_num_bands,
        "use_gcn_norm": bool(use_gcn_norm),
        "gcn_normalize_input": bool(use_gcn_norm),
        "gcn_norm_add_self_loops": bool(use_gcn_norm),
    }


# ---------------------------------------------------------
# Segment-training evaluation / fitting
# ---------------------------------------------------------
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], num_classes: Optional[int] = None) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = list(range(int(num_classes))) if num_classes is not None else None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "conf_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def train_segment_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
) -> Dict[str, Any]:
    model.train()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        logits = out["logits"]
        labels = batch["labels"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


@torch.no_grad()
def evaluate_segment_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    subject_ids: List[str] = []
    segment_ids: List[int] = []
    start_samples: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        logits = out["logits"]
        labels = batch["labels"]
        loss = criterion(logits, labels)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
        subject_ids.extend([str(x) for x in batch["subject_ids"]])
        segment_ids.extend([int(x) for x in batch["segment_ids"]])
        start_samples.extend([int(x) for x in batch["start_samples"]])

    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["y_prob"] = y_prob
    metrics["subject_ids"] = subject_ids
    metrics["segment_ids"] = segment_ids
    metrics["start_samples"] = start_samples
    return metrics


def segment_metrics_to_df(metrics: Dict[str, Any], num_classes: int, split: str) -> pd.DataFrame:
    rows = []
    for sid, seg_id, st, yt, yp, prob in zip(
        metrics["subject_ids"], metrics["segment_ids"], metrics["start_samples"],
        metrics["y_true"], metrics["y_pred"], metrics["y_prob"]
    ):
        row = {
            "split": split,
            "subject_id": str(sid),
            "segment_id": int(seg_id),
            "start_sample": int(st),
            "true_label": int(yt),
            "segment_pred_label": int(yp),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(prob[c])
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_segment_df_to_subject_df(segment_df: pd.DataFrame, num_classes: int, split: str) -> pd.DataFrame:
    rows = []
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    for sid, sdf in segment_df.groupby("subject_id"):
        labels = sdf["true_label"].astype(int).unique().tolist()
        if len(labels) != 1:
            raise ValueError(f"Subject {sid} has mixed labels in segment_df: {labels}")
        mean_prob = sdf[prob_cols].to_numpy(dtype=np.float64).mean(axis=0)
        pred = int(np.argmax(mean_prob))
        row = {
            "split": split,
            "subject_id": str(sid),
            "true_label": int(labels[0]),
            "pred_label": pred,
            "num_segments": int(len(sdf)),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(mean_prob[c])
        rows.append(row)
    return pd.DataFrame(rows)


def subject_df_to_metrics(subject_df: pd.DataFrame, num_classes: int) -> Dict[str, Any]:
    y_true = subject_df["true_label"].astype(int).to_numpy()
    y_pred = subject_df["pred_label"].astype(int).to_numpy()
    metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)
    prob = subject_df[[f"prob_{c}" for c in range(num_classes)]].to_numpy(dtype=np.float64)
    p_true = np.clip(prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    metrics["loss"] = float((-np.log(p_true)).mean())
    metrics["y_prob"] = prob.tolist()
    return metrics


def evaluate_segment_subject_level(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
    split: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    segment_metrics = evaluate_segment_loader(model, loader, criterion, device, num_classes=num_classes)
    segment_df = segment_metrics_to_df(segment_metrics, num_classes=num_classes, split=split)
    subject_df = aggregate_segment_df_to_subject_df(segment_df, num_classes=num_classes, split=split)
    subject_metrics = subject_df_to_metrics(subject_df, num_classes=num_classes)
    return subject_metrics, subject_df, segment_df, segment_metrics


def fit_segment_baseline_subject_es(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: Union[str, torch.device],
    num_classes: int,
    epochs: int,
    patience: int,
    start_epoch: int,
    min_delta: float,
    top_k: int,
    save_path: Optional[str],
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, Any]] = None

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        file_prefix = os.path.splitext(os.path.basename(save_path))[0] + "_topk"
    else:
        save_dir = None
        file_prefix = "segment_topk"

    if TopKEarlyStopping is not None:
        early_stopper = TopKEarlyStopping(
            patience=patience,
            start_epoch=start_epoch,
            min_delta=min_delta,
            top_k=top_k,
            save_dir=save_dir,
            verbose=verbose,
            file_prefix=file_prefix,
        )
    else:
        early_stopper = None
        best_loss = float("inf")
        bad_epochs = 0

    for epoch in range(1, int(epochs) + 1):
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        train_seg_metrics = train_segment_one_epoch(model, train_loader, optimizer, criterion, device, num_classes)
        train_subject_metrics, _, _, _ = evaluate_segment_subject_level(model, train_loader, criterion, device, num_classes, split="train")
        val_subject_metrics, _, _, val_seg_metrics = evaluate_segment_subject_level(model, val_loader, criterion, device, num_classes, split="val")

        row = {
            "epoch": int(epoch),
            "train_seg_loss": float(train_seg_metrics["loss"]),
            "train_seg_acc": float(train_seg_metrics["accuracy"]),
            "train_seg_bal_acc": float(train_seg_metrics["balanced_accuracy"]),
            "train_seg_macro_f1": float(train_seg_metrics["macro_f1"]),
            "train_loss": float(train_subject_metrics["loss"]),
            "train_acc": float(train_subject_metrics["accuracy"]),
            "train_bal_acc": float(train_subject_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_subject_metrics["macro_f1"]),
            "val_seg_loss": float(val_seg_metrics["loss"]),
            "val_seg_acc": float(val_seg_metrics["accuracy"]),
            "val_seg_bal_acc": float(val_seg_metrics["balanced_accuracy"]),
            "val_seg_macro_f1": float(val_seg_metrics["macro_f1"]),
            "val_loss": float(val_subject_metrics["loss"]),
            "val_acc": float(val_subject_metrics["accuracy"]),
            "val_bal_acc": float(val_subject_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_subject_metrics["macro_f1"]),
        }
        history.append(row)

        if verbose:
            print(
                f"Epoch [{epoch:03d}/{epochs}] | "
                f"TrainSeg loss={row['train_seg_loss']:.4f}, acc={row['train_seg_acc']:.4f} | "
                f"TrainSubj loss={row['train_loss']:.4f}, bal={row['train_bal_acc']:.4f}, f1={row['train_macro_f1']:.4f} || "
                f"ValSubj loss={row['val_loss']:.4f}, acc={row['val_acc']:.4f}, "
                f"bal={row['val_bal_acc']:.4f}, f1={row['val_macro_f1']:.4f}"
            )

        if early_stopper is not None:
            should_stop = early_stopper(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=float(val_subject_metrics["loss"]),
                val_bal_acc=float(val_subject_metrics["balanced_accuracy"]),
                val_macro_f1=float(val_subject_metrics["macro_f1"]),
                extra_state={"history": history},
            )
        else:
            improved = float(val_subject_metrics["loss"]) < best_loss - float(min_delta)
            if improved:
                best_loss = float(val_subject_metrics["loss"])
                bad_epochs = 0
                best_state = copy.deepcopy(model.state_dict())
                if save_path is not None:
                    torch.save({"epoch": epoch, "model_state_dict": best_state, "optimizer_state_dict": optimizer.state_dict()}, save_path)
            elif epoch >= start_epoch:
                bad_epochs += 1
            should_stop = bad_epochs >= patience

        if should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

    if early_stopper is not None:
        best_meta = early_stopper.get_best_checkpoint()
        if best_meta is not None and best_meta.get("path") is not None and os.path.exists(best_meta["path"]):
            checkpoint = torch.load(best_meta["path"], map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            best_state = checkpoint["model_state_dict"]
            if save_path is not None and best_meta["path"] != save_path:
                torch.save(checkpoint, save_path)
        elif best_state is not None:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    best_val_metrics, _, _, _ = evaluate_segment_subject_level(model, val_loader, criterion, device, num_classes, split="val")
    return model, best_val_metrics, history, best_state


# ---------------------------------------------------------
# MIL prediction saving
# ---------------------------------------------------------
def rows_to_prediction_df(rows: Sequence[Dict[str, Any]], num_classes: int, split: str) -> pd.DataFrame:
    records = []
    for r in rows:
        prob = np.asarray(r["prob"], dtype=np.float32).reshape(-1)
        emb = np.asarray(r.get("embedding", []), dtype=np.float32).reshape(-1)
        rec = {
            "split": split,
            "subject_id": str(r["subject_id"]),
            "true_label": int(r["label"]),
            "pred_label": int(r["pred"]),
        }
        for i in range(num_classes):
            rec[f"prob_{i}"] = float(prob[i])
        if emb.size > 0:
            rec["embedding_json"] = json.dumps(emb.tolist())
        records.append(rec)
    return pd.DataFrame(records)


def save_mil_predictions(model: nn.Module, loader: DataLoader, device: Union[str, torch.device], csv_path: str, num_classes: int, split: str) -> pd.DataFrame:
    rows = collect_subject_embeddings(model, loader, device)
    df = rows_to_prediction_df(rows, num_classes=num_classes, split=split)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    return df


# ---------------------------------------------------------
# Main training runner
# ---------------------------------------------------------
def run_caueeg_linkx_training(
    *,
    training_approach: str,
    dataset_path: str,
    fixed_edges,
    channel_names: Sequence[str] = CAUEEG_EEG19,
    task: str = "dementia-no-overlap",
    file_format: str = "feather",
    out_h5: str = "caueeg_master_linkx.h5",
    feature_families: Sequence[str] = ("relative_band_power", "statistical"),
    connectivity_metric: str = "coherence",
    connectivity_band: Optional[int] = 2,
    encoder_type: str = "linkx",
    mil_pool_type: str = "gated",
    filter_method: str = "fixed",
    segment_selection_strategy: str = "original_random_k",
    cleancluster_manifest_path: Optional[str] = None,
    level: str = "segment",
    macro_duration_sec: float = 60.0,
    level_reduce: str = "mean",
    base_k: int = 10,
    seed: int = 42,
    batch_size: int = 8,
    test_batch_size: Optional[int] = None,
    epochs: int = 200,
    patience: int = 20,
    start_epoch: int = 20,
    min_delta: float = 1e-3,
    top_k: int = 3,
    lr: float = 3e-4,
    weight_decay: float = 5e-3,
    dropout: float = 0.3,
    graph_emb_dim: int = 64,
    attn_dim: int = 64,
    gnn_hidden_dim: int = 64,
    node_hidden_dims: Sequence[int] = (256, 128),
    edge_hidden_dims: Sequence[int] = (128, 64),
    branch_emb_dim: int = 64,
    edge_mode: str = "topology_weighted",
    device: str = "cuda",
    rebuild_h5: bool = False,
    output_root: str = "graph/results_caueeg_linkx_all_training",
    use_split_prefix: bool = True,
    bad_ids: Optional[Iterable[str]] = ("00587", "00781", "01301"),
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
    test_code: bool = False,
    test_n_subjects: int = 30,
    use_gcn_norm: bool = False,
) -> Dict[str, Any]:
    training_approach = str(training_approach).lower()
    if training_approach not in {"mil", "segment_k", "segment_all"}:
        raise ValueError("training_approach must be one of: mil, segment_k, segment_all")

    set_global_seed(seed)
    device_t = torch.device(device if torch.cuda.is_available() or str(device).startswith("cpu") else "cpu")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_tag = f"k{base_k}" if training_approach != "segment_all" else "all"
    run_name = (
        f"{timestamp}_seed{seed}_{training_approach}_{level}_{segment_selection_strategy}_"
        f"{connectivity_metric}_band{connectivity_band}_{k_tag}_{encoder_type}_{mil_pool_type}_{filter_method}"
    )
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config, train_graphs, val_graphs, test_graphs, num_classes = build_split_graphs(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        out_h5=out_h5,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        encoder_type=encoder_type,
        filter_method=filter_method,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        rebuild_h5=rebuild_h5,
        use_split_prefix=use_split_prefix,
        bad_ids=bad_ids,
        crop_len=crop_len,
        step=step,
        latency=latency,
        test_code=test_code,
        test_n_subjects=test_n_subjects,
    )

    train_graphs, train_dataset_mode = select_train_graphs_by_strategy(
        train_graphs,
        strategy=segment_selection_strategy,
        base_k=base_k,
        seed=seed,
        run_dir=run_dir,
        cleancluster_manifest_path=cleancluster_manifest_path,
    )
    summarize_graph_pool(train_graphs, f"train_graphs_after_{segment_selection_strategy}")

    train_graphs = convert_segment_graphs_to_level(train_graphs, level=level, macro_duration_sec=macro_duration_sec, reduce=level_reduce)
    val_graphs = convert_segment_graphs_to_level(val_graphs, level=level, macro_duration_sec=macro_duration_sec, reduce=level_reduce)
    test_graphs = convert_segment_graphs_to_level(test_graphs, level=level, macro_duration_sec=macro_duration_sec, reduce=level_reduce)
    summarize_graph_pool(train_graphs, f"train_{level}_instances")
    summarize_graph_pool(val_graphs, f"val_{level}_instances")
    summarize_graph_pool(test_graphs, f"test_{level}_instances")

    encoder_l = str(encoder_type).lower()
    multiband_encoder = encoder_l in {"linkx_cnn5", "cnn5"}
    if test_batch_size is None:
        test_batch_size = max(1, batch_size // 2) if training_approach == "mil" else max(16, batch_size)

    log_config = {
        "training_approach": training_approach,
        "dataset_path": dataset_path,
        "task": task,
        "file_format": file_format,
        "out_h5": out_h5,
        "feature_families": list(feature_families),
        "connectivity_metric": connectivity_metric,
        "connectivity_band": connectivity_band,
        "encoder_type": encoder_type,
        "mil_pool_type": mil_pool_type,
        "filter_method": filter_method,
        "segment_selection_strategy": segment_selection_strategy,
        "level": level,
        "base_k": base_k,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "start_epoch": start_epoch,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "graph_emb_dim": graph_emb_dim,
        "attn_dim": attn_dim,
        "edge_mode": edge_mode,
        "use_gcn_norm": use_gcn_norm,
        "bad_ids": list(bad_ids) if bad_ids is not None else None,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w") as f:
        json.dump(make_jsonable(log_config), f, indent=2)

    ckpt_path = os.path.join(run_dir, "best_model.pt")
    criterion = nn.CrossEntropyLoss()

    if training_approach == "mil":
        if train_dataset_mode == "label_aware_random":
            train_dataset = LabelAwareSubjectBagDataset(
                train_graphs,
                train=True,
                base_k=None,
                k_by_label={label: int(base_k) for label in range(num_classes)},
                max_k_per_subject=300,
                seed=seed,
                return_segment_ids=True,
            )
        else:
            train_dataset = SubjectBagGraphDataset(train_graphs, max_segments_per_subject=None, train=True)

        val_dataset = LabelAwareSubjectBagDataset(val_graphs, train=False, eval_k_per_subject=None, seed=seed)
        test_dataset = LabelAwareSubjectBagDataset(test_graphs, train=False, eval_k_per_subject=None, seed=seed)

        collate_fn = collate_subject_bags_multiband if multiband_encoder else collate_subject_bags
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True)

        model_kwargs = build_model_kwargs(
            num_node_features=train_dataset.num_node_features,
            num_classes=num_classes,
            num_nodes=train_dataset.num_nodes,
            encoder_type=encoder_type,
            edge_mode=edge_mode,
            graph_emb_dim=graph_emb_dim,
            dropout=dropout,
            mil_pool_type=mil_pool_type,
            attn_dim=attn_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            node_hidden_dims=node_hidden_dims,
            edge_hidden_dims=edge_hidden_dims,
            branch_emb_dim=branch_emb_dim,
            cnn_num_bands=getattr(train_graphs[0], "conn_stack", torch.empty(0)).shape[0] if multiband_encoder else None,
            use_gcn_norm=use_gcn_norm,
        )
        model = SubjectMILClassifier(**_filter_kwargs_for_class(SubjectMILClassifier, model_kwargs)).to(device_t)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        model, val_metrics, history, best_state = _call_with_supported_kwargs(
            fit_mil_baseline,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_t,
            epochs=epochs,
            patience=patience,
            save_path=ckpt_path,
            start_epoch=start_epoch,
            min_delta=min_delta,
            top_k=top_k,
            verbose=True,
        )

        train_metrics = evaluate(model, train_loader, criterion, device_t)
        val_metrics = evaluate(model, val_loader, criterion, device_t)
        test_metrics = evaluate(model, test_loader, criterion, device_t)

        save_history_csv(history, os.path.join(run_dir, "history.csv"))
        summary_rows = []
        for split, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
            summary_rows.append({
                "split": split,
                "loss": float(metrics["loss"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "macro_f1": float(metrics["macro_f1"]),
                "confusion_matrix": metrics["conf_matrix"],
            })
        save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))
        save_mil_predictions(model, val_loader, device_t, os.path.join(run_dir, "val_predictions.csv"), num_classes, "val")
        test_pred_df = save_mil_predictions(model, test_loader, device_t, os.path.join(run_dir, "test_predictions.csv"), num_classes, "test")

        summary_test = {
            "encoder_type": encoder_type,
            "training_approach": "mil",
            "mil_pool_type": mil_pool_type,
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "confusion_matrix": test_metrics["conf_matrix"],
        }

    else:
        if training_approach == "segment_k" and train_dataset_mode == "label_aware_random":
            train_dataset = SubjectBalancedSegmentKDataset(train_graphs, k=base_k, seed=seed, fill_with_replacement=True)
        else:
            # segment_all, or a strategy that already fixed selected k segments.
            train_dataset = GraphSegmentDataset(train_graphs)

        val_dataset = GraphSegmentDataset(val_graphs)
        test_dataset = GraphSegmentDataset(test_graphs)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_segments, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_graph_segments, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=collate_graph_segments, num_workers=0, pin_memory=True)

        # num_node_features/num_nodes are available in both datasets.
        first_graph = train_graphs[0]
        model_kwargs = build_model_kwargs(
            num_node_features=int(first_graph.x.shape[-1]),
            num_classes=num_classes,
            num_nodes=int(first_graph.x.shape[0]),
            encoder_type=encoder_type,
            edge_mode=edge_mode,
            graph_emb_dim=graph_emb_dim,
            dropout=dropout,
            mil_pool_type="mean",
            attn_dim=attn_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            node_hidden_dims=node_hidden_dims,
            edge_hidden_dims=edge_hidden_dims,
            branch_emb_dim=branch_emb_dim,
            cnn_num_bands=getattr(first_graph, "conn_stack", torch.empty(0)).shape[0] if multiband_encoder else None,
            use_gcn_norm=use_gcn_norm,
        )
        model = SegmentGraphClassifierFromMIL(**model_kwargs).to(device_t)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        model, val_metrics, history, best_state = fit_segment_baseline_subject_es(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device_t,
            num_classes=num_classes,
            epochs=epochs,
            patience=patience,
            start_epoch=start_epoch,
            min_delta=min_delta,
            top_k=top_k,
            save_path=ckpt_path,
            verbose=True,
        )

        train_metrics, train_subject_df, train_segment_df, train_seg_metrics = evaluate_segment_subject_level(model, train_loader, criterion, device_t, num_classes, "train")
        val_metrics, val_subject_df, val_segment_df, val_seg_metrics = evaluate_segment_subject_level(model, val_loader, criterion, device_t, num_classes, "val")
        test_metrics, test_subject_df, test_segment_df, test_seg_metrics = evaluate_segment_subject_level(model, test_loader, criterion, device_t, num_classes, "test")

        save_history_csv(history, os.path.join(run_dir, "history.csv"))
        train_subject_df.to_csv(os.path.join(run_dir, "train_predictions_subject_agg.csv"), index=False)
        val_subject_df.to_csv(os.path.join(run_dir, "val_predictions_subject_agg.csv"), index=False)
        test_subject_df.to_csv(os.path.join(run_dir, "test_predictions_subject_agg.csv"), index=False)
        train_segment_df.to_csv(os.path.join(run_dir, "train_predictions_segment.csv"), index=False)
        val_segment_df.to_csv(os.path.join(run_dir, "val_predictions_segment.csv"), index=False)
        test_segment_df.to_csv(os.path.join(run_dir, "test_predictions_segment.csv"), index=False)

        summary_rows = []
        for split, subj_m, seg_m in [
            ("train", train_metrics, train_seg_metrics),
            ("val", val_metrics, val_seg_metrics),
            ("test", test_metrics, test_seg_metrics),
        ]:
            summary_rows.append({
                "split": split,
                "loss": float(subj_m["loss"]),
                "accuracy": float(subj_m["accuracy"]),
                "balanced_accuracy": float(subj_m["balanced_accuracy"]),
                "macro_f1": float(subj_m["macro_f1"]),
                "segment_loss": float(seg_m["loss"]),
                "segment_accuracy": float(seg_m["accuracy"]),
                "segment_balanced_accuracy": float(seg_m["balanced_accuracy"]),
                "segment_macro_f1": float(seg_m["macro_f1"]),
                "confusion_matrix": subj_m["conf_matrix"],
            })
        save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))
        test_pred_df = test_subject_df

        summary_test = {
            "encoder_type": encoder_type,
            "training_approach": training_approach,
            "mil_pool_type": "None",
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "confusion_matrix": test_metrics["conf_matrix"],
        }

    summary_test.update({
        "feature_families": list(feature_families),
        "topology": filter_method,
        "connectivity_metric": connectivity_metric,
        "connectivity_band": connectivity_band,
        "edge_mode": edge_mode,
        "segment_selection_strategy": segment_selection_strategy,
        "level": level,
        "macro_duration_sec": macro_duration_sec,
        "level_reduce": level_reduce,
        "base_k": base_k,
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "start_epoch": start_epoch,
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "graph_emb_dim": graph_emb_dim,
        "attn_dim": attn_dim if training_approach == "mil" else "None",
        "use_gcn_norm": use_gcn_norm,
        "seed": seed,
    })
    save_summary_metrics_csv([summary_test], os.path.join(run_dir, "summary_test.csv"))

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "history": history,
        "best_state": best_state,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_pred_df": test_pred_df,
        "summary_test": summary_test,
        "run_dir": run_dir,
    }


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified CAUEEG LinkX/MIL/segment training script.")
    parser.add_argument("--training_approach", type=str, default="segment_all", choices=["mil", "segment_k", "segment_all"])
    parser.add_argument("--dataset_path", type=str, default="/home/anphan/Downloads/caueeg-dataset/")
    parser.add_argument("--task", type=str, default="dementia-no-overlap")
    parser.add_argument("--file_format", type=str, default="edf", choices=["edf", "feather", "memmap", "np"])
    parser.add_argument("--out_h5", type=str, default="/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5")
    parser.add_argument("--rebuild_h5", action="store_true")
    parser.add_argument("--output_root", type=str, default="/home/anphan/Documents/CAUEEG/result_all_training")

    parser.add_argument("--feature_families_str", type=str, default="relative_band_power,statistical")
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=parse_optional_int, default=2)
    parser.add_argument("--topology", type=str, default="fixed")
    parser.add_argument("--encoder_type", type=str, default="linkx")
    parser.add_argument("--mil_pool_type", type=str, default="mean")
    parser.add_argument("--edge_mode", type=str, default="topology_weighted")

    parser.add_argument("--segment_selection_strategy", type=str, default="original_random_k")
    parser.add_argument("--cleancluster_manifest_path", type=str, default="/home/anphan/Documents/CAUEEG/visualize-merged_sliding_random/global_segment_clusters/global_cluster_manifest.csv")
    parser.add_argument("--level", type=str, default="segment", choices=["segment", "macro", "subject"])
    parser.add_argument("--macro_duration_sec", type=float, default=60.0)
    parser.add_argument("--level_reduce", type=str, default="mean")

    parser.add_argument("--base_k", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--start_epoch", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-3)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--graph_emb_dim", type=int, default=64)
    parser.add_argument("--attn_dim", type=int, default=64)
    parser.add_argument("--gnn_hidden_dim", type=int, default=64)
    parser.add_argument("--branch_emb_dim", type=int, default=64)
    parser.add_argument("--use_gcn_norm", action="store_true")

    parser.add_argument("--crop_len", type=int, default=CROP_LEN)
    parser.add_argument("--latency", type=int, default=LATENCY)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument("--use_split_prefix", type=parse_bool, default=True)
    parser.add_argument("--bad_ids_str", type=str, default="00587,00781,01301")
    parser.add_argument("--seeds", type=str, default="15,42,100")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_code", action="store_true")
    parser.add_argument("--test_n_subjects", type=int, default=30)
    return parser.parse_args()


def load_fixed_edges_from_config(channel_names: Sequence[str]) -> set[Tuple[int, int]]:
    try:
        import config as project_config
        fixed_pairs = getattr(project_config, "MONOFIXEDGES")
        return _normalize_fixed_edges(fixed_pairs, n_channels=len(channel_names), channel_names=channel_names)
    except Exception as e:
        print(f"[WARN] Could not load config.MONOFIXEDGES ({e}). Falling back to empty fixed_edges.")
        return set()


def main() -> None:
    args = parse_args()
    feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    bad_ids = [x.strip() for x in args.bad_ids_str.split(",") if x.strip()] if args.bad_ids_str else None
    step = int(args.crop_len * (1.0 - float(args.overlap)))
    out_h5 = args.out_h5
    if out_h5 is None:
        out_h5 = os.path.join(args.output_root, f"caueeg_{args.task}_{args.file_format}_features.h5")

    channel_names = CAUEEG_EEG19
    fixed_edges = load_fixed_edges_from_config(channel_names)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root = os.path.join(
        args.output_root,
        f"{timestamp}_{args.training_approach}_{args.encoder_type}_{args.segment_selection_strategy}_{args.connectivity_metric}",
    )
    os.makedirs(experiment_root, exist_ok=True)

    summary_rows = []
    for seed in seeds:
        out = run_caueeg_linkx_training(
            training_approach=args.training_approach,
            dataset_path=args.dataset_path,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            task=args.task,
            file_format=args.file_format,
            out_h5=out_h5,
            feature_families=feature_families,
            connectivity_metric=args.connectivity_metric,
            connectivity_band=args.connectivity_band,
            encoder_type=args.encoder_type,
            mil_pool_type=args.mil_pool_type,
            filter_method=args.topology,
            segment_selection_strategy=args.segment_selection_strategy,
            cleancluster_manifest_path=args.cleancluster_manifest_path,
            level=args.level,
            macro_duration_sec=args.macro_duration_sec,
            level_reduce=args.level_reduce,
            base_k=args.base_k,
            seed=seed,
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            epochs=args.epochs,
            patience=args.patience,
            start_epoch=args.start_epoch,
            min_delta=args.min_delta,
            top_k=args.top_k,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            graph_emb_dim=args.graph_emb_dim,
            attn_dim=args.attn_dim,
            gnn_hidden_dim=args.gnn_hidden_dim,
            branch_emb_dim=args.branch_emb_dim,
            edge_mode=args.edge_mode,
            device=args.device,
            rebuild_h5=args.rebuild_h5,
            output_root=experiment_root,
            use_split_prefix=args.use_split_prefix,
            bad_ids=bad_ids,
            crop_len=args.crop_len,
            step=step,
            latency=args.latency,
            test_code=args.test_code,
            test_n_subjects=args.test_n_subjects,
            use_gcn_norm=args.use_gcn_norm,
        )
        summary_rows.append(out["summary_test"])

    save_seed_aggregation(summary_rows, os.path.join(experiment_root, "agg_seed_results"))
    print(f"Done. Experiment root: {experiment_root}")


if __name__ == "__main__":
    main()
