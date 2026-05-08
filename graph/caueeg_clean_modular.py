from __future__ import annotations

"""
caueeg_clean_modular.py

A cleaner CAUEEG experiment runner that keeps caueeg_removenoise.py as the
backbone idea:
  1) build/reuse H5
  2) build a segment graph pool
  3) optionally keep only CleanCluster-clean segments
  4) convert the clean pool to segment / macro / subject level
  5) plug in dense.py and gnn.py encoders
  6) train either direct subject-level classifiers or MIL over segment/macro bags

This file intentionally does NOT replace your old LinkX runner. It is a modular
runner for experiments where you want to vary:
  - data level: segment | macro | subject
  - model family: node_only | connectivity_only | dense_dual_branch |
                  fixed_graph_gnn | fused_graph_bank_gnn | dual_branch_graph
  - node pooling and graph pooling separately

Expected to live next to your project modules:
  caueeg_loader_min.py, master_builder.py, mil_full_std.py, mil_utils.py,
  dense.py, gnn.py, metrics.py, utils_all.py or utils.py.
"""

import argparse
import copy
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
try:
    from caueeg_loader_min import load_caueeg_task_datasets
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import load_caueeg_task_datasets from caueeg_loader_min.py") from exc

try:
    from master_builder import build_master_eeg_dataset
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import build_master_eeg_dataset from master_builder.py") from exc

try:
    from mil_full_std import load_h5_payload_for_subjects
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import load_h5_payload_for_subjects from mil_full_std.py") from exc

try:
    from mil_utils import build_graphs_from_payload
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import build_graphs_from_payload from mil_utils.py") from exc

try:
    from utils_all import set_global_seed
except Exception:  # pragma: no cover
    try:
        from utils import set_seed as set_global_seed
    except Exception:
        def set_global_seed(seed: int) -> None:
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

try:
    from pipeline.dense import (
        ConnectivityOnlyCNN,
        ConnectivityOnlyMLP,
        DualBranchDenseModel,
        NodeOnlyMLP,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import dense.py model classes.") from exc

try:
    from pipeline.gnn import (
        DualBranchGraphModel,
        FusedGraphBankGNN,
        SimpleFixedGraphGNN,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError("Could not import gnn.py model classes.") from exc


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
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
BAD_SERIALS = {"00587", "00781", "01301", "train_00587", "train_00781", "train_01301"}


# -----------------------------------------------------------------------------
# Config dataclass
# -----------------------------------------------------------------------------
@dataclass
class CleanModularConfig:
    # data
    dataset_path: str
    out_h5: str
    task: str = "dementia"
    file_format: str = "feather"
    rebuild_h5: bool = False

    # feature/connectivity/graph construction
    feature_families: Tuple[str, ...] = ("relative_band_power", "statistical", "wavelet_energy")
    connectivity_metric: str = "wpli"
    connectivity_band: Optional[int] = 2
    topology: str = "fixed"
    standardize_features: bool = True

    # clean data pool
    segment_selection_strategy: str = "original_random_k"
    cleancluster_manifest_path: Optional[str] = None
    clean_k: int = 20
    base_k: Optional[int] = 10
    max_k_per_subject: int = 300
    apply_clean_to_eval: bool = False

    # level/aggregation
    level: str = "segment"                  # segment | macro | subject
    macro_duration_sec: float = 60.0
    aggregation: str = "gated_attention_mil" # none | mean_mil | gated_attention_mil

    # model
    model_family: str = "fixed_graph_gnn"    # node_only | connectivity_only | dense_dual_branch | fixed_graph_gnn | fused_graph_bank_gnn | dual_branch_graph
    connectivity_encoder_type: str = "cnn"   # cnn | mlp
    backbone: str = "gcn"                    # gcn | sage | gatv2
    node_pool: str = "flatten"               # dense: flatten/mean/max/sum; dual-graph: mean/max/add/attention/gated_attention/...
    graph_pool: str = "mean"                 # mean/max/add/mean_max_concat/mean_add_concat/attention/gated_attention
    hidden_dim: int = 64
    emb_dim: int = 64
    num_layers: int = 2
    gat_heads: int = 4
    dropout: float = 0.3
    use_edge_weight: bool = True
    use_batchnorm: bool = True
    fusion_mode: str = "concat"

    # graph bank
    bank_specs: Optional[List[Dict[str, Any]]] = None
    bank_fusion_mode: str = "summary_gated"   # static | summary_gated
    bank_topology_rule: str = "union"         # union | intersection | vote
    bank_vote_threshold: float = 0.5
    bank_fusion_temperature: float = 1.0
    bank_hidden_dim: int = 64

    # train
    seed: int = 42
    batch_size: int = 8
    epochs: int = 200
    lr: float = 3e-3
    weight_decay: float = 5e-3
    patience: int = 50
    start_epoch: int = 50
    min_delta: float = 1e-3
    use_lr_scheduler: bool = True
    scheduler_patience: int = 20
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6
    device: str = "cuda"
    output_root: str = "graph/results_caueeg_clean_modular"
    graph_node_pooling: str = "mean"
    graph_readout: str = "mean"
    dense_node_readout: str = "mean"
    graph_node_pool_ratio: float = 0.8
# -----------------------------------------------------------------------------
# Basic helpers from the old backbone
# -----------------------------------------------------------------------------
def dataset_to_subject_records_limited(
    dataset,
    *,
    limit: int = 5,
    bad_ids: Optional[set] = None,
):
    """
    Convert only the first `limit` valid CAUEEG recordings into subject records.

    This is useful for test-code/debug mode because it avoids iterating through
    the full train/val/test datasets.
    """
    bad_ids = set() if bad_ids is None else set(bad_ids)

    records = []
    subject_ids = []

    for sample in dataset:
        serial = str(sample["serial"])

        if serial in bad_ids:
            continue

        signal = sample["signal"]              # [21, T]
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal)

        if len(windows) == 0:
            continue

        rec = {
            "subject_id": serial,
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
            },
        }

        records.append(rec)
        subject_ids.append(serial)

        if len(subject_ids) >= int(limit):
            break

    if len(records) == 0:
        raise RuntimeError(
            f"test_code mode selected 0 valid records. "
            f"limit={limit}, bad_ids={bad_ids}"
        )

    return records, subject_ids
def _normalize_fixed_edges(
    fixed_edges: Optional[Sequence[Tuple[Any, Any]]],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set[tuple[int, int]]:
    if fixed_edges is None:
        return set()

    fixed_pairs: set[tuple[int, int]] = set()
    name_to_idx = None
    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(f"channel_names length {len(channel_names)} != n_channels={n_channels}")
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


def segment_recording(
    signal: np.ndarray,
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
) -> tuple[list[np.ndarray], list[int]]:
    """CAUEEG recording -> 19-channel windows."""
    x = np.asarray(signal, dtype=np.float32)
    x = x[:19]  # drop EKG + photic
    total_len = x.shape[-1]
    starts = list(range(latency, total_len - crop_len + 1, step))
    windows = [x[:, s:s + crop_len].astype(np.float32, copy=False) for s in starts]
    return windows, starts


def dataset_to_subject_records(dataset) -> tuple[list[dict[str, Any]], list[str]]:
    """Convert CauEegDataset split into records accepted by build_master_eeg_dataset()."""
    records: list[dict[str, Any]] = []
    subject_ids: list[str] = []

    for sample in dataset:
        signal = sample["signal"]
        serial = str(sample["serial"])
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal)
        if len(windows) == 0:
            continue

        records.append({
            "subject_id": serial,
            "label": label,
            "class_id": label,
            "sampling_rate": SFREQ,
            "channel_names": CAUEEG_EEG19,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": list(range(len(windows))),
            "recording_info": {"serial": serial, "age": age},
        })
        subject_ids.append(serial)

    return records, subject_ids


def collect_required_connectivity_metrics(
    bank_specs: Optional[Sequence[Mapping[str, Any]]],
    default_connectivity_metric: str,
) -> list[str]:
    metrics = {str(default_connectivity_metric)}
    if bank_specs is not None:
        for spec in bank_specs:
            metrics.add(str(spec.get("connectivity_metric", default_connectivity_metric)))
    return sorted(metrics)


def graph_key(g: Data) -> tuple[str, int]:
    return str(g.subject_id), int(g.segment_id)


def summarize_graph_pool(graphs: Sequence[Data], name: str) -> None:
    subject_to_count: dict[str, int] = defaultdict(int)
    label_to_subjects: dict[int, set[str]] = defaultdict(set)

    for g in graphs:
        sid = str(g.subject_id)
        y = int(g.y.view(-1)[0].item())
        subject_to_count[sid] += 1
        label_to_subjects[y].add(sid)

    counts = np.array(list(subject_to_count.values()), dtype=np.int64)
    print(f"\n[{name}]")
    print(f"num graphs: {len(graphs)}")
    print(f"num subjects: {len(subject_to_count)}")
    if len(counts) > 0:
        print(f"instances per subject: min={counts.min()}, mean={counts.mean():.2f}, max={counts.max()}")
    print("subjects per label:", {k: len(v) for k, v in label_to_subjects.items()})


# -----------------------------------------------------------------------------
# CleanCluster selection helpers
# -----------------------------------------------------------------------------
def load_cleancluster_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    required = {"subject_id", "segment_id", "keep_clean", "kmeans_cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CleanCluster manifest missing columns: {missing}")

    if df["keep_clean"].dtype != bool:
        df["keep_clean"] = df["keep_clean"].astype(str).str.lower().isin(["true", "1", "yes"])

    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df["kmeans_cluster_id"] = df["kmeans_cluster_id"].astype(int)
    return df


def filter_graphs_by_manifest_keep_clean(graphs: Sequence[Data], manifest_df: pd.DataFrame) -> list[Data]:
    clean_keys = set(
        manifest_df.loc[manifest_df["keep_clean"], ["subject_id", "segment_id"]]
        .itertuples(index=False, name=None)
    )
    out = [g for g in graphs if graph_key(g) in clean_keys]
    if len(out) == 0:
        raise RuntimeError("No graphs remain after CleanCluster filtering.")
    return out


def select_clean_kmeans_graphs_from_manifest(
    graphs: Sequence[Data],
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    distance_col: str = "kmeans_centroid_distance",
) -> list[Data]:
    graph_lookup = {graph_key(g): g for g in graphs}
    clean_df = manifest_df[manifest_df["keep_clean"]].copy()
    clean_df = clean_df[
        clean_df.apply(lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup, axis=1)
    ].copy()
    if len(clean_df) == 0:
        raise RuntimeError("No clean rows match the provided graphs.")

    selected_keys: list[tuple[str, int]] = []
    for sid, sdf in clean_df.groupby("subject_id"):
        chosen_rows = []
        for _, cdf in sdf.groupby("kmeans_cluster_id"):
            if distance_col in cdf.columns:
                row = cdf.sort_values(distance_col, ascending=True).iloc[0]
            elif "iforest_score" in cdf.columns:
                row = cdf.sort_values("iforest_score", ascending=False).iloc[0]
            else:
                row = cdf.sample(n=1, random_state=seed).iloc[0]
            chosen_rows.append(row)

        chosen_df = pd.DataFrame(chosen_rows)
        if len(chosen_df) > k:
            if "cluster_size" in chosen_df.columns:
                chosen_df = chosen_df.sort_values("cluster_size", ascending=False).head(k)
            else:
                chosen_df = chosen_df.sample(n=k, random_state=seed)

        if len(chosen_df) < k:
            chosen_pairs = set(zip(chosen_df["subject_id"].astype(str), chosen_df["segment_id"].astype(int)))
            remaining = sdf[
                ~sdf.apply(
                    lambda r: (str(r["subject_id"]), int(r["segment_id"])) in chosen_pairs,
                    axis=1,
                )
            ]
            need = k - len(chosen_df)
            if len(remaining) > 0:
                fill_df = remaining.sample(n=min(need, len(remaining)), random_state=seed)
                chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        for _, row in chosen_df.iterrows():
            selected_keys.append((str(row["subject_id"]), int(row["segment_id"])))

    out = [graph_lookup[key] for key in selected_keys if key in graph_lookup]
    if len(out) == 0:
        raise RuntimeError("No graphs selected by CleanCluster KMeans strategy.")
    return out


def weighted_sample_clean_graphs_from_manifest(
    graphs: Sequence[Data],
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    weight_col: str = "sampling_weight",
) -> list[Data]:
    rng = np.random.default_rng(seed)
    graph_lookup = {graph_key(g): g for g in graphs}
    df = manifest_df[manifest_df["keep_clean"]].copy()
    if weight_col not in df.columns:
        raise KeyError(f"Manifest missing weight column: {weight_col}")

    selected_graphs: list[Data] = []
    for _, sdf in df.groupby("subject_id"):
        sdf = sdf.copy()
        sdf = sdf[
            sdf.apply(lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup, axis=1)
        ]
        if len(sdf) == 0:
            continue

        n = min(k, len(sdf))
        weights = sdf[weight_col].to_numpy(dtype=np.float64)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 1e-8, None)
        probs = weights / weights.sum()
        chosen_pos = rng.choice(np.arange(len(sdf)), size=n, replace=False, p=probs)
        chosen = sdf.iloc[chosen_pos]
        for _, row in chosen.iterrows():
            selected_graphs.append(graph_lookup[(str(row["subject_id"]), int(row["segment_id"]))])

    if len(selected_graphs) == 0:
        raise RuntimeError("No graphs selected by weighted CleanCluster sampling.")
    return selected_graphs


def apply_segment_selection(
    graphs: list[Data],
    *,
    selection_strategy: str,
    manifest_df: Optional[pd.DataFrame],
    clean_k: int,
    seed: int,
) -> tuple[list[Data], str]:
    """
    Returns selected graphs and training bag sampling mode.
    mode='random_k' means the dataset will sample base_k each epoch.
    mode='fixed_all' means selected graphs are fixed and all used in each bag.
    """
    strategy = selection_strategy.lower()
    if strategy == "original_random_k":
        return graphs, "random_k"

    if manifest_df is None:
        raise ValueError(f"cleancluster_manifest_path is required for {selection_strategy}")

    if strategy == "clean_random_k":
        return filter_graphs_by_manifest_keep_clean(graphs, manifest_df), "random_k"

    if strategy == "clean_kmeans_k":
        return select_clean_kmeans_graphs_from_manifest(graphs, manifest_df, k=clean_k, seed=seed), "fixed_all"

    if strategy == "all_clean":
        return filter_graphs_by_manifest_keep_clean(graphs, manifest_df), "fixed_all"

    if strategy == "clean_weighted_k":
        return weighted_sample_clean_graphs_from_manifest(graphs, manifest_df, k=clean_k, seed=seed), "fixed_all"

    raise ValueError(
        f"Unknown segment_selection_strategy={selection_strategy!r}. "
        "Use one of: original_random_k, clean_random_k, clean_kmeans_k, all_clean, clean_weighted_k."
    )


# -----------------------------------------------------------------------------
# Graph construction helpers
# -----------------------------------------------------------------------------
def build_graph_bank_from_specs(
    payload: Mapping[str, Any],
    subject_ids: Sequence[str],
    *,
    feature_families: Sequence[str],
    default_connectivity_metric: str,
    default_connectivity_band: Optional[int],
    default_filter_method: str,
    default_fixed_edges: Optional[Sequence[Tuple[int, int]]],
    channel_names: Sequence[str],
    bank_specs: Sequence[Mapping[str, Any]],
    standardize_features: bool = True,
) -> tuple[list[Data], list[str]]:
    """Build repeated candidate graphs and attach adj_bank/topology_bank to base graphs."""
    if bank_specs is None or len(bank_specs) == 0:
        raise ValueError("bank_specs must contain at least one candidate.")

    candidate_names: list[str] = []
    candidate_graph_lists: list[list[Data]] = []

    for spec_idx, spec in enumerate(bank_specs):
        name = str(spec.get("name", f"cand_{spec_idx}"))
        cand_metric = spec.get("connectivity_metric", default_connectivity_metric)
        cand_band = spec["connectivity_band"] if "connectivity_band" in spec else default_connectivity_band
        cand_filter_method = spec.get("filter_method", default_filter_method)
        cand_fixed_edges = spec.get("fixed_edges", default_fixed_edges)

        gs = build_graphs_from_payload(
            payload,
            subject_ids,
            feature_families=feature_families,
            connectivity_metric=cand_metric,
            connectivity_band=cand_band,
            filter_method=cand_filter_method,
            fixed_edges=cand_fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=standardize_features,
        )
        candidate_names.append(name)
        candidate_graph_lists.append(gs)

    def _full_key(g: Data) -> tuple[str, int, int]:
        return (
            str(getattr(g, "subject_id", "")),
            int(getattr(g, "segment_id", -1)),
            int(getattr(g, "start_sample", -1)),
        )

    candidate_maps: list[dict[tuple[str, int, int], Data]] = []
    for gs in candidate_graph_lists:
        candidate_maps.append({_full_key(g): g for g in gs})

    base_graphs = candidate_graph_lists[0]
    for g in base_graphs:
        key = _full_key(g)
        bank_adj: list[torch.Tensor] = []
        bank_topo: list[torch.Tensor] = []
        for cand_name, gmap in zip(candidate_names, candidate_maps):
            if key not in gmap:
                raise KeyError(f"Graph key {key} missing in candidate {cand_name!r}.")
            gg = gmap[key]
            if not hasattr(gg, "adj"):
                raise ValueError(f"Candidate {cand_name!r} graph is missing dense adj.")
            adj = gg.adj.detach().cpu().float() if torch.is_tensor(gg.adj) else torch.tensor(gg.adj, dtype=torch.float32)
            bank_adj.append(adj)
            bank_topo.append((adj.abs() > 0).float())

        g.adj_bank = torch.stack(bank_adj, dim=0)          # [K, N, N]
        g.topology_bank = torch.stack(bank_topo, dim=0)    # [K, N, N]
        g.topology_names = list(candidate_names)
    return base_graphs, candidate_names


def ensure_graph_dense_attrs(g: Data) -> Data:
    """Ensure g.adj, g.edge_weight, and g.edge_attr exist and are consistent."""
    if not hasattr(g, "adj") or g.adj is None:
        n = int(g.x.shape[0])
        adj = torch.zeros((n, n), dtype=torch.float32)
        if hasattr(g, "edge_index") and g.edge_index is not None:
            ew = getattr(g, "edge_weight", None)
            if ew is None:
                ew = torch.ones(g.edge_index.shape[1], dtype=torch.float32)
            adj[g.edge_index[0].cpu(), g.edge_index[1].cpu()] = ew.detach().cpu().float()
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


def make_graph_from_dense(
    *,
    x: torch.Tensor,
    adj: torch.Tensor,
    y: int,
    subject_id: str,
    level: str,
    instance_id: str,
    segment_id: Optional[int] = None,
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None,
    adj_bank: Optional[torch.Tensor] = None,
    topology_bank: Optional[torch.Tensor] = None,
    topology_names: Optional[list[str]] = None,
) -> Data:
    adj = torch.nan_to_num(adj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)
    edge_index, edge_weight = dense_to_sparse(adj)

    g = Data(
        x=x.float(),
        edge_index=edge_index.long(),
        y=torch.tensor([int(y)], dtype=torch.long),
    )
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
    if end_sample is not None:
        g.end_sample = int(end_sample)
    if adj_bank is not None:
        g.adj_bank = adj_bank.float()
        g.topology_bank = (adj_bank.abs() > 0).float() if topology_bank is None else topology_bank.float()
        g.topology_names = list(topology_names or [f"cand_{i}" for i in range(adj_bank.shape[0])])
    return g


def reduce_stack(xs: Sequence[torch.Tensor], how: str = "mean") -> torch.Tensor:
    stack = torch.stack([x.float() for x in xs], dim=0)
    how = how.lower()
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
    raise ValueError(f"Unsupported reduce method: {how}")


def convert_segment_graphs_to_level(
    graphs: Sequence[Data],
    *,
    level: str,
    macro_duration_sec: float = 60.0,
    sfreq: float = SFREQ,
    reduce: str = "mean",
) -> list[Data]:
    """
    Convert clean segment graphs to segment/macro/subject instances.

    Important: call this AFTER cleaning/segment selection, so macro/subject
    summaries are built only from the selected pool.
    """
    level = level.lower()
    graphs = [ensure_graph_dense_attrs(copy.copy(g)) for g in graphs]

    if level == "segment":
        for g in graphs:
            g.level = "segment"
            g.instance_id = f"{g.subject_id}_seg{int(getattr(g, 'segment_id', 0))}"
        return list(graphs)

    grouped: dict[tuple[str, int], list[Data]] = defaultdict(list)
    macro_len_samples = max(int(round(macro_duration_sec * sfreq)), 1)

    for g in graphs:
        sid = str(g.subject_id)
        if level == "subject":
            group_id = 0
        elif level == "macro":
            start = int(getattr(g, "start_sample", 0))
            group_id = start // macro_len_samples
        else:
            raise ValueError("level must be one of: segment, macro, subject")
        grouped[(sid, group_id)].append(g)

    out: list[Data] = []
    for (sid, group_id), gs in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        y_values = [int(g.y.view(-1)[0].item()) for g in gs]
        if len(set(y_values)) != 1:
            raise ValueError(f"Mixed labels inside group {(sid, group_id)}: {set(y_values)}")
        y = y_values[0]

        x = reduce_stack([g.x.detach().cpu() for g in gs], how=reduce)
        adj = reduce_stack([g.adj.detach().cpu() for g in gs], how=reduce)

        # Average graph bank if present.
        adj_bank = None
        topology_bank = None
        topology_names = None
        if hasattr(gs[0], "adj_bank"):
            adj_bank = reduce_stack([g.adj_bank.detach().cpu() for g in gs], how=reduce)  # [K,N,N]
            topology_bank = (adj_bank.abs() > 0).float()
            topology_names = list(getattr(gs[0], "topology_names", [f"cand_{i}" for i in range(adj_bank.shape[0])]))

        starts = [int(getattr(g, "start_sample", 0)) for g in gs]
        segs = [int(getattr(g, "segment_id", -1)) for g in gs]
        instance_id = f"{sid}_{level}{group_id}"
        new_g = make_graph_from_dense(
            x=x,
            adj=adj,
            y=y,
            subject_id=sid,
            level=level,
            instance_id=instance_id,
            segment_id=min(segs) if segs else None,
            start_sample=min(starts) if starts else None,
            adj_bank=adj_bank,
            topology_bank=topology_bank,
            topology_names=topology_names,
        )
        new_g.num_source_segments = len(gs)
        new_g.source_segment_ids = segs
        out.append(new_g)

    return out


# -----------------------------------------------------------------------------
# Build clean graph pools
# -----------------------------------------------------------------------------
def build_segment_graph_pool(
    cfg: CleanModularConfig,
    *,
    fixed_edges: Optional[Sequence[Tuple[int, int]]],
    channel_names: Sequence[str],
    test_code: bool = False,
    test_n_subjects: int = 5,
) -> dict[str, Any]:
    """Build/reuse H5, load payload, build segment-level graph pool, then clean-select."""
    # # 1) Official split
    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=cfg.dataset_path,
        task=cfg.task,
        load_event=False,
        file_format=cfg.file_format,
        transform=None,
        verbose=False,
    )

    # # 2) Convert to H5 records
    # train_records, train_ids = dataset_to_subject_records(train_set)
    # val_records, val_ids = dataset_to_subject_records(val_set)
    # test_records, test_ids = dataset_to_subject_records(test_set)
    # all_records = train_records + val_records + test_records

    # 2) convert each recording into subject-like records
    # In test_code mode, only convert the first N valid subjects per split.
    if test_code:
        print(f"[TEST_CODE] Using first {test_n_subjects} valid subjects per split.")

        train_records, train_ids = dataset_to_subject_records_limited(
            train_set,
            limit=test_n_subjects,
            bad_ids=BAD_SERIALS,
        )
        val_records, val_ids = dataset_to_subject_records_limited(
            val_set,
            limit=test_n_subjects,
            bad_ids=BAD_SERIALS,
        )
        test_records, test_ids = dataset_to_subject_records_limited(
            test_set,
            limit=test_n_subjects,
            bad_ids=BAD_SERIALS,
        )
    else:
        train_records, train_ids = dataset_to_subject_records(train_set)
        val_records, val_ids = dataset_to_subject_records(val_set)
        test_records, test_ids = dataset_to_subject_records(test_set)

    all_records = train_records + val_records + test_records

    # In normal mode, this filters bad subjects.
    # In test_code mode, bad subjects were already skipped by dataset_to_subject_records_limited,
    # but keeping this is harmless.
    train_ids_filter = [sid for sid in train_ids if sid not in BAD_SERIALS]
    val_ids_filter   = [sid for sid in val_ids if sid not in BAD_SERIALS]
    test_ids_filter  = [sid for sid in test_ids if sid not in BAD_SERIALS]
    all_ids_filter   = train_ids_filter + val_ids_filter + test_ids_filter

    if test_code:
        print("[TEST_CODE] train_ids_filter:", train_ids_filter)
        print("[TEST_CODE] val_ids_filter:", val_ids_filter)
        print("[TEST_CODE] test_ids_filter:", test_ids_filter)


    num_classes = len(sorted({int(r["label"]) for r in all_records}))

    # 3) Build/reuse H5
    need_build = cfg.rebuild_h5 or (not os.path.isfile(cfg.out_h5))
    required_metrics = collect_required_connectivity_metrics(
        cfg.bank_specs if cfg.model_family in {"fused_graph_bank_gnn", "dual_branch_graph"} else None,
        cfg.connectivity_metric,
    )
    if need_build:
        print(f"[H5] Building master file: {cfg.out_h5}")
        build_master_eeg_dataset(
            subject_records=all_records,
            output_h5_path=cfg.out_h5,
            feature_families=list(cfg.feature_families),
            connectivity_metrics=required_metrics,
            overwrite=True,
            skip_bad_segments=False,
            target_sampling_rate=None,
            qc_input_unit="auto",
        )
    else:
        print(f"[H5] Reusing existing master file: {cfg.out_h5}")

    train_ids_suf = ["train_" + sid for sid in train_ids_filter]
    val_ids_suf = ["val_" + sid for sid in val_ids_filter]
    test_ids_suf = ["test_" + sid for sid in test_ids_filter]
    all_ids_suf = train_ids_suf + val_ids_suf + test_ids_suf

    # Do not preselect a band when graph bank candidates need different bands.
    payload_connectivity_band = None if cfg.bank_specs else cfg.connectivity_band

    payload = load_h5_payload_for_subjects(
        h5_path=cfg.out_h5,
        subject_ids=all_ids_suf,
        feature_families=list(cfg.feature_families),
        connectivity_metrics=required_metrics,
        connectivity_band=payload_connectivity_band,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    # 4) Build segment graphs
    if cfg.bank_specs and cfg.model_family in {"fused_graph_bank_gnn", "dual_branch_graph"}:
        train_graphs, topology_names = build_graph_bank_from_specs(
            payload,
            train_ids_suf,
            feature_families=cfg.feature_families,
            default_connectivity_metric=cfg.connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=cfg.topology,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=cfg.bank_specs,
            standardize_features=cfg.standardize_features,
        )
        val_graphs, _ = build_graph_bank_from_specs(
            payload,
            val_ids_suf,
            feature_families=cfg.feature_families,
            default_connectivity_metric=cfg.connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=cfg.topology,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=cfg.bank_specs,
            standardize_features=cfg.standardize_features,
        )
        test_graphs, _ = build_graph_bank_from_specs(
            payload,
            test_ids_suf,
            feature_families=cfg.feature_families,
            default_connectivity_metric=cfg.connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=cfg.topology,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=cfg.bank_specs,
            standardize_features=cfg.standardize_features,
        )
    else:
        topology_names = None
        train_graphs = build_graphs_from_payload(
            payload,
            train_ids_suf,
            feature_families=cfg.feature_families,
            connectivity_metric=cfg.connectivity_metric,
            connectivity_band=cfg.connectivity_band,
            filter_method=cfg.topology,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=cfg.standardize_features,
        )
        val_graphs = build_graphs_from_payload(
            payload,
            val_ids_suf,
            feature_families=cfg.feature_families,
            connectivity_metric=cfg.connectivity_metric,
            connectivity_band=cfg.connectivity_band,
            filter_method=cfg.topology,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=cfg.standardize_features,
        )
        test_graphs = build_graphs_from_payload(
            payload,
            test_ids_suf,
            feature_families=cfg.feature_families,
            connectivity_metric=cfg.connectivity_metric,
            connectivity_band=cfg.connectivity_band,
            filter_method=cfg.topology,
            fixed_edges=fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=cfg.standardize_features,
        )

    train_graphs = [ensure_graph_dense_attrs(g) for g in train_graphs]
    val_graphs = [ensure_graph_dense_attrs(g) for g in val_graphs]
    test_graphs = [ensure_graph_dense_attrs(g) for g in test_graphs]

    summarize_graph_pool(train_graphs, "train_graphs_original")
    summarize_graph_pool(val_graphs, "val_graphs_original")
    summarize_graph_pool(test_graphs, "test_graphs_original")

    # 5) Clean segment selection
    manifest_df = None
    if cfg.segment_selection_strategy.lower() != "original_random_k" or cfg.apply_clean_to_eval:
        if cfg.cleancluster_manifest_path is None:
            raise ValueError("cleancluster_manifest_path is required for clean segment selection.")
        manifest_df = load_cleancluster_manifest(cfg.cleancluster_manifest_path)

    train_graphs_selected, train_sample_mode = apply_segment_selection(
        train_graphs,
        selection_strategy=cfg.segment_selection_strategy,
        manifest_df=manifest_df,
        clean_k=cfg.clean_k,
        seed=cfg.seed,
    )

    if cfg.apply_clean_to_eval:
        assert manifest_df is not None
        val_graphs = filter_graphs_by_manifest_keep_clean(val_graphs, manifest_df)
        test_graphs = filter_graphs_by_manifest_keep_clean(test_graphs, manifest_df)

    summarize_graph_pool(train_graphs_selected, f"train_graphs_after_{cfg.segment_selection_strategy}")

    return {
        "train_graphs": train_graphs_selected,
        "val_graphs": val_graphs,
        "test_graphs": test_graphs,
        "train_sample_mode": train_sample_mode,
        "num_classes": num_classes,
        "topology_names": topology_names,
    }


# -----------------------------------------------------------------------------
# Datasets and collate
# -----------------------------------------------------------------------------
class FlatGraphDataset(Dataset):
    """One graph = one training/eval sample."""
    def __init__(self, graphs: Sequence[Data]):
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("FlatGraphDataset received no graphs.")
        self.num_nodes = int(self.graphs[0].x.shape[0])
        self.num_node_features = int(self.graphs[0].x.shape[1])

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]


class SubjectBagGraphDataset(Dataset):
    """One subject = one bag of segment/macro graphs."""
    def __init__(
        self,
        graphs: Sequence[Data],
        *,
        train: bool,
        sample_k: Optional[int] = None,
        max_k_per_subject: Optional[int] = None,
        seed: int = 42,
    ):
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("SubjectBagGraphDataset received no graphs.")
        self.train = bool(train)
        self.sample_k = sample_k
        self.max_k_per_subject = max_k_per_subject
        self.rng = np.random.default_rng(seed)

        self.subject_to_graphs: dict[str, list[Data]] = defaultdict(list)
        self.subject_labels: dict[str, int] = {}
        for g in self.graphs:
            sid = str(g.subject_id)
            y = int(g.y.view(-1)[0].item())
            self.subject_to_graphs[sid].append(g)
            if sid in self.subject_labels and self.subject_labels[sid] != y:
                raise ValueError(f"Subject {sid} has mixed labels.")
            self.subject_labels[sid] = y

        self.subject_ids = sorted(self.subject_to_graphs.keys())
        self.num_nodes = int(self.graphs[0].x.shape[0])
        self.num_node_features = int(self.graphs[0].x.shape[1])

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.subject_ids[idx]
        graphs = list(self.subject_to_graphs[sid])

        if self.max_k_per_subject is not None and len(graphs) > self.max_k_per_subject:
            pos = self.rng.choice(len(graphs), size=self.max_k_per_subject, replace=False)
            graphs = [graphs[int(i)] for i in pos]

        if self.train and self.sample_k is not None and len(graphs) > self.sample_k:
            pos = self.rng.choice(len(graphs), size=self.sample_k, replace=False)
            graphs = [graphs[int(i)] for i in pos]

        return {
            "subject_id": sid,
            "label": self.subject_labels[sid],
            "graphs": graphs,
        }


def _stack_optional_attr(graphs: Sequence[Data], attr: str) -> Optional[torch.Tensor]:
    if not hasattr(graphs[0], attr):
        return None
    vals = []
    for g in graphs:
        v = getattr(g, attr, None)
        if v is None:
            return None
        vals.append(v.detach().cpu().float() if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32))
    return torch.stack(vals, dim=0)


def collate_flat_graphs(graphs: Sequence[Data]) -> dict[str, Any]:
    graphs = [ensure_graph_dense_attrs(copy.copy(g)) for g in graphs]
    pyg_batch = Batch.from_data_list(graphs)
    node_features = torch.stack([g.x.float() for g in graphs], dim=0)       # [B,N,F]
    connectivity = torch.stack([g.adj.float() for g in graphs], dim=0)      # [B,N,N]
    labels = torch.tensor([int(g.y.view(-1)[0].item()) for g in graphs], dtype=torch.long)
    subject_ids = [str(g.subject_id) for g in graphs]
    instance_ids = [str(getattr(g, "instance_id", f"{g.subject_id}_{i}")) for i, g in enumerate(graphs)]

    adj_bank = _stack_optional_attr(graphs, "adj_bank")
    topology_bank = _stack_optional_attr(graphs, "topology_bank")

    return {
        "pyg_batch": pyg_batch,
        "node_features": node_features,
        "connectivity": connectivity.unsqueeze(1),  # [B,1,N,N]
        "adj_bank": adj_bank,
        "topology_bank": topology_bank,
        "labels": labels,
        "subject_ids": subject_ids,
        "instance_ids": instance_ids,
        "bag_indices": None,
    }


def collate_subject_bags(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    flat_graphs: list[Data] = []
    bag_indices: list[int] = []
    labels: list[int] = []
    subject_ids: list[str] = []

    for b, row in enumerate(rows):
        gs = list(row["graphs"])
        if len(gs) == 0:
            raise ValueError(f"Empty graph bag for subject {row['subject_id']}")
        flat_graphs.extend(gs)
        bag_indices.extend([b] * len(gs))
        labels.append(int(row["label"]))
        subject_ids.append(str(row["subject_id"]))

    batch = collate_flat_graphs(flat_graphs)
    batch["bag_indices"] = torch.tensor(bag_indices, dtype=torch.long)
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    batch["subject_ids"] = subject_ids
    batch["num_bags"] = len(rows)
    return batch


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and callable(getattr(v, "to")):
            try:
                out[k] = v.to(device)
            except Exception:
                out[k] = v
        else:
            out[k] = v
    return out


# -----------------------------------------------------------------------------
# Model wrappers
# -----------------------------------------------------------------------------
def normalize_dense_node_readout(pool: str) -> str:
    """
    Dense node-feature readout for NodeOnlyMLP / DualBranchDenseModel.
    This happens before an MLP and does not use graph edges.
    """
    p = str(pool).lower()
    if p == "add":
        p = "sum"
    if p not in {"flatten", "mean", "max", "sum"}:
        raise ValueError(
            f"Dense node readout supports flatten/mean/max/sum only, got {pool!r}."
        )
    return p


def normalize_graph_node_pooling(pooling: str) -> str:
    """
    Optional GNN node coarsening after message passing.
    This is where TopKPool and SAGPool belong.
    """
    p = str(pooling).lower()
    aliases = {
        "none": "none",
        "no": "none",
        "off": "none",
        "topk": "topk",
        "topkpool": "topk",
        "topk_pool": "topk",
        "sag": "sagpool",
        "sagpool": "sagpool",
        "sag_pool": "sagpool",
    }
    if p not in aliases:
        raise ValueError(
            f"Graph node pooling supports none/topk/sagpool only, got {pooling!r}."
        )
    return aliases[p]


def normalize_graph_readout(pool: str) -> str:
    """
    Final graph-level readout after GNN message passing / optional coarsening.
    """
    p = str(pool).lower()
    aliases = {
        "sum": "add",
        "add": "add",
        "mean": "mean",
        "max": "max",
        "mean_max": "mean_max_concat",
        "mean_max_concat": "mean_max_concat",
        "mean_add": "mean_add_concat",
        "mean_add_concat": "mean_add_concat",
        "attention": "attention",
        "attn": "attention",
        "gated": "gated_attention",
        "gated_attention": "gated_attention",
    }
    if p not in aliases:
        raise ValueError(
            "Graph readout supports mean/max/add/mean_max_concat/"
            f"mean_add_concat/attention/gated_attention, got {pool!r}."
        )
    return aliases[p]


def normalize_graph_pool(pool: str) -> str:
    p = pool.lower()
    alias = {
        "sum": "add",
        "mean_max": "mean_max_concat",
        "mean_add": "mean_add_concat",
        "gated": "gated_attention",
    }
    return alias.get(p, p)


class GatedMILPool(nn.Module):
    def __init__(self, emb_dim: int, attn_dim: int = 64):
        super().__init__()
        self.v = nn.Linear(emb_dim, attn_dim)
        self.u = nn.Linear(emb_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, z: torch.Tensor, bag_indices: torch.Tensor, num_bags: int) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.w(torch.tanh(self.v(z)) * torch.sigmoid(self.u(z))).squeeze(-1)  # [M]
        out = []
        attn_all = torch.zeros_like(scores)
        for b in range(num_bags):
            mask = bag_indices == b
            if not torch.any(mask):
                out.append(torch.zeros(z.shape[1], device=z.device, dtype=z.dtype))
                continue
            a = torch.softmax(scores[mask], dim=0)
            attn_all[mask] = a
            out.append(torch.sum(z[mask] * a.unsqueeze(-1), dim=0))
        return torch.stack(out, dim=0), attn_all


def mean_mil_pool(z: torch.Tensor, bag_indices: torch.Tensor, num_bags: int) -> torch.Tensor:
    out = []
    for b in range(num_bags):
        mask = bag_indices == b
        if torch.any(mask):
            out.append(z[mask].mean(dim=0))
        else:
            out.append(torch.zeros(z.shape[1], device=z.device, dtype=z.dtype))
    return torch.stack(out, dim=0)


class CleanModularModel(nn.Module):
    """
    Wraps dense.py or gnn.py base models and optionally adds subject-level MIL.
    """
    def __init__(
        self,
        *,
        base_model: nn.Module,
        model_family: str,
        embedding_dim: int,
        num_classes: int,
        aggregation: str = "none",
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.model_family = model_family
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.aggregation = aggregation.lower()
        if self.aggregation not in {"none", "mean_mil", "gated_attention_mil"}:
            raise ValueError("aggregation must be none, mean_mil, or gated_attention_mil")

        if self.aggregation == "gated_attention_mil":
            self.mil_pool = GatedMILPool(self.embedding_dim, attn_dim=attn_dim)
        else:
            self.mil_pool = None

        # We use this classifier after MIL pooling. For aggregation='none', use the base classifier.
        self.subject_classifier = nn.Linear(self.embedding_dim, self.num_classes)

    def _forward_base(self, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, Any]:
        family = self.model_family

        if family in {"node_only", "connectivity_only", "dense_dual_branch"}:
            out = self.base_model(
                node_features=batch["node_features"],
                connectivity=batch["connectivity"],
                return_dict=True,
            )
        elif family == "fixed_graph_gnn":
            out = self.base_model(batch["pyg_batch"], return_dict=True)
        elif family in {"fused_graph_bank_gnn", "dual_branch_graph"}:
            out = self.base_model(
                batch["pyg_batch"],
                adj_bank=batch.get("adj_bank", None),
                topology_bank=batch.get("topology_bank", None),
                return_dict=True,
            )
        else:
            raise ValueError(f"Unknown model_family={family!r}")

        logits = out.logits if hasattr(out, "logits") else out["logits"]
        emb = out.embedding if hasattr(out, "embedding") else out["embedding"]
        return logits, emb, out

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        inst_logits, inst_emb, raw_out = self._forward_base(batch)

        if self.aggregation == "none":
            return {
                "logits": inst_logits,
                "embedding": inst_emb,
                "instance_logits": inst_logits,
                "instance_embedding": inst_emb,
                "attention_weights": None,
                "raw_out": raw_out,
            }

        bag_indices = batch.get("bag_indices", None)
        if bag_indices is None:
            raise ValueError("MIL aggregation requires bag_indices in the batch.")
        num_bags = int(batch.get("num_bags", int(bag_indices.max().item()) + 1))

        if self.aggregation == "mean_mil":
            subj_emb = mean_mil_pool(inst_emb, bag_indices, num_bags)
            attn = None
        else:
            assert self.mil_pool is not None
            subj_emb, attn = self.mil_pool(inst_emb, bag_indices, num_bags)

        subj_logits = self.subject_classifier(subj_emb)
        return {
            "logits": subj_logits,
            "embedding": subj_emb,
            "instance_logits": inst_logits,
            "instance_embedding": inst_emb,
            "attention_weights": attn,
            "raw_out": raw_out,
        }


def build_base_model(
    *,
    cfg: CleanModularConfig,
    num_nodes: int,
    num_node_features: int,
    num_classes: int,
    num_candidates: Optional[int],
) -> tuple[nn.Module, int]:
    family = cfg.model_family.lower()
    emb_dim = int(cfg.emb_dim)
    graph_pool = normalize_graph_pool(cfg.graph_pool)
    graph_node_pooling = cfg.graph_node_pooling
    graph_readout = cfg.graph_readout
    dense_node_readout = cfg.dense_node_readout
    graph_node_pool_ratio = cfg.graph_node_pool_ratio

    if family == "node_only":
        model = NodeOnlyMLP(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_classes=num_classes,
            readout=normalize_dense_node_pool(cfg.node_pool),
            hidden_dims=(256, 128),
            emb_dim=emb_dim,
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
        )
        return model, emb_dim

    if family == "connectivity_only":
        if cfg.connectivity_encoder_type.lower() == "cnn":
            model = ConnectivityOnlyCNN(
                num_bands=1,
                num_classes=num_classes,
                emb_dim=emb_dim,
                dropout=cfg.dropout,
                use_batchnorm=cfg.use_batchnorm,
            )
        else:
            model = ConnectivityOnlyMLP(
                num_nodes=num_nodes,
                num_classes=num_classes,
                num_bands=1,
                emb_dim=emb_dim,
                dropout=cfg.dropout,
                use_batchnorm=cfg.use_batchnorm,
            )
        return model, emb_dim

    if family == "dense_dual_branch":
        model = DualBranchDenseModel(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_bands=1,
            node_readout=normalize_dense_node_pool(cfg.node_pool),
            node_emb_dim=emb_dim,
            connectivity_encoder_type=cfg.connectivity_encoder_type.lower(),
            connectivity_emb_dim=emb_dim,
            fusion_mode=cfg.fusion_mode,
            fusion_emb_dim=emb_dim,
            dropout=cfg.dropout,
            use_batchnorm=cfg.use_batchnorm,
        )
        return model, emb_dim


    if family == "fixed_graph_gnn":
        model = SimpleFixedGraphGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            backbone=cfg.backbone,
            hidden_dim=cfg.hidden_dim,
            graph_emb_dim=emb_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            gat_heads=cfg.gat_heads,
            use_edge_weight=cfg.use_edge_weight,
            use_batchnorm=cfg.use_batchnorm,
            node_pooling_type=graph_node_pooling,
            node_pool_ratio=graph_node_pool_ratio,
            readout_type=graph_readout,
            readout_hidden_dim=cfg.hidden_dim,
            readout_dropout=cfg.dropout,
            return_attention_weights=False,
        )
        return model, emb_dim

    if family == "fused_graph_bank_gnn":
        if not num_candidates:
            raise ValueError("fused_graph_bank_gnn requires bank_specs / num_candidates.")
        model = FusedGraphBankGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_candidates=int(num_candidates),
            backbone=cfg.backbone,
            hidden_dim=cfg.hidden_dim,
            graph_emb_dim=emb_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            gat_heads=cfg.gat_heads,
            use_edge_weight=cfg.use_edge_weight,
            use_batchnorm=cfg.use_batchnorm,
            node_pooling_type=graph_node_pooling,
            node_pool_ratio=graph_node_pool_ratio,
            readout_type=graph_readout,
            readout_hidden_dim=cfg.hidden_dim,
            readout_dropout=cfg.dropout,
            fusion_mode=cfg.bank_fusion_mode,
            topology_rule=cfg.bank_topology_rule,
            vote_threshold=cfg.bank_vote_threshold,
            fusion_temperature=cfg.bank_fusion_temperature,
            fusion_hidden_dim=cfg.bank_hidden_dim,
        )
        return model, emb_dim

    if family == "dual_branch_graph":
        use_graph_bank = bool(num_candidates)
        model = DualBranchGraphModel(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes if use_graph_bank else None,
            use_graph_bank=use_graph_bank,
            num_candidates=int(num_candidates) if use_graph_bank else None,
            backbone=cfg.backbone,
            hidden_dim=cfg.hidden_dim,
            graph_emb_dim=emb_dim,
            node_emb_dim=emb_dim,
            fusion_emb_dim=emb_dim,
            num_layers=cfg.num_layers,
            # dropout=cfg.dropout,
            gat_heads=cfg.gat_heads,
            use_edge_weight=cfg.use_edge_weight,
            use_batchnorm=cfg.use_batchnorm,
            node_readout_type=graph_readout,
            graph_readout_type=graph_readout,
            readout_hidden_dim=cfg.hidden_dim,
            readout_dropout=cfg.dropout,
            fusion_mode=cfg.fusion_mode,
            graph_bank_fusion_mode=cfg.bank_fusion_mode,
            topology_rule=cfg.bank_topology_rule,
            vote_threshold=cfg.bank_vote_threshold,
            fusion_temperature=cfg.bank_fusion_temperature,
            graph_bank_hidden_dim=cfg.bank_hidden_dim,
            return_attention_weights=False,
            node_hidden_dims= (emb_dim, cfg.hidden_dim),
            node_dropout= 0.2,
            graph_dropout = 0.2,
            node_pooling_type = graph_node_pooling,
            node_pool_ratio = graph_node_pool_ratio,
            fusion_dropout = 0.2,
        )
        return model, emb_dim

    raise ValueError(f"Unknown model_family={cfg.model_family!r}")


def make_loaders(
    *,
    train_graphs: Sequence[Data],
    val_graphs: Sequence[Data],
    test_graphs: Sequence[Data],
    cfg: CleanModularConfig,
    train_sample_mode: str,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, int]]:
    level = cfg.level.lower()
    aggregation = cfg.aggregation.lower()
    if level == "subject":
        aggregation = "none"

    if aggregation == "none":
        train_ds = FlatGraphDataset(train_graphs)
        val_ds = FlatGraphDataset(val_graphs)
        test_ds = FlatGraphDataset(test_graphs)
        collate_fn = collate_flat_graphs
    else:
        sample_k = cfg.base_k if train_sample_mode == "random_k" else None
        train_ds = SubjectBagGraphDataset(
            train_graphs,
            train=True,
            sample_k=sample_k,
            max_k_per_subject=cfg.max_k_per_subject,
            seed=cfg.seed,
        )
        val_ds = SubjectBagGraphDataset(val_graphs, train=False, sample_k=None, seed=cfg.seed)
        test_ds = SubjectBagGraphDataset(test_graphs, train=False, sample_k=None, seed=cfg.seed)
        collate_fn = collate_subject_bags

    info = {
        "num_nodes": int(train_ds.num_nodes),
        "num_node_features": int(train_ds.num_node_features),
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, info


# -----------------------------------------------------------------------------
# Train/evaluate
# -----------------------------------------------------------------------------
def summarize_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(y_prob.shape[1]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "conf_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    losses = []
    probs_all = []
    y_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        logits = out["logits"]
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        probs_all.append(torch.softmax(logits.detach(), dim=-1).cpu().numpy())
        y_all.append(labels.detach().cpu().numpy())

    y_true = np.concatenate(y_all, axis=0)
    y_prob = np.concatenate(probs_all, axis=0)
    metrics = summarize_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    return metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses = []
    probs_all = []
    y_all = []
    subject_ids_all: list[str] = []
    emb_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["labels"]
        out = model(batch)
        logits = out["logits"]
        loss = criterion(logits, labels)

        losses.append(float(loss.detach().cpu().item()))
        probs_all.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
        y_all.append(labels.detach().cpu().numpy())
        emb_all.append(out["embedding"].detach().cpu().numpy())
        subject_ids_all.extend([str(x) for x in batch.get("subject_ids", [])])

    y_true = np.concatenate(y_all, axis=0)
    y_prob = np.concatenate(probs_all, axis=0)
    metrics = summarize_metrics(y_true, y_prob)
    metrics["loss"] = float(np.mean(losses))
    metrics["subject_ids"] = subject_ids_all
    metrics["embedding"] = np.concatenate(emb_all, axis=0).tolist() if emb_all else []
    return metrics


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    cfg: CleanModularConfig,
    run_dir: str,
) -> tuple[nn.Module, dict[str, Any], list[dict[str, Any]], dict[str, torch.Tensor]]:
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = None
    if cfg.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience,
            threshold=cfg.min_delta,
            min_lr=cfg.scheduler_min_lr,
        )

    best_state = copy.deepcopy(model.state_dict())
    best_score = -np.inf
    best_metrics: dict[str, Any] = {}
    no_improve = 0
    history: list[dict[str, Any]] = []
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, cfg.epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate_model(model, val_loader, criterion, device)

        if scheduler is not None and epoch >= 10:
            scheduler.step(val_m["loss"])

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_m["loss"],
            "train_accuracy": train_m["accuracy"],
            "train_balanced_accuracy": train_m["balanced_accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "val_loss": val_m["loss"],
            "val_accuracy": val_m["accuracy"],
            "val_balanced_accuracy": val_m["balanced_accuracy"],
            "val_macro_f1": val_m["macro_f1"],
        }
        history.append(row)

        # Prefer balanced accuracy, then macro-F1, after warmup.
        score = float(val_m["balanced_accuracy"] + 1e-4 * val_m["macro_f1"])
        improved = epoch >= cfg.start_epoch and score > best_score + cfg.min_delta
        if improved:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = val_m
            no_improve = 0
            torch.save({"model_state_dict": best_state, "epoch": epoch, "val_metrics": val_m}, ckpt_path)
        elif epoch >= cfg.start_epoch:
            no_improve += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_m['loss']:.4f} bal {train_m['balanced_accuracy']:.4f} f1 {train_m['macro_f1']:.4f} | "
            f"val loss {val_m['loss']:.4f} bal {val_m['balanced_accuracy']:.4f} f1 {val_m['macro_f1']:.4f} | "
            f"no_improve {no_improve}"
        )

        if epoch >= cfg.start_epoch and no_improve >= cfg.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model, best_metrics, history, best_state


# -----------------------------------------------------------------------------
# Save/plot helpers
# -----------------------------------------------------------------------------
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


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(obj), f, indent=2)


def save_history_csv(history: list[dict[str, Any]], csv_path: str) -> pd.DataFrame:
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)
    # print(f"Saved history: {csv_path}")
    return df


def save_summary_metrics_csv(summary_rows: Sequence[Mapping[str, Any]], csv_path: str) -> pd.DataFrame:
    rows = []
    for row in summary_rows:
        r = make_jsonable(dict(row))
        if "conf_matrix" in r:
            r["confusion_matrix_json"] = json.dumps(r["conf_matrix"])
            del r["conf_matrix"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    # print(f"Saved summary metrics: {csv_path}")
    return df


def save_predictions_csv(metrics: Mapping[str, Any], csv_path: str, num_classes: int) -> pd.DataFrame:
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    y_prob = np.asarray(metrics["y_prob"], dtype=float)
    subject_ids = list(metrics.get("subject_ids", [f"sample_{i}" for i in range(len(y_true))]))
    emb = metrics.get("embedding", None)

    rows = []
    for i in range(len(y_true)):
        rec = {
            "subject_id": subject_ids[i] if i < len(subject_ids) else f"sample_{i}",
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
        }
        for c in range(num_classes):
            rec[f"prob_{c}"] = float(y_prob[i, c])
        if emb is not None and i < len(emb):
            rec["embedding_json"] = json.dumps(make_jsonable(emb[i]))
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    # print(f"Saved predictions: {csv_path}")
    return df


def _safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))


def plot_baseline_style(metrics: Mapping[str, Any], class_names: Sequence[str], output_dir: str, prefix: str = "test") -> None:
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    y_prob = np.asarray(metrics["y_prob"], dtype=float)
    num_classes = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_norm = _safe_divide(cm, cm.sum(axis=1, keepdims=True))

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion.png"), dpi=300, bbox_inches="tight")
    plt.close()

    if num_classes >= 2:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        plt.figure(figsize=(6, 5))
        for c in range(num_classes):
            # roc_curve can fail if the class is absent in this split.
            try:
                fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")
            except Exception:
                continue
        try:
            fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            auc_micro = auc(fpr_micro, tpr_micro)
            plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro-average (AUC={auc_micro:.3f})")
        except Exception:
            pass
        plt.plot([0, 1], [0, 1], linestyle=":")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_roc_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()


def save_seed_aggregation(summary_rows: Sequence[Mapping[str, Any]], output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for row in summary_rows:
        r = make_jsonable(dict(row))
        if isinstance(r.get("feature_families"), list):
            r["feature_families"] = ",".join(map(str, r["feature_families"]))
        if "conf_matrix" in r:
            r["confusion_matrix_json"] = json.dumps(r["conf_matrix"])
            del r["conf_matrix"]
        rows.append(r)

    df = pd.DataFrame(rows)
    raw_path = os.path.join(output_dir, "all_seed_results.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "macro_f1"] if c in df.columns]
    variant_cols = [
        "encoder_type", "training_approach", "mil_pool_type", 
        "feature_families", "topology", "connectivity_metric", "connectivity_band",
        "segment_selection_strategy", "graph_bank"
        "base_k", "batch_size", "lr",
        "dropout", "weight_decay", "emb_dim", 

        "node_pool", "graph_pool",
        "bank_fusion_mode", "bank_topology_rule", "graph_node_pooling", 
        "dense_node_readout","graph_node_pool_ratio", "seed",
    ]
    # Do not group by seed for aggregate across seeds.
    variant_cols = [c for c in variant_cols if c in df.columns and c != "seed"]

    agg = df.groupby(variant_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
    agg.columns = [col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in agg.columns]
    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_mean_std"] = agg.apply(
                lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}" if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
                axis=1,
            )
    agg_path = os.path.join(output_dir, "aggregate_seed_results.csv")
    agg.to_csv(agg_path, index=False)
    # print(f"Saved per-seed results: {raw_path}")
    print(f"Saved aggregate results: {agg_path}")
    return df, agg


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def run_one_seed(
    cfg: CleanModularConfig,
    *,
    fixed_edges: Optional[Sequence[Tuple[int, int]]],
    channel_names: Sequence[str] = CAUEEG_EEG19,
    test_code: bool = False,
    test_n_subjects: int = 5,
) -> dict[str, Any]:
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
    cfg.device = str(device)

    run_name = (
        f"seed{cfg.seed}_{cfg.level}_{cfg.model_family}_{cfg.backbone}_"
        f"npool{cfg.node_pool}_gpool{cfg.graph_pool}_{cfg.aggregation}"
    )
    run_dir = os.path.join(cfg.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    save_json(asdict(cfg), os.path.join(run_dir, "config.json"))

    pool = build_segment_graph_pool(cfg, fixed_edges=fixed_edges, channel_names=channel_names, test_code=test_code, test_n_subjects=test_n_subjects)

    train_graphs = convert_segment_graphs_to_level(
        pool["train_graphs"], level=cfg.level, macro_duration_sec=cfg.macro_duration_sec
    )
    val_graphs = convert_segment_graphs_to_level(
        pool["val_graphs"], level=cfg.level, macro_duration_sec=cfg.macro_duration_sec
    )
    test_graphs = convert_segment_graphs_to_level(
        pool["test_graphs"], level=cfg.level, macro_duration_sec=cfg.macro_duration_sec
    )

    summarize_graph_pool(train_graphs, f"train_{cfg.level}_instances")
    summarize_graph_pool(val_graphs, f"val_{cfg.level}_instances")
    summarize_graph_pool(test_graphs, f"test_{cfg.level}_instances")

    effective_cfg = copy.deepcopy(cfg)
    if effective_cfg.level.lower() == "subject":
        effective_cfg.aggregation = "none"

    train_loader, val_loader, test_loader, data_info = make_loaders(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        test_graphs=test_graphs,
        cfg=effective_cfg,
        train_sample_mode=pool["train_sample_mode"],
    )

    num_candidates = None
    if cfg.bank_specs and cfg.model_family in {"fused_graph_bank_gnn", "dual_branch_graph"}:
        num_candidates = len(cfg.bank_specs)

    base_model, emb_dim = build_base_model(
        cfg=effective_cfg,
        num_nodes=data_info["num_nodes"],
        num_node_features=data_info["num_node_features"],
        num_classes=pool["num_classes"],
        num_candidates=num_candidates,
    )
    model = CleanModularModel(
        base_model=base_model,
        model_family=cfg.model_family,
        embedding_dim=emb_dim,
        num_classes=pool["num_classes"],
        aggregation=effective_cfg.aggregation,
        attn_dim=cfg.hidden_dim,
    )

    model, val_best, history, best_state = fit_model(
        model,
        train_loader,
        val_loader,
        cfg=effective_cfg,
        run_dir=run_dir,
    )

    criterion = nn.CrossEntropyLoss()
    train_metrics = evaluate_model(model, train_loader, criterion, device)
    val_metrics = evaluate_model(model, val_loader, criterion, device)
    test_metrics = evaluate_model(model, test_loader, criterion, device)

    class_names = [f"class_{i}" for i in range(pool["num_classes"])]
    if cfg.task == "abnormal" and pool["num_classes"] == 2:
        class_names = ["normal", "abnormal"]
    elif cfg.task == "dementia" and pool["num_classes"] == 3:
        class_names = ["normal", "mci", "dementia"]

    plot_baseline_style(test_metrics, class_names, run_dir, prefix="test")
    save_history_csv(history, os.path.join(run_dir, "history.csv"))
    save_predictions_csv(val_metrics, os.path.join(run_dir, "val_predictions.csv"), num_classes=pool["num_classes"])
    save_predictions_csv(test_metrics, os.path.join(run_dir, "test_predictions.csv"), num_classes=pool["num_classes"])

    summary_rows = [
        {"split": "train", **{k: train_metrics[k] for k in ["loss", "accuracy", "balanced_accuracy", "macro_f1", "conf_matrix"]}},
        {"split": "val", **{k: val_metrics[k] for k in ["loss", "accuracy", "balanced_accuracy", "macro_f1", "conf_matrix"]}},
        {"split": "test", **{k: test_metrics[k] for k in ["loss", "accuracy", "balanced_accuracy", "macro_f1", "conf_matrix"]}},
    ]
    save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))

    summary_test = {
        "seed": cfg.seed,
        "encoder_type": str(cfg.model_family) + "_" + str(cfg.backbone),
        "training_approach": "MIL-subject (" + str(cfg.level) + ")",
        "mil_pool_type": effective_cfg.aggregation,
        "feature_families": list(cfg.feature_families),
        "topology": cfg.topology,
        "connectivity_metric": cfg.connectivity_metric,
        "connectivity_band": cfg.connectivity_band,
        "segment_selection_strategy": cfg.segment_selection_strategy,
        "graph_bank":  cfg.connectivity_metric,
        # "backbone": cfg.backbone,
        "node_pool": cfg.node_pool,
        "graph_pool": cfg.graph_pool,
        "bank_fusion_mode": cfg.bank_fusion_mode,
        "bank_topology_rule": cfg.bank_topology_rule,
        "graph_node_pooling" : graph_node_pooling,
        "dense_node_readout": dense_node_readout ,
        "graph_node_pool_ratio": graph_node_pool_ratio,

        "base_k": cfg.base_k,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "patience": cfg.patience,
        "start_epoch": cfg.start_epoch,
        "lr": cfg.lr,
        "dropout": cfg.dropout,
        "weight_decay": cfg.weight_decay,
        "use_lr_scheduler": cfg.use_lr_scheduler,
        "emb_dim": cfg.emb_dim,
        "hidden_dim": cfg.hidden_dim,
        "accuracy": float(test_metrics["accuracy"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "conf_matrix": test_metrics["conf_matrix"],
        "run_dir": run_dir,
    }
    save_summary_metrics_csv([summary_test], os.path.join(run_dir, "summary_test.csv"))

    return {
        "model": model,
        "run_dir": run_dir,
        "history": history,
        "best_state": best_state,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "summary_test": [summary_test],
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def nullable_int(val: str) -> Optional[int]:
    if val is None:
        return None
    if str(val).lower() == "none":
        return None
    return int(val)


def parse_bank_specs(s: Optional[str]) -> Optional[list[dict[str, Any]]]:
    if s is None or str(s).strip() == "":
        return None
    s = str(s)
    if os.path.isfile(s):
        with open(s, "r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(s)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CleanCluster + modular CAUEEG dense/GNN runner")

    parser.add_argument("--dataset_path", type=str, default="/home/anphan/Downloads/caueeg-dataset/")
    parser.add_argument("--out_h5", type=str, default="/home/anphan/Documents/caueeg_randomcrop_master_dementia_seed42.h5")
    parser.add_argument("--task", type=str, default="dementia")
    parser.add_argument("--file_format", type=str, default="edf")
    parser.add_argument("--rebuild_h5", action="store_true")

    parser.add_argument("--feature_families_str", type=str, default="relative_band_power,statistical")
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=nullable_int, default=2)
    parser.add_argument("--topology", type=str, default="fixed")

    parser.add_argument("--segment_selection_strategy", type=str, default="clean_weighted_k",
                        choices=["original_random_k", "clean_random_k", "clean_kmeans_k", "all_clean", "clean_weighted_k"])
    parser.add_argument("--cleancluster_manifest_path", type=str, default="/home/anphan/Documents/CAUEEG/visualize/segment_selection/cleancluster/cleancluster_manifest.csv")
    parser.add_argument("--clean_k", type=int, default=20)
    parser.add_argument("--base_k", type=nullable_int, default=10)
    parser.add_argument("--apply_clean_to_eval", action="store_true")

    parser.add_argument("--level", type=str, default="segment", choices=["segment", "macro", "subject"])
    parser.add_argument("--macro_duration_sec", type=float, default=60.0)
    parser.add_argument("--aggregation", type=str, default="mean_mil",
                        choices=["none", "mean_mil", "gated_attention_mil"])

    parser.add_argument("--model_family", type=str, default="fixed_graph_gnn",
                        choices=["node_only", "connectivity_only", "dense_dual_branch", "fixed_graph_gnn", "fused_graph_bank_gnn", "dual_branch_graph"])
    parser.add_argument("--connectivity_encoder_type", type=str, default="cnn", choices=["cnn", "mlp"])
    parser.add_argument("--backbone", type=str, default="gatv2", choices=["gcn", "sage", "gatv2"])
    parser.add_argument("--node_pool", type=str, default="mean")
    # parser.add_argument("--graph_pool", type=str, default="mean")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fusion_mode", type=str, default="concat", choices=["concat", "gated"])

    # parser.add_argument("--bank_specs_json", type=str, default=None,
                        # help="Either a JSON string or a path to a JSON file containing graph-bank specs.")
    parser.add_argument("--bank_fusion_mode", type=str, default="summary_gated", choices=["static", "summary_gated"])
    parser.add_argument("--bank_topology_rule", type=str, default="vote", choices=["union", "intersection", "vote"])
    parser.add_argument("--bank_vote_threshold", type=float, default=0.5)
    parser.add_argument("--bank_fusion_temperature", type=float, default=1.0)
    parser.add_argument("--bank_hidden_dim", type=int, default=64)

    parser.add_argument("--seeds", type=str, default="15,42,100")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--start_epoch", type=int, default=50)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--no_lr_scheduler", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_root", type=str, default="/home/anphan/Documents/EEG_Project/CAUEEG/result_clean_modular")
    parser.add_argument(
        "--dense_node_pool",
        type=str,
        default="flatten",
        choices=["flatten", "mean", "max", "sum"],
    )

    parser.add_argument(
        "--graph_node_pooling",
        type=str,
        default="sagpool",
        choices=["none", "topk", "sagpool"],
    )

    parser.add_argument(
        "--graph_node_pool_ratio",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--graph_pool",
        type=str,
        default="mean_max_concat",
        choices=[
            "mean",
            "max",
            "add",
            "sum",
            "mean_max_concat",
            "mean_add_concat",
            "attention",
            "gated_attention",
        ],
    )


    parser.add_argument(
        "--test_code",
        action="store_true",
        help="Debug mode: use only first N valid subjects from each split.",
    )

    parser.add_argument(
        "--test_n_subjects",
        type=int,
        default=5,
        help="Number of valid subjects per split to use in --test_code mode.",
    )
    return parser.parse_args()

def find_existing_run_ignore_timestamp(output_base: str, run_core: str):
    """
    Return an existing run folder whose name matches run_core
    after ignoring the timestamp prefix.

    Example:
        existing folder:
        20260429_120530_dementia_segment_clean_weighted_k_wpli_k10_...

        run_core:
        dementia_segment_clean_weighted_k_wpli_k10_...
    """
    base = Path(output_base)
    if not base.exists():
        return None

    suffix = "_" + run_core

    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue

        if p.name == run_core or p.name.endswith(suffix):
            return p

    return None
    
def main() -> None:
    args = parse_args()

    # Fixed edges from your config.py if available.
    try:
        import config
        raw_fixed_edges = getattr(config, "MONOFIXEDGES", None)
    except Exception:
        raw_fixed_edges = None
    fixed_edges = _normalize_fixed_edges(raw_fixed_edges, n_channels=19, channel_names=CAUEEG_EEG19)

    feature_families = tuple(x.strip() for x in args.feature_families_str.split(",") if x.strip())
    # bank_specs = parse_bank_specs(args.bank_specs_json)
    bank_specs = [
        {"name": "wpli_theta_full", 
        "connectivity_metric": "wpli", 
        "connectivity_band": 1, 
        "filter_method": "full"},
        {"name": "wpli_alpha_full", 
        "connectivity_metric": "wpli", 
        "connectivity_band": 2, 
        "filter_method": "full"},
        {"name": "coherence_alpha_full", 
        "connectivity_metric": "coherence", 
        "connectivity_band": 2, 
        "filter_method": "full"},
        {"name": "coherence_theta_full", 
        "connectivity_metric": "coherence", 
        "connectivity_band": 1, 
        "filter_method": "full"},
    ]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    k_tag = f"k{args.base_k}" if args.segment_selection_strategy != "all_clean" else "allclean"
    run_name = (
        f"{args.task}_{args.level}_{args.segment_selection_strategy}_"
        f"{args.connectivity_metric}_{k_tag}_{args.model_family}_{args.backbone}_"
        f"npool{args.node_pool}_gpool{args.graph_pool}_{args.aggregation}_{args.topology}"
    )


    existing_run = find_existing_run_ignore_timestamp(args.output_root, run_name)

    if existing_run is not None:
        print("=" * 80)
        print("[SKIP] Existing run found. Skip this run and move to next bash-loop item.")
        print(f"[SKIP] Existing folder: {existing_run}")
        print(f"[SKIP] Run core: {run_core}")
        print("=" * 80)
        sys.exit(0)

    output_root = os.path.join(args.output_root, f"{timestamp}_{run_name}")
    os.makedirs(output_root, exist_ok=True)

    use_lr_scheduler = args.use_lr_scheduler and not args.no_lr_scheduler

    agg_seed_results: list[dict[str, Any]] = []
    for seed in seeds:
        cfg = CleanModularConfig(
            dataset_path=args.dataset_path,
            out_h5=args.out_h5,
            task=args.task,
            file_format=args.file_format,
            rebuild_h5=args.rebuild_h5,
            feature_families=feature_families,
            connectivity_metric=args.connectivity_metric,
            connectivity_band=args.connectivity_band,
            topology=args.topology,
            segment_selection_strategy=args.segment_selection_strategy,
            cleancluster_manifest_path=args.cleancluster_manifest_path,
            clean_k=args.clean_k,
            base_k=args.base_k,
            apply_clean_to_eval=args.apply_clean_to_eval,
            level=args.level,
            macro_duration_sec=args.macro_duration_sec,
            aggregation=args.aggregation,
            model_family=args.model_family,
            connectivity_encoder_type=args.connectivity_encoder_type,
            backbone=args.backbone,
            node_pool=args.node_pool,
            graph_pool=args.graph_pool,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            num_layers=args.num_layers,
            gat_heads=args.gat_heads,
            dropout=args.dropout,
            fusion_mode=args.fusion_mode,
            bank_specs=bank_specs,
            bank_fusion_mode=args.bank_fusion_mode,
            bank_topology_rule=args.bank_topology_rule,
            bank_vote_threshold=args.bank_vote_threshold,
            bank_fusion_temperature=args.bank_fusion_temperature,
            bank_hidden_dim=args.bank_hidden_dim,
            seed=seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            start_epoch=args.start_epoch,
            use_lr_scheduler=use_lr_scheduler,
            device=args.device,
            output_root=output_root,
            graph_node_pooling = normalize_graph_node_pooling(args.graph_node_pooling),
            graph_readout = normalize_graph_readout(args.graph_pool),
            dense_node_readout = normalize_dense_node_readout(args.dense_node_pool),
            graph_node_pool_ratio = args.graph_node_pool_ratio
        )
        out = run_one_seed(cfg, fixed_edges=fixed_edges, channel_names=CAUEEG_EEG19, test_code=args.test_code,test_n_subjects=6)
        agg_seed_results.extend(out["summary_test"])

    agg_dir = os.path.join(output_root, "agg_seed_results")
    _, agg_df = save_seed_aggregation(agg_seed_results, output_dir=agg_dir)

    print("run_name", run_name)
    print("\nAggregate across seeds:")
    cols = [c for c in ["accuracy_mean_std", "balanced_accuracy_mean_std", "macro_f1_mean_std"] if c in agg_df.columns]
    if cols:
        print(agg_df[cols])
    else:
        print(agg_df)


if __name__ == "__main__":
    main()
