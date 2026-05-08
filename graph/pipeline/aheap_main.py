from __future__ import annotations

"""
AHEAP-specific H5-first experiment runner and experiment-ladder builder.

Why this file exists
--------------------
The generic `main.py` is already close to a reusable H5 runner, but for the
AHEAP study we want a runner that is explicitly organized around the main design
axes of your paper/project:

1. graph construction level: segment / macro / subject
2. representation source / model family
3. topology strategy
4. edge-weight strategy
5. connectivity source (metric / band / multiband / graph-bank)
6. subject aggregation strategy
7. graph-level readout / pooling

This module keeps those axes explicit and also fixes two practical issues that
matter for real experiments:
- it is AHEAP-only and therefore simpler to reason about
- trainable MIL aggregation heads are stored inside the model wrapper, instead
  of being recreated inside a stateless helper on every forward pass

The code intentionally reuses the source modules that already exist in the
project:
- master_builder.py
- dense.py
- gnn.py
- models_mil.py
- trainer.py
- evaluate.py
- utils.py

Typical usage
-------------
1) Build a ladder:
    ladder = build_default_aheap_ladder(
        h5_path="/path/to/master.h5",
        output_root="/path/to/results",
    )

2) Run one ladder item:
    run_aheap_ladder_item(ladder[0])

3) Run the full ladder:
    for spec in ladder:
        run_aheap_ladder_item(spec)
"""

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dense import (  # noqa: E402
    ConnectivityOnlyCNN,
    ConnectivityOnlyMLP,
    DualBranchDenseModel,
    NodeOnlyMLP,
)
from evaluate import (  # noqa: E402
    save_predictions_csv,
    save_summary_json,
    summarize_cv_results,
    summarize_fold_results,
)
from master_builder import list_available_groups, load_selected_groups  # noqa: E402
from models_mil import (  # noqa: E402
    AttentionMILPool,
    GatedAttentionMILPool,
    SubjectFusionHead,
    aggregate_subject_predictions,
)
from trainer import Trainer  # noqa: E402
from utils import ensure_dir, get_device, load_yaml_config, set_seed  # noqa: E402

try:  # noqa: E402
    from data_config import MONOFIXEDGES
except Exception:  # pragma: no cover
    MONOFIXEDGES = []


DEFAULT_BAND_ORDER: list[str] = ["delta", "theta", "alpha", "beta", "gamma"]
_GRAPH_FAMILIES = {
    "simple_fixed_graph_gnn",
    "fixed_graph_gnn",
    "gnn",
    "fused_graph_bank_gnn",
    "dual_branch_graph_model",
}
_CONNECTIVITY_FAMILIES = {
    "connectivity_only_mlp",
    "connectivity_only_cnn",
    "dual_branch_dense",
    "simple_fixed_graph_gnn",
    "fixed_graph_gnn",
    "gnn",
    "fused_graph_bank_gnn",
    "dual_branch_graph_model",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class AHEAPExperimentSpec:
    """One ladder item."""

    name: str
    config: dict[str, Any]
    output_dir: str
    split_seeds: list[int] = field(default_factory=lambda: [101])
    train_seeds: list[int] = field(default_factory=lambda: [11])
    notes: str = ""


# -----------------------------------------------------------------------------
# Small config / JSON helpers
# -----------------------------------------------------------------------------


def _normalize_name(name: Any) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


_ALIAS_MAP = {
    "node_only": "node_only_mlp",
    "node_only_dense": "node_only_mlp",
    "connectivity_only": "connectivity_only_mlp",
    "connectivity_only_dense": "connectivity_only_mlp",
    "fixed_graph": "simple_fixed_graph_gnn",
    "simple_gnn": "simple_fixed_graph_gnn",
    "graph_bank_gnn": "fused_graph_bank_gnn",
    "dual_branch_graph": "dual_branch_graph_model",
    "no_mil": "none",
    "light_subject_fusion": "subject_fusion",
    "light_fusion": "subject_fusion",
    "sum": "add",
}


def _canonical_name(name: Any) -> str:
    key = _normalize_name(name)
    return _ALIAS_MAP.get(key, key)


def _cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], Mapping)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _read_config(path: str | os.PathLike) -> dict[str, Any]:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return dict(load_yaml_config(path))
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))


def _save_json(data: Any, path: str | os.PathLike) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)
    return str(path)


# -----------------------------------------------------------------------------
# AHEAP folds
# -----------------------------------------------------------------------------


def make_aheap_folds(
    *,
    subject_ids: Sequence[str],
    labels: Sequence[int],
    n_splits: int,
    val_ratio: float,
    split_seed: int,
) -> list[dict[str, Any]]:
    """Stratified outer CV + inner validation split for AHEAP."""
    subject_ids = [str(x) for x in subject_ids]
    y = np.asarray(labels, dtype=np.int64)
    if len(subject_ids) != len(y):
        raise ValueError("subject_ids and labels length mismatch.")

    min_count = min(Counter(y.tolist()).values())
    if int(n_splits) > min_count:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest class count={min_count}."
        )

    outer = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(split_seed))
    idx = np.arange(len(subject_ids), dtype=np.int64)

    folds: list[dict[str, Any]] = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer.split(idx, y)):
        train_val_ids = [subject_ids[int(i)] for i in train_val_idx]
        train_val_y = y[train_val_idx]
        test_ids = [subject_ids[int(i)] for i in test_idx]

        inner_ok = len(train_val_ids) >= 2 and len(set(train_val_y.tolist())) >= 2
        if inner_ok and all(v >= 2 for v in Counter(train_val_y.tolist()).values()):
            inner = StratifiedShuffleSplit(
                n_splits=1,
                test_size=float(val_ratio),
                random_state=int(split_seed) + 1000 + int(fold_idx),
            )
            rel_train_idx, rel_val_idx = next(inner.split(np.arange(len(train_val_ids)), train_val_y))
        else:
            rng = np.random.RandomState(int(split_seed) + 1000 + int(fold_idx))
            perm = np.arange(len(train_val_ids))
            rng.shuffle(perm)
            n_val = max(1, int(round(len(train_val_ids) * float(val_ratio))))
            rel_val_idx = np.sort(perm[:n_val])
            rel_train_idx = np.sort(perm[n_val:])
            if len(rel_train_idx) == 0:
                rel_train_idx = rel_val_idx[:1]
                rel_val_idx = rel_val_idx[1:]

        train_ids = [train_val_ids[int(i)] for i in rel_train_idx]
        val_ids = [train_val_ids[int(i)] for i in rel_val_idx]

        folds.append(
            {
                "fold": int(fold_idx),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "split_name": f"fold_{fold_idx}",
            }
        )
    return folds


# -----------------------------------------------------------------------------
# H5 loading helpers
# -----------------------------------------------------------------------------


def _collect_required_connectivity_metrics(cfg: Mapping[str, Any]) -> list[str]:
    family = _canonical_name(_cfg_get(cfg, "experiment", "model_family", default="node_only_mlp"))
    graph_cfg = dict(_cfg_get(cfg, "graph", default={}) or {})
    conn_cfg = dict(_cfg_get(cfg, "connectivity", default={}) or {})

    metrics: set[str] = set()

    primary_metric = conn_cfg.get("primary_metric", conn_cfg.get("metric", None))
    if primary_metric is not None:
        metrics.add(str(primary_metric))

    for metric in conn_cfg.get("metrics", []) or []:
        metrics.add(str(metric))

    if family in _CONNECTIVITY_FAMILIES:
        if primary_metric is None and family in {"connectivity_only_mlp", "connectivity_only_cnn", "dual_branch_dense"}:
            raise ValueError(
                "Connectivity-based model family requires connectivity.primary_metric or connectivity.metric."
            )

    topology = _canonical_name(graph_cfg.get("topology", "fixed"))
    edge_weight = _canonical_name(graph_cfg.get("edge_weight", "binary"))
    graph_metric = graph_cfg.get("connectivity_metric", primary_metric)
    if topology.startswith("connectivity") or topology in {"full", "topk", "mst", "threshold"}:
        if graph_metric is not None:
            metrics.add(str(graph_metric))
    if edge_weight in {"connectivity", "normalized", "similarity", "topology_weight", "fused_weights", "fused"}:
        if graph_metric is not None:
            metrics.add(str(graph_metric))

    for spec in graph_cfg.get("graph_bank_specs", []) or []:
        spec = dict(spec)
        for key in (
            "metric",
            "connectivity_metric",
            "topology_metric",
            "edge_weight_metric",
        ):
            value = spec.get(key, None)
            if value is not None:
                metrics.add(str(value))

    return sorted(metrics)


def load_aheap_h5_payload(
    cfg: Mapping[str, Any],
    subject_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    h5_path = _cfg_get(cfg, "dataset", "h5_path")
    if not h5_path:
        raise ValueError("dataset.h5_path is required.")

    feature_families = list(_cfg_get(cfg, "features", "families", default=[]))
    if not feature_families:
        raise ValueError("features.families is required.")

    connectivity_metrics = _collect_required_connectivity_metrics(cfg)
    payload = load_selected_groups(
        h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        subject_ids=list(subject_ids),
    )
    return payload


# -----------------------------------------------------------------------------
# Array / representation helpers
# -----------------------------------------------------------------------------


def _aggregate_array(x: np.ndarray, mode: str) -> np.ndarray:
    mode = _canonical_name(mode)
    if mode == "mean":
        return np.mean(x, axis=0).astype(np.float32)
    if mode == "median":
        return np.median(x, axis=0).astype(np.float32)
    if mode == "std":
        return np.std(x, axis=0).astype(np.float32)
    if mode == "max":
        return np.max(x, axis=0).astype(np.float32)
    if mode == "min":
        return np.min(x, axis=0).astype(np.float32)
    if mode == "sum" or mode == "add":
        return np.sum(x, axis=0).astype(np.float32)
    raise ValueError(f"Unsupported aggregation mode={mode!r}")


def _concat_feature_families(subject_entry: Mapping[str, Any], feature_families: Sequence[str]) -> np.ndarray:
    if not feature_families:
        raise ValueError("At least one feature family is required.")
    feats = [np.asarray(subject_entry["features"][fam], dtype=np.float32) for fam in feature_families]
    ref_shape = feats[0].shape[:2]
    for fam, arr in zip(feature_families, feats):
        if arr.ndim != 3:
            raise ValueError(f"Feature family {fam!r} must have shape [W, N, F], got {arr.shape}")
        if arr.shape[:2] != ref_shape:
            raise ValueError(f"Feature family {fam!r} shape {arr.shape} does not align with {ref_shape}")
    return np.concatenate(feats, axis=-1).astype(np.float32)


def _band_to_index(band: int | str | None, band_order: Sequence[str] = DEFAULT_BAND_ORDER) -> int | None:
    if band is None:
        return None
    if isinstance(band, str):
        band_key = str(band).lower()
        if band_key in {"all", "multi", "multiband", "bank"}:
            return None
        if band_key not in list(band_order):
            raise KeyError(f"Unknown band={band!r}. Expected one of {list(band_order)}.")
        return list(band_order).index(band_key)
    return int(band)


def _select_band_if_needed(x: np.ndarray, band: int | str | None) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3:
        return arr
    band_idx = _band_to_index(band)
    if band_idx is None:
        return arr
    return arr[int(band_idx)].astype(np.float32)


def _resolve_primary_connectivity(
    connectivity_map: Mapping[str, np.ndarray],
    *,
    metric: str | None,
    band: int | str | None,
    multiband_reduce: str = "mean",
    allow_multiband: bool = False,
) -> np.ndarray | None:
    if metric is None:
        if len(connectivity_map) == 0:
            return None
        if len(connectivity_map) > 1:
            raise ValueError(
                "Multiple connectivity metrics are available. Provide an explicit primary metric."
            )
        metric = next(iter(connectivity_map.keys()))

    if metric not in connectivity_map:
        raise KeyError(f"Connectivity metric {metric!r} not found in instance connectivity map.")

    arr = np.asarray(connectivity_map[metric], dtype=np.float32)
    if arr.ndim == 2:
        return arr

    if arr.ndim != 3:
        raise ValueError(f"Expected connectivity array [N,N] or [B,N,N], got {arr.shape}")

    if band is not None and _band_to_index(band) is not None:
        return _select_band_if_needed(arr, band)

    if allow_multiband:
        return arr

    return _aggregate_array(arr, multiband_reduce)


def _build_groups_for_subject(
    subject_entry: Mapping[str, Any],
    *,
    graph_level: str,
    windows_per_macro: Optional[int],
    feature_agg: str,
    connectivity_agg: str,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
) -> list[dict[str, Any]]:
    graph_level = _canonical_name(graph_level)
    node_x_all = _concat_feature_families(subject_entry, feature_families)

    conn_all: dict[str, np.ndarray] = {}
    for metric in connectivity_metrics:
        if metric not in subject_entry.get("connectivity", {}):
            continue
        conn_all[metric] = np.asarray(subject_entry["connectivity"][metric], dtype=np.float32)

    num_windows = int(node_x_all.shape[0])
    seg_ids = np.asarray(subject_entry.get("segment_id", np.arange(num_windows)), dtype=np.int64)
    start_samples = np.asarray(subject_entry.get("start_sample", np.arange(num_windows)), dtype=np.int64)
    end_samples = np.asarray(subject_entry.get("end_sample", np.arange(num_windows)), dtype=np.int64)

    if graph_level == "segment":
        groups = [[i] for i in range(num_windows)]
        group_ids = [f"seg_{int(seg_ids[i])}" for i in range(num_windows)]
    elif graph_level == "subject":
        groups = [list(range(num_windows))]
        group_ids = ["subject"]
    elif graph_level == "macro":
        wp = int(windows_per_macro or 10)
        if wp < 1:
            raise ValueError("macro.windows_per_macro must be >= 1")
        groups = [list(range(i, min(i + wp, num_windows))) for i in range(0, num_windows, wp)]
        group_ids = [f"macro_{i}" for i in range(len(groups))]
    else:
        raise ValueError(f"Unsupported graph_level={graph_level!r}")

    out: list[dict[str, Any]] = []
    for gid, idxs in zip(group_ids, groups):
        x = node_x_all[idxs]
        x_red = x[0] if len(idxs) == 1 else _aggregate_array(x, feature_agg)

        instance_connectivity: dict[str, np.ndarray] = {}
        for metric, values in conn_all.items():
            sub = values[idxs]
            c_red = sub[0] if len(idxs) == 1 else _aggregate_array(sub, connectivity_agg)
            instance_connectivity[metric] = np.asarray(c_red, dtype=np.float32)

        out.append(
            {
                "instance_id": gid,
                "group_indices": list(idxs),
                "node_features": np.asarray(x_red, dtype=np.float32),
                "connectivity_map": instance_connectivity,
                "start_sample": int(start_samples[idxs[0]]),
                "end_sample": int(end_samples[idxs[-1]]),
            }
        )
    return out


# -----------------------------------------------------------------------------
# Dense bag dataset / collate
# -----------------------------------------------------------------------------


class DenseSubjectBagDataset(Dataset):
    def __init__(
        self,
        bags: Sequence[dict[str, Any]],
        *,
        max_instances: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.bags = [copy.deepcopy(b) for b in bags]
        self.max_instances = None if max_instances is None else int(max_instances)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bag = copy.deepcopy(self.bags[int(idx)])
        instances = list(bag["instances"])
        if self.max_instances is not None and len(instances) > self.max_instances:
            rng = random.Random(self.seed + 1000003 * (idx + 1))
            chosen = sorted(rng.sample(range(len(instances)), self.max_instances))
            instances = [instances[i] for i in chosen]
        bag["instances"] = instances
        return bag



def dense_subject_bag_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    batch = list(batch)
    batch_size = len(batch)
    max_k = max(len(item["instances"]) for item in batch)
    first_inst = batch[0]["instances"][0]
    node_shape = tuple(np.asarray(first_inst["node_features"]).shape)
    has_connectivity = first_inst.get("connectivity") is not None
    conn_shape = None if not has_connectivity else tuple(np.asarray(first_inst["connectivity"]).shape)

    node_tensor = torch.zeros((batch_size, max_k, *node_shape), dtype=torch.float32)
    conn_tensor = None
    if has_connectivity:
        assert conn_shape is not None
        conn_tensor = torch.zeros((batch_size, max_k, *conn_shape), dtype=torch.float32)

    mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    subject_ids: list[str] = []
    instance_ids: list[list[str]] = []

    for b_idx, item in enumerate(batch):
        labels[b_idx] = int(item["label"])
        subject_ids.append(str(item["subject_id"]))
        this_ids: list[str] = []
        for k_idx, inst in enumerate(item["instances"]):
            mask[b_idx, k_idx] = True
            node_tensor[b_idx, k_idx] = torch.as_tensor(inst["node_features"], dtype=torch.float32)
            if conn_tensor is not None:
                conn_tensor[b_idx, k_idx] = torch.as_tensor(inst["connectivity"], dtype=torch.float32)
            this_ids.append(str(inst.get("instance_id", f"inst_{k_idx}")))
        instance_ids.append(this_ids)

    out = {
        "node_features": node_tensor,
        "mask": mask,
        "labels": labels,
        "subject_ids": subject_ids,
        "instance_ids": instance_ids,
    }
    if conn_tensor is not None:
        out["connectivity"] = conn_tensor
    return out


# -----------------------------------------------------------------------------
# Graph helpers / graph bag dataset
# -----------------------------------------------------------------------------


def _import_pyg():
    try:
        from torch_geometric.data import Batch, Data
        from torch_geometric.utils import dense_to_sparse
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Graph experiments require torch_geometric in the runtime environment."
        ) from exc
    return Data, Batch, dense_to_sparse



def _norm_channel(name: str) -> str:
    x = str(name).strip().upper()
    alias = {"FZ": "Fz", "CZ": "Cz", "PZ": "Pz", "FP1": "Fp1", "FP2": "Fp2"}
    return alias.get(x, str(name).strip())



def _normalize_fixed_edges(
    fixed_edges: Optional[Sequence[tuple[Any, Any]]],
    *,
    channel_names: Sequence[str],
) -> list[tuple[int, int]]:
    if fixed_edges is None:
        return []
    name_to_idx = {_norm_channel(ch): i for i, ch in enumerate(channel_names)}
    pairs: set[tuple[int, int]] = set()
    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            uu = name_to_idx.get(_norm_channel(str(u)))
            vv = name_to_idx.get(_norm_channel(str(v)))
            if uu is None or vv is None:
                continue
        else:
            uu, vv = int(u), int(v)
        if uu == vv:
            continue
        pairs.add(tuple(sorted((uu, vv))))
    return sorted(pairs)



def _topk_union_topology(adj: np.ndarray, topk: int) -> np.ndarray:
    n = adj.shape[0]
    base = np.abs(adj).copy()
    np.fill_diagonal(base, -np.inf)
    topo = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argsort(base[i])[-int(topk):]
        topo[i, idx] = 1.0
    topo = ((topo + topo.T) > 0).astype(np.float32)
    np.fill_diagonal(topo, 0.0)
    return topo



def _maximum_spanning_tree_topology(adj: np.ndarray) -> np.ndarray:
    n = adj.shape[0]
    weights = np.abs(adj).astype(np.float32)
    np.fill_diagonal(weights, 0.0)
    edges: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((float(weights[i, j]), i, j))
    edges.sort(reverse=True)

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    topo = np.zeros((n, n), dtype=np.float32)
    used = 0
    for w, i, j in edges:
        if w <= 0:
            continue
        if union(i, j):
            topo[i, j] = topo[j, i] = 1.0
            used += 1
            if used == n - 1:
                break

    if used < n - 1:
        for i in range(n - 1):
            topo[i, i + 1] = topo[i + 1, i] = 1.0
    return topo



def _feature_similarity_topology(x: np.ndarray, topk: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)
    sim = x_norm @ x_norm.T
    topo = _topk_union_topology(sim, topk=topk)
    return topo.astype(np.float32), sim.astype(np.float32)



def _minmax_normalize_matrix(adj: np.ndarray) -> np.ndarray:
    x = np.asarray(adj, dtype=np.float32)
    off = x[~np.eye(x.shape[0], dtype=bool)]
    if off.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(off.min())
    hi = float(off.max())
    if math.isclose(lo, hi):
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    np.fill_diagonal(y, 0.0)
    return y.astype(np.float32)



def _make_fixed_topology(
    n_nodes: int,
    *,
    fixed_edges: Optional[Sequence[tuple[Any, Any]]],
    channel_names: Sequence[str],
) -> np.ndarray:
    pairs = _normalize_fixed_edges(fixed_edges, channel_names=channel_names)
    topo = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if not pairs:
        topo[:] = 1.0
    else:
        for i, j in pairs:
            topo[i, j] = topo[j, i] = 1.0
    np.fill_diagonal(topo, 0.0)
    return topo



def _resolve_graph_metric_and_band(graph_cfg: Mapping[str, Any]) -> tuple[str | None, int | str | None]:
    metric = graph_cfg.get("connectivity_metric", graph_cfg.get("metric", None))
    band = graph_cfg.get("connectivity_band", graph_cfg.get("band", None))
    return None if metric is None else str(metric), band



def _make_topology_and_weights(
    *,
    node_features: np.ndarray,
    connectivity_map: Mapping[str, np.ndarray],
    channel_names: Sequence[str],
    graph_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = node_features.shape[0]
    topology = _canonical_name(graph_cfg.get("topology", "fixed"))
    edge_weight = _canonical_name(graph_cfg.get("edge_weight", "connectivity"))
    topk = int(graph_cfg.get("topk", 3))
    threshold = float(graph_cfg.get("threshold", 0.0))
    reduce_mode = _canonical_name(graph_cfg.get("connectivity_reduce", "mean"))
    metric, band = _resolve_graph_metric_and_band(graph_cfg)

    fixed_edges_cfg = graph_cfg.get("fixed_edges", None)
    fixed_edges = MONOFIXEDGES if fixed_edges_cfg in {None, "mono_fixed"} else fixed_edges_cfg

    primary_matrix = _resolve_primary_connectivity(
        connectivity_map,
        metric=metric,
        band=band,
        multiband_reduce=reduce_mode,
        allow_multiband=False,
    )

    if topology == "fixed":
        topo = _make_fixed_topology(n_nodes, fixed_edges=fixed_edges, channel_names=channel_names)
        weight_source = np.asarray(primary_matrix, dtype=np.float32) if primary_matrix is not None else topo.copy()

    elif topology in {"connectivity", "connectivity_full", "full"}:
        if primary_matrix is None:
            raise ValueError("Connectivity-based topology requires a connectivity matrix.")
        topo = np.ones((n_nodes, n_nodes), dtype=np.float32)
        np.fill_diagonal(topo, 0.0)
        weight_source = np.asarray(primary_matrix, dtype=np.float32)

    elif topology in {"connectivity_topk", "topk"}:
        if primary_matrix is None:
            raise ValueError("Top-k topology requires a connectivity matrix.")
        weight_source = np.asarray(primary_matrix, dtype=np.float32)
        topo = _topk_union_topology(weight_source, topk=topk)

    elif topology in {"connectivity_mst", "mst"}:
        if primary_matrix is None:
            raise ValueError("MST topology requires a connectivity matrix.")
        weight_source = np.asarray(primary_matrix, dtype=np.float32)
        topo = _maximum_spanning_tree_topology(weight_source)

    elif topology in {"connectivity_threshold", "threshold"}:
        if primary_matrix is None:
            raise ValueError("Threshold topology requires a connectivity matrix.")
        weight_source = np.asarray(primary_matrix, dtype=np.float32)
        topo = (np.abs(weight_source) >= float(threshold)).astype(np.float32)
        np.fill_diagonal(topo, 0.0)

    elif topology in {"feature_induced", "feature_induced_topk"}:
        topo, weight_source = _feature_similarity_topology(node_features, topk=topk)

    else:
        raise ValueError(f"Unsupported graph.topology={topology!r}")

    weight_source = np.nan_to_num(weight_source, nan=0.0, posinf=0.0, neginf=0.0)
    weight_source = 0.5 * (weight_source + weight_source.T)
    np.fill_diagonal(weight_source, 0.0)

    if edge_weight in {"binary", "none"}:
        adj = topo.astype(np.float32)
    elif edge_weight in {"connectivity", "topology_weight", "similarity"}:
        adj = topo * weight_source
    elif edge_weight == "normalized":
        adj = topo * _minmax_normalize_matrix(np.abs(weight_source))
    else:
        raise ValueError(f"Unsupported graph.edge_weight={edge_weight!r}")

    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    adj = 0.5 * (adj + adj.T)
    np.fill_diagonal(adj, 0.0)
    return topo.astype(np.float32), adj.astype(np.float32)



def _fuse_bank_numpy(
    adj_bank: np.ndarray,
    topo_bank: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    topology_rule: str = "union",
    vote_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    weights_arr = np.ones((adj_bank.shape[0],), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
    weights_arr = weights_arr / np.clip(weights_arr.sum(), 1e-8, None)
    fused_adj = np.sum(adj_bank * weights_arr[:, None, None], axis=0)

    topology_rule = _canonical_name(topology_rule)
    if topology_rule == "union":
        fused_topo = topo_bank.max(axis=0)
    elif topology_rule == "intersection":
        fused_topo = topo_bank.min(axis=0)
    elif topology_rule == "vote":
        voted = np.sum(topo_bank * weights_arr[:, None, None], axis=0)
        fused_topo = (voted >= float(vote_threshold)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported topology_rule={topology_rule!r}")

    fused_adj = 0.5 * (fused_adj + fused_adj.T)
    fused_topo = ((fused_topo + fused_topo.T) > 0).astype(np.float32)
    np.fill_diagonal(fused_adj, 0.0)
    np.fill_diagonal(fused_topo, 0.0)
    fused_adj = fused_adj * fused_topo
    return fused_adj.astype(np.float32), fused_topo.astype(np.float32)



def _build_graph_bank(
    *,
    node_features: np.ndarray,
    connectivity_map: Mapping[str, np.ndarray],
    channel_names: Sequence[str],
    graph_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    specs = graph_cfg.get("graph_bank_specs", None)
    primary_metric = graph_cfg.get("connectivity_metric", graph_cfg.get("metric", None))

    if specs is None:
        if primary_metric is None:
            raise ValueError("graph_bank_specs is missing and no primary connectivity metric was provided.")
        if primary_metric not in connectivity_map:
            raise KeyError(f"Primary metric {primary_metric!r} not found for automatic graph bank.")
        conn = np.asarray(connectivity_map[str(primary_metric)], dtype=np.float32)
        if conn.ndim != 3:
            raise ValueError(
                "Automatic graph bank construction requires multiband connectivity [B,N,N]."
            )
        specs = [{"name": f"{primary_metric}_{band}", "metric": primary_metric, "band": band} for band in DEFAULT_BAND_ORDER[: conn.shape[0]]]

    adj_bank: list[np.ndarray] = []
    topo_bank: list[np.ndarray] = []
    names: list[str] = []

    for idx, spec in enumerate(specs):
        spec = dict(spec)
        cand_name = str(spec.get("name", f"candidate_{idx}"))

        cand_cfg = dict(graph_cfg)
        metric = spec.get("metric", spec.get("connectivity_metric", spec.get("topology_metric", spec.get("edge_weight_metric", primary_metric))))
        band = spec.get("band", spec.get("connectivity_band", None))
        cand_cfg["connectivity_metric"] = metric
        if band is not None:
            cand_cfg["connectivity_band"] = band

        if "topology" in spec:
            cand_cfg["topology"] = spec["topology"]
        if "edge_weight" in spec:
            cand_cfg["edge_weight"] = spec["edge_weight"]
        if "topk" in spec:
            cand_cfg["topk"] = spec["topk"]
        if "threshold" in spec:
            cand_cfg["threshold"] = spec["threshold"]
        if "connectivity_reduce" in spec:
            cand_cfg["connectivity_reduce"] = spec["connectivity_reduce"]

        topo, adj = _make_topology_and_weights(
            node_features=node_features,
            connectivity_map=connectivity_map,
            channel_names=channel_names,
            graph_cfg=cand_cfg,
        )
        topo_bank.append(topo)
        adj_bank.append(adj)
        names.append(cand_name)

    return (
        np.stack(adj_bank, axis=0).astype(np.float32),
        np.stack(topo_bank, axis=0).astype(np.float32),
        names,
    )



def _instance_to_pyg_data(
    *,
    node_features: np.ndarray,
    connectivity_map: Mapping[str, np.ndarray],
    channel_names: Sequence[str],
    label: int,
    subject_id: str,
    instance_id: str,
    graph_cfg: Mapping[str, Any],
    require_graph_bank: bool,
):
    Data, _, dense_to_sparse = _import_pyg()

    graph_cfg = dict(graph_cfg)
    need_bank = bool(require_graph_bank) or _canonical_name(graph_cfg.get("topology", "fixed")) == "fused_bank" or _canonical_name(graph_cfg.get("edge_weight", "binary")) in {"fused_weights", "fused"}

    adj_bank = None
    topo_bank = None
    bank_names = None
    if need_bank:
        adj_bank, topo_bank, bank_names = _build_graph_bank(
            node_features=node_features,
            connectivity_map=connectivity_map,
            channel_names=channel_names,
            graph_cfg=graph_cfg,
        )
        fused_adj, fused_topo = _fuse_bank_numpy(
            adj_bank,
            topo_bank,
            weights=np.asarray(graph_cfg.get("static_bank_weights", None), dtype=np.float32) if graph_cfg.get("static_bank_weights", None) is not None else None,
            topology_rule=str(graph_cfg.get("topology_rule", "union")),
            vote_threshold=float(graph_cfg.get("vote_threshold", 0.5)),
        )
        topo, adj = fused_topo, fused_adj
    else:
        topo, adj = _make_topology_and_weights(
            node_features=node_features,
            connectivity_map=connectivity_map,
            channel_names=channel_names,
            graph_cfg=graph_cfg,
        )

    edge_index, edge_weight = dense_to_sparse(torch.as_tensor(adj, dtype=torch.float32))

    data = Data(
        x=torch.as_tensor(node_features, dtype=torch.float32),
        edge_index=edge_index.long(),
        y=torch.tensor([int(label)], dtype=torch.long),
    )
    data.edge_weight = edge_weight.float()
    data.edge_attr = edge_weight.view(-1, 1).float()
    data.adj = torch.as_tensor(adj, dtype=torch.float32)
    data.topology = torch.as_tensor(topo, dtype=torch.float32)
    data.subject_id = str(subject_id)
    data.instance_id = str(instance_id)

    if adj_bank is not None and topo_bank is not None:
        data.adj_bank = torch.as_tensor(adj_bank, dtype=torch.float32)
        data.topology_bank = torch.as_tensor(topo_bank, dtype=torch.float32)
        data.bank_names = list(bank_names or [f"candidate_{i}" for i in range(adj_bank.shape[0])])

    return data


class GraphSubjectBagDataset(Dataset):
    def __init__(
        self,
        bags: Sequence[dict[str, Any]],
        *,
        max_instances: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.bags = [copy.deepcopy(b) for b in bags]
        self.max_instances = None if max_instances is None else int(max_instances)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bag = copy.deepcopy(self.bags[int(idx)])
        graphs = list(bag["graphs"])
        if self.max_instances is not None and len(graphs) > self.max_instances:
            rng = random.Random(self.seed + 1000003 * (idx + 1))
            chosen = sorted(rng.sample(range(len(graphs)), self.max_instances))
            graphs = [graphs[i] for i in chosen]
        bag["graphs"] = graphs
        return bag



def graph_subject_bag_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _, Batch, _ = _import_pyg()
    flat_graphs = []
    bag_indices = []
    labels = []
    subject_ids = []
    for bag_idx, item in enumerate(batch):
        labels.append(int(item["label"]))
        subject_ids.append(str(item["subject_id"]))
        graphs = list(item["graphs"])
        if len(graphs) == 0:
            raise ValueError(f"Subject {item['subject_id']!r} has an empty graph bag.")
        for g in graphs:
            flat_graphs.append(g)
            bag_indices.append(int(bag_idx))
    pyg_batch = Batch.from_data_list(flat_graphs)
    return {
        "pyg_batch": pyg_batch,
        "bag_indices": torch.as_tensor(bag_indices, dtype=torch.long),
        "labels": torch.as_tensor(labels, dtype=torch.long),
        "subject_ids": subject_ids,
    }


# -----------------------------------------------------------------------------
# Build dense / graph bags from H5 payload
# -----------------------------------------------------------------------------


def _build_dense_subject_bags(
    *,
    payload: Mapping[str, Mapping[str, Any]],
    subject_ids: Sequence[str],
    cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    graph_level = _cfg_get(cfg, "experiment", "graph_level", default="segment")
    feature_families = list(_cfg_get(cfg, "features", "families", default=[]))
    connectivity_metrics = _collect_required_connectivity_metrics(cfg)

    windows_per_macro = _cfg_get(cfg, "macro", "windows_per_macro", default=10)
    feature_agg = _cfg_get(cfg, "macro", "feature_aggregation", default="mean")
    connectivity_agg = _cfg_get(cfg, "macro", "connectivity_aggregation", default="mean")

    conn_cfg = dict(_cfg_get(cfg, "connectivity", default={}) or {})
    primary_metric = conn_cfg.get("primary_metric", conn_cfg.get("metric", None))
    primary_band = conn_cfg.get("primary_band", conn_cfg.get("band", None))
    allow_multiband = bool(conn_cfg.get("allow_multiband", True))
    multiband_reduce = str(conn_cfg.get("multiband_reduce", "mean"))

    bags: list[dict[str, Any]] = []
    for sid in subject_ids:
        subj = payload[sid]
        raw_instances = _build_groups_for_subject(
            subj,
            graph_level=graph_level,
            windows_per_macro=windows_per_macro,
            feature_agg=feature_agg,
            connectivity_agg=connectivity_agg,
            feature_families=feature_families,
            connectivity_metrics=connectivity_metrics,
        )

        instances: list[dict[str, Any]] = []
        for inst in raw_instances:
            conn = _resolve_primary_connectivity(
                inst["connectivity_map"],
                metric=None if primary_metric is None else str(primary_metric),
                band=primary_band,
                multiband_reduce=multiband_reduce,
                allow_multiband=allow_multiband,
            )
            instances.append(
                {
                    "instance_id": inst["instance_id"],
                    "node_features": np.asarray(inst["node_features"], dtype=np.float32),
                    "connectivity": None if conn is None else np.asarray(conn, dtype=np.float32),
                    "start_sample": int(inst["start_sample"]),
                    "end_sample": int(inst["end_sample"]),
                }
            )

        bags.append(
            {
                "subject_id": sid,
                "label": int(subj["label"]),
                "instances": instances,
            }
        )
    return bags



def _build_graph_subject_bags(
    *,
    payload: Mapping[str, Mapping[str, Any]],
    subject_ids: Sequence[str],
    cfg: Mapping[str, Any],
    require_graph_bank: bool,
) -> list[dict[str, Any]]:
    graph_level = _cfg_get(cfg, "experiment", "graph_level", default="segment")
    feature_families = list(_cfg_get(cfg, "features", "families", default=[]))
    connectivity_metrics = _collect_required_connectivity_metrics(cfg)

    windows_per_macro = _cfg_get(cfg, "macro", "windows_per_macro", default=10)
    feature_agg = _cfg_get(cfg, "macro", "feature_aggregation", default="mean")
    connectivity_agg = _cfg_get(cfg, "macro", "connectivity_aggregation", default="mean")
    graph_cfg = dict(_cfg_get(cfg, "graph", default={}) or {})
    conn_cfg = dict(_cfg_get(cfg, "connectivity", default={}) or {})
    if "connectivity_metric" not in graph_cfg and (conn_cfg.get("primary_metric", conn_cfg.get("metric", None)) is not None):
        graph_cfg["connectivity_metric"] = conn_cfg.get("primary_metric", conn_cfg.get("metric", None))
    if "connectivity_band" not in graph_cfg and conn_cfg.get("primary_band", conn_cfg.get("band", None)) is not None:
        graph_cfg["connectivity_band"] = conn_cfg.get("primary_band", conn_cfg.get("band", None))

    bags: list[dict[str, Any]] = []
    for sid in subject_ids:
        subj = payload[sid]
        channel_names = list(subj["channel_names"])
        raw_instances = _build_groups_for_subject(
            subj,
            graph_level=graph_level,
            windows_per_macro=windows_per_macro,
            feature_agg=feature_agg,
            connectivity_agg=connectivity_agg,
            feature_families=feature_families,
            connectivity_metrics=connectivity_metrics,
        )
        graphs = [
            _instance_to_pyg_data(
                node_features=np.asarray(inst["node_features"], dtype=np.float32),
                connectivity_map=inst["connectivity_map"],
                channel_names=channel_names,
                label=int(subj["label"]),
                subject_id=sid,
                instance_id=str(inst["instance_id"]),
                graph_cfg=graph_cfg,
                require_graph_bank=require_graph_bank,
            )
            for inst in raw_instances
        ]
        bags.append({"subject_id": sid, "label": int(subj["label"]), "graphs": graphs})
    return bags


# -----------------------------------------------------------------------------
# Instance model builders
# -----------------------------------------------------------------------------


def _build_dense_instance_model(cfg: Mapping[str, Any], sample_bag: Mapping[str, Any], num_classes: int) -> nn.Module:
    family = _canonical_name(_cfg_get(cfg, "experiment", "model_family", default="node_only_mlp"))
    dense_cfg = dict(_cfg_get(cfg, "dense", default={}) or {})

    first_inst = sample_bag["instances"][0]
    node_x = np.asarray(first_inst["node_features"], dtype=np.float32)
    num_nodes, num_node_features = int(node_x.shape[0]), int(node_x.shape[1])
    conn = first_inst.get("connectivity", None)
    num_bands = 1
    if conn is not None:
        conn = np.asarray(conn, dtype=np.float32)
        if conn.ndim == 3:
            num_bands = int(conn.shape[0])

    if family == "node_only_mlp":
        return NodeOnlyMLP(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_classes=num_classes,
            readout=dense_cfg.get("node_readout", "flatten"),
            hidden_dims=tuple(dense_cfg.get("hidden_dims", [256, 128])),
            emb_dim=int(dense_cfg.get("emb_dim", 128)),
            dropout=float(dense_cfg.get("dropout", 0.2)),
            use_batchnorm=bool(dense_cfg.get("use_batchnorm", False)),
        )

    if family == "connectivity_only_mlp":
        return ConnectivityOnlyMLP(
            num_nodes=num_nodes,
            num_bands=num_bands,
            num_classes=num_classes,
            hidden_dims=tuple(dense_cfg.get("hidden_dims", [256, 128])),
            emb_dim=int(dense_cfg.get("emb_dim", 128)),
            dropout=float(dense_cfg.get("dropout", 0.2)),
            flatten_mode=dense_cfg.get("flatten_mode", "upper_triangle"),
            symmetrize=bool(dense_cfg.get("symmetrize", True)),
            include_diagonal=bool(dense_cfg.get("include_diagonal", False)),
            use_batchnorm=bool(dense_cfg.get("use_batchnorm", False)),
        )

    if family == "connectivity_only_cnn":
        return ConnectivityOnlyCNN(
            num_bands=num_bands,
            num_classes=num_classes,
            emb_dim=int(dense_cfg.get("emb_dim", 128)),
            conv_channels=tuple(dense_cfg.get("conv_channels", [16, 32, 64])),
            kernel_sizes=tuple(dense_cfg.get("kernel_sizes", [3, 3, 3])),
            dropout=float(dense_cfg.get("dropout", 0.2)),
            adaptive_pool_output_size=int(dense_cfg.get("adaptive_pool_output_size", 1)),
            use_batchnorm=bool(dense_cfg.get("use_batchnorm", True)),
        )

    if family == "dual_branch_dense":
        return DualBranchDenseModel(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_bands=num_bands,
            node_readout=dense_cfg.get("node_readout", "flatten"),
            node_hidden_dims=tuple(dense_cfg.get("node_hidden_dims", [256, 128])),
            node_emb_dim=int(dense_cfg.get("node_emb_dim", 128)),
            connectivity_encoder_type=dense_cfg.get("connectivity_encoder_type", "mlp"),
            connectivity_hidden_dims=tuple(dense_cfg.get("connectivity_hidden_dims", [256, 128])),
            connectivity_emb_dim=int(dense_cfg.get("connectivity_emb_dim", 128)),
            connectivity_flatten_mode=dense_cfg.get("connectivity_flatten_mode", "upper_triangle"),
            connectivity_symmetrize=bool(dense_cfg.get("connectivity_symmetrize", True)),
            connectivity_include_diagonal=bool(dense_cfg.get("connectivity_include_diagonal", False)),
            connectivity_conv_channels=tuple(dense_cfg.get("connectivity_conv_channels", [16, 32, 64])),
            connectivity_kernel_sizes=tuple(dense_cfg.get("connectivity_kernel_sizes", [3, 3, 3])),
            fusion_mode=dense_cfg.get("fusion_mode", "concat"),
            fusion_emb_dim=int(dense_cfg.get("fusion_emb_dim", 128)),
            dropout=float(dense_cfg.get("dropout", 0.2)),
            use_batchnorm=bool(dense_cfg.get("use_batchnorm", False)),
        )

    raise ValueError(f"Unsupported dense model_family={family!r}")



def _build_graph_instance_model(cfg: Mapping[str, Any], sample_bag: Mapping[str, Any], num_classes: int) -> nn.Module:
    family = _canonical_name(_cfg_get(cfg, "experiment", "model_family", default="simple_fixed_graph_gnn"))
    gnn_cfg = dict(_cfg_get(cfg, "gnn", default={}) or {})

    first_graph = sample_bag["graphs"][0]
    num_nodes = int(first_graph.x.shape[0])
    num_node_features = int(first_graph.x.shape[1])

    try:
        from gnn import DualBranchGraphModel, FusedGraphBankGNN, SimpleFixedGraphGNN
    except Exception as exc:  # pragma: no cover
        raise ImportError("Could not import graph models from gnn.py") from exc

    if family in {"simple_fixed_graph_gnn", "fixed_graph_gnn", "gnn"}:
        return SimpleFixedGraphGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            backbone=gnn_cfg.get("backbone", "gcn"),
            hidden_dim=int(gnn_cfg.get("hidden_dim", 64)),
            graph_emb_dim=int(gnn_cfg.get("graph_emb_dim", 128)),
            num_layers=int(gnn_cfg.get("num_layers", 2)),
            dropout=float(gnn_cfg.get("dropout", 0.2)),
            gat_heads=int(gnn_cfg.get("gat_heads", 4)),
            use_edge_weight=bool(gnn_cfg.get("use_edge_weight", True)),
            use_batchnorm=bool(gnn_cfg.get("use_batchnorm", True)),
            node_pooling_type=gnn_cfg.get("node_pooling_type", "none"),
            node_pool_ratio=float(gnn_cfg.get("node_pool_ratio", 0.8)),
            readout_type=gnn_cfg.get("readout_type", "mean"),
            readout_hidden_dim=int(gnn_cfg.get("readout_hidden_dim", 64)),
            readout_dropout=float(gnn_cfg.get("readout_dropout", 0.0)),
            return_attention_weights=bool(gnn_cfg.get("return_attention_weights", False)),
        )

    if family == "fused_graph_bank_gnn":
        first_adj_bank = getattr(first_graph, "adj_bank", None)
        if first_adj_bank is None:
            raise ValueError("FusedGraphBankGNN requires graphs with adj_bank attached.")
        num_candidates = int(first_adj_bank.shape[0])
        return FusedGraphBankGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_candidates=num_candidates,
            backbone=gnn_cfg.get("backbone", "gcn"),
            hidden_dim=int(gnn_cfg.get("hidden_dim", 64)),
            graph_emb_dim=int(gnn_cfg.get("graph_emb_dim", 128)),
            num_layers=int(gnn_cfg.get("num_layers", 2)),
            dropout=float(gnn_cfg.get("dropout", 0.2)),
            gat_heads=int(gnn_cfg.get("gat_heads", 4)),
            use_edge_weight=bool(gnn_cfg.get("use_edge_weight", True)),
            use_batchnorm=bool(gnn_cfg.get("use_batchnorm", True)),
            node_pooling_type=gnn_cfg.get("node_pooling_type", "none"),
            node_pool_ratio=float(gnn_cfg.get("node_pool_ratio", 0.8)),
            readout_type=gnn_cfg.get("readout_type", "mean"),
            readout_hidden_dim=int(gnn_cfg.get("readout_hidden_dim", 64)),
            readout_dropout=float(gnn_cfg.get("readout_dropout", 0.0)),
            return_attention_weights=bool(gnn_cfg.get("return_attention_weights", False)),
            fusion_mode=gnn_cfg.get("graph_bank_fusion_mode", "static"),
            topology_rule=gnn_cfg.get("topology_rule", "union"),
            vote_threshold=float(gnn_cfg.get("vote_threshold", 0.5)),
            fusion_temperature=float(gnn_cfg.get("fusion_temperature", 1.0)),
            fusion_hidden_dim=int(gnn_cfg.get("fusion_hidden_dim", 64)),
        )

    if family == "dual_branch_graph_model":
        use_graph_bank = bool(gnn_cfg.get("use_graph_bank", False))
        num_candidates = None
        if use_graph_bank:
            first_adj_bank = getattr(first_graph, "adj_bank", None)
            if first_adj_bank is None:
                raise ValueError("DualBranchGraphModel(use_graph_bank=True) requires adj_bank on graphs.")
            num_candidates = int(first_adj_bank.shape[0])
        return DualBranchGraphModel(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes,
            use_graph_bank=use_graph_bank,
            num_candidates=num_candidates,
            node_hidden_dims=tuple(gnn_cfg.get("node_hidden_dims", [128, 64])),
            node_emb_dim=int(gnn_cfg.get("node_emb_dim", 128)),
            node_dropout=float(gnn_cfg.get("node_dropout", 0.2)),
            backbone=gnn_cfg.get("backbone", "gcn"),
            hidden_dim=int(gnn_cfg.get("hidden_dim", 64)),
            graph_emb_dim=int(gnn_cfg.get("graph_emb_dim", 128)),
            num_layers=int(gnn_cfg.get("num_layers", 2)),
            graph_dropout=float(gnn_cfg.get("graph_dropout", 0.2)),
            gat_heads=int(gnn_cfg.get("gat_heads", 4)),
            use_edge_weight=bool(gnn_cfg.get("use_edge_weight", True)),
            use_batchnorm=bool(gnn_cfg.get("use_batchnorm", True)),
            node_pooling_type=gnn_cfg.get("node_pooling_type", "none"),
            node_pool_ratio=float(gnn_cfg.get("node_pool_ratio", 0.8)),
            node_readout_type=gnn_cfg.get("node_readout_type", "mean"),
            graph_readout_type=gnn_cfg.get("graph_readout_type", "mean"),
            readout_hidden_dim=int(gnn_cfg.get("readout_hidden_dim", 64)),
            readout_dropout=float(gnn_cfg.get("readout_dropout", 0.0)),
            return_attention_weights=bool(gnn_cfg.get("return_attention_weights", False)),
            graph_bank_fusion_mode=gnn_cfg.get("graph_bank_fusion_mode", "static"),
            topology_rule=gnn_cfg.get("topology_rule", "union"),
            vote_threshold=float(gnn_cfg.get("vote_threshold", 0.5)),
            fusion_temperature=float(gnn_cfg.get("fusion_temperature", 1.0)),
            graph_bank_hidden_dim=int(gnn_cfg.get("graph_bank_hidden_dim", 64)),
            fusion_mode=gnn_cfg.get("fusion_mode", "concat"),
            fusion_emb_dim=int(gnn_cfg.get("fusion_emb_dim", 128)),
            fusion_dropout=float(gnn_cfg.get("fusion_dropout", 0.2)),
        )

    raise ValueError(f"Unsupported graph model_family={family!r}")


# -----------------------------------------------------------------------------
# Trainable subject aggregation wrappers
# -----------------------------------------------------------------------------


class DenseSubjectModel(nn.Module):
    def __init__(self, instance_model: nn.Module, *, aggregation: str, mil_cfg: Mapping[str, Any], num_classes: int):
        super().__init__()
        self.instance_model = instance_model
        self.aggregation = _canonical_name(aggregation)
        self.pool: nn.Module | None = None
        self.fusion_head: SubjectFusionHead | None = None
        emb_dim = int(self.instance_model.classifier.in_features)

        attn_dim = int(mil_cfg.get("attention_dim", 128))
        mil_dropout = float(mil_cfg.get("dropout", 0.0))

        if self.aggregation == "attention_mil":
            self.pool = AttentionMILPool(in_dim=emb_dim, attn_dim=attn_dim, dropout=mil_dropout)
        elif self.aggregation == "gated_attention_mil":
            self.pool = GatedAttentionMILPool(in_dim=emb_dim, attn_dim=attn_dim, dropout=mil_dropout)
        elif self.aggregation == "subject_fusion":
            self.fusion_head = SubjectFusionHead(
                in_dim=emb_dim,
                num_classes=int(num_classes),
                hidden_dim=int(mil_cfg.get("fusion_hidden_dim", 128)),
                fusion_dim=int(mil_cfg.get("fusion_dim", 128)),
                instance_logit_dim=int(num_classes),
                dropout=float(mil_cfg.get("fusion_dropout", 0.2)),
                use_mean_max=bool(mil_cfg.get("fusion_use_mean_max", True)),
            )

    def forward_bag(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        x = batch["node_features"]
        mask = batch["mask"]
        conn = batch.get("connectivity", None)
        labels = batch["labels"]
        subject_ids = batch["subject_ids"]

        bsz, max_k = int(x.shape[0]), int(x.shape[1])
        flat_mask = mask.reshape(-1)
        x_flat = x.reshape(bsz * max_k, *x.shape[2:])[flat_mask]
        conn_flat = None
        if conn is not None:
            conn_flat = conn.reshape(bsz * max_k, *conn.shape[2:])[flat_mask]

        out = self.instance_model(
            node_features=x_flat,
            connectivity=conn_flat,
            metadata={"subject_ids": subject_ids},
            return_dict=True,
        )

        emb_dim = int(out.embedding.shape[-1])
        num_classes = int(out.logits.shape[-1])
        grouped_emb = torch.zeros((bsz, max_k, emb_dim), device=out.embedding.device, dtype=out.embedding.dtype)
        grouped_logits = torch.zeros((bsz, max_k, num_classes), device=out.logits.device, dtype=out.logits.dtype)
        grouped_emb.reshape(-1, emb_dim)[flat_mask] = out.embedding
        grouped_logits.reshape(-1, num_classes)[flat_mask] = out.logits

        agg = aggregate_subject_predictions(
            instance_embeddings=grouped_emb,
            instance_logits=grouped_logits,
            mask=mask.to(device=grouped_emb.device),
            method=self.aggregation,
            classifier=self.instance_model.classifier,
            pool=self.pool,
            fusion_head=self.fusion_head,
        )
        return {
            "logits": agg["subject_logits"],
            "probs": agg["subject_prob"],
            "preds": agg["subject_pred"],
            "targets": labels.to(device=grouped_emb.device),
            "subject_ids": list(subject_ids),
            "attention_weights": agg.get("attention_weights"),
        }


class GraphSubjectModel(nn.Module):
    def __init__(self, instance_model: nn.Module, *, aggregation: str, mil_cfg: Mapping[str, Any], num_classes: int):
        super().__init__()
        self.instance_model = instance_model
        self.aggregation = _canonical_name(aggregation)
        self.pool: nn.Module | None = None
        self.fusion_head: SubjectFusionHead | None = None
        emb_dim = int(self.instance_model.classifier.in_features)

        attn_dim = int(mil_cfg.get("attention_dim", 128))
        mil_dropout = float(mil_cfg.get("dropout", 0.0))

        if self.aggregation == "attention_mil":
            self.pool = AttentionMILPool(in_dim=emb_dim, attn_dim=attn_dim, dropout=mil_dropout)
        elif self.aggregation == "gated_attention_mil":
            self.pool = GatedAttentionMILPool(in_dim=emb_dim, attn_dim=attn_dim, dropout=mil_dropout)
        elif self.aggregation == "subject_fusion":
            self.fusion_head = SubjectFusionHead(
                in_dim=emb_dim,
                num_classes=int(num_classes),
                hidden_dim=int(mil_cfg.get("fusion_hidden_dim", 128)),
                fusion_dim=int(mil_cfg.get("fusion_dim", 128)),
                instance_logit_dim=int(num_classes),
                dropout=float(mil_cfg.get("fusion_dropout", 0.2)),
                use_mean_max=bool(mil_cfg.get("fusion_use_mean_max", True)),
            )

    def forward_bag(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        pyg_batch = batch["pyg_batch"]
        bag_indices = batch["bag_indices"]
        labels = batch["labels"]
        subject_ids = batch["subject_ids"]

        out = self.instance_model(pyg_batch, return_dict=True)
        agg = aggregate_subject_predictions(
            instance_embeddings=out.embedding,
            instance_logits=out.logits,
            bag_indices=bag_indices.to(device=out.embedding.device),
            method=self.aggregation,
            classifier=self.instance_model.classifier,
            pool=self.pool,
            fusion_head=self.fusion_head,
        )
        return {
            "logits": agg["subject_logits"],
            "probs": agg["subject_prob"],
            "preds": agg["subject_pred"],
            "targets": labels.to(device=out.embedding.device),
            "subject_ids": list(subject_ids),
            "attention_weights": agg.get("attention_weights"),
        }


# -----------------------------------------------------------------------------
# Prediction table helpers
# -----------------------------------------------------------------------------


def _prediction_result_to_df(
    pred: Mapping[str, Any],
    *,
    split: str,
    fold: int,
    split_seed: int,
    train_seed: int,
    source_level: str,
) -> pd.DataFrame:
    y_true = np.asarray(pred["y_true"], dtype=np.int64).reshape(-1)
    probs = pred.get("probs", None)
    logits = pred.get("logits", None)
    y_pred = pred.get("y_pred", None)
    subject_ids = pred.get("subject_ids", None)

    n = int(len(y_true))
    if subject_ids is None:
        subject_ids = [f"subject_{i}" for i in range(n)]
    else:
        subject_ids = list(subject_ids)

    rows: list[dict[str, Any]] = []
    probs_np = None if probs is None else np.asarray(probs, dtype=np.float64)
    logits_np = None if logits is None else np.asarray(logits, dtype=np.float64)
    preds_np = None if y_pred is None else np.asarray(y_pred, dtype=np.int64).reshape(-1)

    for i in range(n):
        row = {
            "subject_id": str(subject_ids[i]),
            "true_label": int(y_true[i]),
            "split": str(split),
            "fold": int(fold),
            "split_seed": int(split_seed),
            "train_seed": int(train_seed),
            "source_level": str(source_level),
        }
        if preds_np is not None:
            row["pred_label"] = int(preds_np[i])
        if probs_np is not None:
            for c in range(probs_np.shape[1]):
                row[f"prob_{c}"] = float(probs_np[i, c])
        if logits_np is not None:
            for c in range(logits_np.shape[1]):
                row[f"logit_{c}"] = float(logits_np[i, c])
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# One-fold runner
# -----------------------------------------------------------------------------


def _run_one_fold(
    *,
    cfg: Mapping[str, Any],
    payload: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    fold: int,
    split_seed: int,
    train_seed: int,
    output_dir: str | os.PathLike,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_dir = Path(output_dir) / f"fold_{int(fold):02d}"
    ensure_dir(fold_dir)

    set_seed(int(train_seed) + int(fold))

    family = _canonical_name(_cfg_get(cfg, "experiment", "model_family", default="node_only_mlp"))
    graph_level = _canonical_name(_cfg_get(cfg, "experiment", "graph_level", default="segment"))
    aggregation = _canonical_name(_cfg_get(cfg, "experiment", "subject_aggregation", default="mean_mil"))

    train_cfg = dict(_cfg_get(cfg, "train", default={}) or {})
    mil_cfg = dict(_cfg_get(cfg, "mil", default={}) or {})
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_bag_limit = train_cfg.get("train_max_instances_per_subject", None)
    eval_bag_limit = train_cfg.get("eval_max_instances_per_subject", None)
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    epochs = int(train_cfg.get("epochs", 50))
    patience = train_cfg.get("patience", 10)
    loss_name = str(train_cfg.get("loss_name", "cross_entropy"))
    monitor = str(train_cfg.get("monitor", "balanced_accuracy"))

    is_graph = family in _GRAPH_FAMILIES
    require_graph_bank = family == "fused_graph_bank_gnn" or (
        family == "dual_branch_graph_model" and bool(_cfg_get(cfg, "gnn", "use_graph_bank", default=False))
    )

    num_classes = len(sorted({int(payload[sid]["label"]) for sid in payload.keys()}))

    if is_graph:
        train_bags = _build_graph_subject_bags(payload=payload, subject_ids=train_ids, cfg=cfg, require_graph_bank=require_graph_bank)
        val_bags = _build_graph_subject_bags(payload=payload, subject_ids=val_ids, cfg=cfg, require_graph_bank=require_graph_bank)
        test_bags = _build_graph_subject_bags(payload=payload, subject_ids=test_ids, cfg=cfg, require_graph_bank=require_graph_bank)

        train_ds = GraphSubjectBagDataset(train_bags, max_instances=train_bag_limit, seed=train_seed)
        val_ds = GraphSubjectBagDataset(val_bags, max_instances=eval_bag_limit, seed=train_seed)
        test_ds = GraphSubjectBagDataset(test_bags, max_instances=eval_bag_limit, seed=train_seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=graph_subject_bag_collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=graph_subject_bag_collate)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=graph_subject_bag_collate)

        instance_model = _build_graph_instance_model(cfg, train_bags[0], num_classes=num_classes)
        model = GraphSubjectModel(instance_model, aggregation=aggregation, mil_cfg=mil_cfg, num_classes=num_classes).to(device)
        forward_fn = lambda m, batch, trainer: m.forward_bag(batch)

    else:
        train_bags = _build_dense_subject_bags(payload=payload, subject_ids=train_ids, cfg=cfg)
        val_bags = _build_dense_subject_bags(payload=payload, subject_ids=val_ids, cfg=cfg)
        test_bags = _build_dense_subject_bags(payload=payload, subject_ids=test_ids, cfg=cfg)

        train_ds = DenseSubjectBagDataset(train_bags, max_instances=train_bag_limit, seed=train_seed)
        val_ds = DenseSubjectBagDataset(val_bags, max_instances=eval_bag_limit, seed=train_seed)
        test_ds = DenseSubjectBagDataset(test_bags, max_instances=eval_bag_limit, seed=train_seed)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=dense_subject_bag_collate)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dense_subject_bag_collate)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dense_subject_bag_collate)

        instance_model = _build_dense_instance_model(cfg, train_bags[0], num_classes=num_classes)
        model = DenseSubjectModel(instance_model, aggregation=aggregation, mil_cfg=mil_cfg, num_classes=num_classes).to(device)
        forward_fn = lambda m, batch, trainer: m.forward_bag(batch)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    trainer = Trainer(
        model,
        optimizer=optimizer,
        device=device,
        loss_name=loss_name,
        num_classes=num_classes,
        monitor=monitor,
        early_stopping_patience=None if patience is None else int(patience),
        checkpoint_dir=fold_dir,
        checkpoint_name="best_model.pt",
        save_best_only=True,
        forward_fn=forward_fn,
        verbose=bool(train_cfg.get("verbose", True)),
    )

    fit_out = trainer.fit(train_loader, val_loader, num_epochs=epochs, start_epoch=1)
    best_ckpt = fit_out.get("best_checkpoint_path", None)
    if best_ckpt:
        trainer.load_checkpoint(best_ckpt, load_optimizer=False, load_scheduler=False)

    preds_by_split = {
        "train": trainer.predict(train_loader, split_name="train", compute_loss=True, compute_metrics=True),
        "val": trainer.predict(val_loader, split_name="val", compute_loss=True, compute_metrics=True),
        "test": trainer.predict(test_loader, split_name="test", compute_loss=True, compute_metrics=True),
    }

    fold_rows: list[dict[str, Any]] = []
    fold_summary_out: dict[str, Any] = {
        "fold": int(fold),
        "best_epoch": fit_out.get("best_epoch"),
        "best_monitor_value": fit_out.get("best_monitor_value"),
        "best_checkpoint_path": fit_out.get("best_checkpoint_path"),
    }

    for split_name, pred in preds_by_split.items():
        pred_df = _prediction_result_to_df(
            pred,
            split=split_name,
            fold=fold,
            split_seed=split_seed,
            train_seed=train_seed,
            source_level=str(graph_level),
        )
        save_predictions_csv(pred_df, fold_dir / f"{split_name}_subject_predictions.csv")

        split_summary = summarize_fold_results(pred_df)
        split_summary["fold"] = int(fold)
        split_summary["split_seed"] = int(split_seed)
        split_summary["train_seed"] = int(train_seed)
        fold_rows.append(split_summary)
        fold_summary_out[split_name] = split_summary

    _save_json(_jsonable(fit_out), fold_dir / "fit_summary.json")
    _save_json(_jsonable(fold_summary_out), fold_dir / "fold_summary.json")

    return fold_rows, fold_summary_out


# -----------------------------------------------------------------------------
# Top-level AHEAP runner
# -----------------------------------------------------------------------------


def run_aheap_experiment(
    *,
    config: Mapping[str, Any] | None = None,
    config_path: str | os.PathLike | None = None,
    output_dir: str | os.PathLike,
    split_seed: int,
    train_seed: int,
) -> None:
    if config is None and config_path is None:
        raise ValueError("Provide either config or config_path.")
    cfg = dict(config) if config is not None else _read_config(config_path)  # type: ignore[arg-type]

    dataset_name = _canonical_name(_cfg_get(cfg, "dataset", "name", default="aheap"))
    if dataset_name != "aheap":
        raise ValueError(f"run_aheap_experiment only supports dataset.name='aheap', got {dataset_name!r}")

    h5_path = _cfg_get(cfg, "dataset", "h5_path")
    if not h5_path:
        raise ValueError("dataset.h5_path is required.")

    output_dir = str(output_dir)
    ensure_dir(output_dir)
    device = torch.device(get_device(_cfg_get(cfg, "train", "device", default=None)))
    set_seed(int(train_seed))

    meta = {
        "config_path": None if config_path is None else str(config_path),
        "dataset_name": dataset_name,
        "h5_path": str(h5_path),
        "split_seed": int(split_seed),
        "train_seed": int(train_seed),
        "device": str(device),
    }
    _save_json(meta, Path(output_dir) / "run_meta.json")
    _save_json(_jsonable(cfg), Path(output_dir) / "resolved_config.json")

    available = list_available_groups(h5_path)
    payload_index = load_aheap_h5_payload(cfg, available["subjects"])
    subject_ids = sorted(payload_index.keys())
    labels = [int(payload_index[sid]["label"]) for sid in subject_ids]

    folds = make_aheap_folds(
        subject_ids=subject_ids,
        labels=labels,
        n_splits=int(_cfg_get(cfg, "evaluation", "n_splits", default=5)),
        val_ratio=float(_cfg_get(cfg, "evaluation", "val_ratio", default=0.2)),
        split_seed=int(split_seed),
    )
    payload = load_aheap_h5_payload(cfg, subject_ids)

    all_rows: list[dict[str, Any]] = []
    all_fold_summaries: list[dict[str, Any]] = []
    for spec in folds:
        fold_rows, fold_summary = _run_one_fold(
            cfg=cfg,
            payload=payload,
            train_ids=spec["train_ids"],
            val_ids=spec["val_ids"],
            test_ids=spec["test_ids"],
            fold=int(spec["fold"]),
            split_seed=int(split_seed),
            train_seed=int(train_seed),
            output_dir=output_dir,
            device=device,
        )
        all_rows.extend(fold_rows)
        all_fold_summaries.append(fold_summary)

    rows_df = pd.DataFrame(all_rows)
    if not rows_df.empty:
        rows_df.to_csv(Path(output_dir) / "all_fold_split_summaries.csv", index=False)
        cv_summary = summarize_cv_results(rows_df.to_dict("records"), group_by=("split",))
    else:
        cv_summary = {}
    save_summary_json(cv_summary, Path(output_dir) / "cv_summary.json")
    _save_json(_jsonable(all_fold_summaries), Path(output_dir) / "all_fold_summaries.json")

    print(f"[done] Results saved to: {output_dir}")


# -----------------------------------------------------------------------------
# Experiment ladder helpers
# -----------------------------------------------------------------------------


def make_base_aheap_config(
    *,
    h5_path: str | os.PathLike,
    feature_families: Sequence[str] = ("relative_band_power", "hjorth"),
    connectivity_metric: str = "coherence",
    connectivity_band: str | None = "alpha",
) -> dict[str, Any]:
    """Base config that all ladder items override."""
    return {
        "dataset": {
            "name": "aheap",
            "h5_path": str(h5_path),
        },
        "features": {
            "families": list(feature_families),
        },
        "connectivity": {
            "primary_metric": str(connectivity_metric),
            "primary_band": connectivity_band,
            "metrics": ["pearson", "coherence", "pli", "wpli"],
            "allow_multiband": True,
            "multiband_reduce": "mean",
        },
        "experiment": {
            "graph_level": "segment",
            "model_family": "node_only_mlp",
            "subject_aggregation": "mean_mil",
        },
        "macro": {
            "windows_per_macro": 10,
            "feature_aggregation": "mean",
            "connectivity_aggregation": "mean",
        },
        "graph": {
            "topology": "fixed",
            "edge_weight": "binary",
            "fixed_edges": "mono_fixed",
            "topk": 3,
            "threshold": 0.2,
            "connectivity_reduce": "mean",
            "topology_rule": "union",
            "vote_threshold": 0.5,
        },
        "dense": {
            "node_readout": "flatten",
            "hidden_dims": [256, 128],
            "emb_dim": 128,
            "dropout": 0.2,
            "use_batchnorm": False,
            "connectivity_encoder_type": "mlp",
            "connectivity_hidden_dims": [256, 128],
            "connectivity_emb_dim": 128,
            "connectivity_flatten_mode": "upper_triangle",
            "connectivity_symmetrize": True,
            "connectivity_include_diagonal": False,
            "connectivity_conv_channels": [16, 32, 64],
            "connectivity_kernel_sizes": [3, 3, 3],
            "fusion_mode": "concat",
            "fusion_emb_dim": 128,
        },
        "gnn": {
            "backbone": "gcn",
            "hidden_dim": 64,
            "graph_emb_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "gat_heads": 4,
            "use_edge_weight": True,
            "use_batchnorm": True,
            "node_pooling_type": "none",
            "node_pool_ratio": 0.8,
            "readout_type": "mean",
            "readout_hidden_dim": 64,
            "readout_dropout": 0.0,
            "return_attention_weights": False,
            "graph_bank_fusion_mode": "summary_gated",
            "fusion_hidden_dim": 64,
            "use_graph_bank": False,
            "node_readout_type": "mean",
            "graph_readout_type": "mean",
            "fusion_mode": "concat",
            "fusion_emb_dim": 128,
            "fusion_dropout": 0.2,
        },
        "mil": {
            "attention_dim": 128,
            "dropout": 0.0,
            "fusion_hidden_dim": 128,
            "fusion_dim": 128,
            "fusion_dropout": 0.2,
            "fusion_use_mean_max": True,
        },
        "train": {
            "device": None,
            "batch_size": 8,
            "num_workers": 0,
            "train_max_instances_per_subject": None,
            "eval_max_instances_per_subject": None,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 40,
            "patience": 10,
            "loss_name": "cross_entropy",
            "monitor": "balanced_accuracy",
            "verbose": True,
        },
        "evaluation": {
            "n_splits": 5,
            "val_ratio": 0.2,
        },
    }



def make_band_graph_bank_specs(
    *,
    metric: str,
    bands: Sequence[str] = DEFAULT_BAND_ORDER,
    topology: str = "connectivity_topk",
    edge_weight: str = "connectivity",
    topk: int = 3,
) -> list[dict[str, Any]]:
    return [
        {
            "name": f"{metric}_{band}",
            "metric": str(metric),
            "band": str(band),
            "topology": str(topology),
            "edge_weight": str(edge_weight),
            "topk": int(topk),
        }
        for band in bands
    ]



def make_metric_graph_bank_specs(
    *,
    metrics: Sequence[str],
    band: str = "alpha",
    topology: str = "connectivity_topk",
    edge_weight: str = "connectivity",
    topk: int = 3,
) -> list[dict[str, Any]]:
    return [
        {
            "name": f"{metric}_{band}",
            "metric": str(metric),
            "band": str(band),
            "topology": str(topology),
            "edge_weight": str(edge_weight),
            "topk": int(topk),
        }
        for metric in metrics
    ]



def _add_spec(
    ladder: list[AHEAPExperimentSpec],
    *,
    name: str,
    output_root: str | os.PathLike,
    base_cfg: Mapping[str, Any],
    override: Mapping[str, Any],
    split_seeds: Sequence[int],
    train_seeds: Sequence[int],
    notes: str = "",
) -> None:
    cfg = _deep_update(base_cfg, override)
    ladder.append(
        AHEAPExperimentSpec(
            name=name,
            config=cfg,
            output_dir=str(Path(output_root) / name),
            split_seeds=[int(x) for x in split_seeds],
            train_seeds=[int(x) for x in train_seeds],
            notes=notes,
        )
    )



def build_default_aheap_ladder(
    *,
    h5_path: str | os.PathLike,
    output_root: str | os.PathLike,
    split_seeds: Sequence[int] = (101,),
    train_seeds: Sequence[int] = (11,),
    feature_families: Sequence[str] = ("relative_band_power", "hjorth"),
) -> list[AHEAPExperimentSpec]:
    """
    Build a staged, manageable ladder instead of a blind Cartesian product.

    The ladder is organized by the experiment axes requested by the user.
    Each block changes a small number of axes at a time so the comparisons are
    interpretable.
    """
    base = make_base_aheap_config(
        h5_path=h5_path,
        feature_families=feature_families,
    )
    ladder: list[AHEAPExperimentSpec] = []

    # ------------------------------------------------------------------
    # Block 1: graph construction level x subject aggregation on node-only
    # ------------------------------------------------------------------
    for graph_level, aggregation in [
        ("segment", "mean_mil"),
        ("segment", "gated_attention_mil"),
        ("macro", "mean_mil"),
        ("macro", "subject_fusion"),
        ("subject", "none"),
    ]:
        _add_spec(
            ladder,
            name=f"B1_node_{graph_level}_{aggregation}",
            output_root=output_root,
            base_cfg=base,
            override={
                "experiment": {
                    "graph_level": graph_level,
                    "model_family": "node_only_mlp",
                    "subject_aggregation": aggregation,
                },
            },
            split_seeds=split_seeds,
            train_seeds=train_seeds,
            notes="Node-only baselines across graph construction level and subject aggregation.",
        )

    # ------------------------------------------------------------------
    # Block 2: connectivity-only dense baselines
    # ------------------------------------------------------------------
    for metric, band in [("coherence", "alpha"), ("pli", "alpha"), ("wpli", "alpha")]:
        for family in ["connectivity_only_mlp", "connectivity_only_cnn"]:
            _add_spec(
                ladder,
                name=f"B2_{family}_{metric}_{band}",
                output_root=output_root,
                base_cfg=base,
                override={
                    "experiment": {
                        "graph_level": "segment",
                        "model_family": family,
                        "subject_aggregation": "mean_mil",
                    },
                    "connectivity": {
                        "primary_metric": metric,
                        "primary_band": band,
                        "allow_multiband": family == "connectivity_only_cnn",
                    },
                },
                split_seeds=split_seeds,
                train_seeds=train_seeds,
                notes="Connectivity-only dense baselines over metric and encoder type.",
            )

    _add_spec(
        ladder,
        name="B2_connectivity_only_cnn_coherence_multiband",
        output_root=output_root,
        base_cfg=base,
        override={
            "experiment": {
                "graph_level": "segment",
                "model_family": "connectivity_only_cnn",
                "subject_aggregation": "mean_mil",
            },
            "connectivity": {
                "primary_metric": "coherence",
                "primary_band": None,
                "allow_multiband": True,
            },
        },
        split_seeds=split_seeds,
        train_seeds=train_seeds,
        notes="Connectivity-only multiband CNN baseline.",
    )

    # ------------------------------------------------------------------
    # Block 3: dense dual-branch
    # ------------------------------------------------------------------
    for metric, band in [("coherence", "alpha"), ("coherence", None), ("pli", "alpha")]:
        metric_tag = metric if band is None else f"{metric}_{band}"
        _add_spec(
            ladder,
            name=f"B3_dual_branch_dense_{metric_tag}",
            output_root=output_root,
            base_cfg=base,
            override={
                "experiment": {
                    "graph_level": "segment",
                    "model_family": "dual_branch_dense",
                    "subject_aggregation": "gated_attention_mil",
                },
                "connectivity": {
                    "primary_metric": metric,
                    "primary_band": band,
                    "allow_multiband": band is None,
                },
                "dense": {
                    "connectivity_encoder_type": "cnn" if band is None else "mlp",
                },
            },
            split_seeds=split_seeds,
            train_seeds=train_seeds,
            notes="Dense node+connectivity dual-branch models.",
        )

    # ------------------------------------------------------------------
    # Block 4: simple fixed-graph GNN, topology x edge weights x readout
    # ------------------------------------------------------------------
    for topology, edge_weight in [
        ("fixed", "binary"),
        ("fixed", "connectivity"),
        ("fixed", "normalized"),
        ("connectivity_topk", "connectivity"),
        ("connectivity_mst", "connectivity"),
        ("feature_induced", "similarity"),
    ]:
        for readout in ["mean", "mean_max_concat", "attention"]:
            _add_spec(
                ladder,
                name=f"B4_gnn_{topology}_{edge_weight}_{readout}",
                output_root=output_root,
                base_cfg=base,
                override={
                    "experiment": {
                        "graph_level": "segment",
                        "model_family": "simple_fixed_graph_gnn",
                        "subject_aggregation": "gated_attention_mil",
                    },
                    "graph": {
                        "topology": topology,
                        "edge_weight": edge_weight,
                        "connectivity_metric": "coherence",
                        "connectivity_band": "alpha",
                        "topk": 3,
                    },
                    "gnn": {
                        "readout_type": readout,
                    },
                },
                split_seeds=split_seeds,
                train_seeds=train_seeds,
                notes="Simple GNN sweep over topology strategy, edge weights, and graph readout.",
            )

    # ------------------------------------------------------------------
    # Block 5: fused graph bank GNN
    # ------------------------------------------------------------------
    _add_spec(
        ladder,
        name="B5_fused_bank_bands_coherence",
        output_root=output_root,
        base_cfg=base,
        override={
            "experiment": {
                "graph_level": "segment",
                "model_family": "fused_graph_bank_gnn",
                "subject_aggregation": "gated_attention_mil",
            },
            "graph": {
                "topology": "fused_bank",
                "edge_weight": "fused_weights",
                "connectivity_metric": "coherence",
                "graph_bank_specs": make_band_graph_bank_specs(
                    metric="coherence",
                    bands=DEFAULT_BAND_ORDER,
                    topology="connectivity_topk",
                    edge_weight="connectivity",
                    topk=3,
                ),
            },
            "gnn": {
                "readout_type": "attention",
                "graph_bank_fusion_mode": "summary_gated",
                "topology_rule": "union",
            },
        },
        split_seeds=split_seeds,
        train_seeds=train_seeds,
        notes="Graph bank over coherence bands.",
    )

    _add_spec(
        ladder,
        name="B5_fused_bank_metrics_alpha",
        output_root=output_root,
        base_cfg=base,
        override={
            "experiment": {
                "graph_level": "segment",
                "model_family": "fused_graph_bank_gnn",
                "subject_aggregation": "gated_attention_mil",
            },
            "graph": {
                "topology": "fused_bank",
                "edge_weight": "fused_weights",
                "connectivity_metric": "coherence",
                "graph_bank_specs": make_metric_graph_bank_specs(
                    metrics=["coherence", "pli", "wpli"],
                    band="alpha",
                    topology="connectivity_topk",
                    edge_weight="connectivity",
                    topk=3,
                ),
            },
            "gnn": {
                "readout_type": "attention",
                "graph_bank_fusion_mode": "summary_gated",
                "topology_rule": "vote",
                "vote_threshold": 0.5,
            },
        },
        split_seeds=split_seeds,
        train_seeds=train_seeds,
        notes="Graph bank over metrics at alpha band.",
    )

    # ------------------------------------------------------------------
    # Block 6: dual-branch graph model
    # ------------------------------------------------------------------
    _add_spec(
        ladder,
        name="B6_dual_branch_graph_fixed",
        output_root=output_root,
        base_cfg=base,
        override={
            "experiment": {
                "graph_level": "segment",
                "model_family": "dual_branch_graph_model",
                "subject_aggregation": "gated_attention_mil",
            },
            "graph": {
                "topology": "fixed",
                "edge_weight": "connectivity",
                "connectivity_metric": "coherence",
                "connectivity_band": "alpha",
            },
            "gnn": {
                "use_graph_bank": False,
                "node_readout_type": "mean",
                "graph_readout_type": "attention",
                "fusion_mode": "gated",
            },
        },
        split_seeds=split_seeds,
        train_seeds=train_seeds,
        notes="Dual-branch graph model on one graph.",
    )

    _add_spec(
        ladder,
        name="B6_dual_branch_graph_bank",
        output_root=output_root,
        base_cfg=base,
        override={
            "experiment": {
                "graph_level": "segment",
                "model_family": "dual_branch_graph_model",
                "subject_aggregation": "gated_attention_mil",
            },
            "graph": {
                "topology": "fused_bank",
                "edge_weight": "fused_weights",
                "connectivity_metric": "coherence",
                "graph_bank_specs": make_band_graph_bank_specs(
                    metric="coherence",
                    bands=DEFAULT_BAND_ORDER,
                    topology="connectivity_topk",
                    edge_weight="connectivity",
                    topk=3,
                ),
            },
            "gnn": {
                "use_graph_bank": True,
                "node_readout_type": "mean",
                "graph_readout_type": "attention",
                "fusion_mode": "gated",
                "graph_bank_fusion_mode": "summary_gated",
            },
        },
        split_seeds=split_seeds,
        train_seeds=train_seeds,
        notes="Dual-branch graph model with learned graph-bank fusion.",
    )

    # ------------------------------------------------------------------
    # Block 7: macro/subject graph follow-up on stronger families
    # ------------------------------------------------------------------
    for graph_level, aggregation in [("macro", "subject_fusion"), ("subject", "none")]:
        _add_spec(
            ladder,
            name=f"B7_dual_branch_graph_bank_{graph_level}_{aggregation}",
            output_root=output_root,
            base_cfg=base,
            override={
                "experiment": {
                    "graph_level": graph_level,
                    "model_family": "dual_branch_graph_model",
                    "subject_aggregation": aggregation,
                },
                "graph": {
                    "topology": "fused_bank",
                    "edge_weight": "fused_weights",
                    "connectivity_metric": "coherence",
                    "graph_bank_specs": make_band_graph_bank_specs(
                        metric="coherence",
                        bands=DEFAULT_BAND_ORDER,
                        topology="connectivity_topk",
                        edge_weight="connectivity",
                        topk=3,
                    ),
                },
                "gnn": {
                    "use_graph_bank": True,
                    "node_readout_type": "mean",
                    "graph_readout_type": "attention",
                    "fusion_mode": "gated",
                    "graph_bank_fusion_mode": "summary_gated",
                },
            },
            split_seeds=split_seeds,
            train_seeds=train_seeds,
            notes="Follow-up on larger graph-construction levels using a stronger graph model.",
        )

    return ladder


# -----------------------------------------------------------------------------
# Ladder execution helpers
# -----------------------------------------------------------------------------


def run_aheap_ladder_item(spec: AHEAPExperimentSpec) -> None:
    ensure_dir(spec.output_dir)
    _save_json(_jsonable(spec.config), Path(spec.output_dir) / "ladder_config.json")
    _save_json(
        {
            "name": spec.name,
            "split_seeds": spec.split_seeds,
            "train_seeds": spec.train_seeds,
            "notes": spec.notes,
        },
        Path(spec.output_dir) / "ladder_meta.json",
    )

    for split_seed in spec.split_seeds:
        for train_seed in spec.train_seeds:
            run_dir = Path(spec.output_dir) / f"split_seed_{int(split_seed):03d}" / f"train_seed_{int(train_seed):03d}"
            run_aheap_experiment(
                config=spec.config,
                output_dir=run_dir,
                split_seed=int(split_seed),
                train_seed=int(train_seed),
            )



def write_ladder_summary(ladder: Sequence[AHEAPExperimentSpec], path: str | os.PathLike) -> str:
    rows = []
    for spec in ladder:
        rows.append(
            {
                "name": spec.name,
                "output_dir": spec.output_dir,
                "split_seeds": list(spec.split_seeds),
                "train_seeds": list(spec.train_seeds),
                "notes": spec.notes,
                "model_family": _cfg_get(spec.config, "experiment", "model_family"),
                "graph_level": _cfg_get(spec.config, "experiment", "graph_level"),
                "subject_aggregation": _cfg_get(spec.config, "experiment", "subject_aggregation"),
                "topology": _cfg_get(spec.config, "graph", "topology"),
                "edge_weight": _cfg_get(spec.config, "graph", "edge_weight"),
                "connectivity_metric": _cfg_get(spec.config, "connectivity", "primary_metric"),
                "connectivity_band": _cfg_get(spec.config, "connectivity", "primary_band"),
            }
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AHEAP H5-first experiment runner and ladder builder")
    sub = p.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run-config", help="Run one config file")
    p_run.add_argument("--config", required=True, help="Path to config (.json/.yaml)")
    p_run.add_argument("--output-dir", required=True, help="Output directory")
    p_run.add_argument("--split-seed", type=int, default=101)
    p_run.add_argument("--train-seed", type=int, default=11)

    p_ladder = sub.add_parser("write-default-ladder", help="Write the default AHEAP ladder configs")
    p_ladder.add_argument("--h5-path", required=True)
    p_ladder.add_argument("--output-root", required=True)
    p_ladder.add_argument("--split-seeds", type=int, nargs="*", default=[15, 42, 100])
    p_ladder.add_argument("--train-seeds", type=int, nargs="*", default=[11])

    p_run_ladder = sub.add_parser("run-default-ladder", help="Build and run the default AHEAP ladder")
    p_run_ladder.add_argument("--h5-path", required=True)
    p_run_ladder.add_argument("--output-root", required=True)
    p_run_ladder.add_argument("--split-seeds", type=int, nargs="*", default=[15, 42, 100])
    p_run_ladder.add_argument("--train-seeds", type=int, nargs="*", default=[11])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.command == "run-config":
        run_aheap_experiment(
            config_path=args.config,
            output_dir=args.output_dir,
            split_seed=args.split_seed,
            train_seed=args.train_seed,
        )

    elif args.command == "write-default-ladder":
        ladder = build_default_aheap_ladder(
            h5_path=args.h5_path,
            output_root=args.output_root,
            split_seeds=args.split_seeds,
            train_seeds=args.train_seeds,
        )
        ensure_dir(args.output_root)
        for spec in ladder:
            spec_dir = Path(spec.output_dir)
            ensure_dir(spec_dir)
            _save_json(_jsonable(spec.config), spec_dir / "ladder_config.json")
        summary_path = write_ladder_summary(ladder, Path(args.output_root) / "ladder_summary.csv")
        print(f"[done] Wrote {len(ladder)} ladder items to {args.output_root}")
        print(f"[done] Summary: {summary_path}")

    elif args.command == "run-default-ladder":
        ladder = build_default_aheap_ladder(
            h5_path=args.h5_path,
            output_root=args.output_root,
            split_seeds=args.split_seeds,
            train_seeds=args.train_seeds,
        )
        summary_path = write_ladder_summary(ladder, Path(args.output_root) / "ladder_summary.csv")
        print(f"[info] Running {len(ladder)} ladder items")
        print(f"[info] Summary: {summary_path}")
        for spec in ladder:
            print(f"[run] {spec.name}")
            run_aheap_ladder_item(spec)
