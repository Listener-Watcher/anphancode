from __future__ import annotations

"""
H5-first experiment runner for the reusable EEG dementia ladder.

Key behavior
------------
- Reuses an existing master H5 instead of recomputing preprocessing/features/connectivity.
- AHEAP uses split-seed-driven CV splits.
- CAUEEG uses the official fixed train/validation/test JSON split.
- Supports dense and graph model families on segment / macro / subject levels.
- Treats subject aggregation as a separate axis via simple mean / MIL / subject fusion.

This file is intentionally practical and self-contained so it can sit next to the
existing project modules without forcing a big refactor.
"""

import argparse
import json
import math
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from dense import (
    ConnectivityOnlyCNN,
    ConnectivityOnlyMLP,
    DualBranchDenseModel,
    NodeOnlyMLP,
)
from evaluate import (
    save_predictions_csv,
    save_summary_json,
    summarize_cv_results,
    summarize_fold_results,
)
from models_mil import aggregate_subject_predictions
from trainer import Trainer
from utils import ensure_dir, get_device, load_yaml_config, set_seed




try:
    from data_config import MONOFIXEDGES
except Exception:
    MONOFIXEDGES = []


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from master_builder import list_available_groups, load_selected_groups

# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------


def _read_config(path: str | os.PathLike) -> dict[str, Any]:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return dict(load_yaml_config(path))
    with open(path, "r", encoding="utf-8") as f:
        return dict(json.load(f))



def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj



def _cfg_get(cfg: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur



def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


# -----------------------------------------------------------------------------
# Dataset split helpers
# -----------------------------------------------------------------------------


def _load_caueeg_split_json(dataset_path: str | os.PathLike, task: str) -> dict[str, Any]:
    task = str(task).lower()
    json_path = Path(dataset_path) / f"{task}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"CAUEEG split file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)



def _resolve_h5_subject_id(
    raw_subject_id: str,
    *,
    available_subject_ids: set[str],
    split_name: str | None = None,
    prefix_mode: str = "auto",
) -> str:
    sid = str(raw_subject_id)
    split_name = None if split_name is None else str(split_name).lower()
    mode = _normalize_name(prefix_mode)

    candidates: list[str] = []
    if mode in {"none", "raw"}:
        candidates = [sid]
    elif mode in {"split", "prefixed"}:
        if split_name is None:
            raise ValueError("prefix_mode='split' requires split_name.")
        candidates = [f"{split_name}_{sid}"]
    else:
        # auto: try the explicit split prefix first, then raw id.
        if split_name is not None:
            candidates.append(f"{split_name}_{sid}")
            if split_name == "validation":
                candidates.append(f"val_{sid}")
            if split_name == "val":
                candidates.append(f"validation_{sid}")
        candidates.extend([sid, f"train_{sid}", f"val_{sid}", f"validation_{sid}", f"test_{sid}"])

    for cand in candidates:
        if cand in available_subject_ids:
            return cand

    raise KeyError(
        f"Could not resolve H5 subject id for raw id {sid!r}. Tried: {candidates}."
    )



def _make_aheap_folds(
    *,
    subject_ids: Sequence[str],
    labels: Sequence[int],
    n_splits: int,
    val_ratio: float,
    split_seed: int,
) -> list[dict[str, Any]]:
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

    fold_specs: list[dict[str, Any]] = []
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

        fold_specs.append(
            {
                "fold": int(fold_idx),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
                "split_name": f"fold_{fold_idx}",
            }
        )

    return fold_specs



def _make_caueeg_fixed_split(
    *,
    dataset_path: str | os.PathLike,
    task: str,
    available_subject_ids: set[str],
    prefix_mode: str = "auto",
) -> list[dict[str, Any]]:
    split_json = _load_caueeg_split_json(dataset_path, task)

    def _resolve_split(entries: Sequence[Mapping[str, Any]], split_name: str) -> list[str]:
        out: list[str] = []
        for row in entries:
            raw_id = str(row["serial"])
            out.append(
                _resolve_h5_subject_id(
                    raw_id,
                    available_subject_ids=available_subject_ids,
                    split_name=split_name,
                    prefix_mode=prefix_mode,
                )
            )
        return out

    train_ids = _resolve_split(split_json["train_split"], "train")
    val_ids = _resolve_split(split_json["validation_split"], "val")
    test_ids = _resolve_split(split_json["test_split"], "test")

    return [
        {
            "fold": 0,
            "train_ids": train_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "split_name": "official_caueeg_split",
        }
    ]


# -----------------------------------------------------------------------------
# H5 payload loading
# -----------------------------------------------------------------------------


def _should_load_connectivity(cfg: Mapping[str, Any]) -> bool:
    family = _normalize_name(_cfg_get(cfg, "experiment", "model_family", default="node_only"))
    topology = _normalize_name(_cfg_get(cfg, "graph", "topology", default="fixed"))
    edge_weight = _normalize_name(_cfg_get(cfg, "graph", "edge_weight", default="binary"))

    if family in {
        "connectivity_only_mlp",
        "connectivity_only_cnn",
        "dual_branch_dense",
        "simple_fixed_graph_gnn",
        "fused_graph_bank_gnn",
        "dual_branch_graph_model",
    }:
        return True
    if topology.startswith("connectivity") or topology.startswith("feature_induced"):
        return True
    if edge_weight in {"connectivity", "normalized", "similarity", "topology_weight", "fused"}:
        return True
    return False



def _load_h5_payload(cfg: Mapping[str, Any], subject_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    h5_path = _cfg_get(cfg, "dataset", "h5_path")
    if not h5_path:
        raise ValueError("dataset.h5_path is required.")

    feature_families = list(_cfg_get(cfg, "features", "families", default=[]))
    connectivity_metric = _cfg_get(cfg, "connectivity", "metric", default=None)
    connectivity_metrics = [str(connectivity_metric)] if (connectivity_metric and _should_load_connectivity(cfg)) else []

    payload = load_selected_groups(
        h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        subject_ids=list(subject_ids),
    )
    return payload


# -----------------------------------------------------------------------------
# Representation helpers
# -----------------------------------------------------------------------------


def _aggregate_array(x: np.ndarray, mode: str) -> np.ndarray:
    mode = _normalize_name(mode)
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
    raise ValueError(f"Unsupported aggregation mode={mode!r}")



def _concat_feature_families(subject_entry: Mapping[str, Any], feature_families: Sequence[str]) -> np.ndarray:
    if not feature_families:
        raise ValueError("At least one feature family is required for node-feature models.")
    feats = [np.asarray(subject_entry["features"][fam], dtype=np.float32) for fam in feature_families]
    ref_shape = feats[0].shape[:2]
    for fam, arr in zip(feature_families, feats):
        if arr.ndim != 3:
            raise ValueError(f"Feature family {fam!r} must have shape [W, N, F], got {arr.shape}")
        if arr.shape[:2] != ref_shape:
            raise ValueError(f"Feature family {fam!r} shape {arr.shape} does not align with {ref_shape}")
    return np.concatenate(feats, axis=-1).astype(np.float32)



def _select_connectivity_tensor(
    subject_entry: Mapping[str, Any],
    *,
    connectivity_metric: Optional[str],
    connectivity_band: Optional[int | str],
) -> Optional[np.ndarray]:
    if connectivity_metric is None:
        return None
    if connectivity_metric not in subject_entry.get("connectivity", {}):
        raise KeyError(f"Connectivity metric {connectivity_metric!r} not found in payload entry.")

    conn = np.asarray(subject_entry["connectivity"][connectivity_metric], dtype=np.float32)
    if conn.ndim == 4 and connectivity_band is not None:
        if isinstance(connectivity_band, str):
            band_names = ["delta", "theta", "alpha", "beta", "gamma"]
            key = str(connectivity_band).lower()
            if key not in band_names:
                raise KeyError(f"Unknown connectivity band {connectivity_band!r}")
            band_idx = band_names.index(key)
        else:
            band_idx = int(connectivity_band)
        conn = conn[:, band_idx]
    return conn.astype(np.float32)



def _reduce_connectivity_for_graph(conn: Optional[np.ndarray], reduce_mode: str = "mean") -> Optional[np.ndarray]:
    if conn is None:
        return None
    if conn.ndim == 3:
        return conn.astype(np.float32)
    if conn.ndim == 4:
        return _aggregate_array(conn, reduce_mode)
    raise ValueError(f"Unexpected connectivity tensor shape {conn.shape}")



def _build_groups_for_subject(
    subject_entry: Mapping[str, Any],
    *,
    graph_level: str,
    windows_per_macro: Optional[int],
    feature_agg: str,
    connectivity_agg: str,
    connectivity_reduce_for_graph: str,
    feature_families: Sequence[str],
    connectivity_metric: Optional[str],
    connectivity_band: Optional[int | str],
) -> list[dict[str, Any]]:
    graph_level = _normalize_name(graph_level)
    node_x_all = _concat_feature_families(subject_entry, feature_families)
    conn_all = _select_connectivity_tensor(
        subject_entry,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
    )

    num_windows = node_x_all.shape[0]
    seg_ids = np.asarray(subject_entry.get("segment_id", np.arange(num_windows)), dtype=np.int64)
    start_samples = np.asarray(subject_entry.get("start_sample", np.arange(num_windows)), dtype=np.int64)

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

        conn_red: Optional[np.ndarray] = None
        if conn_all is not None:
            c = conn_all[idxs]
            c_red = c[0] if len(idxs) == 1 else _aggregate_array(c, connectivity_agg)
            if c_red.ndim == 4:
                c_red = _aggregate_array(c_red, connectivity_reduce_for_graph)
            conn_red = c_red.astype(np.float32)

        out.append(
            {
                "instance_id": gid,
                "group_indices": idxs,
                "node_features": x_red.astype(np.float32),
                "connectivity": None if conn_red is None else conn_red.astype(np.float32),
                "start_sample": int(start_samples[idxs[0]]),
            }
        )
    return out


# -----------------------------------------------------------------------------
# Dense bag dataset
# -----------------------------------------------------------------------------


class DenseSubjectBagDataset(Dataset):
    def __init__(
        self,
        bags: Sequence[dict[str, Any]],
        *,
        max_instances: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.bags = [deepcopy(b) for b in bags]
        self.max_instances = None if max_instances is None else int(max_instances)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bag = deepcopy(self.bags[int(idx)])
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
    conn_tensor = None if conn_shape is None else torch.zeros((batch_size, max_k, *conn_shape), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_k), dtype=torch.bool)
    labels = torch.zeros((batch_size,), dtype=torch.long)
    subject_ids: list[str] = []
    instance_ids: list[list[str]] = []

    for b_idx, item in enumerate(batch):
        labels[b_idx] = int(item["label"])
        subject_ids.append(str(item["subject_id"]))
        cur_ids: list[str] = []
        for k_idx, inst in enumerate(item["instances"]):
            node_tensor[b_idx, k_idx] = torch.as_tensor(inst["node_features"], dtype=torch.float32)
            if conn_tensor is not None and inst.get("connectivity") is not None:
                conn_tensor[b_idx, k_idx] = torch.as_tensor(inst["connectivity"], dtype=torch.float32)
            mask[b_idx, k_idx] = True
            cur_ids.append(str(inst["instance_id"]))
        instance_ids.append(cur_ids)

    out = {
        "node_features": node_tensor,
        "labels": labels,
        "mask": mask,
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



def _make_topology_and_weights(
    *,
    node_features: np.ndarray,
    connectivity: Optional[np.ndarray],
    channel_names: Sequence[str],
    graph_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = node_features.shape[0]
    topology = _normalize_name(graph_cfg.get("topology", "fixed"))
    edge_weight = _normalize_name(graph_cfg.get("edge_weight", "connectivity"))
    topk = int(graph_cfg.get("topk", 3))
    threshold = float(graph_cfg.get("threshold", 0.0))
    fixed_edges_cfg = graph_cfg.get("fixed_edges", None)
    fixed_edges = MONOFIXEDGES if fixed_edges_cfg in {None, "mono_fixed"} else fixed_edges_cfg

    if topology == "fixed":
        topo = _make_fixed_topology(n_nodes, fixed_edges=fixed_edges, channel_names=channel_names)
        weight_source = np.asarray(connectivity, dtype=np.float32) if connectivity is not None else topo.copy()

    elif topology in {"connectivity", "connectivity_full", "full"}:
        if connectivity is None:
            raise ValueError("Connectivity-based topology requires a connectivity matrix.")
        topo = np.ones((n_nodes, n_nodes), dtype=np.float32)
        np.fill_diagonal(topo, 0.0)
        weight_source = np.asarray(connectivity, dtype=np.float32)

    elif topology in {"connectivity_topk", "topk"}:
        if connectivity is None:
            raise ValueError("Top-k topology requires a connectivity matrix.")
        topo = _topk_union_topology(np.asarray(connectivity, dtype=np.float32), topk=topk)
        weight_source = np.asarray(connectivity, dtype=np.float32)

    elif topology in {"connectivity_mst", "mst"}:
        if connectivity is None:
            raise ValueError("MST topology requires a connectivity matrix.")
        topo = _maximum_spanning_tree_topology(np.asarray(connectivity, dtype=np.float32))
        weight_source = np.asarray(connectivity, dtype=np.float32)

    elif topology in {"connectivity_threshold", "threshold"}:
        if connectivity is None:
            raise ValueError("Threshold topology requires a connectivity matrix.")
        weight_source = np.asarray(connectivity, dtype=np.float32)
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



def _build_graph_bank(
    *,
    node_features: np.ndarray,
    connectivity_tensor: Optional[np.ndarray],
    channel_names: Sequence[str],
    graph_cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    specs = graph_cfg.get("graph_bank_specs", None)
    if specs is None:
        if connectivity_tensor is None or connectivity_tensor.ndim != 3:
            raise ValueError(
                "graph_bank_specs is missing and connectivity tensor is not multiband [B,N,N]."
            )
        specs = [{"band_index": i} for i in range(connectivity_tensor.shape[0])]

    adj_bank: list[np.ndarray] = []
    topo_bank: list[np.ndarray] = []
    for spec in specs:
        spec = dict(spec)
        band_index = spec.get("band_index", None)
        band_name = spec.get("band", None)

        conn_matrix: Optional[np.ndarray] = None
        if connectivity_tensor is not None:
            if connectivity_tensor.ndim == 2:
                conn_matrix = connectivity_tensor
            elif connectivity_tensor.ndim == 3:
                if band_name is not None:
                    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
                    band_index = band_order.index(str(band_name).lower())
                if band_index is None:
                    raise ValueError("Each graph bank candidate needs band_index/band for multiband connectivity.")
                conn_matrix = connectivity_tensor[int(band_index)]
            else:
                raise ValueError(f"Unexpected connectivity tensor for graph bank: {connectivity_tensor.shape}")

        cand_graph_cfg = dict(graph_cfg)
        cand_graph_cfg.update(spec)
        topo, adj = _make_topology_and_weights(
            node_features=node_features,
            connectivity=conn_matrix,
            channel_names=channel_names,
            graph_cfg=cand_graph_cfg,
        )
        topo_bank.append(topo)
        adj_bank.append(adj)

    return np.stack(adj_bank, axis=0).astype(np.float32), np.stack(topo_bank, axis=0).astype(np.float32)



def _instance_to_pyg_data(
    *,
    node_features: np.ndarray,
    connectivity: Optional[np.ndarray],
    channel_names: Sequence[str],
    label: int,
    subject_id: str,
    instance_id: str,
    graph_cfg: Mapping[str, Any],
    require_graph_bank: bool,
):
    Data, _, dense_to_sparse = _import_pyg()

    conn_for_graph = None
    if connectivity is not None:
        if connectivity.ndim == 2:
            conn_for_graph = connectivity
        elif connectivity.ndim == 3:
            reduce_mode = _normalize_name(graph_cfg.get("connectivity_reduce", "mean"))
            conn_for_graph = _aggregate_array(connectivity, reduce_mode)
        else:
            raise ValueError(f"Unexpected connectivity instance shape {connectivity.shape}")

    topo, adj = _make_topology_and_weights(
        node_features=node_features,
        connectivity=conn_for_graph,
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

    if require_graph_bank:
        adj_bank, topo_bank = _build_graph_bank(
            node_features=node_features,
            connectivity_tensor=connectivity,
            channel_names=channel_names,
            graph_cfg=graph_cfg,
        )
        data.adj_bank = torch.as_tensor(adj_bank, dtype=torch.float32)
        data.topology_bank = torch.as_tensor(topo_bank, dtype=torch.float32)

    return data


class GraphSubjectBagDataset(Dataset):
    def __init__(
        self,
        bags: Sequence[dict[str, Any]],
        *,
        max_instances: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        self.bags = [deepcopy(b) for b in bags]
        self.max_instances = None if max_instances is None else int(max_instances)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        bag = deepcopy(self.bags[int(idx)])
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
    connectivity_metric = _cfg_get(cfg, "connectivity", "metric", default=None)
    connectivity_band = _cfg_get(cfg, "connectivity", "band", default=None)
    windows_per_macro = _cfg_get(cfg, "macro", "windows_per_macro", default=10)
    feature_agg = _cfg_get(cfg, "macro", "feature_aggregation", default="mean")
    connectivity_agg = _cfg_get(cfg, "macro", "connectivity_aggregation", default="mean")
    connectivity_reduce = _cfg_get(cfg, "graph", "connectivity_reduce", default="mean")

    bags: list[dict[str, Any]] = []
    for sid in subject_ids:
        subj = payload[sid]
        instances = _build_groups_for_subject(
            subj,
            graph_level=graph_level,
            windows_per_macro=windows_per_macro,
            feature_agg=feature_agg,
            connectivity_agg=connectivity_agg,
            connectivity_reduce_for_graph=connectivity_reduce,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
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
    connectivity_metric = _cfg_get(cfg, "connectivity", "metric", default=None)
    connectivity_band = _cfg_get(cfg, "connectivity", "band", default=None)
    windows_per_macro = _cfg_get(cfg, "macro", "windows_per_macro", default=10)
    feature_agg = _cfg_get(cfg, "macro", "feature_aggregation", default="mean")
    connectivity_agg = _cfg_get(cfg, "macro", "connectivity_aggregation", default="mean")
    connectivity_reduce = _cfg_get(cfg, "graph", "connectivity_reduce", default="mean")
    graph_cfg = dict(_cfg_get(cfg, "graph", default={}) or {})

    bags: list[dict[str, Any]] = []
    for sid in subject_ids:
        subj = payload[sid]
        channel_names = list(subj["channel_names"])
        instances = _build_groups_for_subject(
            subj,
            graph_level=graph_level,
            windows_per_macro=windows_per_macro,
            feature_agg=feature_agg,
            connectivity_agg=connectivity_agg,
            connectivity_reduce_for_graph=connectivity_reduce,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
        )
        graphs = [
            _instance_to_pyg_data(
                node_features=np.asarray(inst["node_features"], dtype=np.float32),
                connectivity=None if inst.get("connectivity") is None else np.asarray(inst["connectivity"], dtype=np.float32),
                channel_names=channel_names,
                label=int(subj["label"]),
                subject_id=sid,
                instance_id=str(inst["instance_id"]),
                graph_cfg=graph_cfg,
                require_graph_bank=require_graph_bank,
            )
            for inst in instances
        ]
        bags.append({"subject_id": sid, "label": int(subj["label"]), "graphs": graphs})
    return bags


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------


def _build_dense_model(cfg: Mapping[str, Any], sample_bag: Mapping[str, Any], num_classes: int) -> nn.Module:
    family = _normalize_name(_cfg_get(cfg, "experiment", "model_family", default="node_only"))
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

    if family in {"node_only", "node_only_mlp", "node_only_dense"}:
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



def _build_gnn_model(cfg: Mapping[str, Any], sample_bag: Mapping[str, Any], num_classes: int) -> nn.Module:
    family = _normalize_name(_cfg_get(cfg, "experiment", "model_family", default="simple_fixed_graph_gnn"))
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
# Forward hooks for bag-level subject training
# -----------------------------------------------------------------------------


def _build_dense_bag_forward(aggregation: str):
    aggregation = _normalize_name(aggregation)

    def _forward(model: nn.Module, batch: Mapping[str, Any], trainer: Trainer) -> dict[str, Any]:
        x = batch["node_features"]  # [B, K, N, F]
        mask = batch["mask"]        # [B, K]
        conn = batch.get("connectivity", None)
        labels = batch["labels"]
        subject_ids = batch["subject_ids"]

        bsz, max_k = int(x.shape[0]), int(x.shape[1])
        flat_mask = mask.reshape(-1)
        x_flat = x.reshape(bsz * max_k, *x.shape[2:])[flat_mask]
        conn_flat = None
        if conn is not None:
            conn_flat = conn.reshape(bsz * max_k, *conn.shape[2:])[flat_mask]

        out = model(
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
            method=aggregation,
            classifier=model.classifier,
        )
        return {
            "logits": agg["subject_logits"],
            "probs": agg["subject_prob"],
            "preds": agg["subject_pred"],
            "targets": labels.to(device=grouped_emb.device),
            "subject_ids": list(subject_ids),
            "attention_weights": agg.get("attention_weights"),
        }

    return _forward



def _build_graph_bag_forward(aggregation: str):
    aggregation = _normalize_name(aggregation)

    def _forward(model: nn.Module, batch: Mapping[str, Any], trainer: Trainer) -> dict[str, Any]:
        pyg_batch = batch["pyg_batch"]
        bag_indices = batch["bag_indices"].to(trainer.device)
        labels = batch["labels"].to(trainer.device)
        subject_ids = list(batch["subject_ids"])

        out = model(pyg_batch, return_dict=True)
        emb = out.embedding
        logits = out.logits

        agg = aggregate_subject_predictions(
            instance_embeddings=emb,
            instance_logits=logits,
            bag_indices=bag_indices,
            method=aggregation,
            classifier=model.classifier,
        )
        return {
            "logits": agg["subject_logits"],
            "probs": agg["subject_prob"],
            "preds": agg["subject_pred"],
            "targets": labels,
            "subject_ids": subject_ids,
            "attention_weights": agg.get("attention_weights"),
        }

    return _forward


# -----------------------------------------------------------------------------
# Prediction table helpers
# -----------------------------------------------------------------------------


def _prediction_result_to_df(
    pred: Mapping[str, Any],
    *,
    split: str,
    fold: int,
    split_seed: int,
    source_level: str,
) -> pd.DataFrame:
    y_true = np.asarray(pred["y_true"], dtype=np.int64).reshape(-1)
    probs = pred.get("probs", None)
    logits = pred.get("logits", None)
    y_pred = pred.get("y_pred", None)
    subject_ids = pred.get("subject_ids", None)

    n = int(len(y_true))
    if subject_ids is None:
        subject_ids = [f"{split}_subject_{i}" for i in range(n)]
    if y_pred is None:
        if probs is not None:
            y_pred = np.asarray(probs).argmax(axis=1)
        elif logits is not None:
            y_pred = np.asarray(logits).argmax(axis=1)
        else:
            raise ValueError("Prediction result is missing y_pred/probs/logits.")

    df = pd.DataFrame(
        {
            "subject_id": [str(x) for x in subject_ids],
            "true_label": y_true.astype(np.int64),
            "pred_label": np.asarray(y_pred, dtype=np.int64).reshape(-1),
            "split": split,
            "fold": int(fold),
            "split_seed": int(split_seed),
            "source_level": str(source_level),
        }
    )
    if probs is not None:
        probs_np = np.asarray(probs)
        for c in range(probs_np.shape[1]):
            df[f"prob_{c}"] = probs_np[:, c]
    elif logits is not None:
        logits_np = np.asarray(logits)
        for c in range(logits_np.shape[1]):
            df[f"logit_{c}"] = logits_np[:, c]
    return df


# -----------------------------------------------------------------------------
# One split run
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
    family = _normalize_name(_cfg_get(cfg, "experiment", "model_family", default="node_only"))
    aggregation = _normalize_name(_cfg_get(cfg, "experiment", "aggregation", default="none"))
    graph_level = _cfg_get(cfg, "experiment", "graph_level", default="segment")

    train_cfg = dict(_cfg_get(cfg, "train", default={}) or {})
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 0))
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = train_cfg.get("early_stopping_patience", 15)
    monitor = str(train_cfg.get("monitor", "balanced_accuracy"))
    loss_name = str(train_cfg.get("loss_name", "cross_entropy"))
    train_bag_limit = train_cfg.get("max_instances_per_subject_train", None)
    eval_bag_limit = train_cfg.get("max_instances_per_subject_eval", None)

    fold_dir = Path(output_dir) / f"fold_{int(fold):02d}"
    ensure_dir(fold_dir)

    is_graph = family in {
        "simple_fixed_graph_gnn",
        "fixed_graph_gnn",
        "fused_graph_bank_gnn",
        "dual_branch_graph_model",
        "gnn",
    }
    require_graph_bank = family == "fused_graph_bank_gnn" or (
        family == "dual_branch_graph_model" and bool(_cfg_get(cfg, "gnn", "use_graph_bank", default=False))
    )

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

        num_classes = len(sorted({int(payload[sid]["label"]) for sid in payload.keys()}))
        model = _build_gnn_model(cfg, train_bags[0], num_classes=num_classes).to(device)
        forward_fn = _build_graph_bag_forward(aggregation)

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

        num_classes = len(sorted({int(payload[sid]["label"]) for sid in payload.keys()}))
        model = _build_dense_model(cfg, train_bags[0], num_classes=num_classes).to(device)
        forward_fn = _build_dense_bag_forward(aggregation)

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
            source_level=str(graph_level),
        )
        save_predictions_csv(pred_df, fold_dir / f"{split_name}_subject_predictions.csv")

        split_summary = summarize_fold_results(pred_df)
        split_summary["fold"] = int(fold)
        split_summary["split_seed"] = int(split_seed)
        split_summary["train_seed"] = int(train_seed)
        fold_rows.append(split_summary)
        fold_summary_out[split_name] = split_summary

    with open(fold_dir / "fit_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(fit_out), f, indent=2)
    with open(fold_dir / "fold_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(fold_summary_out), f, indent=2)

    return fold_rows, fold_summary_out


# -----------------------------------------------------------------------------
# Top-level experiment driver
# -----------------------------------------------------------------------------


def run_experiment(
    *,
    config_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    split_seed: int,
    train_seed: int,
) -> None:
    cfg = _read_config(config_path)
    dataset_name = _normalize_name(_cfg_get(cfg, "dataset", "name", default="aheap"))
    h5_path = _cfg_get(cfg, "dataset", "h5_path")
    if not h5_path:
        raise ValueError("dataset.h5_path is required.")

    ensure_dir(output_dir)
    device = torch.device(get_device(_cfg_get(cfg, "train", "device", default=None)))
    set_seed(int(train_seed))

    meta = {
        "config_path": str(config_path),
        "dataset_name": dataset_name,
        "h5_path": str(h5_path),
        "split_seed": int(split_seed),
        "train_seed": int(train_seed),
        "device": str(device),
    }
    with open(Path(output_dir) / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(meta), f, indent=2)
    with open(Path(output_dir) / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(cfg), f, indent=2)

    available = list_available_groups(h5_path)
    available_subject_ids = set(available["subjects"])

    if dataset_name == "aheap":
        payload_index = _load_h5_payload(cfg, available["subjects"])
        subject_ids = sorted(payload_index.keys())
        labels = [int(payload_index[sid]["label"]) for sid in subject_ids]
        folds = _make_aheap_folds(
            subject_ids=subject_ids,
            labels=labels,
            n_splits=int(_cfg_get(cfg, "evaluation", "n_splits", default=5)),
            val_ratio=float(_cfg_get(cfg, "evaluation", "val_ratio", default=0.2)),
            split_seed=int(split_seed),
        )
        needed_subjects = subject_ids

    elif dataset_name == "caueeg":
        dataset_path = _cfg_get(cfg, "dataset", "dataset_path")
        task = _cfg_get(cfg, "dataset", "task", default="dementia")
        prefix_mode = _cfg_get(cfg, "dataset", "subject_id_prefix_mode", default="auto")
        folds = _make_caueeg_fixed_split(
            dataset_path=dataset_path,
            task=task,
            available_subject_ids=available_subject_ids,
            prefix_mode=prefix_mode,
        )
        needed_subjects = sorted({sid for spec in folds for key in ("train_ids", "val_ids", "test_ids") for sid in spec[key]})

    else:
        raise ValueError(f"Unsupported dataset.name={dataset_name!r}")

    payload = _load_h5_payload(cfg, needed_subjects)

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

    with open(Path(output_dir) / "all_fold_summaries.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(all_fold_summaries), f, indent=2)

    print(f"[done] Results saved to: {output_dir}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H5-first EEG ladder runner")
    p.add_argument("--config", required=True, help="Path to experiment config (.json/.yaml)")
    p.add_argument("--output-dir", required=True, help="Directory for this run")
    p.add_argument("--split-seed", type=int, default=101, help="Used for AHEAP CV splits. Ignored for CAUEEG official split.")
    p.add_argument("--train-seed", type=int, default=11, help="Training seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        config_path=args.config,
        output_dir=args.output_dir,
        split_seed=args.split_seed,
        train_seed=args.train_seed,
    )
