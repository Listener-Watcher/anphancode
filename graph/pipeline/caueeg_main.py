from __future__ import annotations

"""
caueeg_main.py

H5-based experiment runner for CAUEEG.

Main goals
----------
- reuse precomputed H5 features/connectivity instead of recomputing them
- support three graph-construction levels:
    1) segment graph   : one short EEG window = one graph / instance
    2) macro graph     : one larger block = one graph / instance
    3) subject graph   : one whole subject = one graph / instance
- treat the following as separate design axes:
    * model family
    * subject aggregation strategy
    * topology strategy
    * edge-weight strategy
    * connectivity source / band usage
    * graph readout

This file is intentionally written as a practical experiment harness, not as a
single monolithic script. The important entry points are:

- run_caueeg_experiment(...)
- run_caueeg_ladder(...)
- build_default_caueeg_ladder(...)

Expected project modules
------------------------
This file is designed to sit next to the current project modules, especially:
- dense.py
- gnn.py
- models_mil.py
- metrics.py
- evaluate.py
- visualize.py
- utils.py
- graph_construction.py

It also expects CAUEEG task JSON files (for official train/val/test splits) and
an H5 file that stores precomputed window-level features/connectivity.
"""

import copy
import sys
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Literal, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
try:
    from pipeline.utils import ensure_dir, get_device, make_run_name, set_seed
except ImportError:  # pragma: no cover
    from utils import ensure_dir, get_device, make_run_name, set_seed

try:
    from pipeline.dense import (
        ConnectivityOnlyCNN,
        ConnectivityOnlyMLP,
        DualBranchDenseModel,
        NodeOnlyMLP,
    )
except ImportError:  # pragma: no cover
    from dense import (
        ConnectivityOnlyCNN,
        ConnectivityOnlyMLP,
        DualBranchDenseModel,
        NodeOnlyMLP,
    )

try:
    from pipeline.gnn import DualBranchGraphModel, FusedGraphBankGNN, SimpleFixedGraphGNN, GraphReadout
except ImportError:  # pragma: no cover
    from gnn import DualBranchGraphModel, FusedGraphBankGNN, SimpleFixedGraphGNN, GraphReadout

try:
    from pipeline.models_mil import (
        GatedAttentionMILPool,
        MeanMILPool,
        SubjectFusionHead,
        aggregate_subject_predictions,
    )
except ImportError:  # pragma: no cover
    from models_mil import (
        GatedAttentionMILPool,
        MeanMILPool,
        SubjectFusionHead,
        aggregate_subject_predictions,
    )

try:
    from pipeline.metrics import summarize_classification_metrics
except ImportError:  # pragma: no cover
    from metrics import summarize_classification_metrics

try:
    from pipeline.evaluate import aggregate_instance_predictions_to_subject
except ImportError:  # pragma: no cover
    from evaluate import aggregate_instance_predictions_to_subject

try:
    from pipeline.visualize import plot_confusion_matrix, plot_training_curves
except ImportError:  # pragma: no cover
    from visualize import plot_confusion_matrix, plot_training_curves

GRAPH_IMPORT_ERROR: Exception | None = None
try:
    from graph_construction import (
        DEFAULT_BANDS,
        GraphSample,
        build_connectivity_topology,
        build_feature_induced_topology,
        build_fixed_topology,
        build_graph_bank,
        fuse_graph_bank,
        to_pyg_data,
    )
except Exception as exc:  # pragma: no cover
    GRAPH_IMPORT_ERROR = exc
    DEFAULT_BANDS = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    GraphSample = Any  # type: ignore[assignment]
# native reusable readout block



ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
# ----- legacy encoders from old MIL stack -----
from mil_full_std import (
    RawNodeMLPEncoder,
    RawNodeEdgeMLPEncoder,
    RawNodeAdjCNNEncoder,
    RawNodeMultiBandCNNEncoder,
    MultiBandCNNEncoder,
)

from mil_utils import (
    GNNEncoder,
    GraphSAGEEncoder,
    GCNIIEncoder,
    H2GCNLikeEncoder,
)

# optional: only if these exist in your local version
try:
    from mil_full_std import GNNEncoder_GAT, HybridGNNEncoder
except Exception:
    GNNEncoder_GAT = None
    HybridGNNEncoder = None



# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
GraphLevel = Literal["segment", "macro", "subject"]
ModelFamily = Literal[
    "node_only",
    "connectivity_only",
    "dense_dual_branch",
    "fixed_graph_gnn",
    "fused_graph_bank_gnn",
    "dual_branch_graph",
    "legacy_encoder",
]
AggregationStrategy = Literal["none", "mean_mil", "gated_attention_mil", "subject_fusion"]
TopologyStrategy = Literal["fixed", "connectivity", "feature_induced", "fused_bank", "residual_learning"]
EdgeWeightStrategy = Literal["binary", "connectivity", "normalized", "fused"]
ConnectivityEncoderType = Literal["mlp", "cnn"]
ReductionMode = Literal["mean", "median", "std", "max", "min"]


@dataclass(slots=True)
class LevelConfig:
    graph_level: GraphLevel = "segment"
    macro_duration_sec: float = 300.0
    feature_reduce: ReductionMode = "mean"
    connectivity_reduce: ReductionMode = "mean"


@dataclass(slots=True)
class TopologyConfig:
    strategy: TopologyStrategy = "connectivity"
    fixed_edge_pairs: Optional[list[tuple[int, int]]] = None
    fixed_adjacency: Optional[np.ndarray] = None
    topology_metric: Optional[str] = "coherence"
    topology_band: Optional[int | str] = "alpha"
    similarity: str = "cosine"
    topology_kwargs: dict[str, Any] = field(default_factory=lambda: {"mode": "topk", "topk": 4})
    graph_bank_specs: Optional[list[dict[str, Any]]] = None
    fuse_method: str = "mean"
    fuse_topology_rule: str = "union"
    fuse_vote_threshold: float = 0.5
    primary_candidate: int | str = 0


@dataclass(slots=True)
class EdgeWeightConfig:
    strategy: EdgeWeightStrategy = "connectivity"
    edge_metric: Optional[str] = "coherence"
    edge_band: Optional[int | str] = "alpha"
    normalize_mode: str = "none"  # none | absmax | minmax | row_l1
    fused_sources: tuple[tuple[str, Optional[int | str]], ...] = ()
    fused_method: str = "mean"


@dataclass(slots=True)
class ConnectivityTensorConfig:
    metrics: tuple[str, ...] = ("coherence",)
    bands: Optional[tuple[int | str, ...]] = ("alpha",)


@dataclass(slots=True)
class ModelConfig:
    family: ModelFamily = "node_only"
    connectivity_encoder_type: ConnectivityEncoderType = "cnn"
    backbone: str = "gcn"
    hidden_dim: int = 64
    emb_dim: int = 128
    dropout: float = 0.2
    num_layers: int = 2
    gat_heads: int = 4
    use_edge_weight: bool = True
    use_batchnorm: bool = True
    graph_readout: str = "mean"
    fusion_mode: str = "concat"
    graph_bank_fusion_mode: str = "summary_gated"
    encoder_type: str | None = None          # legacy encoder name
    encoder_source: str = "native"           # "native" or "legacy"
    legacy_graph_pool: str = "mean"          # for old GNN-like encoders
    legacy_node_pool: str = "mean"           # for old node-only encoders
    legacy_num_bands: int = 5
    legacy_use_multiband: bool = False
    node_pooling_type: str = "none"      # none | topk | sagpool
    node_pool_ratio: float = 0.8
    legacy_graph_readout: str = "mean"   # for rewritten legacy wrappers
    legacy_align_native_readout: bool = False


@dataclass(slots=True)
class AggregationConfig:
    strategy: AggregationStrategy = "none"
    posthoc_eval_vote: str = "soft_vote"
    attn_dim: int = 128
    train_max_instances_per_subject: Optional[int] = None
    eval_max_instances_per_subject: Optional[int] = None


@dataclass(slots=True)
class TrainConfig:
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 15
    monitor: str = "balanced_accuracy"  # loss | accuracy | balanced_accuracy | macro_f1
    monitor_mode: str = "max"          # min | max
    seed: int = 42
    num_workers: int = 0


@dataclass(slots=True)
class CAUEEGExperimentSpec:
    name: str
    task: str = "dementia"
    dataset_path: str = ""
    h5_path: str = ""
    feature_families: tuple[str, ...] = ("relative_band_power", "hjorth", "statistical")
    connectivity_metrics_to_load: tuple[str, ...] = ("coherence",)
    level: LevelConfig = field(default_factory=LevelConfig)
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    edge_weights: EdgeWeightConfig = field(default_factory=EdgeWeightConfig)
    connectivity_tensor: ConnectivityTensorConfig = field(default_factory=ConnectivityTensorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output_root: str = "./results_caueeg"
    class_names: Optional[tuple[str, ...]] = None


@dataclass(slots=True)
class H5SubjectEntry:
    subject_id: str
    label: int
    channel_names: list[str]
    segment_id: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray
    features: dict[str, np.ndarray]
    connectivity: dict[str, np.ndarray]
    connectivity_band_names: dict[str, Optional[list[str]]]


@dataclass(slots=True)
class PreparedInstance:
    subject_id: str
    label: int
    level: str
    instance_id: str
    node_features: np.ndarray
    connectivity_tensor: np.ndarray
    graph_sample: Any | None = None
    pyg_data: Data | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# CAUEEG task split helpers
# ---------------------------------------------------------------------
def _read_json(path: str | os.PathLike) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_caueeg_task_splits(dataset_path: str, task: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Load official CAUEEG task splits directly from task JSON.

    This is deliberately lightweight because we only need split membership and
    class labels; signal loading is not required when experiments use H5.
    """
    task = str(task).lower()
    task_json = os.path.join(dataset_path, f"{task}.json")
    task_dict = _read_json(task_json)

    config = {k: v for k, v in task_dict.items() if k not in {"train_split", "validation_split", "test_split"}}
    train_rows = list(task_dict["train_split"])
    val_rows = list(task_dict["validation_split"])
    test_rows = list(task_dict["test_split"])
    return config, train_rows, val_rows, test_rows


def _serial_from_split_row(row: Mapping[str, Any]) -> str:
    for key in ("serial", "subject_id", "id"):
        if key in row:
            return str(row[key])
    raise KeyError(f"Could not find serial-like key in row keys={list(row.keys())}")


def _label_from_split_row(row: Mapping[str, Any]) -> int:
    for key in ("class_label", "label", "class_id", "target"):
        if key in row:
            return int(row[key])
    raise KeyError(f"Could not find class label key in row keys={list(row.keys())}")


def list_h5_subject_ids(h5_path: str) -> list[str]:
    with h5py.File(h5_path, "r") as h5f:
        return sorted(list(h5f["subjects"].keys()))


def resolve_h5_subject_ids_for_split(
    h5_path: str,
    split_rows: Sequence[Mapping[str, Any]],
    split_name: str,
) -> list[tuple[str, int, str]]:
    """
    Resolve CAUEEG split serials against subject IDs stored in the H5.

    Returns
    -------
    list of tuples
        [(h5_subject_id, label, raw_serial), ...]

    The helper supports both of these conventions:
    - subject_id == serial
    - subject_id == f"{split}_{serial}"  (used by older CAUEEG H5 builders)
    """
    available = set(list_h5_subject_ids(h5_path))
    resolved: list[tuple[str, int, str]] = []

    for row in split_rows:
        raw_serial = _serial_from_split_row(row)
        if raw_serial in {"00587", "00781", "01301"}:
            continue
        label = _label_from_split_row(row)
        candidates = [
            raw_serial,
            f"{split_name}_{raw_serial}",
            f"{split_name.lower()}_{raw_serial}",
            f"{split_name.upper()}_{raw_serial}",
        ]

        chosen = None
        for cand in candidates:
            if cand in available:
                chosen = cand
                break

        if chosen is None:
            raise KeyError(
                f"Could not resolve serial={raw_serial!r} for split={split_name!r} in H5 {h5_path}. "
                f"Tried {candidates[:2]}..."
            )
        resolved.append((chosen, label, raw_serial))

    return resolved


# ---------------------------------------------------------------------
# H5 payload helpers
# ---------------------------------------------------------------------
def load_h5_entries(
    h5_path: str,
    subject_ids: Sequence[str],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
) -> dict[str, H5SubjectEntry]:
    """
    Load only the needed precomputed content from the CAUEEG H5 file.
    """
    entries: dict[str, H5SubjectEntry] = {}
    bad_ids = {"train_00587", "train_00781", "train_01301"}
    with h5py.File(h5_path, "r") as h5f:
        for sid in subject_ids:
            # print(sid)
            if sid in bad_ids:
                print(f"Skip {sid}")
                continue
            grp = h5f[f"subjects/{sid}"]
            entry_features: dict[str, np.ndarray] = {}
            entry_connectivity: dict[str, np.ndarray] = {}
            entry_band_names: dict[str, Optional[list[str]]] = {}

            for fam in feature_families:
                entry_features[fam] = np.asarray(grp[f"windows/features/{fam}"][:], dtype=np.float32)
                # print("Loaded feature", fam)

            for metric in connectivity_metrics:
                # print("Loaded metric", metric)

                ds = grp[f"windows/connectivity/{metric}"]
                entry_connectivity[metric] = np.asarray(ds[:], dtype=np.float32)
                band_names = ds.attrs.get("band_names", None)
                entry_band_names[metric] = _normalize_band_names(band_names)
                # if band_names is None:
                #     entry_band_names[metric] = None
                # else:
                #     out_names: list[str] = []
                #     for x in band_names:
                #         if isinstance(x, bytes):
                #             out_names.append(x.decode("utf-8"))
                #         else:
                #             out_names.append(str(x))
                #     entry_band_names[metric] = out_names

            meta = grp["metadata"]
            ch_names = [
                x.decode("utf-8") if isinstance(x, bytes) else str(x)
                for x in grp["metadata/channel_names"][:]
            ]

            entries[sid] = H5SubjectEntry(
                subject_id=sid,
                label=int(meta.attrs["label"]),
                channel_names=ch_names,
                segment_id=np.asarray(grp["windows/raw/segment_id"][:], dtype=np.int64),
                start_sample=np.asarray(grp["windows/raw/start_sample"][:], dtype=np.int64),
                end_sample=np.asarray(grp["windows/raw/end_sample"][:], dtype=np.int64),
                features=entry_features,
                connectivity=entry_connectivity,
                connectivity_band_names=entry_band_names,
            )
    return entries
def _normalize_band_names(raw):
    if raw is None:
        return None

    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        else:
            out = []
            for x in raw.tolist():
                if isinstance(x, (bytes, np.bytes_)):
                    out.append(x.decode("utf-8"))
                else:
                    out.append(str(x))
            return out

    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")

    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        return [s]

    if isinstance(raw, (list, tuple)):
        out = []
        for x in raw:
            if isinstance(x, (bytes, np.bytes_)):
                out.append(x.decode("utf-8"))
            else:
                out.append(str(x))
        return out

    return [str(raw)]

# ---------------------------------------------------------------------
# Generic array helpers
# ---------------------------------------------------------------------
def reduce_array(x: np.ndarray, how: ReductionMode, axis: int = 0) -> np.ndarray:
    how = str(how).lower()
    if how == "mean":
        return np.mean(x, axis=axis)
    if how == "median":
        return np.median(x, axis=axis)
    if how == "std":
        return np.std(x, axis=axis)
    if how == "max":
        return np.max(x, axis=axis)
    if how == "min":
        return np.min(x, axis=axis)
    raise ValueError(f"Unsupported reduction {how!r}")


def zscore_node_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def normalize_edge_weight_matrix(w: np.ndarray, mode: str = "none", eps: float = 1e-8) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32)
    mode = str(mode).lower()

    if mode == "none":
        return w

    if mode == "absmax":
        denom = float(np.max(np.abs(w)))
        if denom < eps:
            return w.copy()
        return (w / denom).astype(np.float32)

    if mode == "minmax":
        w_min = float(np.min(w))
        w_max = float(np.max(w))
        if abs(w_max - w_min) < eps:
            return np.zeros_like(w, dtype=np.float32)
        return ((w - w_min) / (w_max - w_min)).astype(np.float32)

    if mode == "row_l1":
        denom = np.sum(np.abs(w), axis=1, keepdims=True)
        denom = np.where(denom < eps, 1.0, denom)
        return (w / denom).astype(np.float32)

    raise ValueError(f"Unsupported normalize_mode={mode!r}")


def _resolve_single_band_index(arr: np.ndarray, band_names: Optional[Sequence[str]], band: int | str) -> tuple[int, str]:
    if isinstance(band, (int, np.integer)):
        idx = int(band)
        if idx < 0 or idx >= arr.shape[0]:
            raise IndexError(f"Band index {idx} is out of range for array with shape {arr.shape}")
        if band_names is None:
            return idx, f"band_{idx}"
        return idx, str(band_names[idx])

    band_str = str(band)
    if band_names is not None:
        names = [str(x) for x in band_names]
        if band_str not in names:
            raise KeyError(f"Band {band_str!r} not in available bands {names}")
        return names.index(band_str), band_str

    default_names = list(DEFAULT_BANDS.keys())
    if arr.shape[0] == len(default_names) and band_str in default_names:
        return default_names.index(band_str), band_str

    raise KeyError(f"Band {band_str!r} requested but band names are unavailable")


def select_band_tensor(
    values: np.ndarray,
    band_names: Optional[Sequence[str]],
    bands: Optional[Sequence[int | str] | int | str],
) -> np.ndarray:
    """
    Select a connectivity tensor as [C, N, N].

    Input values may be [N, N] or [B, N, N].
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected [N,N] or [B,N,N], got {arr.shape}")

    if bands is None:
        return arr

    if isinstance(bands, (int, str, np.integer)):
        idx, _ = _resolve_single_band_index(arr, band_names, bands)
        return arr[idx: idx + 1]

    out = []
    for band in bands:
        idx, _ = _resolve_single_band_index(arr, band_names, band)
        out.append(arr[idx])
    return np.stack(out, axis=0).astype(np.float32)


def stack_connectivity_channels(
    connectivity_sources: Mapping[str, np.ndarray],
    band_names_map: Mapping[str, Optional[Sequence[str]]],
    cfg: ConnectivityTensorConfig,
) -> np.ndarray:
    """
    Build dense connectivity tensor for dense models.

    Output shape: [C, N, N]
    where C may correspond to:
    - one selected band from one metric
    - many bands from one metric
    - stacked bands across many metrics
    """
    channels: list[np.ndarray] = []
    for metric in cfg.metrics:
        if metric not in connectivity_sources:
            raise KeyError(f"Metric {metric!r} not found in connectivity sources")
        values = connectivity_sources[metric]
        selected = select_band_tensor(values, band_names_map.get(metric, None), cfg.bands)
        channels.append(selected)

    out = np.concatenate(channels, axis=0)
    return out.astype(np.float32)


def aggregate_feature_families(
    entry: H5SubjectEntry,
    window_indices: np.ndarray,
    feature_families: Sequence[str],
    reduce_mode: ReductionMode,
) -> np.ndarray:
    feat_parts: list[np.ndarray] = []
    for fam in feature_families:
        x = np.asarray(entry.features[fam], dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Feature family {fam!r} expected [W,N,F], got {x.shape}")
        x_sel = x[window_indices]
        if len(window_indices) == 1:
            feat_parts.append(x_sel[0])
        else:
            feat_parts.append(reduce_array(x_sel, reduce_mode, axis=0).astype(np.float32))
    return np.concatenate(feat_parts, axis=-1).astype(np.float32)


def aggregate_connectivity_sources(
    entry: H5SubjectEntry,
    window_indices: np.ndarray,
    connectivity_metrics: Sequence[str],
    reduce_mode: ReductionMode,
) -> tuple[dict[str, np.ndarray], dict[str, Optional[list[str]]]]:
    sources: dict[str, np.ndarray] = {}
    names: dict[str, Optional[list[str]]] = {}
    for metric in connectivity_metrics:
        arr = np.asarray(entry.connectivity[metric], dtype=np.float32)
        arr_sel = arr[window_indices]
        if len(window_indices) == 1:
            sources[metric] = arr_sel[0]
        else:
            sources[metric] = reduce_array(arr_sel, reduce_mode, axis=0).astype(np.float32)
        names[metric] = entry.connectivity_band_names.get(metric, None)
    return sources, names


def build_macro_groups(
    start_sample: np.ndarray,
    *,
    sfreq: float,
    macro_duration_sec: float,
) -> dict[int, np.ndarray]:
    block_len = int(round(float(sfreq) * float(macro_duration_sec)))
    if block_len < 1:
        raise ValueError("macro_duration_sec leads to block length < 1")

    macro_ids = np.floor_divide(np.asarray(start_sample, dtype=np.int64), block_len).astype(np.int64)
    out: dict[int, list[int]] = defaultdict(list)
    for idx, mid in enumerate(macro_ids.tolist()):
        out[int(mid)].append(int(idx))
    return {mid: np.asarray(idxs, dtype=np.int64) for mid, idxs in out.items()}


# ---------------------------------------------------------------------
# Graph assembly from precomputed H5 content
# ---------------------------------------------------------------------
def _require_graph_helpers() -> None:
    if GRAPH_IMPORT_ERROR is not None:
        raise ImportError(
            "graph_construction.py (or one of its dependencies) could not be imported. "
            "Graph-model experiment families require the graph construction helpers."
        ) from GRAPH_IMPORT_ERROR


def resolve_connectivity_matrix(
    connectivity_sources: Mapping[str, np.ndarray],
    band_names_map: Mapping[str, Optional[Sequence[str]]],
    metric: str,
    band: Optional[int | str],
) -> tuple[np.ndarray, Optional[str]]:
    if metric not in connectivity_sources:
        raise KeyError(f"Metric {metric!r} not found in connectivity sources: {list(connectivity_sources.keys())}")
    arr = np.asarray(connectivity_sources[metric], dtype=np.float32)

    if arr.ndim == 2:
        return arr.astype(np.float32), None
    if arr.ndim != 3:
        raise ValueError(f"Expected [N,N] or [B,N,N], got {arr.shape}")

    if band is None:
        return np.mean(arr, axis=0).astype(np.float32), "mean_all_bands"

    idx, band_name = _resolve_single_band_index(arr, band_names_map.get(metric, None), band)
    return arr[idx].astype(np.float32), band_name


def fuse_weight_sources(
    connectivity_sources: Mapping[str, np.ndarray],
    band_names_map: Mapping[str, Optional[Sequence[str]]],
    fused_sources: Sequence[tuple[str, Optional[int | str]]],
    method: str = "mean",
) -> np.ndarray:
    mats: list[np.ndarray] = []
    for metric, band in fused_sources:
        mat, _ = resolve_connectivity_matrix(connectivity_sources, band_names_map, metric, band)
        mats.append(np.asarray(mat, dtype=np.float32))
    if len(mats) == 0:
        raise ValueError("fused_sources is empty for edge_weight_strategy='fused'")
    stack = np.stack(mats, axis=0)
    return reduce_array(stack, str(method).lower(), axis=0).astype(np.float32)


def build_graph_sample_from_precomputed(
    *,
    subject_id: str,
    label: int,
    level: str,
    node_features: np.ndarray,
    connectivity_sources: Mapping[str, np.ndarray],
    band_names_map: Mapping[str, Optional[Sequence[str]]],
    topology_cfg: TopologyConfig,
    edge_cfg: EdgeWeightConfig,
    metadata: Optional[dict[str, Any]] = None,
    segment_id: Optional[int] = None,
    macro_id: Optional[int] = None,
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None,
) -> Any:
    _require_graph_helpers()

    if topology_cfg.strategy == "residual_learning":
        raise NotImplementedError(
            "Residual topology learning is intentionally left for later; "
            "the current experiment harness exposes a clear placeholder for it."
        )

    X = np.asarray(node_features, dtype=np.float32)
    n_nodes = int(X.shape[0])
    metadata = dict(metadata or {})

    graph_bank = None
    primary_adj: np.ndarray
    primary_topology: np.ndarray
    primary_edge_matrix: np.ndarray

    if topology_cfg.strategy == "fused_bank":
        candidate_specs = topology_cfg.graph_bank_specs
        if not candidate_specs:
            raise ValueError("topology.strategy='fused_bank' requires topology.graph_bank_specs")

        graph_bank = build_graph_bank(
            node_features=X,
            connectivity_sources={
                m: (np.asarray(v, dtype=np.float32), band_names_map.get(m, None))
                for m, v in connectivity_sources.items()
            },
            candidate_specs=candidate_specs,
            fixed_topology=topology_cfg.fixed_edge_pairs if topology_cfg.fixed_edge_pairs is not None else topology_cfg.fixed_adjacency,
        )
        fused = fuse_graph_bank(
            graph_bank,
            method=topology_cfg.fuse_method,
            topology_rule=topology_cfg.fuse_topology_rule,
            vote_threshold=float(topology_cfg.fuse_vote_threshold),
            select_index=int(topology_cfg.primary_candidate) if isinstance(topology_cfg.primary_candidate, int) else 0,
            output_name="fused_primary",
        )
        primary_adj = np.asarray(fused.adjacency, dtype=np.float32)
        primary_topology = np.asarray(fused.topology, dtype=np.float32)
        primary_edge_matrix = np.asarray(fused.edge_weight_matrix, dtype=np.float32)

    else:
        if topology_cfg.strategy == "fixed":
            topo_result = build_fixed_topology(
                n_nodes,
                edge_pairs=topology_cfg.fixed_edge_pairs,
                adjacency=topology_cfg.fixed_adjacency,
                complete_if_missing=True,
                undirected=True,
                include_self_loops=False,
            )
            topology_metric_name = None
            topology_band_name = None

        elif topology_cfg.strategy == "connectivity":
            if topology_cfg.topology_metric is None:
                raise ValueError("topology.topology_metric must be set when strategy='connectivity'")
            topo_matrix, topology_band_name = resolve_connectivity_matrix(
                connectivity_sources,
                band_names_map,
                topology_cfg.topology_metric,
                topology_cfg.topology_band,
            )
            topo_result = build_connectivity_topology(topo_matrix, **dict(topology_cfg.topology_kwargs))
            topology_metric_name = topology_cfg.topology_metric

        elif topology_cfg.strategy == "feature_induced":
            topo_result = build_feature_induced_topology(
                X,
                similarity=str(topology_cfg.similarity).lower(),
                **dict(topology_cfg.topology_kwargs),
            )
            topology_metric_name = None
            topology_band_name = None

        else:
            raise ValueError(f"Unsupported topology strategy {topology_cfg.strategy!r}")

        if edge_cfg.strategy == "binary":
            edge_matrix = np.asarray(topo_result.topology, dtype=np.float32)

        elif edge_cfg.strategy in {"connectivity", "normalized"}:
            if edge_cfg.edge_metric is None:
                raise ValueError("edge_weights.edge_metric must be set for connectivity/normalized edge weights")
            edge_matrix, _ = resolve_connectivity_matrix(
                connectivity_sources,
                band_names_map,
                edge_cfg.edge_metric,
                edge_cfg.edge_band,
            )
            if edge_cfg.strategy == "normalized":
                edge_matrix = normalize_edge_weight_matrix(edge_matrix, mode=edge_cfg.normalize_mode)

        elif edge_cfg.strategy == "fused":
            edge_matrix = fuse_weight_sources(
                connectivity_sources,
                band_names_map,
                fused_sources=edge_cfg.fused_sources,
                method=edge_cfg.fused_method,
            )

        else:
            raise ValueError(f"Unsupported edge-weight strategy {edge_cfg.strategy!r}")

        primary_topology = np.asarray(topo_result.topology, dtype=np.float32)
        primary_edge_matrix = np.asarray(edge_matrix, dtype=np.float32)
        primary_adj = (primary_topology * primary_edge_matrix).astype(np.float32)

        metadata.update(
            {
                "topology_metric": topology_metric_name,
                "topology_band": topology_band_name,
            }
        )

    graph = GraphSample(
        node_features=X,
        adjacency=primary_adj,
        subject_id=str(subject_id),
        label=f"class_{int(label)}",
        label_id=int(label),
        level=str(level),
        segment_id=segment_id,
        macro_id=macro_id,
        start_sample=start_sample,
        end_sample=end_sample,
        topology=primary_topology,
        edge_weight_matrix=primary_edge_matrix,
        graph_bank=graph_bank,
        metadata=metadata,
    )
    return graph


def build_instances_for_entry(entry: H5SubjectEntry, spec: CAUEEGExperimentSpec) -> list[PreparedInstance]:
    """
    Build PreparedInstance objects at the requested level using only H5 content.
    """
    level = spec.level.graph_level
    n_windows = len(entry.segment_id)
    if n_windows == 0:
        return []

    if level == "segment":
        groups = {int(seg_id): np.asarray([i], dtype=np.int64) for i, seg_id in enumerate(entry.segment_id.tolist())}
    elif level == "macro":
        groups = build_macro_groups(
            entry.start_sample,
            sfreq=200.0,
            macro_duration_sec=spec.level.macro_duration_sec,
        )
    elif level == "subject":
        groups = {0: np.arange(n_windows, dtype=np.int64)}
    else:
        raise ValueError(f"Unsupported graph level {level!r}")

    instances: list[PreparedInstance] = []

    for group_id, window_indices in groups.items():
        node_features = aggregate_feature_families(
            entry,
            window_indices,
            feature_families=spec.feature_families,
            reduce_mode=spec.level.feature_reduce,
        )
        node_features = zscore_node_features(node_features)

        connectivity_sources, band_names_map = aggregate_connectivity_sources(
            entry,
            window_indices,
            connectivity_metrics=spec.connectivity_metrics_to_load,
            reduce_mode=spec.level.connectivity_reduce,
        )

        connectivity_tensor = stack_connectivity_channels(
            connectivity_sources,
            band_names_map,
            spec.connectivity_tensor,
        )

        start_sample = int(np.min(entry.start_sample[window_indices]))
        end_sample = int(np.max(entry.end_sample[window_indices]))

        graph_sample = None
        pyg_data = None
        graph_like_families = {
            "fixed_graph_gnn",
            "fused_graph_bank_gnn",
            "dual_branch_graph",
            "legacy_encoder",
        }

        if spec.model.family in graph_like_families:
            graph_sample = build_graph_sample_from_precomputed(
                subject_id=entry.subject_id,
                label=entry.label,
                level=level,
                node_features=node_features,
                connectivity_sources=connectivity_sources,
                band_names_map=band_names_map,
                topology_cfg=spec.topology,
                edge_cfg=spec.edge_weights,
                metadata={"window_indices": window_indices.tolist()},
                segment_id=int(group_id) if level == "segment" else None,
                macro_id=int(group_id) if level == "macro" else None,
                start_sample=start_sample,
                end_sample=end_sample,
            )
            pyg_data = to_pyg_data(graph_sample)

        if level == "segment":
            instance_id = f"seg_{int(group_id)}"
        elif level == "macro":
            instance_id = f"macro_{int(group_id)}"
        else:
            instance_id = "subject"

        instances.append(
            PreparedInstance(
                subject_id=entry.subject_id,
                label=entry.label,
                level=level,
                instance_id=instance_id,
                node_features=node_features.astype(np.float32),
                connectivity_tensor=connectivity_tensor.astype(np.float32),
                graph_sample=graph_sample,
                pyg_data=pyg_data,
                metadata={
                    "window_indices": window_indices.tolist(),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                },
            )
        )

    return instances


# ---------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------
class InstanceDataset(Dataset):
    def __init__(self, instances: Sequence[PreparedInstance]):
        self.instances = list(instances)

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, idx: int) -> PreparedInstance:
        return self.instances[idx]


class SubjectBagDataset(Dataset):
    """
    Subject-grouped dataset with deterministic optional subsampling.
    """

    def __init__(
        self,
        instances: Sequence[PreparedInstance],
        *,
        train: bool,
        max_instances_per_subject: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.train = bool(train)
        self.max_instances_per_subject = None if max_instances_per_subject is None else int(max_instances_per_subject)
        self.seed = int(seed)
        self.epoch = 0

        grouped: dict[str, list[PreparedInstance]] = defaultdict(list)
        labels: dict[str, int] = {}
        for inst in instances:
            grouped[inst.subject_id].append(inst)
            labels[inst.subject_id] = int(inst.label)

        self.subject_ids = sorted(grouped.keys())
        self.subject_to_instances = {
            sid: sorted(grouped[sid], key=lambda x: (x.level, x.instance_id))
            for sid in self.subject_ids
        }
        self.subject_to_label = labels

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.subject_ids[idx]
        instances = list(self.subject_to_instances[sid])

        if self.max_instances_per_subject is not None and len(instances) > self.max_instances_per_subject:
            rng = np.random.default_rng(self.seed + 100003 * self.epoch + idx)
            chosen = np.sort(rng.choice(len(instances), size=self.max_instances_per_subject, replace=False))
            instances = [instances[int(i)] for i in chosen.tolist()]

        return {
            "subject_id": sid,
            "label": self.subject_to_label[sid],
            "instances": instances,
        }


def _stack_if_not_empty(arrs: Sequence[np.ndarray], dtype: torch.dtype = torch.float32) -> Optional[torch.Tensor]:
    if len(arrs) == 0:
        return None
    return torch.tensor(np.stack(arrs, axis=0), dtype=dtype)


def collate_instances(batch: Sequence[PreparedInstance]) -> dict[str, Any]:
    node_features = _stack_if_not_empty([x.node_features for x in batch], dtype=torch.float32)
    connectivity = _stack_if_not_empty([x.connectivity_tensor for x in batch], dtype=torch.float32)
    labels = torch.tensor([int(x.label) for x in batch], dtype=torch.long)
    subject_ids = [x.subject_id for x in batch]
    instance_ids = [x.instance_id for x in batch]

    pyg_batch = None
    pyg_list = [x.pyg_data for x in batch if x.pyg_data is not None]
    if len(pyg_list) == len(batch) and len(pyg_list) > 0:
        pyg_batch = Batch.from_data_list(pyg_list)

    dense_adj = None
    if all(x.graph_sample is not None for x in batch):
        dense_adj = torch.tensor(
            np.stack([x.graph_sample.adjacency for x in batch], axis=0),
            dtype=torch.float32,
        )

    conn_stack = None
    if connectivity is not None:
        # same tensor already used by dense models: [B, C, N, N]
        conn_stack = connectivity.clone()

    return {
        "node_features": node_features,
        "connectivity": connectivity,
        "labels": labels,
        "subject_ids": subject_ids,
        "instance_ids": instance_ids,
        "pyg_batch": pyg_batch,
        "dense_adj": dense_adj,
        "conn_stack": conn_stack,
    }


def collate_subject_bags_generic(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flat_instances: list[PreparedInstance] = []
    bag_sizes: list[int] = []
    labels: list[int] = []
    subject_ids: list[str] = []
    instance_subject_ids: list[str] = []
    instance_ids: list[str] = []

    for row in batch:
        sid = str(row["subject_id"])
        instances = list(row["instances"])
        bag_sizes.append(len(instances))
        labels.append(int(row["label"]))
        subject_ids.append(sid)
        flat_instances.extend(instances)
        instance_subject_ids.extend([sid] * len(instances))
        instance_ids.extend([inst.instance_id for inst in instances])

    flat = collate_instances(flat_instances)
    flat["bag_sizes"] = torch.tensor(bag_sizes, dtype=torch.long)
    flat["labels"] = torch.tensor(labels, dtype=torch.long)
    flat["subject_ids"] = subject_ids
    flat["instance_subject_ids"] = instance_subject_ids
    flat["instance_ids"] = instance_ids
    return flat


# ---------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------
def normalize_graph_readout_name(name: str) -> str:
    name = str(name).lower().replace("+", "_")
    mapping = {
        "mean": "mean",
        "sum": "add",
        "add": "add",
        "max": "max",
        "mean_max": "mean_max_concat",
        "meanmax": "mean_max_concat",
        "mean_add": "mean_add_concat",
        "attention": "attention",
        "attn": "attention",
        "gated_attention": "gated_attention",
    }
    if name not in mapping:
        raise ValueError(f"Unsupported graph readout name {name!r}")
    return mapping[name]

class LegacyMLPNodeWithGraphReadout(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dims=(128,64), emb_dim=128,
                 dropout=0.2, readout_type="mean"):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(), nn.Dropout(dropout),
        )
        self.readout = GraphReadout(
            input_dim=hidden_dims[-1],
            readout_type=readout_type,
            output_dim=emb_dim,
            hidden_dim=64,
            dropout=0.0,
            return_attention_weights=True,
        )
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, batch):
        pyg_batch = batch["pyg_batch"] if isinstance(batch, dict) else batch
        h = self.node_mlp(pyg_batch.x)
        graph_emb, attn = self.readout(h, pyg_batch.batch, return_attention_weights=True)
        logits = self.classifier(graph_emb)
        return {"logits": logits, "embedding": graph_emb, "attention_weights": attn}

class SubjectClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExperimentSystem(nn.Module):
    def __init__(
        self,
        *,
        base_model: nn.Module,
        model_family: ModelFamily,
        aggregation_cfg: AggregationConfig,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.model_family = model_family
        self.aggregation_cfg = aggregation_cfg
        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)

        self.subject_classifier: Optional[nn.Module] = None
        self.pool: Optional[nn.Module] = None
        self.fusion_head: Optional[nn.Module] = None

        if aggregation_cfg.strategy == "mean_mil":
            self.pool = MeanMILPool()
            self.subject_classifier = SubjectClassifierHead(self.embedding_dim, self.num_classes, dropout=dropout)
        elif aggregation_cfg.strategy == "gated_attention_mil":
            self.pool = GatedAttentionMILPool(self.embedding_dim, attn_dim=aggregation_cfg.attn_dim, dropout=dropout)
            self.subject_classifier = SubjectClassifierHead(self.embedding_dim, self.num_classes, dropout=dropout)
        elif aggregation_cfg.strategy == "subject_fusion":
            self.fusion_head = SubjectFusionHead(
                in_dim=self.embedding_dim,
                num_classes=self.num_classes,
                instance_logit_dim=self.num_classes,
                hidden_dim=max(64, self.embedding_dim),
                fusion_dim=max(64, self.embedding_dim),
                dropout=dropout,
            )

    def forward_base(self, batch: Mapping[str, Any]) -> Any:
        if self.model_family in {"node_only", "connectivity_only", "dense_dual_branch"}:
            return self.base_model(
                node_features=batch["node_features"],
                connectivity=batch["connectivity"],
                return_dict=True,
            )

        if self.model_family == "legacy_encoder":
            return self.base_model(batch)

        return self.base_model(batch["pyg_batch"], return_dict=True)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        base_out = self.forward_base(batch)
        # logits = base_out.logits
        # embedding = base_out.embedding


        # base_out = self.forward_base(batch)

        if isinstance(base_out, dict):
            logits = base_out["logits"]
            embedding = base_out["embedding"]
            base_attention = base_out.get("attention_weights", None)
        else:
            logits = base_out.logits
            embedding = base_out.embedding
            base_attention = getattr(base_out, "graph_attention_weights", None)

        if self.aggregation_cfg.strategy == "none":
            return {
                "logits": logits,
                "targets": batch["labels"],
                "subject_ids": batch.get("subject_ids", None),
                "instance_ids": batch.get("instance_ids", None),
                "instance_logits": logits,
                "instance_embedding": embedding,
                "attention_weights": base_attention,
            }

        agg = aggregate_subject_predictions(
            instance_embeddings=embedding,
            instance_logits=logits if self.aggregation_cfg.strategy == "subject_fusion" else None,
            subject_ids=batch["instance_subject_ids"],
            method=self.aggregation_cfg.strategy,
            classifier=self.subject_classifier,
            pool=self.pool,
            fusion_head=self.fusion_head,
            sort_subjects=False,
        )
        return {
            "logits": agg["subject_logits"],
            "targets": batch["labels"],
            "subject_ids": agg["subject_keys"],
            "instance_ids": batch.get("instance_ids", None),
            "instance_subject_ids": batch["instance_subject_ids"],
            "instance_logits": logits,
            "instance_embedding": embedding,
            "attention_weights": agg["attention_weights"],
        }


def build_base_model(
    spec: CAUEEGExperimentSpec,
    *,
    num_nodes: int,
    num_node_features: int,
    num_bands: int,
    num_classes: int,
) -> tuple[nn.Module, int]:
    fam = spec.model.family
    readout = normalize_graph_readout_name(spec.model.graph_readout)

    if fam == "node_only":
        model = NodeOnlyMLP(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_classes=num_classes,
            readout="flatten",
            emb_dim=spec.model.emb_dim,
            dropout=spec.model.dropout,
            use_batchnorm=spec.model.use_batchnorm,
        )
        return model, int(spec.model.emb_dim)

    if fam == "connectivity_only":
        if spec.model.connectivity_encoder_type == "cnn":
            model = ConnectivityOnlyCNN(
                num_bands=num_bands,
                num_classes=num_classes,
                emb_dim=spec.model.emb_dim,
                dropout=spec.model.dropout,
                use_batchnorm=spec.model.use_batchnorm,
            )
        else:
            model = ConnectivityOnlyMLP(
                num_nodes=num_nodes,
                num_bands=num_bands,
                num_classes=num_classes,
                emb_dim=spec.model.emb_dim,
                dropout=spec.model.dropout,
                use_batchnorm=spec.model.use_batchnorm,
            )
        return model, int(spec.model.emb_dim)

    if fam == "dense_dual_branch":
        model = DualBranchDenseModel(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_bands=num_bands,
            connectivity_encoder_type=spec.model.connectivity_encoder_type,
            node_emb_dim=spec.model.emb_dim,
            connectivity_emb_dim=spec.model.emb_dim,
            fusion_emb_dim=spec.model.emb_dim,
            dropout=spec.model.dropout,
            use_batchnorm=spec.model.use_batchnorm,
            fusion_mode=spec.model.fusion_mode,
        )
        return model, int(spec.model.emb_dim)

    if fam == "fixed_graph_gnn":
        model = SimpleFixedGraphGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            backbone=spec.model.backbone,
            hidden_dim=spec.model.hidden_dim,
            graph_emb_dim=spec.model.emb_dim,
            num_layers=spec.model.num_layers,
            dropout=spec.model.dropout,
            gat_heads=spec.model.gat_heads,
            use_edge_weight=spec.model.use_edge_weight,
            use_batchnorm=spec.model.use_batchnorm,
            readout_type=readout,
            return_attention_weights=readout in {"attention", "gated_attention"},
            node_pooling_type=spec.model.node_pooling_type,     # none | topk | sagpool
            node_pool_ratio=spec.model.node_pool_ratio,
        )
        return model, int(spec.model.emb_dim)

    if fam == "fused_graph_bank_gnn":
        num_candidates = 0
        if spec.topology.graph_bank_specs is not None:
            num_candidates = len(spec.topology.graph_bank_specs)
        if num_candidates < 1:
            raise ValueError("fused_graph_bank_gnn requires topology.graph_bank_specs")
        model = FusedGraphBankGNN(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_candidates=num_candidates,
            backbone=spec.model.backbone,
            hidden_dim=spec.model.hidden_dim,
            graph_emb_dim=spec.model.emb_dim,
            num_layers=spec.model.num_layers,
            dropout=spec.model.dropout,
            gat_heads=spec.model.gat_heads,
            use_edge_weight=spec.model.use_edge_weight,
            use_batchnorm=spec.model.use_batchnorm,
            readout_type=readout,
            fusion_mode=spec.model.graph_bank_fusion_mode,
            return_attention_weights=readout in {"attention", "gated_attention"},
            node_pooling_type=spec.model.node_pooling_type,     # none | topk | sagpool
            node_pool_ratio=spec.model.node_pool_ratio,
        )
        return model, int(spec.model.emb_dim)

    if fam == "dual_branch_graph":
        use_graph_bank = spec.topology.strategy == "fused_bank"
        num_candidates = len(spec.topology.graph_bank_specs or []) if use_graph_bank else None
        model = DualBranchGraphModel(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes,
            use_graph_bank=use_graph_bank,
            num_candidates=num_candidates,
            hidden_dim=spec.model.hidden_dim,
            graph_emb_dim=spec.model.emb_dim,
            node_emb_dim=spec.model.emb_dim,
            num_layers=spec.model.num_layers,
            graph_dropout=spec.model.dropout,
            node_dropout=spec.model.dropout,
            use_edge_weight=spec.model.use_edge_weight,
            use_batchnorm=spec.model.use_batchnorm,
            backbone=spec.model.backbone,
            graph_readout_type=readout,
            node_readout_type=readout,
            graph_bank_fusion_mode=spec.model.graph_bank_fusion_mode,
            fusion_mode=spec.model.fusion_mode,
            fusion_emb_dim=spec.model.emb_dim,
            return_attention_weights=readout in {"attention", "gated_attention"},
            node_pooling_type=spec.model.node_pooling_type,     # none | topk | sagpool
            node_pool_ratio=spec.model.node_pool_ratio,
        )
        return model, int(spec.model.emb_dim)
    if fam == "legacy_encoder":
        model = LegacyEncoderClassifier(
            num_node_features=num_node_features,
            num_classes=num_classes,
            num_nodes=num_nodes,
            encoder_type=spec.model.encoder_type,
            graph_emb_dim=spec.model.emb_dim,
            dropout=spec.model.dropout,
            gnn_hidden_dim=spec.model.hidden_dim,
            node_hidden_dims=(256, 128),
            edge_hidden_dims=(128, 64),
            branch_emb_dim=64,
            edge_mode="topology_weighted",
            graph_pool=spec.model.legacy_graph_pool,
            num_bands=spec.model.legacy_num_bands,
            graph_readout=spec.model.graph_readout,
        )
        return model, int(spec.model.emb_dim)
    raise ValueError(f"Unsupported model family {fam!r}")

class LegacyEncoderClassifier(nn.Module):
    """
    Adapter that lets caueeg_main.py use legacy encoder_type models
    from mil_full_std.py / mil_utils.py on subject-level or macro-level graphs.
    """

    def __init__(
        self,
        *,
        num_node_features: int,
        num_classes: int,
        num_nodes: int,
        encoder_type: str,
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        gnn_hidden_dim: int = 64,
        node_hidden_dims=(256, 128),
        edge_hidden_dims=(128, 64),
        branch_emb_dim: int = 64,
        edge_mode: str = "topology_weighted",
        graph_pool: str = "mean",
        num_bands: int = 5,
        graph_readout: str = "mean",   # optional extra post-encoder head later
    ):
        super().__init__()
        enc = str(encoder_type).lower()
        self.encoder_type = enc
        self.num_nodes = int(num_nodes)
        self.graph_readout = graph_readout

        if enc == "linkx":
            self.encoder = RawNodeEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                use_upper_triangle=True,
                symmetrize_adj=True,
            )

        elif enc == "mlp_node":
            self.encoder = RawNodeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                proj_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif enc == "linkx_cnn":
            self.encoder = RawNodeAdjCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif enc == "linkx_cnn5":
            self.encoder = RawNodeMultiBandCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_bands=num_bands,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif enc == "cnn5":
            self.encoder = MultiBandCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_bands=num_bands,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif enc == "gnn":
            self.encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif enc == "sage":
            self.encoder = GraphSAGEEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=2,
                dropout=dropout,
                pool=graph_pool,
                jk_mode="last",
            )

        elif enc == "gcn2":
            self.encoder = GCNIIEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=8,
                dropout=dropout,
                alpha=0.1,
                theta=0.5,
                shared_weights=True,
                pool=graph_pool,
                use_edge_weight=True,
            )

        elif enc == "h2gcn":
            self.encoder = H2GCNLikeEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=2,
                dropout=dropout,
                pool=graph_pool,
            )

        elif enc == "gat":
            if GNNEncoder_GAT is None:
                raise ImportError("GNNEncoder_GAT not available in this environment.")
            self.encoder = GNNEncoder_GAT(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                num_layers=3,
                dropout=dropout,
                heads=4,
                edge_dim=1,
                pooling=graph_pool,
            )

        elif enc == "hybrid":
            if HybridGNNEncoder is None:
                raise ImportError("HybridGNNEncoder not available in this environment.")
            self.encoder = HybridGNNEncoder(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                gat_layers=2,
                cheb_layers=2,
                dropout=dropout,
                heads=4,
                edge_dim=1,
                pooling=graph_pool,
            )

        else:
            raise ValueError(f"Unsupported legacy encoder_type={encoder_type!r}")

        self.classifier = nn.Linear(graph_emb_dim, num_classes)

    def _get_dense_adj(self, batch):
        adj = batch.get("dense_adj", None)
        if adj is None:
            raise ValueError(f"{self.encoder_type} requires batch['dense_adj']")
        return adj.float()

    def _get_conn_stack(self, batch):
        conn_stack = batch.get("conn_stack", None)
        if conn_stack is None:
            raise ValueError(f"{self.encoder_type} requires batch['conn_stack']")
        return conn_stack.float()

    def forward(self, batch):
        if isinstance(batch, dict):
            pyg_batch = batch["pyg_batch"]
        else:
            pyg_batch = batch
            batch = {"pyg_batch": pyg_batch}

        if self.encoder_type == "linkx_cnn":
            emb = self.encoder(pyg_batch, self._get_dense_adj(pyg_batch))
        elif self.encoder_type in {"linkx_cnn5", "cnn5"}:
            emb = self.encoder(pyg_batch, self._get_conn_stack(pyg_batch))
        else:
            emb = self.encoder(pyg_batch)

        logits = self.classifier(emb)
        return {
            "logits": logits,
            "embedding": emb,
        }
# ---------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------
def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and callable(getattr(v, "to")):
            try:
                out[k] = v.to(device, non_blocking=True)
            except TypeError:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


class EarlyStopper:
    def __init__(self, monitor: str, mode: str, patience: int) -> None:
        self.monitor = str(monitor)
        self.mode = str(mode).lower()
        self.patience = int(patience)
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.bad_epochs = 0

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return float(value) < float(self.best_value)
        return float(value) > float(self.best_value)

    def step(self, value: float, epoch: int) -> bool:
        if self.is_better(value):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def build_prediction_dataframe(
    *,
    y_true: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    pred: np.ndarray,
    subject_ids: Sequence[Any],
    instance_ids: Optional[Sequence[Any]],
    source_level: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "subject_id": list(subject_ids),
            "true_label": y_true.astype(int),
            "pred_label": pred.astype(int),
            "source_level": source_level,
        }
    )
    if instance_ids is not None:
        df["instance_id"] = list(instance_ids)
    for c in range(probs.shape[1]):
        df[f"prob_{c}"] = probs[:, c]
    for c in range(logits.shape[1]):
        df[f"logit_{c}"] = logits[:, c]
    return df


def collect_epoch_outputs(
    system: ExperimentSystem,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    train = optimizer is not None
    system.train(mode=train)

    total_loss = 0.0
    n_batches = 0
    y_true_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    subject_ids_all: list[Any] = []
    instance_ids_all: list[Any] = []
    attention_all: list[np.ndarray] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = system(batch)
            loss = F.cross_entropy(out["logits"], out["targets"])
            if train:
                loss.backward()
                optimizer.step()

        n_batches += 1
        total_loss += float(loss.detach().cpu().item())

        probs = torch.softmax(out["logits"], dim=-1)
        pred = torch.argmax(probs, dim=-1)

        y_true_all.append(out["targets"].detach().cpu().numpy())
        logits_all.append(out["logits"].detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())

        subject_ids = out.get("subject_ids", None)
        if subject_ids is not None:
            subject_ids_all.extend(list(subject_ids))

        instance_ids = out.get("instance_ids", None)
        if instance_ids is not None:
            instance_ids_all.extend(list(instance_ids))

        attn = out.get("attention_weights", None)
        if attn is not None and torch.is_tensor(attn):
            attention_all.append(attn.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.empty((0,), dtype=np.int64)
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, 0), dtype=np.float32)
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, 0), dtype=np.float32)
    pred = np.concatenate(pred_all, axis=0) if pred_all else np.empty((0,), dtype=np.int64)

    metrics = summarize_classification_metrics(
        y_true=y_true,
        y_pred=pred,
        probs=probs,
        logits=logits,
        num_classes=probs.shape[1] if probs.ndim == 2 and probs.shape[0] > 0 else None,
    )

    return {
        "loss": total_loss / max(n_batches, 1),
        "metrics": metrics,
        "y_true": y_true,
        "logits": logits,
        "probs": probs,
        "pred": pred,
        "subject_ids": subject_ids_all if len(subject_ids_all) > 0 else None,
        "instance_ids": instance_ids_all if len(instance_ids_all) > 0 else None,
        "attention": attention_all if len(attention_all) > 0 else None,
    }


def maybe_build_subject_table(
    epoch_out: Mapping[str, Any],
    *,
    spec: CAUEEGExperimentSpec,
) -> tuple[Optional[pd.DataFrame], dict[str, Any]]:
    """
    Build subject-level table/metrics.

    For aggregation='none' on segment/macro levels:
      - training is instance-level
      - validation/test subject metrics are computed by posthoc voting

    For aggregation != 'none' or subject-level graphs:
      - outputs are already subject-level
    """
    subject_ids = epoch_out.get("subject_ids", None)
    if subject_ids is None:
        return None, epoch_out["metrics"]

    if spec.aggregation.strategy == "none" and spec.level.graph_level in {"segment", "macro"}:
        df_instance = build_prediction_dataframe(
            y_true=np.asarray(epoch_out["y_true"]),
            logits=np.asarray(epoch_out["logits"]),
            probs=np.asarray(epoch_out["probs"]),
            pred=np.asarray(epoch_out["pred"]),
            subject_ids=subject_ids,
            instance_ids=epoch_out.get("instance_ids", None),
            source_level=spec.level.graph_level,
        )
        df_subject = aggregate_instance_predictions_to_subject(
            predictions=df_instance,
            method=spec.aggregation.posthoc_eval_vote,
            group_cols=["subject_id"],
        )
        prob_cols = sorted([c for c in df_subject.columns if c.startswith("prob_")], key=lambda x: int(x.split("_")[1]))
        y_true = df_subject["true_label"].to_numpy(dtype=np.int64)
        y_pred = df_subject["pred_label"].to_numpy(dtype=np.int64)
        probs = df_subject[prob_cols].to_numpy(dtype=np.float64)
        metrics = summarize_classification_metrics(y_true=y_true, y_pred=y_pred, probs=probs)
        return df_subject, metrics

    df_subject = build_prediction_dataframe(
        y_true=np.asarray(epoch_out["y_true"]),
        logits=np.asarray(epoch_out["logits"]),
        probs=np.asarray(epoch_out["probs"]),
        pred=np.asarray(epoch_out["pred"]),
        subject_ids=subject_ids,
        instance_ids=None,
        source_level="subject",
    )
    return df_subject, epoch_out["metrics"]


# ---------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------
def validate_spec(spec: CAUEEGExperimentSpec) -> None:
    if not spec.dataset_path:
        raise ValueError("spec.dataset_path must be set")
    if not spec.h5_path:
        raise ValueError("spec.h5_path must be set")

    if spec.model.family == "macro_mvgnn":
        if spec.level.graph_level != "macro":
            raise ValueError("macro_mvgnn requires level.graph_level='macro'.")
        if spec.aggregation.strategy == "none":
            raise ValueError("macro_mvgnn is a bag-level subject model; use a non-'none' aggregation strategy.")
        return

    if spec.level.graph_level == "subject" and spec.aggregation.strategy != "none":
        raise ValueError("Subject-level graphs already produce one graph per subject; use aggregation.strategy='none'.")

    if spec.model.family in {"fixed_graph_gnn", "fused_graph_bank_gnn", "dual_branch_graph"}:
        _require_graph_helpers()

    if spec.model.family == "fused_graph_bank_gnn" and spec.topology.strategy != "fused_bank":
        raise ValueError("fused_graph_bank_gnn expects topology.strategy='fused_bank'.")

    if spec.model.family == "dual_branch_graph" and spec.topology.strategy == "residual_learning":
        raise NotImplementedError("Residual topology learning placeholder exists, but the model is not implemented yet.")

    if spec.topology.strategy == "fused_bank" and not spec.topology.graph_bank_specs:
        raise ValueError("topology.strategy='fused_bank' requires topology.graph_bank_specs")


def infer_class_names(task: str, num_classes: int) -> tuple[str, ...]:
    task = str(task).lower()
    if task == "abnormal" and num_classes == 2:
        return ("normal", "abnormal")
    if task == "dementia" and num_classes == 3:
        return ("normal", "mci", "dementia")
    return tuple(f"class_{i}" for i in range(num_classes))


def prepare_split_instances(spec: CAUEEGExperimentSpec) -> dict[str, list[PreparedInstance]]:
    validate_spec(spec)

    _, train_rows, val_rows, test_rows = load_caueeg_task_splits(spec.dataset_path, spec.task)
    split_to_rows = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }

    split_to_resolved = {
        split: resolve_h5_subject_ids_for_split(spec.h5_path, rows, split)
        for split, rows in split_to_rows.items()
    }

    all_subject_ids = []
    for rows in split_to_resolved.values():
        all_subject_ids.extend([sid for sid, _, _ in rows])
    if ["train_00587"] in all_subject_ids:
        print("Still contain train_00587")
    if ["train_00781"] in all_subject_ids:
        print("Still contain train_00781")
    if ["train_01301"] in all_subject_ids:
        print("Still contain train_01301")



    entries = load_h5_entries(
        spec.h5_path,
        all_subject_ids,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
    )

    split_instances: dict[str, list[PreparedInstance]] = {"train": [], "val": [], "test": []}
    for split, rows in split_to_resolved.items():
        for sid, _, _ in rows:
            split_instances[split].extend(build_instances_for_entry(entries[sid], spec))
    return split_instances


def make_loaders(
    split_instances: Mapping[str, Sequence[PreparedInstance]],
    spec: CAUEEGExperimentSpec,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "num_workers": int(spec.train.num_workers),
        "pin_memory": True,
    }

    if spec.aggregation.strategy == "none":
        train_ds = InstanceDataset(split_instances["train"])
        val_ds = InstanceDataset(split_instances["val"])
        test_ds = InstanceDataset(split_instances["test"])

        train_loader = DataLoader(
            train_ds,
            batch_size=int(spec.train.batch_size),
            shuffle=True,
            collate_fn=collate_instances,
            **common,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=int(spec.train.batch_size),
            shuffle=False,
            collate_fn=collate_instances,
            **common,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=int(spec.train.batch_size),
            shuffle=False,
            collate_fn=collate_instances,
            **common,
        )
        return train_loader, val_loader, test_loader

    train_ds = SubjectBagDataset(
        split_instances["train"],
        train=True,
        max_instances_per_subject=spec.aggregation.train_max_instances_per_subject,
        seed=spec.train.seed,
    )
    val_ds = SubjectBagDataset(
        split_instances["val"],
        train=False,
        max_instances_per_subject=spec.aggregation.eval_max_instances_per_subject,
        seed=spec.train.seed,
    )
    test_ds = SubjectBagDataset(
        split_instances["test"],
        train=False,
        max_instances_per_subject=spec.aggregation.eval_max_instances_per_subject,
        seed=spec.train.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(spec.train.batch_size),
        shuffle=True,
        collate_fn=collate_subject_bags_generic,
        **common,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(spec.train.batch_size),
        shuffle=False,
        collate_fn=collate_subject_bags_generic,
        **common,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=int(spec.train.batch_size),
        shuffle=False,
        collate_fn=collate_subject_bags_generic,
        **common,
    )
    return train_loader, val_loader, test_loader


def run_caueeg_experiment(spec: CAUEEGExperimentSpec) -> dict[str, Any]:
    validate_spec(spec)
    if spec.model.family == "macro_mvgnn":
        from macro_mvgnn_new import run_macro_mvgnn_experiment
        return run_macro_mvgnn_experiment(spec)
    set_seed(spec.train.seed)

    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    split_instances = prepare_split_instances(spec)
    train_loader, val_loader, test_loader = make_loaders(split_instances, spec)

    if len(split_instances["train"]) == 0:
        raise ValueError("Training split produced zero instances.")

    sample = split_instances["train"][0]
    num_nodes = int(sample.node_features.shape[0])
    num_node_features = int(sample.node_features.shape[1])
    num_bands = int(sample.connectivity_tensor.shape[0])
    num_classes = len(sorted({int(x.label) for x in split_instances["train"] + split_instances["val"] + split_instances["test"]}))
    class_names = spec.class_names or infer_class_names(spec.task, num_classes)

    base_model, embedding_dim = build_base_model(
        spec,
        num_nodes=num_nodes,
        num_node_features=num_node_features,
        num_bands=num_bands,
        num_classes=num_classes,
    )
    system = ExperimentSystem(
        base_model=base_model,
        model_family=spec.model.family,
        aggregation_cfg=spec.aggregation,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout=spec.model.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(system.parameters(), lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    stopper = EarlyStopper(spec.train.monitor, spec.train.monitor_mode, spec.train.patience)

    run_dir = ensure_dir(
        os.path.join(
            spec.output_root,
            make_run_name(
                spec.name,
                # spec.task,
                # spec.level.graph_level,
                # spec.model.family,
                spec.aggregation.strategy,
                timestamp=True,
            ),
        )
    )

    history_train: list[dict[str, Any]] = []
    history_val: list[dict[str, Any]] = []
    best_state: Optional[dict[str, Any]] = None
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, spec.train.epochs + 1):
        if spec.aggregation.strategy != "none" and hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        train_out = collect_epoch_outputs(system, train_loader, device=device, optimizer=optimizer)
        val_out = collect_epoch_outputs(system, val_loader, device=device, optimizer=None)

        _, train_metrics_subject = maybe_build_subject_table(train_out, spec=spec)
        _, val_metrics_subject = maybe_build_subject_table(val_out, spec=spec)

        history_train.append(
            {
                "epoch": epoch,
                "loss": float(train_out["loss"]),
                "accuracy": float(train_metrics_subject["accuracy"]),
                "balanced_accuracy": float(train_metrics_subject["balanced_accuracy"]),
                "macro_f1": float(train_metrics_subject["macro_f1"]),
            }
        )
        history_val.append(
            {
                "epoch": epoch,
                "loss": float(val_out["loss"]),
                "accuracy": float(val_metrics_subject["accuracy"]),
                "balanced_accuracy": float(val_metrics_subject["balanced_accuracy"]),
                "macro_f1": float(val_metrics_subject["macro_f1"]),
            }
        )

        if spec.train.monitor == "loss":
            monitor_value = float(val_out["loss"])
        else:
            monitor_value = float(val_metrics_subject[spec.train.monitor])

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_out['loss']:.4f}, bal_acc={train_metrics_subject['balanced_accuracy']:.4f}, macro_f1={train_metrics_subject['macro_f1']:.4f} | "
            f"val loss={val_out['loss']:.4f}, bal_acc={val_metrics_subject['balanced_accuracy']:.4f}, macro_f1={val_metrics_subject['macro_f1']:.4f}"
        )

        should_stop = stopper.step(monitor_value, epoch)
        if stopper.best_epoch == epoch:
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(system.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "monitor": spec.train.monitor,
                "monitor_value": monitor_value,
                "val_loss": float(val_out["loss"]),
                "val_metrics": copy.deepcopy(val_metrics_subject),
                "spec": asdict(spec),
            }
            torch.save(best_state, best_ckpt_path)

        if should_stop:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best {spec.train.monitor}={stopper.best_value:.6f} at epoch {stopper.best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was saved during training.")

    best_loaded = torch.load(best_ckpt_path, map_location=device)
    system.load_state_dict(best_loaded["model_state_dict"])

    train_final = collect_epoch_outputs(system, train_loader, device=device, optimizer=None)
    val_final = collect_epoch_outputs(system, val_loader, device=device, optimizer=None)
    test_final = collect_epoch_outputs(system, test_loader, device=device, optimizer=None)

    train_subject_df, train_subject_metrics = maybe_build_subject_table(train_final, spec=spec)
    val_subject_df, val_subject_metrics = maybe_build_subject_table(val_final, spec=spec)
    test_subject_df, test_subject_metrics = maybe_build_subject_table(test_final, spec=spec)

    pd.DataFrame(history_train).to_csv(os.path.join(run_dir, "history_train.csv"), index=False)
    pd.DataFrame(history_val).to_csv(os.path.join(run_dir, "history_val.csv"), index=False)

    if train_subject_df is not None:
        train_subject_df.to_csv(os.path.join(run_dir, "train_subject_predictions.csv"), index=False)
    if val_subject_df is not None:
        val_subject_df.to_csv(os.path.join(run_dir, "val_subject_predictions.csv"), index=False)
    if test_subject_df is not None:
        test_subject_df.to_csv(os.path.join(run_dir, "test_subject_predictions.csv"), index=False)

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "best_epoch": best_loaded["epoch"],
                "monitor": spec.train.monitor,
                "monitor_value": best_loaded["monitor_value"],
                "train_metrics": train_subject_metrics,
                "val_metrics": val_subject_metrics,
                "test_metrics": test_subject_metrics,
                "class_names": list(class_names),
                "spec": asdict(spec),
            },
            f,
            indent=2,
        )

    try:
        plot_training_curves(
            {"train": history_train, "val": history_val},
            save_path=os.path.join(run_dir, "training_curves.png"),
            metric_keys=("loss", "balanced_accuracy", "macro_f1"),
            title=f"{spec.name} training curves",
        )
    except Exception:
        pass

    try:
        if test_subject_df is not None:
            plot_confusion_matrix(
                test_subject_df["true_label"].to_numpy(dtype=np.int64),
                test_subject_df["pred_label"].to_numpy(dtype=np.int64),
                save_path=os.path.join(run_dir, "test_confusion_matrix.png"),
                class_names=class_names,
                normalize=True,
                title=f"{spec.name} test confusion matrix",
            )
    except Exception:
        pass

    return {
        "run_dir": run_dir,
        "best_epoch": best_loaded["epoch"],
        "best_checkpoint_path": best_ckpt_path,
        "train_metrics": train_subject_metrics,
        "val_metrics": val_subject_metrics,
        "test_metrics": test_subject_metrics,
        "class_names": class_names,
        "spec": spec,
    }







def run_caueeg_ladder(specs: Sequence[CAUEEGExperimentSpec]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in specs:
        print("=" * 100)
        print(f"Running experiment: {spec.name}")
        print("=" * 100)
        out = run_caueeg_experiment(spec)
        results.append(out)
    return results


def select_bucket_winners(df, top_k_per_bucket=1):
    bucket_cols = ["graph_level", "model_family"]
    winners = (
        df.sort_values(
            by=["val_bal_acc", "val_macro_f1"], #, "val_loss"
            ascending=[False, False], # True
        )
        .groupby(bucket_cols, as_index=False, group_keys=False)
        .head(top_k_per_bucket)
        .reset_index(drop=True)
    )
    print(winners)
    return winners

if __name__ == "__main__":
#     # Example usage:
#     #   python caueeg_main.py
#     # Then edit dataset_path / h5_path to your local paths.
#     example_dataset_path = "/mnt/data/anphan/CAUEEG/caueeg-dataset"
#     # example_h5_path = "/mnt/data/anphan/CAUEEG/caueeg_master_linkx.h5"
#     example_h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
#     leaderboard = '/home/anphan/Documents/EEG_Project/CAUEEG/results_pipeline/leaderboard.csv'
#     leaderboard_df = pd.read_csv(leaderboard)

#     if os.path.exists(example_dataset_path) and os.path.exists(example_h5_path):
#         ladder = build_new_caueeg_ladder(
#             dataset_path=example_dataset_path,
#             h5_path=example_h5_path,
#             task="dementia",
#             output_root="/home/anphan/Documents/EEG_Project/CAUEEG/results_pipeline",
#         )

#         for i, spec in enumerate(ladder):
#             print(f"{i:02d} - {spec.name}")
#             run_caueeg_ladder([spec])   # wrap in [] because run_caueeg_ladder expects a list of specs
#     else:
#         print("Edit example_dataset_path and example_h5_path in caueeg_main.py before running.")


    spec = CAUEEGExperimentSpec(
        name="macro_mvgnn_default",
        task="dementia",
        dataset_path="/mnt/data/anphan/CAUEEG/caueeg-dataset",
        h5_path="/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5",
        feature_families=("relative_band_power", "statistical"),
        connectivity_metrics_to_load=("coherence", "pli", "wpli"),
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=60.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        model=ModelConfig(
            family="macro_mvgnn",
            backbone="gatv2",
            hidden_dim=64,
            emb_dim=128,
            num_layers=2,
            dropout=0.2,
            gat_heads=4,
            graph_readout="gated_attention",
            graph_bank_fusion_mode="summary_gated",
        ),
        aggregation=AggregationConfig(
            strategy="gated_attention_mil",
            attn_dim=128,
        ),
        train=TrainConfig(
            batch_size=8,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=120,
            patience=30,
            monitor="macro_f1",
            monitor_mode="max",
            seed=42,
        ),
        output_root="./results_caueeg",
    )

    run_caueeg_ladder([spec])