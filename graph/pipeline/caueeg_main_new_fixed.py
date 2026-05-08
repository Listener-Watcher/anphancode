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
    from pipeline.gnn import DualBranchGraphModel, FusedGraphBankGNN, SimpleFixedGraphGNN
except ImportError:  # pragma: no cover
    from gnn import DualBranchGraphModel, FusedGraphBankGNN, SimpleFixedGraphGNN

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
    "macro_mvgnn",
]
AggregationStrategy = Literal["none", "mean_mil", "gated_attention_mil", "subject_fusion"]
TopologyStrategy = Literal["fixed", "connectivity", "feature_induced", "fused_bank", "residual_learning"]
EdgeWeightStrategy = Literal["binary", "connectivity", "normalized", "fused"]
ConnectivityEncoderType = Literal["mlp", "cnn"]
ReductionMode = Literal["mean", "median", "std", "max", "min"]


def default_fixed_edge_pairs_19() -> list[tuple[int, int]]:
    """
    Simple hand-crafted 10-20 neighbor graph for CAUEEG 19 EEG channels.

    Channel order:
    0 Fp1, 1 F3, 2 C3, 3 P3, 4 O1,
    5 Fp2, 6 F4, 7 C4, 8 P4, 9 O2,
    10 F7, 11 T3, 12 T5, 13 F8, 14 T4,
    15 T6, 16 FZ, 17 CZ, 18 PZ
    """
    return [
        (0, 1), (0, 10), (0, 5),
        (5, 6), (5, 13),

        (10, 1), (10, 11),
        (11, 2), (11, 12),
        (12, 3), (12, 4),

        (13, 6), (13, 14),
        (14, 7), (14, 15),
        (15, 8), (15, 9),

        (1, 2), (2, 3), (3, 4),
        (6, 7), (7, 8), (8, 9),

        (1, 16), (6, 16),
        (2, 17), (7, 17),
        (3, 18), (8, 18),

        (16, 17), (17, 18),
    ]


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
        if spec.model.family in {"fixed_graph_gnn", "fused_graph_bank_gnn", "dual_branch_graph"}:
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

    return {
        "node_features": node_features,
        "connectivity": connectivity,
        "labels": labels,
        "subject_ids": subject_ids,
        "instance_ids": instance_ids,
        "pyg_batch": pyg_batch,
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

        return self.base_model(batch["pyg_batch"], return_dict=True)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        base_out = self.forward_base(batch)
        logits = base_out.logits
        embedding = base_out.embedding

        if self.aggregation_cfg.strategy == "none":
            return {
                "logits": logits,
                "targets": batch["labels"],
                "subject_ids": batch.get("subject_ids", None),
                "instance_ids": batch.get("instance_ids", None),
                "instance_logits": logits,
                "instance_embedding": embedding,
                "attention_weights": None,
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
        )
        return model, int(spec.model.emb_dim)

    raise ValueError(f"Unsupported model family {fam!r}")


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
    split_name_map = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }

    split_to_resolved = {
        split: resolve_h5_subject_ids_for_split(spec.h5_path, rows, split_name_map[split])
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


# ---------------------------------------------------------------------
# Ladder helpers
# ---------------------------------------------------------------------
# def default_graph_bank_specs(
#     *,
#     metrics: Sequence[str] = ("coherence",),
#     bands: Sequence[int | str] = ("theta", "alpha", "beta"),
# ) -> list[dict[str, Any]]:
#     specs: list[dict[str, Any]] = []
#     for metric in metrics:
#         for band in bands:
#             specs.append(
#                 {
#                     "name": f"{metric}_{band}",
#                     "topology_mode": "connectivity",
#                     "edge_weight_mode": "connectivity",
#                     "connectivity_metric": metric,
#                     "band": band,
#                     "topology_kwargs": {"mode": "topk", "topk": 4},
#                 }
#             )
#     return specs


# def build_default_caueeg_ladder(
#     *,
#     dataset_path: str,
#     h5_path: str,
#     task: str = "dementia",
#     output_root: str = "./results_caueeg",
# ) -> list[CAUEEGExperimentSpec]:
#     """
#     Curated ladder rather than a naive full Cartesian product.

#     The blocks are ordered so that simpler, lower-risk comparisons happen first.
#     """
#     base = CAUEEGExperimentSpec(
#         name="base",
#         task=task,
#         dataset_path=dataset_path,
#         h5_path=h5_path,
#         output_root=output_root,
#         feature_families=("relative_band_power", "hjorth", "statistical"),
#         connectivity_metrics_to_load=("coherence",),
#         connectivity_tensor=ConnectivityTensorConfig(metrics=("coherence",), bands=(2,)),
#         train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),
#     )

#     specs: list[CAUEEGExperimentSpec] = []

#     # --------------------------------------------------
#     # Block 1: subject-level dense baselines
#     # --------------------------------------------------
#     specs.append(
#         replace(
#             base,
#             name="subject_node_only",
#             level=LevelConfig(graph_level="subject"),
#             model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
#             aggregation=AggregationConfig(strategy="none"),
#         )
#     )
#     specs.append(
#         replace(
#             base,
#             name="subject_connectivity_cnn",
#             level=LevelConfig(graph_level="subject"),
#             model=ModelConfig(family="connectivity_only", connectivity_encoder_type="cnn", emb_dim=64, dropout=0.2),
#             aggregation=AggregationConfig(strategy="none"),
#             connectivity_tensor=ConnectivityTensorConfig(metrics=("coherence",), bands=(1, 2, 3)),
#         )
#     )
#     specs.append(
#         replace(
#             base,
#             name="subject_dense_dual_branch",
#             level=LevelConfig(graph_level="subject"),
#             model=ModelConfig(family="dense_dual_branch", connectivity_encoder_type="cnn", emb_dim=64, dropout=0.2),
#             aggregation=AggregationConfig(strategy="none"),
#             connectivity_tensor=ConnectivityTensorConfig(metrics=("coherence",), bands=(1, 2, 3)),
#         )
#     )

#     # --------------------------------------------------
#     # Block 2: subject-level graph baselines
#     # --------------------------------------------------
#     specs.append(
#         replace(
#             base,
#             name="subject_fixed_graph_gnn",
#             level=LevelConfig(graph_level="subject"),
#             topology=TopologyConfig(strategy="fixed"),
#             edge_weights=EdgeWeightConfig(strategy="binary"),
#             model=ModelConfig(family="fixed_graph_gnn", backbone="gcn", graph_readout="mean_max", emb_dim=64),
#             aggregation=AggregationConfig(strategy="none"),
#         )
#     )
#     specs.append(
#         replace(
#             base,
#             name="subject_dual_branch_graph",
#             level=LevelConfig(graph_level="subject"),
#             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
#             edge_weights=EdgeWeightConfig(strategy="connectivity", edge_metric="coherence", edge_band=2),
#             model=ModelConfig(family="dual_branch_graph", backbone="gcn", graph_readout="mean_max", emb_dim=64, fusion_mode="gated"),
#             aggregation=AggregationConfig(strategy="none"),
#         )
#     )

#     # --------------------------------------------------
#     # Block 3: segment-level + subject aggregation
#     # --------------------------------------------------
#     specs.append(
#         replace(
#             base,
#             name="segment_fixed_graph_mean_mil",
#             level=LevelConfig(graph_level="segment"),
#             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
#             edge_weights=EdgeWeightConfig(strategy="connectivity", edge_metric="coherence", edge_band=2),
#             model=ModelConfig(family="fixed_graph_gnn", backbone="gcn", graph_readout="mean", emb_dim=64),
#             aggregation=AggregationConfig(strategy="mean_mil", train_max_instances_per_subject=100, eval_max_instances_per_subject=None),
#             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),
#         )
#     )
#     specs.append(
#         replace(
#             base,
#             name="segment_fixed_graph_gated_mil",
#             level=LevelConfig(graph_level="segment"),
#             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
#             edge_weights=EdgeWeightConfig(strategy="connectivity", edge_metric="coherence", edge_band=2),
#             model=ModelConfig(family="fixed_graph_gnn", backbone="gcn", graph_readout="attention", emb_dim=64),
#             aggregation=AggregationConfig(strategy="gated_attention_mil", attn_dim=64, train_max_instances_per_subject=100),
#             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),

#         )
#     )

#     # --------------------------------------------------
#     # Block 4: macro graphs + light subject fusion
#     # --------------------------------------------------
#     specs.append(
#         replace(
#             base,
#             name="macro_dual_branch_subject_fusion",
#             level=LevelConfig(graph_level="macro", macro_duration_sec=300.0, feature_reduce="mean", connectivity_reduce="mean"),
#             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
#             edge_weights=EdgeWeightConfig(strategy="normalized", edge_metric="coherence", edge_band=2, normalize_mode="absmax"),
#             model=ModelConfig(family="dual_branch_graph", backbone="gcn", graph_readout="mean_max", emb_dim=64, fusion_mode="gated"),
#             aggregation=AggregationConfig(strategy="subject_fusion", train_max_instances_per_subject=None),
#             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),            
#         )
#     )

#     # --------------------------------------------------
#     # Block 5: fused graph bank
#     # --------------------------------------------------
#     bank_specs = default_graph_bank_specs(metrics=("coherence",), bands=(1, 2, 3))
#     specs.append(
#         replace(
#             base,
#             name="segment_graph_bank_gnn",
#             level=LevelConfig(graph_level="segment"),
#             topology=TopologyConfig(
#                 strategy="fused_bank",
#                 graph_bank_specs=bank_specs,
#                 fuse_method="mean",
#                 fuse_topology_rule="union",
#                 primary_candidate=0,
#             ),
#             edge_weights=EdgeWeightConfig(strategy="fused", fused_sources=(("coherence", 1), ("coherence", 2), ("coherence", 3)), fused_method="mean"),
#             model=ModelConfig(family="fused_graph_bank_gnn", backbone="gcn", graph_readout="attention", emb_dim=64, graph_bank_fusion_mode="summary_gated"),
#             aggregation=AggregationConfig(strategy="gated_attention_mil", attn_dim=64, train_max_instances_per_subject=100),
#             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),                    
#         )
#     )
#     specs.append(
#         replace(
#             base,
#             name="segment_dual_branch_graph_bank",
#             level=LevelConfig(graph_level="segment"),
#             topology=TopologyConfig(
#                 strategy="fused_bank",
#                 graph_bank_specs=bank_specs,
#                 fuse_method="mean",
#                 fuse_topology_rule="union",
#                 primary_candidate=0,
#             ),
#             edge_weights=EdgeWeightConfig(strategy="fused", fused_sources=(("coherence", 1), ("coherence", 2), ("coherence", 3)), fused_method="mean"),
#             model=ModelConfig(family="dual_branch_graph", backbone="gcn", graph_readout="attention", emb_dim=64, fusion_mode="gated", graph_bank_fusion_mode="summary_gated"),
#             aggregation=AggregationConfig(strategy="gated_attention_mil", attn_dim=64, train_max_instances_per_subject=100),
#             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),                    

#         )
#     )

#     return specs




def default_fixed_edge_pairs_19() -> list[tuple[int, int]]:
    """
    Simple hand-crafted 10-20 neighbor graph for CAUEEG 19 EEG channels.

    Channel order:
    0 Fp1, 1 F3, 2 C3, 3 P3, 4 O1,
    5 Fp2, 6 F4, 7 C4, 8 P4, 9 O2,
    10 F7, 11 T3, 12 T5, 13 F8, 14 T4,
    15 T6, 16 FZ, 17 CZ, 18 PZ
    """
    return [
        (0, 1), (0, 10), (0, 5),
        (5, 6), (5, 13),

        (10, 1), (10, 11),
        (11, 2), (11, 12),
        (12, 3), (12, 4),

        (13, 6), (13, 14),
        (14, 7), (14, 15),
        (15, 8), (15, 9),

        (1, 2), (2, 3), (3, 4),
        (6, 7), (7, 8), (8, 9),

        (1, 16), (6, 16),
        (2, 17), (7, 17),
        (3, 18), (8, 18),

        (16, 17), (17, 18),
    ]


def default_graph_bank_specs() -> list[dict[str, Any]]:
    """
    Stronger graph-bank candidate set.
    Use mostly integer bands to avoid string-band parsing issues.

    Band index map:
      0 delta
      1 theta
      2 alpha
      3 beta
      4 gamma
    """
    return [
        {
            "name": "coh_alpha_topk",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": 2,
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "coh_alpha_mst",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": 2,
            "topology_kwargs": {"mode": "mst"},
        },
        {
            "name": "coh_theta_topk",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": 1,
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "wpli_alpha_topk",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "wpli",
            "band": 2,
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "plv_fixed",
            "topology_mode": "fixed",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "plv",
            "band": 2,
            # "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        # {
        #     "name": "pearson_topk",
        #     "topology_mode": "connectivity",
        #     "edge_weight_mode": "connectivity",
        #     "connectivity_metric": "pearson",
        #     "band": None,
        #     "topology_kwargs": {"mode": "topk", "topk": 4},
        # },
        {
            "name": "feature_cosine_topk",
            "topology_mode": "feature_induced",
            "edge_weight_mode": "topology_weight",
            "similarity": "cosine",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
    ]


def build_new_caueeg_ladder(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_caueeg",
) -> list[CAUEEGExperimentSpec]:
    """
    Expanded Block-0 ladder.

    Goals:
    - keep subject-level baselines
    - add much stronger segment/macro coverage
    - test more than just connectivity topk
    - include fixed / mst / threshold / feature-induced / fused-bank
    """

    # integer band ids to avoid string-band parsing issues
    DELTA = 0
    THETA = 1
    ALPHA = 2
    BETA = 3
    GAMMA = 4

    fixed_edges = default_fixed_edge_pairs_19()

    base = CAUEEGExperimentSpec(
        name="base",
        task=task,
        dataset_path=dataset_path,
        h5_path=h5_path,
        output_root=output_root,
        feature_families=("relative_band_power", "statistical"), #"hjorth", 
        connectivity_metrics_to_load=("coherence", "wpli", "plv"), # "pearson", ),
        connectivity_tensor=ConnectivityTensorConfig(
            metrics=("wpli",),
            bands=(DELTA, THETA, ALPHA, BETA, GAMMA),
        ),
        train=TrainConfig(
            batch_size=16,
            epochs=200,
            patience=30,
            lr=1e-3,
            weight_decay=5e-3,
            seed=42,
        ),
    )

    subject_train = TrainConfig(
        batch_size=16,
        epochs=200,
        patience=30,
        lr=1e-3,
        weight_decay=5e-3,
        seed=42,
    )

    segment_train = TrainConfig(
        batch_size=8,
        epochs=200,
        patience=30,
        lr=1e-3,
        weight_decay=5e-3,
        seed=42,
    )

    macro_train = TrainConfig(
        batch_size=8,
        epochs=200,
        patience=30,
        lr=1e-3,
        weight_decay=5e-3,
        seed=42,
    )

    specs: list[CAUEEGExperimentSpec] = []

    def add(name: str, **kwargs):
        specs.append(replace(base, name=name, **kwargs))

    # ==================================================
    # Block 1: subject-level dense baselines
    # ==================================================
    # add(
    #     "subject_node_only",
    #     level=LevelConfig(graph_level="subject"),
    #     model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
    #     aggregation=AggregationConfig(strategy="none"),
    #     train=subject_train,
    # )

    # add(
    #     "subject_connectivity_cnn",
    #     level=LevelConfig(graph_level="subject"),
    #     model=ModelConfig(
    #         family="connectivity_only",
    #         connectivity_encoder_type="cnn",
    #         emb_dim=64,
    #         dropout=0.2,
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
    #     train=subject_train,
    # )

    # add(
    #     "subject_dense_dual_branch",
    #     level=LevelConfig(graph_level="subject"),
    #     model=ModelConfig(
    #         family="dense_dual_branch",
    #         connectivity_encoder_type="cnn",
    #         emb_dim=64,
    #         dropout=0.2,
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
    #     train=subject_train,
    # )


    add(
        "subject_graph_bank",
        level=LevelConfig(graph_level="subject"),
        topology=TopologyConfig(
            strategy="fused_bank",
            graph_bank_specs=default_graph_bank_specs(),
            fuse_method="mean",
            fuse_topology_rule="union",
            primary_candidate=0,
        ),
        edge_weights=EdgeWeightConfig(
            strategy="fused",
            fused_sources=(
                ("coherence", THETA),
                ("coherence", ALPHA),
                ("coherence", BETA),
                ("wpli", ALPHA),
            ),
            fused_method="mean",
        ),
        model=ModelConfig(
            family="fused_graph_bank_gnn",
            backbone="gatv2",
            graph_readout="attention",
            emb_dim=64,
            graph_bank_fusion_mode="summary_gated",
        ),
        aggregation=AggregationConfig(
            strategy="none",
            # attn_dim=64,
            # train_max_instances_per_subject=100,
        ),
        train=subject_train,
    )

    # # ==================================================
    # # Block 2: subject-level graph baselines
    # # ==================================================
    # add(
    #     "subject_fixed_binary_gnn",
    #     level=LevelConfig(graph_level="subject"),
    #     topology=TopologyConfig(
    #         strategy="fixed",
    #         fixed_edge_pairs=fixed_edges,
    #     ),
    #     edge_weights=EdgeWeightConfig(strategy="binary"),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gcn",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     train=subject_train,
    # )

    add(
        "subject_connectivity_topk_gatv2",
        level=LevelConfig(graph_level="subject"),
        topology=TopologyConfig(
            strategy="connectivity",
            topology_metric="wpli",
            topology_band=ALPHA,
            topology_kwargs={"mode": "topk", "topk": 4},
        ),
        edge_weights=EdgeWeightConfig(
            strategy="connectivity",
            edge_metric="wpli",
            edge_band=ALPHA,
        ),
        model=ModelConfig(
            family="fixed_graph_gnn",
            backbone="gatv2",
            graph_readout="mean_max",
            emb_dim=64,
        ),
        aggregation=AggregationConfig(strategy="none"),
        train=subject_train,
    )

    add(
        "subject_connectivity_mst_gatv2",
        level=LevelConfig(graph_level="subject"),
        topology=TopologyConfig(
            strategy="connectivity",
            topology_metric="wpli",
            topology_band=ALPHA,
            topology_kwargs={"mode": "mst"},
        ),
        edge_weights=EdgeWeightConfig(
            strategy="connectivity",
            edge_metric="wpli",
            edge_band=ALPHA,
        ),
        model=ModelConfig(
            family="fixed_graph_gnn",
            backbone="gatv2",
            graph_readout="mean_max",
            emb_dim=64,
        ),
        aggregation=AggregationConfig(strategy="none"),
        train=subject_train,
    )

    add(
        "subject_connectivity_fixed_gatv2",
        level=LevelConfig(graph_level="subject"),
        topology=TopologyConfig(
            strategy="fixed",
            fixed_edge_pairs=fixed_edges,
            # topology_metric="wpli",
            # topology_band=ALPHA,
                # add(
    #     "subject_fixed_binary_gnn",
    #     level=LevelConfig(graph_level="subject"),
    #     topology=TopologyConfig(
    #         strategy="fixed",
    #         fixed_edge_pairs=fixed_edges,
    #     ),
    #     edge_weights=EdgeWeightConfig(strategy="binary"),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gcn",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     train=subject_train,
    # )
            # topology_kwargs={"mode": "mst"},
        ),
        edge_weights=EdgeWeightConfig(
            strategy="connectivity",
            edge_metric="wpli",
            edge_band=ALPHA,
        ),
        model=ModelConfig(
            family="fixed_graph_gnn",
            backbone="gatv2",
            graph_readout="mean_max",
            emb_dim=64,
        ),
        aggregation=AggregationConfig(strategy="none"),
        train=subject_train,
    )

    # add(
    #     "subject_connectivity_threshold_gnn",
    #     level=LevelConfig(graph_level="subject"),
    #     topology=TopologyConfig(
    #         strategy="connectivity",
    #         topology_metric="coherence",
    #         topology_band=ALPHA,
    #         topology_kwargs={"mode": "threshold", "threshold": 0.30},
    #     ),
    #     edge_weights=EdgeWeightConfig(
    #         strategy="normalized",
    #         edge_metric="coherence",
    #         edge_band=ALPHA,
    #         normalize_mode="absmax",
    #     ),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gcn",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     train=subject_train,
    # )

    # add(
    #     "subject_feature_induced_gatv2",
    #     level=LevelConfig(graph_level="subject"),
    #     topology=TopologyConfig(
    #         strategy="feature_induced",
    #         similarity="cosine",
    #         topology_kwargs={"mode": "topk", "topk": 4},
    #     ),
    #     edge_weights=EdgeWeightConfig(strategy="binary"),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gatv2",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     train=subject_train,
    # )

    # add(
    #     "subject_dual_branch_graph_topk",
    #     level=LevelConfig(graph_level="subject"),
    #     topology=TopologyConfig(
    #         strategy="connectivity",
    #         topology_metric="coherence",
    #         topology_band=ALPHA,
    #         topology_kwargs={"mode": "topk", "topk": 4},
    #     ),
    #     edge_weights=EdgeWeightConfig(
    #         strategy="connectivity",
    #         edge_metric="coherence",
    #         edge_band=ALPHA,
    #     ),
    #     model=ModelConfig(
    #         family="dual_branch_graph",
    #         backbone="gcn",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #         fusion_mode="gated",
    #     ),
    #     aggregation=AggregationConfig(strategy="none"),
    #     train=subject_train,
    # )

    # ==================================================
    # Block 3: segment-level dense + MIL
    # ==================================================
    add(
        "segment_node_only_mean_mil",
        level=LevelConfig(graph_level="segment"),
        model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    add(
        "segment_node_only_gated_mil",
        level=LevelConfig(graph_level="segment"),
        model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
        aggregation=AggregationConfig(
            strategy="gated_attention_mil",
            attn_dim=64,
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    add(
        "segment_connectivity_only_mean_mil",
        level=LevelConfig(graph_level="segment"),
        model=ModelConfig(
            family="connectivity_only",
            connectivity_encoder_type="cnn",
            emb_dim=64,
            dropout=0.2,
        ),
        connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    add(
        "segment_dense_dual_branch_gated_mil",
        level=LevelConfig(graph_level="segment"),
        model=ModelConfig(
            family="dense_dual_branch",
            connectivity_encoder_type="cnn",
            emb_dim=64,
            dropout=0.2,
        ),
        connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
        aggregation=AggregationConfig(
            strategy="gated_attention_mil",
            attn_dim=64,
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )


    add(
        "segment_dense_dual_branch_mean_mil",
        level=LevelConfig(graph_level="segment"),
        model=ModelConfig(
            family="dense_dual_branch",
            connectivity_encoder_type="cnn",
            emb_dim=64,
            dropout=0.2,
        ),
        connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            attn_dim=64,
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    # ==================================================
    # Block 4: segment-level graph + MIL
    # ==================================================
    add(
        "segment_connectivity_fixed_mean_mil",
        level=LevelConfig(graph_level="segment"),
        topology=TopologyConfig(
            strategy="fixed",
            fixed_edge_pairs=fixed_edges,
        ),
        edge_weights=EdgeWeightConfig(            
            strategy="connectivity",
            edge_metric="wpli",
            edge_band=ALPHA,
        ),
        model=ModelConfig(
            family="fixed_graph_gnn",
            backbone="gcn",
            graph_readout="mean",
            emb_dim=64,
        ),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    add(
        "segment_connectivity_topk_mean_mil",
        level=LevelConfig(graph_level="segment"),
        topology=TopologyConfig(
            strategy="connectivity",
            topology_metric="wpli",
            topology_band=ALPHA,
            topology_kwargs={"mode": "topk", "topk": 4},
        ),
        edge_weights=EdgeWeightConfig(
            strategy="connectivity",
            edge_metric="wpli",
            edge_band=ALPHA,
        ),
        model=ModelConfig(
            family="fixed_graph_gnn",
            backbone="gatv2",
            graph_readout="attention",
            emb_dim=64,
        ),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            attn_dim=64,
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    add(
        "segment_connectivity_mst_mean_mil",
        level=LevelConfig(graph_level="segment"),
        topology=TopologyConfig(
            strategy="connectivity",
            topology_metric="wpli",
            topology_band=ALPHA,
            topology_kwargs={"mode": "mst"},
        ),
        edge_weights=EdgeWeightConfig(
            strategy="connectivity",
            edge_metric="coherence",
            edge_band=ALPHA,
        ),
        model=ModelConfig(
            family="fixed_graph_gnn",
            backbone="gatv2",
            graph_readout="attention",
            emb_dim=64,
        ),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            attn_dim=64,
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    # add(
    #     "segment_connectivity_threshold_gated_mil",
    #     level=LevelConfig(graph_level="segment"),
    #     topology=TopologyConfig(
    #         strategy="connectivity",
    #         topology_metric="coherence",
    #         topology_band=ALPHA,
    #         topology_kwargs={"mode": "threshold", "threshold": 0.30},
    #     ),
    #     edge_weights=EdgeWeightConfig(
    #         strategy="normalized",
    #         edge_metric="coherence",
    #         edge_band=ALPHA,
    #         normalize_mode="absmax",
    #     ),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gcn",
    #         graph_readout="attention",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(
    #         strategy="gated_attention_mil",
    #         attn_dim=64,
    #         train_max_instances_per_subject=100,
    #     ),
    #     train=segment_train,
    # )

    # add(
    #     "segment_feature_induced_mean_mil",
    #     level=LevelConfig(graph_level="segment"),
    #     topology=TopologyConfig(
    #         strategy="feature_induced",
    #         similarity="cosine",
    #         topology_kwargs={"mode": "topk", "topk": 4},
    #     ),
    #     edge_weights=EdgeWeightConfig(strategy="binary"),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gatv2",
    #         graph_readout="attention",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(
    #         strategy="mean_mil",
    #         attn_dim=64,
    #         train_max_instances_per_subject=100,
    #     ),
    #     train=segment_train,
    # )

    # add(
    #     "segment_dual_branch_graph_topk",
    #     level=LevelConfig(graph_level="segment"),
    #     topology=TopologyConfig(
    #         strategy="connectivity",
    #         topology_metric="coherence",
    #         topology_band=ALPHA,
    #         topology_kwargs={"mode": "topk", "topk": 4},
    #     ),
    #     edge_weights=EdgeWeightConfig(
    #         strategy="connectivity",
    #         edge_metric="coherence",
    #         edge_band=ALPHA,
    #     ),
    #     model=ModelConfig(
    #         family="dual_branch_graph",
    #         backbone="gcn",
    #         graph_readout="attention",
    #         emb_dim=64,
    #         fusion_mode="gated",
    #     ),
    #     aggregation=AggregationConfig(
    #         strategy="gated_attention_mil",
    #         attn_dim=64,
    #         train_max_instances_per_subject=100,
    #     ),
    #     train=segment_train,
    # )

    add(
        "segment_graph_bank_mean_mil",
        level=LevelConfig(graph_level="segment"),
        topology=TopologyConfig(
            strategy="fused_bank",
            graph_bank_specs=default_graph_bank_specs(),
            fuse_method="mean",
            fuse_topology_rule="union",
            primary_candidate=0,
        ),
        edge_weights=EdgeWeightConfig(
            strategy="fused",
            fused_sources=(
                ("coherence", THETA),
                ("coherence", ALPHA),
                ("coherence", BETA),
                ("wpli", ALPHA),
            ),
            fused_method="mean",
        ),
        model=ModelConfig(
            family="fused_graph_bank_gnn",
            backbone="gatv2",
            graph_readout="attention",
            emb_dim=64,
            graph_bank_fusion_mode="summary_gated",
        ),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            attn_dim=64,
            train_max_instances_per_subject=100,
        ),
        train=segment_train,
    )

    # ==================================================
    # Block 5: macro-level dense + subject fusion
    # ==================================================
    add(
        "macro_node_only_subject_fusion",
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
        aggregation=AggregationConfig(strategy="subject_fusion"),
        train=macro_train,
    )

    add(
        "macro_connectivity_only_subject_fusion",
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        model=ModelConfig(
            family="connectivity_only",
            connectivity_encoder_type="cnn",
            emb_dim=64,
            dropout=0.2,
        ),
        connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
        aggregation=AggregationConfig(strategy="subject_fusion"),
        train=macro_train,
    )

    add(
        "macro_dense_dual_branch_subject_fusion",
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        model=ModelConfig(
            family="dense_dual_branch",
            connectivity_encoder_type="cnn",
            emb_dim=64,
            dropout=0.2,
        ),
        connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
        aggregation=AggregationConfig(strategy="subject_fusion"),
        train=macro_train,
    )

    # # ==================================================
    # # Block 6: macro-level graph + subject fusion
    # # ==================================================
    # add(
    #     "macro_connectivity_topk_subject_fusion",
    #     level=LevelConfig(
    #         graph_level="macro",
    #         macro_duration_sec=300.0,
    #         feature_reduce="mean",
    #         connectivity_reduce="mean",
    #     ),
    #     topology=TopologyConfig(
    #         strategy="connectivity",
    #         topology_metric="wpli",
    #         topology_band=ALPHA,
    #         topology_kwargs={"mode": "topk", "topk": 4},
    #     ),
    #     edge_weights=EdgeWeightConfig(
    #         strategy="normalized",
    #         edge_metric="wpli",
    #         edge_band=ALPHA,
    #         normalize_mode="absmax",
    #     ),
    #     model=ModelConfig(
    #         family="dual_branch_graph",
    #         backbone="gatv2",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #         fusion_mode="gated",
    #     ),
    #     aggregation=AggregationConfig(strategy="subject_fusion"),
    #     train=macro_train,
    # )

    add(
        "macro_connectivity_mst_subject_fusion",
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        topology=TopologyConfig(
            strategy="connectivity",
            topology_metric="wpli",
            topology_band=ALPHA,
            topology_kwargs={"mode": "mst"},
        ),
        edge_weights=EdgeWeightConfig(
            strategy="normalized",
            edge_metric="wpli",
            edge_band=ALPHA,
            normalize_mode="absmax",
        ),
        model=ModelConfig(
            family="dual_branch_graph",
            backbone="gatv2",
            graph_readout="mean_max",
            emb_dim=64,
            fusion_mode="gated",
        ),
        aggregation=AggregationConfig(strategy="subject_fusion"),
        train=macro_train,
    )

    add(
        "macro_connectivity_fixed_subject_fusion",
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        topology=TopologyConfig(
            strategy="fixed",
            fixed_edge_pairs=fixed_edges,
        ),
        edge_weights=EdgeWeightConfig(
            strategy="normalized",
            edge_metric="wpli",
            edge_band=ALPHA,
            normalize_mode="absmax",
        ),
        model=ModelConfig(
            family="dual_branch_graph",
            backbone="gatv2",
            graph_readout="mean_max",
            emb_dim=64,
            fusion_mode="gated",
        ),
        aggregation=AggregationConfig(strategy="subject_fusion"),
        train=macro_train,
    )

    # add(
    #     "macro_feature_induced_subject_fusion",
    #     level=LevelConfig(
    #         graph_level="macro",
    #         macro_duration_sec=300.0,
    #         feature_reduce="mean",
    #         connectivity_reduce="mean",
    #     ),
    #     topology=TopologyConfig(
    #         strategy="feature_induced",
    #         similarity="cosine",
    #         topology_kwargs={"mode": "topk", "topk": 4},
    #     ),
    #     edge_weights=EdgeWeightConfig(strategy="binary"),
    #     model=ModelConfig(
    #         family="fixed_graph_gnn",
    #         backbone="gcn",
    #         graph_readout="mean_max",
    #         emb_dim=64,
    #     ),
    #     aggregation=AggregationConfig(strategy="subject_fusion"),
    #     train=macro_train,
    # )

    add(
        "macro_graph_bank_subject_fusion",
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        topology=TopologyConfig(
            strategy="fused_bank",
            graph_bank_specs=default_graph_bank_specs(),
            fuse_method="mean",
            fuse_topology_rule="union",
            primary_candidate=0,
        ),
        edge_weights=EdgeWeightConfig(
            strategy="fused",
            fused_sources=(
                ("coherence", THETA),
                ("coherence", ALPHA),
                ("coherence", BETA),
                ("wpli", ALPHA),
            ),
            fused_method="mean",
        ),
        model=ModelConfig(
            family="fused_graph_bank_gnn",
            backbone="gcn",
            graph_readout="attention",
            emb_dim=64,
            graph_bank_fusion_mode="summary_gated",
        ),
        aggregation=AggregationConfig(strategy="subject_fusion"),
        train=macro_train,
    )

    return specs

def run_caueeg_ladder(specs: Sequence[CAUEEGExperimentSpec]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in specs:
        print("=" * 100)
        print(f"Running experiment: {spec.name}")
        print("=" * 100)
        out = run_caueeg_experiment(spec)
        results.append(out)
    return results


# import copy

# def build_stage2_caueeg_ladder(default_ladder, leaderboard_df):
#     name_to_spec = {spec.name: spec for spec in default_ladder}
#     winners = select_bucket_winners(leaderboard_df, top_k_per_bucket=1)

#     new_ladder = []

#     for _, row in winners.iterrows():
#         base_spec = copy.deepcopy(name_to_spec[row["name"]])

#         # expand one axis at a time
#         if row["model_family"] in {"fixed_graph_gnn", "dual_branch_graph", "fused_graph_bank_gnn"}:
#             for readout in ["mean", "mean_max_concat", "attention"]:
#                 spec2 = copy.deepcopy(base_spec)
#                 spec2.name = f"{base_spec.name}_readout_{readout}"
#                 spec2.readout = readout
#                 new_ladder.append(spec2)

#         if row["graph_level"] in {"segment", "macro"}:
#             for agg in ["mean_mil", "gated_attention_mil"]:
#                 spec2 = copy.deepcopy(base_spec)
#                 spec2.name = f"{base_spec.name}_agg_{agg}"
#                 spec2.subject_aggregation = agg
#                 new_ladder.append(spec2)

#     return new_ladder

# stage2_ladder = build_stage2_caueeg_ladder(default_ladder, leaderboard)
# for i, spec in enumerate(stage2_ladder):
#     print(i, spec.name)

# stage2_results = run_caueeg_ladder(stage2_ladder)


def summarize_subject_macro_bags(
    bags,
    *,
    split_name: str,
    sfreq: float = 200.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subject_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []

    for bag in bags:
        subject_id = str(bag.subject_id)
        label = int(bag.label)
        num_macros = int(len(bag.macros))

        macro_segment_counts: list[int] = []
        macro_start_samples: list[int] = []
        macro_end_samples: list[int] = []

        for macro in bag.macros:
            num_segments = int(macro.node_feature_seq.shape[0])
            start_sample = macro.metadata.get("start_sample", None)
            end_sample = macro.metadata.get("end_sample", None)
            window_indices = macro.metadata.get("window_indices", [])

            macro_segment_counts.append(num_segments)
            if start_sample is not None:
                macro_start_samples.append(int(start_sample))
            if end_sample is not None:
                macro_end_samples.append(int(end_sample))

            macro_duration_sec = None
            if start_sample is not None and end_sample is not None:
                macro_duration_sec = (int(end_sample) - int(start_sample)) / float(sfreq)

            macro_rows.append(
                {
                    "split": split_name,
                    "subject_id": subject_id,
                    "label": label,
                    "macro_id": int(macro.macro_id),
                    "num_segments_in_macro": num_segments,
                    "macro_start_sample": None if start_sample is None else int(start_sample),
                    "macro_end_sample": None if end_sample is None else int(end_sample),
                    "macro_duration_sec": macro_duration_sec,
                    "window_indices": json.dumps(window_indices),
                }
            )

        recording_start = min(macro_start_samples) if macro_start_samples else None
        recording_end = max(macro_end_samples) if macro_end_samples else None

        recording_duration_sec = None
        if recording_start is not None and recording_end is not None:
            recording_duration_sec = (recording_end - recording_start) / float(sfreq)

        total_segments = int(sum(macro_segment_counts))

        subject_rows.append(
            {
                "split": split_name,
                "subject_id": subject_id,
                "label": label,
                "num_segments_total": total_segments,
                "num_macros": num_macros,
                "recording_start_sample": recording_start,
                "recording_end_sample": recording_end,
                "recording_duration_sec": recording_duration_sec,
                "recording_duration_min": None if recording_duration_sec is None else recording_duration_sec / 60.0,
                "segments_per_macro": json.dumps(macro_segment_counts),
                "min_segments_per_macro": min(macro_segment_counts) if macro_segment_counts else None,
                "max_segments_per_macro": max(macro_segment_counts) if macro_segment_counts else None,
                "mean_segments_per_macro": float(np.mean(macro_segment_counts)) if macro_segment_counts else None,
            }
        )

    return pd.DataFrame(subject_rows), pd.DataFrame(macro_rows)


def summarize_macro_distribution_by_class(subject_df: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_df.groupby(["split", "label"], as_index=False)
        .agg(
            num_subjects=("subject_id", "count"),
            mean_num_segments_total=("num_segments_total", "mean"),
            std_num_segments_total=("num_segments_total", "std"),
            min_num_segments_total=("num_segments_total", "min"),
            max_num_segments_total=("num_segments_total", "max"),
            mean_num_macros=("num_macros", "mean"),
            std_num_macros=("num_macros", "std"),
            min_num_macros=("num_macros", "min"),
            max_num_macros=("num_macros", "max"),
            mean_recording_duration_min=("recording_duration_min", "mean"),
            std_recording_duration_min=("recording_duration_min", "std"),
            min_recording_duration_min=("recording_duration_min", "min"),
            max_recording_duration_min=("recording_duration_min", "max"),
            mean_min_segments_per_macro=("min_segments_per_macro", "mean"),
            mean_max_segments_per_macro=("max_segments_per_macro", "mean"),
            mean_mean_segments_per_macro=("mean_segments_per_macro", "mean"),
        )
    )


def inspect_macro_distribution_from_spec(
    spec: CAUEEGExperimentSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from macro_mvgnn_new_fixed import build_subject_macro_bags

    _, train_rows, val_rows, test_rows = load_caueeg_task_splits(spec.dataset_path, spec.task)

    train_pairs = resolve_h5_subject_ids_for_split(spec.h5_path, train_rows, "train")
    val_pairs = resolve_h5_subject_ids_for_split(spec.h5_path, val_rows, "val")
    test_pairs = resolve_h5_subject_ids_for_split(spec.h5_path, test_rows, "test")

    train_ids = [sid for sid, _, _ in train_pairs]
    val_ids = [sid for sid, _, _ in val_pairs]
    test_ids = [sid for sid, _, _ in test_pairs]

    all_ids = train_ids + val_ids + test_ids
    entries = load_h5_entries(
        spec.h5_path,
        all_ids,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
    )

    train_entries = {sid: entries[sid] for sid in train_ids if sid in entries}
    val_entries = {sid: entries[sid] for sid in val_ids if sid in entries}
    test_entries = {sid: entries[sid] for sid in test_ids if sid in entries}

    sfreq = 200.0

    train_bags = build_subject_macro_bags(
        train_entries,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
        graph_bank_specs=spec.topology.graph_bank_specs,
        macro_duration_sec=spec.level.macro_duration_sec,
        sfreq=sfreq,
        connectivity_reduce=spec.level.connectivity_reduce,
    )
    val_bags = build_subject_macro_bags(
        val_entries,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
        graph_bank_specs=spec.topology.graph_bank_specs,
        macro_duration_sec=spec.level.macro_duration_sec,
        sfreq=sfreq,
        connectivity_reduce=spec.level.connectivity_reduce,
    )
    test_bags = build_subject_macro_bags(
        test_entries,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
        graph_bank_specs=spec.topology.graph_bank_specs,
        macro_duration_sec=spec.level.macro_duration_sec,
        sfreq=sfreq,
        connectivity_reduce=spec.level.connectivity_reduce,
    )

    train_subject_df, train_macro_df = summarize_subject_macro_bags(train_bags, split_name="train", sfreq=sfreq)
    val_subject_df, val_macro_df = summarize_subject_macro_bags(val_bags, split_name="val", sfreq=sfreq)
    test_subject_df, test_macro_df = summarize_subject_macro_bags(test_bags, split_name="test", sfreq=sfreq)

    subject_df = pd.concat([train_subject_df, val_subject_df, test_subject_df], ignore_index=True)
    macro_df = pd.concat([train_macro_df, val_macro_df, test_macro_df], ignore_index=True)
    class_df = summarize_macro_distribution_by_class(subject_df)
    return subject_df, macro_df, class_df



if __name__ == "__main__":
    example_dataset_path = "/mnt/data/anphan/CAUEEG/caueeg-dataset"
    example_h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"

    spec = CAUEEGExperimentSpec(
        name="macro_mvgnn_default",
        task="dementia",
        dataset_path=example_dataset_path,
        h5_path=example_h5_path,
        feature_families=("relative_band_power", "statistical"),
        connectivity_metrics_to_load=("coherence", "wpli"),
        level=LevelConfig(
            graph_level="macro",
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
        ),
        model=ModelConfig(
            family="macro_mvgnn",
            backbone="gatv2",
            hidden_dim=64,
            emb_dim=64,
            num_layers=2,
            dropout=0.3,
            gat_heads=4,
            graph_readout="gated_attention",
            graph_bank_fusion_mode="summary_gated",
        ),
        aggregation=AggregationConfig(
            strategy="mean_mil",
            attn_dim=64,
        ),
        train=TrainConfig(
            batch_size=8,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=300,
            patience=150,
            monitor="macro_f1",
            monitor_mode="max",
            seed=42,
        ),
        output_root="./results_caueeg",
    )

    INSPECT_ONLY = True

    if INSPECT_ONLY:
        subject_df, macro_df, class_df = inspect_macro_distribution_from_spec(spec)

        print("\n=== Subject summary ===")
        print(subject_df.head(20).to_string(index=False))

        print("\n=== Class summary ===")
        print(class_df.to_string(index=False))

        print("\n=== Macro summary ===")
        print(macro_df.head(30).to_string(index=False))

        subject_df.to_csv("macro_subject_summary.csv", index=False)
        macro_df.to_csv("macro_detail_summary.csv", index=False)
        class_df.to_csv("macro_class_summary.csv", index=False)
    else:
        run_caueeg_ladder([spec])
