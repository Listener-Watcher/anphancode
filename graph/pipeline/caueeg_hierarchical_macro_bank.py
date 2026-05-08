from __future__ import annotations

"""
Hierarchical macro-level connectivity-bank model for CAUEEG.

What this module gives you
--------------------------
1. H5-first data preparation that reuses the official CAUEEG task splits.
2. Subject bags where each subject contains several macro instances.
3. Each macro instance contains:
   - node features: aggregated from all windows inside the macro
   - connectivity bank: [num_bands, num_metrics, num_nodes, num_nodes]
4. A new model that:
   - learns metric weights inside each band
   - learns band weights after graph encoding
   - uses a graph branch at macro level
   - uses a global residual connectivity branch at subject level
   - fuses both branches for final subject classification

This is designed as a practical prototype that can run next to your current
project files without requiring you to rewrite `caueeg_main.py` immediately.
"""

import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
from metrics import summarize_classification_metrics
from trainer import Trainer
from utils import ensure_dir, get_device, set_seed


DEFAULT_BAND_ORDER: tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")


# -----------------------------------------------------------------------------
# Minimal H5 / split helpers copied here to avoid importing caueeg_main.py
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class _H5SubjectEntry:
    subject_id: str
    label: int
    channel_names: list[str]
    segment_id: np.ndarray
    start_sample: np.ndarray
    end_sample: np.ndarray
    features: dict[str, np.ndarray]
    connectivity: dict[str, np.ndarray]
    connectivity_band_names: dict[str, Optional[list[str]]]


def load_caueeg_task_splits(dataset_path: str, task: str) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task = str(task).lower()
    task_json = os.path.join(dataset_path, f"{task}.json")
    with open(task_json, "r", encoding="utf-8") as f:
        task_dict = json.load(f)
    config = {k: v for k, v in task_dict.items() if k not in {"train_split", "validation_split", "test_split"}}
    return config, list(task_dict["train_split"]), list(task_dict["validation_split"]), list(task_dict["test_split"])


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
    import h5py
    with h5py.File(h5_path, "r") as h5f:
        return sorted(list(h5f["subjects"].keys()))


def resolve_h5_subject_ids_for_split(
    h5_path: str,
    split_rows: Sequence[Mapping[str, Any]],
    split_name: str,
) -> list[tuple[str, int, str]]:
    available = set(list_h5_subject_ids(h5_path))
    resolved: list[tuple[str, int, str]] = []
    for row in split_rows:
        raw_serial = _serial_from_split_row(row)
        if raw_serial in {"00587", "00781", "01301"}:
            continue
        label = _label_from_split_row(row)
        candidates = [raw_serial, f"{split_name}_{raw_serial}", f"{split_name.lower()}_{raw_serial}", f"{split_name.upper()}_{raw_serial}"]
        chosen = None
        for cand in candidates:
            if cand in available:
                chosen = cand
                break
        if chosen is None:
            raise KeyError(f"Could not resolve serial={raw_serial!r} for split={split_name!r} in H5 {h5_path}.")
        resolved.append((chosen, label, raw_serial))
    return resolved


def load_h5_entries(
    h5_path: str,
    subject_ids: Sequence[str],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
) -> dict[str, _H5SubjectEntry]:
    import h5py
    entries: dict[str, _H5SubjectEntry] = {}
    bad_ids = {"train_00587", "train_00781", "train_01301"}
    with h5py.File(h5_path, "r") as h5f:
        for sid in subject_ids:
            if sid in bad_ids:
                continue
            grp = h5f[f"subjects/{sid}"]
            entry_features: dict[str, np.ndarray] = {}
            entry_connectivity: dict[str, np.ndarray] = {}
            entry_band_names: dict[str, Optional[list[str]]] = {}

            for fam in feature_families:
                entry_features[fam] = np.asarray(grp[f"windows/features/{fam}"][:], dtype=np.float32)

            for metric in connectivity_metrics:
                ds = grp[f"windows/connectivity/{metric}"]
                entry_connectivity[metric] = np.asarray(ds[:], dtype=np.float32)
                band_names = ds.attrs.get("band_names", None)
                if band_names is None:
                    entry_band_names[metric] = None
                else:
                    out_names: list[str] = []
                    for x in band_names:
                        out_names.append(x.decode("utf-8") if isinstance(x, bytes) else str(x))
                    entry_band_names[metric] = out_names

            meta = grp["metadata"]
            ch_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in grp["metadata/channel_names"][:]]
            entries[sid] = _H5SubjectEntry(
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


def reduce_array(x: np.ndarray, how: str, axis: int = 0) -> np.ndarray:
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


def aggregate_feature_families(
    entry: _H5SubjectEntry,
    window_indices: np.ndarray,
    feature_families: Sequence[str],
    reduce_mode: str,
) -> np.ndarray:
    feat_parts: list[np.ndarray] = []
    for fam in feature_families:
        x = np.asarray(entry.features[fam], dtype=np.float32)
        x_sel = x[window_indices]
        if len(window_indices) == 1:
            feat_parts.append(x_sel[0])
        else:
            feat_parts.append(reduce_array(x_sel, reduce_mode, axis=0).astype(np.float32))
    return np.concatenate(feat_parts, axis=-1).astype(np.float32)


def aggregate_connectivity_sources(
    entry: _H5SubjectEntry,
    window_indices: np.ndarray,
    connectivity_metrics: Sequence[str],
    reduce_mode: str,
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


def build_macro_groups(start_sample: np.ndarray, *, sfreq: float, macro_duration_sec: float) -> dict[int, np.ndarray]:
    block_len = int(round(float(sfreq) * float(macro_duration_sec)))
    if block_len < 1:
        raise ValueError("macro_duration_sec leads to block length < 1")
    macro_ids = np.floor_divide(np.asarray(start_sample, dtype=np.int64), block_len).astype(np.int64)
    out: dict[int, list[int]] = {}
    for idx, mid in enumerate(macro_ids.tolist()):
        out.setdefault(int(mid), []).append(int(idx))
    return {mid: np.asarray(idxs, dtype=np.int64) for mid, idxs in out.items()}


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class HierBankDataConfig:
    dataset_path: str
    h5_path: str
    task: str = "dementia"
    feature_families: tuple[str, ...] = ("relative_band_power", "hjorth", "statistical")
    connectivity_metrics: tuple[str, ...] = ("coherence", "pli", "wpli", "pearson", "spearman")
    bands: tuple[str, ...] = DEFAULT_BAND_ORDER
    macro_duration_sec: float = 300.0
    feature_reduce: str = "mean"
    connectivity_reduce: str = "mean"
    global_connectivity_reduce: str = "mean"
    broadcast_nonband_metrics: bool = True


@dataclass(slots=True)
class HierBankModelConfig:
    num_classes: int = 3
    graph_hidden_dim: int = 64
    graph_emb_dim: int = 128
    graph_num_layers: int = 2
    graph_dropout: float = 0.2
    graph_topk: int = 4
    scorer_type: str = "cnn"          # cnn | mlp
    scorer_hidden_dim: int = 64
    band_attention_dim: int = 64
    macro_attention_dim: int = 128
    global_branch_hidden_dim: int = 128
    global_branch_emb_dim: int = 128
    fusion_hidden_dim: int = 128
    fusion_dropout: float = 0.2


@dataclass(slots=True)
class HierBankTrainConfig:
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 15
    monitor: str = "balanced_accuracy"
    monitor_mode: str = "max"
    device: str = "cuda"
    max_grad_norm: Optional[float] = 5.0
    use_amp: bool = False
    output_root: str = "./results_caueeg_hier_bank"
    experiment_name: str = "macro_hier_bank_dual_branch"


@dataclass(slots=True)
class HierBankExperimentConfig:
    data: HierBankDataConfig
    model: HierBankModelConfig = field(default_factory=HierBankModelConfig)
    train: HierBankTrainConfig = field(default_factory=HierBankTrainConfig)


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------
@dataclass(slots=True)
class SubjectMacroBag:
    subject_id: str
    label: int
    macro_node_features: np.ndarray          # [K, N, F]
    macro_connectivity_bank: np.ndarray      # [K, B, M, N, N]
    global_connectivity_bank: np.ndarray     # [B, M, N, N]
    macro_ids: np.ndarray                    # [K]
    macro_start_samples: np.ndarray          # [K]
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
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


def _save_json(data: Any, path: str | os.PathLike) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)
    return str(path)


def _resolve_band_indices(
    available_band_names: Optional[Sequence[str]],
    requested_bands: Sequence[str],
    num_bands: int,
) -> list[int]:
    if available_band_names is None:
        default = list(DEFAULT_BAND_ORDER)
        if num_bands == len(default):
            available_band_names = default
        else:
            raise KeyError(
                "Band names are unavailable and the metric is banded with a non-default number of bands."
            )

    names = [str(x) for x in available_band_names]
    out: list[int] = []
    for band in requested_bands:
        if band not in names:
            raise KeyError(f"Requested band {band!r} not found in available bands {names}.")
        out.append(names.index(band))
    return out


def _prepare_metric_bank(
    values: np.ndarray,
    *,
    available_band_names: Optional[Sequence[str]],
    requested_bands: Sequence[str],
    broadcast_nonband_metrics: bool,
) -> np.ndarray:
    """
    Return one metric bank as [B, N, N].

    - If values are already banded [B_all, N, N], select the requested bands.
    - If values are 2D [N, N], either broadcast it across requested bands or raise.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 2:
        if not broadcast_nonband_metrics:
            raise ValueError(
                "Received a non-banded connectivity matrix but broadcast_nonband_metrics=False."
            )
        return np.stack([arr for _ in requested_bands], axis=0).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected connectivity values [N,N] or [B,N,N], got {arr.shape}.")

    band_idx = _resolve_band_indices(available_band_names, requested_bands, num_bands=arr.shape[0])
    return np.stack([arr[i] for i in band_idx], axis=0).astype(np.float32)


def _build_macro_bank_for_entry(
    entry: Any,
    cfg: HierBankDataConfig,
) -> SubjectMacroBag:
    groups = build_macro_groups(
        entry.start_sample,
        sfreq=200.0,
        macro_duration_sec=float(cfg.macro_duration_sec),
    )
    if len(groups) == 0:
        raise ValueError(f"Subject {entry.subject_id!r} has no macro groups.")

    macro_node_features: list[np.ndarray] = []
    macro_connectivity_banks: list[np.ndarray] = []
    macro_ids: list[int] = []
    macro_starts: list[int] = []

    for macro_id, window_idx in sorted(groups.items(), key=lambda kv: kv[0]):
        node_x = aggregate_feature_families(
            entry,
            window_idx,
            feature_families=cfg.feature_families,
            reduce_mode=str(cfg.feature_reduce),
        )
        node_x = zscore_node_features(node_x)

        conn_sources, conn_names = aggregate_connectivity_sources(
            entry,
            window_idx,
            connectivity_metrics=cfg.connectivity_metrics,
            reduce_mode=str(cfg.connectivity_reduce),
        )

        metric_banks: list[np.ndarray] = []
        for metric in cfg.connectivity_metrics:
            metric_banks.append(
                _prepare_metric_bank(
                    conn_sources[metric],
                    available_band_names=conn_names.get(metric, None),
                    requested_bands=cfg.bands,
                    broadcast_nonband_metrics=bool(cfg.broadcast_nonband_metrics),
                )
            )
        # [M, B, N, N] -> [B, M, N, N]
        metric_stack = np.stack(metric_banks, axis=0).astype(np.float32)
        macro_bank = np.transpose(metric_stack, (1, 0, 2, 3)).astype(np.float32)

        macro_node_features.append(node_x)
        macro_connectivity_banks.append(macro_bank)
        macro_ids.append(int(macro_id))
        macro_starts.append(int(np.min(entry.start_sample[window_idx])))

    macro_node_features_arr = np.stack(macro_node_features, axis=0).astype(np.float32)
    macro_connectivity_arr = np.stack(macro_connectivity_banks, axis=0).astype(np.float32)

    if macro_connectivity_arr.shape[0] == 1:
        global_bank = macro_connectivity_arr[0].copy()
    else:
        global_bank = reduce_array(
            macro_connectivity_arr,
            str(cfg.global_connectivity_reduce),
            axis=0,
        ).astype(np.float32)

    return SubjectMacroBag(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        macro_node_features=macro_node_features_arr,
        macro_connectivity_bank=macro_connectivity_arr,
        global_connectivity_bank=global_bank,
        macro_ids=np.asarray(macro_ids, dtype=np.int64),
        macro_start_samples=np.asarray(macro_starts, dtype=np.int64),
        metadata={
            "bands": list(cfg.bands),
            "metrics": list(cfg.connectivity_metrics),
            "num_macros": int(len(macro_ids)),
        },
    )


def prepare_subject_macro_bags(
    cfg: HierBankDataConfig,
) -> dict[str, list[SubjectMacroBag]]:
    """
    Build train/val/test subject bags directly from the H5 file and official task JSON.
    """
    _, train_rows, val_rows, test_rows = load_caueeg_task_splits(cfg.dataset_path, cfg.task)
    split_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
    resolved = {
        split: resolve_h5_subject_ids_for_split(cfg.h5_path, rows, split)
        for split, rows in split_rows.items()
    }

    all_subject_ids: list[str] = []
    for rows in resolved.values():
        all_subject_ids.extend([sid for sid, _, _ in rows])

    entries = load_h5_entries(
        cfg.h5_path,
        all_subject_ids,
        feature_families=cfg.feature_families,
        connectivity_metrics=cfg.connectivity_metrics,
    )

    out: dict[str, list[SubjectMacroBag]] = {"train": [], "val": [], "test": []}
    for split, rows in resolved.items():
        for sid, _, _ in rows:
            bag = _build_macro_bank_for_entry(entries[sid], cfg)
            out[split].append(bag)
    return out


# -----------------------------------------------------------------------------
# Dataset / collate
# -----------------------------------------------------------------------------
class SubjectMacroBankDataset(Dataset):
    def __init__(self, bags: Sequence[SubjectMacroBag]) -> None:
        self.bags = list(bags)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> SubjectMacroBag:
        return self.bags[idx]


def collate_subject_macro_banks(batch: Sequence[SubjectMacroBag]) -> dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    bsz = len(batch)
    max_k = max(int(x.macro_node_features.shape[0]) for x in batch)
    num_nodes = int(batch[0].macro_node_features.shape[1])
    node_feat_dim = int(batch[0].macro_node_features.shape[2])
    num_bands = int(batch[0].macro_connectivity_bank.shape[1])
    num_metrics = int(batch[0].macro_connectivity_bank.shape[2])

    macro_node_features = torch.zeros((bsz, max_k, num_nodes, node_feat_dim), dtype=torch.float32)
    macro_connectivity_bank = torch.zeros((bsz, max_k, num_bands, num_metrics, num_nodes, num_nodes), dtype=torch.float32)
    global_connectivity_bank = torch.zeros((bsz, num_bands, num_metrics, num_nodes, num_nodes), dtype=torch.float32)
    mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    labels = torch.zeros((bsz,), dtype=torch.long)

    subject_ids: list[str] = []
    macro_ids: list[np.ndarray] = []

    for i, bag in enumerate(batch):
        k = int(bag.macro_node_features.shape[0])
        macro_node_features[i, :k] = torch.from_numpy(bag.macro_node_features)
        macro_connectivity_bank[i, :k] = torch.from_numpy(bag.macro_connectivity_bank)
        global_connectivity_bank[i] = torch.from_numpy(bag.global_connectivity_bank)
        mask[i, :k] = True
        labels[i] = int(bag.label)
        subject_ids.append(str(bag.subject_id))
        macro_ids.append(np.asarray(bag.macro_ids, dtype=np.int64))

    return {
        "macro_node_features": macro_node_features,
        "macro_connectivity_bank": macro_connectivity_bank,
        "global_connectivity_bank": global_connectivity_bank,
        "mask": mask,
        "labels": labels,
        "subject_ids": subject_ids,
        "macro_ids": macro_ids,
    }


# -----------------------------------------------------------------------------
# Model pieces
# -----------------------------------------------------------------------------
class MatrixScorer(nn.Module):
    """Score one connectivity matrix and return a scalar logit."""

    def __init__(self, num_nodes: int, *, scorer_type: str = "cnn", hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.scorer_type = str(scorer_type).lower()
        if self.scorer_type not in {"cnn", "mlp"}:
            raise ValueError("scorer_type must be one of {'cnn', 'mlp'}." )

        if self.scorer_type == "cnn":
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(16 * 4 * 4, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )
        else:
            tri = self.num_nodes * (self.num_nodes - 1) // 2
            self.register_buffer("iu", torch.triu_indices(self.num_nodes, self.num_nodes, offset=1), persistent=False)
            self.net = nn.Sequential(
                nn.Linear(tri, int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., N, N]
        if x.ndim < 2:
            raise ValueError(f"Expected at least 2 dims for matrix scorer, got {tuple(x.shape)}")
        if x.shape[-2] != self.num_nodes or x.shape[-1] != self.num_nodes:
            raise ValueError(
                f"Expected last dims [{self.num_nodes}, {self.num_nodes}], got {tuple(x.shape)}"
            )

        flat = x.reshape(-1, self.num_nodes, self.num_nodes)
        if self.scorer_type == "cnn":
            logits = self.net(flat.unsqueeze(1)).reshape(*x.shape[:-2], 1)
        else:
            feat = flat[:, self.iu[0], self.iu[1]]
            logits = self.net(feat).reshape(*x.shape[:-2], 1)
        return logits.squeeze(-1)


class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_dim), int(out_dim))
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [B, N, F], adj: [B, N, N]
        n = adj.shape[-1]
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        a = adj + eye
        deg = a.sum(dim=-1)
        deg_inv_sqrt = deg.clamp_min(1e-6).pow(-0.5)
        a_norm = deg_inv_sqrt.unsqueeze(-1) * a * deg_inv_sqrt.unsqueeze(-2)
        out = torch.bmm(a_norm, self.linear(x))
        out = F.relu(out)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)
        return out


class DenseGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_node_features: int,
        hidden_dim: int,
        emb_dim: int,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be >= 1")
        layers: list[nn.Module] = []
        prev = int(num_node_features)
        for _ in range(int(num_layers) - 1):
            layers.append(DenseGCNLayer(prev, int(hidden_dim), dropout=float(dropout)))
            prev = int(hidden_dim)
        layers.append(DenseGCNLayer(prev, int(emb_dim), dropout=float(dropout)))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, adj)
        return h.mean(dim=1)


class MacroAttentionPool(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(int(in_dim), int(attn_dim)),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(attn_dim), 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, K, D], mask: [B, K]
        scores = self.att(x).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=1)
        return pooled, weights


class GlobalConnectivityResidualBranch(nn.Module):
    def __init__(self, num_bands: int, num_nodes: int, hidden_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.num_bands = int(num_bands)
        self.num_nodes = int(num_nodes)
        self.net = nn.Sequential(
            nn.Conv2d(self.num_bands, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(emb_dim)),
        )

    def forward(self, fused_band_mats: torch.Tensor) -> torch.Tensor:
        # fused_band_mats: [B, Bd, N, N]
        return self.net(fused_band_mats)


class GatedFusionHead(nn.Module):
    def __init__(self, in_dim_a: int, in_dim_b: int, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.proj_a = nn.Linear(int(in_dim_a), int(hidden_dim))
        self.proj_b = nn.Linear(int(in_dim_b), int(hidden_dim))
        self.gate = nn.Sequential(
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        za = self.proj_a(a)
        zb = self.proj_b(b)
        gate = self.gate(torch.cat([za, zb], dim=-1))
        fused = gate * za + (1.0 - gate) * zb
        logits = self.classifier(fused)
        return fused, logits


# -----------------------------------------------------------------------------
# Main model
# -----------------------------------------------------------------------------
class HierarchicalMacroConnectivityBankNet(nn.Module):
    """
    Macro-level hierarchical connectivity-bank model.

    Batch keys expected
    -------------------
    - macro_node_features: [B, K, N, F]
    - macro_connectivity_bank: [B, K, Bd, M, N, N]
    - global_connectivity_bank: [B, Bd, M, N, N]
    - mask: [B, K]
    - labels: [B]
    - subject_ids: list[str]
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_bands: int,
        num_metrics: int,
        cfg: HierBankModelConfig,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_bands = int(num_bands)
        self.num_metrics = int(num_metrics)
        self.cfg = cfg
        self.graph_topk = int(cfg.graph_topk)

        self.metric_scorer = MatrixScorer(
            self.num_nodes,
            scorer_type=str(cfg.scorer_type),
            hidden_dim=int(cfg.scorer_hidden_dim),
        )
        self.band_attention = nn.Sequential(
            nn.Linear(int(cfg.graph_emb_dim), int(cfg.band_attention_dim)),
            nn.Tanh(),
            nn.Linear(int(cfg.band_attention_dim), 1),
        )
        self.graph_encoder = DenseGraphEncoder(
            num_node_features=self.num_node_features,
            hidden_dim=int(cfg.graph_hidden_dim),
            emb_dim=int(cfg.graph_emb_dim),
            num_layers=int(cfg.graph_num_layers),
            dropout=float(cfg.graph_dropout),
        )
        self.macro_pool = MacroAttentionPool(
            in_dim=int(cfg.graph_emb_dim),
            attn_dim=int(cfg.macro_attention_dim),
            dropout=float(cfg.graph_dropout),
        )
        self.global_branch = GlobalConnectivityResidualBranch(
            num_bands=self.num_bands,
            num_nodes=self.num_nodes,
            hidden_dim=int(cfg.global_branch_hidden_dim),
            emb_dim=int(cfg.global_branch_emb_dim),
        )
        self.fusion_head = GatedFusionHead(
            in_dim_a=int(cfg.graph_emb_dim),
            in_dim_b=int(cfg.global_branch_emb_dim),
            hidden_dim=int(cfg.fusion_hidden_dim),
            num_classes=int(cfg.num_classes),
            dropout=float(cfg.fusion_dropout),
        )

    @staticmethod
    def _clean_adjacency(adj: torch.Tensor) -> torch.Tensor:
        adj = 0.5 * (adj + adj.transpose(-1, -2))
        eye = torch.eye(adj.shape[-1], device=adj.device, dtype=torch.bool)
        adj = adj.masked_fill(eye.unsqueeze(0).expand(adj.shape[0], -1, -1), 0.0)
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        return adj

    def _topk_sparsify(self, adj: torch.Tensor) -> torch.Tensor:
        # adj: [B, N, N]
        adj = self._clean_adjacency(adj)
        if self.graph_topk <= 0 or self.graph_topk >= adj.shape[-1]:
            return adj

        scores = adj.abs()
        topk_val, topk_idx = torch.topk(scores, k=int(self.graph_topk), dim=-1)
        del topk_val
        mask = torch.zeros_like(adj, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        mask = mask | mask.transpose(-1, -2)
        out = torch.where(mask, adj, torch.zeros_like(adj))
        return self._clean_adjacency(out)

    def _fuse_metrics_within_band(self, bank: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # bank: [*, Bd, M, N, N]
        scores = self.metric_scorer(bank)                      # [*, Bd, M]
        weights = torch.softmax(scores, dim=-1)
        fused = torch.sum(weights.unsqueeze(-1).unsqueeze(-1) * bank, dim=-3)  # [*, Bd, N, N]
        return fused, weights

    def _encode_macro_band_graphs(
        self,
        node_features: torch.Tensor,
        fused_band_adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # node_features: [B, K, N, F]
        # fused_band_adj: [B, K, Bd, N, N]
        # mask: [B, K]
        bsz, max_k, _, feat_dim = node_features.shape
        _ = feat_dim
        band_embs: list[torch.Tensor] = []

        for b in range(self.num_bands):
            x_band = node_features.reshape(bsz * max_k, self.num_nodes, self.num_node_features)
            adj_band = fused_band_adj[:, :, b].reshape(bsz * max_k, self.num_nodes, self.num_nodes)
            valid = mask.reshape(-1)
            x_valid = x_band[valid]
            adj_valid = self._topk_sparsify(adj_band[valid])
            emb_valid = self.graph_encoder(x_valid, adj_valid)

            emb_all = torch.zeros(
                (bsz * max_k, emb_valid.shape[-1]),
                dtype=emb_valid.dtype,
                device=emb_valid.device,
            )
            emb_all[valid] = emb_valid
            band_embs.append(emb_all.reshape(bsz, max_k, -1))

        band_embs_t = torch.stack(band_embs, dim=2)  # [B, K, Bd, D]
        band_scores = self.band_attention(band_embs_t).squeeze(-1)  # [B, K, Bd]
        band_weights = torch.softmax(band_scores, dim=2)
        macro_emb = torch.sum(band_weights.unsqueeze(-1) * band_embs_t, dim=2)  # [B, K, D]
        return macro_emb, band_weights

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        x = batch["macro_node_features"]
        macro_bank = batch["macro_connectivity_bank"]
        global_bank = batch["global_connectivity_bank"]
        mask = batch["mask"]
        labels = batch["labels"]
        subject_ids = batch["subject_ids"]

        # Metric fusion per macro and per band.
        macro_fused_band_adj, macro_metric_weights = self._fuse_metrics_within_band(macro_bank)
        macro_emb, band_weights = self._encode_macro_band_graphs(x, macro_fused_band_adj, mask)
        subject_graph_emb, macro_attention = self.macro_pool(macro_emb, mask)

        # Global residual connectivity branch.
        global_fused_band_adj, global_metric_weights = self._fuse_metrics_within_band(global_bank)
        global_emb = self.global_branch(global_fused_band_adj)

        fused_emb, logits = self.fusion_head(subject_graph_emb, global_emb)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
            "targets": labels,
            "subject_ids": list(subject_ids),
            "attention_weights": macro_attention,
            "embedding": fused_emb,
            "aux": {
                "macro_metric_weights": macro_metric_weights,
                "band_weights": band_weights,
                "macro_attention": macro_attention,
                "global_metric_weights": global_metric_weights,
                "macro_fused_band_adj": macro_fused_band_adj,
                "global_fused_band_adj": global_fused_band_adj,
                "subject_graph_embedding": subject_graph_emb,
                "global_embedding": global_emb,
            },
        }


# -----------------------------------------------------------------------------
# Runner helpers
# -----------------------------------------------------------------------------
def _prediction_result_to_df(pred: Mapping[str, Any], *, split: str) -> pd.DataFrame:
    y_true = np.asarray(pred.get("y_true"), dtype=np.int64)
    y_pred = np.asarray(pred.get("y_pred"), dtype=np.int64)
    probs = np.asarray(pred.get("probs"), dtype=np.float64)
    subject_ids = list(pred.get("subject_ids") or [])

    rows: list[dict[str, Any]] = []
    for i in range(len(y_true)):
        rec = {
            "split": split,
            "subject_id": subject_ids[i] if i < len(subject_ids) else f"{split}_{i}",
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
        }
        if probs.ndim == 2:
            for c in range(probs.shape[1]):
                rec[f"prob_{c}"] = float(probs[i, c])
        rows.append(rec)
    return pd.DataFrame(rows)


def _save_prediction_csv(pred: Mapping[str, Any], path: str | os.PathLike, *, split: str) -> pd.DataFrame:
    df = _prediction_result_to_df(pred, split=split)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _save_attention_artifacts(pred: Mapping[str, Any], path: str | os.PathLike) -> None:
    attn = pred.get("attention_weights", None)
    if attn is None:
        return
    arr = np.asarray(attn, dtype=np.float32)
    np.save(path, arr)


def run_hierarchical_macro_bank_experiment(
    cfg: HierBankExperimentConfig,
) -> dict[str, Any]:
    set_seed(int(cfg.train.seed), deterministic=True, benchmark=False)
    device = get_device(cfg.train.device)

    split_bags = prepare_subject_macro_bags(cfg.data)
    train_loader = DataLoader(
        SubjectMacroBankDataset(split_bags["train"]),
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_subject_macro_banks,
    )
    val_loader = DataLoader(
        SubjectMacroBankDataset(split_bags["val"]),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_subject_macro_banks,
    )
    test_loader = DataLoader(
        SubjectMacroBankDataset(split_bags["test"]),
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        num_workers=int(cfg.train.num_workers),
        collate_fn=collate_subject_macro_banks,
    )

    first_bag = split_bags["train"][0]
    num_nodes = int(first_bag.macro_node_features.shape[1])
    num_node_features = int(first_bag.macro_node_features.shape[2])
    num_bands = int(first_bag.macro_connectivity_bank.shape[1])
    num_metrics = int(first_bag.macro_connectivity_bank.shape[2])

    model = HierarchicalMacroConnectivityBankNet(
        num_nodes=num_nodes,
        num_node_features=num_node_features,
        num_bands=num_bands,
        num_metrics=num_metrics,
        cfg=cfg.model,
    ).to(device)

    run_dir = Path(cfg.train.output_root) / cfg.train.experiment_name
    ensure_dir(run_dir)

    optimizer = AdamW(model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay))
    trainer = Trainer(
        model,
        optimizer=optimizer,
        device=device,
        loss_name="cross_entropy",
        num_classes=int(cfg.model.num_classes),
        monitor=str(cfg.train.monitor),
        monitor_mode=str(cfg.train.monitor_mode),
        early_stopping_patience=int(cfg.train.patience),
        checkpoint_dir=run_dir,
        checkpoint_name="best_model.pt",
        save_best_only=True,
        forward_fn=lambda m, batch, trainer: m(batch),
        max_grad_norm=cfg.train.max_grad_norm,
        use_amp=bool(cfg.train.use_amp),
        verbose=True,
    )

    fit_out = trainer.fit(train_loader, val_loader, num_epochs=int(cfg.train.epochs), start_epoch=1)
    best_ckpt = fit_out.get("best_checkpoint_path", None)
    if best_ckpt:
        trainer.load_checkpoint(best_ckpt, load_optimizer=False, load_scheduler=False)

    preds = {
        "train": trainer.predict(train_loader, split_name="train", compute_loss=True, compute_metrics=True),
        "val": trainer.predict(val_loader, split_name="val", compute_loss=True, compute_metrics=True),
        "test": trainer.predict(test_loader, split_name="test", compute_loss=True, compute_metrics=True),
    }

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "best_epoch": fit_out.get("best_epoch"),
        "best_monitor_value": fit_out.get("best_monitor_value"),
        "best_checkpoint_path": fit_out.get("best_checkpoint_path"),
        "config": {
            "data": asdict(cfg.data),
            "model": asdict(cfg.model),
            "train": asdict(cfg.train),
        },
    }

    for split_name, pred in preds.items():
        pred_csv = _save_prediction_csv(pred, run_dir / f"{split_name}_subject_predictions.csv", split=split_name)
        _save_attention_artifacts(pred, run_dir / f"{split_name}_macro_attention.npy")
        metric_summary = summarize_classification_metrics(
            y_true=pred["y_true"],
            y_pred=pred["y_pred"],
            probs=pred["probs"],
            logits=pred["logits"],
            num_classes=int(cfg.model.num_classes),
        )
        summary[f"{split_name}_metrics"] = metric_summary
        summary[f"{split_name}_num_subjects"] = int(len(pred_csv))

    _save_json(summary, run_dir / "summary.json")
    return summary


def build_recommended_hierarchical_bank_config(
    *,
    dataset_path: str,
    h5_path: str,
    output_root: str,
    task: str = "dementia",
) -> HierBankExperimentConfig:
    return HierBankExperimentConfig(
        data=HierBankDataConfig(
            dataset_path=dataset_path,
            h5_path=h5_path,
            task=task,
            feature_families=("relative_band_power", "hjorth", "statistical"),
            connectivity_metrics=("coherence", "pli", "wpli", "pearson", "spearman"),
            bands=("delta", "theta", "alpha", "beta", "gamma"),
            macro_duration_sec=300.0,
            feature_reduce="mean",
            connectivity_reduce="mean",
            global_connectivity_reduce="mean",
            broadcast_nonband_metrics=True,
        ),
        model=HierBankModelConfig(
            num_classes=3,
            graph_hidden_dim=64,
            graph_emb_dim=128,
            graph_num_layers=2,
            graph_dropout=0.2,
            graph_topk=4,
            scorer_type="cnn",
            scorer_hidden_dim=64,
            band_attention_dim=64,
            macro_attention_dim=128,
            global_branch_hidden_dim=128,
            global_branch_emb_dim=128,
            fusion_hidden_dim=128,
            fusion_dropout=0.2,
        ),
        train=HierBankTrainConfig(
            seed=42,
            batch_size=8,
            num_workers=0,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=60,
            patience=15,
            monitor="balanced_accuracy",
            monitor_mode="max",
            device="cuda",
            max_grad_norm=5.0,
            use_amp=False,
            output_root=output_root,
            experiment_name="macro_hier_bank_dual_branch_v1",
        ),
    )


if __name__ == "__main__":
    example_dataset_path = "/mnt/data/anphan/CAUEEG/caueeg-dataset"
    example_h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    if os.path.exists(example_dataset_path) and os.path.exists(example_h5_path):
        cfg = build_recommended_hierarchical_bank_config(
            dataset_path=example_dataset_path,
            h5_path=example_h5_path,
            output_root="/home/anphan/Documents/EEG_Project/CAUEEG/results_pipeline/results_caueeg_hier_bank",
            task="dementia",
        )
        out = run_hierarchical_macro_bank_experiment(cfg)
        print(json.dumps(_jsonable(out), indent=2))
    else:
        print("Edit the example dataset/h5 paths before running this module.")
