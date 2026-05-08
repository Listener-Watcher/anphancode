"""
caueeg_softvote_baseline_main.py

Standalone CAUEEG baseline runner for the ORIGINAL non-graph ensemble style:

    branch 1: raw EEG segment model
    branch 2: feature-family MLP segment model, e.g. relative_band_power
    branch 3: feature-family MLP segment model, e.g. hjorth

Each branch is trained as an independent segment-level classifier with its own
optimizer/checkpoint. At inference, segment probabilities are averaged:

    p_ensemble(segment) = mean_b p_b(segment)

Then subject/recording prediction is computed by soft voting over all segments:

    p_subject = mean_segments p_ensemble(segment)
    y_subject = argmax(p_subject)

There is NO embedding fusion, NO shared encoder, and NO MIL pooling in this file.

Example: original-style three-branch baseline
--------------------------------------------
python caueeg_softvote_baseline_main.py \
  --dataset_path /home/anphan/Downloads/caueeg-dataset \
  --task dementia-no-overlap \
  --file_format feather \
  --out_h5 /home/anphan/Documents/caueeg_softvote_baseline.h5 \
  --branches raw_eeg relative_band_power hjorth \
  --segment_train_policy label_aware_k \
  --base_k 8 \
  --seeds 15 42 100 \
  --rebuild_h5

Feature-only three-branch baseline
----------------------------------
python caueeg_softvote_baseline_main.py \
  --dataset_path /home/anphan/Downloads/caueeg-dataset \
  --task dementia-no-overlap \
  --file_format feather \
  --out_h5 /home/anphan/Documents/caueeg_softvote_baseline.h5 \
  --branches relative_band_power hjorth statistical \
  --segment_train_policy label_aware_k \
  --base_k 8 \
  --seeds 15 42 100
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
try:
    from caueeg_loader_min import load_caueeg_task_datasets
except Exception:  # pragma: no cover
    try:
        from caueeg_script import load_caueeg_task_datasets
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import load_caueeg_task_datasets. Put this file in the same "
            "folder as caueeg_loader_min.py, or adjust the import."
        ) from e

try:
    from master_builder import build_master_eeg_dataset
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import build_master_eeg_dataset from master_builder.py. "
        "Put this file in the same folder as master_builder.py."
    ) from e

try:
    from mil_full_std import load_h5_payload_for_subjects
except Exception as e:  # pragma: no cover
    raise ImportError(
        "Could not import load_h5_payload_for_subjects from mil_full_std.py. "
        "Put this file in the same folder as mil_full_std.py."
    ) from e

try:
    from utils_all import set_global_seed as project_set_global_seed
except Exception:  # pragma: no cover
    project_set_global_seed = None


# ---------------------------------------------------------------------
# CAUEEG constants aligned with your LinkX adapter
# ---------------------------------------------------------------------
CAUEEG_EEG19 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4",
    "T6", "FZ", "CZ", "PZ",
]

SFREQ = 200.0
DEFAULT_CROP_LEN = 2000       # 10 seconds at 200 Hz
DEFAULT_LATENCY = 2000        # skip first 10 seconds
DEFAULT_OVERLAP = 0.5


# ---------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------
def set_seed(seed: int) -> None:
    seed = int(seed)
    if project_set_global_seed is not None:
        project_set_global_seed(seed)
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def stable_int_from_string(x: str) -> int:
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)


def parse_str_list(values: Optional[Sequence[str]]) -> List[str]:
    if values is None:
        return []
    out: List[str] = []
    for v in values:
        for item in str(v).replace(",", " ").split():
            item = item.strip()
            if item:
                out.append(item)
    return out


def parse_bad_ids(values: Optional[Sequence[str]]) -> set[str]:
    return set(parse_str_list(values))


def infer_feature_families_from_branches(branches: Sequence[str]) -> List[str]:
    return [b for b in branches if b != "raw_eeg"]


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------
def segment_recording(
    signal: np.ndarray,
    crop_len: int = DEFAULT_CROP_LEN,
    overlap: float = DEFAULT_OVERLAP,
    latency: int = DEFAULT_LATENCY,
) -> Tuple[List[np.ndarray], List[int]]:
    """signal: [21,T] -> list of [19,crop_len] windows."""
    x = np.asarray(signal, dtype=np.float32)[:19]
    step = int(round(crop_len * (1.0 - float(overlap))))
    step = max(step, 1)

    total_len = x.shape[-1]
    starts = list(range(int(latency), total_len - int(crop_len) + 1, step))
    windows = [x[:, s:s + int(crop_len)].astype(np.float32, copy=False) for s in starts]
    return windows, starts


def dataset_to_subject_records(
    dataset: Dataset,
    split_prefix: str,
    *,
    crop_len: int,
    overlap: float,
    latency: int,
    bad_ids: Optional[set[str]] = None,
) -> Tuple[List[dict], List[str]]:
    """Convert CAUEEG split into records accepted by build_master_eeg_dataset()."""
    bad_ids = set() if bad_ids is None else set(bad_ids)
    records: List[dict] = []
    subject_ids: List[str] = []

    for sample in dataset:
        serial = str(sample["serial"])
        if serial in bad_ids:
            continue

        signal = sample["signal"]
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(
            signal,
            crop_len=crop_len,
            overlap=overlap,
            latency=latency,
        )
        if len(windows) == 0:
            continue

        sid = f"{split_prefix}_{serial}"
        records.append({
            "subject_id": sid,
            "label": label,
            "class_id": label,
            "sampling_rate": SFREQ,
            "channel_names": CAUEEG_EEG19,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": list(range(len(windows))),
            "recording_info": {
                "serial": serial,
                "split": split_prefix,
                "age": age,
            },
        })
        subject_ids.append(sid)

    return records, subject_ids


def prepare_h5_and_payload(args) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], List[str], int]:
    branches = list(args.branches)
    feature_families = infer_feature_families_from_branches(branches)
    use_raw_eeg = "raw_eeg" in branches
    bad_ids = parse_bad_ids(args.bad_ids)

    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=args.dataset_path,
        task=args.task,
        load_event=False,
        file_format=args.file_format,
        transform=None,
        verbose=False,
    )

    train_records, train_ids = dataset_to_subject_records(
        train_set,
        "train",
        crop_len=args.crop_len,
        overlap=args.overlap,
        latency=args.latency,
        bad_ids=bad_ids,
    )
    val_records, val_ids = dataset_to_subject_records(
        val_set,
        "val",
        crop_len=args.crop_len,
        overlap=args.overlap,
        latency=args.latency,
        bad_ids=bad_ids,
    )
    test_records, test_ids = dataset_to_subject_records(
        test_set,
        "test",
        crop_len=args.crop_len,
        overlap=args.overlap,
        latency=args.latency,
        bad_ids=bad_ids,
    )

    all_records = train_records + val_records + test_records
    all_ids = train_ids + val_ids + test_ids
    if len(all_records) == 0:
        raise RuntimeError("No valid CAUEEG records after filtering/windowing.")

    num_classes = len(sorted({int(r["label"]) for r in all_records}))

    out_h5 = Path(args.out_h5)
    need_build = bool(args.rebuild_h5) or (not out_h5.is_file())
    if need_build:
        out_h5.parent.mkdir(parents=True, exist_ok=True)
        build_metric = str(args.build_connectivity_metric).lower()
        connectivity_metrics = [] if build_metric in {"", "none", "null"} else [args.build_connectivity_metric]

        print(f"[H5] Building: {out_h5}")
        print(f"[H5] feature_families={feature_families}")
        print(f"[H5] connectivity_metrics_for_build={connectivity_metrics}")

        try:
            build_master_eeg_dataset(
                subject_records=all_records,
                output_h5_path=str(out_h5),
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                overwrite=True,
                skip_bad_segments=False,
                target_sampling_rate=None,
                qc_input_unit="auto",
            )
        except Exception:
            if len(connectivity_metrics) == 0:
                print("[H5] Empty connectivity_metrics failed. Retrying with ['pearson'] for builder compatibility.")
                build_master_eeg_dataset(
                    subject_records=all_records,
                    output_h5_path=str(out_h5),
                    feature_families=feature_families,
                    connectivity_metrics=["pearson"],
                    overwrite=True,
                    skip_bad_segments=False,
                    target_sampling_rate=None,
                    qc_input_unit="auto",
                )
            else:
                raise
    else:
        print(f"[H5] Reusing: {out_h5}")

    payload = load_h5_payload_for_subjects(
        h5_path=str(out_h5),
        subject_ids=all_ids,
        feature_families=feature_families,
        connectivity_metrics=[],
        connectivity_band=None,
        load_raw_for_alignment=bool(use_raw_eeg),
        load_bad_segment_flag=False,
    )

    missing = [sid for sid in all_ids if sid not in payload]
    if missing:
        raise KeyError(
            f"H5 payload is missing {len(missing)} requested subject ids. "
            f"Examples: {missing[:5]}. Rebuild with --rebuild_h5 if ids/features changed."
        )

    return payload, train_ids, val_ids, test_ids, num_classes


# ---------------------------------------------------------------------
# Convert H5 payload into arrays for separate branches
# ---------------------------------------------------------------------
def zscore_per_segment_node_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: [W,C,F]. Z-score over channels, separately for each window and feature."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def zscore_raw_eeg_per_window_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: [W,C,T]. Z-score over time within each window/channel."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


@dataclass
class SubjectArrays:
    subject_id: str
    label: int
    feature_by_family: Dict[str, np.ndarray]  # family -> [W, C*F_family]
    raw_eeg: Optional[np.ndarray]             # [W,1,C,T]
    segment_ids: np.ndarray                   # [W]
    start_samples: np.ndarray                 # [W]


def payload_to_subject_arrays(
    payload: Dict[str, Dict[str, Any]],
    subject_ids: Sequence[str],
    branches: Sequence[str],
    *,
    standardize_features: bool,
    standardize_raw_eeg: bool,
) -> List[SubjectArrays]:
    feature_families = infer_feature_families_from_branches(branches)
    use_raw_eeg = "raw_eeg" in branches
    out: List[SubjectArrays] = []

    for sid in subject_ids:
        subj = payload[sid]
        label = int(subj["label"])
        segment_ids = np.asarray(subj["segment_id"], dtype=np.int64)
        start_samples = np.asarray(subj["start_sample"], dtype=np.int64)
        n_windows = len(segment_ids)

        feature_by_family: Dict[str, np.ndarray] = {}
        for fam in feature_families:
            if fam not in subj.get("features", {}):
                raise KeyError(f"Subject {sid} is missing feature family {fam!r} in H5 payload.")
            xf = np.asarray(subj["features"][fam], dtype=np.float32)  # [W,C,F]
            if xf.ndim != 3:
                raise ValueError(f"Feature family {fam!r} for {sid} must be [W,C,F], got {xf.shape}")
            if xf.shape[0] != n_windows:
                raise ValueError(f"Feature/window mismatch for {sid}/{fam}: {xf.shape[0]} vs {n_windows}")
            if standardize_features:
                xf = zscore_per_segment_node_features(xf)
            feature_by_family[fam] = xf.reshape(xf.shape[0], -1).astype(np.float32)

        raw = None
        if use_raw_eeg:
            if subj.get("raw_eeg", None) is None:
                raise KeyError(
                    f"Subject {sid} has no raw_eeg in payload. Use branch raw_eeg only with raw windows in the H5/payload."
                )
            raw0 = np.asarray(subj["raw_eeg"], dtype=np.float32)  # [W,C,T]
            if raw0.shape[0] != n_windows:
                raise ValueError(f"Raw/window mismatch for {sid}: {raw0.shape[0]} vs {n_windows}")
            if standardize_raw_eeg:
                raw0 = zscore_raw_eeg_per_window_channel(raw0)
            raw = raw0[:, None, :, :].astype(np.float32)

        out.append(SubjectArrays(
            subject_id=str(sid),
            label=label,
            feature_by_family=feature_by_family,
            raw_eeg=raw,
            segment_ids=segment_ids,
            start_samples=start_samples,
        ))

    return out


# ---------------------------------------------------------------------
# Segment dataset for one independent branch
# ---------------------------------------------------------------------
def compute_label_aware_k(subjects: Sequence[SubjectArrays], base_k: int, max_k: Optional[int]) -> Dict[int, int]:
    label_to_subjects: Dict[int, List[str]] = defaultdict(list)
    for s in subjects:
        label_to_subjects[int(s.label)].append(s.subject_id)
    n_subjects_per_label = {label: len(v) for label, v in label_to_subjects.items()}
    max_subjects = max(n_subjects_per_label.values())
    target_segments_per_class = max_subjects * int(base_k)

    out: Dict[int, int] = {}
    for label, n_subj in n_subjects_per_label.items():
        k = int(math.ceil(target_segments_per_class / n_subj))
        if max_k is not None:
            k = min(k, int(max_k))
        out[int(label)] = k
    return out


class BranchSegmentDataset(Dataset):
    """One item = one segment for one specific branch."""

    def __init__(
        self,
        subjects: Sequence[SubjectArrays],
        branch: str,
        *,
        train: bool,
        train_policy: str,
        base_k: Optional[int],
        max_k_per_subject: Optional[int],
        seed: int,
    ):
        self.subjects = list(subjects)
        self.branch = str(branch)
        self.train = bool(train)
        self.train_policy = str(train_policy).lower()
        self.base_k = None if base_k is None or int(base_k) <= 0 else int(base_k)
        self.max_k_per_subject = max_k_per_subject
        self.seed = int(seed)
        self.epoch = 0
        self.subject_by_id = {s.subject_id: s for s in self.subjects}
        if len(self.subjects) == 0:
            raise ValueError("subjects is empty")

        self.k_by_label = None
        if self.train and self.train_policy == "label_aware_k":
            if self.base_k is None:
                raise ValueError("base_k must be > 0 for train_policy='label_aware_k'")
            self.k_by_label = compute_label_aware_k(self.subjects, self.base_k, self.max_k_per_subject)

        self.rows: List[Tuple[str, int]] = []
        self._refresh_rows()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        if self.train and self.train_policy in {"random_k", "label_aware_k"}:
            self._refresh_rows()

    def _sample_indices_for_subject(self, s: SubjectArrays) -> List[int]:
        n = len(s.segment_ids)
        if n == 0:
            return []
        if not self.train or self.train_policy == "all":
            return list(range(n))

        if self.train_policy == "random_k":
            if self.base_k is None:
                raise ValueError("base_k must be > 0 for train_policy='random_k'")
            k = self.base_k
        elif self.train_policy == "label_aware_k":
            assert self.k_by_label is not None
            k = self.k_by_label[int(s.label)]
        else:
            raise ValueError("train_policy must be one of: all, random_k, label_aware_k")

        subject_seed = self.seed + 1000003 * self.epoch + stable_int_from_string(s.subject_id)
        rng = random.Random(subject_seed)
        if n >= k:
            return rng.sample(range(n), k)
        return list(range(n)) + [rng.randrange(n) for _ in range(k - n)]

    def _refresh_rows(self) -> None:
        rows: List[Tuple[str, int]] = []
        for s in self.subjects:
            for idx in self._sample_indices_for_subject(s):
                rows.append((s.subject_id, int(idx)))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def _get_x(self, s: SubjectArrays, w: int) -> Tensor:
        if self.branch == "raw_eeg":
            if s.raw_eeg is None:
                raise KeyError("raw_eeg branch requested, but subject has no raw_eeg array")
            return torch.tensor(s.raw_eeg[w], dtype=torch.float32)
        if self.branch not in s.feature_by_family:
            raise KeyError(f"Feature branch {self.branch!r} not found for subject {s.subject_id}")
        return torch.tensor(s.feature_by_family[self.branch][w], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, w = self.rows[int(idx)]
        s = self.subject_by_id[sid]
        return {
            "x": self._get_x(s, w),
            "label": torch.tensor(int(s.label), dtype=torch.long),
            "subject_id": s.subject_id,
            "segment_id": torch.tensor(int(s.segment_ids[w]), dtype=torch.long),
            "start_sample": torch.tensor(int(s.start_samples[w]), dtype=torch.long),
        }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Independent branch models
# ---------------------------------------------------------------------
class FeatureBranchMLP(nn.Module):
    """
    Feature-family branch, matching the original baseline spirit:
        flatten feature -> Linear(512) -> ELU -> Linear(128) -> ELU -> Dropout -> classifier.
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 512),
            nn.ELU(),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, int(num_classes)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=int(emb_size),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int, expansion: int = 4, dropout: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(emb_size), int(expansion * emb_size)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(expansion * emb_size), int(emb_size)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.block = nn.Sequential(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(int(emb_size)),
                MultiHeadAttentionBlock(int(emb_size), int(num_heads), float(dropout)),
                nn.Dropout(float(dropout)),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(int(emb_size)),
                FeedForwardBlock(int(emb_size), expansion=4, dropout=float(dropout)),
                nn.Dropout(float(dropout)),
            )),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class RawEEGTransformerClassifier(nn.Module):
    """
    Raw EEG branch based on the uploaded baseline's EEGTransformer idea:
        Conv temporal patching -> channel convolution -> token projection -> transformer -> FC classifier.

    Input: [B, 1, 19, crop_len]
    """

    def __init__(
        self,
        num_classes: int,
        *,
        crop_len: int,
        n_channels: int = 19,
        emb_size: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.crop_len = int(crop_len)
        self.emb_size = int(emb_size)

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 20), stride=(1, 10)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(self.n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 40), stride=(1, 20)),
            nn.Dropout(float(dropout)),
        )
        self.projection = nn.Conv2d(64, self.emb_size, kernel_size=(1, 1), stride=(1, 1))
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(self.emb_size, num_heads=num_heads, dropout=dropout)
            for _ in range(int(depth))
        ])

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.crop_len)
            tokens = self._tokens(dummy)
            fc_dim = int(tokens.reshape(1, -1).shape[1])

        self.classifier = nn.Sequential(
            nn.Linear(fc_dim, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, int(num_classes)),
        )

    def _tokens(self, x: Tensor) -> Tensor:
        h = self.shallownet(x)                # [B,64,1,L]
        h = self.projection(h)                # [B,E,1,L]
        h = h.squeeze(2).transpose(1, 2)      # [B,L,E]
        return h.contiguous()

    def forward(self, x: Tensor) -> Tensor:
        h = self._tokens(x)
        h = self.transformer(h)
        h = h.contiguous().view(h.size(0), -1)
        return self.classifier(h)


# ---------------------------------------------------------------------
# Metrics and prediction utilities
# ---------------------------------------------------------------------
def safe_probs_from_logits(logits: Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def nll_loss_from_probs(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.clip(probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    try:
        return float(log_loss(y_true, probs, labels=list(range(num_classes))))
    except Exception:
        idx = np.arange(len(y_true))
        return float(-np.mean(np.log(probs[idx, y_true] + 1e-8)))


def compute_metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "loss": nll_loss_from_probs(y_true, probs, num_classes),
        "conf_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }

    if num_classes == 2:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            out["roc_auc"] = float("nan")
    else:
        try:
            out["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except Exception:
            out["roc_auc_ovr_macro"] = float("nan")
    return out


def save_probs_dataframe(
    df: pd.DataFrame,
    save_path: str,
    *,
    probs: np.ndarray,
    prefix: str = "prob",
) -> pd.DataFrame:
    out = df.copy()
    for c in range(probs.shape[1]):
        out[f"{prefix}_{c}"] = probs[:, c]
    out.to_csv(save_path, index=False)
    return out


@torch.no_grad()
def collect_branch_segment_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    branch: str,
) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch["x"])
        probs = safe_probs_from_logits(logits)
        preds = probs.argmax(axis=1)
        labels = batch["label"].detach().cpu().numpy().astype(int)
        seg_ids = batch["segment_id"].detach().cpu().numpy().astype(int)
        starts = batch["start_sample"].detach().cpu().numpy().astype(int)
        subject_ids = list(batch["subject_id"])

        for i in range(len(labels)):
            row = {
                "branch": branch,
                "subject_id": subject_ids[i],
                "segment_id": int(seg_ids[i]),
                "start_sample": int(starts[i]),
                "true_label": int(labels[i]),
                "pred_label": int(preds[i]),
            }
            for c in range(num_classes):
                row[f"prob_{c}"] = float(probs[i, c])
            rows.append(row)
    return pd.DataFrame(rows)


def soft_vote_segment_predictions(branch_dfs: Dict[str, pd.DataFrame], num_classes: int) -> pd.DataFrame:
    if len(branch_dfs) == 0:
        raise ValueError("No branch predictions provided.")

    keys = ["subject_id", "segment_id", "start_sample", "true_label"]
    branches = list(branch_dfs.keys())
    base = branch_dfs[branches[0]][keys].copy()
    base = base.sort_values(keys).reset_index(drop=True)

    prob_stack = []
    for branch in branches:
        df = branch_dfs[branch].copy().sort_values(keys).reset_index(drop=True)
        if len(df) != len(base):
            raise ValueError(f"Branch {branch!r} prediction count mismatch: {len(df)} vs {len(base)}")
        if not df[keys].equals(base[keys]):
            mismatch = pd.concat([base[keys].head(), df[keys].head()], axis=1)
            raise ValueError(f"Branch {branch!r} segment keys do not align. Example:\n{mismatch}")
        prob_stack.append(df[[f"prob_{c}" for c in range(num_classes)]].to_numpy(dtype=np.float64))

    avg_prob = np.mean(np.stack(prob_stack, axis=0), axis=0)
    avg_prob = np.clip(avg_prob, 1e-8, None)
    avg_prob = avg_prob / avg_prob.sum(axis=1, keepdims=True)

    out = base.copy()
    for c in range(num_classes):
        out[f"prob_{c}"] = avg_prob[:, c]
    out["pred_label"] = avg_prob.argmax(axis=1).astype(int)
    out["num_branches"] = len(branches)
    out["branches"] = "+".join(branches)
    return out


def aggregate_segments_to_subjects(segment_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    for sid, sdf in segment_df.groupby("subject_id", sort=True):
        probs = sdf[prob_cols].to_numpy(dtype=np.float64)
        mean_prob = probs.mean(axis=0)
        mean_prob = np.clip(mean_prob, 1e-8, None)
        mean_prob = mean_prob / mean_prob.sum()
        y = int(sdf["true_label"].iloc[0])
        row = {
            "subject_id": sid,
            "true_label": y,
            "pred_label": int(mean_prob.argmax()),
            "num_segments": int(len(sdf)),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(mean_prob[c])
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_from_prediction_df(df: pd.DataFrame, num_classes: int) -> Dict[str, Any]:
    y = df["true_label"].to_numpy(dtype=int)
    probs = df[[f"prob_{c}" for c in range(num_classes)]].to_numpy(dtype=np.float64)
    return compute_metrics_from_probs(y, probs, num_classes)


# ---------------------------------------------------------------------
# Training one independent branch
# ---------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience: int, start_epoch: int, min_delta: float, save_path: str):
        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.save_path = save_path
        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.counter = 0
        self.should_stop = False

    def step(self, model: nn.Module, epoch: int, val_loss: float, extra: Optional[dict] = None) -> bool:
        val_loss = float(val_loss)
        improved = val_loss < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.best_epoch = int(epoch)
            payload = {
                "epoch": int(epoch),
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "val_loss": float(val_loss),
            }
            if extra:
                payload.update(extra)
            torch.save(payload, self.save_path)
            if epoch >= self.start_epoch:
                self.counter = 0
        else:
            if epoch >= self.start_epoch:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
        return self.should_stop


def get_branch_input_dim(subjects: Sequence[SubjectArrays], branch: str) -> int:
    if len(subjects) == 0:
        raise ValueError("subjects is empty")
    s = subjects[0]
    if branch == "raw_eeg":
        return -1
    if branch not in s.feature_by_family:
        raise KeyError(f"Branch {branch!r} is not available in feature_by_family")
    return int(s.feature_by_family[branch].shape[1])


def build_branch_model(args, branch: str, train_subjects: Sequence[SubjectArrays], num_classes: int) -> nn.Module:
    if branch == "raw_eeg":
        return RawEEGTransformerClassifier(
            num_classes=num_classes,
            crop_len=args.crop_len,
            n_channels=19,
            emb_size=args.raw_emb_size,
            depth=args.raw_depth,
            num_heads=args.raw_heads,
            dropout=args.raw_dropout,
        )
    input_dim = get_branch_input_dim(train_subjects, branch)
    return FeatureBranchMLP(input_dim=input_dim, num_classes=num_classes, dropout=args.feature_dropout)


def compute_class_weights_from_subjects(subjects: Sequence[SubjectArrays], num_classes: int, device: torch.device) -> Optional[Tensor]:
    labels = np.asarray([int(s.label) for s in subjects], dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        return None
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_branch(
    args,
    branch: str,
    train_subjects: Sequence[SubjectArrays],
    val_subjects: Sequence[SubjectArrays],
    *,
    num_classes: int,
    seed_dir: str,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    branch_dir = os.path.join(seed_dir, f"branch_{branch}")
    os.makedirs(branch_dir, exist_ok=True)

    model = build_branch_model(args, branch, train_subjects, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights_from_subjects(train_subjects, num_classes, device) if args.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_dataset = BranchSegmentDataset(
        train_subjects,
        branch,
        train=True,
        train_policy=args.segment_train_policy,
        base_k=args.base_k,
        max_k_per_subject=args.max_k_per_subject,
        seed=seed,
    )
    val_dataset = BranchSegmentDataset(
        val_subjects,
        branch,
        train=False,
        train_policy="all",
        base_k=None,
        max_k_per_subject=None,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    ckpt_path = os.path.join(branch_dir, "best_branch_model.pt")
    stopper = EarlyStopping(
        patience=args.patience,
        start_epoch=args.start_epoch,
        min_delta=args.min_delta,
        save_path=ckpt_path,
    )

    history: List[Dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        train_losses: List[float] = []
        train_correct = 0
        train_n = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["x"])
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
            train_correct += int((logits.argmax(dim=1) == batch["label"]).sum().detach().cpu().item())
            train_n += int(batch["label"].numel())

        model.eval()
        val_losses: List[float] = []
        val_probs: List[np.ndarray] = []
        val_labels: List[np.ndarray] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                logits = model(batch["x"])
                loss = criterion(logits, batch["label"])
                val_losses.append(float(loss.detach().cpu().item()))
                val_probs.append(safe_probs_from_logits(logits))
                val_labels.append(batch["label"].detach().cpu().numpy().astype(int))

        val_probs_np = np.concatenate(val_probs, axis=0)
        val_labels_np = np.concatenate(val_labels, axis=0)
        val_metrics = compute_metrics_from_probs(val_labels_np, val_probs_np, num_classes)
        val_loss = float(np.mean(val_losses))
        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        train_acc = float(train_correct / max(train_n, 1))

        row = {
            "epoch": int(epoch),
            "branch": branch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"[{branch}] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        stop = stopper.step(
            model,
            epoch,
            val_loss,
            extra={
                "branch": branch,
                "val_accuracy": float(val_metrics["accuracy"]),
                "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "val_macro_f1": float(val_metrics["macro_f1"]),
            },
        )
        if stop:
            print(f"[{branch}] Early stopping at epoch {epoch}; best_epoch={stopper.best_epoch}, best_val_loss={stopper.best_val_loss:.6f}")
            break

    pd.DataFrame(history).to_csv(os.path.join(branch_dir, "training_history.csv"), index=False)

    if not os.path.isfile(ckpt_path):
        raise RuntimeError(f"No checkpoint was saved for branch {branch!r}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return {
        "branch": branch,
        "model": model,
        "checkpoint_path": ckpt_path,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss": float(ckpt.get("val_loss", float("nan"))),
        "branch_dir": branch_dir,
    }


# ---------------------------------------------------------------------
# One seed run: train all branches independently, then soft vote
# ---------------------------------------------------------------------
def class_names_for_task(task: str, num_classes: int) -> List[str]:
    task = str(task).lower()
    if task.startswith("abnormal"):
        return ["normal", "abnormal"]
    if task.startswith("dementia"):
        return ["normal", "mci", "dementia"]
    return [f"class_{i}" for i in range(num_classes)]


def save_config(args, seed_dir: str, seed: int) -> None:
    cfg = vars(args).copy()
    cfg["seed"] = int(seed)
    with open(os.path.join(seed_dir, "config.json"), "w") as f:
        json.dump(make_jsonable(cfg), f, indent=2)


def run_one_seed(
    args,
    payload,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    test_ids: Sequence[str],
    *,
    num_classes: int,
    seed: int,
    root_run_dir: str,
) -> Dict[str, Any]:
    set_seed(seed)
    seed_dir = os.path.join(root_run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    save_config(args, seed_dir, seed)

    train_subjects = payload_to_subject_arrays(
        payload,
        train_ids,
        args.branches,
        standardize_features=args.standardize_features,
        standardize_raw_eeg=args.standardize_raw_eeg,
    )
    val_subjects = payload_to_subject_arrays(
        payload,
        val_ids,
        args.branches,
        standardize_features=args.standardize_features,
        standardize_raw_eeg=args.standardize_raw_eeg,
    )
    test_subjects = payload_to_subject_arrays(
        payload,
        test_ids,
        args.branches,
        standardize_features=args.standardize_features,
        standardize_raw_eeg=args.standardize_raw_eeg,
    )

    device = torch.device(args.device if torch.cuda.is_available() or str(args.device).startswith("cpu") else "cpu")
    print(f"[seed={seed}] device={device}")

    branch_artifacts: Dict[str, Dict[str, Any]] = {}
    for branch in args.branches:
        print(f"\n========== Training independent branch: {branch} ==========")
        branch_artifacts[branch] = train_one_branch(
            args,
            branch,
            train_subjects,
            val_subjects,
            num_classes=num_classes,
            seed_dir=seed_dir,
            seed=seed,
            device=device,
        )

    # Inference on all val/test segments for every branch.
    branch_val_dfs: Dict[str, pd.DataFrame] = {}
    branch_test_dfs: Dict[str, pd.DataFrame] = {}

    for branch, artifact in branch_artifacts.items():
        model = artifact["model"]
        val_dataset = BranchSegmentDataset(
            val_subjects,
            branch,
            train=False,
            train_policy="all",
            base_k=None,
            max_k_per_subject=None,
            seed=seed,
        )
        test_dataset = BranchSegmentDataset(
            test_subjects,
            branch,
            train=False,
            train_policy="all",
            base_k=None,
            max_k_per_subject=None,
            seed=seed,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        val_df = collect_branch_segment_predictions(model, val_loader, device=device, num_classes=num_classes, branch=branch)
        test_df = collect_branch_segment_predictions(model, test_loader, device=device, num_classes=num_classes, branch=branch)

        val_df.to_csv(os.path.join(artifact["branch_dir"], "val_segment_predictions.csv"), index=False)
        test_df.to_csv(os.path.join(artifact["branch_dir"], "test_segment_predictions.csv"), index=False)

        branch_val_dfs[branch] = val_df
        branch_test_dfs[branch] = test_df

    val_segment_ens = soft_vote_segment_predictions(branch_val_dfs, num_classes)
    test_segment_ens = soft_vote_segment_predictions(branch_test_dfs, num_classes)

    val_subject_ens = aggregate_segments_to_subjects(val_segment_ens, num_classes)
    test_subject_ens = aggregate_segments_to_subjects(test_segment_ens, num_classes)

    val_segment_ens.to_csv(os.path.join(seed_dir, "val_segment_predictions_softvote.csv"), index=False)
    test_segment_ens.to_csv(os.path.join(seed_dir, "test_segment_predictions_softvote.csv"), index=False)
    val_subject_ens.to_csv(os.path.join(seed_dir, "val_subject_predictions_softvote.csv"), index=False)
    test_subject_ens.to_csv(os.path.join(seed_dir, "test_subject_predictions_softvote.csv"), index=False)

    val_subject_metrics = metrics_from_prediction_df(val_subject_ens, num_classes)
    test_subject_metrics = metrics_from_prediction_df(test_subject_ens, num_classes)
    val_segment_metrics = metrics_from_prediction_df(val_segment_ens, num_classes)
    test_segment_metrics = metrics_from_prediction_df(test_segment_ens, num_classes)

    summary = {
        "baseline_type": "three_separate_model_softvote",
        "training_approach": "segment_ce_independent_branches_softvote_subject",
        "branches": list(args.branches),
        "feature_families": infer_feature_families_from_branches(args.branches),
        "use_raw_eeg": "raw_eeg" in args.branches,
        "task": args.task,
        "num_classes": int(num_classes),
        "class_names": class_names_for_task(args.task, num_classes),
        "seed": int(seed),
        "segment_train_policy": args.segment_train_policy,
        "base_k": args.base_k,
        "max_k_per_subject": args.max_k_per_subject,
        "crop_len": args.crop_len,
        "overlap": args.overlap,
        "latency": args.latency,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "start_epoch": args.start_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "standardize_features": bool(args.standardize_features),
        "standardize_raw_eeg": bool(args.standardize_raw_eeg),
        "accuracy": float(test_subject_metrics["accuracy"]),
        "balanced_accuracy": float(test_subject_metrics["balanced_accuracy"]),
        "macro_f1": float(test_subject_metrics["macro_f1"]),
        "loss": float(test_subject_metrics["loss"]),
        "val_accuracy": float(val_subject_metrics["accuracy"]),
        "val_balanced_accuracy": float(val_subject_metrics["balanced_accuracy"]),
        "val_macro_f1": float(val_subject_metrics["macro_f1"]),
        "val_loss": float(val_subject_metrics["loss"]),
        "segment_accuracy": float(test_segment_metrics["accuracy"]),
        "segment_balanced_accuracy": float(test_segment_metrics["balanced_accuracy"]),
        "segment_macro_f1": float(test_segment_metrics["macro_f1"]),
        "val_segment_accuracy": float(val_segment_metrics["accuracy"]),
        "val_segment_balanced_accuracy": float(val_segment_metrics["balanced_accuracy"]),
        "val_segment_macro_f1": float(val_segment_metrics["macro_f1"]),
        "confusion_matrix": test_subject_metrics["conf_matrix"],
        "segment_confusion_matrix": test_segment_metrics["conf_matrix"],
        "run_dir": seed_dir,
    }

    pd.DataFrame([normalize_summary_row(summary)]).to_csv(os.path.join(seed_dir, "summary_test.csv"), index=False)
    with open(os.path.join(seed_dir, "summary_test.json"), "w") as f:
        json.dump(make_jsonable(summary), f, indent=2)

    print("\n[Subject-level test soft-vote metrics]")
    print(json.dumps(make_jsonable({k: summary[k] for k in ["accuracy", "balanced_accuracy", "macro_f1", "loss"]}), indent=2))

    return summary


# ---------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------
def make_jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {k: make_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [make_jsonable(v) for v in x]
    return x


def normalize_summary_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k in {"confusion_matrix", "segment_confusion_matrix", "class_names", "branches", "feature_families"}:
            out[k if k not in {"confusion_matrix", "segment_confusion_matrix"} else f"{k}_json"] = json.dumps(make_jsonable(v))
        elif isinstance(v, (list, tuple, dict)):
            out[k] = json.dumps(make_jsonable(v))
        else:
            out[k] = make_jsonable(v)
    return out


def save_seed_aggregation(summary_rows: Sequence[Dict[str, Any]], output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    rows = [normalize_summary_row(r) for r in summary_rows]
    df = pd.DataFrame(rows)
    raw_path = os.path.join(output_dir, "all_seed_results.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = [
        c for c in [
            "accuracy", "balanced_accuracy", "macro_f1", "loss",
            "segment_accuracy", "segment_balanced_accuracy", "segment_macro_f1",
            "val_accuracy", "val_balanced_accuracy", "val_macro_f1", "val_loss",
        ] if c in df.columns
    ]
    variant_cols = [
        "baseline_type", "training_approach", "branches", "feature_families", "use_raw_eeg",
        "segment_train_policy", "base_k", "crop_len", "overlap", "latency",
        "batch_size", "epochs", "patience", "start_epoch", "lr", "weight_decay", "task",
    ]
    variant_cols = [c for c in variant_cols if c in df.columns]

    agg = df.groupby(variant_cols, dropna=False)[metric_cols].agg(["mean", "std", "min", "max", "count"]).reset_index()
    agg.columns = [col[0] if col[1] == "" else f"{col[0]}_{col[1]}" for col in agg.columns]

    for m in metric_cols:
        mean_col, std_col = f"{m}_mean", f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_mean_std"] = agg.apply(
                lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}" if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
                axis=1,
            )

    agg_path = os.path.join(output_dir, "aggregate_seed_results.csv")
    agg.to_csv(agg_path, index=False)

    if "confusion_matrix_json" in df.columns:
        cms = [np.asarray(json.loads(s), dtype=float) for s in df["confusion_matrix_json"].dropna()]
        if len(cms) > 0:
            stack = np.stack(cms, axis=0)
            cm_info = {
                "num_seeds": int(len(cms)),
                "confusion_matrix_sum": stack.sum(axis=0).astype(int).tolist(),
                "confusion_matrix_mean": stack.mean(axis=0).tolist(),
                "confusion_matrix_std": stack.std(axis=0, ddof=1).tolist() if len(cms) > 1 else np.zeros_like(stack[0]).tolist(),
            }
            with open(os.path.join(output_dir, "aggregate_confusion_matrix.json"), "w") as f:
                json.dump(cm_info, f, indent=2)

    print(f"Saved per-seed results: {raw_path}")
    print(f"Saved aggregate results: {agg_path}")
    return df, agg


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CAUEEG three-separate-model soft-voting baseline. No fusion, no MIL.")

    # Data
    p.add_argument("--dataset_path", type=str, default="/home/anphan/Downloads/caueeg-dataset/")
    p.add_argument("--task", type=str, default="dementia-no-overlap", choices=["abnormal", "dementia", "abnormal-no-overlap", "dementia-no-overlap"])
    p.add_argument("--file_format", type=str, default="edf", choices=["edf", "feather", "memmap", "np"])
    p.add_argument("--out_h5", type=str, default="/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5")
    p.add_argument("--rebuild_h5", action="store_true")
    p.add_argument("--build_connectivity_metric", type=str, default="none")
    p.add_argument("--bad_ids", nargs="*", default=["00587", "00781", "01301"])
    p.add_argument("--crop_len", type=int, default=DEFAULT_CROP_LEN)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--latency", type=int, default=DEFAULT_LATENCY)
    p.add_argument("--standardize_features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--standardize_raw_eeg", action=argparse.BooleanOptionalAction, default=True)

    # Ensemble branches. raw_eeg means raw EEG CNN-transformer. Any other string is treated as an H5 feature family.
    p.add_argument("--branches", nargs="+", default=["raw_eeg", "relative_band_power", "hjorth"])

    # Segment sampling
    p.add_argument("--segment_train_policy", type=str, default="label_aware_k", choices=["all", "random_k", "label_aware_k"])
    p.add_argument("--base_k", type=int, default=8, help="Used by random_k/label_aware_k. Use 0 only with --segment_train_policy all.")
    p.add_argument("--max_k_per_subject", type=int, default=300)

    # Training
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=128, help="Segment batch size for training each branch.")
    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--start_epoch", type=int, default=50)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=True)

    # Architecture
    p.add_argument("--feature_dropout", type=float, default=0.3)
    p.add_argument("--raw_emb_size", type=int, default=64)
    p.add_argument("--raw_depth", type=int, default=3)
    p.add_argument("--raw_heads", type=int, default=4)
    p.add_argument("--raw_dropout", type=float, default=0.5)

    # Runtime/output
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output_root", type=str, default="graph/results_caueeg_softvote_baseline")
    p.add_argument("--run_name", type=str, default=None)

    return p


def validate_args(args) -> None:
    args.branches = parse_str_list(args.branches)
    if len(args.branches) == 0:
        raise ValueError("Provide at least one branch, e.g. --branches raw_eeg relative_band_power hjorth")
    if len(set(args.branches)) != len(args.branches):
        raise ValueError(f"Duplicate branches are not allowed: {args.branches}")
    if len(args.branches) != 3:
        print(f"[Warning] You requested {len(args.branches)} branches: {args.branches}. "
              "The original baseline uses 3 branches, but the code supports any number >=1.")
    if args.segment_train_policy in {"random_k", "label_aware_k"} and int(args.base_k) <= 0:
        raise ValueError("base_k must be > 0 for random_k or label_aware_k.")


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)

    payload, train_ids, val_ids, test_ids, num_classes = prepare_h5_and_payload(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    branch_tag = "-".join(args.branches)
    k_tag = f"k{args.base_k}" if args.segment_train_policy != "all" else "allseg"
    run_name = args.run_name or f"{timestamp}_{args.task}_softvote_{branch_tag}_{k_tag}"
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run directory: {run_dir}")
    print(f"Branches: {args.branches}")
    print(f"Feature families for H5: {infer_feature_families_from_branches(args.branches)}")
    print("NO fusion. NO MIL. Each branch trains separately; probabilities are averaged.")

    summaries = []
    for seed in args.seeds:
        print(f"\n==================== Seed {seed} ====================")
        summaries.append(run_one_seed(
            args,
            payload,
            train_ids,
            val_ids,
            test_ids,
            num_classes=num_classes,
            seed=int(seed),
            root_run_dir=run_dir,
        ))

    save_seed_aggregation(summaries, os.path.join(run_dir, "agg_seed_results"))


if __name__ == "__main__":
    main()
