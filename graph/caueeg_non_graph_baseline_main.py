"""
caueeg_non_graph_baseline_main.py

Non-graph CAUEEG baseline aligned with the MIL-LinkX pipeline.

What this file provides
-----------------------
1) Controlled non-graph MIL baseline:
      feature families (+ optional raw EEG) -> segment encoder -> MIL pooling -> subject prediction

2) Non-MIL segment-training baseline:
      feature families (+ optional raw EEG) -> segment classifier -> subject prediction by soft voting

The data preparation intentionally mirrors the CAUEEG LinkX adapter:
- official CAUEEG task split from load_caueeg_task_datasets(...)
- first 19 EEG channels only
- 10-second windows at 200 Hz: crop_len=2000
- latency/start skip = 2000 samples
- 50% overlap by default
- H5 feature payload loaded through load_h5_payload_for_subjects(...)
- feature_families are selected by command-line argument
- validation/test use all segments by default
- per-seed outputs and seed aggregation are saved

Example
-------
MIL controlled baseline:
python caueeg_non_graph_baseline_main.py \
  --dataset_path /home/anphan/Downloads/caueeg-dataset \
  --task dementia-no-overlap \
  --file_format feather \
  --out_h5 /home/anphan/Documents/caueeg_baseline_features.h5 \
  --training_mode mil \
  --feature_families relative_band_power statistical wavelet_energy \
  --base_k 8 \
  --seeds 15 42 100

Segment/non-MIL baseline:
python caueeg_non_graph_baseline_main.py \
  --dataset_path /home/anphan/Downloads/caueeg-dataset \
  --task dementia-no-overlap \
  --file_format feather \
  --out_h5 /home/anphan/Documents/caueeg_baseline_features.h5 \
  --training_mode segment \
  --feature_families relative_band_power hjorth \
  --segment_train_policy label_aware_k \
  --base_k 8 \
  --seeds 15 42 100
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
# In your repository, caueeg_linkx_mil.py imports these modules.  This
# file follows the same convention.  The fallbacks are only for cases
# where the loader file has a different local name.
try:
    from caueeg_loader_min import load_caueeg_task_datasets
except Exception:  # pragma: no cover
    try:
        from caueeg_script import load_caueeg_task_datasets
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Could not import load_caueeg_task_datasets. Put this file in the same "
            "folder as caueeg_loader_min.py, or adjust the import at the top."
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
# CAUEEG settings copied from the LinkX adapter
# ---------------------------------------------------------------------
CAUEEG_EEG19 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4",
    "T6", "FZ", "CZ", "PZ",
]

SFREQ = 200.0
DEFAULT_CROP_LEN = 2000       # 10 seconds at 200 Hz
DEFAULT_LATENCY = 2000        # skip first 10 seconds, same as CEEDNet/adapter style
DEFAULT_OVERLAP = 0.5


# ---------------------------------------------------------------------
# Reproducibility
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
    # Stable across Python processes, unlike built-in hash(...).
    import hashlib
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


# ---------------------------------------------------------------------
# Data preparation: same windowing as LinkX adapter
# ---------------------------------------------------------------------
def segment_recording(
    signal: np.ndarray,
    crop_len: int = DEFAULT_CROP_LEN,
    overlap: float = DEFAULT_OVERLAP,
    latency: int = DEFAULT_LATENCY,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    signal: [21, T] from CAUEEG
    return windows using first 19 EEG channels only, each [19, crop_len].
    """
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
    """
    Convert CAUEEG Dataset split into records accepted by build_master_eeg_dataset().

    We explicitly prefix the recording id with train_/val_/test_ so H5 subject ids
    are stable and match the current LinkX pipeline convention.
    """
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
        rec = {
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
        }
        records.append(rec)
        subject_ids.append(sid)

    return records, subject_ids


def prepare_h5_and_payload(args) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str], List[str], int]:
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
        print(f"[H5] feature_families={args.feature_families}")
        print(f"[H5] connectivity_metrics_for_build={connectivity_metrics}")

        try:
            build_master_eeg_dataset(
                subject_records=all_records,
                output_h5_path=str(out_h5),
                feature_families=args.feature_families,
                connectivity_metrics=connectivity_metrics,
                overwrite=True,
                skip_bad_segments=False,
                target_sampling_rate=None,
                qc_input_unit="auto",
            )
        except Exception:
            # Some local versions of master_builder expect at least one connectivity metric.
            # Retry with Pearson only when the user requested no connectivity.
            if len(connectivity_metrics) == 0:
                print("[H5] Empty connectivity_metrics failed. Retrying with ['pearson'] for builder compatibility.")
                build_master_eeg_dataset(
                    subject_records=all_records,
                    output_h5_path=str(out_h5),
                    feature_families=args.feature_families,
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
        feature_families=args.feature_families,
        connectivity_metrics=[],
        connectivity_band=None,
        load_raw_for_alignment=bool(args.use_raw_eeg),
        load_bad_segment_flag=False,
    )

    missing = [sid for sid in all_ids if sid not in payload]
    if missing:
        raise KeyError(
            f"H5 payload is missing {len(missing)} requested subject ids. "
            f"Examples: {missing[:5]}. If your old H5 used different ids, rebuild it with --rebuild_h5."
        )

    return payload, train_ids, val_ids, test_ids, num_classes


# ---------------------------------------------------------------------
# Feature extraction from H5 payload
# ---------------------------------------------------------------------
def zscore_per_segment_node_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: [W, C, F]. Z-score over channels, separately for each window and feature."""
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def zscore_raw_eeg_per_window_channel(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """x: [W, C, T]. Z-score over time within each window/channel."""
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
    features: Optional[np.ndarray]       # [W, D] or None
    raw_eeg: Optional[np.ndarray]        # [W, 1, C, T] or None
    segment_ids: np.ndarray              # [W]
    start_samples: np.ndarray            # [W]


def payload_to_subject_arrays(
    payload: Dict[str, Dict[str, Any]],
    subject_ids: Sequence[str],
    feature_families: Sequence[str],
    *,
    use_raw_eeg: bool,
    standardize_features: bool,
    standardize_raw_eeg: bool,
) -> List[SubjectArrays]:
    out: List[SubjectArrays] = []

    for sid in subject_ids:
        subj = payload[sid]
        label = int(subj["label"])
        segment_ids = np.asarray(subj["segment_id"], dtype=np.int64)
        start_samples = np.asarray(subj["start_sample"], dtype=np.int64)

        feat_flat = None
        if len(feature_families) > 0:
            feat_list = []
            for fam in feature_families:
                if fam not in subj["features"]:
                    raise KeyError(f"Subject {sid} is missing feature family {fam!r} in H5 payload.")
                xf = np.asarray(subj["features"][fam], dtype=np.float32)  # [W, C, F]
                if xf.ndim != 3:
                    raise ValueError(f"Feature {fam!r} for {sid} must be [W,C,F], got {xf.shape}")
                feat_list.append(xf)
            feat = np.concatenate(feat_list, axis=-1).astype(np.float32)  # [W, C, F_total]
            if standardize_features:
                feat = zscore_per_segment_node_features(feat)
            feat_flat = feat.reshape(feat.shape[0], -1).astype(np.float32)  # [W, C*F_total]

        raw = None
        if use_raw_eeg:
            if subj.get("raw_eeg", None) is None:
                raise KeyError(
                    f"Subject {sid} has no raw_eeg in payload. Load with --use_raw_eeg and rebuild/use an H5 with raw windows."
                )
            raw0 = np.asarray(subj["raw_eeg"], dtype=np.float32)  # [W, C, T]
            if standardize_raw_eeg:
                raw0 = zscore_raw_eeg_per_window_channel(raw0)
            raw = raw0[:, None, :, :].astype(np.float32)  # [W, 1, C, T]

        if feat_flat is None and raw is None:
            raise ValueError("No input branch is enabled. Provide feature_families or use --use_raw_eeg.")

        n_windows = len(segment_ids)
        if feat_flat is not None and feat_flat.shape[0] != n_windows:
            raise ValueError(f"Feature/window mismatch for {sid}: {feat_flat.shape[0]} vs {n_windows}")
        if raw is not None and raw.shape[0] != n_windows:
            raise ValueError(f"Raw/window mismatch for {sid}: {raw.shape[0]} vs {n_windows}")

        out.append(
            SubjectArrays(
                subject_id=str(sid),
                label=label,
                features=feat_flat,
                raw_eeg=raw,
                segment_ids=segment_ids,
                start_samples=start_samples,
            )
        )

    return out


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
def compute_label_aware_k(subjects: Sequence[SubjectArrays], base_k: int, max_k: Optional[int]) -> Dict[int, int]:
    label_to_subjects: Dict[int, List[str]] = defaultdict(list)
    for s in subjects:
        label_to_subjects[int(s.label)].append(s.subject_id)

    n_subjects_per_label = {label: len(v) for label, v in label_to_subjects.items()}
    max_subjects = max(n_subjects_per_label.values())
    target_segments_per_class = max_subjects * int(base_k)

    k_by_label: Dict[int, int] = {}
    for label, n_subj in n_subjects_per_label.items():
        k = int(math.ceil(target_segments_per_class / n_subj))
        if max_k is not None:
            k = min(k, int(max_k))
        k_by_label[int(label)] = k
    return k_by_label


class SegmentDataset(Dataset):
    """One item = one EEG segment/window."""

    def __init__(
        self,
        subjects: Sequence[SubjectArrays],
        *,
        train: bool,
        train_policy: str = "all",
        base_k: Optional[int] = None,
        max_k_per_subject: Optional[int] = None,
        seed: int = 42,
    ):
        self.subjects = list(subjects)
        self.train = bool(train)
        self.train_policy = str(train_policy).lower()
        self.seed = int(seed)
        self.epoch = 0
        self.base_k = None if base_k is None or int(base_k) <= 0 else int(base_k)
        self.max_k_per_subject = max_k_per_subject
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, w = self.rows[int(idx)]
        s = self.subject_by_id[sid]
        item: Dict[str, Any] = {
            "subject_id": s.subject_id,
            "label": torch.tensor(int(s.label), dtype=torch.long),
            "segment_id": torch.tensor(int(s.segment_ids[w]), dtype=torch.long),
            "start_sample": torch.tensor(int(s.start_samples[w]), dtype=torch.long),
        }
        if s.features is not None:
            item["features"] = torch.tensor(s.features[w], dtype=torch.float32)
        if s.raw_eeg is not None:
            item["raw_eeg"] = torch.tensor(s.raw_eeg[w], dtype=torch.float32)
        return item


class SubjectMILDataset(Dataset):
    """One item = one recording/subject bag containing multiple segments."""

    def __init__(
        self,
        subjects: Sequence[SubjectArrays],
        *,
        train: bool,
        base_k: Optional[int] = None,
        max_k_per_subject: Optional[int] = None,
        eval_k_per_subject: Optional[int] = None,
        seed: int = 42,
    ):
        self.subjects = sorted(list(subjects), key=lambda s: s.subject_id)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.base_k = None if base_k is None or int(base_k) <= 0 else int(base_k)
        self.max_k_per_subject = max_k_per_subject
        self.eval_k_per_subject = None if eval_k_per_subject is None or int(eval_k_per_subject) <= 0 else int(eval_k_per_subject)

        if len(self.subjects) == 0:
            raise ValueError("subjects is empty")

        self.k_by_label = None
        if self.train and self.base_k is not None:
            self.k_by_label = compute_label_aware_k(self.subjects, self.base_k, self.max_k_per_subject)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.subjects)

    def _choose_indices(self, s: SubjectArrays) -> List[int]:
        n = len(s.segment_ids)
        if n == 0:
            return []

        if self.train:
            if self.base_k is None:
                return list(range(n))
            assert self.k_by_label is not None
            k = self.k_by_label[int(s.label)]
            subject_seed = self.seed + 1000003 * self.epoch + stable_int_from_string(s.subject_id)
        else:
            if self.eval_k_per_subject is None:
                return list(range(n))
            k = self.eval_k_per_subject
            subject_seed = self.seed + stable_int_from_string(s.subject_id)

        rng = random.Random(subject_seed)
        if n >= k:
            return rng.sample(range(n), k)
        return list(range(n)) + [rng.randrange(n) for _ in range(k - n)]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.subjects[int(idx)]
        chosen = self._choose_indices(s)
        item: Dict[str, Any] = {
            "subject_id": s.subject_id,
            "label": torch.tensor(int(s.label), dtype=torch.long),
            "segment_ids": torch.tensor(s.segment_ids[chosen], dtype=torch.long),
            "start_samples": torch.tensor(s.start_samples[chosen], dtype=torch.long),
        }
        if s.features is not None:
            item["features"] = torch.tensor(s.features[chosen], dtype=torch.float32)  # [K,D]
        if s.raw_eeg is not None:
            item["raw_eeg"] = torch.tensor(s.raw_eeg[chosen], dtype=torch.float32)    # [K,1,C,T]
        return item


def collate_mil_bags(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = torch.stack([b["label"] for b in batch], dim=0)
    subject_ids = [b["subject_id"] for b in batch]
    bag_sizes = torch.tensor([len(b["segment_ids"]) for b in batch], dtype=torch.long)

    out: Dict[str, Any] = {
        "labels": labels,
        "subject_ids": subject_ids,
        "bag_sizes": bag_sizes,
        "segment_ids": [b["segment_ids"] for b in batch],
        "start_samples": [b["start_samples"] for b in batch],
    }
    if "features" in batch[0]:
        out["features"] = torch.cat([b["features"] for b in batch], dim=0)  # [sumK,D]
    if "raw_eeg" in batch[0]:
        out["raw_eeg"] = torch.cat([b["raw_eeg"] for b in batch], dim=0)    # [sumK,1,C,T]
    return out


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def make_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_dims:
        layers.extend([
            nn.Linear(prev, int(h)),
            nn.LayerNorm(int(h)),
            nn.ELU(),
            nn.Dropout(float(dropout)),
        ])
        prev = int(h)
    return nn.Sequential(*layers), prev


class FeatureMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], emb_dim: int, dropout: float):
        super().__init__()
        self.mlp, last_dim = make_mlp(input_dim, hidden_dims, dropout)
        self.proj = nn.Linear(last_dim, emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        h = self.mlp(x)
        return F.elu(self.proj(h))


class RawEEGTransformerEncoder(nn.Module):
    """
    Lightweight raw-EEG Conformer/CNN-transformer-style encoder.
    Input: [B, 1, C, T].
    """

    def __init__(
        self,
        n_channels: int = 19,
        emb_size: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        out_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.patch = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 20), stride=(1, 10)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=(self.n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 40), stride=(1, 20)),
            nn.Dropout(float(dropout)),
            nn.Conv2d(64, emb_size, kernel_size=(1, 1)),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.proj = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, out_dim),
            nn.ELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # patch: [B,E,1,L] -> [B,L,E]
        h = self.patch(x)
        h = h.squeeze(2).transpose(1, 2).contiguous()
        h = self.transformer(h)
        h = h.mean(dim=1)
        return self.proj(h)


class SegmentEncoder(nn.Module):
    """Shared segment encoder used by both segment training and MIL."""

    def __init__(
        self,
        feature_dim: Optional[int],
        *,
        use_raw_eeg: bool,
        feature_hidden_dims: Sequence[int],
        feature_emb_dim: int,
        raw_emb_dim: int,
        segment_emb_dim: int,
        dropout: float,
        raw_depth: int,
        raw_heads: int,
    ):
        super().__init__()
        self.use_features = feature_dim is not None and int(feature_dim) > 0
        self.use_raw_eeg = bool(use_raw_eeg)

        if not self.use_features and not self.use_raw_eeg:
            raise ValueError("SegmentEncoder needs at least one branch: features or raw_eeg")

        branch_dims = []
        if self.use_features:
            self.feature_encoder = FeatureMLPEncoder(
                input_dim=int(feature_dim),
                hidden_dims=feature_hidden_dims,
                emb_dim=feature_emb_dim,
                dropout=dropout,
            )
            branch_dims.append(feature_emb_dim)
        else:
            self.feature_encoder = None

        if self.use_raw_eeg:
            self.raw_encoder = RawEEGTransformerEncoder(
                n_channels=19,
                emb_size=64,
                depth=raw_depth,
                num_heads=raw_heads,
                out_dim=raw_emb_dim,
                dropout=dropout,
            )
            branch_dims.append(raw_emb_dim)
        else:
            self.raw_encoder = None

        self.fusion = nn.Sequential(
            nn.Linear(sum(branch_dims), segment_emb_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: Optional[Tensor] = None, raw_eeg: Optional[Tensor] = None) -> Tensor:
        embs = []
        if self.use_features:
            if features is None:
                raise ValueError("features branch is enabled but features is None")
            embs.append(self.feature_encoder(features))
        if self.use_raw_eeg:
            if raw_eeg is None:
                raise ValueError("raw_eeg branch is enabled but raw_eeg is None")
            embs.append(self.raw_encoder(raw_eeg))
        h = torch.cat(embs, dim=-1) if len(embs) > 1 else embs[0]
        return self.fusion(h)


class SegmentClassifier(nn.Module):
    def __init__(self, encoder: SegmentEncoder, segment_emb_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(segment_emb_dim, segment_emb_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(segment_emb_dim, num_classes),
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        h = self.encoder(
            features=batch.get("features", None),
            raw_eeg=batch.get("raw_eeg", None),
        )
        logits = self.classifier(h)
        return {"logits": logits, "segment_emb": h}


class GatedAttentionMILPool(nn.Module):
    def __init__(self, input_dim: int, attn_dim: int):
        super().__init__()
        self.v = nn.Linear(input_dim, attn_dim)
        self.u = nn.Linear(input_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: Tensor, bag_sizes: Tensor) -> Tuple[Tensor, List[Tensor]]:
        bag_embs = []
        attn_list = []
        start = 0
        for size in bag_sizes.detach().cpu().tolist():
            end = start + int(size)
            hi = h[start:end]
            scores = self.w(torch.tanh(self.v(hi)) * torch.sigmoid(self.u(hi))).squeeze(-1)
            attn = torch.softmax(scores, dim=0)
            bag_embs.append(torch.sum(attn.unsqueeze(-1) * hi, dim=0))
            attn_list.append(attn)
            start = end
        return torch.stack(bag_embs, dim=0), attn_list


class NonGraphMILClassifier(nn.Module):
    def __init__(
        self,
        encoder: SegmentEncoder,
        segment_emb_dim: int,
        num_classes: int,
        *,
        mil_pool_type: str,
        attn_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.mil_pool_type = str(mil_pool_type).lower()
        if self.mil_pool_type not in {"mean", "gated"}:
            raise ValueError("mil_pool_type must be 'mean' or 'gated'")
        self.gated_pool = GatedAttentionMILPool(segment_emb_dim, attn_dim) if self.mil_pool_type == "gated" else None
        self.classifier = nn.Sequential(
            nn.Linear(segment_emb_dim, segment_emb_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(segment_emb_dim, num_classes),
        )

    def _mean_pool(self, h: Tensor, bag_sizes: Tensor) -> Tuple[Tensor, List[Tensor]]:
        bag_embs = []
        attn_list = []
        start = 0
        for size in bag_sizes.detach().cpu().tolist():
            end = start + int(size)
            hi = h[start:end]
            bag_embs.append(hi.mean(dim=0))
            attn_list.append(torch.full((int(size),), 1.0 / max(int(size), 1), device=h.device, dtype=h.dtype))
            start = end
        return torch.stack(bag_embs, dim=0), attn_list

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        h = self.encoder(
            features=batch.get("features", None),
            raw_eeg=batch.get("raw_eeg", None),
        )
        bag_sizes = batch["bag_sizes"]
        if self.mil_pool_type == "gated":
            bag_emb, attn_list = self.gated_pool(h, bag_sizes)
        else:
            bag_emb, attn_list = self._mean_pool(h, bag_sizes)
        logits = self.classifier(bag_emb)
        return {"logits": logits, "bag_emb": bag_emb, "segment_emb": h, "attn_list": attn_list}


# ---------------------------------------------------------------------
# Metrics, prediction collection, and aggregation
# ---------------------------------------------------------------------
def safe_prob(logits: Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def subject_nll_loss(y_true: np.ndarray, probs: np.ndarray, num_classes: int) -> float:
    y_true = np.asarray(y_true, dtype=int)
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

    metrics = {
        "loss": subject_nll_loss(y_true, probs, num_classes=num_classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "conf_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": probs,
    }
    if num_classes == 2 and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            metrics["roc_auc"] = float("nan")
    elif num_classes > 2 and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
        except Exception:
            metrics["roc_auc_ovr_macro"] = float("nan")
    return metrics


@torch.no_grad()
def predict_segments(model: SegmentClassifier, loader: DataLoader, device: torch.device, num_classes: int) -> pd.DataFrame:
    model.eval()
    rows: List[Dict[str, Any]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        probs = safe_prob(out["logits"])
        preds = probs.argmax(axis=1)
        labels = batch["label"].detach().cpu().numpy().astype(int)
        segment_ids = batch["segment_id"].detach().cpu().numpy().astype(int)
        start_samples = batch["start_sample"].detach().cpu().numpy().astype(int)
        subject_ids = list(batch["subject_id"])

        for i in range(len(labels)):
            row = {
                "subject_id": subject_ids[i],
                "segment_id": int(segment_ids[i]),
                "start_sample": int(start_samples[i]),
                "true_label": int(labels[i]),
                "pred_label": int(preds[i]),
            }
            for c in range(num_classes):
                row[f"prob_{c}"] = float(probs[i, c])
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_segment_predictions_to_subjects(seg_df: pd.DataFrame, num_classes: int) -> pd.DataFrame:
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    rows = []
    for sid, sdf in seg_df.groupby("subject_id"):
        prob = sdf[prob_cols].to_numpy(dtype=float).mean(axis=0)
        row = {
            "subject_id": sid,
            "true_label": int(sdf["true_label"].iloc[0]),
            "pred_label": int(np.argmax(prob)),
            "num_segments": int(len(sdf)),
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(prob[c])
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def predict_mil_subjects(
    model: NonGraphMILClassifier,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    *,
    save_attention: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    model.eval()
    pred_rows: List[Dict[str, Any]] = []
    attn_rows: List[Dict[str, Any]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)
        probs = safe_prob(out["logits"])
        preds = probs.argmax(axis=1)
        labels = batch["labels"].detach().cpu().numpy().astype(int)
        subject_ids = list(batch["subject_ids"])

        for i, sid in enumerate(subject_ids):
            row = {
                "subject_id": sid,
                "true_label": int(labels[i]),
                "pred_label": int(preds[i]),
                "num_segments": int(batch["bag_sizes"][i].detach().cpu().item()),
            }
            for c in range(num_classes):
                row[f"prob_{c}"] = float(probs[i, c])
            pred_rows.append(row)

            if save_attention:
                seg_ids = batch["segment_ids"][i].detach().cpu().numpy().astype(int)
                starts = batch["start_samples"][i].detach().cpu().numpy().astype(int)
                attn = out["attn_list"][i].detach().cpu().numpy().reshape(-1)
                for j in range(len(seg_ids)):
                    attn_rows.append({
                        "subject_id": sid,
                        "true_label": int(labels[i]),
                        "subject_pred": int(preds[i]),
                        "segment_id": int(seg_ids[j]),
                        "start_sample": int(starts[j]),
                        "attention": float(attn[j]),
                    })

    attn_df = pd.DataFrame(attn_rows) if save_attention else None
    return pd.DataFrame(pred_rows), attn_df


def metrics_from_prediction_df(df: pd.DataFrame, num_classes: int) -> Dict[str, Any]:
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    return compute_metrics_from_probs(
        y_true=df["true_label"].to_numpy(dtype=int),
        probs=df[prob_cols].to_numpy(dtype=float),
        num_classes=num_classes,
    )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------
def make_class_weights(subjects: Sequence[SubjectArrays], num_classes: int, device: torch.device) -> Optional[Tensor]:
    labels = np.asarray([s.label for s in subjects], dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


class SimpleEarlyStopper:
    """
    Selection rule aligned with your current MIL runs:
      1) highest val_bal_acc
      2) highest val_macro_f1
      3) lowest val_loss
    Patience starts after start_epoch.
    """

    def __init__(self, patience: int, start_epoch: int, min_delta: float):
        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.best_key: Optional[Tuple[float, float, float]] = None
        self.best_epoch = 0
        self.counter = 0

    @staticmethod
    def key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
        return (
            float(metrics["balanced_accuracy"]),
            float(metrics["macro_f1"]),
            -float(metrics["loss"]),
        )

    def step(self, epoch: int, metrics: Dict[str, Any]) -> Tuple[bool, bool]:
        current = self.key(metrics)
        improved = False
        if self.best_key is None:
            improved = True
        elif current[0] > self.best_key[0] + self.min_delta:
            improved = True
        elif abs(current[0] - self.best_key[0]) <= self.min_delta and current[1:] > self.best_key[1:]:
            improved = True

        if improved:
            self.best_key = current
            self.best_epoch = int(epoch)
            if epoch >= self.start_epoch:
                self.counter = 0
        elif epoch >= self.start_epoch:
            self.counter += 1

        should_stop = epoch >= self.start_epoch and self.counter >= self.patience
        return improved, should_stop


def build_encoder(args, feature_dim: Optional[int]) -> SegmentEncoder:
    return SegmentEncoder(
        feature_dim=feature_dim,
        use_raw_eeg=bool(args.use_raw_eeg),
        feature_hidden_dims=args.feature_hidden_dims,
        feature_emb_dim=args.feature_emb_dim,
        raw_emb_dim=args.raw_emb_dim,
        segment_emb_dim=args.segment_emb_dim,
        dropout=args.dropout,
        raw_depth=args.raw_depth,
        raw_heads=args.raw_heads,
    )


def train_segment_baseline(
    args,
    train_subjects: Sequence[SubjectArrays],
    val_subjects: Sequence[SubjectArrays],
    test_subjects: Sequence[SubjectArrays],
    num_classes: int,
    run_dir: str,
    seed: int,
) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_ds = SegmentDataset(
        train_subjects,
        train=True,
        train_policy=args.segment_train_policy,
        base_k=args.base_k,
        max_k_per_subject=args.max_k_per_subject,
        seed=seed,
    )
    val_ds = SegmentDataset(val_subjects, train=False, train_policy="all", seed=seed)
    test_ds = SegmentDataset(test_subjects, train=False, train_policy="all", seed=seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    feature_dim = train_subjects[0].features.shape[1] if train_subjects[0].features is not None else None
    encoder = build_encoder(args, feature_dim)
    model = SegmentClassifier(encoder, args.segment_emb_dim, num_classes, args.dropout).to(device)

    weight = make_class_weights(train_subjects, num_classes, device) if args.use_class_weight else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    stopper = SimpleEarlyStopper(args.patience, args.start_epoch, args.min_delta)
    best_state = None
    history: List[Dict[str, Any]] = []
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch - 1)

        model.train()
        total_loss = 0.0
        total_n = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.item()) * int(batch["label"].numel())
            total_n += int(batch["label"].numel())

        train_seg_df = predict_segments(model, train_loader, device, num_classes)
        val_seg_df = predict_segments(model, val_loader, device, num_classes)
        train_sub_df = aggregate_segment_predictions_to_subjects(train_seg_df, num_classes)
        val_sub_df = aggregate_segment_predictions_to_subjects(val_seg_df, num_classes)
        train_metrics = metrics_from_prediction_df(train_sub_df, num_classes)
        val_metrics = metrics_from_prediction_df(val_sub_df, num_classes)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_bal_acc": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train subj bal={row['train_bal_acc']:.4f} f1={row['train_macro_f1']:.4f} | "
            f"Val subj loss={row['val_loss']:.4f} bal={row['val_bal_acc']:.4f} f1={row['val_macro_f1']:.4f}"
        )

        improved, should_stop = stopper.step(epoch, val_metrics)
        if improved:
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "val_metrics": {k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)},
                "history": copy.deepcopy(history),
            }
            torch.save(best_state, ckpt_path)
        if should_stop:
            print(f"Early stopping at epoch {epoch}; best epoch={stopper.best_epoch}")
            break

    if best_state is None and os.path.isfile(ckpt_path):
        best_state = torch.load(ckpt_path, map_location="cpu")
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    val_seg_df = predict_segments(model, val_loader, device, num_classes)
    test_seg_df = predict_segments(model, test_loader, device, num_classes)
    val_sub_df = aggregate_segment_predictions_to_subjects(val_seg_df, num_classes)
    test_sub_df = aggregate_segment_predictions_to_subjects(test_seg_df, num_classes)

    val_seg_df.to_csv(os.path.join(run_dir, "val_segment_predictions.csv"), index=False)
    test_seg_df.to_csv(os.path.join(run_dir, "test_segment_predictions.csv"), index=False)
    val_sub_df.to_csv(os.path.join(run_dir, "val_subject_predictions.csv"), index=False)
    test_sub_df.to_csv(os.path.join(run_dir, "test_subject_predictions.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    val_metrics = metrics_from_prediction_df(val_sub_df, num_classes)
    test_metrics = metrics_from_prediction_df(test_sub_df, num_classes)

    return {
        "model": model,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_pred_df": val_sub_df,
        "test_pred_df": test_sub_df,
    }


def train_mil_baseline(
    args,
    train_subjects: Sequence[SubjectArrays],
    val_subjects: Sequence[SubjectArrays],
    test_subjects: Sequence[SubjectArrays],
    num_classes: int,
    run_dir: str,
    seed: int,
) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    train_ds = SubjectMILDataset(
        train_subjects,
        train=True,
        base_k=args.base_k,
        max_k_per_subject=args.max_k_per_subject,
        eval_k_per_subject=None,
        seed=seed,
    )
    val_ds = SubjectMILDataset(val_subjects, train=False, eval_k_per_subject=args.eval_k_per_subject, seed=seed)
    test_ds = SubjectMILDataset(test_subjects, train=False, eval_k_per_subject=args.eval_k_per_subject, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_mil_bags, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mil_bags, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_mil_bags, num_workers=args.num_workers, pin_memory=True)

    feature_dim = train_subjects[0].features.shape[1] if train_subjects[0].features is not None else None
    encoder = build_encoder(args, feature_dim)
    model = NonGraphMILClassifier(
        encoder,
        args.segment_emb_dim,
        num_classes,
        mil_pool_type=args.mil_pool_type,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)

    weight = make_class_weights(train_subjects, num_classes, device) if args.use_class_weight else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.lr_patience)

    stopper = SimpleEarlyStopper(args.patience, args.start_epoch, args.min_delta)
    best_state = None
    history: List[Dict[str, Any]] = []
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch - 1)

        model.train()
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            loss = criterion(out["logits"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        train_pred_df, _ = predict_mil_subjects(model, train_loader, device, num_classes, save_attention=False)
        val_pred_df, _ = predict_mil_subjects(model, val_loader, device, num_classes, save_attention=False)
        train_metrics = metrics_from_prediction_df(train_pred_df, num_classes)
        val_metrics = metrics_from_prediction_df(val_pred_df, num_classes)
        scheduler.step(val_metrics["loss"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_bal_acc": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_bal_acc": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"Train bal={row['train_bal_acc']:.4f} f1={row['train_macro_f1']:.4f} | "
            f"Val loss={row['val_loss']:.4f} bal={row['val_bal_acc']:.4f} f1={row['val_macro_f1']:.4f}"
        )

        improved, should_stop = stopper.step(epoch, val_metrics)
        if improved:
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "val_metrics": {k: v for k, v in val_metrics.items() if not isinstance(v, np.ndarray)},
                "history": copy.deepcopy(history),
            }
            torch.save(best_state, ckpt_path)
        if should_stop:
            print(f"Early stopping at epoch {epoch}; best epoch={stopper.best_epoch}")
            break

    if best_state is None and os.path.isfile(ckpt_path):
        best_state = torch.load(ckpt_path, map_location="cpu")
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    val_pred_df, val_attn_df = predict_mil_subjects(model, val_loader, device, num_classes, save_attention=args.save_attention)
    test_pred_df, test_attn_df = predict_mil_subjects(model, test_loader, device, num_classes, save_attention=args.save_attention)

    val_pred_df.to_csv(os.path.join(run_dir, "val_subject_predictions.csv"), index=False)
    test_pred_df.to_csv(os.path.join(run_dir, "test_subject_predictions.csv"), index=False)
    if val_attn_df is not None:
        val_attn_df.to_csv(os.path.join(run_dir, "val_attention.csv"), index=False)
    if test_attn_df is not None:
        test_attn_df.to_csv(os.path.join(run_dir, "test_attention.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)

    val_metrics = metrics_from_prediction_df(val_pred_df, num_classes)
    test_metrics = metrics_from_prediction_df(test_pred_df, num_classes)

    return {
        "model": model,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
    }


# ---------------------------------------------------------------------
# Saving and aggregation
# ---------------------------------------------------------------------
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
    if isinstance(row.get("feature_hidden_dims"), list):
        row["feature_hidden_dims"] = ",".join(map(str, row["feature_hidden_dims"]))
    if "confusion_matrix" in row:
        row["confusion_matrix_json"] = json.dumps(row["confusion_matrix"])
        del row["confusion_matrix"]
    return row


def save_seed_aggregation(summary_rows: Sequence[Dict[str, Any]], output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    rows = [normalize_summary_row(r) for r in summary_rows]
    df = pd.DataFrame(rows)
    raw_path = os.path.join(output_dir, "all_seed_results.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "macro_f1", "loss"] if c in df.columns]
    variant_cols = [
        "baseline_type",
        "training_mode",
        "mil_pool_type",
        "feature_families",
        "use_raw_eeg",
        "segment_train_policy",
        "base_k",
        "eval_k_per_subject",
        "batch_size",
        "epochs",
        "patience",
        "start_epoch",
        "lr",
        "dropout",
        "weight_decay",
        "segment_emb_dim",
        "feature_emb_dim",
        "raw_emb_dim",
        "attn_dim",
        "task",
        "crop_len",
        "overlap",
        "latency",
    ]
    variant_cols = [c for c in variant_cols if c in df.columns]

    agg = (
        df.groupby(variant_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
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


def save_run_config(args, run_dir: str, seed: int) -> None:
    cfg = vars(args).copy()
    cfg["seed"] = int(seed)
    cfg = make_jsonable(cfg)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------
# Main experiment orchestration
# ---------------------------------------------------------------------
def class_names_for_task(task: str, num_classes: int) -> List[str]:
    task = str(task).lower()
    if task.startswith("abnormal"):
        return ["normal", "abnormal"]
    if task.startswith("dementia"):
        return ["normal", "mci", "dementia"]
    return [f"class_{i}" for i in range(num_classes)]


def run_one_seed(args, payload, train_ids, val_ids, test_ids, num_classes: int, seed: int, root_run_dir: str) -> Dict[str, Any]:
    set_seed(seed)

    train_subjects = payload_to_subject_arrays(
        payload,
        train_ids,
        args.feature_families,
        use_raw_eeg=args.use_raw_eeg,
        standardize_features=args.standardize_features,
        standardize_raw_eeg=args.standardize_raw_eeg,
    )
    val_subjects = payload_to_subject_arrays(
        payload,
        val_ids,
        args.feature_families,
        use_raw_eeg=args.use_raw_eeg,
        standardize_features=args.standardize_features,
        standardize_raw_eeg=args.standardize_raw_eeg,
    )
    test_subjects = payload_to_subject_arrays(
        payload,
        test_ids,
        args.feature_families,
        use_raw_eeg=args.use_raw_eeg,
        standardize_features=args.standardize_features,
        standardize_raw_eeg=args.standardize_raw_eeg,
    )

    seed_dir = os.path.join(root_run_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    save_run_config(args, seed_dir, seed)

    if args.training_mode == "mil":
        result = train_mil_baseline(args, train_subjects, val_subjects, test_subjects, num_classes, seed_dir, seed)
        training_approach = "MIL-subject"
    elif args.training_mode == "segment":
        result = train_segment_baseline(args, train_subjects, val_subjects, test_subjects, num_classes, seed_dir, seed)
        training_approach = "segment-CE-softvote-subject"
    else:
        raise ValueError("training_mode must be 'mil' or 'segment'")

    test_metrics = result["test_metrics"]
    val_metrics = result["val_metrics"]

    summary = {
        "baseline_type": "non_graph",
        "training_mode": args.training_mode,
        "training_approach": training_approach,
        "mil_pool_type": args.mil_pool_type if args.training_mode == "mil" else "none",
        "feature_families": list(args.feature_families),
        "use_raw_eeg": bool(args.use_raw_eeg),
        "segment_train_policy": args.segment_train_policy,
        "accuracy": float(test_metrics["accuracy"]),
        "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "macro_f1": float(test_metrics["macro_f1"]),
        "loss": float(test_metrics["loss"]),
        "val_accuracy": float(val_metrics["accuracy"]),
        "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        "val_macro_f1": float(val_metrics["macro_f1"]),
        "val_loss": float(val_metrics["loss"]),
        "confusion_matrix": test_metrics["conf_matrix"],
        "task": args.task,
        "num_classes": int(num_classes),
        "class_names": class_names_for_task(args.task, num_classes),
        "seed": int(seed),
        "base_k": args.base_k,
        "eval_k_per_subject": args.eval_k_per_subject,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "start_epoch": args.start_epoch,
        "lr": args.lr,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "segment_emb_dim": args.segment_emb_dim,
        "feature_emb_dim": args.feature_emb_dim,
        "raw_emb_dim": args.raw_emb_dim,
        "attn_dim": args.attn_dim,
        "feature_hidden_dims": list(args.feature_hidden_dims),
        "crop_len": args.crop_len,
        "overlap": args.overlap,
        "latency": args.latency,
        "run_dir": seed_dir,
    }

    pd.DataFrame([normalize_summary_row(summary)]).to_csv(os.path.join(seed_dir, "summary_test.csv"), index=False)
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CAUEEG non-graph baseline aligned with LinkX/MIL pipeline")

    # Data
    p.add_argument("--dataset_path", type=str, default="/home/anphan/Downloads/caueeg-dataset/")
    p.add_argument("--task", type=str, default="dementia-no-overlap", choices=["abnormal", "dementia", "abnormal-no-overlap", "dementia-no-overlap"])
    p.add_argument("--file_format", type=str, default="edf", choices=["edf", "feather", "memmap", "np"])
    p.add_argument("--out_h5", type=str, default="/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5")
    p.add_argument("--rebuild_h5", action="store_true")
    p.add_argument("--build_connectivity_metric", type=str, default="none", help="Use 'none' for feature-only H5; use pearson if your master_builder requires a connectivity metric.")
    p.add_argument("--feature_families", nargs="+", default=["relative_band_power", "statistical", "wavelet_energy"])
    p.add_argument("--use_raw_eeg", action="store_true", help="Add raw EEG CNN-transformer branch.")
    p.add_argument("--bad_ids", nargs="*", default=["00587", "00781", "01301"])
    p.add_argument("--crop_len", type=int, default=DEFAULT_CROP_LEN)
    p.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    p.add_argument("--latency", type=int, default=DEFAULT_LATENCY)
    p.add_argument("--standardize_features", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--standardize_raw_eeg", action=argparse.BooleanOptionalAction, default=True)

    # Experiment
    p.add_argument("--training_mode", type=str, default="mil", choices=["mil", "segment"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--output_root", type=str, default="graph/results_caueeg_non_graph_baseline")
    p.add_argument("--run_name", type=str, default=None)

    # Segment selection
    p.add_argument("--base_k", type=int, default=10, help="For MIL train bags and optional segment train sampling. Use 0 for all train segments/bag segments.")
    p.add_argument("--max_k_per_subject", type=int, default=300)
    p.add_argument("--eval_k_per_subject", type=int, default=0, help="0 means use all val/test segments.")
    p.add_argument("--segment_train_policy", type=str, default="label_aware_k", choices=["all", "random_k", "label_aware_k"])

    # Model
    p.add_argument("--mil_pool_type", type=str, default="mean", choices=["mean", "gated"])
    p.add_argument("--feature_hidden_dims", nargs="+", type=int, default=[64, 32])
    p.add_argument("--feature_emb_dim", type=int, default=64)
    p.add_argument("--raw_emb_dim", type=int, default=64)
    p.add_argument("--segment_emb_dim", type=int, default=64)
    p.add_argument("--attn_dim", type=int, default=64)
    p.add_argument("--raw_depth", type=int, default=3)
    p.add_argument("--raw_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)

    # Optimization
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8, help="For MIL this is subjects/bags; for segment mode this is segments.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-3)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--start_epoch", type=int, default=50)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--lr_patience", type=int, default=8)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--use_class_weight", action="store_true")

    # System/output
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_attention", action="store_true")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    args.feature_families = parse_str_list(args.feature_families)
    args.bad_ids = parse_str_list(args.bad_ids)
    args.eval_k_per_subject = None if args.eval_k_per_subject is None or int(args.eval_k_per_subject) <= 0 else int(args.eval_k_per_subject)
    args.base_k = None if args.base_k is None or int(args.base_k) <= 0 else int(args.base_k)

    if len(args.feature_families) == 0 and not args.use_raw_eeg:
        raise ValueError("No input selected: provide --feature_families or set --use_raw_eeg")

    payload, train_ids, val_ids, test_ids, num_classes = prepare_h5_and_payload(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fam_tag = "raw" if len(args.feature_families) == 0 else "-".join(args.feature_families)
    raw_tag = "rawEEG" if args.use_raw_eeg else "noRaw"
    k_tag = "all" if args.base_k is None else f"k{args.base_k}"
    run_name = args.run_name or f"{timestamp}_{args.task}_nongraph_{args.training_mode}_{fam_tag}_{raw_tag}_{k_tag}"
    root_run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(root_run_dir, exist_ok=True)

    with open(os.path.join(root_run_dir, "root_config.json"), "w") as f:
        json.dump(make_jsonable(vars(args)), f, indent=2)

    summary_rows = []
    for seed in args.seeds:
        print("=" * 80)
        print(f"Running seed={seed} | mode={args.training_mode} | features={args.feature_families} | raw={args.use_raw_eeg}")
        print("=" * 80)
        summary = run_one_seed(args, payload, train_ids, val_ids, test_ids, num_classes, int(seed), root_run_dir)
        summary_rows.append(summary)

    save_seed_aggregation(summary_rows, root_run_dir)
    print(f"Done. Results saved to: {root_run_dir}")


if __name__ == "__main__":
    main()
