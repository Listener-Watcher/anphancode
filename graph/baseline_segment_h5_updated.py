#!/usr/bin/env python3
"""
baseline_segment_h5.py

Segment-level baseline runner for the EEG raw + RBP + Hjorth architecture from
model.py/main.py, adapted to the same split/seed/fold/evaluation style used by
mil_full_std.py.

Training unit:
    one EEG window/segment = one training example

Evaluation:
    1) segment-level metrics
    2) subject-level metrics by soft-voting / averaging segment probabilities

Expected H5 layout from master_builder.py:
    subjects/<sid>/metadata.attrs['label']
    subjects/<sid>/windows/raw/eeg                    [W, C, T]
    subjects/<sid>/windows/raw/segment_id             [W]
    subjects/<sid>/windows/features/relative_band_power [W, C, F]
    subjects/<sid>/windows/features/hjorth              [W, C, F]

Example:
python baseline_segment_h5.py \
  --h5_path /path/to/master.h5 \
  --output_dir ./baseline_segment_results \
  --baseline_model ensemble_vote \
  --split_seeds 15 42 101 \
  --k_folds 10 \
  --val_ratio 0.1 \
  --epochs 100 \
  --patience 20 \
  --batch_size 64
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

try:
    import matplotlib.pyplot as plt
except Exception:  # plotting is optional
    plt = None

from models.resnet_1d import ResNet1D

# =============================================================================
# Reproducibility / IO
# =============================================================================

def set_global_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | os.PathLike) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x


def stable_int_from_string(x: str) -> int:
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)

class AHEAPResNet1DWrapper(nn.Module):
    """
    Wrapper for CEEDNet ResNet1D so it can be used with your current
    train_model(model, train_loader, ..., feature_type='eeg_feature') function.

    Your train_model calls model(x), but CEEDNet ResNet1D calls model(x, age).
    For AHEAP, we set use_age='no' and pass dummy age.
    """

    def __init__(
        self,
        num_classes,
        seq_length,
        in_channels=19,

        block="basic",
        conv_layers=[1, 1, 1, 1],
        base_channels=16,
        fc_stages=1,
        dropout=0.5,
        activation="relu",
    ):
        super().__init__()

        self.model = ResNet1D(
            block=block,
            conv_layers=conv_layers,
            in_channels=in_channels,
            out_dims=num_classes,
            seq_length=seq_length,
            base_channels=base_channels,
            use_age="no",        # important for AHEAP if age is unavailable
            fc_stages=fc_stages,
            dropout=dropout,
            activation=activation,
            base_pool="max",
            final_pool="average",
        )

    def forward(self, x):
        # Your current EEG input is likely [B, 1, 19, T].
        # ResNet1D expects [B, 19, T].
        if x.ndim == 4:
            x = x.squeeze(1)

        age = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        return self.model(x, age)

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, num_heads: int, dropout: float):
        super().__init__()
        if emb_size % num_heads != 0:
            raise ValueError(f"emb_size={emb_size} must be divisible by num_heads={num_heads}")
        self.emb_size = int(emb_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.emb_size // self.num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.queries(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.keys(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.values(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        energy = torch.matmul(q, k.transpose(-2, -1))
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        att = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        att = self.att_drop(att)
        out = torch.matmul(att, v).transpose(1, 2).contiguous().view(b, n, self.emb_size)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return x + self.fn(x, **kwargs)


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int, drop_p: float):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, emb_size: int, num_heads: int = 4, drop_p: float = 0.5,
                 forward_expansion: int = 4, forward_drop_p: float = 0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p),
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p),
            )),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int, emb_size: int):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size: int = 64, n_channels: int = 19):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 32, (1, 40), (1, 10)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 40), (1, 20)),
            nn.Dropout(0.5),
        )
        self.projection = nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, C, T]
        x = self.shallownet(x)
        x = self.projection(x)             # [B, E, 1, W]
        x = x.flatten(2).transpose(1, 2)   # [B, W, E]
        return x


class ClassificationHead(nn.Module):
    def __init__(self, num_classes: int, fc: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fc, 128),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.contiguous().view(x.size(0), -1))


class ClassificationHeadCombine(nn.Module):
    def __init__(self, fc: int = 512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fc, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.contiguous().view(x.size(0), -1))


class EEGTransformer(nn.Sequential):
    def __init__(self, emb_size: int = 64, n_channels: int = 19, fc: int = 128,
                 depth: int = 3, n_classes: int = 2):
        super().__init__(PatchEmbedding(emb_size, n_channels), TransformerEncoder(depth, emb_size), ClassificationHead(n_classes, fc))


class EEGTransCombine(nn.Sequential):
    def __init__(self, emb_size: int = 64, n_channels: int = 19, fc: int = 512, depth: int = 3):
        super().__init__(PatchEmbedding(emb_size, n_channels), TransformerEncoder(depth, emb_size), ClassificationHeadCombine(fc))


class RBPFeatureExtractor(nn.Module):
    def __init__(self, n_classes: int = 2, n_channels: int = 19, t: int = 1, b: int = 5):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(t * b * n_channels, 128)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(128, 64)
        self.elu2 = nn.ELU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        x = self.flatten(x)
        x = self.elu1(self.fc1(x))
        features = self.elu2(self.fc2(x))
        out = self.classifier(self.dropout(features))
        return (out, features) if return_feature else out


class HJFeatureExtractor(nn.Module):
    def __init__(self, n_classes: int = 2, n_channels: int = 19, t: int = 1):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(n_channels * 3 * t, 128)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(128, 64)
        self.elu2 = nn.ELU()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor, return_feature: bool = False):
        x = self.flatten(x)
        x = self.elu1(self.fc1(x))
        features = self.elu2(self.fc2(x))
        out = self.classifier(self.dropout(features))
        return (out, features) if return_feature else out


class RBPCombine(nn.Module):
    def __init__(self, n_channels: int = 19, t: int = 1, b: int = 5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(t * b * n_channels, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class HJCombine(nn.Module):
    def __init__(self, n_channels: int = 19, t: int = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_channels * 3 * t, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.flatten(1))


class AlzheimerClassifier(nn.Module):
    def __init__(self, n_classes: int = 2, channels: int = 19, fc: int = 512, t: int = 1, b: int = 5,
                 emb_size: int = 64, depth: int = 3, hj_t: Optional[int] = None):
        super().__init__()
        self.eeg_model = EEGTransCombine(emb_size=emb_size, n_channels=channels, fc=fc, depth=depth)
        self.rbp_model = RBPCombine(n_channels=channels, t=t, b=b)
        self.hj_model = HJCombine(n_channels=channels, t=(t if hj_t is None else hj_t))
        self.fc = nn.Sequential(
            nn.Linear(128 * 3, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, eeg: torch.Tensor, rbp: torch.Tensor, hj: torch.Tensor) -> torch.Tensor:
        eeg_features = self.eeg_model(eeg)
        rbp_features = self.rbp_model(rbp)
        hj_features = self.hj_model(hj)
        return self.fc(torch.cat([eeg_features, rbp_features, hj_features], dim=1))


def infer_eeg_fc(n_channels: int, raw_timepoints: int, emb_size: int, depth: int, device: str = "cpu") -> int:
    patch = PatchEmbedding(emb_size=emb_size, n_channels=n_channels).to(device)
    enc = TransformerEncoder(depth=depth, emb_size=emb_size).to(device)
    patch.eval(); enc.eval()
    with torch.no_grad():
        dummy = torch.zeros(2, 1, n_channels, raw_timepoints, device=device)
        out = enc(patch(dummy))
        return int(out.reshape(out.size(0), -1).shape[1])


# =============================================================================
# H5 metadata and datasets
# =============================================================================

def decode_str_array(arr: np.ndarray) -> List[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def list_subjects_and_labels(h5_path: str, subject_ids: Optional[Sequence[str]] = None) -> Tuple[List[str], List[int]]:
    with h5py.File(h5_path, "r") as h5f:
        available = sorted(list(h5f.get("subjects", {}).keys()))
        selected = available if subject_ids is None else [str(s) for s in subject_ids if str(s) in available]
        labels = [int(h5f[f"subjects/{sid}/metadata"].attrs["label"]) for sid in selected]
    if not selected:
        raise ValueError("No matching subjects found in H5.")
    return selected, labels


def get_h5_shapes(h5_path: str, rbp_family: str, hj_family: str, subject_id: Optional[str] = None) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5f:
        sid = subject_id or sorted(list(h5f["subjects"].keys()))[0]
        grp = h5f[f"subjects/{sid}"]
        raw_shape = tuple(grp["windows/raw/eeg"].shape)       # [W, C, T]
        rbp_shape = tuple(grp[f"windows/features/{rbp_family}"].shape)  # [W, C, ...]
        hj_shape = tuple(grp[f"windows/features/{hj_family}"].shape)
        ch_names = decode_str_array(grp["metadata/channel_names"][:])
    return {"subject_id": sid, "raw_shape": raw_shape, "rbp_shape": rbp_shape, "hj_shape": hj_shape, "channel_names": ch_names}


def resolve_feature_dims(rbp_sample_shape: Sequence[int], hj_sample_shape: Sequence[int], rbp_b: Optional[int]) -> Tuple[int, int, int, int]:
    # sample shapes exclude window dimension, normally [C, F]
    n_channels = int(rbp_sample_shape[0])
    rbp_per_ch = int(np.prod(rbp_sample_shape[1:]))
    if rbp_b is None:
        if rbp_per_ch % 5 == 0:
            b = 5
        else:
            b = rbp_per_ch
    else:
        b = int(rbp_b)
    if rbp_per_ch % b != 0:
        raise ValueError(f"RBP per-channel feature dim {rbp_per_ch} is not divisible by b={b}.")
    rbp_t = rbp_per_ch // b

    hj_per_ch = int(np.prod(hj_sample_shape[1:]))
    if hj_per_ch % 3 != 0:
        raise ValueError(f"Hjorth per-channel feature dim must be 3*t, got {hj_per_ch}.")
    hj_t = hj_per_ch // 3
    if hj_t != rbp_t:
        print(f"[Info] inferred rbp_t={rbp_t}, hj_t={hj_t}. The runner will use separate rbp_t and hj_t.")
    return n_channels, rbp_t, b, hj_t


class H5SegmentDataset(Dataset):
    def __init__(self, h5_path: str, subject_ids: Sequence[str], *, rbp_family: str, hj_family: str,
                 train: bool = False, segment_policy: str = "all", base_k: Optional[int] = None,
                 seed: int = 42):
        self.h5_path = str(h5_path)
        self.rbp_family = str(rbp_family)
        self.hj_family = str(hj_family)
        self.train = bool(train)
        self.segment_policy = str(segment_policy)
        self.base_k = None if base_k is None else int(base_k)
        self.seed = int(seed)
        self._h5: Optional[h5py.File] = None
        self.index: List[Tuple[str, int]] = []
        self.subject_labels: Dict[str, int] = {}

        with h5py.File(self.h5_path, "r") as h5f:
            for sid in subject_ids:
                sid = str(sid)
                if sid not in h5f["subjects"]:
                    continue
                grp = h5f[f"subjects/{sid}"]
                n = int(grp["windows/raw/eeg"].shape[0])
                y = int(grp["metadata"].attrs["label"])
                self.subject_labels[sid] = y
                if self.train and self.segment_policy == "random_k" and self.base_k is not None:
                    rng = random.Random(self.seed + stable_int_from_string(sid))
                    if n >= self.base_k:
                        chosen = rng.sample(range(n), self.base_k)
                    else:
                        chosen = list(range(n)) + [rng.randrange(n) for _ in range(self.base_k - n)]
                    self.index.extend((sid, int(i)) for i in chosen)
                else:
                    self.index.extend((sid, i) for i in range(n))
        if len(self.index) == 0:
            raise ValueError("Segment dataset is empty.")

    def __len__(self) -> int:
        return len(self.index)

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    def __del__(self):
        self.close()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sid, wi = self.index[idx]
        h5f = self._open()
        grp = h5f[f"subjects/{sid}"]
        eeg = grp["windows/raw/eeg"][wi].astype(np.float32)              # [C, T]
        rbp = grp[f"windows/features/{self.rbp_family}"][wi].astype(np.float32)
        hj = grp[f"windows/features/{self.hj_family}"][wi].astype(np.float32)
        label = int(grp["metadata"].attrs["label"])
        seg_id = int(grp["windows/raw/segment_id"][wi]) if "segment_id" in grp["windows/raw"] else int(wi)
        start = int(grp["windows/raw/start_sample"][wi]) if "start_sample" in grp["windows/raw"] else -1
        return {
            "eeg": torch.from_numpy(eeg).unsqueeze(0),   # [1, C, T]
            "rbp": torch.from_numpy(rbp),
            "hj": torch.from_numpy(hj),
            "label": torch.tensor(label, dtype=torch.long),
            "subject_id": sid,
            "segment_id": seg_id,
            "start_sample": start,
        }


def segment_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "eeg": torch.stack([b["eeg"] for b in batch], dim=0).float(),
        "rbp": torch.stack([b["rbp"] for b in batch], dim=0).float(),
        "hj": torch.stack([b["hj"] for b in batch], dim=0).float(),
        "label": torch.stack([b["label"] for b in batch], dim=0).long(),
        "subject_id": [b["subject_id"] for b in batch],
        "segment_id": [int(b["segment_id"]) for b in batch],
        "start_sample": [int(b["start_sample"]) for b in batch],
    }


# =============================================================================
# Split helpers
# =============================================================================

def balanced_kfold_split(subject_ids: Sequence[str], labels: Sequence[int], seed: int, k: int) -> List[List[str]]:
    subject_ids = np.asarray(subject_ids)
    y = np.asarray(labels, dtype=np.int64)
    min_count = min(Counter(y.tolist()).values())
    if k > min_count:
        raise ValueError(f"k_folds={k} exceeds smallest class count={min_count}.")
    skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=int(seed))
    folds = []
    for fold_idx, (_, test_idx) in enumerate(skf.split(subject_ids, y)):
        test_subjects = subject_ids[test_idx].tolist()
        folds.append(test_subjects)
        counts = Counter(y[test_idx].tolist())
        print(f"Fold {fold_idx}: Subjects={test_subjects}, counts={dict(counts)}")
    return folds


def stratified_split_subjects(train_subjects: Sequence[str], label_map: Mapping[str, int], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    train_subjects = list(train_subjects)
    y = np.asarray([int(label_map[s]) for s in train_subjects], dtype=np.int64)
    counts = Counter(y.tolist())
    if len(counts) >= 2 and all(v >= 2 for v in counts.values()):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed))
        tr_idx, va_idx = next(splitter.split(np.arange(len(train_subjects)), y))
    else:
        rng = np.random.RandomState(int(seed))
        perm = np.arange(len(train_subjects))
        rng.shuffle(perm)
        n_val = max(1, int(round(len(train_subjects) * float(val_ratio))))
        va_idx = np.sort(perm[:n_val])
        tr_idx = np.sort(perm[n_val:])
    return [train_subjects[i] for i in tr_idx], [train_subjects[i] for i in va_idx]


# =============================================================================
# Model factory, training, evaluation
# =============================================================================

def build_segment_model(model_name: str, *, num_classes: int, n_channels: int, raw_timepoints: int,
                        rbp_t: int, rbp_b: int, hj_t: int, emb_size: int, depth: int, device: torch.device) -> nn.Module:
    model_name = model_name.lower()
    fc = infer_eeg_fc(n_channels, raw_timepoints, emb_size, depth, device=str(device))
    print(f"[Model] inferred raw EEG fc={fc}, n_channels={n_channels}, T={raw_timepoints}, rbp_t={rbp_t}, rbp_b={rbp_b}, hj_t={hj_t}")
    # if model_name == "eeg":
    #     return EEGTransformer(emb_size=emb_size, n_channels=n_channels, fc=fc, depth=depth, n_classes=num_classes)
    if model_name == "eeg":
        model_eeg = AHEAPResNet1DWrapper(
            num_classes=num_classes,
            seq_length=raw_timepoints,
            in_channels=n_channels,
            conv_layers=[1, 1, 1, 1],   # smaller than ResNet18 [2,2,2,2]
            base_channels=16,           # much smaller than 64
            fc_stages=1,
            dropout=0.5,
            activation="relu",
        )
        return model_eeg
    if model_name == "rbp":
        return RBPFeatureExtractor(n_classes=num_classes, n_channels=n_channels, t=rbp_t, b=rbp_b)
    if model_name in {"hj", "hjorth"}:
        return HJFeatureExtractor(n_classes=num_classes, n_channels=n_channels, t=hj_t)
    if model_name == "combine":
        return AlzheimerClassifier(n_classes=num_classes, channels=n_channels, fc=fc, t=rbp_t, b=rbp_b, emb_size=emb_size, depth=depth, hj_t=hj_t)
    raise ValueError("baseline_model must be one of: eegresnet, eeg, rbp, hj, combine, ensemble_vote")


def forward_model(model: nn.Module, batch: Mapping[str, Any], model_name: str, device: torch.device) -> torch.Tensor:
    eeg = batch["eeg"].to(device, non_blocking=True)
    rbp = batch["rbp"].to(device, non_blocking=True)
    hj = batch["hj"].to(device, non_blocking=True)
    model_name = model_name.lower()
    if model_name.startswith("eeg"):
        return model(eeg)
    if model_name == "rbp":
        return model(rbp)
    if model_name in {"hj", "hjorth"}:
        return model(hj)
    return model(eeg, rbp, hj)


def compute_metrics(y_true: Sequence[int], probs: np.ndarray, num_classes: int) -> Dict[str, Any]:
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    labels = list(range(num_classes))
    return {
        "loss": float("nan"),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "macro_f1": float(f1_score(y_true_arr, y_pred, average="macro", zero_division=0)),
        "conf_matrix": confusion_matrix(y_true_arr, y_pred, labels=labels),
    }


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device, model_name: str) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_n = 0
    ys, ps = [], []
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = forward_model(model, batch, model_name, device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        n = labels.numel()
        total_loss += float(loss.detach().cpu().item()) * n
        total_n += n
        ys.extend(labels.detach().cpu().numpy().tolist())
        ps.append(torch.softmax(logits.detach(), dim=-1).cpu().numpy())
    probs = np.concatenate(ps, axis=0)
    m = compute_metrics(ys, probs, probs.shape[1])
    m["loss"] = total_loss / max(total_n, 1)
    return m


def evaluate_segments(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
                      model_name: str, *, split: str, fold: int, split_seed: int, num_classes: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    model.eval()
    rows = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            logits = forward_model(model, batch, model_name, device)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            total_loss += float(loss.detach().cpu().item()) * labels.numel()
            total_n += labels.numel()
            for i in range(labels.numel()):
                row = {
                    "split_seed": int(split_seed),
                    "fold": int(fold),
                    "split": split,
                    "source_level": "segment",
                    "subject_id": batch["subject_id"][i],
                    "segment_id": int(batch["segment_id"][i]),
                    "start_sample": int(batch["start_sample"][i]),
                    "true_label": int(labels[i].detach().cpu().item()),
                    "pred_label": int(preds[i]),
                }
                for c in range(num_classes):
                    row[f"prob_{c}"] = float(probs[i, c])
                rows.append(row)
    df = pd.DataFrame(rows)
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    metrics = compute_metrics(df["true_label"].tolist(), df[prob_cols].to_numpy(), num_classes)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics, df




def is_ensemble_vote_name(model_name: str) -> bool:
    return str(model_name).lower() in {"ensemble_vote", "voting", "soft_vote_ensemble"}


def build_ensemble_vote_models(*, num_classes: int, n_channels: int, raw_timepoints: int,
                               rbp_t: int, rbp_b: int, hj_t: int,
                               emb_size: int, depth: int, device: torch.device) -> Dict[str, nn.Module]:
    """
    Build the three independent baseline models used by the Voting baseline:
      1) raw EEG Transformer/Conformer-like branch
      2) RBP MLP branch
      3) Hjorth MLP branch

    These models are trained separately. At inference, probabilities are averaged:
        p_segment = (p_eeg + p_rbp + p_hj) / 3
    """
    return {
        "eeg": build_segment_model("eeg", num_classes=num_classes, n_channels=n_channels,
                                   raw_timepoints=raw_timepoints, rbp_t=rbp_t, rbp_b=rbp_b,
                                   hj_t=hj_t, emb_size=emb_size, depth=depth, device=device),
        "rbp": build_segment_model("rbp", num_classes=num_classes, n_channels=n_channels,
                                   raw_timepoints=raw_timepoints, rbp_t=rbp_t, rbp_b=rbp_b,
                                   hj_t=hj_t, emb_size=emb_size, depth=depth, device=device),
        "hj": build_segment_model("hj", num_classes=num_classes, n_channels=n_channels,
                                  raw_timepoints=raw_timepoints, rbp_t=rbp_t, rbp_b=rbp_b,
                                  hj_t=hj_t, emb_size=emb_size, depth=depth, device=device),
    }


def evaluate_ensemble_vote_segments(models: Mapping[str, nn.Module], loader: DataLoader, criterion: nn.Module,
                                    device: torch.device, *, split: str, fold: int, split_seed: int,
                                    num_classes: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Evaluate the requested ensemble-voting baseline.

    Step 1: get probability from each separately trained model for each segment.
    Step 2: average the three probabilities to get final segment probability.
    Step 3: return segment-level prediction table. Subject-level soft vote is handled
            by aggregate_segment_df_to_subject(...), which averages these segment probabilities.
    """
    for m in models.values():
        m.eval()

    rows = []
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)

            logits_eeg = forward_model(models["eeg"], batch, "eeg", device)
            logits_rbp = forward_model(models["rbp"], batch, "rbp", device)
            logits_hj = forward_model(models["hj"], batch, "hj", device)

            prob_eeg = torch.softmax(logits_eeg, dim=-1)
            prob_rbp = torch.softmax(logits_rbp, dim=-1)
            prob_hj = torch.softmax(logits_hj, dim=-1)
            probs_t = (prob_eeg + prob_rbp + prob_hj) / 3.0

            # This loss is only for logging; the ensemble itself is not trained jointly.
            loss = (
                criterion(logits_eeg, labels)
                + criterion(logits_rbp, labels)
                + criterion(logits_hj, labels)
            ) / 3.0

            probs = probs_t.detach().cpu().numpy()
            preds = probs.argmax(axis=1)
            total_loss += float(loss.detach().cpu().item()) * labels.numel()
            total_n += labels.numel()

            for i in range(labels.numel()):
                row = {
                    "split_seed": int(split_seed),
                    "fold": int(fold),
                    "split": split,
                    "source_level": "segment",
                    "subject_id": batch["subject_id"][i],
                    "segment_id": int(batch["segment_id"][i]),
                    "start_sample": int(batch["start_sample"][i]),
                    "true_label": int(labels[i].detach().cpu().item()),
                    "pred_label": int(preds[i]),
                }
                for c in range(num_classes):
                    row[f"prob_eeg_{c}"] = float(prob_eeg[i, c].detach().cpu().item())
                    row[f"prob_rbp_{c}"] = float(prob_rbp[i, c].detach().cpu().item())
                    row[f"prob_hj_{c}"] = float(prob_hj[i, c].detach().cpu().item())
                    row[f"prob_{c}"] = float(probs[i, c])
                rows.append(row)

    df = pd.DataFrame(rows)
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    metrics = compute_metrics(df["true_label"].tolist(), df[prob_cols].to_numpy(), num_classes)
    metrics["loss"] = total_loss / max(total_n, 1)
    return metrics, df

def aggregate_segment_df_to_subject(df: pd.DataFrame, num_classes: int, method: str = "soft_vote") -> pd.DataFrame:
    prob_cols = [f"prob_{c}" for c in range(num_classes)]
    out_rows = []
    group_cols = ["split_seed", "fold", "split", "subject_id"]
    for keys, g in df.groupby(group_cols, sort=False, dropna=False):
        probs = g[prob_cols].to_numpy(dtype=np.float64)
        if method in {"soft_vote", "mean_prob", "mean"}:
            p = probs.mean(axis=0)
        elif method == "hard_vote":
            counts = np.bincount(g["pred_label"].to_numpy(dtype=np.int64), minlength=num_classes).astype(np.float64)
            p = counts / max(counts.sum(), 1.0)
        else:
            raise ValueError(f"Unknown subject aggregation method={method}")
        row = dict(zip(group_cols, keys))
        row["source_level"] = "subject"
        row["aggregation_method"] = method
        row["true_label"] = int(g["true_label"].iloc[0])
        row["pred_label"] = int(np.argmax(p))
        row["num_segments"] = int(len(g))
        for c in range(num_classes):
            row[f"prob_{c}"] = float(p[c])
        out_rows.append(row)
    return pd.DataFrame(out_rows)


class EarlyStopper:
    def __init__(self, patience: int, start_epoch: int = 0, min_delta: float = 0.0):
        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self.best_epoch = -1

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        improved = float(val_loss) < self.best_loss - self.min_delta
        if improved:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        if epoch >= self.start_epoch:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def fit_segment_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, *, optimizer: torch.optim.Optimizer,
                      criterion: nn.Module, device: torch.device, model_name: str, epochs: int, patience: int,
                      start_epoch: int, min_delta: float) -> Tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    stopper = EarlyStopper(patience=patience, start_epoch=start_epoch, min_delta=min_delta)
    history: List[Dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, model_name)
        val_m, _ = evaluate_segments(model, val_loader, criterion, device, model_name, split="val", fold=-1, split_seed=-1,
                                     num_classes=getattr(model, "num_classes", 0) or len(set([])) + 2)
        # fix num_classes inference from logits is not available here; recompute below is okay for scalar metrics except conf labels.
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_m.items() if k != "conf_matrix"})
        row.update({f"val_{k}": v for k, v in val_m.items() if k != "conf_matrix"})
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train loss={train_m['loss']:.4f}, bal_acc={train_m['balanced_accuracy']:.4f}, "
            f"macro_f1={train_m['macro_f1']:.4f} | val loss={val_m['loss']:.4f}, "
            f"bal_acc={val_m['balanced_accuracy']:.4f}, macro_f1={val_m['macro_f1']:.4f}"
        )
        if stopper.step(epoch, val_m["loss"], model):
            print(f"Early stopping at epoch {epoch}. Best val_loss={stopper.best_loss:.6f} epoch={stopper.best_epoch}")
            break
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return model, history, {"best_epoch": stopper.best_epoch, "best_val_loss": stopper.best_loss}


# =============================================================================
# Saving / plotting
# =============================================================================

def save_history_csv(history: Sequence[Mapping[str, Any]], path: str) -> None:
    pd.DataFrame(list(history)).to_csv(path, index=False)


def save_summary_metrics_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    pd.DataFrame([{k: json.dumps(jsonable(v)) if k == "confusion_matrix" else v for k, v in r.items()} for r in rows]).to_csv(path, index=False)


def plot_confusion(metrics: Mapping[str, Any], class_names: Sequence[str], save_path: str, title: str) -> None:
    if plt is None:
        return
    cm = np.asarray(metrics["conf_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_history(history: Sequence[Mapping[str, Any]], save_path: str) -> None:
    if plt is None or len(history) == 0:
        return
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    for ax, metric in zip(axes, ["loss", "balanced_accuracy", "macro_f1"]):
        for split in ["train", "val"]:
            col = f"{split}_{metric}"
            if col in df.columns:
                ax.plot(df["epoch"], df[col], marker="o", label=split)
        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def aggregate_fold_summaries_to_seed(fold_rows: Sequence[Mapping[str, Any]], seed: int) -> Dict[str, Any]:
    df = pd.DataFrame(list(fold_rows))
    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1", "loss"]
    out = {"seed": int(seed), "num_folds": int(len(df))}
    for col in metric_cols:
        if col in df.columns:
            out[col] = float(df[col].mean())
            out[f"{col}_std_across_folds"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    cms = [np.asarray(cm) for cm in df["confusion_matrix"].tolist()] if "confusion_matrix" in df.columns else []
    if cms:
        out["confusion_matrix"] = np.sum(cms, axis=0)
    return out


def save_seed_aggregation(seed_rows: Sequence[Mapping[str, Any]], output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)
    raw = pd.DataFrame([{k: json.dumps(jsonable(v)) if k == "confusion_matrix" else v for k, v in r.items()} for r in seed_rows])
    raw.to_csv(os.path.join(output_dir, "all_seed_results.csv"), index=False)
    metric_cols = [c for c in ["accuracy", "balanced_accuracy", "macro_f1", "loss"] if c in raw.columns]
    rows = []
    for col in metric_cols:
        vals = raw[col].astype(float)
        rows.append({"metric": col, "mean": vals.mean(), "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                     "min": vals.min(), "max": vals.max(), "count": len(vals),
                     "mean_std": f"{vals.mean():.4f} ± {(vals.std(ddof=1) if len(vals) > 1 else 0.0):.4f}"})
    agg = pd.DataFrame(rows)
    agg.to_csv(os.path.join(output_dir, "aggregate_seed_results.csv"), index=False)
    return raw, agg


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(args: argparse.Namespace) -> str:
    if getattr(args, "torch_threads", None) is not None:
        torch.set_num_threads(int(args.torch_threads))
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    subject_ids, labels = list_subjects_and_labels(args.h5_path)
    num_classes = int(max(labels)) + 1
    class_names = args.class_names if args.class_names else [f"class_{i}" for i in range(num_classes)]

    shapes = get_h5_shapes(args.h5_path, args.rbp_family, args.hj_family, subject_ids[0])
    raw_shape = shapes["raw_shape"]
    rbp_shape = shapes["rbp_shape"]
    hj_shape = shapes["hj_shape"]
    n_channels, rbp_t, rbp_b, hj_t = resolve_feature_dims(rbp_shape[1:], hj_shape[1:], args.rbp_b)
    raw_timepoints = int(raw_shape[2])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_baseline_segment_{args.baseline_model}_k{args.k_folds}_val{args.val_ratio}"
    output_dir = ensure_dir(os.path.join(args.output_dir, run_name))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(jsonable(vars(args) | {"num_classes": num_classes, "shapes": shapes}), f, indent=2)

    all_seed_summaries = []
    all_fold_summary_rows = []
    for split_seed in args.split_seeds:
        set_global_seed(split_seed)
        seed_dir = ensure_dir(os.path.join(output_dir, f"seed{split_seed}"))
        folds = balanced_kfold_split(subject_ids, labels, split_seed, args.k_folds)
        fold_test_rows = []
        for fold, test_subjects in enumerate(folds):
            print(f"\n========== Seed {split_seed} | Fold {fold} ==========")
            run_dir = ensure_dir(os.path.join(seed_dir, f"fold{fold}"))
            test_set = set(test_subjects)
            trainval_subjects = [s for s in subject_ids if s not in test_set]
            label_map = dict(zip(subject_ids, labels))
            train_subjects, val_subjects = stratified_split_subjects(trainval_subjects, label_map, args.val_ratio, split_seed + fold)
            print(f"#Train subjects={len(train_subjects)} #Val={len(val_subjects)} #Test={len(test_subjects)}")

            train_ds = H5SegmentDataset(args.h5_path, train_subjects, rbp_family=args.rbp_family, hj_family=args.hj_family,
                                        train=True, segment_policy=args.train_segment_policy, base_k=args.base_k, seed=split_seed + fold)
            val_ds = H5SegmentDataset(args.h5_path, val_subjects, rbp_family=args.rbp_family, hj_family=args.hj_family)
            test_ds = H5SegmentDataset(args.h5_path, test_subjects, rbp_family=args.rbp_family, hj_family=args.hj_family)
            g = torch.Generator().manual_seed(int(split_seed) + 1234 + int(fold))
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                      pin_memory=torch.cuda.is_available(), collate_fn=segment_collate, generator=g)
            val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size or args.batch_size, shuffle=False, num_workers=args.num_workers,
                                    pin_memory=torch.cuda.is_available(), collate_fn=segment_collate)
            test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size or args.batch_size, shuffle=False, num_workers=args.num_workers,
                                     pin_memory=torch.cuda.is_available(), collate_fn=segment_collate)

            set_global_seed(args.train_seed if args.train_seed is not None else split_seed + fold)
            criterion = nn.CrossEntropyLoss()

            if is_ensemble_vote_name(args.baseline_model):
                models = build_ensemble_vote_models(
                    num_classes=num_classes,
                    n_channels=n_channels,
                    raw_timepoints=raw_timepoints,
                    rbp_t=rbp_t,
                    rbp_b=rbp_b,
                    hj_t=hj_t,
                    emb_size=args.emb_size,
                    depth=args.depth,
                    device=device,
                )
                models = {name: m.to(device) for name, m in models.items()}
                for m in models.values():
                    setattr(m, "num_classes", num_classes)

                history: Dict[str, List[Dict[str, Any]]] = {}
                best_meta: Dict[str, Any] = {}
                branch_lr = {
                    "eeg": args.lr_eeg if args.lr_eeg is not None else args.lr,
                    "rbp": args.lr_rbp if args.lr_rbp is not None else args.lr,
                    "hj": args.lr_hj if args.lr_hj is not None else args.lr,
                }

                for branch_name in ["eeg", "rbp", "hj"]:
                    print(f"\n--- Training ensemble branch: {branch_name} ---")
                    opt = torch.optim.AdamW(models[branch_name].parameters(), lr=branch_lr[branch_name], weight_decay=args.weight_decay)
                    models[branch_name], hist_b, meta_b = fit_segment_model(
                        models[branch_name],
                        train_loader,
                        val_loader,
                        optimizer=opt,
                        criterion=criterion,
                        device=device,
                        model_name=branch_name,
                        epochs=args.epochs,
                        patience=args.patience,
                        start_epoch=args.start_epoch,
                        min_delta=args.min_delta,
                    )
                    history[branch_name] = hist_b
                    best_meta[branch_name] = meta_b
                    save_history_csv(hist_b, os.path.join(run_dir, f"history_{branch_name}.csv"))

                torch.save(
                    {
                        "model_state_dicts": {name: m.state_dict() for name, m in models.items()},
                        "best_meta": best_meta,
                        "args": vars(args),
                        "ensemble_rule": "segment_prob = mean(softmax(eeg), softmax(rbp), softmax(hj)); subject_prob = mean(segment_prob)",
                    },
                    os.path.join(run_dir, "best_model.pt"),
                )
                model_for_eval = models
            else:
                model = build_segment_model(args.baseline_model, num_classes=num_classes, n_channels=n_channels, raw_timepoints=raw_timepoints,
                                            rbp_t=rbp_t, rbp_b=rbp_b, hj_t=hj_t, emb_size=args.emb_size, depth=args.depth, device=device).to(device)
                setattr(model, "num_classes", num_classes)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                model, history, best_meta = fit_segment_model(model, train_loader, val_loader, optimizer=optimizer, criterion=criterion,
                                                              device=device, model_name=args.baseline_model, epochs=args.epochs,
                                                              patience=args.patience, start_epoch=args.start_epoch, min_delta=args.min_delta)
                torch.save({"model_state_dict": model.state_dict(), "best_meta": best_meta, "args": vars(args)}, os.path.join(run_dir, "best_model.pt"))
                model_for_eval = model

            split_outputs = {}
            summary_rows = []
            for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
                if is_ensemble_vote_name(args.baseline_model):
                    seg_metrics, seg_df = evaluate_ensemble_vote_segments(
                        model_for_eval,
                        loader,
                        criterion,
                        device,
                        split=split_name,
                        fold=fold,
                        split_seed=split_seed,
                        num_classes=num_classes,
                    )
                else:
                    seg_metrics, seg_df = evaluate_segments(model_for_eval, loader, criterion, device, args.baseline_model, split=split_name,
                                                            fold=fold, split_seed=split_seed, num_classes=num_classes)
                subj_df = aggregate_segment_df_to_subject(seg_df, num_classes=num_classes, method=args.subject_aggregation)
                subj_metrics = compute_metrics(subj_df["true_label"].tolist(), subj_df[[f"prob_{c}" for c in range(num_classes)]].to_numpy(), num_classes)
                subj_metrics["loss"] = float("nan")
                seg_df.to_csv(os.path.join(run_dir, f"{split_name}_segment_predictions.csv"), index=False)
                subj_df.to_csv(os.path.join(run_dir, f"{split_name}_subject_predictions.csv"), index=False)
                split_outputs[(split_name, "segment")] = seg_metrics
                split_outputs[(split_name, "subject")] = subj_metrics
                for level, m in [("segment", seg_metrics), ("subject", subj_metrics)]:
                    summary_rows.append({
                        "split": split_name,
                        "level": level,
                        "fold": fold,
                        "loss": float(m["loss"]),
                        "accuracy": float(m["accuracy"]),
                        "balanced_accuracy": float(m["balanced_accuracy"]),
                        "macro_f1": float(m["macro_f1"]),
                        "confusion_matrix": m["conf_matrix"],
                    })

            if isinstance(history, Mapping):
                # For ensemble_vote each branch has its own history file; also save a compact marker.
                pd.DataFrame([{"branch": k, "num_epochs": len(v)} for k, v in history.items()]).to_csv(os.path.join(run_dir, "history.csv"), index=False)
            else:
                save_history_csv(history, os.path.join(run_dir, "history.csv"))
            save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))
            if not isinstance(history, Mapping):
                plot_history(history, os.path.join(run_dir, "history.png"))
            plot_confusion(split_outputs[("test", "subject")], class_names, os.path.join(run_dir, "test_subject_confusion_matrix.png"), "Test subject confusion")
            plot_confusion(split_outputs[("test", "segment")], class_names, os.path.join(run_dir, "test_segment_confusion_matrix.png"), "Test segment confusion")

            test_subject_metrics = split_outputs[("test", "subject")]
            summary_test = {
                "encoder_type": f"baseline_{args.baseline_model}",
                "training_approach": "segment_based",
                "subject_aggregation": args.subject_aggregation,
                "split_seed": int(split_seed),
                "fold": int(fold),
                "loss": float(test_subject_metrics["loss"]),
                "accuracy": float(test_subject_metrics["accuracy"]),
                "balanced_accuracy": float(test_subject_metrics["balanced_accuracy"]),
                "macro_f1": float(test_subject_metrics["macro_f1"]),
                "confusion_matrix": test_subject_metrics["conf_matrix"],
                "baseline_model": args.baseline_model,
                "base_k": args.base_k,
                "train_segment_policy": args.train_segment_policy,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "patience": args.patience,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            }
            save_summary_metrics_csv([summary_test], os.path.join(run_dir, "summary_test.csv"))
            fold_test_rows.append(summary_test)
            all_fold_summary_rows.append(summary_test)

        seed_summary = aggregate_fold_summaries_to_seed(fold_test_rows, seed=split_seed)
        all_seed_summaries.append(seed_summary)

    save_summary_metrics_csv(all_fold_summary_rows, os.path.join(output_dir, "fold_metrics_all_seeds.csv"))
    _, agg_df = save_seed_aggregation(all_seed_summaries, os.path.join(output_dir, "agg_seed_results"))
    agg_df.to_csv(os.path.join(output_dir, "overall_summary_test.csv"), index=False)
    print("\nAggregate across seeds:")
    print(agg_df)
    return output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment-level H5 baseline runner aligned with mil_full_std outputs.")
    p.add_argument("--h5_path", default="/home/anphan/Documents/aheap_master.h5")
    p.add_argument("--output_dir", default="/home/anphan/Documents/AHEAP_data/baseline_resnet")

    p.add_argument("--baseline_model", choices=["eeg", "eegresnet", "rbp", "hj", "combine", "ensemble_vote", "voting"], default="eeg")
    p.add_argument("--rbp_family", default="relative_band_power")
    p.add_argument("--hj_family", default="hjorth")
    p.add_argument("--rbp_b", type=int, default=None, help="RBP band count. Default: infer 5 when possible, else use feature dim.")
    p.add_argument("--split_seeds", type=int, nargs="+", default=[15, 42, 100])
    p.add_argument("--train_seed", type=int, default=None)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--train_segment_policy", choices=["all", "random_k"], default="random_k")
    p.add_argument("--base_k", type=int, default=10, help="Used only with --train_segment_policy random_k.")
    p.add_argument("--subject_aggregation", choices=["soft_vote", "hard_vote"], default="soft_vote")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--start_epoch", type=int, default=100)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4, help="Default LR. Also used for combine/eeg unless branch-specific LR is set.")
    p.add_argument("--lr_eeg", type=float, default=1e-4)
    p.add_argument("--lr_rbp", type=float, default=3e-3)
    p.add_argument("--lr_hj", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--emb_size", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--torch_threads", type=int, default=None, help="Optional CPU torch thread cap, e.g. 1 for debugging on CPU.")
    p.add_argument("--class_names", nargs="*", default=None)
    return p.parse_args()


if __name__ == "__main__":
    out = run_experiment(parse_args())
    print(f"Saved results to: {out}")
