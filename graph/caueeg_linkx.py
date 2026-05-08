# graph/build_caueeg_randomcrop_master.py

import os
import json
import random
import hashlib
from pathlib import Path
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from caueeg.caueeg_script import (
    load_caueeg_config,
    load_caueeg_task_datasets,
    calculate_signal_statistics,
)
from caueeg.pipeline import (
    EegRandomCrop,
    EegDropChannels,
    EegToTensor,
    eeg_collate_fn,
)
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from master_builder import build_master_eeg_dataset, DEFAULT_BANDS, FEATURE_REGISTRY, CONNECTIVITY_REGISTRY
CAUEEG_BI23_PAIRS = [
    ("F7", "Fp1"),
    ("F3", "Fp1"),
    ("Fp1", "FZ"), 
    ("F8", "Fp2"), 
    ("F4", "Fp2"),
    ("FZ", "Fp2"),
    ("Fp1", "Fp2"), 
    ("F7", "F3"),
    ("F3", "FZ"),
    ("F4", "FZ"), 
    ("F8", "F4"), 
    ("F3", "C3"),
    ("FZ", "CZ"),
    ("F4", "C4"), 
    ("C3", "T3"),
    ("C3", "CZ"),
    ("C4", "CZ"),
    ("C4", "T4"), 
    ("T3", "T5"),
    ("C3", "P3"),
    ("CZ", "PZ"),
    ("C4", "P4"), 
    ("T4", "T6"), 
    ("P3", "PZ"),
    ("P4", "PZ"),
    ("P4", "O2"), 
    ("P3", "O1"),
    ("T6", "O2"), 
    ("T5", "O1"),
    ("O1", "O2")
]

# This list above has 23 pairs.
CAUEEG_BI23_NAMES = [f"{a}-{b}" for a, b in CAUEEG_BI23_PAIRS]

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_int(x: str) -> int:
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)

def segment_recording(signal, crop_len=2000, overlap=0.5, latency=2000):
    x = np.asarray(signal, dtype=np.float32)
    x = x[:19]  # keep EEG only
    step = int(crop_len * (1.0 - overlap))
    total_len = x.shape[-1]
    starts = list(range(latency, total_len - crop_len + 1, step))
    windows = [x[:, s:s + crop_len].astype(np.float32, copy=False) for s in starts]
    return windows, starts

# ---------------------------------------------------------
# Channel helpers
# ---------------------------------------------------------
def strip_avg_suffix(ch: str) -> str:
    ch = str(ch).strip()
    return ch[:-4] if ch.endswith("-AVG") else ch


def get_caueeg_channel_info(dataset_path: str):
    config = load_caueeg_config(dataset_path)
    signal_header = list(config["signal_header"])

    idx_ekg = signal_header.index("EKG")
    idx_photic = signal_header.index("Photic")
    drop_idx = [idx_ekg, idx_photic]

    eeg19_raw = [ch for ch in signal_header if ch not in ["EKG", "Photic"]]
    eeg19_clean = [strip_avg_suffix(ch) for ch in eeg19_raw]

    return config, signal_header, eeg19_clean, drop_idx


# ---------------------------------------------------------
# Official-style stats computation
# ---------------------------------------------------------
def build_stats_transform(crop_length: int, latency: int, drop_idx):
    return transforms.Compose([
        EegRandomCrop(
            crop_length=crop_length,
            length_limit=10**7,
            multiple=1,
            latency=latency,
            segment_simulation=False,
            return_timing=False,
        ),
        EegDropChannels(drop_idx),
        EegToTensor(),
    ])

def compute_train_signal_stats(
    dataset_path: str,
    task: str,
    file_format: str,
    crop_length: int,
    latency: int,
    seed: int,
):
    set_global_seed(seed)
    _, _, _, drop_idx = get_caueeg_channel_info(dataset_path)

    stats_transform = build_stats_transform(
        crop_length=crop_length,
        latency=latency,
        drop_idx=drop_idx,
    )

    _, train_set, _, _ = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=stats_transform,
        verbose=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=eeg_collate_fn,
    )

    mean_sum = None
    std_sum = None
    n_count = 0

    for _ in range(5):
        for sample in train_loader:
            signal = sample["signal"]                      # [N, C, T]
            std, mean = torch.std_mean(signal, dim=-1, keepdim=True)   # [N, C, 1]

            mean_batch = mean.sum(dim=0, keepdim=True)    # [1, C, 1]
            std_batch = std.sum(dim=0, keepdim=True)      # [1, C, 1]

            if mean_sum is None:
                mean_sum = torch.zeros_like(mean_batch)
                std_sum = torch.zeros_like(std_batch)

            mean_sum += mean_batch
            std_sum += std_batch
            n_count += signal.shape[0]

    signal_mean = mean_sum / n_count
    signal_std = std_sum / n_count

    return signal_mean.detach().cpu().numpy(), signal_std.detach().cpu().numpy()
def compute_train_crops_per_record(train_dataset, target_total_train_crops: int) -> tuple[int, int]:
    if target_total_train_crops is None or int(target_total_train_crops) <= 0:
        raise ValueError("target_total_train_crops must be a positive integer.")

    n_train_records = len(train_dataset)
    if n_train_records <= 0:
        raise ValueError("Training dataset is empty.")

    train_crops_per_record = math.ceil(int(target_total_train_crops) / n_train_records)
    actual_total_train_crops = train_crops_per_record * n_train_records
    return train_crops_per_record, actual_total_train_crops
# ---------------------------------------------------------
# Deterministic random crop bank
# ---------------------------------------------------------
def sample_random_crop_starts(
    signal_len: int,
    crop_length: int,
    latency: int,
    n_crops: int,
    serial: str,
    split_name: str,
    seed: int,
):
    hi = signal_len - crop_length
    if hi <= latency:
        return []

    rng = np.random.default_rng(seed + stable_int(f"{split_name}:{serial}"))
    starts = rng.integers(low=latency, high=hi, size=n_crops, endpoint=False)
    return [int(x) for x in starts]


def normalize_crop_dataset_mode(x: np.ndarray, signal_mean: np.ndarray, signal_std: np.ndarray):
    # x shape: [C, T]
    # signal_mean/std from official stats code have broadcastable shape [1, C, 1]
    return ((x - signal_mean.squeeze(0)) / (signal_std.squeeze(0) + 1e-8)).astype(np.float32)


def build_subject_records_from_split(
    dataset,
    split_name: str,
    eeg19_clean,
    drop_idx,
    signal_mean,
    signal_std,
    crop_length: int,
    latency: int,
    n_crops_per_record: int,
    seed: int,
):
    records = []
    kept_ids = []

    for sample in dataset:
        signal = np.asarray(sample["signal"], dtype=np.float32)  # [21, T]
        serial = str(sample["serial"])
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        # drop EKG + photic
        signal = np.delete(signal, drop_idx, axis=0)  # [19, T]

        starts = sample_random_crop_starts(
            signal_len=signal.shape[-1],
            crop_length=crop_length,
            latency=latency,
            n_crops=n_crops_per_record,
            serial=serial,
            split_name=split_name,
            seed=seed,
        )

        if len(starts) == 0:
            continue

        windows = []
        segment_ids = []
        segment_meta = []

        for i, st in enumerate(starts):
            crop = signal[:, st:st + crop_length].astype(np.float32, copy=False)
            crop = normalize_crop_dataset_mode(crop, signal_mean, signal_std)

            windows.append(crop)
            segment_ids.append(i)
            segment_meta.append({
                "serial": serial,
                "split": split_name,
                "crop_start": st,
                "crop_length": crop_length,
                "latency": latency,
                "normalization": "dataset",
            })

        rec = {
            "subject_id": f"{split_name}_{serial}",
            "label": label,
            "class_id": label,
            "sampling_rate": 200.0,
            "channel_names": eeg19_clean,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": segment_ids,
            "segment_metadata": segment_meta,
            "recording_info": {
                "serial": serial,
                "age": age,
                "split": split_name,
                "n_random_crops": len(windows),
            },
        }
        records.append(rec)
        kept_ids.append(f"{split_name}_{serial}")

    return records, kept_ids


# ---------------------------------------------------------
# Main builder
# ---------------------------------------------------------
def build_caueeg_randomcrop_master(
    dataset_path: str,
    task: str = "dementia",
    file_format: str = "feather",
    output_h5_path: str = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master.h5",
    seed: int = 42,
    crop_length: int = 2000,
    latency: int = 2000,
    target_total_train_crops: int = 100_000,
    eval_crops_per_record: int = 8,
    feature_families=None,
    connectivity_metrics=None,
    overwrite: bool = False,
):
    set_global_seed(seed)

    if feature_families is None:
        feature_families = list(FEATURE_REGISTRY.keys())

    if connectivity_metrics is None:
        connectivity_metrics = list(CONNECTIVITY_REGISTRY.keys())

    config, signal_header, eeg19_clean, drop_idx = get_caueeg_channel_info(dataset_path)

    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )
    train_crops_per_record, actual_total_train_crops = compute_train_crops_per_record(
        train_dataset=train_set,
        target_total_train_crops=target_total_train_crops,
    )

    print(f"Requested total train crops: {target_total_train_crops}")
    print(f"Train recordings: {len(train_set)}")
    print(f"Computed train_crops_per_record: {train_crops_per_record}")
    print(f"Actual total stored train crops: {actual_total_train_crops}")
    print(f"Eval crops per record: {eval_crops_per_record}")
    signal_mean, signal_std = compute_train_signal_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
    )

    train_records, train_ids = build_subject_records_from_split(
        train_set, "train",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        n_crops_per_record=train_crops_per_record,
        seed=seed,
    )
    val_records, val_ids = build_subject_records_from_split(
        val_set, "val",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        n_crops_per_record=eval_crops_per_record,
        seed=seed,
    )
    test_records, test_ids = build_subject_records_from_split(
        test_set, "test",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        n_crops_per_record=eval_crops_per_record,
        seed=seed,
    )

    all_records = train_records + val_records + test_records
    meta_json = output_h5_path.replace(".h5", "_meta.json")
    with open(meta_json, "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "task": task,
                "file_format": file_format,
                "seed": seed,
                "crop_length": crop_length,
                "latency": latency,
                "input_norm": "dataset",
                "target_total_train_crops": int(target_total_train_crops),
                "train_crops_per_record": int(train_crops_per_record),
                "actual_total_train_crops": int(actual_total_train_crops),
                "eval_crops_per_record": int(eval_crops_per_record),
                "bands": {k: list(v) for k, v in DEFAULT_BANDS.items()},
                "feature_families": list(feature_families),
                "connectivity_metrics": list(connectivity_metrics),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            f,
            indent=2,
        )
    np.savez(
        output_h5_path.replace(".h5", "_dataset_norm_stats.npz"),
        signal_mean=signal_mean,
        signal_std=signal_std,
    )

    build_master_eeg_dataset(
        subject_records=all_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=DEFAULT_BANDS,
        overwrite=overwrite,
        skip_bad_segments=False,
        target_sampling_rate=None,
        qc_input_unit="auto",
    )

    total_windows_all = sum(len(r["windows"]) for r in all_records)
    total_windows_train = sum(len(r["windows"]) for r in train_records)
    total_windows_val = sum(len(r["windows"]) for r in val_records)
    total_windows_test = sum(len(r["windows"]) for r in test_records)

    print(f"Saved H5: {output_h5_path}")
    print(f"Saved meta: {meta_json}")
    print(f"Saved norm stats: {output_h5_path.replace('.h5', '_dataset_norm_stats.npz')}")
    print(f"Train records: {len(train_records)} | stored train crops: {total_windows_train}")
    print(f"Val records: {len(val_records)} | stored val crops: {total_windows_val}")
    print(f"Test records: {len(test_records)} | stored test crops: {total_windows_test}")
    print(f"Total crop windows stored: {total_windows_all}")






def sample_sliding_window_starts(
    signal_len: int,
    crop_length: int,
    latency: int,
    overlap: float,
):
    """
    Deterministic sliding-window starts.

    Example:
      crop_length=2000, overlap=0.5
      step=1000
      starts = 2000, 3000, 4000, ...
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    step = int(round(crop_length * (1.0 - overlap)))
    if step <= 0:
        raise ValueError(f"Invalid step={step}. Check crop_length={crop_length}, overlap={overlap}")

    last_start = signal_len - crop_length
    if last_start < latency:
        return []

    starts = list(range(int(latency), int(last_start) + 1, step))
    return [int(s) for s in starts]


def build_sliding_subject_records_from_split(
    dataset,
    split_name: str,
    eeg19_clean,
    drop_idx,
    signal_mean,
    signal_std,
    crop_length: int,
    latency: int,
    overlap: float,
):
    """
    Same schema as build_subject_records_from_split(...), but using sliding starts.

    Output subject IDs:
      train_00001
      val_00001
      test_00001
    """
    records = []
    kept_ids = []

    for sample in dataset:
        signal = np.asarray(sample["signal"], dtype=np.float32)  # [21, T]
        serial = str(sample["serial"])
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        # Important: use the same channel dropping method as the random builder.
        signal = np.delete(signal, drop_idx, axis=0)  # [19, T]

        starts = sample_sliding_window_starts(
            signal_len=signal.shape[-1],
            crop_length=crop_length,
            latency=latency,
            overlap=overlap,
        )

        if len(starts) == 0:
            continue

        windows = []
        segment_ids = []
        segment_meta = []

        for i, st in enumerate(starts):
            crop = signal[:, st:st + crop_length].astype(np.float32, copy=False)

            # Important: same normalization as random-crop H5.
            crop = normalize_crop_dataset_mode(crop, signal_mean, signal_std)

            windows.append(crop)
            segment_ids.append(i)
            segment_meta.append({
                "serial": serial,
                "split": split_name,
                "crop_start": int(st),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
                "step": int(round(crop_length * (1.0 - overlap))),
                "window_mode": "sliding",
                "normalization": "dataset",
            })

        sid = f"{split_name}_{serial}"

        rec = {
            "subject_id": sid,
            "label": label,
            "class_id": label,
            "sampling_rate": 200.0,
            "channel_names": eeg19_clean,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": segment_ids,
            "segment_metadata": segment_meta,
            "recording_info": {
                "serial": serial,
                "age": age,
                "split": split_name,
                "window_mode": "sliding",
                "n_sliding_windows": len(windows),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
            },
        }

        records.append(rec)
        kept_ids.append(sid)

    return records, kept_ids


def load_or_compute_norm_stats(
    dataset_path: str,
    task: str,
    file_format: str,
    crop_length: int,
    latency: int,
    seed: int,
    norm_stats_npz: Optional[str] = None,
):
    """
    To make sliding and random H5 perfectly compatible, reuse the random H5
    normalization stats if possible.
    """
    if norm_stats_npz is not None and os.path.exists(norm_stats_npz):
        stats = np.load(norm_stats_npz)
        signal_mean = stats["signal_mean"]
        signal_std = stats["signal_std"]
        print(f"Loaded existing norm stats: {norm_stats_npz}")
        return signal_mean, signal_std

    print("Norm stats file not provided/found. Recomputing stats.")
    return compute_train_signal_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
    )


def build_caueeg_sliding_master_compatible(
    dataset_path: str,
    task: str = "dementia",
    file_format: str = "feather",
    output_h5_path: str = "/mnt/data/anphan/CAUEEG/caueeg_sliding_master_dementia_compatible.h5",
    seed: int = 42,
    crop_length: int = 2000,
    latency: int = 2000,
    overlap: float = 0.5,
    feature_families=None,
    connectivity_metrics=None,
    norm_stats_npz: Optional[str] = None,
    overwrite: bool = False,
):
    """
    Sliding-window H5 builder compatible with your random-crop H5.

    This fixes:
      1. subject IDs: train_00001 / val_00001 / test_00001
      2. same normalization as random H5
      3. same feature families
      4. same connectivity metrics
      5. same H5 schema
    """
    set_global_seed(seed)

    if feature_families is None:
        feature_families = list(FEATURE_REGISTRY.keys())

    if connectivity_metrics is None:
        connectivity_metrics = list(CONNECTIVITY_REGISTRY.keys())

    config, signal_header, eeg19_clean, drop_idx = get_caueeg_channel_info(dataset_path)

    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    signal_mean, signal_std = load_or_compute_norm_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
        norm_stats_npz=norm_stats_npz,
    )

    train_records, train_ids = build_sliding_subject_records_from_split(
        train_set,
        "train",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
    )

    val_records, val_ids = build_sliding_subject_records_from_split(
        val_set,
        "val",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
    )

    test_records, test_ids = build_sliding_subject_records_from_split(
        test_set,
        "test",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
    )

    all_records = train_records + val_records + test_records

    meta_json = output_h5_path.replace(".h5", "_meta.json")
    with open(meta_json, "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "task": task,
                "file_format": file_format,
                "seed": seed,
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
                "step": int(round(crop_length * (1.0 - overlap))),
                "window_mode": "sliding",
                "input_norm": "dataset",
                "norm_stats_npz_source": norm_stats_npz,
                "bands": {k: list(v) for k, v in DEFAULT_BANDS.items()},
                "feature_families": list(feature_families),
                "connectivity_metrics": list(connectivity_metrics),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            f,
            indent=2,
        )

    np.savez(
        output_h5_path.replace(".h5", "_dataset_norm_stats.npz"),
        signal_mean=signal_mean,
        signal_std=signal_std,
    )

    build_master_eeg_dataset(
        subject_records=all_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=DEFAULT_BANDS,
        overwrite=overwrite,
        skip_bad_segments=False,
        target_sampling_rate=None,
        qc_input_unit="auto",
    )

    total_train = sum(len(r["windows"]) for r in train_records)
    total_val = sum(len(r["windows"]) for r in val_records)
    total_test = sum(len(r["windows"]) for r in test_records)
    total_all = total_train + total_val + total_test

    print(f"Saved sliding H5: {output_h5_path}")
    print(f"Saved meta: {meta_json}")
    print(f"Saved norm stats: {output_h5_path.replace('.h5', '_dataset_norm_stats.npz')}")
    print(f"Train records: {len(train_records)} | sliding windows: {total_train}")
    print(f"Val records:   {len(val_records)} | sliding windows: {total_val}")
    print(f"Test records:  {len(test_records)} | sliding windows: {total_test}")
    print(f"Total sliding windows stored: {total_all}")

def make_bipolar_signal_from_mono(
    mono_signal: np.ndarray,
    mono_channel_names,
    bipolar_pairs=CAUEEG_BI23_PAIRS,
):
    """
    mono_signal: [19, T]
    return: bipolar_signal [23, T]
    """
    mono_signal = np.asarray(mono_signal, dtype=np.float32)
    name_to_idx = {str(ch): i for i, ch in enumerate(mono_channel_names)}

    bipolar = []
    bipolar_names = []

    for a, b in bipolar_pairs:
        if a not in name_to_idx or b not in name_to_idx:
            raise KeyError(f"Missing channel for bipolar pair {a}-{b}")

        x = mono_signal[name_to_idx[a]] - mono_signal[name_to_idx[b]]
        bipolar.append(x.astype(np.float32, copy=False))
        bipolar_names.append(f"{a}-{b}")

    return np.stack(bipolar, axis=0).astype(np.float32), bipolar_names


def raw21_to_mono19(signal, drop_idx):
    """
    raw CAUEEG signal [21, T] -> mono EEG [19, T]
    """
    signal = np.asarray(signal, dtype=np.float32)
    return np.delete(signal, drop_idx, axis=0).astype(np.float32, copy=False)


def raw21_to_bi23(signal, drop_idx, mono_channel_names):
    """
    raw CAUEEG signal [21, T] -> bipolar EEG [23, T]
    """
    mono = raw21_to_mono19(signal, drop_idx)
    bi23, bi23_names = make_bipolar_signal_from_mono(
        mono,
        mono_channel_names=mono_channel_names,
        bipolar_pairs=CAUEEG_BI23_PAIRS,
    )
    return bi23, bi23_names


def compute_train_signal_stats_bipolar(
    dataset_path: str,
    task: str,
    file_format: str,
    crop_length: int,
    latency: int,
    seed: int,
    n_passes: int = 5,
):
    """
    Compute dataset normalization stats on bipolar random crops.

    Output:
      signal_mean: [1, 23, 1]
      signal_std:  [1, 23, 1]
    """
    set_global_seed(seed)

    _, _, mono19_names, drop_idx = get_caueeg_channel_info(dataset_path)

    _, train_set, _, _ = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    mean_sum = None
    std_sum = None
    n_count = 0

    for pass_id in range(n_passes):
        for sample in train_set:
            signal = np.asarray(sample["signal"], dtype=np.float32)
            serial = str(sample["serial"])

            bi23, _ = raw21_to_bi23(
                signal,
                drop_idx=drop_idx,
                mono_channel_names=mono19_names,
            )

            starts = sample_random_crop_starts(
                signal_len=bi23.shape[-1],
                crop_length=crop_length,
                latency=latency,
                n_crops=1,
                serial=serial,
                split_name=f"stats_pass_{pass_id}",
                seed=seed,
            )

            if len(starts) == 0:
                continue

            st = starts[0]
            crop = bi23[:, st:st + crop_length]  # [23, T]

            crop_t = torch.tensor(crop, dtype=torch.float32).unsqueeze(0)  # [1, 23, T]
            std, mean = torch.std_mean(crop_t, dim=-1, keepdim=True)       # [1, 23, 1]

            if mean_sum is None:
                mean_sum = torch.zeros_like(mean)
                std_sum = torch.zeros_like(std)

            mean_sum += mean
            std_sum += std
            n_count += 1

    if n_count == 0:
        raise RuntimeError("No crops were found when computing bipolar normalization stats.")

    signal_mean = mean_sum / n_count
    signal_std = std_sum / n_count

    return signal_mean.cpu().numpy(), signal_std.cpu().numpy()

def build_subject_records_from_split_bipolar(
    dataset,
    split_name: str,
    mono19_names,
    drop_idx,
    signal_mean,
    signal_std,
    crop_length: int,
    latency: int,
    n_crops_per_record: int,
    seed: int,
):
    records = []
    kept_ids = []

    for sample in dataset:
        signal = np.asarray(sample["signal"], dtype=np.float32)  # [21, T]
        serial = str(sample["serial"])
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        # raw [21, T] -> bipolar [23, T]
        signal, bi23_names = raw21_to_bi23(
            signal,
            drop_idx=drop_idx,
            mono_channel_names=mono19_names,
        )

        starts = sample_random_crop_starts(
            signal_len=signal.shape[-1],
            crop_length=crop_length,
            latency=latency,
            n_crops=n_crops_per_record,
            serial=serial,
            split_name=split_name,
            seed=seed,
        )

        if len(starts) == 0:
            continue

        windows = []
        segment_ids = []
        segment_meta = []

        for i, st in enumerate(starts):
            crop = signal[:, st:st + crop_length].astype(np.float32, copy=False)
            crop = normalize_crop_dataset_mode(crop, signal_mean, signal_std)

            windows.append(crop)
            segment_ids.append(i)
            segment_meta.append({
                "serial": serial,
                "split": split_name,
                "crop_start": int(st),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "normalization": "dataset_bipolar",
                "montage": "bi23",
            })

        sid = f"{split_name}_{serial}"

        rec = {
            "subject_id": sid,
            "label": label,
            "class_id": label,
            "sampling_rate": 200.0,
            "channel_names": bi23_names,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": segment_ids,
            "segment_metadata": segment_meta,
            "recording_info": {
                "serial": serial,
                "age": age,
                "split": split_name,
                "n_random_crops": len(windows),
                "montage": "bi23",
            },
        }

        records.append(rec)
        kept_ids.append(sid)

    return records, kept_ids
def build_caueeg_combined_sliding_random_master(
    dataset_path,
    task="dementia",
    file_format="feather",
    output_h5_path="/mnt/data/anphan/CAUEEG/caueeg_combined_sliding_random_dementia_seed42.h5",
    seed=42,
    crop_length=2000,
    latency=2000,
    overlap=0.5,
    target_total_train_random_crops=100000,
    eval_random_crops_per_record=0,
    eval_use_random=False,
    feature_families=None,
    connectivity_metrics=None,
    overwrite=False,
):
    """
    Build one H5 containing sliding + random windows.

    Recommended:
      train: sliding + random
      val:   sliding only
      test:  sliding only

    To do random TTA for val/test, set:
      eval_use_random=True
      eval_random_crops_per_record=8
    """
    set_global_seed(seed)

    if feature_families is None:
        feature_families = list(FEATURE_REGISTRY.keys())

    if connectivity_metrics is None:
        connectivity_metrics = list(CONNECTIVITY_REGISTRY.keys())

    config, signal_header, eeg19_clean, drop_idx = get_caueeg_channel_info(dataset_path)

    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    train_random_crops_per_record, actual_total_train_random_crops = compute_train_crops_per_record(
        train_dataset=train_set,
        target_total_train_crops=target_total_train_random_crops,
    )

    print(f"Window mode: sliding + random")
    print(f"Sliding overlap: {overlap}")
    print(f"Train records: {len(train_set)}")
    print(f"Train random crops per record: {train_random_crops_per_record}")
    print(f"Actual total train random crops before duplicate removal: {actual_total_train_random_crops}")
    print(f"Eval use random: {eval_use_random}")
    print(f"Eval random crops per record: {eval_random_crops_per_record}")

    signal_mean, signal_std = compute_train_signal_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
    )

    train_records, train_ids = build_subject_records_from_split_combined(
        train_set,
        "train",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
        n_random_crops_per_record=train_random_crops_per_record,
        seed=seed,
        use_sliding=True,
        use_random=True,
    )

    val_records, val_ids = build_subject_records_from_split_combined(
        val_set,
        "val",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
        n_random_crops_per_record=eval_random_crops_per_record,
        seed=seed,
        use_sliding=True,
        use_random=eval_use_random,
    )

    test_records, test_ids = build_subject_records_from_split_combined(
        test_set,
        "test",
        eeg19_clean=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        overlap=overlap,
        n_random_crops_per_record=eval_random_crops_per_record,
        seed=seed,
        use_sliding=True,
        use_random=eval_use_random,
    )

    all_records = train_records + val_records + test_records

    meta_json = output_h5_path.replace(".h5", "_meta.json")
    with open(meta_json, "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "task": task,
                "file_format": file_format,
                "seed": int(seed),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "overlap": float(overlap),
                "step": int(round(crop_length * (1.0 - overlap))),
                "window_mode": "sliding_plus_random",
                "train_mode": "sliding_plus_random",
                "eval_mode": "sliding_plus_random" if eval_use_random else "sliding_only",
                "input_norm": "dataset",
                "target_total_train_random_crops": int(target_total_train_random_crops),
                "train_random_crops_per_record": int(train_random_crops_per_record),
                "actual_total_train_random_crops_before_dedup": int(actual_total_train_random_crops),
                "eval_random_crops_per_record": int(eval_random_crops_per_record),
                "bands": {k: list(v) for k, v in DEFAULT_BANDS.items()},
                "feature_families": list(feature_families),
                "connectivity_metrics": list(connectivity_metrics),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            f,
            indent=2,
        )

    np.savez(
        output_h5_path.replace(".h5", "_dataset_norm_stats.npz"),
        signal_mean=signal_mean,
        signal_std=signal_std,
    )

    build_master_eeg_dataset(
        subject_records=all_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=DEFAULT_BANDS,
        overwrite=overwrite,
        skip_bad_segments=False,
        target_sampling_rate=None,
        qc_input_unit="auto",
    )

    total_train = sum(len(r["windows"]) for r in train_records)
    total_val = sum(len(r["windows"]) for r in val_records)
    total_test = sum(len(r["windows"]) for r in test_records)
    total_all = total_train + total_val + total_test

    print(f"Saved combined H5: {output_h5_path}")
    print(f"Saved meta: {meta_json}")
    print(f"Saved norm stats: {output_h5_path.replace('.h5', '_dataset_norm_stats.npz')}")
    print(f"Train records: {len(train_records)} | combined windows: {total_train}")
    print(f"Val records:   {len(val_records)} | windows: {total_val}")
    print(f"Test records:  {len(test_records)} | windows: {total_test}")
    print(f"Total windows stored: {total_all}")



def build_caueeg_randomcrop_master_bipolar23(
    dataset_path: str,
    task: str = "dementia",
    file_format: str = "feather",
    output_h5_path: str = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_bi23_seed42.h5",
    seed: int = 42,
    crop_length: int = 2000,
    latency: int = 2000,
    target_total_train_crops: int = 100_000,
    eval_crops_per_record: int = 8,
    feature_families=None,
    connectivity_metrics=None,
    overwrite: bool = False,
):
    set_global_seed(seed)

    if feature_families is None:
        feature_families = list(FEATURE_REGISTRY.keys())

    if connectivity_metrics is None:
        connectivity_metrics = list(CONNECTIVITY_REGISTRY.keys())

    config, signal_header, mono19_names, drop_idx = get_caueeg_channel_info(dataset_path)

    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    train_crops_per_record, actual_total_train_crops = compute_train_crops_per_record(
        train_dataset=train_set,
        target_total_train_crops=target_total_train_crops,
    )

    print(f"Montage: bipolar 23")
    print(f"Requested total train crops: {target_total_train_crops}")
    print(f"Train recordings: {len(train_set)}")
    print(f"Computed train_crops_per_record: {train_crops_per_record}")
    print(f"Actual total stored train crops: {actual_total_train_crops}")
    print(f"Eval crops per record: {eval_crops_per_record}")

    signal_mean, signal_std = compute_train_signal_stats_bipolar(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
        n_passes=5,
    )

    train_records, train_ids = build_subject_records_from_split_bipolar(
        train_set,
        "train",
        mono19_names=mono19_names,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        n_crops_per_record=train_crops_per_record,
        seed=seed,
    )

    val_records, val_ids = build_subject_records_from_split_bipolar(
        val_set,
        "val",
        mono19_names=mono19_names,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        n_crops_per_record=eval_crops_per_record,
        seed=seed,
    )

    test_records, test_ids = build_subject_records_from_split_bipolar(
        test_set,
        "test",
        mono19_names=mono19_names,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        crop_length=crop_length,
        latency=latency,
        n_crops_per_record=eval_crops_per_record,
        seed=seed,
    )

    all_records = train_records + val_records + test_records

    meta_json = output_h5_path.replace(".h5", "_meta.json")
    with open(meta_json, "w") as f:
        json.dump(
            {
                "dataset_path": dataset_path,
                "task": task,
                "file_format": file_format,
                "seed": int(seed),
                "crop_length": int(crop_length),
                "latency": int(latency),
                "input_norm": "dataset_bipolar",
                "montage": "bi23",
                "channel_names": CAUEEG_BI23_NAMES,
                "bipolar_pairs": [list(p) for p in CAUEEG_BI23_PAIRS],
                "target_total_train_crops": int(target_total_train_crops),
                "train_crops_per_record": int(train_crops_per_record),
                "actual_total_train_crops": int(actual_total_train_crops),
                "eval_crops_per_record": int(eval_crops_per_record),
                "bands": {k: list(v) for k, v in DEFAULT_BANDS.items()},
                "feature_families": list(feature_families),
                "connectivity_metrics": list(connectivity_metrics),
                "train_ids": train_ids,
                "val_ids": val_ids,
                "test_ids": test_ids,
            },
            f,
            indent=2,
        )

    np.savez(
        output_h5_path.replace(".h5", "_dataset_norm_stats.npz"),
        signal_mean=signal_mean,
        signal_std=signal_std,
    )

    build_master_eeg_dataset(
        subject_records=all_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=DEFAULT_BANDS,
        overwrite=overwrite,
        skip_bad_segments=False,
        target_sampling_rate=None,
        qc_input_unit="auto",
    )

    total_train = sum(len(r["windows"]) for r in train_records)
    total_val = sum(len(r["windows"]) for r in val_records)
    total_test = sum(len(r["windows"]) for r in test_records)
    total_all = total_train + total_val + total_test

    print(f"Saved bipolar H5: {output_h5_path}")
    print(f"Saved meta: {meta_json}")
    print(f"Saved norm stats: {output_h5_path.replace('.h5', '_dataset_norm_stats.npz')}")
    print(f"Train records: {len(train_records)} | stored train crops: {total_train}")
    print(f"Val records: {len(val_records)} | stored val crops: {total_val}")
    print(f"Test records: {len(test_records)} | stored test crops: {total_test}")
    print(f"Total crop windows stored: {total_all}")

if __name__ == "__main__":

    # ---------------------------------------------------------
    # Bipolar montage helpers
    # ---------------------------------------------------------
    task="dementia"
    file_format="edf"
    dataset_path="/home/anphan/Downloads/caueeg-dataset/"
    out_h5 = "/home/anphan/Documents/caueeg_randomcrop_bi23_dementia_seed42.h5"

    build_caueeg_randomcrop_master_bipolar23(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        output_h5_path=out_h5,
        seed=42,
        crop_length=2000,
        latency=2000,
        target_total_train_crops=100_000,
        eval_crops_per_record=8,
        feature_families=None,
        connectivity_metrics=None,
        overwrite=True,
    )


    # build_caueeg_combined_sliding_random_master(
    #     dataset_path=dataset_path,
    #     task=task,
    #     file_format=file_format,
    #     output_h5_path=out_h5,
    #     seed=42,
    #     crop_length=2000,
    #     latency=2000,
    #     overlap=0.5,
    #     target_total_train_random_crops=100000,

    #     # Recommended: keep val/test deterministic
    #     eval_use_random=False,
    #     eval_random_crops_per_record=0,

    #     feature_families=None,
    #     connectivity_metrics=None,
    #     overwrite=True,
    # )



    # random_h5 = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    # random_norm_stats = random_h5.replace(".h5", "_dataset_norm_stats.npz")

    # sliding_h5 = "/mnt/data/anphan/CAUEEG/caueeg_sliding_master_dementia_seed42_overlap50_compatible.h5"

    # build_caueeg_sliding_master_compatible(
    #     dataset_path="/mnt/data/anphan/CAUEEG/caueeg-dataset",
    #     task="dementia",
    #     file_format="feather",
    #     output_h5_path=sliding_h5,
    #     seed=42,
    #     crop_length=2000,
    #     latency=2000,
    #     overlap=0.5,

    #     # Important: use same schema as random H5
    #     feature_families=None,
    #     connectivity_metrics=None,

    #     # Important: reuse the random H5 normalization stats
    #     norm_stats_npz=random_norm_stats,

    #     overwrite=True,
    # )
    # build_caueeg_randomcrop_master(
    #     dataset_path="/mnt/data/anphan/CAUEEG/caueeg-dataset",
    #     task="dementia",
    #     file_format="feather",
    #     output_h5_path="/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5",
    #     seed=42,
    #     crop_length=2000,
    #     latency=2000,
    #     target_total_train_crops=100_000,
    #     eval_crops_per_record=8,
    #     feature_families=None,
    #     connectivity_metrics=None,
    #     overwrite=False,
    # )

    #############################################
    # Requested total train crops: 100000
    # Train recordings: 950
    # Computed train_crops_per_record: 106
    # Actual total stored train crops: 100700
    # Eval crops per record: 8
    # Saved H5: /mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5
    # Saved meta: /mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42_meta.json
    # Saved norm stats: /mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42_dataset_norm_stats.npz
    # Train records: 950 | stored train crops: 100700
    # Val records: 119 | stored val crops: 952
    # Test records: 118 | stored test crops: 944
    # Total crop windows stored: 102596
    #############################################














    # import h5py

    # # 2. Define the missing records
    # missing_records = {
    #     "train_00587": {
    #         "features": ["relative_band_power", "hjorth", "statistical"],
    #         "connectivity": ["coherence"]
    #     },
    #     "train_00781": {
    #         "features": [],
    #         "connectivity": ["coherence"]
    #     },
    #     "train_01301": {
    #         "features": ["relative_band_power", "hjorth", "statistical"],
    #         "connectivity": ["coherence"]
    #     }
    # }
    # 2. Updated missing records based on your latest scan
    # missing_records = {
    #     "train_00587": {
    #         "features": [
    #             "log_band_power", 
    #             "spectral_entropy", 
    #             "higuchi_fd", 
    #             "wavelet_energy"
    #         ],
    #         "connectivity": [
    #             "pearson", 
    #             "spearman", 
    #             "plv", 
    #             "pli", 
    #             "wpli"
    #         ]
    #     },
    #     "train_00781": {
    #         "features": [
    #             "wavelet_energy"
    #         ],
    #         "connectivity": [
    #             "pearson", 
    #             "spearman", 
    #             "plv", 
    #             "pli", 
    #             "wpli"
    #         ]
    #     },
    #     "train_01301": {
    #         "features": [
    #             "log_band_power", 
    #             "spectral_entropy", 
    #             "higuchi_fd", 
    #             "wavelet_energy"
    #         ],
    #         "connectivity": [
    #             "pearson", 
    #             "spearman", 
    #             "plv", 
    #             "pli", 
    #             "wpli"
    #         ]
    #     }
    # }
    # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42_backup.h5"

    # # 3. Open the file and patch
    # with h5py.File(h5_path, "a") as h5f:
    #     bands = DEFAULT_BANDS 
        
    #     for subject_id, missing in missing_records.items():
    #         print(f"\nProcessing {subject_id}...")
    #         subject_path = f"subjects/{subject_id}"
            
    #         if subject_path not in h5f:
    #             print(f"  Warning: {subject_id} not found. Skipping.")
    #             continue
                
    #         subj_grp = h5f[subject_path]
            
    #         # Extract raw EEG data; expected shape (n_windows, n_channels, n_time)
    #         raw_eeg = subj_grp["windows/raw/eeg"][:]
    #         n_windows = raw_eeg.shape[0]
            
    #         # Fetch the sampling rate
    #         try:
    #             stored_sampling_rate = subj_grp["metadata"].attrs.get("stored_sampling_rate", 500.0)
    #         except KeyError:
    #             stored_sampling_rate = 500.0 
                
    #         feat_grp = subj_grp.require_group("windows/features")
    #         conn_grp = subj_grp.require_group("windows/connectivity")

    #         # --- PROCESS MISSING FEATURES ---
    #         for family in missing["features"]:
    #             if family in feat_grp:
    #                 print(f"  Skipping features/{family}: Already exists.")
    #                 continue
                    
    #             print(f"  Computing features/{family}...")
    #             spec = FEATURE_REGISTRY[family]
    #             dataset = None
                
    #             for idx in range(n_windows):
    #                 x = raw_eeg[idx] 
    #                 values, meta = spec["fn"](x, stored_sampling_rate, bands)
    #                 values = np.asarray(values, dtype=np.float32)
                    
    #                 if dataset is None:
    #                     dataset = feat_grp.create_dataset(
    #                         name=family,
    #                         shape=(n_windows,) + values.shape,
    #                         dtype=np.float32
    #                     )
    #                     dataset.attrs["description"] = meta.get("description", spec.get("description", ""))
    #                     dataset.attrs["feature_names"] = meta.get("feature_names", [])
    #                     dataset.attrs["shape_description"] = ["num_windows", "num_channels", "num_features"]
                    
    #                 dataset[idx] = values

    #         # --- PROCESS MISSING CONNECTIVITY ---
    #         for metric in missing["connectivity"]:
    #             if metric in conn_grp:
    #                 print(f"  Skipping connectivity/{metric}: Already exists.")
    #                 continue
                    
    #             print(f"  Computing connectivity/{metric}...")
    #             spec = CONNECTIVITY_REGISTRY[metric]
    #             dataset = None
                
    #             for idx in range(n_windows):
    #                 x = raw_eeg[idx] 
    #                 values, meta = spec["fn"](x, stored_sampling_rate, bands)
    #                 values = np.asarray(values, dtype=np.float32)
                    
    #                 if dataset is None:
    #                     dataset = conn_grp.create_dataset(
    #                         name=metric,
    #                         shape=(n_windows,) + values.shape,
    #                         dtype=np.float32
    #                     )
    #                     dataset.attrs["description"] = meta.get("description", spec.get("description", ""))
    #                     if "band_names" in meta:
    #                         dataset.attrs["band_names"] = meta["band_names"]
                    
    #                 dataset[idx] = values

    #     print("\nPatching of remaining features complete!")