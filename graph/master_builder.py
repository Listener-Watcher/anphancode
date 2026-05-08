from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from scipy import signal, stats


# -----------------------------------------------------------------------------
# Default EEG bands
# -----------------------------------------------------------------------------
DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# -----------------------------------------------------------------------------
# QC configuration aligned with clean_data.py
# -----------------------------------------------------------------------------
POSTERIOR_CHANNELS = {"P3", "P4", "Pz", "O1", "O2", "T5", "T6"}

CLEAN_DATA_QC_THRESHOLDS: Dict[str, float] = {
    "max_abs_uv": 200.0,
    "max_ptp_uv": 300.0,
    "global_std_uv": 80.0,
    "flat_std_uv": 0.05,
    "flat_channel_frac": 0.30,
    "kurtosis_thr": 8.0,
    "high_kurtosis_frac": 0.50,
    "slow_fast_ratio_thr": 2.5,
    "posterior_alpha_rel_thr": 0.20,
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def _as_numpy_float32(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32, copy=False)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _as_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


def _json_dumps(obj: Any) -> str:
    return json.dumps(_as_jsonable(obj), ensure_ascii=False)


def _safe_subject_key(subject_id: Any) -> str:
    return str(subject_id).replace("/", "__")


def _vlen_str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _require_2d_window(window: np.ndarray) -> np.ndarray:
    arr = _as_numpy_float32(window)
    if arr.ndim != 2:
        raise ValueError(f"EEG window must have shape [channels, time], got {arr.shape}")
    return arr


def _window_chunks(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    # Chunk one window at a time to support selective loading.
    return (1,) + tuple(shape[1:])


def _normalize_target_sfreq(target_sfreq: Optional[float]) -> Optional[float]:
    if target_sfreq is None:
        return None
    target = float(target_sfreq)
    if not math.isfinite(target) or target <= 0:
        raise ValueError(f"target_sampling_rate must be a positive finite number, got {target_sfreq!r}")
    return target


def _resample_window(window: np.ndarray, orig_sfreq: float, target_sfreq: Optional[float]) -> np.ndarray:
    """
    Resample one EEG window along the time axis with anti-alias filtering.

    If ``target_sfreq`` is None or matches ``orig_sfreq``, the window is returned
    unchanged. Otherwise scipy.signal.resample_poly is used.
    """
    x = _require_2d_window(window)
    target = _normalize_target_sfreq(target_sfreq)
    orig = float(orig_sfreq)
    if target is None or math.isclose(orig, target, rel_tol=1e-9, abs_tol=1e-9):
        return x.astype(np.float32, copy=False)

    ratio = Fraction(target / orig).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    y = signal.resample_poly(x, up=up, down=down, axis=-1, padtype="line")

    expected_len = int(round(x.shape[-1] * target / orig))
    if y.shape[-1] > expected_len:
        y = y[..., :expected_len]
    elif y.shape[-1] < expected_len:
        pad_width = [(0, 0)] * y.ndim
        pad_width[-1] = (0, expected_len - y.shape[-1])
        y = np.pad(y, pad_width, mode="edge")

    return y.astype(np.float32, copy=False)

def _to_hdf5_attr_value(value):
    if value is None:
        return _json_dumps(None)
    if isinstance(value, (dict, list, tuple)):
        return _json_dumps(value)
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_hdf5_attr_value(value.item())
        return _json_dumps(value.tolist())
    if torch.is_tensor(value):
        if value.ndim == 0:
            return _to_hdf5_attr_value(value.item())
        return _json_dumps(value.detach().cpu().tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
# -----------------------------------------------------------------------------
# Bipolar helper adapted from feature_extract.py
# -----------------------------------------------------------------------------

def compute_bipolar_segment(
    eeg_segment: np.ndarray,
    channel_names: Sequence[str],
    bipolar_pairs: Sequence[Tuple[str, str]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert referential EEG into bipolar montage.

    This mirrors the logic currently used in feature_extract.py, but is kept
    here so the new builder stays self-contained.
    """
    eeg_segment = _require_2d_window(eeg_segment)
    ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}

    bipolar_data: List[np.ndarray] = []
    bipolar_names: List[str] = []
    for ch_a, ch_b in bipolar_pairs:
        if ch_a not in ch_to_idx or ch_b not in ch_to_idx:
            raise ValueError(f"Missing channel for bipolar pair: {ch_a}-{ch_b}")
        x = eeg_segment[ch_to_idx[ch_a]] - eeg_segment[ch_to_idx[ch_b]]
        bipolar_data.append(x.astype(np.float32, copy=False))
        bipolar_names.append(f"{ch_a}-{ch_b}")

    bipolar_segment = np.stack(bipolar_data, axis=0).astype(np.float32, copy=False)
    return bipolar_segment, bipolar_names


# -----------------------------------------------------------------------------
# Subject input helpers
# -----------------------------------------------------------------------------

def iter_subject_records_from_list(records: Sequence[Mapping[str, Any]]) -> Iterator[Mapping[str, Any]]:
    """
    Expected input record format:
    [
        {
            "subject_id": ...,
            "label": ...,
            "windows": [np.ndarray[C, T], ...],
            "start_samples": [...],
            "channel_names": [...],
            ...
        }
    ]
    """
    for record in records:
        yield record



def iter_subject_records_from_master_dir(
    master_dir: str | os.PathLike,
    manifest_name: str = "subject_manifest.csv",
    use_subject_column: Optional[str] = "use_subject",
) -> Iterator[Dict[str, Any]]:
    """
    Adapter for the same subject-level directory structure used by feature_extract.py:
      master_dir/
        subject_manifest.csv
        sub-001.pt
        sub-002.pt
        ...

    Each subject .pt is expected to contain keys like:
      - subject_id
      - class_id
      - channel_names
      - montage_type (optional)
      - session_info (optional)
      - segments: list of dicts with eeg_segment, segment_id, start_sample, ...
    """
    import pandas as pd

    master_dir = Path(master_dir)
    manifest_path = master_dir / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if use_subject_column is not None and use_subject_column in manifest.columns:
        manifest = manifest[manifest[use_subject_column] == 1].copy()

    for _, row in manifest.iterrows():
        subject_id = row["subject_id"]
        pt_path = master_dir / f"{subject_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing subject file: {pt_path}")

        obj = torch.load(pt_path, map_location="cpu", weights_only=False)
        segments = obj.get("segments", [])

        windows = []
        start_samples = []
        end_samples = []
        segment_ids = []
        segment_metadata = []

        for seg in segments:
            eeg_segment = _as_numpy_float32(seg["eeg_segment"])
            start_sample = int(seg.get("start_sample", 0))
            end_sample = int(seg.get("end_sample", start_sample + eeg_segment.shape[-1]))

            windows.append(eeg_segment)
            start_samples.append(start_sample)
            end_samples.append(end_sample)
            segment_ids.append(int(seg.get("segment_id", len(segment_ids))))
            segment_metadata.append({
                k: _as_jsonable(v)
                for k, v in seg.items()
                if k not in {"eeg_segment"}
            })

        yield {
            "subject_id": obj.get("subject_id", subject_id),
            "label": obj.get("class_id", row.get("class_id")),
            "class_id": obj.get("class_id", row.get("class_id")),
            "sampling_rate": obj.get("sampling_rate", row.get("sampling_rate", 500)),
            "channel_names": obj.get("channel_names"),
            "montage_type": obj.get("montage_type", row.get("montage_type")),
            "session_info": obj.get("session_info", None),
            "recording_info": {k: _as_jsonable(v) for k, v in obj.items() if k not in {"segments"}},
            "windows": windows,
            "start_samples": start_samples,
            "end_samples": end_samples,
            "segment_ids": segment_ids,
            "segment_metadata": segment_metadata,
        }


# -----------------------------------------------------------------------------
# QC helpers
# -----------------------------------------------------------------------------

# def _bandpower_from_psd_qc(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
#     idx = (freqs >= fmin) & (freqs < fmax)
#     if not np.any(idx):
#         return np.zeros(psd.shape[0], dtype=np.float32)
#     return np.trapezoid(psd[:, idx], freqs[idx], axis=1).astype(np.float32)


# def _convert_window_to_microvolts(window: np.ndarray, input_unit: str = "auto") -> np.ndarray:
#     x = _require_2d_window(window)
#     unit = str(input_unit).lower()
#     if unit not in {"auto", "uv", "microvolt", "microvolts", "v", "volt", "volts"}:
#         raise ValueError(f"Unsupported input_unit={input_unit!r}. Use 'auto', 'uv', or 'v'.")
#     if unit in {"uv", "microvolt", "microvolts"}:
#         return x.astype(np.float32, copy=False)
#     if unit in {"v", "volt", "volts"}:
#         return (x * 1e6).astype(np.float32, copy=False)

#     max_abs = float(np.max(np.abs(x))) if x.size > 0 else 0.0
#     if max_abs < 1.0:
#         return (x * 1e6).astype(np.float32, copy=False)
#     return x.astype(np.float32, copy=False)


# def compute_window_qc(
#     window: np.ndarray,
#     sfreq: float,
#     ch_names: Sequence[str],
#     *,
#     input_unit: str = "auto",
#     qc_thresholds: Optional[Mapping[str, float]] = None,
# ) -> Dict[str, Any]:
#     """
#     Compute QC metrics using the same logic and thresholds as clean_data.py.

#     clean_data.py expects microvolt windows, computes amplitude-, kurtosis-,
#     and PSD-based QC summaries, and flags a window as noisy when any of the
#     soft thresholds are exceeded. This function mirrors that behavior and also
#     returns backward-compatible aliases used by the earlier builder.
#     """
#     thresholds = dict(CLEAN_DATA_QC_THRESHOLDS)
#     if qc_thresholds is not None:
#         thresholds.update({str(k): float(v) for k, v in qc_thresholds.items()})

#     window_uv = _convert_window_to_microvolts(window, input_unit=input_unit)
#     eps = 1e-12

#     ch_std = np.std(window_uv, axis=1)
#     ch_ptp = np.ptp(window_uv, axis=1)
#     ch_absmax = np.max(np.abs(window_uv), axis=1)
#     ch_kurt = stats.kurtosis(window_uv, axis=1, fisher=False, bias=False, nan_policy="omit")
#     ch_kurt = np.nan_to_num(ch_kurt, nan=0.0, posinf=0.0, neginf=0.0)

#     nperseg = min(window_uv.shape[1], max(256, int(2 * sfreq)))
#     freqs, psd = signal.welch(window_uv, fs=sfreq, axis=-1, nperseg=nperseg)

#     total_power = _bandpower_from_psd_qc(psd, freqs, 0.5, 45.0) + eps
#     delta = _bandpower_from_psd_qc(psd, freqs, 0.5, 4.0)
#     theta = _bandpower_from_psd_qc(psd, freqs, 4.0, 8.0)
#     alpha = _bandpower_from_psd_qc(psd, freqs, 8.0, 13.0)
#     beta = _bandpower_from_psd_qc(psd, freqs, 13.0, 30.0)
#     hf = _bandpower_from_psd_qc(psd, freqs, 30.0, 45.0)
#     lf = _bandpower_from_psd_qc(psd, freqs, 1.0, 20.0) + eps

#     rel_delta = delta / total_power
#     rel_theta = theta / total_power
#     rel_alpha = alpha / total_power
#     rel_beta = beta / total_power
#     rel_hf = hf / total_power

#     posterior_idx = [i for i, ch in enumerate(ch_names) if ch in POSTERIOR_CHANNELS]
#     if len(posterior_idx) > 0:
#         posterior_alpha_rel = float(np.mean(rel_alpha[posterior_idx]))
#     else:
#         posterior_alpha_rel = float(np.mean(rel_alpha))

#     slow_fast_ratio = float((np.mean(rel_delta) + np.mean(rel_theta)) / (np.mean(rel_alpha) + np.mean(rel_beta) + eps))

#     metrics: Dict[str, Any] = {
#         "max_abs_uv": float(np.max(ch_absmax)),
#         "max_ptp_uv": float(np.max(ch_ptp)),
#         "peak_to_peak_uv": float(np.max(ch_ptp)),
#         "median_ptp_uv": float(np.median(ch_ptp)),
#         "global_std_uv": float(np.std(window_uv)),
#         "flat_channel_frac": float(np.mean(ch_std < thresholds["flat_std_uv"])),
#         "high_kurtosis_frac": float(np.mean(ch_kurt > thresholds["kurtosis_thr"])),
#         "rel_delta_mean": float(np.mean(rel_delta)),
#         "rel_theta_mean": float(np.mean(rel_theta)),
#         "rel_alpha_mean": float(np.mean(rel_alpha)),
#         "rel_beta_mean": float(np.mean(rel_beta)),
#         "rel_hf_mean": float(np.mean(rel_hf)),
#         "posterior_alpha_rel": posterior_alpha_rel,
#         "slow_fast_ratio": slow_fast_ratio,
#         "hf_lf_ratio": float(np.mean(hf) / np.mean(lf)),
#         "input_unit": input_unit,
#     }

#     reasons: List[str] = []
#     if metrics["max_abs_uv"] > thresholds["max_abs_uv"]:
#         reasons.append(f"max_abs_uv>{thresholds['max_abs_uv']}")
#     if metrics["max_ptp_uv"] > thresholds["max_ptp_uv"]:
#         reasons.append(f"max_ptp_uv>{thresholds['max_ptp_uv']}")
#     if metrics["global_std_uv"] > thresholds["global_std_uv"]:
#         reasons.append(f"global_std_uv>{thresholds['global_std_uv']}")
#     if metrics["flat_channel_frac"] > thresholds["flat_channel_frac"]:
#         reasons.append(f"flat_channel_frac>{thresholds['flat_channel_frac']}")
#     if metrics["high_kurtosis_frac"] > thresholds["high_kurtosis_frac"]:
#         reasons.append(f"high_kurtosis_frac>{thresholds['high_kurtosis_frac']}")

#     noise_flag = bool(len(reasons) > 0)
#     qc_dict = dict(metrics)
#     qc_dict["thresholds"] = dict(thresholds)
#     qc_dict["triggered"] = list(reasons)

#     metrics.update({
#         "noise_flag": noise_flag,
#         "bad_segment_flag": noise_flag,
#         "artifact_reasons": ";".join(reasons),
#         "reason": ";".join(reasons),
#         "qc_dict": qc_dict,
#     })
#     return metrics


# -----------------------------------------------------------------------------
# Spectral helpers
# -----------------------------------------------------------------------------

def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    return (freqs >= lo) & (freqs < hi)



def _welch_psd(window: np.ndarray, sfreq: float) -> Tuple[np.ndarray, np.ndarray]:
    x = _require_2d_window(window)
    nperseg = min(x.shape[-1], int(sfreq * 2))
    noverlap = nperseg // 2
    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        axis=-1,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return freqs.astype(np.float32), psd.astype(np.float32)



def _band_power_from_psd(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Mapping[str, Tuple[float, float]],
    log_scale: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    band_names = list(bands.keys())
    out = np.zeros((psd.shape[0], len(band_names)), dtype=np.float32)

    for b_idx, band_name in enumerate(band_names):
        mask = _band_mask(freqs, bands[band_name])
        if not np.any(mask):
            continue
        power = np.trapz(psd[:, mask], freqs[mask], axis=-1)
        # power = np.trapezoid(psd[:, mask], freqs[mask], axis=-1)
        if log_scale:
            power = np.log1p(np.maximum(power, 0.0))
        out[:, b_idx] = power.astype(np.float32)
    return out, band_names



def _relative_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Mapping[str, Tuple[float, float]],
) -> Tuple[np.ndarray, List[str]]:
    abs_power, band_names = _band_power_from_psd(freqs, psd, bands, log_scale=False)
    total_power = abs_power.sum(axis=1, keepdims=True)
    rel = abs_power / np.clip(total_power, 1e-8, None)
    return rel.astype(np.float32), band_names



def _bandpass_filter(sig_1d: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    lo, hi = band
    nyq = sfreq / 2.0
    hi = min(hi, nyq - 1e-3)
    if lo <= 0 or hi <= lo:
        raise ValueError(f"Invalid band {band} for sfreq={sfreq}")
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, sig_1d).astype(np.float32)



def _analytic_phase(window: np.ndarray, sfreq: float, band: Tuple[float, float]) -> np.ndarray:
    x = _require_2d_window(window)
    phases = np.zeros_like(x, dtype=np.float32)
    for ch in range(x.shape[0]):
        xf = _bandpass_filter(x[ch], sfreq, band)
        phases[ch] = np.angle(signal.hilbert(xf)).astype(np.float32)
    return phases


# -----------------------------------------------------------------------------
# Node feature extractors
# -----------------------------------------------------------------------------

def feature_relative_band_power(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    values, band_names = _relative_band_power(freqs, psd, bands)
    return values, {
        "feature_names": [f"rbp_{b}" for b in band_names],
        "description": "Relative band power per channel.",
    }



def feature_absolute_band_power(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    values, band_names = _band_power_from_psd(freqs, psd, bands, log_scale=False)
    return values, {
        "feature_names": [f"abs_power_{b}" for b in band_names],
        "description": "Absolute band power per channel.",
    }



def feature_log_band_power(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    values, band_names = _band_power_from_psd(freqs, psd, bands, log_scale=True)
    return values, {
        "feature_names": [f"log_power_{b}" for b in band_names],
        "description": "Log-transformed band power per channel.",
    }



def feature_hjorth(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)

    var_x = np.var(x, axis=-1)
    var_dx = np.var(dx, axis=-1)
    var_ddx = np.var(ddx, axis=-1)

    activity = var_x
    mobility = np.sqrt(var_dx / np.clip(var_x, 1e-8, None))
    complexity = np.sqrt(var_ddx / np.clip(var_dx, 1e-8, None)) / np.clip(mobility, 1e-8, None)

    values = np.stack([activity, mobility, complexity], axis=-1).astype(np.float32)
    return values, {
        "feature_names": ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"],
        "description": "Hjorth activity, mobility, complexity per channel.",
    }



def feature_statistical(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    mean = np.mean(x, axis=-1)
    std = np.std(x, axis=-1)
    skew = stats.skew(x, axis=-1, bias=False)
    kurt = stats.kurtosis(x, axis=-1, fisher=True, bias=False)
    min_v = np.min(x, axis=-1)
    max_v = np.max(x, axis=-1)
    ptp = np.ptp(x, axis=-1)

    values = np.stack([mean, std, skew, kurt, min_v, max_v, ptp], axis=-1).astype(np.float32)
    return values, {
        "feature_names": ["mean", "std", "skew", "kurtosis", "min", "max", "ptp"],
        "description": "Basic statistical features per channel.",
    }



def feature_spectral_entropy(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    freqs, psd = _welch_psd(window, sfreq)
    p = psd / np.clip(psd.sum(axis=-1, keepdims=True), 1e-8, None)
    entropy = -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1) / np.log(p.shape[-1])
    values = entropy[:, None].astype(np.float32)
    return values, {
        "feature_names": ["spectral_entropy"],
        "description": "Normalized spectral entropy per channel.",
    }



def _higuchi_fd_1d(x: np.ndarray, kmax: int = 8) -> float:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 4:
        return float("nan")

    lk = []
    xk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if idx.size < 2:
                continue
            ll = np.sum(np.abs(np.diff(x[idx])))
            norm = (n - 1) / (((n - m - 1) // k) * k)
            lm.append((ll * norm) / k)
        if len(lm) == 0:
            continue
        lk.append(np.mean(lm))
        xk.append(1.0 / k)

    if len(lk) < 2:
        return float("nan")
    coeffs = np.polyfit(np.log(xk), np.log(lk), deg=1)
    return float(coeffs[0])



def feature_hfd(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    values = np.array([_higuchi_fd_1d(ch) for ch in x], dtype=np.float32)[:, None]
    return values, {
        "feature_names": ["higuchi_fd"],
        "description": "Higuchi fractal dimension per channel.",
    }



def feature_wavelet_energy(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise ImportError("wavelet features require PyWavelets (pywt). Install it or remove 'wavelet_energy'.") from exc

    x = _require_2d_window(window)
    level = 5
    values = []
    for ch in x:
        coeffs = pywt.wavedec(ch, wavelet="db4", level=level)
        energies = [float(np.sum(np.square(c))) for c in coeffs]
        values.append(energies)
    values = np.asarray(values, dtype=np.float32)
    names = [f"wavelet_energy_a{level}"] + [f"wavelet_energy_d{i}" for i in range(level, 0, -1)]
    return values, {
        "feature_names": names,
        "description": "Wavelet sub-band energies per channel.",
    }


# -----------------------------------------------------------------------------
# Connectivity extractors
# -----------------------------------------------------------------------------

def _symmetrize(mat: np.ndarray) -> np.ndarray:
    mat = 0.5 * (mat + mat.T)
    np.fill_diagonal(mat, 0.0)
    return mat.astype(np.float32)



def conn_pearson(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    mat = np.corrcoef(x)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return _symmetrize(mat), {"description": "Pearson correlation connectivity.", "band_names": None}



def conn_spearman(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    ranks = np.apply_along_axis(stats.rankdata, 1, x)
    mat = np.corrcoef(ranks)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return _symmetrize(mat), {"description": "Spearman correlation connectivity.", "band_names": None}



def conn_coherence(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    band_names = list(bands.keys())
    n_channels = x.shape[0]
    out = np.zeros((len(band_names), n_channels, n_channels), dtype=np.float32)
    nperseg = min(x.shape[-1], int(sfreq * 2))
    noverlap = nperseg // 2

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            freqs, coh = signal.coherence(x[i], x[j], fs=sfreq, nperseg=nperseg, noverlap=noverlap)
            for b_idx, band_name in enumerate(band_names):
                mask = _band_mask(freqs, bands[band_name])
                val = float(np.mean(coh[mask])) if np.any(mask) else 0.0
                out[b_idx, i, j] = val
                out[b_idx, j, i] = val
    return out, {"description": "Band-averaged magnitude-squared coherence.", "band_names": band_names}



def _phase_connectivity(
    window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, Tuple[float, float]],
    mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = _require_2d_window(window)
    band_names = list(bands.keys())
    n_channels = x.shape[0]
    out = np.zeros((len(band_names), n_channels, n_channels), dtype=np.float32)

    for b_idx, band_name in enumerate(band_names):
        phases = _analytic_phase(x, sfreq, bands[band_name])
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                dphi = phases[i] - phases[j]
                if mode == "plv":
                    val = np.abs(np.mean(np.exp(1j * dphi)))
                elif mode == "pli":
                    val = np.abs(np.mean(np.sign(np.sin(dphi))))
                elif mode == "wpli":
                    im = np.sin(dphi)
                    denom = np.mean(np.abs(im))
                    val = np.abs(np.mean(im)) / max(denom, 1e-8)
                else:
                    raise ValueError(f"Unsupported mode: {mode}")
                out[b_idx, i, j] = float(val)
                out[b_idx, j, i] = float(val)
    return out, {"description": f"Band-wise {mode.upper()} connectivity.", "band_names": band_names}



def conn_plv(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _phase_connectivity(window, sfreq, bands, mode="plv")



def conn_pli(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _phase_connectivity(window, sfreq, bands, mode="pli")



def conn_wpli(window: np.ndarray, sfreq: float, bands: Mapping[str, Tuple[float, float]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    return _phase_connectivity(window, sfreq, bands, mode="wpli")


# -----------------------------------------------------------------------------
# Registries
# -----------------------------------------------------------------------------

FeatureExtractor = Callable[[np.ndarray, float, Mapping[str, Tuple[float, float]]], Tuple[np.ndarray, Dict[str, Any]]]
ConnectivityExtractor = Callable[[np.ndarray, float, Mapping[str, Tuple[float, float]]], Tuple[np.ndarray, Dict[str, Any]]]


FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "relative_band_power": {
        "fn": feature_relative_band_power,
        "description": "Relative band power per channel.",
        "kind": "node",
    },
    # "absolute_band_power": {
    #     "fn": feature_absolute_band_power,
    #     "description": "Absolute band power per channel.",
    #     "kind": "node",
    # },
    "log_band_power": {
        "fn": feature_log_band_power,
        "description": "Log band power per channel.",
        "kind": "node",
    },
    "hjorth": {
        "fn": feature_hjorth,
        "description": "Hjorth parameters per channel.",
        "kind": "node",
    },
    "statistical": {
        "fn": feature_statistical,
        "description": "Statistical per-channel features.",
        "kind": "node",
    },
    "spectral_entropy": {
        "fn": feature_spectral_entropy,
        "description": "Spectral entropy per channel.",
        "kind": "node",
    },
    "higuchi_fd": {
        "fn": feature_hfd,
        "description": "Higuchi fractal dimension per channel.",
        "kind": "node",
    },
    "wavelet_energy": {
        "fn": feature_wavelet_energy,
        "description": "Wavelet energies per channel.",
        "kind": "node",
    },
}


CONNECTIVITY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "pearson": {"fn": conn_pearson, "description": "Pearson correlation.", "kind": "connectivity"},
    "spearman": {"fn": conn_spearman, "description": "Spearman correlation.", "kind": "connectivity"},
    "coherence": {"fn": conn_coherence, "description": "Band-wise coherence.", "kind": "connectivity"},
    "plv": {"fn": conn_plv, "description": "Band-wise phase-locking value.", "kind": "connectivity"},
    "pli": {"fn": conn_pli, "description": "Band-wise phase-lag index.", "kind": "connectivity"},
    "wpli": {"fn": conn_wpli, "description": "Band-wise weighted phase-lag index.", "kind": "connectivity"},
}


# -----------------------------------------------------------------------------
# HDF5 writing helpers
# -----------------------------------------------------------------------------

def _write_string_dataset(group: h5py.Group, name: str, values: Sequence[str]) -> None:
    arr = np.asarray(list(values), dtype=object)
    if name in group:
        del group[name]
    group.create_dataset(name, data=arr, dtype=_vlen_str_dtype())



def _init_registry_attrs(h5f: h5py.File, bands: Mapping[str, Tuple[float, float]]) -> None:
    h5f.attrs["bands_json"] = _json_dumps({k: list(v) for k, v in bands.items()})
    h5f.attrs["feature_registry_json"] = _json_dumps({
        k: {kk: vv for kk, vv in v.items() if kk != "fn"}
        for k, v in FEATURE_REGISTRY.items()
    })
    h5f.attrs["connectivity_registry_json"] = _json_dumps({
        k: {kk: vv for kk, vv in v.items() if kk != "fn"}
        for k, v in CONNECTIVITY_REGISTRY.items()
    })



def _create_subject_group(
    h5f: h5py.File,
    subject_record: Mapping[str, Any],
    n_windows: int,
    window_shape: Tuple[int, int],
    feature_names: Sequence[str],
    connectivity_names: Sequence[str],
    bands: Mapping[str, Tuple[float, float]],
) -> h5py.Group:
    subject_id = _safe_subject_key(subject_record["subject_id"])
    grp = h5f.require_group(f"subjects/{subject_id}")

    # Metadata group
    meta = grp.require_group("metadata")
    meta.attrs["subject_id"] = str(subject_record["subject_id"])
    meta.attrs["label"] = int(subject_record.get("label", subject_record.get("class_id")))
    meta.attrs["class_id"] = int(subject_record.get("class_id", subject_record.get("label")))
    meta.attrs["sampling_rate"] = float(subject_record["sampling_rate"])
    meta.attrs["stored_sampling_rate"] = float(subject_record.get("stored_sampling_rate", subject_record["sampling_rate"]))
    meta.attrs["original_sampling_rate"] = float(subject_record.get("original_sampling_rate", subject_record["sampling_rate"]))
    meta.attrs["num_windows"] = int(n_windows)
    meta.attrs["num_channels"] = int(window_shape[0])
    meta.attrs["num_timepoints"] = int(window_shape[1])
    meta.attrs["stored_num_timepoints"] = int(window_shape[1])
    meta.attrs["montage_type"] = "" if subject_record.get("montage_type") is None else str(subject_record.get("montage_type"))
    meta.attrs["session_info_json"] = _json_dumps(subject_record.get("session_info"))
    meta.attrs["recording_info_json"] = _json_dumps(subject_record.get("recording_info"))

    if "channel_names" in meta:
        del meta["channel_names"]
    _write_string_dataset(meta, "channel_names", [str(x) for x in subject_record["channel_names"]])

    windows_grp = grp.require_group("windows")
    raw_grp = windows_grp.require_group("raw")
    # qc_grp = windows_grp.require_group("qc")
    feat_grp = windows_grp.require_group("features")
    conn_grp = windows_grp.require_group("connectivity")

    # Raw windows
    if "eeg" in raw_grp:
        del raw_grp["eeg"]
    raw_grp.create_dataset(
        "eeg",
        shape=(n_windows, window_shape[0], window_shape[1]),
        dtype=np.float32,
        chunks=_window_chunks((n_windows, window_shape[0], window_shape[1])),
        compression="gzip",
        compression_opts=4,
    )

    for name in ["segment_id", "start_sample", "end_sample"]:
        if name in raw_grp:
            del raw_grp[name]
    raw_grp.create_dataset("segment_id", shape=(n_windows,), dtype=np.int64)
    raw_grp.create_dataset("start_sample", shape=(n_windows,), dtype=np.int64)
    raw_grp.create_dataset("end_sample", shape=(n_windows,), dtype=np.int64)

    # QC datasets (aligned with clean_data.py plus backward-compatible aliases)
    # qc_fields = [
    #     ("max_abs_uv", np.float32),
    #     ("max_ptp_uv", np.float32),
    #     ("peak_to_peak_uv", np.float32),
    #     ("median_ptp_uv", np.float32),
    #     ("global_std_uv", np.float32),
    #     ("flat_channel_frac", np.float32),
    #     ("high_kurtosis_frac", np.float32),
    #     ("rel_delta_mean", np.float32),
    #     ("rel_theta_mean", np.float32),
    #     ("rel_alpha_mean", np.float32),
    #     ("rel_beta_mean", np.float32),
    #     ("rel_hf_mean", np.float32),
    #     ("posterior_alpha_rel", np.float32),
    #     ("slow_fast_ratio", np.float32),
    #     ("hf_lf_ratio", np.float32),
    #     ("noise_flag", np.uint8),
    #     ("bad_segment_flag", np.uint8),
    # ]
    # for qname, qdtype in qc_fields:
    #     if qname in qc_grp:
    #         del qc_grp[qname]
    #     qc_grp.create_dataset(qname, shape=(n_windows,), dtype=qdtype)
    # for sname in ["artifact_reasons", "reason", "qc_json"]:
    #     if sname in qc_grp:
    #         del qc_grp[sname]
    #     qc_grp.create_dataset(sname, shape=(n_windows,), dtype=_vlen_str_dtype())

    # Placeholder feature/connectivity groups; actual datasets created lazily from first window
    feat_grp.attrs["requested_families_json"] = _json_dumps(list(feature_names))
    conn_grp.attrs["requested_metrics_json"] = _json_dumps(list(connectivity_names))
    conn_grp.attrs["bands_json"] = _json_dumps({k: list(v) for k, v in bands.items()})

    return grp



def _ensure_window_dataset(
    parent: h5py.Group,
    name: str,
    n_windows: int,
    per_window_shape: Tuple[int, ...],
    attrs: Optional[Mapping[str, Any]] = None,
) -> h5py.Dataset:
    if name not in parent:
        ds = parent.create_dataset(
            name,
            shape=(n_windows,) + tuple(per_window_shape),
            dtype=np.float32,
            chunks=_window_chunks((n_windows,) + tuple(per_window_shape)),
            compression="gzip",
            compression_opts=4,
        )
        if attrs:
            for k, v in attrs.items():
                ds.attrs[k] = _to_hdf5_attr_value(v)
            # for k, v in attrs.items():
            #     ds.attrs[k] = _json_dumps(v) if isinstance(v, (dict, list, tuple)) else v
        return ds
    return parent[name]


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------

def build_master_eeg_dataset(
    subject_records: Iterable[Mapping[str, Any]],
    output_h5_path: str | os.PathLike,
    *,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
    overwrite: bool = False,
    skip_bad_segments: bool = False,
    bipolar_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    montage_type_if_bipolar: str = "bipolar",
    target_sampling_rate: Optional[float] = None,
    qc_kwargs: Optional[Mapping[str, Any]] = None,
    qc_input_unit: str = "auto",
) -> str:
    """
    Build a reusable HDF5 master EEG dataset.

    Parameters
    ----------
    subject_records:
        Iterable of subject dictionaries.
    output_h5_path:
        Output HDF5 path.
    feature_families:
        Names from FEATURE_REGISTRY.
    connectivity_metrics:
        Names from CONNECTIVITY_REGISTRY.
    bands:
        Mapping like {"delta": (1,4), ...}
    overwrite:
        If True, recreate the HDF5 file.
    skip_bad_segments:
        If True, windows flagged by QC are skipped entirely.
    bipolar_pairs:
        Optional montage conversion pairs. If provided, each input window is converted
        before feature/connectivity extraction.
    target_sampling_rate:
        Optional sampling rate to store in the master dataset. Use None to keep the
        original rate, or pass a specific rate such as 100 or 250 to resample each
        window before storing and before feature/connectivity extraction. QC is still
        computed on the pre-resampled signal so it stays aligned with clean_data.py.
    qc_kwargs:
        Optional threshold overrides for QC. These are merged into the
        clean_data.py defaults.
    qc_input_unit:
        One of {"auto", "uv", "v"}. clean_data.py computes QC in microvolts,
        so this controls whether the incoming stored windows should be converted
        before QC metrics are computed.
    """
    output_h5_path = str(output_h5_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_h5_path)), exist_ok=True)

    feature_families = list(feature_families or FEATURE_REGISTRY.keys())
    connectivity_metrics = list(connectivity_metrics or CONNECTIVITY_REGISTRY.keys())
    bands = dict(bands or DEFAULT_BANDS)
    # qc_kwargs = dict(qc_kwargs or {})
    target_sampling_rate = _normalize_target_sfreq(target_sampling_rate)

    unknown_features = sorted(set(feature_families) - set(FEATURE_REGISTRY.keys()))
    unknown_conn = sorted(set(connectivity_metrics) - set(CONNECTIVITY_REGISTRY.keys()))
    if unknown_features:
        raise ValueError(f"Unknown feature families: {unknown_features}")
    if unknown_conn:
        raise ValueError(f"Unknown connectivity metrics: {unknown_conn}")

    if overwrite and os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    mode = "a" if os.path.exists(output_h5_path) else "w"
    with h5py.File(output_h5_path, mode) as h5f:
        _init_registry_attrs(h5f, bands)
        h5f.attrs["storage_format"] = "HDF5"
        h5f.attrs["builder_version"] = "1.2"
        # h5f.attrs["qc_alignment"] = "clean_data.py"
        # h5f.attrs["qc_thresholds_json"] = _json_dumps(dict(CLEAN_DATA_QC_THRESHOLDS))
        h5f.attrs["target_sampling_rate"] = -1.0 if target_sampling_rate is None else float(target_sampling_rate)

        for subject_record in subject_records:
            subject_id = _safe_subject_key(subject_record["subject_id"])
            subject_path = f"subjects/{subject_id}"
            if subject_path in h5f and not overwrite:
                # Resume-friendly behavior.
                continue
            if subject_path in h5f and overwrite:
                del h5f[subject_path]

            source_windows = subject_record["windows"]
            if len(source_windows) == 0:
                continue

            original_sampling_rate = float(subject_record.get("sampling_rate", 500.0))
            stored_sampling_rate = original_sampling_rate if target_sampling_rate is None else float(target_sampling_rate)
            channel_names = list(subject_record.get("channel_names") or [f"ch_{i}" for i in range(_require_2d_window(source_windows[0]).shape[0])])

            processed_windows: List[Dict[str, np.ndarray]] = []
            if bipolar_pairs is not None:
                bipolar_names: Optional[List[str]] = None
                for w in source_windows:
                    qc_window, bipolar_names = compute_bipolar_segment(w, channel_names, bipolar_pairs)
                    stored_window = _resample_window(qc_window, original_sampling_rate, stored_sampling_rate)
                    processed_windows.append({"qc_window": qc_window, "stored_window": stored_window})
                subject_record = dict(subject_record)
                subject_record["channel_names"] = bipolar_names
                subject_record["montage_type"] = montage_type_if_bipolar
                channel_names = list(bipolar_names or channel_names)
            else:
                for w in source_windows:
                    qc_window = _require_2d_window(w)
                    stored_window = _resample_window(qc_window, original_sampling_rate, stored_sampling_rate)
                    processed_windows.append({"qc_window": qc_window, "stored_window": stored_window})

            first_window = _require_2d_window(processed_windows[0]["stored_window"])
            n_channels, n_time = first_window.shape
            n_windows = len(processed_windows)

            segment_ids = list(subject_record.get("segment_ids") or range(n_windows))
            start_samples = list(subject_record.get("start_samples") or [0] * n_windows)
            default_window_len = int(round(n_time * original_sampling_rate / stored_sampling_rate))
            end_samples = list(subject_record.get("end_samples") or [s + default_window_len for s in start_samples])
            if len(segment_ids) != n_windows or len(start_samples) != n_windows or len(end_samples) != n_windows:
                raise ValueError(f"Metadata lengths do not match number of windows for subject {subject_record['subject_id']}")

            subject_record = dict(subject_record)
            subject_record["sampling_rate"] = stored_sampling_rate
            subject_record["stored_sampling_rate"] = stored_sampling_rate
            subject_record["original_sampling_rate"] = original_sampling_rate
            subject_record["channel_names"] = channel_names
            grp = _create_subject_group(
                h5f,
                subject_record,
                n_windows=n_windows,
                window_shape=(n_channels, n_time),
                feature_names=feature_families,
                connectivity_names=connectivity_metrics,
                bands=bands,
            )

            raw_grp = grp["windows/raw"]
            # qc_grp = grp["windows/qc"]
            feat_grp = grp["windows/features"]
            conn_grp = grp["windows/connectivity"]

            written_idx = 0
            for idx, prepared in enumerate(processed_windows):
                x = _require_2d_window(prepared["stored_window"])
                qc_source = _require_2d_window(prepared["qc_window"])
                if x.shape != (n_channels, n_time):
                    raise ValueError(
                        f"Inconsistent stored window shape for subject {subject_record['subject_id']}: expected {(n_channels, n_time)}, got {x.shape}"
                    )

                # qc = compute_window_qc(
                #     qc_source,
                #     sfreq=original_sampling_rate,
                #     ch_names=channel_names,
                #     input_unit=qc_input_unit,
                #     qc_thresholds=qc_kwargs,
                # )
                # qc["stored_sampling_rate"] = stored_sampling_rate
                # qc["original_sampling_rate"] = original_sampling_rate
                # qc["stored_num_timepoints"] = int(x.shape[-1])
                # qc["original_num_timepoints"] = int(qc_source.shape[-1])
                # if skip_bad_segments and qc["bad_segment_flag"]:
                #     continue

                raw_grp["eeg"][written_idx] = x
                raw_grp["segment_id"][written_idx] = int(segment_ids[idx])
                raw_grp["start_sample"][written_idx] = int(start_samples[idx])
                raw_grp["end_sample"][written_idx] = int(end_samples[idx])

                # qc_grp["max_abs_uv"][written_idx] = qc["max_abs_uv"]
                # qc_grp["max_ptp_uv"][written_idx] = qc["max_ptp_uv"]
                # qc_grp["peak_to_peak_uv"][written_idx] = qc["peak_to_peak_uv"]
                # qc_grp["median_ptp_uv"][written_idx] = qc["median_ptp_uv"]
                # qc_grp["global_std_uv"][written_idx] = qc["global_std_uv"]
                # qc_grp["flat_channel_frac"][written_idx] = qc["flat_channel_frac"]
                # qc_grp["high_kurtosis_frac"][written_idx] = qc["high_kurtosis_frac"]
                # qc_grp["rel_delta_mean"][written_idx] = qc["rel_delta_mean"]
                # qc_grp["rel_theta_mean"][written_idx] = qc["rel_theta_mean"]
                # qc_grp["rel_alpha_mean"][written_idx] = qc["rel_alpha_mean"]
                # qc_grp["rel_beta_mean"][written_idx] = qc["rel_beta_mean"]
                # qc_grp["rel_hf_mean"][written_idx] = qc["rel_hf_mean"]
                # qc_grp["posterior_alpha_rel"][written_idx] = qc["posterior_alpha_rel"]
                # qc_grp["slow_fast_ratio"][written_idx] = qc["slow_fast_ratio"]
                # qc_grp["hf_lf_ratio"][written_idx] = qc["hf_lf_ratio"]
                # qc_grp["noise_flag"][written_idx] = int(qc["noise_flag"])
                # qc_grp["bad_segment_flag"][written_idx] = int(qc["bad_segment_flag"])
                # qc_grp["artifact_reasons"][written_idx] = qc["artifact_reasons"]
                # qc_grp["reason"][written_idx] = qc["reason"]
                # qc_grp["qc_json"][written_idx] = _json_dumps(qc["qc_dict"])

                for family in feature_families:
                    spec = FEATURE_REGISTRY[family]
                    values, meta = spec["fn"](x, stored_sampling_rate, bands)
                    values = _as_numpy_float32(values)
                    ds = _ensure_window_dataset(
                        feat_grp,
                        family,
                        n_windows=n_windows,
                        per_window_shape=tuple(values.shape),
                        attrs={
                            "description": meta.get("description", spec["description"]),
                            "feature_names": meta.get("feature_names", []),
                            "shape_description": ["num_windows", "num_channels", "num_features"],
                        },
                    )
                    ds[written_idx] = values

                for metric in connectivity_metrics:
                    spec = CONNECTIVITY_REGISTRY[metric]
                    values, meta = spec["fn"](x, stored_sampling_rate, bands)
                    values = _as_numpy_float32(values)
                    shape_desc = ["num_windows"]
                    if values.ndim == 2:
                        shape_desc += ["num_channels", "num_channels"]
                    elif values.ndim == 3:
                        shape_desc += ["num_bands", "num_channels", "num_channels"]
                    else:
                        shape_desc += [f"dim_{i}" for i in range(values.ndim)]
                    ds = _ensure_window_dataset(
                        conn_grp,
                        metric,
                        n_windows=n_windows,
                        per_window_shape=tuple(values.shape),
                        attrs={
                            "description": meta.get("description", spec["description"]),
                            "band_names": meta.get("band_names"),
                            "shape_description": shape_desc,
                        },
                    )
                    ds[written_idx] = values

                written_idx += 1

            # If bad segments were skipped, shrink datasets to the actual number written.
            if written_idx != n_windows:
                _truncate_subject_group(grp, written_idx)

            grp["metadata"].attrs["num_windows"] = int(written_idx)

    return output_h5_path



def _truncate_subject_group(grp: h5py.Group, new_size: int) -> None:
    """
    Repack a subject group in-place after skipping segments.
    HDF5 datasets are immutable in shape unless maxshape is set; for simplicity,
    this helper copies the first `new_size` rows into replacement datasets.
    """
    for subpath in ["windows/raw", "windows/qc", "windows/features", "windows/connectivity"]:
        parent = grp[subpath]
        for name in list(parent.keys()):
            ds = parent[name]
            data = ds[:new_size]
            attrs = dict(ds.attrs)
            del parent[name]
            if data.dtype.kind in {"S", "U", "O"}:
                new_ds = parent.create_dataset(name, data=data, dtype=_vlen_str_dtype())
            else:
                new_ds = parent.create_dataset(
                    name,
                    data=data,
                    compression="gzip" if data.ndim >= 2 else None,
                    compression_opts=4 if data.ndim >= 2 else None,
                    chunks=_window_chunks(data.shape) if data.ndim >= 2 else None,
                )
            for k, v in attrs.items():
                new_ds.attrs[k] = v


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------

def list_available_groups(h5_path: str | os.PathLike) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as h5f:
        out = {
            "bands": json.loads(h5f.attrs["bands_json"]),
            "features": json.loads(h5f.attrs["feature_registry_json"]),
            "connectivity": json.loads(h5f.attrs["connectivity_registry_json"]),
            "subjects": list(h5f.get("subjects", {}).keys()),
        }
    return out



def _iter_subject_ids(h5f: h5py.File, subject_ids: Optional[Sequence[str]]) -> List[str]:
    existing = list(h5f.get("subjects", {}).keys())
    if subject_ids is None:
        return existing
    wanted = [_safe_subject_key(s) for s in subject_ids]
    return [s for s in wanted if s in existing]



def load_feature_family(
    h5_path: str | os.PathLike,
    family: str,
    subject_ids: Optional[Sequence[str]] = None,
    include_raw_metadata: bool = True,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as h5f:
        for sid in _iter_subject_ids(h5f, subject_ids):
            grp = h5f[f"subjects/{sid}"]
            ds = grp[f"windows/features/{family}"]
            entry: Dict[str, Any] = {
                "values": ds[:],
                "feature_names": json.loads(ds.attrs.get("feature_names", "[]")) if isinstance(ds.attrs.get("feature_names", None), str) else ds.attrs.get("feature_names", []),
                "shape": ds.shape,
            }
            if include_raw_metadata:
                entry["segment_id"] = grp["windows/raw/segment_id"][:]
                entry["start_sample"] = grp["windows/raw/start_sample"][:]
                entry["end_sample"] = grp["windows/raw/end_sample"][:]
                entry["label"] = int(grp["metadata"].attrs["label"])
                entry["channel_names"] = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in grp["metadata/channel_names"][:]]
            out[sid] = entry
    return out



def load_connectivity_metric(
    h5_path: str | os.PathLike,
    metric: str,
    subject_ids: Optional[Sequence[str]] = None,
    band: Optional[str | int] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as h5f:
        for sid in _iter_subject_ids(h5f, subject_ids):
            grp = h5f[f"subjects/{sid}"]
            ds = grp[f"windows/connectivity/{metric}"]
            values = ds[:]
            band_names_raw = ds.attrs.get("band_names")
            band_names: Optional[List[str]] = None
            if isinstance(band_names_raw, str):
                band_names = json.loads(band_names_raw)
            elif band_names_raw is not None:
                band_names = list(band_names_raw)

            if band is not None and values.ndim == 4:
                if isinstance(band, str):
                    if band_names is None or band not in band_names:
                        raise KeyError(f"Band '{band}' not found for metric '{metric}'")
                    band_idx = band_names.index(band)
                else:
                    band_idx = int(band)
                values = values[:, band_idx]

            out[sid] = {
                "values": values,
                "band_names": band_names,
                "shape": values.shape,
                "label": int(grp["metadata"].attrs["label"]),
            }
    return out



def load_selected_groups(
    h5_path: str | os.PathLike,
    *,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    subject_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as h5f:
        for sid in _iter_subject_ids(h5f, subject_ids):
            grp = h5f[f"subjects/{sid}"]
            subj_entry: Dict[str, Any] = {
                "label": int(grp["metadata"].attrs["label"]),
                "segment_id": grp["windows/raw/segment_id"][:],
                "start_sample": grp["windows/raw/start_sample"][:],
                "end_sample": grp["windows/raw/end_sample"][:],
                "channel_names": [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in grp["metadata/channel_names"][:]],
                "features": {},
                "connectivity": {},
            }
            for family in feature_families or []:
                subj_entry["features"][family] = grp[f"windows/features/{family}"][:]
            for metric in connectivity_metrics or []:
                subj_entry["connectivity"][metric] = grp[f"windows/connectivity/{metric}"][:]
            out[sid] = subj_entry
    return out


# -----------------------------------------------------------------------------
# Example usage helpers
# -----------------------------------------------------------------------------

def example_usage_snippets() -> Dict[str, str]:
    return {
        "build_from_list": (
            "records = [\n"
            "    {\n"
            "        'subject_id': 'sub-001',\n"
            "        'label': 1,\n"
            "        'sampling_rate': 500,\n"
            "        'channel_names': ['Fp1', 'Fp2', 'F3'],\n"
            "        'windows': [np.random.randn(3, 2000).astype(np.float32) for _ in range(10)],\n"
            "        'start_samples': [i * 1000 for i in range(10)],\n"
            "    }\n"
            "]\n"
            "build_master_eeg_dataset(records, 'master_eeg.h5', overwrite=True)"
        ),
        "build_from_existing_master_dir": (
            "subject_iter = iter_subject_records_from_master_dir('/path/to/master_clean_data')\n"
            "build_master_eeg_dataset(subject_iter, 'master_eeg.h5', overwrite=True)"
        ),
        "load_one_feature": (
            "rbp = load_feature_family('master_eeg.h5', 'relative_band_power')\n"
            "x_sub1 = rbp['sub-001']['values']   # [num_windows, num_channels, num_bands]"
        ),
        "load_one_connectivity": (
            "pli_alpha = load_connectivity_metric('master_eeg.h5', 'pli', band='alpha')\n"
            "adj_sub1 = pli_alpha['sub-001']['values']   # [num_windows, num_channels, num_channels]"
        ),
        "load_several_groups": (
            "data = load_selected_groups(\n"
            "    'master_eeg.h5',\n"
            "    feature_families=['relative_band_power', 'hjorth'],\n"
            "    connectivity_metrics=['pearson', 'coherence'],\n"
            ")\n"
            "sub = data['sub-001']\n"
            "rbp = sub['features']['relative_band_power']\n"
            "hj = sub['features']['hjorth']\n"
            "pearson = sub['connectivity']['pearson']\n"
            "coh = sub['connectivity']['coherence']"
        ),
    }
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from master_builder import list_available_groups, load_feature_family

# Edit this if your class ids are different
LABEL_MAP = {
    0: "C",
    1: "A",
    2: "F",
}


def build_subject_feature_tables(h5_path: str, family: str):
    """
    Load one feature family from the master H5 and build:
    1) subject_df: one row per subject, features averaged over windows and channels
    2) channel_df: one row per subject-channel, features averaged over windows
    3) flat_df: one row per subject, flattened channel x feature vector for PCA
    """
    loaded = load_feature_family(h5_path, family)

    subject_rows = []
    channel_rows = []
    flat_rows = []

    for sid, item in loaded.items():
        x = np.asarray(item["values"], dtype=np.float32)   # [W, C, F]
        if x.ndim != 3:
            raise ValueError(
                f"Expected feature tensor [num_windows, num_channels, num_features], got {x.shape}"
            )

        feature_names = list(item["feature_names"])
        channel_names = list(item["channel_names"])
        label = int(item["label"])
        class_name = LABEL_MAP.get(label, str(label))

        # subject-level average over windows -> [C, F]
        x_sub = x.mean(axis=0)

        # global subject summary: also average over channels -> [F]
        x_global = x_sub.mean(axis=0)

        row = {
            "subject_id": sid,
            "label": label,
            "class_name": class_name,
        }
        row.update({fname: float(val) for fname, val in zip(feature_names, x_global)})
        subject_rows.append(row)

        # channel-level subject summary
        for c_idx, ch in enumerate(channel_names):
            crow = {
                "subject_id": sid,
                "label": label,
                "class_name": class_name,
                "channel": ch,
            }
            crow.update({fname: float(val) for fname, val in zip(feature_names, x_sub[c_idx])})
            channel_rows.append(crow)

        # flattened subject vector for PCA: [C * F]
        flat = x_sub.reshape(-1)
        frow = {
            "subject_id": sid,
            "label": label,
            "class_name": class_name,
        }
        for c_idx, ch in enumerate(channel_names):
            for f_idx, fname in enumerate(feature_names):
                frow[f"{ch}__{fname}"] = float(x_sub[c_idx, f_idx])
        flat_rows.append(frow)

    subject_df = pd.DataFrame(subject_rows)
    channel_df = pd.DataFrame(channel_rows)
    flat_df = pd.DataFrame(flat_rows)

    return subject_df, channel_df, flat_df


def plot_class_mean_heatmap(subject_df: pd.DataFrame, feature_cols: list[str], title: str, output_dir):
    class_means = subject_df.groupby("class_name")[feature_cols].mean()

    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(feature_cols)), 4))
    im = ax.imshow(class_means.values, aspect="auto")

    ax.set_xticks(np.arange(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(class_means.index)))
    ax.set_yticklabels(class_means.index)
    ax.set_title(title)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_class_mean_heatmap-{feature_cols}.png", dpi=300)
    plt.close()


def plot_subject_boxplots(subject_df: pd.DataFrame, feature_cols: list[str], output_dir, title_prefix: str = ""):
    classes = sorted(subject_df["class_name"].unique())
    n = len(feature_cols)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, feat in zip(axes, feature_cols):
        data = [
            subject_df.loc[subject_df["class_name"] == c, feat].dropna().values
            for c in classes
        ]
        ax.boxplot(data, labels=classes)
        ax.set_title(f"{title_prefix}{feat}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Value")

    for ax in axes[len(feature_cols):]:
        ax.axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{output_dir}/plot_subject_boxplots-{feature_cols}.png", dpi=300)
    plt.close()


def plot_pca(flat_df: pd.DataFrame, output_dir, title: str = "PCA by class"):
    meta_cols = {"subject_id", "label", "class_name"}
    feature_cols = [c for c in flat_df.columns if c not in meta_cols]

    X = flat_df[feature_cols].to_numpy(dtype=np.float32)
    y = flat_df["class_name"].to_numpy()

    Xz = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2).fit_transform(Xz)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls in sorted(np.unique(y)):
        idx = y == cls
        ax.scatter(Z[idx, 0], Z[idx, 1], label=cls, s=60)

    for i, sid in enumerate(flat_df["subject_id"].tolist()):
        ax.annotate(sid, (Z[i, 0], Z[i, 1]), fontsize=8, alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{output_dir}/plot_pca.png", dpi=300)
    plt.close()


def plot_channel_feature(channel_df: pd.DataFrame, feature_name: str, channel: str, output_dir):
    sub = channel_df[channel_df["channel"] == channel].copy()
    classes = sorted(sub["class_name"].unique())
    data = [sub.loc[sub["class_name"] == c, feature_name].dropna().values for c in classes]


    print("classes =", classes)
    print("len(classes) =", len(classes))
    print("data shapes =", [np.asarray(d).shape for d in data])



    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=classes)
    plt.title(f"{feature_name} at channel {channel}")
    plt.xlabel("Class")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/plot_channel_feature{feature_name}.png", dpi=300)
    plt.close()
    # plt.show()



# def plot_channel_feature(channel_df, feature_name="rbp_alpha", channel="Pz", output_dir=None):
#     import os
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt

#     # ---- basic checks ----
#     if "channel" not in channel_df.columns:
#         raise ValueError("channel_df must contain a 'channel' column.")
#     if "class_name" not in channel_df.columns:
#         raise ValueError("channel_df must contain a 'class_name' column.")
#     if feature_name not in channel_df.columns:
#         raise ValueError(
#             f"Feature '{feature_name}' not found. "
#             f"Available example columns: {channel_df.columns.tolist()[:20]}"
#         )

#     available_channels = sorted(channel_df["channel"].dropna().unique().tolist())
#     if channel not in available_channels:
#         raise ValueError(
#             f"Channel '{channel}' not found. "
#             f"Available channels: {available_channels}"
#         )

#     sub = channel_df[channel_df["channel"] == channel].copy()
#     classes = sorted(sub["class_name"].dropna().unique().tolist())

#     plot_data = []
#     plot_labels = []

#     for c in classes:
#         vals = sub.loc[sub["class_name"] == c, feature_name]

#         # If duplicate column names exist, vals may be a DataFrame instead of Series
#         if isinstance(vals, pd.DataFrame):
#             if vals.shape[1] != 1:
#                 raise ValueError(
#                     f"Feature '{feature_name}' appears multiple times in channel_df.columns. "
#                     f"Please make feature names unique."
#                 )
#             vals = vals.iloc[:, 0]

#         vals = pd.to_numeric(vals, errors="coerce").dropna().to_numpy().ravel()

#         if vals.size > 0:
#             plot_data.append(vals)
#             plot_labels.append(str(c))

#     if len(plot_data) == 0:
#         raise ValueError(
#             f"No non-empty data found for feature='{feature_name}' at channel='{channel}'."
#         )

#     print("plot_labels:", plot_labels)
#     print("n_labels:", len(plot_labels))
#     print("n_groups:", len(plot_data))
#     print("group_shapes:", [d.shape for d in plot_data])

#     plt.figure(figsize=(6, 4))
#     plt.boxplot(plot_data, tick_labels=plot_labels)   # use tick_labels instead of labels
#     plt.title(f"{feature_name} at channel {channel}")
#     plt.xlabel("Class")
#     plt.ylabel("Value")
#     plt.tight_layout()

#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         save_path = os.path.join(output_dir, f"{feature_name}_{channel}_boxplot.png")
#         plt.savefig(save_path, dpi=300, bbox_inches="tight")
#         print("Saved:", save_path)

#     plt.show()
#     plt.close()

# -----------------------------------------------------------------------------
# Raw .set -> segmented subject_records helpers
# Add these imports near the top of master_builder.py if missing:
# import re
# import glob
# import logging
# import mne
# import pandas as pd
# -----------------------------------------------------------------------------

import re
import glob
import logging
from pathlib import Path

# import mne
import pandas as pd


def extract_subject_id_from_set_path(file_path: str | os.PathLike) -> str:
    """
    Extract subject id like 'sub-001' from a .set file path.
    """
    m = re.search(r"(sub-\d+)", str(file_path))
    if m is None:
        raise ValueError(f"Could not extract subject_id from path: {file_path}")
    return m.group(1)


# def read_eeglab_set(
#     file_path: str | os.PathLike,
#     *,
#     preload: bool = True,
#     eeg_only: bool = True,
# ) -> mne.io.BaseRaw:
#     """
#     Read one EEGLAB .set file with MNE.

#     Returns
#     -------
#     raw : mne.io.BaseRaw
#         EEG-only raw object if eeg_only=True.
#     """
#     logging.getLogger("mne").setLevel(logging.ERROR)
#     raw = mne.io.read_raw_eeglab(str(file_path), preload=preload, verbose="ERROR")

#     if eeg_only:
#         picks = mne.pick_types(raw.info, eeg=True, exclude=[])
#         raw.pick(picks)

#     return raw


def sliding_window_indices(
    n_samples: int,
    window_samples: int,
    step_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return aligned sliding window [start, end) indices.
    """
    if window_samples <= 0:
        raise ValueError(f"window_samples must be > 0, got {window_samples}")
    if step_samples <= 0:
        raise ValueError(f"step_samples must be > 0, got {step_samples}")
    if n_samples < window_samples:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    starts = np.arange(0, n_samples - window_samples + 1, step_samples, dtype=np.int64)
    ends = starts + window_samples
    return starts, ends


def segment_continuous_eeg(
    eeg: np.ndarray,
    sfreq: float,
    *,
    window_sec: float,
    overlap: float,
) -> Tuple[List[np.ndarray], List[int], List[int], List[int]]:
    """
    Segment continuous EEG into fixed windows.

    Parameters
    ----------
    eeg : np.ndarray
        Shape [num_channels, num_timepoints].
    sfreq : float
        Sampling rate.
    window_sec : float
        Window length in seconds.
    overlap : float
        Overlap ratio in [0, 1). Example: 0.5 means 50% overlap.

    Returns
    -------
    windows, segment_ids, start_samples, end_samples
    """
    eeg = _require_2d_window(eeg)

    if not (0.0 <= float(overlap) < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    if float(window_sec) <= 0:
        raise ValueError(f"window_sec must be > 0, got {window_sec}")

    window_samples = int(round(float(window_sec) * float(sfreq)))
    step_samples = int(round(window_samples * (1.0 - float(overlap))))

    if step_samples < 1:
        raise ValueError(
            f"Overlap too large: window_sec={window_sec}, overlap={overlap}, "
            f"window_samples={window_samples}, step_samples={step_samples}"
        )

    starts, ends = sliding_window_indices(
        n_samples=eeg.shape[1],
        window_samples=window_samples,
        step_samples=step_samples,
    )

    windows: List[np.ndarray] = []
    segment_ids: List[int] = []
    start_samples: List[int] = []
    end_samples: List[int] = []

    for seg_id, (s, e) in enumerate(zip(starts.tolist(), ends.tolist())):
        windows.append(eeg[:, s:e].astype(np.float32, copy=False))
        segment_ids.append(int(seg_id))
        start_samples.append(int(s))
        end_samples.append(int(e))

    return windows, segment_ids, start_samples, end_samples


def _encode_label_value(
    label_value: Any,
    label_to_int: Optional[Mapping[Any, int]] = None,
) -> int:
    """
    Convert a label into an integer class id.
    """
    if label_to_int is not None:
        if label_value not in label_to_int:
            raise KeyError(f"Label {label_value!r} not found in label_to_int.")
        return int(label_to_int[label_value])

    if isinstance(label_value, (int, np.integer)):
        return int(label_value)

    raise ValueError(
        "Label must already be an integer, or you must provide label_to_int "
        f"to encode non-integer labels. Got {label_value!r}"
    )


def load_subject_label_map_from_tsv(
    tsv_path: str | os.PathLike,
    *,
    subject_col: str = "participant_id",
    label_col: str = "Group",
    sep: str = "\t",
    label_to_int: Optional[Mapping[Any, int]] = None,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """
    Read subject labels from a TSV/CSV-like table and return:
      - subject_id -> int class_id
      - metadata dict containing the original label mapping

    If label_to_int is None and the label column is non-numeric, labels are
    encoded by sorted unique values.
    """
    df = pd.read_csv(tsv_path, sep=sep)

    if subject_col not in df.columns:
        raise KeyError(f"Missing subject column '{subject_col}' in {tsv_path}")
    if label_col not in df.columns:
        raise KeyError(f"Missing label column '{label_col}' in {tsv_path}")

    raw_labels = df[label_col].tolist()

    if label_to_int is None:
        # project-specific default for AHEAP
        label_to_int = {"C": 0, "A": 1, "F": 2}
        
    subject_to_label: Dict[str, int] = {}
    for _, row in df.iterrows():
        sid = str(row[subject_col])
        subject_to_label[sid] = _encode_label_value(row[label_col], label_to_int)

    meta = {
        "subject_col": subject_col,
        "label_col": label_col,
        "label_to_int": dict(label_to_int),
    }
    return subject_to_label, meta


def build_subject_record_from_set_file(
    file_path: str | os.PathLike,
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float,
    overlap: float,
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
    session_info: Optional[Mapping[str, Any]] = None,
    extra_recording_info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Read one raw .set file and convert it into the subject_record format that
    build_master_eeg_dataset(...) already expects.

    Notes
    -----
    - Raw MNE EEGLAB values are typically in Volts.
    - We keep raw windows in their original physical scale here.
    - QC conversion to microvolts is already handled later by compute_window_qc(...)
      through qc_input_unit='auto' or 'v'.
    """
    file_path = str(file_path)
    subject_id = extract_subject_id_from_set_path(file_path)

    if subject_id not in subject_label_map:
        raise KeyError(f"Missing label for subject {subject_id}")

    raw = read_eeglab_set(file_path, preload=True, eeg_only=True)

    sfreq = float(raw.info["sfreq"])
    if expected_sfreq is not None and not math.isclose(sfreq, float(expected_sfreq), rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Unexpected sfreq for {subject_id}: got {sfreq}, expected {expected_sfreq}"
        )

    ch_names = list(raw.ch_names)
    if rename_channels is not None:
        ch_names = [rename_channels.get(ch, ch) for ch in ch_names]

    eeg = raw.get_data().astype(np.float32, copy=False)   # usually Volts from MNE
    windows, segment_ids, start_samples, end_samples = segment_continuous_eeg(
        eeg,
        sfreq,
        window_sec=window_sec,
        overlap=overlap,
    )

    recording_info = {
        "source_file": file_path,
        "n_channels_raw": int(eeg.shape[0]),
        "n_samples_raw": int(eeg.shape[1]),
        "duration_sec_raw": float(eeg.shape[1] / sfreq),
        "window_sec": float(window_sec),
        "overlap": float(overlap),
        "task_name": Path(file_path).stem,
        "mne_info_description": str(raw.info.get("description", "")),
    }
    if extra_recording_info is not None:
        recording_info.update({str(k): _as_jsonable(v) for k, v in extra_recording_info.items()})

    label_int = int(subject_label_map[subject_id])

    return {
        "subject_id": subject_id,
        "label": label_int,
        "class_id": label_int,
        "sampling_rate": sfreq,
        "channel_names": ch_names,
        "montage_type": "referential",
        "session_info": None if session_info is None else dict(session_info),
        "recording_info": recording_info,
        "windows": windows,
        "segment_ids": segment_ids,
        "start_samples": start_samples,
        "end_samples": end_samples,
    }


def iter_subject_records_from_set_files(
    file_paths: Sequence[str | os.PathLike],
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float,
    overlap: float,
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Yield subject_records directly from raw .set files.
    """
    for file_path in file_paths:
        yield build_subject_record_from_set_file(
            file_path=file_path,
            subject_label_map=subject_label_map,
            window_sec=window_sec,
            overlap=overlap,
            rename_channels=rename_channels,
            expected_sfreq=expected_sfreq,
        )


def find_set_files_under_derivatives(
    derivatives_root: str | os.PathLike,
    *,
    pattern: str = "sub-*/eeg/*.set",
) -> List[str]:
    """
    Find derivative .set files such as:
      derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set
    """
    derivatives_root = str(derivatives_root)
    paths = sorted(glob.glob(os.path.join(derivatives_root, pattern)))
    if len(paths) == 0:
        raise FileNotFoundError(
            f"No .set files found under {derivatives_root} with pattern {pattern}"
        )
    return paths


def build_master_eeg_dataset_from_set_files(
    file_paths: Sequence[str | os.PathLike],
    output_h5_path: str | os.PathLike,
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float = 4.0,
    overlap: float = 0.5,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
    overwrite: bool = False,
    skip_bad_segments: bool = True,
    bipolar_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    montage_type_if_bipolar: str = "bipolar",
    target_sampling_rate: Optional[float] = None,
    qc_kwargs: Optional[Mapping[str, Any]] = None,
    qc_input_unit: str = "auto",
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
) -> str:
    """
    End-to-end helper:
      raw .set -> segmentation -> QC cleaning -> feature extraction ->
      connectivity extraction -> HDF5 master file
    """
    subject_records = iter_subject_records_from_set_files(
        file_paths=file_paths,
        subject_label_map=subject_label_map,
        window_sec=window_sec,
        overlap=overlap,
        rename_channels=rename_channels,
        expected_sfreq=expected_sfreq,
    )

    return build_master_eeg_dataset(
        subject_records=subject_records,
        output_h5_path=output_h5_path,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=bands,
        overwrite=overwrite,
        skip_bad_segments=skip_bad_segments,
        bipolar_pairs=bipolar_pairs,
        montage_type_if_bipolar=montage_type_if_bipolar,
        target_sampling_rate=target_sampling_rate,
        qc_kwargs=qc_kwargs,
        qc_input_unit=qc_input_unit,
    )


def build_master_eeg_dataset_from_derivatives_dir(
    derivatives_root: str | os.PathLike,
    output_h5_path: str | os.PathLike,
    *,
    subject_label_map: Mapping[str, int],
    window_sec: float = 4.0,
    overlap: float = 0.5,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    bands: Optional[Mapping[str, Tuple[float, float]]] = None,
    overwrite: bool = False,
    skip_bad_segments: bool = True,
    bipolar_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    montage_type_if_bipolar: str = "bipolar",
    target_sampling_rate: Optional[float] = None,
    qc_kwargs: Optional[Mapping[str, Any]] = None,
    qc_input_unit: str = "auto",
    rename_channels: Optional[Mapping[str, str]] = None,
    expected_sfreq: Optional[float] = None,
    pattern: str = "sub-*/eeg/*.set",
) -> str:
    """
    Convenience wrapper for the common derivatives directory layout.
    """
    file_paths = find_set_files_under_derivatives(
        derivatives_root=derivatives_root,
        pattern=pattern,
    )

    return build_master_eeg_dataset_from_set_files(
        file_paths=file_paths,
        output_h5_path=output_h5_path,
        subject_label_map=subject_label_map,
        window_sec=window_sec,
        overlap=overlap,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=bands,
        overwrite=overwrite,
        skip_bad_segments=skip_bad_segments,
        bipolar_pairs=bipolar_pairs,
        montage_type_if_bipolar=montage_type_if_bipolar,
        target_sampling_rate=target_sampling_rate,
        qc_kwargs=qc_kwargs,
        qc_input_unit=qc_input_unit,
        rename_channels=rename_channels,
        expected_sfreq=expected_sfreq,
    )

# if __name__ == "__main__":
#     import argparse
#     import config
#     # parser = argparse.ArgumentParser(description="Build modular HDF5 master EEG dataset.")
#     # parser.add_argument("--master_dir", type=str, help="Directory containing subject_manifest.csv and subject .pt files.")
#     # parser.add_argument("--output", type=str, required=True, help="Output .h5 path.")
#     # parser.add_argument("--overwrite", action="store_true", help="Rebuild output file from scratch.")
#     # parser.add_argument("--skip_bad_segments", action="store_true", help="Drop QC-flagged windows.")
#     # parser.add_argument("--qc_input_unit", type=str, default="auto", help="QC input unit: auto, uv, or v.")
#     # args = parser.parse_args()
#     master_dir = '/home/anphan/Documents/EEG_Project/AHEAP_data/master_clean_data'
#     output = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
#     if master_dir:
#         subject_iter = iter_subject_records_from_master_dir(master_dir)
#         build_master_eeg_dataset(
#             subject_iter,
#             output,
#             overwrite=True,
#             skip_bad_segments=True,
#             target_sampling_rate=250,
#             bipolar_pairs=config.bi23_channel_names,
#             montage_type_if_bipolar="bi23",
#             qc_input_unit="auto",
#         )
#     else:
#         raise ValueError("Please provide --master_dir or import the module and call build_master_eeg_dataset(...) directly.")

if __name__ == "__main__":
    import argparse
    import config

    parser = argparse.ArgumentParser(description="Build master EEG HDF5 directly from raw .set files.")
    # parser.add_argument("--derivatives_root", type=str, required=True)
    # parser.add_argument("--participants_tsv", type=str, required=True)
    # parser.add_argument("--output", type=str, required=True)

    parser.add_argument("--window_sec", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--montage", type=str, default="mono", choices=["mono", "bi23", "bi30"])

    parser.add_argument("--feature_families", type=str, default=",".join(FEATURE_REGISTRY.keys()))
    parser.add_argument("--connectivity_metrics", type=str, default=",".join(CONNECTIVITY_REGISTRY.keys()))

    parser.add_argument("--label_col", type=str, default="Group")
    parser.add_argument("--subject_col", type=str, default="participant_id")

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_bad_segments", action="store_true")
    parser.add_argument("--target_sampling_rate", type=int, default=250)
    parser.add_argument("--qc_input_unit", type=str, default="v", choices=["auto", "uv", "v"])


    args = parser.parse_args()

    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    connectivity_metrics = [x.strip() for x in args.connectivity_metrics.split(",") if x.strip()]


    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    class_set ="all3" 
    root_path = f'/mnt/data/anphan/AHEAP_data/all_h5_master_files_{args.target_sampling_rate}hz'
    os.makedirs(root_path,exist_ok=True)
    folder_name = f'{args.montage}_duration{args.window_sec}_overlap{args.overlap}'
    output_dir = os.path.join(root_path, folder_name)
    os.makedirs(output_dir,exist_ok=True)
    h5_output_file = os.path.join(output_dir, 'master.h5')
    # num_classes, class_labels, class_names = get_class(class_set, dataset)
    # data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)


    subject_label_map, label_meta = load_subject_label_map_from_tsv(
        tsv_path,
        subject_col=args.subject_col,
        label_col=args.label_col,
        sep="\t",
        label_to_int={"C": 0, "A": 1, "F": 2},
    )
    # print(subject_label_map)
    # print(label_meta)
    if args.montage == "mono":
        bipolar_pairs = None
        montage_type_if_bipolar = "referential"
    elif args.montage == "bi23":
        bipolar_pairs = config.bi23_channel_names   # must be list[tuple[str, str]]
        montage_type_if_bipolar = "bi23"
    elif args.montage == "bi30":
        bipolar_pairs = config.bi30_channel_names   # must be list[tuple[str, str]]
        montage_type_if_bipolar = "bi30"
    else:
        raise ValueError(f"Unsupported montage: {args.montage}")

    out_path = build_master_eeg_dataset_from_derivatives_dir(
        derivatives_root=data_dir,
        output_h5_path=h5_output_file,
        subject_label_map=subject_label_map,
        window_sec=args.window_sec,
        overlap=args.overlap,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        bands=DEFAULT_BANDS,
        overwrite=args.overwrite,
        skip_bad_segments=args.skip_bad_segments,
        bipolar_pairs=bipolar_pairs,
        montage_type_if_bipolar=montage_type_if_bipolar,
        target_sampling_rate=args.target_sampling_rate,
        qc_kwargs=None,
        qc_input_unit=args.qc_input_unit,
        expected_sfreq=500,
    )

    print(f"Saved master dataset to: {out_path}")
    print("Label encoding:", label_meta["label_to_int"])

# if __name__ == "__main__":
    # h5_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
    # # output_dir = '/home/anphan/Documents/EEG_Project/AHEAP_data/EDA/master_full_data_mono_250hz'
    # # os.makedirs(output_dir,exist_ok=True)


    # with h5py.File(h5_path, "r") as f:
    #     sid = list(f["subjects"].keys())[0]   # pick first subject
    #     raw = f[f"subjects/{sid}/windows/raw/eeg"][0]      # [C, T]
    #     qc_max_abs_uv = f[f"subjects/{sid}/windows/qc/max_abs_uv"][0]

    #     raw_max_abs = float(np.max(np.abs(raw)))

    #     print("subject:", sid)
    #     print("raw max abs:", raw_max_abs)
    #     print("qc max_abs_uv:", float(qc_max_abs_uv))
    #     print("ratio qc/raw:", float(qc_max_abs_uv / max(raw_max_abs, 1e-12)))
    #     x = f[f"subjects/{sid}/windows/raw/eeg"][:5]   # first 5 windows
    #     print("global min:", x.min())
    #     print("global max:", x.max())
    #     print("median abs:", np.median(np.abs(x)))
    #     print("95th pct abs:", np.percentile(np.abs(x), 95))
    


    # import mne
    # import numpy as np

    # file_path = "/mnt/data/anphan/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"

    # raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose="ERROR")
    # raw.pick(mne.pick_types(raw.info, eeg=True, exclude=[]))

    # data_v = raw.get_data()          # what MNE gives you
    # data_uv = data_v * 1e6           # converted to microvolts

    # print("sfreq:", raw.info["sfreq"])
    # print("shape:", data_v.shape)
    # print("channels:", raw.ch_names)

    # print("\n--- raw.get_data() scale ---")
    # print("min:", data_v.min())
    # print("max:", data_v.max())
    # print("max abs:", np.max(np.abs(data_v)))
    # print("median abs:", np.median(np.abs(data_v)))
    # print("95th pct abs:", np.percentile(np.abs(data_v), 95))

    # print("\n--- after *1e6 (microvolts) ---")
    # print("min uV:", data_uv.min())
    # print("max uV:", data_uv.max())
    # print("max abs uV:", np.max(np.abs(data_uv)))
    # print("median abs uV:", np.median(np.abs(data_uv)))
    # print("95th pct abs uV:", np.percentile(np.abs(data_uv), 95))
    # See what is available
    # info = list_available_groups(h5_path)
    # print("Available feature families:", list(info["features"].keys()))
    # print("Available connectivity metrics:", list(info["connectivity"].keys()))

    # # Example 1: relative band power
    # family = "statistical"
    # subject_df, channel_df, flat_df = build_subject_feature_tables(h5_path, family)



    # print("channel_df columns:", channel_df.columns.tolist())
    # print("nrows =", len(channel_df))

    # if "channel" in channel_df.columns:
    #     print("unique channels:", sorted(channel_df["channel"].dropna().astype(str).unique().tolist()))

    # if "class_name" in channel_df.columns:
    #     print("class_name counts:")
    #     print(channel_df["class_name"].value_counts(dropna=False))
    # else:
    #     print("No class_name column found")

    # if "label" in channel_df.columns:
    #     print("label counts:")
    #     print(channel_df["label"].value_counts(dropna=False))
    # elif "class_id" in channel_df.columns:
    #     print("class_id counts:")
    #     print(channel_df["class_id"].value_counts(dropna=False))

    # print(channel_df.head())



    # feature_cols = [c for c in subject_df.columns if c not in ["subject_id", "label", "class_name"]]

    # # Heatmap of class means
    # plot_class_mean_heatmap(
    #     subject_df,
    #     feature_cols,
    #     title=f"Class mean features: {family} (subject average over windows/channels)",
    #     output_dir=output_dir
    # )

    # # Boxplots for selected features
    # # selected = [c for c in feature_cols if c in ["rbp_delta", "rbp_theta", "rbp_alpha", "rbp_beta", "rbp_gamma"]]
    # plot_subject_boxplots(subject_df, feature_cols, title_prefix=f"{family}: ", output_dir=output_dir)

    # # PCA using flattened channel x feature subject vectors
    # plot_pca(flat_df, title=f"PCA of {family} (subject-level)", output_dir=output_dir)

    # # Channel-specific view
    # if "rbp_alpha" in channel_df.columns:
    #     plot_channel_feature(channel_df, feature_name="rbp_alpha", channel="Pz", output_dir=output_dir)

    # subject_df, channel_df, flat_df = build_subject_feature_tables(h5_path, "hjorth")
    # subject_df, channel_df, flat_df = build_subject_feature_tables(h5_path, "statistical")
    # build_subject_feature_tables(h5_path, "spectral_entropy")
    # build_subject_feature_tables(h5_path, "higuchi_fd")
    # build_subject_feature_tables(h5_path, "wavelet_energy")