# preprocessing.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import signal, stats

try:
    from .dataset import SubjectRecord
except ImportError:
    try:
        from .datasets import SubjectRecord
    except ImportError:
        SubjectRecord = Any  # type: ignore[misc,assignment]


ArrayLike = np.ndarray
ReferenceMode = Literal["none", "average", "channels", "custom"]
ZScoreMode = Literal[
    "none",
    "global",
    "per_channel",
    "subject_wise_feature",
    "channel_wise_feature",
]


DEFAULT_QC_THRESHOLDS: dict[str, float] = {
    "max_abs_uv": 200.0,
    "max_ptp_uv": 300.0,
    "global_std_uv": 80.0,
    "flat_std_uv": 0.05,
    "flat_channel_frac": 0.30,
    "kurtosis_thr": 8.0,
    "high_kurtosis_frac": 0.50,
}

POSTERIOR_CHANNELS: set[str] = {"P3", "P4", "Pz", "O1", "O2", "T5", "T6"}


@dataclass(slots=True)
class PreparedSubjectWindows:
    """
    Output bundle from :func:`prepare_subject_windows`.

    Attributes
    ----------
    subject_id:
        Subject identifier.
    dataset_name:
        Dataset source name.
    label:
        Canonical label name.
    label_id:
        Integer label ID.
    sfreq:
        Sampling rate after preprocessing.
    channels:
        Channel names aligned to the window tensors.
    windows:
        Final kept windows with shape [num_valid_windows, num_channels, num_timepoints].
    window_df:
        Per-window metadata aligned row-wise with ``windows``.
    macro_df:
        Per-window macro-group mapping aligned by ``segment_id``.
    qc_df:
        QC table for all candidate windows before dropping.
    metadata:
        Additional run metadata and preprocessing settings.
    """

    subject_id: str
    dataset_name: str
    label: str
    label_id: int
    sfreq: float
    channels: list[str]
    windows: np.ndarray
    window_df: pd.DataFrame
    macro_df: pd.DataFrame
    qc_df: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


def bandpass_filter_signal(
    signal_array: np.ndarray,
    sfreq: float,
    low_freq: float | None,
    high_freq: float | None,
    *,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass / highpass / lowpass filtering.

    Parameters
    ----------
    signal_array:
        Input array. Typical shapes are [channels, time] or [windows, channels, time].
    sfreq:
        Sampling rate in Hz.
    low_freq:
        Low cutoff in Hz. Use None for lowpass only.
    high_freq:
        High cutoff in Hz. Use None for highpass only.
    order:
        Butterworth filter order.
    axis:
        Time axis.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as input.
    """
    x = _as_float32_array(signal_array, name="signal_array")
    sfreq = _validate_positive_float(sfreq, "sfreq")
    order = _validate_positive_int(order, "order")
    axis = _normalize_axis(axis, x.ndim)

    if low_freq is None and high_freq is None:
        return x.copy()

    nyquist = sfreq / 2.0

    if low_freq is not None:
        low_freq = float(low_freq)
        if low_freq <= 0 or low_freq >= nyquist:
            raise ValueError(
                f"low_freq must be in (0, {nyquist}), got {low_freq}."
            )

    if high_freq is not None:
        high_freq = float(high_freq)
        if high_freq <= 0 or high_freq >= nyquist:
            raise ValueError(
                f"high_freq must be in (0, {nyquist}), got {high_freq}."
            )

    if low_freq is not None and high_freq is not None and low_freq >= high_freq:
        raise ValueError(
            f"Expected low_freq < high_freq, got low_freq={low_freq} and high_freq={high_freq}."
        )

    if low_freq is not None and high_freq is not None:
        wn: float | tuple[float, float] = (low_freq, high_freq)
        btype = "bandpass"
    elif low_freq is not None:
        wn = low_freq
        btype = "highpass"
    else:
        wn = float(high_freq)  # type: ignore[arg-type]
        btype = "lowpass"

    sos = signal.butter(
        N=order,
        Wn=wn,
        btype=btype,
        fs=sfreq,
        output="sos",
    )
    y = signal.sosfiltfilt(sos, x, axis=axis)
    return y.astype(np.float32, copy=False)


def notch_filter_signal(
    signal_array: np.ndarray,
    sfreq: float,
    notch_freqs: float | Sequence[float] | None,
    *,
    quality_factor: float = 30.0,
    axis: int = -1,
) -> np.ndarray:
    """
    Apply one or more zero-phase notch filters.

    Parameters
    ----------
    signal_array:
        Input array. Typical shapes are [channels, time] or [windows, channels, time].
    sfreq:
        Sampling rate in Hz.
    notch_freqs:
        One frequency or a sequence of frequencies in Hz.
        Use None to skip notch filtering.
    quality_factor:
        IIR notch quality factor.
    axis:
        Time axis.

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as input.
    """
    x = _as_float32_array(signal_array, name="signal_array")
    sfreq = _validate_positive_float(sfreq, "sfreq")
    quality_factor = _validate_positive_float(quality_factor, "quality_factor")
    axis = _normalize_axis(axis, x.ndim)

    if notch_freqs is None:
        return x.copy()

    if isinstance(notch_freqs, (int, float, np.integer, np.floating)):
        freq_list = [float(notch_freqs)]
    else:
        freq_list = [float(freq) for freq in notch_freqs]

    y = x.copy()
    nyquist = sfreq / 2.0
    for freq in freq_list:
        if freq <= 0 or freq >= nyquist:
            raise ValueError(
                f"Each notch frequency must be in (0, {nyquist}), got {freq}."
            )
        b, a = signal.iirnotch(w0=freq, Q=quality_factor, fs=sfreq)
        y = signal.filtfilt(b, a, y, axis=axis)

    return y.astype(np.float32, copy=False)


def reference_signal(
    signal_array: np.ndarray,
    *,
    mode: ReferenceMode = "none",
    channel_names: Sequence[str] | None = None,
    ref_channels: Sequence[str | int] | None = None,
    custom_reference_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Apply a simple referencing strategy to [channels, time] EEG.

    Parameters
    ----------
    signal_array:
        Input EEG array with shape [channels, time].
    mode:
        Referencing mode:
        - "none": return unchanged
        - "average": subtract average across all channels
        - "channels": subtract mean of selected reference channels
        - "custom": call ``custom_reference_fn``
    channel_names:
        Channel names used when ``ref_channels`` are provided as strings.
    ref_channels:
        Reference channels as names or indices for mode="channels".
    custom_reference_fn:
        Custom function for mode="custom".

    Returns
    -------
    np.ndarray
        Re-referenced signal with shape [channels, time].
    """
    x = _require_2d_signal(signal_array, name="signal_array")
    mode = str(mode).lower()

    if mode == "none":
        return x.copy()

    if mode == "average":
        ref = x.mean(axis=0, keepdims=True)
        return (x - ref).astype(np.float32, copy=False)

    if mode == "channels":
        ref_idx = _resolve_channel_indices(
            ref_channels=ref_channels,
            channel_names=channel_names,
            n_channels=x.shape[0],
        )
        if len(ref_idx) == 0:
            raise ValueError("mode='channels' requires at least one reference channel.")
        ref = x[ref_idx].mean(axis=0, keepdims=True)
        return (x - ref).astype(np.float32, copy=False)

    if mode == "custom":
        if custom_reference_fn is None:
            raise ValueError("mode='custom' requires custom_reference_fn.")
        y = custom_reference_fn(x.copy())
        y = _require_2d_signal(y, name="custom-referenced signal")
        if y.shape != x.shape:
            raise ValueError(
                f"custom_reference_fn must preserve shape {x.shape}, got {y.shape}."
            )
        return y.astype(np.float32, copy=False)

    raise ValueError(
        f"Unsupported reference mode {mode!r}. "
        "Use one of {'none', 'average', 'channels', 'custom'}."
    )


def zscore_signal(
    x: np.ndarray,
    *,
    mode: ZScoreMode = "per_channel",
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Z-score normalize raw EEG or subject-level feature tensors.

    Supported shapes and modes
    --------------------------
    Raw EEG:
        - [channels, time]
        - [windows, channels, time]

    Feature tensors:
        - [windows, channels, features]

    Modes
    -----
    - "none"
    - "global"
    - "per_channel"
    - "subject_wise_feature"
    - "channel_wise_feature"

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray]]
        Normalized array and normalization statistics.
    """
    arr = _as_float32_array(x, name="x")
    eps = _validate_positive_float(eps, "eps")

    mode = str(mode).lower()
    if mode == "none":
        stats_dict = {
            "mean": np.zeros((1,), dtype=np.float32),
            "std": np.ones((1,), dtype=np.float32),
        }
        return arr.copy(), stats_dict

    if mode == "global":
        mean = np.array([arr.mean()], dtype=np.float32)
        std = _safe_std(np.array([arr.std()], dtype=np.float32), eps=eps)
        y = (arr - mean[0]) / std[0]
        return y.astype(np.float32, copy=False), {"mean": mean, "std": std}

    if mode == "per_channel":
        if arr.ndim == 2:
            mean = arr.mean(axis=1, keepdims=True).astype(np.float32)
            std = _safe_std(arr.std(axis=1, keepdims=True).astype(np.float32), eps=eps)
            y = (arr - mean) / std
            return y.astype(np.float32, copy=False), {"mean": mean, "std": std}

        if arr.ndim == 3:
            # Per-channel normalization across the last dimension.
            mean = arr.mean(axis=-1, keepdims=True).astype(np.float32)
            std = _safe_std(arr.std(axis=-1, keepdims=True).astype(np.float32), eps=eps)
            y = (arr - mean) / std
            return y.astype(np.float32, copy=False), {"mean": mean, "std": std}

        raise ValueError(
            "mode='per_channel' expects an array with 2 or 3 dimensions."
        )

    if mode == "subject_wise_feature":
        _require_3d_array(arr, name="x")
        flat = arr.reshape(-1, arr.shape[-1]).astype(np.float32)
        mean = flat.mean(axis=0, keepdims=True).astype(np.float32)
        std = _safe_std(flat.std(axis=0, keepdims=True).astype(np.float32), eps=eps)
        y = ((flat - mean) / std).reshape(arr.shape).astype(np.float32)
        return y, {"mean": mean.squeeze(0), "std": std.squeeze(0)}

    if mode == "channel_wise_feature":
        _require_3d_array(arr, name="x")
        mean = arr.mean(axis=0).astype(np.float32)
        std = _safe_std(arr.std(axis=0).astype(np.float32), eps=eps)
        y = ((arr - mean[None, :, :]) / std[None, :, :]).astype(np.float32)
        return y, {"mean": mean, "std": std}

    raise ValueError(
        f"Unsupported z-score mode {mode!r}. "
        "Use one of {'none', 'global', 'per_channel', 'subject_wise_feature', 'channel_wise_feature'}."
    )


def segment_signal(
    signal_array: np.ndarray,
    sfreq: float,
    *,
    window_sec: float,
    overlap: float = 0.0,
    start_offset_sec: float = 0.0,
    end_offset_sec: float = 0.0,
    keep_partial: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Segment one [channels, time] signal into short windows.

    Parameters
    ----------
    signal_array:
        Input EEG array with shape [channels, time].
    sfreq:
        Sampling rate in Hz.
    window_sec:
        Window length in seconds.
    overlap:
        Fractional overlap in [0, 1).
    start_offset_sec:
        Trim this much time from the start before segmenting.
    end_offset_sec:
        Trim this much time from the end before segmenting.
    keep_partial:
        If True, keeps a final shorter window and pads it with edge values
        to full length.

    Returns
    -------
    tuple[np.ndarray, pandas.DataFrame]
        Windows with shape [num_windows, channels, time] and per-window metadata.
    """
    x = _require_2d_signal(signal_array, name="signal_array")
    sfreq = _validate_positive_float(sfreq, "sfreq")
    window_sec = _validate_positive_float(window_sec, "window_sec")

    if not (0.0 <= float(overlap) < 1.0):
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")

    start_offset_samples = int(round(_validate_nonnegative_float(start_offset_sec, "start_offset_sec") * sfreq))
    end_offset_samples = int(round(_validate_nonnegative_float(end_offset_sec, "end_offset_sec") * sfreq))

    n_total = x.shape[1]
    start_idx = start_offset_samples
    end_idx = n_total - end_offset_samples

    if start_idx >= end_idx:
        raise ValueError(
            f"Offsets remove the entire signal: start_idx={start_idx}, end_idx={end_idx}, n_total={n_total}."
        )

    x_use = x[:, start_idx:end_idx]
    window_samples = int(round(window_sec * sfreq))
    if window_samples < 1:
        raise ValueError("window_sec is too small for the given sampling rate.")

    step_samples = int(round(window_samples * (1.0 - overlap)))
    if step_samples < 1:
        raise ValueError("overlap is too large; resulting step size is < 1 sample.")

    starts = list(range(0, max(x_use.shape[1] - window_samples + 1, 0), step_samples))

    windows: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for seg_idx, local_start in enumerate(starts):
        local_end = local_start + window_samples
        if local_end > x_use.shape[1]:
            continue

        w = x_use[:, local_start:local_end]
        global_start = start_idx + local_start
        global_end = start_idx + local_end

        windows.append(w.astype(np.float32, copy=False))
        rows.append(
            {
                "segment_id": seg_idx,
                "start_sample": int(global_start),
                "end_sample": int(global_end),
                "start_sec": float(global_start / sfreq),
                "end_sec": float(global_end / sfreq),
                "window_sec": float(window_samples / sfreq),
                "num_samples": int(window_samples),
            }
        )

    if keep_partial and (len(rows) == 0 or rows[-1]["end_sample"] < end_idx):
        last_start = 0 if len(rows) == 0 else rows[-1]["start_sample"] + step_samples
        last_end = min(last_start + window_samples, end_idx)
        if last_end > last_start:
            w = x[:, last_start:last_end]
            if w.shape[1] < window_samples:
                pad_width = [(0, 0), (0, window_samples - w.shape[1])]
                w = np.pad(w, pad_width, mode="edge")
            seg_idx = len(rows)
            windows.append(w.astype(np.float32, copy=False))
            rows.append(
                {
                    "segment_id": seg_idx,
                    "start_sample": int(last_start),
                    "end_sample": int(last_end),
                    "start_sec": float(last_start / sfreq),
                    "end_sec": float(last_end / sfreq),
                    "window_sec": float((last_end - last_start) / sfreq),
                    "num_samples": int(last_end - last_start),
                }
            )

    if len(windows) == 0:
        empty_windows = np.empty((0, x.shape[0], window_samples), dtype=np.float32)
        empty_df = pd.DataFrame(
            columns=[
                "segment_id",
                "start_sample",
                "end_sample",
                "start_sec",
                "end_sec",
                "window_sec",
                "num_samples",
            ]
        )
        return empty_windows, empty_df

    windows_arr = np.stack(windows, axis=0).astype(np.float32, copy=False)
    window_df = pd.DataFrame(rows)
    return windows_arr, window_df


def build_macro_groups(
    window_df: pd.DataFrame,
    sfreq: float,
    *,
    macro_duration_sec: float,
    macro_step_sec: float | None = None,
) -> pd.DataFrame:
    """
    Assign short windows to macro blocks.

    This function uses each window center to produce a unique macro assignment.
    That keeps the mapping simple and stable for later macro-graph construction.

    Parameters
    ----------
    window_df:
        DataFrame from :func:`segment_signal` or a compatible table with
        ``segment_id``, ``start_sample``, and ``end_sample``.
    sfreq:
        Sampling rate in Hz.
    macro_duration_sec:
        Macro-block duration in seconds, e.g. 300 for 5 minutes.
    macro_step_sec:
        Step between successive macro blocks. Defaults to ``macro_duration_sec``
        for non-overlapping macro blocks.

    Returns
    -------
    pandas.DataFrame
        One row per input window with macro assignment metadata.
    """
    _validate_positive_float(sfreq, "sfreq")
    macro_duration_sec = _validate_positive_float(macro_duration_sec, "macro_duration_sec")
    if macro_step_sec is None:
        macro_step_sec = macro_duration_sec
    macro_step_sec = _validate_positive_float(macro_step_sec, "macro_step_sec")

    required = {"segment_id", "start_sample", "end_sample"}
    missing = required.difference(window_df.columns)
    if missing:
        raise KeyError(f"window_df is missing required columns: {sorted(missing)}")

    if len(window_df) == 0:
        return pd.DataFrame(
            columns=[
                "segment_id",
                "macro_id",
                "macro_start_sample",
                "macro_end_sample",
                "macro_start_sec",
                "macro_end_sec",
                "window_index_in_macro",
                "num_windows_in_macro",
            ]
        )

    df = window_df.copy().reset_index(drop=True)
    macro_duration_samples = int(round(macro_duration_sec * sfreq))
    macro_step_samples = int(round(macro_step_sec * sfreq))
    if macro_duration_samples < 1 or macro_step_samples < 1:
        raise ValueError("Macro duration/step produce fewer than 1 sample.")

    center_sample = ((df["start_sample"].to_numpy() + df["end_sample"].to_numpy()) / 2.0).astype(np.int64)
    macro_id = np.floor_divide(center_sample, macro_step_samples).astype(np.int64)
    macro_start_sample = macro_id * macro_step_samples
    macro_end_sample = macro_start_sample + macro_duration_samples

    out = pd.DataFrame(
        {
            "segment_id": df["segment_id"].astype(np.int64),
            "macro_id": macro_id,
            "macro_start_sample": macro_start_sample,
            "macro_end_sample": macro_end_sample,
            "macro_start_sec": macro_start_sample / sfreq,
            "macro_end_sec": macro_end_sample / sfreq,
        }
    )

    # Stable ordering inside each macro group.
    out = out.sort_values(["macro_id", "segment_id"]).reset_index(drop=True)
    out["window_index_in_macro"] = out.groupby("macro_id").cumcount().astype(np.int64)
    out["num_windows_in_macro"] = out.groupby("macro_id")["segment_id"].transform("count").astype(np.int64)
    return out


def drop_bad_windows(
    windows: np.ndarray,
    sfreq: float,
    *,
    channel_names: Sequence[str] | None = None,
    window_df: pd.DataFrame | None = None,
    input_unit: str = "auto",
    qc_thresholds: Mapping[str, float] | None = None,
    min_valid_windows: int = 1,
    max_windows_keep: int | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    QC-filter segmented windows.

    Parameters
    ----------
    windows:
        Window tensor with shape [num_windows, channels, time].
    sfreq:
        Sampling rate in Hz.
    channel_names:
        Optional channel names.
    window_df:
        Optional metadata table aligned with ``windows``.
    input_unit:
        "auto", "uv", or "v".
    qc_thresholds:
        Optional threshold overrides.
    min_valid_windows:
        Minimum valid windows required for the subject to be considered usable.
    max_windows_keep:
        Optional cap on kept valid windows.
    random_state:
        Seed used when subsampling valid windows.

    Returns
    -------
    tuple
        ``(windows_kept, window_df_kept, qc_df_all, summary_dict)``
    """
    arr = _as_float32_array(windows, name="windows")
    _require_3d_array(arr, name="windows")
    sfreq = _validate_positive_float(sfreq, "sfreq")
    min_valid_windows = _validate_positive_int(min_valid_windows, "min_valid_windows")

    if window_df is None:
        window_df = pd.DataFrame({"segment_id": np.arange(arr.shape[0], dtype=np.int64)})
    else:
        if len(window_df) != arr.shape[0]:
            raise ValueError(
                f"window_df length ({len(window_df)}) must match number of windows ({arr.shape[0]})."
            )
        window_df = window_df.reset_index(drop=True).copy()

    if channel_names is None:
        channel_names = [f"Ch{i + 1}" for i in range(arr.shape[1])]
    channel_names = [str(ch) for ch in channel_names]
    if len(channel_names) != arr.shape[1]:
        raise ValueError(
            f"channel_names length ({len(channel_names)}) must match num_channels ({arr.shape[1]})."
        )

    thresholds = dict(DEFAULT_QC_THRESHOLDS)
    if qc_thresholds is not None:
        thresholds.update({str(k): float(v) for k, v in qc_thresholds.items()})

    qc_rows: list[dict[str, Any]] = []
    valid_mask = np.ones(arr.shape[0], dtype=bool)

    for i in range(arr.shape[0]):
        metrics = _compute_window_qc(
            arr[i],
            sfreq=sfreq,
            channel_names=channel_names,
            input_unit=input_unit,
            qc_thresholds=thresholds,
        )
        flags = _flag_window(metrics, qc_thresholds=thresholds)
        row = {**metrics, **flags}
        if "segment_id" in window_df.columns:
            row["segment_id"] = int(window_df.loc[i, "segment_id"])
        row["window_index"] = i
        qc_rows.append(row)
        valid_mask[i] = not bool(flags["bad_segment_flag"])

    qc_df = pd.DataFrame(qc_rows)

    valid_indices = np.flatnonzero(valid_mask)
    subject_keep = len(valid_indices) >= min_valid_windows

    if subject_keep and max_windows_keep is not None and len(valid_indices) > int(max_windows_keep):
        rng = np.random.default_rng(random_state)
        valid_indices = np.sort(
            rng.choice(valid_indices, size=int(max_windows_keep), replace=False)
        )

    if not subject_keep:
        kept_windows = np.empty((0, arr.shape[1], arr.shape[2]), dtype=np.float32)
        kept_df = window_df.iloc[0:0].copy()
    else:
        kept_windows = arr[valid_indices].astype(np.float32, copy=False)
        kept_df = window_df.iloc[valid_indices].reset_index(drop=True)

    summary = {
        "num_windows_total": int(arr.shape[0]),
        "num_windows_valid": int(len(valid_indices)) if subject_keep else int(valid_mask.sum()),
        "num_windows_removed": int(arr.shape[0] - valid_mask.sum()),
        "fraction_removed": float(1.0 - valid_mask.mean()) if arr.shape[0] > 0 else 0.0,
        "subject_keep": bool(subject_keep),
        "min_valid_windows": int(min_valid_windows),
        "thresholds": thresholds,
    }

    return kept_windows, kept_df, qc_df, summary


def prepare_subject_windows(
    subject: SubjectRecord,
    *,
    apply_bandpass: bool = True,
    bandpass_low_freq: float | None = 0.5,
    bandpass_high_freq: float | None = 45.0,
    bandpass_order: int = 4,
    apply_notch: bool = False,
    notch_freqs: float | Sequence[float] | None = None,
    notch_quality_factor: float = 30.0,
    reference_mode: ReferenceMode = "none",
    ref_channels: Sequence[str | int] | None = None,
    custom_reference_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    signal_norm_mode: ZScoreMode = "none",
    signal_norm_eps: float = 1e-8,
    window_sec: float = 4.0,
    overlap: float = 0.5,
    start_offset_sec: float = 0.0,
    end_offset_sec: float = 0.0,
    keep_partial_window: bool = False,
    apply_qc: bool = True,
    qc_input_unit: str = "auto",
    qc_thresholds: Mapping[str, float] | None = None,
    min_valid_windows: int = 1,
    max_windows_keep: int | None = None,
    random_state: int | None = None,
    macro_duration_sec: float | None = None,
    macro_step_sec: float | None = None,
) -> PreparedSubjectWindows:
    """
    Full subject-level preprocessing helper.

    Pipeline
    --------
    1. start from ``subject.raw_signal``
    2. optional bandpass / notch / referencing
    3. segment into short windows
    4. optional QC-based window dropping
    5. optional subject-level z-score normalization of the kept windows
    6. optional macro grouping

    Notes
    -----
    QC is computed before optional z-score normalization so amplitude-based QC
    stays meaningful.

    Returns
    -------
    PreparedSubjectWindows
        Final prepared subject bundle.
    """
    if getattr(subject, "raw_signal", None) is None:
        raise ValueError(
            f"Subject {getattr(subject, 'subject_id', '<unknown>')!r} has no raw_signal loaded."
        )
    if getattr(subject, "sfreq", None) is None:
        raise ValueError(
            f"Subject {getattr(subject, 'subject_id', '<unknown>')!r} has no sampling rate."
        )

    sfreq = float(subject.sfreq)
    channels = [str(ch) for ch in subject.channels]
    raw = _require_2d_signal(np.asarray(subject.raw_signal), name="subject.raw_signal")

    x = raw.copy()

    if apply_bandpass:
        x = bandpass_filter_signal(
            x,
            sfreq=sfreq,
            low_freq=bandpass_low_freq,
            high_freq=bandpass_high_freq,
            order=bandpass_order,
        )

    if apply_notch:
        x = notch_filter_signal(
            x,
            sfreq=sfreq,
            notch_freqs=notch_freqs,
            quality_factor=notch_quality_factor,
        )

    x = reference_signal(
        x,
        mode=reference_mode,
        channel_names=channels,
        ref_channels=ref_channels,
        custom_reference_fn=custom_reference_fn,
    )

    # Segment the amplitude-preserving signal first so QC remains meaningful.
    windows_for_qc, window_df = segment_signal(
        x,
        sfreq=sfreq,
        window_sec=window_sec,
        overlap=overlap,
        start_offset_sec=start_offset_sec,
        end_offset_sec=end_offset_sec,
        keep_partial=keep_partial_window,
    )

    if apply_qc:
        kept_windows_amp, kept_window_df, qc_df, qc_summary = drop_bad_windows(
            windows_for_qc,
            sfreq=sfreq,
            channel_names=channels,
            window_df=window_df,
            input_unit=qc_input_unit,
            qc_thresholds=qc_thresholds,
            min_valid_windows=min_valid_windows,
            max_windows_keep=max_windows_keep,
            random_state=random_state,
        )
    else:
        kept_windows_amp = windows_for_qc
        kept_window_df = window_df.copy().reset_index(drop=True)
        qc_df = pd.DataFrame(
            {
                "window_index": np.arange(len(window_df), dtype=np.int64),
                "segment_id": kept_window_df.get(
                    "segment_id",
                    pd.Series(np.arange(len(window_df), dtype=np.int64)),
                ),
                "bad_segment_flag": np.zeros(len(window_df), dtype=np.uint8),
                "noise_flag": np.zeros(len(window_df), dtype=np.uint8),
                "artifact_reasons": [""] * len(window_df),
            }
        )
        qc_summary = {
            "num_windows_total": int(len(window_df)),
            "num_windows_valid": int(len(window_df)),
            "num_windows_removed": 0,
            "fraction_removed": 0.0,
            "subject_keep": True,
            "min_valid_windows": int(min_valid_windows),
            "thresholds": dict(DEFAULT_QC_THRESHOLDS if qc_thresholds is None else qc_thresholds),
        }

    # Apply optional z-score normalization after QC.
    if signal_norm_mode != "none" and len(kept_windows_amp) > 0:
        kept_windows, norm_stats = zscore_signal(
            kept_windows_amp,
            mode=signal_norm_mode,
            eps=signal_norm_eps,
        )
    else:
        kept_windows = kept_windows_amp.astype(np.float32, copy=False)
        norm_stats = {
            "mean": np.zeros((1,), dtype=np.float32),
            "std": np.ones((1,), dtype=np.float32),
        }

    if macro_duration_sec is not None and len(kept_window_df) > 0:
        macro_df = build_macro_groups(
            kept_window_df,
            sfreq=sfreq,
            macro_duration_sec=macro_duration_sec,
            macro_step_sec=macro_step_sec,
        )
    else:
        macro_df = pd.DataFrame(
            {
                "segment_id": kept_window_df.get(
                    "segment_id",
                    pd.Series(dtype=np.int64),
                )
            }
        )
        if len(macro_df) > 0:
            macro_df["macro_id"] = 0
            macro_df["macro_start_sample"] = 0
            macro_df["macro_end_sample"] = kept_window_df["end_sample"].max()
            macro_df["macro_start_sec"] = 0.0
            macro_df["macro_end_sec"] = float(macro_df["macro_end_sample"].iloc[0] / sfreq)
            macro_df["window_index_in_macro"] = np.arange(len(macro_df), dtype=np.int64)
            macro_df["num_windows_in_macro"] = len(macro_df)

    metadata = {
        "preprocessing": {
            "apply_bandpass": bool(apply_bandpass),
            "bandpass_low_freq": bandpass_low_freq,
            "bandpass_high_freq": bandpass_high_freq,
            "bandpass_order": int(bandpass_order),
            "apply_notch": bool(apply_notch),
            "notch_freqs": None if notch_freqs is None else (
                float(notch_freqs) if isinstance(notch_freqs, (int, float, np.integer, np.floating))
                else [float(freq) for freq in notch_freqs]
            ),
            "notch_quality_factor": float(notch_quality_factor),
            "reference_mode": reference_mode,
            "ref_channels": None if ref_channels is None else [str(x) for x in ref_channels],
            "signal_norm_mode": signal_norm_mode,
            "window_sec": float(window_sec),
            "overlap": float(overlap),
            "start_offset_sec": float(start_offset_sec),
            "end_offset_sec": float(end_offset_sec),
            "macro_duration_sec": None if macro_duration_sec is None else float(macro_duration_sec),
            "macro_step_sec": None if macro_step_sec is None else float(macro_step_sec),
        },
        "qc_summary": qc_summary,
        "signal_norm_stats": norm_stats,
    }

    return PreparedSubjectWindows(
        subject_id=str(subject.subject_id),
        dataset_name=str(subject.dataset_name),
        label=str(subject.label),
        label_id=int(subject.label_id),
        sfreq=sfreq,
        channels=channels,
        windows=kept_windows,
        window_df=kept_window_df.reset_index(drop=True),
        macro_df=macro_df.reset_index(drop=True),
        qc_df=qc_df.reset_index(drop=True),
        metadata=metadata,
    )


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _as_float32_array(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return arr


def _require_2d_signal(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = _as_float32_array(x, name=name)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [channels, time], got {arr.shape}.")
    return arr


def _require_3d_array(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = _as_float32_array(x, name=name)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have 3 dimensions, got {arr.shape}.")
    return arr


def _validate_positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return value


def _validate_nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}.")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _normalize_axis(axis: int, ndim: int) -> int:
    axis = int(axis)
    if axis < 0:
        axis += ndim
    if not (0 <= axis < ndim):
        raise ValueError(f"axis={axis} is out of bounds for ndim={ndim}.")
    return axis


def _safe_std(std: np.ndarray, *, eps: float) -> np.ndarray:
    out = np.asarray(std, dtype=np.float32).copy()
    out[out < eps] = 1.0
    return out


def _resolve_channel_indices(
    *,
    ref_channels: Sequence[str | int] | None,
    channel_names: Sequence[str] | None,
    n_channels: int,
) -> list[int]:
    if ref_channels is None:
        raise ValueError("ref_channels must be provided for mode='channels'.")

    if channel_names is not None:
        name_to_idx = {str(ch): i for i, ch in enumerate(channel_names)}
    else:
        name_to_idx = {}

    idx: list[int] = []
    for item in ref_channels:
        if isinstance(item, (int, np.integer)):
            item_idx = int(item)
        else:
            key = str(item)
            if key not in name_to_idx:
                raise KeyError(f"Reference channel {key!r} not found in channel_names.")
            item_idx = name_to_idx[key]

        if not (0 <= item_idx < n_channels):
            raise IndexError(
                f"Reference channel index {item_idx} is out of range for n_channels={n_channels}."
            )
        idx.append(item_idx)
    return idx


def _convert_window_to_microvolts(window: np.ndarray, *, input_unit: str) -> np.ndarray:
    x = _require_2d_signal(window, name="window")
    unit = str(input_unit).lower()
    if unit not in {"auto", "uv", "microvolt", "microvolts", "v", "volt", "volts"}:
        raise ValueError(
            f"Unsupported input_unit={input_unit!r}. Use 'auto', 'uv', or 'v'."
        )

    if unit in {"uv", "microvolt", "microvolts"}:
        return x.astype(np.float32, copy=False)

    if unit in {"v", "volt", "volts"}:
        return (x * 1e6).astype(np.float32, copy=False)

    max_abs = float(np.max(np.abs(x))) if x.size > 0 else 0.0
    if max_abs < 1.0:
        return (x * 1e6).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)


def _compute_window_qc(
    window: np.ndarray,
    *,
    sfreq: float,
    channel_names: Sequence[str],
    input_unit: str,
    qc_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """
    Compute lightweight per-window QC metrics.

    The metrics intentionally stay close to your existing QC logic:
    amplitude, peak-to-peak, global std, flatness, and kurtosis-based spike checks.
    """
    x_uv = _convert_window_to_microvolts(window, input_unit=input_unit)

    ch_std = np.std(x_uv, axis=1)
    ch_ptp = np.ptp(x_uv, axis=1)
    ch_absmax = np.max(np.abs(x_uv), axis=1)

    ch_kurt = stats.kurtosis(
        x_uv,
        axis=1,
        fisher=False,
        bias=False,
        nan_policy="omit",
    )
    ch_kurt = np.nan_to_num(ch_kurt, nan=0.0, posinf=0.0, neginf=0.0)

    metrics = {
        "max_abs_uv": float(np.max(ch_absmax)),
        "max_ptp_uv": float(np.max(ch_ptp)),
        "peak_to_peak_uv": float(np.max(ch_ptp)),
        "global_std_uv": float(np.std(x_uv)),
        "flat_channel_frac": float(np.mean(ch_std < qc_thresholds["flat_std_uv"])),
        "high_kurtosis_frac": float(np.mean(ch_kurt > qc_thresholds["kurtosis_thr"])),
        "input_unit": input_unit,
        "num_channels": int(len(channel_names)),
        "num_samples": int(x_uv.shape[1]),
    }
    return metrics


def _flag_window(
    metrics: Mapping[str, Any],
    *,
    qc_thresholds: Mapping[str, float],
) -> dict[str, Any]:
    reasons: list[str] = []

    if float(metrics["max_abs_uv"]) > qc_thresholds["max_abs_uv"]:
        reasons.append(f"max_abs_uv>{qc_thresholds['max_abs_uv']}")
    if float(metrics["max_ptp_uv"]) > qc_thresholds["max_ptp_uv"]:
        reasons.append(f"max_ptp_uv>{qc_thresholds['max_ptp_uv']}")
    if float(metrics["global_std_uv"]) > qc_thresholds["global_std_uv"]:
        reasons.append(f"global_std_uv>{qc_thresholds['global_std_uv']}")
    if float(metrics["flat_channel_frac"]) > qc_thresholds["flat_channel_frac"]:
        reasons.append(f"flat_channel_frac>{qc_thresholds['flat_channel_frac']}")
    if float(metrics["high_kurtosis_frac"]) > qc_thresholds["high_kurtosis_frac"]:
        reasons.append(f"high_kurtosis_frac>{qc_thresholds['high_kurtosis_frac']}")

    bad = len(reasons) > 0
    return {
        "noise_flag": bool(bad),
        "bad_segment_flag": bool(bad),
        "artifact_reasons": ";".join(reasons),
    }

if __name__ == "__main__":
    from dataset import load_dataset
    import data_config as config
    print("AHEAP data")

    aheap_records = load_dataset(
        "aheap",
        root_dir=config.AHEAP_DIR,
        set_glob="**/*eyesclosed*.set",
        participants_path=config.AHEAP_TSV_PATH,
        verbose=True,
    )

    subject = aheap_records[0]

    prepared = prepare_subject_windows(
        subject,
        apply_bandpass=True,
        bandpass_low_freq=0.5,
        bandpass_high_freq=45.0,
        apply_notch=False,
        reference_mode="average",
        window_sec=4.0,
        overlap=0.5,
        apply_qc=True,
        qc_input_unit="auto",
        min_valid_windows=10,
        macro_duration_sec=300.0,   # 5 minutes
    )

    print(prepared.subject_id)
    print(prepared.windows.shape)         # [num_valid_windows, C, T]
    print(prepared.window_df.head())
    num_macros = prepared.macro_df["macro_id"].nunique()
    print("num_macros:", num_macros)
    print(prepared.macro_df.head())
    print(prepared.metadata["qc_summary"])

    print("CAUEEG data")
    records = load_dataset(
        "caueeg",
        root_dir=config.CAUEEG_DIR,
        task="dementia",
        split="train",
        file_format="feather",
        load_signal=True,
        drop_channels=["EKG", "Photic"],
        sampling_rate=200.0,
    )

    subject = records[0]

    prepared = prepare_subject_windows(
        subject,
        apply_bandpass=True,
        bandpass_low_freq=0.5,
        bandpass_high_freq=45.0,
        apply_notch=False,
        reference_mode="average",
        window_sec=4.0,
        overlap=0.5,
        apply_qc=True,
        qc_input_unit="auto",
        min_valid_windows=10,
        macro_duration_sec=300.0,   # 5 minutes
    )

    print(prepared.subject_id)
    print(prepared.windows.shape)         # [num_valid_windows, C, T]
    print(prepared.window_df.head())
    num_macros = prepared.macro_df["macro_id"].nunique()
    print("num_macros:", num_macros)
    print(prepared.macro_df.head())
    print(prepared.metadata["qc_summary"])