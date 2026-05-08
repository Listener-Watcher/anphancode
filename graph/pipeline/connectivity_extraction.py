# connectivity.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import signal, stats


AggregationMode = Literal["mean", "median", "std", "max", "min"]
ThresholdMode = Literal["abs", "positive", "negative"]
NormalizeMode = Literal["none", "minmax", "zscore"]


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


CONNECTIVITY_ALIASES: dict[str, str] = {
    "pearson": "pearson",
    "corr": "pearson",
    "correlation": "pearson",
    "spearman": "spearman",
    "coherence": "coherence",
    "coh": "coherence",
    "pli": "pli",
    "wpli": "wpli",
}


@dataclass(slots=True)
class ConnectivityResult:
    """
    Container for one connectivity output.

    Attributes
    ----------
    name:
        Canonical connectivity metric name.
    values:
        Connectivity matrix/tensor. Usually:
        - [N, N] for single-matrix metrics
        - [B, N, N] for bandwise metrics
    band_names:
        Ordered band names when `values` is bandwise.
    metadata:
        Extra information for debugging and downstream use.
    """

    name: str
    values: np.ndarray
    band_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float32)
        if arr.ndim not in (2, 3):
            raise ValueError(
                f"ConnectivityResult.values must have shape [N, N] or [B, N, N], got {arr.shape}."
            )
        if arr.shape[-1] != arr.shape[-2]:
            raise ValueError(
                f"Connectivity matrices must be square in the last two dims, got {arr.shape}."
            )
        if arr.ndim == 3 and self.band_names is not None and len(self.band_names) != arr.shape[0]:
            raise ValueError(
                f"band_names length ({len(self.band_names)}) does not match num_bands ({arr.shape[0]})."
            )
        self.values = arr.astype(np.float32, copy=False)


def compute_pearson_connectivity(
    eeg_window: np.ndarray,
) -> np.ndarray:
    """
    Compute Pearson correlation connectivity from one EEG window.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].

    Returns
    -------
    np.ndarray
        Connectivity matrix of shape [num_channels, num_channels].
    """
    x = _require_2d_window(eeg_window)
    if x.shape[1] < 2:
        raise ValueError("Pearson connectivity requires at least 2 timepoints.")
    adj = np.corrcoef(x).astype(np.float32)
    return np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)


def compute_spearman_connectivity(
    eeg_window: np.ndarray,
) -> np.ndarray:
    """
    Compute Spearman correlation connectivity from one EEG window.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].

    Returns
    -------
    np.ndarray
        Connectivity matrix of shape [num_channels, num_channels].
    """
    x = _require_2d_window(eeg_window)
    if x.shape[1] < 2:
        raise ValueError("Spearman connectivity requires at least 2 timepoints.")

    # Rank each channel across time, then compute Pearson on ranks.
    ranked = np.stack(
        [stats.rankdata(ch, method="average") for ch in x],
        axis=0,
    ).astype(np.float32)
    adj = np.corrcoef(ranked).astype(np.float32)
    return np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)


def compute_coherence_connectivity(
    eeg_window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
    *,
    nperseg: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute band-wise magnitude-squared coherence connectivity.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    sfreq:
        Sampling rate in Hz.
    bands:
        Frequency band dictionary. Defaults to `DEFAULT_BANDS`.
    nperseg:
        Optional segment length for Welch/coherence.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Bandwise connectivity tensor [B, N, N] and band names.
    """
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    band_dict = dict(DEFAULT_BANDS if bands is None else bands)
    band_names = list(band_dict.keys())

    n_channels = x.shape[0]
    out = np.zeros((len(band_names), n_channels, n_channels), dtype=np.float32)

    nperseg_use = _resolve_nperseg(x.shape[-1], sfreq, nperseg)

    for i in range(n_channels):
        out[:, i, i] = 1.0
        for j in range(i + 1, n_channels):
            freqs, cxy = signal.coherence(
                x[i],
                x[j],
                fs=sfreq,
                nperseg=nperseg_use,
            )
            vals: list[float] = []
            for band_name in band_names:
                low, high = band_dict[band_name]
                mask = (freqs >= low) & (freqs < high)
                val = float(np.mean(cxy[mask])) if np.any(mask) else 0.0
                vals.append(val)

            vals_arr = np.asarray(vals, dtype=np.float32)
            out[:, i, j] = vals_arr
            out[:, j, i] = vals_arr

    return out, band_names


def compute_pli_connectivity(
    eeg_window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
    *,
    filter_order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute band-wise phase-lag index (PLI) connectivity.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    sfreq:
        Sampling rate in Hz.
    bands:
        Frequency band dictionary. Defaults to `DEFAULT_BANDS`.
    filter_order:
        Butterworth bandpass order for band-limited analytic phases.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Bandwise connectivity tensor [B, N, N] and band names.
    """
    return _phase_connectivity(
        eeg_window,
        sfreq,
        bands=bands,
        mode="pli",
        filter_order=filter_order,
    )


def compute_wpli_connectivity(
    eeg_window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
    *,
    filter_order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute band-wise weighted phase-lag index (wPLI) connectivity.

    Notes
    -----
    This uses the same simple phase-difference-based version already present in
    your source builder: it is a practical placeholder aligned with the current
    project, and can be refined later with a cross-spectrum-based formulation.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    sfreq:
        Sampling rate in Hz.
    bands:
        Frequency band dictionary. Defaults to `DEFAULT_BANDS`.
    filter_order:
        Butterworth bandpass order for band-limited analytic phases.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Bandwise connectivity tensor [B, N, N] and band names.
    """
    return _phase_connectivity(
        eeg_window,
        sfreq,
        bands=bands,
        mode="wpli",
        filter_order=filter_order,
    )


def extract_connectivity_for_window(
    eeg_window: np.ndarray,
    sfreq: float,
    *,
    metrics: Sequence[str] | None = None,
    bands: Mapping[str, tuple[float, float]] | None = None,
    coherence_nperseg: int | None = None,
    phase_filter_order: int = 4,
    postprocess: bool = False,
    postprocess_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, ConnectivityResult]:
    """
    Extract one or more connectivity metrics from one EEG window.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    sfreq:
        Sampling rate in Hz.
    metrics:
        Requested metrics. Defaults to all supported metrics:
        ["pearson", "spearman", "coherence", "pli", "wpli"].
    bands:
        Frequency band dictionary for band-wise metrics.
    coherence_nperseg:
        Optional segment length for coherence.
    phase_filter_order:
        Butterworth order for PLI/wPLI bandpass filtering.
    postprocess:
        Whether to apply `postprocess_connectivity_matrix` to each result.
    postprocess_kwargs:
        Keyword arguments passed to `postprocess_connectivity_matrix`.

    Returns
    -------
    dict[str, ConnectivityResult]
        Mapping from metric name to connectivity result.
    """
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    metric_list = _normalize_metric_names(
        metrics or ("pearson", "spearman", "coherence", "pli", "wpli")
    )
    band_dict = dict(DEFAULT_BANDS if bands is None else bands)

    results: dict[str, ConnectivityResult] = {}

    for metric in metric_list:
        if metric == "pearson":
            values = compute_pearson_connectivity(x)
            band_names = None
            meta = {
                "description": "Pearson correlation connectivity.",
                "signed": True,
                "bandwise": False,
            }

        elif metric == "spearman":
            values = compute_spearman_connectivity(x)
            band_names = None
            meta = {
                "description": "Spearman correlation connectivity.",
                "signed": True,
                "bandwise": False,
            }

        elif metric == "coherence":
            values, band_names = compute_coherence_connectivity(
                x,
                sfreq,
                bands=band_dict,
                nperseg=coherence_nperseg,
            )
            meta = {
                "description": "Band-wise coherence connectivity.",
                "signed": False,
                "bandwise": True,
            }

        elif metric == "pli":
            values, band_names = compute_pli_connectivity(
                x,
                sfreq,
                bands=band_dict,
                filter_order=phase_filter_order,
            )
            meta = {
                "description": "Band-wise phase-lag index connectivity.",
                "signed": False,
                "bandwise": True,
            }

        elif metric == "wpli":
            values, band_names = compute_wpli_connectivity(
                x,
                sfreq,
                bands=band_dict,
                filter_order=phase_filter_order,
            )
            meta = {
                "description": "Band-wise weighted phase-lag index connectivity.",
                "signed": False,
                "bandwise": True,
            }

        else:
            raise ValueError(f"Unsupported connectivity metric {metric!r}.")

        if postprocess:
            values = postprocess_connectivity_matrix(
                values,
                **(dict(postprocess_kwargs or {})),
            )

        results[metric] = ConnectivityResult(
            name=metric,
            values=values,
            band_names=band_names,
            metadata=meta,
        )

    return results


def aggregate_connectivity_across_windows(
    connectivity_values: np.ndarray | Sequence[np.ndarray],
    *,
    aggregation: AggregationMode = "mean",
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Aggregate connectivity across windows for macro or subject graphs.

    Supported input shapes
    ----------------------
    - [W, N, N]
    - [W, B, N, N]
    - sequence of [N, N] arrays
    - sequence of [B, N, N] arrays

    Parameters
    ----------
    connectivity_values:
        Connectivity matrices/tensors across windows.
    aggregation:
        Aggregation method across windows.
    weights:
        Optional weights for weighted mean aggregation only.

    Returns
    -------
    np.ndarray
        Aggregated connectivity:
        - [N, N] if input was per-window single-matrix
        - [B, N, N] if input was per-window bandwise
    """
    x = _stack_connectivity_windows(connectivity_values)

    if x.shape[0] == 1:
        return x[0].astype(np.float32, copy=False)

    aggregation = str(aggregation).lower()

    if weights is not None:
        if aggregation != "mean":
            raise ValueError("weights are only supported when aggregation='mean'.")
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(w) != x.shape[0]:
            raise ValueError(
                f"weights length ({len(w)}) must match num_windows ({x.shape[0]})."
            )
        w = w / np.clip(w.sum(), 1e-8, None)
        out = np.tensordot(w, x, axes=(0, 0))
        return np.asarray(out, dtype=np.float32)

    if aggregation == "mean":
        out = np.mean(x, axis=0)
    elif aggregation == "median":
        out = np.median(x, axis=0)
    elif aggregation == "std":
        out = np.std(x, axis=0)
    elif aggregation == "max":
        out = np.max(x, axis=0)
    elif aggregation == "min":
        out = np.min(x, axis=0)
    else:
        raise ValueError(
            f"Unsupported aggregation {aggregation!r}. "
            "Use one of {'mean', 'median', 'std', 'max', 'min'}."
        )

    return np.asarray(out, dtype=np.float32)


def build_connectivity_tensor(
    eeg_data: np.ndarray,
    sfreq: float,
    *,
    metrics: Sequence[str] | None = None,
    bands: Mapping[str, tuple[float, float]] | None = None,
    aggregate_windows: AggregationMode | None = None,
    stack_metrics: bool = True,
    broadcast_nonband_to_bands: bool = True,
    coherence_nperseg: int | None = None,
    phase_filter_order: int = 4,
    postprocess: bool = False,
    postprocess_kwargs: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray | dict[str, ConnectivityResult], dict[str, Any]]:
    """
    Build connectivity outputs from one window or many windows.

    Supported inputs
    ----------------
    - segment level: `eeg_data` shape [N, T]
      -> one metric: [N, N] or [B, N, N]
      -> many metrics with stack_metrics=True:
           [M, N, N] or [M, B, N, N]
    - many windows: `eeg_data` shape [W, N, T]
      -> if aggregate_windows is None:
           returns per-window-per-metric dict
      -> else:
           returns macro/subject connectivity after window aggregation

    Parameters
    ----------
    eeg_data:
        One EEG window or a stack of windows.
    sfreq:
        Sampling rate in Hz.
    metrics:
        Requested connectivity metrics.
    bands:
        Frequency band dictionary for bandwise metrics.
    aggregate_windows:
        Optional aggregation across windows for macro or subject graphs.
    stack_metrics:
        If True, stack multiple metrics into a metric bank tensor.
    broadcast_nonband_to_bands:
        If True and some requested metrics are bandwise while others are not,
        repeat non-bandwise metrics across bands so stacking works.
    coherence_nperseg:
        Optional segment length for coherence.
    phase_filter_order:
        Butterworth order for PLI/wPLI.
    postprocess:
        Whether to apply postprocessing.
    postprocess_kwargs:
        Keyword arguments passed to `postprocess_connectivity_matrix`.

    Returns
    -------
    tuple
        `(connectivity_output, metadata)`
    """
    arr = np.asarray(eeg_data, dtype=np.float32)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    metric_list = _normalize_metric_names(
        metrics or ("pearson", "spearman", "coherence", "pli", "wpli")
    )

    if arr.ndim == 2:
        result_dict = extract_connectivity_for_window(
            arr,
            sfreq,
            metrics=metric_list,
            bands=bands,
            coherence_nperseg=coherence_nperseg,
            phase_filter_order=phase_filter_order,
            postprocess=postprocess,
            postprocess_kwargs=postprocess_kwargs,
        )

        if len(metric_list) == 1:
            result = result_dict[metric_list[0]]
            meta = {
                "graph_level": "segment",
                "metric_names": [result.name],
                "band_names": result.band_names,
                "shape": tuple(result.values.shape),
            }
            return result.values, meta

        if not stack_metrics:
            meta = {
                "graph_level": "segment",
                "metric_names": list(result_dict.keys()),
                "stacked": False,
            }
            return result_dict, meta

        stacked, metric_names, band_names = _stack_metric_bank(
            result_dict,
            broadcast_nonband_to_bands=broadcast_nonband_to_bands,
        )
        meta = {
            "graph_level": "segment",
            "metric_names": metric_names,
            "band_names": band_names,
            "stacked": True,
            "shape": tuple(stacked.shape),
        }
        return stacked, meta

    if arr.ndim != 3:
        raise ValueError(
            f"eeg_data must have shape [N, T] or [W, N, T], got {arr.shape}."
        )

    per_metric_windows: dict[str, list[np.ndarray]] = {metric: [] for metric in metric_list}
    metric_band_names: dict[str, list[str] | None] = {metric: None for metric in metric_list}

    for w_idx in range(arr.shape[0]):
        result_dict = extract_connectivity_for_window(
            arr[w_idx],
            sfreq,
            metrics=metric_list,
            bands=bands,
            coherence_nperseg=coherence_nperseg,
            phase_filter_order=phase_filter_order,
            postprocess=postprocess,
            postprocess_kwargs=postprocess_kwargs,
        )
        for metric in metric_list:
            per_metric_windows[metric].append(result_dict[metric].values)
            if metric_band_names[metric] is None:
                metric_band_names[metric] = result_dict[metric].band_names

    if aggregate_windows is None:
        # Return per-window results as a dict so downstream code can decide how to use them.
        out_dict: dict[str, ConnectivityResult] = {}
        for metric in metric_list:
            stacked_metric = _stack_connectivity_windows(per_metric_windows[metric])
            out_dict[metric] = ConnectivityResult(
                name=metric,
                values=stacked_metric,
                band_names=metric_band_names[metric],
                metadata={"graph_level": "window_stack"},
            )
        meta = {
            "graph_level": "window_stack",
            "metric_names": metric_list,
            "num_windows": int(arr.shape[0]),
            "stacked": False,
        }
        return out_dict, meta

    aggregated_dict: dict[str, ConnectivityResult] = {}
    for metric in metric_list:
        agg_values = aggregate_connectivity_across_windows(
            per_metric_windows[metric],
            aggregation=aggregate_windows,
        )
        aggregated_dict[metric] = ConnectivityResult(
            name=metric,
            values=agg_values,
            band_names=metric_band_names[metric],
            metadata={
                "graph_level": "aggregated",
                "aggregation": str(aggregate_windows),
                "num_windows": int(arr.shape[0]),
            },
        )

    if len(metric_list) == 1:
        result = aggregated_dict[metric_list[0]]
        meta = {
            "graph_level": "aggregated",
            "aggregation": str(aggregate_windows),
            "metric_names": [result.name],
            "band_names": result.band_names,
            "num_windows": int(arr.shape[0]),
            "shape": tuple(result.values.shape),
        }
        return result.values, meta

    if not stack_metrics:
        meta = {
            "graph_level": "aggregated",
            "aggregation": str(aggregate_windows),
            "metric_names": metric_list,
            "num_windows": int(arr.shape[0]),
            "stacked": False,
        }
        return aggregated_dict, meta

    stacked, metric_names, band_names = _stack_metric_bank(
        aggregated_dict,
        broadcast_nonband_to_bands=broadcast_nonband_to_bands,
    )
    meta = {
        "graph_level": "aggregated",
        "aggregation": str(aggregate_windows),
        "metric_names": metric_names,
        "band_names": band_names,
        "num_windows": int(arr.shape[0]),
        "stacked": True,
        "shape": tuple(stacked.shape),
    }
    return stacked, meta


def postprocess_connectivity_matrix(
    connectivity: np.ndarray,
    *,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    nan_to_num: bool = True,
    threshold: float | None = None,
    threshold_mode: ThresholdMode = "abs",
    normalize: NormalizeMode = "none",
    clip_min: float | None = None,
    clip_max: float | None = None,
    flatten_upper_triangle: bool = False,
    copy: bool = True,
) -> np.ndarray:
    """
    Postprocess connectivity matrices/tensors.

    Supported input shapes
    ----------------------
    - [N, N]
    - [B, N, N]
    - [W, N, N]
    - [W, B, N, N]

    Parameters
    ----------
    connectivity:
        Connectivity array with square last two dims.
    symmetrize:
        If True, replace A with 0.5 * (A + A^T).
    zero_diagonal:
        If True, zero the diagonal of each matrix.
    nan_to_num:
        If True, replace NaN/Inf with finite values.
    threshold:
        Optional threshold applied after symmetrization/diagonal cleanup.
    threshold_mode:
        - "abs": keep entries with |A| >= threshold
        - "positive": keep entries with A >= threshold
        - "negative": keep entries with A <= -threshold
    normalize:
        - "none"
        - "minmax"
        - "zscore"
      Applied independently to each matrix over all entries.
    clip_min, clip_max:
        Optional clipping after normalization.
    flatten_upper_triangle:
        If True, return upper-triangle vectors with shape [..., E].
    copy:
        If True, operate on a copy.

    Returns
    -------
    np.ndarray
        Postprocessed matrix/tensor or flattened upper-triangle features.
    """
    x = np.asarray(connectivity, dtype=np.float32)
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError(
            f"Connectivity input must end with square dims [N, N], got {x.shape}."
        )

    y = x.copy() if copy else x

    if nan_to_num:
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if symmetrize:
        y = 0.5 * (y + np.swapaxes(y, -1, -2))

    if zero_diagonal:
        _zero_last_two_diagonals_inplace(y)

    if threshold is not None:
        thr = float(threshold)
        if thr < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}.")
        if threshold_mode == "abs":
            mask = np.abs(y) >= thr
        elif threshold_mode == "positive":
            mask = y >= thr
        elif threshold_mode == "negative":
            mask = y <= -thr
        else:
            raise ValueError(f"Unsupported threshold_mode {threshold_mode!r}.")
        y = np.where(mask, y, 0.0).astype(np.float32, copy=False)

    normalize = str(normalize).lower()
    if normalize != "none":
        y = _normalize_per_matrix(y, mode=normalize)

    if clip_min is not None or clip_max is not None:
        y = np.clip(
            y,
            a_min=-np.inf if clip_min is None else float(clip_min),
            a_max=np.inf if clip_max is None else float(clip_max),
        ).astype(np.float32, copy=False)

    if flatten_upper_triangle:
        y = _flatten_upper_triangle(y)

    return y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _require_2d_window(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"EEG window must have shape [num_channels, num_timepoints], got {arr.shape}.")
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError(f"EEG window is too small, got {arr.shape}.")
    return arr


def _validate_positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return value


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def _normalize_metric_names(metrics: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for metric in metrics:
        key = str(metric).strip().lower()
        if key not in CONNECTIVITY_ALIASES:
            raise ValueError(
                f"Unsupported connectivity metric {metric!r}. "
                f"Available metric names/aliases: {sorted(CONNECTIVITY_ALIASES.keys())}"
            )
        canonical = CONNECTIVITY_ALIASES[key]
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)

    return out


def _resolve_nperseg(n_samples: int, sfreq: float, nperseg: int | None) -> int:
    if nperseg is None:
        return min(n_samples, max(256, int(round(2.0 * sfreq))))
    return min(n_samples, _validate_positive_int(nperseg, "nperseg"))


def _bandpass_filter_1d(
    x: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    *,
    order: int = 4,
) -> np.ndarray:
    low, high = float(band[0]), float(band[1])
    nyq = sfreq / 2.0

    if low <= 0 or high <= 0 or low >= high:
        raise ValueError(f"Invalid band {band}.")
    if high >= nyq:
        high = nyq - 1e-3
    if low >= high:
        raise ValueError(f"Band {band} is invalid for sfreq={sfreq}.")

    sos = signal.butter(
        N=order,
        Wn=[low, high],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    try:
        y = signal.sosfiltfilt(sos, x)
    except ValueError:
        # Fallback for unusually short inputs.
        y = signal.sosfilt(sos, x)
    return np.asarray(y, dtype=np.float32)


def _analytic_phase(
    eeg_window: np.ndarray,
    sfreq: float,
    band: tuple[float, float],
    *,
    filter_order: int = 4,
) -> np.ndarray:
    x = _require_2d_window(eeg_window)
    filtered = np.stack(
        [
            _bandpass_filter_1d(ch, sfreq, band, order=filter_order)
            for ch in x
        ],
        axis=0,
    ).astype(np.float32)
    analytic = signal.hilbert(filtered, axis=-1)
    return np.angle(analytic).astype(np.float32)


def _phase_connectivity(
    eeg_window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None,
    *,
    mode: Literal["pli", "wpli"],
    filter_order: int = 4,
) -> tuple[np.ndarray, list[str]]:
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    band_dict = dict(DEFAULT_BANDS if bands is None else bands)
    band_names = list(band_dict.keys())

    n_channels = x.shape[0]
    out = np.zeros((len(band_names), n_channels, n_channels), dtype=np.float32)

    for b_idx, band_name in enumerate(band_names):
        phases = _analytic_phase(
            x,
            sfreq,
            band_dict[band_name],
            filter_order=filter_order,
        )
        for i in range(n_channels):
            out[b_idx, i, i] = 1.0
            for j in range(i + 1, n_channels):
                dphi = phases[i] - phases[j]

                if mode == "pli":
                    val = float(np.abs(np.mean(np.sign(np.sin(dphi)))))
                elif mode == "wpli":
                    # Matches the current simple project version in the source builder.
                    im = np.sin(dphi)
                    denom = float(np.mean(np.abs(im)))
                    val = float(np.abs(np.mean(im)) / max(denom, 1e-8))
                else:
                    raise ValueError(f"Unsupported phase connectivity mode {mode!r}.")

                out[b_idx, i, j] = val
                out[b_idx, j, i] = val

    return out, band_names


def _stack_connectivity_windows(
    connectivity_values: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    if isinstance(connectivity_values, np.ndarray):
        arr = np.asarray(connectivity_values, dtype=np.float32)
        if arr.ndim in (3, 4):
            return arr.astype(np.float32, copy=False)
        if arr.ndim == 2:
            return arr[None, ...].astype(np.float32, copy=False)
        raise ValueError(
            f"connectivity_values must have shape [W,N,N], [W,B,N,N], [N,N], or [B,N,N], got {arr.shape}."
        )

    mats = [np.asarray(m, dtype=np.float32) for m in connectivity_values]
    if len(mats) == 0:
        raise ValueError("connectivity_values must not be empty.")

    ref_shape = mats[0].shape
    if len(ref_shape) not in (2, 3):
        raise ValueError(f"Each connectivity item must have shape [N,N] or [B,N,N], got {ref_shape}.")

    for idx, mat in enumerate(mats[1:], start=1):
        if mat.shape != ref_shape:
            raise ValueError(
                f"Connectivity item at index {idx} has shape {mat.shape}, expected {ref_shape}."
            )

    return np.stack(mats, axis=0).astype(np.float32, copy=False)


def _stack_metric_bank(
    result_dict: Mapping[str, ConnectivityResult],
    *,
    broadcast_nonband_to_bands: bool,
) -> tuple[np.ndarray, list[str], list[str] | None]:
    metric_names = list(result_dict.keys())
    arrays = [np.asarray(result_dict[name].values, dtype=np.float32) for name in metric_names]

    ndim_set = {arr.ndim for arr in arrays}
    if ndim_set == {2}:
        stacked = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        return stacked, metric_names, None

    if ndim_set == {3}:
        ref_bands = arrays[0].shape[0]
        band_names = result_dict[metric_names[0]].band_names
        for idx, arr in enumerate(arrays[1:], start=1):
            if arr.shape != arrays[0].shape:
                raise ValueError(
                    f"Bandwise metric {metric_names[idx]!r} has shape {arr.shape}, "
                    f"expected {arrays[0].shape} for stacking."
                )
        stacked = np.stack(arrays, axis=0).astype(np.float32, copy=False)
        return stacked, metric_names, band_names

    # Mixed 2D and 3D metrics.
    if not broadcast_nonband_to_bands:
        raise ValueError(
            "Cannot stack mixed bandwise and non-bandwise metrics unless "
            "broadcast_nonband_to_bands=True."
        )

    band_counts = [arr.shape[0] for arr in arrays if arr.ndim == 3]
    ref_bands = band_counts[0]
    if any(b != ref_bands for b in band_counts):
        raise ValueError(
            f"Bandwise metrics have inconsistent num_bands: {band_counts}."
        )

    band_names: list[str] | None = None
    for name in metric_names:
        bn = result_dict[name].band_names
        if bn is not None:
            band_names = list(bn)
            break

    converted: list[np.ndarray] = []
    for arr in arrays:
        if arr.ndim == 2:
            converted.append(np.repeat(arr[None, ...], ref_bands, axis=0))
        else:
            converted.append(arr)
    stacked = np.stack(converted, axis=0).astype(np.float32, copy=False)
    return stacked, metric_names, band_names


def _zero_last_two_diagonals_inplace(x: np.ndarray) -> None:
    n = x.shape[-1]
    idx = np.arange(n)
    x[..., idx, idx] = 0.0


def _normalize_per_matrix(x: np.ndarray, *, mode: NormalizeMode) -> np.ndarray:
    mode = str(mode).lower()
    y = x.astype(np.float32, copy=False)
    leading_shape = y.shape[:-2]
    flat = y.reshape((-1, y.shape[-2], y.shape[-1]))

    out = np.empty_like(flat, dtype=np.float32)
    for i in range(flat.shape[0]):
        mat = flat[i]
        if mode == "minmax":
            mn = float(np.min(mat))
            mx = float(np.max(mat))
            if mx - mn < 1e-8:
                out[i] = np.zeros_like(mat, dtype=np.float32)
            else:
                out[i] = ((mat - mn) / (mx - mn)).astype(np.float32)
        elif mode == "zscore":
            mu = float(np.mean(mat))
            sd = float(np.std(mat))
            if sd < 1e-8:
                out[i] = np.zeros_like(mat, dtype=np.float32)
            else:
                out[i] = ((mat - mu) / sd).astype(np.float32)
        else:
            raise ValueError(f"Unsupported normalize mode {mode!r}.")
    return out.reshape(y.shape).astype(np.float32, copy=False)


def _flatten_upper_triangle(x: np.ndarray) -> np.ndarray:
    n = x.shape[-1]
    iu = np.triu_indices(n, k=1)
    flat = x[..., iu[0], iu[1]]
    return np.asarray(flat, dtype=np.float32)

if __name__ == "__main__":
    import numpy as np

    import data_config as config
    from dataset import load_dataset
    from preprocessing import prepare_subject_windows
    # ================================================================
    # AHEAP EXAMPLE
    # ================================================================
    aheap_records = load_dataset(
        "aheap",
        root_dir=config.AHEAP_DIR,
        set_glob="**/*.set",
        participants_path=config.AHEAP_TSV_PATH,
        load_signal=True,
        verbose=False,
    )

    if len(aheap_records) > 0:
        aheap_subject = aheap_records[0]

        aheap_prepared = prepare_subject_windows(
            aheap_subject,
            apply_bandpass=True,
            bandpass_low_freq=0.5,
            bandpass_high_freq=45.0,
            apply_notch=False,
            reference_mode="average",
            signal_norm_mode="none",
            window_sec=4.0,
            overlap=0.5,
            apply_qc=True,
            qc_input_unit="auto",
            min_valid_windows=5,
            macro_duration_sec=300.0,   # 5 minutes
        )
        window = aheap_prepared.windows[0]
        segment_conn, segment_meta = build_connectivity_tensor(
            window,
            sfreq=500.0,
            metrics=["coherence"],
            bands={
                "delta": (1.0, 4.0),
                "theta": (4.0, 8.0),
                "alpha": (8.0, 13.0),
                "beta": (13.0, 30.0),
                "gamma": (30.0, 45.0),
            },
            postprocess=True,
            postprocess_kwargs={
                "symmetrize": True,
                "zero_diagonal": True,
            },
        )

        print("segment coherence shape:", segment_conn.shape)   # [B, 19, 19]
        print(segment_meta)


    else:
        print("No AHEAP records were loaded.")

    # ================================================================
    # CAUEEG EXAMPLE
    # ================================================================
    caueeg_records = load_dataset(
        "caueeg",
        root_dir=config.CAUEEG_DIR,
        task="dementia",
        split="train",
        file_format="feather",
        load_signal=True,
        verbose=False,
        drop_channels=["EKG", "Photic"],
        sampling_rate=200.0,
    )

    if len(caueeg_records) > 0:
        caueeg_subject = caueeg_records[0]

        prepared = prepare_subject_windows(
            caueeg_subject,
            apply_bandpass=True,
            bandpass_low_freq=0.5,
            bandpass_high_freq=45.0,
            apply_notch=False,
            reference_mode="average",
            signal_norm_mode="none",
            window_sec=10.0,            # CAUEEG baseline-style crop length
            overlap=0.5,
            apply_qc=True,
            qc_input_unit="auto",
            min_valid_windows=5,
            macro_duration_sec=300.0,   # 5 minutes
        )
        if "macro_id" in prepared.macro_df.columns and len(prepared.macro_df) > 0:
            macro_ids = sorted(prepared.macro_df["macro_id"].unique().tolist())
            first_macro_id = macro_ids[0]

            macro_segment_ids = (
                prepared.macro_df.loc[prepared.macro_df["macro_id"] == first_macro_id, "segment_id"]
                .astype(int)
                .tolist()
            )

            macro_window_df = prepared.window_df.copy()
            macro_window_df["segment_id"] = macro_window_df["segment_id"].astype(int)

            macro_indices = macro_window_df.index[
                macro_window_df["segment_id"].isin(macro_segment_ids)
            ].to_numpy()

            macro_windows = prepared.windows[macro_indices]  # [W_macro, N, T]
            # macro_windows = np.stack(
            #     [
            #         np.random.randn(19, 2000).astype(np.float32)
            #         for _ in range(8)
            #     ],
            #     axis=0,
            # )  # [W, N, T]

            macro_conn, macro_meta = build_connectivity_tensor(
                macro_windows,
                sfreq=200.0,
                metrics=["pearson", "coherence"],
                aggregate_windows="mean",
                stack_metrics=True,
                broadcast_nonband_to_bands=True,
                postprocess=True,
                postprocess_kwargs={
                    "symmetrize": True,
                    "zero_diagonal": True,
                },
            )

            print("macro connectivity bank shape:", macro_conn.shape)  # usually [M, B, 19, 19]
            print(macro_meta)


    else:
        print("No CAUEEG records were loaded.")
    # --------------------------------------------------
    # Example 1: one EEG window -> segment-level connectivity
    # --------------------------------------------------
    # sfreq = 500.0
    # window = np.random.randn(19, 2000).astype(np.float32)

    # segment_conn, segment_meta = build_connectivity_tensor(
    #     window,
    #     sfreq,
    #     metrics=["coherence"],
    #     bands={
    #         "delta": (1.0, 4.0),
    #         "theta": (4.0, 8.0),
    #         "alpha": (8.0, 13.0),
    #         "beta": (13.0, 30.0),
    #         "gamma": (30.0, 45.0),
    #     },
    #     postprocess=True,
    #     postprocess_kwargs={
    #         "symmetrize": True,
    #         "zero_diagonal": True,
    #     },
    # )

    # print("segment coherence shape:", segment_conn.shape)   # [B, 19, 19]
    # print(segment_meta)

    # # --------------------------------------------------
    # # Example 2: many short windows in one macro block
    # # -> macro-level aggregated connectivity
    # # --------------------------------------------------
    # macro_windows = np.stack(
    #     [
    #         np.random.randn(19, 2000).astype(np.float32)
    #         for _ in range(8)
    #     ],
    #     axis=0,
    # )  # [W, N, T]

    # macro_conn, macro_meta = build_connectivity_tensor(
    #     macro_windows,
    #     sfreq,
    #     metrics=["pearson", "coherence"],
    #     aggregate_windows="mean",
    #     stack_metrics=True,
    #     broadcast_nonband_to_bands=True,
    #     postprocess=True,
    #     postprocess_kwargs={
    #         "symmetrize": True,
    #         "zero_diagonal": True,
    #     },
    # )

    # print("macro connectivity bank shape:", macro_conn.shape)  # usually [M, B, 19, 19]
    # print(macro_meta)