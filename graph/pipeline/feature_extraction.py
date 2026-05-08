# features.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from scipy import signal, stats
import numpy as np
import pandas as pd

from dataset import load_dataset
from preprocessing import prepare_subject_windows

AggregationMode = Literal["mean", "median", "std", "max", "min"]
BandpowerMode = Literal["relative", "absolute", "log"]


DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


FEATURE_GROUP_ALIASES: dict[str, str] = {
    "time_domain": "time_domain",
    "statistical": "time_domain",
    "hjorth": "hjorth",
    "bandpower": "relative_band_power",
    "relative_band_power": "relative_band_power",
    "log_band_power": "log_band_power",
    "entropy": "entropy",
    "spectral_entropy": "entropy",
    "energies": "energies",
    "wavelet_energy": "energies",
}


@dataclass(slots=True)
class FeatureGroupResult:
    """
    Container for one feature family.

    Attributes
    ----------
    name:
        Canonical feature family name.
    values:
        Feature matrix of shape [num_nodes, num_features].
    feature_names:
        Ordered feature names aligned with the last dimension of ``values``.
    metadata:
        Additional metadata for debugging and downstream use.
    """

    name: str
    values: np.ndarray
    feature_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.values, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"FeatureGroupResult.values must have shape [num_nodes, num_features], got {arr.shape}."
            )
        if arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"feature_names length ({len(self.feature_names)}) does not match "
                f"num_features ({arr.shape[1]})."
            )
        self.values = arr.astype(np.float32, copy=False)


def compute_time_domain_features(
    eeg_window: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute channel-wise time-domain/statistical features from one EEG window.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix [num_channels, num_features] and feature names.
    """
    x = _require_2d_window(eeg_window)

    mean = np.mean(x, axis=1)
    std = np.std(x, axis=1)
    skew = stats.skew(x, axis=1, bias=False, nan_policy="omit")
    kurt = stats.kurtosis(x, axis=1, fisher=False, bias=False, nan_policy="omit")
    minimum = np.min(x, axis=1)
    maximum = np.max(x, axis=1)
    ptp = np.ptp(x, axis=1)
    rms = np.sqrt(np.mean(np.square(x), axis=1))
    line_length = np.sum(np.abs(np.diff(x, axis=1)), axis=1) / max(x.shape[1] - 1, 1)
    zero_cross_rate = np.mean(np.diff(np.signbit(x), axis=1), axis=1)

    feats = np.stack(
        [
            mean,
            std,
            np.nan_to_num(skew, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(kurt, nan=0.0, posinf=0.0, neginf=0.0),
            minimum,
            maximum,
            ptp,
            rms,
            line_length,
            zero_cross_rate.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    names = [
        "td_mean",
        "td_std",
        "td_skew",
        "td_kurtosis",
        "td_min",
        "td_max",
        "td_ptp",
        "td_rms",
        "td_line_length",
        "td_zero_cross_rate",
    ]
    return feats, names


def compute_hjorth_features(
    eeg_window: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute channel-wise Hjorth parameters from one EEG window.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix [num_channels, 3] and feature names.
    """
    x = _require_2d_window(eeg_window)
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)

    var0 = np.var(x, axis=1)
    var1 = np.var(dx, axis=1)
    var2 = np.var(ddx, axis=1)

    eps = 1e-8
    activity = var0
    mobility = np.sqrt(var1 / np.maximum(var0, eps))
    complexity = np.sqrt(var2 / np.maximum(var1, eps)) / np.maximum(mobility, eps)

    feats = np.stack([activity, mobility, complexity], axis=1).astype(np.float32, copy=False)
    names = ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
    return feats, names


def compute_bandpower_features(
    eeg_window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
    *,
    mode: BandpowerMode = "relative",
    nperseg: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute channel-wise bandpower features from one EEG window.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    sfreq:
        Sampling rate in Hz.
    bands:
        Frequency band dictionary. Defaults to ``DEFAULT_BANDS``.
    mode:
        - "relative": relative band power
        - "absolute": absolute band power
        - "log": log(1 + absolute power)
    nperseg:
        Optional Welch nperseg.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix [num_channels, num_bands] and feature names.
    """
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    band_dict = dict(DEFAULT_BANDS if bands is None else bands)

    freqs, psd = _welch_psd(x, sfreq=sfreq, nperseg=nperseg)

    band_names = list(band_dict.keys())
    abs_power = np.zeros((x.shape[0], len(band_names)), dtype=np.float32)

    for b_idx, band_name in enumerate(band_names):
        low, high = band_dict[band_name]
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            abs_power[:, b_idx] = np.trapezoid(psd[:, mask], freqs[mask], axis=-1).astype(np.float32)

    if mode == "absolute":
        feats = abs_power
        names = [f"absp_{band_name}" for band_name in band_names]
    elif mode == "log":
        feats = np.log1p(np.maximum(abs_power, 0.0)).astype(np.float32)
        names = [f"logbp_{band_name}" for band_name in band_names]
    elif mode == "relative":
        total_power = abs_power.sum(axis=1, keepdims=True)
        feats = abs_power / np.clip(total_power, 1e-8, None)
        feats = feats.astype(np.float32, copy=False)
        names = [f"rbp_{band_name}" for band_name in band_names]
    else:
        raise ValueError(f"Unsupported bandpower mode {mode!r}.")

    return feats.astype(np.float32, copy=False), names


def compute_entropy_features(
    eeg_window: np.ndarray,
    sfreq: float,
    bands: Mapping[str, tuple[float, float]] | None = None,
    *,
    amplitude_bins: int = 32,
    nperseg: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute channel-wise entropy-style features from one EEG window.

    Implemented features
    --------------------
    - normalized spectral entropy from Welch PSD
    - amplitude histogram entropy
    - Gaussian differential entropy approximation

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    sfreq:
        Sampling rate in Hz.
    bands:
        Optional band dictionary. When provided, spectral entropy is computed
        over the full span covered by these bands.
    amplitude_bins:
        Number of bins for amplitude entropy.
    nperseg:
        Optional Welch nperseg.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix [num_channels, 3] and feature names.
    """
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    amplitude_bins = _validate_positive_int(amplitude_bins, "amplitude_bins")

    freqs, psd = _welch_psd(x, sfreq=sfreq, nperseg=nperseg)

    if bands is not None and len(bands) > 0:
        low_all = min(float(lo) for lo, _ in bands.values())
        high_all = max(float(hi) for _, hi in bands.values())
        mask = (freqs >= low_all) & (freqs < high_all)
        psd_use = psd[:, mask] if np.any(mask) else psd
    else:
        psd_use = psd

    psd_prob = psd_use / np.clip(psd_use.sum(axis=1, keepdims=True), 1e-8, None)
    spectral_entropy = stats.entropy(psd_prob, axis=1)
    if psd_prob.shape[1] > 1:
        spectral_entropy = spectral_entropy / np.log(psd_prob.shape[1])

    amp_entropy = np.zeros((x.shape[0],), dtype=np.float32)
    for ch_idx in range(x.shape[0]):
        counts, _ = np.histogram(x[ch_idx], bins=amplitude_bins)
        probs = counts.astype(np.float64) / max(counts.sum(), 1)
        probs = probs[probs > 0]
        amp_entropy[ch_idx] = float(
            -np.sum(probs * np.log(probs)) / max(np.log(amplitude_bins), 1e-8)
        )

    var = np.var(x, axis=1)
    diff_entropy = 0.5 * np.log(2.0 * np.pi * np.e * np.maximum(var, 1e-8))

    feats = np.stack(
        [
            np.nan_to_num(spectral_entropy, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(amp_entropy, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(diff_entropy, nan=0.0, posinf=0.0, neginf=0.0),
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    names = ["entropy_spectral", "entropy_amplitude", "entropy_differential"]
    return feats, names


def compute_energy_features(
    eeg_window: np.ndarray,
    *,
    wavelet: str = "db4",
    level: int = 5,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute channel-wise wavelet sub-band energies from one EEG window.

    Notes
    -----
    This follows the same basic idea as your previous source:
    compute DWT coefficients per channel, then store the energy of each
    coefficient set. The group name is intentionally shortened to ``energies``.

    Parameters
    ----------
    eeg_window:
        EEG array of shape [num_channels, num_timepoints].
    wavelet:
        PyWavelets wavelet name.
    level:
        DWT decomposition level.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Feature matrix [num_channels, level + 1] and feature names.

    Raises
    ------
    ImportError
        If PyWavelets is not installed.
    """
    try:
        import pywt  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Energy features require PyWavelets (pywt). "
            "Install it or remove 'energies' from the feature groups."
        ) from exc

    x = _require_2d_window(eeg_window)
    level = _validate_positive_int(level, "level")

    values: list[list[float]] = []
    for ch in x:
        coeffs = pywt.wavedec(ch, wavelet=wavelet, level=level)
        energies = [float(np.sum(np.square(c))) for c in coeffs]
        values.append(energies)

    feats = np.asarray(values, dtype=np.float32)

    # wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_1]
    names = [f"energies_a{level}"] + [f"energies_d{i}" for i in range(level, 0, -1)]
    return feats, names


def extract_node_features_for_window(
    eeg_window: np.ndarray,
    sfreq: float,
    *,
    feature_groups: Sequence[str] | None = None,
    bands: Mapping[str, tuple[float, float]] | None = None,
    bandpower_nperseg: int | None = None,
    amplitude_entropy_bins: int = 32,
    energy_wavelet: str = "db4",
    energy_level: int = 5,
) -> dict[str, FeatureGroupResult]:
    """
    Extract channel-wise feature groups from one EEG window.

    Supported canonical group names
    -------------------------------
    - "time_domain"
    - "hjorth"
    - "relative_band_power"
    - "log_band_power"
    - "entropy"
    - "energies"

    Returns
    -------
    dict[str, FeatureGroupResult]
        Mapping from canonical group name to extracted feature group result.
    """
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")
    band_dict = dict(DEFAULT_BANDS if bands is None else bands)

    requested = _normalize_requested_groups(
        feature_groups or ("time_domain", "hjorth", "relative_band_power", "entropy", "energies")
    )

    out: dict[str, FeatureGroupResult] = {}

    if "time_domain" in requested:
        values, names = compute_time_domain_features(x)
        out["time_domain"] = FeatureGroupResult(
            name="time_domain",
            values=values,
            feature_names=names,
            metadata={"description": "Channel-wise time-domain/statistical features."},
        )

    if "hjorth" in requested:
        values, names = compute_hjorth_features(x)
        out["hjorth"] = FeatureGroupResult(
            name="hjorth",
            values=values,
            feature_names=names,
            metadata={"description": "Channel-wise Hjorth parameters."},
        )

    if "relative_band_power" in requested:
        values, names = compute_bandpower_features(
            x,
            sfreq,
            band_dict,
            mode="relative",
            nperseg=bandpower_nperseg,
        )
        out["relative_band_power"] = FeatureGroupResult(
            name="relative_band_power",
            values=values,
            feature_names=names,
            metadata={"description": "Channel-wise relative band power.", "bands": dict(band_dict)},
        )

    if "log_band_power" in requested:
        values, names = compute_bandpower_features(
            x,
            sfreq,
            band_dict,
            mode="log",
            nperseg=bandpower_nperseg,
        )
        out["log_band_power"] = FeatureGroupResult(
            name="log_band_power",
            values=values,
            feature_names=names,
            metadata={"description": "Channel-wise log band power.", "bands": dict(band_dict)},
        )

    if "entropy" in requested:
        values, names = compute_entropy_features(
            x,
            sfreq,
            bands=band_dict,
            amplitude_bins=amplitude_entropy_bins,
            nperseg=bandpower_nperseg,
        )
        out["entropy"] = FeatureGroupResult(
            name="entropy",
            values=values,
            feature_names=names,
            metadata={"description": "Channel-wise entropy features."},
        )

    if "energies" in requested:
        values, names = compute_energy_features(
            x,
            wavelet=energy_wavelet,
            level=energy_level,
        )
        out["energies"] = FeatureGroupResult(
            name="energies",
            values=values,
            feature_names=names,
            metadata={
                "description": "Channel-wise wavelet sub-band energies.",
                "wavelet": energy_wavelet,
                "level": int(energy_level),
            },
        )

    return out


def aggregate_features_across_windows(
    feature_matrices: np.ndarray | Sequence[np.ndarray],
    *,
    aggregation: AggregationMode = "mean",
    weights: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """
    Aggregate feature matrices across windows for macro or subject graphs.

    Parameters
    ----------
    feature_matrices:
        Either:
        - array of shape [num_windows, num_nodes, num_features], or
        - sequence of [num_nodes, num_features] arrays.
    aggregation:
        Aggregation method across windows.
    weights:
        Optional weights for weighted mean aggregation only.

    Returns
    -------
    np.ndarray
        Aggregated feature matrix with shape [num_nodes, num_features].
    """
    x = _stack_feature_matrices(feature_matrices)

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
        return out.astype(np.float32, copy=False)

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


def build_feature_matrix(
    eeg_data: np.ndarray,
    sfreq: float,
    *,
    feature_groups: Sequence[str] | None = None,
    bands: Mapping[str, tuple[float, float]] | None = None,
    aggregate_windows: AggregationMode | None = None,
    bandpower_nperseg: int | None = None,
    amplitude_entropy_bins: int = 32,
    energy_wavelet: str = "db4",
    energy_level: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Build a final node feature matrix from one window or many windows.

    Supported inputs
    ----------------
    - segment level: ``eeg_data`` shape [num_nodes, num_timepoints]
      -> returns [num_nodes, num_features]
    - many windows: ``eeg_data`` shape [num_windows, num_nodes, num_timepoints]
      -> if ``aggregate_windows is None`` returns [num_windows, num_nodes, num_features]
      -> else returns [num_nodes, num_features]

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Final feature matrix and metadata.
    """
    arr = np.asarray(eeg_data, dtype=np.float32)
    sfreq = _validate_positive_float(sfreq, "sfreq")

    selected_groups = _normalize_requested_groups(
        feature_groups or ("time_domain", "hjorth", "relative_band_power", "entropy", "energies")
    )

    if arr.ndim == 2:
        feature_dict = extract_node_features_for_window(
            arr,
            sfreq,
            feature_groups=selected_groups,
            bands=bands,
            bandpower_nperseg=bandpower_nperseg,
            amplitude_entropy_bins=amplitude_entropy_bins,
            energy_wavelet=energy_wavelet,
            energy_level=energy_level,
        )
        matrix, feature_names = select_feature_groups(feature_dict, selected_groups, concatenate=True)
        meta = {
            "graph_level": "segment",
            "selected_groups": selected_groups,
            "feature_names": feature_names,
            "num_nodes": int(matrix.shape[0]),
            "num_features": int(matrix.shape[1]),
            "num_windows": 1,
        }
        return matrix, meta

    if arr.ndim != 3:
        raise ValueError(
            f"eeg_data must have shape [N, T] or [W, N, T], got {arr.shape}."
        )

    per_window_mats: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for w_idx in range(arr.shape[0]):
        feature_dict = extract_node_features_for_window(
            arr[w_idx],
            sfreq,
            feature_groups=selected_groups,
            bands=bands,
            bandpower_nperseg=bandpower_nperseg,
            amplitude_entropy_bins=amplitude_entropy_bins,
            energy_wavelet=energy_wavelet,
            energy_level=energy_level,
        )
        matrix_w, names_w = select_feature_groups(feature_dict, selected_groups, concatenate=True)
        per_window_mats.append(matrix_w)

        if feature_names is None:
            feature_names = names_w
        elif feature_names != names_w:
            raise ValueError("Feature names changed across windows, which should not happen.")

    stacked = np.stack(per_window_mats, axis=0).astype(np.float32, copy=False)

    if aggregate_windows is None:
        meta = {
            "graph_level": "segment_stack",
            "selected_groups": selected_groups,
            "feature_names": list(feature_names or []),
            "num_windows": int(stacked.shape[0]),
            "num_nodes": int(stacked.shape[1]),
            "num_features": int(stacked.shape[2]),
        }
        return stacked, meta

    aggregated = aggregate_features_across_windows(
        stacked,
        aggregation=aggregate_windows,
    )
    meta = {
        "graph_level": "aggregated",
        "selected_groups": selected_groups,
        "feature_names": list(feature_names or []),
        "aggregation": str(aggregate_windows),
        "num_windows": int(stacked.shape[0]),
        "num_nodes": int(aggregated.shape[0]),
        "num_features": int(aggregated.shape[1]),
    }
    return aggregated, meta


def select_feature_groups(
    feature_groups_data: Mapping[str, FeatureGroupResult | np.ndarray],
    selected_groups: Sequence[str],
    *,
    concatenate: bool = True,
) -> tuple[np.ndarray, list[str]] | dict[str, FeatureGroupResult | np.ndarray]:
    """
    Select a subset of feature groups and optionally concatenate them.

    Parameters
    ----------
    feature_groups_data:
        Mapping from feature group name to ``FeatureGroupResult`` or raw feature array.
    selected_groups:
        Requested feature groups.
    concatenate:
        If True, concatenate along the last dimension.
        If False, return the selected mapping.

    Returns
    -------
    tuple[np.ndarray, list[str]] or dict
        Concatenated feature matrix and feature names, or the selected mapping.
    """
    requested = _normalize_requested_groups(selected_groups)

    selected: dict[str, FeatureGroupResult | np.ndarray] = {}
    for group_name in requested:
        if group_name not in feature_groups_data:
            raise KeyError(
                f"Requested feature group {group_name!r} not found. "
                f"Available groups: {list(feature_groups_data.keys())}"
            )
        selected[group_name] = feature_groups_data[group_name]

    if not concatenate:
        return selected

    arrays: list[np.ndarray] = []
    feature_names: list[str] = []
    leading_shape: tuple[int, ...] | None = None

    for group_name in requested:
        item = selected[group_name]
        if isinstance(item, FeatureGroupResult):
            arr = item.values
            names = list(item.feature_names)
        else:
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim < 2:
                raise ValueError(
                    f"Raw feature group arrays must have at least 2 dimensions, got {arr.shape}."
                )
            names = [f"{group_name}_{i}" for i in range(arr.shape[-1])]

        if leading_shape is None:
            leading_shape = arr.shape[:-1]
        elif leading_shape != arr.shape[:-1]:
            raise ValueError(
                f"Feature group {group_name!r} has incompatible leading shape {arr.shape[:-1]}; "
                f"expected {leading_shape}."
            )

        arrays.append(arr.astype(np.float32, copy=False))
        feature_names.extend(names)

    matrix = np.concatenate(arrays, axis=-1).astype(np.float32, copy=False)
    return matrix, feature_names


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _require_2d_window(eeg_window: np.ndarray) -> np.ndarray:
    arr = np.asarray(eeg_window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(
            f"EEG window must have shape [num_channels, num_timepoints], got {arr.shape}."
        )
    if arr.shape[0] < 1 or arr.shape[1] < 2:
        raise ValueError(f"EEG window is too small, got shape {arr.shape}.")
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


def _normalize_requested_groups(feature_groups: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    for name in feature_groups:
        key = str(name).strip().lower()
        if key not in FEATURE_GROUP_ALIASES:
            raise ValueError(
                f"Unsupported feature group {name!r}. "
                f"Available canonical/alias names: {sorted(FEATURE_GROUP_ALIASES.keys())}"
            )
        canonical = FEATURE_GROUP_ALIASES[key]
        if canonical not in seen:
            out.append(canonical)
            seen.add(canonical)

    return out


def _welch_psd(
    eeg_window: np.ndarray,
    *,
    sfreq: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = _require_2d_window(eeg_window)
    sfreq = _validate_positive_float(sfreq, "sfreq")

    if nperseg is None:
        nperseg_use = min(x.shape[-1], max(256, int(round(2.0 * sfreq))))
    else:
        nperseg_use = min(x.shape[-1], _validate_positive_int(nperseg, "nperseg"))

    noverlap_use = nperseg_use // 2 if nperseg_use > 1 else 0

    freqs, psd = signal.welch(
        x,
        fs=sfreq,
        axis=-1,
        nperseg=nperseg_use,
        noverlap=noverlap_use,
        detrend="constant",
        scaling="density",
    )
    return freqs.astype(np.float32), psd.astype(np.float32)


def _stack_feature_matrices(
    feature_matrices: np.ndarray | Sequence[np.ndarray],
) -> np.ndarray:
    if isinstance(feature_matrices, np.ndarray):
        arr = np.asarray(feature_matrices, dtype=np.float32)
        if arr.ndim == 2:
            return arr[None, ...]
        if arr.ndim != 3:
            raise ValueError(
                f"feature_matrices must have shape [W, N, F] or [N, F], got {arr.shape}."
            )
        return arr.astype(np.float32, copy=False)

    matrices = [np.asarray(m, dtype=np.float32) for m in feature_matrices]
    if len(matrices) == 0:
        raise ValueError("feature_matrices must not be empty.")

    ref_shape = matrices[0].shape
    if len(ref_shape) != 2:
        raise ValueError(f"Each feature matrix must have shape [N, F], got {ref_shape}.")

    for idx, mat in enumerate(matrices[1:], start=1):
        if mat.shape != ref_shape:
            raise ValueError(
                f"Feature matrix at index {idx} has shape {mat.shape}, expected {ref_shape}."
            )

    return np.stack(matrices, axis=0).astype(np.float32, copy=False)

# if __name__ == "__main__":
    # import numpy as np

    # sfreq = 500.0
    # window = np.random.randn(19, 2000).astype(np.float32)

    # # One segment window -> one [N, F] node feature matrix
    # feature_dict = extract_node_features_for_window(
    #     window,
    #     sfreq,
    #     feature_groups=["relative_band_power", "hjorth", "time_domain", "entropy", "energies"],
    # )

    # for name, result in feature_dict.items():
    #     print(name, result.values.shape)

    # segment_x, segment_meta = build_feature_matrix(
    #     window,
    #     sfreq,
    #     feature_groups=["relative_band_power", "hjorth", "energies"],
    # )
    # print(segment_x.shape)
    # print(segment_meta["feature_names"][:8])

    # # Many windows -> one macro/subject feature matrix by aggregation
    # windows = np.stack([window, 0.9 * window, 1.1 * window], axis=0)

    # macro_x, macro_meta = build_feature_matrix(
    #     windows,
    #     sfreq,
    #     feature_groups=["relative_band_power", "hjorth", "energies"],
    #     aggregate_windows="mean",
    # )
    # print(macro_x.shape)
    # print(macro_meta["aggregation"])

# ---------------------------------------------------------------------
# Example usage: test feature extraction on AHEAP and CAUEEG
# ---------------------------------------------------------------------
# from __future__ import annotations


# from features import build_feature_matrix


def _print_feature_summary(
    name: str,
    x: np.ndarray,
    meta: dict,
) -> None:
    print(f"\n[{name}]")
    print("shape:", x.shape)
    print("graph_level:", meta.get("graph_level"))
    print("selected_groups:", meta.get("selected_groups"))
    print("num_features:", meta.get("num_features"))
    print("first feature names:", meta.get("feature_names", [])[:8])


def _extract_segment_macro_subject_features(
    prepared,
    *,
    feature_groups: list[str],
) -> None:
    """
    Demonstrate segment-level, macro-level, and subject-level feature extraction
    from one prepared subject.
    """
    print("\n" + "=" * 80)
    print(f"Subject: {prepared.subject_id} | dataset={prepared.dataset_name} | label={prepared.label}")
    print(f"Valid windows: {prepared.windows.shape[0]}")
    print(f"Channels: {prepared.windows.shape[1]}")
    print(f"Window length (samples): {prepared.windows.shape[2]}")
    print("=" * 80)

    if prepared.windows.shape[0] == 0:
        print("No valid windows for this subject.")
        return

    # --------------------------------------------------
    # 1) Segment level
    # one short window = one graph
    # --------------------------------------------------
    segment_window = prepared.windows[0]  # [N, T]
    segment_x, segment_meta = build_feature_matrix(
        segment_window,
        prepared.sfreq,
        feature_groups=feature_groups,
    )
    _print_feature_summary("SEGMENT", segment_x, segment_meta)

    # --------------------------------------------------
    # 2) Macro level
    # one macro block = many short windows -> aggregated
    # --------------------------------------------------
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

        macro_x, macro_meta = build_feature_matrix(
            macro_windows,
            prepared.sfreq,
            feature_groups=feature_groups,
            aggregate_windows="mean",
        )
        _print_feature_summary(f"MACRO (macro_id={first_macro_id})", macro_x, macro_meta)

        print("macro_id:", first_macro_id)
        print("num_windows_in_macro:", macro_windows.shape[0])
    else:
        print("\n[MACRO]")
        print("No macro grouping available for this subject.")

    # --------------------------------------------------
    # 3) Subject level
    # all valid windows of one subject -> aggregated
    # --------------------------------------------------
    subject_x, subject_meta = build_feature_matrix(
        prepared.windows,
        prepared.sfreq,
        feature_groups=feature_groups,
        aggregate_windows="mean",
    )
    _print_feature_summary("SUBJECT", subject_x, subject_meta)

    print("num_windows_used_for_subject:", prepared.windows.shape[0])


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # Choose a compact set of feature groups to verify everything works.
    # You can add/remove groups here.
    # -----------------------------------------------------------------
    import data_config as config
    feature_groups = [
        "relative_band_power",
        "hjorth",
        "statistical",
        "entropy",
        "energies",
    ]

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

        _extract_segment_macro_subject_features(
            aheap_prepared,
            feature_groups=feature_groups,
        )
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

        caueeg_prepared = prepare_subject_windows(
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

        _extract_segment_macro_subject_features(
            caueeg_prepared,
            feature_groups=feature_groups,
        )
    else:
        print("No CAUEEG records were loaded.")