"""
caueeg_h5_eda.py

High-density EDA for the CAUEEG-style master H5 files used in the
LINKX/MIL EEG pipeline.

Expected H5 structure, matching the current project loader:

subjects/{subject_id}/metadata.attrs['label']
subjects/{subject_id}/metadata/channel_names
subjects/{subject_id}/windows/raw/segment_id
subjects/{subject_id}/windows/raw/start_sample
subjects/{subject_id}/windows/raw/end_sample
subjects/{subject_id}/windows/features/{family}          # [W, C, F]
subjects/{subject_id}/windows/connectivity/{metric}      # [W, C, C] or [W, B, C, C]

Main outputs:
- scalar feature violin/raincloud-style plots
- connectivity class x band heatmap grids
- connectivity distribution boxplots
- channel-wise feature bar plots
- split/class feature summary table
- subject-level artifact/outlier table

Dependencies:
    pip install h5py numpy pandas matplotlib seaborn openpyxl
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------

DEFAULT_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]

# Edit this if your extraction code uses a different order.
# If H5 datasets contain attrs['feature_names'], those names override this map.
DEFAULT_FEATURE_NAME_MAP: dict[str, list[str]] = {
    # One feature per channel
    "higuchi_fd": ["higuchi_fd"],
    "spectral_entropy": ["spectral_entropy"],

    # Three Hjorth parameters per channel
    "hjorth": ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"],

    # Five EEG bands per channel
    "log_band_power": [
        "log_power_delta", "log_power_theta", "log_power_alpha",
        "log_power_beta", "log_power_gamma",
    ],
    "relative_band_power": ["rbp_delta", "rbp_theta", "rbp_alpha", "rbp_beta", "rbp_gamma"],
    "rbp": ["rbp_delta", "rbp_theta", "rbp_alpha", "rbp_beta", "rbp_gamma"],
    "band_power": DEFAULT_BAND_NAMES,

    # Statistical features per channel
    "statistical": ["mean", "std", "skew", "kurtosis", "min", "max", "ptp"],
    "stats": ["mean", "std", "skew", "kurtosis", "min", "max", "ptp"],

    # Discrete wavelet energy features per channel
    "wavelet_energy": [
        "wavelet_energy_a5", "wavelet_energy_d5", "wavelet_energy_d4",
        "wavelet_energy_d3", "wavelet_energy_d2", "wavelet_energy_d1",
    ],
}

# Common CAUEEG dementia task label conventions. Override from CLI or function call
# if your H5 uses a different mapping.
DEFAULT_CLASS_MAP = {
    0: "HC",
    1: "MCI",
    2: "AD",       # CAUEEG paper calls this class "dementia"; use AD if your labels are AD/MCI/HC.
}

DEFAULT_PALETTE = "colorblind"


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _decode_one(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def decode_str_list(arr: Iterable[Any]) -> list[str]:
    """Decode an H5 string array into Python strings."""
    return [_decode_one(x) for x in arr]


def safe_get_attrs(group_or_ds: h5py.Group | h5py.Dataset) -> dict[str, Any]:
    """Return H5 attrs as a plain dict with decoded strings where possible."""
    out = {}
    for k, v in group_or_ds.attrs.items():
        if isinstance(v, bytes):
            out[k] = v.decode("utf-8")
        elif isinstance(v, np.ndarray) and v.dtype.kind in {"S", "O"}:
            out[k] = decode_str_list(v)
        else:
            out[k] = v
    return out


def load_class_map(class_map_json: Optional[str]) -> dict[int, str]:
    """
    Load a class map from a JSON string or JSON file.

    Examples
    --------
    --class_map '{"0":"HC","1":"MCI","2":"AD"}'
    --class_map /path/to/class_map.json
    """
    if not class_map_json:
        return dict(DEFAULT_CLASS_MAP)

    p = Path(class_map_json)
    if p.exists():
        raw = json.loads(p.read_text())
    else:
        raw = json.loads(class_map_json)
    return {int(k): str(v) for k, v in raw.items()}


def read_split_lookup(split_csv: Optional[str | Path]) -> dict[str, str]:
    """
    Optional external subject split table.

    The CSV should contain columns: subject_id, data_split.
    Use this when the H5 subject IDs do not encode the split.
    """
    if split_csv is None:
        return {}
    df = pd.read_csv(split_csv)
    required = {"subject_id", "data_split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split_csv is missing columns: {missing}")
    return dict(zip(df["subject_id"].astype(str), df["data_split"].astype(str)))


def infer_data_split(
    sid: str,
    grp: h5py.Group,
    split_lookup: Optional[Mapping[str, str]] = None,
) -> str:
    """
    Infer Train/Val/Test from, in priority order:
    1) external split lookup
    2) metadata attrs such as data_split/split/split_name
    3) recording_info attrs, if present
    4) subject ID prefix such as train_00587 or test-00587
    5) 'Unknown'
    """
    sid = str(sid)
    if split_lookup and sid in split_lookup:
        return str(split_lookup[sid])

    candidate_attr_names = ("data_split", "split", "split_name", "set")
    for path in ("metadata", "recording_info"):
        if path in grp:
            attrs = safe_get_attrs(grp[path])
            for name in candidate_attr_names:
                if name in attrs:
                    return normalize_split_name(attrs[name])

    m = re.match(r"^(train|training|val|valid|validation|test)[_\-:/]", sid, flags=re.IGNORECASE)
    if m:
        return normalize_split_name(m.group(1))

    return "Unknown"


def normalize_split_name(x: Any) -> str:
    """Normalize split names for plotting."""
    s = str(x).strip().lower()
    if s in {"train", "training"}:
        return "Train"
    if s in {"val", "valid", "validation"}:
        return "Val"
    if s == "test":
        return "Test"
    return str(x)



def sanitize_h5_name(x: Any) -> str:
    """
    Clean feature/band names stored as JSON-like H5 attrs.

    This fixes names like '["rbp_delta"', '"rbp_theta"', and '"ptp"]'
    into 'rbp_delta', 'rbp_theta', and 'ptp'.
    """
    s = _decode_one(x)
    for _ in range(4):
        s = s.strip()
        s = s.strip("[](){}")
        s = s.strip()
        s = s.strip("'\"")
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_+\-./]", "", s)
    s = s.strip("_")
    return s


def coerce_h5_name_sequence(raw: Any) -> list[str]:
    """
    Convert H5 attrs that may be stored as a real list, NumPy array, JSON
    string, Python-list string, or comma-separated string into clean names.
    """
    if raw is None:
        return []

    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            raw = raw.item()
        else:
            values = raw.tolist()
            return [sanitize_h5_name(v) for v in values if sanitize_h5_name(v)]

    if isinstance(raw, (bytes, np.bytes_)):
        raw = raw.decode("utf-8")

    if isinstance(raw, str):
        text = raw.strip()
        # Common case: attrs['feature_names'] = '["mean", "std", ...]'
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return [sanitize_h5_name(v) for v in parsed if sanitize_h5_name(v)]
                if isinstance(parsed, str):
                    return [sanitize_h5_name(parsed)]
            except Exception:
                pass
        # Fallback for malformed list strings.
        values = [v for v in text.split(",") if v.strip()]
        return [sanitize_h5_name(v) for v in values if sanitize_h5_name(v)]

    if isinstance(raw, (list, tuple)):
        return [sanitize_h5_name(v) for v in raw if sanitize_h5_name(v)]

    return [sanitize_h5_name(raw)]


def infer_feature_names(
    ds: h5py.Dataset,
    family: str,
    n_features: int,
    feature_name_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> list[str]:
    """
    Infer feature names for one feature family.

    Priority:
    1) H5 attrs: feature_names / names / columns
    2) user/default feature_name_map
    3) fallback feature_00, feature_01, ...
    """
    attrs = safe_get_attrs(ds)
    for key in ("feature_names", "names", "columns"):
        if key in attrs:
            names = coerce_h5_name_sequence(attrs[key])
            if len(names) == n_features:
                return names

    fmap: dict[str, Sequence[str]] = dict(DEFAULT_FEATURE_NAME_MAP)
    if feature_name_map:
        fmap.update(feature_name_map)
    if family in fmap and len(fmap[family]) == n_features:
        return [sanitize_h5_name(x) for x in fmap[family]]

    return [f"{sanitize_h5_name(family)}_{i:02d}" for i in range(n_features)]


def infer_band_names(ds: h5py.Dataset, n_bands: int) -> list[str]:
    """Infer band names from H5 attrs, otherwise use delta/theta/alpha/beta/gamma."""
    attrs = safe_get_attrs(ds)
    for key in ("band_names", "bands"):
        if key in attrs:
            names = attrs[key]
            if isinstance(names, str):
                names = [x.strip() for x in names.split(",") if x.strip()]
            names = [str(x) for x in list(names)]
            if len(names) == n_bands:
                return names
    if n_bands == len(DEFAULT_BAND_NAMES):
        return list(DEFAULT_BAND_NAMES)
    return [f"band_{i}" for i in range(n_bands)]


def list_h5_inventory(h5_path: str | Path) -> dict[str, Any]:
    """Return available subjects, feature families, connectivity metrics, and channels."""
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as h5f:
        if "subjects" not in h5f:
            raise KeyError("Expected top-level group 'subjects' in H5 file.")
        subject_ids = list(h5f["subjects"].keys())
        if not subject_ids:
            raise ValueError("No subjects found in H5 file.")

        sid0 = subject_ids[0]
        grp0 = h5f[f"subjects/{sid0}"]
        feature_families = []
        if "windows/features" in grp0:
            feature_families = list(grp0["windows/features"].keys())
        connectivity_metrics = []
        if "windows/connectivity" in grp0:
            connectivity_metrics = list(grp0["windows/connectivity"].keys())
        channels = []
        if "metadata/channel_names" in grp0:
            channels = decode_str_list(grp0["metadata/channel_names"][:])

    return {
        "num_subjects": len(subject_ids),
        "subject_ids": subject_ids,
        "feature_families": feature_families,
        "connectivity_metrics": connectivity_metrics,
        "channel_names": channels,
    }


def upper_triangle_values(mat: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    """Flatten upper triangle of a square matrix."""
    mat = np.asarray(mat, dtype=np.float64)
    k = 0 if include_diagonal else 1
    iu = np.triu_indices(mat.shape[-1], k=k)
    return mat[iu]


def clean_array(x: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Convert inf to NaN by default; useful before pandas/seaborn."""
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, fill_value)
    return x


# ---------------------------------------------------------------------
# H5 -> scalar feature dataframe
# ---------------------------------------------------------------------

def load_scalar_feature_wide_df(
    h5_path: str | Path,
    *,
    feature_families: Optional[Sequence[str]] = None,
    class_map: Optional[Mapping[int, str]] = None,
    split_lookup: Optional[Mapping[str, str]] = None,
    feature_name_map: Optional[Mapping[str, Sequence[str]]] = None,
    include_bad_segment_flag: bool = True,
    drop_bad_segments: bool = False,
    max_subjects: Optional[int] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load scalar node features into a wide dataframe.

    Output row grain:
        one row = one subject × one window × one channel

    Metadata columns:
        subject_id, class_id, class_label, data_split, window_id,
        segment_id, start_sample, end_sample, channel, channel_idx

    Feature columns:
        {family}__{feature_name}

    Returns
    -------
    wide_df : pd.DataFrame
    feature_cols : list[str]
    """
    h5_path = str(h5_path)
    class_map = dict(class_map or DEFAULT_CLASS_MAP)
    split_lookup = dict(split_lookup or {})
    frames: list[pd.DataFrame] = []
    all_feature_cols: list[str] = []

    with h5py.File(h5_path, "r") as h5f:
        subject_ids = list(h5f["subjects"].keys())
        if max_subjects is not None:
            subject_ids = subject_ids[: int(max_subjects)]

        for sid in subject_ids:
            if sid in {"train_00587", "train_00781", "train_01301"}:
                continue
            grp = h5f[f"subjects/{sid}"]
            if "windows/features" not in grp:
                continue

            families = list(feature_families or grp["windows/features"].keys())
            families = [f for f in families if f in grp["windows/features"]]
            if not families:
                continue

            label = int(grp["metadata"].attrs.get("label", -1))
            class_label = class_map.get(label, f"class_{label}")
            data_split = infer_data_split(sid, grp, split_lookup=split_lookup)

            channel_names = decode_str_list(grp["metadata/channel_names"][:]) if "metadata/channel_names" in grp else None

            # Use the first family to define W and C.
            ref = np.asarray(grp[f"windows/features/{families[0]}"][:], dtype=np.float32)
            if ref.ndim != 3:
                raise ValueError(f"Feature tensor {sid}/{families[0]} must be [W,C,F], got {ref.shape}")
            W, C, _ = ref.shape
            if channel_names is None or len(channel_names) != C:
                channel_names = [f"ch_{i}" for i in range(C)]

            segment_id = grp["windows/raw/segment_id"][:].astype(int) if "windows/raw/segment_id" in grp else np.arange(W)
            start_sample = grp["windows/raw/start_sample"][:].astype(int) if "windows/raw/start_sample" in grp else np.full(W, -1)
            end_sample = grp["windows/raw/end_sample"][:].astype(int) if "windows/raw/end_sample" in grp else np.full(W, -1)

            bad_flag = None
            if include_bad_segment_flag and "windows/qc/bad_segment_flag" in grp:
                bad_flag = grp["windows/qc/bad_segment_flag"][:].astype(int)

            # Metadata grid [W*C rows]
            win_idx = np.repeat(np.arange(W), C)
            ch_idx = np.tile(np.arange(C), W)
            df = pd.DataFrame({
                "subject_id": str(sid),
                "class_id": label,
                "class_label": class_label,
                "data_split": data_split,
                "window_id": win_idx,
                "segment_id": segment_id[win_idx],
                "start_sample": start_sample[win_idx],
                "end_sample": end_sample[win_idx],
                "channel_idx": ch_idx,
                "channel": [channel_names[i] for i in ch_idx],
            })
            if bad_flag is not None:
                df["bad_segment_flag"] = bad_flag[win_idx]

            for fam in families:
                ds = grp[f"windows/features/{fam}"]
                x = np.asarray(ds[:], dtype=np.float32)
                if x.ndim != 3:
                    raise ValueError(f"Feature tensor {sid}/{fam} must be [W,C,F], got {x.shape}")
                if x.shape[:2] != (W, C):
                    raise ValueError(f"Feature tensor {sid}/{fam} has [W,C]={x.shape[:2]}, expected {(W,C)}")

                names = infer_feature_names(ds, fam, x.shape[-1], feature_name_map=feature_name_map)
                x2 = clean_array(x).reshape(W * C, x.shape[-1])
                for j, fname in enumerate(names):
                    col = f"{fam}__{fname}"
                    df[col] = x2[:, j]
                    if col not in all_feature_cols:
                        all_feature_cols.append(col)

            if drop_bad_segments and "bad_segment_flag" in df.columns:
                df = df[df["bad_segment_flag"].fillna(0).astype(int) == 0].copy()

            frames.append(df)

    if not frames:
        raise ValueError("No scalar feature rows were loaded. Check feature_families and H5 structure.")

    wide_df = pd.concat(frames, ignore_index=True)
    wide_df = add_derived_scalar_features(wide_df, all_feature_cols)
    for c in wide_df.columns:
        if c not in all_feature_cols and "alpha_theta_ratio" in c:
            all_feature_cols.append(c)
    return wide_df, all_feature_cols


def add_derived_scalar_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Add derived clinical features such as alpha/theta ratio when possible."""
    df = df.copy()

    possible_alpha = [
        "relative_band_power__rbp_alpha",
        "relative_band_power__alpha",
        "rbp__rbp_alpha",
        "rbp__alpha",
        "band_power__alpha",
        "log_band_power__log_power_alpha",
    ]
    possible_theta = [
        "relative_band_power__rbp_theta",
        "relative_band_power__theta",
        "rbp__rbp_theta",
        "rbp__theta",
        "band_power__theta",
        "log_band_power__log_power_theta",
    ]
    alpha_col = next((c for c in possible_alpha if c in df.columns), None)
    theta_col = next((c for c in possible_theta if c in df.columns), None)
    if alpha_col and theta_col:
        out_col = "derived__alpha_theta_ratio"
        df[out_col] = df[alpha_col] / (df[theta_col].abs() + 1e-8)
        if out_col not in feature_cols:
            feature_cols.append(out_col)

    return df


def wide_to_long_features(
    wide_df: pd.DataFrame,
    feature_cols: Sequence[str],
    *,
    max_rows: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Convert wide scalar feature dataframe to long/tidy format for seaborn."""
    meta_cols = [
        "subject_id", "class_id", "class_label", "data_split", "window_id",
        "segment_id", "start_sample", "end_sample", "channel_idx", "channel",
    ]
    if "bad_segment_flag" in wide_df.columns:
        meta_cols.append("bad_segment_flag")

    df = wide_df[meta_cols + list(feature_cols)].copy()
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=int(max_rows), random_state=random_state)

    long_df = df.melt(
        id_vars=meta_cols,
        value_vars=list(feature_cols),
        var_name="feature_key",
        value_name="value",
    )
    long_df[["feature_family", "feature_name"]] = long_df["feature_key"].str.split("__", n=1, expand=True)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df[np.isfinite(long_df["value"].to_numpy(dtype=float, na_value=np.nan))].copy()
    return long_df


# ---------------------------------------------------------------------
# Plot 1: Global scalar feature distributions
# ---------------------------------------------------------------------

def plot_global_feature_distributions(
    scalar_long_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    features: Optional[Sequence[str]] = None,
    standardize_per_feature: bool = True,
    max_features_per_fig: int = 12,
    sample_per_feature: Optional[int] = 5000,
    palette: str = DEFAULT_PALETTE,
    violin_inner: str = "quartile",
    add_strip: bool = True,
    dpi: int = 300,
) -> list[str]:
    """
    Plot high-density violin/raincloud-style distributions.

    Facet = feature_key, x = class_label, y = value or z(value), hue = data_split.
    To compare many features on very different scales, standardize_per_feature=True
    is recommended.
    """
    out_dir = ensure_dir(output_dir)
    df = scalar_long_df.copy()
    if features is not None:
        df = df[df["feature_key"].isin(features)].copy()
    if df.empty:
        raise ValueError("No rows available for global feature distribution plot.")

    if standardize_per_feature:
        def _zscore(s: pd.Series) -> pd.Series:
            mu = s.mean(skipna=True)
            sd = s.std(skipna=True)
            return (s - mu) / (sd + 1e-8)
        df["plot_value"] = df.groupby("feature_key", group_keys=False)["value"].transform(_zscore)
        y_label = "Feature value z-score within feature"
    else:
        df["plot_value"] = df["value"]
        y_label = "Feature value"

    if sample_per_feature is not None:
        # Avoid pandas GroupBy.apply deprecation and keep plotting memory bounded.
        sampled_groups = []
        for _, g in df.groupby("feature_key", sort=False):
            sampled_groups.append(g.sample(n=min(len(g), int(sample_per_feature)), random_state=42))
        df = pd.concat(sampled_groups, ignore_index=True) if sampled_groups else df.iloc[0:0].copy()

    feature_list = list(pd.unique(df["feature_key"]))
    paths: list[str] = []
    for page, start in enumerate(range(0, len(feature_list), max_features_per_fig), start=1):
        page_features = feature_list[start:start + max_features_per_fig]
        page_df = df[df["feature_key"].isin(page_features)].copy()

        ncols = min(4, len(page_features))
        nrows = math.ceil(len(page_features) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
        axes_flat = axes.ravel()

        for ax, fkey in zip(axes_flat, page_features):
            sub = page_df[page_df["feature_key"] == fkey]
            sns.violinplot(
                data=sub,
                x="class_label",
                y="plot_value",
                hue="data_split",
                palette=palette,
                cut=0,
                inner=violin_inner,
                dodge=True,
                linewidth=0.8,
                ax=ax,
            )
            if add_strip:
                sns.stripplot(
                    data=sub.sample(n=min(len(sub), 2000), random_state=42),
                    x="class_label",
                    y="plot_value",
                    hue="data_split",
                    palette=palette,
                    dodge=True,
                    size=1.2,
                    alpha=0.18,
                    linewidth=0,
                    ax=ax,
                    legend=False,
                )
            ax.set_title(fkey, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel(y_label)
            ax.tick_params(axis="x", rotation=20)
            # Keep only one legend outside later.
            if ax.get_legend() is not None:
                ax.get_legend().remove()

        for ax in axes_flat[len(page_features):]:
            ax.set_visible(False)

        handles, labels = axes_flat[0].get_legend_handles_labels() if axes_flat[0].get_legend() else ([], [])
        # Rebuild legend from the first axis artists.
        if not handles:
            handles, labels = axes_flat[0].get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            fig.legend(unique.values(), unique.keys(), title="Split", loc="upper center", ncol=3, frameon=False)

        fig.suptitle("Global scalar feature distributions by class and split", y=1.02, fontsize=14)
        fig.tight_layout()
        path = out_dir / f"plot1_global_feature_distribution_page{page:02d}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(path))

    return paths


# ---------------------------------------------------------------------
# Connectivity loading and aggregation
# ---------------------------------------------------------------------

def compute_mean_connectivity_by_group(
    h5_path: str | Path,
    *,
    metrics: Optional[Sequence[str]] = None,
    class_map: Optional[Mapping[int, str]] = None,
    split_lookup: Optional[Mapping[str, str]] = None,
    split: Optional[str] = None,
    subject_weighted: bool = True,
    use_absolute: bool = False,
    zero_diagonal: bool = True,
    symmetrize: bool = True,
    max_subjects: Optional[int] = None,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], list[str], list[str]]:
    """
    Stream H5 connectivity matrices and average by metric × class × band.

    Returns
    -------
    mean_conn[metric][class_label][band_name] = [C, C] averaged matrix
    channel_names
    class_order

    Notes
    -----
    With subject_weighted=True, each subject contributes one matrix per band
    after averaging over its windows. This avoids giving more weight to subjects
    with more windows.
    """
    h5_path = str(h5_path)
    class_map = dict(class_map or DEFAULT_CLASS_MAP)
    split_lookup = dict(split_lookup or {})

    sums: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    channel_names: list[str] = []
    seen_classes: list[str] = []
    band_names_by_metric: dict[str, list[str]] = {}

    with h5py.File(h5_path, "r") as h5f:
        subject_ids = list(h5f["subjects"].keys())
        if max_subjects is not None:
            subject_ids = subject_ids[: int(max_subjects)]

        for sid in subject_ids:
            if sid in {"train_00587", "train_00781", "train_01301"}:
                continue
            grp = h5f[f"subjects/{sid}"]
            if "windows/connectivity" not in grp:
                continue
            data_split = infer_data_split(sid, grp, split_lookup=split_lookup)
            if split is not None and normalize_split_name(data_split) != normalize_split_name(split):
                continue

            label = int(grp["metadata"].attrs.get("label", -1))
            class_label = class_map.get(label, f"class_{label}")
            if class_label not in seen_classes:
                seen_classes.append(class_label)

            if not channel_names and "metadata/channel_names" in grp:
                channel_names = decode_str_list(grp["metadata/channel_names"][:])

            metric_list = list(metrics or grp["windows/connectivity"].keys())
            for metric in metric_list:
                if metric not in grp["windows/connectivity"]:
                    continue
                ds = grp[f"windows/connectivity/{metric}"]
                arr = np.asarray(ds[:], dtype=np.float64)
                if arr.ndim == 3:
                    arr = arr[:, None, :, :]  # [W, 1, C, C]
                if arr.ndim != 4:
                    raise ValueError(f"Connectivity {sid}/{metric} must be [W,C,C] or [W,B,C,C], got {arr.shape}")

                W, B, C, C2 = arr.shape
                if C != C2:
                    raise ValueError(f"Connectivity {sid}/{metric} is not square: {arr.shape}")
                bnames = infer_band_names(ds, B)
                band_names_by_metric[metric] = bnames

                arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
                if symmetrize:
                    arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
                if use_absolute:
                    arr = np.abs(arr)
                if zero_diagonal:
                    idx = np.arange(C)
                    arr[:, :, idx, idx] = np.nan

                if subject_weighted:
                    subject_mean = np.nanmean(arr, axis=0)  # [B,C,C]
                    for b, bname in enumerate(bnames):
                        mat = subject_mean[b]
                        if bname not in sums[metric][class_label]:
                            sums[metric][class_label][bname] = np.zeros_like(mat, dtype=np.float64)
                            counts[metric][class_label][bname] = 0
                        sums[metric][class_label][bname] += np.nan_to_num(mat, nan=0.0)
                        counts[metric][class_label][bname] += 1
                else:
                    # Window-weighted: each window contributes equally.
                    for b, bname in enumerate(bnames):
                        mat = np.nanmean(arr[:, b], axis=0)
                        if bname not in sums[metric][class_label]:
                            sums[metric][class_label][bname] = np.zeros_like(mat, dtype=np.float64)
                            counts[metric][class_label][bname] = 0
                        sums[metric][class_label][bname] += np.nan_to_num(mat, nan=0.0)
                        counts[metric][class_label][bname] += 1

    mean_conn: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(lambda: defaultdict(dict))
    for metric, class_dict in sums.items():
        for cls, band_dict in class_dict.items():
            for bname, mat_sum in band_dict.items():
                n = max(counts[metric][cls][bname], 1)
                mat = mat_sum / n
                if zero_diagonal:
                    np.fill_diagonal(mat, np.nan)
                mean_conn[metric][cls][bname] = mat

    return mean_conn, channel_names, seen_classes


def connectivity_subject_summary_df(
    h5_path: str | Path,
    *,
    metrics: Optional[Sequence[str]] = None,
    class_map: Optional[Mapping[int, str]] = None,
    split_lookup: Optional[Mapping[str, str]] = None,
    use_absolute: bool = True,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    max_subjects: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a compact subject-level connectivity summary dataframe.

    Rows include:
    - one 'global' row per subject × metric × band: mean upper-triangle connectivity
    - one 'channel' row per subject × metric × band × channel: channel mean connectivity strength

    This is much smaller than storing all edges/windows and is suitable for boxplots.
    """
    h5_path = str(h5_path)
    class_map = dict(class_map or DEFAULT_CLASS_MAP)
    split_lookup = dict(split_lookup or {})
    rows: list[dict[str, Any]] = []

    with h5py.File(h5_path, "r") as h5f:
        subject_ids = list(h5f["subjects"].keys())
        if max_subjects is not None:
            subject_ids = subject_ids[: int(max_subjects)]

        for sid in subject_ids:
            if sid in {"train_00587", "train_00781", "train_01301"}:
                continue
            grp = h5f[f"subjects/{sid}"]
            if "windows/connectivity" not in grp:
                continue
            label = int(grp["metadata"].attrs.get("label", -1))
            class_label = class_map.get(label, f"class_{label}")
            data_split = infer_data_split(sid, grp, split_lookup=split_lookup)
            channel_names = decode_str_list(grp["metadata/channel_names"][:]) if "metadata/channel_names" in grp else None

            metric_list = list(metrics or grp["windows/connectivity"].keys())
            for metric in metric_list:
                if metric not in grp["windows/connectivity"]:
                    continue
                ds = grp[f"windows/connectivity/{metric}"]
                arr = np.asarray(ds[:], dtype=np.float64)
                if arr.ndim == 3:
                    arr = arr[:, None, :, :]
                if arr.ndim != 4:
                    raise ValueError(f"Connectivity {sid}/{metric} must be [W,C,C] or [W,B,C,C], got {arr.shape}")

                W, B, C, _ = arr.shape
                if channel_names is None or len(channel_names) != C:
                    channel_names = [f"ch_{i}" for i in range(C)]
                bnames = infer_band_names(ds, B)

                arr = np.nan_to_num(arr, nan=np.nan, posinf=np.nan, neginf=np.nan)
                if symmetrize:
                    arr = 0.5 * (arr + np.swapaxes(arr, -1, -2))
                if use_absolute:
                    arr = np.abs(arr)
                if zero_diagonal:
                    idx = np.arange(C)
                    arr[:, :, idx, idx] = np.nan

                # Average windows first so every subject has equal weight.
                subj_band_mean = np.nanmean(arr, axis=0)  # [B,C,C]

                for b, bname in enumerate(bnames):
                    mat = subj_band_mean[b]
                    global_value = np.nanmean(upper_triangle_values(mat, include_diagonal=False))
                    rows.append({
                        "level": "global",
                        "subject_id": str(sid),
                        "class_id": label,
                        "class_label": class_label,
                        "data_split": data_split,
                        "metric": metric,
                        "band": bname,
                        "channel": "ALL",
                        "channel_idx": -1,
                        "value": float(global_value),
                    })

                    # Channel mean strength to all other channels.
                    ch_strength = np.nanmean(mat, axis=1)
                    for ci, val in enumerate(ch_strength):
                        rows.append({
                            "level": "channel",
                            "subject_id": str(sid),
                            "class_id": label,
                            "class_label": class_label,
                            "data_split": data_split,
                            "metric": metric,
                            "band": bname,
                            "channel": channel_names[ci],
                            "channel_idx": ci,
                            "value": float(val),
                        })

    out = pd.DataFrame(rows)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[np.isfinite(out["value"].to_numpy(dtype=float, na_value=np.nan))].copy()
    return out



def _aggregate_connectivity_values(
    df: pd.DataFrame,
    group_cols: list[str],
    *,
    value_col: str,
    summary_level: str,
) -> pd.DataFrame:
    """Internal helper for grouped connectivity summary statistics."""
    grouped = df.groupby(group_cols, dropna=False, observed=True)
    out = grouped.agg(
        n_values=(value_col, "count"),
        n_subjects=("subject_id", pd.Series.nunique),
        mean=(value_col, "mean"),
        std=(value_col, "std"),
        min=(value_col, "min"),
        q05=(value_col, lambda x: x.quantile(0.05)),
        q25=(value_col, lambda x: x.quantile(0.25)),
        median=(value_col, "median"),
        q75=(value_col, lambda x: x.quantile(0.75)),
        q95=(value_col, lambda x: x.quantile(0.95)),
        max=(value_col, "max"),
    ).reset_index()

    if "channel" in df.columns and summary_level == "channel":
        ch = grouped["channel"].nunique().reset_index(name="n_channels")
        out = out.merge(ch, on=group_cols, how="left")
    else:
        out["n_channels"] = np.nan

    out["sem"] = out["std"] / np.sqrt(out["n_values"].clip(lower=1))
    out["cv"] = out["std"] / out["mean"].replace(0, np.nan)
    out["iqr"] = out["q75"] - out["q25"]
    out["range"] = out["max"] - out["min"]
    out["summary_level"] = summary_level
    return out


def make_connectivity_split_class_summary(
    conn_summary_df: pd.DataFrame,
    *,
    level: str = "channel",
    include_band_level: bool = True,
    include_all_bands: bool = True,
    include_all_splits: bool = True,
    include_all_classes: bool = False,
    value_col: str = "value",
) -> pd.DataFrame:
    """
    Summarize connectivity by data_split, class_label, metric, and band.

    level='channel' aggregates all subject-channel mean connectivity values.
    This matches the table requested for pooling all subjects and all channels.

    level='global' aggregates one subject-level global connectivity value per
    subject, metric, and band, so each subject contributes one value.
    """
    required = {"level", "data_split", "class_label", "subject_id", "metric", "band", value_col}
    missing = sorted(required - set(conn_summary_df.columns))
    if missing:
        raise KeyError(f"conn_summary_df is missing required columns: {missing}")

    summary_level = str(level).lower()
    df = conn_summary_df[conn_summary_df["level"].astype(str).str.lower() == summary_level].copy()
    if df.empty:
        raise ValueError(f"No rows found for connectivity level={level!r}.")

    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[np.isfinite(df[value_col].to_numpy(dtype=float, na_value=np.nan))].copy()
    if df.empty:
        raise ValueError(f"No finite connectivity values found for level={level!r}.")

    variants = [("base", df)]
    if include_all_splits:
        tmp = df.copy()
        tmp["data_split"] = "ALL"
        variants.append(("all_splits", tmp))
    if include_all_classes:
        tmp = df.copy()
        tmp["class_label"] = "ALL"
        variants.append(("all_classes", tmp))
        if include_all_splits:
            tmp2 = df.copy()
            tmp2["data_split"] = "ALL"
            tmp2["class_label"] = "ALL"
            variants.append(("all_splits_classes", tmp2))

    pieces: list[pd.DataFrame] = []
    for _, work in variants:
        if include_band_level:
            pieces.append(_aggregate_connectivity_values(
                work, ["data_split", "class_label", "metric", "band"],
                value_col=value_col, summary_level=summary_level,
            ))
        if include_all_bands:
            pooled = work.copy()
            pooled["band"] = "ALL"
            pieces.append(_aggregate_connectivity_values(
                pooled, ["data_split", "class_label", "metric", "band"],
                value_col=value_col, summary_level=summary_level,
            ))

    out = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    ordered_cols = [
        "summary_level", "data_split", "class_label", "metric", "band",
        "n_values", "n_subjects", "n_channels",
        "mean", "std", "sem", "cv", "min", "q05", "q25",
        "median", "q75", "q95", "max", "iqr", "range",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]
    out = out.drop_duplicates().sort_values(
        ["summary_level", "data_split", "class_label", "metric", "band"]
    ).reset_index(drop=True)
    return out
# ---------------------------------------------------------------------
# Plot 2A: Connectivity heatmap facet grid
# ---------------------------------------------------------------------

def plot_connectivity_grid(
    mean_conn: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: str | Path,
    *,
    metric: str,
    channel_names: Optional[Sequence[str]] = None,
    class_order: Optional[Sequence[str]] = None,
    band_order: Optional[Sequence[str]] = None,
    cmap: str = "viridis",
    robust_quantile: float = 0.98,
    center_zero: bool = False,
    dpi: int = 300,
) -> str:
    """
    Plot class × band grid of averaged Channel × Channel connectivity heatmaps.
    """
    out_dir = ensure_dir(output_dir)
    if metric not in mean_conn:
        raise KeyError(f"Metric {metric!r} not available in mean_conn.")

    metric_dict = mean_conn[metric]
    classes = list(class_order or metric_dict.keys())
    bands = list(band_order or next(iter(metric_dict.values())).keys())

    # Global color scale for fair class/band comparison.
    all_vals = []
    for cls in classes:
        for band in bands:
            if cls in metric_dict and band in metric_dict[cls]:
                all_vals.append(metric_dict[cls][band].reshape(-1))
    vals = np.concatenate(all_vals) if all_vals else np.array([0.0])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        vals = np.array([0.0])
    vmax = float(np.nanquantile(np.abs(vals) if center_zero else vals, robust_quantile))
    if center_zero:
        vmin = -vmax
    else:
        vmin = float(np.nanquantile(vals, 1.0 - robust_quantile))

    nrows = len(classes)
    ncols = len(bands)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)

    for r, cls in enumerate(classes):
        for c, band in enumerate(bands):
            ax = axes[r, c]
            if cls not in metric_dict or band not in metric_dict[cls]:
                ax.set_visible(False)
                continue
            mat = metric_dict[cls][band]
            sns.heatmap(
                mat,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar=(c == ncols - 1),
                xticklabels=channel_names if channel_names else False,
                yticklabels=channel_names if channel_names else False,
            )
            ax.set_title(f"{cls} | {band}", fontsize=10)
            if r != nrows - 1:
                ax.set_xticklabels([])
            else:
                ax.tick_params(axis="x", labelrotation=90, labelsize=6)
            if c != 0:
                ax.set_yticklabels([])
            else:
                ax.tick_params(axis="y", labelsize=6)

    fig.suptitle(f"Mean connectivity heatmaps: {metric}", y=1.01, fontsize=14)
    fig.tight_layout()
    path = out_dir / f"plot2_connectivity_heatmap_grid_{metric}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------
# Plot 2B: Connectivity boxplots
# ---------------------------------------------------------------------

def plot_connectivity_boxplots(
    conn_summary_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    level: str = "global",
    metrics: Optional[Sequence[str]] = None,
    palette: str = DEFAULT_PALETTE,
    dpi: int = 300,
) -> str:
    """
    Boxplot summary for connectivity values.

    For level='global':
        row = metric, x = band, hue = class_label, y = subject-level mean edge value.

    For level='channel':
        row = metric, col = band, x = channel, hue = class_label, y = subject-level channel strength.
        This can be dense, but is useful for finding channel-specific connectivity differences.
    """
    out_dir = ensure_dir(output_dir)
    df = conn_summary_df[conn_summary_df["level"] == level].copy()
    if metrics is not None:
        df = df[df["metric"].isin(metrics)].copy()
    if df.empty:
        raise ValueError(f"No connectivity summary rows for level={level!r}.")

    if level == "global":
        metric_list = list(pd.unique(df["metric"]))
        nrows = len(metric_list)
        fig, axes = plt.subplots(nrows, 1, figsize=(10, max(3.5, 3.0 * nrows)), squeeze=False)
        for ax, metric in zip(axes.ravel(), metric_list):
            sub = df[df["metric"] == metric]
            sns.boxplot(data=sub, x="band", y="value", hue="class_label", palette=palette, ax=ax)
            sns.stripplot(
                data=sub.sample(n=min(len(sub), 3000), random_state=42),
                x="band", y="value", hue="class_label", palette=palette,
                dodge=True, alpha=0.25, size=2, linewidth=0, ax=ax, legend=False,
            )
            ax.set_title(metric)
            ax.set_xlabel("Frequency band")
            ax.set_ylabel("Mean edge connectivity")
            handles, labels = ax.get_legend_handles_labels()
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        if handles:
            unique = dict(zip(labels, handles))
            fig.legend(unique.values(), unique.keys(), title="Class", loc="upper center", ncol=3, frameon=False)
        fig.suptitle("Connectivity distributions: subject-level mean edge value", y=1.02)
        fig.tight_layout()
        path = out_dir / "plot2_connectivity_boxplots_global.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    if level == "channel":
        # For readability, generate one figure per metric.
        paths = []
        for metric in pd.unique(df["metric"]):
            sub_metric = df[df["metric"] == metric].copy()
            g = sns.catplot(
                data=sub_metric,
                kind="box",
                x="channel",
                y="value",
                hue="class_label",
                col="band",
                col_wrap=3,
                palette=palette,
                height=4.0,
                aspect=1.5,
                sharey=True,
            )
            g.set_xticklabels(rotation=90, fontsize=7)
            g.set_axis_labels("Channel", "Mean channel connectivity strength")
            g.fig.suptitle(f"Channel-wise connectivity strength: {metric}", y=1.02)
            path = out_dir / f"plot2_connectivity_boxplots_channel_{metric}.png"
            g.fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(g.fig)
            paths.append(str(path))
        # Return first path to keep signature simple; all paths are saved.
        return paths[0]

    raise ValueError("level must be 'global' or 'channel'.")


# ---------------------------------------------------------------------
# Plot 3: Channel-wise feature bar plot
# ---------------------------------------------------------------------

def plot_channel_feature_bar(
    scalar_wide_df: pd.DataFrame,
    output_path: str | Path,
    *,
    feature_key: str,
    split: Optional[str] = None,
    subject_weighted: bool = True,
    palette: str = DEFAULT_PALETTE,
    dpi: int = 300,
) -> str:
    """
    Grouped bar chart by channel and class for one feature.

    Y-axis = mean feature value, X-axis = EEG channel, hue = class_label.
    Error bars are standard deviation across subjects when subject_weighted=True.
    """
    if feature_key not in scalar_wide_df.columns:
        raise KeyError(f"feature_key {feature_key!r} not found. Available example: {list(scalar_wide_df.columns[-10:])}")

    df = scalar_wide_df.copy()
    if split is not None:
        df = df[df["data_split"].map(normalize_split_name) == normalize_split_name(split)].copy()
    df[feature_key] = pd.to_numeric(df[feature_key], errors="coerce")
    df = df[np.isfinite(df[feature_key].to_numpy(dtype=float, na_value=np.nan))].copy()

    if subject_weighted:
        # Average windows per subject/channel first; then plot distribution across subjects.
        df = (
            df.groupby(["subject_id", "class_label", "data_split", "channel_idx", "channel"], as_index=False)[feature_key]
              .mean()
        )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(13, 5))
    ax = sns.barplot(
        data=df,
        x="channel",
        y=feature_key,
        hue="class_label",
        palette=palette,
        errorbar="sd",
    )
    ax.set_title(f"Channel-wise feature mean: {feature_key}" + (f" ({split})" if split else ""))
    ax.set_xlabel("EEG channel")
    ax.set_ylabel(feature_key)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Class", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(out_path)


# ---------------------------------------------------------------------
# Plot 3 extension: generate channel-wise bar plots for every scalar feature
# ---------------------------------------------------------------------

def _safe_filename(text: str, max_len: int = 180) -> str:
    """Convert an arbitrary feature key into a safe file-name stem."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    stem = stem.strip("._-") or "feature"
    return stem[:max_len]


def _resolve_plot3_feature_keys(
    scalar_wide_df: pd.DataFrame,
    feature_keys: Optional[Sequence[str]] = None,
) -> list[str]:
    """
    Resolve feature keys for Plot 3.

    If feature_keys is None, every scalar feature column using the family__feature
    naming convention is used. Metadata and derived subject/split columns are excluded.
    """
    if feature_keys is None:
        keys = [
            c for c in scalar_wide_df.columns
            if "__" in str(c)
            and not str(c).endswith("__mean")
            and not str(c).endswith("__std")
            and not str(c).endswith("__class_z")
        ]
    else:
        keys = [str(c) for c in feature_keys]

    missing = [c for c in keys if c not in scalar_wide_df.columns]
    if missing:
        raise KeyError(
            "Some Plot 3 feature keys were not found: "
            f"{missing[:10]}" + (" ..." if len(missing) > 10 else "")
        )

    # Keep only columns that have at least one finite numeric value.
    out: list[str] = []
    for col in keys:
        vals = pd.to_numeric(scalar_wide_df[col], errors="coerce")
        if np.isfinite(vals.to_numpy(dtype=float, na_value=np.nan)).any():
            out.append(col)
    return out


def _prepare_plot3_subject_channel_df(
    scalar_wide_df: pd.DataFrame,
    feature_keys: Sequence[str],
    *,
    split: Optional[str] = None,
    subject_weighted: bool = True,
) -> pd.DataFrame:
    """
    Prepare a compact dataframe for all Plot 3 figures.

    The raw window table can be very large. For fair channel-wise class plots,
    this function first averages all windows within each subject/channel. That
    prevents subjects with many windows from dominating the error bars.
    """
    meta_cols = ["subject_id", "class_label", "data_split", "channel_idx", "channel"]
    missing_meta = [c for c in meta_cols if c not in scalar_wide_df.columns]
    if missing_meta:
        raise KeyError(f"scalar_wide_df is missing required metadata columns: {missing_meta}")

    feature_keys = list(feature_keys)
    df = scalar_wide_df[meta_cols + feature_keys].copy()
    if split is not None:
        df = df[df["data_split"].map(normalize_split_name) == normalize_split_name(split)].copy()

    for col in feature_keys:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    if subject_weighted:
        df = (
            df.groupby(meta_cols, as_index=False, observed=True)[feature_keys]
              .mean()
        )
    return df


def plot_all_channel_feature_bars(
    scalar_wide_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_keys: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    subject_weighted: bool = True,
    palette: str = DEFAULT_PALETTE,
    dpi: int = 300,
) -> list[str]:
    """
    Generate Plot 3 as one grouped channel-wise bar plot for every scalar feature.

    Returns
    -------
    list[str]
        Saved paths, one PNG per feature.
    """
    feature_keys = _resolve_plot3_feature_keys(scalar_wide_df, feature_keys)
    if not feature_keys:
        return []

    out_dir = ensure_dir(Path(output_dir) / "plot3_channel_feature_bars_all")
    plot_df = _prepare_plot3_subject_channel_df(
        scalar_wide_df,
        feature_keys,
        split=split,
        subject_weighted=subject_weighted,
    )

    paths: list[str] = []
    for i, feature_key in enumerate(feature_keys, start=1):
        fdf = plot_df[["subject_id", "class_label", "data_split", "channel_idx", "channel", feature_key]].dropna(subset=[feature_key])
        if fdf.empty:
            continue

        out_path = out_dir / f"plot3_{i:03d}_{_safe_filename(feature_key)}.png"
        plt.figure(figsize=(13, 5))
        ax = sns.barplot(
            data=fdf,
            x="channel",
            y=feature_key,
            hue="class_label",
            palette=palette,
            errorbar="sd",
        )
        title = f"Channel-wise feature mean: {feature_key}"
        if split is not None:
            title += f" ({normalize_split_name(split)})"
        ax.set_title(title)
        ax.set_xlabel("EEG channel")
        ax.set_ylabel(feature_key)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Class", frameon=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        paths.append(str(out_path))

    # Save a small index so it is easy to map feature names to files.
    index_path = out_dir / "plot3_feature_file_index.csv"
    pd.DataFrame({
        "feature_key": feature_keys,
        "file": [str(out_dir / f"plot3_{i:03d}_{_safe_filename(k)}.png") for i, k in enumerate(feature_keys, start=1)],
    }).to_csv(index_path, index=False)
    return paths


def plot_channel_feature_grid_pages(
    scalar_wide_df: pd.DataFrame,
    output_dir: str | Path,
    *,
    feature_keys: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    subject_weighted: bool = True,
    features_per_page: int = 6,
    ncols: int = 2,
    palette: str = DEFAULT_PALETTE,
    dpi: int = 300,
) -> list[str]:
    """
    High-density Plot 3: multiple channel-wise feature bar plots per page.

    This is usually easier to scan than opening 30+ individual PNG files.
    """
    feature_keys = _resolve_plot3_feature_keys(scalar_wide_df, feature_keys)
    if not feature_keys:
        return []

    features_per_page = max(1, int(features_per_page))
    ncols = max(1, int(ncols))
    out_dir = ensure_dir(Path(output_dir) / "plot3_channel_feature_grid_pages")
    plot_df = _prepare_plot3_subject_channel_df(
        scalar_wide_df,
        feature_keys,
        split=split,
        subject_weighted=subject_weighted,
    )

    paths: list[str] = []
    for page_start in range(0, len(feature_keys), features_per_page):
        page_features = feature_keys[page_start:page_start + features_per_page]
        nrows = int(math.ceil(len(page_features) / ncols))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8.5 * ncols, 4.2 * nrows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        legend_handles = None
        legend_labels = None
        for ax, feature_key in zip(axes_flat, page_features):
            fdf = plot_df[["class_label", "channel", feature_key]].dropna(subset=[feature_key])
            if fdf.empty:
                ax.set_visible(False)
                continue
            sns.barplot(
                data=fdf,
                x="channel",
                y=feature_key,
                hue="class_label",
                palette=palette,
                errorbar="sd",
                ax=ax,
            )
            ax.set_title(feature_key, fontsize=10)
            ax.set_xlabel("")
            ax.set_ylabel("Mean ± SD")
            ax.tick_params(axis="x", rotation=60, labelsize=8)

            handles, labels = ax.get_legend_handles_labels()
            if handles and legend_handles is None:
                legend_handles, legend_labels = handles, labels
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()

        for ax in axes_flat[len(page_features):]:
            ax.set_visible(False)

        title = "Plot 3: Channel-wise feature means by class"
        if split is not None:
            title += f" ({normalize_split_name(split)})"
        fig.suptitle(title, fontsize=14, y=0.995)
        if legend_handles is not None:
            fig.legend(
                legend_handles,
                legend_labels,
                title="Class",
                loc="upper right",
                frameon=False,
                bbox_to_anchor=(0.995, 0.985),
            )
        fig.tight_layout(rect=[0, 0, 0.98, 0.965])
        page_id = page_start // features_per_page + 1
        out_path = out_dir / f"plot3_channel_feature_grid_page{page_id:02d}.png"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out_path))

    return paths


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------

def make_global_split_class_summary(
    scalar_wide_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """
    Table 1: group by [data_split, class_label] and calculate mean/std/min/max
    for all scalar feature columns.

    Returns long-format table:
        data_split, class_label, feature_key, n, mean, std, min, max
    """
    rows = []
    for (split, cls), g in scalar_wide_df.groupby(["data_split", "class_label"], dropna=False):
        for col in feature_cols:
            if col not in g.columns:
                continue
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            vals = vals[np.isfinite(vals)]
            rows.append({
                "data_split": split,
                "class_label": cls,
                "feature_key": col,
                "n": int(vals.shape[0]),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "min": float(vals.min()) if len(vals) else np.nan,
                "max": float(vals.max()) if len(vals) else np.nan,
            })
    return pd.DataFrame(rows)


def _resolve_existing_feature_columns(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    out = []
    for c in candidates:
        if c in df.columns and c not in out:
            out.append(c)
    return out


def make_subject_outlier_tracker(
    scalar_wide_df: pd.DataFrame,
    *,
    artifact_feature_candidates: Sequence[str] = (
        "statistical__ptp",
        "stats__ptp",
        "statistical__kurtosis",
        "stats__kurtosis",
    ),
    clinical_feature_candidates: Sequence[str] = (
        "derived__alpha_theta_ratio",
        "relative_band_power__rbp_alpha",
        "relative_band_power__rbp_theta",
        "relative_band_power__alpha",
        "relative_band_power__theta",
        "rbp__rbp_alpha",
        "rbp__rbp_theta",
        "rbp__alpha",
        "rbp__theta",
    ),
    z_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Table 2: subject-level artifact and clinical summary with outlier flag.

    A subject is flagged as Outlier if any available artifact feature mean has
    class-normalized z-score > z_threshold.
    """
    artifact_cols = _resolve_existing_feature_columns(scalar_wide_df, artifact_feature_candidates)
    clinical_cols = _resolve_existing_feature_columns(scalar_wide_df, clinical_feature_candidates)

    # Robust fallback for older runs where H5 feature-name attrs were malformed
    # or feature names use prefixes such as rbp_alpha/log_power_alpha.
    if not artifact_cols:
        artifact_cols = [
            c for c in scalar_wide_df.columns
            if "__" in c and re.search(r"(^|__)(ptp|kurtosis)$", sanitize_h5_name(c), flags=re.IGNORECASE)
        ]
    if not clinical_cols:
        clinical_cols = [c for c in scalar_wide_df.columns if c == "derived__alpha_theta_ratio"]
        clinical_cols += [
            c for c in scalar_wide_df.columns
            if "__" in c and re.search(r"(alpha|theta)", sanitize_h5_name(c), flags=re.IGNORECASE)
            and ("relative_band_power" in c or "rbp" in c or "log_band_power" in c)
        ]

    key_cols = artifact_cols + [c for c in clinical_cols if c not in artifact_cols]
    if not key_cols:
        raise ValueError(
            "No requested artifact/clinical feature columns were found in scalar_wide_df. "
            f"Available feature-like columns include: {[c for c in scalar_wide_df.columns if '__' in c][:20]}"
        )

    # Average and std per subject. Use named aggregation to avoid fragile
    # pandas MultiIndex column flattening across versions.
    agg_spec = {}
    for col in key_cols:
        agg_spec[f"{col}__mean"] = (col, "mean")
        agg_spec[f"{col}__std"] = (col, "std")

    subject_mean = (
        scalar_wide_df
        .groupby(["class_label", "subject_id", "data_split"], as_index=False)
        .agg(**agg_spec)
    )
    # Class-level distribution of subject means for artifact columns.
    out = subject_mean.copy()
    out["outlier_reasons"] = ""
    out["is_outlier"] = False

    for col in artifact_cols:
        mean_col = f"{col}__mean"
        if mean_col not in out.columns:
            continue
        class_stats = out.groupby("class_label")[mean_col].agg(["mean", "std"]).rename(columns={"mean": "class_mean", "std": "class_std"})
        out = out.merge(class_stats, left_on="class_label", right_index=True, how="left")
        z_col = f"{col}__class_z"
        out[z_col] = (out[mean_col] - out["class_mean"]) / (out["class_std"].replace(0, np.nan) + 1e-8)
        flag = out[z_col] > float(z_threshold)
        out.loc[flag, "is_outlier"] = True
        out.loc[flag, "outlier_reasons"] = out.loc[flag, "outlier_reasons"].astype(str) + f"{col} z>{z_threshold}; "
        out = out.drop(columns=["class_mean", "class_std"])

    # Put important columns first.
    first_cols = ["class_label", "subject_id", "data_split", "is_outlier", "outlier_reasons"]
    remaining = [c for c in out.columns if c not in first_cols]
    return out[first_cols + remaining].sort_values(["is_outlier", "class_label", "subject_id"], ascending=[False, True, True])


def export_summary_tables(
    output_dir: str | Path,
    *,
    global_summary: pd.DataFrame,
    subject_outliers: pd.DataFrame,
    prefix: str = "eda",
) -> dict[str, str]:
    """Export summary tables to CSV and one Excel workbook."""
    out_dir = ensure_dir(output_dir)
    paths = {
        "global_summary_csv": str(out_dir / f"{prefix}_table1_global_split_class_summary.csv"),
        "subject_outliers_csv": str(out_dir / f"{prefix}_table2_subject_outlier_tracker.csv"),
        "excel": str(out_dir / f"{prefix}_summary_tables.xlsx"),
    }
    global_summary.to_csv(paths["global_summary_csv"], index=False)
    subject_outliers.to_csv(paths["subject_outliers_csv"], index=False)
    with pd.ExcelWriter(paths["excel"], engine="openpyxl") as writer:
        global_summary.to_excel(writer, index=False, sheet_name="split_class_summary")
        subject_outliers.to_excel(writer, index=False, sheet_name="subject_outliers")
    return paths


# ---------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------

def run_caueeg_h5_eda(
    h5_path: str | Path,
    output_dir: str | Path,
    *,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    class_map: Optional[Mapping[int, str]] = None,
    split_csv: Optional[str | Path] = None,
    example_channel_feature: Optional[str] = None,
    plot3_features: Optional[Sequence[str]] = None,
    plot3_mode: str = "all",
    plot3_features_per_page: int = 6,
    max_subjects: Optional[int] = None,
    max_rows_for_plot1: int = 250_000,
    palette: str = DEFAULT_PALETTE,
) -> dict[str, Any]:
    """
    Run the full EDA pipeline.

    Returns a dictionary with saved paths and loaded dataframes for interactive use.
    """
    out_dir = ensure_dir(output_dir)
    fig_dir = ensure_dir(out_dir / "figures")
    table_dir = ensure_dir(out_dir / "tables")

    sns.set_theme(style="whitegrid", context="notebook", palette=palette)

    class_map = dict(class_map or DEFAULT_CLASS_MAP)
    split_lookup = read_split_lookup(split_csv)

    inventory = list_h5_inventory(h5_path)
    print("[Inventory]", json.dumps({k: v for k, v in inventory.items() if k != "subject_ids"}, indent=2))

    # feature_families = list(feature_families or inventory["feature_families"])
    connectivity_metrics = list(connectivity_metrics or inventory["connectivity_metrics"])

    # ----- scalar features -----
    scalar_wide_df, feature_cols = load_scalar_feature_wide_df(
        h5_path,
        feature_families=feature_families,
        class_map=class_map,
        split_lookup=split_lookup,
        max_subjects=max_subjects,
    )
    scalar_long_df = wide_to_long_features(
        scalar_wide_df,
        feature_cols,
        max_rows=max_rows_for_plot1,
    )

    plot1_paths = plot_global_feature_distributions(
        scalar_long_df,
        fig_dir,
        palette=palette,
    )

    ----- channel-wise feature bar plots -----
    plot3_mode:
      "example" -> one feature only, compatible with the old script
      "all"     -> one PNG per feature + high-density grid pages
      "both"    -> do both
    plot3_mode = str(plot3_mode).lower()
    if plot3_mode not in {"example", "all", "both", "none"}:
        raise ValueError("plot3_mode must be one of: 'none', 'example', 'all', 'both'.")

    if example_channel_feature is None:
        preferred = [
            "relative_band_power__rbp_beta",
            "relative_band_power__beta",
            "rbp__rbp_beta",
            "rbp__beta",
            "band_power__beta",
            "log_band_power__log_power_beta",
            feature_cols[0] if feature_cols else None,
        ]
        example_channel_feature = next((x for x in preferred if x and x in scalar_wide_df.columns), None)

    plot3_path = None
    plot3_paths: list[str] = []
    plot3_grid_paths: list[str] = []

    if plot3_mode in {"example", "both"} and example_channel_feature is not None:
        plot3_path = plot_channel_feature_bar(
            scalar_wide_df,
            fig_dir / f"plot3_channel_feature_bar_{example_channel_feature.replace('__','_')}.png",
            feature_key=example_channel_feature,
            palette=palette,
        )

    if plot3_mode in {"all", "both"}:
        selected_plot3_features = list(plot3_features or feature_cols)
        plot3_paths = plot_all_channel_feature_bars(
            scalar_wide_df,
            fig_dir,
            feature_keys=selected_plot3_features,
            palette=palette,
        )
        plot3_grid_paths = plot_channel_feature_grid_pages(
            scalar_wide_df,
            fig_dir,
            feature_keys=selected_plot3_features,
            features_per_page=plot3_features_per_page,
            palette=palette,
        )
        if plot3_path is None and plot3_paths:
            plot3_path = plot3_paths[0]

    # ----- summary tables -----
    table1 = make_global_split_class_summary(scalar_wide_df, feature_cols)
    table2 = make_subject_outlier_tracker(scalar_wide_df)
    table_paths = export_summary_tables(table_dir, global_summary=table1, subject_outliers=table2)

    # ----- connectivity plots -----
    connectivity_paths: dict[str, Any] = {}
    if connectivity_metrics:
        mean_conn, channel_names, class_order = compute_mean_connectivity_by_group(
            h5_path,
            metrics=connectivity_metrics,
            class_map=class_map,
            split_lookup=split_lookup,
            max_subjects=max_subjects,
        )
        heatmap_paths = []
        for metric in connectivity_metrics:
            if metric in mean_conn:
                heatmap_paths.append(plot_connectivity_grid(
                    mean_conn,
                    fig_dir,
                    metric=metric,
                    channel_names=channel_names,
                    class_order=class_order,
                ))
        conn_summary = connectivity_subject_summary_df(
            h5_path,
            metrics=connectivity_metrics,
            class_map=class_map,
            split_lookup=split_lookup,
            max_subjects=max_subjects,
        )
        conn_summary_path = table_dir / "connectivity_subject_summary.csv"
        conn_summary.to_csv(conn_summary_path, index=False)

        # New summary tables:
        # 1) subject-level global connectivity stats by split/class/metric/band
        # 2) channel-level connectivity stats pooled across all subjects and channels
        conn_global_stats = make_connectivity_split_class_summary(conn_summary, level="global")
        conn_channel_stats = make_connectivity_split_class_summary(conn_summary, level="channel")

        conn_global_stats_path = table_dir / "connectivity_global_split_class_summary.csv"
        conn_channel_stats_path = table_dir / "connectivity_channel_split_class_summary.csv"
        conn_stats_excel_path = table_dir / "connectivity_summary_tables.xlsx"

        conn_global_stats.to_csv(conn_global_stats_path, index=False)
        conn_channel_stats.to_csv(conn_channel_stats_path, index=False)
        with pd.ExcelWriter(conn_stats_excel_path, engine="openpyxl") as writer:
            conn_summary.to_excel(writer, index=False, sheet_name="subject_summary")
            conn_global_stats.to_excel(writer, index=False, sheet_name="global_split_class")
            conn_channel_stats.to_excel(writer, index=False, sheet_name="channel_split_class")

        conn_box_global = plot_connectivity_boxplots(conn_summary, fig_dir, level="global", palette=palette)
        conn_box_channel = plot_connectivity_boxplots(conn_summary, fig_dir, level="channel", palette=palette)
        # connectivity_paths = {
        #     "heatmap_paths": heatmap_paths,
        #     "summary_csv": str(conn_summary_path),
        #     "global_split_class_summary_csv": str(conn_global_stats_path),
        #     "channel_split_class_summary_csv": str(conn_channel_stats_path),
        #     "summary_excel": str(conn_stats_excel_path),
        #     "boxplot_global": conn_box_global,
        #     "boxplot_channel_first": conn_box_channel,
        # }

    # return {
    #     "inventory": inventory,
    #     "feature_cols": feature_cols,
    #     "scalar_wide_df": scalar_wide_df,
    #     "scalar_long_df": scalar_long_df,
    #     "plot1_paths": plot1_paths,
    #     "plot3_path": plot3_path,
    #     "plot3_paths": plot3_paths,
    #     "plot3_grid_paths": plot3_grid_paths,
    #     "table_paths": table_paths,
    #     "connectivity_paths": connectivity_paths,
    # }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_comma_list(x: Optional[str]) -> Optional[list[str]]:
    if x is None or str(x).strip() == "":
        return None
    return [s.strip() for s in str(x).split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="High-density EDA for CAUEEG master H5 feature/connectivity files.")
    # parser.add_argument("--h5_path", type=str, required=True, help="Path to master H5 file.")
    # parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--feature_families", type=str, default=None, help="Comma-separated feature families. Default: all in H5.")
    parser.add_argument("--connectivity_metrics", type=str, default=None, help="Comma-separated connectivity metrics. Default: all in H5.")
    parser.add_argument("--class_map", type=str, default=None, help="JSON string or file, e.g. '{\"0\":\"HC\",\"1\":\"MCI\",\"2\":\"AD\"}'.")
    parser.add_argument("--split_csv", type=str, default=None, help="Optional CSV with subject_id,data_split.")
    parser.add_argument("--example_channel_feature", type=str, default=None, help="Feature key for Plot 3 example mode, e.g. relative_band_power__rbp_beta.")
    parser.add_argument("--plot3_mode", type=str, default="all", choices=["none", "example", "all", "both"], help="Plot 3 generation mode. Default: all.")
    parser.add_argument("--plot3_features", type=str, default=None, help="Optional comma-separated feature keys for Plot 3. Default: all scalar features.")
    parser.add_argument("--plot3_features_per_page", type=int, default=6, help="Number of channel-wise feature plots per high-density Plot 3 page.")
    parser.add_argument("--max_subjects", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--max_rows_for_plot1", type=int, default=250000, help="Subsample rows before long-format plot1.")
    parser.add_argument("--palette", type=str, default=DEFAULT_PALETTE, help="Seaborn color palette.")
    args = parser.parse_args()
    import os
    # h5_path = "/home/anphan/Documents/caueeg_randomcrop_master_dementia_seed42.h5"
    h5_path = "/home/anphan/Documents/caueeg_merged_sliding_random_trainonly.h5"
    root_path = "/home/anphan/Documents/CAUEEG"
    output_dir = os.path.join(root_path,'visualize-merged_sliding_random')
    os.makedirs(output_dir,exist_ok = True)
    run_caueeg_h5_eda(
        h5_path=h5_path,
        output_dir=output_dir,
        feature_families=parse_comma_list(args.feature_families),
        connectivity_metrics=parse_comma_list(args.connectivity_metrics),
        class_map=load_class_map(args.class_map),
        split_csv=args.split_csv,
        example_channel_feature=args.example_channel_feature,
        plot3_features=parse_comma_list(args.plot3_features),
        plot3_mode=args.plot3_mode,
        plot3_features_per_page=args.plot3_features_per_page,
        max_subjects=args.max_subjects,
        max_rows_for_plot1=args.max_rows_for_plot1,
        palette=args.palette,
    )

    # print("\nSaved outputs:")
    # print(json.dumps({
    #     "plot1_paths": result["plot1_paths"],
    #     "plot3_path": result["plot3_path"],
    #     "plot3_num_individual_paths": len(result.get("plot3_paths", [])),
    #     "plot3_individual_dir": str(Path(args.output_dir) / "figures" / "plot3_channel_feature_bars_all"),
    #     "plot3_grid_paths": result.get("plot3_grid_paths", []),
    #     "table_paths": result["table_paths"],
    #     "connectivity_paths": result["connectivity_paths"],
    # }, indent=2))


if __name__ == "__main__":
    main()
