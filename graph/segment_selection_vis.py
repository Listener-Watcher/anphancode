from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, List

import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import joblib

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# Import from your existing source.
# Run this script from the same folder where mil_full_std.py is importable,
# or add graph/ to PYTHONPATH.
from mil_full_std import load_h5_payload_for_subjects


CAUEEG_EEG19 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4",
    "T6", "FZ", "CZ", "PZ",
]


# =========================================================
# Config
# =========================================================

@dataclass
class H5WindowSelectionConfig:
    h5_path: str
    output_dir: str

    # Feature families stored in your H5.
    statistical_family: str = "statistical"
    hjorth_family: str = "hjorth"
    spectral_families: Tuple[str, ...] = ("relative_band_power", "wavelet_energy")

    # Connectivity.
    connectivity_metric: str = "wpli"
    alpha_band_index: int = 2  # delta=0, theta=1, alpha=2, beta=3, gamma=4

    # Feature indices inside statistical feature tensor [W, C, F].
    # Based on your stated order:
    # mean, max, min, skew, kurtosis, ptp
    statistical_kurtosis_idx: int = 4
    statistical_ptp_idx: int = 5

    # Hjorth tensor usually: activity, mobility, complexity.
    hjorth_complexity_idx: int = 2

    # Selection.
    n_select: int = 10
    anomaly_contamination: float = 0.20
    pca_components: int = 5
    n_clusters: int = 10
    seed: int = 42

    # Plotting.
    embedding_method: str = "pca"  # "pca" or "tsne"
    max_subject_plots: Optional[int] = 20
    dpi: int = 200

    # CleanCluster manifest rule.
    artifact_cluster_anomaly_fraction_threshold: float = 1.0
    artifact_cluster_min_size: int = 1

    # IForest feature choice.
    use_abs_kurtosis_for_iforest: bool = True
    use_hjorth_for_iforest: bool = True
    # Connectivity noise scoring.
    use_connectivity_noise_score: bool = True
    connectivity_noise_weight: float = 0.50
    node_noise_weight: float = 0.50

    # Weighted sampling.
    sampling_temperature: float = 1.0
    min_sampling_weight: float = 0.05
# =========================================================
# H5 inspection
# =========================================================

def list_h5_subject_ids(h5_path: str) -> List[str]:
    with h5py.File(h5_path, "r") as h5f:
        return list(h5f["subjects"].keys())


def inspect_h5_structure(h5_path: str, max_subjects: int = 2) -> None:
    """
    Print the H5 structure for a few subjects.
    This is useful before running the full analysis.
    """
    with h5py.File(h5_path, "r") as h5f:
        print("\nTop-level keys:", list(h5f.keys()))
        subject_ids = list(h5f["subjects"].keys())
        print(f"Number of subjects: {len(subject_ids)}")

        for sid in subject_ids[:max_subjects]:
            if sid in {"train_00587", "train_00781", "train_01301"}:
                continue
            print("\n" + "=" * 80)
            print("Subject:", sid)
            grp = h5f[f"subjects/{sid}"]

            print("metadata attrs:", dict(grp["metadata"].attrs))
            if "channel_names" in grp["metadata"]:
                ch = grp["metadata/channel_names"][:]
                ch = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ch]
                print("channel_names:", ch)

            if "windows" in grp:
                print("windows keys:", list(grp["windows"].keys()))

            if "features" in grp["windows"]:
                print("feature families:", list(grp["windows/features"].keys()))
                for fam in grp["windows/features"].keys():
                    print(f"  {fam}: shape={grp['windows/features'][fam].shape}")

            if "connectivity" in grp["windows"]:
                print("connectivity metrics:", list(grp["windows/connectivity"].keys()))
                for metric in grp["windows/connectivity"].keys():
                    print(f"  {metric}: shape={grp['windows/connectivity'][metric].shape}")


# =========================================================
# Small numeric helpers
# =========================================================
def robust_zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Robust z-score using median and MAD.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if mad < eps:
        q1, q3 = np.nanpercentile(x, [25, 75])
        iqr = q3 - q1
        scale = iqr / 1.349 if iqr > eps else np.nanstd(x)
    else:
        scale = 1.4826 * mad

    if scale < eps:
        return np.zeros_like(x, dtype=np.float64)

    return (x - med) / scale


def positive_robust_zscore(x: np.ndarray) -> np.ndarray:
    """
    Negative z-scores are not noisy.
    Positive z-scores mean larger-than-normal value within subject.
    """
    z = robust_zscore(x)
    return np.maximum(z, 0.0)

def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return StandardScaler().fit_transform(X)


def _clean_connectivity_stack(A: np.ndarray) -> np.ndarray:
    """
    A: [W, N, N]
    """
    A = np.asarray(A, dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + np.transpose(A, (0, 2, 1)))

    for w in range(A.shape[0]):
        np.fill_diagonal(A[w], 0.0)

    return A


def _safe_random_choice(
    candidate_idx: np.ndarray,
    n_select: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidate_idx = np.asarray(candidate_idx, dtype=int)
    if len(candidate_idx) == 0:
        return np.array([], dtype=int)
    size = min(n_select, len(candidate_idx))
    return rng.choice(candidate_idx, size=size, replace=False)


def _flatten_window_features(x: np.ndarray) -> np.ndarray:
    """
    Convert [W, C, F] to [W, C*F].
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected feature tensor [W, C, F], got {x.shape}")
    return x.reshape(x.shape[0], -1)


def _select_band_connectivity(conn: np.ndarray, band_index: int) -> np.ndarray:
    """
    Handles:
      [W, B, N, N] -> [W, N, N]
      [W, N, N]    -> [W, N, N]
    """
    conn = np.asarray(conn, dtype=np.float32)

    if conn.ndim == 4:
        if not (0 <= band_index < conn.shape[1]):
            raise ValueError(
                f"band_index={band_index} is invalid for connectivity shape {conn.shape}"
            )
        return conn[:, band_index]

    if conn.ndim == 3:
        return conn

    raise ValueError(f"Connectivity must be [W,B,N,N] or [W,N,N], got {conn.shape}")

def compute_upper_triangle_mean(A: np.ndarray) -> np.ndarray:
    """
    A: [W, N, N]
    return: [W], mean upper-triangle connectivity strength per segment.
    """
    A = np.asarray(A, dtype=np.float32)
    if A.ndim != 3:
        raise ValueError(f"Expected A with shape [W,N,N], got {A.shape}")

    n = A.shape[-1]
    iu = np.triu_indices(n, k=1)
    return A[:, iu[0], iu[1]].mean(axis=1)


def build_iforest_artifact_features(
    ptp_score: np.ndarray,
    kurtosis_score: np.ndarray,
    hjorth_complexity: np.ndarray,
    *,
    use_abs_kurtosis: bool = True,
    use_hjorth: bool = True,
) -> np.ndarray:
    cols = [np.asarray(ptp_score, dtype=np.float64)]

    if use_abs_kurtosis:
        cols.append(np.abs(np.asarray(kurtosis_score, dtype=np.float64)))
    else:
        cols.append(np.asarray(kurtosis_score, dtype=np.float64))

    if use_hjorth:
        cols.append(np.asarray(hjorth_complexity, dtype=np.float64))

    return np.column_stack(cols)

def compute_connectivity_noise_features(state: SubjectWindowState) -> pd.DataFrame:
    """
    Compute segment-level connectivity quality/noise features from alpha-wPLI.

    These are not disease labels. They are only suspiciousness scores.
    """
    A = np.asarray(state.alpha_wpli, dtype=np.float32)  # [W, N, N]

    if A.ndim != 3:
        raise ValueError(f"state.alpha_wpli must be [W,N,N], got {A.shape}")

    W, N, _ = A.shape
    iu = np.triu_indices(N, k=1)

    edges = A[:, iu[0], iu[1]]  # [W, E]
    edges = np.nan_to_num(edges, nan=0.0, posinf=0.0, neginf=0.0)

    # Basic edge distribution features.
    conn_mean = edges.mean(axis=1)
    conn_std = edges.std(axis=1)
    conn_abs_mean = np.abs(edges).mean(axis=1)
    conn_abs_max = np.abs(edges).max(axis=1)

    # Fraction of very large edges within each segment.
    # Threshold is subject-level 95th percentile over all segment edges.
    edge_thr = np.nanpercentile(np.abs(edges), 95)
    conn_high_edge_fraction = (np.abs(edges) >= edge_thr).mean(axis=1)

    # Distance to subject median connectivity.
    median_edges = np.nanmedian(edges, axis=0, keepdims=True)
    conn_median_distance = np.linalg.norm(edges - median_edges, axis=1) / np.sqrt(edges.shape[1])

    # Temporal jump: how different this segment is from previous/next segment.
    # If windows are ordered by start_sample, big jumps may indicate transient artifacts.
    order = np.argsort(state.start_sample)
    temporal_jump = np.zeros(W, dtype=np.float64)

    ordered_edges = edges[order]
    if W >= 2:
        diffs = np.linalg.norm(np.diff(ordered_edges, axis=0), axis=1) / np.sqrt(edges.shape[1])

        # assign jump to both neighboring segments
        temporal_jump_ordered = np.zeros(W, dtype=np.float64)
        temporal_jump_ordered[1:] += diffs
        temporal_jump_ordered[:-1] += diffs
        counts = np.ones(W, dtype=np.float64)
        counts[1:-1] = 2.0
        temporal_jump_ordered = temporal_jump_ordered / counts

        temporal_jump[order] = temporal_jump_ordered

    return pd.DataFrame({
        "segment_index": np.arange(W, dtype=int),
        "conn_mean": conn_mean,
        "conn_std": conn_std,
        "conn_abs_mean": conn_abs_mean,
        "conn_abs_max": conn_abs_max,
        "conn_high_edge_fraction": conn_high_edge_fraction,
        "conn_median_distance": conn_median_distance,
        "conn_temporal_jump": temporal_jump,
    })

def compute_segment_noise_scores(
    state: SubjectWindowState,
    cfg: H5WindowSelectionConfig,
) -> pd.DataFrame:
    """
    Build node noise score, connectivity noise score, combined score,
    and final sampling weight for each segment.
    """
    W = len(state.segment_id)

    # -------------------------
    # Node noise
    # -------------------------
    node_ptp_z = positive_robust_zscore(state.ptp_score)
    node_abs_kurt_z = positive_robust_zscore(np.abs(state.kurtosis_score))
    node_hjorth_z = positive_robust_zscore(state.hjorth_complexity)

    node_noise_score = np.nanmean(
        np.column_stack([
            node_ptp_z,
            node_abs_kurt_z,
            node_hjorth_z,
        ]),
        axis=1,
    )

    # -------------------------
    # Connectivity noise
    # -------------------------
    conn_df = compute_connectivity_noise_features(state)

    conn_noise_cols = [
        "conn_std",
        "conn_abs_max",
        "conn_high_edge_fraction",
        "conn_median_distance",
        "conn_temporal_jump",
    ]

    conn_z_list = []
    for col in conn_noise_cols:
        conn_z_list.append(positive_robust_zscore(conn_df[col].to_numpy()))

    conn_noise_score = np.nanmean(np.column_stack(conn_z_list), axis=1)

    # -------------------------
    # Combined score
    # -------------------------
    node_w = float(cfg.node_noise_weight)
    conn_w = float(cfg.connectivity_noise_weight)

    denom = max(node_w + conn_w, 1e-8)

    if not cfg.use_connectivity_noise_score:
        combined_noise_score = node_noise_score
    else:
        combined_noise_score = (
            node_w * node_noise_score + conn_w * conn_noise_score
        ) / denom

    # -------------------------
    # Convert noise score to sampling weight
    # -------------------------
    temp = max(float(cfg.sampling_temperature), 1e-6)

    sampling_weight = np.exp(-combined_noise_score / temp)
    sampling_weight = np.clip(
        sampling_weight,
        float(cfg.min_sampling_weight),
        1.0,
    )

    out = pd.DataFrame({
        "segment_index": np.arange(W, dtype=int),

        "node_ptp_z": node_ptp_z,
        "node_abs_kurtosis_z": node_abs_kurt_z,
        "node_hjorth_complexity_z": node_hjorth_z,
        "node_noise_score": node_noise_score,

        "connectivity_noise_score": conn_noise_score,
        "combined_noise_score": combined_noise_score,
        "sampling_weight": sampling_weight,
    })

    out = out.merge(conn_df, on="segment_index", how="left")

    return out
# =========================================================
# Subject state
# =========================================================

@dataclass
class SubjectWindowState:
    subject_id: str
    label: int
    segment_id: np.ndarray
    start_sample: np.ndarray
    channel_names: List[str]

    ptp_score: np.ndarray
    kurtosis_score: np.ndarray
    hjorth_complexity: np.ndarray

    spectral_X: np.ndarray
    spectral_pca_5d: np.ndarray
    embedding_2d: np.ndarray

    kmeans_labels: np.ndarray
    iforest_is_clean: np.ndarray
    iforest_score: np.ndarray

    alpha_wpli: np.ndarray  # [W, N, N]


def build_subject_state_from_payload_entry(
    sid: str,
    entry: Dict[str, Any],
    cfg: H5WindowSelectionConfig,
) -> SubjectWindowState:
    """
    Convert one subject payload entry into arrays needed by the three selection methods.
    """
    features = entry["features"]
    connectivity = entry["connectivity"]

    if cfg.statistical_family not in features:
        raise KeyError(f"{sid}: missing feature family {cfg.statistical_family!r}")

    if cfg.hjorth_family not in features:
        raise KeyError(f"{sid}: missing feature family {cfg.hjorth_family!r}")

    if cfg.connectivity_metric not in connectivity:
        raise KeyError(f"{sid}: missing connectivity metric {cfg.connectivity_metric!r}")

    stat = np.asarray(features[cfg.statistical_family], dtype=np.float32)  # [W, C, F_stat]
    hjorth = np.asarray(features[cfg.hjorth_family], dtype=np.float32)     # [W, C, F_hjorth]

    if stat.ndim != 3:
        raise ValueError(f"{sid}: statistical must be [W,C,F], got {stat.shape}")
    if hjorth.ndim != 3:
        raise ValueError(f"{sid}: hjorth must be [W,C,F], got {hjorth.shape}")

    if cfg.statistical_kurtosis_idx >= stat.shape[-1]:
        raise ValueError(
            f"{sid}: kurtosis index {cfg.statistical_kurtosis_idx} "
            f"out of range for statistical shape {stat.shape}"
        )

    if cfg.statistical_ptp_idx >= stat.shape[-1]:
        raise ValueError(
            f"{sid}: ptp index {cfg.statistical_ptp_idx} "
            f"out of range for statistical shape {stat.shape}"
        )

    if cfg.hjorth_complexity_idx >= hjorth.shape[-1]:
        raise ValueError(
            f"{sid}: hjorth complexity index {cfg.hjorth_complexity_idx} "
            f"out of range for hjorth shape {hjorth.shape}"
        )

    # Window-level artifact scores: average across channels.
    kurtosis_score = stat[:, :, cfg.statistical_kurtosis_idx].mean(axis=1)
    ptp_score = stat[:, :, cfg.statistical_ptp_idx].mean(axis=1)
    hjorth_complexity = hjorth[:, :, cfg.hjorth_complexity_idx].mean(axis=1)

    # Spectral features for diversity.
    spectral_blocks = []
    used_spectral_families = []

    for fam in cfg.spectral_families:
        if fam in features:
            spectral_blocks.append(_flatten_window_features(features[fam]))
            used_spectral_families.append(fam)

    if len(spectral_blocks) == 0:
        raise KeyError(
            f"{sid}: none of spectral_families={cfg.spectral_families} exist in H5. "
            "Use relative_band_power or rebuild H5 with wavelet_energy."
        )

    if "wavelet_energy" not in used_spectral_families:
        warnings.warn(
            f"{sid}: wavelet_energy not found. Diversity uses {used_spectral_families}. "
            "If you want RBP + wavelet energy, rebuild the H5 with wavelet_energy."
        )

    spectral_X = np.concatenate(spectral_blocks, axis=1)
    spectral_Xz = _standardize(spectral_X)

    n_windows = spectral_X.shape[0]

    # PCA to 5D for KMeans.
    pca_dim = min(cfg.pca_components, n_windows, spectral_Xz.shape[1])
    pca_model = PCA(n_components=pca_dim, random_state=cfg.seed)
    spectral_pca = pca_model.fit_transform(spectral_Xz)

    if pca_dim < cfg.pca_components:
        pad = np.zeros((n_windows, cfg.pca_components - pca_dim), dtype=np.float32)
        spectral_pca_5d = np.concatenate([spectral_pca, pad], axis=1)
    else:
        spectral_pca_5d = spectral_pca

    # 2D embedding for plot.
    if cfg.embedding_method.lower() == "tsne" and n_windows >= 5:
        perplexity = min(30, max(2, (n_windows - 1) // 3))
        embedding_2d = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=cfg.seed,
        ).fit_transform(spectral_pca_5d)
    else:
        emb_dim = min(2, n_windows, spectral_Xz.shape[1])
        emb = PCA(n_components=emb_dim, random_state=cfg.seed).fit_transform(spectral_Xz)
        if emb.shape[1] == 1:
            embedding_2d = np.column_stack([emb[:, 0], np.zeros(n_windows)])
        else:
            embedding_2d = emb

    # KMeans clusters.
    k = min(cfg.n_clusters, n_windows)
    kmeans_labels = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=cfg.seed,
    ).fit_predict(spectral_pca_5d)

    # Isolation Forest.
    # artifact_X = np.column_stack([ptp_score, kurtosis_score])

    # Isolation Forest.
    artifact_X = build_iforest_artifact_features(
        ptp_score=ptp_score,
        kurtosis_score=kurtosis_score,
        hjorth_complexity=hjorth_complexity,
        use_abs_kurtosis=cfg.use_abs_kurtosis_for_iforest,
        use_hjorth=cfg.use_hjorth_for_iforest,
    )
    # artifact_Xz = _standardize(artifact_X)


    # artifact_Xz = _standardize(artifact_X)

    # iforest = IsolationForest(
    #     n_estimators=200,
    #     contamination=cfg.anomaly_contamination,
    #     random_state=cfg.seed,
    # )
    # iforest_pred = iforest.fit_predict(artifact_Xz)
    # iforest_is_clean = iforest_pred == 1

    # # Higher score = more normal, lower score = more anomalous.
    # iforest_score = iforest.score_samples(artifact_Xz)

    iforest_is_clean, iforest_score, severity_threshold = fit_iforest_adaptive_anomaly_labels(
        artifact_X=artifact_X,
        seed=cfg.seed,
        tau=3.5,
    )


    alpha_conn = _select_band_connectivity(
        connectivity[cfg.connectivity_metric],
        band_index=cfg.alpha_band_index,
    )
    alpha_conn = _clean_connectivity_stack(alpha_conn)

    return SubjectWindowState(
        subject_id=str(sid),
        label=int(entry["label"]),
        segment_id=np.asarray(entry["segment_id"], dtype=np.int64),
        start_sample=np.asarray(entry["start_sample"], dtype=np.int64),
        channel_names=list(entry.get("channel_names", CAUEEG_EEG19)),

        ptp_score=ptp_score,
        kurtosis_score=kurtosis_score,
        hjorth_complexity=hjorth_complexity,

        spectral_X=spectral_X,
        spectral_pca_5d=spectral_pca_5d,
        embedding_2d=embedding_2d,

        kmeans_labels=kmeans_labels,
        iforest_is_clean=iforest_is_clean,
        iforest_score=iforest_score,

        alpha_wpli=alpha_conn,
    )


# =========================================================
# Selection methods
# =========================================================



def fit_global_segment_clusterer(
    X_train_segments,
    *,
    n_clusters=8,
    pca_dim=8,
    seed=42,
    save_path="global_segment_clusterer.joblib",
):
    """
    X_train_segments: [num_train_segments, feature_dim]
    """

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_train_segments)

    pca_dim = min(pca_dim, Xz.shape[0], Xz.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xp = pca.fit_transform(Xz)

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    )
    labels = kmeans.fit_predict(Xp)

    model = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "pca_dim": pca_dim,
        "seed": seed,
    }

    joblib.dump(model, save_path)

    return model, labels

def robust_mad_threshold(x: np.ndarray, tau: float = 3.5) -> float:
    """
    Robust threshold: median + tau * MAD.
    Use for artifact severity where higher = noisier.
    """
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))

    if mad < 1e-8:
        # Fallback if all values are almost identical.
        q1, q3 = np.percentile(x, [25, 75])
        iqr = q3 - q1
        return q3 + 1.5 * iqr

    # 1.4826 makes MAD comparable to std under normality.
    return med + tau * 1.4826 * mad

def select_baseline_random(state: SubjectWindowState, cfg: H5WindowSelectionConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    all_idx = np.arange(len(state.segment_id))
    return _safe_random_choice(all_idx, cfg.n_select, rng)


def select_clean_iforest(state: SubjectWindowState, cfg: H5WindowSelectionConfig) -> np.ndarray:
    """
    Randomly select from non-artifact windows.
    If fewer than n_select clean windows remain, fill with least-anomalous remaining windows.
    """
    rng = np.random.default_rng(cfg.seed)

    clean_idx = np.where(state.iforest_is_clean)[0]
    selected = list(_safe_random_choice(clean_idx, cfg.n_select, rng))

    if len(selected) < cfg.n_select:
        selected_set = set(selected)
        remaining = np.array([i for i in range(len(state.segment_id)) if i not in selected_set])

        # Higher score = cleaner.
        remaining_sorted = remaining[np.argsort(state.iforest_score[remaining])[::-1]]
        need = cfg.n_select - len(selected)
        selected.extend(remaining_sorted[:need].tolist())

    return np.asarray(selected, dtype=int)


def fit_iforest_adaptive_anomaly_labels(
    artifact_X: np.ndarray,
    seed: int = 42,
    tau: float = 3.5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns:
        iforest_is_clean: bool [W]
        iforest_score: raw sklearn score, higher = cleaner
        severity_threshold: adaptive threshold, higher severity = noisier
    """
    artifact_Xz = _standardize(artifact_X)

    iforest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=seed,
    )

    iforest.fit(artifact_Xz)

    # sklearn score: higher = cleaner
    iforest_score = iforest.score_samples(artifact_Xz)

    # convert to severity: higher = noisier
    artifact_severity = -iforest_score

    severity_threshold = robust_mad_threshold(artifact_severity, tau=tau)

    iforest_is_clean = artifact_severity <= severity_threshold

    return iforest_is_clean, iforest_score, severity_threshold

def select_diversity_kmeans(state: SubjectWindowState, cfg: H5WindowSelectionConfig) -> np.ndarray:
    """
    Pick one representative window per cluster.
    Representative = nearest to cluster centroid in PCA space.
    """
    rng = np.random.default_rng(cfg.seed)

    X = state.spectral_pca_5d
    labels = state.kmeans_labels
    unique_clusters = np.unique(labels)

    selected = []

    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        Xc = X[idx]
        centroid = Xc.mean(axis=0, keepdims=True)
        dist = np.linalg.norm(Xc - centroid, axis=1)
        chosen = idx[np.argmin(dist)]
        selected.append(int(chosen))

    selected = selected[:cfg.n_select]

    if len(selected) < cfg.n_select:
        selected_set = set(selected)
        remaining = np.array([i for i in range(len(state.segment_id)) if i not in selected_set])
        fill = _safe_random_choice(remaining, cfg.n_select - len(selected), rng)
        selected.extend(fill.tolist())

    return np.asarray(selected, dtype=int)


def run_selection_methods(
    state: SubjectWindowState,
    cfg: H5WindowSelectionConfig,
) -> Dict[str, np.ndarray]:
    return {
        "Baseline_Random": select_baseline_random(state, cfg),
        "Clean_IForest": select_clean_iforest(state, cfg),
        "Diversity_KMeans": select_diversity_kmeans(state, cfg),
    }


# =========================================================
# Summaries
# =========================================================

def summarize_subject(
    state: SubjectWindowState,
    selections: Dict[str, np.ndarray],
    cfg: H5WindowSelectionConfig,
) -> pd.DataFrame:
    rows = []

    max_clusters = min(cfg.n_clusters, cfg.n_select)

    for method_name, idx in selections.items():
        idx = np.asarray(idx, dtype=int)
        represented_clusters = np.unique(state.kmeans_labels[idx])
        cluster_coverage = 100.0 * len(represented_clusters) / max_clusters

        rows.append({
            "subject_id": state.subject_id,
            "label": state.label,
            "Method_Name": method_name,
            "Num_Selected": int(len(idx)),
            "Selected_Segment_IDs": ",".join(map(str, state.segment_id[idx].tolist())),
            "Selected_Start_Samples": ",".join(map(str, state.start_sample[idx].tolist())),
            "Mean_Selected_PTP": float(np.mean(state.ptp_score[idx])),
            "Mean_Selected_Kurtosis": float(np.mean(state.kurtosis_score[idx])),
            "Cluster_Coverage_Percentage": float(cluster_coverage),
            "Mean_Hjorth_Complexity": float(np.mean(state.hjorth_complexity[idx])),
        })

    return pd.DataFrame(rows)


def summarize_global(per_subject_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "Mean_Selected_PTP",
        "Mean_Selected_Kurtosis",
        "Cluster_Coverage_Percentage",
        "Mean_Hjorth_Complexity",
    ]

    return (
        per_subject_df
        .groupby("Method_Name", as_index=False)[metric_cols]
        .mean()
        .sort_values("Method_Name")
        .reset_index(drop=True)
    )


def add_cleancluster_flags_to_cluster_df(
    cluster_df: pd.DataFrame,
    *,
    anomaly_fraction_threshold: float = 1.0,
    min_cluster_size: int = 1,
) -> pd.DataFrame:
    """
    Mark artifact-only clusters.

    Default rule:
        artifact_cluster = iforest_anomaly_fraction >= 1.0

    This is intentionally conservative: remove only clusters where all segments
    are IForest anomalies.
    """
    df = cluster_df.copy()

    df["artifact_cluster"] = (
        (df["iforest_anomaly_fraction"] >= float(anomaly_fraction_threshold))
        & (df["cluster_size"] >= int(min_cluster_size))
    )

    df["clean_cluster"] = ~df["artifact_cluster"]

    return df


def build_subject_cleancluster_manifest(
    state: SubjectWindowState,
    cluster_df: pd.DataFrame,
    cfg: H5WindowSelectionConfig,
) -> pd.DataFrame:
    """
    Build one row per segment/window.

    This is the main reusable manifest for MIL filtering.
    """
    cluster_df = add_cleancluster_flags_to_cluster_df(
        cluster_df,
        anomaly_fraction_threshold=cfg.artifact_cluster_anomaly_fraction_threshold,
        min_cluster_size=cfg.artifact_cluster_min_size,
    )

    cluster_info = cluster_df.set_index("cluster_id").to_dict(orient="index")

    alpha_wpli_strength = compute_upper_triangle_mean(state.alpha_wpli)

    rows = []
    noise_df = compute_segment_noise_scores(state, cfg)
    
    noise_info = noise_df.set_index("segment_index").to_dict(orient="index")
    
    for local_idx in range(len(state.segment_id)):
        cluster_id = int(state.kmeans_labels[local_idx])
        info = cluster_info[cluster_id]

        iforest_is_anomaly = bool(not state.iforest_is_clean[local_idx])
        artifact_cluster = bool(info["artifact_cluster"])
        keep_clean = not artifact_cluster

        rows.append({
            "subject_id": state.subject_id,
            "split": infer_split_from_subject_id(state.subject_id),
            "label": int(state.label),

            # local row index inside payload arrays; this is important for filtering later
            "segment_index": int(local_idx),

            # metadata from H5
            "segment_id": int(state.segment_id[local_idx]),
            "start_sample": int(state.start_sample[local_idx]),

            # KMeans info
            "kmeans_cluster_id": cluster_id,
            "cluster_size": int(info["cluster_size"]),
            "cluster_fraction": float(info["cluster_fraction"]),
            "cluster_anomaly_fraction": float(info["iforest_anomaly_fraction"]),
            "artifact_cluster": artifact_cluster,
            "clean_cluster": bool(info["clean_cluster"]),

            # IForest info
            "iforest_is_anomaly": iforest_is_anomaly,
            "iforest_is_clean": bool(state.iforest_is_clean[local_idx]),
            "iforest_score": float(state.iforest_score[local_idx]),

            # segment-level quality features
            "ptp": float(state.ptp_score[local_idx]),
            "kurtosis": float(state.kurtosis_score[local_idx]),
            "abs_kurtosis": float(abs(state.kurtosis_score[local_idx])),
            "hjorth_complexity": float(state.hjorth_complexity[local_idx]),
            "alpha_wpli_strength": float(alpha_wpli_strength[local_idx]),

            # final reusable flag
            "keep_clean": bool(keep_clean),

            # noise scores for weighted sampling
            "node_noise_score": float(noise_info[local_idx]["node_noise_score"]),
            "connectivity_noise_score": float(noise_info[local_idx]["connectivity_noise_score"]),
            "combined_noise_score": float(noise_info[local_idx]["combined_noise_score"]),
            "sampling_weight": float(noise_info[local_idx]["sampling_weight"]),

            # connectivity diagnostics
            "conn_mean": float(noise_info[local_idx]["conn_mean"]),
            "conn_std": float(noise_info[local_idx]["conn_std"]),
            "conn_abs_mean": float(noise_info[local_idx]["conn_abs_mean"]),
            "conn_abs_max": float(noise_info[local_idx]["conn_abs_max"]),
            "conn_high_edge_fraction": float(noise_info[local_idx]["conn_high_edge_fraction"]),
            "conn_median_distance": float(noise_info[local_idx]["conn_median_distance"]),
            "conn_temporal_jump": float(noise_info[local_idx]["conn_temporal_jump"]),
        })

    return pd.DataFrame(rows)


def summarize_cleancluster_subjects(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per subject summarizing how much was removed.
    """
    rows = []

    for sid, g in manifest_df.groupby("subject_id"):
        if sid in {"train_00587", "train_00781", "train_01301"}:
            continue
        keep = g["keep_clean"].to_numpy(dtype=bool)
        artifact = g["artifact_cluster"].to_numpy(dtype=bool)

        rows.append({
            "subject_id": sid,
            "split": g["split"].iloc[0],
            "label": int(g["label"].iloc[0]),

            "num_segments_total": int(len(g)),
            "num_segments_clean": int(keep.sum()),
            "num_segments_removed": int((~keep).sum()),
            "removed_fraction": float((~keep).mean()),

            "num_clusters_total": int(g["kmeans_cluster_id"].nunique()),
            "num_artifact_clusters": int(g.loc[g["artifact_cluster"], "kmeans_cluster_id"].nunique()),

            "mean_ptp_all": float(g["ptp"].mean()),
            "mean_ptp_clean": float(g.loc[g["keep_clean"], "ptp"].mean()) if keep.any() else np.nan,
            "mean_ptp_removed": float(g.loc[~g["keep_clean"], "ptp"].mean()) if (~keep).any() else np.nan,

            "mean_abs_kurtosis_all": float(g["abs_kurtosis"].mean()),
            "mean_abs_kurtosis_clean": float(g.loc[g["keep_clean"], "abs_kurtosis"].mean()) if keep.any() else np.nan,
            "mean_abs_kurtosis_removed": float(g.loc[~g["keep_clean"], "abs_kurtosis"].mean()) if (~keep).any() else np.nan,

            "mean_hjorth_complexity_all": float(g["hjorth_complexity"].mean()),
            "mean_hjorth_complexity_clean": float(g.loc[g["keep_clean"], "hjorth_complexity"].mean()) if keep.any() else np.nan,
            "mean_hjorth_complexity_removed": float(g.loc[~g["keep_clean"], "hjorth_complexity"].mean()) if (~keep).any() else np.nan,
        })

    return pd.DataFrame(rows)


def summarize_cleancluster_by_class(subject_summary_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "num_segments_total",
        "num_segments_clean",
        "num_segments_removed",
        "removed_fraction",
        "num_clusters_total",
        "num_artifact_clusters",
        "mean_ptp_all",
        "mean_ptp_clean",
        "mean_ptp_removed",
        "mean_abs_kurtosis_all",
        "mean_abs_kurtosis_clean",
        "mean_abs_kurtosis_removed",
        "mean_hjorth_complexity_all",
        "mean_hjorth_complexity_clean",
        "mean_hjorth_complexity_removed",
    ]

    return (
        subject_summary_df
        .groupby(["split", "label"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["split", "label"])
        .reset_index(drop=True)
    )


def save_cleancluster_metadata(
    cfg: H5WindowSelectionConfig,
    save_path: str | os.PathLike,
) -> str:
    """
    Save the rule/config that generated the manifest.
    """
    import json
    from dataclasses import asdict

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    meta = asdict(cfg)
    meta["note"] = (
        "CleanCluster manifest. keep_clean=False means the segment belongs to a "
        "KMeans cluster whose IForest anomaly fraction exceeds the configured threshold."
    )

    with open(save_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(save_path)


def build_cleancluster_manifest(
    cfg: H5WindowSelectionConfig,
    subject_ids: Optional[Sequence[str]] = None,
    *,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the CleanCluster manifest once.

    Output files:
        cleancluster_manifest.csv
        cleancluster_subject_summary.csv
        cleancluster_class_summary.csv
        cleancluster_cluster_diagnostics.csv
        cleancluster_config.json

    Returns
    -------
    manifest_df:
        One row per segment.
    subject_summary_df:
        One row per subject.
    class_summary_df:
        One row per split x class.
    cluster_df_all:
        One row per subject x KMeans cluster.
    """
    output_dir = Path(cfg.output_dir)
    clean_dir = output_dir / "cleancluster"
    clean_dir.mkdir(parents=True, exist_ok=True)

    if subject_ids is None:
        subject_ids = list_h5_subject_ids(cfg.h5_path)

    requested_families = set()
    requested_families.add(cfg.statistical_family)
    requested_families.add(cfg.hjorth_family)
    requested_families.update(cfg.spectral_families)

    payload = load_h5_payload_for_subjects(
        h5_path=cfg.h5_path,
        subject_ids=subject_ids,
        feature_families=sorted(requested_families),
        connectivity_metrics=[cfg.connectivity_metric],
        connectivity_band=None,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    all_manifest_rows = []
    all_cluster_dfs = []
    failed_subjects = []

    for sid, entry in payload.items():
        if sid in {"train_00587", "train_00781", "train_01301"}:
            continue
        try:
            state = build_subject_state_from_payload_entry(sid, entry, cfg)

            cluster_df = compute_subject_cluster_diagnostics(state, cfg)
            cluster_df = add_cleancluster_flags_to_cluster_df(
                cluster_df,
                anomaly_fraction_threshold=cfg.artifact_cluster_anomaly_fraction_threshold,
                min_cluster_size=cfg.artifact_cluster_min_size,
            )

            manifest_subj = build_subject_cleancluster_manifest(
                state=state,
                cluster_df=cluster_df,
                cfg=cfg,
            )

            all_cluster_dfs.append(cluster_df)
            all_manifest_rows.append(manifest_subj)

        except Exception as e:
            failed_subjects.append({"subject_id": sid, "error": str(e)})
            print(f"[WARN] CleanCluster skip subject={sid}: {e}")

    if len(all_manifest_rows) == 0:
        raise RuntimeError("No subjects were successfully processed for CleanCluster manifest.")

    manifest_df = pd.concat(all_manifest_rows, ignore_index=True)
    cluster_df_all = pd.concat(all_cluster_dfs, ignore_index=True)

    subject_summary_df = summarize_cleancluster_subjects(manifest_df)
    class_summary_df = summarize_cleancluster_by_class(subject_summary_df)

    if save_outputs:
        manifest_df.to_csv(clean_dir / "cleancluster_manifest.csv", index=False)
        subject_summary_df.to_csv(clean_dir / "cleancluster_subject_summary.csv", index=False)
        class_summary_df.to_csv(clean_dir / "cleancluster_class_summary.csv", index=False)
        cluster_df_all.to_csv(clean_dir / "cleancluster_cluster_diagnostics.csv", index=False)

        if len(failed_subjects) > 0:
            pd.DataFrame(failed_subjects).to_csv(clean_dir / "cleancluster_failed_subjects.csv", index=False)

        save_cleancluster_metadata(cfg, clean_dir / "cleancluster_config.json")

        print(f"\nSaved CleanCluster manifest to: {clean_dir / 'cleancluster_manifest.csv'}")
        print(f"Saved subject summary to: {clean_dir / 'cleancluster_subject_summary.csv'}")
        print(f"Saved class summary to: {clean_dir / 'cleancluster_class_summary.csv'}")
        print(f"Saved cluster diagnostics to: {clean_dir / 'cleancluster_cluster_diagnostics.csv'}")

    return manifest_df, subject_summary_df, class_summary_df, cluster_df_all


def validate_cleancluster_manifest(
    manifest_df: pd.DataFrame,
    *,
    min_clean_segments_warning: int = 10,
) -> pd.DataFrame:
    """
    Print and return subjects with too few clean segments.
    """
    subject_summary = summarize_cleancluster_subjects(manifest_df)

    bad = subject_summary[
        subject_summary["num_segments_clean"] < int(min_clean_segments_warning)
    ].copy()

    print("\nCleanCluster manifest validation")
    print("Total subjects:", subject_summary["subject_id"].nunique())
    print("Total segments:", len(manifest_df))
    print("Kept clean segments:", int(manifest_df["keep_clean"].sum()))
    print("Removed segments:", int((~manifest_df["keep_clean"]).sum()))
    print("Mean removed fraction:", float(subject_summary["removed_fraction"].mean()))

    if len(bad) > 0:
        print(f"\n[WARN] {len(bad)} subjects have fewer than {min_clean_segments_warning} clean segments.")
        print(bad[["subject_id", "label", "num_segments_total", "num_segments_clean", "removed_fraction"]].head(20))

    return bad
# =========================================================
# Plotting
# =========================================================

def plot_feature_distributions(
    state: SubjectWindowState,
    selections: Dict[str, np.ndarray],
    axes,
) -> None:
    marker_map = {
        "Baseline_Random": "X",
        "Clean_IForest": "D",
        "Diversity_KMeans": "*",
    }

    y_offsets = {
        "Baseline_Random": -0.005,
        "Clean_IForest": -0.010,
        "Diversity_KMeans": -0.015,
    }

    ax = axes[0]
    sns.histplot(state.ptp_score, kde=True, bins=30, stat="density", ax=ax, color="lightgray")
    ax.set_title("PTP distribution")
    ax.set_xlabel("Mean PTP across channels")

    for name, idx in selections.items():
        ax.scatter(
            state.ptp_score[idx],
            np.full(len(idx), y_offsets[name]),
            marker=marker_map[name],
            s=120 if name == "Diversity_KMeans" else 80,
            edgecolor="black",
            linewidth=0.8,
            label=name,
            zorder=5,
        )
    ax.legend(fontsize=8)

    ax = axes[1]
    sns.histplot(state.kurtosis_score, kde=True, bins=30, stat="density", ax=ax, color="lightgray")
    ax.set_title("Kurtosis distribution")
    ax.set_xlabel("Mean kurtosis across channels")

    for name, idx in selections.items():
        ax.scatter(
            state.kurtosis_score[idx],
            np.full(len(idx), y_offsets[name]),
            marker=marker_map[name],
            s=120 if name == "Diversity_KMeans" else 80,
            edgecolor="black",
            linewidth=0.8,
            label=name,
            zorder=5,
        )
    ax.legend(fontsize=8)


def plot_embedding(
    state: SubjectWindowState,
    selections: Dict[str, np.ndarray],
    ax,
    cfg: H5WindowSelectionConfig,
) -> None:
    emb = state.embedding_2d

    sns.scatterplot(
        x=emb[:, 0],
        y=emb[:, 1],
        hue=state.kmeans_labels,
        palette="tab10",
        s=50,
        ax=ax,
        legend="brief",
    )

    marker_map = {
        "Baseline_Random": "X",
        "Clean_IForest": "D",
        "Diversity_KMeans": "*",
    }

    for name, idx in selections.items():
        ax.scatter(
            emb[idx, 0],
            emb[idx, 1],
            marker=marker_map[name],
            s=250 if name == "Diversity_KMeans" else 130,
            edgecolor="black",
            linewidth=1.1,
            label=f"{name} selected",
            zorder=10,
        )

    ax.set_title(f"{cfg.embedding_method.upper()} spectral embedding")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)


def plot_connectivity_heatmaps(
    state: SubjectWindowState,
    selections: Dict[str, np.ndarray],
    axes,
) -> None:
    mean_mats = {
        name: state.alpha_wpli[idx].mean(axis=0)
        for name, idx in selections.items()
    }

    all_vals = np.concatenate([m.reshape(-1) for m in mean_mats.values()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    for ax, (name, mat) in zip(axes, mean_mats.items()):
        sns.heatmap(
            mat,
            ax=ax,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            square=True,
            cbar=True,
            xticklabels=state.channel_names,
            yticklabels=state.channel_names,
        )
        ax.set_title(f"Mean alpha wPLI\n{name}")
        ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)


def plot_subject_dashboard(
    state: SubjectWindowState,
    selections: Dict[str, np.ndarray],
    cfg: H5WindowSelectionConfig,
    save_path: str | os.PathLike,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_context("talk")

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.0, 1.2, 1.3])

    ax_ptp = fig.add_subplot(gs[0, 0])
    ax_kurt = fig.add_subplot(gs[0, 1])
    ax_emb = fig.add_subplot(gs[0:2, 2])

    plot_feature_distributions(state, selections, axes=[ax_ptp, ax_kurt])
    plot_embedding(state, selections, ax_emb, cfg)

    heat_axes = [
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
    ]
    plot_connectivity_heatmaps(state, selections, heat_axes)

    fig.suptitle(
        f"Window-selection comparison | subject={state.subject_id} | label={state.label}",
        fontsize=18,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_global_summary(global_df: pd.DataFrame, save_path: str | os.PathLike, dpi: int = 200) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = global_df.melt(
        id_vars="Method_Name",
        value_vars=[
            "Mean_Selected_PTP",
            "Mean_Selected_Kurtosis",
            "Cluster_Coverage_Percentage",
            "Mean_Hjorth_Complexity",
        ],
        var_name="Metric",
        value_name="Value",
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(data=plot_df, x="Metric", y="Value", hue="Method_Name", ax=ax)
    ax.set_title("Global comparison of window-selection strategies")
    ax.set_xlabel("")
    ax.set_ylabel("Mean over subjects")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)

def filter_subject_ids_by_split(subject_ids, split: str):
    split = str(split).lower()

    if split in {"all", "none"}:
        return list(subject_ids)

    prefix_map = {
        "train": "train_",
        "val": "val_",
        "validation": "val_",
        "test": "test_",
    }

    if split not in prefix_map:
        raise ValueError(
            f"Unknown split={split!r}. Use one of: all, train, val, validation, test."
        )

    prefix = prefix_map[split]
    out = [sid for sid in subject_ids if str(sid).startswith(prefix) and sid not in {"train_00587", "train_00781", "train_01301"}]

    if len(out) == 0:
        raise ValueError(
            f"No subject IDs found with prefix {prefix!r}. "
            "Check your H5 subject IDs."
        )

    return out


def infer_split_from_subject_id(subject_id: str) -> str:
    sid = str(subject_id)
    if sid.startswith("train_"):
        return "train"
    if sid.startswith("val_"):
        return "val"
    if sid.startswith("test_"):
        return "test"
    return "unknown"


def compute_subject_cluster_diagnostics(
    state: SubjectWindowState,
    cfg: H5WindowSelectionConfig,
) -> pd.DataFrame:
    rows = []

    labels = state.kmeans_labels
    unique_clusters = np.unique(labels)

    # alpha wPLI edge strength per segment
    # Use upper triangle mean to avoid double counting.
    n_nodes = state.alpha_wpli.shape[-1]
    iu = np.triu_indices(n_nodes, k=1)
    alpha_wpli_strength = state.alpha_wpli[:, iu[0], iu[1]].mean(axis=1)

    abs_kurtosis = np.abs(state.kurtosis_score)

    for c in unique_clusters:
        idx = np.where(labels == c)[0]

        rows.append({
            "subject_id": state.subject_id,
            "label": state.label,
            "cluster_id": int(c),
            "cluster_size": int(len(idx)),
            "cluster_fraction": float(len(idx) / len(labels)),

            "mean_ptp": float(np.mean(state.ptp_score[idx])),
            "median_ptp": float(np.median(state.ptp_score[idx])),

            "mean_kurtosis": float(np.mean(state.kurtosis_score[idx])),
            "mean_abs_kurtosis": float(np.mean(abs_kurtosis[idx])),

            "mean_hjorth_complexity": float(np.mean(state.hjorth_complexity[idx])),

            # Higher score = cleaner.
            "mean_iforest_score": float(np.mean(state.iforest_score[idx])),

            # Fraction of this cluster marked anomalous by IForest.
            "iforest_anomaly_fraction": float(np.mean(~state.iforest_is_clean[idx])),

            "mean_alpha_wpli_strength": float(np.mean(alpha_wpli_strength[idx])),

            "segment_ids": ",".join(map(str, state.segment_id[idx].tolist())),
            "start_samples": ",".join(map(str, state.start_sample[idx].tolist())),
        })

    return pd.DataFrame(rows)

def plot_kmeans_cluster_diagnostics(
    state: SubjectWindowState,
    cluster_df: pd.DataFrame,
    save_path: str | os.PathLike,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    emb = state.embedding_2d
    labels = state.kmeans_labels

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. PCA scatter colored by KMeans cluster
    ax = axes[0, 0]
    sns.scatterplot(
        x=emb[:, 0],
        y=emb[:, 1],
        hue=labels,
        palette="tab10",
        s=70,
        ax=ax,
        legend="brief",
    )
    ax.set_title("KMeans clusters in spectral PCA space")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")

    # 2. PCA scatter colored by PTP
    ax = axes[0, 1]
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=state.ptp_score,
        s=70,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
    )
    fig.colorbar(sc, ax=ax)
    ax.set_title("PTP score over spectral PCA")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")

    # 3. PCA scatter colored by abs kurtosis
    ax = axes[0, 2]
    abs_kurtosis = np.abs(state.kurtosis_score)
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=abs_kurtosis,
        s=70,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
    )
    fig.colorbar(sc, ax=ax)
    ax.set_title("Absolute kurtosis over spectral PCA")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")

    # 4. PCA scatter colored by IForest score
    ax = axes[1, 0]
    sc = ax.scatter(
        emb[:, 0],
        emb[:, 1],
        c=state.iforest_score,
        s=70,
        cmap="viridis",
        edgecolor="black",
        linewidth=0.3,
    )
    fig.colorbar(sc, ax=ax)
    ax.set_title("Isolation Forest score\nhigher = cleaner")
    ax.set_xlabel("PCA dim 1")
    ax.set_ylabel("PCA dim 2")

    # 5. Cluster-level artifact fraction
    ax = axes[1, 1]
    sns.barplot(
        data=cluster_df,
        x="cluster_id",
        y="iforest_anomaly_fraction",
        ax=ax,
    )
    ax.set_title("IForest anomaly fraction by KMeans cluster")
    ax.set_xlabel("KMeans cluster")
    ax.set_ylabel("Anomaly fraction")

    # 6. Cluster size
    ax = axes[1, 2]
    sns.barplot(
        data=cluster_df,
        x="cluster_id",
        y="cluster_size",
        ax=ax,
    )
    ax.set_title("Cluster size")
    ax.set_xlabel("KMeans cluster")
    ax.set_ylabel("Number of segments")

    fig.suptitle(
        f"KMeans cluster diagnostics | subject={state.subject_id} | label={state.label}",
        fontsize=18,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)

def plot_iforest_diagnostics(
    state: SubjectWindowState,
    save_path: str | os.PathLike,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    emb = state.embedding_2d
    is_anomaly = ~state.iforest_is_clean

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. PCA scatter clean vs anomaly
    ax = axes[0, 0]
    plot_df = pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "IForest_Label": np.where(is_anomaly, "Anomaly", "Clean"),
        "PTP": state.ptp_score,
        "Kurtosis": state.kurtosis_score,
        "Abs_Kurtosis": np.abs(state.kurtosis_score),
        "Hjorth_Complexity": state.hjorth_complexity,
        "IForest_Score": state.iforest_score,
    })

    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue="IForest_Label",
        style="IForest_Label",
        s=80,
        ax=ax,
    )
    ax.set_title("IForest clean vs anomaly\nin spectral PCA space")

    # 2. PTP boxplot
    ax = axes[0, 1]
    sns.boxplot(data=plot_df, x="IForest_Label", y="PTP", ax=ax)
    sns.stripplot(data=plot_df, x="IForest_Label", y="PTP", ax=ax, color="black", alpha=0.4)
    ax.set_title("PTP: clean vs anomaly")

    # 3. Abs kurtosis boxplot
    ax = axes[0, 2]
    sns.boxplot(data=plot_df, x="IForest_Label", y="Abs_Kurtosis", ax=ax)
    sns.stripplot(data=plot_df, x="IForest_Label", y="Abs_Kurtosis", ax=ax, color="black", alpha=0.4)
    ax.set_title("Abs kurtosis: clean vs anomaly")

    # 4. Hjorth complexity boxplot
    ax = axes[1, 0]
    sns.boxplot(data=plot_df, x="IForest_Label", y="Hjorth_Complexity", ax=ax)
    sns.stripplot(data=plot_df, x="IForest_Label", y="Hjorth_Complexity", ax=ax, color="black", alpha=0.4)
    ax.set_title("Hjorth complexity: clean vs anomaly")

    # 5. Mean alpha wPLI clean
    ax = axes[1, 1]
    if np.any(~is_anomaly):
        mean_clean = state.alpha_wpli[~is_anomaly].mean(axis=0)
    else:
        mean_clean = np.zeros_like(state.alpha_wpli[0])

    sns.heatmap(
        mean_clean,
        ax=ax,
        cmap="viridis",
        square=True,
        cbar=True,
        xticklabels=state.channel_names,
        yticklabels=state.channel_names,
    )
    ax.set_title("Mean alpha wPLI: clean")

    # 6. Mean alpha wPLI anomaly
    ax = axes[1, 2]
    if np.any(is_anomaly):
        mean_anom = state.alpha_wpli[is_anomaly].mean(axis=0)
    else:
        mean_anom = np.zeros_like(state.alpha_wpli[0])

    sns.heatmap(
        mean_anom,
        ax=ax,
        cmap="viridis",
        square=True,
        cbar=True,
        xticklabels=state.channel_names,
        yticklabels=state.channel_names,
    )
    ax.set_title("Mean alpha wPLI: anomaly")

    fig.suptitle(
        f"Isolation Forest diagnostics | subject={state.subject_id} | label={state.label}",
        fontsize=18,
        y=1.02,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def compute_subject_iforest_summary(state: SubjectWindowState) -> Dict[str, float]:
    is_anomaly = ~state.iforest_is_clean

    return {
        "subject_id": state.subject_id,
        "label": int(state.label),
        "num_segments": int(len(state.segment_id)),
        "num_iforest_anomaly": int(np.sum(is_anomaly)),
        "iforest_anomaly_fraction": float(np.mean(is_anomaly)),

        "mean_ptp_all": float(np.mean(state.ptp_score)),
        "mean_ptp_clean": float(np.mean(state.ptp_score[~is_anomaly])) if np.any(~is_anomaly) else np.nan,
        "mean_ptp_anomaly": float(np.mean(state.ptp_score[is_anomaly])) if np.any(is_anomaly) else np.nan,

        "mean_abs_kurtosis_all": float(np.mean(np.abs(state.kurtosis_score))),
        "mean_abs_kurtosis_clean": float(np.mean(np.abs(state.kurtosis_score[~is_anomaly]))) if np.any(~is_anomaly) else np.nan,
        "mean_abs_kurtosis_anomaly": float(np.mean(np.abs(state.kurtosis_score[is_anomaly]))) if np.any(is_anomaly) else np.nan,

        "mean_hjorth_complexity_all": float(np.mean(state.hjorth_complexity)),
        "mean_hjorth_complexity_clean": float(np.mean(state.hjorth_complexity[~is_anomaly])) if np.any(~is_anomaly) else np.nan,
        "mean_hjorth_complexity_anomaly": float(np.mean(state.hjorth_complexity[is_anomaly])) if np.any(is_anomaly) else np.nan,
    }

def summarize_iforest_by_class(iforest_subject_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "iforest_anomaly_fraction",
        "mean_ptp_all",
        "mean_ptp_clean",
        "mean_ptp_anomaly",
        "mean_abs_kurtosis_all",
        "mean_abs_kurtosis_clean",
        "mean_abs_kurtosis_anomaly",
        "mean_hjorth_complexity_all",
        "mean_hjorth_complexity_clean",
        "mean_hjorth_complexity_anomaly",
    ]

    return (
        iforest_subject_df
        .groupby("label", as_index=False)[metric_cols]
        .mean()
        .sort_values("label")
        .reset_index(drop=True)
    )

def plot_iforest_class_summary(
    iforest_subject_df: pd.DataFrame,
    save_path: str | os.PathLike,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.boxplot(
        data=iforest_subject_df,
        x="label",
        y="iforest_anomaly_fraction",
        ax=axes[0],
    )
    sns.stripplot(
        data=iforest_subject_df,
        x="label",
        y="iforest_anomaly_fraction",
        ax=axes[0],
        color="black",
        alpha=0.4,
    )
    axes[0].set_title("IForest anomaly fraction by class")

    sns.boxplot(
        data=iforest_subject_df,
        x="label",
        y="mean_ptp_anomaly",
        ax=axes[1],
    )
    sns.stripplot(
        data=iforest_subject_df,
        x="label",
        y="mean_ptp_anomaly",
        ax=axes[1],
        color="black",
        alpha=0.4,
    )
    axes[1].set_title("PTP of anomalous segments by class")

    sns.boxplot(
        data=iforest_subject_df,
        x="label",
        y="mean_hjorth_complexity_anomaly",
        ax=axes[2],
    )
    sns.stripplot(
        data=iforest_subject_df,
        x="label",
        y="mean_hjorth_complexity_anomaly",
        ax=axes[2],
        color="black",
        alpha=0.4,
    )
    axes[2].set_title("Hjorth complexity of anomalous segments by class")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)

def flag_artifact_like_clusters(
    cluster_df: pd.DataFrame,
    anomaly_fraction_threshold: float = 0.60,
    high_ptp_quantile: float = 0.75,
    high_abs_kurtosis_quantile: float = 0.75,
) -> pd.DataFrame:
    df = cluster_df.copy()

    ptp_thr = df["mean_ptp"].quantile(high_ptp_quantile)
    kurt_thr = df["mean_abs_kurtosis"].quantile(high_abs_kurtosis_quantile)

    df["artifact_like_cluster"] = (
        (df["iforest_anomaly_fraction"] >= anomaly_fraction_threshold)
        | (
            (df["mean_ptp"] >= ptp_thr)
            & (df["mean_abs_kurtosis"] >= kurt_thr)
        )
    )

    return df

def summarize_artifact_clusters_by_class(cluster_df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (label, subject_id), g in cluster_df_all.groupby(["label", "subject_id"]):
        total_segments = g["cluster_size"].sum()
        artifact_segments = g.loc[g["artifact_like_cluster"], "cluster_size"].sum()

        rows.append({
            "label": int(label),
            "subject_id": subject_id,
            "num_clusters": int(len(g)),
            "num_artifact_like_clusters": int(g["artifact_like_cluster"].sum()),
            "artifact_like_cluster_fraction": float(g["artifact_like_cluster"].mean()),
            "artifact_segment_fraction": float(artifact_segments / total_segments),
        })

    subject_level = pd.DataFrame(rows)

    class_level = (
        subject_level
        .groupby("label", as_index=False)[
            [
                "artifact_like_cluster_fraction",
                "artifact_segment_fraction",
            ]
        ]
        .mean()
    )

    return subject_level, class_level

# =========================================================
# Main runner
# =========================================================

def analyze_h5_window_selection(
    cfg: H5WindowSelectionConfig,
    subject_ids: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_cluster_dfs = []
    all_iforest_rows = []

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "class_level").mkdir(parents=True, exist_ok=True)
    (output_dir / "subject_dashboards").mkdir(parents=True, exist_ok=True)
    (output_dir / "cluster_analysis").mkdir(parents=True, exist_ok=True)

    if subject_ids is None:
        subject_ids = list_h5_subject_ids(cfg.h5_path)

    # Load all feature families that may be needed.
    requested_families = set()
    print("cfg.statistical_family", cfg.statistical_family)
    print("cfg.hjorth_family", cfg.hjorth_family)
    print("cfg.spectral_families", cfg.spectral_families)
    requested_families.add(cfg.statistical_family)
    requested_families.add(cfg.hjorth_family)
    requested_families.update(cfg.spectral_families)

    # Important:
    # connectivity_band=None because we want to handle both [W,B,N,N] and [W,N,N] ourselves.
    payload = load_h5_payload_for_subjects(
        h5_path=cfg.h5_path,
        subject_ids=subject_ids,
        feature_families=sorted(requested_families),
        connectivity_metrics=[cfg.connectivity_metric],
        connectivity_band=None,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    all_subject_rows = []
    plotted = 0

    for sid, entry in payload.items():
        if sid in {"train_00587", "train_00781", "train_01301"}:
            continue
        try:
            state = build_subject_state_from_payload_entry(sid, entry, cfg)
            selections = run_selection_methods(state, cfg)
            cluster_df = compute_subject_cluster_diagnostics(state, cfg)
            cluster_df = flag_artifact_like_clusters(cluster_df)

            all_cluster_dfs.append(cluster_df)
            all_iforest_rows.append(compute_subject_iforest_summary(state))

            # plot_kmeans_cluster_diagnostics(
            #     state,
            #     cluster_df,
            #     output_dir / "cluster_analysis" / f"{sid}_kmeans_cluster_diagnostics.png",
            # )

            # plot_iforest_diagnostics(
            #     state,
            #     output_dir / "cluster_analysis" / f"{sid}_iforest_diagnostics.png",
            # )
            subject_df = summarize_subject(state, selections, cfg)
            all_subject_rows.append(subject_df)

            if cfg.max_subject_plots is None or plotted < cfg.max_subject_plots:
                fig_path = output_dir / "subject_dashboards" / f"{sid}_window_selection.png"
                plot_subject_dashboard(state, selections, cfg, fig_path)
                plotted += 1

        except Exception as e:
            print(f"[WARN] Skip subject={sid}: {e}")
    
    if len(all_subject_rows) == 0:
        raise RuntimeError("No subjects were successfully analyzed.")

    cluster_df_all = pd.concat(all_cluster_dfs, ignore_index=True)
    iforest_subject_df = pd.DataFrame(all_iforest_rows)

    cluster_df_all.to_csv(
        output_dir / "class_level" / "subject_kmeans_cluster_diagnostics.csv",
        index=False,
    )

    iforest_subject_df.to_csv(
        output_dir / "class_level" / "subject_iforest_summary.csv",
        index=False,
    )

    iforest_class_df = summarize_iforest_by_class(iforest_subject_df)
    iforest_class_df.to_csv(
        output_dir / "class_level" / "class_level_iforest_summary.csv",
        index=False,
    )

    plot_iforest_class_summary(
        iforest_subject_df,
        output_dir / "class_level" / "class_level_iforest_boxplots.png",
    )

    artifact_subject_df, artifact_class_df = summarize_artifact_clusters_by_class(cluster_df_all)

    artifact_subject_df.to_csv(
        output_dir / "class_level" / "subject_artifact_cluster_summary.csv",
        index=False,
    )

    artifact_class_df.to_csv(
        output_dir / "class_level" / "class_level_artifact_cluster_summary.csv",
        index=False,
    )


    per_subject_df = pd.concat(all_subject_rows, ignore_index=True)
    global_df = summarize_global(per_subject_df)

    per_subject_path = output_dir / "window_selection_per_subject_summary.csv"
    global_path = output_dir / "window_selection_global_summary.csv"

    per_subject_df.to_csv(per_subject_path, index=False)
    global_df.to_csv(global_path, index=False)

    plot_global_summary(
        global_df,
        output_dir / "window_selection_global_summary.png",
        dpi=cfg.dpi,
    )

    print(f"\nSaved per-subject summary to: {per_subject_path}")
    print(f"Saved global summary to: {global_path}")
    print(f"Saved dashboard plots to: {output_dir / 'subject_dashboards'}")

    return per_subject_df, global_df


# =========================================================
# CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # parser.add_argument("--h5_path", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, default="./window_selection_analysis_h5")

    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--alpha_band_index", type=int, default=2)

    parser.add_argument("--n_select", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=10)
    # parser.add_argument("--anomaly_contamination", type=float, default=0.20)

    parser.add_argument("--embedding_method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--max_subject_plots", type=int, default=20)

    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--max_inspect_subjects", type=int, default=2)

    # Optional subject subset.
    parser.add_argument(
        "--subject_ids",
        type=str,
        default=None,
        help="Comma-separated subject IDs. If omitted, analyze all subjects in H5.",
    )

    # Use this if your statistical order differs.
    parser.add_argument("--stat_kurtosis_idx", type=int, default=4)
    parser.add_argument("--stat_ptp_idx", type=int, default=5)
    parser.add_argument("--hjorth_complexity_idx", type=int, default=2)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["all", "train", "val", "test"],
        help="Which H5 split to analyze based on subject ID prefix.",
    )

    parser.add_argument(
        "--build_cleancluster_manifest",
        action="store_true",
        help="Build CleanCluster manifest before/without visualization.",
    )

    parser.add_argument(
        "--artifact_cluster_anomaly_fraction_threshold",
        type=float,
        default=1.0,
        help="Remove KMeans clusters whose IForest anomaly fraction is at least this value.",
    )

    parser.add_argument(
        "--artifact_cluster_min_size",
        type=int,
        default=1,
        help="Minimum cluster size for artifact-cluster removal.",
    )

    parser.add_argument(
        "--manifest_only",
        action="store_true",
        help="Only build CleanCluster manifest and skip dashboard visualization.",
    )


    return parser.parse_args()

def load_cleancluster_manifest_for_visualization(manifest_path: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    required = {
        "subject_id",
        "label",
        "segment_id",
        "keep_clean",
        "sampling_weight",
        "ptp",
        "abs_kurtosis",
        "hjorth_complexity",
        "combined_noise_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest is missing required columns: {missing}")

    if df["keep_clean"].dtype != bool:
        df["keep_clean"] = (
            df["keep_clean"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    df["subject_id"] = df["subject_id"].astype(str)
    return df


def summarize_sampling_weights_by_subject(
    manifest_df: pd.DataFrame,
    *,
    clean_only: bool = True,
) -> pd.DataFrame:
    df = manifest_df.copy()
    if clean_only:
        df = df[df["keep_clean"]].copy()

    rows = []
    for sid, g in df.groupby("subject_id"):
        if sid in {"train_00587", "train_00781", "train_01301"}:
            continue
        w = g["sampling_weight"].to_numpy(dtype=float)

        rows.append({
            "subject_id": sid,
            "split": g["split"].iloc[0] if "split" in g.columns else "unknown",
            "label": int(g["label"].iloc[0]),
            "num_segments": int(len(g)),
            "weight_min": float(np.min(w)),
            "weight_q25": float(np.percentile(w, 25)),
            "weight_median": float(np.median(w)),
            "weight_mean": float(np.mean(w)),
            "weight_q75": float(np.percentile(w, 75)),
            "weight_max": float(np.max(w)),
            "weight_std": float(np.std(w)),
            "effective_sample_size": float((w.sum() ** 2) / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def summarize_top_weight_segments(
    manifest_df: pd.DataFrame,
    *,
    top_k: int = 10,
    clean_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        top_segments_df:
            One row per top-weight segment.
        top_subject_summary_df:
            One row per subject summarizing the top-weight segments.
    """
    df = manifest_df.copy()
    if clean_only:
        df = df[df["keep_clean"]].copy()

    top_rows = []

    for sid, g in df.groupby("subject_id"):
        if sid in {"train_00587", "train_00781", "train_01301"}:
            continue
        g_top = (
            g.sort_values("sampling_weight", ascending=False)
             .head(top_k)
             .copy()
        )
        g_top["top_rank"] = np.arange(1, len(g_top) + 1)
        top_rows.append(g_top)

    if len(top_rows) == 0:
        raise RuntimeError("No top-weight segments found.")

    top_segments_df = pd.concat(top_rows, ignore_index=True)

    metric_cols = [
        "sampling_weight",
        "combined_noise_score",
        "node_noise_score",
        "connectivity_noise_score",
        "ptp",
        "abs_kurtosis",
        "hjorth_complexity",
    ]

    optional_cols = [
        "conn_std",
        "conn_abs_max",
        "conn_high_edge_fraction",
        "conn_median_distance",
        "conn_temporal_jump",
        "alpha_wpli_strength",
    ]
    metric_cols += [c for c in optional_cols if c in top_segments_df.columns]

    top_subject_summary_df = (
        top_segments_df
        .groupby(["subject_id", "label"], as_index=False)[metric_cols]
        .agg(["mean", "std", "min", "max"])
    )

    top_subject_summary_df.columns = [
        "_".join([x for x in col if x])
        for col in top_subject_summary_df.columns.to_flat_index()
    ]
    top_subject_summary_df = top_subject_summary_df.reset_index()

    return top_segments_df, top_subject_summary_df

def plot_sampling_weight_distribution_by_subject(
    manifest_df: pd.DataFrame,
    save_path: str | os.PathLike,
    *,
    clean_only: bool = True,
    max_subjects: int | None = 30,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = manifest_df.copy()
    if clean_only:
        df = df[df["keep_clean"]].copy()

    subject_ids = sorted(df["subject_id"].unique())
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    plot_df = df[df["subject_id"].isin(subject_ids)].copy()

    fig, ax = plt.subplots(figsize=(max(14, len(subject_ids) * 0.45), 6))

    sns.boxplot(
        data=plot_df,
        x="subject_id",
        y="sampling_weight",
        hue="label",
        dodge=False,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x="subject_id",
        y="sampling_weight",
        color="black",
        alpha=0.35,
        size=3,
        ax=ax,
    )

    ax.set_title("Sampling weight distribution by subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Sampling weight")
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_weight_vs_quality_features(
    manifest_df: pd.DataFrame,
    save_path: str | os.PathLike,
    *,
    clean_only: bool = True,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = manifest_df.copy()
    if clean_only:
        df = df[df["keep_clean"]].copy()

    features = [
        "combined_noise_score",
        "node_noise_score",
        "connectivity_noise_score",
        "ptp",
        "abs_kurtosis",
        "hjorth_complexity",
    ]

    features = [c for c in features if c in df.columns]

    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, col in zip(axes, features):
        sns.scatterplot(
            data=df,
            x=col,
            y="sampling_weight",
            hue="label",
            alpha=0.65,
            s=25,
            ax=ax,
        )
        ax.set_title(f"Weight vs {col}")
        ax.set_ylabel("Sampling weight")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)



def plot_top_weight_segment_statistics(
    top_segments_df: pd.DataFrame,
    save_path: str | os.PathLike,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = [
        "sampling_weight",
        "combined_noise_score",
        "ptp",
        "abs_kurtosis",
        "hjorth_complexity",
    ]

    optional = [
        "connectivity_noise_score",
        "conn_std",
        "conn_abs_max",
        "conn_median_distance",
        "conn_temporal_jump",
    ]
    metrics += [c for c in optional if c in top_segments_df.columns]
    metrics = [c for c in metrics if c in top_segments_df.columns]

    plot_df = top_segments_df.melt(
        id_vars=["subject_id", "label", "segment_id", "top_rank"],
        value_vars=metrics,
        var_name="metric",
        value_name="value",
    )

    fig, ax = plt.subplots(figsize=(16, 7))

    sns.boxplot(
        data=plot_df,
        x="metric",
        y="value",
        hue="label",
        ax=ax,
    )

    ax.set_title("Statistics of top-weight segments by class")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_top_weight_segments_per_subject(
    top_segments_df: pd.DataFrame,
    save_path: str | os.PathLike,
    *,
    max_subjects: int | None = 30,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = top_segments_df.copy()

    subject_ids = sorted(df["subject_id"].unique())
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    df = df[df["subject_id"].isin(subject_ids)].copy()

    pivot = df.pivot_table(
        index="subject_id",
        columns="top_rank",
        values="sampling_weight",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, max(5, len(pivot) * 0.35)))

    sns.heatmap(
        pivot,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar=True,
        ax=ax,
    )

    ax.set_title("Top sampling weights per subject")
    ax.set_xlabel("Top-rank segment")
    ax.set_ylabel("Subject")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def generate_sampling_weight_report(
    manifest_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    top_k: int = 10,
    clean_only: bool = True,
    max_subjects_plot: int | None = 30,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_cleancluster_manifest_for_visualization(manifest_path)

    weight_subject_summary = summarize_sampling_weights_by_subject(
        manifest_df,
        clean_only=clean_only,
    )

    top_segments_df, top_subject_summary_df = summarize_top_weight_segments(
        manifest_df,
        top_k=top_k,
        clean_only=clean_only,
    )

    weight_subject_summary.to_csv(
        output_dir / "sampling_weight_subject_summary.csv",
        index=False,
    )

    top_segments_df.to_csv(
        output_dir / f"top{top_k}_sampling_weight_segments.csv",
        index=False,
    )

    top_subject_summary_df.to_csv(
        output_dir / f"top{top_k}_sampling_weight_subject_summary.csv",
        index=False,
    )

    paths = {}

    paths["weight_distribution_by_subject"] = plot_sampling_weight_distribution_by_subject(
        manifest_df,
        output_dir / "sampling_weight_distribution_by_subject.png",
        clean_only=clean_only,
        max_subjects=max_subjects_plot,
    )

    paths["weight_vs_quality"] = plot_weight_vs_quality_features(
        manifest_df,
        output_dir / "sampling_weight_vs_quality_features.png",
        clean_only=clean_only,
    )

    paths["top_weight_stats"] = plot_top_weight_segment_statistics(
        top_segments_df,
        output_dir / f"top{top_k}_sampling_weight_segment_statistics.png",
    )

    paths["top_weight_heatmap"] = plot_top_weight_segments_per_subject(
        top_segments_df,
        output_dir / f"top{top_k}_sampling_weights_per_subject.png",
        max_subjects=max_subjects_plot,
    )

    print("\nSaved sampling-weight report to:", output_dir)
    for k, v in paths.items():
        print(f"{k}: {v}")

    return {
        "manifest_df": manifest_df,
        "weight_subject_summary": weight_subject_summary,
        "top_segments_df": top_segments_df,
        "top_subject_summary_df": top_subject_summary_df,
        "paths": paths,
    }



def _to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def load_manifest(manifest_path: str | Path, clean_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    required = {"subject_id", "label", "segment_id", "sampling_weight", "keep_clean"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["keep_clean"] = _to_bool_series(df["keep_clean"])
    df["sampling_weight"] = pd.to_numeric(df["sampling_weight"], errors="coerce")

    if "split" not in df.columns:
        df["split"] = df["subject_id"].str.extract(r"^(train|val|test)_", expand=False).fillna("unknown")

    if clean_only:
        df = df[df["keep_clean"]].copy()

    df = df.dropna(subset=["sampling_weight"])
    return df


def summarize_sampling_weights(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sid, g in df.groupby("subject_id"):
        w = g["sampling_weight"].to_numpy(dtype=float)
        w = np.clip(w, 0, None)

        ess = (w.sum() ** 2) / np.sum(w ** 2) if np.sum(w ** 2) > 0 else np.nan

        rows.append({
            "subject_id": sid,
            "split": g["split"].iloc[0],
            "label": int(g["label"].iloc[0]),
            "num_clean_segments": int(len(g)),
            "weight_min": float(np.min(w)),
            "weight_q25": float(np.percentile(w, 25)),
            "weight_median": float(np.median(w)),
            "weight_mean": float(np.mean(w)),
            "weight_q75": float(np.percentile(w, 75)),
            "weight_max": float(np.max(w)),
            "weight_std": float(np.std(w)),
            "effective_sample_size": float(ess),
            "effective_sample_fraction": float(ess / len(w)) if len(w) > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def plot_weight_box_by_subject(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    max_subjects: int | None = 50,
    sort_by: str = "subject_id",
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_sampling_weights(df)

    if sort_by == "weight_mean":
        subject_order = summary.sort_values("weight_mean")["subject_id"].tolist()
    elif sort_by == "effective_sample_fraction":
        subject_order = summary.sort_values("effective_sample_fraction")["subject_id"].tolist()
    else:
        subject_order = sorted(df["subject_id"].unique())

    if max_subjects is not None:
        subject_order = subject_order[:max_subjects]

    plot_df = df[df["subject_id"].isin(subject_order)].copy()

    fig, ax = plt.subplots(figsize=(max(14, 0.45 * len(subject_order)), 6))

    sns.boxplot(
        data=plot_df,
        x="subject_id",
        y="sampling_weight",
        order=subject_order,
        hue="label",
        dodge=False,
        ax=ax,
    )

    sns.stripplot(
        data=plot_df,
        x="subject_id",
        y="sampling_weight",
        order=subject_order,
        color="black",
        alpha=0.35,
        size=3,
        ax=ax,
    )

    ax.set_title("Sampling weight distribution by subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Sampling weight")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_weight_violin_by_subject(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    max_subjects: int | None = 50,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    subject_order = sorted(df["subject_id"].unique())
    if max_subjects is not None:
        subject_order = subject_order[:max_subjects]

    plot_df = df[df["subject_id"].isin(subject_order)].copy()

    fig, ax = plt.subplots(figsize=(max(14, 0.45 * len(subject_order)), 6))

    sns.violinplot(
        data=plot_df,
        x="subject_id",
        y="sampling_weight",
        order=subject_order,
        inner="quartile",
        cut=0,
        ax=ax,
    )

    ax.set_title("Sampling weight distribution by subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Sampling weight")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_weight_histograms_faceted(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    max_subjects: int | None = 30,
    bins: int = 20,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    subject_order = sorted(df["subject_id"].unique())
    if max_subjects is not None:
        subject_order = subject_order[:max_subjects]

    plot_df = df[df["subject_id"].isin(subject_order)].copy()

    g = sns.FacetGrid(
        plot_df,
        col="subject_id",
        col_wrap=5,
        sharex=True,
        sharey=False,
        height=2.5,
    )
    g.map_dataframe(sns.histplot, x="sampling_weight", bins=bins, kde=True)
    g.set_titles("{col_name}")
    g.set_axis_labels("Sampling weight", "Count")

    g.fig.suptitle("Sampling weight histogram per subject", y=1.02)
    g.fig.tight_layout()
    g.fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(g.fig)

    return str(save_path)


def plot_effective_sample_size(
    summary_df: pd.DataFrame,
    save_path: str | Path,
    *,
    max_subjects: int | None = 80,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.sort_values("effective_sample_fraction").copy()
    if max_subjects is not None:
        plot_df = plot_df.head(max_subjects)

    fig, ax = plt.subplots(figsize=(max(14, 0.35 * len(plot_df)), 6))

    sns.barplot(
        data=plot_df,
        x="subject_id",
        y="effective_sample_fraction",
        hue="label",
        dodge=False,
        ax=ax,
    )

    ax.set_title("Effective sample fraction by subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("ESS / number of clean segments")
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def generate_weight_distribution_report(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    clean_only: bool = True,
    max_subjects: int | None = 50,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(manifest_path, clean_only=clean_only)
    summary = summarize_sampling_weights(df)

    summary_path = output_dir / "sampling_weight_subject_summary.csv"
    summary.to_csv(summary_path, index=False)

    paths = {
        "box": plot_weight_box_by_subject(
            df,
            output_dir / "sampling_weight_box_by_subject.png",
            max_subjects=max_subjects,
            sort_by="subject_id",
        ),
        "box_sorted_by_ess": plot_weight_box_by_subject(
            df,
            output_dir / "sampling_weight_box_by_subject_sorted_by_ess.png",
            max_subjects=max_subjects,
            sort_by="effective_sample_fraction",
        ),
        "violin": plot_weight_violin_by_subject(
            df,
            output_dir / "sampling_weight_violin_by_subject.png",
            max_subjects=max_subjects,
        ),
        "histograms": plot_weight_histograms_faceted(
            df,
            output_dir / "sampling_weight_histograms_by_subject.png",
            max_subjects=min(max_subjects, 30) if max_subjects is not None else 30,
        ),
        "ess": plot_effective_sample_size(
            summary,
            output_dir / "effective_sample_fraction_by_subject.png",
            max_subjects=max_subjects,
        ),
    }

    print(f"\nSaved summary CSV: {summary_path}")
    for name, path in paths.items():
        print(f"{name}: {path}")


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--manifest_path", type=str, required=True)
#     parser.add_argument("--output_dir", type=str, required=True)
#     parser.add_argument("--max_subjects", type=int, default=50)
#     parser.add_argument("--include_removed", action="store_true")
#     return parser.parse_args()


if __name__ == "__main__":
    # root = "/home/anphan/Documents/CAUEEG/visualize/segment_selection"
    # manifest_path = os.path.join(root, "cleancluster/cleancluster_manifest.csv")
    # output_dir = os.path.join(root, "weight_visualize")
    # os.makedirs(output_dir, exist_ok=True)
    # max_subjects = 20
    # include_removed = True

    # generate_weight_distribution_report(
    #     manifest_path=manifest_path,
    #     output_dir=output_dir,
    #     clean_only=not include_removed,
    #     max_subjects=max_subjects,
    # )

#####################################################3
    args = parse_args()
    # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_merged_sliding_random_trainonly.h5"
    h5_path = "/home/anphan/Documents/caueeg_randomcrop_master_dementia_seed42.h5"

    output_dir = "/home/anphan/Documents/CAUEEG/visualize/segment_selection"
    os.makedirs(output_dir, exist_ok=True)


    class_level_path = os.path.join(output_dir, "class_level")
    os.makedirs(class_level_path, exist_ok = True)


    if args.inspect:
        inspect_h5_structure(h5_path, max_subjects=args.max_inspect_subjects)

    # subject_ids = None
    # if args.subject_ids is not None and args.subject_ids.strip():
    #     subject_ids = [x.strip() for x in args.subject_ids.split(",") if x.strip()]
    all_subject_ids = list_h5_subject_ids(h5_path)

    if args.subject_ids is not None and args.subject_ids.strip():
        subject_ids = [x.strip() for x in args.subject_ids.split(",") if x.strip()]
    else:
        subject_ids = filter_subject_ids_by_split(all_subject_ids, args.split)

    print(f"\nAnalyzing split={args.split}")
    print(f"Number of selected subjects: {len(subject_ids)}")
    print("First few subject IDs:", subject_ids[:10])
    # sample_id = subject_ids[:10]
    max_subject_plots = args.max_subject_plots
    if max_subject_plots < 0:
        max_subject_plots = None

    cfg = H5WindowSelectionConfig(
        h5_path=h5_path,
        output_dir=output_dir,

        connectivity_metric=args.connectivity_metric,
        alpha_band_index=args.alpha_band_index,

        n_select=args.n_select,
        n_clusters=args.n_clusters,
        # anomaly_contamination=args.anomaly_contamination,

        embedding_method=args.embedding_method,
        max_subject_plots=max_subject_plots,

        statistical_kurtosis_idx=args.stat_kurtosis_idx,
        statistical_ptp_idx=args.stat_ptp_idx,
        hjorth_complexity_idx=args.hjorth_complexity_idx,
        artifact_cluster_anomaly_fraction_threshold=args.artifact_cluster_anomaly_fraction_threshold,
        artifact_cluster_min_size=args.artifact_cluster_min_size,

    )

    # per_subject_df, global_df = analyze_h5_window_selection(cfg, subject_ids=subject_ids)


    # if args.build_cleancluster_manifest or args.manifest_only:
    manifest_df, clean_subject_df, clean_class_df, cluster_df_all = build_cleancluster_manifest(
        cfg,
        subject_ids=subject_ids,
        save_outputs=True,
    )

    validate_cleancluster_manifest(
        manifest_df,
        min_clean_segments_warning=args.n_select,
    )

    print("\nCleanCluster class summary:")
    print(clean_class_df)

    generate_sampling_weight_report(
        manifest_path=Path(cfg.output_dir) / "cleancluster" / "cleancluster_manifest.csv",
        output_dir=Path(cfg.output_dir) / "cleancluster" / "sampling_weight_report",
        top_k=args.n_select,
        clean_only=True,
        max_subjects_plot=30,
    )
    if not args.manifest_only:
        per_subject_df, global_df = analyze_h5_window_selection(cfg, subject_ids=subject_ids)

        print("\nGlobal summary:")
        print(global_df)

########################################



#############
# rule: combined_noise_score ↑  →  sampling_weight ↓
# PTP ↑                 →  sampling_weight ↓
# abs kurtosis ↑        →  sampling_weight ↓
# Hjorth complexity ↑   →  sampling_weight ↓
# Flat high weights:
#     subject is mostly clean.

# Many low weights:
#     subject has many borderline/noisy segments.

# Very small effective sample size:
#     only a few segments dominate the sampling distribution.

# All top ranks close to 1:
#     many good clean segments.

# Only rank 1 high, others much lower:
#     clean pool may still contain many borderline segments.

# Many subjects with low top weights:
#     the cleaning rule may be too weak, or the subject is noisy overall.