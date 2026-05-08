#!/usr/bin/env python3
"""
Cluster-quality diagnostics for global KMeans segment clusters.

Main goals:
1. Check cluster compactness / separation if embeddings are available.
2. Check whether clusters are dominated by a few subjects.
3. Check class distribution per cluster.
4. Detect noise-like or suspicious clusters.
5. Save CSV summaries and diagnostic plots.

Expected manifest columns:
    subject_id
    segment_id
    kmeans_cluster_id
    keep_clean

Optional useful columns:
    true_label / label / class_label / y
    kmeans_centroid_distance
    iforest_score
    sampling_weight
    cluster_size

Optional embedding/features:
    - Provide --features_csv with feature columns, or
    - Provide --embeddings_npz with an array named X by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    mutual_info_score,
    normalized_mutual_info_score,
)

try:
    from scipy.stats import chi2_contingency
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# Basic helpers
# =========================================================

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def detect_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return (
        s.astype(str)
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def entropy_from_counts(counts: np.ndarray, normalize: bool = True) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0

    p = counts / total
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))

    if normalize:
        max_ent = np.log(len(counts)) if len(counts) > 1 else 1.0
        if max_ent <= 0:
            return 0.0
        ent = ent / max_ent

    return float(ent)


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def cramers_v_from_contingency(table: pd.DataFrame) -> Optional[float]:
    if not SCIPY_AVAILABLE:
        return None

    arr = table.to_numpy(dtype=np.float64)
    if arr.sum() <= 0:
        return None

    chi2, p, dof, expected = chi2_contingency(arr)
    n = arr.sum()
    r, k = arr.shape
    denom = n * max(min(r - 1, k - 1), 1)
    return float(np.sqrt(chi2 / denom))


# =========================================================
# Loading
# =========================================================

def load_manifest(
    manifest_path: str | Path,
    cluster_col: str = "kmeans_cluster_id",
    subject_col: str = "subject_id",
    segment_col: str = "segment_id",
    clean_col: str = "keep_clean",
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    required = [cluster_col, subject_col, segment_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")

    df[subject_col] = df[subject_col].astype(str)
    df[segment_col] = df[segment_col].astype(int)
    df[cluster_col] = df[cluster_col].astype(int)

    if clean_col in df.columns:
        df[clean_col] = normalize_bool_series(df[clean_col])
    else:
        df[clean_col] = True

    return df


def infer_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Conservative auto-detection.

    It only uses numeric columns with common embedding/feature prefixes.
    This avoids accidentally using label, cluster_id, distance, or scores as features.
    """
    prefixes = (
        "emb_", "embedding_", "feat_", "feature_",
        "x_", "z_", "pc_", "pca_", "dim_",
    )

    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in prefixes):
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)

    return cols


def load_feature_matrix(
    manifest_df: pd.DataFrame,
    *,
    features_csv: Optional[str] = None,
    embeddings_npz: Optional[str] = None,
    embedding_key: str = "X",
    feature_cols: Optional[Sequence[str]] = None,
    subject_col: str = "subject_id",
    segment_col: str = "segment_id",
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Returns:
        X, used_feature_cols

    Rules:
    - If features_csv has subject_id and segment_id, join by those keys.
    - Otherwise, it must have the same row order/length as manifest.
    - embeddings_npz must have the same row order/length as manifest.
    """
    if features_csv is None and embeddings_npz is None:
        if feature_cols is None:
            feature_cols = infer_feature_columns(manifest_df)
        if len(feature_cols) == 0:
            return None, []

        X = manifest_df[list(feature_cols)].to_numpy(dtype=np.float32)
        return np.nan_to_num(X), list(feature_cols)

    if features_csv is not None:
        feat_df = pd.read_csv(features_csv)

        if subject_col in feat_df.columns and segment_col in feat_df.columns:
            feat_df[subject_col] = feat_df[subject_col].astype(str)
            feat_df[segment_col] = feat_df[segment_col].astype(int)

            merged = manifest_df[[subject_col, segment_col]].merge(
                feat_df,
                on=[subject_col, segment_col],
                how="left",
                validate="one_to_one",
            )

            if feature_cols is None:
                feature_cols = infer_feature_columns(merged)
                if len(feature_cols) == 0:
                    meta_cols = {
                        subject_col, segment_col,
                        "label", "true_label", "class_label", "y",
                        "kmeans_cluster_id", "keep_clean",
                        "kmeans_centroid_distance", "iforest_score",
                        "sampling_weight", "cluster_size",
                    }
                    feature_cols = [
                        c for c in merged.columns
                        if c not in meta_cols and pd.api.types.is_numeric_dtype(merged[c])
                    ]

            X = merged[list(feature_cols)].to_numpy(dtype=np.float32)

        else:
            if len(feat_df) != len(manifest_df):
                raise ValueError(
                    f"features_csv has {len(feat_df)} rows, but manifest has {len(manifest_df)} rows. "
                    "Provide subject_id and segment_id in features_csv for safe joining."
                )

            if feature_cols is None:
                feature_cols = infer_feature_columns(feat_df)
                if len(feature_cols) == 0:
                    feature_cols = [
                        c for c in feat_df.columns
                        if pd.api.types.is_numeric_dtype(feat_df[c])
                    ]

            X = feat_df[list(feature_cols)].to_numpy(dtype=np.float32)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, list(feature_cols)

    if embeddings_npz is not None:
        data = np.load(embeddings_npz, allow_pickle=True)
        if embedding_key not in data:
            raise KeyError(f"{embeddings_npz} does not contain key {embedding_key!r}. Available: {list(data.keys())}")

        X = np.asarray(data[embedding_key], dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Embedding array must be 2D [num_segments, dim], got {X.shape}")

        if len(X) != len(manifest_df):
            raise ValueError(
                f"Embedding array has {len(X)} rows, but manifest has {len(manifest_df)} rows. "
                "Make sure they are in the same order."
            )

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        names = [f"emb_{i}" for i in range(X.shape[1])]
        return X, names

    return None, []


# =========================================================
# Per-cluster summaries
# =========================================================

def compute_subject_cluster_distribution(
    df: pd.DataFrame,
    *,
    subject_col: str,
    cluster_col: str,
    label_col: Optional[str],
) -> pd.DataFrame:
    counts = pd.crosstab(df[subject_col], df[cluster_col])
    pct = counts.div(counts.sum(axis=1), axis=0)

    rows = []
    for sid in counts.index:
        row_counts = counts.loc[sid].to_numpy(dtype=np.float64)
        row_pct = pct.loc[sid].to_numpy(dtype=np.float64)

        dominant_idx = int(np.argmax(row_counts))
        dominant_cluster = counts.columns[dominant_idx]
        dominant_share = float(row_pct[dominant_idx])
        n_clusters_present = int((row_counts > 0).sum())
        entropy_norm = entropy_from_counts(row_counts, normalize=True)

        row = {
            subject_col: sid,
            "n_segments": int(row_counts.sum()),
            "n_clusters_present": n_clusters_present,
            "dominant_cluster": int(dominant_cluster),
            "dominant_cluster_share": dominant_share,
            "cluster_entropy_norm": entropy_norm,
        }

        if label_col is not None:
            labels = df.loc[df[subject_col] == sid, label_col].dropna().unique()
            row[label_col] = labels[0] if len(labels) > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_cluster_summary(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    subject_col: str,
    label_col: Optional[str],
    clean_col: str,
    distance_col: Optional[str],
    iforest_col: Optional[str],
    min_cluster_size: int = 20,
    subject_dom_threshold: float = 0.40,
    low_clean_threshold: float = 0.70,
) -> pd.DataFrame:
    global_label_dist = None
    if label_col is not None:
        global_label_dist = df[label_col].value_counts(normalize=True).to_dict()

    rows = []

    for cid, cdf in df.groupby(cluster_col):
        n_segments = len(cdf)
        subj_counts = cdf[subject_col].value_counts()
        n_subjects = len(subj_counts)
        top_subject = str(subj_counts.index[0])
        top_subject_count = int(subj_counts.iloc[0])
        top_subject_share = top_subject_count / max(n_segments, 1)
        subject_entropy_norm = entropy_from_counts(subj_counts.to_numpy(), normalize=True)

        row = {
            "cluster_id": int(cid),
            "n_segments": int(n_segments),
            "n_subjects": int(n_subjects),
            "top_subject": top_subject,
            "top_subject_count": top_subject_count,
            "top_subject_share": float(top_subject_share),
            "subject_entropy_norm": float(subject_entropy_norm),
            "clean_ratio": float(cdf[clean_col].mean()) if clean_col in cdf.columns else 1.0,
        }

        if label_col is not None:
            label_counts = cdf[label_col].value_counts().sort_index()
            majority_label = label_counts.idxmax()
            majority_label_count = int(label_counts.max())
            majority_label_share = majority_label_count / max(n_segments, 1)
            class_entropy_norm = entropy_from_counts(label_counts.to_numpy(), normalize=True)

            # Enrichment relative to global label distribution.
            enrichments = {}
            for lab, local_count in label_counts.items():
                local_p = local_count / max(n_segments, 1)
                global_p = global_label_dist.get(lab, 0.0)
                enrichments[lab] = safe_div(local_p, global_p)

            enriched_label = max(enrichments, key=enrichments.get)
            max_enrichment = enrichments[enriched_label]

            row.update({
                "n_classes_present": int(len(label_counts)),
                "majority_label": majority_label,
                "majority_label_count": majority_label_count,
                "majority_label_share": float(majority_label_share),
                "class_entropy_norm": float(class_entropy_norm),
                "enriched_label": enriched_label,
                "max_class_enrichment": float(max_enrichment),
            })

            for lab, count in label_counts.items():
                row[f"class_{lab}_count"] = int(count)
                row[f"class_{lab}_pct"] = float(count / max(n_segments, 1))

        if distance_col is not None and distance_col in cdf.columns:
            dist = pd.to_numeric(cdf[distance_col], errors="coerce").dropna()
            if len(dist) > 0:
                row.update({
                    "centroid_distance_mean": float(dist.mean()),
                    "centroid_distance_std": float(dist.std(ddof=0)),
                    "centroid_distance_median": float(dist.median()),
                    "centroid_distance_p90": float(dist.quantile(0.90)),
                    "centroid_distance_max": float(dist.max()),
                })

        if iforest_col is not None and iforest_col in cdf.columns:
            sc = pd.to_numeric(cdf[iforest_col], errors="coerce").dropna()
            if len(sc) > 0:
                row.update({
                    "iforest_score_mean": float(sc.mean()),
                    "iforest_score_std": float(sc.std(ddof=0)),
                    "iforest_score_median": float(sc.median()),
                    "iforest_score_min": float(sc.min()),
                })

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)

    # Add flags.
    out["flag_tiny_cluster"] = out["n_segments"] < min_cluster_size
    out["flag_subject_dominated"] = out["top_subject_share"] >= subject_dom_threshold
    out["flag_low_subject_diversity"] = out["subject_entropy_norm"] < 0.50
    out["flag_low_clean_ratio"] = out["clean_ratio"] < low_clean_threshold

    if "centroid_distance_mean" in out.columns:
        q75 = out["centroid_distance_mean"].quantile(0.75)
        q25 = out["centroid_distance_mean"].quantile(0.25)
        iqr = q75 - q25
        far_cutoff = q75 + 1.5 * iqr
        out["flag_far_from_centroid"] = out["centroid_distance_mean"] > far_cutoff
    else:
        out["flag_far_from_centroid"] = False

    if label_col is not None and "majority_label_share" in out.columns:
        out["flag_class_pure_and_suspicious"] = (
            (out["majority_label_share"] >= 0.80)
            & (
                out["flag_subject_dominated"]
                | (out["n_subjects"] < 5)
            )
        )
    else:
        out["flag_class_pure_and_suspicious"] = False

    def recommend(r):
        if r["flag_low_clean_ratio"] or r["flag_far_from_centroid"] or r["flag_tiny_cluster"]:
            return "exclude_or_inspect_as_noise"
        if r["flag_subject_dominated"] or r["flag_low_subject_diversity"]:
            return "inspect_subject_fingerprint"
        if label_col is not None and r.get("majority_label_share", 0.0) >= 0.70:
            return "possible_disease_or_confound_cluster"
        return "useful_common_state_cluster"

    out["recommendation"] = out.apply(recommend, axis=1)
    return out


def compute_global_cluster_metrics(
    df: pd.DataFrame,
    X: Optional[np.ndarray],
    *,
    cluster_col: str,
    label_col: Optional[str],
    max_metric_samples: int = 20000,
    seed: int = 42,
) -> dict:
    metrics = {}

    clusters = df[cluster_col].to_numpy()
    metrics["n_segments"] = int(len(df))
    metrics["n_clusters"] = int(pd.Series(clusters).nunique())

    if label_col is not None:
        y = df[label_col].to_numpy()
        metrics["mutual_info_cluster_label"] = float(mutual_info_score(y, clusters))
        metrics["normalized_mutual_info_cluster_label"] = float(normalized_mutual_info_score(y, clusters))

        table = pd.crosstab(df[cluster_col], df[label_col])
        metrics["cramers_v_cluster_label"] = cramers_v_from_contingency(table)

        if SCIPY_AVAILABLE:
            chi2, p, dof, expected = chi2_contingency(table.to_numpy())
            metrics["chi2_cluster_label"] = float(chi2)
            metrics["chi2_p_value_cluster_label"] = float(p)
            metrics["chi2_dof_cluster_label"] = int(dof)

    if X is not None:
        X = np.asarray(X, dtype=np.float32)
        valid = np.isfinite(X).all(axis=1)
        Xv = X[valid]
        cv = clusters[valid]

        # Need at least 2 clusters and more samples than clusters.
        if len(np.unique(cv)) >= 2 and len(Xv) > len(np.unique(cv)):
            if len(Xv) > max_metric_samples:
                rng = np.random.default_rng(seed)
                idx = rng.choice(len(Xv), size=max_metric_samples, replace=False)
                X_eval = Xv[idx]
                c_eval = cv[idx]
            else:
                X_eval = Xv
                c_eval = cv

            # Silhouette can be slow; sample handles that.
            metrics["silhouette_score"] = float(silhouette_score(X_eval, c_eval))
            metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_eval, c_eval))
            metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_eval, c_eval))
            metrics["cluster_metric_n_used"] = int(len(X_eval))
        else:
            metrics["cluster_metric_warning"] = "Not enough valid samples/clusters for silhouette/DB/CH."

    return metrics


def save_representative_segments(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    cluster_col: str,
    subject_col: str,
    segment_col: str,
    label_col: Optional[str],
    distance_col: Optional[str],
    iforest_col: Optional[str],
) -> pd.DataFrame:
    rows = []

    for cid, cdf in df.groupby(cluster_col):
        # Closest to centroid.
        if distance_col is not None and distance_col in cdf.columns:
            tmp = cdf.copy()
            tmp[distance_col] = pd.to_numeric(tmp[distance_col], errors="coerce")
            tmp = tmp.dropna(subset=[distance_col])

            if len(tmp) > 0:
                closest = tmp.sort_values(distance_col, ascending=True).iloc[0]
                farthest = tmp.sort_values(distance_col, ascending=False).iloc[0]

                for rep_type, r in [("closest_to_centroid", closest), ("farthest_from_centroid", farthest)]:
                    row = {
                        "cluster_id": int(cid),
                        "rep_type": rep_type,
                        subject_col: str(r[subject_col]),
                        segment_col: int(r[segment_col]),
                        distance_col: float(r[distance_col]),
                    }
                    if label_col is not None:
                        row[label_col] = r[label_col]
                    rows.append(row)

        # Cleanest by iforest score.
        if iforest_col is not None and iforest_col in cdf.columns:
            tmp = cdf.copy()
            tmp[iforest_col] = pd.to_numeric(tmp[iforest_col], errors="coerce")
            tmp = tmp.dropna(subset=[iforest_col])

            if len(tmp) > 0:
                cleanest = tmp.sort_values(iforest_col, ascending=False).iloc[0]
                row = {
                    "cluster_id": int(cid),
                    "rep_type": "highest_iforest_score",
                    subject_col: str(cleanest[subject_col]),
                    segment_col: int(cleanest[segment_col]),
                    iforest_col: float(cleanest[iforest_col]),
                }
                if label_col is not None:
                    row[label_col] = cleanest[label_col]
                rows.append(row)

    rep_df = pd.DataFrame(rows)
    rep_df.to_csv(out_dir / "representative_segments_by_cluster.csv", index=False)
    return rep_df


# =========================================================
# Plots
# =========================================================

def save_heatmap(
    mat: pd.DataFrame,
    save_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: tuple[float, float] = (10, 6),
    annotate: bool = False,
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_xticklabels([str(c) for c in mat.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(i) for i in mat.index])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if annotate and mat.shape[0] <= 20 and mat.shape[1] <= 20:
        arr = mat.to_numpy()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_bar(
    x,
    y,
    save_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    rotation: int = 45,
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(len(x)), y)
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels([str(v) for v in x], rotation=rotation, ha="right")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    subject_summary: pd.DataFrame,
    out_dir: Path,
    *,
    cluster_col: str,
    subject_col: str,
    label_col: Optional[str],
    distance_col: Optional[str],
    max_subjects_heatmap: int = 80,
):
    # Cluster size.
    save_bar(
        cluster_summary["cluster_id"].to_list(),
        cluster_summary["n_segments"].to_numpy(),
        out_dir / "cluster_size.png",
        title="Cluster size",
        xlabel="Cluster ID",
        ylabel="Number of segments",
    )

    # Clean ratio.
    save_bar(
        cluster_summary["cluster_id"].to_list(),
        cluster_summary["clean_ratio"].to_numpy(),
        out_dir / "cluster_clean_ratio.png",
        title="Clean ratio by cluster",
        xlabel="Cluster ID",
        ylabel="Mean keep_clean",
    )

    # Top subject share.
    save_bar(
        cluster_summary["cluster_id"].to_list(),
        cluster_summary["top_subject_share"].to_numpy(),
        out_dir / "cluster_top_subject_share.png",
        title="Subject domination by cluster",
        xlabel="Cluster ID",
        ylabel="Top subject share",
    )

    # Class distribution.
    if label_col is not None:
        class_counts = pd.crosstab(df[cluster_col], df[label_col]).sort_index()
        class_pct = class_counts.div(class_counts.sum(axis=1), axis=0)

        save_heatmap(
            class_counts,
            out_dir / "class_by_cluster_counts.png",
            title="Class counts by cluster",
            xlabel="Class label",
            ylabel="Cluster ID",
            figsize=(8, 6),
            annotate=True,
        )

        save_heatmap(
            class_pct,
            out_dir / "class_by_cluster_percent.png",
            title="Class percentage by cluster",
            xlabel="Class label",
            ylabel="Cluster ID",
            figsize=(8, 6),
            annotate=True,
        )

    # Subject-cluster distribution.
    subj_cluster = pd.crosstab(df[subject_col], df[cluster_col])
    subj_cluster_pct = subj_cluster.div(subj_cluster.sum(axis=1), axis=0)

    # Sort by dominant cluster and optionally label.
    tmp = subject_summary.copy()
    sort_cols = ["dominant_cluster", "dominant_cluster_share"]
    if label_col is not None and label_col in tmp.columns:
        sort_cols = [label_col] + sort_cols

    sorted_subjects = (
        tmp.sort_values(sort_cols, ascending=[True] * len(sort_cols))[subject_col]
        .astype(str)
        .to_list()
    )
    sorted_subjects = [s for s in sorted_subjects if s in subj_cluster_pct.index]

    if len(sorted_subjects) > max_subjects_heatmap:
        sorted_subjects = sorted_subjects[:max_subjects_heatmap]

    save_heatmap(
        subj_cluster_pct.loc[sorted_subjects],
        out_dir / "subject_cluster_distribution_percent.png",
        title="Subject segment distribution across clusters",
        xlabel="Cluster ID",
        ylabel="Subject ID",
        figsize=(10, max(5, 0.12 * len(sorted_subjects))),
        annotate=False,
    )

    # Centroid distance boxplot by cluster.
    if distance_col is not None and distance_col in df.columns:
        tmp = df[[cluster_col, distance_col]].copy()
        tmp[distance_col] = pd.to_numeric(tmp[distance_col], errors="coerce")
        tmp = tmp.dropna()

        data = [
            tmp.loc[tmp[cluster_col] == cid, distance_col].to_numpy()
            for cid in sorted(tmp[cluster_col].unique())
        ]
        labels = [str(cid) for cid in sorted(tmp[cluster_col].unique())]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.set_title("Centroid distance distribution by cluster")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel(distance_col)
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(out_dir / "centroid_distance_by_cluster_boxplot.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# =========================================================
# Main diagnostic runner
# =========================================================

def run_cluster_quality_diagnostics(
    *,
    manifest_path: str,
    out_dir: str,
    features_csv: Optional[str] = None,
    embeddings_npz: Optional[str] = None,
    embedding_key: str = "X",
    feature_cols: Optional[Sequence[str]] = None,
    cluster_col: str = "kmeans_cluster_id",
    subject_col: str = "subject_id",
    segment_col: str = "segment_id",
    label_col: Optional[str] = None,
    clean_col: str = "keep_clean",
    distance_col: Optional[str] = None,
    iforest_col: Optional[str] = None,
    min_cluster_size: int = 20,
    subject_dom_threshold: float = 0.40,
    low_clean_threshold: float = 0.70,
    max_metric_samples: int = 20000,
    seed: int = 42,
) -> dict:
    out_dir = ensure_dir(out_dir)

    df = load_manifest(
        manifest_path,
        cluster_col=cluster_col,
        subject_col=subject_col,
        segment_col=segment_col,
        clean_col=clean_col,
    )

    # Auto-detect optional columns.
    if label_col is None:
        label_col = detect_col(df, ["true_label", "label", "class_label", "y"])

    if distance_col is None:
        distance_col = detect_col(df, ["kmeans_centroid_distance", "global_cluster_distance", "centroid_distance"])

    if iforest_col is None:
        iforest_col = detect_col(df, ["iforest_score", "isolation_forest_score", "clean_score"])

    # Save normalized manifest.
    df.to_csv(out_dir / "manifest_used_by_diagnostics.csv", index=False)

    X, used_feature_cols = load_feature_matrix(
        df,
        features_csv=features_csv,
        embeddings_npz=embeddings_npz,
        embedding_key=embedding_key,
        feature_cols=feature_cols,
        subject_col=subject_col,
        segment_col=segment_col,
    )

    # Cluster summary.
    cluster_summary = compute_cluster_summary(
        df,
        cluster_col=cluster_col,
        subject_col=subject_col,
        label_col=label_col,
        clean_col=clean_col,
        distance_col=distance_col,
        iforest_col=iforest_col,
        min_cluster_size=min_cluster_size,
        subject_dom_threshold=subject_dom_threshold,
        low_clean_threshold=low_clean_threshold,
    )
    cluster_summary.to_csv(out_dir / "cluster_quality_summary.csv", index=False)

    # Subject distribution.
    subject_summary = compute_subject_cluster_distribution(
        df,
        subject_col=subject_col,
        cluster_col=cluster_col,
        label_col=label_col,
    )
    subject_summary.to_csv(out_dir / "subject_cluster_summary.csv", index=False)

    # Count tables.
    subject_cluster_counts = pd.crosstab(df[subject_col], df[cluster_col])
    subject_cluster_pct = subject_cluster_counts.div(subject_cluster_counts.sum(axis=1), axis=0)
    subject_cluster_counts.to_csv(out_dir / "subject_by_cluster_counts.csv")
    subject_cluster_pct.to_csv(out_dir / "subject_by_cluster_percent.csv")

    if label_col is not None:
        class_cluster_counts = pd.crosstab(df[cluster_col], df[label_col])
        class_cluster_pct = class_cluster_counts.div(class_cluster_counts.sum(axis=1), axis=0)
        class_cluster_counts.to_csv(out_dir / "class_by_cluster_counts.csv")
        class_cluster_pct.to_csv(out_dir / "class_by_cluster_percent.csv")

    # Global metrics.
    global_metrics = compute_global_cluster_metrics(
        df,
        X,
        cluster_col=cluster_col,
        label_col=label_col,
        max_metric_samples=max_metric_samples,
        seed=seed,
    )
    global_metrics["manifest_path"] = str(manifest_path)
    global_metrics["features_csv"] = features_csv
    global_metrics["embeddings_npz"] = embeddings_npz
    global_metrics["used_feature_cols"] = used_feature_cols
    global_metrics["label_col"] = label_col
    global_metrics["distance_col"] = distance_col
    global_metrics["iforest_col"] = iforest_col

    with open(out_dir / "global_cluster_metrics.json", "w") as f:
        json.dump(global_metrics, f, indent=2)

    # Flagged clusters.
    flag_cols = [
        "flag_tiny_cluster",
        "flag_subject_dominated",
        "flag_low_subject_diversity",
        "flag_low_clean_ratio",
        "flag_far_from_centroid",
        "flag_class_pure_and_suspicious",
    ]
    flagged = cluster_summary[cluster_summary[flag_cols].any(axis=1)].copy()
    flagged.to_csv(out_dir / "flagged_clusters.csv", index=False)

    # Representative segments.
    save_representative_segments(
        df,
        out_dir,
        cluster_col=cluster_col,
        subject_col=subject_col,
        segment_col=segment_col,
        label_col=label_col,
        distance_col=distance_col,
        iforest_col=iforest_col,
    )

    # Plots.
    make_plots(
        df,
        cluster_summary,
        subject_summary,
        out_dir,
        cluster_col=cluster_col,
        subject_col=subject_col,
        label_col=label_col,
        distance_col=distance_col,
    )

    print("\nSaved diagnostics to:", out_dir)
    print("\nImportant files:")
    print("  cluster_quality_summary.csv")
    print("  subject_cluster_summary.csv")
    print("  class_by_cluster_percent.csv")
    print("  flagged_clusters.csv")
    print("  representative_segments_by_cluster.csv")
    print("  global_cluster_metrics.json")

    return {
        "manifest": df,
        "cluster_summary": cluster_summary,
        "subject_summary": subject_summary,
        "global_metrics": global_metrics,
        "flagged_clusters": flagged,
    }


# =========================================================
# CLI
# =========================================================

def parse_feature_cols(s: Optional[str]) -> Optional[list[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--manifest_path", type=str, required=True)
    # parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--features_csv", type=str, default=None)
    parser.add_argument("--embeddings_npz", type=str, default=None)
    parser.add_argument("--embedding_key", type=str, default="X")
    parser.add_argument("--feature_cols", type=str, default=None)

    parser.add_argument("--cluster_col", type=str, default="global_cluster_id")
    parser.add_argument("--subject_col", type=str, default="subject_id")
    parser.add_argument("--segment_col", type=str, default="segment_id")
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--clean_col", type=str, default="keep_clean")
    parser.add_argument("--distance_col", type=str, default=None)
    parser.add_argument("--iforest_col", type=str, default=None)

    parser.add_argument("--min_cluster_size", type=int, default=20)
    parser.add_argument("--subject_dom_threshold", type=float, default=0.40)
    parser.add_argument("--low_clean_threshold", type=float, default=0.70)
    parser.add_argument("--max_metric_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    manifest_path = "/home/anphan/Documents/CAUEEG/visualize-merged_sliding_random/global_segment_clusters/global_cluster_manifest.csv"

    out_dir = "/home/anphan/Documents/CAUEEG/visualize/global_segment_clusters/cluster_quality_diagnostics"
    import os
    os.makedirs(out_dir, exist_ok=True)

    run_cluster_quality_diagnostics(
        manifest_path=manifest_path,
        out_dir=out_dir,
        features_csv=args.features_csv,
        embeddings_npz=args.embeddings_npz,
        embedding_key=args.embedding_key,
        feature_cols=parse_feature_cols(args.feature_cols),
        cluster_col=args.cluster_col,
        subject_col=args.subject_col,
        segment_col=args.segment_col,
        label_col=args.label_col,
        clean_col=args.clean_col,
        distance_col=args.distance_col,
        iforest_col=args.iforest_col,
        min_cluster_size=args.min_cluster_size,
        subject_dom_threshold=args.subject_dom_threshold,
        low_clean_threshold=args.low_clean_threshold,
        max_metric_samples=args.max_metric_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()