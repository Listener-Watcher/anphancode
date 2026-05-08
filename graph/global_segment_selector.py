

from __future__ import annotations

import seaborn as sns

import os
import joblib
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)

# ---------------------------------------------------------
# Region-level representation
# ---------------------------------------------------------

DEFAULT_REGION_TO_CHANNELS_MONO = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "FZ", "Fz"],
    "central":   ["C3", "C4", "CZ", "Cz"],
    "parietal":  ["P3", "P4", "PZ", "Pz"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "occipital": ["O1", "O2"],
}


def _graph_tensor_to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _zscore_segment_node_features(x, eps=1e-8):
    """
    x: [N, F]
    Z-score feature columns inside one segment graph.
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def _build_region_indices(channel_names, region_to_channels):
    name_to_idx = {str(ch): i for i, ch in enumerate(channel_names)}
    region_indices = {}

    for region, chs in region_to_channels.items():
        idx = []
        for ch in chs:
            if ch in name_to_idx:
                idx.append(name_to_idx[ch])
        if len(idx) > 0:
            region_indices[region] = idx

    if len(region_indices) == 0:
        raise ValueError("No valid region indices found. Check channel_names and region_to_channels.")

    return region_indices

def plot_pca_kmeans_grid_report(grid_df, save_dir):
    """
    Save plots for PCA/KMeans model selection.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("silhouette_mean", "Silhouette score ↑"),
        ("davies_bouldin_mean", "Davies-Bouldin ↓"),
        ("calinski_harabasz_mean", "Calinski-Harabasz ↑"),
        ("ari_stability_mean", "Cluster stability ARI ↑"),
        ("min_cluster_fraction_mean", "Minimum cluster fraction ↑"),
        ("max_cluster_fraction_mean", "Maximum cluster fraction ↓"),
    ]

    paths = {}

    for metric, title in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))

        for pca_dim, g in grid_df.groupby("pca_dim"):
            g = g.sort_values("n_clusters")
            ax.plot(
                g["n_clusters"],
                g[metric],
                marker="o",
                label=f"PCA dim={pca_dim}",
            )

        ax.set_xlabel("Number of clusters")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        out_path = save_dir / f"grid_{metric}.png"
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)

        paths[metric] = str(out_path)

    # Elbow plot for inertia.
    fig, ax = plt.subplots(figsize=(8, 5))
    for pca_dim, g in grid_df.groupby("pca_dim"):
        g = g.sort_values("n_clusters")
        ax.plot(
            g["n_clusters"],
            g["inertia_mean"],
            marker="o",
            label=f"PCA dim={pca_dim}",
        )

    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("KMeans inertia")
    ax.set_title("KMeans elbow plot")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    out_path = save_dir / "grid_inertia_elbow.png"
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    paths["inertia_elbow"] = str(out_path)

    return paths

def plot_pca_explained_variance(
    X_train,
    save_path,
    *,
    max_components=30,
    seed=42,
):
    """
    Plot PCA explained variance on TRAIN segments only.

    This helps choose candidate PCA dimensions.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X_train, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    max_components = min(int(max_components), Xz.shape[0], Xz.shape[1])
    pca = PCA(n_components=max_components, random_state=seed)
    pca.fit(Xz)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    df = pd.DataFrame({
        "pc": np.arange(1, max_components + 1),
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": cumulative,
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["pc"], df["explained_variance_ratio"], marker="o", label="Individual")
    ax.plot(df["pc"], df["cumulative_explained_variance"], marker="o", label="Cumulative")

    for thr in [0.80, 0.90, 0.95]:
        ax.axhline(thr, linestyle="--", linewidth=1, alpha=0.6)
        ax.text(df["pc"].iloc[-1], thr, f" {int(thr * 100)}%", va="bottom")

    ax.set_xlabel("PCA dimension")
    ax.set_ylabel("Explained variance")
    ax.set_title("PCA explained variance on train segments")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return df
def graph_to_global_segment_feature(
    g,
    *,
    channel_names,
    region_to_channels=DEFAULT_REGION_TO_CHANNELS_MONO,
    include_node_flat=True,
    include_region_mean=True,
    include_region_std=True,
    zscore_segment=True,
):
    """
    Convert one segment graph into one feature vector for global clustering.

    This is only for segment selection/clustering, not the MIL model input.
    """
    x = _graph_tensor_to_numpy(g.x).astype(np.float32)  # [N, F]

    if zscore_segment:
        x = _zscore_segment_node_features(x)

    blocks = []

    if include_node_flat:
        blocks.append(x.reshape(-1))

    if include_region_mean or include_region_std:
        region_indices = _build_region_indices(channel_names, region_to_channels)

        region_parts = []
        for region, idx in region_indices.items():
            xr = x[idx, :]  # [num_region_channels, F]

            if include_region_mean:
                region_parts.append(xr.mean(axis=0))

            if include_region_std:
                region_parts.append(xr.std(axis=0))

        blocks.append(np.concatenate(region_parts, axis=0))

    if len(blocks) == 0:
        raise ValueError("No representation block selected.")

    return np.concatenate(blocks, axis=0).astype(np.float32)


def graphs_to_global_feature_table(
    graphs,
    *,
    channel_names,
    split_name,
    fold=None,
    region_to_channels=DEFAULT_REGION_TO_CHANNELS_MONO,
    include_node_flat=True,
    include_region_mean=True,
    include_region_std=True,
    zscore_segment=True,
):
    """
    Convert graph list to:
        X: [num_segments, D]
        meta_df: one row per segment
    """
    X_rows = []
    meta_rows = []

    for i, g in enumerate(graphs):
        feat = graph_to_global_segment_feature(
            g,
            channel_names=channel_names,
            region_to_channels=region_to_channels,
            include_node_flat=include_node_flat,
            include_region_mean=include_region_mean,
            include_region_std=include_region_std,
            zscore_segment=zscore_segment,
        )

        sid = str(getattr(g, "subject_id", ""))
        seg_id = int(getattr(g, "segment_id", i))
        start_sample = int(getattr(g, "start_sample", -1))
        label = int(g.y.view(-1)[0].item())

        X_rows.append(feat)
        meta_rows.append({
            "fold": fold,
            "split": split_name,
            "subject_id": sid,
            "segment_id": seg_id,
            "start_sample": start_sample,
            "true_label": label,
            "row_index": i,
        })

    X = np.stack(X_rows, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return X, meta_df


def fit_global_segment_clusterer(
    X_train,
    *,
    n_clusters=8,
    pca_dim=8,
    seed=42,
    save_path=None,
):
    """
    Fit global segment-state model on TRAIN segments only.

    Returns a dict containing scaler, PCA, KMeans.
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_train)

    pca_dim = min(int(pca_dim), Xz.shape[0], Xz.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xp = pca.fit_transform(Xz)

    n_clusters = min(int(n_clusters), Xp.shape[0])
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    )
    train_cluster_id = kmeans.fit_predict(Xp)

    model = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "pca_dim": pca_dim,
        "seed": int(seed),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)

    return model, train_cluster_id, Xp


def apply_global_segment_clusterer(X, clusterer):
    """
    Assign global cluster ID and centroid distance to any split.
    """
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xz = clusterer["scaler"].transform(X)
    Xp = clusterer["pca"].transform(Xz)

    kmeans = clusterer["kmeans"]
    labels = kmeans.predict(Xp)

    centers = kmeans.cluster_centers_
    dist = np.linalg.norm(Xp - centers[labels], axis=1)

    return labels.astype(int), dist.astype(np.float32), Xp.astype(np.float32)

def build_global_cluster_manifest_from_graphs(
    train_graphs,
    val_graphs=None,
    test_graphs=None,
    *,
    channel_names,
    output_dir,
    fold=None,
    n_clusters=8,
    pca_dim=8,
    seed=42,
    region_to_channels=DEFAULT_REGION_TO_CHANNELS_MONO,

    # New diagnostic options
    run_model_selection=False,
    model_selection_only=False,
    pca_dims_to_try=(3, 5, 8, 10, 15),
    n_clusters_to_try=(4, 5, 6, 8, 10, 12),
    model_selection_seeds=(15, 42, 100),
):
    """
    Fit global KMeans on train_graphs only, then assign cluster IDs
    to train/val/test graphs.

    If run_model_selection=True:
        generate PCA explained-variance plot and KMeans diagnostic plots
        before fitting the final clusterer.

    If model_selection_only=True:
        only save diagnostic plots/tables and return without creating final manifest.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. Convert TRAIN graphs to clustering matrix
    # -----------------------------------------------------
    train_X, train_meta = graphs_to_global_feature_table(
        train_graphs,
        channel_names=channel_names,
        split_name="train",
        fold=fold,
        region_to_channels=region_to_channels,
    )

    # -----------------------------------------------------
    # 2. RUN PCA/KMEANS DIAGNOSTIC HERE
    # -----------------------------------------------------
    if run_model_selection:
        model_selection_dir = output_dir / "model_selection"

        pca_df, grid_df, plot_paths = run_global_cluster_model_selection_report(
            train_X,
            output_dir=model_selection_dir,
            pca_dims=pca_dims_to_try,
            n_clusters_list=n_clusters_to_try,
            seeds=model_selection_seeds,
        )

        print("\n[PCA/KMeans model selection completed]")
        print("Saved report to:", model_selection_dir)
        print("Current final setting will still use:")
        print(f"  pca_dim={pca_dim}")
        print(f"  n_clusters={n_clusters}")

        if model_selection_only:
            return {
                "train_X": train_X,
                "train_meta": train_meta,
                "pca_df": pca_df,
                "grid_df": grid_df,
                "plot_paths": plot_paths,
                "manifest_df": None,
                "manifest_path": None,
                "clusterer": None,
                "clusterer_path": None,
            }

    # -----------------------------------------------------
    # 3. Fit final clusterer using selected pca_dim/n_clusters
    # -----------------------------------------------------
    clusterer_path = output_dir / (
        f"global_segment_clusterer_fold{fold}.joblib"
        if fold is not None
        else "global_segment_clusterer.joblib"
    )

    clusterer, train_labels, train_Xp = fit_global_segment_clusterer(
        train_X,
        n_clusters=n_clusters,
        pca_dim=pca_dim,
        seed=seed,
        save_path=clusterer_path,
    )

    # -----------------------------------------------------
    # 4. Apply final clusterer to each split
    # -----------------------------------------------------
    all_rows = []

    def _apply_split(graphs, split_name):
        if graphs is None or len(graphs) == 0:
            return None

        X, meta = graphs_to_global_feature_table(
            graphs,
            channel_names=channel_names,
            split_name=split_name,
            fold=fold,
            region_to_channels=region_to_channels,
        )

        labels, dist, Xp = apply_global_segment_clusterer(X, clusterer)

        meta = meta.copy()
        meta["global_cluster_id"] = labels
        meta["global_cluster_distance"] = dist

        # Store first two PCA dims for visualization.
        meta["global_pca1"] = Xp[:, 0]
        meta["global_pca2"] = Xp[:, 1] if Xp.shape[1] > 1 else 0.0

        return meta

    train_df = _apply_split(train_graphs, "train")
    val_df = _apply_split(val_graphs, "val")
    test_df = _apply_split(test_graphs, "test")

    for df in [train_df, val_df, test_df]:
        if df is not None:
            all_rows.append(df)

    manifest_df = pd.concat(all_rows, ignore_index=True)

    # -----------------------------------------------------
    # 5. Save manifest
    # -----------------------------------------------------
    manifest_path = output_dir / (
        f"global_cluster_manifest_fold{fold}.csv"
        if fold is not None
        else "global_cluster_manifest.csv"
    )

    manifest_df.to_csv(manifest_path, index=False)

    return {
        "clusterer": clusterer,
        "clusterer_path": str(clusterer_path),
        "manifest_df": manifest_df,
        "manifest_path": str(manifest_path),
    }

def select_global_cluster_scored_graphs(
    graphs,
    manifest_df,
    *,
    k=10,
    score_col="sampling_weight",
    cluster_col="global_cluster_id",
    distance_col="global_cluster_distance",
):
    """
    Select k segments per subject using global clusters.

    If sampling_weight is not available, use distance-only centrality.
    """
    graph_lookup = {
        (str(g.subject_id), int(g.segment_id)): g
        for g in graphs
    }

    df = manifest_df.copy()
    df = df[
        df.apply(
            lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup,
            axis=1,
        )
    ].copy()

    if len(df) == 0:
        raise RuntimeError("No manifest rows match graph list.")

    selected_keys = []

    for sid, sdf in df.groupby("subject_id"):
        chosen_rows = []

        # cluster allocation: first one per available cluster
        clusters = sorted(sdf[cluster_col].unique())

        for c in clusters:
            cdf = sdf[sdf[cluster_col] == c].copy()

            if score_col in cdf.columns:
                base_score = cdf[score_col].to_numpy(dtype=np.float64)
            else:
                base_score = np.ones(len(cdf), dtype=np.float64)

            dist = cdf[distance_col].to_numpy(dtype=np.float64)
            rep_score = base_score / (1.0 + dist)

            best_pos = int(np.argmax(rep_score))
            chosen_rows.append(cdf.iloc[best_pos])

        chosen_df = pd.DataFrame(chosen_rows)

        # If too many clusters, keep best representatives.
        if len(chosen_df) > k:
            if score_col in chosen_df.columns:
                chosen_df["_rank_score"] = (
                    chosen_df[score_col].astype(float)
                    / (1.0 + chosen_df[distance_col].astype(float))
                )
            else:
                chosen_df["_rank_score"] = 1.0 / (
                    1.0 + chosen_df[distance_col].astype(float)
                )

            chosen_df = chosen_df.sort_values("_rank_score", ascending=False).head(k)

        # If fewer than k clusters, fill from remaining best segments.
        if len(chosen_df) < k:
            chosen_pairs = set(
                zip(chosen_df["subject_id"].astype(str), chosen_df["segment_id"].astype(int))
            )

            remaining = sdf[
                ~sdf.apply(
                    lambda r: (str(r["subject_id"]), int(r["segment_id"])) in chosen_pairs,
                    axis=1,
                )
            ].copy()

            if len(remaining) > 0:
                if score_col in remaining.columns:
                    remaining["_rank_score"] = (
                        remaining[score_col].astype(float)
                        / (1.0 + remaining[distance_col].astype(float))
                    )
                else:
                    remaining["_rank_score"] = 1.0 / (
                        1.0 + remaining[distance_col].astype(float)
                    )

                need = k - len(chosen_df)
                fill_df = remaining.sort_values("_rank_score", ascending=False).head(need)
                chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        for _, row in chosen_df.iterrows():
            selected_keys.append((str(row["subject_id"]), int(row["segment_id"])))

    return [graph_lookup[key] for key in selected_keys if key in graph_lookup]




def _flatten_subject_window_features_from_payload(
    entry,
    feature_families_for_cluster=("relative_band_power", "hjorth"),
):
    """
    Build one segment-level representation matrix from payload.

    entry["features"][fam]: [W, N, F]
    return X: [W, D]
    """
    blocks = []

    for fam in feature_families_for_cluster:
        if fam not in entry["features"]:
            continue

        x = np.asarray(entry["features"][fam], dtype=np.float32)  # [W, N, F]
        if x.ndim != 3:
            raise ValueError(f"Expected {fam} shape [W,N,F], got {x.shape}")

        # Segment-wise feature standardization across nodes.
        mu = x.mean(axis=1, keepdims=True)
        sd = x.std(axis=1, keepdims=True)
        xz = (x - mu) / (sd + 1e-8)

        blocks.append(xz.reshape(xz.shape[0], -1))

    if len(blocks) == 0:
        raise ValueError(
            f"No clustering feature families found from {feature_families_for_cluster}"
        )

    return np.concatenate(blocks, axis=1).astype(np.float32)


def payload_split_to_cluster_matrix(
    payload,
    subject_ids,
    *,
    split_name,
    feature_families_for_cluster=("relative_band_power", "hjorth"),
):
    X_rows = []
    meta_rows = []

    for sid in subject_ids:
        entry = payload[sid]

        X_sid = _flatten_subject_window_features_from_payload(
            entry,
            feature_families_for_cluster=feature_families_for_cluster,
        )

        seg_ids = np.asarray(entry["segment_id"], dtype=np.int64)
        start_samples = np.asarray(entry["start_sample"], dtype=np.int64)
        label = int(entry["label"])

        for i in range(X_sid.shape[0]):
            X_rows.append(X_sid[i])
            meta_rows.append({
                "split": split_name,
                "subject_id": str(sid),
                "segment_index": int(i),
                "segment_id": int(seg_ids[i]),
                "start_sample": int(start_samples[i]),
                "true_label": label,
            })

    X = np.stack(X_rows, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)

    return X, meta_df


def fit_global_segment_clusterer(
    X_train,
    *,
    n_clusters=6,
    pca_dim=8,
    seed=42,
    save_path=None,
):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_train)

    pca_dim = min(int(pca_dim), Xz.shape[0], Xz.shape[1])
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xp = pca.fit_transform(Xz)

    n_clusters = min(int(n_clusters), Xp.shape[0])
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=seed,
    )

    train_cluster_id = kmeans.fit_predict(Xp)

    clusterer = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": kmeans,
        "n_clusters": n_clusters,
        "pca_dim": pca_dim,
        "seed": int(seed),
    }

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clusterer, save_path)

    return clusterer, train_cluster_id, Xp


def apply_global_segment_clusterer(X, clusterer):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Xz = clusterer["scaler"].transform(X)
    Xp = clusterer["pca"].transform(Xz)

    kmeans = clusterer["kmeans"]
    labels = kmeans.predict(Xp)

    centers = kmeans.cluster_centers_
    dist = np.linalg.norm(Xp - centers[labels], axis=1)

    return labels.astype(int), dist.astype(np.float32), Xp.astype(np.float32)


def build_global_cluster_manifest_from_payload(
    payload,
    train_ids,
    val_ids=None,
    test_ids=None,
    *,
    feature_families_for_cluster=("relative_band_power", "hjorth"),
    output_dir,
    fold=None,
    n_clusters=6,
    pca_dim=15,
    seed=42,

    run_model_selection=False,
    model_selection_only=False,
    pca_dims_to_try=(5, 8, 10),
    n_clusters_to_try=(25, 50, 100),
    model_selection_seeds=(15, 42, 100),
):
    """
    Encoder-independent global segment clustering.

    Fit scaler/PCA/KMeans on train payload segments only.
    Then apply to val/test payload segments.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, train_meta = payload_split_to_cluster_matrix(
        payload,
        train_ids,
        split_name="train",
        feature_families_for_cluster=feature_families_for_cluster,
    )

    print("[GLOBAL_CLUSTER] X_train shape:", X_train.shape)

    if run_model_selection:
        model_selection_dir = output_dir / "model_selection"

        pca_df, grid_df, plot_paths = run_global_cluster_model_selection_report(
            X_train,
            output_dir=model_selection_dir,
            pca_dims=pca_dims_to_try,
            n_clusters_list=n_clusters_to_try,
            seeds=model_selection_seeds,
        )

        print("\n[GLOBAL_CLUSTER] PCA/KMeans model-selection report saved to:")
        print(model_selection_dir)
        print("\n[GLOBAL_CLUSTER] Current final setting:")
        print(f"  pca_dim={pca_dim}")
        print(f"  n_clusters={n_clusters}")

        if model_selection_only:
            return {
                "clusterer": None,
                "clusterer_path": None,
                "manifest_df": None,
                "manifest_path": None,
                "X_train": X_train,
                "train_meta": train_meta,
                "pca_df": pca_df,
                "grid_df": grid_df,
                "plot_paths": plot_paths,
            }

    clusterer_path = output_dir / (
        f"global_segment_clusterer_fold{fold}.joblib"
        if fold is not None
        else "global_segment_clusterer.joblib"
    )

    clusterer, _, _ = fit_global_segment_clusterer(
        X_train,
        n_clusters=n_clusters,
        pca_dim=pca_dim,
        seed=seed,
        save_path=clusterer_path,
    )

    all_dfs = []

    def _apply_split(subject_ids, split_name):
        if subject_ids is None or len(subject_ids) == 0:
            return None

        X, meta = payload_split_to_cluster_matrix(
            payload,
            subject_ids,
            split_name=split_name,
            feature_families_for_cluster=feature_families_for_cluster,
        )

        labels, dist, Xp = apply_global_segment_clusterer(X, clusterer)

        meta = meta.copy()
        meta["fold"] = fold
        meta["global_cluster_id"] = labels
        meta["global_cluster_distance"] = dist
        meta["global_pca1"] = Xp[:, 0]
        meta["global_pca2"] = Xp[:, 1] if Xp.shape[1] > 1 else 0.0

        return meta

    for subject_ids, split_name in [
        (train_ids, "train"),
        (val_ids, "val"),
        (test_ids, "test"),
    ]:
        df_split = _apply_split(subject_ids, split_name)
        if df_split is not None:
            all_dfs.append(df_split)

    manifest_df = pd.concat(all_dfs, ignore_index=True)

    manifest_path = output_dir / (
        f"global_cluster_manifest_fold{fold}.csv"
        if fold is not None
        else "global_cluster_manifest.csv"
    )

    manifest_df.to_csv(manifest_path, index=False)

    return {
        "clusterer": clusterer,
        "clusterer_path": str(clusterer_path),
        "manifest_df": manifest_df,
        "manifest_path": str(manifest_path),
    }

def _scatter_categorical(
    df,
    *,
    color_col,
    title,
    save_path,
    x_col="global_pca1",
    y_col="global_pca2",
    max_legend_items=30,
    alpha=0.75,
    s=20,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    cats = sorted(plot_df[color_col].dropna().unique(), key=lambda x: str(x))

    fig, ax = plt.subplots(figsize=(10, 8))

    # If too many subjects, legend becomes unreadable.
    show_legend = len(cats) <= max_legend_items

    cmap = plt.get_cmap("tab20")
    color_map = {
        c: cmap(i % 20)
        for i, c in enumerate(cats)
    }

    for c in cats:
        g = plot_df[plot_df[color_col] == c]
        ax.scatter(
            g[x_col],
            g[y_col],
            s=s,
            alpha=alpha,
            color=color_map[c],
            label=str(c) if show_legend else None,
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("Global PCA dim 1")
    ax.set_ylabel("Global PCA dim 2")
    ax.grid(True, alpha=0.25)

    if show_legend:
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=8,
            title=color_col,
        )
    else:
        ax.text(
            0.02,
            0.98,
            f"{len(cats)} unique {color_col}; legend hidden",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round", alpha=0.15),
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_global_kmeans_three_views(
    manifest_df,
    save_dir,
    *,
    split_filter="train",
    class_names=None,
    max_subject_legend=30,
):
    """
    Generate 3 global KMeans scatter plots:
      1. color by true class
      2. color by subject_id
      3. color by global_cluster_id

    Uses global_pca1/global_pca2 saved in the manifest.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = manifest_df.copy()

    if split_filter is not None:
        df = df[df["split"] == split_filter].copy()

    if "global_pca1" not in df.columns or "global_pca2" not in df.columns:
        raise KeyError("manifest_df must contain global_pca1 and global_pca2.")

    # Human-readable class label.
    if class_names is not None:
        df["true_class_name"] = df["true_label"].map(lambda x: class_names[int(x)])
        class_col = "true_class_name"
    else:
        class_col = "true_label"

    paths = {}

    paths["by_true_class"] = _scatter_categorical(
        df,
        color_col=class_col,
        title=f"Global segment KMeans PCA view | color by true class | split={split_filter}",
        save_path=save_dir / f"global_kmeans_by_true_class_{split_filter}.png",
        max_legend_items=20,
        s=25,
    )

    paths["by_subject"] = _scatter_categorical(
        df,
        color_col="subject_id",
        title=f"Global segment KMeans PCA view | color by subject | split={split_filter}",
        save_path=save_dir / f"global_kmeans_by_subject_{split_filter}.png",
        max_legend_items=max_subject_legend,
        s=18,
        alpha=0.65,
    )

    paths["by_cluster"] = _scatter_categorical(
        df,
        color_col="global_cluster_id",
        title=f"Global segment KMeans PCA view | color by global cluster | split={split_filter}",
        save_path=save_dir / f"global_kmeans_by_cluster_{split_filter}.png",
        max_legend_items=50,
        s=25,
    )

    return paths
def evaluate_pca_kmeans_grid(
    X_train,
    *,
    pca_dims=(3, 5, 8, 10, 15),
    n_clusters_list=(4, 5, 6, 8, 10, 12),
    seeds=(15, 42, 100),
    sample_size_for_silhouette=5000,
):
    """
    Evaluate PCA dimension and KMeans cluster count on TRAIN segments only.

    Returns one row per pca_dim x n_clusters.
    Metrics:
      - inertia: lower is better, but always decreases with more clusters
      - silhouette: higher is better
      - calinski_harabasz: higher is better
      - davies_bouldin: lower is better
      - ari_stability: higher means clusters are stable across seeds
      - min_cluster_fraction: avoids tiny useless clusters
      - max_cluster_fraction: avoids one giant dominant cluster
    """
    X = np.asarray(X_train, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    rows = []

    for pca_dim in pca_dims:
        pca_dim_eff = min(int(pca_dim), Xz.shape[0], Xz.shape[1])

        pca = PCA(n_components=pca_dim_eff, random_state=42)
        Xp = pca.fit_transform(Xz)

        for k in n_clusters_list:
            k_eff = min(int(k), Xp.shape[0] - 1)
            if k_eff < 2:
                continue

            labels_by_seed = []
            seed_rows = []

            for seed in seeds:
                km = KMeans(
                    n_clusters=k_eff,
                    n_init=20,
                    random_state=int(seed),
                )
                labels = km.fit_predict(Xp)
                labels_by_seed.append(labels)

                counts = np.bincount(labels, minlength=k_eff)
                frac = counts / counts.sum()

                # Silhouette can be slow. Use sample_size when many segments exist.
                sil_sample = min(sample_size_for_silhouette, Xp.shape[0])
                sil = silhouette_score(
                    Xp,
                    labels,
                    sample_size=sil_sample if Xp.shape[0] > sil_sample else None,
                    random_state=int(seed),
                )

                seed_rows.append({
                    "inertia": float(km.inertia_),
                    "silhouette": float(sil),
                    "calinski_harabasz": float(calinski_harabasz_score(Xp, labels)),
                    "davies_bouldin": float(davies_bouldin_score(Xp, labels)),
                    "min_cluster_fraction": float(frac.min()),
                    "max_cluster_fraction": float(frac.max()),
                    "num_empty_clusters": int(np.sum(counts == 0)),
                })

            # Stability across KMeans seeds.
            ari_vals = []
            for i in range(len(labels_by_seed)):
                for j in range(i + 1, len(labels_by_seed)):
                    ari_vals.append(adjusted_rand_score(labels_by_seed[i], labels_by_seed[j]))

            seed_df = pd.DataFrame(seed_rows)

            rows.append({
                "pca_dim": int(pca_dim_eff),
                "n_clusters": int(k_eff),

                "explained_variance": float(np.sum(pca.explained_variance_ratio_)),

                "inertia_mean": float(seed_df["inertia"].mean()),
                "inertia_std": float(seed_df["inertia"].std()),

                "silhouette_mean": float(seed_df["silhouette"].mean()),
                "silhouette_std": float(seed_df["silhouette"].std()),

                "calinski_harabasz_mean": float(seed_df["calinski_harabasz"].mean()),
                "calinski_harabasz_std": float(seed_df["calinski_harabasz"].std()),

                "davies_bouldin_mean": float(seed_df["davies_bouldin"].mean()),
                "davies_bouldin_std": float(seed_df["davies_bouldin"].std()),

                "ari_stability_mean": float(np.mean(ari_vals)) if len(ari_vals) > 0 else np.nan,
                "ari_stability_std": float(np.std(ari_vals)) if len(ari_vals) > 0 else np.nan,

                "min_cluster_fraction_mean": float(seed_df["min_cluster_fraction"].mean()),
                "max_cluster_fraction_mean": float(seed_df["max_cluster_fraction"].mean()),
            })

    return pd.DataFrame(rows)
def run_global_cluster_model_selection_report(
    X_train,
    output_dir,
    *,
    pca_dims=(3, 5, 8, 10, 15),
    n_clusters_list=(4, 5, 6, 8, 10, 12),
    seeds=(15, 42, 100),
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pca_df = plot_pca_explained_variance(
        X_train,
        output_dir / "pca_explained_variance.png",
        max_components=max(max(pca_dims), 30),
    )
    pca_df.to_csv(output_dir / "pca_explained_variance.csv", index=False)

    grid_df = evaluate_pca_kmeans_grid(
        X_train,
        pca_dims=pca_dims,
        n_clusters_list=n_clusters_list,
        seeds=seeds,
    )
    grid_df.to_csv(output_dir / "pca_kmeans_grid_metrics.csv", index=False)

    plot_paths = plot_pca_kmeans_grid_report(
        grid_df,
        output_dir / "plots",
    )

    print("\nSaved PCA/KMeans model-selection report to:", output_dir)
    print("\nTop candidates by silhouette + stability:")
    cols = [
        "pca_dim",
        "n_clusters",
        "explained_variance",
        "silhouette_mean",
        "davies_bouldin_mean",
        "ari_stability_mean",
        "min_cluster_fraction_mean",
        "max_cluster_fraction_mean",
    ]
    print(
        grid_df.sort_values(
            ["silhouette_mean", "ari_stability_mean", "min_cluster_fraction_mean"],
            ascending=[False, False, False],
        )[cols].head(10)
    )

    return pca_df, grid_df, plot_paths




def load_global_cluster_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"subject_id", "global_cluster_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing required columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)

    if "split" not in df.columns:
        df["split"] = df["subject_id"].str.extract(
            r"^(train|val|test)_", expand=False
        ).fillna("unknown")

    if "true_label" not in df.columns:
        df["true_label"] = -1

    return df


def make_subject_cluster_count_table(
    df: pd.DataFrame,
    *,
    split: str | None = "train",
    normalize: bool = False,
) -> pd.DataFrame:
    plot_df = df.copy()

    if split is not None:
        plot_df = plot_df[plot_df["split"] == split].copy()

    count_table = pd.crosstab(
        plot_df["subject_id"],
        plot_df["global_cluster_id"],
    )

    # Ensure all clusters appear as columns.
    all_clusters = sorted(df["global_cluster_id"].unique())
    count_table = count_table.reindex(columns=all_clusters, fill_value=0)

    if normalize:
        count_table = count_table.div(count_table.sum(axis=1).replace(0, np.nan), axis=0)
        count_table = count_table.fillna(0.0)

    return count_table


def add_subject_label_to_index(
    table: pd.DataFrame,
    df: pd.DataFrame,
) -> pd.DataFrame:
    label_map = (
        df.groupby("subject_id")["true_label"]
        .first()
        .to_dict()
    )

    out = table.copy()
    out.index = [
        f"{sid} | y={label_map.get(sid, 'NA')}"
        for sid in out.index
    ]
    return out


def plot_subject_cluster_heatmap(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    split: str | None = "train",
    normalize: bool = True,
    max_subjects: int | None = 80,
    sort_by_label: bool = True,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    table = make_subject_cluster_count_table(
        df,
        split=split,
        normalize=normalize,
    )

    meta = (
        df[df["split"] == split].groupby("subject_id")["true_label"].first()
        if split is not None
        else df.groupby("subject_id")["true_label"].first()
    )

    if sort_by_label:
        order = (
            pd.DataFrame({"subject_id": table.index})
            .assign(true_label=lambda x: x["subject_id"].map(meta))
            .sort_values(["true_label", "subject_id"])
            ["subject_id"]
            .tolist()
        )
        table = table.loc[order]

    if max_subjects is not None:
        table = table.iloc[:max_subjects]

    table_labeled = add_subject_label_to_index(table, df)

    fig_height = max(6, 0.28 * len(table_labeled))
    fig_width = max(8, 0.75 * table_labeled.shape[1])

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        table_labeled,
        cmap="viridis",
        annot=True,
        fmt=".2f" if normalize else "d",
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "Proportion of segments" if normalize else "Segment count"},
        ax=ax,
    )

    title_value = "proportion" if normalize else "count"
    ax.set_title(f"Subject distribution over global clusters ({title_value}) | split={split}")
    ax.set_xlabel("Global cluster ID")
    ax.set_ylabel("Subject | true label")

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cluster_class_percentage_and_entropy(
    manifest_path,
    output_dir,
    *,
    cluster_col="global_cluster_id",
    label_col=None,
    class_names=None,
    only_clean=False,
    clean_col="keep_clean",
    count_unit="segment",   # "segment" or "subject"
    normalize_entropy=True,
    prefix="global_cluster_class_entropy",
):
    """
    Compute class percentage + entropy for each global cluster.

    Entropy:
        H = - sum_c p_c log2(p_c)

    If normalize_entropy=True:
        H_norm = H / log2(num_classes)
        => 0 means pure cluster
        => 1 means perfectly mixed cluster
    """

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(manifest_path)

    # -----------------------------
    # infer label column if needed
    # -----------------------------
    if label_col is None:
        candidates = [
            "label",
            "class_label",
            "y",
            "true_label",
            "subject_label",
        ]
        for c in candidates:
            if c in df.columns:
                label_col = c
                break

    if label_col is None:
        raise KeyError(
            "Cannot find label column. Please pass label_col='your_label_column'. "
            f"Available columns: {list(df.columns)}"
        )

    if cluster_col not in df.columns:
        raise KeyError(
            f"Cannot find cluster_col={cluster_col!r}. "
            f"Available columns: {list(df.columns)}"
        )

    # -----------------------------
    # optionally keep only clean rows
    # -----------------------------
    if only_clean:
        if clean_col not in df.columns:
            raise KeyError(f"only_clean=True but {clean_col!r} is not in manifest.")
        if df[clean_col].dtype != bool:
            df[clean_col] = (
                df[clean_col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
        df = df[df[clean_col]].copy()

    # -----------------------------
    # choose counting unit
    # -----------------------------
    if count_unit == "segment":
        # one vote per segment row
        # avoid accidental duplicated segment rows if subject_id/segment_id exist
        dedup_cols = [cluster_col, label_col]
        if "subject_id" in df.columns:
            dedup_cols.append("subject_id")
        if "segment_id" in df.columns:
            dedup_cols.append("segment_id")

        df_count = df[dedup_cols].drop_duplicates()

    elif count_unit == "subject":
        # one vote per subject inside each cluster
        if "subject_id" not in df.columns:
            raise KeyError("count_unit='subject' requires subject_id column.")
        df_count = df[[cluster_col, "subject_id", label_col]].drop_duplicates()

    else:
        raise ValueError("count_unit must be 'segment' or 'subject'.")

    df_count[cluster_col] = df_count[cluster_col].astype(int)
    df_count[label_col] = df_count[label_col].astype(int)

    # -----------------------------
    # counts and percentages
    # -----------------------------
    counts = pd.crosstab(
        df_count[cluster_col],
        df_count[label_col],
    ).sort_index()

    all_labels = sorted(df_count[label_col].unique())
    counts = counts.reindex(columns=all_labels, fill_value=0)

    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0

    # -----------------------------
    # entropy
    # -----------------------------
    probs = counts.div(counts.sum(axis=1), axis=0).to_numpy(dtype=np.float64)
    probs_safe = np.clip(probs, 1e-12, 1.0)

    entropy = -(probs_safe * np.log2(probs_safe)).sum(axis=1)

    if normalize_entropy:
        max_entropy = np.log2(len(all_labels))
        entropy_plot = entropy / max_entropy if max_entropy > 0 else entropy
        entropy_name = "entropy_norm"
        entropy_ylabel = "Normalized entropy"
    else:
        entropy_plot = entropy
        entropy_name = "entropy"
        entropy_ylabel = "Entropy"

    summary_df = counts.copy()
    summary_df.columns = [f"count_class_{c}" for c in summary_df.columns]

    for c in all_labels:
        summary_df[f"percent_class_{c}"] = percentages[c].values

    summary_df["num_samples"] = counts.sum(axis=1).values
    summary_df["entropy"] = entropy
    summary_df["entropy_norm"] = (
        entropy / np.log2(len(all_labels)) if len(all_labels) > 1 else 0.0
    )

    summary_df = summary_df.reset_index()

    summary_csv = os.path.join(output_dir, f"{prefix}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # -----------------------------
    # plotting
    # -----------------------------
    if class_names is None:
        class_names = {c: f"Class {c}" for c in all_labels}
    elif isinstance(class_names, list):
        class_names = {i: name for i, name in enumerate(class_names)}

    x = np.arange(len(percentages.index))
    cluster_labels = percentages.index.astype(str).tolist()

    fig, ax1 = plt.subplots(figsize=(max(10, len(x) * 0.7), 6))

    bottom = np.zeros(len(percentages))

    for c in all_labels:
        vals = percentages[c].to_numpy()
        ax1.bar(
            x,
            vals,
            bottom=bottom,
            label=class_names.get(c, f"Class {c}"),
            alpha=0.85,
        )
        bottom += vals

    ax1.set_xlabel("Global cluster ID")
    ax1.set_ylabel("Class percentage (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cluster_labels, rotation=45, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        entropy_plot,
        marker="o",
        linewidth=2,
        label=entropy_ylabel,
    )

    if normalize_entropy:
        ax2.set_ylim(0, 1.05)

    ax2.set_ylabel(entropy_ylabel)

    # annotate entropy values
    for i, h in enumerate(entropy_plot):
        ax2.text(
            x[i],
            h + 0.02 if normalize_entropy else h,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    title_unit = "segments" if count_unit == "segment" else "subjects"
    title_clean = "clean only" if only_clean else "all"
    ax1.set_title(
        f"Class composition and entropy per global cluster "
        f"({title_unit}, {title_clean})"
    )

    # combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
    )

    fig.tight_layout()

    fig_path = os.path.join(output_dir, f"{prefix}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved summary CSV: {summary_csv}")
    print(f"Saved plot: {fig_path}")

    return summary_df, fig_path
def plot_subject_cluster_stacked_bar(
    df: pd.DataFrame,
    save_path: str | Path,
    *,
    split: str | None = "train",
    normalize: bool = True,
    max_subjects: int | None = 80,
) -> str:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    table = make_subject_cluster_count_table(
        df,
        split=split,
        normalize=normalize,
    )

    meta = (
        df[df["split"] == split].groupby("subject_id")["true_label"].first()
        if split is not None
        else df.groupby("subject_id")["true_label"].first()
    )

    order = (
        pd.DataFrame({"subject_id": table.index})
        .assign(true_label=lambda x: x["subject_id"].map(meta))
        .sort_values(["true_label", "subject_id"])
        ["subject_id"]
        .tolist()
    )
    table = table.loc[order]

    if max_subjects is not None:
        table = table.iloc[:max_subjects]

    fig, ax = plt.subplots(figsize=(max(14, 0.35 * len(table)), 6))

    bottom = np.zeros(len(table), dtype=float)

    for cluster_id in table.columns:
        vals = table[cluster_id].to_numpy(dtype=float)
        ax.bar(
            np.arange(len(table)),
            vals,
            bottom=bottom,
            label=f"cluster {cluster_id}",
        )
        bottom += vals

    ax.set_title(f"Global cluster composition per subject | split={split}")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Proportion of segments" if normalize else "Segment count")
    ax.set_xticks(np.arange(len(table)))
    ax.set_xticklabels(table.index, rotation=90, fontsize=7)
    ax.legend(title="Global cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_subject_cluster_timeline(
    df: pd.DataFrame,
    output_dir: str | Path,
    *,
    split: str | None = "train",
    max_subjects: int | None = 30,
) -> list[str]:
    """
    One small timeline plot per subject:
        x-axis = segment order or start_sample
        y-axis = global_cluster_id
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    if split is not None:
        plot_df = plot_df[plot_df["split"] == split].copy()

    subject_ids = sorted(plot_df["subject_id"].unique())
    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    paths = []

    for sid in subject_ids:
        g = plot_df[plot_df["subject_id"] == sid].copy()

        if "start_sample" in g.columns:
            g = g.sort_values("start_sample")
            x = g["start_sample"].to_numpy()
            xlabel = "start_sample"
        elif "segment_id" in g.columns:
            g = g.sort_values("segment_id")
            x = g["segment_id"].to_numpy()
            xlabel = "segment_id"
        else:
            g = g.reset_index(drop=True)
            x = np.arange(len(g))
            xlabel = "segment_order"

        y = g["global_cluster_id"].to_numpy()
        label = int(g["true_label"].iloc[0]) if "true_label" in g.columns else -1

        fig, ax = plt.subplots(figsize=(10, 3))

        ax.scatter(x, y, s=35)
        ax.plot(x, y, alpha=0.35)

        ax.set_title(f"Global cluster timeline | subject={sid} | label={label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("global_cluster_id")
        ax.set_yticks(sorted(plot_df["global_cluster_id"].unique()))
        ax.grid(True, alpha=0.25)

        fig.tight_layout()

        save_path = output_dir / f"cluster_timeline_{sid}.png"
        fig.savefig(save_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        paths.append(str(save_path))

    return paths


def summarize_subject_cluster_distribution(
    df: pd.DataFrame,
    *,
    split: str | None = "train",
) -> pd.DataFrame:
    plot_df = df.copy()
    if split is not None:
        plot_df = plot_df[plot_df["split"] == split].copy()

    rows = []

    for sid, g in plot_df.groupby("subject_id"):
        counts = g["global_cluster_id"].value_counts().sort_index()
        proportions = counts / counts.sum()

        rows.append({
            "subject_id": sid,
            "split": g["split"].iloc[0],
            "true_label": int(g["true_label"].iloc[0]),
            "num_segments": int(len(g)),
            "num_clusters_present": int(counts.shape[0]),
            "dominant_cluster": int(counts.idxmax()),
            "dominant_cluster_fraction": float(proportions.max()),
            "cluster_entropy": float(
                -(proportions * np.log(proportions + 1e-12)).sum()
            ),
        })

    return pd.DataFrame(rows)


def generate_subject_cluster_distribution_report(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    split: str | None = "train",
    max_subjects: int | None = 80,
    max_timeline_subjects: int | None = 30,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_global_cluster_manifest(manifest_path)

    summary_df = summarize_subject_cluster_distribution(df, split=split)
    summary_path = output_dir / f"subject_cluster_distribution_summary_{split}.csv"
    summary_df.to_csv(summary_path, index=False)

    count_table = make_subject_cluster_count_table(df, split=split, normalize=False)
    prop_table = make_subject_cluster_count_table(df, split=split, normalize=True)

    count_path = output_dir / f"subject_cluster_counts_{split}.csv"
    prop_path = output_dir / f"subject_cluster_proportions_{split}.csv"

    count_table.to_csv(count_path)
    prop_table.to_csv(prop_path)

    heatmap_path = plot_subject_cluster_heatmap(
        df,
        output_dir / f"subject_cluster_heatmap_proportion_{split}.png",
        split=split,
        normalize=True,
        max_subjects=max_subjects,
    )

    count_heatmap_path = plot_subject_cluster_heatmap(
        df,
        output_dir / f"subject_cluster_heatmap_count_{split}.png",
        split=split,
        normalize=False,
        max_subjects=max_subjects,
    )

    stacked_path = plot_subject_cluster_stacked_bar(
        df,
        output_dir / f"subject_cluster_stacked_bar_{split}.png",
        split=split,
        normalize=True,
        max_subjects=max_subjects,
    )

    timeline_paths = plot_subject_cluster_timeline(
        df,
        output_dir / f"subject_timelines_{split}",
        split=split,
        max_subjects=max_timeline_subjects,
    )

    print("\nSaved subject/global-cluster distribution report to:", output_dir)
    print("summary:", summary_path)
    print("counts:", count_path)
    print("proportions:", prop_path)
    print("heatmap proportion:", heatmap_path)
    print("heatmap count:", count_heatmap_path)
    print("stacked bar:", stacked_path)
    print("timeline folder:", output_dir / f"subject_timelines_{split}")

    return {
        "summary_df": summary_df,
        "count_table": count_table,
        "proportion_table": prop_table,
        "paths": {
            "summary_csv": str(summary_path),
            "count_csv": str(count_path),
            "proportion_csv": str(prop_path),
            "heatmap_proportion": heatmap_path,
            "heatmap_count": count_heatmap_path,
            "stacked_bar": stacked_path,
            "timeline_paths": timeline_paths,
        },
    }




if __name__ == "__main__":

    class_names = ["normal", "mci", "dementia"]
    SEED = 42
    N_CLUSTERS=50
    PCA_DIM=10
    root_path = "/home/anphan/Documents/CAUEEG"
    save_path = os.path.join(root_path,'visualize-random')
    os.makedirs(save_path,exist_ok = True)

    # out_h5 = "/home/anphan/Documents/caueeg_merged_sliding_random_trainonly.h5"
    out_h5 = "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
    dataset_path = "/home/anphan/Downloads/caueeg-dataset/"
    task = "dementia-no-overlap"
    file_format = "edf"
    feature_families = ['relative_band_power', 'statistical']
    bad_ids = {"00587", "00781", "01301", "train_00587", "train_00781", "train_01301"}

    from caueeg_removenoise_with_levels import *
    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    train_records, train_ids = dataset_to_subject_records(train_set)
    val_records, val_ids = dataset_to_subject_records(val_set)
    test_records, test_ids = dataset_to_subject_records(test_set)

    all_records = train_records + val_records + test_records

    train_ids_filter = [sid for sid in train_ids if sid not in bad_ids]
    val_ids_filter   = [sid for sid in val_ids if sid not in bad_ids]
    test_ids_filter = [sid for sid in test_ids if sid not in bad_ids]
    all_ids_filter   = train_ids_filter + val_ids_filter + test_ids_filter
    # all_ids = train_ids + val_ids + test_ids

    train_ids_suf = ['train_' + item for item in train_ids_filter]
    val_ids_suf = ['val_' + item for item in val_ids_filter]
    test_ids_suf = ['test_' + item for item in test_ids_filter]

    all_ids_suf = train_ids_suf + val_ids_suf + test_ids_suf

    # 4) load payload
    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=all_ids_suf,
        feature_families=feature_families,
        connectivity_metrics=["wpli"],
        connectivity_band=None,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )
    print("Loading payload....")
    cluster_path=os.path.join(save_path, f"statistical_clusters_PCA{PCA_DIM}_N{N_CLUSTERS}")

    global_out = build_global_cluster_manifest_from_payload(
        payload=payload,
        train_ids=train_ids_suf,
        val_ids=val_ids_suf,
        test_ids=test_ids_suf,
        feature_families_for_cluster=("relative_band_power", "statistical"),
        output_dir=cluster_path,
        fold=None,
        n_clusters=N_CLUSTERS,
        pca_dim=PCA_DIM,
        run_model_selection=False,
        model_selection_only=False,
    )





    global_cluster_manifest_path = global_out["manifest_path"]
    global_manifest_df = global_out["manifest_df"]

    print("[GLOBAL_CLUSTER] Manifest:", global_cluster_manifest_path)

    plot_paths = plot_global_kmeans_three_views(
        global_out["manifest_df"],
        save_dir=os.path.join(cluster_path, "plots"),
        split_filter="train",
        class_names=class_names,
        max_subject_legend=30,
    )

    print(plot_paths)
    plot_global_kmeans_three_views(
        global_out["manifest_df"],
        save_dir=os.path.join(cluster_path, "plots"),
        split_filter="val",
        class_names=class_names,
    )

    plot_global_kmeans_three_views(
        global_out["manifest_df"],
        save_dir=os.path.join(cluster_path, "plots"),
        split_filter="test",
        class_names=class_names,
    )
    for split in ["train", "val", "test"]:
        output_dir = f"{cluster_path}/{split}"
        os.makedirs(output_dir,exist_ok=True)
        max_subjects=80
        max_timeline_subjects=30
        generate_subject_cluster_distribution_report(
            manifest_path=global_cluster_manifest_path,
            output_dir=output_dir,
            split=split,
            max_subjects=max_subjects,
            max_timeline_subjects=max_timeline_subjects,
        )

    summary_df, fig_path = plot_cluster_class_percentage_and_entropy(
        global_cluster_manifest_path,
        cluster_path,
        label_col="true_label",              # change to "class_label" if your CSV uses that
        class_names=class_names,
        only_clean=False,               # set True if you want only keep_clean=True
        count_unit="segment",           # use "subject" if you do not want subjects with more segments to dominate
        normalize_entropy=True,
    )
# if __name__ == "__main__":
