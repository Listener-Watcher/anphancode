
# from lib import *
# from model import *
# from data_utils import *
# from graph_utils import *
# from data_preparation import * 
from utils_all import get_feature_dim_from_string
import config
# from fake_label import *
import os
import json
import argparse
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    pairwise_distances,
)

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# If umap is installed:
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

def run_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X_scaled)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    return Z, pca



def run_umap(X_scaled, n_components=2, n_neighbors=15, min_dist=0.1):
    if not HAS_UMAP:
        raise ImportError("UMAP is not installed. pip install umap-learn")
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=42
    )
    Z = reducer.fit_transform(X_scaled)
    return Z, reducer

# =========================================================
# 6. CLUSTER / OVERLAP METRICS
# =========================================================
def compute_silhouette_by_class(X_scaled, y_class):
    if len(np.unique(y_class)) < 2:
        return np.nan
    return silhouette_score(X_scaled, y_class)


def compute_pairwise_distance_stats(X_scaled, y_class, y_subject):
    """
    Compare distances among:
    - same subject
    - same class but different subject
    - different class
    """
    D = pairwise_distances(X_scaled, metric="euclidean")
    n = D.shape[0]

    same_subject = []
    same_class_diff_subject = []
    diff_class = []

    for i in range(n):
        for j in range(i + 1, n):
            d = D[i, j]
            if y_subject[i] == y_subject[j]:
                same_subject.append(d)
            elif y_class[i] == y_class[j]:
                same_class_diff_subject.append(d)
            else:
                diff_class.append(d)

    stats = {
        "same_subject_mean": np.mean(same_subject) if same_subject else np.nan,
        "same_subject_std": np.std(same_subject) if same_subject else np.nan,
        "same_class_diff_subject_mean": np.mean(same_class_diff_subject) if same_class_diff_subject else np.nan,
        "same_class_diff_subject_std": np.std(same_class_diff_subject) if same_class_diff_subject else np.nan,
        "diff_class_mean": np.mean(diff_class) if diff_class else np.nan,
        "diff_class_std": np.std(diff_class) if diff_class else np.nan,
    }
    return stats, same_subject, same_class_diff_subject, diff_class

# =========================================================
# 7. kNN PURITY: CLASS VS SUBJECT
# =========================================================
def compute_knn_purity(X_scaled, y_class, y_subject, k=10):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X_scaled)
    distances, indices = nbrs.kneighbors(X_scaled)

    # remove self neighbor
    indices = indices[:, 1:]

    class_purity_list = []
    subject_purity_list = []

    for i in range(len(X_scaled)):
        neigh = indices[i]
        class_match = (y_class[neigh] == y_class[i]).mean()
        subject_match = (y_subject[neigh] == y_subject[i]).mean()
        class_purity_list.append(class_match)
        subject_purity_list.append(subject_match)

    return {
        "knn_class_purity_mean": float(np.mean(class_purity_list)),
        "knn_subject_purity_mean": float(np.mean(subject_purity_list)),
        "knn_class_purity_std": float(np.std(class_purity_list)),
        "knn_subject_purity_std": float(np.std(subject_purity_list)),
    }


# =========================================================
# 8. SUBJECT-LEVEL CENTROIDS / HETEROGENEITY
# =========================================================

def analyze_subject_heterogeneity(df, X_scaled):
    rows = []

    for sid, group in df.groupby("subject_id"):
        idx = group.index.values
        X_sub = X_scaled[idx]
        centroid = X_sub.mean(axis=0)
        dists = np.linalg.norm(X_sub - centroid, axis=1)

        rows.append({
            "subject_id": sid,
            "class_id": group["class_id"].iloc[0],
            "n_segments": len(group),
            "within_subject_mean_dist": dists.mean(),
            "within_subject_std_dist": dists.std(),
        })

    return pd.DataFrame(rows)



def build_subject_centroids(df, X_scaled):
    centroids = []
    meta = []

    for sid, group in df.groupby("subject_id"):
        idx = group.index.values
        centroid = X_scaled[idx].mean(axis=0)
        centroids.append(centroid)
        meta.append({
            "subject_id": sid,
            "class_id": group["class_id"].iloc[0],
            "n_segments": len(group)
        })

    centroids = np.vstack(centroids)
    meta_df = pd.DataFrame(meta)
    return meta_df, centroids


# =========================================================
# 9. OPTIONAL: SEGMENT DISTANCE TO CLASS CENTROIDS
# =========================================================
def segment_to_class_centroid_analysis(df, X_scaled):
    """
    For each class, compute centroid.
    Then for each segment, see which class centroid it is closest to.
    This helps check overlap and "mixed" subjects.
    """
    class_centroids = {}
    for c, group in df.groupby("class_id"):
        idx = group.index.values
        class_centroids[c] = X_scaled[idx].mean(axis=0)

    class_names = list(class_centroids.keys())
    centroid_mat = np.vstack([class_centroids[c] for c in class_names])

    D = pairwise_distances(X_scaled, centroid_mat, metric="euclidean")
    nearest_idx = D.argmin(axis=1)
    nearest_class = [class_names[i] for i in nearest_idx]

    out_df = df.copy()
    out_df["nearest_class_centroid"] = nearest_class
    out_df["matches_true_class_centroid"] = (out_df["nearest_class_centroid"] == out_df["class_id"])
    return out_df


def subject_to_class_centroid_match_rate(meta_df, centroids):
    class_centroids = {}
    for c, group in meta_df.groupby("class_id"):
        idx = group.index.values
        class_centroids[c] = centroids[idx].mean(axis=0)

    class_names = list(class_centroids.keys())
    centroid_mat = np.vstack([class_centroids[c] for c in class_names])

    D = pairwise_distances(centroids, centroid_mat, metric="euclidean")
    nearest_idx = D.argmin(axis=1)
    nearest_class = np.array([class_names[i] for i in nearest_idx])
    true_class = meta_df["class_id"].values
    return float((nearest_class == true_class).mean())

# =========================================================
# 1. LOAD DATA
# =========================================================
def load_pt_data(pt_path):
    data = torch.load(pt_path, map_location="cpu")
    if not isinstance(data, list):
        raise ValueError(f"Expected list from {pt_path}, got {type(data)}")
    print(f"Loaded {len(data)} segments from {pt_path}")
    return data


# =========================================================
# 2. BASIC GRAPH / NODE FEATURE SUMMARIES
# =========================================================
def extract_summary_features(item, use_upper_triangle=True):
    """
    Safe feature extractor that works even if number of channels differs.
    Returns one fixed-length vector per segment.
    """
    x = item["node_features"]
    adj = item["adj"]

    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(adj):
        adj = adj.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.float32)      # [N, F]
    adj = np.asarray(adj, dtype=np.float32)  # [N, N]

    N, F = x.shape

    # ---- Node feature summaries ----
    feat_mean = x.mean(axis=0)          # [F]
    feat_std = x.std(axis=0)            # [F]
    feat_min = x.min(axis=0)            # [F]
    feat_max = x.max(axis=0)            # [F]

    # ---- Adjacency summaries ----
    if use_upper_triangle:
        iu = np.triu_indices(N, k=1)
        edges = adj[iu]
    else:
        edges = adj.reshape(-1)

    # Basic edge stats
    edge_mean = np.array([edges.mean()], dtype=np.float32)
    edge_std = np.array([edges.std()], dtype=np.float32)
    edge_min = np.array([edges.min()], dtype=np.float32)
    edge_max = np.array([edges.max()], dtype=np.float32)

    # Density-like stats
    nonzero_ratio = np.array([(np.abs(edges) > 1e-8).mean()], dtype=np.float32)

    # Node strength
    node_strength = adj.sum(axis=1)
    strength_mean = np.array([node_strength.mean()], dtype=np.float32)
    strength_std = np.array([node_strength.std()], dtype=np.float32)
    strength_min = np.array([node_strength.min()], dtype=np.float32)
    strength_max = np.array([node_strength.max()], dtype=np.float32)

    # Degree proxy (nonzero count per node)
    node_degree = (np.abs(adj) > 1e-8).sum(axis=1)
    degree_mean = np.array([node_degree.mean()], dtype=np.float32)
    degree_std = np.array([node_degree.std()], dtype=np.float32)

    # Spectral summaries
    try:
        eigvals = np.linalg.eigvalsh(adj)
        eigvals = np.sort(eigvals)
        eig_summary = np.array([
            eigvals[-1],                # largest
            eigvals[-2] if len(eigvals) >= 2 else eigvals[-1],
            eigvals.mean(),
            eigvals.std(),
            eigvals.min()
        ], dtype=np.float32)
    except:
        eig_summary = np.zeros(5, dtype=np.float32)

    # Combine
    vec = np.concatenate([
        feat_mean, feat_std, feat_min, feat_max,
        edge_mean, edge_std, edge_min, edge_max,
        nonzero_ratio,
        strength_mean, strength_std, strength_min, strength_max,
        degree_mean, degree_std,
        eig_summary
    ])

    return vec


def extract_flatten_plus_summary(item):
    """
    Use only if node ordering / channel ordering is consistent across all segments.
    If channel count differs, this representation will be inconsistent unless you pad/truncate.
    """
    x = item["node_features"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    summary = extract_summary_features(item)

    flat_x = x.reshape(-1)  # [N*F]
    vec = np.concatenate([flat_x, summary])
    return vec


def pad_or_truncate_flatten(item, target_nodes=23):
    """
    Optional way to make flattening consistent if some segments have 19 and some 23 channels.
    This assumes channel ordering is compatible.
    """
    x = item["node_features"]
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    N, F = x.shape
    if N < target_nodes:
        pad = np.zeros((target_nodes - N, F), dtype=np.float32)
        x2 = np.vstack([x, pad])
    else:
        x2 = x[:target_nodes]

    summary = extract_summary_features(item)
    vec = np.concatenate([x2.reshape(-1), summary])
    return vec


# =========================================================
# 3. BUILD DATAFRAME / EMBEDDING MATRIX
# =========================================================
def build_segment_table(data, embedding_mode="summary_only", target_nodes=23):
    rows = []
    X = []

    for item in data:
        sid = item["subject_id"]
        y = item["class_id"]
        seg_id = item.get("segment_id", -1)
        start_sample = item.get("start_sample", -1)

        # choose embedding
        if embedding_mode == "summary_only":
            vec = extract_summary_features(item)
        elif embedding_mode == "flatten_plus_summary":
            vec = extract_flatten_plus_summary(item)
        elif embedding_mode == "padded_flatten_plus_summary":
            vec = pad_or_truncate_flatten(item, target_nodes=target_nodes)
        else:
            raise ValueError(f"Unknown embedding_mode: {embedding_mode}")

        X.append(vec)
        rows.append({
            "subject_id": sid,
            "class_id": y,
            "segment_id": seg_id,
            "start_sample": start_sample,
            "n_channels": item["node_features"].shape[0]
        })

    df = pd.DataFrame(rows)
    X = np.vstack(X)

    return df, X


# =========================================================
# 4. ENCODE LABELS + SCALE
# =========================================================
def prepare_features(df, X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    class_encoder = LabelEncoder()
    y_class = class_encoder.fit_transform(df["class_id"].values)

    subject_encoder = LabelEncoder()
    y_subject = subject_encoder.fit_transform(df["subject_id"].values)

    return X_scaled, y_class, y_subject, class_encoder, subject_encoder


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")

def plot_2d_embedding(Z, labels, title, savepath, filename, label_names=None, alpha=0.6, s=15):
    ensure_dir(savepath)

    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    for lab in unique_labels:
        idx = labels == lab
        name = str(lab) if label_names is None else label_names.get(lab, str(lab))
        plt.scatter(Z[idx, 0], Z[idx, 1], alpha=alpha, s=s, label=name)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(savepath, filename)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_file}")

def plot_distance_histograms(same_subject, same_class_diff_subject, diff_class, savepath, filename, bins=50):
    ensure_dir(savepath)

    plt.figure(figsize=(8, 6))
    plt.hist(same_subject, bins=bins, alpha=0.5, label="same subject", density=True)
    plt.hist(same_class_diff_subject, bins=bins, alpha=0.5, label="same class / different subject", density=True)
    plt.hist(diff_class, bins=bins, alpha=0.5, label="different class", density=True)
    plt.xlabel("Euclidean distance")
    plt.ylabel("Density")
    plt.title("Pairwise distance distributions")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(savepath, filename)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_file}")

def plot_within_subject_heterogeneity(subject_df, savepath, filename):
    ensure_dir(savepath)

    plt.figure(figsize=(8, 6))
    for c in sorted(subject_df["class_id"].unique()):
        vals = subject_df.loc[subject_df["class_id"] == c, "within_subject_mean_dist"].values
        plt.hist(vals, bins=20, alpha=0.5, label=f"class {c}")

    plt.xlabel("Mean distance to subject centroid")
    plt.ylabel("Count")
    plt.title("Within-subject heterogeneity by class")
    plt.legend()
    plt.tight_layout()

    out_file = os.path.join(savepath, filename)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_file}")


import numpy as np

def compute_pairwise_distance_stats_sampled(
    X_scaled,
    y_class,
    y_subject,
    max_pairs_per_group=30000,
    random_state=42
):
    rng = np.random.default_rng(random_state)
    n = len(X_scaled)

    same_subject = []
    same_class_diff_subject = []
    diff_class = []

    # sample random pairs instead of all O(n^2) pairs
    # keep drawing until enough samples are collected
    max_trials = max_pairs_per_group * 80
    trials = 0

    while trials < max_trials:
        i = rng.integers(0, n)
        j = rng.integers(0, n)
        trials += 1

        if i == j:
            continue
        if i > j:
            i, j = j, i

        d = np.linalg.norm(X_scaled[i] - X_scaled[j])

        if y_subject[i] == y_subject[j]:
            if len(same_subject) < max_pairs_per_group:
                same_subject.append(d)
        elif y_class[i] == y_class[j]:
            if len(same_class_diff_subject) < max_pairs_per_group:
                same_class_diff_subject.append(d)
        else:
            if len(diff_class) < max_pairs_per_group:
                diff_class.append(d)

        if (
            len(same_subject) >= max_pairs_per_group and
            len(same_class_diff_subject) >= max_pairs_per_group and
            len(diff_class) >= max_pairs_per_group
        ):
            break

    stats = {
        "same_subject_mean": np.mean(same_subject) if same_subject else np.nan,
        "same_subject_std": np.std(same_subject) if same_subject else np.nan,
        "same_class_diff_subject_mean": np.mean(same_class_diff_subject) if same_class_diff_subject else np.nan,
        "same_class_diff_subject_std": np.std(same_class_diff_subject) if same_class_diff_subject else np.nan,
        "diff_class_mean": np.mean(diff_class) if diff_class else np.nan,
        "diff_class_std": np.std(diff_class) if diff_class else np.nan,
        "n_same_subject": len(same_subject),
        "n_same_class_diff_subject": len(same_class_diff_subject),
        "n_diff_class": len(diff_class),
    }

    return stats, same_subject, same_class_diff_subject, diff_class



def compute_between_within_scatter_ratio(X, y):
    classes = np.unique(y)
    mu = X.mean(axis=0)

    sw = 0.0
    sb = 0.0

    for c in classes:
        Xc = X[y == c]
        muc = Xc.mean(axis=0)
        sw += ((Xc - muc) ** 2).sum()
        sb += len(Xc) * ((muc - mu) ** 2).sum()

    if sw <= 1e-12:
        return np.nan
    return float(sb / sw)

# def run_full_analysis(
#     pt_path,
#     savepath,
#     embedding_mode="summary_only",
#     target_nodes=23,
#     use_umap=True,
#     knn_k=10
# ):
#     ensure_dir(savepath)

#     print("=" * 60)
#     print("Loading data")
#     print("=" * 60)
#     data = load_pt_data(pt_path)
#     # data = torch.load(pt_path, map_location="cpu")

#     print("=" * 60)
#     print("Building segment table")
#     print("=" * 60)
#     df, X = build_segment_table(
#         data,
#         embedding_mode=embedding_mode,
#         target_nodes=target_nodes
#     )

#     print("Segment table shape:", df.shape)
#     print("Feature matrix shape:", X.shape)
#     print(df.head())

#     print("=" * 60)
#     print("Preparing features")
#     print("=" * 60)
#     X_scaled, y_class, y_subject, class_encoder, subject_encoder = prepare_features(df, X)

#     # ------------------------------
#     # PCA on segments
#     # ------------------------------
#     # print("=" * 60)
#     # print("PCA on segments")
#     # print("=" * 60)
#     # Z_pca, pca_model = run_pca(X_scaled, n_components=2)

#     # class_name_map = {i: c for i, c in enumerate(class_encoder.classes_)}

#     # plot_2d_embedding(
#     #     Z_pca, y_class,
#     #     title="PCA of segments colored by class",
#     #     savepath=savepath,
#     #     filename="pca_segments_by_class.png",
#     #     label_names=class_name_map
#     # )

#     # plot_2d_embedding(
#     #     Z_pca, y_subject,
#     #     title="PCA of segments colored by subject",
#     #     savepath=savepath,
#     #     filename="pca_segments_by_subject.png"
#     # )

#     # ------------------------------
#     # UMAP on segments
#     # ------------------------------
#     # if use_umap and HAS_UMAP:
#     #     print("=" * 60)
#     #     print("UMAP on segments")
#     #     print("=" * 60)
#     #     Z_umap, _ = run_umap(X_scaled)

#     #     plot_2d_embedding(
#     #         Z_umap, y_class,
#     #         title="UMAP of segments colored by class",
#     #         savepath=savepath,
#     #         filename="umap_segments_by_class.png",
#     #         label_names=class_name_map
#     #     )

#     #     plot_2d_embedding(
#     #         Z_umap, y_subject,
#     #         title="UMAP of segments colored by subject",
#     #         savepath=savepath,
#     #         filename="umap_segments_by_subject.png"
#     #     )

#     # ------------------------------
#     # Metrics
#     # ------------------------------
#     # print("=" * 60)
#     # print("Clustering / overlap metrics")
#     # print("=" * 60)
#     sil = compute_silhouette_by_class(X_scaled, y_class)
#     # print(f"Silhouette score by class: {sil:.4f}")

#     knn_stats = compute_knn_purity(X_scaled, y_class, y_subject, k=knn_k)
#     # print("kNN purity stats:")
#     # for k, v in knn_stats.items():
#     #     print(f"  {k}: {v:.4f}")

#     dist_stats, same_subject, same_class_diff_subject, diff_class = compute_pairwise_distance_stats(
#         X_scaled, y_class, y_subject
#     )
#     # print("Distance stats:")
#     # for k, v in dist_stats.items():
#     #     print(f"  {k}: {v:.4f}")

#     # plot_distance_histograms(
#     #     same_subject, same_class_diff_subject, diff_class,
#     #     savepath=savepath,
#     #     filename="pairwise_distance_histograms.png"
#     # )

#     # ------------------------------
#     # Subject-level heterogeneity
#     # ------------------------------
#     print("=" * 60)
#     print("Subject-level heterogeneity")
#     print("=" * 60)
#     subject_df = analyze_subject_heterogeneity(df, X_scaled, y_class)
#     print(subject_df.sort_values("within_subject_mean_dist", ascending=False).head(10))

#     plot_within_subject_heterogeneity(
#         subject_df,
#         savepath=savepath,
#         filename="within_subject_heterogeneity.png"
#     )

#     # ------------------------------
#     # Subject centroid analysis
#     # ------------------------------
#     print("=" * 60)
#     print("Subject centroid analysis")
#     print("=" * 60)
#     meta_df, centroids = build_subject_centroids(df, X_scaled)
#     y_sub_class = LabelEncoder().fit_transform(meta_df["class_id"].values)

#     Z_sub_pca, _ = run_pca(centroids, n_components=2)
#     class_names_sub = {i: c for i, c in enumerate(sorted(meta_df["class_id"].unique()))}

#     plot_2d_embedding(
#         Z_sub_pca, y_sub_class,
#         title="PCA of subject centroids colored by class",
#         savepath=savepath,
#         filename="pca_subject_centroids_by_class.png",
#         label_names=class_names_sub
#     )

#     if use_umap and HAS_UMAP:
#         Z_sub_umap, _ = run_umap(centroids)
#         plot_2d_embedding(
#             Z_sub_umap, y_sub_class,
#             title="UMAP of subject centroids colored by class",
#             savepath=savepath,
#             filename="umap_subject_centroids_by_class.png",
#             label_names=class_names_sub
#         )

#     if len(np.unique(y_sub_class)) >= 2:
#         sil_sub = silhouette_score(centroids, y_sub_class)
#         print(f"Subject centroid silhouette by class: {sil_sub:.4f}")

#     # ------------------------------
#     # Segment to class centroid analysis
#     # ------------------------------
#     print("=" * 60)
#     print("Segment nearest class centroid analysis")
#     print("=" * 60)
#     nearest_df = segment_to_class_centroid_analysis(df, X_scaled)
#     match_rate = nearest_df["matches_true_class_centroid"].mean()
#     print(f"Fraction of segments nearest to their true class centroid: {match_rate:.4f}")

#     subject_mix = (
#         nearest_df.groupby(["subject_id", "class_id", "nearest_class_centroid"])
#         .size()
#         .reset_index(name="count")
#     )
#     print("Example subject centroid-mix table:")
#     print(subject_mix.head(20))

#     return {
#         "segment_df": df,
#         "X_raw": X,
#         "X_scaled": X_scaled,
#         "subject_df": subject_df,
#         "nearest_df": nearest_df,
#         "knn_stats": knn_stats,
#         "distance_stats": dist_stats,
#         "segment_silhouette": sil
#     }


# =========================================================
# main per-file analysis
# =========================================================
def analyze_single_pt(
    pt_path,
    savepath,
    embedding_mode="summary_only",
    target_nodes=23,
    use_umap=False,
    knn_k=10,
    max_pairs_per_group=20000,
    classifier_name="logreg",
    n_splits=5,
):
    ensure_dir(savepath)


    last_part = os.path.basename(pt_path)
    parts = last_part.split('_')
    # data_processed_path = os.path.join(pt_path, "data_processed")
    # data_processed_path = pt_path
    
    all_data_path = f"{pt_path}/master_graph_data.pt"



    data = load_pt_data(all_data_path)
    df, X = build_segment_table(
        data,
        embedding_mode=embedding_mode,
        target_nodes=target_nodes,
    )

    X_scaled, y_class, y_subject, class_encoder, subject_encoder = prepare_features(df, X)


    try:
        node_features = parts[1]
        weight_method = parts[2:]
    except ValueError:
        node_features = parts[0]
        weight_method = parts[1:3]
    

    metrics = {
        "pt_path": last_part,
        "feature_name": node_features,
        "n_segments": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_classes": int(df["class_id"].nunique()),
        "embedding_mode": embedding_mode,
    }

    # PCA
    Z_pca, pca_model = run_pca(X_scaled, n_components=2)
    metrics["pca_var_1"] = float(pca_model.explained_variance_ratio_[0])
    metrics["pca_var_2"] = float(pca_model.explained_variance_ratio_[1])

    class_name_map = {i: c for i, c in enumerate(class_encoder.classes_)}
    # plot_2d_embedding(
    #     Z_pca, y_class,
    #     title="PCA of segments colored by class",
    #     savepath=savepath,
    #     filename="pca_segments_by_class.png",
    #     label_names=class_name_map
    # )
    # plot_2d_embedding(
    #     Z_pca, y_subject,
    #     title="PCA of segments colored by subject",
    #     savepath=savepath,
    #     filename="pca_segments_by_subject.png"
    # )

    if use_umap and HAS_UMAP:
        Z_umap, _ = run_umap(X_scaled)
        # plot_2d_embedding(
        #     Z_umap, y_class,
        #     title="UMAP of segments colored by class",
        #     savepath=savepath,
        #     filename="umap_segments_by_class.png",
        #     label_names=class_name_map
        # )
        # plot_2d_embedding(
        #     Z_umap, y_subject,
        #     title="UMAP of segments colored by subject",
        #     savepath=savepath,
        #     filename="umap_segments_by_subject.png"
        # )

    # segment-level screening
    if len(np.unique(y_class)) >= 2:
        metrics["segment_silhouette_by_class"] = safe_float(silhouette_score(X_scaled, y_class))
    else:
        metrics["segment_silhouette_by_class"] = np.nan

    metrics["bw_scatter_ratio_segments"] = compute_between_within_scatter_ratio(X_scaled, y_class)

    knn_stats = compute_knn_purity(X_scaled, y_class, y_subject, k=knn_k)
    metrics.update(knn_stats)

    dist_stats, same_subject, same_class_diff_subject, diff_class = compute_pairwise_distance_stats_sampled(
        X_scaled,
        y_class,
        y_subject,
        max_pairs_per_group=max_pairs_per_group,
        random_state=42,
    )
    metrics.update(dist_stats)

    # plot_distance_histograms(
    #     same_subject,
    #     same_class_diff_subject,
    #     diff_class,
    #     savepath=savepath,
    #     filename="pairwise_distance_histograms.png",
    # )

    nearest_df = segment_to_class_centroid_analysis(df, X_scaled)
    metrics["segment_centroid_match_rate"] = float(nearest_df["matches_true_class_centroid"].mean())
    nearest_df.to_csv(os.path.join(savepath, "segment_nearest_centroid.csv"), index=False)

    # subject-level summaries
    subject_df = analyze_subject_heterogeneity(df, X_scaled)
    subject_df.to_csv(os.path.join(savepath, "subject_heterogeneity.csv"), index=False)
    # plot_within_subject_heterogeneity(
    #     subject_df,
    #     savepath=savepath,
    #     filename="within_subject_heterogeneity.png",
    # )
    metrics["within_subject_mean_dist_mean"] = float(subject_df["within_subject_mean_dist"].mean())
    metrics["within_subject_mean_dist_std"] = float(subject_df["within_subject_mean_dist"].std())

    meta_df, centroids = build_subject_centroids(df, X_scaled)
    meta_df.to_csv(os.path.join(savepath, "subject_centroids_meta.csv"), index=False)

    y_sub_class = LabelEncoder().fit_transform(meta_df["class_id"].values)
    Z_sub_pca, sub_pca = run_pca(centroids, n_components=2)
    metrics["subject_pca_var_1"] = float(sub_pca.explained_variance_ratio_[0])
    metrics["subject_pca_var_2"] = float(sub_pca.explained_variance_ratio_[1])

    class_names_sub = {i: c for i, c in enumerate(sorted(meta_df["class_id"].unique()))}
    # plot_2d_embedding(
    #     Z_sub_pca,
    #     y_sub_class,
    #     title="PCA of subject centroids colored by class",
    #     savepath=savepath,
    #     filename="pca_subject_centroids_by_class.png",
    #     label_names=class_names_sub,
    # )

    if use_umap and HAS_UMAP:
        Z_sub_umap, _ = run_umap(centroids)
        # plot_2d_embedding(
        #     Z_sub_umap,
        #     y_sub_class,
        #     title="UMAP of subject centroids colored by class",
        #     savepath=savepath,
        #     filename="umap_subject_centroids_by_class.png",
        #     label_names=class_names_sub,
        # )

    if len(np.unique(y_sub_class)) >= 2:
        metrics["subject_centroid_silhouette_by_class"] = safe_float(silhouette_score(centroids, y_sub_class))
    else:
        metrics["subject_centroid_silhouette_by_class"] = np.nan

    metrics["bw_scatter_ratio_subjects"] = compute_between_within_scatter_ratio(centroids, y_sub_class)
    metrics["subject_centroid_match_rate"] = subject_to_class_centroid_match_rate(meta_df, centroids)

    # probe classifier
    probe_metrics, fold_df = run_subject_probe_cv(
        meta_df,
        centroids,
        classifier_name=classifier_name,
        n_splits=n_splits,
        random_state=42,
    )
    metrics.update(probe_metrics)

    if fold_df is not None:
        fold_df.to_csv(os.path.join(savepath, "probe_cv_folds.csv"), index=False)

    # final combined score
    metrics["combined_feature_score"] = compute_combined_feature_score(metrics)

    # save all outputs
    # pd.DataFrame([metrics]).to_csv(os.path.join(savepath, "feature_screening_metrics.csv"), index=False)
    # for k, v in metrics.items():
    #     print(k, type(v))

    # save_json(metrics, os.path.join(savepath, "feature_screening_metrics.json"))

    # print("\n===== feature screening metrics =====")
    # for k, v in metrics.items():
    #     print(f"{k}: {v}")

    return metrics

# =========================================================
# subject-level probe classifier
# =========================================================
def run_subject_probe_cv(meta_df, centroids, classifier_name="logreg", n_splits=5, random_state=42):
    y = meta_df["class_id"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    class_counts = pd.Series(y_enc).value_counts()
    min_class_count = int(class_counts.min())
    n_splits = min(n_splits, min_class_count)

    if n_splits < 2:
        return {
            "probe_acc_mean": np.nan,
            "probe_acc_std": np.nan,
            "probe_bal_acc_mean": np.nan,
            "probe_bal_acc_std": np.nan,
            "probe_macro_f1_mean": np.nan,
            "probe_macro_f1_std": np.nan,
            "probe_n_splits": n_splits,
        }, None

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs = []
    bals = []
    f1s = []
    fold_rows = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(centroids, y_enc)):
        Xtr, Xte = centroids[tr_idx], centroids[te_idx]
        ytr, yte = y_enc[tr_idx], y_enc[te_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xte = scaler.transform(Xte)

        if classifier_name == "logreg":
            clf = LogisticRegression(
                max_iter=3000,
                multi_class="auto",
                class_weight="balanced",
                random_state=random_state,
            )
        elif classifier_name == "linearsvm":
            clf = LinearSVC(
                class_weight="balanced",
                random_state=random_state,
                max_iter=5000,
            )
        else:
            raise ValueError(f"Unknown classifier_name: {classifier_name}")

        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)

        acc = accuracy_score(yte, pred)
        bal = balanced_accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")

        accs.append(acc)
        bals.append(bal)
        f1s.append(f1)

        fold_rows.append({
            "fold": fold,
            "acc": acc,
            "balanced_acc": bal,
            "macro_f1": f1,
        })

    metrics = {
        "probe_acc_mean": float(np.mean(accs)),
        "probe_acc_std": float(np.std(accs)),
        "probe_bal_acc_mean": float(np.mean(bals)),
        "probe_bal_acc_std": float(np.std(bals)),
        "probe_macro_f1_mean": float(np.mean(f1s)),
        "probe_macro_f1_std": float(np.std(f1s)),
        "probe_n_splits": int(n_splits),
    }
    fold_df = pd.DataFrame(fold_rows)
    return metrics, fold_df





# =========================================================
# combined score
# =========================================================
def compute_combined_feature_score(metrics):
    """
    Higher is better.
    This is a heuristic ranking score for comparing feature sets.
    """
    score = 0.0

    score += 2.0 * safe_float(metrics.get("probe_macro_f1_mean", np.nan))
    score += 1.5 * safe_float(metrics.get("probe_bal_acc_mean", np.nan))
    score += 1.0 * safe_float(metrics.get("knn_class_purity_mean", np.nan))
    score += 1.0 * safe_float(metrics.get("segment_centroid_match_rate", np.nan))
    score += 0.8 * safe_float(metrics.get("subject_centroid_match_rate", np.nan))
    score += 0.8 * safe_float(metrics.get("bw_scatter_ratio_segments", np.nan))
    score += 0.8 * safe_float(metrics.get("bw_scatter_ratio_subjects", np.nan))

    score -= 0.8 * safe_float(metrics.get("knn_subject_purity_mean", np.nan))
    score -= 0.5 * safe_float(metrics.get("probe_macro_f1_std", np.nan))
    score -= 0.3 * safe_float(metrics.get("probe_bal_acc_std", np.nan))

    return float(score)


# =========================================================
# append one row into global summary csv
# =========================================================
def append_summary_row(metrics, summary_csv_path):
    row_df = pd.DataFrame([metrics])

    if os.path.exists(summary_csv_path):
        old = pd.read_csv(summary_csv_path)
        new = pd.concat([old, row_df], ignore_index=True)
        # keep latest unique path
        new = new.drop_duplicates(subset=["pt_path"], keep="last")
    else:
        new = row_df

    new.to_csv(summary_csv_path, index=False)
    print(f"Updated summary: {summary_csv_path}")





# if __name__ == "__main__":


#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pt_path", type=str, required=True)
#     # parser.add_argument("--savepath", type=str, required=True)
#     # parser.add_argument("--summary_csv", type=str, required=False, default="")
#     parser.add_argument("--embedding_mode", type=str, default="summary_only",
#                         choices=["summary_only", "padded_flatten_plus_summary"])
#     parser.add_argument("--target_nodes", type=int, default=23)
#     parser.add_argument("--use_umap", action="store_true")
#     parser.add_argument("--knn_k", type=int, default=10)
#     parser.add_argument("--max_pairs_per_group", type=int, default=20000)
#     parser.add_argument("--classifier_name", type=str, default="logreg",
#                         choices=["logreg", "linearsvm"])
#     parser.add_argument("--n_splits", type=int, default=5)

#     args = parser.parse_args()

#     root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"
#     save_path = os.path.join(root_path,'vis')
#     os.makedirs(save_path,exist_ok = True)
#     summary_csv = os.path.join(save_path, "summary.csv")

#     saved_subject_dir = os.path.join(root_path,f'all_master_graph_data/{args.pt_path}')
#     last_part = os.path.basename(saved_subject_dir)
#     parts = last_part.split('_')
#     data_processed_path = saved_subject_dir
#     # data_processed_path = os.path.join(saved_subject_dir, "data_processed")
#     all_data_path = f"{data_processed_path}/master_graph_data.pt"

#     if not os.path.exists(all_data_path):
#         raise FileNotFoundError(f"Missing: {all_data_path}")
#     if not os.path.exists(all_data_path):
#         print(f"Skipping: {all_data_path} not found.")
#         sys.exit(1) 

#     print("File found! Processing...")

#     folder_path = os.path.join(save_path,last_part)
#     os.makedirs(folder_path,exist_ok = True)

#     ensure_dir(folder_path)

#     metrics = analyze_single_pt(
#         pt_path=saved_subject_dir,
#         savepath=folder_path,
#         embedding_mode=args.embedding_mode,
#         target_nodes=args.target_nodes,
#         use_umap=args.use_umap,
#         knn_k=args.knn_k,
#         max_pairs_per_group=args.max_pairs_per_group,
#         classifier_name=args.classifier_name,
#         n_splits=args.n_splits,
#     )

#     if summary_csv:
#         append_summary_row(metrics, summary_csv)



    # args = parser.parse_args()
    # # class_set = args.class_set
    # topology = args.topology
    # use_fake_label = args.use_fake_label
    # add_noise = args.add_noise
    # if add_noise:
    #     noise_ratio = args.noise_ratio
    # else:
    #     noise_ratio = None

    # # use_fake_label = False
    # # fake_label_method = "fake_noise_segment_label"
    # # topology = "fixed"
    # # noise_ratio = 0.5
    # fake_label_method = f"fakelabel_noise{add_noise}_{noise_ratio}" if use_fake_label ==True else f"reallabel_noise{add_noise}_{noise_ratio}"

    # if topology == "topk":
    #     topk=3
    # else:
    #     topk=None


    # num_classes, class_labels, class_names = get_class(class_set, dataset)
    # data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    # print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)

    # root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"
    # save_path = os.path.join(root_path,'vis')
    # os.makedirs(save_path,exist_ok = True)

    # saved_subject_dir = os.path.join(root_path,'all_master_graph_data/mono_rbphjorth_coherence_None')
    # last_part = os.path.basename(saved_subject_dir)
    # parts = last_part.split('_')

    # try:
    #     node_features = parts[1]
    #     weight_method = parts[2:]
    #     _ = get_feature_dim_from_string(feature_dim_dict, node_features)
    # except ValueError:
    #     node_features = parts[0]
    #     weight_method = parts[1:3]
    

    # timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # feat, used_features = get_feature_dim_from_string(feature_dim_dict, node_features)
    # folder_name = f"{timestamp}_{model_name}_{last_part}"
    # output_dir = os.path.join(save_path, folder_name)
    # os.makedirs(output_dir,exist_ok = True)
    # log_path = os.path.join(output_dir, f"log.txt")
    
    # data_processed_path = os.path.join(saved_subject_dir, "data_processed")
    # all_data_path = f"{data_processed_path}/master_graph_data.pt"

    # if not os.path.exists(all_data_path):
    #     raise FileNotFoundError(f"Missing: {all_data_path}")
    # if not os.path.exists(all_data_path):
    #     print(f"Skipping: {all_data_path} not found.")
    #     sys.exit(1) 

    # print("File found! Processing...")

    # results = run_full_analysis(
    #     pt_path=all_data_path,
    #     savepath=output_dir,
    #     embedding_mode="summary_only",
    #     use_umap=True,
    #     knn_k=10
    # )