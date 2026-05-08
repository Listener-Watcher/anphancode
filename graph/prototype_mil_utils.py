# prototype_mil_utils.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from torch_geometric.data import Data

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, List

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
# ---------------------------------------------------------
# Default CAUEEG 19-channel region definition
# Use case-insensitive matching, so FZ/Fz both work.
# ---------------------------------------------------------
DEFAULT_REGION_TO_CHANNELS_MONO: Dict[str, List[str]] = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "FZ"],
    "central":   ["C3", "C4", "CZ"],
    "parietal":  ["P3", "P4", "PZ"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "occipital": ["O1", "O2"],
}


@dataclass
class SegmentPrototypeModel:
    """
    Stores the fitted prototype transform.

    The KMeans input vector for each segment is:

        concat(
            flattened node features,        [N * F]
            region-wise mean node features  [R * F]
        )

    where R is the number of brain regions.
    """
    scaler: StandardScaler
    kmeans: MiniBatchKMeans
    pca: Optional[PCA]
    region_to_indices: Dict[str, List[int]]
    region_names: List[str]
    channel_names: List[str]
    input_dim: int
    pca_dim: Optional[int]
    num_prototypes: int


def _normalize_name(x: str) -> str:
    return str(x).strip().lower()


def build_region_to_indices(
    channel_names: Sequence[str],
    region_to_channels: Mapping[str, Sequence[str]],
) -> Tuple[Dict[str, List[int]], List[str]]:
    """
    Convert region -> channel names into region -> node indices.
    Matching is case-insensitive.
    """
    channel_names = [str(c) for c in channel_names]
    name_to_idx = {_normalize_name(ch): i for i, ch in enumerate(channel_names)}

    region_to_indices: Dict[str, List[int]] = {}
    region_names: List[str] = []

    for region, chs in region_to_channels.items():
        idxs = []
        for ch in chs:
            key = _normalize_name(ch)
            if key in name_to_idx:
                idxs.append(name_to_idx[key])

        if len(idxs) == 0:
            # Do not crash, but skip empty region.
            # This is useful when switching mono / bipolar / custom channel names.
            continue

        region = str(region)
        region_to_indices[region] = idxs
        region_names.append(region)

    if len(region_names) == 0:
        raise ValueError(
            "No valid regions found. Check channel_names and region_to_channels."
        )

    return region_to_indices, region_names


def _to_numpy_x(g: Data) -> np.ndarray:
    x = g.x
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"Expected g.x shape [N, F], got {x.shape}")

    return x


def _safe_zscore_segment_x(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Standardize each feature dimension across nodes inside one segment.

    This keeps the prototype vector focused on spatial/relative node pattern,
    not just raw feature scale.
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return ((x - mu) / (sd + eps)).astype(np.float32)


def segment_region_mean_features(
    x: np.ndarray,
    region_to_indices: Mapping[str, Sequence[int]],
    region_names: Sequence[str],
) -> np.ndarray:
    """
    x: [N, F]

    returns:
        region_mean_flat: [R * F]
    """
    region_means = []

    for region in region_names:
        idxs = list(region_to_indices[region])
        region_x = x[idxs]  # [num_region_nodes, F]
        region_means.append(region_x.mean(axis=0))

    return np.concatenate(region_means, axis=0).astype(np.float32)


def segment_prototype_vector(
    g: Data,
    *,
    region_to_indices: Mapping[str, Sequence[int]],
    region_names: Sequence[str],
    include_node_flat: bool = True,
    include_region_mean: bool = True,
    zscore_segment: bool = True,
) -> np.ndarray:
    """
    Build the KMeans input vector for one segment.

    Uses:
        1) flattened node features
        2) region-aware mean node features
    """
    x = _to_numpy_x(g)  # [N, F]

    if zscore_segment:
        x = _safe_zscore_segment_x(x)

    parts = []

    if include_node_flat:
        parts.append(x.reshape(-1))

    if include_region_mean:
        region_mean = segment_region_mean_features(
            x,
            region_to_indices=region_to_indices,
            region_names=region_names,
        )
        parts.append(region_mean)

    if len(parts) == 0:
        raise ValueError("At least one of include_node_flat or include_region_mean must be True.")

    return np.concatenate(parts, axis=0).astype(np.float32)


def collect_prototype_matrix(
    graphs: Sequence[Data],
    *,
    region_to_indices: Mapping[str, Sequence[int]],
    region_names: Sequence[str],
    include_node_flat: bool = True,
    include_region_mean: bool = True,
    zscore_segment: bool = True,
) -> np.ndarray:
    rows = [
        segment_prototype_vector(
            g,
            region_to_indices=region_to_indices,
            region_names=region_names,
            include_node_flat=include_node_flat,
            include_region_mean=include_region_mean,
            zscore_segment=zscore_segment,
        )
        for g in graphs
    ]

    if len(rows) == 0:
        raise ValueError("No graphs provided for prototype fitting.")

    return np.vstack(rows).astype(np.float32)


def fit_segment_prototype_model(
    train_graphs: Sequence[Data],
    *,
    channel_names: Sequence[str],
    region_to_channels: Mapping[str, Sequence[str]] = DEFAULT_REGION_TO_CHANNELS_MONO,
    num_prototypes: int = 8,
    pca_dim: Optional[int] = 32,
    seed: int = 42,
    include_node_flat: bool = True,
    include_region_mean: bool = True,
    zscore_segment: bool = True,
    batch_size: int = 2048,
) -> SegmentPrototypeModel:
    """
    Fit scaler/PCA/KMeans on training graphs only.

    Important:
        For CV, call this separately inside each fold using only fold-train graphs.
    """
    if num_prototypes < 2:
        raise ValueError("num_prototypes should be >= 2.")

    region_to_indices, region_names = build_region_to_indices(
        channel_names=channel_names,
        region_to_channels=region_to_channels,
    )

    Z = collect_prototype_matrix(
        train_graphs,
        region_to_indices=region_to_indices,
        region_names=region_names,
        include_node_flat=include_node_flat,
        include_region_mean=include_region_mean,
        zscore_segment=zscore_segment,
    )

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    pca = None
    final_pca_dim = None

    if pca_dim is not None:
        max_dim = min(Zs.shape[0] - 1, Zs.shape[1])
        if max_dim >= 2:
            final_pca_dim = min(int(pca_dim), max_dim)
            pca = PCA(n_components=final_pca_dim, random_state=seed)
            Zs = pca.fit_transform(Zs)

    kmeans = MiniBatchKMeans(
        n_clusters=int(num_prototypes),
        random_state=seed,
        n_init=20,
        batch_size=int(batch_size),
        reassignment_ratio=0.01,
    )
    kmeans.fit(Zs)

    return SegmentPrototypeModel(
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        region_to_indices=region_to_indices,
        region_names=region_names,
        channel_names=list(channel_names),
        input_dim=int(Z.shape[1]),
        pca_dim=final_pca_dim,
        num_prototypes=int(num_prototypes),
    )


def _soft_assignment_from_distances(
    dist: np.ndarray,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    dist: [G, K]
    returns soft assignment [G, K]
    """
    temperature = max(float(temperature), eps)
    logits = -dist / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    expv = np.exp(logits)
    return (expv / np.clip(expv.sum(axis=1, keepdims=True), eps, None)).astype(np.float32)


def attach_segment_prototypes(
    graphs: Sequence[Data],
    proto_model: SegmentPrototypeModel,
    *,
    temperature: float = 1.0,
    include_node_flat: bool = True,
    include_region_mean: bool = True,
    zscore_segment: bool = True,
) -> Sequence[Data]:
    """
    Attach prototype info to every PyG graph.

    Added fields:
        g.proto_id        : LongTensor [1]
        g.proto_soft      : FloatTensor [1, K]
        g.proto_dist      : FloatTensor [1, K]
        g.proto_dist_log  : FloatTensor [1, K]

    Use shape [1, K], not [K], so PyG Batch concatenates to [G, K].
    """
    if len(graphs) == 0:
        return graphs

    Z = collect_prototype_matrix(
        graphs,
        region_to_indices=proto_model.region_to_indices,
        region_names=proto_model.region_names,
        include_node_flat=include_node_flat,
        include_region_mean=include_region_mean,
        zscore_segment=zscore_segment,
    )

    Zs = proto_model.scaler.transform(Z)
    if proto_model.pca is not None:
        Zs = proto_model.pca.transform(Zs)

    dist = proto_model.kmeans.transform(Zs).astype(np.float32)  # [G, K]
    proto_id = dist.argmin(axis=1).astype(np.int64)
    proto_soft = _soft_assignment_from_distances(dist, temperature=temperature)
    proto_dist_log = np.log1p(dist).astype(np.float32)

    for i, g in enumerate(graphs):
        g.proto_id = torch.tensor([proto_id[i]], dtype=torch.long)
        g.proto_soft = torch.tensor(proto_soft[i][None, :], dtype=torch.float32)
        g.proto_dist = torch.tensor(dist[i][None, :], dtype=torch.float32)
        g.proto_dist_log = torch.tensor(proto_dist_log[i][None, :], dtype=torch.float32)

    return graphs


def summarize_prototype_usage(
    graphs: Sequence[Data],
    num_prototypes: int,
) -> Dict[str, Any]:
    """
    Useful sanity check after attaching prototypes.
    """
    ids = []
    labels = []

    for g in graphs:
        if not hasattr(g, "proto_id"):
            continue
        ids.append(int(g.proto_id.view(-1)[0].item()))

        if hasattr(g, "y") and g.y is not None:
            labels.append(int(g.y.view(-1)[0].item()))
        else:
            labels.append(-1)

    counts = np.bincount(np.asarray(ids, dtype=np.int64), minlength=num_prototypes)

    return {
        "num_graphs": len(ids),
        "num_prototypes": int(num_prototypes),
        "prototype_counts": counts.tolist(),
        "prototype_ratios": (counts / max(counts.sum(), 1)).tolist(),
    }

# prototype_viz_utils.py


def _ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_numpy_x(g):
    x = g.x
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _safe_zscore_segment_x(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return ((x - mu) / (sd + eps)).astype(np.float32)


def _get_graph_label(g) -> int:
    if hasattr(g, "y") and g.y is not None:
        return int(g.y.view(-1)[0].item())
    return -1


def _get_proto_id(g) -> int:
    if not hasattr(g, "proto_id"):
        raise AttributeError("Graph is missing proto_id. Run attach_segment_prototypes(...) first.")
    return int(g.proto_id.view(-1)[0].item())


def _get_subject_id(g) -> str:
    return str(getattr(g, "subject_id", ""))


def _get_segment_id(g) -> int:
    return int(getattr(g, "segment_id", -1))


def _get_start_sample(g) -> int:
    return int(getattr(g, "start_sample", -1))


def make_prototype_segment_table(graphs: Sequence[Any]) -> pd.DataFrame:
    rows = []

    for g in graphs:
        proto_id = _get_proto_id(g)
        dist_to_proto = None

        if hasattr(g, "proto_dist"):
            d = g.proto_dist.view(-1).detach().cpu().numpy()
            if proto_id < len(d):
                dist_to_proto = float(d[proto_id])

        rows.append({
            "subject_id": _get_subject_id(g),
            "segment_id": _get_segment_id(g),
            "start_sample": _get_start_sample(g),
            "true_label": _get_graph_label(g),
            "proto_id": proto_id,
            "dist_to_proto": dist_to_proto,
        })

    return pd.DataFrame(rows)


def plot_prototype_scatter(
    graphs: Sequence[Any],
    proto_model,
    save_path,
    *,
    color_by: str = "proto_id",   # "proto_id" or "true_label"
    method: str = "pca",          # "pca" or "tsne"
    title: Optional[str] = None,
    max_points: Optional[int] = 5000,
    seed: int = 42,
):
    """
    Plot segment prototype vectors in 2D.

    This uses the same transformed feature space as KMeans:
        node-flat + region-mean -> scaler -> optional PCA.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    Z = collect_prototype_matrix(
        graphs,
        region_to_indices=proto_model.region_to_indices,
        region_names=proto_model.region_names,
        include_node_flat=True,
        include_region_mean=True,
        zscore_segment=True,
    )

    Z = proto_model.scaler.transform(Z)

    if proto_model.pca is not None:
        Z = proto_model.pca.transform(Z)

    meta = make_prototype_segment_table(graphs)

    if max_points is not None and len(meta) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(meta), size=max_points, replace=False)
        Z = Z[idx]
        meta = meta.iloc[idx].reset_index(drop=True)

    if method.lower() == "pca":
        reducer = PCA(n_components=2, random_state=seed)
    elif method.lower() == "tsne":
        reducer = TSNE(
            n_components=2,
            random_state=seed,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, len(meta) // 20)),
        )
    else:
        raise ValueError("method must be 'pca' or 'tsne'.")

    Z2 = reducer.fit_transform(Z)

    y = meta[color_by].to_numpy(dtype=int)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(Z2[:, 0], Z2[:, 1], c=y, s=16, alpha=0.75)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title or f"Prototype segment space colored by {color_by} ({method.upper()})")
    ax.grid(True, alpha=0.3)

    legend = ax.legend(*sc.legend_elements(), title=color_by, loc="best", fontsize=8)
    ax.add_artist(legend)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(save_path)


def plot_prototype_class_distribution(
    graphs: Sequence[Any],
    save_path,
    *,
    num_prototypes: int,
    class_names: Optional[Sequence[str]] = None,
    normalize: str = "row",  # "row", "col", or "none"
    title: str = "Class distribution by prototype",
):
    """
    Heatmap: prototype_id x class.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_prototype_segment_table(graphs)

    counts = pd.crosstab(df["proto_id"], df["true_label"])
    counts = counts.reindex(index=range(num_prototypes), fill_value=0)

    classes = sorted(df["true_label"].unique())
    counts = counts.reindex(columns=classes, fill_value=0)

    mat = counts.to_numpy(dtype=np.float64)

    if normalize == "row":
        denom = np.clip(mat.sum(axis=1, keepdims=True), 1e-12, None)
        plot_mat = mat / denom
        fmt = ".2f"
    elif normalize == "col":
        denom = np.clip(mat.sum(axis=0, keepdims=True), 1e-12, None)
        plot_mat = mat / denom
        fmt = ".2f"
    elif normalize == "none":
        plot_mat = mat
        fmt = ".0f"
    else:
        raise ValueError("normalize must be 'row', 'col', or 'none'.")

    if class_names is None:
        xticklabels = [str(c) for c in classes]
    else:
        xticklabels = [class_names[c] if c < len(class_names) else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.5), max(4, num_prototypes * 0.35)))
    im = ax.imshow(plot_mat, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_yticks(np.arange(num_prototypes))
    ax.set_yticklabels([f"P{k}" for k in range(num_prototypes)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Prototype")
    ax.set_title(title)

    for i in range(plot_mat.shape[0]):
        for j in range(plot_mat.shape[1]):
            ax.text(j, i, format(plot_mat[i, j], fmt), ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(save_path), counts


def plot_prototype_region_signature(
    graphs: Sequence[Any],
    proto_model,
    save_path,
    *,
    num_prototypes: int,
    title: str = "Region-aware mean feature signature by prototype",
):
    """
    Heatmap: prototype_id x brain_region.

    For each segment:
        x [N, F] -> region means [R, F] -> average over features -> [R]

    Then average this region vector inside each prototype.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    region_names = list(proto_model.region_names)
    R = len(region_names)

    sums = np.zeros((num_prototypes, R), dtype=np.float64)
    counts = np.zeros(num_prototypes, dtype=np.float64)

    for g in graphs:
        k = _get_proto_id(g)
        x = _to_numpy_x(g)
        x = _safe_zscore_segment_x(x)

        region_flat = segment_region_mean_features(
            x,
            region_to_indices=proto_model.region_to_indices,
            region_names=region_names,
        )

        # region_flat is [R * F]. Convert to [R, F], then average over F.
        F = x.shape[1]
        region_mat = region_flat.reshape(R, F)
        region_sig = region_mat.mean(axis=1)

        sums[k] += region_sig
        counts[k] += 1

    mean_sig = sums / np.clip(counts[:, None], 1e-12, None)

    fig, ax = plt.subplots(figsize=(max(6, R * 1.2), max(4, num_prototypes * 0.35)))
    im = ax.imshow(mean_sig, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(R))
    ax.set_xticklabels(region_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(num_prototypes))
    ax.set_yticklabels([f"P{k} (n={int(counts[k])})" for k in range(num_prototypes)])
    ax.set_xlabel("Brain region")
    ax.set_ylabel("Prototype")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(save_path), pd.DataFrame(mean_sig, columns=region_names)


def plot_prototype_node_feature_heatmaps(
    graphs: Sequence[Any],
    save_dir,
    *,
    num_prototypes: int,
    channel_names: Sequence[str],
    feature_names: Optional[Sequence[str]] = None,
    max_features: Optional[int] = None,
):
    """
    One heatmap per prototype:
        rows = EEG channels
        columns = node features

    This helps answer: what does this prototype look like spatially?
    """
    save_dir = _ensure_dir(save_dir)

    example_x = _to_numpy_x(graphs[0])
    N, F = example_x.shape

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(F)]

    if max_features is not None:
        F_plot = min(max_features, F)
    else:
        F_plot = F

    sums = np.zeros((num_prototypes, N, F), dtype=np.float64)
    counts = np.zeros(num_prototypes, dtype=np.float64)

    for g in graphs:
        k = _get_proto_id(g)
        x = _safe_zscore_segment_x(_to_numpy_x(g))
        sums[k] += x
        counts[k] += 1

    mean_x = sums / np.clip(counts[:, None, None], 1e-12, None)

    paths = []

    for k in range(num_prototypes):
        mat = mean_x[k, :, :F_plot]

        fig, ax = plt.subplots(figsize=(max(7, F_plot * 0.35), max(5, N * 0.25)))
        im = ax.imshow(mat, aspect="auto")
        fig.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(F_plot))
        ax.set_xticklabels(list(feature_names)[:F_plot], rotation=90)
        ax.set_yticks(np.arange(N))
        ax.set_yticklabels(list(channel_names))
        ax.set_title(f"Prototype P{k} node-feature centroid, n={int(counts[k])}")
        ax.set_xlabel("Node feature")
        ax.set_ylabel("Channel")

        fig.tight_layout()
        out = save_dir / f"prototype_{k:02d}_node_feature_heatmap.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(str(out))

    return paths


def save_representative_segments(
    graphs: Sequence[Any],
    save_path,
    *,
    num_prototypes: int,
    top_n: int = 5,
):
    """
    Save closest segments to each prototype.

    This is usually the most useful table for manual inspection.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = make_prototype_segment_table(graphs)

    if "dist_to_proto" not in df.columns or df["dist_to_proto"].isna().all():
        raise ValueError("dist_to_proto is missing. Make sure g.proto_dist exists.")

    rows = []

    for k in range(num_prototypes):
        sub = df[df["proto_id"] == k].copy()
        sub = sub.sort_values("dist_to_proto", ascending=True).head(top_n)
        sub["rank_in_prototype"] = np.arange(1, len(sub) + 1)
        rows.append(sub)

    out = pd.concat(rows, axis=0).reset_index(drop=True)
    out.to_csv(save_path, index=False)

    return str(save_path), out


def plot_attention_by_prototype(
    attention_df: pd.DataFrame,
    save_path,
    *,
    num_prototypes: int,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Expected columns:
        proto_id
        attention
        true_label

    This tells you which prototypes the trained MIL model actually uses.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    required = {"proto_id", "attention", "true_label"}
    missing = required - set(attention_df.columns)
    if missing:
        raise KeyError(f"attention_df missing columns: {missing}")

    summary = (
        attention_df
        .groupby(["proto_id", "true_label"])["attention"]
        .mean()
        .reset_index()
    )

    mat_df = summary.pivot(index="proto_id", columns="true_label", values="attention")
    mat_df = mat_df.reindex(index=range(num_prototypes), fill_value=0.0)
    mat_df = mat_df.fillna(0.0)

    mat = mat_df.to_numpy(dtype=np.float64)
    classes = list(mat_df.columns)

    if class_names is None:
        xticklabels = [str(c) for c in classes]
    else:
        xticklabels = [class_names[int(c)] if int(c) < len(class_names) else str(c) for c in classes]

    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.5), max(4, num_prototypes * 0.35)))
    im = ax.imshow(mat, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(xticklabels, rotation=30, ha="right")
    ax.set_yticks(np.arange(num_prototypes))
    ax.set_yticklabels([f"P{k}" for k in range(num_prototypes)])
    ax.set_xlabel("Class")
    ax.set_ylabel("Prototype")
    ax.set_title("Mean MIL attention by prototype and class")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return str(save_path), mat_df

