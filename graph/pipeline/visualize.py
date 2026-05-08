"""
visualize.py

Reusable plotting functions for:
- training history
- confusion matrices
- ROC / PR curves
- calibration curves
- connectivity heatmaps
- graph structure examples
- 2D embeddings
- attention weights

Design goals
------------
- save figures to disk cleanly
- keep plotting independent from trainer internals
- work for dense, graph, and MIL experiments
- support subject-level result visualization
- support optional segment / macro graph examples
- support both binary and multiclass (2-class / 3-class) settings

Notes
-----
- Uses matplotlib only for plotting.
- Avoids trainer/model-specific assumptions.
- Each function returns the saved path as a string.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


__all__ = [
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_pr_curves",
    "plot_calibration_curve",
    "plot_connectivity_heatmap",
    "plot_graph_structure",
    "plot_embedding_2d",
    "plot_attention_weights",
]


# =========================================================
# Helpers
# =========================================================

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]
MaybeArray = Union[np.ndarray, Sequence[float], Sequence[int], "torch.Tensor"]


def _to_numpy(x: MaybeArray) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_1d(x: MaybeArray, name: str) -> np.ndarray:
    arr = _to_numpy(x)
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _ensure_2d(x: MaybeArray, name: str) -> np.ndarray:
    arr = _to_numpy(x)
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}.")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} is empty, got shape {arr.shape}.")
    return arr


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits)
    denom = np.clip(exp_x.sum(axis=1, keepdims=True), 1e-12, None)
    return exp_x / denom


def _ensure_probs_or_logits(
    probs: Optional[MaybeArray] = None,
    logits: Optional[MaybeArray] = None,
) -> np.ndarray:
    if probs is not None:
        p = _ensure_2d(probs, "probs").astype(np.float64)
        row_sums = np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        return p / row_sums
    if logits is not None:
        z = _ensure_2d(logits, "logits").astype(np.float64)
        return _softmax_numpy(z)
    raise ValueError("Need one of: probs or logits.")


def _infer_class_names(
    num_classes: int,
    class_names: Optional[Sequence[str]] = None,
) -> list[str]:
    if class_names is None:
        return [f"class_{i}" for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(
            f"class_names length ({len(class_names)}) does not match num_classes ({num_classes})."
        )
    return [str(x) for x in class_names]


def _prepare_output_path(
    save_path: Union[str, Path],
    default_suffix: str = ".png",
) -> Path:
    path = Path(save_path)
    if path.suffix == "":
        path = path.with_suffix(default_suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _finalize_figure(
    fig: plt.Figure,
    save_path: Union[str, Path],
    dpi: int = 300,
    close: bool = True,
) -> str:
    path = _prepare_output_path(save_path)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return str(path)


def _extract_history_series(
    history: Mapping[str, Any],
    split: str,
    key: str,
) -> Optional[np.ndarray]:
    if split not in history:
        return None
    split_val = history[split]

    if isinstance(split_val, Mapping):
        if key not in split_val:
            return None
        return np.asarray(split_val[key], dtype=np.float64)

    if isinstance(split_val, Sequence):
        vals = []
        for row in split_val:
            if isinstance(row, Mapping) and key in row:
                vals.append(row[key])
        if len(vals) == 0:
            return None
        return np.asarray(vals, dtype=np.float64)

    return None


def _maybe_get_epochs(history: Mapping[str, Any], split: str) -> Optional[np.ndarray]:
    if split not in history:
        return None
    split_val = history[split]
    if isinstance(split_val, Sequence):
        vals = []
        for row in split_val:
            if isinstance(row, Mapping) and "epoch" in row:
                vals.append(row["epoch"])
        if len(vals) > 0:
            return np.asarray(vals, dtype=np.int64)
    return None


# =========================================================
# Training curves
# =========================================================

def plot_training_curves(
    history: Mapping[str, Any],
    save_path: Union[str, Path],
    *,
    metric_keys: Sequence[str] = ("loss", "accuracy", "balanced_accuracy", "macro_f1"),
    train_key: str = "train",
    val_key: str = "val",
    title: str = "Training curves",
    dpi: int = 300,
) -> str:
    """
    Plot training/validation curves from a history dictionary.

    Supported history shapes
    ------------------------
    1) {"train": [{"epoch": 1, "loss": ...}, ...], "val": [...]}
    2) {"train": {"loss": [...], ...}, "val": {"loss": [...], ...}}

    Returns
    -------
    str
        Saved figure path.
    """
    metric_keys = list(metric_keys)
    n_plots = len(metric_keys)
    if n_plots == 0:
        raise ValueError("metric_keys is empty.")

    fig, axes = plt.subplots(n_plots, 1, figsize=(7, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_keys):
        y_train = _extract_history_series(history, train_key, metric)
        y_val = _extract_history_series(history, val_key, metric)

        x_train = _maybe_get_epochs(history, train_key)
        x_val = _maybe_get_epochs(history, val_key)

        if y_train is None and y_val is None:
            ax.set_visible(False)
            continue

        if y_train is not None:
            if x_train is None or len(x_train) != len(y_train):
                x_train = np.arange(1, len(y_train) + 1)
            ax.plot(x_train, y_train, marker="o", label=train_key)

        if y_val is not None:
            if x_val is None or len(x_val) != len(y_val):
                x_val = np.arange(1, len(y_val) + 1)
            ax.plot(x_val, y_val, marker="o", label=val_key)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    return _finalize_figure(fig, save_path, dpi=dpi)


# =========================================================
# Confusion matrix
# =========================================================

def plot_confusion_matrix(
    y_true: MaybeArray,
    y_pred: MaybeArray,
    save_path: Union[str, Path],
    *,
    class_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    title: str = "Confusion matrix",
    dpi: int = 300,
) -> str:
    """
    Plot confusion matrix for 2-class or multiclass predictions.

    If normalize=True, displayed cell values are row-normalized, while the
    raw counts are also shown in the annotations.
    """
    y_true = _ensure_1d(y_true, "y_true").astype(np.int64)
    y_pred = _ensure_1d(y_pred, "y_pred").astype(np.int64)

    num_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    labels = list(range(num_classes))
    names = _infer_class_names(num_classes, class_names)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_plot = cm.astype(np.float64)

    if normalize:
        row_sums = np.clip(cm_plot.sum(axis=1, keepdims=True), 1.0, None)
        cm_plot = cm_plot / row_sums

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(num_classes):
        for j in range(num_classes):
            if normalize:
                txt = f"{cm[i, j]}\n({cm_plot[i, j]:.2f})"
            else:
                txt = f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center")

    return _finalize_figure(fig, save_path, dpi=dpi)


# =========================================================
# ROC / PR
# =========================================================

def plot_roc_curves(
    y_true: MaybeArray,
    save_path: Union[str, Path],
    *,
    probs: Optional[MaybeArray] = None,
    logits: Optional[MaybeArray] = None,
    class_names: Optional[Sequence[str]] = None,
    title: str = "ROC curves",
    dpi: int = 300,
) -> str:
    """
    Plot ROC curves for binary or multiclass one-vs-rest classification.

    Undefined classes (no positive or no negative samples) are skipped safely.
    """
    y_true = _ensure_1d(y_true, "y_true").astype(np.int64)
    probs = _ensure_probs_or_logits(probs=probs, logits=logits)
    num_classes = probs.shape[1]
    names = _infer_class_names(num_classes, class_names)

    fig, ax = plt.subplots(figsize=(6, 5))

    if num_classes == 2:
        if np.unique(y_true).size >= 2:
            fpr, tpr, _ = roc_curve(y_true, probs[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{names[1]} vs rest (AUC={roc_auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            yc = y_bin[:, c]
            if np.unique(yc).size < 2:
                continue
            fpr, tpr, _ = roc_curve(yc, probs[:, c])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{names[c]} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finalize_figure(fig, save_path, dpi=dpi)


def plot_pr_curves(
    y_true: MaybeArray,
    save_path: Union[str, Path],
    *,
    probs: Optional[MaybeArray] = None,
    logits: Optional[MaybeArray] = None,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Precision-recall curves",
    dpi: int = 300,
) -> str:
    """
    Plot PR curves for binary or multiclass one-vs-rest classification.

    Undefined classes (no positive samples) are skipped safely.
    """
    y_true = _ensure_1d(y_true, "y_true").astype(np.int64)
    probs = _ensure_probs_or_logits(probs=probs, logits=logits)
    num_classes = probs.shape[1]
    names = _infer_class_names(num_classes, class_names)

    fig, ax = plt.subplots(figsize=(6, 5))

    if num_classes == 2:
        positive = (y_true == 1).astype(np.int64)
        if positive.sum() > 0:
            precision, recall, _ = precision_recall_curve(positive, probs[:, 1])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, label=f"{names[1]} vs rest (AUC={pr_auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            yc = y_bin[:, c]
            if yc.sum() == 0:
                continue
            precision, recall, _ = precision_recall_curve(yc, probs[:, c])
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, label=f"{names[c]} (AUC={pr_auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.setTitle = ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finalize_figure(fig, save_path, dpi=dpi)


# =========================================================
# Calibration
# =========================================================

def plot_calibration_curve(
    y_true: MaybeArray,
    save_path: Union[str, Path],
    *,
    probs: Optional[MaybeArray] = None,
    logits: Optional[MaybeArray] = None,
    class_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Calibration curve",
    dpi: int = 300,
) -> str:
    """
    Plot calibration curves.

    Binary:
        plots the positive class curve.
    Multiclass:
        plots one-vs-rest calibration for each class.

    Undefined classes are skipped safely.
    """
    y_true = _ensure_1d(y_true, "y_true").astype(np.int64)
    probs = _ensure_probs_or_logits(probs=probs, logits=logits)
    num_classes = probs.shape[1]
    names = _infer_class_names(num_classes, class_names)

    fig, ax = plt.subplots(figsize=(6, 5))

    if num_classes == 2:
        y_bin = (y_true == 1).astype(np.int64)
        if np.unique(y_bin).size >= 2:
            frac_pos, mean_pred = calibration_curve(
                y_bin,
                probs[:, 1],
                n_bins=n_bins,
                strategy=strategy,
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=names[1])
    else:
        y_oh = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            yc = y_oh[:, c]
            if np.unique(yc).size < 2:
                continue
            frac_pos, mean_pred = calibration_curve(
                yc,
                probs[:, c],
                n_bins=n_bins,
                strategy=strategy,
            )
            ax.plot(mean_pred, frac_pos, marker="o", label=names[c])

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finalize_figure(fig, save_path, dpi=dpi)


# =========================================================
# Connectivity / graph
# =========================================================

def plot_connectivity_heatmap(
    matrix: MaybeArray,
    save_path: Union[str, Path],
    *,
    channel_names: Optional[Sequence[str]] = None,
    title: str = "Connectivity heatmap",
    show_colorbar: bool = True,
    dpi: int = 300,
) -> str:
    """
    Plot a connectivity / adjacency matrix heatmap.

    Supports shape [N, N].
    """
    mat = _ensure_2d(matrix, "matrix").astype(np.float64)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"matrix must be square, got shape {mat.shape}.")

    n = mat.shape[0]
    labels = [str(i) for i in range(n)] if channel_names is None else list(channel_names)
    if len(labels) != n:
        raise ValueError("channel_names length does not match matrix size.")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto")
    if show_colorbar:
        fig.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_title(title)
    ax.set_xlabel("Node / channel")
    ax.set_ylabel("Node / channel")

    return _finalize_figure(fig, save_path, dpi=dpi)


def plot_graph_structure(
    save_path: Union[str, Path],
    *,
    adj_matrix: Optional[MaybeArray] = None,
    edge_index: Optional[MaybeArray] = None,
    edge_weight: Optional[MaybeArray] = None,
    node_labels: Optional[Sequence[str]] = None,
    node_positions: Optional[Mapping[Any, Tuple[float, float]]] = None,
    threshold: Optional[float] = None,
    title: str = "Graph structure",
    dpi: int = 300,
) -> str:
    """
    Plot a graph example from either:
    - dense adjacency matrix
    - edge_index (+ optional edge_weight)

    Notes
    -----
    - Requires networkx. Raises a helpful error if unavailable.
    - If threshold is provided, weak edges are removed before plotting.
    """
    if nx is None:
        raise ImportError("plot_graph_structure requires networkx to be installed.")

    G = nx.Graph()

    if adj_matrix is not None:
        adj = _ensure_2d(adj_matrix, "adj_matrix").astype(np.float64)
        if adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj_matrix must be square, got shape {adj.shape}.")
        n = adj.shape[0]
        for i in range(n):
            G.add_node(i)
        for i in range(n):
            for j in range(i + 1, n):
                w = float(adj[i, j])
                if threshold is not None and abs(w) < threshold:
                    continue
                if abs(w) > 0:
                    G.add_edge(i, j, weight=w)

    elif edge_index is not None:
        ei = _to_numpy(edge_index)
        if ei.ndim != 2 or ei.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {ei.shape}.")
        ew = None if edge_weight is None else _ensure_1d(edge_weight, "edge_weight").astype(np.float64)
        if ew is not None and ew.shape[0] != ei.shape[1]:
            raise ValueError("edge_weight length does not match edge_index.")
        nodes = np.unique(ei)
        for node in nodes.tolist():
            G.add_node(int(node))
        for k in range(ei.shape[1]):
            i = int(ei[0, k])
            j = int(ei[1, k])
            w = 1.0 if ew is None else float(ew[k])
            if threshold is not None and abs(w) < threshold:
                continue
            G.add_edge(i, j, weight=w)
    else:
        raise ValueError("Need one of: adj_matrix or edge_index.")

    if node_positions is None:
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = dict(node_positions)

    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax)

    if node_labels is not None:
        labels = {i: str(lbl) for i, lbl in enumerate(node_labels)}
    else:
        labels = {node: str(node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8)

    ax.set_title(title)
    ax.axis("off")

    return _finalize_figure(fig, save_path, dpi=dpi)


# =========================================================
# Embeddings
# =========================================================

def plot_embedding_2d(
    embeddings: MaybeArray,
    labels: MaybeArray,
    save_path: Union[str, Path],
    *,
    method: str = "pca",
    class_names: Optional[Sequence[str]] = None,
    point_annotations: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    dpi: int = 300,
    **method_kwargs: Any,
) -> str:
    """
    Plot 2D embeddings using PCA or t-SNE.

    Parameters
    ----------
    embeddings : [N, D]
    labels : [N]
    method : {"pca", "tsne"}
    point_annotations : optional sequence[str]
        If provided, annotate each point.

    Returns
    -------
    str
        Saved figure path.
    """
    X = _ensure_2d(embeddings, "embeddings").astype(np.float64)
    y = _ensure_1d(labels, "labels").astype(np.int64)

    if X.shape[0] != y.shape[0]:
        raise ValueError("embeddings and labels must have the same number of rows.")

    method = str(method).lower()
    if method == "pca":
        reducer = PCA(n_components=2, **method_kwargs)
    elif method == "tsne":
        reducer = TSNE(n_components=2, **method_kwargs)
    else:
        raise ValueError(f"method must be 'pca' or 'tsne', got {method!r}")

    X2 = reducer.fit_transform(X)
    num_classes = int(np.max(y)) + 1
    names = _infer_class_names(num_classes, class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    for c in range(num_classes):
        idx = (y == c)
        if np.sum(idx) == 0:
            continue
        ax.scatter(X2[idx, 0], X2[idx, 1], label=names[c], alpha=0.8)

    if point_annotations is not None:
        if len(point_annotations) != X2.shape[0]:
            raise ValueError("point_annotations length mismatch.")
        for i, txt in enumerate(point_annotations):
            ax.text(X2[i, 0], X2[i, 1], str(txt), fontsize=8)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title or f"Embedding 2D ({method.upper()})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return _finalize_figure(fig, save_path, dpi=dpi)


# =========================================================
# Attention
# =========================================================

def plot_attention_weights(
    weights: MaybeArray,
    save_path: Union[str, Path],
    *,
    instance_labels: Optional[Sequence[str]] = None,
    title: str = "Attention weights",
    normalize: bool = False,
    dpi: int = 300,
) -> str:
    """
    Plot attention weights.

    Supported inputs
    ----------------
    - 1D: one bag / one subject attention vector
    - 2D: multiple subjects x instances heatmap

    If normalize=True, each row is normalized to sum to 1.
    """
    arr = _to_numpy(weights).astype(np.float64)

    if arr.ndim == 1:
        x = arr.copy()
        if normalize:
            x = x / np.clip(x.sum(), 1e-12, None)

        labels = [str(i) for i in range(len(x))] if instance_labels is None else list(instance_labels)
        if len(labels) != len(x):
            raise ValueError("instance_labels length mismatch.")

        fig, ax = plt.subplots(figsize=(max(6, len(x) * 0.5), 4))
        ax.bar(np.arange(len(x)), x)
        ax.set_xticks(np.arange(len(x)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel("Instance")
        ax.set_ylabel("Attention weight")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

        return _finalize_figure(fig, save_path, dpi=dpi)

    if arr.ndim == 2:
        x = arr.copy()
        if normalize:
            row_sums = np.clip(x.sum(axis=1, keepdims=True), 1e-12, None)
            x = x / row_sums

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(x, aspect="auto")
        fig.colorbar(im, ax=ax)

        ax.set_xlabel("Instance")
        ax.set_ylabel("Bag / subject")
        ax.set_title(title)

        if instance_labels is not None:
            if len(instance_labels) != x.shape[1]:
                raise ValueError("instance_labels length mismatch.")
            ax.set_xticks(np.arange(x.shape[1]))
            ax.set_xticklabels(list(instance_labels), rotation=45, ha="right")

        return _finalize_figure(fig, save_path, dpi=dpi)

    raise ValueError(f"weights must be 1D or 2D, got shape {arr.shape}.")


# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    outdir = Path("visualize_examples")
    outdir.mkdir(parents=True, exist_ok=True)

    history = {
        "train": [
            {"epoch": 1, "loss": 1.10, "accuracy": 0.42, "balanced_accuracy": 0.40, "macro_f1": 0.39},
            {"epoch": 2, "loss": 0.92, "accuracy": 0.58, "balanced_accuracy": 0.56, "macro_f1": 0.55},
            {"epoch": 3, "loss": 0.78, "accuracy": 0.67, "balanced_accuracy": 0.65, "macro_f1": 0.64},
        ],
        "val": [
            {"epoch": 1, "loss": 1.05, "accuracy": 0.46, "balanced_accuracy": 0.44, "macro_f1": 0.43},
            {"epoch": 2, "loss": 0.98, "accuracy": 0.54, "balanced_accuracy": 0.53, "macro_f1": 0.52},
            {"epoch": 3, "loss": 0.95, "accuracy": 0.57, "balanced_accuracy": 0.55, "macro_f1": 0.54},
        ],
    }
    plot_training_curves(history, outdir / "training_curves.png")

    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 2])
    plot_confusion_matrix(
        y_true,
        y_pred,
        outdir / "confusion_matrix.png",
        class_names=["HC", "AD", "FTD"],
        normalize=True,
    )

    print(f"Saved example figures to: {outdir.resolve()}")
