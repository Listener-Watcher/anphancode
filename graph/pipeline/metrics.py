"""
metrics.py

Reusable evaluation metrics for subject-level dementia classification.

Design goals:
- work for dense, graph, and MIL subject-level outputs
- operate on subject-level predictions by default
- support multiclass first
- handle missing/undefined AUC cases safely
- return plain Python dictionaries from summary helpers for easy logging/saving

Conventions:
- labels are integer-encoded class ids of shape [N]
- predictions can be provided as:
    * hard labels: y_pred
    * probabilities: probs of shape [N, C]
    * logits: logits of shape [N, C]
- if logits are given, softmax is applied internally
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[int], Sequence[float], List[int], List[float]]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _to_numpy(x: Optional[ArrayLike]) -> Optional[np.ndarray]:
    """Convert tensor/list/array to numpy without modifying values."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_1d_labels(y: ArrayLike, name: str = "labels") -> np.ndarray:
    """Convert labels to a flat int64 array."""
    arr = _to_numpy(y)
    if arr is None:
        raise ValueError(f"{name} cannot be None.")
    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr.astype(np.int64, copy=False)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for numpy arrays."""
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits)
    denom = np.clip(exp_x.sum(axis=1, keepdims=True), 1e-12, None)
    return (exp_x / denom).astype(np.float64, copy=False)


def _ensure_2d_scores(
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
) -> np.ndarray:
    """
    Return class probabilities of shape [N, C].

    Priority:
    1) probs
    2) logits -> softmax(logits)
    """
    if probs is not None:
        p = _to_numpy(probs)
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 2:
            raise ValueError(f"probs must have shape [N, C], got {p.shape}.")
        row_sums = p.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("probs contains rows with non-positive sum.")
        if np.any(p < 0):
            raise ValueError("probs contains negative values.")
        return p / row_sums

    if logits is not None:
        z = _to_numpy(logits)
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2:
            raise ValueError(f"logits must have shape [N, C], got {z.shape}.")
        return _softmax_numpy(z)

    raise ValueError("Need one of: probs or logits.")


def _infer_pred_labels(
    y_pred: Optional[ArrayLike] = None,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
) -> np.ndarray:
    """Infer hard labels either directly or from argmax over probs/logits."""
    if y_pred is not None:
        return _ensure_1d_labels(y_pred, name="y_pred")

    p = _ensure_2d_scores(probs=probs, logits=logits)
    return np.argmax(p, axis=1).astype(np.int64, copy=False)


def _infer_class_labels(
    y_true: np.ndarray,
    probs: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None,
    labels: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Determine the class index list to use for multiclass metrics.

    Priority:
    1) explicit labels
    2) explicit num_classes -> [0, ..., C-1]
    3) probs.shape[1]
    4) max(y_true) + 1
    """
    if labels is not None:
        return [int(x) for x in labels]

    if num_classes is not None:
        return list(range(int(num_classes)))

    if probs is not None:
        return list(range(int(probs.shape[1])))

    return list(range(int(np.max(y_true)) + 1))


def _one_hot(y: np.ndarray, class_labels: Sequence[int]) -> np.ndarray:
    """One-hot encode labels according to an explicit class ordering."""
    y = _ensure_1d_labels(y, name="y")
    class_to_idx = {int(c): i for i, c in enumerate(class_labels)}
    out = np.zeros((len(y), len(class_labels)), dtype=np.float64)
    for i, label in enumerate(y):
        if int(label) not in class_to_idx:
            raise ValueError(f"Label {label} not found in class_labels={list(class_labels)}.")
        out[i, class_to_idx[int(label)]] = 1.0
    return out


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))


def _safe_weighted_mean(
    values: Sequence[Optional[float]],
    weights: Sequence[float],
) -> Optional[float]:
    pairs = [(float(v), float(w)) for v, w in zip(values, weights) if v is not None and np.isfinite(v)]
    if len(pairs) == 0:
        return None
    v = np.asarray([p[0] for p in pairs], dtype=np.float64)
    w = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if np.all(w <= 0):
        return float(np.mean(v))
    return float(np.average(v, weights=w))


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def compute_accuracy(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
) -> float:
    """Compute standard accuracy from hard labels or class scores."""
    yt = _ensure_1d_labels(y_true, name="y_true")
    yp = _infer_pred_labels(y_pred=y_pred, probs=probs, logits=logits)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(pred)={len(yp)}.")
    return float(accuracy_score(yt, yp))


def compute_balanced_accuracy(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
) -> float:
    """Compute balanced accuracy for multiclass classification."""
    yt = _ensure_1d_labels(y_true, name="y_true")
    yp = _infer_pred_labels(y_pred=y_pred, probs=probs, logits=logits)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(pred)={len(yp)}.")
    return float(balanced_accuracy_score(yt, yp))


def compute_macro_f1(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    zero_division: float = 0.0,
) -> float:
    """Compute macro-F1 for multiclass classification."""
    yt = _ensure_1d_labels(y_true, name="y_true")
    yp = _infer_pred_labels(y_pred=y_pred, probs=probs, logits=logits)
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(pred)={len(yp)}.")
    return float(f1_score(yt, yp, average="macro", zero_division=zero_division))


def compute_confusion_matrix(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    labels: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
) -> np.ndarray:
    """Compute confusion matrix as a numpy array of shape [C, C]."""
    yt = _ensure_1d_labels(y_true, name="y_true")
    yp = _infer_pred_labels(y_pred=y_pred, probs=probs, logits=logits)

    class_labels = _infer_class_labels(
        y_true=yt,
        probs=_to_numpy(probs) if probs is not None else _to_numpy(logits),
        num_classes=num_classes,
        labels=labels,
    )
    cm = confusion_matrix(yt, yp, labels=class_labels)
    return np.asarray(cm, dtype=np.int64)


def compute_roc_auc(
    y_true: ArrayLike,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    labels: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute safe multiclass ROC-AUC summaries.

    Returns a dictionary with:
    - macro_ovr
    - weighted_ovr
    - per_class
    - valid_classes
    - missing_classes

    If a class has no positive or no negative examples, its per-class ROC-AUC is None.
    """
    yt = _ensure_1d_labels(y_true, name="y_true")
    p = _ensure_2d_scores(probs=probs, logits=logits)

    if len(yt) != p.shape[0]:
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs probs/logits batch={p.shape[0]}.")

    class_labels = _infer_class_labels(y_true=yt, probs=p, num_classes=num_classes, labels=labels)
    if p.shape[1] != len(class_labels):
        raise ValueError(
            f"Score dimension mismatch: scores have C={p.shape[1]} but class_labels has {len(class_labels)} classes."
        )

    y_bin = _one_hot(yt, class_labels)
    per_class: Dict[int, Optional[float]] = {}
    supports: List[int] = []
    valid_classes: List[int] = []
    missing_classes: List[int] = []

    for idx, cls in enumerate(class_labels):
        y_true_bin = y_bin[:, idx]
        score_bin = p[:, idx]
        supports.append(int(y_true_bin.sum()))

        if np.unique(y_true_bin).size < 2:
            per_class[int(cls)] = None
            missing_classes.append(int(cls))
            continue

        try:
            auc_val = roc_auc_score(y_true_bin, score_bin)
            per_class[int(cls)] = float(auc_val)
            valid_classes.append(int(cls))
        except ValueError:
            per_class[int(cls)] = None
            missing_classes.append(int(cls))

    per_class_values = [per_class[int(cls)] for cls in class_labels]
    macro_ovr = _safe_mean(per_class_values)
    weighted_ovr = _safe_weighted_mean(per_class_values, supports)

    return {
        "macro_ovr": macro_ovr,
        "weighted_ovr": weighted_ovr,
        "per_class": per_class,
        "valid_classes": valid_classes,
        "missing_classes": missing_classes,
        "num_valid_classes": len(valid_classes),
    }


def compute_pr_auc(
    y_true: ArrayLike,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    labels: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute safe multiclass PR-AUC summaries using one-vs-rest average precision.

    Returns a dictionary with:
    - macro_ovr
    - weighted_ovr
    - per_class
    - valid_classes
    - missing_classes

    If a class has no positive examples, its per-class PR-AUC is None.
    """
    yt = _ensure_1d_labels(y_true, name="y_true")
    p = _ensure_2d_scores(probs=probs, logits=logits)

    if len(yt) != p.shape[0]:
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs probs/logits batch={p.shape[0]}.")

    class_labels = _infer_class_labels(y_true=yt, probs=p, num_classes=num_classes, labels=labels)
    if p.shape[1] != len(class_labels):
        raise ValueError(
            f"Score dimension mismatch: scores have C={p.shape[1]} but class_labels has {len(class_labels)} classes."
        )

    y_bin = _one_hot(yt, class_labels)
    per_class: Dict[int, Optional[float]] = {}
    supports: List[int] = []
    valid_classes: List[int] = []
    missing_classes: List[int] = []

    for idx, cls in enumerate(class_labels):
        y_true_bin = y_bin[:, idx]
        score_bin = p[:, idx]
        supports.append(int(y_true_bin.sum()))

        if y_true_bin.sum() == 0:
            per_class[int(cls)] = None
            missing_classes.append(int(cls))
            continue

        try:
            ap_val = average_precision_score(y_true_bin, score_bin)
            per_class[int(cls)] = float(ap_val)
            valid_classes.append(int(cls))
        except ValueError:
            per_class[int(cls)] = None
            missing_classes.append(int(cls))

    per_class_values = [per_class[int(cls)] for cls in class_labels]
    macro_ovr = _safe_mean(per_class_values)
    weighted_ovr = _safe_weighted_mean(per_class_values, supports)

    return {
        "macro_ovr": macro_ovr,
        "weighted_ovr": weighted_ovr,
        "per_class": per_class,
        "valid_classes": valid_classes,
        "missing_classes": missing_classes,
        "num_valid_classes": len(valid_classes),
    }


def compute_brier_score(
    y_true: ArrayLike,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    labels: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
) -> float:
    """
    Compute the multiclass Brier score.

    Definition used here:
        mean_i sum_c (p_ic - y_ic)^2

    This is appropriate for subject-level multiclass probability calibration tracking.
    """
    yt = _ensure_1d_labels(y_true, name="y_true")
    p = _ensure_2d_scores(probs=probs, logits=logits)

    if len(yt) != p.shape[0]:
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs probs/logits batch={p.shape[0]}.")

    class_labels = _infer_class_labels(y_true=yt, probs=p, num_classes=num_classes, labels=labels)
    if p.shape[1] != len(class_labels):
        raise ValueError(
            f"Score dimension mismatch: scores have C={p.shape[1]} but class_labels has {len(class_labels)} classes."
        )

    y_bin = _one_hot(yt, class_labels)
    sample_errors = np.sum((p - y_bin) ** 2, axis=1)
    return float(np.mean(sample_errors))


def summarize_classification_metrics(
    y_true: ArrayLike,
    y_pred: Optional[ArrayLike] = None,
    *,
    probs: Optional[ArrayLike] = None,
    logits: Optional[ArrayLike] = None,
    labels: Optional[Sequence[int]] = None,
    num_classes: Optional[int] = None,
    include_auc: bool = True,
) -> Dict[str, Any]:
    """
    Compute a standard subject-level classification summary.

    Returns a plain dictionary suitable for:
    - logging
    - JSON saving
    - CSV row expansion (after flattening if needed)

    The confusion matrix is returned as a nested list for serialization friendliness.
    """
    yt = _ensure_1d_labels(y_true, name="y_true")
    p = None
    if probs is not None or logits is not None:
        p = _ensure_2d_scores(probs=probs, logits=logits)

    pred = _infer_pred_labels(y_pred=y_pred, probs=probs, logits=logits)
    class_labels = _infer_class_labels(y_true=yt, probs=p, num_classes=num_classes, labels=labels)
    cm = compute_confusion_matrix(yt, pred, labels=class_labels)

    summary: Dict[str, Any] = {
        "num_samples": int(len(yt)),
        "class_labels": [int(c) for c in class_labels],
        "accuracy": compute_accuracy(yt, pred),
        "balanced_accuracy": compute_balanced_accuracy(yt, pred),
        "macro_f1": compute_macro_f1(yt, pred),
        "confusion_matrix": cm.tolist(),
    }

    if p is not None:
        summary["brier_score"] = compute_brier_score(
            yt,
            probs=p,
            labels=class_labels,
        )

        if include_auc:
            roc_dict = compute_roc_auc(
                yt,
                probs=p,
                labels=class_labels,
            )
            pr_dict = compute_pr_auc(
                yt,
                probs=p,
                labels=class_labels,
            )

            summary["roc_auc"] = roc_dict
            summary["pr_auc"] = pr_dict
            summary["roc_auc_macro_ovr"] = roc_dict["macro_ovr"]
            summary["pr_auc_macro_ovr"] = pr_dict["macro_ovr"]

    return summary


# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Example 1: logits -> probabilities internally
    logits = torch.tensor([
        [2.2, 0.1, -1.3],
        [0.2, 1.8, -0.4],
        [-0.5, 0.3, 1.9],
        [1.2, 0.8, -0.2],
        [0.1, 1.1, 0.5],
        [-0.3, 0.2, 1.4],
    ], dtype=torch.float32)

    labels = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)

    summary_from_logits = summarize_classification_metrics(
        y_true=labels,
        logits=logits,
        num_classes=3,
    )

    print("Summary from logits:")
    for k, v in summary_from_logits.items():
        print(f"{k}: {v}")

    # Example 2: probabilities + hard labels
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    acc = compute_accuracy(labels, preds)
    bal_acc = compute_balanced_accuracy(labels, preds)
    macro_f1 = compute_macro_f1(labels, preds)
    cm = compute_confusion_matrix(labels, preds, num_classes=3)
    roc = compute_roc_auc(labels, probs=probs, num_classes=3)
    pr = compute_pr_auc(labels, probs=probs, num_classes=3)
    brier = compute_brier_score(labels, probs=probs, num_classes=3)

    print("\nManual metric calls:")
    print("accuracy:", acc)
    print("balanced_accuracy:", bal_acc)
    print("macro_f1:", macro_f1)
    print("confusion_matrix:\n", cm)
    print("roc_auc:", roc)
    print("pr_auc:", pr)
    print("brier_score:", brier)

    # Example training-step style usage
    #
    # batch_logits = model(batch_dict)["logits"]        # [B, C]
    # batch_labels = batch_dict["labels"]               # [B]
    # loss = criterion(batch_logits, batch_labels)
    #
    # with torch.no_grad():
    #     metrics = summarize_classification_metrics(
    #         y_true=batch_labels,
    #         logits=batch_logits,
    #         num_classes=3,
    #     )
    #     print(metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"])
