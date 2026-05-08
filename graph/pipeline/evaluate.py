"""
evaluate.py

Prediction aggregation, fold aggregation, and final evaluation summaries
for subject-level dementia classification.

This module is designed to work with:
- segment-level predictions
- macro-level predictions
- subject-level predictions
- MIL and non-MIL experiments
- 2-class and 3-class settings

Main ideas
----------
1. Convert instance-level predictions (segments / macro instances) into a
   subject-level table.
2. Summarize one fold / one seed / one split cleanly.
3. Aggregate summaries across folds and seeds.
4. Save predictions and summaries in lightweight formats for later analysis.

Recommended prediction table columns
------------------------------------
At minimum:
- subject_id
- true_label

Optional but strongly recommended:
- pred_label
- prob_0, prob_1, ...
- logit_0, logit_1, ...
- split
- fold
- split_seed
- instance_id
- source_level   ("segment", "macro", "subject")

The aggregation functions are intentionally tolerant:
- if probabilities are missing but logits exist, they are converted
- if only hard labels exist, one-hot votes are used as fallback
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    from .metrics import summarize_classification_metrics
except ImportError:  # pragma: no cover
    from metrics import summarize_classification_metrics


__all__ = [
    "aggregate_instance_predictions_to_subject",
    "soft_vote_subject_predictions",
    "mean_vote_subject_predictions",
    "summarize_fold_results",
    "summarize_cv_results",
    "save_predictions_csv",
    "save_summary_json",
]


_DEFAULT_META_COLS: Tuple[str, ...] = ("split_seed", "fold", "split")
_DEFAULT_GROUP_COLS: Tuple[str, ...] = ("split_seed", "fold", "split", "subject_id")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_x = np.exp(logits)
    denom = np.clip(exp_x.sum(axis=1, keepdims=True), 1e-12, None)
    return exp_x / denom


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _ensure_dataframe(
    predictions: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]] = None,
    *,
    subject_ids: Optional[Sequence[Any]] = None,
    true_labels: Optional[Sequence[int]] = None,
    pred_labels: Optional[Sequence[int]] = None,
    probs: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    instance_ids: Optional[Sequence[Any]] = None,
    split: Optional[Union[str, Sequence[Any]]] = None,
    fold: Optional[Union[int, Sequence[Any]]] = None,
    split_seed: Optional[Union[int, Sequence[Any]]] = None,
    source_level: Optional[Union[str, Sequence[Any]]] = None,
) -> pd.DataFrame:
    if predictions is not None:
        if isinstance(predictions, pd.DataFrame):
            return predictions.copy()
        return pd.DataFrame(list(predictions))

    if subject_ids is None or true_labels is None:
        raise ValueError("Need either predictions dataframe/rows or both subject_ids and true_labels.")

    n = len(subject_ids)
    if len(true_labels) != n:
        raise ValueError("subject_ids and true_labels must have the same length.")

    df = pd.DataFrame({
        "subject_id": list(subject_ids),
        "true_label": list(true_labels),
    })

    if pred_labels is not None:
        if len(pred_labels) != n:
            raise ValueError("pred_labels length mismatch.")
        df["pred_label"] = list(pred_labels)

    if probs is not None:
        probs = np.asarray(probs)
        if probs.ndim != 2 or probs.shape[0] != n:
            raise ValueError(f"probs must have shape [N, C], got {probs.shape}.")
        for c in range(probs.shape[1]):
            df[f"prob_{c}"] = probs[:, c]

    if logits is not None:
        logits = np.asarray(logits)
        if logits.ndim != 2 or logits.shape[0] != n:
            raise ValueError(f"logits must have shape [N, C], got {logits.shape}.")
        for c in range(logits.shape[1]):
            df[f"logit_{c}"] = logits[:, c]

    if instance_ids is not None:
        if len(instance_ids) != n:
            raise ValueError("instance_ids length mismatch.")
        df["instance_id"] = list(instance_ids)

    def _broadcast(value: Any, name: str) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            if len(value) != n:
                raise ValueError(f"{name} length mismatch.")
            df[name] = list(value)
        else:
            df[name] = value

    _broadcast(split, "split")
    _broadcast(fold, "fold")
    _broadcast(split_seed, "split_seed")
    _broadcast(source_level, "source_level")

    return df


def _infer_num_classes(
    df: pd.DataFrame,
    *,
    prob_cols: Optional[Sequence[str]] = None,
    logit_cols: Optional[Sequence[str]] = None,
) -> int:
    if prob_cols is not None and len(prob_cols) > 0:
        return len(prob_cols)
    if logit_cols is not None and len(logit_cols) > 0:
        return len(logit_cols)

    label_candidates = []
    for col in ("true_label", "pred_label"):
        if col in df.columns:
            label_candidates.extend(pd.Series(df[col]).dropna().astype(int).tolist())

    if len(label_candidates) == 0:
        raise ValueError("Could not infer num_classes from dataframe.")
    return int(max(label_candidates)) + 1


def _find_score_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    prob_cols = sorted([c for c in df.columns if c.startswith("prob_")], key=lambda x: int(x.split("_")[1]))
    logit_cols = sorted([c for c in df.columns if c.startswith("logit_")], key=lambda x: int(x.split("_")[1]))
    return prob_cols, logit_cols


def _ensure_probability_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    prob_cols, logit_cols = _find_score_columns(df)

    if len(prob_cols) > 0:
        return df, prob_cols

    if len(logit_cols) > 0:
        logits = df[logit_cols].to_numpy(dtype=np.float64)
        probs = _softmax_numpy(logits)
        new_prob_cols = [f"prob_{i}" for i in range(probs.shape[1])]
        for i, col in enumerate(new_prob_cols):
            df[col] = probs[:, i]
        return df, new_prob_cols

    if "pred_label" in df.columns:
        num_classes = _infer_num_classes(df, prob_cols=prob_cols, logit_cols=logit_cols)
        preds = df["pred_label"].to_numpy(dtype=np.int64)
        probs = np.zeros((len(df), num_classes), dtype=np.float64)
        probs[np.arange(len(df)), preds] = 1.0
        new_prob_cols = [f"prob_{i}" for i in range(num_classes)]
        for i, col in enumerate(new_prob_cols):
            df[col] = probs[:, i]
        return df, new_prob_cols

    raise ValueError("Need at least one of: prob_* columns, logit_* columns, or pred_label.")


def _ensure_logit_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    _, logit_cols = _find_score_columns(df)
    return df, logit_cols


def _resolve_group_cols(df: pd.DataFrame, group_cols: Optional[Sequence[str]]) -> list[str]:
    if group_cols is None:
        cols = [c for c in _DEFAULT_GROUP_COLS if c in df.columns]
    else:
        cols = [str(c) for c in group_cols if str(c) in df.columns]

    if "subject_id" not in cols:
        if "subject_id" not in df.columns:
            raise KeyError("Prediction table must contain 'subject_id'.")
        cols.append("subject_id")
    return cols


def _validate_group_labels(df_group: pd.DataFrame, true_label_col: str = "true_label") -> int:
    if true_label_col not in df_group.columns:
        raise KeyError(f"Missing required column: {true_label_col}")
    uniq = pd.Series(df_group[true_label_col]).dropna().unique().tolist()
    if len(uniq) == 0:
        raise ValueError("Group has no valid true label.")
    if len(uniq) > 1:
        raise ValueError(
            f"Group contains multiple true labels: {uniq}. "
            "Instance rows for one subject/fold/split should share the same label."
        )
    return int(uniq[0])


def _normalize_prob_row(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    denom = np.clip(x.sum(), 1e-12, None)
    return x / denom


def _aggregate_common_metadata(df_group: pd.DataFrame, group_cols: Sequence[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col in group_cols:
        values = df_group[col].dropna().unique().tolist()
        if len(values) == 0:
            row[col] = None
        elif len(values) == 1:
            row[col] = values[0]
        else:
            raise ValueError(f"Group column '{col}' has multiple values inside one group: {values}")
    return row


def _aggregate_subject_table_from_scores(
    df: pd.DataFrame,
    *,
    aggregated_scores: np.ndarray,
    group_cols: Sequence[str],
    score_type: str,
    aggregated_logits: Optional[np.ndarray] = None,
    source_level: Optional[str] = None,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    prob_cols = [f"prob_{i}" for i in range(aggregated_scores.shape[1])]
    logit_cols = [f"logit_{i}" for i in range(aggregated_scores.shape[1])] if aggregated_logits is not None else []

    grouped = list(df.groupby(list(group_cols), sort=False, dropna=False))
    if len(grouped) != aggregated_scores.shape[0]:
        raise RuntimeError("Internal grouping mismatch while building subject table.")

    for row_idx, (_, g) in enumerate(grouped):
        row = _aggregate_common_metadata(g, group_cols)
        row["true_label"] = _validate_group_labels(g, true_label_col="true_label")
        row["num_instances"] = int(len(g))
        row["source_level"] = source_level or ("subject" if len(g) == 1 else "aggregated_subject")
        row["aggregation_method"] = score_type

        probs = _normalize_prob_row(aggregated_scores[row_idx])
        pred = int(np.argmax(probs))
        row["pred_label"] = pred
        for c, col in enumerate(prob_cols):
            row[col] = float(probs[c])

        if aggregated_logits is not None:
            for c, col in enumerate(logit_cols):
                row[col] = float(aggregated_logits[row_idx, c])

        rows.append(row)

    return pd.DataFrame(rows)


def soft_vote_subject_predictions(
    predictions: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    *,
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = _ensure_dataframe(predictions)
    df, prob_cols = _ensure_probability_columns(df)
    group_cols = _resolve_group_cols(df, group_cols)

    grouped = list(df.groupby(group_cols, sort=False, dropna=False))
    agg_probs = []
    for _, g in grouped:
        probs = g[prob_cols].to_numpy(dtype=np.float64)
        agg_probs.append(np.mean(probs, axis=0))
    agg_probs = np.asarray(agg_probs, dtype=np.float64)

    return _aggregate_subject_table_from_scores(
        df,
        aggregated_scores=agg_probs,
        group_cols=group_cols,
        score_type="soft_vote",
        source_level="subject",
    )


def mean_vote_subject_predictions(
    predictions: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    *,
    group_cols: Optional[Sequence[str]] = None,
    score_type: str = "auto",
) -> pd.DataFrame:
    df = _ensure_dataframe(predictions)
    group_cols = _resolve_group_cols(df, group_cols)

    score_type = str(score_type).lower()
    if score_type not in {"auto", "logits", "probs"}:
        raise ValueError(f"score_type must be one of ['auto', 'logits', 'probs'], got {score_type!r}")

    df, logit_cols = _ensure_logit_columns(df)
    df_prob, prob_cols = _ensure_probability_columns(df)

    use_logits = False
    if score_type == "logits":
        if len(logit_cols) == 0:
            raise ValueError("score_type='logits' requested, but no logit_* columns found.")
        use_logits = True
    elif score_type == "auto":
        use_logits = len(logit_cols) > 0

    grouped = list(df.groupby(group_cols, sort=False, dropna=False))

    if use_logits:
        agg_logits = []
        agg_probs = []
        for _, g in grouped:
            logits = g[logit_cols].to_numpy(dtype=np.float64)
            mean_logits = np.mean(logits, axis=0)
            agg_logits.append(mean_logits)
            agg_probs.append(_normalize_prob_row(_softmax_numpy(mean_logits[None, :])[0]))
        agg_logits = np.asarray(agg_logits, dtype=np.float64)
        agg_probs = np.asarray(agg_probs, dtype=np.float64)

        return _aggregate_subject_table_from_scores(
            df,
            aggregated_scores=agg_probs,
            group_cols=group_cols,
            score_type="mean_vote_logits",
            aggregated_logits=agg_logits,
            source_level="subject",
        )

    agg_probs = []
    grouped_prob = list(df_prob.groupby(group_cols, sort=False, dropna=False))
    for _, g in grouped_prob:
        probs = g[prob_cols].to_numpy(dtype=np.float64)
        agg_probs.append(np.mean(probs, axis=0))
    agg_probs = np.asarray(agg_probs, dtype=np.float64)

    return _aggregate_subject_table_from_scores(
        df_prob,
        aggregated_scores=agg_probs,
        group_cols=group_cols,
        score_type="mean_vote_probs",
        source_level="subject",
    )


def aggregate_instance_predictions_to_subject(
    predictions: Optional[Union[pd.DataFrame, Sequence[Mapping[str, Any]]]] = None,
    *,
    subject_ids: Optional[Sequence[Any]] = None,
    true_labels: Optional[Sequence[int]] = None,
    pred_labels: Optional[Sequence[int]] = None,
    probs: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    instance_ids: Optional[Sequence[Any]] = None,
    split: Optional[Union[str, Sequence[Any]]] = None,
    fold: Optional[Union[int, Sequence[Any]]] = None,
    split_seed: Optional[Union[int, Sequence[Any]]] = None,
    source_level: Optional[Union[str, Sequence[Any]]] = None,
    method: str = "soft_vote",
    group_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = _ensure_dataframe(
        predictions,
        subject_ids=subject_ids,
        true_labels=true_labels,
        pred_labels=pred_labels,
        probs=probs,
        logits=logits,
        instance_ids=instance_ids,
        split=split,
        fold=fold,
        split_seed=split_seed,
        source_level=source_level,
    )
    group_cols = _resolve_group_cols(df, group_cols)
    method = str(method).lower()

    if method in {"soft_vote", "soft"}:
        return soft_vote_subject_predictions(df, group_cols=group_cols)

    if method in {"mean", "mean_vote"}:
        return mean_vote_subject_predictions(df, group_cols=group_cols, score_type="auto")

    if method in {"identity", "none"}:
        counts = df.groupby(group_cols, sort=False, dropna=False).size().reset_index(name="num_instances")
        if np.any(counts["num_instances"].to_numpy() != 1):
            raise ValueError(
                "method='identity' requires one row per subject group. "
                "Found duplicate rows; use soft_vote or mean_vote instead."
            )

        df_out, prob_cols = _ensure_probability_columns(df)
        probs_arr = df_out[prob_cols].to_numpy(dtype=np.float64)
        probs_arr = probs_arr / np.clip(probs_arr.sum(axis=1, keepdims=True), 1e-12, None)
        df_out["pred_label"] = np.argmax(probs_arr, axis=1).astype(int)
        if "num_instances" not in df_out.columns:
            df_out["num_instances"] = 1
        if "aggregation_method" not in df_out.columns:
            df_out["aggregation_method"] = "identity"
        if "source_level" not in df_out.columns:
            df_out["source_level"] = "subject"
        return df_out.reset_index(drop=True)

    raise ValueError(f"Unknown aggregation method: {method!r}")


def summarize_fold_results(
    predictions: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    *,
    subject_aggregation: str = "soft_vote",
    group_cols: Optional[Sequence[str]] = None,
    return_subject_predictions: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], pd.DataFrame]]:
    df = _ensure_dataframe(predictions)
    group_cols = _resolve_group_cols(df, group_cols)

    counts = df.groupby(group_cols, sort=False, dropna=False).size().to_numpy()
    needs_aggregation = np.any(counts > 1)

    if needs_aggregation:
        subject_df = aggregate_instance_predictions_to_subject(
            df,
            method=subject_aggregation,
            group_cols=group_cols,
        )
    else:
        subject_df = aggregate_instance_predictions_to_subject(
            df,
            method="identity",
            group_cols=group_cols,
        )

    prob_cols, logit_cols = _find_score_columns(subject_df)
    y_true = subject_df["true_label"].to_numpy(dtype=np.int64)
    y_pred = subject_df["pred_label"].to_numpy(dtype=np.int64) if "pred_label" in subject_df.columns else None
    probs_arr = subject_df[prob_cols].to_numpy(dtype=np.float64) if len(prob_cols) > 0 else None
    logits_arr = subject_df[logit_cols].to_numpy(dtype=np.float64) if len(logit_cols) > 0 else None

    metric_summary = summarize_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs_arr,
        logits=logits_arr,
        num_classes=(len(prob_cols) if len(prob_cols) > 0 else None),
    )

    summary: Dict[str, Any] = {
        "num_instances": int(len(df)),
        "num_subjects": int(len(subject_df)),
        "subject_aggregation": subject_aggregation if needs_aggregation else "identity",
        **metric_summary,
    }

    for col in _DEFAULT_META_COLS:
        if col in subject_df.columns:
            values = subject_df[col].dropna().unique().tolist()
            if len(values) == 1:
                summary[col] = values[0]

    if return_subject_predictions:
        return summary, subject_df
    return summary


def summarize_cv_results(
    fold_summaries: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    *,
    metric_keys: Optional[Sequence[str]] = None,
    group_by: Sequence[str] = ("split",),
) -> Dict[str, Any]:
    if isinstance(fold_summaries, pd.DataFrame):
        summary_df = fold_summaries.copy()
    else:
        summary_df = pd.DataFrame(list(fold_summaries))

    if summary_df.empty:
        return {
            "num_entries": 0,
            "group_by": list(group_by),
            "per_entry": [],
            "groups": {},
            "overall": {},
        }

    if metric_keys is None:
        metric_keys = [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "brier_score",
            "roc_auc_macro_ovr",
            "pr_auc_macro_ovr",
        ]

    metric_keys = [m for m in metric_keys if m in summary_df.columns]

    def _aggregate_numeric(df_part: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {"num_entries": int(len(df_part))}
        for key in metric_keys:
            vals = pd.to_numeric(df_part[key], errors="coerce").dropna().to_numpy(dtype=np.float64)
            if len(vals) == 0:
                out[key] = {"mean": None, "std": None, "min": None, "max": None}
            else:
                out[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=0)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        return out

    groups: Dict[str, Any] = {}
    if len(group_by) > 0:
        valid_group_by = [g for g in group_by if g in summary_df.columns]
        if len(valid_group_by) > 0:
            for key, g in summary_df.groupby(valid_group_by, sort=False, dropna=False):
                if not isinstance(key, tuple):
                    key = (key,)
                key_name = "|".join(f"{col}={val}" for col, val in zip(valid_group_by, key))
                groups[key_name] = {
                    "group_values": {col: _jsonable(val) for col, val in zip(valid_group_by, key)},
                    **_aggregate_numeric(g),
                }

    overall = _aggregate_numeric(summary_df)

    return {
        "num_entries": int(len(summary_df)),
        "group_by": list(group_by),
        "metric_keys": list(metric_keys),
        "per_entry": _jsonable(summary_df.to_dict(orient="records")),
        "groups": groups,
        "overall": overall,
    }


def save_predictions_csv(
    predictions: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    path: Union[str, Path],
    *,
    index: bool = False,
) -> str:
    df = _ensure_dataframe(predictions)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return str(path)


def save_summary_json(
    summary: Mapping[str, Any],
    path: Union[str, Path],
    *,
    indent: int = 2,
) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(dict(summary)), f, indent=indent, ensure_ascii=False)
    return str(path)


if __name__ == "__main__":
    segment_rows = [
        {"subject_id": "S1", "true_label": 0, "instance_id": "S1_seg0", "prob_0": 0.80, "prob_1": 0.15, "prob_2": 0.05, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
        {"subject_id": "S1", "true_label": 0, "instance_id": "S1_seg1", "prob_0": 0.70, "prob_1": 0.20, "prob_2": 0.10, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
        {"subject_id": "S1", "true_label": 0, "instance_id": "S1_seg2", "prob_0": 0.60, "prob_1": 0.30, "prob_2": 0.10, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
        {"subject_id": "S2", "true_label": 1, "instance_id": "S2_seg0", "prob_0": 0.10, "prob_1": 0.75, "prob_2": 0.15, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
        {"subject_id": "S2", "true_label": 1, "instance_id": "S2_seg1", "prob_0": 0.15, "prob_1": 0.70, "prob_2": 0.15, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
        {"subject_id": "S3", "true_label": 2, "instance_id": "S3_seg0", "prob_0": 0.20, "prob_1": 0.20, "prob_2": 0.60, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
        {"subject_id": "S3", "true_label": 2, "instance_id": "S3_seg1", "prob_0": 0.25, "prob_1": 0.15, "prob_2": 0.60, "split": "test", "fold": 0, "split_seed": 10, "source_level": "segment"},
    ]
    segment_df = pd.DataFrame(segment_rows)

    subject_df = aggregate_instance_predictions_to_subject(
        segment_df,
        method="soft_vote",
    )
    print("Subject-level predictions:")
    print(subject_df)

    fold_summary, _ = summarize_fold_results(
        segment_df,
        subject_aggregation="soft_vote",
        return_subject_predictions=True,
    )
    print("\nFold summary:")
    print(json.dumps(_jsonable(fold_summary), indent=2))

    cv_summary = summarize_cv_results([fold_summary], group_by=("split",))
    print("\nCV summary:")
    print(json.dumps(_jsonable(cv_summary), indent=2))
