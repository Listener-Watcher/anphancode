# ============================================================
# CAUEEG H5 official-split RF baseline
# ============================================================

import os
import json
import h5py
import argparse
import random
from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from caueeg_loader_min import load_caueeg_task_datasets

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

try:
    from utils_all import set_global_seed
except Exception:
    def set_global_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)


def list_h5_subject_ids(h5_path: str) -> list[str]:
    with h5py.File(h5_path, "r") as f:
        return sorted(list(f["subjects"].keys()))


def get_official_serials_from_dataset(ds):
    """
    Avoid loading EEG signals. CauEegDataset stores annotation rows in data_list.
    """
    if hasattr(ds, "data_list"):
        return [str(row["serial"]) for row in ds.data_list]

    # fallback, slower because it calls __getitem__
    return [str(ds[i]["serial"]) for i in range(len(ds))]


def resolve_h5_subject_ids_from_official_serials(
    h5_path,
    serials,
    split_name,
    *,
    bad_ids=None,
):
    """
    Official split comes from CAUEEG json.
    This function only maps those official serials to the actual H5 keys.

    It does NOT decide train/val/test from H5 key prefixes.
    """
    bad_ids = set() if bad_ids is None else {str(x) for x in bad_ids}

    with h5py.File(h5_path, "r") as f:
        h5_keys = set(f["subjects"].keys())

    out = []
    missing = []

    for serial in serials:
        serial = str(serial)

        if serial in bad_ids:
            continue

        candidates = [
            serial,                  # H5 built without prefix
            f"{split_name}_{serial}", # H5 built with prefix
        ]

        matched = None
        for key in candidates:
            if key in h5_keys:
                matched = key
                break

        if matched is None:
            missing.append(serial)
        else:
            out.append(matched)

    if len(missing) > 0:
        raise KeyError(
            f"{len(missing)} official {split_name} serials were not found in H5. "
            f"First missing: {missing[:10]}"
        )

    return out


def get_official_caueeg_h5_split_ids(
    *,
    dataset_path,
    task,
    file_format,
    h5_path,
    bad_ids=None,
):
    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    train_serials = get_official_serials_from_dataset(train_set)
    val_serials = get_official_serials_from_dataset(val_set)
    test_serials = get_official_serials_from_dataset(test_set)

    train_ids = resolve_h5_subject_ids_from_official_serials(
        h5_path, train_serials, "train", bad_ids=bad_ids
    )
    val_ids = resolve_h5_subject_ids_from_official_serials(
        h5_path, val_serials, "val", bad_ids=bad_ids
    )
    test_ids = resolve_h5_subject_ids_from_official_serials(
        h5_path, test_serials, "test", bad_ids=bad_ids
    )

    print("[Official split]")
    print("train:", len(train_ids))
    print("val  :", len(val_ids))
    print("test :", len(test_ids))

    return config, train_ids, val_ids, test_ids



def zscore_node_features(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Same idea as your graph pipeline:
    standardize each segment graph across nodes, feature by feature.
    x: [num_channels, num_features]
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + eps)


def flatten_adj(
    adj: np.ndarray,
    *,
    use_upper_triangle: bool = True,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
) -> np.ndarray:
    adj = np.asarray(adj, dtype=np.float32)

    if symmetrize:
        adj = 0.5 * (adj + adj.T)

    if zero_diagonal:
        np.fill_diagonal(adj, 0.0)

    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

    if use_upper_triangle:
        iu = np.triu_indices(adj.shape[0], k=1)
        return adj[iu].astype(np.float32)

    return adj.reshape(-1).astype(np.float32)


def h5_subject_to_rf_rows(
    h5f,
    sid: str,
    *,
    feature_families: Sequence[str],
    connectivity_metric: Optional[str] = None,
    connectivity_band: Optional[int] = None,
    include_edges: bool = False,
    standardize_features: bool = True,
    use_upper_triangle: bool = True,
):
    grp = h5f[f"subjects/{sid}"]
    label = int(grp["metadata"].attrs["label"])

    # node features: concatenate families along feature dimension
    feat_list = []
    for fam in feature_families:
        xfam = grp[f"windows/features/{fam}"][:]   # [W, C, F_fam]
        feat_list.append(np.asarray(xfam, dtype=np.float32))

    x_all = np.concatenate(feat_list, axis=-1)      # [W, C, F_total]
    num_windows = x_all.shape[0]

    seg_ids = grp["windows/raw/segment_id"][:].astype(int)
    start_samples = grp["windows/raw/start_sample"][:].astype(int)

    adj_all = None
    if include_edges:
        if connectivity_metric is None:
            raise ValueError("connectivity_metric must be provided when include_edges=True.")

        adj_all = np.asarray(
            grp[f"windows/connectivity/{connectivity_metric}"][:],
            dtype=np.float32,
        )

        # [W, B, C, C] -> select band
        if adj_all.ndim == 4:
            if connectivity_band is None:
                raise ValueError(
                    f"{connectivity_metric} is banded, so connectivity_band must be set."
                )
            adj_all = adj_all[:, int(connectivity_band)]

        if adj_all.ndim != 3:
            raise ValueError(f"Expected adjacency [W,C,C], got {adj_all.shape}")

    rows = []
    for w in range(num_windows):
        x = x_all[w]  # [C, F]

        if standardize_features:
            x = zscore_node_features(x)

        parts = [x.reshape(-1).astype(np.float32)]

        if include_edges:
            parts.append(
                flatten_adj(
                    adj_all[w],
                    use_upper_triangle=use_upper_triangle,
                    symmetrize=True,
                    zero_diagonal=True,
                )
            )

        feat_vec = np.concatenate(parts, axis=0).astype(np.float32)

        rows.append({
            "subject_id": sid,
            "bare_subject_id": sid.split("_", 1)[1] if "_" in sid else sid,
            "segment_id": int(seg_ids[w]),
            "start_sample": int(start_samples[w]),
            "label": label,
            "x": feat_vec,
        })

    return rows


def build_rf_dataframe_from_h5(
    h5_path: str,
    subject_ids: Sequence[str],
    *,
    feature_families: Sequence[str],
    connectivity_metric: Optional[str],
    connectivity_band: Optional[int],
    include_edges: bool,
    standardize_features: bool = True,
    use_upper_triangle: bool = True,
) -> pd.DataFrame:
    rows = []

    with h5py.File(h5_path, "r") as h5f:
        for sid in subject_ids:
            if sid not in h5f["subjects"]:
                raise KeyError(f"{sid} not found in H5.")
            rows.extend(
                h5_subject_to_rf_rows(
                    h5f,
                    sid,
                    feature_families=feature_families,
                    connectivity_metric=connectivity_metric,
                    connectivity_band=connectivity_band,
                    include_edges=include_edges,
                    standardize_features=standardize_features,
                    use_upper_triangle=use_upper_triangle,
                )
            )

    return pd.DataFrame(rows)


def sample_train_segments(
    df: pd.DataFrame,
    *,
    seed: int,
    base_k: Optional[int] = None,
    segment_selection_strategy: str = "original_random_k",
    cleancluster_manifest_path: Optional[str] = None,
):
    """
    RF equivalent of train segment selection.
    Validation/test should still use all segments.
    """
    if base_k is None:
        return df.copy()

    rng = np.random.default_rng(seed)
    strategy = str(segment_selection_strategy).lower()

    work = df.copy()

    if strategy != "original_random_k":
        if cleancluster_manifest_path is None:
            raise ValueError(
                f"cleancluster_manifest_path is required for {strategy}."
            )

        manifest = pd.read_csv(cleancluster_manifest_path)
        manifest["subject_id"] = manifest["subject_id"].astype(str)
        manifest["segment_id"] = manifest["segment_id"].astype(int)

        if "keep_clean" in manifest.columns:
            manifest["keep_clean"] = (
                manifest["keep_clean"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
        else:
            manifest["keep_clean"] = True

        keep_cols = ["subject_id", "segment_id", "keep_clean"]
        if "sampling_weight" in manifest.columns:
            keep_cols.append("sampling_weight")
        if "kmeans_cluster_id" in manifest.columns:
            keep_cols.append("kmeans_cluster_id")

        work = work.merge(
            manifest[keep_cols],
            left_on=["bare_subject_id", "segment_id"],
            right_on=["subject_id", "segment_id"],
            how="left",
            suffixes=("", "_manifest"),
        )

        work = work[work["keep_clean"] == True].copy()

    selected = []

    for sid, g in work.groupby("subject_id", sort=False):
        n = min(int(base_k), len(g))
        if n <= 0:
            continue

        if strategy in {"original_random_k", "clean_random_k"}:
            idx = rng.choice(g.index.to_numpy(), size=n, replace=False)

        elif strategy == "clean_weighted_k":
            if "sampling_weight" in g.columns:
                w = g["sampling_weight"].fillna(0).to_numpy(dtype=np.float64)
                if np.sum(w) <= 0:
                    p = None
                else:
                    p = w / np.sum(w)
            else:
                p = None
            idx = rng.choice(g.index.to_numpy(), size=n, replace=False, p=p)

        elif strategy == "all_clean":
            idx = g.index.to_numpy()

        elif strategy == "clean_kmeans_k":
            # simple representative version: sample across clusters first
            if "kmeans_cluster_id" not in g.columns:
                idx = rng.choice(g.index.to_numpy(), size=n, replace=False)
            else:
                picked = []
                for _, cg in g.groupby("kmeans_cluster_id", sort=True):
                    picked.append(rng.choice(cg.index.to_numpy(), size=1)[0])
                    if len(picked) >= n:
                        break
                if len(picked) < n:
                    remain = np.setdiff1d(g.index.to_numpy(), np.asarray(picked))
                    extra = rng.choice(remain, size=min(n - len(picked), len(remain)), replace=False)
                    picked.extend(extra.tolist())
                idx = np.asarray(picked)

        else:
            raise ValueError(f"Unknown segment_selection_strategy={strategy}")

        selected.append(work.loc[idx])

    if len(selected) == 0:
        raise RuntimeError("Segment selection produced 0 training rows.")

    return pd.concat(selected, axis=0).reset_index(drop=True)


def df_to_xy(df: pd.DataFrame):
    X = np.stack(df["x"].to_numpy(), axis=0).astype(np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    sid = df["subject_id"].to_numpy()
    return X, y, sid


def compute_metrics(y_true, y_pred, num_classes: int):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }


def aggregate_segments_to_subjects(subject_ids, y_true_seg, prob_seg, num_classes: int):
    rows = []

    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        prob = prob_seg[mask].mean(axis=0)
        pred = int(np.argmax(prob))
        true = int(y_true_seg[mask][0])

        row = {
            "subject_id": sid,
            "true_label": true,
            "pred_label": pred,
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = float(prob[c])
        rows.append(row)

    out = pd.DataFrame(rows)
    return out


def evaluate_rf_split(model, df_split: pd.DataFrame, *, split_name: str, seed: int, num_classes: int):
    X, y, sid = df_to_xy(df_split)

    seg_pred = model.predict(X)
    seg_prob_raw = model.predict_proba(X)

    # Ensure probability columns always align to [0, ..., num_classes-1]
    seg_prob = np.zeros((len(X), num_classes), dtype=np.float32)
    for j, cls in enumerate(model.classes_):
        seg_prob[:, int(cls)] = seg_prob_raw[:, j]

    seg_metrics = compute_metrics(y, seg_pred, num_classes)

    sub_df = aggregate_segments_to_subjects(
        subject_ids=sid,
        y_true_seg=y,
        prob_seg=seg_prob,
        num_classes=num_classes,
    )

    sub_metrics = compute_metrics(
        sub_df["true_label"].to_numpy(),
        sub_df["pred_label"].to_numpy(),
        num_classes,
    )

    seg_metrics.update({"split": f"{split_name}_segment", "seed": seed})
    sub_metrics.update({"split": split_name, "seed": seed})

    sub_df.insert(0, "split", split_name)
    sub_df.insert(0, "seed", seed)

    return seg_metrics, sub_metrics, sub_df


def run_caueeg_rf_official_h5(
    *,
    dataset_path: str,
    task: str,
    file_format: str,
    h5_path: str,
    output_root: str,
    feature_families: Sequence[str],
    connectivity_metric: Optional[str] = "wpli",
    connectivity_band: Optional[int] = 2,
    include_edges: bool = False,
    seeds: Sequence[int] = (15, 42, 100),
    base_k: Optional[int] = 10,
    segment_selection_strategy: str = "original_random_k",
    cleancluster_manifest_path: Optional[str] = None,
    n_estimators: int = 300,
    standardize_features: bool = True,
    use_upper_triangle: bool = True,
):
    # train_ids, val_ids, test_ids = infer_split_ids_from_h5(h5_path)
    bad_ids = {"00587", "00781", "01301"}

    config, train_ids, val_ids, test_ids = get_official_caueeg_h5_split_ids(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        h5_path=h5_path,
        bad_ids=bad_ids,
    )
    all_ids = train_ids + val_ids + test_ids

    print(f"train subjects: {len(train_ids)}")
    print(f"val subjects  : {len(val_ids)}")
    print(f"test subjects : {len(test_ids)}")

    # Build full table once. H5 already contains CAUEEG fixed windows.
    full_df = build_rf_dataframe_from_h5(
        h5_path=h5_path,
        subject_ids=all_ids,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        include_edges=include_edges,
        standardize_features=standardize_features,
        use_upper_triangle=use_upper_triangle,
    )

    num_classes = int(full_df["label"].max()) + 1

    train_df_full = full_df[full_df["subject_id"].isin(train_ids)].reset_index(drop=True)
    val_df = full_df[full_df["subject_id"].isin(val_ids)].reset_index(drop=True)
    test_df = full_df[full_df["subject_id"].isin(test_ids)].reset_index(drop=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "node_edge" if include_edges else "node_only"
    fam_tag = "_".join(feature_families)
    run_name = (
        f"{timestamp}_RF_{mode}_{fam_tag}_"
        f"{connectivity_metric}_band{connectivity_band}_"
        f"{segment_selection_strategy}_k{base_k}"
    )
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"log.txt")
    
    with open(log_path, "w") as f:
        f.write(f"data source {h5_path}, task {task}, file_format {file_format}\n")
        f.write(f"segment_selection_strategy {segment_selection_strategy}\n")
        f.write(f"base_k {base_k}\n")
        f.write(f"feature_families {feature_families}\n")

    all_summary_rows = []
    all_pred_rows = []

    for seed in seeds:
        set_global_seed(int(seed))

        seed_dir = os.path.join(run_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        train_df = sample_train_segments(
            train_df_full,
            seed=int(seed),
            base_k=base_k,
            segment_selection_strategy=segment_selection_strategy,
            cleancluster_manifest_path=cleancluster_manifest_path,
        )

        X_train, y_train, _ = df_to_xy(train_df)

        print(f"\nSeed {seed}")
        print("X_train:", X_train.shape)
        print("val rows:", len(val_df), "test rows:", len(test_df))

        rf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(seed),       # important: same seed loop as main pipeline
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        rf.fit(X_train, y_train)

        val_seg_metrics, val_sub_metrics, val_pred_df = evaluate_rf_split(
            rf, val_df, split_name="val", seed=int(seed), num_classes=num_classes
        )
        test_seg_metrics, test_sub_metrics, test_pred_df = evaluate_rf_split(
            rf, test_df, split_name="test", seed=int(seed), num_classes=num_classes
        )

        summary_rows = [
            val_seg_metrics,
            val_sub_metrics,
            test_seg_metrics,
            test_sub_metrics,
        ]

        pd.DataFrame(summary_rows).to_csv(
            os.path.join(seed_dir, "summary_metrics.csv"),
            index=False,
        )
        val_pred_df.to_csv(os.path.join(seed_dir, "val_predictions.csv"), index=False)
        test_pred_df.to_csv(os.path.join(seed_dir, "test_predictions.csv"), index=False)

        # Same style as main pipeline summary_test.csv
        summary_test = {
            "encoder_type": "RandomForest",
            "training_approach": "RF-segment-softvote-subject",
            "feature_mode": mode,
            "accuracy": test_sub_metrics["accuracy"],
            "balanced_accuracy": test_sub_metrics["balanced_accuracy"],
            "macro_f1": test_sub_metrics["macro_f1"],
            "confusion_matrix": json.dumps(test_sub_metrics["confusion_matrix"]),
            "feature_families": ",".join(feature_families),
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "include_edges": include_edges,
            "base_k": base_k,
            "segment_selection_strategy": segment_selection_strategy,
            "n_estimators": n_estimators,
            "seed": int(seed),
        }
        pd.DataFrame([summary_test]).to_csv(
            os.path.join(seed_dir, "summary_test.csv"),
            index=False,
        )

        all_summary_rows.append(summary_test)
        all_pred_rows.append(test_pred_df)

        print(
            f"Test subject soft-vote | "
            f"Acc={test_sub_metrics['accuracy']:.4f} "
            f"BalAcc={test_sub_metrics['balanced_accuracy']:.4f} "
            f"MacroF1={test_sub_metrics['macro_f1']:.4f}"
        )

    all_summary_df = pd.DataFrame(all_summary_rows)
    all_summary_df.to_csv(os.path.join(run_dir, "all_seed_summary_test.csv"), index=False)

    agg = all_summary_df[["accuracy", "balanced_accuracy", "macro_f1"]].agg(["mean", "std"])
    agg.to_csv(os.path.join(run_dir, "overall_summary_test.csv"))

    all_pred_df = pd.concat(all_pred_rows, axis=0, ignore_index=True)
    all_pred_df.to_csv(os.path.join(run_dir, "subject_predictions_all_seeds.csv"), index=False)

    print("\nAggregate across seeds:")
    print(agg)
    print("\nSaved to:", run_dir)

    return {
        "run_dir": run_dir,
        "summary": all_summary_df,
        "agg": agg,
        "predictions": all_pred_df,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out_h5",
        type=str,
        default="/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5",
    )
    parser.add_argument(
        "--feature_families_str",
        type=str,
        default="relative_band_power,statistical",
    )
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument("--base_k", type=int, default=None)
    parser.add_argument(
        "--segment_selection_strategy",
        type=str,
        default="original_random_k",
        choices=[
            "original_random_k",
            "clean_random_k",
            "all_clean",
        ],
    )
    parser.add_argument("--include_edges", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/anphan/Documents/CAUEEG/results-paper",
    )

    args = parser.parse_args()

    feature_families = [
        x.strip()
        for x in args.feature_families_str.split(",")
        if x.strip()
    ]

    if args.segment_selection_strategy.startswith("global_cluster"):
        cleancluster_manifest_path = "/home/anphan/Documents/CAUEEG/visualize-merged_sliding_random/global_segment_clusters/global_cluster_manifest.csv"
    else:
        cleancluster_manifest_path = "/home/anphan/Documents/CAUEEG/visualize/segment_selection/cleancluster/cleancluster_manifest.csv"
    


    task="dementia-no-overlap"
    file_format="edf"
    dataset_path="/home/anphan/Downloads/caueeg-dataset/"

    out = run_caueeg_rf_official_h5(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        h5_path=args.out_h5,
        output_root=args.output_root,
        feature_families=feature_families,
        connectivity_metric=args.connectivity_metric,
        connectivity_band=args.connectivity_band,
        include_edges=args.include_edges,
        seeds=(15, 42, 100),  # same as main pipeline
        base_k=args.base_k,
        segment_selection_strategy=args.segment_selection_strategy,
        cleancluster_manifest_path=cleancluster_manifest_path,
        n_estimators=args.n_estimators,
        standardize_features=True,
    )