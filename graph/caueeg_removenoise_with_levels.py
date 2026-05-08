
import os
import json
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from caueeg_loader_min import *
from master_builder import build_master_eeg_dataset
from utils_all import set_global_seed
from mil_full_std import load_h5_payload_for_subjects, SubjectMILClassifier, fit_mil_baseline

from mil_utils import SubjectBagGraphDataset, LabelAwareSubjectBagDataset, collate_subject_bags, build_graphs_from_payload_multiband, collate_subject_bags_multiband
from mil_utils import (
    collect_subject_embeddings,
    evaluate,
)
from mil_cluster_views import (
    build_augmented_train_dataset,
    debug_bag_dataset,
    smoke_test_augmented_loader,
    train_one_epoch_multiview_mil,
)
from prototype_mil_utils import (
    DEFAULT_REGION_TO_CHANNELS_MONO,
    fit_segment_prototype_model,
    attach_segment_prototypes,
    summarize_prototype_usage,
)
# ---------------------------------------------------------
# CAUEEG channel order: keep only first 19 EEG channels
# ---------------------------------------------------------
CAUEEG_EEG19 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4",
    "T6", "FZ", "CZ", "PZ",
]



SFREQ = 200.0
CROP_LEN = 2000      # 10 sec at 200 Hz
LATENCY = 2000       # skip first 10 sec like CEEDNet
OVERLAP = 0.5
STEP = int(CROP_LEN * (1.0 - OVERLAP))


def is_multiband_or_bank_encoder(encoder_type: str) -> bool:
    encoder_type = str(encoder_type).lower()
    return encoder_type in {
        "linkx_cnn5",
        "cnn5",
        "gnn_bank",
        "cnn_bank",
        "linkx_cnn_bank",
    }


def get_collate_for_encoder(encoder_type: str):
    if is_multiband_or_bank_encoder(encoder_type):
        return collate_subject_bags_multiband
    return collate_subject_bags

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and k == "pyg_batch":
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _batch_attr_to_numpy(pyg_batch, name, default_value=-1):
    """
    Extract graph-level PyG Batch attribute.
    Works for attributes like:
        proto_id, segment_id, start_sample
    """
    if not hasattr(pyg_batch, name):
        return None

    x = getattr(pyg_batch, name)

    if torch.is_tensor(x):
        return x.detach().cpu().view(-1).numpy()

    try:
        return np.asarray(x).reshape(-1)
    except Exception:
        return None


@torch.no_grad()
def save_segment_attention_with_prototypes(
    model,
    loader,
    device,
    save_path,
    num_classes=None,
    split_name="test",
):
    """
    Save one row per segment:

        split
        subject_id
        true_label
        subject_pred
        subject probability columns
        segment_id
        start_sample
        proto_id
        attention

    This assumes model(batch) returns:
        out["logits"]
        out["attn_list"]

    and each graph has:
        g.proto_id
        g.segment_id
        g.start_sample
    """
    model.eval()
    rows = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        out = model(batch)

        if "attn_list" not in out:
            raise KeyError(
                "Model output does not contain 'attn_list'. "
                "Make sure SubjectMILClassifier.forward() returns attn_list."
            )

        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        labels = batch["labels"].detach().cpu().numpy()
        subject_ids = list(batch["subject_ids"])
        bag_sizes = batch["bag_sizes"].detach().cpu().numpy().astype(int)

        pyg_batch = batch["pyg_batch"]

        proto_ids = _batch_attr_to_numpy(pyg_batch, "proto_id")
        segment_ids = _batch_attr_to_numpy(pyg_batch, "segment_id")
        start_samples = _batch_attr_to_numpy(pyg_batch, "start_sample")

        total_graphs = int(bag_sizes.sum())

        if proto_ids is None:
            proto_ids = np.full(total_graphs, -1)
        if segment_ids is None:
            segment_ids = np.arange(total_graphs)
        if start_samples is None:
            start_samples = np.full(total_graphs, -1)

        attn_list = out["attn_list"]

        start = 0
        for b, size in enumerate(bag_sizes):
            end = start + int(size)

            attn = attn_list[b]
            if torch.is_tensor(attn):
                attn = attn.detach().cpu().numpy()
            attn = np.asarray(attn).reshape(-1)

            if len(attn) != size:
                raise ValueError(
                    f"Attention length mismatch for subject {subject_ids[b]}: "
                    f"len(attn)={len(attn)}, bag_size={size}"
                )

            for local_i, global_i in enumerate(range(start, end)):
                row = {
                    "split": split_name,
                    "subject_id": subject_ids[b],
                    "true_label": int(labels[b]),
                    "subject_pred": int(preds[b]),
                    "segment_id": int(segment_ids[global_i]),
                    "start_sample": int(start_samples[global_i]),
                    "proto_id": int(proto_ids[global_i]),
                    "attention": float(attn[local_i]),
                }

                for c in range(probs.shape[1]):
                    row[f"prob_{c}"] = float(probs[b, c])

                rows.append(row)

            start = end

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    return df
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

def subject_id_probe(X, subject_ids):
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X, subject_ids)
    pred = clf.predict(X)
    return accuracy_score(subject_ids, pred)

def embedding_similarity_report(X, y, sid, max_n=3000):
    if len(X) > max_n:
        idx = np.random.choice(len(X), max_n, replace=False)
        X = X[idx]
        y = y[idx]
        sid = sid[idx]

    S = cosine_similarity(X)

    same_subject = sid[:, None] == sid[None, :]
    same_class = y[:, None] == y[None, :]
    eye = np.eye(len(X), dtype=bool)

    same_subject = same_subject & ~eye
    same_class_diff_subject = same_class & (~same_subject) & ~eye
    diff_class = (~same_class) & ~eye

    return {
        "same_subject_cos": float(S[same_subject].mean()) if same_subject.any() else None,
        "same_class_diff_subject_cos": float(S[same_class_diff_subject].mean()) if same_class_diff_subject.any() else None,
        "diff_class_cos": float(S[diff_class].mean()) if diff_class.any() else None,
    }
def linear_probe(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        multi_class="auto",
    )
    clf.fit(X_train, y_train)

    pred_train = clf.predict(X_train)
    pred_val = clf.predict(X_val)

    return {
        "train_acc": accuracy_score(y_train, pred_train),
        "train_bal_acc": balanced_accuracy_score(y_train, pred_train),
        "train_f1": f1_score(y_train, pred_train, average="macro"),
        "val_acc": accuracy_score(y_val, pred_val),
        "val_bal_acc": balanced_accuracy_score(y_val, pred_val),
        "val_f1": f1_score(y_val, pred_val, average="macro"),
    }
@torch.no_grad()
def collect_mil_embeddings(model, loader, device):
    model.eval()

    bag_X, bag_y, bag_sid = [], [], []
    seg_X, seg_y, seg_sid = [], [], []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)

        bag_emb = out["bag_emb"].detach().cpu().numpy()
        graph_emb = out["graph_emb"].detach().cpu().numpy()

        labels = batch["labels"].detach().cpu().numpy()
        subject_ids = list(batch["subject_ids"])
        bag_sizes = batch["bag_sizes"].detach().cpu().numpy().astype(int)

        bag_X.append(bag_emb)
        bag_y.extend(labels.tolist())
        bag_sid.extend(subject_ids)

        start = 0
        for sid, y, size in zip(subject_ids, labels, bag_sizes):
            end = start + int(size)
            seg_X.append(graph_emb[start:end])
            seg_y.extend([int(y)] * int(size))
            seg_sid.extend([sid] * int(size))
            start = end

    bag_X = np.concatenate(bag_X, axis=0)
    seg_X = np.concatenate(seg_X, axis=0)

    return {
        "bag_X": bag_X,
        "bag_y": np.asarray(bag_y),
        "bag_sid": np.asarray(bag_sid),
        "seg_X": seg_X,
        "seg_y": np.asarray(seg_y),
        "seg_sid": np.asarray(seg_sid),
    }
def collect_required_connectivity_metrics(
    bank_specs,
    default_connectivity_metric: str,
):
    metrics = {str(default_connectivity_metric)}
    if bank_specs is None:
        return sorted(metrics)

    for spec in bank_specs:
        m = spec.get("connectivity_metric", default_connectivity_metric)
        metrics.add(str(m))
    return sorted(metrics)
def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set:
    """
    Convert fixed_edges into a set of sorted integer node pairs.
    Supports:
      - integer edges: [(0,1), (1,2)]
      - channel-name edges: [("Fp1","F3"), ("F3","C3")]
    """
    if fixed_edges is None:
        return set()

    fixed_pairs = set()
    name_to_idx = None

    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names has length {len(channel_names)} but n_channels={n_channels}"
            )
        name_to_idx = {name: i for i, name in enumerate(channel_names)}

    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            if name_to_idx is None:
                raise ValueError(
                    "fixed_edges contains channel names, but channel_names was not provided."
                )
            if u not in name_to_idx or v not in name_to_idx:
                continue
            i, j = name_to_idx[u], name_to_idx[v]
        else:
            i, j = int(u), int(v)

        if i == j:
            continue
        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(u, v)} is out of range for {n_channels} nodes.")

        fixed_pairs.add(tuple(sorted((i, j))))

    return fixed_pairs



def dataset_to_subject_records_limited(
    dataset,
    *,
    limit: int = 5,
    bad_ids: Optional[set] = None,
):
    """
    Convert only the first `limit` valid CAUEEG recordings into subject records.

    This is useful for test-code/debug mode because it avoids iterating through
    the full train/val/test datasets.
    """
    bad_ids = set() if bad_ids is None else set(bad_ids)

    records = []
    subject_ids = []

    for sample in dataset:
        serial = str(sample["serial"])

        if serial in bad_ids:
            continue

        signal = sample["signal"]              # [21, T]
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal)

        if len(windows) == 0:
            continue

        rec = {
            "subject_id": serial,
            "label": label,
            "class_id": label,
            "sampling_rate": SFREQ,
            "channel_names": CAUEEG_EEG19,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": list(range(len(windows))),
            "recording_info": {
                "serial": serial,
                "age": age,
            },
        }

        records.append(rec)
        subject_ids.append(serial)

        if len(subject_ids) >= int(limit):
            break

    if len(records) == 0:
        raise RuntimeError(
            f"test_code mode selected 0 valid records. "
            f"limit={limit}, bad_ids={bad_ids}"
        )

    return records, subject_ids

def segment_recording(signal: np.ndarray,
                      crop_len: int = CROP_LEN,
                      step: int = STEP,
                      latency: int = LATENCY):
    """
    signal: [C, T]
    returns:
        windows: list[np.ndarray] each [19, crop_len]
        starts : list[int]
    """
    x = np.asarray(signal, dtype=np.float32)
    x = x[:19]  # drop EKG + photic

    total_len = x.shape[-1]
    starts = list(range(latency, total_len - crop_len + 1, step))

    windows = [x[:, s:s + crop_len].astype(np.float32, copy=False) for s in starts]
    return windows, starts

def build_graph_bank_from_specs(
    payload,
    subject_ids,
    *,
    feature_families,
    default_connectivity_metric,
    default_connectivity_band,
    default_filter_method,
    default_fixed_edges,
    channel_names,
    bank_specs,
    standardize_features=True,
):
    """
    Reuse existing build_graphs_from_payload(...) repeatedly and attach
    a bank [K, N, N] to each graph.

    Each spec can override:
      - name
      - connectivity_metric
      - connectivity_band
      - filter_method
      - fixed_edges
    """
    if bank_specs is None or len(bank_specs) == 0:
        raise ValueError("bank_specs must contain at least one candidate.")

    candidate_names = []
    candidate_graph_lists = []

    for spec_idx, spec in enumerate(bank_specs):
        name = str(spec.get("name", f"cand_{spec_idx}"))
        cand_metric = spec.get("connectivity_metric", default_connectivity_metric)

        # IMPORTANT:
        # do not blindly force one global default band for all candidates
        if "connectivity_band" in spec:
            cand_band = spec["connectivity_band"]
        else:
            cand_band = default_connectivity_band

        cand_filter_method = spec.get("filter_method", default_filter_method)
        cand_fixed_edges = spec.get("fixed_edges", default_fixed_edges)

        gs = build_graphs_from_payload(
            payload,
            subject_ids,
            feature_families=feature_families,
            connectivity_metric=cand_metric,
            connectivity_band=cand_band,
            filter_method=cand_filter_method,
            fixed_edges=cand_fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=standardize_features,
        )

        candidate_names.append(name)
        candidate_graph_lists.append(gs)

    base_graphs = candidate_graph_lists[0]

    def _graph_key(g):
        sid = str(getattr(g, "subject_id", ""))
        seg = int(getattr(g, "segment_id", -1))
        start = int(getattr(g, "start_sample", -1))
        return (sid, seg, start)

    # precompute maps once
    candidate_maps = []
    for cand_name, gs in zip(candidate_names, candidate_graph_lists):
        gmap = {}
        for g in gs:
            gmap[_graph_key(g)] = g
        candidate_maps.append(gmap)


    # attach [K, N, N] bank to each base graph
    for g in base_graphs:
        key = _graph_key(g)

        bank_adj = []
        bank_topo = []

        for cand_name, gmap in zip(candidate_names, candidate_maps):
            if key not in gmap:
                raise KeyError(f"Graph key {key} missing in candidate {cand_name!r}.")
            gg = gmap[key]

            if not hasattr(gg, "adj") or gg.adj is None:
                raise ValueError(
                    f"Candidate {cand_name!r} graph for key {key} is missing dense adj. "
                    "Make sure build_graphs_from_payload(..., attach_dense_adj=True)."
                )

            adj = gg.adj
            if torch.is_tensor(adj):
                adj = adj.detach().cpu().float()
            else:
                adj = torch.tensor(adj, dtype=torch.float32)

            topo = (adj != 0).float()

            bank_adj.append(adj)
            bank_topo.append(topo)

        g.adj_bank = torch.stack(bank_adj, dim=0)          # [K, N, N]
        g.topology_bank = torch.stack(bank_topo, dim=0)    # [K, N, N]
        g.topology_names = list(candidate_names)
        g.conn_stack = g.adj_bank
        g.conn_stack_names = list(candidate_names)
    return base_graphs, candidate_names
def dataset_to_subject_records(dataset):
    """
    Convert CauEegDataset split into records accepted by build_master_eeg_dataset().
    Use recording serial as MIL bag id.
    """
    records = []
    subject_ids = []

    for sample in dataset:
        signal = sample["signal"]              # [21, T]
        serial = str(sample["serial"])         # use recording id, not patient id
        label = int(sample["class_label"])     # task label
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal)

        if len(windows) == 0:
            continue

        rec = {
            "subject_id": serial,
            "label": label,
            "class_id": label,
            "sampling_rate": SFREQ,
            "channel_names": CAUEEG_EEG19,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": list(range(len(windows))),
            "recording_info": {
                "serial": serial,
                "age": age,
            },
        }
        records.append(rec)
        subject_ids.append(serial)

    return records, subject_ids
###------------------------------------ filter segments ----------------------------
def load_cleancluster_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    required = {
        "subject_id",
        "segment_id",
        "keep_clean",
        "kmeans_cluster_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"CleanCluster manifest missing columns: {missing}")

    # Robust bool handling in case CSV stores booleans as strings.
    if df["keep_clean"].dtype != bool:
        df["keep_clean"] = (
            df["keep_clean"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df["kmeans_cluster_id"] = df["kmeans_cluster_id"].astype(int)

    return df
def graph_key(g):
    return str(g.subject_id), int(g.segment_id)


def filter_graphs_by_manifest_keep_clean(
    graphs,
    manifest_df: pd.DataFrame,
):
    """
    Keep only graphs whose manifest row has keep_clean=True.
    """
    clean_keys = set(
        manifest_df.loc[
            manifest_df["keep_clean"],
            ["subject_id", "segment_id"],
        ]
        .itertuples(index=False, name=None)
    )

    out = [g for g in graphs if graph_key(g) in clean_keys]

    if len(out) == 0:
        raise RuntimeError("No graphs remain after CleanCluster filtering.")

    return out

########################## filter by global cluster ##########################
def load_global_cluster_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    required = {
        "subject_id",
        "segment_id",
        "global_cluster_id",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Global cluster manifest missing columns: {missing}")

    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)

    if "split" not in df.columns:
        df["split"] = df["subject_id"].str.extract(
            r"^(train|val|test)_", expand=False
        ).fillna("unknown")

    if "keep_clean" in df.columns and df["keep_clean"].dtype != bool:
        df["keep_clean"] = (
            df["keep_clean"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    return df



def select_global_cluster_random_graphs_from_manifest(
    graphs,
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    split_name: str | None = "train",
    cluster_col: str = "global_cluster_id",
    use_keep_clean_if_available: bool = False,
    fill_with_replacement: bool = True,
    save_selection_path: str | None = None,
):
    """
    Select k training segments per subject using global KMeans clusters.

    Logic:
      1. Match manifest rows to available graph objects.
      2. For each subject, group segments by global_cluster_id.
      3. Randomly pick at least one segment from each available cluster.
      4. If fewer than k selected, keep filling from remaining segments.
      5. If the subject has fewer than k unique segments and fill_with_replacement=True,
         sample extra duplicate segments with replacement.

    Validation/test should NOT use this. They should keep all segments.
    """
    rng = np.random.default_rng(seed)

    graph_lookup = {
        (str(g.subject_id), int(g.segment_id)): g
        for g in graphs
    }

    df = manifest_df.copy()

    if split_name is not None and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == str(split_name).lower()].copy()

    if use_keep_clean_if_available and "keep_clean" in df.columns:
        df = df[df["keep_clean"]].copy()

    # Keep only manifest rows that exist in this graph split.
    df = df[
        df.apply(
            lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup,
            axis=1,
        )
    ].copy()

    if len(df) == 0:
        raise RuntimeError("No global-cluster manifest rows match the provided graphs.")

    selected_graphs = []
    selected_rows = []

    for sid, sdf in df.groupby("subject_id"):
        sdf = sdf.copy()

        # cluster_id -> dataframe
        cluster_groups = {
            int(cid): cdf.copy()
            for cid, cdf in sdf.groupby(cluster_col)
        }

        cluster_ids = list(cluster_groups.keys())
        rng.shuffle(cluster_ids)

        chosen_keys = []
        chosen_manifest_rows = []

        # -------------------------------------------------
        # Pass 1: choose one random segment from each cluster
        # -------------------------------------------------
        for cid in cluster_ids:
            if len(chosen_keys) >= k:
                break

            cdf = cluster_groups[cid]
            row = cdf.sample(n=1, random_state=int(rng.integers(0, 1_000_000_000))).iloc[0]

            key = (str(row["subject_id"]), int(row["segment_id"]))
            if key not in chosen_keys:
                chosen_keys.append(key)
                chosen_manifest_rows.append(row)

        # -------------------------------------------------
        # Pass 2: fill remaining slots from unused segments
        # If subject has only 2-3 clusters, this naturally
        # keeps drawing more segments from those clusters.
        # -------------------------------------------------
        if len(chosen_keys) < k:
            chosen_key_set = set(chosen_keys)

            remaining = sdf[
                ~sdf.apply(
                    lambda r: (str(r["subject_id"]), int(r["segment_id"])) in chosen_key_set,
                    axis=1,
                )
            ].copy()

            need = k - len(chosen_keys)

            if len(remaining) > 0:
                fill_n = min(need, len(remaining))
                fill_df = remaining.sample(
                    n=fill_n,
                    replace=False,
                    random_state=int(rng.integers(0, 1_000_000_000)),
                )

                for _, row in fill_df.iterrows():
                    key = (str(row["subject_id"]), int(row["segment_id"]))
                    chosen_keys.append(key)
                    chosen_manifest_rows.append(row)

        # -------------------------------------------------
        # Pass 3: if fewer than k unique segments exist,
        # optionally sample with replacement.
        # -------------------------------------------------
        if len(chosen_keys) < k and fill_with_replacement:
            need = k - len(chosen_keys)

            fill_df = sdf.sample(
                n=need,
                replace=True,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )

            for _, row in fill_df.iterrows():
                key = (str(row["subject_id"]), int(row["segment_id"]))
                chosen_keys.append(key)
                chosen_manifest_rows.append(row)

        # Convert keys to graph objects.
        # copy.copy avoids sharing the exact same object if replacement created duplicates.
        for local_rank, key in enumerate(chosen_keys[:k]):
            g = copy.copy(graph_lookup[key])
            selected_graphs.append(g)

            row = dict(chosen_manifest_rows[local_rank])
            row["selected_rank"] = local_rank + 1
            row["selection_strategy"] = "global_cluster_random_k"
            selected_rows.append(row)

    if len(selected_graphs) == 0:
        raise RuntimeError("No graphs selected by global cluster random selector.")

    selection_df = pd.DataFrame(selected_rows)

    if save_selection_path is not None:
        os.makedirs(os.path.dirname(save_selection_path), exist_ok=True)
        selection_df.to_csv(save_selection_path, index=False)

    return selected_graphs, selection_df
import os
import copy
import numpy as np
import pandas as pd


def _sample_rows(
    df: pd.DataFrame,
    *,
    n: int,
    rng: np.random.Generator,
    replace: bool,
    weight_col: str | None = None,
) -> pd.DataFrame:
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    if weight_col is not None and weight_col in df.columns:
        w = df[weight_col].to_numpy(dtype=np.float64)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 1e-8, None)
        p = w / w.sum()
    else:
        p = None

    n_eff = n if replace else min(n, len(df))

    idx = rng.choice(
        np.arange(len(df)),
        size=n_eff,
        replace=replace,
        p=p,
    )

    return df.iloc[idx].copy()


def select_global_cluster_proportional_random_graphs_from_manifest(
    graphs,
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    split_name: str | None = "train",
    cluster_col: str = "global_cluster_id",

    # cluster allocation controls
    cluster_weight_alpha: float = 0.5,
    uniform_cluster_mix: float = 0.05,
    ensure_min_one_per_cluster: bool = False,
    min_cluster_size: int = 1,
    max_cluster_fraction: float | None = 0.60,

    # within-cluster sampling controls
    within_cluster_weight_col: str | None = None,
    use_keep_clean_if_available: bool = False,
    fill_with_replacement: bool = True,

    save_selection_path: str | None = None,
):
    """
    Select k training segments per subject using softened proportional sampling
    over that subject's global-cluster distribution.

    Main behavior:
      1. Match manifest rows to available graph objects.
      2. For each subject, compute cluster sizes.
      3. Optionally ignore very tiny clusters for allocation.
      4. Allocate k samples by softened cluster weights:
             weight_c = cluster_size_c ** cluster_weight_alpha
      5. Add a small uniform mixture so rare clusters still have a chance.
      6. Cap dominant clusters using max_cluster_fraction.
      7. Sample segments inside each chosen cluster.
      8. Fill remaining slots from UNUSED segments first.
      9. Only duplicate with replacement if the subject truly has fewer than k usable segments.

    Recommended default:
        cluster_weight_alpha=0.5
        uniform_cluster_mix=0.05
        ensure_min_one_per_cluster=False
        min_cluster_size=1 or 2
        max_cluster_fraction=0.60

    Validation/test should NOT use this selector. They should keep all segments.
    """
    if k is None:
        raise ValueError("k must be an integer, got None.")
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    if cluster_col not in manifest_df.columns:
        raise KeyError(f"manifest_df is missing cluster_col={cluster_col!r}")

    rng = np.random.default_rng(seed)

    graph_lookup = {
        (str(g.subject_id), int(g.segment_id)): g
        for g in graphs
    }

    df = manifest_df.copy()

    # -------------------------------------------------
    # Optional split filtering.
    # -------------------------------------------------
    if split_name is not None and "split" in df.columns:
        df = df[
            df["split"].astype(str).str.lower() == str(split_name).lower()
        ].copy()

    # -------------------------------------------------
    # Optional clean filtering.
    # -------------------------------------------------
    if use_keep_clean_if_available and "keep_clean" in df.columns:
        if df["keep_clean"].dtype != bool:
            df["keep_clean"] = (
                df["keep_clean"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes", "y"])
            )
        df = df[df["keep_clean"]].copy()

    # -------------------------------------------------
    # Keep only manifest rows that exist in the current graph split.
    # -------------------------------------------------
    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df[cluster_col] = df[cluster_col].astype(int)

    df = df[
        df.apply(
            lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup,
            axis=1,
        )
    ].copy()

    if len(df) == 0:
        raise RuntimeError("No global-cluster manifest rows match the provided graphs.")

    selected_graphs = []
    selected_rows = []

    # -------------------------------------------------
    # Per-subject selection.
    # -------------------------------------------------
    for sid, sdf in df.groupby("subject_id"):
        sdf = sdf.copy()

        if len(sdf) == 0:
            continue

        # Drop duplicated manifest rows for the same segment, if any.
        sdf = sdf.drop_duplicates(subset=["subject_id", "segment_id"]).copy()

        full_cluster_sizes = sdf[cluster_col].value_counts().sort_index()

        # Optional tiny-cluster filtering for allocation only.
        # If this removes all clusters, fall back to all clusters.
        alloc_cluster_sizes = full_cluster_sizes[
            full_cluster_sizes >= int(min_cluster_size)
        ]
        if len(alloc_cluster_sizes) == 0:
            alloc_cluster_sizes = full_cluster_sizes

        cluster_ids = alloc_cluster_sizes.index.to_numpy(dtype=int)
        cluster_counts = alloc_cluster_sizes.to_numpy(dtype=np.float64)

        allocation = {int(c): 0 for c in cluster_ids}

        # -------------------------------------------------
        # Step 1: optional cluster coverage.
        # I recommend False for main experiments.
        # -------------------------------------------------
        if ensure_min_one_per_cluster and len(cluster_ids) <= k:
            for c in cluster_ids:
                allocation[int(c)] = 1

        remaining_slots = k - sum(allocation.values())

        # -------------------------------------------------
        # Step 2: softened proportional allocation.
        # -------------------------------------------------
        if remaining_slots > 0:
            weights = np.power(cluster_counts, float(cluster_weight_alpha))
            weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
            weights = np.clip(weights, 1e-8, None)
            weights = weights / weights.sum()

            if uniform_cluster_mix > 0:
                mix = float(uniform_cluster_mix)
                mix = max(0.0, min(1.0, mix))

                uniform = np.ones_like(weights, dtype=np.float64) / len(weights)
                weights = (1.0 - mix) * weights + mix * uniform
                weights = weights / weights.sum()

            if max_cluster_fraction is None:
                max_per_cluster = None
            else:
                max_cluster_fraction = float(max_cluster_fraction)
                if not (0.0 < max_cluster_fraction <= 1.0):
                    raise ValueError(
                        f"max_cluster_fraction must be in (0, 1], got {max_cluster_fraction}"
                    )
                max_per_cluster = max(1, int(np.ceil(max_cluster_fraction * k)))

            for _ in range(remaining_slots):
                available = []
                available_weights = []

                for c, base_w in zip(cluster_ids, weights):
                    c = int(c)
                    n_unique_c = int(full_cluster_sizes.loc[c])

                    # Cannot allocate more unique samples than this cluster owns,
                    # unless replacement is needed later.
                    unique_capacity_ok = allocation[c] < n_unique_c

                    # Optional cap so one dominant cluster cannot occupy the whole bag.
                    cap_ok = True
                    if max_per_cluster is not None:
                        cap_ok = allocation[c] < max_per_cluster

                    if unique_capacity_ok and cap_ok:
                        available.append(c)
                        available_weights.append(base_w)

                # If caps make allocation impossible, relax the cap but still avoid replacement.
                if len(available) == 0:
                    for c, base_w in zip(cluster_ids, weights):
                        c = int(c)
                        n_unique_c = int(full_cluster_sizes.loc[c])
                        if allocation[c] < n_unique_c:
                            available.append(c)
                            available_weights.append(base_w)

                # If no unique capacity remains, stop here.
                # Replacement fill is handled later.
                if len(available) == 0:
                    break

                available_weights = np.asarray(available_weights, dtype=np.float64)
                available_weights = available_weights / available_weights.sum()

                chosen_c = int(
                    rng.choice(
                        np.asarray(available, dtype=int),
                        size=1,
                        replace=False,
                        p=available_weights,
                    )[0]
                )

                allocation[chosen_c] += 1

        # -------------------------------------------------
        # Step 3: sample inside each allocated cluster.
        # -------------------------------------------------
        chosen_rows = []

        for c, n_pick in allocation.items():
            n_pick = int(n_pick)
            if n_pick <= 0:
                continue

            cdf = sdf[sdf[cluster_col] == int(c)].copy()
            if len(cdf) == 0:
                continue

            replace = bool(fill_with_replacement and n_pick > len(cdf))

            sampled = _sample_rows(
                cdf,
                n=n_pick,
                rng=rng,
                replace=replace,
                weight_col=within_cluster_weight_col,
            )

            chosen_rows.append(sampled)

        if len(chosen_rows) > 0:
            chosen_df = pd.concat(chosen_rows, ignore_index=True)
        else:
            chosen_df = sdf.iloc[0:0].copy()

        # -------------------------------------------------
        # Step 4: remove accidental duplicates first.
        # This keeps duplicate replacement only as a true last resort.
        # -------------------------------------------------
        chosen_df = chosen_df.drop_duplicates(
            subset=["subject_id", "segment_id"],
            keep="first",
        ).copy()

        # -------------------------------------------------
        # Step 5: fill from UNUSED unique subject segments.
        # This fixes the bug in the previous version.
        # -------------------------------------------------
        if len(chosen_df) < k:
            need = k - len(chosen_df)

            chosen_keys = set(
                zip(
                    chosen_df["subject_id"].astype(str),
                    chosen_df["segment_id"].astype(int),
                )
            )

            remaining = sdf[
                ~sdf.apply(
                    lambda r: (str(r["subject_id"]), int(r["segment_id"])) in chosen_keys,
                    axis=1,
                )
            ].copy()

            if len(remaining) > 0:
                fill_n = min(need, len(remaining))

                fill_df = _sample_rows(
                    remaining,
                    n=fill_n,
                    rng=rng,
                    replace=False,
                    weight_col=within_cluster_weight_col,
                )

                chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        # -------------------------------------------------
        # Step 6: only now duplicate if subject has fewer than k usable segments.
        # -------------------------------------------------
        if len(chosen_df) < k and fill_with_replacement:
            need = k - len(chosen_df)

            fill_df = _sample_rows(
                sdf,
                n=need,
                rng=rng,
                replace=True,
                weight_col=within_cluster_weight_col,
            )

            chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        chosen_df = chosen_df.head(k).copy()

        # -------------------------------------------------
        # Save useful diagnostics.
        # -------------------------------------------------
        chosen_df["selected_rank"] = np.arange(1, len(chosen_df) + 1)
        chosen_df["selection_strategy"] = "global_cluster_proportional_random_k"
        chosen_df["cluster_weight_alpha"] = float(cluster_weight_alpha)
        chosen_df["uniform_cluster_mix"] = float(uniform_cluster_mix)
        chosen_df["ensure_min_one_per_cluster"] = bool(ensure_min_one_per_cluster)
        chosen_df["min_cluster_size"] = int(min_cluster_size)
        chosen_df["max_cluster_fraction"] = (
            np.nan if max_cluster_fraction is None else float(max_cluster_fraction)
        )
        chosen_df["num_subject_usable_segments"] = int(len(sdf))
        chosen_df["num_subject_clusters"] = int(full_cluster_sizes.shape[0])

        # Useful: selected cluster counts for debugging.
        selected_cluster_counts = chosen_df[cluster_col].value_counts().to_dict()

        for _, row in chosen_df.iterrows():
            key = (str(row["subject_id"]), int(row["segment_id"]))

            if key not in graph_lookup:
                # Should not happen because we filtered earlier.
                continue

            g = copy.copy(graph_lookup[key])
            selected_graphs.append(g)

            row_dict = dict(row)
            row_dict["selected_cluster_counts"] = json.dumps(
                {str(k_): int(v_) for k_, v_ in selected_cluster_counts.items()}
            )
            selected_rows.append(row_dict)

    if len(selected_graphs) == 0:
        raise RuntimeError("No graphs selected by global cluster proportional selector.")

    selection_df = pd.DataFrame(selected_rows)

    if save_selection_path is not None:
        os.makedirs(os.path.dirname(save_selection_path), exist_ok=True)
        selection_df.to_csv(save_selection_path, index=False)

    return selected_graphs, selection_df

#--------------------------------------------------------------------------------------
def summarize_graph_pool(graphs, name: str):
    from collections import defaultdict

    subject_to_count = defaultdict(int)
    label_to_subjects = defaultdict(set)

    for g in graphs:
        sid = str(g.subject_id)
        y = int(g.y.view(-1)[0].item())
        subject_to_count[sid] += 1
        label_to_subjects[y].add(sid)

    counts = np.array(list(subject_to_count.values()), dtype=np.int64)

    print(f"\n[{name}]")
    print(f"num graphs: {len(graphs)}")
    print(f"num subjects: {len(subject_to_count)}")
    print(f"segments per subject: min={counts.min()}, mean={counts.mean():.2f}, max={counts.max()}")
    print("subjects per label:", {k: len(v) for k, v in label_to_subjects.items()})


def select_clean_kmeans_graphs_from_manifest(
    graphs,
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    distance_col: str = "kmeans_centroid_distance",
):
    """
    Select up to k clean KMeans-representative segments per subject.

    Preferred:
        use kmeans_centroid_distance if your manifest has it.

    Fallback:
        use highest iforest_score within each clean cluster.
    """
    rng = np.random.default_rng(seed)

    graph_lookup = {graph_key(g): g for g in graphs}

    clean_df = manifest_df[manifest_df["keep_clean"]].copy()
    clean_df = clean_df[
        clean_df.apply(lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup, axis=1)
    ].copy()

    if len(clean_df) == 0:
        raise RuntimeError("No clean rows match the provided graphs.")

    selected_keys = []

    for sid, sdf in clean_df.groupby("subject_id"):
        chosen_rows = []

        # 1 representative per clean cluster.
        for cid, cdf in sdf.groupby("kmeans_cluster_id"):
            if distance_col in cdf.columns:
                row = cdf.sort_values(distance_col, ascending=True).iloc[0]
            elif "iforest_score" in cdf.columns:
                # Higher IForest score = cleaner.
                row = cdf.sort_values("iforest_score", ascending=False).iloc[0]
            else:
                row = cdf.sample(n=1, random_state=seed).iloc[0]

            chosen_rows.append(row)

        chosen_df = pd.DataFrame(chosen_rows)

        # If more than k clusters, keep largest / most stable clusters first.
        if len(chosen_df) > k:
            if "cluster_size" in chosen_df.columns:
                chosen_df = chosen_df.sort_values(
                    ["cluster_size"],
                    ascending=False,
                ).head(k)
            else:
                chosen_df = chosen_df.sample(n=k, random_state=seed)

        # If fewer than k clusters, fill from remaining clean segments.
        if len(chosen_df) < k:
            chosen_pairs = set(
                zip(chosen_df["subject_id"].astype(str), chosen_df["segment_id"].astype(int))
            )

            remaining = sdf[
                ~sdf.apply(
                    lambda r: (str(r["subject_id"]), int(r["segment_id"])) in chosen_pairs,
                    axis=1,
                )
            ]

            need = k - len(chosen_df)
            if len(remaining) > 0:
                fill_n = min(need, len(remaining))
                fill_df = remaining.sample(n=fill_n, random_state=seed)
                chosen_df = pd.concat([chosen_df, fill_df], ignore_index=True)

        for _, row in chosen_df.iterrows():
            selected_keys.append((str(row["subject_id"]), int(row["segment_id"])))

    out = [graph_lookup[key] for key in selected_keys if key in graph_lookup]

    if len(out) == 0:
        raise RuntimeError("No graphs selected by CleanCluster KMeans strategy.")

    return out


def weighted_sample_clean_graphs_from_manifest(
    graphs,
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    weight_col: str = "sampling_weight",
):
    """
    For each subject, sample k clean graphs using manifest sampling weights.
    """
    rng = np.random.default_rng(seed)

    graph_lookup = {
        (str(g.subject_id), int(g.segment_id)): g
        for g in graphs
    }

    df = manifest_df.copy()
    df = df[df["keep_clean"]].copy()

    if weight_col not in df.columns:
        raise KeyError(f"Manifest missing weight column: {weight_col}")

    selected_graphs = []

    for sid, sdf in df.groupby("subject_id"):
        sdf = sdf.copy()

        # Keep only rows that exist in the current graph split.
        sdf = sdf[
            sdf.apply(
                lambda r: (str(r["subject_id"]), int(r["segment_id"])) in graph_lookup,
                axis=1,
            )
        ]

        if len(sdf) == 0:
            continue

        n = min(k, len(sdf))

        weights = sdf[weight_col].to_numpy(dtype=np.float64)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 1e-8, None)
        probs = weights / weights.sum()

        chosen_pos = rng.choice(
            np.arange(len(sdf)),
            size=n,
            replace=False,
            p=probs,
        )

        chosen = sdf.iloc[chosen_pos]

        for _, row in chosen.iterrows():
            key = (str(row["subject_id"]), int(row["segment_id"]))
            selected_graphs.append(graph_lookup[key])

    if len(selected_graphs) == 0:
        raise RuntimeError("No graphs selected by weighted CleanCluster sampling.")

    return selected_graphs


# ---------------------------------------------------------
# Level conversion helpers: segment -> macro / subject
# Keep the original caueeg_removenoise.py training backbone unchanged.
# ---------------------------------------------------------
def ensure_graph_dense_attrs(g):
    """Ensure g.adj, g.edge_weight, and g.edge_attr are present/consistent."""
    if not hasattr(g, "adj") or g.adj is None:
        n = int(g.x.shape[0])
        adj = torch.zeros((n, n), dtype=torch.float32)
        if hasattr(g, "edge_index") and g.edge_index is not None:
            ew = getattr(g, "edge_weight", None)
            if ew is None:
                ew = torch.ones(g.edge_index.shape[1], dtype=torch.float32)
            adj[g.edge_index[0].detach().cpu(), g.edge_index[1].detach().cpu()] = ew.detach().cpu().float()
        g.adj = adj

    adj = g.adj.detach().cpu().float() if torch.is_tensor(g.adj) else torch.tensor(g.adj, dtype=torch.float32)
    adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)

    edge_index, edge_weight = dense_to_sparse(adj)
    g.edge_index = edge_index.long()
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()
    g.adj = adj.float()
    return g


def reduce_graph_tensor_stack(xs, how="mean"):
    """Reduce a list of graph tensors with the same shape."""
    stack = torch.stack([x.detach().cpu().float() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32) for x in xs], dim=0)
    how = str(how).lower()
    if how == "mean":
        return stack.mean(dim=0)
    if how == "median":
        return stack.median(dim=0).values
    if how == "max":
        return stack.max(dim=0).values
    if how == "min":
        return stack.min(dim=0).values
    if how == "sum":
        return stack.sum(dim=0)
    raise ValueError(f"Unsupported level_reduce={how!r}. Use mean, median, max, min, or sum.")


def make_graph_from_dense_level(
    *,
    x,
    adj,
    y,
    subject_id,
    level,
    instance_id,
    segment_id=None,
    start_sample=None,
    end_sample=None,
    adj_bank=None,
    topology_bank=None,
    topology_names=None,
    conn_stack=None,
):
    """Create a PyG Data object from dense node features and dense adjacency."""
    adj = torch.nan_to_num(adj.float(), nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)
    edge_index, edge_weight = dense_to_sparse(adj)

    g = Data(
        x=x.float(),
        edge_index=edge_index.long(),
        y=torch.tensor([int(y)], dtype=torch.long),
    )
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()
    g.adj = adj.float()
    g.subject_id = str(subject_id)
    g.level = str(level)
    g.instance_id = str(instance_id)

    if segment_id is not None:
        g.segment_id = int(segment_id)
    if start_sample is not None:
        g.start_sample = int(start_sample)
    if end_sample is not None:
        g.end_sample = int(end_sample)

    if adj_bank is not None:
        g.adj_bank = adj_bank.float()
        g.topology_bank = (adj_bank.abs() > 0).float() if topology_bank is None else topology_bank.float()
        g.topology_names = list(topology_names or [f"cand_{i}" for i in range(adj_bank.shape[0])])

    if conn_stack is not None:
        # Needed by linkx_cnn5/cnn5 and bank-CNN collate paths.
        g.conn_stack = conn_stack.float()
        if not hasattr(g, "adj_bank") and conn_stack.ndim == 3:
            # Bank-CNN encoders can also treat conn_stack as a candidate stack.
            g.conn_stack_names = list(topology_names or [f"cand_{i}" for i in range(conn_stack.shape[0])])

    return g


def convert_segment_graphs_to_level(
    graphs,
    *,
    level="segment",
    macro_duration_sec=60.0,
    sfreq=SFREQ,
    reduce="mean",
):
    """
    Convert segment graphs into segment/macro/subject graph instances.

    - segment: no aggregation; one graph per segment.
    - macro: aggregate segments whose start_sample falls in the same macro block.
    - subject: aggregate all segments from one subject/recording into one graph.

    Call this after clean segment selection so macro/subject graphs are built
    only from the selected segment pool.
    """
    import copy as _copy
    from collections import defaultdict as _defaultdict

    level = str(level).lower()
    if level not in {"segment", "macro", "subject"}:
        raise ValueError("level must be one of: segment, macro, subject")

    graphs = [ensure_graph_dense_attrs(_copy.copy(g)) for g in graphs]

    if level == "segment":
        for g in graphs:
            g.level = "segment"
            g.instance_id = f"{g.subject_id}_seg{int(getattr(g, 'segment_id', 0))}"
        return list(graphs)

    macro_len_samples = max(int(round(float(macro_duration_sec) * float(sfreq))), 1)
    grouped = _defaultdict(list)

    for g in graphs:
        sid = str(g.subject_id)
        if level == "subject":
            group_id = 0
        else:
            start = int(getattr(g, "start_sample", 0))
            group_id = start // macro_len_samples
        grouped[(sid, group_id)].append(g)

    out = []
    for (sid, group_id), gs in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        y_values = [int(g.y.view(-1)[0].item()) for g in gs]
        if len(set(y_values)) != 1:
            raise ValueError(f"Mixed labels inside {level} group {(sid, group_id)}: {set(y_values)}")
        y = y_values[0]

        x = reduce_graph_tensor_stack([g.x for g in gs], how=reduce)
        adj = reduce_graph_tensor_stack([g.adj for g in gs], how=reduce)

        adj_bank = None
        topology_bank = None
        topology_names = None
        if hasattr(gs[0], "adj_bank") and getattr(gs[0], "adj_bank") is not None:
            adj_bank = reduce_graph_tensor_stack([g.adj_bank for g in gs], how=reduce)
            topology_bank = (adj_bank.abs() > 0).float()
            topology_names = list(getattr(gs[0], "topology_names", [f"cand_{i}" for i in range(adj_bank.shape[0])]))

        conn_stack = None
        if hasattr(gs[0], "conn_stack") and getattr(gs[0], "conn_stack") is not None:
            conn_stack = reduce_graph_tensor_stack([g.conn_stack for g in gs], how=reduce)

        starts = [int(getattr(g, "start_sample", 0)) for g in gs]
        segs = [int(getattr(g, "segment_id", -1)) for g in gs]
        instance_id = f"{sid}_{level}{group_id}"

        new_g = make_graph_from_dense_level(
            x=x,
            adj=adj,
            y=y,
            subject_id=sid,
            level=level,
            instance_id=instance_id,
            segment_id=min(segs) if segs else None,
            start_sample=min(starts) if starts else None,
            adj_bank=adj_bank,
            topology_bank=topology_bank,
            topology_names=topology_names,
            conn_stack=conn_stack,
        )
        new_g.num_source_segments = len(gs)
        new_g.source_segment_ids = segs
        out.append(new_g)

    return out


def run_caueeg_linkx_mil(
    dataset_path,
    fixed_edges,          
    channel_names,
    bag_aug_mode,
    task="abnormal",
    file_format="feather",
    out_h5="caueeg_master_linkx.h5",
    feature_families = ['relative_band_power', 'statistical', 'wavelet_energy'],
    connectivity_metric="pearson",
    connectivity_band=None,
    encoder_type = "linkx_cnn",
    mil_pool_type = "gated",
    filter_method="fixed",
    base_k=8,
    seed=42,
    batch_size=8,
    epochs=100,
    # lr=1e-3,
    # weight_decay=1e-4,
    device="cuda",
    rebuild_h5=False,
    use_lr_scheduler=True,
    use_center_loss=True,
    output_root="graph/results_caueeg_linkx",
    num_candidates: Optional[int] = None,
    bank_specs: Optional[List[Dict[str, Any]]] = None,
    bank_fusion_mode: str = "static",
    bank_topology_rule: str = "union",
    bank_vote_threshold: float = 0.5,
    bank_fusion_temperature: float = 1.0,
    bank_hidden_dim: int = 64,
    candidate_fusion_mode: str = "concat",
    candidate_fusion_hidden_dim: int = 64,
    candidate_fusion_dropout: float = 0.0,
    share_linkx_weights: bool = False,
    segment_selection_strategy: str = "original_random_k", # original_random_k | clean_random_k | clean_kmeans_k | all_clean
    cleancluster_manifest_path: Optional[str] = None,
    level: str = "segment",
    macro_duration_sec: float = 60.0,
    level_reduce: str = "mean",
    backbone: str = "gatv2",
    use_gcn_norm: bool = False,
    test_code=False,
    use_kmean_prototype: bool = False,
    num_prototypes: int = 8,
    use_contrastive_loss: bool = False,
    use_prototype_classifier: bool = False,

):
    os.makedirs(output_root, exist_ok=True)

    
    run_name = f"seed{seed}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"log.txt")

    bad_ids = {"00587", "00781", "01301", "train_00587", "train_00781", "train_01301"}

    patience=20
    start_epoch=20
    lr=0.0003
    weight_decay=0.005

    min_delta=1e-3
    top_k=3
    edge_mode="topology_weighted"
    graph_emb_dim=64
    dropout=0.3
    attn_dim=64
    set_global_seed(seed)

    max_k_per_subject = 300
    # feature_families = ['relative_band_power', 'statistical', 'wavelet_energy'] #'hjorth', 


    with open(log_path, "w") as f:
        f.write(f"data source {out_h5}, task {task}, file_format {file_format}\n")
        f.write(f"seeds {seed}\n")
        f.write(f"bank_specs {bank_specs}\n")
        f.write(f"candidate_fusion_mode: {candidate_fusion_mode}\n")
        f.write(f"note: bad_ids {bad_ids} \n")
        f.write(f"segment_selection_strategy: {segment_selection_strategy}\n")
        f.write(f"cleancluster_manifest_path: {cleancluster_manifest_path}\n")
        f.write(f"bag_aug_mode: {bag_aug_mode}\n")
        f.write(f"level: {level}, macro_duration_sec: {macro_duration_sec}, level_reduce: {level_reduce}\n")
        f.write(f"topology: {filter_method}, fixed_edges: {fixed_edges}, channel_names: {channel_names}\n")
        f.write(f"feature_families: {feature_families}\nconnectivity_metric: {connectivity_metric}, connectivity_band: {connectivity_band}\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"\n")

        f.write(f"model_name: {encoder_type}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta={min_delta}, top_k={top_k} \n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"use_center_loss {use_center_loss}, use_contrastive_loss {use_contrastive_loss}, use_prototype_classifier {use_prototype_classifier}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        # f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f"dropout={dropout}, use_lr_scheduler={use_lr_scheduler}, use_gcn_norm={use_gcn_norm}, use_kmean_prototype = {use_kmean_prototype}\n")
    

    # 1) official split
    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    # 2) convert each recording into subject-like records
    if test_code:
        test_n_subjects = 30
        epochs = 1
        print(f"[TEST_CODE] Using first {test_n_subjects} valid subjects per split.")

        train_records, train_ids = dataset_to_subject_records_limited(
            train_set,
            limit=test_n_subjects,
            bad_ids=bad_ids,
        )
        val_records, val_ids = dataset_to_subject_records_limited(
            val_set,
            limit=test_n_subjects,
            bad_ids=bad_ids,
        )
        test_records, test_ids = dataset_to_subject_records_limited(
            test_set,
            limit=test_n_subjects,
            bad_ids=bad_ids,
        )
    else:
        train_records, train_ids = dataset_to_subject_records(train_set)
        val_records, val_ids = dataset_to_subject_records(val_set)
        test_records, test_ids = dataset_to_subject_records(test_set)

    all_records = train_records + val_records + test_records

    train_ids_filter = [sid for sid in train_ids if sid not in bad_ids]
    val_ids_filter   = [sid for sid in val_ids if sid not in bad_ids]
    test_ids_filter = [sid for sid in test_ids if sid not in bad_ids]
    all_ids_filter   = train_ids_filter + val_ids_filter + test_ids_filter
    # all_ids = train_ids + val_ids + test_ids

    if test_code:
        print("[TEST_CODE] train_ids_filter:", train_ids_filter)
        print("[TEST_CODE] val_ids_filter:", val_ids_filter)
        print("[TEST_CODE] test_ids_filter:", test_ids_filter)

    if encoder_type in {"linkx_fused_bank", "gnn_bank", "linkx_bank", "cnn_bank", "linkx_cnn_bank"}:
        required_connectivity_metrics = collect_required_connectivity_metrics(
            bank_specs=bank_specs,
            default_connectivity_metric=connectivity_metric,
        )
    else:
        required_connectivity_metrics = [connectivity_metric]



    num_classes = len(sorted({r["label"] for r in all_records}))

    # 3) build or reuse H5
    need_build = rebuild_h5 or (not os.path.isfile(out_h5))
    if need_build:
        print(f"[H5] Building master file: {out_h5}")
        build_master_eeg_dataset(
            subject_records=all_records,
            output_h5_path=out_h5,
            feature_families=feature_families,
            connectivity_metrics=[connectivity_metric],
            overwrite=True,
            skip_bad_segments=False,
            target_sampling_rate=None,
            qc_input_unit="auto",
        )
    else:
        print(f"[H5] Reusing existing master file: {out_h5}")



    train_ids_suf = ['train_' + item for item in train_ids_filter]
    val_ids_suf = ['val_' + item for item in val_ids_filter]
    test_ids_suf = ['test_' + item for item in test_ids_filter]

    all_ids_suf = train_ids_suf + val_ids_suf + test_ids_suf

    payload_connectivity_band = None if encoder_type in {"linkx_fused_bank", "gnn_bank", "linkx_bank", "cnn_bank", "linkx_cnn_bank"} else connectivity_band

    # 4) load payload
    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=all_ids_suf,
        feature_families=feature_families,
        connectivity_metrics=required_connectivity_metrics,
        connectivity_band=payload_connectivity_band,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    # print(payload.keys())


    # 5) graphs
    # if connectivity_band is not None:
    if encoder_type in {"linkx_fused_bank", "linkx_bank", "gnn_bank", "cnn_bank", "linkx_cnn_bank"}:
        train_graphs, topology_names = build_graph_bank_from_specs(
            payload,
            train_ids_suf,
            feature_families=feature_families,
            default_connectivity_metric=connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=filter_method,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )
        val_graphs, _ = build_graph_bank_from_specs(
            payload,
            val_ids_suf,
            feature_families=feature_families,
            default_connectivity_metric=connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=filter_method,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )
        test_graphs, _ = build_graph_bank_from_specs(
            payload,
            test_ids_suf,
            feature_families=feature_families,
            default_connectivity_metric=connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=filter_method,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )
        num_candidates = len(topology_names)

    elif encoder_type not in ["linkx_cnn5", "cnn5"]:
        train_graphs = build_graphs_from_payload(
            payload, train_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )
        val_graphs = build_graphs_from_payload(
            payload, val_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )
        test_graphs = build_graphs_from_payload(
            payload, test_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )

    else:

        train_graphs = build_graphs_from_payload_multiband(
            payload, train_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )
        val_graphs = build_graphs_from_payload_multiband(
            payload, val_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )
        test_graphs = build_graphs_from_payload_multiband(
            payload, test_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )

    summarize_graph_pool(train_graphs, "train_graphs_original")
    summarize_graph_pool(val_graphs, "val_graphs_original")
    summarize_graph_pool(test_graphs, "test_graphs_original")

    selection_strategy = str(segment_selection_strategy).lower()
    manifest_df = None
    if selection_strategy != "original_random_k":
        if cleancluster_manifest_path is None:
            raise ValueError(
                "cleancluster_manifest_path must be provided when "
                f"segment_selection_strategy={segment_selection_strategy!r}"
            )
        manifest_df = load_global_cluster_manifest(cleancluster_manifest_path)

    if selection_strategy == "original_random_k":
        # No change.
        train_graphs_selected = train_graphs
        train_dataset_mode = "label_aware_random"

    elif selection_strategy == "clean_random_k":
        # Clean pool, but random sampling is still done by LabelAwareSubjectBagDataset.
        train_graphs_selected = filter_graphs_by_manifest_keep_clean(
            train_graphs,
            manifest_df,
        )
        train_dataset_mode = "label_aware_random"

    elif selection_strategy == "all_clean":
        # Use all clean segments.
        train_graphs_selected = filter_graphs_by_manifest_keep_clean(
            train_graphs,
            manifest_df,
        )
        train_dataset_mode = "fixed_all_selected"

    elif selection_strategy == "global_cluster_random_k":
        # Training only:
        # randomly select k segments per subject from global clusters.
        train_graphs_selected, selected_df = select_global_cluster_random_graphs_from_manifest(
            train_graphs,
            manifest_df,
            k=clean_k,
            seed=seed,
            split_name="train",
            cluster_col="global_cluster_id",
            use_keep_clean_if_available=False,
            fill_with_replacement=True,
            save_selection_path=os.path.join(
                run_dir,
                "selected_train_segments_global_cluster_random_k.csv",
            ),
        )

        train_dataset_mode = "fixed_all_selected"
    
    elif selection_strategy == "global_cluster_proportional_random_k":
        train_graphs_selected, selected_df = select_global_cluster_proportional_random_graphs_from_manifest(
            train_graphs,
            manifest_df,
            k=base_k,
            seed=seed,
            split_name="train",
            cluster_col="global_cluster_id",

            cluster_weight_alpha=0.5,
            uniform_cluster_mix=0.10,
            ensure_min_one_per_cluster=False,
            within_cluster_weight_col=None,
            use_keep_clean_if_available=False,
            min_cluster_size=1,
            max_cluster_fraction=0.60,
            fill_with_replacement=True,
            save_selection_path=os.path.join(
                run_dir,
                "selected_train_segments_global_cluster_proportional_random_k.csv",
            ),
        )

        train_graphs = train_graphs_selected
        train_dataset_mode = "fixed_all_selected"
        

    elif selection_strategy == "all_raw":

        train_dataset_mode = "fixed_all_selected"
        train_graphs_selected = train_graphs
    
    else:
        raise ValueError(
            f"Unknown segment_selection_strategy={segment_selection_strategy!r}. "
            "Use one of: original_random_k, clean_random_k, clean_kmeans_k, "
            "all_clean, clean_weighted_k, global_cluster_random_k, global_cluster_proportional_random_k."
        )

    train_graphs = train_graphs_selected
    summarize_graph_pool(train_graphs, f"train_graphs_after_{selection_strategy}")

    level = str(level).lower()
    if use_kmean_prototype and level != "segment":
        raise ValueError(
            "use_kmean_prototype currently expects segment-level graphs. "
            "Use level='segment' or disable use_kmean_prototype for macro/subject runs."
        )

    if use_kmean_prototype:

        proto_model = fit_segment_prototype_model(
            train_graphs,
            channel_names=channel_names,   # CAUEEG_EEG19
            region_to_channels=DEFAULT_REGION_TO_CHANNELS_MONO,
            num_prototypes=num_prototypes,
            pca_dim=32,
            seed=seed,
            include_node_flat=True,
            include_region_mean=True,      # important
            zscore_segment=True,
        )

        train_graphs = attach_segment_prototypes(
            train_graphs,
            proto_model,
            temperature=1.0,
            include_node_flat=True,
            include_region_mean=True,
            zscore_segment=True,
        )

        val_graphs = attach_segment_prototypes(
            val_graphs,
            proto_model,
            temperature=1.0,
            include_node_flat=True,
            include_region_mean=True,
            zscore_segment=True,
        )

        test_graphs = attach_segment_prototypes(
            test_graphs,
            proto_model,
            temperature=1.0,
            include_node_flat=True,
            include_region_mean=True,
            zscore_segment=True,
        )
        with open(log_path, "a") as f:
            f.write(f"Train prototype usage: {summarize_prototype_usage(train_graphs, num_prototypes)}")
            f.write(f"Val prototype usage: {summarize_prototype_usage(val_graphs, num_prototypes)}")
            f.write(f"Test prototype usage: {summarize_prototype_usage(test_graphs, num_prototypes)}")
        from prototype_mil_utils import (
            plot_prototype_scatter,
            plot_prototype_class_distribution,
            plot_prototype_region_signature,
            plot_prototype_node_feature_heatmaps,
            save_representative_segments,
        )

        viz_dir = os.path.join(run_dir, "prototype_viz")
        os.makedirs(viz_dir, exist_ok=True)

        class_names = ["normal", "mci", "dementia"]  # for dementia task
        # class_names = ["normal", "abnormal"]      # for abnormal task

        # 1. Segment space colored by prototype
        plot_prototype_scatter(
            train_graphs,
            proto_model,
            os.path.join(viz_dir, "train_segments_pca_by_prototype.png"),
            color_by="proto_id",
            method="pca",
            title="Train segment prototype space: colored by prototype",
        )

        # 2. Segment space colored by class
        plot_prototype_scatter(
            train_graphs,
            proto_model,
            os.path.join(viz_dir, "train_segments_pca_by_class.png"),
            color_by="true_label",
            method="pca",
            title="Train segment prototype space: colored by class",
        )

        # Optional: t-SNE version
        plot_prototype_scatter(
            train_graphs,
            proto_model,
            os.path.join(viz_dir, "train_segments_tsne_by_prototype.png"),
            color_by="proto_id",
            method="tsne",
            title="Train segment prototype space: t-SNE by prototype",
        )

        # 3. Prototype-class distribution
        class_dist_path, class_dist_df = plot_prototype_class_distribution(
            train_graphs,
            os.path.join(viz_dir, "prototype_class_distribution_train.png"),
            num_prototypes=num_prototypes,
            class_names=class_names,
            normalize="row",
        )

        class_dist_df.to_csv(os.path.join(viz_dir, "prototype_class_counts_train.csv"))

        # 4. Region-aware prototype signature
        region_path, region_df = plot_prototype_region_signature(
            train_graphs,
            proto_model,
            os.path.join(viz_dir, "prototype_region_signature_train.png"),
            num_prototypes=num_prototypes,
        )

        region_df.to_csv(os.path.join(viz_dir, "prototype_region_signature_train.csv"), index=True)

        # 5. Node-feature heatmaps per prototype
        plot_prototype_node_feature_heatmaps(
            train_graphs,
            os.path.join(viz_dir, "prototype_node_heatmaps"),
            num_prototypes=num_prototypes,
            channel_names=channel_names,
            feature_names=None,      # replace with feature names if you have them
            max_features=40,         # avoid huge plots if many features
        )

        # 6. Representative segments closest to each prototype
        rep_path, rep_df = save_representative_segments(
            train_graphs,
            os.path.join(viz_dir, "representative_segments_by_prototype.csv"),
            num_prototypes=num_prototypes,
            top_n=5,
        )

        print("Saved prototype visualizations to:", viz_dir)


    # 6) Optional level conversion for data preparation only.
    # Keep the rest of the MIL training backbone exactly the same.
    train_graphs = convert_segment_graphs_to_level(
        train_graphs,
        level=level,
        macro_duration_sec=macro_duration_sec,
        reduce=level_reduce,
    )
    val_graphs = convert_segment_graphs_to_level(
        val_graphs,
        level=level,
        macro_duration_sec=macro_duration_sec,
        reduce=level_reduce,
    )
    test_graphs = convert_segment_graphs_to_level(
        test_graphs,
        level=level,
        macro_duration_sec=macro_duration_sec,
        reduce=level_reduce,
    )

    summarize_graph_pool(train_graphs, f"train_{level}_instances")
    summarize_graph_pool(val_graphs, f"val_{level}_instances")
    summarize_graph_pool(test_graphs, f"test_{level}_instances")


    # 6) MIL bags

    base_collate_fn = get_collate_for_encoder(encoder_type)
    if bag_aug_mode != "none":
        if cleancluster_manifest_path is None:
            raise ValueError(
                "--cleancluster_manifest_path is required when --bag_aug_mode != none"
            )

        if manifest_df is None:
            manifest_df = load_global_cluster_manifest(cleancluster_manifest_path)


    train_dataset, train_collate_fn, requires_multiview_train_loop = build_augmented_train_dataset(
        train_graphs,
        bag_aug_mode=bag_aug_mode,
        base_k=base_k,
        seed=seed,
        max_k_per_subject=max_k_per_subject,

        # This is the key part for different encoder types.
        base_collate_fn=base_collate_fn,

        # Manifest / global cluster information.
        manifest_df=manifest_df,
        cluster_col=args.cluster_col,
        clean_col=args.clean_col,
        weight_col=args.weight_col,

        # Noise-like cluster filtering.
        exclude_noise_clusters=args.exclude_noise_clusters,
        min_cluster_size=args.min_cluster_size,
        min_clean_rate=args.min_clean_rate,

        # Pseudo-bag settings.
        pseudo_bags_per_epoch=args.pseudo_bags_per_epoch,
        pseudo_subjects_per_bag=args.pseudo_subjects_per_bag,
        p_real=args.p_real,

        # Multiview settings.
        multiview_max_cluster_views=args.multiview_max_cluster_views,
        min_segments_per_cluster_view=args.min_segments_per_cluster_view,
        min_cluster_fraction=args.min_cluster_fraction,
    )

    val_dataset = LabelAwareSubjectBagDataset(
        val_graphs,
        train=False,
        eval_k_per_subject=None,
        seed=seed,
    )

    test_dataset = LabelAwareSubjectBagDataset(
        test_graphs,
        train=False,
        eval_k_per_subject=None,
        seed=seed,
    )

    print("Train subject class counts:", np.bincount(train_dataset.subject_labels, minlength=num_classes))
    print("Val subject class counts:", np.bincount(val_dataset.subject_labels, minlength=num_classes))
    print("Test subject class counts:", np.bincount(test_dataset.subject_labels, minlength=num_classes))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=base_collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=base_collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    model = SubjectMILClassifier(
        num_node_features=train_dataset.num_node_features,
        num_classes=num_classes,
        num_nodes=train_dataset.num_nodes,
        encoder_type=encoder_type,
        edge_mode=edge_mode,
        graph_emb_dim=graph_emb_dim,
        dropout=dropout,
        mil_pool_type=mil_pool_type,
        attn_dim=attn_dim,
        cnn_num_bands=num_candidates,

        # graph_pool=graph_pool,
        # gnn_hidden_dim=gnn_hidden_dim,
        # node_hidden_dims=node_hidden_dims,
        # edge_hidden_dims=edge_hidden_dims,
        # branch_emb_dim=branch_emb_dim,

        num_candidates=num_candidates,
        bank_fusion_mode=bank_fusion_mode,
        bank_topology_rule=bank_topology_rule,
        bank_vote_threshold=bank_vote_threshold,
        bank_fusion_temperature=bank_fusion_temperature,
        bank_hidden_dim=bank_hidden_dim,

        candidate_fusion_mode=candidate_fusion_mode,
        candidate_fusion_hidden_dim=candidate_fusion_hidden_dim,
        candidate_fusion_dropout=candidate_fusion_dropout,
        share_linkx_weights=share_linkx_weights,
        graph_backbone=backbone,
        use_gcn_norm = use_gcn_norm,

        use_prototypes=use_kmean_prototype,
        num_prototypes=num_prototypes,
        prototype_emb_dim=16,
        prototype_hidden_dim=64,
        prototype_use_soft=True,
        prototype_use_dist=True,


        gcn_normalize_input=use_gcn_norm,
        gcn_norm_add_self_loops=use_gcn_norm,
        gcn_norm_abs_weights=False,
        gcn_norm_abs_degree=False,
        use_prototype_classifier=use_prototype_classifier,

    ).to(device)

    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    if args.debug_bag_aug:
        debug_bag_dataset(train_dataset, n=10)

        # Create model first, then run one loader smoke test.
        smoke_test_augmented_loader(
            model=model,
            loader=train_loader,
            device=device,
            multiview=requires_multiview_train_loop,
        )

        print("Debug bag augmentation completed. Exiting before training.")
        return
    if use_center_loss:
        from mil_utils import ClassCenterLoss
        center_loss_fn = ClassCenterLoss(
                num_classes=num_classes,
                emb_dim=graph_emb_dim,
                normalize=True,
            ).to(device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(center_loss_fn.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        center_loss_fn = None

    model, val_metrics, history, best_state = fit_mil_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        patience=patience,
        save_path=ckpt_path,
        start_epoch=start_epoch,
        min_delta=min_delta,
        top_k=top_k,
        use_center_loss=use_center_loss,
        center_loss_fn=center_loss_fn,
        verbose=True,
        use_lr_scheduler=use_lr_scheduler,
        lr_scheduler_metric="val_loss",
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=20,
        lr_scheduler_min_lr=1e-6,
        lr_scheduler_threshold=1e-3,
        lr_scheduler_cooldown=0,
        lr_scheduler_start_epoch=10,       # None => use start_epoch
        use_grad_norm_report=False,
        use_contrastive_loss=use_contrastive_loss

        )

    # 7) final evaluation
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    # after:
    # test_metrics = evaluate(model, test_loader, criterion, device)

    from mil_utils import plot_mil_learning_curves, plot_mil_learning_summary, collect_attention_weights
    plot_mil_learning_curves(history, save_dir=os.path.join(run_dir, "learning_curves"))
    plot_mil_learning_summary(
        history,
        save_path=os.path.join(run_dir, "learning_curve_summary.png"),
    )
    if mil_pool_type =="gated":
        attn_df, attn_summary_df = collect_attention_weights(
            model=model,          # or model after loading best_state
            loader=val_loader,
            device=device,
            split_name="val",
        )

        out_dir = os.path.join(run_dir,"attention_analysis")
        os.makedirs(out_dir, exist_ok=True)

        attn_df.to_csv(os.path.join(out_dir, "val_segment_attention.csv"), index=False)
        attn_summary_df.to_csv(os.path.join(out_dir, "val_subject_attention_summary.csv"), index=False)

        print(attn_summary_df.describe())
        plot_attention_summary(attn_summary_df, out_dir)
        plot_attention_per_subject(
            attn_df,
            out_dir=out_dir,
            max_subjects=None,
            sort_by_attention=True,
        )

        collapse_df = attn_summary_df[
            (attn_summary_df["top1_attention"] >= 0.7) |
            (attn_summary_df["effective_num_segments"] <= 2.0)
        ]

        print("Number of collapsed subjects:", len(collapse_df), "/", len(attn_summary_df))
        print(collapse_df[[
            "subject_id",
            "true_label",
            "pred_label",
            "bag_size",
            "top1_attention",
            "effective_num_segments",
            "normalized_entropy",
        ]])

    train_emb = collect_mil_embeddings(model, train_loader, device)
    val_emb = collect_mil_embeddings(model, val_loader, device)

    print("Bag embedding probe:")
    print(linear_probe(
        train_emb["bag_X"], train_emb["bag_y"],
        val_emb["bag_X"], val_emb["bag_y"],
    ))

    print("Segment embedding probe:")
    print(linear_probe(
        train_emb["seg_X"], train_emb["seg_y"],
        val_emb["seg_X"], val_emb["seg_y"],
    ))
    sid_acc = subject_id_probe(train_emb["seg_X"], train_emb["seg_sid"])
    print("Train segment -> subject_id accuracy:", sid_acc)
    print(embedding_similarity_report(
        train_emb["seg_X"],
        train_emb["seg_y"],
        train_emb["seg_sid"],
    ))
    if task == "abnormal":
        class_names = ["normal", "abnormal"]
    elif task in ["dementia", "dementia-no-overlap"]:
        class_names = ["normal", "mci", "dementia"]
    else:
        # fallback
        num_classes = len(np.unique(test_metrics["y_true"]))
        class_names = [f"class_{i}" for i in range(num_classes)]

    plot_linkx_mil_baseline_style(
        metrics=test_metrics,
        class_names=class_names,
        output_dir=run_dir,
        prefix="test"
    )

    if use_kmean_prototype:
        attention_csv_path = os.path.join(run_dir, "segment_attention_with_prototypes.csv")

        attention_df = save_segment_attention_with_prototypes(
            model=model,
            loader=test_loader,
            device=device,
            save_path=attention_csv_path,
            num_classes=num_classes,
            split_name="test",
        )

        print("Saved segment attention with prototypes to:", attention_csv_path)
        print(attention_df.head())

    summary_rows = [
        {
            "split": "train",
            "loss": float(train_metrics["loss"]),
            "accuracy": float(train_metrics["accuracy"]),
            "balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "macro_f1": float(train_metrics["macro_f1"]),
            "confusion_matrix": train_metrics["conf_matrix"],
        },
        {
            "split": "val",
            "loss": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "macro_f1": float(val_metrics["macro_f1"]),
            "confusion_matrix": val_metrics["conf_matrix"],

        },
        {
            "split": "test",
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "confusion_matrix": test_metrics["conf_matrix"],

        },
    ]

    summary_test = [
        {
            "encoder_type": encoder_type,
            "training_approach": "MIL-subject",
            "mil_pool_type": mil_pool_type,
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "confusion_matrix": test_metrics["conf_matrix"],
            "feature_families": feature_families,
            "topology": filter_method,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "edge_mode": edge_mode,
            "level": level,
            "macro_duration_sec": macro_duration_sec,
            "level_reduce": level_reduce,
            "base_k": base_k,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "start_epoch": start_epoch,
            "lr": lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "graph_emb_dim": graph_emb_dim,
            "attn_dim": attn_dim,
            "use_gcn_norm": use_gcn_norm,
            "use_kmean_prototype": use_kmean_prototype,     
            "seed": seed,

        },
    ]

    save_history_csv(history, os.path.join(run_dir, "history.csv"))
    save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))
    save_summary_metrics_csv(summary_test, os.path.join(run_dir, "summary_test.csv"))

    # train_pred_df = save_predictions_csv(
    #     model, train_loader, device,
    #     os.path.join(run_dir, "train_predictions.csv"),
    #     num_classes=num_classes,
    # )
    val_pred_df = save_predictions_csv(
        model, val_loader, device,
        os.path.join(run_dir, "val_predictions.csv"),
        num_classes=num_classes,
    )
    test_pred_df = save_predictions_csv(
        model, test_loader, device,
        os.path.join(run_dir, "test_predictions.csv"),
        num_classes=num_classes,
    )

    if use_kmean_prototype:
        from prototype_mil_utils import plot_attention_by_prototype

        attention_df = pd.read_csv(os.path.join(run_dir, "segment_attention_with_prototypes.csv"))

        plot_attention_by_prototype(
            attention_df,
            os.path.join(run_dir, "mean_attention_by_prototype_and_class.png"),
            num_prototypes=num_prototypes,
            class_names=class_names,
        )

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "history": history,
        "best_state": best_state,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "run_dir": run_dir,
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
        "summary_test": summary_test
    }

# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------

def make_jsonable(x):
    """Convert numpy/torch objects into JSON-safe Python objects."""
    if isinstance(x, dict):
        return {str(k): make_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_jsonable(v) for v in x]
    if isinstance(x, tuple):
        return [make_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    return x


def normalize_summary_row(row):
    """Prepare one summary row for CSV storage."""
    row = make_jsonable(row)

    # Store list-like config as a stable string
    if isinstance(row.get("feature_families"), list):
        row["feature_families"] = ",".join(map(str, row["feature_families"]))

    # Store confusion matrix as JSON string in CSV
    if "confusion_matrix" in row:
        row["confusion_matrix_json"] = json.dumps(row["confusion_matrix"])
        del row["confusion_matrix"]

    return row


def save_seed_aggregation(summary_rows, output_dir):
    """
    Save:
      1) all_seed_results.csv      : one row per seed
      2) aggregate_seed_results.csv: mean/std/min/max/count across seeds
      3) aggregate_confusion_matrix.json
    """
    os.makedirs(output_dir, exist_ok=True)

    rows = [normalize_summary_row(r) for r in summary_rows]
    df = pd.DataFrame(rows)

    raw_path = os.path.join(output_dir, "all_seed_results.csv")
    df.to_csv(raw_path, index=False)

    metric_cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    # These define one experimental variant.
    variant_cols = [
        "encoder_type",
        "training_approach",
        "mil_pool_type",
        "feature_families",
        "topology",
        "connectivity_metric",
        "connectivity_band",
        "edge_mode",
        "base_k",
        "batch_size",
        "epochs",
        "patience",
        "start_epoch",
        "lr",
        "dropout",
        "weight_decay",
        "graph_emb_dim",
        "attn_dim",
        "use_gcn_norm",
    ]
    variant_cols = [c for c in variant_cols if c in df.columns]

    agg = (
        df.groupby(variant_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )

    # Flatten multi-index columns
    agg.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in agg.columns
    ]

    # Add readable mean ± std columns
    for m in metric_cols:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        if mean_col in agg.columns and std_col in agg.columns:
            agg[f"{m}_mean_std"] = agg.apply(
                lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
                if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
                axis=1,
            )

    agg_path = os.path.join(output_dir, "aggregate_seed_results.csv")
    agg.to_csv(agg_path, index=False)

    # Aggregate confusion matrices separately
    cm_path = None
    if "confusion_matrix_json" in df.columns:
        cms = []
        for s in df["confusion_matrix_json"].dropna():
            cms.append(np.asarray(json.loads(s), dtype=float))

        if len(cms) > 0:
            cm_stack = np.stack(cms, axis=0)
            cm_info = {
                "num_seeds": int(len(cms)),
                "confusion_matrix_sum": cm_stack.sum(axis=0).astype(int).tolist(),
                "confusion_matrix_mean": cm_stack.mean(axis=0).tolist(),
                "confusion_matrix_std": cm_stack.std(axis=0, ddof=1).tolist()
                if len(cms) > 1 else np.zeros_like(cm_stack[0]).tolist(),
            }

            cm_path = os.path.join(output_dir, "aggregate_confusion_matrix.json")
            with open(cm_path, "w") as f:
                json.dump(cm_info, f, indent=2)

    print(f"Saved per-seed results: {raw_path}")
    print(f"Saved aggregate results: {agg_path}")
    if cm_path is not None:
        print(f"Saved aggregate confusion matrix: {cm_path}")

    return df, agg

def rows_to_prediction_df(rows, num_classes=None):
    records = []

    for r in rows:
        rec = {
            "subject_id": r["subject_id"],
            "true_label": int(r["label"]),
            "pred_label": int(r["pred"]),
        }

        prob = np.asarray(r["prob"], dtype=np.float32).reshape(-1)
        emb = np.asarray(r["embedding"], dtype=np.float32).reshape(-1)

        if num_classes is None:
            num_classes_local = len(prob)
        else:
            num_classes_local = int(num_classes)

        for i in range(num_classes_local):
            rec[f"prob_{i}"] = float(prob[i])

        # store embedding as one JSON string so CSV stays compact
        rec["embedding_json"] = json.dumps(emb.tolist())
        records.append(rec)

    return pd.DataFrame(records)


def save_predictions_csv(model, loader, device, csv_path, num_classes=None):
    rows = collect_subject_embeddings(model, loader, device)
    df = rows_to_prediction_df(rows, num_classes=num_classes)
    df.to_csv(csv_path, index=False)
    # print(f"Saved predictions: {csv_path}")
    return df


def save_summary_metrics_csv(summary_rows, csv_path):
    df = pd.DataFrame(summary_rows)
    df.to_csv(csv_path, index=False)
    # print(f"Saved summary metrics: {csv_path}")
    return df


def save_history_csv(history, csv_path):
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)
    # print(f"Saved history: {csv_path}")
    return df


def _safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=np.float64),
        where=(b != 0),
    )

def compute_classwise_sens_spec(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    sens = []
    spec = []

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sens.append(float(sensitivity))
        spec.append(float(specificity))

    return np.array(sens), np.array(spec), cm


def plot_linkx_mil_baseline_style(metrics, class_names, output_dir, prefix="test"):
    """
    metrics must come from evaluate(...)
    expects:
      metrics["y_true"]
      metrics["y_pred"]
      metrics["y_prob"]
      metrics["conf_matrix"]
    """
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    y_prob = np.asarray(metrics["y_prob"], dtype=np.float64)

    num_classes = len(class_names)

    # -----------------------------
    # 1) row-normalized confusion
    # -----------------------------
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = _safe_divide(cm, row_sum)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 2) class-wise metrics
    # -----------------------------
    sens, spec, _ = compute_classwise_sens_spec(y_true, y_pred, num_classes)

    x = np.arange(num_classes)
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, sens, width, label="Sensitivity")
    plt.bar(x + width / 2, spec, width, label="Specificity")
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Class-wise Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_classwise_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 3) ROC curve
    # -----------------------------
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(6, 5))

    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")

    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro-average (AUC={auc_micro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


def nullable_int(val):
    if val.lower() == "none":
        return None
    return int(val)


def find_existing_run_ignore_timestamp(output_base: str, run_core: str):
    """
    Return an existing run folder whose name matches run_core
    after ignoring the timestamp prefix.

    Example:
        existing folder:
        20260429_120530_dementia_segment_clean_weighted_k_wpli_k10_...

        run_core:
        dementia_segment_clean_weighted_k_wpli_k10_...
    """
    base = Path(output_base)
    if not base.exists():
        return None

    suffix = "_" + run_core

    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue

        if p.name == run_core or p.name.endswith(suffix):
            return p

    return None



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device", device)

    print('CUDA_VISIBLE_DEVICES =', os.environ.get('CUDA_VISIBLE_DEVICES'))
    print('torch cuda count =', torch.cuda.device_count())
    print('current logical cuda device =', torch.cuda.current_device())
    print('gpu name =', torch.cuda.get_device_name(0))

    root_path = "/home/anphan/Documents/CAUEEG"


    save_path = os.path.join(root_path,'result-bag_aug_mode-paper')
    os.makedirs(save_path,exist_ok = True)



    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--out_h5", type=str, default=None, required=False, help="out_h5")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    # parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--feature_families_str", type=str,  default="relative_band_power,statistical")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument("--use_lr_scheduler", action="store_true")
    parser.add_argument("--test_code", action="store_true")
    parser.add_argument("--use_gcn_norm", action="store_true")
    parser.add_argument("--use_kmean_prototype", action="store_true")
    parser.add_argument("--use_center_loss", action="store_true")
    parser.add_argument("--use_contrastive_loss", action="store_true")
    parser.add_argument("--use_prototype_classifier", action="store_true")
    parser.add_argument("--use_delta_in_graph_bank", action="store_true")



    parser.add_argument("--base_k", type=nullable_int, default=10, required=False, help="base_k")
    parser.add_argument("--level", type=str, default="segment", choices=["segment", "macro", "subject"],
                        help="Data preparation level: segment, macro, or subject")
    parser.add_argument("--macro_duration_sec", type=float, default=60.0,
                        help="Macro block size in seconds when --level macro")
    parser.add_argument("--level_reduce", type=str, default="mean", choices=["mean", "median", "max", "min", "sum"],
                        help="How to aggregate node features/connectivity for macro/subject level")
    parser.add_argument("--segment_selection_strategy", type=str, default="original_random_k", 
        choices=["original_random_k", "global_cluster_random_k", "global_cluster_proportional_random_k", 
                "all_raw", "clean_random_k", "clean_kmeans_k", "all_clean", "clean_weighted_k"])
    parser.add_argument("--candidate_fusion_mode", type=str, default="concat", 
        choices=["concat", "gated", "mean"])

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="LINKX",
        choices=["gnn", "edge_token", "gnn_bank","cnn_bank", "linkx_cnn_bank", 'linkx_bank', 'linkx_fused_bank', 'hybrid', 'gat', "LINKX", "linkx_cnn", "cnn5","linkx_cnn5", "mlp_node", "sage", "gcn2", "h2gcn"]
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="gatv2",
        choices=["gatv2", "gcn", "sage"]
    )

    parser.add_argument(
        "--bag_aug_mode",
        type=str,
        default="none",
        choices=[
            "none",
            "cluster_view_ce",
            "same_class_pseudo",
            "cluster_pseudo",
            "mixed_real_pseudo",
            "multiview_consistency",
        ],
    )

    parser.add_argument("--debug_bag_aug", action="store_true")

    parser.add_argument("--cleancluster_manifest_path", type=str, default=None)
    parser.add_argument("--cluster_col", type=str, default="global_cluster_id")
    parser.add_argument("--clean_col", type=str, default="keep_clean")
    parser.add_argument("--weight_col", type=str, default="sampling_weight")

    parser.add_argument("--exclude_noise_clusters", action="store_true")
    parser.add_argument("--min_cluster_size", type=int, default=50)
    parser.add_argument("--min_clean_rate", type=float, default=0.50)

    parser.add_argument("--pseudo_bags_per_epoch", type=int, default=1000)
    parser.add_argument("--pseudo_subjects_per_bag", type=int, default=5)
    parser.add_argument("--p_real", type=float, default=0.7)
    parser.add_argument("--multiview_max_cluster_views", type=int, default=3)
    parser.add_argument("--min_segments_per_cluster_view", type=int, default=5)
    parser.add_argument("--min_cluster_fraction", type=float, default=0.05)
    parser.add_argument("--lambda_consistency", type=float, default=0.10)
    parser.add_argument("--consistency_temperature", type=float, default=1.0)
    args = parser.parse_args()

    import config
    # channel_names = config.MONO_CHANNELS
    channel_names = CAUEEG_EEG19
    fixed_pairs = config.MONOFIXEDGES
    channel_name = "mono"
    n_channels = 19
    fixed_edges = _normalize_fixed_edges(fixed_pairs, n_channels, channel_names)
    feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    
    seeds_list = [15, 42, 100]
    agg_seed_results = []
    all_seed_results = []
    

    if args.use_delta_in_graph_bank:
        bank_specs = [
            {"name": "wpli_theta_full", 
            "connectivity_metric": "wpli", 
            "connectivity_band": 1, 
            "filter_method": "full"},
            {"name": "wpli_alpha_fixed", 
            "connectivity_metric": "wpli", 
            "connectivity_band": 2, 
            "filter_method": "fixed"},
            {"name": "coherence_alpha_combined", 
            "connectivity_metric": "coherence", 
            "connectivity_band": 2, 
            "filter_method": "combined"},
            {"name": "coherence_theta_topk4", 
            "connectivity_metric": "coherence", 
            "connectivity_band": 1, 
            "filter_method": "topk"},
            {"name": "coherence_delta_full", 
            "connectivity_metric": "coherence", 
            "connectivity_band": 0, 
            "filter_method": "full"},
        ]
    else:
        bank_specs = [
            {"name": "wpli_theta_full", 
            "connectivity_metric": "wpli", 
            "connectivity_band": 1, 
            "filter_method": "full"},
            {"name": "wpli_alpha_fixed", 
            "connectivity_metric": "wpli", 
            "connectivity_band": 2, 
            "filter_method": "fixed"},
            {"name": "coherence_alpha_combined", 
            "connectivity_metric": "coherence", 
            "connectivity_band": 2, 
            "filter_method": "combined"},
            {"name": "coherence_theta_topk4", 
            "connectivity_metric": "coherence", 
            "connectivity_band": 1, 
            "filter_method": "topk"},
        ]


    if args.test_code:
        save_path = os.path.join(save_path,'result_MIL-LinkX-level-testonly')
        os.makedirs(save_path,exist_ok = True)

    if args.use_kmean_prototype:

        save_path = os.path.join(save_path,'use_kmean_prototype')
        os.makedirs(save_path,exist_ok = True)

    if args.use_contrastive_loss:
        save_path = os.path.join(save_path,'use_contrastive_loss')
        os.makedirs(save_path,exist_ok = True)

    task="dementia-no-overlap"
    file_format="edf"
    dataset_path="/home/anphan/Downloads/caueeg-dataset/"
    # out_h5 = args.out_h5
    if args.out_h5 == None:
        out_h5 = "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
    else:
        out_h5 = args.out_h5
    # out_h5 = "/home/anphan/Documents/caueeg_merged_sliding_random_trainonly.h5"
    
    # out_h5="/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    segment_selection_strategy = args.segment_selection_strategy
    k_tag = f"k{args.base_k}" if segment_selection_strategy != "all_clean" else "allclean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_tag = "gcnnorm" if args.use_gcn_norm else "nonorm"
    center_loss = "centerloss" if args.use_center_loss else "ce_loss"
    run_name = (
        f"{args.level}_{segment_selection_strategy}_"
        f"{args.connectivity_metric}_{k_tag}_{args.encoder_type}_{args.mil_pool_type}_{args.topology}_{args.bag_aug_mode}"
    )
    
    cleancluster_manifest_path = "/home/anphan/Documents/CAUEEG/visualize-random/statistical_clusters_PCA5_N8/global_cluster_manifest.csv"
    
    # existing_run = find_existing_run_ignore_timestamp(save_path, run_name)
    
    # if existing_run is not None:
    #     print("=" * 80)
    #     print("[SKIP] Existing run found. Skip this run and move to next bash-loop item.")
    #     print(f"[SKIP] Existing folder: {existing_run}")
    #     print(f"[SKIP] Run core: {run_name}")
    #     print("=" * 80)
    #     sys.exit(0)


    output_dir = os.path.join(save_path, f"{timestamp}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)



    for seed in seeds_list:
        out = run_caueeg_linkx_mil(
            dataset_path=dataset_path,
            fixed_edges=fixed_edges,          
            channel_names=CAUEEG_EEG19,
            bag_aug_mode=args.bag_aug_mode,
            task=task,
            file_format=file_format,
            out_h5=out_h5,
            feature_families = feature_families,
            connectivity_metric = args.connectivity_metric,
            connectivity_band = args.connectivity_band,
            encoder_type = args.encoder_type,
            mil_pool_type = args.mil_pool_type,
            filter_method = args.topology,
            base_k=args.base_k,
            batch_size=16,
            seed=seed,
            use_lr_scheduler=args.use_lr_scheduler,
            use_center_loss=args.use_center_loss,
            # epochs=1,
            # lr=1e-4,
            # weight_decay=1e-3,
            device=device,
            rebuild_h5=False,
            output_root=output_dir,
            candidate_fusion_mode=args.candidate_fusion_mode,
            segment_selection_strategy=segment_selection_strategy,
            cleancluster_manifest_path=cleancluster_manifest_path,
            level=args.level,
            macro_duration_sec=args.macro_duration_sec,
            level_reduce=args.level_reduce,
            bank_specs = bank_specs,
            test_code = args.test_code,
            backbone = args.backbone,
            use_gcn_norm = args.use_gcn_norm,
            use_kmean_prototype = args.use_kmean_prototype,
            use_contrastive_loss = args.use_contrastive_loss,
            use_prototype_classifier = args.use_prototype_classifier,
        )

        # summary_test is a list with one dict
        seed_rows = out["summary_test"]
        # all_seed_results.append(seed_rows)        
        agg_seed_results.extend(seed_rows)

    # agg_dir = os.path.join(out['run_dir'], "all_seed_results.csv")
    # save_summary_metrics_csv(all_seed_results, agg_dir)

    agg_dir = os.path.join(output_dir, "agg_seed_results")
    seed_df, agg_df = save_seed_aggregation(
        agg_seed_results,
        output_dir=agg_dir,
    )

    print("run_name", run_name)

    print("\nAggregate across seeds:")
    print(agg_df[[
        "accuracy_mean_std",
        "balanced_accuracy_mean_std",
        "macro_f1_mean_std",
    ]])

