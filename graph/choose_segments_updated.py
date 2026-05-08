import argparse
import copy
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from entropy import *
from entropy import _normalize_fixed_edges

# =========================================================
# CAUEEG binary task helpers
# =========================================================
# Original CAUEEG-Dementia labels: 0=normal/HC, 1=MCI, 2=dementia.
# The "ad_*" names below follow your experiment naming, but in CAUEEG this is
# the dementia class unless you later filter diagnosis subtypes to pure AD.
CAUEEG_BINARY_SPECS = {
    "none": {"old_to_new": None, "class_names": None, "task_tag": "3class"},
    "ad_hc": {"old_to_new": {0: 0, 2: 1}, "class_names": ["HC", "AD_or_Dementia"], "task_tag": "ad_hc"},
    "ad_mci": {"old_to_new": {1: 0, 2: 1}, "class_names": ["MCI", "AD_or_Dementia"], "task_tag": "ad_mci"},
    "hc_mci": {"old_to_new": {0: 0, 1: 1}, "class_names": ["HC", "MCI"], "task_tag": "hc_mci"},
}


class BinaryRemapCauEegDataset(Dataset):
    """Filter CAUEEG to two old labels and remap them to 0/1."""

    def __init__(self, base_dataset, old_to_new: dict[int, int]):
        self.base_dataset = base_dataset
        self.old_to_new = {int(k): int(v) for k, v in old_to_new.items()}
        self.indices = []
        for i, item in enumerate(base_dataset.data_list):
            y_old = int(item["class_label"])
            if y_old in self.old_to_new:
                self.indices.append(i)
        if len(self.indices) == 0:
            raise RuntimeError(f"No samples found for binary labels {sorted(self.old_to_new.keys())}.")

        # Make the wrapper still look like a CauEegDataset for count/debug code.
        self.data_list = []
        for i in self.indices:
            item = copy.deepcopy(base_dataset.data_list[i])
            y_old = int(item["class_label"])
            item["original_class_label"] = y_old
            item["class_label"] = self.old_to_new[y_old]
            item["binary_class_label"] = self.old_to_new[y_old]
            self.data_list.append(item)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = copy.deepcopy(self.base_dataset[self.indices[idx]])
        y_old = int(sample["class_label"])
        y_new = self.old_to_new[y_old]
        sample["original_class_label"] = y_old
        sample["class_label"] = y_new
        sample["binary_class_label"] = y_new
        return sample


def normalize_binary_pair(binary_pair: Optional[str]) -> str:
    pair = "none" if binary_pair is None else str(binary_pair).lower()
    if pair in {"", "no", "false", "null"}:
        pair = "none"
    if pair not in CAUEEG_BINARY_SPECS:
        raise ValueError(f"Unknown binary_pair={binary_pair!r}; choose from {list(CAUEEG_BINARY_SPECS.keys())}.")
    return pair


def maybe_apply_binary_remap(train_set, val_set, test_set, binary_pair: Optional[str]):
    pair = normalize_binary_pair(binary_pair)
    if pair == "none":
        return train_set, val_set, test_set, None, None, pair
    spec = CAUEEG_BINARY_SPECS[pair]
    old_to_new = spec["old_to_new"]
    class_names = list(spec["class_names"])
    return (
        BinaryRemapCauEegDataset(train_set, old_to_new),
        BinaryRemapCauEegDataset(val_set, old_to_new),
        BinaryRemapCauEegDataset(test_set, old_to_new),
        old_to_new,
        class_names,
        pair,
    )


def summarize_record_labels(records, split_name: str):
    counts = Counter(int(r["label"]) for r in records)
    print(f"[{split_name}] record label counts: {dict(sorted(counts.items()))}")
    return counts


def build_prefixed_label_map(train_records, val_records, test_records) -> dict[str, int]:
    out = {}
    for split_name, records in [("train", train_records), ("val", val_records), ("test", test_records)]:
        for r in records:
            out[f"{split_name}_{r['subject_id']}"] = int(r["label"])
    return out


def override_payload_labels_in_place(payload: dict, label_by_prefixed_id: dict[str, int]):
    missing = []
    for sid, y in label_by_prefixed_id.items():
        if sid in payload:
            payload[sid]["label"] = int(y)
        else:
            missing.append(sid)
    if missing:
        print(f"[label override] warning: {len(missing)} ids missing from payload; first few: {missing[:5]}")


def get_caueeg_class_names(task: str, binary_pair: Optional[str], fallback_num_classes: int):
    pair = normalize_binary_pair(binary_pair)
    if pair != "none":
        return list(CAUEEG_BINARY_SPECS[pair]["class_names"])
    if str(task).startswith("abnormal"):
        return ["normal", "abnormal"]
    if str(task).startswith("dementia"):
        return ["normal", "mci", "dementia"]
    return [f"class_{i}" for i in range(int(fallback_num_classes))]


def load_cleancluster_manifest(manifest_path: str) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

    # Support both old and new manifest names.
    # Old: kmeans_cluster_id / kmeans_centroid_distance
    # New: global_cluster_id / global_cluster_distance
    required_base = {"subject_id", "segment_id"}
    missing = required_base - set(df.columns)
    if missing:
        raise KeyError(f"Cluster manifest missing columns: {missing}")

    if "global_cluster_id" not in df.columns and "kmeans_cluster_id" not in df.columns:
        raise KeyError(
            "Cluster manifest needs one of: 'global_cluster_id' or 'kmeans_cluster_id'. "
            f"Available columns: {list(df.columns)}"
        )

    if "global_cluster_id" not in df.columns:
        df["global_cluster_id"] = df["kmeans_cluster_id"]
    if "kmeans_cluster_id" not in df.columns:
        df["kmeans_cluster_id"] = df["global_cluster_id"]

    if "global_cluster_distance" not in df.columns and "kmeans_centroid_distance" in df.columns:
        df["global_cluster_distance"] = df["kmeans_centroid_distance"]
    if "kmeans_centroid_distance" not in df.columns and "global_cluster_distance" in df.columns:
        df["kmeans_centroid_distance"] = df["global_cluster_distance"]

    # Robust bool handling if present. Some global-cluster manifests may not have keep_clean.
    if "keep_clean" in df.columns and df["keep_clean"].dtype != bool:
        df["keep_clean"] = (
            df["keep_clean"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df["global_cluster_id"] = df["global_cluster_id"].astype(int)
    df["kmeans_cluster_id"] = df["kmeans_cluster_id"].astype(int)

    return df


def _stable_int_from_string(x: str) -> int:
    """Stable integer hash; do not use Python hash(), which changes across runs."""
    import hashlib
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)



def _resolve_cluster_col_for_manifest(manifest_df: Optional[pd.DataFrame], preferred: str = "global_cluster_id") -> Optional[str]:
    if manifest_df is None:
        return None
    if preferred in manifest_df.columns:
        return preferred
    for cand in ["global_cluster_id", "kmeans_cluster_id"]:
        if cand in manifest_df.columns:
            return cand
    return None

def _build_segment_to_cluster_lookup(
    manifest_df: Optional[pd.DataFrame],
    *,
    cluster_col: str = "global_cluster_id",
) -> dict:
    """Return {(subject_id, segment_id): cluster_id}; empty if manifest is unavailable."""
    if manifest_df is None:
        return {}

    cc = _resolve_cluster_col_for_manifest(manifest_df, preferred=cluster_col)
    if cc is None:
        return {}

    required = {"subject_id", "segment_id", cc}
    if not required.issubset(set(manifest_df.columns)):
        return {}

    tmp = manifest_df[["subject_id", "segment_id", cc]].copy()
    tmp["subject_id"] = tmp["subject_id"].astype(str)
    tmp["segment_id"] = tmp["segment_id"].astype(int)
    tmp[cc] = tmp[cc].astype(int)

    return {
        (str(r.subject_id), int(r.segment_id)): int(getattr(r, cc))
        for r in tmp.itertuples(index=False)
    }


def log_selection_diversity_over_epochs(
    dataset,
    *,
    output_dir: str,
    strategy_name: str,
    manifest_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "global_cluster_id",
    num_epochs: int = 20,
    max_subjects: Optional[int] = None,
):
    """
    Probe the training dataset sampler before training and save how much
    segment/cluster diversity it will expose across epochs.

    This works for:
        - LabelAwareSubjectBagDataset
        - ClusterGuidedLabelAwareSubjectBagDataset
        - SubjectBagGraphDataset, if it returns graphs in __getitem__

    Saved files:
        selection_diversity_long.csv
        selection_diversity_epoch_summary.csv
        selection_diversity_subject_summary.csv
        selection_diversity_global_summary.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    n_epochs = max(1, int(num_epochs))
    old_epoch = getattr(dataset, "epoch", None)

    seg_to_cluster = _build_segment_to_cluster_lookup(
        manifest_df,
        cluster_col=cluster_col,
    )

    subject_indices = list(range(len(dataset)))
    if max_subjects is not None:
        subject_indices = subject_indices[: int(max_subjects)]

    rows = []

    for epoch in range(n_epochs):
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for idx in subject_indices:
            item = dataset[idx]
            sid = str(item.get("subject_id", f"idx_{idx}"))
            label = int(item.get("label", -1))
            graphs = item.get("graphs", [])

            for pos, g in enumerate(graphs):
                seg_id = int(getattr(g, "segment_id", item.get("segment_ids", [-1])[pos] if "segment_ids" in item else -1))
                key = (sid, seg_id)
                cid = seg_to_cluster.get(key, np.nan)

                rows.append({
                    "strategy": strategy_name,
                    "epoch": int(epoch),
                    "subject_id": sid,
                    "label": label,
                    "position_in_bag": int(pos),
                    "segment_id": seg_id,
                    "cluster_id": cid,
                })

    if old_epoch is not None and hasattr(dataset, "set_epoch"):
        dataset.set_epoch(old_epoch)

    long_df = pd.DataFrame(rows)
    long_path = os.path.join(output_dir, "selection_diversity_long.csv")
    long_df.to_csv(long_path, index=False)

    if len(long_df) == 0:
        empty = pd.DataFrame([{"strategy": strategy_name, "num_rows": 0}])
        empty.to_csv(os.path.join(output_dir, "selection_diversity_global_summary.csv"), index=False)
        return {
            "long": long_df,
            "epoch_summary": pd.DataFrame(),
            "subject_summary": pd.DataFrame(),
            "global_summary": empty,
        }

    # Epoch-level summary: how much diversity each epoch has.
    epoch_summary = (
        long_df.groupby(["strategy", "epoch"])
        .agg(
            num_draws=("segment_id", "size"),
            num_subjects=("subject_id", "nunique"),
            unique_segments=("segment_id", "nunique"),
            unique_clusters=("cluster_id", lambda x: pd.Series(x).dropna().nunique()),
            mean_bag_size=("position_in_bag", lambda x: float(pd.Series(x).groupby(long_df.loc[x.index, "subject_id"]).count().mean())),
        )
        .reset_index()
    )
    epoch_summary["unique_segment_ratio"] = epoch_summary["unique_segments"] / epoch_summary["num_draws"].clip(lower=1)
    epoch_summary.to_csv(os.path.join(output_dir, "selection_diversity_epoch_summary.csv"), index=False)

    # Subject-level summary across probe epochs.
    subject_summary = (
        long_df.groupby(["strategy", "subject_id", "label"])
        .agg(
            total_draws=("segment_id", "size"),
            unique_segments=("segment_id", "nunique"),
            unique_clusters=("cluster_id", lambda x: pd.Series(x).dropna().nunique()),
            num_epochs=("epoch", "nunique"),
        )
        .reset_index()
    )
    subject_summary["unique_segment_ratio"] = subject_summary["unique_segments"] / subject_summary["total_draws"].clip(lower=1)
    subject_summary.to_csv(os.path.join(output_dir, "selection_diversity_subject_summary.csv"), index=False)

    global_summary = pd.DataFrame([{
        "strategy": strategy_name,
        "probe_epochs": int(n_epochs),
        "num_subjects": int(long_df["subject_id"].nunique()),
        "total_draws": int(len(long_df)),
        "unique_segments_total": int(long_df[["subject_id", "segment_id"]].drop_duplicates().shape[0]),
        "unique_clusters_total": int(long_df["cluster_id"].dropna().nunique()),
        "mean_subject_unique_segment_ratio": float(subject_summary["unique_segment_ratio"].mean()),
        "mean_epoch_unique_segment_ratio": float(epoch_summary["unique_segment_ratio"].mean()),
    }])
    global_summary.to_csv(os.path.join(output_dir, "selection_diversity_global_summary.csv"), index=False)

    print(f"[selection diversity] saved to: {output_dir}")
    print(global_summary.to_string(index=False))

    return {
        "long": long_df,
        "epoch_summary": epoch_summary,
        "subject_summary": subject_summary,
        "global_summary": global_summary,
    }



class ClusterGuidedLabelAwareSubjectBagDataset(Dataset):
    """
    Dynamic cluster-guided MIL sampler.

    This follows the key rule of LabelAwareSubjectBagDataset:
      - compute k_label per class from base_k, class counts, and max_k_per_subject
      - call set_epoch(epoch) from the training loop
      - sample deterministically from seed + epoch + subject_id
      - different epochs can select different segments

    The difference is that segment choice is guided by cluster structure:
      - global_cluster_random_k: cluster-balanced random sampling
      - global_cluster_proportional_random_k: sample according to this subject's cluster proportions
      - label_aligned_greedy_k: prioritize clusters with higher P(subject_label | cluster),
                                lower entropy, and more subject segments
      - mixed_random_label_aligned_k: 50% random_k + 50% label-aligned greedy
    """

    def __init__(
        self,
        graphs,
        manifest_df: pd.DataFrame,
        *,
        strategy: str,
        base_k: int,
        k_by_label: Optional[dict] = None,
        target_segments_per_class: Optional[int] = None,
        max_k_per_subject: Optional[int] = None,
        seed: int = 42,
        cluster_col: str = "global_cluster_id",
        label_col: Optional[str] = None,
        distance_col: str = "global_cluster_distance",
        return_segment_ids: bool = True,
        debug_dir: Optional[str] = None,
        mixed_random_frac: float = 0.50,
    ):
        self.graphs = list(graphs)
        self.strategy = str(strategy).lower()
        self.base_k = int(base_k)
        self.seed = int(seed)
        self.epoch = 0
        self.return_segment_ids = bool(return_segment_ids)
        self.distance_col = distance_col
        self.debug_dir = debug_dir
        self.mixed_random_frac = float(mixed_random_frac)
        if not (0.0 <= self.mixed_random_frac <= 1.0):
            raise ValueError(f"mixed_random_frac must be in [0, 1], got {self.mixed_random_frac}")

        if len(self.graphs) == 0:
            raise RuntimeError("graphs is empty.")

        if self.strategy not in {
            "global_cluster_random_k",
            "global_cluster_proportional_random_k",
            "label_aligned_greedy_k",
            "mixed_random_label_aligned_k",
        }:
            raise ValueError(f"Unsupported cluster-guided strategy: {strategy!r}")

        df = manifest_df.copy()

        # Support old/new cluster column names.
        if cluster_col not in df.columns:
            if "global_cluster_id" in df.columns:
                cluster_col = "global_cluster_id"
            elif "kmeans_cluster_id" in df.columns:
                cluster_col = "kmeans_cluster_id"
            else:
                raise KeyError(
                    "manifest_df needs one of: global_cluster_id or kmeans_cluster_id. "
                    f"Available columns: {list(df.columns)}"
                )

        self.cluster_col = cluster_col

        required = {"subject_id", "segment_id", self.cluster_col}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"manifest_df missing columns: {missing}")

        df["subject_id"] = df["subject_id"].astype(str)
        df["segment_id"] = df["segment_id"].astype(int)
        df[self.cluster_col] = df[self.cluster_col].astype(int)

        # Graph lookup and subject grouping.
        self.graph_lookup = {
            (str(g.subject_id), int(g.segment_id)): g
            for g in self.graphs
        }
        self.graph_label_lookup = {
            (str(g.subject_id), int(g.segment_id)): int(g.y.view(-1)[0].item())
            for g in self.graphs
        }

        df["_key"] = list(zip(df["subject_id"], df["segment_id"]))
        df = df[df["_key"].isin(self.graph_lookup)].copy()

        if len(df) == 0:
            raise RuntimeError("No manifest rows match the provided training graphs.")

        # Label source: use manifest label if available; otherwise graph.y.
        if label_col is None:
            for cand in ["label", "class_label", "y", "true_label", "subject_label"]:
                if cand in df.columns:
                    label_col = cand
                    break

        if label_col is not None and label_col in df.columns:
            df["_label"] = df[label_col].astype(int)
        else:
            df["_label"] = [self.graph_label_lookup[key] for key in df["_key"]]

        self.df = df

        self.subject_to_graphs = defaultdict(list)
        self.subject_to_label = {}
        for g in self.graphs:
            sid = str(g.subject_id)
            y = int(g.y.view(-1)[0].item())
            self.subject_to_graphs[sid].append(g)
            if sid in self.subject_to_label and self.subject_to_label[sid] != y:
                raise ValueError(f"Subject {sid} has inconsistent labels.")
            self.subject_to_label[sid] = y

        # Only subjects with manifest-matched segments are usable for cluster-guided sampling.
        self.subject_ids = sorted(df["subject_id"].unique().tolist())
        self.subject_labels = [self.subject_to_label[sid] for sid in self.subject_ids]

        for sid in self.subject_ids:
            self.subject_to_graphs[sid] = sorted(
                self.subject_to_graphs[sid],
                key=lambda g: (
                    int(getattr(g, "segment_id", 0)),
                    int(getattr(g, "start_sample", 0)) if getattr(g, "start_sample", None) is not None else 0,
                ),
            )

        self.num_node_features = self.graphs[0].x.shape[-1]
        self.summary_input_dim = self.graphs[0].summary_feat.numel() if hasattr(self.graphs[0], "summary_feat") else None
        self.num_nodes = self.graphs[0].x.shape[0]

        # Make sure fixed-node encoders remain safe.
        for i, g in enumerate(self.graphs):
            if g.x.shape[0] != self.num_nodes:
                raise ValueError(
                    f"Expected fixed num_nodes={self.num_nodes}, but graph {i} has {g.x.shape[0]} nodes."
                )

        # label -> subjects, then k_by_label exactly like LabelAwareSubjectBagDataset.
        self.label_to_subjects = defaultdict(list)
        for sid in self.subject_ids:
            self.label_to_subjects[self.subject_to_label[sid]].append(sid)

        if k_by_label is None:
            n_subjects_per_label = {
                label: len(sids) for label, sids in self.label_to_subjects.items()
            }
            if target_segments_per_class is None:
                max_subjects = max(n_subjects_per_label.values())
                target_segments_per_class = max_subjects * self.base_k

            self.k_by_label = {}
            for label, n_subj in n_subjects_per_label.items():
                k_label = math.ceil(target_segments_per_class / n_subj)
                if max_k_per_subject is not None:
                    k_label = min(k_label, max_k_per_subject)
                self.k_by_label[label] = int(k_label)
        else:
            self.k_by_label = {int(k): int(v) for k, v in k_by_label.items()}
            if max_k_per_subject is not None:
                for label in self.k_by_label:
                    self.k_by_label[label] = min(self.k_by_label[label], max_k_per_subject)

        # subject -> cluster -> graph list.
        self.subject_cluster_to_graphs = defaultdict(lambda: defaultdict(list))
        for _, row in df.iterrows():
            sid = str(row["subject_id"])
            seg = int(row["segment_id"])
            cid = int(row[self.cluster_col])
            key = (sid, seg)
            if key in self.graph_lookup:
                self.subject_cluster_to_graphs[sid][cid].append(self.graph_lookup[key])

        self.cluster_info = self._compute_cluster_info(df)

        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)
            pd.DataFrame(
                [{"label": k, "k_label": v, "num_subjects": len(self.label_to_subjects[k])}
                 for k, v in sorted(self.k_by_label.items())]
            ).to_csv(os.path.join(self.debug_dir, f"{self.strategy}_k_by_label.csv"), index=False)
            pd.DataFrame(list(self.cluster_info.values())).to_csv(
                os.path.join(self.debug_dir, f"{self.strategy}_cluster_info.csv"), index=False
            )

    def _compute_cluster_info(self, df: pd.DataFrame) -> dict:
        all_labels = sorted(df["_label"].unique().tolist())
        counts = pd.crosstab(df[self.cluster_col], df["_label"]).sort_index()
        counts = counts.reindex(columns=all_labels, fill_value=0)
        probs = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

        p = probs.to_numpy(dtype=np.float64)
        p_safe = np.clip(p, 1e-12, 1.0)
        entropy = -(p_safe * np.log2(p_safe)).sum(axis=1)
        max_entropy = np.log2(len(all_labels)) if len(all_labels) > 1 else 1.0
        entropy_norm = entropy / max_entropy

        info = {}
        for idx, cid in enumerate(counts.index):
            row = {
                "cluster_id": int(cid),
                "entropy_norm": float(entropy_norm[idx]),
                "num_segments": int(counts.iloc[idx].sum()),
            }
            for y in all_labels:
                row[f"p_class_{y}"] = float(probs.loc[cid, y])
            info[int(cid)] = row
        return info

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.subject_ids)

    def _rng(self, sid: str):
        subject_seed = self.seed + 1000003 * self.epoch + _stable_int_from_string(str(sid))
        return random.Random(subject_seed)

    @staticmethod
    def _sample_with_replacement_if_needed(rng, candidates, k: int):
        candidates = list(candidates)
        if len(candidates) == 0:
            return []
        if len(candidates) >= k:
            return rng.sample(candidates, k)
        out = list(candidates)
        while len(out) < k:
            out.append(rng.choice(candidates))
        return out

    def _sample_cluster_balanced_random(self, sid: str, rng, k: int):
        cluster_to_graphs = self.subject_cluster_to_graphs[sid]
        clusters = list(cluster_to_graphs.keys())
        rng.shuffle(clusters)

        chosen = []
        # Cycle through clusters to avoid one dominant cluster taking all slots.
        while len(chosen) < k and len(clusters) > 0:
            made_progress = False
            for cid in clusters:
                if len(chosen) >= k:
                    break
                gs = cluster_to_graphs[cid]
                if len(gs) == 0:
                    continue
                chosen.append(rng.choice(gs))
                made_progress = True
            if not made_progress:
                break

        if len(chosen) < k:
            all_gs = [g for gs in cluster_to_graphs.values() for g in gs]
            chosen.extend(self._sample_with_replacement_if_needed(rng, all_gs, k - len(chosen)))
        return chosen[:k]

    def _sample_cluster_proportional_random(self, sid: str, rng, k: int):
        cluster_to_graphs = self.subject_cluster_to_graphs[sid]
        clusters = list(cluster_to_graphs.keys())
        if len(clusters) == 0:
            return []

        sizes = [len(cluster_to_graphs[cid]) for cid in clusters]
        total = float(sum(sizes))
        probs = [sz / total for sz in sizes]

        chosen = []
        for _ in range(k):
            cid = rng.choices(clusters, weights=probs, k=1)[0]
            chosen.append(rng.choice(cluster_to_graphs[cid]))
        return chosen

    def _sample_label_aligned_greedy_dynamic(self, sid: str, rng, k: int):
        y = int(self.subject_to_label[sid])
        cluster_to_graphs = self.subject_cluster_to_graphs[sid]
        subject_cluster_count = {cid: len(gs) for cid, gs in cluster_to_graphs.items()}

        def rank_key(cid):
            info = self.cluster_info[int(cid)]
            p_y = float(info.get(f"p_class_{y}", 0.0))
            h = float(info["entropy_norm"])
            n_subj = int(subject_cluster_count.get(cid, 0))
            return (-p_y, h, -n_subj, int(cid))

        ranked_clusters = sorted(cluster_to_graphs.keys(), key=rank_key)
        chosen = []

        # Greedy cluster order is fixed, but segment order inside each cluster changes by epoch.
        for cid in ranked_clusters:
            if len(chosen) >= k:
                break
            gs = list(cluster_to_graphs[cid])
            rng.shuffle(gs)
            for g in gs:
                if len(chosen) >= k:
                    break
                chosen.append(g)

        if len(chosen) < k:
            all_gs = [g for gs in cluster_to_graphs.values() for g in gs]
            chosen.extend(self._sample_with_replacement_if_needed(rng, all_gs, k - len(chosen)))
        return chosen[:k]

    @staticmethod
    def _graph_key(g):
        return (str(getattr(g, "subject_id", "")), int(getattr(g, "segment_id", -1)))

    def _sample_random_from_subject(self, sid: str, rng, k: int, exclude_keys=None):
        """
        Random-k component matching LabelAwareSubjectBagDataset behavior.
        Prefer no duplicates when possible; use replacement only if needed.
        """
        if k <= 0:
            return []

        exclude_keys = set() if exclude_keys is None else set(exclude_keys)
        candidates = [
            g for g in self.subject_to_graphs[sid]
            if self._graph_key(g) not in exclude_keys
        ]

        # If exclusion leaves too few candidates, allow the full subject pool.
        if len(candidates) == 0:
            candidates = list(self.subject_to_graphs[sid])

        return self._sample_with_replacement_if_needed(rng, candidates, k)

    def _sample_mixed_random_label_aligned(self, sid: str, rng, k: int):
        """
        Hybrid strategy:
            50% random_k + 50% label-aligned cluster-guided segments.

        More generally, mixed_random_frac controls the random portion.
        For k=10 and mixed_random_frac=0.5:
            5 random segments + 5 label-aligned greedy segments.
        """
        if k <= 0:
            return []

        k_random = int(round(k * self.mixed_random_frac))
        k_random = min(max(k_random, 0), k)
        k_cluster = k - k_random

        cluster_graphs = self._sample_label_aligned_greedy_dynamic(sid, rng, k_cluster)
        used = {self._graph_key(g) for g in cluster_graphs}
        random_graphs = self._sample_random_from_subject(sid, rng, k_random, exclude_keys=used)

        graphs = list(cluster_graphs) + list(random_graphs)
        rng.shuffle(graphs)

        # Safety fill if something unexpected returns too few.
        if len(graphs) < k:
            used = {self._graph_key(g) for g in graphs}
            graphs.extend(self._sample_random_from_subject(sid, rng, k - len(graphs), exclude_keys=used))

        return graphs[:k]

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        label = int(self.subject_to_label[sid])
        k = int(self.k_by_label[label])
        rng = self._rng(sid)

        if self.strategy == "global_cluster_random_k":
            graphs = self._sample_cluster_balanced_random(sid, rng, k)
        elif self.strategy == "global_cluster_proportional_random_k":
            graphs = self._sample_cluster_proportional_random(sid, rng, k)
        elif self.strategy == "label_aligned_greedy_k":
            graphs = self._sample_label_aligned_greedy_dynamic(sid, rng, k)
        elif self.strategy == "mixed_random_label_aligned_k":
            graphs = self._sample_mixed_random_label_aligned(sid, rng, k)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        if len(graphs) == 0:
            graphs = self._sample_with_replacement_if_needed(rng, self.subject_to_graphs[sid], k)

        out = {
            "subject_id": sid,
            "label": label,
            "graphs": graphs,
        }
        if self.return_segment_ids:
            out["segment_ids"] = [int(getattr(g, "segment_id", -1)) for g in graphs]
        return out






def run_caueeg_linkx_mil_choose_segments(
    dataset_path,
    fixed_edges,          
    channel_names,
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
    segment_selection_strategy: str = "original_random_k", # original_random_k | global_cluster_random_k | global_cluster_proportional_random_k | label_aligned_greedy_k | all_raw
    cleancluster_manifest_path: Optional[str] = None,
    binary_pair: Optional[str] = "none",  # none | ad_hc | ad_mci | hc_mci
    bag_aug_mode: str = "none",
    debug_bag_aug: bool = False,
    cluster_col: str = "global_cluster_id",
    clean_col: str = "keep_clean",
    weight_col: str = "sampling_weight",
    exclude_noise_clusters: bool = False,
    min_cluster_size: int = 50,
    min_clean_rate: float = 0.50,
    pseudo_bags_per_epoch: int = 1000,
    pseudo_subjects_per_bag: int = 5,
    p_real: float = 0.70,
    multiview_max_cluster_views: int = 3,
    min_segments_per_cluster_view: int = 5,
    min_cluster_fraction: float = 0.05,
    lambda_consistency: float = 0.10,
    consistency_temperature: float = 1.0,
    clean_k: int = 10,
    level: str = "segment",
    macro_duration_sec: float = 60.0,
    level_reduce: str = "mean",
    backbone: str = "gatv2",
    use_gcn_norm: bool = False,
    test_code=False,
    arglr: float = 0.003,
    use_soft_targets=False,

):
    os.makedirs(output_root, exist_ok=True)

    
    run_name = f"seed{seed}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"log.txt")

    bad_ids = {"00587", "00781", "01301", "train_00587", "train_00781", "train_01301"}
    patience=20
    start_epoch=1
    epochs=200
    lr=1e-3
    weight_decay=5e-3

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
        f.write(f"binary_pair: {binary_pair}\n")
        f.write(f"bag_aug_mode: {bag_aug_mode}\n")
        f.write(f"level: {level}, macro_duration_sec: {macro_duration_sec}, level_reduce: {level_reduce}\n")
        f.write(f"topology: {filter_method}, fixed_edges: {fixed_edges}, channel_names: {channel_names}\n")
        f.write(f"feature_families: {feature_families}\nconnectivity_metric: {connectivity_metric}, connectivity_band: {connectivity_band}\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"\n")

        f.write(f"model_name: {encoder_type}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta={min_delta}, top_k={top_k} \n")
        f.write(f"batch_size {batch_size}\n")
        # f.write(f"use_center_loss {use_center_loss}, use_contrastive_loss {use_contrastive_loss}, use_prototype_classifier {use_prototype_classifier}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        # f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f"dropout={dropout}, use_lr_scheduler={use_lr_scheduler}, use_soft_targets = {use_soft_targets}\n")
    


    # 1) official split
    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    train_set, val_set, test_set, binary_old_to_new, binary_class_names, binary_pair = maybe_apply_binary_remap(
        train_set,
        val_set,
        test_set,
        binary_pair,
    )

    # 2) convert each recording into subject-like records
    if test_code:
        epochs = 1
        test_n_subjects = 10
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
    summarize_record_labels(train_records, "train")
    summarize_record_labels(val_records, "val")
    summarize_record_labels(test_records, "test")
    label_by_prefixed_id = build_prefixed_label_map(train_records, val_records, test_records)

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



    num_classes = 2 if normalize_binary_pair(binary_pair) != "none" else len(sorted({r["label"] for r in all_records}))
    class_names = get_caueeg_class_names(task, binary_pair, num_classes)
    print(f"num_classes={num_classes}, class_names={class_names}, binary_pair={binary_pair}")

    with open(log_path, "a") as f:
        f.write(f"num_classes: {num_classes}\n")
        f.write(f"class_names: {class_names}\n")

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
    override_payload_labels_in_place(payload, label_by_prefixed_id)

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

    manifest_required_strategies = {
        "clean_random_k",
        "clean_kmeans_k",
        "all_clean",
        "clean_weighted_k",
        "global_cluster_random_k",
        "global_cluster_proportional_random_k",
        "label_aligned_greedy_k",
        "mixed_random_label_aligned_k",
    }

    manifest_df = None
    if selection_strategy in manifest_required_strategies:
        if cleancluster_manifest_path is None:
            raise ValueError(
                "cleancluster_manifest_path must be provided when "
                f"segment_selection_strategy={selection_strategy!r}"
            )
        manifest_df = load_global_cluster_manifest(cleancluster_manifest_path)

    if selection_strategy == "original_random_k":
        # Same as your original random_k baseline:
        # LabelAwareSubjectBagDataset does dynamic epoch-level sampling and class-balanced k_label.
        train_graphs_selected = train_graphs
        train_dataset_mode = "label_aware_random"

    elif selection_strategy == "all_raw":
        # Use all raw training segments. No k sampling.
        train_graphs_selected = train_graphs
        train_dataset_mode = "fixed_all_selected"

    elif selection_strategy == "clean_random_k":
        # Clean pool, but random sampling is still done dynamically by LabelAwareSubjectBagDataset.
        train_graphs_selected = filter_graphs_by_manifest_keep_clean(
            train_graphs,
            manifest_df,
        )
        train_dataset_mode = "label_aware_random"

    elif selection_strategy == "all_clean":
        # Use all clean segments. No k sampling.
        train_graphs_selected = filter_graphs_by_manifest_keep_clean(
            train_graphs,
            manifest_df,
        )
        train_dataset_mode = "fixed_all_selected"

    elif selection_strategy in {
        "global_cluster_random_k",
        "global_cluster_proportional_random_k",
        "label_aligned_greedy_k",
        "mixed_random_label_aligned_k",
    }:
        # New fair comparison:
        # keep the full graph pool, but let ClusterGuidedLabelAwareSubjectBagDataset
        # dynamically choose k_label segments each epoch using the cluster rule.
        train_graphs_selected = train_graphs
        train_dataset_mode = "cluster_guided_label_aware"

    elif selection_strategy == "clean_kmeans_k":
        # Legacy fixed representative baseline. Kept for backward compatibility.
        # For the new fair comparison, prefer global_cluster_random_k / label_aligned_greedy_k.
        train_graphs_selected = select_clean_kmeans_graphs_from_manifest(
            train_graphs,
            manifest_df,
            k=clean_k,
            seed=seed,
        )
        train_dataset_mode = "fixed_all_selected"

    elif selection_strategy == "clean_weighted_k":
        # Legacy fixed weighted baseline. Kept for backward compatibility.
        train_graphs_selected = weighted_sample_clean_graphs_from_manifest(
            train_graphs,
            manifest_df,
            k=clean_k,
            seed=seed,
            weight_col="sampling_weight",
        )
        train_dataset_mode = "fixed_all_selected"

    else:
        raise ValueError(
            f"Unknown segment_selection_strategy={segment_selection_strategy!r}. "
            "Use one of: label_aligned_greedy_k, original_random_k, "
            "global_cluster_random_k, global_cluster_proportional_random_k, all_raw, "
            "clean_random_k, clean_kmeans_k, all_clean, clean_weighted_k."
        )

    train_graphs = train_graphs_selected
    summarize_graph_pool(train_graphs, f"train_graphs_after_{selection_strategy}")

    level = str(level).lower()
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
    bag_aug_mode = str(bag_aug_mode).lower()

    if bag_aug_mode != "none":
        # Bag augmentation modes from entropy.py / mil_cluster_views.
        # train_graphs may already be prefiltered by the segment-selection strategy.
        if cleancluster_manifest_path is None:
            raise ValueError("cleancluster_manifest_path is required when bag_aug_mode != 'none'.")
        if manifest_df is None:
            manifest_df = load_global_cluster_manifest(cleancluster_manifest_path)

        train_dataset, train_collate_fn, requires_multiview_train_loop = build_augmented_train_dataset(
            train_graphs,
            bag_aug_mode=bag_aug_mode,
            base_k=base_k,
            seed=seed,
            max_k_per_subject=max_k_per_subject,
            base_collate_fn=base_collate_fn,
            manifest_df=manifest_df,
            cluster_col=cluster_col,
            clean_col=clean_col,
            weight_col=weight_col,
            exclude_noise_clusters=exclude_noise_clusters,
            min_cluster_size=min_cluster_size,
            min_clean_rate=min_clean_rate,
            pseudo_bags_per_epoch=pseudo_bags_per_epoch,
            pseudo_subjects_per_bag=pseudo_subjects_per_bag,
            p_real=p_real,
            multiview_max_cluster_views=multiview_max_cluster_views,
            min_segments_per_cluster_view=min_segments_per_cluster_view,
            min_cluster_fraction=min_cluster_fraction,
        )
        train_dataset_mode = f"bag_aug_{bag_aug_mode}"

    else:
        requires_multiview_train_loop = False
        train_collate_fn = base_collate_fn

        # Training dataset depends on the segment-selection strategy.
        if train_dataset_mode == "label_aware_random":
            train_dataset = LabelAwareSubjectBagDataset(
                train_graphs,
                train=True,
                base_k=base_k,
                max_k_per_subject=max_k_per_subject,
                seed=seed,
                return_segment_ids=True,
            )

        elif train_dataset_mode == "cluster_guided_label_aware":
            if level != "segment":
                raise ValueError(
                    f"{selection_strategy!r} is a segment-level dynamic sampling strategy. "
                    "Use level='segment', or preselect fixed segments before macro/subject aggregation."
                )
            if base_k is None:
                raise ValueError(
                    f"{selection_strategy!r} needs base_k because it samples k_label segments per subject."
                )

            train_dataset = ClusterGuidedLabelAwareSubjectBagDataset(
                train_graphs,
                manifest_df,
                strategy=selection_strategy,
                base_k=base_k,
                max_k_per_subject=max_k_per_subject,
                seed=seed,
                cluster_col=cluster_col,
                label_col=None,
                distance_col="global_cluster_distance",
                return_segment_ids=True,
                debug_dir=os.path.join(run_dir, f"{selection_strategy}_debug"),
            )

        elif train_dataset_mode == "fixed_all_selected":
            train_dataset = SubjectBagGraphDataset(
                train_graphs,
                max_segments_per_subject=None,
                train=True,
            )

        else:
            raise ValueError(f"Unknown train_dataset_mode={train_dataset_mode!r}")


    # Keep validation/test the same for all approaches.
    # This preserves the original caueeg_linkx_mil evaluation protocol:
    # eval_k_per_subject=None means use all val/test segments.
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
    try:
        log_selection_diversity_over_epochs(
            train_dataset,
            output_dir=os.path.join(run_dir, "selection_diversity"),
            strategy_name=f"{selection_strategy}__{bag_aug_mode}",
            manifest_df=manifest_df,
            cluster_col=cluster_col,
            num_epochs=min(int(epochs), 20),
        )
    except Exception as e:
        print(f"[Selection diversity] Warning: failed to log diversity diagnostics: {e}")
    if encoder_type in ["linkx_cnn5", "cnn5", 'gnn_bank', "cnn_bank", "linkx_cnn_bank"]:

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
    else:

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
        use_gcn_norm = False,
        use_prototypes=False,
        num_prototypes=2,
        prototype_emb_dim=16,
        prototype_hidden_dim=64,
        prototype_use_soft=True,
        prototype_use_dist=True,
        gcn_normalize_input=False,
        gcn_norm_add_self_loops=False,
        gcn_norm_abs_weights=False,
        gcn_norm_abs_degree=False,

    ).to(device)

    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    if debug_bag_aug:
        debug_bag_dataset(train_dataset, n=10)
        smoke_test_augmented_loader(
            model=model,
            loader=train_loader,
            device=device,
            multiview=requires_multiview_train_loop,
        )
        print("Debug bag augmentation completed. Exiting before training.")
        return {
            "model": model,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "history": [],
            "best_state": None,
            "train_metrics": None,
            "val_metrics": None,
            "test_metrics": None,
            "run_dir": run_dir,
            "val_pred_df": None,
            "test_pred_df": None,
            "summary_test": [],
        }

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        verbose=True,
        use_lr_scheduler=use_lr_scheduler,
        use_soft_targets=use_soft_targets,
        lr_scheduler_metric="val_loss",
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=20,
        lr_scheduler_min_lr=1e-6,
        lr_scheduler_threshold=1e-3,
        lr_scheduler_cooldown=0,
        lr_scheduler_start_epoch=10,       # None => use start_epoch
        )

    # 7) final evaluation
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    # after:
    # test_metrics = evaluate(model, test_loader, criterion, device)


    class_names = get_caueeg_class_names(task, binary_pair, num_classes)

    plot_linkx_mil_baseline_style(
        metrics=test_metrics,
        class_names=class_names,
        output_dir=run_dir,
        prefix="test"
    )

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
            "seed": seed,
            "binary_pair": binary_pair,
            "class_names": class_names,
            "bag_aug_mode": bag_aug_mode,
            "train_dataset_mode": train_dataset_mode,
            "use_soft_targets": use_soft_targets

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

# Backward-compatible name used by older bash scripts.
run_caueeg_linkx_mil_entropy = run_caueeg_linkx_mil_choose_segments

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

def _make_groupby_safe_value(x):
    if isinstance(x, (list, tuple, dict)):
        return json.dumps(make_jsonable(x), sort_keys=True)
    if isinstance(x, np.ndarray):
        return json.dumps(make_jsonable(x), sort_keys=True)
    if torch.is_tensor(x):
        return json.dumps(make_jsonable(x), sort_keys=True)
    return x


def _make_groupby_safe_dataframe(df: pd.DataFrame, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(_make_groupby_safe_value)
    return df

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
        "binary_pair",
        "bag_aug_mode",
        "train_dataset_mode",
        "class_names",
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
    df_group = _make_groupby_safe_dataframe(df, variant_cols)
    # Optional debug file: useful if a future config object still causes trouble.
    group_debug_path = os.path.join(output_dir, "aggregation_group_keys_debug.csv")
    df_group[variant_cols + metric_cols].to_csv(group_debug_path, index=False)

    if len(variant_cols) == 0:
        # Fallback: aggregate all rows together if no variant columns exist.
        agg = df_group[metric_cols].agg(["mean", "std", "min", "max", "count"]).T
        agg = agg.reset_index().rename(columns={"index": "metric"})
    else:
        agg = (
            df_group.groupby(variant_cols, dropna=False)[metric_cols]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        # Flatten multi-index columns.
        agg.columns = [
            col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
            for col in agg.columns
        ]

        # Add readable mean ± std columns.
        for m in metric_cols:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in agg.columns and std_col in agg.columns:
                agg[f"{m}_mean_std"] = agg.apply(
                    lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
                    if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
                    axis=1,
                )

    # agg = (
    #     df.groupby(variant_cols, dropna=False)[metric_cols]
    #     .agg(["mean", "std", "min", "max", "count"])
    #     .reset_index()
    # )

    # Flatten multi-index columns
    # agg.columns = [
    #     col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
    #     for col in agg.columns
    # ]

    # # Add readable mean ± std columns
    # for m in metric_cols:
    #     mean_col = f"{m}_mean"
    #     std_col = f"{m}_std"
    #     if mean_col in agg.columns and std_col in agg.columns:
    #         agg[f"{m}_mean_std"] = agg.apply(
    #             lambda r: f"{r[mean_col]:.4f} ± {r[std_col]:.4f}"
    #             if pd.notna(r[std_col]) else f"{r[mean_col]:.4f} ± NA",
    #             axis=1,
    #         )

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
    plt.figure(figsize=(6, 5))

    if num_classes == 2:
        # sklearn.label_binarize returns one column for binary labels, so handle
        # binary explicitly and plot the positive class probability.
        if len(np.unique(y_true)) >= 2 and y_prob.shape[1] >= 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[1]} vs {class_names[0]} (AUC={roc_auc:.3f})")
    else:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for c in range(num_classes):
            yc = y_true_bin[:, c]
            if len(np.unique(yc)) < 2:
                continue
            fpr, tpr, _ = roc_curve(yc, y_prob[:, c])
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


def save_subject_error_analysis_across_seeds(
    seed_prediction_dfs,
    *,
    output_dir: str,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Aggregate subject-level prediction errors across seeds.

    Expects each dataframe to contain:
        subject_id, true_label, pred_label, prob_0, prob_1, ...
    and ideally a seed column. If seed is absent, the list index is used.

    Saved files:
        subject_predictions_all_seeds.csv
        subject_error_summary_by_subject.csv
        subject_error_summary_by_true_pred.csv
        difficult_subjects.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    frames = []
    for i, df in enumerate(seed_prediction_dfs):
        if df is None or len(df) == 0:
            continue
        tmp = df.copy()
        if "seed" not in tmp.columns:
            tmp["seed"] = i
        frames.append(tmp)

    if len(frames) == 0:
        raise RuntimeError("No prediction dataframes were provided for error analysis.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df["subject_id"] = all_df["subject_id"].astype(str)
    all_df["true_label"] = all_df["true_label"].astype(int)
    all_df["pred_label"] = all_df["pred_label"].astype(int)
    all_df["correct"] = all_df["true_label"] == all_df["pred_label"]

    prob_cols = sorted(
        [c for c in all_df.columns if c.startswith("prob_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    num_classes = len(prob_cols)

    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    class_name_map = {i: str(class_names[i]) for i in range(min(num_classes, len(class_names)))}

    all_df["true_name"] = all_df["true_label"].map(class_name_map)
    all_df["pred_name"] = all_df["pred_label"].map(class_name_map)

    all_path = os.path.join(output_dir, "subject_predictions_all_seeds.csv")
    all_df.to_csv(all_path, index=False)

    def _mode_int(x):
        vc = pd.Series(x).value_counts()
        if len(vc) == 0:
            return -1
        # deterministic tie-break: smallest class id
        max_count = vc.max()
        return int(sorted(vc[vc == max_count].index.tolist())[0])

    grouped = all_df.groupby(["subject_id", "true_label"], dropna=False)

    records = []
    for (sid, y), g in grouped:
        rec = {
            "subject_id": sid,
            "true_label": int(y),
            "true_name": class_name_map.get(int(y), str(y)),
            "num_seeds": int(g["seed"].nunique()),
            "num_runs": int(len(g)),
            "correct_count": int(g["correct"].sum()),
            "wrong_count": int((~g["correct"]).sum()),
            "error_rate": float((~g["correct"]).mean()),
            "pred_mode": _mode_int(g["pred_label"]),
            "pred_labels_by_seed": json.dumps(
                g.sort_values("seed")[["seed", "pred_label"]].to_dict(orient="records")
            ),
        }
        rec["pred_mode_name"] = class_name_map.get(rec["pred_mode"], str(rec["pred_mode"]))

        if len(prob_cols) > 0:
            mean_probs = g[prob_cols].mean(axis=0).to_numpy(dtype=float)
            std_probs = g[prob_cols].std(axis=0, ddof=1).fillna(0.0).to_numpy(dtype=float)
            p_safe = np.clip(mean_probs, 1e-12, 1.0)
            rec["mean_prob_entropy"] = float(-(p_safe * np.log2(p_safe)).sum())
            rec["mean_confidence"] = float(np.max(mean_probs))
            for j, c in enumerate(prob_cols):
                rec[f"mean_{c}"] = float(mean_probs[j])
                rec[f"std_{c}"] = float(std_probs[j])

        records.append(rec)

    subject_summary = pd.DataFrame(records)
    subject_summary = subject_summary.sort_values(
        ["error_rate", "wrong_count", "mean_confidence"],
        ascending=[False, False, True],
    )

    subject_path = os.path.join(output_dir, "subject_error_summary_by_subject.csv")
    subject_summary.to_csv(subject_path, index=False)

    pair_summary = (
        all_df.groupby(["true_label", "true_name", "pred_label", "pred_name"])
        .size()
        .reset_index(name="count")
        .sort_values(["true_label", "count"], ascending=[True, False])
    )
    pair_path = os.path.join(output_dir, "subject_error_summary_by_true_pred.csv")
    pair_summary.to_csv(pair_path, index=False)

    difficult = subject_summary[subject_summary["wrong_count"] > 0].copy()
    difficult_path = os.path.join(output_dir, "difficult_subjects.csv")
    difficult.to_csv(difficult_path, index=False)

    print(f"[error analysis] saved all seed predictions: {all_path}")
    print(f"[error analysis] saved subject summary: {subject_path}")
    print(f"[error analysis] saved difficult subjects: {difficult_path}")

    return {
        "all_predictions": all_df,
        "subject_summary": subject_summary,
        "pair_summary": pair_summary,
        "difficult_subjects": difficult,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device", device)
    root_path = "/home/anphan/Documents/CAUEEG"



    save_path = os.path.join(root_path,'result-use_soft_targets')
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
    parser.add_argument("--use_delta_in_graph_bank", action="store_true")
    parser.add_argument("--use_soft_targets", action="store_true")


    parser.add_argument("--base_k", type=nullable_int, default=10, required=False, help="base_k")
    parser.add_argument("--level", type=str, default="segment", choices=["segment", "macro", "subject"],
                        help="Data preparation level: segment, macro, or subject")
    parser.add_argument("--macro_duration_sec", type=float, default=60.0,
                        help="Macro block size in seconds when --level macro")
    parser.add_argument("--level_reduce", type=str, default="mean", choices=["mean", "median", "max", "min", "sum"],
                        help="How to aggregate node features/connectivity for macro/subject level")
    parser.add_argument("--segment_selection_strategy", type=str, default="original_random_k", 
        choices=["label_aligned_greedy_k", "mixed_random_label_aligned_k", "original_random_k", "global_cluster_random_k", "global_cluster_proportional_random_k", 
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
        "--binary_pair",
        type=str,
        default="none",
        choices=["none", "ad_hc", "ad_mci", "hc_mci"],
        help="Optional binary remap from CAUEEG-Dementia labels: ad_hc, ad_mci, hc_mci.",
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
            "mixed_realmultiview_pseudo",
            "multiview_consistency",
        ],
    )
    parser.add_argument("--debug_bag_aug", action="store_true")
    parser.add_argument("--cluster_col", type=str, default="global_cluster_id")
    parser.add_argument("--clean_col", type=str, default="keep_clean")
    parser.add_argument("--weight_col", type=str, default="sampling_weight")
    parser.add_argument("--exclude_noise_clusters", action="store_true")
    parser.add_argument("--min_cluster_size", type=int, default=50)
    parser.add_argument("--min_clean_rate", type=float, default=0.50)
    parser.add_argument("--pseudo_bags_per_epoch", type=int, default=1000)
    parser.add_argument("--pseudo_subjects_per_bag", type=int, default=5)
    parser.add_argument("--p_real", type=float, default=0.70)
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
    
    seed_prediction_dfs = []


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



    task="dementia-no-overlap"
    file_format="edf"
    dataset_path="/home/anphan/Downloads/caueeg-dataset/"
    # out_h5 = args.out_h5
    if args.out_h5 == None:
        out_h5 = "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
    else:
        out_h5 = args.out_h5
    
    segment_selection_strategy = args.segment_selection_strategy
    k_tag = f"k{args.base_k}" if segment_selection_strategy != "all_clean" else "allclean"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    norm_tag = "gcnnorm" if args.use_gcn_norm else "nonorm"
    run_name = (
        f"{args.binary_pair}_{args.level}_{segment_selection_strategy}_"
        f"{args.connectivity_metric}_{k_tag}_{args.encoder_type}_{args.mil_pool_type}_{args.topology}_{args.bag_aug_mode}"
    )
    
    # cleancluster_manifest_path = "/home/anphan/Documents/CAUEEG/visualize-random/statistical_clusters_PCA20_N100/global_cluster_manifest.csv"
    cleancluster_manifest_path = "/home/anphan/Documents/CAUEEG/visualize-random/statistical_clusters_flatten_no_pca_N100/global_cluster_manifest.csv"

    output_dir = os.path.join(save_path, f"{timestamp}_{run_name}")
    os.makedirs(output_dir, exist_ok=True)



    for seed in seeds_list:
        out = run_caueeg_linkx_mil_entropy(
            dataset_path=dataset_path,
            fixed_edges=fixed_edges,          
            channel_names=CAUEEG_EEG19,
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
            # epochs=1,
            # lr=1e-4,
            # weight_decay=1e-3,
            device=device,
            rebuild_h5=False,
            output_root=output_dir,
            candidate_fusion_mode=args.candidate_fusion_mode,
            segment_selection_strategy=segment_selection_strategy,
            cleancluster_manifest_path=cleancluster_manifest_path,
            binary_pair=args.binary_pair,
            bag_aug_mode=args.bag_aug_mode,
            debug_bag_aug=args.debug_bag_aug,
            cluster_col=args.cluster_col,
            clean_col=args.clean_col,
            weight_col=args.weight_col,
            exclude_noise_clusters=args.exclude_noise_clusters,
            min_cluster_size=args.min_cluster_size,
            min_clean_rate=args.min_clean_rate,
            pseudo_bags_per_epoch=args.pseudo_bags_per_epoch,
            pseudo_subjects_per_bag=args.pseudo_subjects_per_bag,
            p_real=args.p_real,
            multiview_max_cluster_views=args.multiview_max_cluster_views,
            min_segments_per_cluster_view=args.min_segments_per_cluster_view,
            min_cluster_fraction=args.min_cluster_fraction,
            lambda_consistency=args.lambda_consistency,
            consistency_temperature=args.consistency_temperature,
            level=args.level,
            macro_duration_sec=args.macro_duration_sec,
            level_reduce=args.level_reduce,
            bank_specs = bank_specs,
            test_code = args.test_code,
            backbone = args.backbone,
            use_gcn_norm = args.use_gcn_norm,
            use_soft_targets = args.use_soft_targets,
        )

        # summary_test is a list with one dict
        seed_rows = out["summary_test"]
        # all_seed_results.append(seed_rows)        
        agg_seed_results.extend(seed_rows)

        # Save per-subject test predictions for error analysis across seeds.
        if "test_pred_df" in out and out["test_pred_df"] is not None:
            pred_df_seed = out["test_pred_df"].copy()
            pred_df_seed["seed"] = seed
            pred_df_seed["run_dir"] = out.get("run_dir", "")
            pred_df_seed["segment_selection_strategy"] = segment_selection_strategy
            pred_df_seed["binary_pair"] = args.binary_pair
            pred_df_seed["bag_aug_mode"] = args.bag_aug_mode
            seed_prediction_dfs.append(pred_df_seed)
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
    save_subject_error_analysis_across_seeds(
        seed_prediction_dfs=seed_prediction_dfs,
        output_dir=agg_dir,
        class_names=get_caueeg_class_names(task, args.binary_pair, 2 if args.binary_pair != "none" else 3))
