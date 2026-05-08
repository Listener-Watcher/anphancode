# graph/mil_cluster_views.py

from __future__ import annotations

import math
import random
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags, move_batch_to_device

# ============================================================
# Basic helpers
# ============================================================
def make_collate_multiview_subject_bags(base_collate_fn):
    """
    base_collate_fn:
        collate_subject_bags for normal encoders
        collate_subject_bags_multiband for bank/multiband encoders
    """

    def collate_multiview_subject_bags(items):
        flat_items = []
        group_ids = []
        parent_subject_ids = []
        view_names_flat = []

        for group_idx, item in enumerate(items):
            sid = item["subject_id"]
            y = int(item["label"])
            views = item["views"]
            view_names = item.get("view_names", [f"view{i}" for i in range(len(views))])

            parent_subject_ids.append(sid)

            for view_idx, graphs in enumerate(views):
                flat_items.append({
                    "subject_id": f"{sid}::{view_names[view_idx]}",
                    "label": y,
                    "graphs": graphs,
                })
                group_ids.append(group_idx)
                view_names_flat.append(view_names[view_idx])

        batch = base_collate_fn(flat_items)
        batch["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
        batch["parent_subject_ids"] = parent_subject_ids
        batch["view_names"] = view_names_flat

        return batch

    return collate_multiview_subject_bags
def get_valid_subject_clusters(
    subject_to_cluster_graphs,
    sid,
    *,
    min_segments_per_cluster_view=5,
    min_cluster_fraction=0.05,
    exclude_cluster_ids=None,
):
    """
    Decide which clusters are strong enough to become separate views
    for one subject.
    """
    exclude_cluster_ids = set() if exclude_cluster_ids is None else set(exclude_cluster_ids)

    cluster_counts = {}
    total = 0

    for (s, cid), graphs in subject_to_cluster_graphs.items():
        if s != sid:
            continue
        if cid in exclude_cluster_ids:
            continue
        n = len(graphs)
        cluster_counts[cid] = n
        total += n

    valid = []
    rare = []

    for cid, n in cluster_counts.items():
        frac = n / max(total, 1)

        if n >= min_segments_per_cluster_view and frac >= min_cluster_fraction:
            valid.append(cid)
        else:
            rare.append(cid)

    valid = sorted(valid, key=lambda c: cluster_counts[c], reverse=True)
    rare = sorted(rare, key=lambda c: cluster_counts[c], reverse=True)

    return valid, rare, cluster_counts

def _stable_int_from_string(x: str) -> int:
    import hashlib
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)


def _get_graph_label(g) -> int:
    y = g.y
    if torch.is_tensor(y):
        return int(y.view(-1)[0].item())
    return int(y)


def _get_graph_sid(g) -> str:
    return str(getattr(g, "subject_id"))


def _get_graph_segment_id(g) -> int:
    return int(getattr(g, "segment_id", -1))


def _get_graph_cluster_id(g, default: int = -1) -> int:
    c = getattr(g, "cluster_id", default)
    if torch.is_tensor(c):
        return int(c.view(-1)[0].item())
    return int(c)


def _get_graph_clean(g, default: bool = True) -> bool:
    x = getattr(g, "keep_clean", default)
    if torch.is_tensor(x):
        return bool(x.view(-1)[0].item())
    return bool(x)


def _sample_with_replacement_if_needed(
    xs: Sequence[Any],
    k: int,
    rng: random.Random,
) -> List[Any]:
    xs = list(xs)
    if len(xs) == 0:
        raise ValueError("Cannot sample from empty list.")
    if len(xs) >= k:
        return [xs[i] for i in rng.sample(range(len(xs)), k)]
    return xs + [xs[rng.randrange(len(xs))] for _ in range(k - len(xs))]


def _safe_subject_id_for_manifest(sid: str) -> List[str]:
    """
    Allow matching either:
      train_00123 <-> 00123
      val_00123   <-> 00123
      test_00123  <-> 00123
    """
    sid = str(sid)
    out = [sid]
    for prefix in ["train_", "val_", "test_"]:
        if sid.startswith(prefix):
            out.append(sid[len(prefix):])
    return list(dict.fromkeys(out))

def sample_cluster_balanced_subject_view(
    subject_to_cluster_graphs,
    sid,
    *,
    k,
    rng,
    valid_clusters,
    rare_clusters=None,
    include_rare=True,
    max_per_cluster=None,
):
    """
    Sample K segments from a subject using cluster-balanced sampling.

    This prevents one dominant cluster from controlling the whole bag.
    """
    rare_clusters = [] if rare_clusters is None else list(rare_clusters)

    clusters = list(valid_clusters)

    if include_rare:
        clusters = clusters + rare_clusters

    clusters = [
        cid for cid in clusters
        if (sid, cid) in subject_to_cluster_graphs
        and len(subject_to_cluster_graphs[(sid, cid)]) > 0
    ]

    if len(clusters) == 0:
        raise RuntimeError(f"No valid clusters for subject {sid}")

    if max_per_cluster is None:
        max_per_cluster = max(1, math.ceil(k / len(clusters)))

    chosen = []

    # round-robin sampling across clusters
    rng.shuffle(clusters)

    while len(chosen) < k:
        progressed = False

        for cid in clusters:
            pool = subject_to_cluster_graphs[(sid, cid)]
            if len(pool) == 0:
                continue

            current_from_cluster = sum(
                1 for g in chosen
                if int(getattr(g, "cluster_id", -1).view(-1)[0].item())
                == int(cid)
            )

            if current_from_cluster >= max_per_cluster:
                continue

            chosen.append(pool[rng.randrange(len(pool))])
            progressed = True

            if len(chosen) >= k:
                break

        if not progressed:
            # Relax cap if K is still not filled.
            max_per_cluster += 1

    return chosen[:k]
# ============================================================
# Attach cluster metadata from manifest
# ============================================================

def infer_noise_clusters_from_manifest(
    manifest_df: pd.DataFrame,
    *,
    cluster_col: str = "kmeans_cluster_id",
    clean_col: str = "keep_clean",
    distance_col: str = "kmeans_centroid_distance",
    min_cluster_size: int = 50,
    min_clean_rate: float = 0.50,
    max_distance_quantile: float = 0.95,
) -> set[int]:
    """
    Conservative noise-cluster detector.

    A cluster is marked noise-like if:
      - too small
      - low clean rate
      - very high centroid distance, if distance_col exists

    Do not use label distribution to decide noise clusters.
    """
    df = manifest_df.copy()

    if cluster_col not in df.columns:
        raise KeyError(f"Missing cluster column: {cluster_col}")

    if clean_col not in df.columns:
        df[clean_col] = True

    if df[clean_col].dtype != bool:
        df[clean_col] = (
            df[clean_col]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    noise_clusters = set()

    if distance_col in df.columns:
        distance_cutoff = df[distance_col].quantile(max_distance_quantile)
    else:
        distance_cutoff = None

    for cid, cdf in df.groupby(cluster_col):
        cid = int(cid)
        n = len(cdf)
        clean_rate = float(cdf[clean_col].mean())

        is_noise = False

        if n < min_cluster_size:
            is_noise = True

        if clean_rate < min_clean_rate:
            is_noise = True

        if distance_cutoff is not None:
            if float(cdf[distance_col].median()) > float(distance_cutoff):
                is_noise = True

        if is_noise:
            noise_clusters.add(cid)

    return noise_clusters


def attach_cluster_metadata_from_manifest(
    graphs: Sequence[Any],
    manifest_df: pd.DataFrame,
    *,
    cluster_col: str = "kmeans_cluster_id",
    clean_col: str = "keep_clean",
    weight_col: str = "sampling_weight",
    distance_col: str = "kmeans_centroid_distance",
    noise_clusters: Optional[set[int]] = None,
    default_cluster: int = -1,
    default_clean: bool = True,
    default_weight: float = 1.0,
) -> List[Any]:
    """
    Attach these graph-level attributes:
      g.cluster_id
      g.keep_clean
      g.segment_weight
      g.is_noise_cluster

    Required manifest columns:
      subject_id, segment_id, cluster_col

    This does not filter graphs. It only attaches metadata.
    """
    df = manifest_df.copy()

    required = {"subject_id", "segment_id", cluster_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Manifest missing columns: {missing}")

    if clean_col not in df.columns:
        df[clean_col] = default_clean
    if weight_col not in df.columns:
        df[weight_col] = default_weight
    if distance_col not in df.columns:
        df[distance_col] = np.nan

    if df[clean_col].dtype != bool:
        df[clean_col] = (
            df[clean_col]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)

    lookup = {}
    for _, row in df.iterrows():
        lookup[(str(row["subject_id"]), int(row["segment_id"]))] = {
            "cluster_id": int(row[cluster_col]),
            "keep_clean": bool(row[clean_col]),
            "segment_weight": float(row[weight_col]),
            "centroid_distance": float(row[distance_col]) if pd.notna(row[distance_col]) else np.nan,
        }

    if noise_clusters is None:
        noise_clusters = set()

    out = []

    matched = 0
    for g in graphs:
        sid = _get_graph_sid(g)
        seg_id = _get_graph_segment_id(g)

        info = None
        for sid_candidate in _safe_subject_id_for_manifest(sid):
            key = (sid_candidate, seg_id)
            if key in lookup:
                info = lookup[key]
                break

        if info is None:
            cid = int(default_cluster)
            keep_clean = bool(default_clean)
            segment_weight = float(default_weight)
            centroid_distance = np.nan
        else:
            matched += 1
            cid = int(info["cluster_id"])
            keep_clean = bool(info["keep_clean"])
            segment_weight = float(info["segment_weight"])
            centroid_distance = float(info["centroid_distance"])

        g.cluster_id = torch.tensor([cid], dtype=torch.long)
        g.keep_clean = torch.tensor([1 if keep_clean else 0], dtype=torch.long)
        g.segment_weight = torch.tensor([segment_weight], dtype=torch.float32)
        g.centroid_distance = torch.tensor(
            [0.0 if np.isnan(centroid_distance) else centroid_distance],
            dtype=torch.float32,
        )
        g.is_noise_cluster = torch.tensor(
            [1 if cid in noise_clusters else 0],
            dtype=torch.long,
        )

        out.append(g)

    print(
        f"[attach_cluster_metadata] matched {matched}/{len(graphs)} graphs "
        f"using manifest cluster column {cluster_col!r}."
    )

    return out


# ============================================================
# Graph index
# ============================================================

class GraphClusterIndex:
    """
    Index graphs by:
      subject
      label
      cluster
      label + cluster

    This is shared by all augmented MIL datasets.
    """

    def __init__(
        self,
        graphs: Sequence[Any],
        *,
        exclude_noise: bool = True,
        clean_only: bool = True,
        min_segments_per_subject_cluster: int = 1,
    ):
        self.graphs = list(graphs)
        self.exclude_noise = bool(exclude_noise)
        self.clean_only = bool(clean_only)
        self.min_segments_per_subject_cluster = int(min_segments_per_subject_cluster)

        self.subject_to_graphs = defaultdict(list)
        self.subject_to_label = {}
        self.subject_cluster_to_graphs = defaultdict(list)
        self.label_to_subjects = defaultdict(list)
        self.label_to_graphs = defaultdict(list)
        self.label_cluster_to_graphs = defaultdict(list)
        self.label_cluster_to_subjects = defaultdict(set)

        for g in self.graphs:
            sid = _get_graph_sid(g)
            y = _get_graph_label(g)
            cid = _get_graph_cluster_id(g)

            if self.clean_only and not _get_graph_clean(g):
                continue

            is_noise = getattr(g, "is_noise_cluster", torch.tensor([0]))
            if torch.is_tensor(is_noise):
                is_noise = bool(is_noise.view(-1)[0].item())
            else:
                is_noise = bool(is_noise)

            if self.exclude_noise and is_noise:
                continue

            self.subject_to_graphs[sid].append(g)

            if sid in self.subject_to_label and self.subject_to_label[sid] != y:
                raise ValueError(f"Subject {sid} has inconsistent labels.")
            self.subject_to_label[sid] = y

            self.subject_cluster_to_graphs[(sid, cid)].append(g)
            self.label_to_graphs[y].append(g)
            self.label_cluster_to_graphs[(y, cid)].append(g)
            self.label_cluster_to_subjects[(y, cid)].add(sid)

        for sid, y in self.subject_to_label.items():
            self.label_to_subjects[y].append(sid)

        # valid subject-cluster pairs
        self.subject_cluster_keys = []
        for key, gs in self.subject_cluster_to_graphs.items():
            if len(gs) >= self.min_segments_per_subject_cluster:
                self.subject_cluster_keys.append(key)

        self.labels = sorted(self.label_to_subjects.keys())
        self.clusters = sorted({
            cid for (_, cid) in self.subject_cluster_to_graphs.keys()
            if cid >= 0
        })

    def summarize(self, name: str = "GraphClusterIndex"):
        print(f"\n[{name}]")
        print("num graphs indexed:", sum(len(v) for v in self.subject_to_graphs.values()))
        print("num subjects:", len(self.subject_to_graphs))
        print("labels:", {y: len(sids) for y, sids in self.label_to_subjects.items()})
        print("num clusters:", len(self.clusters))
        print("num subject-cluster views:", len(self.subject_cluster_keys))

        cluster_counts = Counter()
        for (_, cid), gs in self.subject_cluster_to_graphs.items():
            cluster_counts[cid] += len(gs)

        print("top clusters:", cluster_counts.most_common(10))


# ============================================================
# Dataset 1: cluster-specific subject views
# ============================================================

class ClusterViewBagDataset(Dataset):
    """
    Each item is one subject-cluster view:
        subject S, cluster c -> K segments from S that belong to c

    Output is compatible with collate_subject_bags:
        {"subject_id", "label", "graphs"}
    """

    def __init__(
        self,
        graph_index: GraphClusterIndex,
        *,
        k: int = 10,
        seed: int = 42,
        min_segments_per_view: int = 1,
        return_debug: bool = True,
    ):
        self.index = graph_index
        self.k = int(k)
        self.seed = int(seed)
        self.epoch = 0
        self.return_debug = bool(return_debug)

        self.keys = []
        for sid, cid in self.index.subject_cluster_keys:
            gs = self.index.subject_cluster_to_graphs[(sid, cid)]
            if len(gs) >= min_segments_per_view:
                self.keys.append((sid, cid))

        if len(self.keys) == 0:
            raise RuntimeError("ClusterViewBagDataset has no valid subject-cluster views.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.keys)

    @property
    def subject_labels(self):
        return [self.index.subject_to_label[sid] for sid, _ in self.keys]

    @property
    def num_node_features(self):
        return self.index.graphs[0].x.shape[-1]

    @property
    def num_nodes(self):
        return self.index.graphs[0].x.shape[0]

    def __getitem__(self, idx: int):
        sid, cid = self.keys[idx]
        y = self.index.subject_to_label[sid]
        gs = self.index.subject_cluster_to_graphs[(sid, cid)]

        rng = random.Random(self.seed + 1000003 * self.epoch + idx)
        chosen = _sample_with_replacement_if_needed(gs, self.k, rng)

        out = {
            "subject_id": f"{sid}::cluster{cid}",
            "label": y,
            "graphs": chosen,
        }

        if self.return_debug:
            out["source_subject_id"] = sid
            out["cluster_id"] = cid
            out["bag_type"] = "cluster_view"

        return out


# ============================================================
# Dataset 2: same-class pseudo bags
# ============================================================

class SameClassPseudoBagDataset(Dataset):
    """
    Create pseudo-bags by mixing segments from multiple subjects of the same class.

    Modes:
      cluster_aligned=False:
          any clean segment from same-class subjects

      cluster_aligned=True:
          all segments come from same class AND same global cluster
    """

    def __init__(
        self,
        graph_index: GraphClusterIndex,
        *,
        k: int = 10,
        bags_per_epoch: int = 1000,
        subjects_per_bag: int = 4,
        max_segments_per_source_subject: int = 4,
        cluster_aligned: bool = False,
        seed: int = 42,
        return_debug: bool = True,
    ):
        self.index = graph_index
        self.k = int(k)
        self.bags_per_epoch = int(bags_per_epoch)
        self.subjects_per_bag = int(subjects_per_bag)
        self.max_segments_per_source_subject = int(max_segments_per_source_subject)
        self.cluster_aligned = bool(cluster_aligned)
        self.seed = int(seed)
        self.epoch = 0
        self.return_debug = bool(return_debug)

        self.labels = list(self.index.labels)

        if self.cluster_aligned:
            valid = []
            for (y, cid), gs in self.index.label_cluster_to_graphs.items():
                source_subjects = self.index.label_cluster_to_subjects[(y, cid)]
                if len(gs) >= 30 and len(source_subjects) >= 5:
                    valid.append((y, cid))
            self.valid_label_clusters = valid
            if len(valid) == 0:
                raise RuntimeError("No valid label-cluster combinations for cluster-aligned pseudo-bags.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.bags_per_epoch

    @property
    def subject_labels(self):
        # approximate label list for class weights/debugging
        return [self.labels[i % len(self.labels)] for i in range(self.bags_per_epoch)]

    @property
    def num_node_features(self):
        return self.index.graphs[0].x.shape[-1]

    @property
    def num_nodes(self):
        return self.index.graphs[0].x.shape[0]

    def _sample_same_class_any_cluster(self, y: int, rng: random.Random):
        source_subjects = list(self.index.label_to_subjects[y])
        if len(source_subjects) == 0:
            raise RuntimeError(f"No source subjects for label {y}")

        rng.shuffle(source_subjects)
        source_subjects = source_subjects[: max(1, min(self.subjects_per_bag, len(source_subjects)))]

        chosen = []
        source_used = []

        per_subject_k = max(1, math.ceil(self.k / len(source_subjects)))
        per_subject_k = min(per_subject_k, self.max_segments_per_source_subject)

        for sid in source_subjects:
            gs = self.index.subject_to_graphs[sid]
            if len(gs) == 0:
                continue
            take = min(per_subject_k, self.k - len(chosen))
            chosen.extend(_sample_with_replacement_if_needed(gs, take, rng))
            source_used.append(sid)
            if len(chosen) >= self.k:
                break

        # fill if needed
        pool = self.index.label_to_graphs[y]
        while len(chosen) < self.k:
            chosen.append(pool[rng.randrange(len(pool))])

        return chosen[: self.k], source_used, None

    def _sample_same_class_same_cluster(self, y: int, cid: int, rng: random.Random):
        source_subjects = list(self.index.label_cluster_to_subjects[(y, cid)])
        rng.shuffle(source_subjects)
        source_subjects = source_subjects[: max(1, min(self.subjects_per_bag, len(source_subjects)))]

        chosen = []
        source_used = []

        per_subject_k = max(1, math.ceil(self.k / len(source_subjects)))
        per_subject_k = min(per_subject_k, self.max_segments_per_source_subject)

        for sid in source_subjects:
            gs = self.index.subject_cluster_to_graphs[(sid, cid)]
            if len(gs) == 0:
                continue
            take = min(per_subject_k, self.k - len(chosen))
            chosen.extend(_sample_with_replacement_if_needed(gs, take, rng))
            source_used.append(sid)
            if len(chosen) >= self.k:
                break

        pool = self.index.label_cluster_to_graphs[(y, cid)]
        while len(chosen) < self.k:
            chosen.append(pool[rng.randrange(len(pool))])

        return chosen[: self.k], source_used, cid

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + 1000003 * self.epoch + idx)

        if self.cluster_aligned:
            y, cid = self.valid_label_clusters[idx % len(self.valid_label_clusters)]
            chosen, source_used, cid = self._sample_same_class_same_cluster(y, cid, rng)
            bag_type = "cluster_pseudo"
            subject_id = f"pseudo_y{y}_c{cid}_{idx}"
        else:
            y = self.labels[idx % len(self.labels)]
            chosen, source_used, cid = self._sample_same_class_any_cluster(y, rng)
            bag_type = "same_class_pseudo"
            subject_id = f"pseudo_y{y}_{idx}"

        out = {
            "subject_id": subject_id,
            "label": int(y),
            "graphs": chosen,
        }

        if self.return_debug:
            out["source_subject_ids"] = list(source_used)
            out["cluster_id"] = -1 if cid is None else int(cid)
            out["bag_type"] = bag_type

        return out


# ============================================================
# Dataset 3: mix real subject bags and pseudo bags
# ============================================================

class MixedRealPseudoBagDataset(Dataset):
    """
    Drop-in dataset that sometimes returns a real subject bag and sometimes
    returns a same-class pseudo-bag.

    Compatible with collate_subject_bags and your existing fit_mil_baseline.
    """

    def __init__(
        self,
        real_dataset: Dataset,
        pseudo_dataset: Dataset,
        *,
        p_real: float = 0.70,
        seed: int = 42,
        length: Optional[int] = None,
    ):
        self.real_dataset = real_dataset
        self.pseudo_dataset = pseudo_dataset
        self.p_real = float(p_real)
        self.seed = int(seed)
        self.epoch = 0
        self.length = int(length) if length is not None else max(len(real_dataset), len(pseudo_dataset))

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        if hasattr(self.real_dataset, "set_epoch"):
            self.real_dataset.set_epoch(epoch)
        if hasattr(self.pseudo_dataset, "set_epoch"):
            self.pseudo_dataset.set_epoch(epoch)

    def __len__(self):
        return self.length

    @property
    def subject_labels(self):
        if hasattr(self.real_dataset, "subject_labels"):
            return list(self.real_dataset.subject_labels)
        return []

    @property
    def num_node_features(self):
        return self.real_dataset.num_node_features

    @property
    def num_nodes(self):
        return self.real_dataset.num_nodes

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + 1000003 * self.epoch + idx)

        if rng.random() < self.p_real:
            j = rng.randrange(len(self.real_dataset))
            item = self.real_dataset[j]
            item["bag_type"] = "real"
            return item

        j = rng.randrange(len(self.pseudo_dataset))
        return self.pseudo_dataset[j]


# ============================================================
# Dataset 4: true multiview subject dataset
# ============================================================

class MultiViewClusterSubjectDataset(Dataset):
    """
    Each item is one subject with multiple views:
      - mixed random view
      - up to max_cluster_views cluster-specific views

    Requires custom collate + custom train loop.
    """

    def __init__(
        self,
        graph_index: GraphClusterIndex,
        *,
        k: int = 10,
        max_cluster_views: int = 3,
        include_mixed_view: bool = True,
        seed: int = 42,
        min_segments_per_cluster_view: int = 5,
        min_cluster_fraction: float =0.05,
    ):
        self.index = graph_index
        self.k = int(k)
        self.max_cluster_views = int(max_cluster_views)
        self.include_mixed_view = bool(include_mixed_view)
        self.seed = int(seed)
        self.epoch = 0
        self.min_segments_per_cluster_view = int(min_segments_per_cluster_view)
        self.min_cluster_fraction = float(min_cluster_fraction)
        self.subject_ids = sorted(self.index.subject_to_graphs.keys())
        if len(self.subject_ids) == 0:
            raise RuntimeError("MultiViewClusterSubjectDataset has no subjects.")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.subject_ids)

    @property
    def subject_labels(self):
        return [self.index.subject_to_label[sid] for sid in self.subject_ids]

    @property
    def num_node_features(self):
        return self.index.graphs[0].x.shape[-1]

    @property
    def num_nodes(self):
        return self.index.graphs[0].x.shape[0]

    def __getitem__(self, idx: int):
        sid = self.subject_ids[idx]
        y = self.index.subject_to_label[sid]

        rng = random.Random(self.seed + 1000003 * self.epoch + _stable_int_from_string(sid))

        views = []
        view_names = []

        if self.include_mixed_view:
            mixed_pool = self.index.subject_to_graphs[sid]
            views.append(_sample_with_replacement_if_needed(mixed_pool, self.k, rng))
            view_names.append("mixed")

        # subject-specific clusters
        # subject_clusters = []
        # for (s, cid), gs in self.index.subject_cluster_to_graphs.items():
        #     if s == sid and len(gs) >= self.min_segments_per_cluster_view:
        #         subject_clusters.append(cid)

        # rng.shuffle(subject_clusters)
        # subject_clusters = subject_clusters[: self.max_cluster_views]

        # for cid in subject_clusters:
        #     gs = self.index.subject_cluster_to_graphs[(sid, cid)]
        #     views.append(_sample_with_replacement_if_needed(gs, self.k, rng))
        #     view_names.append(f"cluster_{cid}")
        valid_clusters, rare_clusters, cluster_counts = get_valid_subject_clusters(
            self.index.subject_cluster_to_graphs,
            sid,
            min_segments_per_cluster_view=self.min_segments_per_cluster_view,
            min_cluster_fraction=0.05,
        )
        # 1. Cluster-balanced mixed view
        mixed_view = sample_cluster_balanced_subject_view(
            self.index.subject_cluster_to_graphs,
            sid,
            k=self.k,
            rng=rng,
            valid_clusters=valid_clusters,
            rare_clusters=rare_clusters,
            include_rare=True,
        )
        views.append(mixed_view)
        view_names.append("cluster_balanced_mixed")

        # 2. Only create separate cluster views for strong clusters
        rng.shuffle(valid_clusters)
        selected_clusters = valid_clusters[: self.max_cluster_views]

        for cid in selected_clusters:
            gs = self.index.subject_cluster_to_graphs[(sid, cid)]

            # No tiny-cluster view
            if len(gs) < self.min_segments_per_cluster_view:
                continue

            views.append(_sample_with_replacement_if_needed(gs, self.k, rng))
            view_names.append(f"cluster_{cid}")

        if len(views) == 0:
            # fallback
            mixed_pool = self.index.subject_to_graphs[sid]
            views.append(_sample_with_replacement_if_needed(mixed_pool, self.k, rng))
            view_names.append("fallback_mixed")

        return {
            "subject_id": sid,
            "label": y,
            "views": views,
            "view_names": view_names,
        }


def collate_multiview_subject_bags(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flatten subject views into normal MIL bags, then add group_ids.

    Model sees B_flat bags.
    group_ids maps each view back to original subject in this batch.
    """
    flat_items = []
    group_ids = []
    parent_subject_ids = []
    view_names_flat = []

    for group_idx, item in enumerate(items):
        sid = item["subject_id"]
        y = int(item["label"])
        views = item["views"]
        view_names = item.get("view_names", [f"view{i}" for i in range(len(views))])

        parent_subject_ids.append(sid)

        for view_idx, graphs in enumerate(views):
            flat_items.append({
                "subject_id": f"{sid}::{view_names[view_idx]}",
                "label": y,
                "graphs": graphs,
            })
            group_ids.append(group_idx)
            view_names_flat.append(view_names[view_idx])

    batch = collate_subject_bags(flat_items)
    batch["group_ids"] = torch.tensor(group_ids, dtype=torch.long)
    batch["parent_subject_ids"] = parent_subject_ids
    batch["view_names"] = view_names_flat

    return batch


# ============================================================
# Loss and train loop for multiview consistency
# ============================================================

def symmetric_kl_from_logits(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    log_pa = F.log_softmax(logits_a / temperature, dim=1)
    pa = F.softmax(logits_a / temperature, dim=1)

    log_pb = F.log_softmax(logits_b / temperature, dim=1)
    pb = F.softmax(logits_b / temperature, dim=1)

    kl_ab = F.kl_div(log_pa, pb.detach(), reduction="batchmean")
    kl_ba = F.kl_div(log_pb, pa.detach(), reduction="batchmean")
    return 0.5 * (kl_ab + kl_ba)


def multiview_consistency_loss(
    logits: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    For each subject group, force all views to agree with the first view.

    Uses prediction consistency only, not embedding MSE, to avoid making
    subject fingerprint stronger.
    """
    losses = []

    unique_groups = torch.unique(group_ids)
    for gid in unique_groups:
        idx = torch.where(group_ids == gid)[0]
        if idx.numel() <= 1:
            continue

        ref = logits[idx[0:1]]
        for j in idx[1:]:
            cur = logits[j:j + 1]
            losses.append(symmetric_kl_from_logits(ref, cur, temperature=temperature))

    if len(losses) == 0:
        return logits.sum() * 0.0

    return torch.stack(losses).mean()


def compute_simple_subject_metrics(y_true, y_pred) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_one_epoch_multiview_mil(
    model,
    loader,
    optimizer,
    criterion,
    device,
    *,
    lambda_consistency: float = 0.1,
    consistency_temperature: float = 1.0,
):
    model.train()

    losses = []
    ce_losses = []
    cons_losses = []

    y_true = []
    y_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        out = model(batch)
        logits = out["logits"]
        labels_flat = batch["labels"]
        group_ids = batch["group_ids"].to(device)

        ce_loss = criterion(logits, labels_flat)

        cons_loss = multiview_consistency_loss(
            logits,
            group_ids,
            temperature=consistency_temperature,
        )

        loss = ce_loss + lambda_consistency * cons_loss

        if "reg_loss" in out:
            loss = loss + out["reg_loss"]

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce_loss.item()))
        cons_losses.append(float(cons_loss.item()))

        # Metrics: average logits over views for each original subject
        with torch.no_grad():
            for gid in torch.unique(group_ids):
                idx = torch.where(group_ids == gid)[0]
                mean_logits = logits[idx].mean(dim=0, keepdim=True)
                pred = int(mean_logits.argmax(dim=1).detach().cpu().item())
                true = int(labels_flat[idx[0]].detach().cpu().item())
                y_pred.append(pred)
                y_true.append(true)

    metrics = compute_simple_subject_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["ce_loss"] = float(np.mean(ce_losses)) if ce_losses else 0.0
    metrics["consistency_loss"] = float(np.mean(cons_losses)) if cons_losses else 0.0
    metrics["y_pred"] = y_pred

    return metrics


# ============================================================
# Dataset builder
# ============================================================

def build_augmented_train_dataset(
    train_graphs,
    *,
    bag_aug_mode: str,
    base_k: int,
    seed: int,
    max_k_per_subject: int = 300,

    # encoder-specific collate function
    base_collate_fn=None,

    # manifest / clustering
    manifest_df: Optional[pd.DataFrame] = None,
    cluster_col: str = "kmeans_cluster_id",
    clean_col: str = "keep_clean",
    weight_col: str = "sampling_weight",

    # noise-cluster filtering
    exclude_noise_clusters: bool = True,
    min_cluster_size: int = 50,
    min_clean_rate: float = 0.50,

    # pseudo-bag config
    pseudo_bags_per_epoch: int = 1000,
    pseudo_subjects_per_bag: int = 4,
    p_real: float = 0.70,

    # multiview config
    multiview_max_cluster_views: int = 3,
    min_segments_per_cluster_view: int = 5,
    min_cluster_fraction: float = 0.05,
):
    """
    Build the training dataset for different MIL bag augmentation modes.

    Returns
    -------
    train_dataset
        Dataset object.

    train_collate_fn
        Correct collate function for the selected encoder and bag mode.

    requires_multiview_train_loop
        False:
            can use the normal train_one_epoch / fit_mil_baseline.

        True:
            use train_one_epoch_multiview_mil because the batch has group_ids.

    Modes
    -----
    none:
        Original subject-level MIL bags.

    cluster_view_ce:
        Each subject-cluster pair becomes one MIL bag.

    same_class_pseudo:
        Pseudo-bags mix segments from multiple subjects of the same class.

    cluster_pseudo:
        Pseudo-bags mix segments from multiple subjects with same class
        and same global cluster.

    mixed_real_pseudo:
        Training alternates between real subject bags and cluster-aligned
        same-class pseudo-bags.

    multiview_consistency:
        Each subject produces multiple views:
            mixed cluster-balanced view
            cluster-specific views
        Requires multiview consistency training loop.
    """

    if base_collate_fn is None:
        base_collate_fn = collate_subject_bags

    bag_aug_mode = str(bag_aug_mode).lower()

    valid_modes = {
        "none",
        "real",
        "standard",
        "cluster_view_ce",
        "same_class_pseudo",
        "cluster_pseudo",
        "mixed_real_pseudo",
        "multiview_consistency",
        "mixed_realmultiview_pseudo",
    }

    if bag_aug_mode not in valid_modes:
        raise ValueError(
            f"Unknown bag_aug_mode={bag_aug_mode!r}. "
            f"Valid modes are: {sorted(valid_modes)}"
        )

    # ============================================================
    # 1. Original baseline: normal subject MIL dataset
    # ============================================================
    if bag_aug_mode in {"none", "real", "standard"}:
        ds = LabelAwareSubjectBagDataset(
            train_graphs,
            train=True,
            base_k=base_k,
            max_k_per_subject=max_k_per_subject,
            seed=seed,
            return_segment_ids=True,
        )

        return ds, base_collate_fn, False

    # ============================================================
    # 2. All cluster/pseudo/multiview modes need manifest metadata
    # ============================================================
    if manifest_df is None:
        raise ValueError(
            f"bag_aug_mode={bag_aug_mode!r} requires manifest_df. "
            "Load your global cluster manifest first and pass it here."
        )

    # Detect noise-like clusters conservatively.
    if exclude_noise_clusters:
        noise_clusters = infer_noise_clusters_from_manifest(
            manifest_df,
            cluster_col=cluster_col,
            clean_col=clean_col,
            min_cluster_size=min_cluster_size,
            min_clean_rate=min_clean_rate,
        )
    else:
        noise_clusters = set()

    print(f"[build_augmented_train_dataset] bag_aug_mode={bag_aug_mode}")
    print(f"[build_augmented_train_dataset] exclude_noise_clusters={exclude_noise_clusters}")
    print(f"[build_augmented_train_dataset] noise_clusters={sorted(noise_clusters)}")

    # Attach cluster metadata to graphs.
    train_graphs = attach_cluster_metadata_from_manifest(
        train_graphs,
        manifest_df,
        cluster_col=cluster_col,
        clean_col=clean_col,
        weight_col=weight_col,
        noise_clusters=noise_clusters,
    )

    # Build graph index.
    index = GraphClusterIndex(
        train_graphs,
        exclude_noise=exclude_noise_clusters,
        clean_only=True,
        min_segments_per_subject_cluster=1,
    )

    index.summarize(name=f"index_for_{bag_aug_mode}")

    # ============================================================
    # 3. Cluster-view CE
    # ============================================================
    if bag_aug_mode == "cluster_view_ce":
        ds = ClusterViewBagDataset(
            index,
            k=base_k,
            seed=seed,
            min_segments_per_view=min_segments_per_cluster_view,
            return_debug=True,
        )

        return ds, base_collate_fn, False

    # ============================================================
    # 4. Same-class pseudo-bag
    # ============================================================
    if bag_aug_mode == "same_class_pseudo":
        ds = SameClassPseudoBagDataset(
            index,
            k=base_k,
            bags_per_epoch=pseudo_bags_per_epoch,
            subjects_per_bag=pseudo_subjects_per_bag,
            max_segments_per_source_subject=max(1, base_k // pseudo_subjects_per_bag),
            cluster_aligned=False,
            seed=seed,
            return_debug=True,
        )

        return ds, base_collate_fn, False

    # ============================================================
    # 5. Cluster-aligned same-class pseudo-bag
    # ============================================================
    if bag_aug_mode == "cluster_pseudo":
        ds = SameClassPseudoBagDataset(
            index,
            k=base_k,
            bags_per_epoch=pseudo_bags_per_epoch,
            subjects_per_bag=pseudo_subjects_per_bag,
            max_segments_per_source_subject=max(1, base_k // pseudo_subjects_per_bag),
            cluster_aligned=True,
            seed=seed,
            return_debug=True,
        )

        return ds, base_collate_fn, False

    # ============================================================
    # 6. Mixed real subject bags + cluster-aligned pseudo-bags
    # ============================================================
    if bag_aug_mode == "mixed_real_pseudo":
        real_ds = LabelAwareSubjectBagDataset(
            train_graphs,
            train=True,
            base_k=base_k,
            max_k_per_subject=max_k_per_subject,
            seed=seed,
            return_segment_ids=True,
        )

        pseudo_ds = SameClassPseudoBagDataset(
            index,
            k=base_k,
            bags_per_epoch=pseudo_bags_per_epoch,
            subjects_per_bag=pseudo_subjects_per_bag,
            max_segments_per_source_subject=max(1, base_k // pseudo_subjects_per_bag),
            cluster_aligned=True,
            seed=seed,
            return_debug=True,
        )

        ds = MixedRealPseudoBagDataset(
            real_dataset=real_ds,
            pseudo_dataset=pseudo_ds,
            p_real=p_real,
            seed=seed,
            length=max(len(real_ds), pseudo_bags_per_epoch),
        )

        return ds, base_collate_fn, False

    if bag_aug_mode == "mixed_realmultiview_pseudo":
        real_ds = ClusterViewBagDataset(
            index,
            k=base_k,
            seed=seed,
            min_segments_per_view=min_segments_per_cluster_view,
            return_debug=True,
        )

        pseudo_ds = SameClassPseudoBagDataset(
            index,
            k=base_k,
            bags_per_epoch=pseudo_bags_per_epoch,
            subjects_per_bag=pseudo_subjects_per_bag,
            max_segments_per_source_subject=max(1, base_k // pseudo_subjects_per_bag),
            cluster_aligned=True,
            seed=seed,
            return_debug=True,
        )

        ds = MixedRealPseudoBagDataset(
            real_dataset=real_ds,
            pseudo_dataset=pseudo_ds,
            p_real=p_real,
            seed=seed,
            length=max(len(real_ds), pseudo_bags_per_epoch),
        )

        return ds, base_collate_fn, False
    # ============================================================
    # 7. True multiview consistency
    # ============================================================
    if bag_aug_mode == "multiview_consistency":
        ds = MultiViewClusterSubjectDataset(
            index,
            k=base_k,
            max_cluster_views=multiview_max_cluster_views,
            include_mixed_view=True,
            seed=seed,
            min_segments_per_cluster_view=min_segments_per_cluster_view,
            min_cluster_fraction=min_cluster_fraction,
        )

        multiview_collate_fn = make_collate_multiview_subject_bags(
            base_collate_fn
        )

        return ds, multiview_collate_fn, True

    raise RuntimeError(f"Unhandled bag_aug_mode={bag_aug_mode!r}")

# ============================================================
# Debug/smoke tests before real training
# ============================================================

def debug_bag_dataset(dataset, *, n: int = 5):
    print(f"\n[debug_bag_dataset] dataset={dataset.__class__.__name__}, len={len(dataset)}")

    for i in range(min(n, len(dataset))):
        item = dataset[i]

        if "graphs" in item:
            graphs = item["graphs"]
            clusters = [_get_graph_cluster_id(g) for g in graphs]
            source_sids = sorted({_get_graph_sid(g) for g in graphs})

            print(
                f"item {i}: subject_id={item['subject_id']}, label={item['label']}, "
                f"bag_type={item.get('bag_type', 'NA')}, "
                f"num_graphs={len(graphs)}, "
                f"clusters={Counter(clusters)}, "
                f"source_subjects={source_sids[:5]}{'...' if len(source_sids) > 5 else ''}"
            )

        elif "views" in item:
            print(
                f"item {i}: subject_id={item['subject_id']}, label={item['label']}, "
                f"num_views={len(item['views'])}, view_names={item.get('view_names')}"
            )
            for vname, graphs in zip(item.get("view_names", []), item["views"]):
                clusters = [_get_graph_cluster_id(g) for g in graphs]
                print(f"  view={vname}, num_graphs={len(graphs)}, clusters={Counter(clusters)}")


def smoke_test_augmented_loader(
    model,
    loader,
    device,
    *,
    multiview: bool = False,
):
    """
    Run one forward pass to confirm collate + model interface works.
    """
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    with torch.no_grad():
        out = model(batch)

    print("\n[smoke_test_augmented_loader]")
    print("batch keys:", sorted(batch.keys()))
    print("logits shape:", tuple(out["logits"].shape))
    print("bag_emb shape:", tuple(out["bag_emb"].shape))
    print("graph_emb shape:", tuple(out["graph_emb"].shape))
    print("labels shape:", tuple(batch["labels"].shape))
    print("bag_sizes:", batch["bag_sizes"].detach().cpu().tolist()[:10])

    if multiview:
        print("group_ids:", batch["group_ids"].detach().cpu().tolist()[:20])
        print("view_names:", batch["view_names"][:20])

    return out