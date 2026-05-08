from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

import copy
import json
import sys
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from gnn import FusedGraphBankGNN
from graph_construction import build_graph_bank
from metrics import summarize_classification_metrics
from models_mil import GatedAttentionMILPool, SubjectFusionHead, aggregate_subject_predictions
from utils import ensure_dir, get_device, make_run_name, set_seed

if TYPE_CHECKING:  # pragma: no cover
    from caueeg_main_new import CAUEEGExperimentSpec, H5SubjectEntry


def _cm():
    import caueeg_main_new as cm

    return cm


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)
from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


@dataclass(slots=True)
class SegmentInstance:
    subject_id: str
    label: int
    segment_id: int
    node_features: np.ndarray      # [N, F]
    adj_bank: np.ndarray           # [V, N, N]
    topology_bank: np.ndarray      # [V, N, N]
    view_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# H5 -> segment instances / flat segment graphs
# ---------------------------------------------------------------------


def _normalize_view_name(name: str) -> str:
    return str(name).strip()


def _as_float32_np(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _select_feature_families_for_one_window(
    entry: "H5SubjectEntry",
    window_idx: int,
    feature_families: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    parts: list[np.ndarray] = []
    by_family: dict[str, np.ndarray] = {}

    for fam in feature_families:
        x = np.asarray(entry.features[fam], dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Feature family {fam!r} expected [W,N,F], got {x.shape}.")
        xw = x[window_idx].astype(np.float32, copy=False)  # [N, F_fam]
        parts.append(xw)
        by_family[str(fam)] = xw

    x_all = np.concatenate(parts, axis=-1).astype(np.float32, copy=False)
    return x_all, by_family


_FEATURE_FAMILY_NAME_ALIASES: dict[str, str] = {
    "rbp": "relative_band_power",
    "relative_band_power": "relative_band_power",
    "bandpower": "relative_band_power",
    "hjorth": "hjorth",
    "statistical": "statistical",
    "stats": "statistical",
    "time_domain": "statistical",
    "wavelet": "wavelet_energy",
    "wavelet_energy": "wavelet_energy",
    "entropy": "entropy",
}


def _resolve_feature_family_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    key = str(name).strip().lower()
    return _FEATURE_FAMILY_NAME_ALIASES.get(key, str(name))


_DEFAULT_SKIP_SPEC_KEYS = {
    "feature_family",
}


def _candidate_uses_feature_family(spec: Mapping[str, Any]) -> Optional[str]:
    fam = spec.get("feature_family", None)
    return _resolve_feature_family_name(fam)


def _build_graph_bank_with_optional_family_specific_views(
    *,
    node_features_all: np.ndarray,
    node_features_by_family: Mapping[str, np.ndarray],
    connectivity_sources: Mapping[str, np.ndarray],
    band_names_map: Mapping[str, Optional[Sequence[str]]],
    graph_bank_specs: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build a graph bank from candidate specs.

    Supports specs with optional `feature_family`, in which case only that
    family's node features are used for `feature_induced` candidates.
    """
    payload = {
        metric: (_as_float32_np(mat), band_names_map.get(metric, None))
        for metric, mat in connectivity_sources.items()
    }

    bank = []
    for raw_spec in graph_bank_specs:
        spec = dict(raw_spec)
        fam = _candidate_uses_feature_family(spec)
        if fam is not None:
            if fam not in node_features_by_family:
                raise KeyError(
                    f"Candidate {spec.get('name', '<unnamed>')!r} requests feature_family={fam!r}, "
                    f"but available families are {sorted(node_features_by_family.keys())}."
                )
            node_x = _as_float32_np(node_features_by_family[fam])
            spec = {k: v for k, v in spec.items() if k not in _DEFAULT_SKIP_SPEC_KEYS}
        else:
            node_x = _as_float32_np(node_features_all)

        built = build_graph_bank(
            node_features=node_x,
            connectivity_sources=payload,
            candidate_specs=[spec],
            fixed_topology=None,
        )
        if len(built) != 1:
            raise ValueError(
                f"Expected build_graph_bank(...) to return exactly one candidate for spec={spec}, got {len(built)}."
            )
        bank.extend(built)

    if len(bank) == 0:
        raise ValueError("Graph bank is empty.")

    adj_bank = np.stack([cand.adjacency for cand in bank], axis=0).astype(np.float32, copy=False)
    topology_bank = np.stack([cand.topology for cand in bank], axis=0).astype(np.float32, copy=False)
    view_names = [_normalize_view_name(cand.name) for cand in bank]
    return adj_bank, topology_bank, view_names


def build_segment_instance_from_h5_entry(
    entry: "H5SubjectEntry",
    *,
    window_idx: int,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SegmentInstance:
    cm = _cm()

    node_features_all, node_features_by_family = _select_feature_families_for_one_window(
        entry,
        window_idx,
        feature_families,
    )
    node_features_all = cm.zscore_node_features(node_features_all)

    # Keep per-family matrices available for family-specific feature-induced graphs.
    node_features_by_family = {
        fam: cm.zscore_node_features(x) for fam, x in node_features_by_family.items()
    }

    connectivity_sources, band_names_map = cm.aggregate_connectivity_sources(
        entry,
        np.asarray([window_idx], dtype=np.int64),
        connectivity_metrics=connectivity_metrics,
        reduce_mode="mean",
    )

    if graph_bank_specs is None:
        from macro_mvgnn_new import build_rich_macro_graph_bank

        bank = build_rich_macro_graph_bank(
            channel_names=entry.channel_names,
            node_feature_mean_all=node_features_all,
            node_feature_mean_by_family=node_features_by_family,
            connectivity_sources=connectivity_sources,
            band_names_map=band_names_map,
            feature_families=feature_families,
        )
        adj_bank = np.stack([cand.adjacency for cand in bank], axis=0).astype(np.float32, copy=False)
        topology_bank = np.stack([cand.topology for cand in bank], axis=0).astype(np.float32, copy=False)
        view_names = [_normalize_view_name(cand.name) for cand in bank]
    else:
        adj_bank, topology_bank, view_names = _build_graph_bank_with_optional_family_specific_views(
            node_features_all=node_features_all,
            node_features_by_family=node_features_by_family,
            connectivity_sources=connectivity_sources,
            band_names_map=band_names_map,
            graph_bank_specs=graph_bank_specs,
        )

    return SegmentInstance(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        segment_id=int(entry.segment_id[window_idx]),
        node_features=node_features_all,
        adj_bank=adj_bank,
        topology_bank=topology_bank,
        view_names=list(view_names),
        metadata={
            "window_idx": int(window_idx),
            "start_sample": int(entry.start_sample[window_idx]),
            "end_sample": int(entry.end_sample[window_idx]),
        },
    )


def segment_instance_to_graph(seg: SegmentInstance) -> Data:
    # Placeholder adjacency. The true graph is rebuilt inside FusedGraphBankGNN
    # from adj_bank/topology_bank after learned fusion.
    fallback_adj = np.mean(seg.adj_bank, axis=0).astype(np.float32, copy=False)
    fallback_adj = 0.5 * (fallback_adj + fallback_adj.T)
    np.fill_diagonal(fallback_adj, 0.0)

    edge_index, edge_weight = dense_to_sparse(torch.tensor(fallback_adj, dtype=torch.float32))
    g = Data(
        x=torch.tensor(seg.node_features, dtype=torch.float32),
        edge_index=edge_index.long(),
        y=torch.tensor([int(seg.label)], dtype=torch.long),
    )
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()
    g.adj = torch.tensor(fallback_adj, dtype=torch.float32)

    g.adj_bank = torch.tensor(seg.adj_bank, dtype=torch.float32)
    g.topology_bank = torch.tensor(seg.topology_bank, dtype=torch.float32)
    g.subject_id = str(seg.subject_id)
    g.segment_id = int(seg.segment_id)
    g.view_names = list(seg.view_names)

    for key, value in seg.metadata.items():
        setattr(g, key, value)

    return g


def build_segment_graphs_for_entries(
    entries: Mapping[str, "H5SubjectEntry"],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[Data]:
    graphs: list[Data] = []
    for sid in sorted(entries.keys()):
        entry = entries[sid]
        num_windows = len(entry.segment_id)
        for window_idx in range(num_windows):
            seg = build_segment_instance_from_h5_entry(
                entry,
                window_idx=window_idx,
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                graph_bank_specs=graph_bank_specs,
            )
            graphs.append(segment_instance_to_graph(seg))
    return graphs


# ---------------------------------------------------------------------
# LINKX-style subject-bag dataset with base_k sampling
# ---------------------------------------------------------------------


class GraphBankSubjectBagDataset(Dataset):
    """
    Subject-bag dataset over flat segment graphs.

    Behavior mirrors the practical backbone used in caueeg_linkx_mil.py:
    - training can sample `base_k` segments per subject each epoch/item fetch
    - validation/test can use all segments or a deterministic cap
    - each returned item is one subject bag
    """

    def __init__(
        self,
        graphs: Sequence[Data],
        *,
        train: bool,
        base_k: Optional[int] = None,
        max_k_per_subject: Optional[int] = None,
        eval_k_per_subject: Optional[int] = None,
        seed: int = 42,
        return_segment_ids: bool = True,
    ) -> None:
        super().__init__()
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("graphs must not be empty.")

        self.train = bool(train)
        self.base_k = None if base_k is None else int(base_k)
        self.max_k_per_subject = None if max_k_per_subject is None else int(max_k_per_subject)
        self.eval_k_per_subject = None if eval_k_per_subject is None else int(eval_k_per_subject)
        self.seed = int(seed)
        self.return_segment_ids = bool(return_segment_ids)

        self.subject_to_graphs: dict[str, list[Data]] = defaultdict(list)
        self.subject_labels: dict[str, int] = {}
        for g in self.graphs:
            sid = str(getattr(g, "subject_id"))
            self.subject_to_graphs[sid].append(g)
            label = int(g.y.view(-1)[0].item())
            if sid in self.subject_labels and self.subject_labels[sid] != label:
                raise ValueError(f"Inconsistent labels for subject {sid!r}.")
            self.subject_labels[sid] = label

        self.subject_ids = sorted(self.subject_to_graphs.keys())
        if len(self.subject_ids) == 0:
            raise ValueError("No subject IDs found in graphs.")

        first_g = self.subject_to_graphs[self.subject_ids[0]][0]
        self.num_nodes = int(first_g.x.shape[0])
        self.num_node_features = int(first_g.x.shape[1])
        self.num_views = int(first_g.adj_bank.shape[0])

        self.view_names = list(getattr(first_g, "view_names", []))
        for sid in self.subject_ids:
            for g in self.subject_to_graphs[sid]:
                if int(g.x.shape[0]) != self.num_nodes:
                    raise ValueError("All graphs must have the same number of nodes.")
                if int(g.x.shape[1]) != self.num_node_features:
                    raise ValueError("All graphs must have the same node feature dimension.")
                if int(g.adj_bank.shape[0]) != self.num_views:
                    raise ValueError("All graphs must have the same number of graph-bank views.")
                cur_views = list(getattr(g, "view_names", []))
                if self.view_names and cur_views and cur_views != self.view_names:
                    raise ValueError("All graphs must use the same graph-bank view ordering.")

        # Compatibility with the LINKX MIL runner.
        self.num_node_features = self.num_node_features
        self.num_nodes = self.num_nodes

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _rng_for_subject(self, sid: str, idx: int) -> np.random.Generator:
        sid_hash = abs(hash((sid, idx, self.seed))) % (2**32)
        return np.random.default_rng(sid_hash)

    def _choose_graphs_for_subject(self, sid: str, idx: int) -> list[Data]:
        graphs = self.subject_to_graphs[sid]
        n = len(graphs)
        if n == 0:
            raise ValueError(f"Subject {sid!r} has zero graphs.")

        if self.train:
            if self.base_k is None:
                chosen_k = n
            else:
                chosen_k = self.base_k
                if self.max_k_per_subject is not None:
                    chosen_k = min(chosen_k, self.max_k_per_subject)
                chosen_k = min(chosen_k, n)

            if chosen_k >= n:
                return list(graphs)

            rng = self._rng_for_subject(sid, idx)
            picked = np.sort(rng.choice(np.arange(n), size=chosen_k, replace=False))
            return [graphs[int(i)] for i in picked.tolist()]

        # evaluation
        if self.eval_k_per_subject is None or self.eval_k_per_subject >= n:
            return list(graphs)
        return list(graphs[: self.eval_k_per_subject])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.subject_ids[int(idx)]
        selected = self._choose_graphs_for_subject(sid, int(idx))
        label = int(self.subject_labels[sid])

        out = {
            "subject_id": sid,
            "label": label,
            "graphs": selected,
        }
        if self.return_segment_ids:
            out["segment_ids"] = [int(getattr(g, "segment_id")) for g in selected]
        return out


# ---------------------------------------------------------------------
# Collate for graph-bank subject bags
# ---------------------------------------------------------------------


def collate_subject_bags_graph_bank(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    batch = list(batch)
    if len(batch) == 0:
        raise ValueError("Empty batch.")

    first_graph = batch[0]["graphs"][0]
    num_nodes = int(first_graph.x.shape[0])
    num_node_features = int(first_graph.x.shape[1])
    num_views = int(first_graph.adj_bank.shape[0])
    view_names = list(getattr(first_graph, "view_names", []))

    bsz = len(batch)
    max_segments = max(len(item["graphs"]) for item in batch)

    node_feature_bag = torch.zeros((bsz, max_segments, num_nodes, num_node_features), dtype=torch.float32)
    adj_bank_bag = torch.zeros((bsz, max_segments, num_views, num_nodes, num_nodes), dtype=torch.float32)
    topology_bank_bag = torch.zeros((bsz, max_segments, num_views, num_nodes, num_nodes), dtype=torch.float32)
    segment_mask = torch.zeros((bsz, max_segments), dtype=torch.bool)
    segment_ids = torch.full((bsz, max_segments), fill_value=-1, dtype=torch.long)

    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    subject_ids = [str(item["subject_id"]) for item in batch]

    for b_idx, item in enumerate(batch):
        graphs = list(item["graphs"])
        for s_idx, g in enumerate(graphs):
            cur_view_names = list(getattr(g, "view_names", []))
            if view_names and cur_view_names and cur_view_names != view_names:
                raise ValueError("All graphs in the batch must share the same view ordering.")
            if int(g.x.shape[0]) != num_nodes or int(g.x.shape[1]) != num_node_features:
                raise ValueError("All graphs in the batch must share the same x shape.")
            if int(g.adj_bank.shape[0]) != num_views:
                raise ValueError("All graphs in the batch must share the same number of views.")

            node_feature_bag[b_idx, s_idx] = g.x.float()
            adj_bank_bag[b_idx, s_idx] = g.adj_bank.float()
            topology_bank_bag[b_idx, s_idx] = g.topology_bank.float()
            segment_mask[b_idx, s_idx] = True
            segment_ids[b_idx, s_idx] = int(getattr(g, "segment_id", -1))

    return {
        "node_feature_bag": node_feature_bag,      # [B,S,N,F]
        "adj_bank_bag": adj_bank_bag,              # [B,S,V,N,N]
        "topology_bank_bag": topology_bank_bag,    # [B,S,V,N,N]
        "segment_mask": segment_mask,              # [B,S]
        "segment_ids": segment_ids,                # [B,S]
        "labels": labels,
        "subject_ids": subject_ids,
        "view_names": view_names,
    }


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------


class SubjectClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SegmentGraphBankEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_views: int,
        num_classes: int,
        graph_backbone: str = "gatv2",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
        readout_type: str = "gated_attention",
        fusion_mode: str = "summary_gated",
        topology_rule: str = "vote",
        vote_threshold: float = 0.45,
        fusion_temperature: float = 0.7,
        fusion_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_classes = int(num_classes)
        self.graph_emb_dim = int(graph_emb_dim)

        self.graph_model = FusedGraphBankGNN(
            num_node_features=int(num_node_features),
            num_classes=int(num_classes),
            num_nodes=int(num_nodes),
            num_candidates=int(num_views),
            backbone=str(graph_backbone),
            hidden_dim=int(hidden_dim),
            graph_emb_dim=int(graph_emb_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            gat_heads=int(gat_heads),
            use_edge_weight=True,
            use_batchnorm=True,
            readout_type=str(readout_type),
            fusion_mode=str(fusion_mode),
            topology_rule=str(topology_rule),
            vote_threshold=float(vote_threshold),
            fusion_temperature=float(fusion_temperature),
            fusion_hidden_dim=None if fusion_hidden_dim is None else int(fusion_hidden_dim),
            return_attention_weights=True,
        )

    @staticmethod
    def _dense_to_batch(dense_x: Tensor, fallback_adj: Tensor, labels: Optional[Tensor] = None) -> Batch:
        data_list: list[Data] = []
        for i in range(dense_x.shape[0]):
            adj = 0.5 * (fallback_adj[i] + fallback_adj[i].transpose(-1, -2))
            adj = adj.clone()
            adj.fill_diagonal_(0.0)
            edge_index, edge_weight = dense_to_sparse(adj)
            data = Data(x=dense_x[i], edge_index=edge_index.long())
            data.edge_weight = edge_weight.float()
            data.edge_attr = edge_weight.view(-1, 1).float()
            data.adj = adj.float()
            if labels is not None:
                data.y = labels[i : i + 1].long()
            data_list.append(data)
        return Batch.from_data_list(data_list)

    def forward(
        self,
        *,
        node_features: Tensor,     # [B,N,F]
        adj_bank: Tensor,          # [B,V,N,N]
        topology_bank: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> dict[str, Tensor | None]:
        fallback_adj = torch.mean(adj_bank, dim=1)
        pyg_batch = self._dense_to_batch(node_features, fallback_adj, labels=labels)

        out = self.graph_model(
            pyg_batch,
            adj_bank=adj_bank,
            topology_bank=topology_bank,
            return_dict=True,
            return_attention_weights=True,
        )

        return {
            "logits": out.logits,
            "embedding": out.embedding,
            "view_attention_weights": out.fusion_weights,
            "graph_attention_weights": out.graph_attention_weights,
            "fused_adjacency": None if out.aux is None else out.aux.get("fused_adjacency"),
            "fused_topology": None if out.aux is None else out.aux.get("fused_topology"),
        }


class SubjectSegmentMVGNN(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_views: int,
        num_classes: int,
        graph_backbone: str = "gatv2",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
        graph_readout_type: str = "gated_attention",
        graph_bank_fusion_mode: str = "summary_gated",
        topology_rule: str = "vote",
        vote_threshold: float = 0.45,
        fusion_temperature: float = 0.7,
        fusion_hidden_dim: Optional[int] = None,
        subject_aggregation: str = "gated_attention_mil",
        subject_attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.segment_emb_dim = int(graph_emb_dim)
        self.subject_aggregation = str(subject_aggregation).lower()

        self.segment_encoder = SegmentGraphBankEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_views=num_views,
            num_classes=num_classes,
            graph_backbone=graph_backbone,
            hidden_dim=hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            gat_heads=gat_heads,
            readout_type=graph_readout_type,
            fusion_mode=graph_bank_fusion_mode,
            topology_rule=topology_rule,
            vote_threshold=vote_threshold,
            fusion_temperature=fusion_temperature,
            fusion_hidden_dim=fusion_hidden_dim,
        )

        self.subject_classifier: Optional[nn.Module] = None
        self.subject_pool: Optional[nn.Module] = None
        self.subject_fusion_head: Optional[nn.Module] = None

        if self.subject_aggregation in {"mean_mil", "none"}:
            self.subject_classifier = SubjectClassifierHead(graph_emb_dim, num_classes, dropout=dropout)
        elif self.subject_aggregation == "gated_attention_mil":
            self.subject_pool = GatedAttentionMILPool(graph_emb_dim, attn_dim=subject_attn_dim, dropout=dropout)
            self.subject_classifier = SubjectClassifierHead(graph_emb_dim, num_classes, dropout=dropout)
        elif self.subject_aggregation == "subject_fusion":
            self.subject_fusion_head = SubjectFusionHead(
                in_dim=graph_emb_dim,
                num_classes=num_classes,
                instance_logit_dim=num_classes,
                hidden_dim=max(64, graph_emb_dim),
                fusion_dim=max(64, graph_emb_dim),
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported subject_aggregation={subject_aggregation!r}")

    def forward(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        node_feature_bag = batch["node_feature_bag"]     # [B,S,N,F]
        adj_bank_bag = batch["adj_bank_bag"]             # [B,S,V,N,N]
        topology_bank_bag = batch.get("topology_bank_bag", None)
        segment_mask = batch["segment_mask"]             # [B,S]

        bsz, max_segments = segment_mask.shape
        flat_mask = segment_mask.reshape(-1)
        if not torch.any(flat_mask):
            raise ValueError("No valid segments in batch.")

        flat_x = node_feature_bag.reshape(-1, *node_feature_bag.shape[2:])[flat_mask]
        flat_adj = adj_bank_bag.reshape(-1, *adj_bank_bag.shape[2:])[flat_mask]
        flat_topo = None
        if topology_bank_bag is not None:
            flat_topo = topology_bank_bag.reshape(-1, *topology_bank_bag.shape[2:])[flat_mask]

        seg_out = self.segment_encoder(
            node_features=flat_x,
            adj_bank=flat_adj,
            topology_bank=flat_topo,
        )

        seg_emb = seg_out["embedding"]
        seg_logits = seg_out["logits"]

        grouped_emb = torch.zeros(
            (bsz, max_segments, self.segment_emb_dim),
            dtype=seg_emb.dtype,
            device=seg_emb.device,
        )
        grouped_emb.view(-1, self.segment_emb_dim)[flat_mask] = seg_emb

        grouped_logits = torch.zeros(
            (bsz, max_segments, self.num_classes),
            dtype=seg_logits.dtype,
            device=seg_logits.device,
        )
        grouped_logits.view(-1, self.num_classes)[flat_mask] = seg_logits

        agg = aggregate_subject_predictions(
            instance_embeddings=grouped_emb,
            instance_logits=grouped_logits if self.subject_aggregation == "subject_fusion" else None,
            mask=segment_mask.to(device=grouped_emb.device),
            method=self.subject_aggregation,
            classifier=self.subject_classifier,
            pool=self.subject_pool,
            fusion_head=self.subject_fusion_head,
            sort_subjects=False,
        )

        return {
            "logits": agg["subject_logits"],
            "probs": agg["subject_prob"],
            "pred": agg["subject_pred"],
            "subject_embedding": agg["subject_embedding"],
            "subject_attention_weights": agg["attention_weights"],
            "segment_embeddings_grouped": grouped_emb,
            "segment_logits_grouped": grouped_logits,
            "segment_view_attention_weights": seg_out["view_attention_weights"],   # [num_valid_segments, K]
            "segment_fused_adjacency": seg_out["fused_adjacency"],
            "segment_fused_topology": seg_out["fused_topology"],
            "segment_ids": batch["segment_ids"],
            "subject_ids": batch["subject_ids"],
            "segment_mask": batch["segment_mask"],
            "view_names": batch["view_names"],
        }


# ---------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _flatten_segment_mil_attention(
    *,
    batch_subject_ids: Sequence[str],
    segment_ids: np.ndarray,
    segment_mask: np.ndarray,
    mil_attention: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for b_idx, sid in enumerate(batch_subject_ids):
        valid_ids = segment_ids[b_idx][segment_mask[b_idx]]
        valid_attn = mil_attention[b_idx][segment_mask[b_idx]]
        for seg_id, attn in zip(valid_ids.tolist(), valid_attn.tolist()):
            rows.append(
                {
                    "subject_id": str(sid),
                    "segment_id": int(seg_id),
                    "mil_attention": float(attn),
                }
            )
    return pd.DataFrame(rows)


# @torch.no_grad()
def collect_epoch_outputs_segment(
    model: SubjectSegmentMVGNN,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> dict[str, Any]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    n_batches = 0
    y_true_all: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    subject_ids_all: list[str] = []

    segment_view_attn_all: list[np.ndarray] = []
    segment_subject_ids_all: list[str] = []
    segment_segment_ids_all: list[int] = []
    view_names_ref: Optional[list[str]] = None

    mil_attn_rows: list[pd.DataFrame] = []

    # Need gradients for training.
    grad_ctx = torch.enable_grad if train else torch.no_grad
    with grad_ctx():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["labels"])
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

            if view_names_ref is None:
                view_names_ref = list(batch["view_names"])

            seg_attn = out["segment_view_attention_weights"].detach().cpu().numpy()
            segment_view_attn_all.append(seg_attn)

            seg_mask = batch["segment_mask"].detach().cpu().numpy().astype(bool)
            seg_ids = batch["segment_ids"].detach().cpu().numpy()
            for b_idx, sid in enumerate(batch["subject_ids"]):
                valid_ids = seg_ids[b_idx][seg_mask[b_idx]]
                segment_subject_ids_all.extend([str(sid)] * len(valid_ids))
                segment_segment_ids_all.extend([int(x) for x in valid_ids.tolist()])

            mil_attn = out.get("subject_attention_weights", None)
            if mil_attn is not None:
                mil_attn_np = mil_attn.detach().cpu().numpy()
                mil_attn_rows.append(
                    _flatten_segment_mil_attention(
                        batch_subject_ids=batch["subject_ids"],
                        segment_ids=seg_ids,
                        segment_mask=seg_mask,
                        mil_attention=mil_attn_np,
                    )
                )

            probs = out["probs"]
            pred = out["pred"]

            total_loss += float(loss.detach().cpu().item())
            n_batches += 1

            y_true_all.append(batch["labels"].detach().cpu().numpy())
            logits_all.append(out["logits"].detach().cpu().numpy())
            probs_all.append(probs.detach().cpu().numpy())
            pred_all.append(pred.detach().cpu().numpy())
            subject_ids_all.extend([str(x) for x in batch["subject_ids"]])

    y_true = np.concatenate(y_true_all, axis=0)
    logits = np.concatenate(logits_all, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    pred = np.concatenate(pred_all, axis=0)

    metrics = summarize_classification_metrics(
        y_true=y_true,
        y_pred=pred,
        probs=probs,
        logits=logits,
        num_classes=probs.shape[1],
    )

    mil_df = pd.concat(mil_attn_rows, axis=0, ignore_index=True) if len(mil_attn_rows) > 0 else pd.DataFrame()

    return {
        "loss": total_loss / max(n_batches, 1),
        "metrics": metrics,
        "y_true": y_true,
        "logits": logits,
        "probs": probs,
        "pred": pred,
        "subject_ids": subject_ids_all,
        "segment_view_attention_weights": np.concatenate(segment_view_attn_all, axis=0)
        if len(segment_view_attn_all) > 0
        else np.zeros((0, 0), dtype=np.float32),
        "segment_subject_ids": segment_subject_ids_all,
        "segment_ids_flat": segment_segment_ids_all,
        "view_names": view_names_ref or [],
        "mil_attention_df": mil_df,
    }


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------
def plot_subject_bank_attention_heatmap(subject_df: pd.DataFrame, save_path: str):
    import matplotlib.pyplot as plt

    weight_cols = [c for c in subject_df.columns if c.startswith("weight__")]
    view_names = [c.replace("weight__", "") for c in weight_cols]

    mat = subject_df[weight_cols].to_numpy(dtype=np.float32)

    plt.figure(figsize=(10, max(4, 0.25 * len(subject_df))))
    plt.imshow(mat, aspect="auto")
    plt.colorbar(label="Mean bank attention")
    plt.xticks(range(len(view_names)), view_names, rotation=45, ha="right")
    plt.yticks(range(len(subject_df)), subject_df["subject_id"].astype(str).tolist())
    plt.xlabel("Topology view")
    plt.ylabel("Subject")
    plt.title("Subject-level average topology attention")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def summarize_bank_attention_by_subject_with_std(df: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [c for c in df.columns if c.startswith("weight__")]
    if len(weight_cols) == 0:
        raise ValueError("No weight__ columns found.")

    mean_df = df.groupby("subject_id", as_index=False)[weight_cols].mean()
    std_df = df.groupby("subject_id", as_index=False)[weight_cols].std(ddof=0)

    out = mean_df.copy()
    for col in weight_cols:
        out[f"std__{col.replace('weight__', '')}"] = std_df[col].to_numpy(dtype=np.float32)

    clean_names = [c.replace("weight__", "") for c in weight_cols]
    weight_mat = mean_df[weight_cols].to_numpy(dtype=np.float32)
    out["top1_view_subject"] = [clean_names[int(np.argmax(row))] for row in weight_mat]
    return out

def summarize_bank_attention_by_subject(df: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [c for c in df.columns if c.startswith("weight__")]
    if len(weight_cols) == 0:
        raise ValueError("No weight__ columns found.")

    grouped = df.groupby("subject_id", as_index=False)[weight_cols].mean()

    top1_views = []
    clean_names = [c.replace("weight__", "") for c in weight_cols]
    weight_mat = grouped[weight_cols].to_numpy(dtype=np.float32)

    for row in weight_mat:
        top1_views.append(clean_names[int(np.argmax(row))])

    grouped["top1_view_subject"] = top1_views
    return grouped

def build_segment_attention_dataframe(epoch_out: Mapping[str, Any]) -> pd.DataFrame:
    weights = np.asarray(epoch_out["segment_view_attention_weights"], dtype=np.float32)
    view_names = list(epoch_out["view_names"])

    if weights.ndim != 2:
        raise ValueError(f"Expected segment_view_attention_weights to be 2D, got {weights.shape}.")
    if weights.shape[1] != len(view_names):
        raise ValueError(
            f"weights has {weights.shape[1]} columns but view_names has length {len(view_names)}."
        )

    df = pd.DataFrame(
        {
            "subject_id": list(epoch_out["segment_subject_ids"]),
            "segment_id": np.asarray(epoch_out["segment_ids_flat"], dtype=np.int64),
        }
    )
    for k, name in enumerate(view_names):
        df[f"weight__{name}"] = weights[:, k]
    if len(view_names) > 0 and len(df) > 0:
        df["top1_view"] = [view_names[i] for i in np.argmax(weights, axis=1)]
    else:
        df["top1_view"] = None
    return df


def build_subject_prediction_dataframe(epoch_out: Mapping[str, Any]) -> pd.DataFrame:
    probs = np.asarray(epoch_out["probs"], dtype=np.float32)
    logits = np.asarray(epoch_out["logits"], dtype=np.float32)

    df = pd.DataFrame(
        {
            "subject_id": list(epoch_out["subject_ids"]),
            "true_label": np.asarray(epoch_out["y_true"], dtype=np.int64),
            "pred_label": np.asarray(epoch_out["pred"], dtype=np.int64),
            "source_level": "subject",
        }
    )
    for c in range(probs.shape[1]):
        df[f"prob_{c}"] = probs[:, c]
    for c in range(logits.shape[1]):
        df[f"logit_{c}"] = logits[:, c]
    return df


def summarize_bank_attention(df: pd.DataFrame) -> pd.DataFrame:
    weight_cols = [c for c in df.columns if c.startswith("weight__")]
    rows = []
    top1_counts = df["top1_view"].value_counts(normalize=True).to_dict() if "top1_view" in df.columns else {}

    for col in weight_cols:
        view_name = col.replace("weight__", "")
        rows.append(
            {
                "view_name": view_name,
                "mean_weight": float(df[col].mean()),
                "std_weight": float(df[col].std(ddof=0)),
                "top1_frequency": float(top1_counts.get(view_name, 0.0)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_weight", ascending=False) if len(rows) > 0 else pd.DataFrame(
        columns=["view_name", "mean_weight", "std_weight", "top1_frequency"]
    )


def _save_bank_attention_artifacts(
    *,
    epoch_out: Mapping[str, Any],
    split_name: str,
    run_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    seg_attn_df = build_segment_attention_dataframe(epoch_out)
    seg_attn_path = os.path.join(run_dir, f"{split_name}_segment_attention.csv")
    seg_attn_df.to_csv(seg_attn_path, index=False)

    summary_df = summarize_bank_attention(seg_attn_df)
    summary_path = os.path.join(run_dir, f"{split_name}_bank_attention_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    mil_df = epoch_out.get("mil_attention_df", pd.DataFrame())
    if isinstance(mil_df, pd.DataFrame) and len(mil_df) > 0:
        mil_path = os.path.join(run_dir, f"{split_name}_mil_attention.csv")
        mil_df.to_csv(mil_path, index=False)

    return seg_attn_df, summary_df

def get_static_bank_weights(model: SubjectSegmentMVGNN) -> np.ndarray:
    bank_fusion = model.segment_encoder.graph_model.bank_fusion
    if bank_fusion is None:
        raise ValueError("Model has no bank_fusion module.")
    if getattr(bank_fusion, "fusion_mode", "").lower() != "static":
        raise ValueError("This helper is only for fusion_mode='static'.")
    logits = bank_fusion.candidate_logits.detach().cpu()
    weights = torch.softmax(logits / bank_fusion.temperature, dim=0)
    return weights.numpy()

def plot_static_bank_weights(model, view_names, save_path):
    import matplotlib.pyplot as plt

    weights = get_static_bank_weights(model)

    plt.figure(figsize=(8, 5))
    plt.barh(view_names, weights)
    plt.xlabel("Learned global bank weight")
    plt.title("Static graph-bank weights")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_static_bank_weight_history(df, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    for view_name, sub in df.groupby("view_name"):
        plt.plot(sub["epoch"], sub["weight"], label=view_name)
    plt.xlabel("Epoch")
    plt.ylabel("Global bank weight")
    plt.title("Static bank weights over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _plot_bank_attention_history(history_df: pd.DataFrame, run_dir: str, split_name: str = "val") -> None:
    import matplotlib.pyplot as plt

    if len(history_df) == 0:
        return

    plot_df = history_df[history_df["split"] == split_name].copy()
    if len(plot_df) == 0:
        return

    plt.figure(figsize=(10, 6))
    for view_name, sub in plot_df.groupby("view_name"):
        plt.plot(sub["epoch"], sub["mean_weight"], label=view_name)
    plt.xlabel("Epoch")
    plt.ylabel("Mean bank attention weight")
    plt.title(f"{split_name.capitalize()} bank attention over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{split_name}_bank_attention_over_epochs.png"), dpi=300)
    plt.close()


def _plot_bank_attention_ranking(summary_df: pd.DataFrame, run_dir: str, split_name: str = "val") -> None:
    import matplotlib.pyplot as plt

    if len(summary_df) == 0:
        return

    plot_df = summary_df.sort_values("mean_weight", ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(plot_df["view_name"], plot_df["mean_weight"])
    plt.xlabel("Mean attention weight")
    plt.title(f"{split_name.capitalize()} bank attention ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"{split_name}_bank_attention_ranking.png"), dpi=300)
    plt.close()


# ---------------------------------------------------------------------
# Split preparation
# ---------------------------------------------------------------------


def prepare_segment_split_graphs(spec: "CAUEEGExperimentSpec") -> dict[str, list[Data]]:
    cm = _cm()

    _, train_rows, val_rows, test_rows = cm.load_caueeg_task_splits(spec.dataset_path, spec.task)
    split_to_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
    split_to_resolved = {
        split: cm.resolve_h5_subject_ids_for_split(spec.h5_path, rows, split)
        for split, rows in split_to_rows.items()
    }

    all_subject_ids: list[str] = []
    for rows in split_to_resolved.values():
        all_subject_ids.extend([sid for sid, _, _ in rows])

    entries = cm.load_h5_entries(
        spec.h5_path,
        all_subject_ids,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
    )

    out: dict[str, list[Data]] = {"train": [], "val": [], "test": []}
    for split, rows in split_to_resolved.items():
        split_entries = {sid: entries[sid] for sid, _, _ in rows}
        out[split] = build_segment_graphs_for_entries(
            split_entries,
            feature_families=spec.feature_families,
            connectivity_metrics=spec.connectivity_metrics_to_load,
            graph_bank_specs=spec.topology.graph_bank_specs,
        )
    return out


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------


def run_segment_mvgnn_experiment(spec: "CAUEEGExperimentSpec") -> dict[str, Any]:
    cm = _cm()

    if str(spec.level.graph_level).lower() != "segment":
        raise ValueError("segment_mvgnn requires spec.level.graph_level='segment'.")
    if str(spec.aggregation.strategy).lower() == "none":
        raise ValueError("segment_mvgnn is a subject-level bag model; use a bag-level aggregation strategy.")

    set_seed(spec.train.seed)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")

    split_graphs = prepare_segment_split_graphs(spec)
    if len(split_graphs["train"]) == 0:
        raise ValueError("Training split produced zero segment graphs.")

    base_k = getattr(spec.aggregation, "base_k", None)
    max_k_per_subject = getattr(spec.aggregation, "max_k_per_subject", 300)
    eval_k_per_subject = getattr(spec.aggregation, "eval_k_per_subject", None)


    train_dataset = LabelAwareSubjectBagDataset(
        split_graphs["train"],
        # train_graphs,
        train=True,
        base_k=base_k,
        max_k_per_subject=max_k_per_subject,
        seed=spec.train.seed,
        return_segment_ids=True,
    )
    val_dataset = LabelAwareSubjectBagDataset(
        split_graphs["val"],
        # val_graphs,
        train=False,
        eval_k_per_subject=None,
        seed=spec.train.seed,
    )
    test_dataset = LabelAwareSubjectBagDataset(
        split_graphs["test"],
        # test_graphs,
        train=False,
        eval_k_per_subject=None,
        seed=spec.train.seed,
    )


    # train_dataset = GraphBankSubjectBagDataset(
    #     split_graphs["train"],
    #     train=True,
    #     base_k=base_k,
    #     max_k_per_subject=max_k_per_subject,
    #     seed=spec.train.seed,
    #     return_segment_ids=True,
    # )
    # val_dataset = GraphBankSubjectBagDataset(
    #     split_graphs["val"],
    #     train=False,
    #     eval_k_per_subject=eval_k_per_subject,
    #     seed=spec.train.seed,
    #     return_segment_ids=True,
    # )
    # test_dataset = GraphBankSubjectBagDataset(
    #     split_graphs["test"],
    #     train=False,
    #     eval_k_per_subject=eval_k_per_subject,
    #     seed=spec.train.seed,
    #     return_segment_ids=True,
    # )

    common_loader_kwargs = dict(
        collate_fn=collate_subject_bags_graph_bank,
        num_workers=int(spec.train.num_workers),
        pin_memory=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(spec.train.batch_size),
        shuffle=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(spec.train.batch_size),
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(spec.train.batch_size),
        shuffle=False,
        **common_loader_kwargs,
    )


    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=int(spec.train.batch_size),
    #     shuffle=True,
    #     collate_fn=collate_subject_bags_graph_bank,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=int(spec.train.batch_size),
    #     shuffle=False,
    #     collate_fn=collate_subject_bags,
    #     num_workers=0,
    #     pin_memory=True,
    # )
    # test_loader = DataLoader(
    #     test_dataset,
    #     batch_size=int(spec.train.batch_size),
    #     shuffle=False,
    #     collate_fn=collate_subject_bags,
    #     num_workers=0,
    #     pin_memory=True,
    # )



    sample_graph = split_graphs["train"][0]
    num_classes = len(sorted({int(g.y.view(-1)[0].item()) for split in split_graphs.values() for g in split}))
    class_names = spec.class_names or cm.infer_class_names(spec.task, num_classes)

    model = SubjectSegmentMVGNN(
        num_nodes=int(sample_graph.x.shape[0]),
        num_node_features=int(sample_graph.x.shape[1]),
        num_views=int(sample_graph.adj_bank.shape[0]),
        num_classes=int(num_classes),
        graph_backbone=str(spec.model.backbone),
        hidden_dim=int(spec.model.hidden_dim),
        graph_emb_dim=int(spec.model.emb_dim),
        num_layers=int(spec.model.num_layers),
        dropout=float(spec.model.dropout),
        gat_heads=int(spec.model.gat_heads),
        graph_readout_type=str(spec.model.graph_readout),
        graph_bank_fusion_mode=str(spec.model.graph_bank_fusion_mode),
        topology_rule=str(spec.topology.fuse_topology_rule),
        vote_threshold=float(getattr(spec.topology, "fuse_vote_threshold", 0.45)),
        fusion_temperature=float(getattr(spec.model, "fusion_temperature", 0.7)),
        fusion_hidden_dim=getattr(spec.model, "fusion_hidden_dim", None),
        subject_aggregation=str(spec.aggregation.strategy),
        subject_attn_dim=int(spec.aggregation.attn_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    stopper = cm.EarlyStopper(spec.train.monitor, spec.train.monitor_mode, spec.train.patience)

    run_dir = ensure_dir(
        os.path.join(
            spec.output_root,
            make_run_name(spec.name, spec.model.family, spec.aggregation.strategy, timestamp=True),
        )
    )
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")

    history_train: list[dict[str, Any]] = []
    history_val: list[dict[str, Any]] = []
    bank_attention_history_rows: list[dict[str, Any]] = []

    for epoch in range(1, spec.train.epochs + 1):
        train_out = collect_epoch_outputs_segment(model, train_loader, device=device, optimizer=optimizer)
        val_out = collect_epoch_outputs_segment(model, val_loader, device=device, optimizer=None)

        train_metrics = train_out["metrics"]
        val_metrics = val_out["metrics"]

        history_train.append(
            {
                "epoch": epoch,
                "loss": float(train_out["loss"]),
                "accuracy": float(train_metrics["accuracy"]),
                "balanced_accuracy": float(train_metrics["balanced_accuracy"]),
                "macro_f1": float(train_metrics["macro_f1"]),
            }
        )
        history_val.append(
            {
                "epoch": epoch,
                "loss": float(val_out["loss"]),
                "accuracy": float(val_metrics["accuracy"]),
                "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
                "macro_f1": float(val_metrics["macro_f1"]),
            }
        )
        val_attn_df = pd.DataFrame()
        val_attn_summary = pd.DataFrame()

        if str(spec.model.graph_bank_fusion_mode).lower() == "static":
            append_static_bank_weight_history(
                bank_attention_history_rows,
                model,
                getattr(sample_graph, "view_names", []),
                epoch,
                split="val",
            )
        else:
            val_attn_df = build_segment_attention_dataframe(val_out)
            val_attn_summary = summarize_bank_attention(val_attn_df)
            if len(val_attn_summary) > 0:
                for _, row in val_attn_summary.iterrows():
                    bank_attention_history_rows.append(
                        {
                            "epoch": epoch,
                            "split": "val",
                            "view_name": str(row["view_name"]),
                            "mean_weight": float(row["mean_weight"]),
                            "std_weight": float(row["std_weight"]),
                            "top1_frequency": float(row["top1_frequency"]),
                        }
                    )

        monitor_value = float(val_out["loss"]) if spec.train.monitor == "loss" else float(val_metrics[spec.train.monitor])

        should_stop = stopper.step(monitor_value, epoch)
        if stopper.best_epoch == epoch:
            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "monitor": spec.train.monitor,
                "monitor_value": monitor_value,
                "val_loss": float(val_out["loss"]),
                "val_metrics": copy.deepcopy(val_metrics),
                "spec": cm.asdict(spec),
            }
            torch.save(best_state, best_ckpt_path)

            if str(spec.model.graph_bank_fusion_mode).lower() != "static":
                if len(val_attn_df) > 0:
                    val_attn_df.to_csv(os.path.join(run_dir, "val_segment_attention_best.csv"), index=False)
                if len(val_attn_summary) > 0:
                    val_attn_summary.to_csv(os.path.join(run_dir, "val_bank_attention_summary_best.csv"), index=False)
            
            mil_df = val_out.get("mil_attention_df", pd.DataFrame())
            if isinstance(mil_df, pd.DataFrame) and len(mil_df) > 0:
                mil_df.to_csv(os.path.join(run_dir, "val_mil_attention_best.csv"), index=False)

        if should_stop:
            break

    best_loaded = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_loaded["model_state_dict"])

    train_final = collect_epoch_outputs_segment(model, train_loader, device=device, optimizer=None)
    val_final = collect_epoch_outputs_segment(model, val_loader, device=device, optimizer=None)
    test_final = collect_epoch_outputs_segment(model, test_loader, device=device, optimizer=None)

    train_df = build_subject_prediction_dataframe(train_final)
    val_df = build_subject_prediction_dataframe(val_final)
    test_df = build_subject_prediction_dataframe(test_final)

    train_df.to_csv(os.path.join(run_dir, "train_subject_predictions.csv"), index=False)
    val_df.to_csv(os.path.join(run_dir, "val_subject_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(run_dir, "test_subject_predictions.csv"), index=False)
    pd.DataFrame(history_train).to_csv(os.path.join(run_dir, "history_train.csv"), index=False)
    pd.DataFrame(history_val).to_csv(os.path.join(run_dir, "history_val.csv"), index=False)

    train_seg_attn_df, train_seg_attn_summary = _save_bank_attention_artifacts(
        epoch_out=train_final,
        split_name="train",
        run_dir=run_dir,
    )
    val_seg_attn_df, val_seg_attn_summary = _save_bank_attention_artifacts(
        epoch_out=val_final,
        split_name="val",
        run_dir=run_dir,
    )
    test_seg_attn_df, test_seg_attn_summary = _save_bank_attention_artifacts(
        epoch_out=test_final,
        split_name="test",
        run_dir=run_dir,
    )

    bank_history_df = pd.DataFrame(bank_attention_history_rows)
    bank_history_df.to_csv(os.path.join(run_dir, "bank_attention_history.csv"), index=False)
    


    is_static_fusion = str(spec.model.graph_bank_fusion_mode).lower() == "static"

    if is_static_fusion:

        static_hist_df = pd.DataFrame(bank_attention_history_rows)
        static_hist_df.to_csv(os.path.join(run_dir, "bank_attention_history.csv"), index=False)

        plot_static_bank_weights(
            model,
            getattr(sample_graph, "view_names", []),
            os.path.join(run_dir, "static_bank_weights.png"),
        )
        plot_static_bank_weight_history(
            bank_history_df,
            os.path.join(run_dir, "static_bank_weight_history.png"),
        )

    else:
        train_subject_attn_df = summarize_bank_attention_by_subject(train_seg_attn_df)
        val_subject_attn_df = summarize_bank_attention_by_subject(val_seg_attn_df)
        test_subject_attn_df = summarize_bank_attention_by_subject(test_seg_attn_df)

        train_subject_attn_df.to_csv(os.path.join(run_dir, "train_subject_bank_attention.csv"), index=False)
        val_subject_attn_df.to_csv(os.path.join(run_dir, "val_subject_bank_attention.csv"), index=False)
        test_subject_attn_df.to_csv(os.path.join(run_dir, "test_subject_bank_attention.csv"), index=False)

        plot_subject_bank_attention_heatmap(
            val_subject_attn_df,
            os.path.join(run_dir, "val_subject_bank_attention_heatmap.png"),
        )
        plot_subject_bank_attention_heatmap(
            test_subject_attn_df,
            os.path.join(run_dir, "test_subject_bank_attention_heatmap.png"),
        )
        # Final summary copies for convenience / plotting.
        val_seg_attn_summary.to_csv(os.path.join(run_dir, "val_bank_attention_summary_final.csv"), index=False)
        test_seg_attn_summary.to_csv(os.path.join(run_dir, "test_bank_attention_summary_final.csv"), index=False)

        _plot_bank_attention_history(bank_history_df, run_dir, split_name="val")
        _plot_bank_attention_ranking(val_seg_attn_summary, run_dir, split_name="val")
        _plot_bank_attention_ranking(test_seg_attn_summary, run_dir, split_name="test")

    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": run_dir,
                "best_epoch": best_loaded["epoch"],
                "monitor": spec.train.monitor,
                "monitor_value": best_loaded["monitor_value"],
                "train_metrics": train_final["metrics"],
                "val_metrics": val_final["metrics"],
                "test_metrics": test_final["metrics"],
                "class_names": list(class_names),
                "view_names": list(getattr(sample_graph, "view_names", [])),
                "model_family": str(spec.model.family),
                "base_k": base_k,
                "max_k_per_subject": max_k_per_subject,
                "spec": cm.asdict(spec),
            },
            f,
            indent=2,
        )

    return {
        "run_dir": run_dir,
        "best_epoch": best_loaded["epoch"],
        "best_checkpoint_path": best_ckpt_path,
        "train_metrics": train_final["metrics"],
        "val_metrics": val_final["metrics"],
        "test_metrics": test_final["metrics"],
        "class_names": class_names,
        "view_names": tuple(getattr(sample_graph, "view_names", [])),
        "train_segment_attention": train_seg_attn_df,
        "val_segment_attention": val_seg_attn_df,
        "test_segment_attention": test_seg_attn_df,
        "spec": spec,
    }

def append_static_bank_weight_history(history_rows, model, view_names, epoch, split="val"):
    weights = get_static_bank_weights(model)
    for name, w in zip(view_names, weights):
        history_rows.append({
            "epoch": int(epoch),
            "split": str(split),
            "view_name": str(name),
            "weight": float(w),
        })

# ---------------------------------------------------------------------
# Bank-definition helper
# ---------------------------------------------------------------------


def build_segment_graph_bank_specs(fixed_edge_pairs: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {
            "name": "structural_local_binary",
            "topology_mode": "fixed",
            "edge_weight_mode": "binary",
            "edge_pairs": list(fixed_edge_pairs),
        },
        {
            "name": "coherence_theta_mst",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": "theta",
            "topology_kwargs": {"mode": "mst"},
        },
        {
            "name": "coherence_alpha_topk4",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": "alpha",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        # {
        #     "name": "coherence_beta_topk4",
        #     "topology_mode": "connectivity",
        #     "edge_weight_mode": "connectivity",
        #     "connectivity_metric": "coherence",
        #     "band": "beta",
        #     "topology_kwargs": {"mode": "topk", "topk": 4},
        # },
        {
            "name": "wpli_alpha_topk4",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "wpli",
            "band": "alpha",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        # {
        #     "name": "feature_rbp_cosine_topk4",
        #     "topology_mode": "feature_induced",
        #     "edge_weight_mode": "topology_weight",
        #     "similarity": "cosine",
        #     "topology_kwargs": {"mode": "topk", "topk": 4},
        #     "feature_family": "relative_band_power",
        # },
        # {
        #     "name": "feature_hjorth_cosine_topk4",
        #     "topology_mode": "feature_induced",
        #     "edge_weight_mode": "topology_weight",
        #     "similarity": "cosine",
        #     "topology_kwargs": {"mode": "topk", "topk": 4},
        #     "feature_family": "hjorth",
        # },
        {
            "name": "structural_local_coherence_alpha",
            "topology_mode": "fixed",
            "edge_weight_mode": "connectivity",
            "edge_weight_metric": "coherence",
            "edge_weight_band": "alpha",
            "edge_pairs": list(fixed_edge_pairs),
        },
        {
            "name": "structural_local_wpli_alpha",
            "topology_mode": "fixed",
            "edge_weight_mode": "connectivity",
            "edge_weight_metric": "wpli",
            "edge_weight_band": "alpha",
            "edge_pairs": list(fixed_edge_pairs),
        },
    ]



if __name__ == "__main__":
    from caueeg_main_new import (
        CAUEEGExperimentSpec,
        LevelConfig,
        TopologyConfig,
        ModelConfig,
        AggregationConfig,
        TrainConfig,
        default_fixed_edge_pairs_19,
    )
    DATASET_PATH = "/home/anphan/Downloads/caueeg-dataset/"
    H5_PATH = "/home/anphan/Documents/caueeg_randomcrop_master_dementia_seed42.h5"
    graph_bank_fusion_mode = "summary_gated"
    # graph_bank_fusion_mode = "static"

    fixed_edges = default_fixed_edge_pairs_19()
    graph_bank_specs = build_segment_graph_bank_specs(fixed_edges)

    spec = CAUEEGExperimentSpec(
        name=f"segment_mvgnn_gatv2_{graph_bank_fusion_mode}",
        task="dementia",
        dataset_path=DATASET_PATH,
        h5_path=H5_PATH,
        feature_families=("relative_band_power", "statistical"),
        connectivity_metrics_to_load=("coherence", "wpli"),
        level=LevelConfig(graph_level="segment"),
        topology=TopologyConfig(
            strategy="fused_bank",
            graph_bank_specs=graph_bank_specs,
            fuse_topology_rule="vote",
            fuse_vote_threshold=0.45,
        ),
        model=ModelConfig(
            family="segment_mvgnn",
            backbone="gatv2",
            hidden_dim=64,
            emb_dim=128,
            num_layers=2,
            dropout=0.2,
            gat_heads=4,
            graph_readout="gated_attention",
            graph_bank_fusion_mode=graph_bank_fusion_mode,
            fusion_temperature=0.7,
            fusion_hidden_dim=64,
        ),
        aggregation=AggregationConfig(
            strategy="gated_attention_mil",
            attn_dim=128,
            base_k=8,                  # number of segments sampled per subject during training
            max_k_per_subject=64,      # extra cap for training
            eval_k_per_subject=None, 
        ),
        train=TrainConfig(
            batch_size=8,
            lr=1e-3,
            weight_decay=5e-4,
            epochs=200,
            patience=100,
            monitor="balanced_accuracy",
            monitor_mode="max",
            seed=42,
            num_workers=0,
        ),
        output_root="/home/anphan/Documents/CAUEEG/results_segment_mvgnn",
    )

    out = run_segment_mvgnn_experiment(spec)
    print("Run dir:", out["run_dir"])
    print("Test metrics:", out["test_metrics"])