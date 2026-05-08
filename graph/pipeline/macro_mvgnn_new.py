from __future__ import annotations

"""
macro_mvgnn.py

Macro-level multi-view GNN for CAUEEG using H5-first inputs.

Main ideas
----------
1) Segment -> macro node aggregation with learnable attention.
2) A richer macro graph bank that is not connectivity-only by default.
3) Macro -> subject aggregation with existing MIL utilities.
4) A dedicated runner that can be dispatched from caueeg_main.py.

The default graph bank mixes several relational views:
- connectivity-derived views (metric x band)
- fixed structural-local prior
- fixed structural-local prior with connectivity weights
- simple hemisphere prior
- simple region-clique prior
- feature-induced graphs built separately from each feature family

This keeps the bank closer to a true multi-view graph bank rather than a
pure bank of connectivity matrices.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

import copy
import json
import os
import re
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
from graph_construction import GraphBankCandidate, build_graph_bank
from metrics import summarize_classification_metrics
from models_mil import GatedAttentionMILPool, SubjectFusionHead, aggregate_subject_predictions
from utils import ensure_dir, get_device, make_run_name, set_seed
from visualize import plot_confusion_matrix, plot_training_curves

if TYPE_CHECKING:  # pragma: no cover
    from caueeg_main import CAUEEGExperimentSpec, H5SubjectEntry


DEFAULT_BANDS: tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")
DEFAULT_BANDWISE_METRICS: tuple[str, ...] = ("coherence", "pli", "wpli")
DEFAULT_REGION_TO_CHANNELS: dict[str, tuple[str, ...]] = {
    "frontal": ("Fp1", "Fp2", "F3", "F4", "F7", "F8", "FZ"),
    "central": ("C3", "C4", "CZ"),
    "parietal": ("P3", "P4", "PZ"),
    "temporal": ("T3", "T4", "T5", "T6"),
    "occipital": ("O1", "O2"),
}


def _cm():
    import caueeg_main_new as cm

    return cm


# -----------------------------------------------------------------------------
# Dataclasses for macro bags
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MacroInstance:
    subject_id: str
    label: int
    macro_id: int
    node_feature_seq: np.ndarray   # [K, N, F_total]
    adj_bank: np.ndarray           # [V, N, N]
    topology_bank: np.ndarray      # [V, N, N]
    view_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubjectMacroBag:
    subject_id: str
    label: int
    macros: list[MacroInstance]
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Small topology helpers
# -----------------------------------------------------------------------------


def _normalize_channel_name(name: str) -> str:
    return str(name).strip().upper()


def _default_fixed_edge_pairs(channel_names: Sequence[str]) -> list[tuple[int, int]]:
    cm = _cm()
    if len(channel_names) == 19:
        return list(cm.default_fixed_edge_pairs_19())
    return []


def _infer_hemi_partner(ch_name: str) -> Optional[str]:
    tokens = str(ch_name).split("-")
    out: list[str] = []
    for tok in tokens:
        m = re.match(r"^([A-Za-z]+)(\d+)$", tok)
        if m is None:
            return None
        base, num = m.group(1), int(m.group(2))
        out.append(f"{base}{num + 1}" if num % 2 == 1 else f"{base}{num - 1}")
    return "-".join(out)


def _build_hemisphere_adjacency(channel_names: Sequence[str]) -> np.ndarray:
    names = [_normalize_channel_name(x) for x in channel_names]
    name_to_idx = {name: i for i, name in enumerate(names)}
    adj = np.zeros((len(names), len(names)), dtype=np.float32)

    for name, i in name_to_idx.items():
        partner = _infer_hemi_partner(name)
        if partner is None or partner not in name_to_idx:
            continue
        j = name_to_idx[partner]
        if i == j:
            continue
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


def _build_region_clique_adjacency(
    channel_names: Sequence[str],
    region_to_channels: Mapping[str, Sequence[str]] = DEFAULT_REGION_TO_CHANNELS,
) -> np.ndarray:
    names = [_normalize_channel_name(x) for x in channel_names]
    name_to_idx = {name: i for i, name in enumerate(names)}
    adj = np.zeros((len(names), len(names)), dtype=np.float32)

    for _, region_channels in region_to_channels.items():
        idxs = [name_to_idx[_normalize_channel_name(ch)] for ch in region_channels if _normalize_channel_name(ch) in name_to_idx]
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                u, v = idxs[i], idxs[j]
                adj[u, v] = 1.0
                adj[v, u] = 1.0

    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


# -----------------------------------------------------------------------------
# Graph-bank builders
# -----------------------------------------------------------------------------


def make_metric_band_graph_bank_specs(
    *,
    bandwise_metrics: Sequence[str] = DEFAULT_BANDWISE_METRICS,
    bands: Sequence[int | str] = DEFAULT_BANDS,
    topology_mode: str = "connectivity",
    edge_weight_mode: str = "connectivity",
    topology_kwargs: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    topo_kwargs = dict(topology_kwargs or {"mode": "topk", "topk": 6})
    specs: list[dict[str, Any]] = []
    for metric in bandwise_metrics:
        for band in bands:
            specs.append(
                {
                    "name": f"{metric}_{band}",
                    "topology_mode": topology_mode,
                    "edge_weight_mode": edge_weight_mode,
                    "connectivity_metric": str(metric),
                    "band": band,
                    "topology_kwargs": dict(topo_kwargs),
                }
            )
    return specs


def build_rich_macro_graph_bank(
    *,
    channel_names: Sequence[str],
    node_feature_mean_all: np.ndarray,
    node_feature_mean_by_family: Mapping[str, np.ndarray],
    connectivity_sources: Mapping[str, np.ndarray],
    band_names_map: Mapping[str, Optional[Sequence[str]]],
    feature_families: Sequence[str],
    fixed_edge_pairs: Optional[Sequence[tuple[int, int]]] = None,
    bands: Sequence[int | str] = DEFAULT_BANDS,
    bandwise_metrics: Sequence[str] = DEFAULT_BANDWISE_METRICS,
    connectivity_topk: int = 4,
    feature_similarity: str = "cosine",
    feature_topk: int = 4,
    include_structural_binary: bool = True,
    include_structural_weighted: bool = True,
    include_hemisphere_binary: bool = True,
    include_region_binary: bool = True,
    include_feature_induced: bool = True,
    extra_candidate_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[GraphBankCandidate]:
    """
    Build a multi-view graph bank that is not connectivity-only.

    Strategy
    --------
    1) connectivity views: one candidate per metric-band pair
    2) structural prior views: binary and connectivity-weighted
    3) hemisphere / region binary priors
    4) feature-induced graphs built independently per feature family
    5) optional extra candidate specs appended at the end
    """
    payload = {
        metric: (np.asarray(values, dtype=np.float32), band_names_map.get(metric, None))
        for metric, values in connectivity_sources.items()
    }
    bank: list[GraphBankCandidate] = []

    available_bandwise_metrics = [
        metric for metric in bandwise_metrics
        if metric in connectivity_sources and np.asarray(connectivity_sources[metric]).ndim == 3
    ]

    if fixed_edge_pairs is None:
        fixed_edge_pairs = _default_fixed_edge_pairs(channel_names)

    # 1) raw connectivity views
    connectivity_specs = make_metric_band_graph_bank_specs(
        bandwise_metrics=available_bandwise_metrics,
        bands=bands,
        topology_mode="connectivity",
        edge_weight_mode="connectivity",
        topology_kwargs={"mode": "topk", "topk": int(connectivity_topk)},
    )
    if connectivity_specs:
        bank.extend(
            build_graph_bank(
                node_features=node_feature_mean_all,
                connectivity_sources=payload,
                candidate_specs=connectivity_specs,
                fixed_topology=None,
            )
        )

    # 2) structural-local prior views
    if fixed_edge_pairs:
        structural_specs: list[dict[str, Any]] = []
        if include_structural_binary:
            structural_specs.append(
                {
                    "name": "structural_local_binary",
                    "topology_mode": "fixed",
                    "edge_weight_mode": "binary",
                    "edge_pairs": list(fixed_edge_pairs),
                }
            )
        if include_structural_weighted:
            for metric in available_bandwise_metrics:
                for band in bands:
                    structural_specs.append(
                        {
                            "name": f"structural_local_{metric}_{band}",
                            "topology_mode": "fixed",
                            "edge_weight_mode": "connectivity",
                            "edge_weight_metric": str(metric),
                            "edge_weight_band": band,
                            "edge_pairs": list(fixed_edge_pairs),
                        }
                    )
        if structural_specs:
            bank.extend(
                build_graph_bank(
                    node_features=node_feature_mean_all,
                    connectivity_sources=payload,
                    candidate_specs=structural_specs,
                    fixed_topology=None,
                )
            )

    # 3) fixed hemisphere and region priors
    fixed_specs: list[dict[str, Any]] = []
    if include_hemisphere_binary:
        fixed_specs.append(
            {
                "name": "structural_hemisphere_binary",
                "topology_mode": "fixed",
                "edge_weight_mode": "binary",
                "adjacency": _build_hemisphere_adjacency(channel_names),
            }
        )
    if include_region_binary:
        fixed_specs.append(
            {
                "name": "region_clique_binary",
                "topology_mode": "fixed",
                "edge_weight_mode": "binary",
                "adjacency": _build_region_clique_adjacency(channel_names),
            }
        )
    if fixed_specs:
        bank.extend(
            build_graph_bank(
                node_features=node_feature_mean_all,
                connectivity_sources=payload,
                candidate_specs=fixed_specs,
                fixed_topology=None,
            )
        )

    # 4) feature-induced candidates per family, not just on the concatenated features
    if include_feature_induced:
        for fam in feature_families:
            if fam not in node_feature_mean_by_family:
                continue
            fam_x = np.asarray(node_feature_mean_by_family[fam], dtype=np.float32)
            fam_specs = [
                {
                    "name": f"feature_{fam}_{feature_similarity}_topk{feature_topk}",
                    "topology_mode": "feature_induced",
                    "edge_weight_mode": "topology_weight",
                    "similarity": str(feature_similarity),
                    "topology_kwargs": {"mode": "topk", "topk": int(feature_topk)},
                }
            ]
            bank.extend(
                build_graph_bank(
                    node_features=fam_x,
                    connectivity_sources=payload,
                    candidate_specs=fam_specs,
                    fixed_topology=None,
                )
            )

    # 5) optional extra candidates from spec
    if extra_candidate_specs:
        bank.extend(
            build_graph_bank(
                node_features=node_feature_mean_all,
                connectivity_sources=payload,
                candidate_specs=list(extra_candidate_specs),
                fixed_topology=None,
            )
        )

    if len(bank) == 0:
        raise ValueError("Graph bank is empty. Check connectivity metrics, bands, and feature families.")

    return bank


# -----------------------------------------------------------------------------
# H5 -> macro bag builders
# -----------------------------------------------------------------------------


def _concat_feature_families_no_reduce(
    entry: "H5SubjectEntry",
    window_indices: np.ndarray,
    feature_families: Sequence[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    parts: list[np.ndarray] = []
    by_family: dict[str, np.ndarray] = {}
    for fam in feature_families:
        x = np.asarray(entry.features[fam], dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Feature family {fam!r} expected [W,N,F], got {x.shape}.")
        fam_x = x[window_indices].astype(np.float32)
        parts.append(fam_x)
        by_family[fam] = fam_x
    return np.concatenate(parts, axis=-1).astype(np.float32), by_family


def build_macro_instance_from_h5_entry(
    entry: "H5SubjectEntry",
    *,
    macro_id: int,
    window_indices: np.ndarray,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    connectivity_reduce: str = "mean",
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> MacroInstance:
    """
    Build one macro instance from one subject H5 entry.

    Node features remain segment-level [K,N,F_total] for learned temporal pooling.
    The graph bank is aggregated at macro level across the same K windows.
    """
    cm = _cm()

    node_feature_seq, node_feature_seq_by_family = _concat_feature_families_no_reduce(
        entry,
        window_indices,
        feature_families,
    )

    connectivity_sources, band_names_map = cm.aggregate_connectivity_sources(
        entry,
        window_indices,
        connectivity_metrics=connectivity_metrics,
        reduce_mode=connectivity_reduce,
    )

    node_feature_mean_by_family = {
        fam: cm.reduce_array(seq, "mean", axis=0).astype(np.float32)
        for fam, seq in node_feature_seq_by_family.items()
    }
    node_feature_mean_all = cm.reduce_array(node_feature_seq, "mean", axis=0).astype(np.float32)

    if graph_bank_specs is None:
        bank = build_rich_macro_graph_bank(
            channel_names=entry.channel_names,
            node_feature_mean_all=node_feature_mean_all,
            node_feature_mean_by_family=node_feature_mean_by_family,
            connectivity_sources=connectivity_sources,
            band_names_map=band_names_map,
            feature_families=feature_families,
        )
    else:
        bank = build_graph_bank(
            node_features=node_feature_mean_all,
            connectivity_sources={
                metric: (np.asarray(mat, dtype=np.float32), band_names_map.get(metric, None))
                for metric, mat in connectivity_sources.items()
            },
            candidate_specs=graph_bank_specs,
            fixed_topology=None,
        )

    adj_bank = np.stack([cand.adjacency for cand in bank], axis=0).astype(np.float32)
    topology_bank = np.stack([cand.topology for cand in bank], axis=0).astype(np.float32)
    view_names = [cand.name for cand in bank]

    start_sample = int(np.min(entry.start_sample[window_indices]))
    end_sample = int(np.max(entry.end_sample[window_indices]))

    return MacroInstance(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        macro_id=int(macro_id),
        node_feature_seq=node_feature_seq,
        adj_bank=adj_bank,
        topology_bank=topology_bank,
        view_names=view_names,
        metadata={
            "window_indices": window_indices.tolist(),
            "start_sample": start_sample,
            "end_sample": end_sample,
        },
    )


def build_subject_macro_bag(
    entry: "H5SubjectEntry",
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    macro_duration_sec: float = 60.0,
    sfreq: float = 200.0,
    connectivity_reduce: str = "mean",
) -> SubjectMacroBag:
    cm = _cm()
    groups = cm.build_macro_groups(
        entry.start_sample,
        sfreq=float(sfreq),
        macro_duration_sec=float(macro_duration_sec),
    )

    macros: list[MacroInstance] = []
    for macro_id, window_indices in sorted(groups.items(), key=lambda kv: int(kv[0])):
        macros.append(
            build_macro_instance_from_h5_entry(
                entry,
                macro_id=int(macro_id),
                window_indices=np.asarray(window_indices, dtype=np.int64),
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                graph_bank_specs=graph_bank_specs,
                connectivity_reduce=connectivity_reduce,
            )
        )
    # print(f"Subject {str(entry.subject_id)}| label={int(entry.label)} | num_macros = {len(macros)}")
    # for macro in macros:
    #     print(
    #         f"  macro_id={macro.macro_id} | num_segments={macro.node_feature_seq.shape[0]}"
    #     )
    return SubjectMacroBag(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        macros=macros,
        metadata={"num_macros": len(macros)},
    )


def build_subject_macro_bags(
    entries: Mapping[str, "H5SubjectEntry"],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
    macro_duration_sec: float = 60.0,
    sfreq: float = 200.0,
    connectivity_reduce: str = "mean",
) -> list[SubjectMacroBag]:
    bags: list[SubjectMacroBag] = []
    for sid in sorted(entries.keys()):
        bags.append(
            build_subject_macro_bag(
                entries[sid],
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                graph_bank_specs=graph_bank_specs,
                macro_duration_sec=macro_duration_sec,
                sfreq=sfreq,
                connectivity_reduce=connectivity_reduce,
            )
        )
    return bags


# -----------------------------------------------------------------------------
# Dataset / collate
# -----------------------------------------------------------------------------


class SubjectMacroBagDataset(Dataset):
    def __init__(self, bags: Sequence[SubjectMacroBag]):
        self.bags = list(bags)
        if len(self.bags) == 0:
            raise ValueError("bags must not be empty.")

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> SubjectMacroBag:
        return self.bags[idx]


class SubjectMacroBagCollate:
    def __call__(self, batch: Sequence[SubjectMacroBag]) -> dict[str, Any]:
        batch = list(batch)
        if len(batch) == 0:
            raise ValueError("Empty batch.")

        bsz = len(batch)
        max_macros = max(len(item.macros) for item in batch)
        max_segments = max(macro.node_feature_seq.shape[0] for item in batch for macro in item.macros)

        first_macro = batch[0].macros[0]
        num_nodes = int(first_macro.node_feature_seq.shape[1])
        num_node_features = int(first_macro.node_feature_seq.shape[2])
        num_views = int(first_macro.adj_bank.shape[0])

        node_feature_bag = torch.zeros(
            (bsz, max_macros, max_segments, num_nodes, num_node_features),
            dtype=torch.float32,
        )
        segment_mask = torch.zeros((bsz, max_macros, max_segments), dtype=torch.bool)
        adj_bank_bag = torch.zeros((bsz, max_macros, num_views, num_nodes, num_nodes), dtype=torch.float32)
        topology_bank_bag = torch.zeros((bsz, max_macros, num_views, num_nodes, num_nodes), dtype=torch.float32)
        macro_mask = torch.zeros((bsz, max_macros), dtype=torch.bool)
        macro_ids = torch.full((bsz, max_macros), fill_value=-1, dtype=torch.long)

        labels = torch.tensor([int(item.label) for item in batch], dtype=torch.long)
        subject_ids = [str(item.subject_id) for item in batch]
        view_names = list(first_macro.view_names)

        for b_idx, bag in enumerate(batch):
            for m_idx, macro in enumerate(bag.macros):
                seq = torch.as_tensor(macro.node_feature_seq, dtype=torch.float32)
                k = int(seq.shape[0])
                node_feature_bag[b_idx, m_idx, :k] = seq
                segment_mask[b_idx, m_idx, :k] = True
                adj_bank_bag[b_idx, m_idx] = torch.as_tensor(macro.adj_bank, dtype=torch.float32)
                topology_bank_bag[b_idx, m_idx] = torch.as_tensor(macro.topology_bank, dtype=torch.float32)
                macro_mask[b_idx, m_idx] = True
                macro_ids[b_idx, m_idx] = int(macro.macro_id)

        return {
            "node_feature_bag": node_feature_bag,
            "segment_mask": segment_mask,
            "adj_bank_bag": adj_bank_bag,
            "topology_bank_bag": topology_bank_bag,
            "macro_mask": macro_mask,
            "macro_ids": macro_ids,
            "labels": labels,
            "subject_ids": subject_ids,
            "view_names": view_names,
        }


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------


def _masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = torch.softmax(scores, dim=dim)
    attn = torch.where(mask, attn, torch.zeros_like(attn))
    denom = attn.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return attn / denom


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


# -----------------------------------------------------------------------------
# Segment -> macro pooling
# -----------------------------------------------------------------------------


class NodeTemporalAttentionPool(nn.Module):
    """
    Attention over segments for each node independently.

    Input
    -----
    x    : [B, K, N, F]
    mask : [B, K]

    Output
    ------
    pooled_node_features : [B, N, F]
    attn                 : [B, N, K]
    """

    def __init__(self, in_dim: int, attn_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.v = nn.Linear(in_dim, attn_dim)
        self.u = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B,K,N,F], got {tuple(x.shape)}")

        bsz, num_segments, num_nodes, _ = x.shape
        if mask is None:
            mask = torch.ones((bsz, num_segments), dtype=torch.bool, device=x.device)
        if tuple(mask.shape) != (bsz, num_segments):
            raise ValueError(f"mask must have shape {(bsz, num_segments)}, got {tuple(mask.shape)}")

        x_node = x.permute(0, 2, 1, 3)  # [B, N, K, F]
        x_drop = self.dropout(x_node)
        gated = torch.tanh(self.v(x_drop)) * torch.sigmoid(self.u(x_drop))
        scores = self.w(gated).squeeze(-1)  # [B, N, K]

        mask_node = mask.unsqueeze(1).expand(-1, num_nodes, -1)
        attn = _masked_softmax(scores, mask_node, dim=-1)
        pooled = torch.sum(attn.unsqueeze(-1) * x_node, dim=2)  # [B, N, F]
        return pooled, attn

class NodeTemporalMeanPool(nn.Module):
    """
    Simple masked mean over segments for each node.

    Input
    -----
    x    : [B, K, N, F]
    mask : [B, K]

    Output
    ------
    pooled_node_features : [B, N, F]
    attn                 : [B, N, K]   # uniform weights over valid segments
    """
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B,K,N,F], got {tuple(x.shape)}")

        bsz, num_segments, num_nodes, _ = x.shape
        if mask is None:
            mask = torch.ones((bsz, num_segments), dtype=torch.bool, device=x.device)
        if tuple(mask.shape) != (bsz, num_segments):
            raise ValueError(f"mask must have shape {(bsz, num_segments)}, got {tuple(mask.shape)}")

        mask_f = mask.float().unsqueeze(-1).unsqueeze(-1)   # [B,K,1,1]
        denom = mask_f.sum(dim=1).clamp_min(1.0)            # [B,1,1]
        pooled = (x * mask_f).sum(dim=1) / denom            # [B,N,F]

        # for compatibility with existing outputs
        attn = mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1.0)   # [B,K]
        attn = attn.unsqueeze(1).expand(-1, num_nodes, -1)           # [B,N,K]

        return pooled, attn
# -----------------------------------------------------------------------------
# Macro encoder
# -----------------------------------------------------------------------------


class MacroGraphBankEncoder(nn.Module):
    """
    Encode one macro with:
      1) segment -> macro node attention
      2) learned graph-bank fusion through FusedGraphBankGNN
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_views: int,
        num_classes: int,
        graph_backbone: str = "gcn",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
        readout_type: str = "gated_attention",
        fusion_mode: str = "summary_gated",
        topology_rule: str = "union",
        segment_attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_views = int(num_views)
        self.num_classes = int(num_classes)
        self.graph_emb_dim = int(graph_emb_dim)
        self.node_pool = NodeTemporalMeanPool()
        # self.node_pool = NodeTemporalAttentionPool(
        #     in_dim=self.num_node_features,
        #     attn_dim=int(segment_attn_dim),
        #     dropout=dropout,
        # )
        self.graph_model = FusedGraphBankGNN(
            num_node_features=self.num_node_features,
            num_classes=self.num_classes,
            num_nodes=self.num_nodes,
            num_candidates=self.num_views,
            backbone=str(graph_backbone),
            hidden_dim=int(hidden_dim),
            graph_emb_dim=self.graph_emb_dim,
            num_layers=int(num_layers),
            dropout=float(dropout),
            gat_heads=int(gat_heads),
            use_edge_weight=True,
            use_batchnorm=True,
            readout_type=str(readout_type),
            fusion_mode=str(fusion_mode),
            topology_rule=str(topology_rule),
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
        node_feature_seq: Tensor,
        segment_mask: Tensor,
        adj_bank: Tensor,
        topology_bank: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> dict[str, Tensor] | Tensor:
        if node_feature_seq.dim() != 4:
            raise ValueError(f"node_feature_seq must have shape [B,K,N,F], got {tuple(node_feature_seq.shape)}.")
        if segment_mask.dim() != 2:
            raise ValueError(f"segment_mask must have shape [B,K], got {tuple(segment_mask.shape)}.")
        if adj_bank.dim() != 4:
            raise ValueError(f"adj_bank must have shape [B,V,N,N], got {tuple(adj_bank.shape)}.")

        macro_x, segment_attn = self.node_pool(node_feature_seq, segment_mask)
        fallback_adj = torch.mean(adj_bank, dim=1)
        pyg_batch = self._dense_to_batch(macro_x, fallback_adj, labels=labels)

        out = self.graph_model(
            pyg_batch,
            adj_bank=adj_bank,
            topology_bank=topology_bank,
            return_dict=True,
            return_attention_weights=True,
        )

        if not return_dict:
            return out.logits

        return {
            "logits": out.logits,
            "embedding": out.embedding,
            "macro_node_features": macro_x,
            "segment_attention_weights": segment_attn,
            "view_attention_weights": out.fusion_weights,
            "graph_attention_weights": out.graph_attention_weights,
            "fused_adjacency": None if out.aux is None else out.aux.get("fused_adjacency"),
            "fused_topology": None if out.aux is None else out.aux.get("fused_topology"),
        }


# -----------------------------------------------------------------------------
# Subject-level macro MV-GNN
# -----------------------------------------------------------------------------


class SubjectMacroMVGNN(nn.Module):
    """
    End-to-end subject classifier.

    Flow
    ----
    segments -> macro node attention -> learned graph-bank fusion -> macro embedding
    -> subject MIL attention / subject fusion -> subject logits
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_views: int,
        num_classes: int,
        graph_backbone: str = "gcn",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
        graph_readout_type: str = "gated_attention",
        graph_bank_fusion_mode: str = "summary_gated",
        topology_rule: str = "union",
        segment_attn_dim: int = 128,
        subject_aggregation: str = "gated_attention_mil",
        subject_attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.subject_aggregation = str(subject_aggregation).lower()
        self.macro_emb_dim = int(graph_emb_dim)

        self.macro_encoder = MacroGraphBankEncoder(
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
            segment_attn_dim=segment_attn_dim,
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
            raise ValueError(
                f"Unsupported subject_aggregation={subject_aggregation!r}. "
                "Use 'none', 'mean_mil', 'gated_attention_mil', or 'subject_fusion'."
            )

    def forward(
        self,
        batch: Optional[Mapping[str, Any]] = None,
        *,
        node_feature_bag: Optional[Tensor] = None,
        segment_mask: Optional[Tensor] = None,
        adj_bank_bag: Optional[Tensor] = None,
        topology_bank_bag: Optional[Tensor] = None,
        macro_mask: Optional[Tensor] = None,
        macro_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        subject_ids: Optional[Sequence[str]] = None,
        return_dict: bool = True,
        **_: Any,
    ) -> dict[str, Any] | Tensor:
        if batch is not None:
            node_feature_bag = batch["node_feature_bag"]
            segment_mask = batch["segment_mask"]
            adj_bank_bag = batch["adj_bank_bag"]
            topology_bank_bag = batch.get("topology_bank_bag", None)
            macro_mask = batch["macro_mask"]
            macro_ids = batch.get("macro_ids", None)
            labels = batch.get("labels", None)
            subject_ids = batch.get("subject_ids", None)

        assert node_feature_bag is not None
        assert segment_mask is not None
        assert adj_bank_bag is not None
        assert macro_mask is not None

        if node_feature_bag.dim() != 5:
            raise ValueError(f"node_feature_bag must have shape [B,M,K,N,F], got {tuple(node_feature_bag.shape)}.")
        if segment_mask.dim() != 3:
            raise ValueError(f"segment_mask must have shape [B,M,K], got {tuple(segment_mask.shape)}.")
        if adj_bank_bag.dim() != 5:
            raise ValueError(f"adj_bank_bag must have shape [B,M,V,N,N], got {tuple(adj_bank_bag.shape)}.")
        if macro_mask.dim() != 2:
            raise ValueError(f"macro_mask must have shape [B,M], got {tuple(macro_mask.shape)}.")

        bsz, max_macros = macro_mask.shape
        emb_dim = self.macro_emb_dim

        flat_mask = macro_mask.reshape(-1)
        if not torch.any(flat_mask):
            raise ValueError("No valid macros in batch.")

        flat_node_seq = node_feature_bag.reshape(-1, *node_feature_bag.shape[2:])[flat_mask]
        flat_seg_mask = segment_mask.reshape(-1, segment_mask.shape[-1])[flat_mask]
        flat_adj_bank = adj_bank_bag.reshape(-1, *adj_bank_bag.shape[2:])[flat_mask]
        flat_topology_bank = None
        if topology_bank_bag is not None:
            flat_topology_bank = topology_bank_bag.reshape(-1, *topology_bank_bag.shape[2:])[flat_mask]

        macro_out = self.macro_encoder(
            node_feature_seq=flat_node_seq,
            segment_mask=flat_seg_mask,
            adj_bank=flat_adj_bank,
            topology_bank=flat_topology_bank,
            return_dict=True,
        )

        macro_emb = macro_out["embedding"]
        macro_logits = macro_out["logits"]

        grouped_emb = torch.zeros((bsz, max_macros, emb_dim), dtype=macro_emb.dtype, device=macro_emb.device)
        grouped_logits = torch.zeros((bsz, max_macros, self.num_classes), dtype=macro_logits.dtype, device=macro_logits.device)
        grouped_emb.view(-1, emb_dim)[flat_mask] = macro_emb
        grouped_logits.view(-1, self.num_classes)[flat_mask] = macro_logits

        agg = aggregate_subject_predictions(
            instance_embeddings=grouped_emb,
            instance_logits=grouped_logits if self.subject_aggregation == "subject_fusion" else None,
            mask=macro_mask.to(device=grouped_emb.device),
            method=self.subject_aggregation,
            classifier=self.subject_classifier,
            pool=self.subject_pool,
            fusion_head=self.subject_fusion_head,
            sort_subjects=False,
        )

        if not return_dict:
            return agg["subject_logits"]

        segment_attn = torch.zeros(
            (bsz, max_macros, node_feature_bag.shape[-2], node_feature_bag.shape[2]),
            dtype=macro_out["segment_attention_weights"].dtype,
            device=macro_out["segment_attention_weights"].device,
        )
        view_attn = torch.zeros(
            (bsz, max_macros, adj_bank_bag.shape[2]),
            dtype=macro_out["view_attention_weights"].dtype,
            device=macro_out["view_attention_weights"].device,
        )
        segment_attn.view(-1, segment_attn.shape[-2], segment_attn.shape[-1])[flat_mask] = macro_out[
            "segment_attention_weights"
        ]
        view_attn.view(-1, view_attn.shape[-1])[flat_mask] = macro_out["view_attention_weights"]

        return {
            "logits": agg["subject_logits"],
            "probs": agg["subject_prob"],
            "preds": agg["subject_pred"],
            "embedding": agg["subject_embedding"],
            "macro_logits": grouped_logits,
            "macro_embeddings": grouped_emb,
            "macro_attention_weights": agg.get("attention_weights"),
            "segment_attention_weights": segment_attn,
            "view_attention_weights": view_attn,
            "subject_ids": list(subject_ids) if subject_ids is not None else None,
            "macro_ids": macro_ids,
            "targets": labels,
        }


# -----------------------------------------------------------------------------
# CAUEEG spec helpers and runner
# -----------------------------------------------------------------------------


class EarlyStopper:
    def __init__(self, monitor: str, mode: str, patience: int) -> None:
        self.monitor = str(monitor)
        self.mode = str(mode).lower()
        self.patience = int(patience)
        self.best_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.bad_epochs = 0

    def is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return float(value) < float(self.best_value)
        return float(value) > float(self.best_value)

    def step(self, value: float, epoch: int) -> bool:
        if self.is_better(value):
            self.best_value = float(value)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to") and callable(getattr(v, "to")):
            try:
                out[k] = v.to(device, non_blocking=True)
            except TypeError:
                out[k] = v.to(device)
        else:
            out[k] = v
    return out


def collect_epoch_outputs_macro(
    model: SubjectMacroMVGNN,
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

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)
            loss = F.cross_entropy(out["logits"], batch["labels"])
            if train:
                loss.backward()
                optimizer.step()

        n_batches += 1
        total_loss += float(loss.detach().cpu().item())

        probs = torch.softmax(out["logits"], dim=-1)
        pred = torch.argmax(probs, dim=-1)

        y_true_all.append(batch["labels"].detach().cpu().numpy())
        logits_all.append(out["logits"].detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())
        subject_ids_all.extend(list(batch["subject_ids"]))

    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.empty((0,), dtype=np.int64)
    logits = np.concatenate(logits_all, axis=0) if logits_all else np.empty((0, 0), dtype=np.float32)
    probs = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, 0), dtype=np.float32)
    pred = np.concatenate(pred_all, axis=0) if pred_all else np.empty((0,), dtype=np.int64)

    metrics = summarize_classification_metrics(
        y_true=y_true,
        y_pred=pred,
        probs=probs,
        logits=logits,
        num_classes=probs.shape[1] if probs.ndim == 2 and probs.shape[0] > 0 else None,
    )

    return {
        "loss": total_loss / max(n_batches, 1),
        "metrics": metrics,
        "y_true": y_true,
        "logits": logits,
        "probs": probs,
        "pred": pred,
        "subject_ids": subject_ids_all,
    }


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


def prepare_macro_split_bags(spec: "CAUEEGExperimentSpec") -> dict[str, list[SubjectMacroBag]]:
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

    out: dict[str, list[SubjectMacroBag]] = {"train": [], "val": [], "test": []}
    for split, rows in split_to_resolved.items():
        split_entries = {sid: entries[sid] for sid, _, _ in rows}
        out[split] = build_subject_macro_bags(
            split_entries,
            feature_families=spec.feature_families,
            connectivity_metrics=spec.connectivity_metrics_to_load,
            graph_bank_specs=spec.topology.graph_bank_specs,
            macro_duration_sec=float(spec.level.macro_duration_sec),
            sfreq=200.0,
            connectivity_reduce=str(spec.level.connectivity_reduce),
        )
    return out


def make_macro_loaders(
    split_bags: Mapping[str, Sequence[SubjectMacroBag]],
    spec: "CAUEEGExperimentSpec",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    common = {
        "num_workers": int(spec.train.num_workers),
        "pin_memory": True,
        "collate_fn": SubjectMacroBagCollate(),
    }
    train_loader = DataLoader(
        SubjectMacroBagDataset(split_bags["train"]),
        batch_size=int(spec.train.batch_size),
        shuffle=True,
        **common,
    )
    val_loader = DataLoader(
        SubjectMacroBagDataset(split_bags["val"]),
        batch_size=int(spec.train.batch_size),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        SubjectMacroBagDataset(split_bags["test"]),
        batch_size=int(spec.train.batch_size),
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def build_macro_mvgnn_model(
    spec: "CAUEEGExperimentSpec",
    *,
    sample_macro: MacroInstance,
    num_classes: int,
) -> SubjectMacroMVGNN:
    return SubjectMacroMVGNN(
        num_nodes=int(sample_macro.node_feature_seq.shape[1]),
        num_node_features=int(sample_macro.node_feature_seq.shape[2]),
        num_views=int(sample_macro.adj_bank.shape[0]),
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
        segment_attn_dim=int(spec.aggregation.attn_dim),
        subject_aggregation=str(spec.aggregation.strategy),
        subject_attn_dim=int(spec.aggregation.attn_dim),
    )


def run_macro_mvgnn_experiment(spec: "CAUEEGExperimentSpec") -> dict[str, Any]:
    cm = _cm()
    if str(spec.level.graph_level).lower() != "macro":
        raise ValueError("macro_mvgnn requires spec.level.graph_level='macro'.")
    if str(spec.aggregation.strategy).lower() == "none":
        raise ValueError("macro_mvgnn is a subject-level macro-bag model; use a bag-level aggregation strategy.")

    set_seed(spec.train.seed)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")

    split_bags = prepare_macro_split_bags(spec)
    train_loader, val_loader, test_loader = make_macro_loaders(split_bags, spec)

    if len(split_bags["train"]) == 0 or len(split_bags["train"][0].macros) == 0:
        raise ValueError("Training split produced zero macro bags.")

    sample_macro = split_bags["train"][0].macros[0]
    num_classes = len(sorted({int(bag.label) for split in split_bags.values() for bag in split}))
    class_names = spec.class_names or cm.infer_class_names(spec.task, num_classes)

    model = build_macro_mvgnn_model(spec, sample_macro=sample_macro, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    stopper = EarlyStopper(spec.train.monitor, spec.train.monitor_mode, spec.train.patience)

    run_dir = ensure_dir(
        os.path.join(
            spec.output_root,
            make_run_name(spec.name, spec.aggregation.strategy, timestamp=True),
        )
    )

    history_train: list[dict[str, Any]] = []
    history_val: list[dict[str, Any]] = []
    best_state: Optional[dict[str, Any]] = None
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")

    for epoch in range(1, spec.train.epochs + 1):
        train_out = collect_epoch_outputs_macro(model, train_loader, device=device, optimizer=optimizer)
        val_out = collect_epoch_outputs_macro(model, val_loader, device=device, optimizer=None)

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

        if spec.train.monitor == "loss":
            monitor_value = float(val_out["loss"])
        else:
            monitor_value = float(val_metrics[spec.train.monitor])

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_out['loss']:.4f}, bal_acc={train_metrics['balanced_accuracy']:.4f}, macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val loss={val_out['loss']:.4f}, bal_acc={val_metrics['balanced_accuracy']:.4f}, macro_f1={val_metrics['macro_f1']:.4f}"
        )

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
                "spec": cm.asdict(spec) if hasattr(cm, "asdict") else None,
            }
            torch.save(best_state, best_ckpt_path)

        if should_stop:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best {spec.train.monitor}={stopper.best_value:.6f} at epoch {stopper.best_epoch}."
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was saved during training.")

    best_loaded = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_loaded["model_state_dict"])

    train_final = collect_epoch_outputs_macro(model, train_loader, device=device, optimizer=None)
    val_final = collect_epoch_outputs_macro(model, val_loader, device=device, optimizer=None)
    test_final = collect_epoch_outputs_macro(model, test_loader, device=device, optimizer=None)

    train_df = build_subject_prediction_dataframe(train_final)
    val_df = build_subject_prediction_dataframe(val_final)
    test_df = build_subject_prediction_dataframe(test_final)

    pd.DataFrame(history_train).to_csv(os.path.join(run_dir, "history_train.csv"), index=False)
    pd.DataFrame(history_val).to_csv(os.path.join(run_dir, "history_val.csv"), index=False)
    train_df.to_csv(os.path.join(run_dir, "train_subject_predictions.csv"), index=False)
    val_df.to_csv(os.path.join(run_dir, "val_subject_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(run_dir, "test_subject_predictions.csv"), index=False)


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
                "view_names": list(sample_macro.view_names),
                "model_family": str(spec.model.family),
                "spec": cm.asdict(spec),
            },
            f,
            indent=2,
        )

    try:
        plot_training_curves(
            {"train": history_train, "val": history_val},
            save_path=os.path.join(run_dir, "training_curves.png"),
            metric_keys=("loss", "balanced_accuracy", "macro_f1"),
            title=f"{spec.name} training curves",
        )
    except Exception:
        pass

    try:
        plot_confusion_matrix(
            test_df["true_label"].to_numpy(dtype=np.int64),
            test_df["pred_label"].to_numpy(dtype=np.int64),
            save_path=os.path.join(run_dir, "test_confusion_matrix.png"),
            class_names=class_names,
            normalize=True,
            title=f"{spec.name} test confusion matrix",
        )
    except Exception:
        pass

    return {
        "run_dir": run_dir,
        "best_epoch": best_loaded["epoch"],
        "best_checkpoint_path": best_ckpt_path,
        "train_metrics": train_final["metrics"],
        "val_metrics": val_final["metrics"],
        "test_metrics": test_final["metrics"],
        "class_names": class_names,
        "view_names": tuple(sample_macro.view_names),
        "spec": spec,
    }


__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_BANDWISE_METRICS",
    "MacroInstance",
    "SubjectMacroBag",
    "SubjectMacroBagDataset",
    "SubjectMacroBagCollate",
    "NodeTemporalAttentionPool",
    "MacroGraphBankEncoder",
    "SubjectMacroMVGNN",
    "make_metric_band_graph_bank_specs",
    "build_rich_macro_graph_bank",
    "build_macro_instance_from_h5_entry",
    "build_subject_macro_bag",
    "build_subject_macro_bags",
    "prepare_macro_split_bags",
    "build_macro_mvgnn_model",
    "run_macro_mvgnn_experiment",
]

