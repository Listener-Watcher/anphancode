from __future__ import annotations

"""
macro_mvgnn.py

Macro-level multi-view graph model for the EEG codebase.

What this module adds
---------------------
1) Segment -> macro node-feature aggregation with trainable attention.
2) A differentiable graph-bank fusion stage that reuses FusedGraphBankGNN.
3) Macro -> subject aggregation with the existing MIL utilities.
4) H5-first data builders that reuse helpers already present in caueeg_main.py.

Design notes
------------
- This module intentionally reuses the current project code instead of replacing it.
- Connectivity views are built from the precomputed H5 tensors.
- The graph-bank fusion is learned end-to-end through FusedGraphBankGNN.
- Segment-to-macro node aggregation is learned end-to-end through
  NodeTemporalAttentionPool.

Important practical note
------------------------
In the current codebase, coherence / PLI / wPLI are band-wise, but Pearson and
Spearman are not. So the natural view bank is usually:
    3 band-wise metrics x 5 bands + 2 single-matrix metrics = 17 views
rather than a strict 25-view bank, unless you later add band-pass Pearson /
Spearman or PLV to the H5 build.
"""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from caueeg_main import (
    H5SubjectEntry,
    build_macro_groups,
    reduce_array,
    aggregate_connectivity_sources,
    load_caueeg_task_splits,
    resolve_h5_subject_ids_for_split,
    load_h5_entries,
)
from gnn import FusedGraphBankGNN
from graph_construction import build_graph_bank
from models_mil import GatedAttentionMILPool, SubjectFusionHead, aggregate_subject_predictions



import os
from torch.utils.data import DataLoader
from torch.optim import AdamW
from trainer import Trainer



DEFAULT_BANDS: tuple[str, ...] = ("delta", "theta", "alpha", "beta", "gamma")
DEFAULT_BANDWISE_METRICS: tuple[str, ...] = ("coherence", "pli", "wpli")
# DEFAULT_NONBAND_METRICS: tuple[str, ...] = ("pearson", "spearman")


# -----------------------------------------------------------------------------
# Dataclasses for macro bags
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class MacroInstance:
    subject_id: str
    label: int
    macro_id: int
    node_feature_seq: np.ndarray   # [K, N, F]
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
# Graph-bank spec builder
# -----------------------------------------------------------------------------


def make_metric_band_graph_bank_specs(
    *,
    bandwise_metrics: Sequence[str] = DEFAULT_BANDWISE_METRICS,
    # nonband_metrics: Sequence[str] = DEFAULT_NONBAND_METRICS,
    bands: Sequence[int | str] = DEFAULT_BANDS,
    topology_mode: str = "connectivity",
    edge_weight_mode: str = "connectivity",
    topology_kwargs: Optional[Mapping[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Build a view bank specification that matches the current codebase.

    By default this creates:
      - 15 band-wise views from coherence / pli / wpli
      - 2 single-matrix views from pearson / spearman

    yielding 17 views total.
    """
    topo_kwargs = dict(topology_kwargs or {"mode": "topk", "topk": 6})
    specs: list[dict[str, Any]] = []

    for metric in bandwise_metrics:
        for band in bands:
            band_name = str(band)
            specs.append(
                {
                    "name": f"{metric}_{band_name}",
                    "topology_mode": topology_mode,
                    "edge_weight_mode": edge_weight_mode,
                    "connectivity_metric": str(metric),
                    "band": band,
                    "topology_kwargs": dict(topo_kwargs),
                }
            )

    # for metric in nonband_metrics:
    #     specs.append(
    #         {
    #             "name": str(metric),
    #             "topology_mode": topology_mode,
    #             "edge_weight_mode": edge_weight_mode,
    #             "connectivity_metric": str(metric),
    #             "topology_kwargs": dict(topo_kwargs),
    #         }
    #     )

    return specs


# -----------------------------------------------------------------------------
# H5 -> macro bag builders
# -----------------------------------------------------------------------------


def _concat_feature_families_no_reduce(
    entry: H5SubjectEntry,
    window_indices: np.ndarray,
    feature_families: Sequence[str],
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for fam in feature_families:
        x = np.asarray(entry.features[fam], dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Feature family {fam!r} expected [W,N,F], got {x.shape}.")
        parts.append(x[window_indices])
    return np.concatenate(parts, axis=-1).astype(np.float32)  # [K, N, F]


def build_macro_instance_from_h5_entry(
    entry: H5SubjectEntry,
    *,
    macro_id: int,
    window_indices: np.ndarray,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Sequence[Mapping[str, Any]],
    connectivity_reduce: str = "mean",
) -> MacroInstance:
    """
    Build one macro instance from a subject H5 entry.

    Node features stay at segment level: [K, N, F].
    Connectivity is aggregated across the same K windows to form one macro graph bank.
    """
    node_feature_seq = _concat_feature_families_no_reduce(entry, window_indices, feature_families)

    connectivity_sources, band_names_map = aggregate_connectivity_sources(
        entry,
        window_indices,
        connectivity_metrics=connectivity_metrics,
        reduce_mode=connectivity_reduce,
    )

    # For any feature-induced candidate, the bank constructor needs one [N, F] matrix.
    # We use the simple mean feature matrix here. The trainable segment->macro attention
    # happens later inside the model.
    node_feature_mean = reduce_array(node_feature_seq, "mean", axis=0).astype(np.float32)

    bank = build_graph_bank(
        node_features=node_feature_mean,
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
    entry: H5SubjectEntry,
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Sequence[Mapping[str, Any]],
    macro_duration_sec: float = 60.0,
    sfreq: float = 200.0,
    connectivity_reduce: str = "mean",
) -> SubjectMacroBag:
    groups = build_macro_groups(
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

    return SubjectMacroBag(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        macros=macros,
        metadata={"num_macros": len(macros)},
    )


def build_subject_macro_bags(
    entries: Mapping[str, H5SubjectEntry],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Sequence[Mapping[str, Any]],
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

        bsz, num_segments, num_nodes, feat_dim = x.shape
        if mask is None:
            mask = torch.ones((bsz, num_segments), dtype=torch.bool, device=x.device)
        if tuple(mask.shape) != (bsz, num_segments):
            raise ValueError(
                f"mask must have shape {(bsz, num_segments)}, got {tuple(mask.shape)}"
            )

        x_node = x.permute(0, 2, 1, 3)  # [B, N, K, F]
        x_drop = self.dropout(x_node)
        gated = torch.tanh(self.v(x_drop)) * torch.sigmoid(self.u(x_drop))
        scores = self.w(gated).squeeze(-1)  # [B, N, K]

        mask_node = mask.unsqueeze(1).expand(-1, num_nodes, -1)
        attn = _masked_softmax(scores, mask_node, dim=-1)
        pooled = torch.sum(attn.unsqueeze(-1) * x_node, dim=2)  # [B, N, F]
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

        self.node_pool = NodeTemporalAttentionPool(
            in_dim=self.num_node_features,
            attn_dim=int(segment_attn_dim),
            dropout=dropout,
        )
        self.graph_model = FusedGraphBankGNN(
            num_node_features=self.num_node_features,
            num_classes=self.num_classes,
            num_nodes=self.num_nodes,
            num_candidates=self.num_views,
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
            raise ValueError(
                f"node_feature_seq must have shape [B,K,N,F], got {tuple(node_feature_seq.shape)}."
            )
        if segment_mask.dim() != 2:
            raise ValueError(
                f"segment_mask must have shape [B,K], got {tuple(segment_mask.shape)}."
            )
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
        *,
        node_feature_bag: Tensor,
        segment_mask: Tensor,
        adj_bank_bag: Tensor,
        topology_bank_bag: Optional[Tensor] = None,
        macro_mask: Tensor,
        macro_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        subject_ids: Optional[Sequence[str]] = None,
        return_dict: bool = True,
        **_: Any,
    ) -> dict[str, Any] | Tensor:
        if node_feature_bag.dim() != 5:
            raise ValueError(
                f"node_feature_bag must have shape [B,M,K,N,F], got {tuple(node_feature_bag.shape)}."
            )
        if segment_mask.dim() != 3:
            raise ValueError(
                f"segment_mask must have shape [B,M,K], got {tuple(segment_mask.shape)}."
            )
        if adj_bank_bag.dim() != 5:
            raise ValueError(
                f"adj_bank_bag must have shape [B,M,V,N,N], got {tuple(adj_bank_bag.shape)}."
            )
        if macro_mask.dim() != 2:
            raise ValueError(
                f"macro_mask must have shape [B,M], got {tuple(macro_mask.shape)}."
            )

        bsz, max_macros = macro_mask.shape
        emb_dim = int(self.macro_encoder.graph_model.classifier.in_features)

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

        num_valid_macros = int(flat_mask.sum().item())
        macro_emb = macro_out["embedding"]
        macro_logits = macro_out["logits"]

        grouped_emb = torch.zeros(
            (bsz, max_macros, emb_dim),
            dtype=macro_emb.dtype,
            device=macro_emb.device,
        )
        grouped_logits = torch.zeros(
            (bsz, max_macros, self.num_classes),
            dtype=macro_logits.dtype,
            device=macro_logits.device,
        )
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
            "num_valid_macros": num_valid_macros,
        }


__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_BANDWISE_METRICS",
    "DEFAULT_NONBAND_METRICS",
    "MacroInstance",
    "SubjectMacroBag",
    "SubjectMacroBagDataset",
    "SubjectMacroBagCollate",
    "NodeTemporalAttentionPool",
    "MacroGraphBankEncoder",
    "SubjectMacroMVGNN",
    "make_metric_band_graph_bank_specs",
    "build_macro_instance_from_h5_entry",
    "build_subject_macro_bag",
    "build_subject_macro_bags",
]


if __name__ == "__main__":


    # -------------------------------------------------
    # paths
    # -------------------------------------------------
    dataset_path = "/mnt/data/anphan/CAUEEG/caueeg-dataset"
    h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------------------------------------------------
    # task split
    # -------------------------------------------------
    task = "dementia"
    config, train_rows, val_rows, test_rows = load_caueeg_task_splits(dataset_path, task)

    train_pairs = resolve_h5_subject_ids_for_split(h5_path, train_rows, "train")
    val_pairs   = resolve_h5_subject_ids_for_split(h5_path, val_rows, "validation")
    test_pairs  = resolve_h5_subject_ids_for_split(h5_path, test_rows, "test")

    train_ids = [sid for sid, _, _ in train_pairs]
    val_ids   = [sid for sid, _, _ in val_pairs]
    test_ids  = [sid for sid, _, _ in test_pairs]

    # -------------------------------------------------
    # what to load from H5
    # -------------------------------------------------
    feature_families = ("relative_band_power", "statistical")
    connectivity_metrics = ("coherence", "pli", "wpli")

    # -------------------------------------------------
    # graph-bank definition
    # default = 17 views:
    #   15 = 3 bandwise metrics x 5 bands
    #    2 = pearson + spearman
    # -------------------------------------------------
    graph_bank_specs = make_metric_band_graph_bank_specs(
        bandwise_metrics=("coherence", "pli", "wpli"),
        bands=("delta", "theta", "alpha", "beta", "gamma"),
        topology_mode="connectivity",
        edge_weight_mode="connectivity",
        topology_kwargs={"mode": "topk", "topk": 4},
    )

    # -------------------------------------------------
    # load H5 entries
    # -------------------------------------------------
    train_entries = load_h5_entries(
        h5_path=h5_path,
        subject_ids=train_ids,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
    )
    val_entries = load_h5_entries(
        h5_path=h5_path,
        subject_ids=val_ids,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
    )
    test_entries = load_h5_entries(
        h5_path=h5_path,
        subject_ids=test_ids,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
    )

    # -------------------------------------------------
    # build macro bags
    # macro_duration_sec controls the macro level
    # -------------------------------------------------
    macro_duration_sec = 60.0   # try 60, 120, 300
    sfreq = 200.0               # CAUEEG

    train_bags = build_subject_macro_bags(
        train_entries,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        graph_bank_specs=graph_bank_specs,
        macro_duration_sec=macro_duration_sec,
        sfreq=sfreq,
        connectivity_reduce="mean",
    )
    val_bags = build_subject_macro_bags(
        val_entries,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        graph_bank_specs=graph_bank_specs,
        macro_duration_sec=macro_duration_sec,
        sfreq=sfreq,
        connectivity_reduce="mean",
    )
    test_bags = build_subject_macro_bags(
        test_entries,
        feature_families=feature_families,
        connectivity_metrics=connectivity_metrics,
        graph_bank_specs=graph_bank_specs,
        macro_duration_sec=macro_duration_sec,
        sfreq=sfreq,
        connectivity_reduce="mean",
    )

    # -------------------------------------------------
    # loaders
    # -------------------------------------------------
    collate_fn = SubjectMacroBagCollate()

    train_loader = DataLoader(
        SubjectMacroBagDataset(train_bags),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SubjectMacroBagDataset(val_bags),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SubjectMacroBagDataset(test_bags),
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # -------------------------------------------------
    # infer dimensions from one macro
    # -------------------------------------------------
    sample_macro = train_bags[0].macros[0]
    num_nodes = sample_macro.node_feature_seq.shape[1]
    num_node_features = sample_macro.node_feature_seq.shape[2]
    num_views = sample_macro.adj_bank.shape[0]
    num_classes = 3

    print("num_nodes =", num_nodes)
    print("num_node_features =", num_node_features)
    print("num_views =", num_views)

    # -------------------------------------------------
    # model
    # -------------------------------------------------
    model = SubjectMacroMVGNN(
        num_nodes=num_nodes,
        num_node_features=num_node_features,
        num_views=num_views,
        num_classes=num_classes,
        graph_backbone="gatv2",                 # try: gcn, sage, gatv2
        hidden_dim=64,
        graph_emb_dim=128,
        num_layers=2,
        dropout=0.2,
        gat_heads=4,
        graph_readout_type="gated_attention", # try: mean, max, gated_attention
        graph_bank_fusion_mode="summary_gated",
        topology_rule="union",
        segment_attn_dim=128,
        subject_aggregation="gated_attention_mil",  # try: mean_mil, gated_attention_mil, subject_fusion
        subject_attn_dim=128,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        loss_name="cross_entropy",
        monitor="macro-f1",
        monitor_mode="max",
        early_stopping_patience=60,
        checkpoint_dir="./macro_mvgnn_runs",
        checkpoint_name="best_macro_mvgnn.pt",
        use_amp=False,
        num_classes=num_classes,
    )

    history = trainer.fit(train_loader, val_loader, num_epochs=120)

    test_result = trainer.validate(test_loader)
    print(test_result)