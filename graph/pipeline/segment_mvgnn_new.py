from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, TYPE_CHECKING

import copy
import json
import os

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
from visualize import plot_confusion_matrix, plot_training_curves

if TYPE_CHECKING:  # pragma: no cover
    from caueeg_main_new import CAUEEGExperimentSpec, H5SubjectEntry


def _cm():
    import caueeg_main_new as cm
    return cm


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
def segment_instance_to_graph(seg: SegmentInstance) -> Data:
    # placeholder single adjacency just to satisfy a PyG Data object;
    # real graph will be rebuilt from adj_bank/topology_bank inside the model
    fallback_adj = np.mean(seg.adj_bank, axis=0).astype(np.float32)
    np.fill_diagonal(fallback_adj, 0.0)
    edge_index, edge_weight = dense_to_sparse(torch.tensor(fallback_adj, dtype=torch.float32))

    g = Data(
        x=torch.tensor(seg.node_features, dtype=torch.float32),
        edge_index=edge_index.long(),
        y=torch.tensor([seg.label], dtype=torch.long),
    )
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()
    g.adj = torch.tensor(fallback_adj, dtype=torch.float32)

    g.adj_bank = torch.tensor(seg.adj_bank, dtype=torch.float32)
    g.topology_bank = torch.tensor(seg.topology_bank, dtype=torch.float32)

    g.subject_id = str(seg.subject_id)
    g.segment_id = int(seg.segment_id)
    g.view_names = list(seg.view_names)

    for k, v in seg.metadata.items():
        setattr(g, k, v)

    return g
def build_segment_graphs_for_entries(
    entries: Mapping[str, "H5SubjectEntry"],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[Data]:
    graphs = []
    for sid in sorted(entries.keys()):
        entry = entries[sid]
        for window_idx in range(len(entry.segment_id)):
            seg = build_segment_instance_from_h5_entry(
                entry,
                window_idx=window_idx,
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                graph_bank_specs=graph_bank_specs,
            )
            graphs.append(segment_instance_to_graph(seg))
    return graphs
@dataclass(slots=True)
class SegmentInstance:
    subject_id: str
    label: int
    segment_id: int
    node_features: np.ndarray     # [N, F]
    adj_bank: np.ndarray          # [V, N, N]
    topology_bank: np.ndarray     # [V, N, N]
    view_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubjectSegmentBag:
    subject_id: str
    label: int
    segments: list[SegmentInstance]
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class AggregationConfig:
    strategy: AggregationStrategy = "none"
    posthoc_eval_vote: str = "soft_vote"
    attn_dim: int = 128
    train_max_instances_per_subject: Optional[int] = None
    eval_max_instances_per_subject: Optional[int] = None

    base_k: Optional[int] = None
    max_k_per_subject: Optional[int] = 300
# ---------------------------------------------------------------------
# H5 -> segment instances
# ---------------------------------------------------------------------

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
        xw = x[window_idx].astype(np.float32)   # [N, F_fam]
        parts.append(xw)
        by_family[fam] = xw

    x_all = np.concatenate(parts, axis=-1).astype(np.float32)
    return x_all, by_family


def build_segment_instance_from_h5_entry(
    entry: "H5SubjectEntry",
    *,
    window_idx: int,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SegmentInstance:
    cm = _cm()

    node_features, node_features_by_family = _select_feature_families_for_one_window(
        entry,
        window_idx,
        feature_families,
    )
    node_features = cm.zscore_node_features(node_features)

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
            node_feature_mean_all=node_features,
            node_feature_mean_by_family=node_features_by_family,
            connectivity_sources=connectivity_sources,
            band_names_map=band_names_map,
            feature_families=feature_families,
        )
    else:
        bank = build_graph_bank(
            node_features=node_features,
            connectivity_sources={
                metric: (np.asarray(mat, dtype=np.float32), band_names_map.get(metric, None))
                for metric, mat in connectivity_sources.items()
            },
            candidate_specs=list(graph_bank_specs),
            fixed_topology=None,
        )

    adj_bank = np.stack([cand.adjacency for cand in bank], axis=0).astype(np.float32)
    topology_bank = np.stack([cand.topology for cand in bank], axis=0).astype(np.float32)
    view_names = [cand.name for cand in bank]

    return SegmentInstance(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        segment_id=int(entry.segment_id[window_idx]),
        node_features=node_features,
        adj_bank=adj_bank,
        topology_bank=topology_bank,
        view_names=view_names,
        metadata={
            "window_idx": int(window_idx),
            "start_sample": int(entry.start_sample[window_idx]),
            "end_sample": int(entry.end_sample[window_idx]),
        },
    )


def build_subject_segment_bag(
    entry: "H5SubjectEntry",
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> SubjectSegmentBag:
    segments: list[SegmentInstance] = []
    for window_idx in range(len(entry.segment_id)):
        segments.append(
            build_segment_instance_from_h5_entry(
                entry,
                window_idx=window_idx,
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                graph_bank_specs=graph_bank_specs,
            )
        )

    return SubjectSegmentBag(
        subject_id=str(entry.subject_id),
        label=int(entry.label),
        segments=segments,
        metadata={"num_segments": len(segments)},
    )


def build_subject_segment_bags(
    entries: Mapping[str, "H5SubjectEntry"],
    *,
    feature_families: Sequence[str],
    connectivity_metrics: Sequence[str],
    graph_bank_specs: Optional[Sequence[Mapping[str, Any]]] = None,
) -> list[SubjectSegmentBag]:
    out: list[SubjectSegmentBag] = []
    for sid in sorted(entries.keys()):
        out.append(
            build_subject_segment_bag(
                entries[sid],
                feature_families=feature_families,
                connectivity_metrics=connectivity_metrics,
                graph_bank_specs=graph_bank_specs,
            )
        )
    return out

class SubjectSegmentBagDataset(Dataset):
    def __init__(self, bags: Sequence[SubjectSegmentBag]):
        self.bags = list(bags)
        if len(self.bags) == 0:
            raise ValueError("bags must not be empty.")

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> SubjectSegmentBag:
        return self.bags[idx]


class SubjectSegmentBagCollate:
    def __call__(self, batch: Sequence[SubjectSegmentBag]) -> dict[str, Any]:
        batch = list(batch)
        if len(batch) == 0:
            raise ValueError("Empty batch.")

        bsz = len(batch)
        max_segments = max(len(item.segments) for item in batch)

        first_seg = batch[0].segments[0]
        num_nodes = int(first_seg.node_features.shape[0])
        num_node_features = int(first_seg.node_features.shape[1])
        num_views = int(first_seg.adj_bank.shape[0])

        node_feature_bag = torch.zeros((bsz, max_segments, num_nodes, num_node_features), dtype=torch.float32)
        adj_bank_bag = torch.zeros((bsz, max_segments, num_views, num_nodes, num_nodes), dtype=torch.float32)
        topology_bank_bag = torch.zeros((bsz, max_segments, num_views, num_nodes, num_nodes), dtype=torch.float32)
        segment_mask = torch.zeros((bsz, max_segments), dtype=torch.bool)
        segment_ids = torch.full((bsz, max_segments), fill_value=-1, dtype=torch.long)

        labels = torch.tensor([int(item.label) for item in batch], dtype=torch.long)
        subject_ids = [str(item.subject_id) for item in batch]
        view_names = list(first_seg.view_names)

        for b_idx, bag in enumerate(batch):
            for s_idx, seg in enumerate(bag.segments):
                node_feature_bag[b_idx, s_idx] = torch.as_tensor(seg.node_features, dtype=torch.float32)
                adj_bank_bag[b_idx, s_idx] = torch.as_tensor(seg.adj_bank, dtype=torch.float32)
                topology_bank_bag[b_idx, s_idx] = torch.as_tensor(seg.topology_bank, dtype=torch.float32)
                segment_mask[b_idx, s_idx] = True
                segment_ids[b_idx, s_idx] = int(seg.segment_id)

        return {
            "node_feature_bag": node_feature_bag,   # [B,S,N,F]
            "adj_bank_bag": adj_bank_bag,           # [B,S,V,N,N]
            "topology_bank_bag": topology_bank_bag, # [B,S,V,N,N]
            "segment_mask": segment_mask,           # [B,S]
            "segment_ids": segment_ids,
            "labels": labels,
            "subject_ids": subject_ids,
            "view_names": view_names,
        }

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
        vote_threshold=0.45,
        fusion_temperature=0.7
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
    ) -> dict[str, Tensor]:
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
        vote_threshold=0.45,
        fusion_temperature=0.7,
        subject_aggregation: str = "gated_attention_mil",
        subject_attn_dim: int = 128,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.segment_emb_dim = int(graph_emb_dim)
        self.subject_aggregation = str(subject_aggregation).lower()

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
            return_attention_weights=True,
        )
        self.subject_classifier = None
        self.subject_pool = None
        self.subject_fusion_head = None

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

        # agg = aggregate_subject_predictions(
        #     instance_embeddings=seg_emb,
        #     instance_logits=seg_logits if self.subject_aggregation == "subject_fusion" else None,
        #     subject_ids=[
        #         sid
        #         for b_idx, sid in enumerate(batch["subject_ids"])
        #         for s_idx in range(max_segments)
        #         if bool(segment_mask[b_idx, s_idx].item())
        #     ],
        #     method=self.subject_aggregation,
        #     classifier=self.subject_classifier,
        #     pool=self.subject_pool,
        #     fusion_head=self.subject_fusion_head,
        #     sort_subjects=False,
        # )
        # return {
        #     "logits": agg["subject_logits"],
        #     "probs": agg["subject_prob"],
        #     "pred": agg["subject_pred"],
        #     "subject_embedding": agg["subject_embedding"],
        #     "subject_attention_weights": agg["attention_weights"],
        #     "segment_embeddings_flat": seg_emb,
        #     "segment_logits_flat": seg_logits,
        #     "segment_view_attention_weights": seg_out["view_attention_weights"],
        #     "subject_ids": batch["subject_ids"],
        # }
        # inside SubjectSegmentMVGNN.forward(...)
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
            "subject_attention_weights": agg["attention_weights"],   # MIL over segments
            "segment_embeddings_grouped": grouped_emb,
            "segment_logits_grouped": grouped_logits,
            "segment_view_attention_weights": seg_out["view_attention_weights"],  # [num_valid_segments, K]
            "segment_fused_adjacency": seg_out["fused_adjacency"],
            "segment_fused_topology": seg_out["fused_topology"],
            "segment_ids": batch["segment_ids"],
            "subject_ids": batch["subject_ids"],
            "segment_mask": batch["segment_mask"],
            "view_names": batch["view_names"],
        }
def move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


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
    y_true_all, logits_all, probs_all, pred_all = [], [], [], []
    subject_ids_all: list[str] = []
    segment_view_attn_all = []
    segment_subject_ids_all = []
    segment_segment_ids_all = []
    view_names_ref = None
    subject_mil_attn_all = []
    subject_segment_ids_all = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            out = model(batch)

            if view_names_ref is None:
                view_names_ref = list(batch["view_names"])

            # bank attention for valid segments
            seg_attn = out["segment_view_attention_weights"].detach().cpu().numpy()   # [num_valid_segments, K]
            segment_view_attn_all.append(seg_attn)

            seg_mask = batch["segment_mask"].detach().cpu().numpy()
            seg_ids = batch["segment_ids"].detach().cpu().numpy()
            for b_idx, sid in enumerate(batch["subject_ids"]):
                valid_ids = seg_ids[b_idx][seg_mask[b_idx]]
                segment_subject_ids_all.extend([sid] * len(valid_ids))
                segment_segment_ids_all.extend(valid_ids.tolist())

            # MIL attention over segments, if available
            if out["subject_attention_weights"] is not None:
                mil_attn = out["subject_attention_weights"].detach().cpu().numpy()   # [B, S]
                subject_mil_attn_all.append(mil_attn)
                subject_segment_ids_all.append(seg_ids)

            loss = F.cross_entropy(out["logits"], batch["labels"])

            if train:
                loss.backward()
                optimizer.step()

        probs = out["probs"]
        pred = out["pred"]

        total_loss += float(loss.detach().cpu().item())
        n_batches += 1

        y_true_all.append(batch["labels"].detach().cpu().numpy())
        logits_all.append(out["logits"].detach().cpu().numpy())
        probs_all.append(probs.detach().cpu().numpy())
        pred_all.append(pred.detach().cpu().numpy())
        subject_ids_all.extend(list(batch["subject_ids"]))

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

    return {
        "loss": total_loss / max(n_batches, 1),
        "metrics": metrics,
        "y_true": y_true,
        "logits": logits,
        "probs": probs,
        "pred": pred,
        "subject_ids": subject_ids_all,
        "segment_view_attention_weights": np.concatenate(segment_view_attn_all, axis=0),
        "segment_subject_ids": segment_subject_ids_all,
        "segment_ids_flat": segment_segment_ids_all,
        "view_names": view_names_ref,
    }
def build_segment_attention_dataframe(epoch_out):
    weights = np.asarray(epoch_out["segment_view_attention_weights"], dtype=np.float32)
    view_names = list(epoch_out["view_names"])

    df = pd.DataFrame({
        "subject_id": list(epoch_out["segment_subject_ids"]),
        "segment_id": np.asarray(epoch_out["segment_ids_flat"], dtype=np.int64),
    })
    for k, name in enumerate(view_names):
        df[f"weight__{name}"] = weights[:, k]
    df["top1_view"] = [view_names[i] for i in np.argmax(weights, axis=1)]
    return df
def build_subject_prediction_dataframe(epoch_out: Mapping[str, Any]) -> pd.DataFrame:
    probs = np.asarray(epoch_out["probs"], dtype=np.float32)
    logits = np.asarray(epoch_out["logits"], dtype=np.float32)

    df = pd.DataFrame({
        "subject_id": list(epoch_out["subject_ids"]),
        "true_label": np.asarray(epoch_out["y_true"], dtype=np.int64),
        "pred_label": np.asarray(epoch_out["pred"], dtype=np.int64),
        "source_level": "subject",
    })
    for c in range(probs.shape[1]):
        df[f"prob_{c}"] = probs[:, c]
    for c in range(logits.shape[1]):
        df[f"logit_{c}"] = logits[:, c]
    return df
def summarize_bank_attention(df):
    weight_cols = [c for c in df.columns if c.startswith("weight__")]
    rows = []
    top1_counts = df["top1_view"].value_counts(normalize=True).to_dict()

    for col in weight_cols:
        view_name = col.replace("weight__", "")
        rows.append({
            "view_name": view_name,
            "mean_weight": float(df[col].mean()),
            "std_weight": float(df[col].std(ddof=0)),
            "top1_frequency": float(top1_counts.get(view_name, 0.0)),
        })
    return pd.DataFrame(rows).sort_values("mean_weight", ascending=False)
# def prepare_segment_split_bags(spec: "CAUEEGExperimentSpec") -> dict[str, list[SubjectSegmentBag]]:
#     cm = _cm()

#     _, train_rows, val_rows, test_rows = cm.load_caueeg_task_splits(spec.dataset_path, spec.task)
#     split_to_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
#     split_to_resolved = {
#         split: cm.resolve_h5_subject_ids_for_split(spec.h5_path, rows, split)
#         for split, rows in split_to_rows.items()
#     }

#     all_subject_ids: list[str] = []
#     for rows in split_to_resolved.values():
#         all_subject_ids.extend([sid for sid, _, _ in rows])

#     entries = cm.load_h5_entries(
#         spec.h5_path,
#         all_subject_ids,
#         feature_families=spec.feature_families,
#         connectivity_metrics=spec.connectivity_metrics_to_load,
#     )

#     out: dict[str, list[SubjectSegmentBag]] = {"train": [], "val": [], "test": []}
#     for split, rows in split_to_resolved.items():
#         split_entries = {sid: entries[sid] for sid, _, _ in rows}
#         out[split] = build_subject_segment_bags(
#             split_entries,
#             feature_families=spec.feature_families,
#             connectivity_metrics=spec.connectivity_metrics_to_load,
#             graph_bank_specs=spec.topology.graph_bank_specs,
#         )
#     return out
def prepare_segment_split_graphs(spec: "CAUEEGExperimentSpec") -> dict[str, list[Data]]:
    cm = _cm()

    _, train_rows, val_rows, test_rows = cm.load_caueeg_task_splits(spec.dataset_path, spec.task)
    split_to_rows = {"train": train_rows, "val": val_rows, "test": test_rows}
    split_to_resolved = {
        split: cm.resolve_h5_subject_ids_for_split(spec.h5_path, rows, split)
        for split, rows in split_to_rows.items()
    }

    all_subject_ids = []
    for rows in split_to_resolved.values():
        all_subject_ids.extend([sid for sid, _, _ in rows])

    entries = cm.load_h5_entries(
        spec.h5_path,
        all_subject_ids,
        feature_families=spec.feature_families,
        connectivity_metrics=spec.connectivity_metrics_to_load,
    )

    out = {"train": [], "val": [], "test": []}
    for split, rows in split_to_resolved.items():
        split_entries = {sid: entries[sid] for sid, _, _ in rows}
        out[split] = build_segment_graphs_for_entries(
            split_entries,
            feature_families=spec.feature_families,
            connectivity_metrics=spec.connectivity_metrics_to_load,
            graph_bank_specs=spec.topology.graph_bank_specs,
        )
    return out

def run_segment_mvgnn_experiment(spec: "CAUEEGExperimentSpec") -> dict[str, Any]:
    cm = _cm()

    if str(spec.level.graph_level).lower() != "segment":
        raise ValueError("segment_mvgnn requires spec.level.graph_level='segment'.")
    if str(spec.aggregation.strategy).lower() == "none":
        raise ValueError("segment_mvgnn is a subject-level bag model; use a bag-level aggregation strategy.")

    set_seed(spec.train.seed)
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")

    # split_bags = prepare_segment_split_bags(spec)
    # split_bags = prepare_segment_split_bags(spec)
    # train_loader = DataLoader(SubjectSegmentBagDataset(...), collate_fn=SubjectSegmentBagCollate(), ...)


    # collate = SubjectSegmentBagCollate()
    # train_loader = DataLoader(SubjectSegmentBagDataset(split_bags["train"]), batch_size=int(spec.train.batch_size), shuffle=True, collate_fn=collate, num_workers=int(spec.train.num_workers), pin_memory=True)
    # val_loader   = DataLoader(SubjectSegmentBagDataset(split_bags["val"]),   batch_size=int(spec.train.batch_size), shuffle=False, collate_fn=collate, num_workers=int(spec.train.num_workers), pin_memory=True)
    # test_loader  = DataLoader(SubjectSegmentBagDataset(split_bags["test"]),  batch_size=int(spec.train.batch_size), shuffle=False, collate_fn=collate, num_workers=int(spec.train.num_workers), pin_memory=True)
    from mil_utils import LabelAwareSubjectBagDataset

    split_graphs = prepare_segment_split_graphs(spec)

    # base_k = getattr(spec.aggregation, "base_k", None)
    # max_k_per_subject = getattr(spec.aggregation, "max_k_per_subject", 300)
    base_k = spec.aggregation.base_k
    max_k_per_subject = spec.aggregation.max_k_per_subject
    if base_k is None:
        train_dataset = LabelAwareSubjectBagDataset(
            split_graphs["train"],
            train=True,
            base_k=None,
            max_k_per_subject=None,
            seed=spec.train.seed,
            return_segment_ids=True,
        )
    else:
        train_dataset = LabelAwareSubjectBagDataset(
            split_graphs["train"],
            train=True,
            base_k=base_k,
            max_k_per_subject=max_k_per_subject,
            seed=spec.train.seed,
            return_segment_ids=True,
        )

    val_dataset = LabelAwareSubjectBagDataset(
        split_graphs["val"],
        train=False,
        eval_k_per_subject=None,
        seed=spec.train.seed,
        return_segment_ids=True,
    )

    test_dataset = LabelAwareSubjectBagDataset(
        split_graphs["test"],
        train=False,
        eval_k_per_subject=None,
        seed=spec.train.seed,
        return_segment_ids=True,
    )
    sample_seg = split_bags["train"][0].segments[0]
    num_classes = len(sorted({int(bag.label) for split in split_bags.values() for bag in split}))
    class_names = spec.class_names or cm.infer_class_names(spec.task, num_classes)

    model = SubjectSegmentMVGNN(
        num_nodes=int(sample_seg.node_features.shape[0]),
        num_node_features=int(sample_seg.node_features.shape[1]),
        num_views=int(sample_seg.adj_bank.shape[0]),
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
        subject_aggregation=str(spec.aggregation.strategy),
        subject_attn_dim=int(spec.aggregation.attn_dim),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.train.lr, weight_decay=spec.train.weight_decay)
    stopper = cm.EarlyStopper(spec.train.monitor, spec.train.monitor_mode, spec.train.patience)

    run_dir = ensure_dir(os.path.join(spec.output_root, make_run_name(spec.name, spec.model.family, spec.aggregation.strategy, timestamp=True)))
    best_ckpt_path = os.path.join(run_dir, "best_model.pt")

    history_train, history_val = [], []
    best_state = None

    for epoch in range(1, spec.train.epochs + 1):
        train_out = collect_epoch_outputs_segment(model, train_loader, device=device, optimizer=optimizer)
        val_out   = collect_epoch_outputs_segment(model, val_loader, device=device, optimizer=None)

        train_metrics = train_out["metrics"]
        val_metrics = val_out["metrics"]

        history_train.append({"epoch": epoch, "loss": float(train_out["loss"]), "accuracy": float(train_metrics["accuracy"]), "balanced_accuracy": float(train_metrics["balanced_accuracy"]), "macro_f1": float(train_metrics["macro_f1"])})
        history_val.append({"epoch": epoch, "loss": float(val_out["loss"]), "accuracy": float(val_metrics["accuracy"]), "balanced_accuracy": float(val_metrics["balanced_accuracy"]), "macro_f1": float(val_metrics["macro_f1"])})

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

        if should_stop:
            break

    best_loaded = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_loaded["model_state_dict"])

    train_final = collect_epoch_outputs_segment(model, train_loader, device=device, optimizer=None)
    val_final   = collect_epoch_outputs_segment(model, val_loader, device=device, optimizer=None)
    test_final  = collect_epoch_outputs_segment(model, test_loader, device=device, optimizer=None)

    train_df = build_subject_prediction_dataframe(train_final)
    val_df   = build_subject_prediction_dataframe(val_final)
    test_df  = build_subject_prediction_dataframe(test_final)

    train_df.to_csv(os.path.join(run_dir, "train_subject_predictions.csv"), index=False)
    val_df.to_csv(os.path.join(run_dir, "val_subject_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(run_dir, "test_subject_predictions.csv"), index=False)
    pd.DataFrame(history_train).to_csv(os.path.join(run_dir, "history_train.csv"), index=False)
    pd.DataFrame(history_val).to_csv(os.path.join(run_dir, "history_val.csv"), index=False)


    build_segment_attention_dataframe(epoch_out)
    import matplotlib.pyplot as plt
    import pandas as pd

    hist = pd.read_csv(os.path.join(run_dir, "bank_attention_history.csv"))
    val_hist = hist[hist["split"] == "val"]

    plt.figure(figsize=(10, 6))
    for view_name, sub in val_hist.groupby("view_name"):
        plt.plot(sub["epoch"], sub["mean_weight"], label=view_name)
    plt.xlabel("Epoch")
    plt.ylabel("Mean bank attention weight")
    plt.title("Validation bank attention over epochs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_bank_attention_over_epochs.png"), dpi=300)
    plt.close()
    final_val = pd.read_csv(os.path.join(run_dir, "val_bank_attention_summary_final.csv"))
    final_val = final_val.sort_values("mean_weight", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(final_val["view_name"], final_val["mean_weight"])
    plt.xlabel("Mean attention weight")
    plt.title("Final validation bank attention ranking")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_bank_attention_ranking.png"), dpi=300)
    plt.close()
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
                "view_names": list(sample_seg.view_names),
                "model_family": str(spec.model.family),
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
        "view_names": tuple(sample_seg.view_names),
        "spec": spec,
    }


def build_segment_graph_bank_specs(fixed_edge_pairs):
    return [
        # 1. pure structural prior
        {
            "name": "structural_local_binary",
            "topology_mode": "fixed",
            "edge_weight_mode": "binary",
            "edge_pairs": list(fixed_edge_pairs),
        },

        # 2-4. connectivity views
        {
            "name": "coherence_theta_topk4",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": "theta",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "coherence_alpha_topk4",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": "alpha",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "coherence_beta_topk4",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": "beta",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },

        # 5-6. phase-lag views
        # {
        #     "name": "pli_alpha_topk4",
        #     "topology_mode": "connectivity",
        #     "edge_weight_mode": "connectivity",
        #     "connectivity_metric": "pli",
        #     "band": "alpha",
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

        # 7-8. feature-induced topologies
        {
            "name": "feature_rbp_cosine_topk4",
            "topology_mode": "feature_induced",
            "edge_weight_mode": "topology_weight",
            "similarity": "cosine",
            "topology_kwargs": {"mode": "topk", "topk": 4},
            "feature_family": "relative_band_power",   # handled outside if needed
        },
        # {
        #     "name": "feature_hjorth_cosine_topk4",
        #     "topology_mode": "feature_induced",
        #     "edge_weight_mode": "topology_weight",
        #     "similarity": "cosine",
        #     "topology_kwargs": {"mode": "topk", "topk": 4},
        #     "feature_family": "hjorth",
        # },

        # 9-10. structural topology + connectivity weights
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

    fixed_edges = default_fixed_edge_pairs_19()
    graph_bank_specs = build_segment_graph_bank_specs(fixed_edges)

    spec = CAUEEGExperimentSpec(
        name="segment_mvgnn_gatv2_bank10",
        task="dementia",
        dataset_path=DATASET_PATH,
        h5_path=H5_PATH,
        feature_families=("relative_band_power", "hjorth", "statistical"),
        connectivity_metrics_to_load=("coherence", "pli", "wpli"),
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
            graph_bank_fusion_mode="summary_gated",
        ),
        aggregation=AggregationConfig(
            strategy="gated_attention_mil",
            attn_dim=128,
            train_max_instances_per_subject=64,
            eval_max_instances_per_subject=128,
        ),
        train=TrainConfig(
            batch_size=8,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=80,
            patience=20,
            monitor="balanced_accuracy",
            monitor_mode="max",
            seed=42,
            num_workers=0,
        ),
        output_root="./results_segment_mvgnn",
    )

    out = run_segment_mvgnn_experiment(spec)
    print(out["test_metrics"])