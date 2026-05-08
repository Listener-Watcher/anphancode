
from lib import *
from data_preparation import * 
from utils_all import *
from model import Gat_block, Cheb_block
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import copy
import config
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Sequence, Any
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphNorm, SAGEConv, GCN2Conv
from torch_geometric.utils import subgraph, dense_to_sparse, to_dense_adj, to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
# from hypergraph import build_hypergraphs_from_master_region_topology, EEGHypergraphNet_basic, EEGHypergraphNet
from sklearn.manifold import TSNE
import pickle
from collections import Counter
from torch_geometric.nn import (
    GATConv,
    ChebConv,
    BatchNorm
)
import math
import hashlib
from torch import Tensor
import os
import pandas as pd
import matplotlib.pyplot as plt

def supervised_contrastive_loss(
    embeddings,
    labels,
    temperature=0.2,
    eps=1e-8,
):
    """
    embeddings: [B, D] subject/bag embeddings
    labels:     [B]

    Positive pairs:
        different subjects in the same class

    Negative pairs:
        subjects from different classes
    """
    device = embeddings.device
    labels = labels.view(-1)

    z = F.normalize(embeddings, dim=1)
    logits = torch.matmul(z, z.T) / temperature  # [B, B]

    B = labels.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=device)

    same_class = labels[:, None].eq(labels[None, :])
    positive_mask = same_class & (~eye)

    # no positive pairs in this batch
    if positive_mask.sum() == 0:
        return embeddings.sum() * 0.0

    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    # exclude self from denominator
    denominator_mask = ~eye
    exp_logits = torch.exp(logits) * denominator_mask.float()

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + eps)

    pos_count = positive_mask.sum(dim=1)
    valid = pos_count > 0

    mean_log_prob_pos = (
        (positive_mask.float() * log_prob).sum(dim=1)
        / pos_count.clamp_min(1)
    )

    loss = -mean_log_prob_pos[valid].mean()
    return loss
def _to_numpy_attr(pyg_batch, name, total_graphs, default_value=-1):
    if not hasattr(pyg_batch, name):
        return np.full(total_graphs, default_value)

    x = getattr(pyg_batch, name)

    if torch.is_tensor(x):
        x = x.detach().cpu().view(-1).numpy()
    else:
        x = np.asarray(x).reshape(-1)

    if len(x) != total_graphs:
        return np.full(total_graphs, default_value)

    return x


@torch.no_grad()
def collect_attention_weights(model, loader, device, split_name="val"):
    """
    Collect segment-level attention weights from a MIL model.

    Assumes model(batch) returns:
        out["logits"]
        out["attn_list"]

    Assumes batch contains:
        batch["subject_ids"]
        batch["labels"]
        batch["bag_sizes"]
        batch["pyg_batch"]
    """
    model.eval()
    rows = []
    summary_rows = []

    for batch in loader:
        # move batch to device
        batch_device = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch_device[k] = v.to(device)
            elif hasattr(v, "to") and k == "pyg_batch":
                batch_device[k] = v.to(device)
            else:
                batch_device[k] = v

        out = model(batch_device)

        if "attn_list" not in out:
            raise KeyError(
                "Model output does not contain out['attn_list']. "
                "Make sure your SubjectMILClassifier returns attention weights."
            )

        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        labels = batch_device["labels"].detach().cpu().numpy()
        subject_ids = list(batch_device["subject_ids"])
        bag_sizes = batch_device["bag_sizes"].detach().cpu().numpy().astype(int)

        total_graphs = int(bag_sizes.sum())
        pyg_batch = batch_device["pyg_batch"]

        segment_ids = _to_numpy_attr(pyg_batch, "segment_id", total_graphs)
        start_samples = _to_numpy_attr(pyg_batch, "start_sample", total_graphs)

        global_start = 0

        for b, size in enumerate(bag_sizes):
            global_end = global_start + int(size)

            attn = out["attn_list"][b]
            attn = attn.detach().cpu().numpy().reshape(-1)

            if len(attn) != size:
                raise ValueError(
                    f"Attention length mismatch for subject {subject_ids[b]}: "
                    f"len(attn)={len(attn)}, bag_size={size}"
                )

            attn_sum = attn.sum()
            if attn_sum > 0:
                attn = attn / attn_sum

            top1 = float(attn.max())
            top3 = float(np.sort(attn)[-min(3, len(attn)):].sum())
            entropy = float(-(attn * np.log(attn + 1e-12)).sum())
            norm_entropy = float(entropy / np.log(len(attn))) if len(attn) > 1 else 0.0
            effective_n = float(1.0 / np.sum(attn ** 2))

            summary_rows.append({
                "split": split_name,
                "subject_id": subject_ids[b],
                "true_label": int(labels[b]),
                "pred_label": int(preds[b]),
                "bag_size": int(size),
                "top1_attention": top1,
                "top3_attention": top3,
                "effective_num_segments": effective_n,
                "effective_fraction": effective_n / float(size),
                "normalized_entropy": norm_entropy,
            })

            for local_i, global_i in enumerate(range(global_start, global_end)):
                row = {
                    "split": split_name,
                    "subject_id": subject_ids[b],
                    "true_label": int(labels[b]),
                    "pred_label": int(preds[b]),
                    "segment_rank_in_bag": int(local_i),
                    "segment_id": int(segment_ids[global_i]),
                    "start_sample": int(start_samples[global_i]),
                    "attention": float(attn[local_i]),
                }

                for c in range(probs.shape[1]):
                    row[f"prob_{c}"] = float(probs[b, c])

                rows.append(row)

            global_start = global_end

    attn_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(summary_rows)

    return attn_df, summary_df

def plot_attention_summary(summary_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Top-1 attention histogram
    plt.figure(figsize=(7, 5))
    plt.hist(summary_df["top1_attention"], bins=20)
    plt.xlabel("Max attention weight per subject")
    plt.ylabel("Number of subjects")
    plt.title("Distribution of Top-1 Attention")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_top1_attention.png"), dpi=300)
    plt.close()

    # 2. Effective number of segments
    plt.figure(figsize=(7, 5))
    plt.hist(summary_df["effective_num_segments"], bins=20)
    plt.xlabel("Effective number of attended segments")
    plt.ylabel("Number of subjects")
    plt.title("Effective Number of Segments")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_effective_num_segments.png"), dpi=300)
    plt.close()

    # 3. Normalized entropy
    plt.figure(figsize=(7, 5))
    plt.hist(summary_df["normalized_entropy"], bins=20)
    plt.xlabel("Normalized attention entropy")
    plt.ylabel("Number of subjects")
    plt.title("Attention Entropy")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "hist_attention_entropy.png"), dpi=300)
    plt.close()


def plot_attention_per_subject(attn_df, out_dir, max_subjects=None, sort_by_attention=True):
    subject_dir = os.path.join(out_dir, "per_subject")
    os.makedirs(subject_dir, exist_ok=True)

    subject_ids = attn_df["subject_id"].unique().tolist()

    if max_subjects is not None:
        subject_ids = subject_ids[:max_subjects]

    for sid in subject_ids:
        sdf = attn_df[attn_df["subject_id"] == sid].copy()

        if sort_by_attention:
            sdf = sdf.sort_values("attention", ascending=False)
            x = np.arange(len(sdf))
            xlabel = "Segments sorted by attention"
        else:
            sdf = sdf.sort_values("segment_rank_in_bag")
            x = sdf["segment_rank_in_bag"].to_numpy()
            xlabel = "Segment index in bag"

        true_label = int(sdf["true_label"].iloc[0])
        pred_label = int(sdf["pred_label"].iloc[0])
        max_attn = float(sdf["attention"].max())

        plt.figure(figsize=(10, 4))
        plt.bar(x, sdf["attention"].to_numpy())
        plt.xlabel(xlabel)
        plt.ylabel("Attention weight")
        plt.title(
            f"Subject {sid} | true={true_label}, pred={pred_label}, "
            f"max_attn={max_attn:.3f}"
        )
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        safe_sid = str(sid).replace("/", "_")
        plt.savefig(os.path.join(subject_dir, f"attention_subject_{safe_sid}.png"), dpi=300)
        plt.close()

def build_graphs_from_payload_multiband(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    sparse_adj_reduction="mean",   # "mean" over 5 bands for the PyG graph edges
):
    """
    Build one PyG graph per window, but also attach a multiband connectivity tensor.

    Expected:
      payload[sid]["features"][fam]         -> [W, N, F_fam]
      payload[sid]["connectivity"][metric]  -> [W, B, N, N]  (for this CNN case)
                                              or [W, N, N]   (will be promoted to B=1)

    Output graph fields:
      g.x          : [N, F_total]
      g.edge_index : sparse edges built from reduced adjacency
      g.edge_weight
      g.edge_attr
      g.adj        : [N, N] reduced dense adjacency
      g.conn_stack : [B, N, N] full multiband tensor for CNN branch
    """
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        # ---------- node features ----------
        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")

            xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
            if xfam.ndim != 3:
                raise ValueError(
                    f"Feature family {fam!r} for subject {sid!r} must have shape [W, N, F], got {xfam.shape}"
                )

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                    raise ValueError(
                        f"Feature family {fam!r} for subject {sid!r} has incompatible shape {xfam.shape}; "
                        f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                    )

            feat_list.append(xfam)

        if len(feat_list) == 0:
            raise ValueError("feature_families is empty")

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
        num_windows = node_x_all.shape[0]
        num_nodes = node_x_all.shape[1]

        # ---------- metadata ----------
        seg_ids = np.asarray(subj.get("segment_id", np.arange(num_windows)), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(num_windows, -1)), dtype=np.int64)

        if len(seg_ids) != num_windows:
            raise ValueError(
                f"segment_id length mismatch for subject {sid!r}: got {len(seg_ids)}, expected {num_windows}"
            )
        if len(start_samples) != num_windows:
            raise ValueError(
                f"start_sample length mismatch for subject {sid!r}: got {len(start_samples)}, expected {num_windows}"
            )

        # ---------- connectivity source ----------
        if edge_source != "connectivity":
            raise ValueError("For this simple multiband CNN version, use edge_source='connectivity'.")

        if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
            raise KeyError(
                f"payload[{sid!r}]['connectivity'] missing metric {connectivity_metric!r}"
            )

        conn_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

        # allow either [W, N, N] or [W, B, N, N]
        if conn_all.ndim == 3:
            conn_all = conn_all[:, None, :, :]   # -> [W, 1, N, N]
        elif conn_all.ndim != 4:
            raise ValueError(
                f"Connectivity tensor for subject {sid!r} must have shape [W, B, N, N] or [W, N, N], got {conn_all.shape}"
            )

        if conn_all.shape[0] != num_windows:
            raise ValueError(
                f"Connectivity window count mismatch for subject {sid!r}: {conn_all.shape[0]} vs {num_windows}"
            )
        if conn_all.shape[2] != num_nodes or conn_all.shape[3] != num_nodes:
            raise ValueError(
                f"Connectivity node count mismatch for subject {sid!r}: {conn_all.shape} vs num_nodes={num_nodes}"
            )

        # reduce bands -> one dense adjacency for PyG graph edges
        if sparse_adj_reduction == "mean":
            adj_all = conn_all.mean(axis=1).astype(np.float32)   # [W, N, N]
        else:
            raise ValueError(f"Unsupported sparse_adj_reduction={sparse_adj_reduction!r}")

        # ---------- build one graph per window ----------
        for w in range(num_windows):
            x = node_x_all[w]                      # [N, F_total]
            conn_stack = conn_all[w].copy()        # [B, N, N]
            adj = adj_all[w].copy()                # [N, N]

            # clean multiband tensor
            conn_stack = np.nan_to_num(conn_stack, nan=0.0, posinf=0.0, neginf=0.0)
            if symmetrize_adj:
                conn_stack = 0.5 * (conn_stack + np.transpose(conn_stack, (0, 2, 1)))
            if zero_diagonal:
                for b in range(conn_stack.shape[0]):
                    np.fill_diagonal(conn_stack[b], 0.0)

            # clean reduced adjacency for graph edges
            if symmetrize_adj:
                adj = 0.5 * (adj + adj.T)
            if zero_diagonal:
                np.fill_diagonal(adj, 0.0)
            adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

            edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))

            g = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long),
            )

            g.edge_weight = edge_weight
            g.edge_attr = edge_weight.view(-1, 1)

            if attach_dense_adj:
                g.adj = torch.tensor(adj, dtype=torch.float32)

            g.conn_stack = torch.tensor(conn_stack, dtype=torch.float32)   # [B, N, N]

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            graphs.append(g)

    return graphs

def build_graphs_from_payload_region_clique(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    edge_source="connectivity",   # "connectivity" or "aligned_adj"
    channel_names=None,
    region_to_channels=None,
    standardize_features=True,
    hyperedge_weight_mode="mean_abs_adj",
    clique_combine_mode="sum",
    keep_empty_hyperedges=False,
    attach_dense_adj=True,
):
    """
    Payload -> region hypergraph weights -> clique adjacency -> ordinary PyG graphs.
    """

    if region_to_channels is None:
        raise ValueError("region_to_channels must be provided.")
    if channel_names is None:
        raise ValueError("channel_names must be provided.")

    region_names, hyperedge_members, node_to_region_mask = _build_region_members_from_channel_names(
        channel_names=channel_names,
        region_to_channels=region_to_channels,
        keep_empty_hyperedges=keep_empty_hyperedges,
    )

    n_nodes = len(channel_names)
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        # ---------- node features ----------
        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")

            xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
            if xfam.ndim != 3:
                raise ValueError(
                    f"Feature family {fam!r} for subject {sid!r} must have shape [W, N, F], got {xfam.shape}"
                )

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                    raise ValueError(
                        f"Feature family {fam!r} for subject {sid!r} has incompatible shape {xfam.shape}; "
                        f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                    )

            feat_list.append(xfam)

        if len(feat_list) == 0:
            raise ValueError("feature_families is empty")

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
        num_windows = node_x_all.shape[0]

        if node_x_all.shape[1] != n_nodes:
            raise ValueError(
                f"Node count mismatch for {sid!r}: features use {node_x_all.shape[1]} nodes, "
                f"but channel_names has {n_nodes} nodes."
            )

        # ---------- adjacency source ----------
        if edge_source == "connectivity":
            if connectivity_metric is None:
                raise ValueError("connectivity_metric must be provided when edge_source='connectivity'")
            if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
                raise KeyError(
                    f"payload[{sid!r}]['connectivity'] missing metric {connectivity_metric!r}"
                )
            adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

            # support [W, B, N, N] if band slicing was not done earlier
            if adj_all.ndim == 4:
                if connectivity_band is None:
                    raise ValueError(
                        f"Connectivity for {sid!r}/{connectivity_metric!r} is banded [W,B,N,N], "
                        "but connectivity_band is None."
                    )
                band_idx = int(connectivity_band) if not isinstance(connectivity_band, str) else connectivity_band
                if isinstance(band_idx, str):
                    raise ValueError(
                        "String band selection is not supported here unless you also pass band-name metadata. "
                        "Prefer slicing earlier in load_h5_payload_for_subjects(...)."
                    )
                adj_all = adj_all[:, band_idx]

        elif edge_source == "aligned_adj":
            if subj.get("aligned_adj", None) is None:
                raise ValueError(
                    f"edge_source='aligned_adj' but payload[{sid!r}]['aligned_adj'] is None"
                )
            adj_all = np.asarray(subj["aligned_adj"], dtype=np.float32)

        else:
            raise ValueError(f"Unsupported edge_source={edge_source!r}")

        if adj_all.ndim != 3:
            raise ValueError(
                f"Adjacency tensor for subject {sid!r} must have shape [W, N, N], got {adj_all.shape}"
            )
        if adj_all.shape[0] != num_windows:
            raise ValueError(
                f"Adjacency window count mismatch for subject {sid!r}: {adj_all.shape[0]} vs {num_windows}"
            )
        if adj_all.shape[1] != n_nodes or adj_all.shape[2] != n_nodes:
            raise ValueError(
                f"Adjacency node count mismatch for subject {sid!r}: {adj_all.shape} vs expected {n_nodes}"
            )

        seg_ids = np.asarray(subj.get("segment_id", np.arange(num_windows)), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(num_windows, -1)), dtype=np.int64)

        # ---------- build one graph per window ----------
        for w in range(num_windows):
            x = node_x_all[w]                  # [N, F]
            adj_orig = adj_all[w].copy()       # [N, N]
            adj_orig = np.nan_to_num(adj_orig, nan=0.0, posinf=0.0, neginf=0.0)

            if standardize_features:
                x = _zscore_per_feature(x)

            hedge_w = _compute_region_hyperedge_weights(
                adj=adj_orig,
                hyperedge_members=hyperedge_members,
                hyperedge_weight_mode=hyperedge_weight_mode,
            )

            clique_adj = hypergraph_to_clique_adj(
                num_nodes=n_nodes,
                hyperedge_members=hyperedge_members,
                hyperedge_weight=hedge_w,
                combine_mode=clique_combine_mode,
                remove_self_loops=True,
            )

            edge_index, edge_weight = dense_to_sparse(torch.from_numpy(clique_adj))
            edge_weight = edge_weight.to(torch.float32)

            g = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index.long(),
                y=torch.tensor([label], dtype=torch.long),
            )

            # keep compatibility with your normal graph path
            g.edge_weight = edge_weight
            g.edge_attr = edge_weight.view(-1, 1)

            if attach_dense_adj:
                g.adj = torch.tensor(clique_adj, dtype=torch.float32)

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            # debug / inspection
            g.region_names = list(region_names)
            g.node_to_region_mask = torch.tensor(node_to_region_mask, dtype=torch.float32)
            g.hyperedge_weight = torch.tensor(hedge_w, dtype=torch.float32)

            graphs.append(g)

    return graphs

# =========================================================
# Topology helpers for build_graphs_from_payload
# =========================================================
from typing import Optional, Sequence, Tuple, List, Union

EdgeEndpoint = Union[int, str]
EdgeSpec = Sequence[Tuple[EdgeEndpoint, EdgeEndpoint]]

def make_mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float):
    layers = []
    prev = input_dim
    for h in hidden_dims:
        layers.extend([
            nn.Linear(prev, h),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev = h
    return nn.Sequential(*layers), prev
    
def _to_2d_features(x) -> torch.Tensor:
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        raise ValueError(f"Expected node features [N, F], got shape {tuple(x.shape)}")
    return x


def make_identity_adj(n: int) -> torch.Tensor:
    return torch.eye(n, dtype=torch.float32)


def make_random_adj_like_with_weights(adj: torch.Tensor, undirected: bool = True) -> torch.Tensor:
    n = int(adj.shape[0])
    out = torch.rand((n, n), dtype=torch.float32)
    if undirected:
        out = 0.5 * (out + out.T)
    out.fill_diagonal_(0.0)
    return out


def permute_graph_consistently(x: torch.Tensor, adj: torch.Tensor):
    n = int(x.shape[0])
    perm = torch.randperm(n)
    return x[perm], adj[perm][:, perm], perm


def permute_adj_only(adj: torch.Tensor):
    n = int(adj.shape[0])
    perm = torch.randperm(n)
    return adj[perm][:, perm], perm


def dense_adj_to_candidate_edges(adj: torch.Tensor, undirected: bool = True):
    """
    Convert dense adjacency to candidate undirected edges + weights.

    Returns
    -------
    edge_index : torch.LongTensor [2, E_sparse]
        Symmetric sparse edge_index for the current dense adjacency.
    edge_attr : torch.FloatTensor [E_sparse]
        Symmetric edge weights matching edge_index.
    edge_list : list[tuple[int, int]]
        Undirected edge list with i < j.
    edge_weights : list[float]
        One weight per undirected edge in edge_list.
    """
    adj = torch.as_tensor(adj, dtype=torch.float32)
    n = int(adj.shape[0])

    if undirected:
        iu = torch.triu_indices(n, n, offset=1)
        w = adj[iu[0], iu[1]]
        mask = torch.abs(w) > 1e-12

        row = iu[0][mask]
        col = iu[1][mask]
        w = w[mask]

        edge_list = [(int(i), int(j)) for i, j in zip(row.tolist(), col.tolist())]
        edge_weights = [float(x) for x in w.tolist()]

        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float32)
        else:
            edge_index = torch.cat(
                [
                    torch.stack([row, col], dim=0),
                    torch.stack([col, row], dim=0),
                ],
                dim=1,
            ).long()
            edge_attr = torch.cat([w, w], dim=0).float()

    else:
        row, col = torch.nonzero(torch.abs(adj) > 1e-12, as_tuple=True)
        w = adj[row, col]
        edge_index = torch.stack([row, col], dim=0).long()
        edge_attr = w.float()
        edge_list = [(int(i), int(j)) for i, j in zip(row.tolist(), col.tolist())]
        edge_weights = [float(x) for x in w.tolist()]

    return edge_index, edge_attr, edge_list, edge_weights


def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    channel_names: Optional[Sequence[str]],
    n_channels: int,
) -> set[tuple[int, int]]:
    if fixed_edges is None:
        return set()

    out = set()

    if channel_names is not None:
        name_to_idx = {str(ch): i for i, ch in enumerate(channel_names)}
    else:
        name_to_idx = None

    for a, b in fixed_edges:
        if isinstance(a, str) or isinstance(b, str):
            if name_to_idx is None:
                raise ValueError("fixed_edges contains channel names but channel_names was not provided.")
            if a not in name_to_idx or b not in name_to_idx:
                raise ValueError(f"Unknown channel in fixed_edges: {(a, b)}")
            i, j = name_to_idx[a], name_to_idx[b]
        else:
            i, j = int(a), int(b)

        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(a, b)} resolved to invalid node indices {(i, j)}")

        if i == j:
            continue

        if i > j:
            i, j = j, i

        out.add((i, j))

    return out


def _maximum_spanning_tree_edges(
    edge_list: Sequence[Tuple[int, int]],
    edge_weights: Sequence[float],
    n_channels: int,
) -> set[tuple[int, int]]:
    """
    Kruskal maximum spanning forest on undirected edges.
    Uses raw weights in descending order.
    If you prefer abs-weight MST, change the sort key to abs(x[2]).
    """
    parent = list(range(n_channels))
    rank = [0] * n_channels

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    edges = [
        (int(i), int(j), float(w))
        for (i, j), w in zip(edge_list, edge_weights)
    ]
    edges.sort(key=lambda x: x[2], reverse=True)

    chosen = set()
    for i, j, _ in edges:
        if union(i, j):
            if i > j:
                i, j = j, i
            chosen.add((i, j))

    return chosen


def _topk_edges(
    edge_list: Sequence[Tuple[int, int]],
    edge_weights: Sequence[float],
    topk: Optional[int] = None,
    top_percent: Optional[float] = None,
) -> set[tuple[int, int]]:
    if len(edge_list) == 0:
        return set()

    if topk is None and top_percent is None:
        raise ValueError("Provide topk or top_percent for filter_method='topk'")

    pairs = [(tuple(map(int, e)), float(w)) for e, w in zip(edge_list, edge_weights)]
    pairs.sort(key=lambda x: x[1], reverse=True)

    if top_percent is not None:
        if not (0 < top_percent <= 1):
            raise ValueError(f"top_percent must be in (0, 1], got {top_percent}")
        k = max(1, int(np.ceil(len(pairs) * top_percent)))
    else:
        k = int(topk)
        if k < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

    k = min(k, len(pairs))
    return {tuple(sorted(p[0])) for p in pairs[:k]}


def _build_dense_adj_from_selected_edges(
    selected_edges: set[tuple[int, int]],
    edge_to_weight: dict[tuple[int, int], float],
    n_channels: int,
    undirected: bool = True,
) -> np.ndarray:
    adj = np.zeros((n_channels, n_channels), dtype=np.float32)

    for i, j in selected_edges:
        w = float(edge_to_weight[(i, j)])
        adj[i, j] = w
        if undirected:
            adj[j, i] = w

    np.fill_diagonal(adj, 0.0)
    return adj


def apply_edge_filter(
    edge_index,
    edge_attr,
    edge_list,
    edge_weights,
    n_channels: int,
    filter_method: str = "mst",
    topk: Optional[int] = 2,
    top_percent: Optional[float] = None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    undirected: bool = True,
):
    """
    Supported filter_method:
      - 'full' / 'none'
      - 'mst'
      - 'fixed'
      - 'topk'
      - 'reconnect' : fixed U mst
      - 'combined'  : fixed U topk
      - 'overlap'   : fixed ∩ topk

    Note:
      If your previous reconnect/combined/overlap semantics were different,
      keep this structure and replace only the set-combination logic below.
    """
    method = str(filter_method).lower()
    edge_to_weight = {
        tuple(sorted((int(i), int(j)))): float(w)
        for (i, j), w in zip(edge_list, edge_weights)
    }

    full_edges = set(edge_to_weight.keys())
    fixed_set = _normalize_fixed_edges(
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        n_channels=n_channels,
    )

    if method in {"full", "none", "dense", "all"}:
        selected = full_edges

    elif method in {"mst", "maxst", "maximum_spanning_tree"}:
        selected = _maximum_spanning_tree_edges(edge_list, edge_weights, n_channels=n_channels)

    elif method == "fixed":
        selected = fixed_set

    elif method == "topk":
        selected = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)

    elif method == "reconnect":
        mst_set = _maximum_spanning_tree_edges(edge_list, edge_weights, n_channels=n_channels)
        selected = fixed_set | mst_set

    elif method == "combined":
        topk_set = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)
        selected = fixed_set | topk_set

    elif method == "overlap":
        topk_set = _topk_edges(edge_list, edge_weights, topk=topk, top_percent=top_percent)
        selected = fixed_set & topk_set

    else:
        raise ValueError(f"Unknown filter_method={filter_method!r}")

    # keep only edges that actually exist in the candidate graph
    selected = {e for e in selected if e in edge_to_weight}

    final_adj = _build_dense_adj_from_selected_edges(
        selected_edges=selected,
        edge_to_weight=edge_to_weight,
        n_channels=n_channels,
        undirected=undirected,
    )

    final_edge_index, final_edge_weight = dense_to_sparse(torch.tensor(final_adj, dtype=torch.float32))
    return final_edge_index.long(), final_edge_weight.float(), final_adj

def _stable_int_from_string(x: str) -> int:
    """
    Stable integer hash from a string.
    Do NOT use Python's built-in hash(), because it is randomized across runs.
    """
    s = str(x).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)
def _get_label_tensor(batch_dict):
    if "labels" in batch_dict:
        return batch_dict["labels"]
    if "y" in batch_dict:
        return batch_dict["y"]
    if "bag_labels" in batch_dict:
        return batch_dict["bag_labels"]
    raise KeyError("Cannot find labels in batch_dict")

def _get_subject_ids(batch_dict, batch_size):
    if "subject_ids" in batch_dict:
        return list(batch_dict["subject_ids"])
    if "subject_id" in batch_dict:
        x = batch_dict["subject_id"]
        return list(x) if isinstance(x, (list, tuple)) else [x] * batch_size
    return [f"subject_{i}" for i in range(batch_size)]

def collect_subject_embeddings(model, loader, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch_dict in loader:
            batch_dict = move_batch_to_device(batch_dict, device)
            out = model(batch_dict)

            bag_emb = out["bag_emb"]          # [B, D]
            logits = out["logits"]            # [B, C]
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            labels = _get_label_tensor(batch_dict)
            B = labels.shape[0]
            subject_ids = _get_subject_ids(batch_dict, B)

            for i in range(B):
                rows.append({
                    "subject_id": subject_ids[i],
                    "label": int(labels[i].detach().cpu().item()),
                    "pred": int(preds[i].detach().cpu().item()),
                    "prob": probs[i].detach().cpu().numpy(),
                    "embedding": bag_emb[i].detach().cpu().numpy(),
                })

    return rows

def move_batch_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to"):   # handles PyG Batch
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out
class LabelAwareSubjectBagDataset(Dataset):
    """
    Subject-level MIL dataset with deterministic per-subject segment subsampling.

    Each dataset item is still:
        {
            "subject_id": ...,
            "label": ...,
            "graphs": [...]
        }

    But for training, only k_label segments are sampled per subject,
    where k_label can depend on the subject label.

    Sampling is deterministic given:
        seed + epoch + subject_id

    So:
      - same seed + same epoch => same sampled segments
      - different epoch => different sampled segments, but reproducible
    """

    def __init__(
        self,
        graphs,
        train: bool = True,
        base_k: int = None,
        k_by_label: dict = None,
        target_segments_per_class: int = None,
        max_k_per_subject: int = None,
        eval_k_per_subject: int = None,
        seed: int = 42,
        sort_graphs_by: str = "segment_id",   # for deterministic graph ordering
        return_segment_ids: bool = False,
    ):
        self.train = train
        self.seed = int(seed)
        self.epoch = 0
        self.return_segment_ids = return_segment_ids
        self.eval_k_per_subject = eval_k_per_subject

        self.subject_to_graphs = defaultdict(list)
        self.subject_to_label = {}

        for g in graphs:
            sid = g.subject_id
            y = int(g.y.item()) if g.y.numel() == 1 else int(g.y[0].item())

            self.subject_to_graphs[sid].append(g)

            if sid in self.subject_to_label and self.subject_to_label[sid] != y:
                raise ValueError(f"Subject {sid} has inconsistent labels.")
            self.subject_to_label[sid] = y

        # Stable subject order
        self.subject_ids = sorted(self.subject_to_graphs.keys())
        self.subject_labels = [self.subject_to_label[sid] for sid in self.subject_ids]

        # Stable graph order inside each subject
        for sid in self.subject_ids:
            if sort_graphs_by == "segment_id":
                self.subject_to_graphs[sid] = sorted(
                    self.subject_to_graphs[sid],
                    key=lambda g: (
                        getattr(g, "segment_id", 0),
                        getattr(g, "start_sample", 0) if getattr(g, "start_sample", None) is not None else 0,
                    ),
                )
            elif sort_graphs_by == "start_sample":
                self.subject_to_graphs[sid] = sorted(
                    self.subject_to_graphs[sid],
                    key=lambda g: (
                        getattr(g, "start_sample", 0) if getattr(g, "start_sample", None) is not None else 0,
                        getattr(g, "segment_id", 0),
                    ),
                )
            else:
                raise ValueError(f"Unsupported sort_graphs_by={sort_graphs_by}")

        if len(graphs) == 0:
            raise ValueError("graphs is empty.")

        self.num_node_features = graphs[0].x.shape[-1]
        self.summary_input_dim = graphs[0].summary_feat.numel() if hasattr(graphs[0], "summary_feat") else None
        self.num_nodes = graphs[0].x.shape[0]

        # make sure all graphs have the same number of nodes
        for i, g in enumerate(graphs):
            if g.x.shape[0] != self.num_nodes:
                raise ValueError(
                    f"RawNodeEdgeMLPEncoder requires fixed num_nodes, "
                    f"but graph {i} has {g.x.shape[0]} nodes while expected {self.num_nodes}."
                )

        # label -> list[subject_id]
        self.label_to_subjects = defaultdict(list)
        for sid in self.subject_ids:
            self.label_to_subjects[self.subject_to_label[sid]].append(sid)

        # For train mode: compute k per label
        if self.train:
            if k_by_label is None:
                if base_k is None:
                    raise ValueError("Provide base_k or k_by_label for training dataset.")

                n_subjects_per_label = {
                    label: len(sids) for label, sids in self.label_to_subjects.items()
                }

                if target_segments_per_class is None:
                    max_subjects = max(n_subjects_per_label.values())
                    target_segments_per_class = max_subjects * base_k

                self.k_by_label = {}
                for label, n_subj in n_subjects_per_label.items():
                    k_label = math.ceil(target_segments_per_class / n_subj)
                    if max_k_per_subject is not None:
                        k_label = min(k_label, max_k_per_subject)
                    self.k_by_label[label] = k_label
            else:
                self.k_by_label = dict(k_by_label)
                if max_k_per_subject is not None:
                    for label in self.k_by_label:
                        self.k_by_label[label] = min(self.k_by_label[label], max_k_per_subject)
        else:
            self.k_by_label = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.subject_ids)

    def _sample_graphs_for_subject(self, sid):
        graphs = self.subject_to_graphs[sid]
        label = self.subject_to_label[sid]

        # ---------- train ----------
        if self.train:
            k = self.k_by_label[label]

            # subject-specific deterministic RNG
            subject_seed = self.seed + 1000003 * self.epoch + _stable_int_from_string(sid)
            rng = random.Random(subject_seed)

            n = len(graphs)
            if n >= k:
                chosen_idx = rng.sample(range(n), k)
            else:
                chosen_idx = list(range(n)) + [rng.randrange(n) for _ in range(k - n)]

            chosen_graphs = [graphs[i] for i in chosen_idx]
            return chosen_graphs, chosen_idx

        # ---------- eval ----------
        if self.eval_k_per_subject is None:
            chosen_idx = list(range(len(graphs)))
            return graphs, chosen_idx

        k = self.eval_k_per_subject
        n = len(graphs)

        # deterministic eval subset
        subject_seed = self.seed + _stable_int_from_string(sid)
        rng = random.Random(subject_seed)

        if n >= k:
            chosen_idx = rng.sample(range(n), k)
        else:
            chosen_idx = list(range(n)) + [rng.randrange(n) for _ in range(k - n)]

        chosen_graphs = [graphs[i] for i in chosen_idx]
        return chosen_graphs, chosen_idx

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        graphs, chosen_idx = self._sample_graphs_for_subject(sid)

        out = {
            "subject_id": sid,
            "label": self.subject_to_label[sid],
            "graphs": graphs,
        }

        if self.return_segment_ids:
            seg_ids = []
            for g in graphs:
                seg_ids.append(getattr(g, "segment_id", None))
            out["segment_ids"] = seg_ids
            out["chosen_idx"] = chosen_idx

        return out


import itertools
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from master_builder import load_feature_family, load_connectivity_metric


def _zscore_per_feature(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Per-graph standardization across nodes, feature by feature.
    x: [num_nodes, num_node_features]
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return ((x - mu) / sd).astype(np.float32)


def _build_region_members_from_channel_names(
    channel_names,
    region_to_channels,
    keep_empty_hyperedges: bool = False,
):
    """
    Build fixed region hyperedges once from channel names.
    Returns:
        region_names: list[str]
        hyperedge_members: list[list[int]]
        node_to_region_mask: [N, R] float32
    """
    name_to_idx = {name: i for i, name in enumerate(channel_names)}

    region_names = []
    hyperedge_members = []

    for region_name, ch_list in region_to_channels.items():
        members = []
        for ch in ch_list:
            if ch in name_to_idx:
                members.append(name_to_idx[ch])

        members = sorted(set(members))

        if len(members) == 0:
            if keep_empty_hyperedges:
                region_names.append(region_name)
                hyperedge_members.append([])
            continue

        region_names.append(region_name)
        hyperedge_members.append(members)

    if len(region_names) == 0:
        raise ValueError("No valid region hyperedges could be built.")

    n_nodes = len(channel_names)
    node_to_region_mask = np.zeros((n_nodes, len(region_names)), dtype=np.float32)
    for h_idx, members in enumerate(hyperedge_members):
        for node_idx in members:
            node_to_region_mask[node_idx, h_idx] = 1.0

    return region_names, hyperedge_members, node_to_region_mask


def _compute_region_hyperedge_weights(
    adj: np.ndarray,
    hyperedge_members,
    hyperedge_weight_mode: str = "mean_adj",
) -> np.ndarray:
    """
    Compute one scalar weight per region/hyperedge from the original pairwise adjacency.

    Modes:
      - mean_adj
      - mean_abs_adj
      - ones
    """
    adj = np.asarray(adj, dtype=np.float32)
    weights = []

    for members in hyperedge_members:
        if len(members) <= 1:
            weights.append(1.0)
            continue

        vals = []
        for i, j in itertools.combinations(members, 2):
            vals.append(adj[i, j])

        vals = np.asarray(vals, dtype=np.float32)

        if vals.size == 0:
            w = 1.0
        elif hyperedge_weight_mode == "mean_adj":
            w = float(vals.mean())
        elif hyperedge_weight_mode == "mean_abs_adj":
            w = float(np.abs(vals).mean())
        elif hyperedge_weight_mode == "ones":
            w = 1.0
        else:
            raise ValueError(f"Unknown hyperedge_weight_mode={hyperedge_weight_mode}")

        weights.append(w)

    return np.asarray(weights, dtype=np.float32)


def hypergraph_to_clique_adj(
    num_nodes: int,
    hyperedge_members,
    hyperedge_weight,
    combine_mode: str = "sum",
    remove_self_loops: bool = True,
) -> np.ndarray:
    """
    Convert region hyperedges into a pairwise clique-expanded adjacency.

    If two nodes co-occur in a hyperedge h, connect them with weight w_h.
    If they co-occur in multiple hyperedges, combine by:
      - sum
      - max
    """
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    hyperedge_weight = np.asarray(hyperedge_weight, dtype=np.float32).reshape(-1)
    if len(hyperedge_weight) != len(hyperedge_members):
        raise ValueError("hyperedge_weight and hyperedge_members size mismatch.")

    for h_idx, members in enumerate(hyperedge_members):
        if len(members) <= 1:
            continue

        w = float(hyperedge_weight[h_idx])

        for i, j in itertools.combinations(members, 2):
            if combine_mode == "sum":
                adj[i, j] += w
                adj[j, i] += w
            elif combine_mode == "max":
                adj[i, j] = max(adj[i, j], w)
                adj[j, i] = max(adj[j, i], w)
            else:
                raise ValueError(f"Unknown combine_mode={combine_mode}")

    if remove_self_loops:
        np.fill_diagonal(adj, 0.0)

    return adj.astype(np.float32)


def build_graphs_from_master_h5_region_clique(
    h5_path: str,
    feature_families,
    connectivity_metric: str,
    connectivity_band=None,          # int or str or None
    channel_names=None,              # required for stable region definition
    region_to_channels=None,         # dict: {"frontal":[...], ...}
    subject_ids=None,
    standardize_features: bool = True,
    hyperedge_weight_mode: str = "mean_abs_adj",
    clique_combine_mode: str = "sum",
    keep_empty_hyperedges: bool = False,
):
    """
    H5 -> true region hypergraph weights -> clique adjacency -> ordinary PyG graph Data

    Output graphs are compatible with the existing RawNodeEdgeMLPEncoder:
      g.x          : [N, F]
      g.edge_index : [2, E]
      g.edge_attr  : [E]
      g.y          : [1]
      g.subject_id, g.segment_id, g.start_sample, ...
    """
    if region_to_channels is None:
        raise ValueError("region_to_channels must be provided.")
    if channel_names is None:
        raise ValueError("channel_names must be provided.")

    feature_families = list(feature_families)
    if len(feature_families) == 0:
        raise ValueError("feature_families is empty.")

    # ---- load feature groups ----
    feature_payloads = {}
    for i, fam in enumerate(feature_families):
        feature_payloads[fam] = load_feature_family(
            h5_path,
            fam,
            subject_ids=subject_ids,
            include_raw_metadata=(i == 0),   # only first one needs metadata
        )

    # ---- load selected connectivity metric/band ----
    conn_payload = load_connectivity_metric(
        h5_path,
        connectivity_metric,
        subject_ids=subject_ids,
        band=connectivity_band,
    )

    # ---- fixed region topology once ----
    region_names, hyperedge_members, node_to_region_mask = _build_region_members_from_channel_names(
        channel_names=channel_names,
        region_to_channels=region_to_channels,
        keep_empty_hyperedges=keep_empty_hyperedges,
    )

    n_nodes = len(channel_names)
    graphs = []

    # iterate by subjects available in first feature family
    for sid in feature_payloads[feature_families[0]].keys():
        meta_ref = feature_payloads[feature_families[0]][sid]
        label = int(meta_ref["label"])

        feat_list = []
        for fam in feature_families:
            if sid not in feature_payloads[fam]:
                raise KeyError(f"Missing subject {sid} in feature family {fam}")
            feat_list.append(np.asarray(feature_payloads[fam][sid]["values"], dtype=np.float32))

        # [W, N, F_total]
        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)

        if sid not in conn_payload:
            raise KeyError(f"Missing subject {sid} in connectivity metric {connectivity_metric}")
        adj_all = np.asarray(conn_payload[sid]["values"], dtype=np.float32)

        if node_x_all.ndim != 3:
            raise ValueError(f"Expected node features [W,N,F], got shape {node_x_all.shape} for {sid}")
        if adj_all.ndim != 3:
            raise ValueError(f"Expected adjacency [W,N,N], got shape {adj_all.shape} for {sid}")

        if node_x_all.shape[0] != adj_all.shape[0]:
            raise ValueError(
                f"Window count mismatch for {sid}: "
                f"features={node_x_all.shape[0]}, connectivity={adj_all.shape[0]}"
            )
        if node_x_all.shape[1] != n_nodes or adj_all.shape[1] != n_nodes or adj_all.shape[2] != n_nodes:
            raise ValueError(
                f"Node/channel mismatch for {sid}: "
                f"expected {n_nodes} nodes from channel_names."
            )

        seg_ids = np.asarray(meta_ref["segment_id"])
        start_samples = np.asarray(meta_ref["start_sample"])

        for w in range(node_x_all.shape[0]):
            x = node_x_all[w]          # [N, F]
            adj_orig = adj_all[w]      # [N, N]

            if standardize_features:
                x = _zscore_per_feature(x)

            # 1) region hyperedge weights from original pairwise adjacency
            hedge_w = _compute_region_hyperedge_weights(
                adj=adj_orig,
                hyperedge_members=hyperedge_members,
                hyperedge_weight_mode=hyperedge_weight_mode,
            )

            # 2) clique expansion -> pairwise topology for current RawNodeEdgeMLPEncoder
            clique_adj = hypergraph_to_clique_adj(
                num_nodes=n_nodes,
                hyperedge_members=hyperedge_members,
                hyperedge_weight=hedge_w,
                combine_mode=clique_combine_mode,
                remove_self_loops=True,
            )

            edge_index, edge_attr = dense_to_sparse(torch.from_numpy(clique_adj))
            edge_attr = edge_attr.to(torch.float32)

            g = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index.long(),
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long),
            )

            # optional dense adjacency for debugging / summary features
            g.adj = torch.tensor(clique_adj, dtype=torch.float32)

            # metadata
            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            # keep hypergraph-derived metadata for debugging
            g.region_names = list(region_names)
            g.node_to_region_mask = torch.tensor(node_to_region_mask, dtype=torch.float32)
            g.hyperedge_weight = torch.tensor(hedge_w, dtype=torch.float32)

            graphs.append(g)

    return graphs
    
class HybridGNNEncoder(nn.Module):
    """
    Segment graph -> graph embedding
    Converted from EEGGNN_Hybrid_old while keeping the architecture as similar as possible.
    """
    def __init__(self,
                 in_channels=18, 
                 hidden_channels=64, 
                 emb_dim=128,
                 gat_layers=2, 
                 cheb_layers=2,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(HybridGNNEncoder, self).__init__()
        
        self.dropout = dropout
        self.pooling = pooling.lower()

        # ----- Spatial branch (GAT) -----
        # kept exactly like original
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn_gat1 = BatchNorm(hidden_channels * heads)

        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.bn_gat2 = BatchNorm(hidden_channels)
        
        # ----- Spectral branch (Chebyshev Conv) -----
        # kept exactly like original
        self.cheb1 = ChebConv(in_channels, hidden_channels, K=3)
        self.bn_cheb1 = BatchNorm(hidden_channels)

        self.cheb2 = ChebConv(hidden_channels, hidden_channels, K=4)
        self.bn_cheb2 = BatchNorm(hidden_channels)
        
        # ----- Fusion projection head -----
        # original: Linear(hidden_channels * 2, num_classes)
        # now:      graph embedding head
        self.fc1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, emb_dim)

    def forward(self, data_batch: Batch):
        x = data_batch.x
        edge_index = data_batch.edge_index
        batch = data_batch.batch

        # ----- Spatial path -----
        xs = F.relu(self.bn_gat1(self.gat1(x, edge_index)))
        xs = F.dropout(xs, p=self.dropout, training=self.training)
        xs = F.relu(self.bn_gat2(self.gat2(xs, edge_index)))
        xs = F.dropout(xs, p=self.dropout, training=self.training)

        # ----- Spectral path -----
        xp = F.relu(self.bn_cheb1(self.cheb1(x, edge_index)))
        xp = F.dropout(xp, p=self.dropout, training=self.training)
        xp = F.relu(self.bn_cheb2(self.cheb2(xp, edge_index)))
        xp = F.dropout(xp, p=self.dropout, training=self.training)

        # ----- Global pooling -----
        if self.pooling == "mean":
            xs = global_mean_pool(xs, batch)
            xp = global_mean_pool(xp, batch)
        elif self.pooling == "max":
            xs = global_max_pool(xs, batch)
            xp = global_max_pool(xp, batch)
        elif self.pooling == "sum":
            xs = global_add_pool(xs, batch)
            xp = global_add_pool(xp, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # ----- Fusion -----
        x_cat = torch.cat([xs, xp], dim=1)

        # ----- Projection head -----
        x_cat = F.relu(self.fc1(x_cat))
        x_cat = F.dropout(x_cat, p=self.dropout, training=self.training)
        graph_emb = self.fc2(x_cat)

        return graph_emb


class GNNEncoder_GAT(nn.Module):
    """
    Segment graph -> graph embedding
    Converted from EEGGNN_GAT while keeping the architecture as similar as possible.

    Original classifier:
        pooled graph -> fc1 -> fc2 -> logits

    Encoder version:
        pooled graph -> fc1 -> fc2 -> graph embedding
    """
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 emb_dim=128,
                 num_layers=3,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(GNNEncoder_GAT, self).__init__()
        
        self.dropout = dropout
        self.pooling = pooling.lower()
        self.num_layers = num_layers

        # ----- Input projection -----
        self.input_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ----- Stack of GAT blocks -----
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(Gat_block(
                hidden_channels=hidden_channels,
                concat=True,
                edge_dim=edge_dim,
                heads=heads
            ))

        # ----- Projection head -----
        # Same role/position as classifier head, but now outputs embedding
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, emb_dim)

    def forward(self, data_batch: Batch):
        x = data_batch.x
        edge_index = data_batch.edge_index
        batch = data_batch.batch

        # support either edge_attr or edge_weight stored in the batch
        edge_attr = getattr(data_batch, "edge_attr", None)
        if edge_attr is None:
            edge_attr = getattr(data_batch, "edge_weight", None)

        # Ensure edge_attr shape for GATv2Conv(edge_dim=1)
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # ----- Input projection -----
        x = self.input_mlp(x)

        # ----- GAT layers -----
        for conv in self.gat_layers:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- Global pooling -----
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # ----- Projection head -----
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = self.fc2(x)

        return graph_emb




def extract_summary_features_from_pyg(g, use_upper_triangle=True, symmetrize=True):
    """
    Build the same kind of summary feature vector from a PyG graph.

    Expected graph attributes:
      - g.x          : [N, F]
      - either g.adj OR g.edge_index (+ optional g.edge_attr)

    Returns:
      np.ndarray [summary_dim]
    """
    # node features
    x = g.x
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)

    N, F = x.shape

    # adjacency
    if hasattr(g, "adj") and g.adj is not None:
        adj = g.adj
        if torch.is_tensor(adj):
            adj = adj.detach().cpu().numpy()
        adj = np.asarray(adj, dtype=np.float32)
    else:
        adj = np.zeros((N, N), dtype=np.float32)

        edge_index = g.edge_index.detach().cpu().numpy()
        if hasattr(g, "edge_attr") and g.edge_attr is not None:
            edge_attr = g.edge_attr.detach().cpu().numpy()
            if edge_attr.ndim > 1:
                edge_attr = edge_attr[:, 0]
            edge_attr = edge_attr.astype(np.float32)
        else:
            edge_attr = np.ones(edge_index.shape[1], dtype=np.float32)

        for k in range(edge_index.shape[1]):
            i = int(edge_index[0, k])
            j = int(edge_index[1, k])
            adj[i, j] = edge_attr[k]

        if symmetrize:
            adj = 0.5 * (adj + adj.T)

    feat_mean = x.mean(axis=0)
    feat_std = x.std(axis=0)
    feat_min = x.min(axis=0)
    feat_max = x.max(axis=0)

    if use_upper_triangle:
        iu = np.triu_indices(N, k=1)
        edges = adj[iu]
    else:
        edges = adj.reshape(-1)

    edge_mean = np.array([edges.mean()], dtype=np.float32)
    edge_std = np.array([edges.std()], dtype=np.float32)
    edge_min = np.array([edges.min()], dtype=np.float32)
    edge_max = np.array([edges.max()], dtype=np.float32)
    nonzero_ratio = np.array([(np.abs(edges) > 1e-8).mean()], dtype=np.float32)

    node_strength = adj.sum(axis=1)
    strength_mean = np.array([node_strength.mean()], dtype=np.float32)
    strength_std = np.array([node_strength.std()], dtype=np.float32)
    strength_min = np.array([node_strength.min()], dtype=np.float32)
    strength_max = np.array([node_strength.max()], dtype=np.float32)

    node_degree = (np.abs(adj) > 1e-8).sum(axis=1)
    degree_mean = np.array([node_degree.mean()], dtype=np.float32)
    degree_std = np.array([node_degree.std()], dtype=np.float32)

    try:
        eigvals = np.linalg.eigvalsh(adj)
        eigvals = np.sort(eigvals)
        eig_summary = np.array([
            eigvals[-1],
            eigvals[-2] if len(eigvals) >= 2 else eigvals[-1],
            eigvals.mean(),
            eigvals.std(),
            eigvals.min(),
        ], dtype=np.float32)
    except Exception:
        eig_summary = np.zeros(5, dtype=np.float32)

    summary_feat = np.concatenate([
        feat_mean, feat_std, feat_min, feat_max,
        edge_mean, edge_std, edge_min, edge_max,
        nonzero_ratio,
        strength_mean, strength_std, strength_min, strength_max,
        degree_mean, degree_std,
        eig_summary,
    ]).astype(np.float32)

    return summary_feat


def attach_summary_features_to_graphs(graphs, use_upper_triangle=True, symmetrize=True):
    for g in graphs:
        if hasattr(g, "summary_feat") and g.summary_feat is not None:
            continue

        summary_feat = extract_summary_features_from_pyg(
            g,
            use_upper_triangle=use_upper_triangle,
            symmetrize=symmetrize,
        )
        g.summary_feat = torch.tensor(summary_feat, dtype=torch.float32)

    return graphs



class SummaryMLPEncoder(nn.Module):
    """
    Input:
        summary_x: [num_graphs, summary_input_dim]

    Output:
        graph_emb: [num_graphs, graph_emb_dim]
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        emb_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(prev, emb_dim)

    def forward(self, summary_x):
        h = self.mlp(summary_x)
        emb = self.proj(h)
        return emb
def load_segment_records(pt_path: str):
    """
    Supports:
      - torch.save(list_of_dicts)
      - torch.save({"data": list_of_dicts, ...})
    """
    obj = torch.load(pt_path, map_location="cpu")

    if isinstance(obj, dict) and "data" in obj:
        records = obj["data"]
    elif isinstance(obj, list):
        records = obj
    else:
        raise ValueError("Unsupported .pt format. Expect list[dict] or dict with key 'data'.")

    if len(records) == 0:
        raise ValueError("No records found in the file.")

    return records

def split_records_by_subject(
    records: List[dict],
    train_subject_ids: List[str],
    val_subject_ids: List[str],
) -> Tuple[List[dict], List[dict]]:
    train_subject_ids = set(train_subject_ids)
    val_subject_ids = set(val_subject_ids)

    train_records, val_records = [], []
    for r in records:
        sid = r["subject_id"]
        if sid in train_subject_ids:
            train_records.append(r)
        elif sid in val_subject_ids:
            val_records.append(r)

    return train_records, val_records

def compute_class_weights_from_subjects(subject_labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(subject_labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_subject_metrics(y_true, y_pred) -> Dict:
    return {
        # "y_pred": y_pred,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "conf_matrix": confusion_matrix(y_true, y_pred),
    }

def collate_subject_bags(batch: List[dict]) -> Dict:
    all_graphs = []
    all_summary = []
    all_full_adj = []

    all_adj_bank = []
    all_topology_bank = []

    bag_sizes = []
    labels = []
    subject_ids = []
    segment_ids_per_subject = []
    topology_names = None

    for item in batch:
        gs = item["graphs"]
        all_graphs.extend(gs)
        bag_sizes.append(len(gs))
        labels.append(int(item["label"]))
        subject_ids.append(item["subject_id"])

        if "segment_ids" in item:
            segment_ids_per_subject.append(item["segment_ids"])

        for g in gs:

            # full dense adjacency for mlp_raw edge_mode='full_adj'
            if hasattr(g, "adj") and g.adj is not None:
                adj = g.adj
                if torch.is_tensor(adj):
                    adj = adj.detach().cpu()
                else:
                    adj = torch.tensor(adj, dtype=torch.float32)
                all_full_adj.append(adj.float())


            if hasattr(g, "adj_bank") and g.adj_bank is not None:
                bank = g.adj_bank
                if torch.is_tensor(bank):
                    bank = bank.detach().cpu()
                else:
                    bank = torch.tensor(bank, dtype=torch.float32)
                all_adj_bank.append(bank.float())

            if hasattr(g, "topology_bank") and g.topology_bank is not None:
                topo = g.topology_bank
                if torch.is_tensor(topo):
                    topo = topo.detach().cpu()
                else:
                    topo = torch.tensor(topo, dtype=torch.float32)
                all_topology_bank.append(topo.float())

            if topology_names is None and hasattr(g, "topology_names"):
                topology_names = list(g.topology_names)

        #     if not hasattr(g, "summary_feat"):
        #         raise AttributeError("Graph is missing summary_feat. Run attach_summary_features_to_graphs(...) first.")
        #     sf = g.summary_feat
        #     if torch.is_tensor(sf):
        #         sf = sf.detach().cpu().numpy()
        #     all_summary.append(np.asarray(sf, dtype=np.float32))

    pyg_batch = Batch.from_data_list(all_graphs)

    out = {
        "pyg_batch": pyg_batch,
        # "summary_x": torch.tensor(np.stack(all_summary, axis=0), dtype=torch.float32),
        "bag_sizes": torch.tensor(bag_sizes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "subject_ids": subject_ids,
    }

    if len(segment_ids_per_subject) > 0:
        out["segment_ids_per_subject"] = segment_ids_per_subject

    if len(all_full_adj) == len(all_graphs):
        out["full_adj"] = torch.stack(all_full_adj, dim=0)   # [num_graphs, N, N]

    if len(all_adj_bank) == len(all_graphs):
        out["adj_bank"] = torch.stack(all_adj_bank, dim=0)          # [num_graphs, K, N, N]

    if len(all_topology_bank) == len(all_graphs):
        out["topology_bank"] = torch.stack(all_topology_bank, dim=0)  # [num_graphs, K, N, N]

    if topology_names is not None:
        out["topology_names"] = topology_names

    return out

def collate_subject_bags_multiband(batch: List[dict]) -> Dict:
    all_graphs = []
    all_summary = []
    all_full_adj = []
    all_conn_stack = []
    bag_sizes = []
    labels = []
    subject_ids = []
    segment_ids_per_subject = []

    for item in batch:
        gs = item["graphs"]
        all_graphs.extend(gs)
        bag_sizes.append(len(gs))
        labels.append(int(item["label"]))
        subject_ids.append(item["subject_id"])

        if "segment_ids" in item:
            segment_ids_per_subject.append(item["segment_ids"])

        for g in gs:
            if hasattr(g, "adj") and g.adj is not None:
                adj = g.adj
                if torch.is_tensor(adj):
                    adj = adj.detach().cpu()
                else:
                    adj = torch.tensor(adj, dtype=torch.float32)
                all_full_adj.append(adj.float())

            if hasattr(g, "conn_stack") and g.conn_stack is not None:
                cs = g.conn_stack
                if torch.is_tensor(cs):
                    cs = cs.detach().cpu()
                else:
                    cs = torch.tensor(cs, dtype=torch.float32)
                all_conn_stack.append(cs.float())

    pyg_batch = Batch.from_data_list(all_graphs)

    out = {
        "pyg_batch": pyg_batch,
        "bag_sizes": torch.tensor(bag_sizes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "subject_ids": subject_ids,
    }

    if len(segment_ids_per_subject) > 0:
        out["segment_ids_per_subject"] = segment_ids_per_subject

    if len(all_full_adj) == len(all_graphs):
        out["full_adj"] = torch.stack(all_full_adj, dim=0).float()   # [num_graphs, N, N]

    if len(all_conn_stack) == len(all_graphs):
        out["conn_stack"] = torch.stack(all_conn_stack, dim=0).float()   # [num_graphs, B, N, N]

    return out
from typing import Optional, Sequence, Dict, Any, List, Tuple
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


# =========================================================
# Stage-2 graph-level builders from H5 payload
#   - segment
#   - macro
#   - subject
# =========================================================

def _ensure_nonempty_feature_families(feature_families: Sequence[str]) -> List[str]:
    ff = [str(x) for x in feature_families]
    if len(ff) == 0:
        raise ValueError("feature_families is empty.")
    return ff


def _safe_subject_payload(payload: Dict[str, Dict[str, Any]], sid: str) -> Dict[str, Any]:
    if sid not in payload:
        raise KeyError(f"Subject {sid!r} not found in payload")
    subj = payload[sid]
    if "features" not in subj:
        raise KeyError(f"payload[{sid!r}] is missing 'features'")
    return subj


def _get_effective_channel_names(
    subj: Dict[str, Any],
    channel_names: Optional[Sequence[str]] = None,
) -> Optional[List[str]]:
    if channel_names is not None:
        return [str(x) for x in channel_names]
    if "channel_names" in subj and subj["channel_names"] is not None:
        return [str(x) for x in subj["channel_names"]]
    return None


def _aggregate_array(
    x: np.ndarray,
    mode: str = "mean",
    axis: int = 0,
) -> np.ndarray:
    mode = str(mode).lower()
    if mode == "mean":
        return np.mean(x, axis=axis)
    if mode == "median":
        return np.median(x, axis=axis)
    if mode == "max":
        return np.max(x, axis=axis)
    if mode == "min":
        return np.min(x, axis=axis)
    if mode == "std":
        return np.std(x, axis=axis)
    raise ValueError(f"Unsupported aggregation mode={mode!r}. Use one of ['mean','median','max','min','std'].")


def _standardize_node_features_per_graph(x: np.ndarray) -> np.ndarray:
    """
    Standardize node features within one graph:
      x: [N, F]
    z-score per feature dimension across nodes.
    """
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    return (x - mu) / (sd + 1e-8)


def _prepare_subject_node_feature_tensor(
    subj: Dict[str, Any],
    feature_families: Sequence[str],
) -> np.ndarray:
    """
    Returns:
        node_x_all: [W, N, F_total]
    """
    feat_list = []
    ref_w = None
    ref_n = None

    for fam in feature_families:
        if fam not in subj["features"]:
            raise KeyError(f"payload subject is missing feature family {fam!r}")

        xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
        if xfam.ndim != 3:
            raise ValueError(
                f"Feature family {fam!r} must have shape [W, N, F], got {xfam.shape}"
            )

        if ref_w is None:
            ref_w, ref_n = xfam.shape[:2]
        else:
            if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                raise ValueError(
                    f"Feature family {fam!r} has incompatible shape {xfam.shape}; "
                    f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                )

        feat_list.append(xfam)

    if len(feat_list) == 0:
        raise ValueError("feature_families is empty")

    node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
    return node_x_all


def _prepare_subject_adjacency_tensor(
    subj: Dict[str, Any],
    *,
    connectivity_metric: Optional[str] = None,
    connectivity_band=None,
    edge_source: str = "connectivity",
) -> np.ndarray:
    """
    Returns:
        adj_all: [W, N, N]
    """
    edge_source = str(edge_source).lower()

    if edge_source == "connectivity":
        if connectivity_metric is None:
            raise ValueError("connectivity_metric must be provided when edge_source='connectivity'")
        if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
            raise KeyError(f"payload subject connectivity missing metric {connectivity_metric!r}")

        adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

        # allow either [W, N, N] or [W, B, N, N]
        if adj_all.ndim == 4:
            if connectivity_band is None:
                raise ValueError(
                    f"Connectivity metric {connectivity_metric!r} is banded [W,B,N,N], "
                    "but connectivity_band is None."
                )
            if isinstance(connectivity_band, str):
                raise ValueError(
                    "String connectivity_band is not supported here. "
                    "Please pass an integer band index or slice earlier when loading the payload."
                )
            band_idx = int(connectivity_band)
            if band_idx < 0 or band_idx >= adj_all.shape[1]:
                raise IndexError(
                    f"connectivity_band={band_idx} out of range for tensor with shape {adj_all.shape}"
                )
            adj_all = adj_all[:, band_idx]   # [W, N, N]

        elif adj_all.ndim != 3:
            raise ValueError(
                f"Connectivity tensor must have shape [W, N, N] or [W, B, N, N], got {adj_all.shape}"
            )

        return adj_all.astype(np.float32)

    elif edge_source == "aligned_adj":
        if subj.get("aligned_adj", None) is None:
            raise ValueError("edge_source='aligned_adj' but payload subject has aligned_adj=None")
        adj_all = np.asarray(subj["aligned_adj"], dtype=np.float32)
        if adj_all.ndim != 3:
            raise ValueError(f"aligned_adj must have shape [W, N, N], got {adj_all.shape}")
        return adj_all.astype(np.float32)

    else:
        raise ValueError(f"Unsupported edge_source={edge_source!r}")


def _validate_window_alignment(
    node_x_all: np.ndarray,
    adj_all: np.ndarray,
    sid: str,
) -> None:
    if node_x_all.ndim != 3:
        raise ValueError(f"node_x_all for {sid!r} must be [W, N, F], got {node_x_all.shape}")
    if adj_all.ndim != 3:
        raise ValueError(f"adj_all for {sid!r} must be [W, N, N], got {adj_all.shape}")

    if node_x_all.shape[0] != adj_all.shape[0]:
        raise ValueError(
            f"Window count mismatch for {sid!r}: features have {node_x_all.shape[0]} windows "
            f"but adjacency has {adj_all.shape[0]}"
        )
    if node_x_all.shape[1] != adj_all.shape[1] or adj_all.shape[1] != adj_all.shape[2]:
        raise ValueError(
            f"Node count mismatch for {sid!r}: node_x_all shape {node_x_all.shape}, adj_all shape {adj_all.shape}"
        )


def _group_window_indices_by_macro(
    start_samples: np.ndarray,
    *,
    macro_size_samples: int,
) -> List[Tuple[int, np.ndarray]]:
    """
    Non-overlapping macro bins using start_sample // macro_size_samples.

    Returns
    -------
    groups : list[(macro_id, window_indices)]
        Sorted by macro_id.
    """
    if macro_size_samples < 1:
        raise ValueError(f"macro_size_samples must be >= 1, got {macro_size_samples}")

    start_samples = np.asarray(start_samples, dtype=np.int64).reshape(-1)
    if start_samples.size == 0:
        return []

    groups: Dict[int, List[int]] = {}
    for idx, s in enumerate(start_samples.tolist()):
        macro_id = int(s // macro_size_samples)
        groups.setdefault(macro_id, []).append(int(idx))

    out: List[Tuple[int, np.ndarray]] = []
    for macro_id in sorted(groups.keys()):
        out.append((int(macro_id), np.asarray(groups[macro_id], dtype=np.int64)))
    return out


def _make_graph_data(
    *,
    sid: str,
    label: int,
    x: np.ndarray,
    adj: np.ndarray,
    filter_method: str = "mst",
    fixed_edges=None,
    channel_names: Optional[Sequence[str]] = None,
    topk: Optional[int] = 2,
    top_percent: Optional[float] = None,
    undirected: bool = True,
    zero_diagonal: bool = True,
    symmetrize_adj: bool = True,
    attach_dense_adj: bool = True,
    standardize_features: bool = True,
    level: str = "segment",
    segment_id: Optional[int] = None,
    macro_id: Optional[int] = None,
    start_sample: Optional[int] = None,
    end_sample: Optional[int] = None,
    extra_attrs: Optional[Dict[str, Any]] = None,
) -> Data:
    """
    Build one PyG Data object from one [N,F] + [N,N] graph.
    This preserves the same filtering path as the existing segment builder:
      dense adj -> candidate edges -> apply_edge_filter(...) -> final sparse graph.
    """
    x = np.asarray(x, dtype=np.float32)
    adj = np.asarray(adj, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"x must have shape [N, F], got {x.shape}")
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must have shape [N, N], got {adj.shape}")
    if adj.shape[0] != x.shape[0]:
        raise ValueError(f"x and adj node counts do not match: {x.shape} vs {adj.shape}")

    if standardize_features:
        x = _standardize_node_features_per_graph(x)

    adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    if symmetrize_adj:
        adj = 0.5 * (adj + adj.T)
    if zero_diagonal:
        np.fill_diagonal(adj, 0.0)

    edge_index0, edge_attr0, edge_list0, edge_weights0 = dense_adj_to_candidate_edges(
        torch.tensor(adj, dtype=torch.float32),
        undirected=undirected,
    )

    edge_index, edge_weight, final_adj = apply_edge_filter(
        edge_index=edge_index0,
        edge_attr=edge_attr0,
        edge_list=edge_list0,
        edge_weights=edge_weights0,
        n_channels=x.shape[0],
        filter_method=filter_method,
        topk=topk,
        top_percent=top_percent,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        undirected=undirected,
    )

    g = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index.long(),
        y=torch.tensor([int(label)], dtype=torch.long),
    )
    g.edge_weight = edge_weight.float()
    g.edge_attr = edge_weight.view(-1, 1).float()

    if attach_dense_adj:
        g.adj = torch.tensor(final_adj, dtype=torch.float32)

    g.subject_id = str(sid)
    g.level = str(level)

    if segment_id is not None:
        g.segment_id = int(segment_id)
    if macro_id is not None:
        g.macro_id = int(macro_id)
    if start_sample is not None:
        g.start_sample = int(start_sample)
    if end_sample is not None:
        g.end_sample = int(end_sample)

    if extra_attrs is not None:
        for k, v in extra_attrs.items():
            setattr(g, k, v)

    return g


def build_segment_graphs_from_payload(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    filter_method: str = "mst",
    fixed_edges=None,
    channel_names=None,
    topk: Optional[int] = 2,
    top_percent: Optional[float] = None,
    undirected: bool = True,
    standardize_features: bool = True,
):
    """
    Build one PyG graph per window from H5-loaded payload.

    This is the stage-2 segment builder and should preserve the old logic.
    """
    graphs = []
    feature_families = _ensure_nonempty_feature_families(feature_families)

    for sid in subject_ids:
        subj = _safe_subject_payload(payload, sid)
        label = int(subj["label"])

        node_x_all = _prepare_subject_node_feature_tensor(subj, feature_families)
        adj_all = _prepare_subject_adjacency_tensor(
            subj,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_source=edge_source,
        )
        _validate_window_alignment(node_x_all, adj_all, sid)

        seg_ids = np.asarray(subj.get("segment_id", np.arange(node_x_all.shape[0])), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(node_x_all.shape[0], -1)), dtype=np.int64)
        end_samples = np.asarray(subj.get("end_sample", np.full(node_x_all.shape[0], -1)), dtype=np.int64)

        if len(seg_ids) != node_x_all.shape[0]:
            raise ValueError(
                f"segment_id length mismatch for subject {sid!r}: {len(seg_ids)} vs {node_x_all.shape[0]}"
            )
        if len(start_samples) != node_x_all.shape[0]:
            raise ValueError(
                f"start_sample length mismatch for subject {sid!r}: {len(start_samples)} vs {node_x_all.shape[0]}"
            )
        if len(end_samples) != node_x_all.shape[0]:
            raise ValueError(
                f"end_sample length mismatch for subject {sid!r}: {len(end_samples)} vs {node_x_all.shape[0]}"
            )

        eff_channel_names = _get_effective_channel_names(subj, channel_names)

        for w in range(node_x_all.shape[0]):
            g = _make_graph_data(
                sid=str(sid),
                label=label,
                x=node_x_all[w],
                adj=adj_all[w],
                filter_method=filter_method,
                fixed_edges=fixed_edges,
                channel_names=eff_channel_names,
                topk=topk,
                top_percent=top_percent,
                undirected=undirected,
                zero_diagonal=zero_diagonal,
                symmetrize_adj=symmetrize_adj,
                attach_dense_adj=attach_dense_adj,
                standardize_features=standardize_features,
                level="segment",
                segment_id=int(seg_ids[w]),
                macro_id=None,
                start_sample=int(start_samples[w]),
                end_sample=int(end_samples[w]),
            )
            graphs.append(g)

    return graphs


def build_macro_graphs_from_payload(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    filter_method: str = "mst",
    fixed_edges=None,
    channel_names=None,
    topk: Optional[int] = 2,
    top_percent: Optional[float] = None,
    undirected: bool = True,
    standardize_features: bool = True,
    macro_seconds: float = 300.0,
    sfreq: float = 200.0,
    macro_reduce_node: str = "mean",
    macro_reduce_adj: str = "mean",
):
    """
    Build one PyG graph per macro block by aggregating many windows.

    Macro assignment:
        macro_id = start_sample // macro_size_samples
    """
    graphs = []
    feature_families = _ensure_nonempty_feature_families(feature_families)

    macro_size_samples = int(round(float(macro_seconds) * float(sfreq)))
    if macro_size_samples < 1:
        raise ValueError(f"macro_size_samples must be >= 1, got {macro_size_samples}")

    for sid in subject_ids:
        subj = _safe_subject_payload(payload, sid)
        label = int(subj["label"])

        node_x_all = _prepare_subject_node_feature_tensor(subj, feature_families)    # [W, N, F]
        adj_all = _prepare_subject_adjacency_tensor(
            subj,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_source=edge_source,
        )                                                                           # [W, N, N]
        _validate_window_alignment(node_x_all, adj_all, sid)

        seg_ids = np.asarray(subj.get("segment_id", np.arange(node_x_all.shape[0])), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(node_x_all.shape[0], -1)), dtype=np.int64)
        end_samples = np.asarray(subj.get("end_sample", np.full(node_x_all.shape[0], -1)), dtype=np.int64)

        eff_channel_names = _get_effective_channel_names(subj, channel_names)
        macro_groups = _group_window_indices_by_macro(
            start_samples,
            macro_size_samples=macro_size_samples,
        )

        for macro_id, window_idx in macro_groups:
            macro_x = _aggregate_array(node_x_all[window_idx], mode=macro_reduce_node, axis=0)   # [N, F]
            macro_adj = _aggregate_array(adj_all[window_idx], mode=macro_reduce_adj, axis=0)      # [N, N]

            g = _make_graph_data(
                sid=str(sid),
                label=label,
                x=macro_x,
                adj=macro_adj,
                filter_method=filter_method,
                fixed_edges=fixed_edges,
                channel_names=eff_channel_names,
                topk=topk,
                top_percent=top_percent,
                undirected=undirected,
                zero_diagonal=zero_diagonal,
                symmetrize_adj=symmetrize_adj,
                attach_dense_adj=attach_dense_adj,
                standardize_features=standardize_features,
                level="macro",
                segment_id=None,
                macro_id=int(macro_id),
                start_sample=int(start_samples[window_idx].min()) if len(window_idx) > 0 else None,
                end_sample=int(end_samples[window_idx].max()) if len(window_idx) > 0 else None,
                extra_attrs={
                    "segment_ids": seg_ids[window_idx].astype(np.int64).tolist(),
                    "source_window_indices": window_idx.astype(np.int64).tolist(),
                    "num_source_segments": int(len(window_idx)),
                },
            )
            graphs.append(g)

    return graphs


def build_subject_graphs_from_payload(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    filter_method: str = "mst",
    fixed_edges=None,
    channel_names=None,
    topk: Optional[int] = 2,
    top_percent: Optional[float] = None,
    undirected: bool = True,
    standardize_features: bool = True,
    subject_reduce_node: str = "mean",
    subject_reduce_adj: str = "mean",
):
    """
    Build one PyG graph per subject by aggregating all windows in that subject.
    """
    graphs = []
    feature_families = _ensure_nonempty_feature_families(feature_families)

    for sid in subject_ids:
        subj = _safe_subject_payload(payload, sid)
        label = int(subj["label"])

        node_x_all = _prepare_subject_node_feature_tensor(subj, feature_families)    # [W, N, F]
        adj_all = _prepare_subject_adjacency_tensor(
            subj,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_source=edge_source,
        )                                                                           # [W, N, N]
        _validate_window_alignment(node_x_all, adj_all, sid)

        seg_ids = np.asarray(subj.get("segment_id", np.arange(node_x_all.shape[0])), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(node_x_all.shape[0], -1)), dtype=np.int64)
        end_samples = np.asarray(subj.get("end_sample", np.full(node_x_all.shape[0], -1)), dtype=np.int64)

        eff_channel_names = _get_effective_channel_names(subj, channel_names)

        subj_x = _aggregate_array(node_x_all, mode=subject_reduce_node, axis=0)      # [N, F]
        subj_adj = _aggregate_array(adj_all, mode=subject_reduce_adj, axis=0)         # [N, N]

        g = _make_graph_data(
            sid=str(sid),
            label=label,
            x=subj_x,
            adj=subj_adj,
            filter_method=filter_method,
            fixed_edges=fixed_edges,
            channel_names=eff_channel_names,
            topk=topk,
            top_percent=top_percent,
            undirected=undirected,
            zero_diagonal=zero_diagonal,
            symmetrize_adj=symmetrize_adj,
            attach_dense_adj=attach_dense_adj,
            standardize_features=standardize_features,
            level="subject",
            segment_id=None,
            macro_id=None,
            start_sample=int(start_samples.min()) if len(start_samples) > 0 else None,
            end_sample=int(end_samples.max()) if len(end_samples) > 0 else None,
            extra_attrs={
                "segment_ids": seg_ids.astype(np.int64).tolist(),
                "num_source_segments": int(len(seg_ids)),
            },
        )
        graphs.append(g)

    return graphs


def build_graphs_from_payload_by_level(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    filter_method: str = "mst",
    fixed_edges=None,
    channel_names=None,
    topk: Optional[int] = 2,
    top_percent: Optional[float] = None,
    undirected: bool = True,
    standardize_features: bool = True,
    graph_level: str = "segment",
    macro_seconds: float = 300.0,
    sfreq: float = 200.0,
    macro_reduce_node: str = "mean",
    macro_reduce_adj: str = "mean",
    subject_reduce_node: str = "mean",
    subject_reduce_adj: str = "mean",
):
    """
    Wrapper dispatcher for stage-2 graph level selection.

    graph_level:
      - "segment": one graph per stored window
      - "macro"  : one graph per macro time block
      - "subject": one graph per subject
    """
    level = str(graph_level).lower()

    common_kwargs = dict(
        payload=payload,
        subject_ids=subject_ids,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        edge_source=edge_source,
        zero_diagonal=zero_diagonal,
        symmetrize_adj=symmetrize_adj,
        attach_dense_adj=attach_dense_adj,
        filter_method=filter_method,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        topk=topk,
        top_percent=top_percent,
        undirected=undirected,
        standardize_features=standardize_features,
    )

    if level == "segment":
        return build_segment_graphs_from_payload(**common_kwargs)

    if level == "macro":
        return build_macro_graphs_from_payload(
            **common_kwargs,
            macro_seconds=macro_seconds,
            sfreq=sfreq,
            macro_reduce_node=macro_reduce_node,
            macro_reduce_adj=macro_reduce_adj,
        )

    if level == "subject":
        return build_subject_graphs_from_payload(
            **common_kwargs,
            subject_reduce_node=subject_reduce_node,
            subject_reduce_adj=subject_reduce_adj,
        )

    raise ValueError(
        f"Unsupported graph_level={graph_level!r}. "
        "Use one of ['segment', 'macro', 'subject']."
    )

def build_graphs_from_payload(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    connectivity_band=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,

    # new topology args
    undirected=True,
    filter_method="mst",             # "mst", "fixed", "topk", "reconnect", "combined", "overlap", "full"
    topk=4,
    top_percent=None,
    fixed_edges: Optional[EdgeSpec] = None,
    channel_names: Optional[Sequence[str]] = None,
    corruption_mode=None,            # None, "identity", "random", "permute_consistent", "permute_adj_only"
    standardize_features=True,

    region_to_channels=None,
    hyperedge_weight_mode="mean_abs_adj",
    clique_combine_mode="sum",
    keep_empty_hyperedges=False,
    # optional node-feature augmentation from adjacency
    add_graph_theory_to_node_features=False,
):
    """
    Build one PyG graph per window from payload, using the selected topology
    instead of blindly converting the full adjacency to sparse.

    payload[sid] must contain:
      - "label"
      - "features"[family] -> [W, N, F_family]
      - "segment_id" -> [W]
      - "start_sample" -> [W]
      - optionally:
          "connectivity"[metric] -> [W, N, N]
          "aligned_adj" -> [W, N, N]
          "channel_names" -> list[str]
    """
    
    if filter_method == "hypergraph":
        return build_graphs_from_payload_region_clique(
            payload=payload,
            subject_ids=subject_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            edge_source=edge_source,
            channel_names=channel_names,
            region_to_channels=region_to_channels,
            standardize_features=standardize_features,
            hyperedge_weight_mode=hyperedge_weight_mode,
            clique_combine_mode=clique_combine_mode,
            keep_empty_hyperedges=keep_empty_hyperedges,
            attach_dense_adj=attach_dense_adj,
        )
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        # -------------------------------------------------
        # node features
        # -------------------------------------------------
        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")

            xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
            if xfam.ndim != 3:
                raise ValueError(
                    f"Feature family {fam!r} for subject {sid!r} must have shape [W, N, F], got {xfam.shape}"
                )

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                    raise ValueError(
                        f"Feature family {fam!r} for subject {sid!r} has incompatible shape {xfam.shape}; "
                        f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                    )

            feat_list.append(xfam)

        if len(feat_list) == 0:
            raise ValueError("feature_families is empty")

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
        num_windows = node_x_all.shape[0]
        num_nodes = node_x_all.shape[1]

        # -------------------------------------------------
        # metadata
        # -------------------------------------------------
        seg_ids = np.asarray(subj.get("segment_id", np.arange(num_windows)), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(num_windows, -1)), dtype=np.int64)

        if len(seg_ids) != num_windows:
            raise ValueError(
                f"segment_id length mismatch for subject {sid!r}: got {len(seg_ids)}, expected {num_windows}"
            )
        if len(start_samples) != num_windows:
            raise ValueError(
                f"start_sample length mismatch for subject {sid!r}: got {len(start_samples)}, expected {num_windows}"
            )

        # prefer subject-level channel names if available
        entry_channel_names = subj.get("channel_names", channel_names)

        # -------------------------------------------------
        # adjacency source
        # -------------------------------------------------
        if edge_source == "connectivity":
            if connectivity_metric is None:
                adj_all = None
            else:
                if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
                    raise KeyError(
                        f"payload[{sid!r}]['connectivity'] missing metric {connectivity_metric!r}"
                    )
                adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

                # support [W, B, N, N] if band slicing was not done earlier
                if adj_all.ndim == 4:
                    if connectivity_band is None:
                        raise ValueError(
                            f"Connectivity for {sid!r}/{connectivity_metric!r} is banded [W,B,N,N], "
                            "but connectivity_band is None."
                        )
                    band_idx = int(connectivity_band) if not isinstance(connectivity_band, str) else connectivity_band
                    if isinstance(band_idx, str):
                        raise ValueError(
                            "String band selection is not supported here unless you also pass band-name metadata. "
                            "Prefer slicing earlier in load_h5_payload_for_subjects(...)."
                        )
                    adj_all = adj_all[:, band_idx]



        elif edge_source == "aligned_adj":
            if subj.get("aligned_adj", None) is None:
                raise ValueError(
                    f"edge_source='aligned_adj' but payload[{sid!r}]['aligned_adj'] is None"
                )
            adj_all = np.asarray(subj["aligned_adj"], dtype=np.float32)

        else:
            raise ValueError(f"Unsupported edge_source={edge_source!r}")

        if adj_all is not None:
            if adj_all.ndim != 3:
                raise ValueError(
                    f"Adjacency tensor for subject {sid!r} must have shape [W, N, N], got {adj_all.shape}"
                )
            if adj_all.shape[0] != num_windows:
                raise ValueError(
                    f"Adjacency window count mismatch for subject {sid!r}: {adj_all.shape[0]} vs {num_windows}"
                )
            if adj_all.shape[1] != num_nodes or adj_all.shape[2] != num_nodes:
                raise ValueError(
                    f"Adjacency node count mismatch for subject {sid!r}: {adj_all.shape} vs num_nodes={num_nodes}"
                )

        # -------------------------------------------------
        # build one graph per window
        # -------------------------------------------------
        for w in range(num_windows):
            x = _to_2d_features(node_x_all[w])   # [N, F]

            if standardize_features:
                x = torch.from_numpy(_zscore_per_feature(x.numpy()))

            if adj_all is None:
                adj_full = np.eye(num_nodes, dtype=np.float32)
            else:
                adj_full = np.asarray(adj_all[w], dtype=np.float32).copy()

            if symmetrize_adj:
                adj_full = 0.5 * (adj_full + adj_full.T)

            if zero_diagonal:
                np.fill_diagonal(adj_full, 0.0)

            adj_full = np.nan_to_num(adj_full, nan=0.0, posinf=0.0, neginf=0.0)
            adj_full_t = torch.tensor(adj_full, dtype=torch.float32)

            # --------------------------------
            # optional corruption on full adj
            # --------------------------------
            if corruption_mode == "identity":
                adj_used = make_identity_adj(num_nodes)

            elif corruption_mode == "random":
                adj_used = make_random_adj_like_with_weights(adj_full_t, undirected=undirected)

            elif corruption_mode == "permute_consistent":
                x, adj_used, _ = permute_graph_consistently(x, adj_full_t)

            elif corruption_mode == "permute_adj_only":
                adj_used, _ = permute_adj_only(adj_full_t)

            elif corruption_mode is None:
                adj_used = adj_full_t.clone()

            else:
                raise ValueError(f"Unknown corruption_mode={corruption_mode}")

            adj_used = adj_used.clone()
            if zero_diagonal:
                adj_used.fill_diagonal_(0.0)

            # --------------------------------
            # candidate edges from current full matrix
            # --------------------------------
            edge_index, edge_attr, edge_list, edge_weights = dense_adj_to_candidate_edges(
                adj_used,
                undirected=undirected,
            )

            # --------------------------------
            # apply topology filter
            # --------------------------------
            final_edge_index, final_edge_weight, final_adj = apply_edge_filter(
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_list=edge_list,
                edge_weights=edge_weights,
                n_channels=num_nodes,
                filter_method=filter_method,
                topk=topk,
                top_percent=top_percent,
                fixed_edges=fixed_edges,
                channel_names=entry_channel_names,
                undirected=undirected,
            )

            # --------------------------------
            # node augmentation from filtered topology
            # --------------------------------
            x_np = x.detach().cpu().numpy().astype(np.float32)

            if add_graph_theory_to_node_features:
                x_aug, _ = append_weighted_graph_theory_to_node_features(
                    node_features=x_np,
                    adj=final_adj,          # <-- filtered topology, not full dense adj
                    signed_input=False,
                )
            else:
                x_aug = x_np

            g = Data(
                x=torch.tensor(x_aug, dtype=torch.float32),
                edge_index=final_edge_index,
                y=torch.tensor([label], dtype=torch.long),
            )

            # pipeline compatibility
            g.edge_weight = final_edge_weight
            g.edge_attr = final_edge_weight.view(-1, 1)

            if attach_dense_adj:
                g.adj = torch.tensor(final_adj, dtype=torch.float32)

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            graphs.append(g)

    return graphs
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _prepare_adjacency(
    adj: np.ndarray,
    *,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Prepare one dense weighted adjacency matrix.

    Parameters
    ----------
    adj : [N, N]
        Full weighted adjacency.
    symmetrize : bool
        If True, replace A with 0.5 * (A + A.T).
    zero_diagonal : bool
        If True, set diagonal to zero.
    """
    A = _as_float32(adj).copy()

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"adj must be square [N, N], got shape={A.shape}")

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    if symmetrize:
        A = 0.5 * (A + A.T)

    if zero_diagonal:
        np.fill_diagonal(A, 0.0)

    # remove tiny numerical noise
    A[np.abs(A) < eps] = 0.0
    return A.astype(np.float32, copy=False)


def _power_iteration_eigenvector_centrality(
    W: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Eigenvector centrality for a nonnegative symmetric matrix W.
    Returns a nonnegative vector of shape [N].
    """
    N = W.shape[0]
    if N == 0:
        return np.zeros((0,), dtype=np.float32)

    if np.all(W <= eps):
        return np.zeros((N,), dtype=np.float32)

    v = np.ones((N,), dtype=np.float32) / np.sqrt(max(N, 1))

    for _ in range(max_iter):
        v_next = W @ v
        norm = np.linalg.norm(v_next)
        if norm < eps:
            return np.zeros((N,), dtype=np.float32)

        v_next = v_next / norm

        if np.max(np.abs(v_next - v)) < tol:
            v = v_next
            break

        v = v_next

    # keep scale stable
    v = np.maximum(v, 0.0)
    s = v.sum()
    if s > eps:
        v = v / s

    return v.astype(np.float32, copy=False)


def _weighted_local_clustering(
    W_nonneg: np.ndarray,
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Weighted local clustering coefficient per node for a nonnegative matrix.

    Uses an Onnela-style triangle intensity:
        C_i = 2 / (k_i (k_i - 1)) * sum_{j<k in N(i)} (w_ij w_ik w_jk)^(1/3)

    W_nonneg should be:
      - symmetric
      - nonnegative
      - zero diagonal
    """
    N = W_nonneg.shape[0]
    C = np.zeros((N,), dtype=np.float32)

    if N < 3:
        return C

    max_w = float(np.max(W_nonneg))
    if max_w < eps:
        return C

    # normalize to [0, 1] for stability
    W = W_nonneg / max_w

    for i in range(N):
        nbrs = np.where(W[i] > eps)[0]
        nbrs = nbrs[nbrs != i]
        k = len(nbrs)

        if k < 2:
            C[i] = 0.0
            continue

        tri_sum = 0.0
        for a in range(k):
            j = nbrs[a]
            wij = W[i, j]
            for b in range(a + 1, k):
                l = nbrs[b]
                wil = W[i, l]
                wjl = W[j, l]
                if wjl <= eps:
                    continue
                tri_sum += float((wij * wil * wjl) ** (1.0 / 3.0))

        C[i] = float((2.0 * tri_sum) / max(k * (k - 1), 1))

    return C.astype(np.float32, copy=False)


def compute_weighted_node_graph_theory_features(
    adj: np.ndarray,
    *,
    signed_input: Optional[bool] = None,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    include_strength: bool = True,
    include_abs_strength: bool = True,
    include_pos_neg_strength: bool = True,
    include_mean_abs_edge: bool = True,
    include_weighted_clustering: bool = True,
    include_eigenvector_centrality: bool = True,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute per-node weighted graph-theory features from one full adjacency matrix.

    Parameters
    ----------
    adj : [N, N]
        Full weighted adjacency for one segment.
    signed_input : bool or None
        - True: treat adjacency as signed (e.g. Pearson, Spearman)
        - False: treat adjacency as nonnegative (e.g. coherence, PLI, PLV, wPLI)
        - None: auto-detect from the matrix
    symmetrize, zero_diagonal :
        Standard adjacency cleanup.
    include_* :
        Toggle feature groups.

    Returns
    -------
    features : [N, K] float32
    meta : dict
        Includes feature_names and description.
    """
    A = _prepare_adjacency(
        adj,
        symmetrize=symmetrize,
        zero_diagonal=zero_diagonal,
        eps=eps,
    )
    N = A.shape[0]

    if signed_input is None:
        signed_input = bool(np.any(A < -eps))

    A_abs = np.abs(A)
    A_pos = np.maximum(A, 0.0)
    A_neg = np.maximum(-A, 0.0)

    parts: List[np.ndarray] = []
    feature_names: List[str] = []

    if include_strength:
        strength = A.sum(axis=1, dtype=np.float32)
        parts.append(strength[:, None])
        feature_names.append("gt_strength")

    if include_abs_strength:
        abs_strength = A_abs.sum(axis=1, dtype=np.float32)
        parts.append(abs_strength[:, None])
        feature_names.append("gt_abs_strength")

    if include_pos_neg_strength and signed_input:
        pos_strength = A_pos.sum(axis=1, dtype=np.float32)
        neg_strength = A_neg.sum(axis=1, dtype=np.float32)
        parts.append(pos_strength[:, None])
        parts.append(neg_strength[:, None])
        feature_names.extend(["gt_pos_strength", "gt_neg_strength"])

    if include_mean_abs_edge:
        denom = max(N - 1, 1)
        mean_abs_edge = A_abs.sum(axis=1, dtype=np.float32) / float(denom)
        parts.append(mean_abs_edge[:, None])
        feature_names.append("gt_mean_abs_edge")

    if include_weighted_clustering:
        # for signed matrices, use abs(A) for a stable nonnegative weighted clustering
        clustering_base = A_abs if signed_input else np.maximum(A, 0.0)
        weighted_clustering = _weighted_local_clustering(clustering_base, eps=eps)
        parts.append(weighted_clustering[:, None])
        feature_names.append("gt_weighted_clustering")

    if include_eigenvector_centrality:
        # for signed matrices, use abs(A); for unsigned, use nonnegative A
        eig_base = A_abs if signed_input else np.maximum(A, 0.0)
        eig_cent = _power_iteration_eigenvector_centrality(eig_base, eps=eps)
        parts.append(eig_cent[:, None])
        feature_names.append("gt_eigenvector_centrality")

    if len(parts) == 0:
        raise ValueError("No graph-theory features selected.")

    features = np.concatenate(parts, axis=1).astype(np.float32, copy=False)

    meta = {
        "feature_names": feature_names,
        "description": (
            "Per-node weighted graph-theory features computed from one full weighted adjacency. "
            "For signed adjacencies, clustering and eigenvector centrality use abs(A)."
        ),
        "num_nodes": int(N),
        "num_features": int(features.shape[1]),
        "signed_input": bool(signed_input),
    }
    return features, meta


def append_weighted_graph_theory_to_node_features(
    node_features: np.ndarray,
    adj: np.ndarray,
    *,
    base_feature_names: Optional[Sequence[str]] = None,
    **graph_theory_kwargs,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Concatenate weighted node graph-theory features onto existing node features.

    Parameters
    ----------
    node_features : [N, F]
    adj : [N, N]

    Returns
    -------
    x_aug : [N, F + K]
    meta : dict with combined feature_names
    """
    X = _as_float32(node_features)
    if X.ndim != 2:
        raise ValueError(f"node_features must be [N, F], got shape={X.shape}")

    gt_x, gt_meta = compute_weighted_node_graph_theory_features(
        adj,
        **graph_theory_kwargs,
    )

    if X.shape[0] != gt_x.shape[0]:
        raise ValueError(
            f"node_features and adjacency disagree on num_nodes: "
            f"{X.shape[0]} vs {gt_x.shape[0]}"
        )

    x_aug = np.concatenate([X, gt_x], axis=1).astype(np.float32, copy=False)

    if base_feature_names is None:
        base_feature_names = [f"base_feat_{i}" for i in range(X.shape[1])]

    meta = {
        "feature_names": list(base_feature_names) + list(gt_meta["feature_names"]),
        "description": "Base node features concatenated with weighted node graph-theory features.",
        "base_dim": int(X.shape[1]),
        "graph_theory_dim": int(gt_x.shape[1]),
        "total_dim": int(x_aug.shape[1]),
        "graph_theory_meta": gt_meta,
    }
    return x_aug, meta

def build_graphs_from_payload_graphtheoryfeature(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
):
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        # ---------- node features ----------
        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")

            xfam = np.asarray(subj["features"][fam], dtype=np.float32)   # [W, N, F_fam]
            if xfam.ndim != 3:
                raise ValueError(
                    f"Feature family {fam!r} for subject {sid!r} must have shape [W, N, F], got {xfam.shape}"
                )

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[0] != ref_w or xfam.shape[1] != ref_n:
                    raise ValueError(
                        f"Feature family {fam!r} for subject {sid!r} has incompatible shape {xfam.shape}; "
                        f"expected same [W, N] as previous families = [{ref_w}, {ref_n}]"
                    )

            feat_list.append(xfam)

        if len(feat_list) == 0:
            raise ValueError("feature_families is empty")

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)   # [W, N, F_total]
        num_windows = node_x_all.shape[0]
        num_nodes = node_x_all.shape[1]

        # ---------- metadata ----------
        seg_ids = np.asarray(
            subj.get("segment_id", np.arange(num_windows)),
            dtype=np.int64,
        )
        start_samples = np.asarray(
            subj.get("start_sample", np.full(num_windows, -1)),
            dtype=np.int64,
        )

        if len(seg_ids) != num_windows:
            raise ValueError(
                f"segment_id length mismatch for subject {sid!r}: got {len(seg_ids)}, expected {num_windows}"
            )
        if len(start_samples) != num_windows:
            raise ValueError(
                f"start_sample length mismatch for subject {sid!r}: got {len(start_samples)}, expected {num_windows}"
            )

        # ---------- adjacency source ----------
        if edge_source == "connectivity":
            if connectivity_metric is None:
                adj_all = None
            else:
                if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
                    raise KeyError(
                        f"payload[{sid!r}]['connectivity'] missing metric {connectivity_metric!r}"
                    )
                adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

        elif edge_source == "aligned_adj":
            if subj.get("aligned_adj", None) is None:
                raise ValueError(
                    f"edge_source='aligned_adj' but payload[{sid!r}]['aligned_adj'] is None"
                )
            adj_all = np.asarray(subj["aligned_adj"], dtype=np.float32)

        else:
            raise ValueError(f"Unsupported edge_source={edge_source!r}")

        if adj_all is not None:
            if adj_all.ndim != 3:
                raise ValueError(
                    f"Adjacency tensor for subject {sid!r} must have shape [W, N, N], got {adj_all.shape}"
                )
            if adj_all.shape[0] != num_windows:
                raise ValueError(
                    f"Adjacency window count mismatch for subject {sid!r}: {adj_all.shape[0]} vs {num_windows}"
                )
            if adj_all.shape[1] != num_nodes or adj_all.shape[2] != num_nodes:
                raise ValueError(
                    f"Adjacency node count mismatch for subject {sid!r}: {adj_all.shape} vs num_nodes={num_nodes}"
                )

        # ---------- build one graph per window ----------
        for w in range(num_windows):
            x = node_x_all[w]   # [N, F_total]

            if adj_all is None:
                adj = np.eye(num_nodes, dtype=np.float32)
            else:
                adj = np.asarray(adj_all[w], dtype=np.float32).copy()

            if symmetrize_adj:
                adj = 0.5 * (adj + adj.T)

            if zero_diagonal:
                np.fill_diagonal(adj, 0.0)

            # optional: clean NaN/Inf
            adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)

            edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))



            x_aug, _ = append_weighted_graph_theory_to_node_features(
                node_features=x,          # [N, F]
                adj=adj,             # [N, N]
                signed_input=False,
            )

            g = Data(
                x=torch.tensor(x_aug, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long),
            )
            # For current pipeline compatibility:
            # - GCN path uses edge_weight
            # - GAT / other paths can use edge_attr
            g.edge_weight = edge_weight
            g.edge_attr = edge_weight.view(-1, 1)

            if attach_dense_adj:
                g.adj = torch.tensor(adj, dtype=torch.float32)

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            graphs.append(g)

    return graphs
# =========================================================
# Model
# =========================================================
class GNNEncoder(nn.Module):
    """
    Segment graph -> graph embedding
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)

        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.norm2 = GraphNorm(hidden_dim)

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )
        self.dropout = dropout

    def forward(self, data_batch: Batch) -> torch.Tensor:
        x = data_batch.x
        edge_index = data_batch.edge_index
        edge_weight = getattr(data_batch, "edge_weight", None)
        batch = data_batch.batch

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.norm2(x, batch)
        x = F.relu(x)

        graph_emb = global_mean_pool(x, batch)   # [num_graphs, hidden_dim]
        graph_emb = self.graph_proj(graph_emb)   # [num_graphs, emb_dim]
        return graph_emb


# =========================================================



class ConstrainedWeightedMeanMIL(nn.Module):
    """
    Learnable weighted mean MIL pooling with anti-collapse constraints.

    Compared with gated attention:
      - still learns segment weights
      - keeps a uniform mean-pooling component
      - penalizes focusing on too few segments
    """

    def __init__(
        self,
        in_dim: int,
        attn_dim: int = 64,
        dropout: float = 0.2,
        temperature: float = 2.0,
        gamma_max: float = 0.6,
        min_effective_frac: float = 0.35,
        min_entropy: float = 0.75,
        max_weight: float | None = None,
        lambda_entropy: float = 0.01,
        lambda_effective: float = 0.01,
        lambda_max_weight: float = 0.01,
        segment_dropout: float = 0.2,
        init_seed: int | None = None,   # <-- add this
    ):
        super().__init__()

        self.temperature = float(temperature)
        self.gamma_max = float(gamma_max)
        self.min_effective_frac = float(min_effective_frac)
        self.min_entropy = float(min_entropy)
        self.max_weight = max_weight

        self.lambda_entropy = float(lambda_entropy)
        self.lambda_effective = float(lambda_effective)
        self.lambda_max_weight = float(lambda_max_weight)
        self.segment_dropout = float(segment_dropout)

        self.score_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, attn_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, 1),
        )

        # Learn how far to move away from pure mean pooling.
        # gamma = gamma_max * sigmoid(mix_logit)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))

        self.last_reg_loss = None
        self.last_diagnostics = {}

        self.reset_parameters(init_seed=init_seed)

    def reset_parameters(self, init_seed: int | None = None):
        """
        Deterministic initialization using the same pipeline seed.

        If init_seed is provided, this temporarily sets the RNG state,
        initializes this module, then restores the previous RNG state.

        This prevents earlier random operations in the pipeline from changing
        model initialization.
        """

        def _reset():
            for m in self.modules():
                if m is self:
                    continue

                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            # Start close to mean pooling.
            # gamma = gamma_max * sigmoid(-2) ≈ 0.12 * gamma_max
            nn.init.constant_(self.mix_logit, -2.0)

        if init_seed is None:
            _reset()
            return

        # Save current RNG states
        cpu_rng_state = torch.random.get_rng_state()

        cuda_rng_states = None
        if torch.cuda.is_available():
            cuda_rng_states = torch.cuda.get_rng_state_all()

        # Temporarily use the same global pipeline seed
        torch.manual_seed(int(init_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(init_seed))

        _reset()

        # Restore previous RNG states so this does not disturb the rest of pipeline
        torch.random.set_rng_state(cpu_rng_state)

        if torch.cuda.is_available() and cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
            
    def _regularization(self, w: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        w: [K], sum = 1
        """
        k = w.numel()

        entropy = -(w * torch.log(w + 1e-12)).sum()
        norm_entropy = entropy / math.log(k) if k > 1 else entropy * 0.0

        effective_n = 1.0 / torch.sum(w ** 2).clamp_min(1e-12)
        target_effective_n = self.min_effective_frac * float(k)

        entropy_penalty = F.relu(self.min_entropy - norm_entropy).pow(2)
        effective_penalty = F.relu(target_effective_n - effective_n).pow(2) / float(k * k)

        if self.max_weight is None:
            # Dynamic cap. For K=10, cap around 0.25.
            max_w_allowed = max(2.5 / float(k), 0.15)
        else:
            max_w_allowed = float(self.max_weight)

        max_weight_penalty = F.relu(w.max() - max_w_allowed).pow(2)

        reg = (
            self.lambda_entropy * entropy_penalty
            + self.lambda_effective * effective_penalty
            + self.lambda_max_weight * max_weight_penalty
        )

        diag = {
            "norm_entropy": float(norm_entropy.detach().cpu()),
            "effective_n": float(effective_n.detach().cpu()),
            "top1_weight": float(w.max().detach().cpu()),
            "gamma": float((self.gamma_max * torch.sigmoid(self.mix_logit)).detach().cpu()),
        }

        return reg, diag

    def forward(
        self,
        graph_emb: torch.Tensor,
        bag_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        graph_emb: [total_segments, D]
        bag_sizes: [num_subjects]
        """
        bag_embs = []
        weight_list = []
        reg_losses = []

        diag_accum = {
            "norm_entropy": [],
            "effective_n": [],
            "top1_weight": [],
            "gamma": [],
        }

        gamma = self.gamma_max * torch.sigmoid(self.mix_logit)

        start = 0
        for size in bag_sizes.tolist():
            size = int(size)
            end = start + size

            h = graph_emb[start:end]  # [K, D]
            k = h.shape[0]

            scores = self.score_net(h).squeeze(-1)  # [K]

            # Segment dropout before softmax.
            # This prevents repeatedly relying on the same few segments.
            if self.training and self.segment_dropout > 0 and k > 1:
                keep = torch.rand(k, device=h.device) > self.segment_dropout
                if keep.sum() == 0:
                    keep[torch.randint(0, k, (1,), device=h.device)] = True
                scores = scores.masked_fill(~keep, -1e9)

            attn = torch.softmax(scores / self.temperature, dim=0)

            uniform = torch.full_like(attn, 1.0 / float(k))

            # Constrained learnable weighted mean.
            w = (1.0 - gamma) * uniform + gamma * attn
            w = w / (w.sum() + 1e-12)

            z = torch.sum(w.unsqueeze(-1) * h, dim=0)

            reg, diag = self._regularization(w)

            bag_embs.append(z)
            weight_list.append(w)
            reg_losses.append(reg)

            for key in diag_accum:
                diag_accum[key].append(diag[key])

            start = end

        bag_embs = torch.stack(bag_embs, dim=0)

        if len(reg_losses) > 0:
            self.last_reg_loss = torch.stack(reg_losses).mean()
        else:
            self.last_reg_loss = graph_emb.sum() * 0.0

        self.last_diagnostics = {
            key: float(sum(vals) / max(len(vals), 1))
            for key, vals in diag_accum.items()
        }

        return bag_embs, weight_list
class GatedAttentionMIL(nn.Module):
    """
    Ilse-style gated attention:
      a_i = w^T [tanh(Vh_i) * sigmoid(Uh_i)]
    """
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(
        self,
        graph_emb: torch.Tensor,
        bag_sizes: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        graph_emb: [total_graphs_in_batch, D]
        bag_sizes: [num_bags]

        Returns:
          bag_embs: [num_bags, D]
          attn_list: list of attention weights for each bag
        """
        bag_embs = []
        attn_list = []

        start = 0
        for size in bag_sizes.tolist():
            end = start + size
            h = graph_emb[start:end]  # [size, D]

            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # [size, 1]
            a = torch.softmax(a.squeeze(-1), dim=0)                       # [size]

            z = torch.sum(a.unsqueeze(-1) * h, dim=0)                     # [D]

            bag_embs.append(z)
            attn_list.append(a)
            start = end

        bag_embs = torch.stack(bag_embs, dim=0)
        return bag_embs, attn_list

class MeanMILPool(nn.Module):
    def forward(self, graph_emb: torch.Tensor, bag_sizes: torch.Tensor):
        bag_embs = []
        start = 0
        dummy_attn = []
        for size in bag_sizes.tolist():
            end = start + size
            h = graph_emb[start:end]
            z = h.mean(dim=0)
            bag_embs.append(z)
            dummy_attn.append(torch.ones(size, device=h.device) / size)
            start = end
        bag_embs = torch.stack(bag_embs, dim=0)
        return bag_embs, dummy_attn        



class SubjectBagGraphDataset(Dataset):
    def __init__(self, graphs, max_segments_per_subject=None, train=True):
        self.train = train
        self.max_segments_per_subject = max_segments_per_subject
        self.subject_to_graphs = defaultdict(list)
        self.subject_to_label = {}

        for g in graphs:
            sid = g.subject_id
            y = int(g.y.item()) if g.y.numel() == 1 else int(g.y[0].item())
            self.subject_to_graphs[sid].append(g)
            self.subject_to_label[sid] = y

        self.subject_ids = sorted(self.subject_to_graphs.keys())
        self.subject_labels = [self.subject_to_label[sid] for sid in self.subject_ids]
        self.num_node_features = graphs[0].x.shape[-1]
        self.num_nodes = graphs[0].x.shape[0]

        # make sure all graphs have the same number of nodes
        for i, g in enumerate(graphs):
            if g.x.shape[0] != self.num_nodes:
                raise ValueError(
                    f"RawNodeEdgeMLPEncoder requires fixed num_nodes, "
                    f"but graph {i} has {g.x.shape[0]} nodes while expected {self.num_nodes}."
                )
        # optional convenience
        self.summary_input_dim = graphs[0].summary_feat.numel() if hasattr(graphs[0], "summary_feat") else None

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        sid = self.subject_ids[idx]
        graphs = self.subject_to_graphs[sid]

        if self.max_segments_per_subject is not None and len(graphs) > self.max_segments_per_subject:
            if self.train:
                chosen = np.random.choice(len(graphs), self.max_segments_per_subject, replace=False)
                graphs = [graphs[i] for i in chosen]
            else:
                graphs = graphs[:self.max_segments_per_subject]

        return {
            "subject_id": sid,
            "label": self.subject_to_label[sid],
            "graphs": graphs,
        }

def metrics_to_row(metrics: dict, split_seed: int, fold: int, split_name: str):
    return {
        "split_seed": split_seed,
        # "train_seed": train_seed,
        "fold": fold,
        "split": split_name,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "conf_matrix": json.dumps(metrics["conf_matrix"].tolist()),
    }

def predictions_to_rows(metrics: dict, split_seed: int, fold: int, split_name: str, num_classes: int):
    rows = []
    for sid, y_true, y_pred, probs in zip(
        metrics["subject_ids"],
        metrics["y_true"],
        metrics["y_pred"],
        metrics["y_prob"],
    ):
        row = {
            "split_seed": split_seed,
            # "train_seed": train_seed,
            "fold": fold,
            "split": split_name,
            "subject_id": sid,
            "true_label": y_true,
            "pred_label": y_pred,
        }
        for c in range(num_classes):
            row[f"prob_{c}"] = probs[c]
        rows.append(row)
    return rows


from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_subject_embeddings_tsne(
    subject_rows,
    level,
    output_dir,
    split=None,   # e.g. "Training", "Validation", "Test"
    color_by="label",
    title="Subject Embeddings (t-SNE)",
    class_names=None,
    segment_point_size=16,
    subject_point_size=80,
    random_state=42,
):
    """
    subject_rows: list of dicts, each containing at least
        - embedding
        - subject_id
        - label
        - pred

    Behavior:
    - level == "subject":
        keep same style as before
    - level == "segment":
        color by subject
        marker shape by class
        no text labels on points
        hide legend if split == "Training"
    """

    if len(subject_rows) == 0:
        raise ValueError("subject_rows is empty")

    X = np.stack([r["embedding"] for r in subject_rows], axis=0)
    labels = np.array([r["label"] for r in subject_rows], dtype=int)
    subject_ids = np.array([str(r["subject_id"]) for r in subject_rows])

    unique_label_ids = sorted(np.unique(labels))
    if class_names is None:
        class_names = {cls: f"Class {cls}" for cls in unique_label_ids}

    perplexity = min(10, len(subject_rows) - 1)
    if perplexity < 1:
        raise ValueError("Not enough rows for t-SNE")

    Z2 = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity
    ).fit_transform(X)
    show_legend = not (split is not None and str(split).lower() == "train")

    plt.figure(figsize=(9, 7))

    # =====================================================
    # SUBJECT LEVEL: keep same behavior
    # =====================================================
    if level == "subject":
        if color_by == "label":
            c = labels
        elif color_by == "pred":
            preds = np.array([r["pred"] for r in subject_rows], dtype=int)
            
            c = preds
        else:
            raise ValueError("For level='subject', color_by must be 'label' or 'pred'")

        unique_classes = sorted(np.unique(c))
        for cls in unique_classes:
            idx = np.where(c == cls)[0]
            plt.scatter(
                Z2[idx, 0],
                Z2[idx, 1],
                s=subject_point_size,
                alpha=0.9,
                label=class_names.get(cls, f"Class {cls}")
            )

        for i, sid in enumerate(subject_ids):
            short_id = sid.replace("sub-", "s")
            plt.text(Z2[i, 0], Z2[i, 1], short_id, fontsize=5)

        if show_legend:
            plt.legend(title=color_by, loc="best")

        save_name = f"{split}_{level}_embeddings_tsne.png"

    # =====================================================
    # SEGMENT LEVEL
    #   - color by subject
    #   - shape by class
    #   - border line around each dot
    #   - no legend for Training
    # =====================================================
    elif level == "segment":
        unique_subjects = sorted(np.unique(subject_ids))

        # One color per subject
        cmap = plt.cm.get_cmap("gist_ncar", len(unique_subjects))
        subject_to_color = {
            sid: cmap(i) for i, sid in enumerate(unique_subjects)
        }

        # One label per subject
        subject_to_label = {}
        for sid, lbl in zip(subject_ids, labels):
            if sid in subject_to_label and subject_to_label[sid] != int(lbl):
                raise ValueError(f"Subject {sid} has inconsistent labels")
            subject_to_label[sid] = int(lbl)

        # Marker shape by class
        # class 0 -> circle, class 1 -> diamond, class 2 -> X
        marker_map = {
            0: "o",
            1: "D",
            2: "X",
        }

        fallback_markers = ["o", "D", "X", "^", "s", "P", "v", "<", ">"]

        for sid in unique_subjects:
            idx = np.where(subject_ids == sid)[0]
            lbl = subject_to_label[sid]

            marker = marker_map.get(lbl, fallback_markers[lbl % len(fallback_markers)])
            color = subject_to_color[sid]
            class_text = class_names.get(lbl, f"Class {lbl}")
            legend_text = f"{sid} - label {class_text}"

            plt.scatter(
                Z2[idx, 0],
                Z2[idx, 1],
                s=segment_point_size,
                marker=marker,
                c=[color],
                alpha=0.8,
                edgecolors="black",
                linewidths=0.35,
                label=legend_text,
            )

        if show_legend:
            plt.legend(
                title="Subject / class",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                fontsize=8
            )

        save_name = f"{split}_{level}_embeddings_tsne.png"

    else:
        raise ValueError("level must be 'subject' or 'segment'")

    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_subject_rows(subject_rows, save_path_pkl, save_path_csv=None):
    os.makedirs(os.path.dirname(save_path_pkl), exist_ok=True)

    with open(save_path_pkl, "wb") as f:
        pickle.dump(subject_rows, f)

    if save_path_csv is not None:
        csv_rows = []
        for r in subject_rows:
            row = {
                "subject_id": r["subject_id"],
                "label": r["label"],
                "pred": r["pred"],
            }
            prob = np.asarray(r["prob"])
            for j in range(len(prob)):
                row[f"prob_{j}"] = float(prob[j])
            csv_rows.append(row)

        df = pd.DataFrame(csv_rows)
        df.to_csv(save_path_csv, index=False)


def load_all_fold_data(pkl_path):
    with open(pkl_path, "rb") as f:
        all_fold_data = pickle.load(f)
    return all_fold_data

def rows_to_map(rows):
    return {r["subject_id"]: r for r in rows}


def orthogonal_procrustes_align(X_source, X_target):
    """
    Learn an orthogonal transform mapping source -> target.
    """
    mu_s = X_source.mean(axis=0, keepdims=True)
    mu_t = X_target.mean(axis=0, keepdims=True)

    Xs = X_source - mu_s
    Xt = X_target - mu_t

    M = Xs.T @ Xt
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt

    def transform(X_new):
        return (X_new - mu_s) @ R + mu_t

    return transform
def align_oof_test_embeddings_across_folds(all_fold_data, reference_fold=0):
    ref_entry = None
    for fd in all_fold_data:
        if fd["fold"] == reference_fold:
            ref_entry = fd
            break

    if ref_entry is None:
        raise ValueError(f"reference_fold={reference_fold} not found")

    ref_train_map = rows_to_map(ref_entry["train_rows"])
    aligned_rows = []

    for fd in all_fold_data:
        fold_idx = fd["fold"]

        if fold_idx == reference_fold:
            for r in fd["test_rows"]:
                rr = dict(r)
                rr["aligned_embedding"] = np.asarray(r["embedding"], dtype=np.float32)
                rr["source_fold"] = fold_idx
                aligned_rows.append(rr)
            continue

        cur_train_map = rows_to_map(fd["train_rows"])
        shared_anchor_ids = sorted(set(ref_train_map.keys()) & set(cur_train_map.keys()))

        if len(shared_anchor_ids) < 3:
            raise ValueError(
                f"Fold {fold_idx} has only {len(shared_anchor_ids)} shared training subjects "
                f"with reference fold {reference_fold}"
            )

        X_source = np.stack(
            [np.asarray(cur_train_map[sid]["embedding"], dtype=np.float32) for sid in shared_anchor_ids],
            axis=0
        )
        X_target = np.stack(
            [np.asarray(ref_train_map[sid]["embedding"], dtype=np.float32) for sid in shared_anchor_ids],
            axis=0
        )

        transform = orthogonal_procrustes_align(X_source, X_target)

        for r in fd["test_rows"]:
            rr = dict(r)
            emb = np.asarray(r["embedding"], dtype=np.float32)[None, :]
            rr["aligned_embedding"] = transform(emb)[0]
            rr["source_fold"] = fold_idx
            aligned_rows.append(rr)

    return aligned_rows
def plot_aligned_subject_embeddings_umap(
    aligned_rows,
    class_names=None,
    embedding_key="aligned_embedding",
    title="Aligned OOF Subject Embeddings (UMAP)",
    annotate_subject_ids=True,
    save_path=None
):
    try:
        import umap
    except ImportError:
        raise ImportError("Please install umap-learn first: pip install umap-learn")

    X = np.stack([np.asarray(r[embedding_key], dtype=np.float32) for r in aligned_rows], axis=0)
    y = np.array([r["label"] for r in aligned_rows])
    subject_ids = [r["subject_id"] for r in aligned_rows]

    short_ids = [s.replace('sub-', 's') for s in subject_ids]
    unique_classes = sorted(np.unique(y))
    if class_names is None:
        class_names = {cls: f"Class {cls}" for cls in unique_classes}

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.2, random_state=42)
    Z2 = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for cls in unique_classes:
        idx = np.where(y == cls)[0]
        plt.scatter(
            Z2[idx, 0],
            Z2[idx, 1],
            s=80,
            alpha=0.65,
            label=class_names.get(cls, f"Class {cls}")
        )

    if annotate_subject_ids:
        for i, sid in enumerate(short_ids):
            plt.text(Z2[i, 0], Z2[i, 1], str(sid), fontsize=5)

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="True class", loc="best")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def save_fold_subject_embeddings(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    fold_idx,
    save_dir="cv_subject_embeddings"
):
    os.makedirs(save_dir, exist_ok=True)

    train_subject_rows_f = collect_subject_embeddings(model, train_loader, device)
    val_subject_rows_f = collect_subject_embeddings(model, val_loader, device)
    test_subject_rows_f  = collect_subject_embeddings(model, test_loader, device)

    train_pkl = os.path.join(save_dir, f"fold_{fold_idx}_train_subject_rows.pkl")
    train_csv = os.path.join(save_dir, f"fold_{fold_idx}_train_subject_rows.csv")

    val_pkl = os.path.join(save_dir, f"fold_{fold_idx}_val_subject_rows.pkl")
    val_csv = os.path.join(save_dir, f"fold_{fold_idx}_val_subject_rows.csv")

    test_pkl = os.path.join(save_dir, f"fold_{fold_idx}_test_subject_rows.pkl")
    test_csv = os.path.join(save_dir, f"fold_{fold_idx}_test_subject_rows.csv")

    save_subject_rows(train_subject_rows_f, train_pkl, train_csv)
    save_subject_rows(val_subject_rows_f, val_pkl, val_csv)
    save_subject_rows(test_subject_rows_f, test_pkl, test_csv)

    print(f"Saved fold {fold_idx}:")
    # print(" ", train_pkl)
    print(" ", test_pkl)

    return train_subject_rows_f, val_subject_rows_f, test_subject_rows_f

def collect_segment_embeddings(model, loader, device):
    """
    Save one row per segment graph, not one row per subject bag.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for batch_dict in loader:
            batch_dict = move_batch_to_device(batch_dict, device)
            out = model(batch_dict)

            graph_emb = out["graph_emb"].detach().cpu().numpy()   # [num_graphs_total, D]
            bag_sizes = batch_dict["bag_sizes"].detach().cpu().numpy().tolist()
            labels = batch_dict["labels"].detach().cpu().numpy().tolist()
            subject_ids = list(batch_dict["subject_ids"])

            start = 0
            for sid, y, size in zip(subject_ids, labels, bag_sizes):
                end = start + size
                for local_seg_idx in range(size):
                    rows.append({
                        "subject_id": sid,
                        "label": int(y),
                        "segment_idx_in_bag": int(local_seg_idx),
                        "embedding": graph_emb[start + local_seg_idx].copy(),
                    })
                start = end

    return rows


def segment_fingerprint_metrics(segment_rows):
    """
    Compare:
      - same-subject similarity
      - same-class but different-subject similarity
      - different-class similarity
      - nearest-neighbor subject retrieval
      - nearest-neighbor class retrieval excluding same subject
    """
    X = np.stack([r["embedding"] for r in segment_rows], axis=0)
    y = np.array([r["label"] for r in segment_rows])
    sids = np.array([r["subject_id"] for r in segment_rows])

    # normalize for cosine
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)

    same_subject_means = []
    same_class_other_subject_means = []
    diff_class_means = []
    nn_same_subject = []
    nn_same_class_other_subject = []

    n = len(segment_rows)
    for i in range(n):
        same_subj = (sids == sids[i])
        same_cls = (y == y[i])
        other_subj = ~same_subj
        not_self = np.ones(n, dtype=bool)
        not_self[i] = False

        mask_same_subject = same_subj & not_self
        mask_same_class_other_subject = same_cls & other_subj
        mask_diff_class = ~same_cls

        if mask_same_subject.any():
            same_subject_means.append(S[i, mask_same_subject].mean())
        if mask_same_class_other_subject.any():
            same_class_other_subject_means.append(S[i, mask_same_class_other_subject].mean())
        if mask_diff_class.any():
            diff_class_means.append(S[i, mask_diff_class].mean())

        # nearest neighbor overall
        j = np.argmax(S[i])
        nn_same_subject.append(int(sids[j] == sids[i]))

        # nearest neighbor among different subjects only
        s_tmp = S[i].copy()
        s_tmp[same_subj] = -np.inf
        if np.isfinite(s_tmp).any():
            j2 = np.argmax(s_tmp)
            nn_same_class_other_subject.append(int(y[j2] == y[i]))

    out = {
        "mean_cosine_same_subject": float(np.mean(same_subject_means)) if same_subject_means else np.nan,
        "mean_cosine_same_class_other_subject": float(np.mean(same_class_other_subject_means)) if same_class_other_subject_means else np.nan,
        "mean_cosine_diff_class": float(np.mean(diff_class_means)) if diff_class_means else np.nan,
        "top1_same_subject_retrieval": float(np.mean(nn_same_subject)) if nn_same_subject else np.nan,
        "top1_same_class_other_subject_retrieval": float(np.mean(nn_same_class_other_subject)) if nn_same_class_other_subject else np.nan,
    }
    return out


def run_subject_id_probe(train_segment_rows, val_segment_rows):
    """
    This measures how strongly subject identity is encoded in graph embeddings.
    """
    X_train = np.stack([r["embedding"] for r in train_segment_rows], axis=0)
    y_train = np.array([r["subject_id"] for r in train_segment_rows])

    X_val = np.stack([r["embedding"] for r in val_segment_rows], axis=0)
    y_val = np.array([r["subject_id"] for r in val_segment_rows])

    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_val)

    return {
        "subject_id_probe_acc": float(accuracy_score(y_val, pred))
    }


def run_disease_probe(train_segment_rows, test_segment_rows):
    """
    Train disease probe on train subjects, evaluate on unseen test subjects.
    """
    X_train = np.stack([r["embedding"] for r in train_segment_rows], axis=0)
    y_train = np.array([r["label"] for r in train_segment_rows])

    X_test = np.stack([r["embedding"] for r in test_segment_rows], axis=0)
    y_test = np.array([r["label"] for r in test_segment_rows])

    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    return {
        "disease_probe_acc": float(accuracy_score(y_test, pred)),
        "disease_probe_bal_acc": float(balanced_accuracy_score(y_test, pred)),
        "disease_probe_macro_f1": float(f1_score(y_test, pred, average="macro")),
    }

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    losses = []
    y_true = []
    y_pred = []
    y_prob = []
    subject_ids_all = []
    attn_dump = {}

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        out = model(batch)

        logits = out["logits"]
        labels = batch["labels"]

        loss = criterion(logits, labels)
        if "reg_loss" in out:
            loss = loss + out["reg_loss"]
        losses.append(loss.item())

        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        # preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        y_prob.extend(probs.cpu().numpy().tolist())
        subject_ids_all.extend(batch["subject_ids"])

        for sid, attn in zip(batch["subject_ids"], out["attn_list"]):
            attn_dump[sid] = attn.detach().cpu().numpy()

    metrics = compute_subject_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    metrics["subject_ids"] = subject_ids_all
    metrics["attention"] = attn_dump
    return metrics

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D] subject embeddings
        labels:     [B]
        """
        device = embeddings.device
        labels = labels.view(-1, 1)

        z = F.normalize(embeddings, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature  # [B, B]

        # Remove self-comparison
        logits_mask = torch.ones_like(sim, device=device)
        logits_mask.fill_diagonal_(0)

        # Positive mask: same class, not itself
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask * logits_mask

        # Numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp_min(1e-8))

        # Average log-probability over positives
        pos_count = pos_mask.sum(dim=1)

        valid = pos_count > 0
        if valid.sum() == 0:
            return embeddings.sum() * 0.0

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1.0)

        loss = -mean_log_prob_pos[valid].mean()
        return loss
def grad_norm_report(model):
    groups = {}

    for name, p in model.named_parameters():
        if p.grad is None:
            continue

        if "graph_encoder" in name:
            group = "graph_encoder"
        elif "mil_pool" in name or "attention" in name or "attn" in name:
            group = "mil_pool"
        elif "classifier" in name:
            group = "classifier"
        elif "node" in name:
            group = "node_branch"
        elif "edge" in name or "adj" in name or "conn" in name:
            group = "connectivity_branch"
        else:
            group = "other"

        g = p.grad.detach().norm().item()
        groups.setdefault(group, []).append(g)

    return {
        k: {
            "mean_grad_norm": float(np.mean(v)),
            "max_grad_norm": float(np.max(v)),
        }
        for k, v in groups.items()
    }
def count_supcon_pairs(labels):
    labels = labels.detach().cpu()
    total_pos = 0
    per_class = {}

    for c in labels.unique().tolist():
        n = int((labels == c).sum())
        per_class[int(c)] = n
        total_pos += n * (n - 1)

    return per_class, total_pos


def soft_cross_entropy(logits, soft_targets, sample_weights=None):
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(soft_targets * log_probs).sum(dim=1)

    if sample_weights is not None:
        loss = loss * sample_weights

    return loss.mean()


def make_caueeg_soft_targets(y, device):
    # label order: 0=HC/normal, 1=MCI, 2=dementia
    table = torch.tensor([
        [0.90, 0.10, 0.00],
        [0.15, 0.70, 0.15],
        [0.00, 0.10, 0.90],
    ], dtype=torch.float32, device=device)

    return table[y]
def make_sample_weights(y, device):
    # 0=HC, 1=MCI, 2=Dementia
    class_weights = torch.tensor([1.0, 1.5, 1.0], device=device)
    return class_weights[y]
def ordinal_score_loss(logits, y):
    probs = torch.softmax(logits, dim=1)

    class_scores = torch.arange(
        logits.size(1),
        device=logits.device,
        dtype=torch.float32,
    )

    pred_score = (probs * class_scores).sum(dim=1)
    true_score = y.float()

    return F.smooth_l1_loss(pred_score, true_score)
def train_one_epoch(model, loader, optimizer, criterion, device, use_soft_targets=False, use_grad_norm_report=False, center_loss_fn = None, use_center_loss=False):
    model.train()
    losses = []
    y_true = []
    y_pred = []
    center_losses = []
    ce_losses = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        out = model(batch)
        if "pool_diagnostics" in out:
            print(out["pool_diagnostics"])
        # print(out["graph_emb"].shape) 
        # print(out["bag_emb"].shape)
        # print(out["logits"].shape) 
        logits = out["logits"]
        subject_emb = out["bag_emb"]
        avg_logits = logits.detach().mean(dim=0).cpu().numpy()
        # print("Avg logits:", avg_logits)
        labels = batch["labels"]
        if use_soft_targets:
            soft_y = make_caueeg_soft_targets(labels, logits.device)
            # loss = soft_cross_entropy(logits, soft_y)
            sample_weights = make_sample_weights(labels, logits.device)

            loss = soft_cross_entropy(
                logits,
                soft_y,
                sample_weights=sample_weights,
            )

            # Add ordinal severity auxiliary loss
            # loss_cls = soft_cross_entropy(logits, soft_y, sample_weights)
            # loss_ord = ordinal_score_loss(logits, y)

            # loss = loss_cls + 0.2 * loss_ord


        # per_class, total_pos = count_supcon_pairs(labels)
        # print("batch class counts:", per_class, "positive pairs:", total_pos)
        elif use_center_loss:
            lambda_center = 0.001
            ce_loss = criterion(logits, labels)
            center_loss = center_loss_fn(subject_emb, labels)
            loss = ce_loss + lambda_center * center_loss
            center_losses.append(float(center_loss.detach().cpu().item()))
            ce_losses.append(float(ce_loss.detach().cpu().item()))
        else:
            loss = criterion(logits, labels)
        
        if "reg_loss" in out:
            loss = loss + out["reg_loss"]
        
        loss.backward()
        if use_grad_norm_report:
            print("Gradient report:", grad_norm_report(model))

        optimizer.step()
        losses.append(loss.item())
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_subject_metrics(y_true, y_pred)
    # metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    if use_center_loss:
        metrics["ce_loss"] = float(np.mean(ce_losses)) if ce_losses else 0.0
        metrics["center_loss"] = float(np.mean(center_losses)) if center_losses else 0.0


    return metrics    

import numpy as np

def train_one_epoch_subject_invariant(
    model,
    loader,
    optimizer,
    criterion,
    device,
    lambda_supcon=0.01,
    supcon_temperature=0.2,
):
    model.train()

    losses = []
    ce_losses = []
    supcon_losses = []

    y_true = []
    y_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        out = model(batch)
        if "pool_diagnostics" in out:
            print(out["pool_diagnostics"])
        logits = out["logits"]
        labels = batch["labels"]
        per_class, total_pos = count_supcon_pairs(labels)
        print("batch class counts:", per_class, "positive pairs:", total_pos)
        ce_loss = criterion(logits, labels)

        # Use subject-level bag embedding, not segment graph_emb
        bag_emb = out["bag_emb"]
        supcon_loss = supervised_contrastive_loss(
            bag_emb,
            labels,
            temperature=supcon_temperature,
        )

        loss = ce_loss + lambda_supcon * supcon_loss

        if "reg_loss" in out:
            loss = loss + out["reg_loss"]

        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        ce_losses.append(float(ce_loss.item()))
        supcon_losses.append(float(supcon_loss.item()))

        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_subject_metrics(y_true, y_pred)
    metrics["y_pred"] = y_pred
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["ce_loss"] = float(np.mean(ce_losses)) if ce_losses else 0.0
    metrics["supcon_loss"] = float(np.mean(supcon_losses)) if supcon_losses else 0.0

    return metrics


# =========================
# Helper functions
# =========================
def extract_edge_weight(pyg_batch: Batch) -> Optional[Tensor]:
    """
    Safely extract scalar edge weights from pyg_batch.edge_attr if present.
    Returns shape [num_edges] or None.
    """
    edge_attr = getattr(pyg_batch, "edge_attr", None)
    if edge_attr is None:
        return None

    if edge_attr.dim() == 1:
        return edge_attr.float()

    if edge_attr.size(-1) == 1:
        return edge_attr.squeeze(-1).float()

    # if multi-dimensional edge_attr exists, use the first channel
    return edge_attr[:, 0].float()


def pool_nodes(x: Tensor, batch: Tensor, pool: str = "mean") -> Tensor:
    """
    x:     [num_nodes, feat_dim]
    batch: [num_nodes]
    """
    if pool == "mean":
        return global_mean_pool(x, batch)
    if pool == "max":
        return global_max_pool(x, batch)
    if pool == "add":
        return global_add_pool(x, batch)
    raise ValueError(f"Unknown pool='{pool}'. Use one of ['mean', 'max', 'add'].")


def pool_single_graph(x: Tensor, pool: str = "mean") -> Tensor:
    """
    x: [num_nodes_in_one_graph, feat_dim]
    returns [1, feat_dim]
    """
    if pool == "mean":
        return x.mean(dim=0, keepdim=True)
    if pool == "max":
        return x.max(dim=0, keepdim=True).values
    if pool == "add":
        return x.sum(dim=0, keepdim=True)
    raise ValueError(f"Unknown pool='{pool}'. Use one of ['mean', 'max', 'add'].")


# =========================
# 1) GraphSAGE Encoder
# =========================
class GraphSAGEEncoder(nn.Module):
    """
    Simple inductive baseline.
    Ignores edge weights in this version.
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pool: str = "mean",
        jk_mode: str = "last",   # "last" or "cat"
    ):
        super().__init__()
        assert num_layers >= 1
        assert jk_mode in {"last", "cat"}

        self.dropout = dropout
        self.pool = pool
        self.jk_mode = jk_mode

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        in_dim = num_node_features
        for _ in range(num_layers):
            self.convs.append(SAGEConv(in_dim, hidden_dim, aggr="mean"))
            self.norms.append(GraphNorm(hidden_dim))
            in_dim = hidden_dim

        out_dim = hidden_dim if jk_mode == "last" else hidden_dim * num_layers
        self.proj = nn.Linear(out_dim, graph_emb_dim)

    def forward(self, pyg_batch: Batch) -> Tensor:
        x = pyg_batch.x
        edge_index = pyg_batch.edge_index
        batch = pyg_batch.batch

        layer_outs = []
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            layer_outs.append(x)

        if self.jk_mode == "cat":
            x = torch.cat(layer_outs, dim=-1)
        else:
            x = layer_outs[-1]

        x = self.proj(x)
        graph_emb = pool_nodes(x, batch, pool=self.pool)
        return graph_emb


# =========================
# 2) GCNII Encoder
# =========================
class GCNIIEncoder(nn.Module):
    """
    Better for testing whether deeper residual graph propagation helps.
    Uses edge weights if available.
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 8,
        dropout: float = 0.2,
        alpha: float = 0.1,
        theta: float = 0.5,
        shared_weights: bool = True,
        pool: str = "mean",
        use_edge_weight: bool = True,
    ):
        super().__init__()
        assert num_layers >= 1

        self.dropout = dropout
        self.pool = pool
        self.use_edge_weight = use_edge_weight

        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    channels=hidden_dim,
                    alpha=alpha,
                    theta=theta,
                    layer=layer_idx + 1,
                    shared_weights=shared_weights,
                    normalize=True,
                )
            )
            self.norms.append(GraphNorm(hidden_dim))

        self.out_proj = nn.Linear(hidden_dim, graph_emb_dim)

    def forward(self, pyg_batch: Batch) -> Tensor:
        x = pyg_batch.x
        edge_index = pyg_batch.edge_index
        batch = pyg_batch.batch

        edge_weight = extract_edge_weight(pyg_batch) if self.use_edge_weight else None

        x0 = self.input_proj(x)
        x0 = F.relu(x0)
        x = F.dropout(x0, p=self.dropout, training=self.training)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, x0, edge_index, edge_weight=edge_weight)
            x = norm(x, batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.out_proj(x)
        graph_emb = pool_nodes(x, batch, pool=self.pool)
        return graph_emb


# =========================
# 3) H2GCN-like Encoder
# =========================
class H2GCNLikeEncoder(nn.Module):
    """
    - separate ego information from neighbor information
    - use 1-hop and strict 2-hop neighborhoods
    - combine intermediate representations
    - this version binarizes adjacency
    - it processes one graph at a time inside a batched PyG Batch
    """

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        pool: str = "mean",
    ):
        super().__init__()
        assert num_layers >= 1

        self.dropout = dropout
        self.pool = pool
        self.num_layers = num_layers

        self.input_proj = nn.Linear(num_node_features, hidden_dim)

        # each layer sees:
        # [ego_init, 1-hop(current), 2-hop(current)]
        self.linears = nn.ModuleList([
            nn.Linear(hidden_dim * 3, hidden_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

        # concatenate outputs from all layers + initial embedding
        self.out_proj = nn.Linear(hidden_dim * (num_layers + 1), graph_emb_dim)

    @staticmethod
    def _sym_norm(adj: Tensor) -> Tensor:
        """
        Symmetric normalization: D^{-1/2} A D^{-1/2}
        adj: [N, N]
        """
        deg = adj.sum(dim=-1)
        deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
        return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]

    def _build_h2_mats(self, adj: Tensor) -> tuple[Tensor, Tensor]:
        """
        Build:
        - A1: strict 1-hop adjacency (no self-loop)
        - A2: strict 2-hop adjacency excluding 1-hop and self
        """
        n = adj.size(0)
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype)

        adj = (adj > 0).float()
        adj = adj * (1.0 - eye)  # remove self-loops

        # strict 2-hop
        two_hop = ((adj @ adj) > 0).float()
        two_hop = two_hop * (1.0 - adj) * (1.0 - eye)

        a1 = self._sym_norm(adj)
        a2 = self._sym_norm(two_hop)
        return a1, a2

    def _forward_one_graph(self, xg: Tensor, edge_index_g: Tensor) -> Tensor:
        """
        xg:           [N, F]
        edge_index_g: [2, E]
        returns graph embedding [1, graph_emb_dim]
        """
        n = xg.size(0)

        if n == 1:
            # degenerate graph
            x0 = self.input_proj(xg)
            x0 = F.relu(x0)
            x_out = self.out_proj(x0)
            return pool_single_graph(x_out, pool=self.pool)

        adj = to_dense_adj(edge_index_g, max_num_nodes=n).squeeze(0)
        a1, a2 = self._build_h2_mats(adj)

        x0 = self.input_proj(xg)
        x0 = F.relu(x0)

        h = x0
        reps = [x0]

        for linear, norm in zip(self.linears, self.norms):
            h1 = a1 @ h
            h2 = a2 @ h

            h = torch.cat([x0, h1, h2], dim=-1)
            h = linear(h)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

            reps.append(h)

        h = torch.cat(reps, dim=-1)
        h = self.out_proj(h)

        graph_emb = pool_single_graph(h, pool=self.pool)
        return graph_emb

    def forward(self, pyg_batch: Batch) -> Tensor:
        x = pyg_batch.x
        edge_index = pyg_batch.edge_index
        batch = pyg_batch.batch

        graph_embs = []
        num_graphs = int(batch.max().item()) + 1

        for gid in range(num_graphs):
            node_ids = (batch == gid).nonzero(as_tuple=False).view(-1)
            xg = x[node_ids]

            edge_index_g, _ = subgraph(
                subset=node_ids,
                edge_index=edge_index,
                edge_attr=None,
                relabel_nodes=True,
                num_nodes=x.size(0),
            )

            graph_emb_g = self._forward_one_graph(xg, edge_index_g)
            graph_embs.append(graph_emb_g)

        return torch.cat(graph_embs, dim=0)


class RawNodeEdgeMLPEncoder(nn.Module):
    """
    Non-GNN graph encoder using:
      - raw node features
      - raw adjacency / edge weights

    Assumes:
      - fixed number of nodes per graph
      - consistent node ordering across graphs
    """
    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_upper_triangle: bool = True,
        symmetrize_adj: bool = True,
        edge_mode: str = "topology_weighted",
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.use_upper_triangle = use_upper_triangle
        self.symmetrize_adj = symmetrize_adj
        self.edge_mode = edge_mode.lower()


        if self.edge_mode not in ["topology_weighted", "topology_binary", "full_adj"]:
            raise ValueError(f"Unsupported edge_mode={edge_mode}")

        node_input_dim = num_nodes * num_node_features
        if use_upper_triangle:
            edge_input_dim = num_nodes * (num_nodes - 1) // 2
        else:
            edge_input_dim = num_nodes * num_nodes

        # Node branch
        self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # Edge branch
        self.edge_mlp, edge_last_dim = make_mlp(edge_input_dim, edge_hidden_dims, dropout)
        self.edge_proj = nn.Linear(edge_last_dim, branch_emb_dim)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _get_topology_weighted_adj(self, pyg_batch):
        edge_attr = getattr(pyg_batch, "edge_attr", None)
        if edge_attr is None:
            edge_attr = getattr(pyg_batch, "edge_weight", None)

        if edge_attr is not None:
            if edge_attr.dim() > 1:
                if edge_attr.size(-1) == 1:
                    edge_attr = edge_attr.squeeze(-1)
                else:
                    edge_attr = edge_attr[:, 0]

        adj = to_dense_adj(
            pyg_batch.edge_index,
            batch=pyg_batch.batch,
            edge_attr=edge_attr,
            max_num_nodes=self.num_nodes,
        )
        return adj

    def _get_topology_binary_adj(self, pyg_batch):
        num_edges = pyg_batch.edge_index.size(1)
        binary_edge_attr = torch.ones(
            num_edges,
            device=pyg_batch.edge_index.device,
            dtype=pyg_batch.x.dtype,
        )

        adj = to_dense_adj(
            pyg_batch.edge_index,
            batch=pyg_batch.batch,
            edge_attr=binary_edge_attr,
            max_num_nodes=self.num_nodes,
        )
        return adj

    def forward(self, pyg_batch):
        """
        pyg_batch.x         : [total_nodes, F]
        pyg_batch.batch     : [total_nodes]
        pyg_batch.edge_index: [2, total_edges]
        pyg_batch.edge_attr : [total_edges, 1] or [total_edges] or absent
        """
        # ----- node branch -----
        dense_x, mask = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [num_graphs, N, F]

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}"
            )

        node_x = dense_x.reshape(dense_x.size(0), -1)  # [num_graphs, N*F]
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)              # [num_graphs, branch_emb_dim]

        # ----- edge branch -----
        if self.edge_mode == "topology_weighted":
            adj = self._get_topology_weighted_adj(pyg_batch)

        elif self.edge_mode == "topology_binary":
            adj = self._get_topology_binary_adj(pyg_batch)

        else:
            raise ValueError(f"Unsupported edge_mode={self.edge_mode}")

        if self.symmetrize_adj:
            adj = 0.5 * (adj + adj.transpose(1, 2))

        if self.use_upper_triangle:
            iu = torch.triu_indices(
                self.num_nodes, self.num_nodes, offset=1, device=adj.device
            )
            edge_x = adj[:, iu[0], iu[1]]             # [num_graphs, N*(N-1)/2]
        else:
            edge_x = adj.reshape(adj.size(0), -1)     # [num_graphs, N*N]

        edge_h = self.edge_mlp(edge_x)
        edge_emb = self.edge_proj(edge_h)             # [num_graphs, branch_emb_dim]

        # ----- fuse -----
        fused = torch.cat([node_emb, edge_emb], dim=1)
        graph_emb = self.fusion(fused)                # [num_graphs, emb_dim]

        return graph_emb



class RawNodeMLPEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        proj_dim: int = 128,   # use 64 for strict ablation, 128 for capacity-matched
        emb_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        node_input_dim = num_nodes * num_node_features
        self.num_nodes = num_nodes

        self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, proj_dim)

        self.fusion = nn.Sequential(
            nn.Linear(proj_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, pyg_batch):
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")

        node_x = dense_x.reshape(dense_x.size(0), -1)
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)
        graph_emb = self.fusion(node_emb)
        return graph_emb
# =========================================================
# Graph-level MLP encoder
# =========================================================
class MLPGraphEncoder(nn.Module):
    """
    Turn each graph into one embedding by:
      1) pooling node features inside the graph
      2) applying an MLP

    This lets you compare MLP vs GNN while keeping the same
    batch_dict["pyg_batch"] interface.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        emb_dim: int = 128,
        dropout: float = 0.2,
        node_pool: str = "mean",   # "mean", "sum", "max"
    ):
        super().__init__()

        self.node_pool = node_pool

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h

        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(prev, emb_dim)

    def pool_nodes(self, x, batch):
        if self.node_pool == "mean":
            return global_mean_pool(x, batch)
        elif self.node_pool == "sum":
            return global_add_pool(x, batch)
        elif self.node_pool == "max":
            return global_max_pool(x, batch)
        else:
            raise ValueError(f"Unsupported node_pool={self.node_pool}")

    def forward(self, pyg_batch):
        # pyg_batch.x      : [num_nodes_total, in_dim]
        # pyg_batch.batch  : [num_nodes_total]
        graph_x = self.pool_nodes(pyg_batch.x, pyg_batch.batch)   # [num_graphs, in_dim]
        h = self.mlp(graph_x)                                     # [num_graphs, last_hidden]
        emb = self.proj(h)                                        # [num_graphs, emb_dim]
        return emb


def plot_mil_learning_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(history)
    df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)

    epochs = df["epoch"]

    # -----------------------------
    # 1. Loss curve
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, df["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, df["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_loss.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 2. Accuracy curve
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, df["train_acc"], marker="o", label="train_acc")
    plt.plot(epochs, df["val_acc"], marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training / Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_accuracy.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 3. Balanced accuracy curve
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, df["train_bal_acc"], marker="o", label="train_bal_acc")
    plt.plot(epochs, df["val_bal_acc"], marker="o", label="val_bal_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title("Training / Validation Balanced Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_balanced_accuracy.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 4. Macro-F1 curve
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, df["train_macro_f1"], marker="o", label="train_macro_f1")
    plt.plot(epochs, df["val_macro_f1"], marker="o", label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Training / Validation Macro-F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "curve_macro_f1.png"), dpi=300)
    plt.close()

    # -----------------------------
    # 5. Learning-rate curve
    # -----------------------------
    if "lr" in df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, df["lr"], marker="o", label="lr_before_scheduler")

        if "lr_after_scheduler" in df.columns:
            plt.plot(
                epochs,
                df["lr_after_scheduler"],
                marker="o",
                label="lr_after_scheduler",
            )

        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.yscale("log")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "curve_learning_rate.png"), dpi=300)
        plt.close()

    return df

def plot_mil_learning_summary(history, save_path):
    df = pd.DataFrame(history)
    epochs = df["epoch"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    items = [
        ("loss", "Loss"),
        ("acc", "Accuracy"),
        ("bal_acc", "Balanced Accuracy"),
        ("macro_f1", "Macro-F1"),
    ]

    for ax, (key, title) in zip(axes.ravel(), items):
        ax.plot(epochs, df[f"train_{key}"], marker="o", label=f"train_{key}")
        ax.plot(epochs, df[f"val_{key}"], marker="o", label=f"val_{key}")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("MIL Learning Curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassCenterLoss(nn.Module):
    def __init__(self, num_classes, emb_dim, normalize=True):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, emb_dim))
        self.normalize = normalize

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D] subject embeddings after MIL pooling
        labels:     [B]
        """
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
            centers = F.normalize(self.centers, dim=1)
        else:
            centers = self.centers

        target_centers = centers[labels]
        loss = ((embeddings - target_centers) ** 2).sum(dim=1).mean()
        return loss