from __future__ import annotations

from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from mil_utils import (
    GNNEncoder,
    RawNodeEdgeMLPEncoder,
    GraphSAGEEncoder,
    GCNIIEncoder,
    H2GCNLikeEncoder,
)


# =========================================================
# Region / candidate defaults
# =========================================================

DEFAULT_REGION_TO_CHANNELS: Dict[str, List[str]] = {
    "frontal":   ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"],
    "central":   ["C3", "Cz", "C4"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "parietal":  ["P3", "Pz", "P4"],
    "occipital": ["O1", "O2"],
}

# Candidate name, metric, band
DEFAULT_CONN_CANDIDATES: List[Tuple[str, str, Optional[str]]] = [
    # ("wpli-theta",      "wpli",      "theta"),
    # ("wpli-alpha",      "wpli",      "alpha"),
    # ("wpli-beta",       "wpli",      "beta"),
    ("pli-theta",       "pli",       "theta"),
    ("pli-alpha",       "pli",       "alpha"),
    ("coherence-alpha", "coherence", "alpha"),
    ("pearson-alpha",   "pearson",   "alpha"),
    # ("spearman-alpha",  "spearman",  "alpha"),
]


# =========================================================
# Small helpers
# =========================================================

def _as_torch_float(x: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _symmetrize_last2(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(-1, -2))


def _lookup_connectivity_candidate(
    connectivity_sources: Mapping[str, Any],
    metric: str,
    band: Optional[str],
    candidate_name: str,
) -> Any:
    """
    Supported input styles:
      1) flat candidate-name dict:
           {"wpli-alpha": adj, "coherence-alpha": adj, ...}
      2) flat tuple-key dict:
           {("wpli", "alpha"): adj, ...}
      3) nested metric/band dict:
           {"wpli": {"alpha": adj, "beta": adj}, "pearson": {"alpha": adj}}
      4) metric-only dict (fallback when band-specific entry is unavailable):
           {"pearson": adj}

    The fallback in (4) is useful if a metric is not stored band-wise.
    """
    if candidate_name in connectivity_sources:
        return connectivity_sources[candidate_name]

    tuple_key = (metric, band)
    if tuple_key in connectivity_sources:
        return connectivity_sources[tuple_key]

    if metric in connectivity_sources:
        value = connectivity_sources[metric]
        if isinstance(value, Mapping):
            if band in value:
                return value[band]
            if candidate_name in value:
                return value[candidate_name]
        else:
            return value

    raise KeyError(
        f"Could not find connectivity candidate '{candidate_name}' "
        f"(metric={metric!r}, band={band!r}) in connectivity_sources."
    )


def _pack_flat_segments_to_padded(
    seg_emb_flat: torch.Tensor,
    bag_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert flattened segment embeddings into padded [B, T, D].

    Inputs
    ------
    seg_emb_flat : [sum_T, D]
    bag_sizes    : [B]

    Returns
    -------
    seg_emb_padded : [B, T_max, D]
    bag_mask       : [B, T_max]  True for valid segments
    """
    if seg_emb_flat.ndim != 2:
        raise ValueError(f"seg_emb_flat must have shape [sum_T, D], got {tuple(seg_emb_flat.shape)}")

    B = int(bag_sizes.numel())
    T_max = int(bag_sizes.max().item()) if B > 0 else 0
    D = int(seg_emb_flat.size(-1))

    padded = seg_emb_flat.new_zeros((B, T_max, D))
    mask = torch.zeros((B, T_max), dtype=torch.bool, device=seg_emb_flat.device)

    start = 0
    for b, size in enumerate(bag_sizes.tolist()):
        end = start + size
        padded[b, :size] = seg_emb_flat[start:end]
        mask[b, :size] = True
        start = end

    if start != seg_emb_flat.size(0):
        raise ValueError(
            f"bag_sizes sum to {start}, but seg_emb_flat has {seg_emb_flat.size(0)} rows."
        )

    return padded, mask


# =========================================================
# 1. Channel -> region adjacency
# =========================================================

def channel_adj_to_region_adj(
    channel_adj: np.ndarray | torch.Tensor,
    channel_names: Sequence[str],
    region_to_channels: Optional[Mapping[str, Sequence[str]]] = None,
    *,
    reduce: str = "mean",
    include_within_region_on_diag: bool = True,
) -> torch.Tensor:
    """
    Convert a channel-level adjacency [N, N] into a region-level adjacency [R, R].

    Rules
    -----
    - off-diagonal entry (r, s): mean over all channel pairs between regions r and s
    - diagonal entry (r, r): mean over within-region off-diagonal pairs if possible
    - if a region has only one channel, its diagonal falls back to 0.0

    Parameters
    ----------
    channel_adj:
        Dense connectivity matrix [N, N].
    channel_names:
        Channel order corresponding to the rows/cols of channel_adj.
    region_to_channels:
        Mapping like {"frontal": [...], ...}. Defaults to DEFAULT_REGION_TO_CHANNELS.
    reduce:
        Currently only "mean" is supported.
    include_within_region_on_diag:
        Whether to summarize within-region connectivity on the diagonal.

    Returns
    -------
    region_adj : torch.Tensor [R, R]
    """
    if reduce != "mean":
        raise ValueError(f"Unsupported reduce={reduce!r}; only 'mean' is supported.")

    region_to_channels = dict(region_to_channels or DEFAULT_REGION_TO_CHANNELS)
    region_names = list(region_to_channels.keys())

    A = _as_torch_float(channel_adj)
    if A.ndim != 2 or A.size(0) != A.size(1):
        raise ValueError(f"channel_adj must be square [N, N], got {tuple(A.shape)}")
    A = _symmetrize_last2(A)

    name_to_idx = {str(ch): i for i, ch in enumerate(channel_names)}
    region_indices: Dict[str, List[int]] = {}
    for region_name, ch_list in region_to_channels.items():
        idx = [name_to_idx[ch] for ch in ch_list if ch in name_to_idx]
        if len(idx) == 0:
            raise ValueError(f"Region {region_name!r} has no valid channels in channel_names.")
        region_indices[region_name] = idx

    R = len(region_names)
    region_adj = torch.zeros((R, R), dtype=A.dtype, device=A.device)

    for i, region_i in enumerate(region_names):
        idx_i = region_indices[region_i]
        for j, region_j in enumerate(region_names):
            idx_j = region_indices[region_j]

            if i == j:
                if not include_within_region_on_diag:
                    value = A.new_tensor(0.0)
                else:
                    sub = A[idx_i][:, idx_i]
                    if sub.size(0) <= 1:
                        value = A.new_tensor(0.0)
                    else:
                        tri = torch.triu_indices(sub.size(0), sub.size(1), offset=1, device=sub.device)
                        vals = sub[tri[0], tri[1]]
                        value = vals.mean() if vals.numel() > 0 else A.new_tensor(0.0)
            else:
                sub = A[idx_i][:, idx_j]
                value = sub.mean() if sub.numel() > 0 else A.new_tensor(0.0)

            region_adj[i, j] = value

    region_adj = _symmetrize_last2(region_adj)
    return region_adj


# =========================================================
# 2. Candidate bank builder
# =========================================================

def build_conn_bank_for_segment(
    connectivity_sources: Mapping[str, Any],
    channel_names: Sequence[str],
    region_to_channels: Optional[Mapping[str, Sequence[str]]] = None,
    candidate_specs: Optional[Sequence[Tuple[str, str, Optional[str]]]] = None,
) -> torch.Tensor:
    """
    Build a candidate regional connectivity bank for one segment.

    Parameters
    ----------
    connectivity_sources:
        Per-segment connectivity matrices. Supported forms are described in
        _lookup_connectivity_candidate(...).
    channel_names:
        Channel names aligned to the channel-level matrices.
    region_to_channels:
        Region definition. Defaults to DEFAULT_REGION_TO_CHANNELS.
    candidate_specs:
        Sequence of (candidate_name, metric, band). Defaults to DEFAULT_CONN_CANDIDATES.

    Returns
    -------
    conn_bank : torch.Tensor [K, R, R]
    """
    region_to_channels = dict(region_to_channels or DEFAULT_REGION_TO_CHANNELS)
    candidate_specs = list(candidate_specs or DEFAULT_CONN_CANDIDATES)

    region_graphs: List[torch.Tensor] = []
    for candidate_name, metric, band in candidate_specs:
        channel_adj = _lookup_connectivity_candidate(
            connectivity_sources=connectivity_sources,
            metric=metric,
            band=band,
            candidate_name=candidate_name,
        )
        region_adj = channel_adj_to_region_adj(
            channel_adj=channel_adj,
            channel_names=channel_names,
            region_to_channels=region_to_channels,
            reduce="mean",
            include_within_region_on_diag=True,
        )
        region_graphs.append(region_adj)

    conn_bank = torch.stack(region_graphs, dim=0)  # [K, R, R]
    return conn_bank


# =========================================================
# Optional collate for the new model
# =========================================================

def collate_subject_bags_v2(batch: List[dict]) -> Dict[str, Any]:
    """
    New bag format for SubjectMILClassifierV2.

    Each dataset item is expected to look like:
        {
            "subject_id": str,
            "label": int,
            "graphs": list[PyG Data],
        }

    Required per-segment graph attributes for the new branch:
        g.x         : [N, F]
        g.conn_bank : [K, R, R]

    Returned keys
    -------------
    - x_bag     : [B, T_max, N, F]
    - conn_bank : [B, T_max, K, R, R]
    - bag_mask  : [B, T_max]         True for valid segment positions
    - y         : [B]

    Compatibility keys retained for the existing node-graph encoder path:
    - pyg_batch
    - bag_sizes
    - labels
    - subject_ids
    """
    all_graphs: List[Any] = []
    bag_sizes: List[int] = []
    labels: List[int] = []
    subject_ids: List[str] = []
    segment_ids_per_subject: List[List[Any]] = []

    if len(batch) == 0:
        raise ValueError("Empty batch passed to collate_subject_bags_v2")

    # infer shapes from the first graph
    first_graph = batch[0]["graphs"][0]
    if not hasattr(first_graph, "conn_bank"):
        raise AttributeError(
            "Each graph must have attribute 'conn_bank' with shape [K, R, R] "
            "for SubjectMILClassifierV2."
        )

    x0 = _as_torch_float(first_graph.x)
    c0 = _as_torch_float(first_graph.conn_bank)
    if c0.ndim != 3:
        raise ValueError(f"g.conn_bank must have shape [K, R, R], got {tuple(c0.shape)}")

    N, F_dim = int(x0.size(0)), int(x0.size(1))
    K, R1, R2 = int(c0.size(0)), int(c0.size(1)), int(c0.size(2))
    if R1 != R2:
        raise ValueError(f"g.conn_bank must be square per candidate, got {tuple(c0.shape)}")
    R = R1

    B = len(batch)
    T_max = max(len(item["graphs"]) for item in batch)

    x_bag = torch.zeros((B, T_max, N, F_dim), dtype=torch.float32)
    conn_bank = torch.zeros((B, T_max, K, R, R), dtype=torch.float32)
    bag_mask = torch.zeros((B, T_max), dtype=torch.bool)

    for b, item in enumerate(batch):
        gs = item["graphs"]
        all_graphs.extend(gs)
        bag_sizes.append(len(gs))
        labels.append(int(item["label"]))
        subject_ids.append(item["subject_id"])

        seg_ids_here: List[Any] = []
        for t, g in enumerate(gs):
            gx = _as_torch_float(g.x)
            gc = _as_torch_float(g.conn_bank)

            if gx.shape != (N, F_dim):
                raise ValueError(
                    f"All graphs must have the same x shape. Expected {(N, F_dim)}, got {tuple(gx.shape)}"
                )
            if gc.shape != (K, R, R):
                raise ValueError(
                    f"All graphs must have the same conn_bank shape. Expected {(K, R, R)}, got {tuple(gc.shape)}"
                )

            x_bag[b, t] = gx
            conn_bank[b, t] = gc
            bag_mask[b, t] = True
            seg_ids_here.append(getattr(g, "segment_id", None))

        segment_ids_per_subject.append(seg_ids_here)

    pyg_batch = Batch.from_data_list(all_graphs)

    return {
        "pyg_batch": pyg_batch,
        "x_bag": x_bag,                                   # [B, T, N, F]
        "conn_bank": conn_bank,                           # [B, T, K, R, R]
        "bag_mask": bag_mask,                             # [B, T]
        "bag_sizes": torch.tensor(bag_sizes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "y": torch.tensor(labels, dtype=torch.long),      # alias requested by user
        "subject_ids": subject_ids,
        "segment_ids_per_subject": segment_ids_per_subject,
    }


# =========================================================
# 3. Candidate graph mixer
# =========================================================

class CandidateGraphMixer(nn.Module):
    def __init__(self, num_candidates: int):
        super().__init__()
        self.num_candidates = int(num_candidates)
        self.logits_alpha = nn.Parameter(torch.zeros(self.num_candidates, dtype=torch.float32))

    def alpha(self) -> torch.Tensor:
        return torch.softmax(self.logits_alpha, dim=0)

    def forward(self, conn_bank: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        conn_bank: [B, T, K, R, R] or [S, K, R, R]

        Returns
        -------
        mixed_adj : same leading dims without K -> [..., R, R]
        alpha     : [K]
        """
        if conn_bank.ndim not in (4, 5):
            raise ValueError(
                f"conn_bank must have shape [S,K,R,R] or [B,T,K,R,R], got {tuple(conn_bank.shape)}"
            )

        alpha = self.alpha().to(device=conn_bank.device, dtype=conn_bank.dtype)
        view_shape = [1] * (conn_bank.ndim - 3) + [self.num_candidates, 1, 1]
        mixed_adj = (conn_bank * alpha.view(*view_shape)).sum(dim=-3)
        mixed_adj = _symmetrize_last2(mixed_adj)
        return mixed_adj, alpha


# =========================================================
# 4. Sparse region mask
# =========================================================

class SparseRegionMask(nn.Module):
    def __init__(self, num_regions: int):
        super().__init__()
        self.num_regions = int(num_regions)
        self.mask_logits = nn.Parameter(torch.zeros((self.num_regions, self.num_regions), dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        sym_logits = 0.5 * (self.mask_logits + self.mask_logits.t())
        mask = torch.sigmoid(sym_logits)
        mask = 0.5 * (mask + mask.t())
        return mask


# =========================================================
# 5. Connectivity encoder
# =========================================================

class ConnectivityEncoder(nn.Module):
    def __init__(
        self,
        num_regions: int = 5,
        hidden_dim: int = 32,
        emb_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_regions = int(num_regions)
        triu_idx = torch.triu_indices(self.num_regions, self.num_regions, offset=0)
        self.register_buffer("triu_row", triu_idx[0], persistent=False)
        self.register_buffer("triu_col", triu_idx[1], persistent=False)

        upper_dim = self.num_regions * (self.num_regions + 1) // 2
        self.mlp = nn.Sequential(
            nn.Linear(upper_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        adj: [B, T, R, R] or [S, R, R]

        Returns
        -------
        emb: [B, T, 64] or [S, 64]
        """
        if adj.ndim not in (3, 4):
            raise ValueError(f"adj must have shape [S,R,R] or [B,T,R,R], got {tuple(adj.shape)}")
        if adj.size(-1) != self.num_regions or adj.size(-2) != self.num_regions:
            raise ValueError(
                f"Expected last dims {(self.num_regions, self.num_regions)}, got {tuple(adj.shape[-2:])}"
            )

        adj = _symmetrize_last2(adj)
        flat = adj[..., self.triu_row, self.triu_col]
        out = self.mlp(flat)
        return out


# =========================================================
# 6. Attention MIL pooling on padded bags
# =========================================================

class AttentionMILPool(nn.Module):
    """
    Gated-attention MIL pooling on padded subject bags.

    Inputs
    ------
    seg_emb  : [B, T, D]
    bag_mask : [B, T]  True means valid segment
    """
    def __init__(self, in_dim: int, attn_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, seg_emb: torch.Tensor, bag_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if seg_emb.ndim != 3:
            raise ValueError(f"seg_emb must have shape [B, T, D], got {tuple(seg_emb.shape)}")
        if bag_mask.ndim != 2:
            raise ValueError(f"bag_mask must have shape [B, T], got {tuple(bag_mask.shape)}")
        if seg_emb.shape[:2] != bag_mask.shape:
            raise ValueError(
                f"seg_emb first two dims {tuple(seg_emb.shape[:2])} must match bag_mask {tuple(bag_mask.shape)}"
            )

        score = self.w(torch.tanh(self.V(seg_emb)) * torch.sigmoid(self.U(seg_emb))).squeeze(-1)  # [B, T]
        score = score.masked_fill(~bag_mask, -1e9)

        attn = torch.softmax(score, dim=1)
        attn = attn * bag_mask.to(attn.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-8)

        bag_emb = torch.sum(attn.unsqueeze(-1) * seg_emb, dim=1)  # [B, D]
        return bag_emb, attn


# =========================================================
# 7. Gated residual fusion head
# =========================================================

class GatedFusionHead(nn.Module):
    def __init__(
        self,
        node_dim: int,
        conn_dim: int,
        num_classes: int,
        dropout: float = 0.2,
        gate_bias_init: float = -2.0,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.conn_dim = int(conn_dim)

        self.conn_proj = nn.Sequential(
            nn.Linear(self.conn_dim, self.node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.node_dim, self.node_dim),
        )

        self.gate_fc1 = nn.Linear(self.node_dim + self.conn_dim, self.node_dim)
        self.gate_fc2 = nn.Linear(self.node_dim, self.node_dim)
        nn.init.constant_(self.gate_fc2.bias, gate_bias_init)

        self.classifier = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.node_dim, num_classes),
        )

    def forward(self, z_node_subj: torch.Tensor, z_conn_subj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        proj_conn = self.conn_proj(z_conn_subj)                            # [B, D_node]
        gate_in = torch.cat([z_node_subj, z_conn_subj], dim=-1)           # [B, D_node + D_conn]
        gate = torch.sigmoid(self.gate_fc2(F.relu(self.gate_fc1(gate_in))))
        z_final = z_node_subj + gate * proj_conn
        logits = self.classifier(z_final)
        return logits, z_final, gate


# =========================================================
# Build the current node encoder without redesigning it
# =========================================================

def _build_existing_node_graph_encoder(
    *,
    encoder_type: str,
    num_node_features: int,
    num_nodes: Optional[int],
    graph_emb_dim: int,
    dropout: float,
    graph_pool: str,
    gnn_hidden_dim: int,
    sage_layers: int,
    gcn2_layers: int,
    gcn2_alpha: float,
    gcn2_theta: float,
    gcn2_shared_weights: bool,
    gcn2_use_edge_weight: bool,
    h2gcn_layers: int,
    node_hidden_dims: Sequence[int],
    edge_hidden_dims: Sequence[int],
    branch_emb_dim: int,
) -> nn.Module:
    encoder_type = encoder_type.lower()

    if encoder_type == "sage":
        return GraphSAGEEncoder(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=sage_layers,
            dropout=dropout,
            pool=graph_pool,
            jk_mode="last",
        )

    if encoder_type == "gcn2":
        return GCNIIEncoder(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=gcn2_layers,
            dropout=dropout,
            alpha=gcn2_alpha,
            theta=gcn2_theta,
            shared_weights=gcn2_shared_weights,
            pool=graph_pool,
            use_edge_weight=gcn2_use_edge_weight,
        )

    if encoder_type == "h2gcn":
        return H2GCNLikeEncoder(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=h2gcn_layers,
            dropout=dropout,
            pool=graph_pool,
        )

    if encoder_type == "gnn":
        return GNNEncoder(
            in_dim=num_node_features,
            hidden_dim=gnn_hidden_dim,
            emb_dim=graph_emb_dim,
            dropout=dropout,
        )

    if encoder_type == "linkx":
        if num_nodes is None:
            raise ValueError("num_nodes must be provided when encoder_type='linkx'")
        return RawNodeEdgeMLPEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            node_hidden_dims=node_hidden_dims,
            edge_hidden_dims=edge_hidden_dims,
            branch_emb_dim=branch_emb_dim,
            emb_dim=graph_emb_dim,
            dropout=dropout,
            use_upper_triangle=True,
            symmetrize_adj=True,
        )

    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}. Choose from ['gnn', 'linkx', 'sage', 'gcn2', 'h2gcn']."
    )


# =========================================================
# 8. Two-branch MIL classifier
# =========================================================

class SubjectMILClassifierV2(nn.Module):
    """
    Existing node-feature MIL branch
      + regional connectivity MIL helper branch
      + gated residual fusion.

    Expected batch_dict from collate_subject_bags_v2:
      x_bag     : [B, T, 19, F]   (currently provided for clarity / debugging)
      conn_bank : [B, T, K, 5, 5]
      bag_mask  : [B, T]
      labels    : [B]
      pyg_batch : flattened PyG batch used by the existing node graph encoder
      bag_sizes : [B]
    """
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "linkx",
        num_nodes: Optional[int] = 19,
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",
        gnn_hidden_dim: int = 64,
        sage_layers: int = 2,
        gcn2_layers: int = 8,
        gcn2_alpha: float = 0.1,
        gcn2_theta: float = 0.5,
        gcn2_shared_weights: bool = True,
        gcn2_use_edge_weight: bool = True,
        h2gcn_layers: int = 2,
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        attn_dim: int = 128,
        num_conn_candidates: int = 8,
        num_regions: int = 5,
        conn_emb_dim: int = 64,
        lambda_mask: float = 1e-3,
    ):
        super().__init__()

        self.num_conn_candidates = int(num_conn_candidates)
        self.num_regions = int(num_regions)
        self.conn_emb_dim = int(conn_emb_dim)
        self.node_emb_dim = int(graph_emb_dim)
        self.lambda_mask = float(lambda_mask)

        # ----- Branch A: preserve the current node-feature graph encoder -----
        self.node_graph_encoder = _build_existing_node_graph_encoder(
            encoder_type=encoder_type,
            num_node_features=num_node_features,
            num_nodes=num_nodes,
            graph_emb_dim=graph_emb_dim,
            dropout=dropout,
            graph_pool=graph_pool,
            gnn_hidden_dim=gnn_hidden_dim,
            sage_layers=sage_layers,
            gcn2_layers=gcn2_layers,
            gcn2_alpha=gcn2_alpha,
            gcn2_theta=gcn2_theta,
            gcn2_shared_weights=gcn2_shared_weights,
            gcn2_use_edge_weight=gcn2_use_edge_weight,
            h2gcn_layers=h2gcn_layers,
            node_hidden_dims=node_hidden_dims,
            edge_hidden_dims=edge_hidden_dims,
            branch_emb_dim=branch_emb_dim,
        )
        self.node_mil_pool = AttentionMILPool(in_dim=graph_emb_dim, attn_dim=attn_dim)

        # ----- Branch B: candidate bank -> mix -> sparse mask -> small MLP encoder -----
        self.graph_mixer = CandidateGraphMixer(num_candidates=num_conn_candidates)
        self.region_mask_layer = SparseRegionMask(num_regions=num_regions)
        self.conn_encoder = ConnectivityEncoder(
            num_regions=num_regions,
            hidden_dim=32,
            emb_dim=conn_emb_dim,
            dropout=dropout,
        )
        self.conn_mil_pool = AttentionMILPool(in_dim=conn_emb_dim, attn_dim=attn_dim)

        # ----- Fusion -----
        self.fusion_head = GatedFusionHead(
            node_dim=graph_emb_dim,
            conn_dim=conn_emb_dim,
            num_classes=num_classes,
            dropout=dropout,
            gate_bias_init=-2.0,
        )

    def forward(self, batch_dict: Mapping[str, Any]) -> Dict[str, Any]:
        required = ["pyg_batch", "bag_sizes", "conn_bank", "bag_mask"]
        missing = [k for k in required if k not in batch_dict]
        if len(missing) > 0:
            raise KeyError(f"SubjectMILClassifierV2 missing required batch keys: {missing}")

        # =====================================================
        # Branch A: node-feature MIL (existing segment encoder)
        # =====================================================
        node_seg_flat = self.node_graph_encoder(batch_dict["pyg_batch"])    # [sum_T, D_node]
        node_seg_padded, inferred_mask = _pack_flat_segments_to_padded(
            node_seg_flat,
            batch_dict["bag_sizes"],
        )

        bag_mask = batch_dict["bag_mask"]
        if bag_mask.shape != inferred_mask.shape:
            raise ValueError(
                f"bag_mask shape {tuple(bag_mask.shape)} does not match inferred shape {tuple(inferred_mask.shape)}"
            )
        if not torch.equal(bag_mask.bool(), inferred_mask.bool()):
            raise ValueError("bag_mask does not match bag_sizes / flattened segment count.")

        z_node_subj, node_attn = self.node_mil_pool(node_seg_padded, bag_mask)   # [B, D_node], [B, T]

        # =====================================================
        # Branch B: connectivity MIL helper branch
        # =====================================================
        conn_bank = batch_dict["conn_bank"]                                      # [B, T, K, R, R]
        if conn_bank.ndim != 5:
            raise ValueError(f"conn_bank must have shape [B, T, K, R, R], got {tuple(conn_bank.shape)}")

        mixed_adj, alpha = self.graph_mixer(conn_bank)                            # [B, T, R, R], [K]
        region_mask = self.region_mask_layer()                                    # [R, R]
        final_adj = mixed_adj * region_mask.view(1, 1, self.num_regions, self.num_regions)
        final_adj = _symmetrize_last2(final_adj)

        z_conn_seg = self.conn_encoder(final_adj)                                 # [B, T, 64]
        z_conn_seg = z_conn_seg * bag_mask.unsqueeze(-1).to(z_conn_seg.dtype)
        z_conn_subj, conn_attn = self.conn_mil_pool(z_conn_seg, bag_mask)         # [B, 64], [B, T]

        # =====================================================
        # Fusion + classification
        # =====================================================
        logits, z_final, gate = self.fusion_head(z_node_subj, z_conn_subj)

        reg_loss = self.lambda_mask * region_mask.abs().mean()

        node_attn_list = [node_attn[b, bag_mask[b]].detach() for b in range(node_attn.size(0))]
        conn_attn_list = [conn_attn[b, bag_mask[b]].detach() for b in range(conn_attn.size(0))]

        aux = {
            "alpha": alpha,
            "region_mask": region_mask,
            "node_attn": node_attn,
            "conn_attn": conn_attn,
            "gate": gate,
            "mixed_adj": mixed_adj,
            "final_adj": final_adj,
            "z_node_subj": z_node_subj,
            "z_conn_subj": z_conn_subj,
        }

        return {
            "logits": logits,
            "bag_emb": z_final,
            "graph_emb": node_seg_flat,      # keep existing downstream utilities working
            "graph_emb_node": node_seg_flat,
            "conn_seg_emb": z_conn_seg,
            "node_attn_list": node_attn_list,
            "conn_attn_list": conn_attn_list,
            "attn_list": node_attn_list,     # compatibility with existing evaluate(...)
            "reg_loss": reg_loss,
            "aux": aux,
        }


# =========================================================
# Small sanity-check example
# =========================================================
if __name__ == "__main__":
    B, T, N, F_dim = 2, 4, 19, 8
    K, R = 8, 5

    # fake candidate bank per segment
    fake_conn_bank = torch.randn(B, T, K, R, R)
    fake_conn_bank = 0.5 * (fake_conn_bank + fake_conn_bank.transpose(-1, -2))

    mixer = CandidateGraphMixer(num_candidates=K)
    sparse_mask = SparseRegionMask(num_regions=R)
    conn_encoder = ConnectivityEncoder(num_regions=R)
    pool = AttentionMILPool(in_dim=64, attn_dim=32)

    mixed, alpha = mixer(fake_conn_bank)
    masked = mixed * sparse_mask().view(1, 1, R, R)
    conn_emb = conn_encoder(masked)
    bag_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
    subj_emb, attn = pool(conn_emb, bag_mask)

    print("mixed:", tuple(mixed.shape))
    print("alpha:", tuple(alpha.shape), "sum=", float(alpha.sum().item()))
    print("masked:", tuple(masked.shape))
    print("conn_emb:", tuple(conn_emb.shape))
    print("subj_emb:", tuple(subj_emb.shape))
    print("attn:", tuple(attn.shape))
