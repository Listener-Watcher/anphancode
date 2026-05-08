from __future__ import annotations

import copy
import re
import runpy
from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse, to_dense_batch

import config
import mil_utils as mu
import mil_full_std as mf


# ---------------------------------------------------------------------
# Keep originals so we can delegate for old behavior / single-topology
# ---------------------------------------------------------------------
_ORIGINAL_BUILD_GRAPHS = mu.build_graphs_from_payload
_ORIGINAL_COLLATE = mu.collate_subject_bags
_ORIGINAL_SUBJECT_MIL = mf.SubjectMILClassifier


# ---------------------------------------------------------------------
# Region maps
# ---------------------------------------------------------------------
DEFAULT_REGION_TO_CHANNELS_MONO: Dict[str, List[str]] = {
    "frontal":   ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"],
    "central":   ["C3", "C4", "Cz"],
    "parietal":  ["P3", "P4", "Pz"],
    "temporal":  ["T3", "T4", "T5", "T6"],
    "occipital": ["O1", "O2"],
}


# ---------------------------------------------------------------------
# Runtime config
# Encoded inside existing --topology string so mil_full_std.py stays unchanged
#
# Supported examples:
#   --topology none
#   --topology fixed
#   --topology structural_local
#   --topology structural_hemisphere
#   --topology connectivity
#   --topology feature_induced|k=4|metric=cosine
#   --topology region_graph|region=region_clique
#   --topology multi:structural_local+feature_induced|fusion=mean|k=4|metric=cosine
#   --topology multi:structural_local+connectivity+feature_induced|fusion=learned_global_weights|k=4|metric=cosine
# ---------------------------------------------------------------------
VALID_TOPOLOGY_NAMES = {
    "structural_local",
    "structural_hemisphere",
    "connectivity",
    "feature_induced",
    "region_graph",
}
VALID_FUSIONS = {"mean", "learned_global_weights", "per_graph_attention"}


@dataclass
class TopologyRuntimeConfig:
    topology_mode: str = "single"  # none | single | multi
    topology_names: Tuple[str, ...] = ("connectivity",)
    topology_fusion_mode: str = "mean"
    feature_graph_k: int = 4
    feature_graph_metric: str = "cosine"
    feature_graph_rbf_gamma: Optional[float] = None
    region_graph_mode: str = "region_clique"
    debug_topology: bool = False


_CURRENT_CFG = TopologyRuntimeConfig()
_REGION_TO_CHANNELS = DEFAULT_REGION_TO_CHANNELS_MONO.copy()


# ---------------------------------------------------------------------
# Parsing existing args.topology string
# ---------------------------------------------------------------------
def _dedupe_keep_order(xs: Sequence[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        x = str(x).strip()
        if not x or x in seen:
            continue
        out.append(x)
        seen.add(x)
    return out


def parse_topology_spec(spec: Optional[str]) -> TopologyRuntimeConfig:
    if spec is None:
        return TopologyRuntimeConfig()

    s = str(spec).strip()
    if s == "":
        return TopologyRuntimeConfig()

    # Backward-friendly aliases
    alias = {
        "fixed": "structural_local",
        "local": "structural_local",
        "hemisphere": "structural_hemisphere",
        "feature": "feature_induced",
        "region": "region_graph",
    }
    s = alias.get(s, s)

    parts = [p.strip() for p in s.split("|") if p.strip()]
    head = parts[0]
    opts = {}

    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            opts[k.strip().lower()] = v.strip()

    if head == "none":
        cfg = TopologyRuntimeConfig(
            topology_mode="none",
            topology_names=(),
        )
    elif head.startswith("multi:"):
        names = head[len("multi:"):].split("+")
        names = _dedupe_keep_order(names)
        cfg = TopologyRuntimeConfig(
            topology_mode="multi",
            topology_names=tuple(names),
        )
    elif head.startswith("single:"):
        name = head[len("single:"):].strip()
        cfg = TopologyRuntimeConfig(
            topology_mode="single",
            topology_names=(name,),
        )
    else:
        cfg = TopologyRuntimeConfig(
            topology_mode="single",
            topology_names=(head,),
        )

    bad = [x for x in cfg.topology_names if x not in VALID_TOPOLOGY_NAMES]
    if bad:
        raise ValueError(
            f"Unknown topology names: {bad}. Valid: {sorted(VALID_TOPOLOGY_NAMES)}"
        )

    fusion = opts.get("fusion", cfg.topology_fusion_mode)
    if fusion not in VALID_FUSIONS:
        raise ValueError(f"Invalid fusion={fusion!r}. Valid: {sorted(VALID_FUSIONS)}")

    metric = opts.get("metric", cfg.feature_graph_metric).lower()
    if metric not in {"cosine", "rbf"}:
        raise ValueError("feature graph metric must be 'cosine' or 'rbf'")

    region_mode = opts.get("region", cfg.region_graph_mode).lower()
    if region_mode not in {"region_clique", "region_hypergraph_proxy"}:
        raise ValueError("region mode must be 'region_clique' or 'region_hypergraph_proxy'")

    gamma = opts.get("gamma", None)
    gamma = None if gamma is None else float(gamma)

    debug_flag = opts.get("debug", "0").lower() in {"1", "true", "yes", "y"}

    return TopologyRuntimeConfig(
        topology_mode=cfg.topology_mode,
        topology_names=cfg.topology_names,
        topology_fusion_mode=fusion,
        feature_graph_k=int(opts.get("k", cfg.feature_graph_k)),
        feature_graph_metric=metric,
        feature_graph_rbf_gamma=gamma,
        region_graph_mode=region_mode,
        debug_topology=debug_flag,
    )


def _set_runtime_cfg_from_filter_method(filter_method: Optional[str]) -> TopologyRuntimeConfig:
    global _CURRENT_CFG
    _CURRENT_CFG = parse_topology_spec(filter_method)
    return _CURRENT_CFG


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _clean_adj(
    adj: np.ndarray,
    *,
    symmetrize: bool = True,
    zero_diagonal: bool = True,
    eps: float = 1e-8,
) -> np.ndarray:
    A = np.asarray(adj, dtype=np.float32).copy()
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency must be square [N, N], got {A.shape}")

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    if symmetrize:
        A = 0.5 * (A + A.T)
    if zero_diagonal:
        np.fill_diagonal(A, 0.0)

    A[np.abs(A) < eps] = 0.0
    return A.astype(np.float32, copy=False)


def _debug_topology(name: str, A: np.ndarray, debug: bool = False):
    if not debug:
        return
    nnz = int((np.abs(A) > 1e-8).sum())
    density = float(nnz) / max(A.size, 1)
    print(
        f"[topology={name}] shape={tuple(A.shape)} "
        f"nnz={nnz} density={density:.4f} min={A.min():.4f} max={A.max():.4f}"
    )


def _infer_hemi_partner(ch_name: str) -> Optional[str]:
    tokens = str(ch_name).split("-")
    out = []
    for tok in tokens:
        m = re.match(r"^([A-Za-z]+)(\d+)$", tok)
        if m is None:
            return None
        base, num = m.group(1), int(m.group(2))
        out.append(f"{base}{num + 1}" if num % 2 == 1 else f"{base}{num - 1}")
    return "-".join(out)


def _infer_hemi_pairs(channel_names: Sequence[str]) -> List[Tuple[int, int]]:
    name_to_idx = {str(ch): i for i, ch in enumerate(channel_names)}
    pairs = []
    for ch, i in name_to_idx.items():
        partner = _infer_hemi_partner(ch)
        if partner is None or partner not in name_to_idx:
            continue
        j = name_to_idx[partner]
        if i < j:
            pairs.append((i, j))
    return pairs


def _build_structural_local(num_nodes: int, fixed_edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    if fixed_edges is None:
        raise ValueError("fixed_edges must be provided for structural_local")
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i, j in fixed_edges:
        i, j = int(i), int(j)
        if i == j:
            continue
        if not (0 <= i < num_nodes and 0 <= j < num_nodes):
            raise ValueError(f"Fixed edge {(i, j)} is out of range for num_nodes={num_nodes}")
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _build_structural_hemisphere(
    channel_names: Sequence[str],
    fixed_edges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    A = _build_structural_local(len(channel_names), fixed_edges)
    for i, j in _infer_hemi_pairs(channel_names):
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def _build_feature_induced(
    x: np.ndarray,
    *,
    k: int,
    metric: str,
    rbf_gamma: Optional[float],
) -> np.ndarray:
    X = np.asarray(x, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Node feature matrix must be [N, F], got {X.shape}")

    N = X.shape[0]
    k = min(max(int(k), 1), max(N - 1, 1))

    if metric == "cosine":
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        denom = np.clip(denom, 1e-8, None)
        Xn = X / denom
        sim = Xn @ Xn.T
        sim = 0.5 * (sim + 1.0)
    elif metric == "rbf":
        if rbf_gamma is None:
            rbf_gamma = 1.0 / max(X.shape[1], 1)
        diff = X[:, None, :] - X[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        sim = np.exp(-float(rbf_gamma) * dist2).astype(np.float32)
    else:
        raise ValueError(f"Unknown feature graph metric={metric!r}")

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        row = sim[i].copy()
        row[i] = -np.inf
        idx = np.argpartition(row, -k)[-k:]
        A[i, idx] = sim[i, idx]

    A = np.maximum(A, A.T)
    return A.astype(np.float32)


def _build_region_graph(
    channel_names: Sequence[str],
    region_to_channels: Mapping[str, Sequence[str]],
    mode: str,
) -> np.ndarray:
    region_names, hyperedge_members, node_to_region_mask = mu._build_region_members_from_channel_names(
        channel_names=channel_names,
        region_to_channels=region_to_channels,
        keep_empty_hyperedges=False,
    )

    num_nodes = len(channel_names)

    if mode == "region_clique":
        hedge_w = np.ones((len(hyperedge_members),), dtype=np.float32)
        return mu.hypergraph_to_clique_adj(
            num_nodes=num_nodes,
            hyperedge_members=hyperedge_members,
            hyperedge_weight=hedge_w,
            combine_mode="sum",
            remove_self_loops=True,
        )

    if mode == "region_hypergraph_proxy":
        H = node_to_region_mask.astype(np.float32)
        region_size = np.clip(H.sum(axis=0, keepdims=True), 1.0, None)
        Hn = H / np.sqrt(region_size)
        A = Hn @ Hn.T
        np.fill_diagonal(A, 0.0)
        return A.astype(np.float32)

    raise ValueError(f"Unknown region graph mode={mode!r}")


def _build_topology_dict(
    *,
    x: np.ndarray,
    source_adj: Optional[np.ndarray],
    fixed_edges: Optional[Sequence[Tuple[int, int]]],
    channel_names: Sequence[str],
    cfg: TopologyRuntimeConfig,
) -> "OrderedDict[str, np.ndarray]":
    out = OrderedDict()
    num_nodes = x.shape[0]

    for name in cfg.topology_names:
        if name == "structural_local":
            A = _build_structural_local(num_nodes, fixed_edges)
        elif name == "structural_hemisphere":
            A = _build_structural_hemisphere(channel_names, fixed_edges)
        elif name == "connectivity":
            if source_adj is None:
                raise ValueError("Topology 'connectivity' requested but source adjacency is None")
            A = np.asarray(source_adj, dtype=np.float32)
        elif name == "feature_induced":
            A = _build_feature_induced(
                x,
                k=cfg.feature_graph_k,
                metric=cfg.feature_graph_metric,
                rbf_gamma=cfg.feature_graph_rbf_gamma,
            )
        elif name == "region_graph":
            A = _build_region_graph(
                channel_names=channel_names,
                region_to_channels=_REGION_TO_CHANNELS,
                mode=cfg.region_graph_mode,
            )
        else:
            raise ValueError(f"Unsupported topology name={name!r}")

        A = _clean_adj(A, symmetrize=True, zero_diagonal=True)
        _debug_topology(name, A, debug=cfg.debug_topology)
        out[name] = A

    return out


# ---------------------------------------------------------------------
# Patched graph builder
# Uses current mil_full_std.py call shape:
#   build_graphs_from_payload(..., filter_method=args.topology, fixed_edges=..., channel_names=...)
# ---------------------------------------------------------------------
def patched_build_graphs_from_payload(
    payload,
    subject_ids,
    feature_families,
    connectivity_metric=None,
    edge_source="connectivity",
    zero_diagonal=True,
    symmetrize_adj=True,
    attach_dense_adj=True,
    filter_method=None,
    fixed_edges=None,
    channel_names=None,
    undirected=True,
    standardize_features=False,
    **kwargs,
):
    cfg = _set_runtime_cfg_from_filter_method(filter_method)

    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        label = int(subj["label"])

        feat_list = []
        ref_w = None
        ref_n = None

        for fam in feature_families:
            if fam not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] missing family {fam!r}")
            xfam = np.asarray(subj["features"][fam], dtype=np.float32)
            if xfam.ndim != 3:
                raise ValueError(f"Expected [W, N, F] for family {fam!r}, got {xfam.shape}")

            if ref_w is None:
                ref_w, ref_n = xfam.shape[:2]
            else:
                if xfam.shape[:2] != (ref_w, ref_n):
                    raise ValueError(
                        f"Feature family {fam!r} shape mismatch for {sid}: "
                        f"{xfam.shape[:2]} vs {(ref_w, ref_n)}"
                    )
            feat_list.append(xfam)

        node_x_all = np.concatenate(feat_list, axis=-1).astype(np.float32)
        num_windows = node_x_all.shape[0]
        num_nodes = node_x_all.shape[1]

        seg_ids = np.asarray(subj.get("segment_id", np.arange(num_windows)), dtype=np.int64)
        start_samples = np.asarray(subj.get("start_sample", np.full(num_windows, -1)), dtype=np.int64)

        if edge_source == "connectivity":
            if connectivity_metric is None:
                source_adj_all = None
            else:
                if "connectivity" not in subj or connectivity_metric not in subj["connectivity"]:
                    raise KeyError(f"Missing connectivity metric {connectivity_metric!r} for subject {sid!r}")
                source_adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)
        elif edge_source == "aligned_adj":
            source_adj_all = np.asarray(subj["aligned_adj"], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported edge_source={edge_source!r}")

        if channel_names is None:
            if "channel_names" in subj:
                channel_names_local = list(subj["channel_names"])
            else:
                channel_names_local = [f"ch_{i}" for i in range(num_nodes)]
        else:
            channel_names_local = list(channel_names)

        if len(channel_names_local) != num_nodes:
            raise ValueError(
                f"channel_names length mismatch: len(channel_names)={len(channel_names_local)} vs num_nodes={num_nodes}"
            )

        for w in range(num_windows):
            x = node_x_all[w]
            if standardize_features:
                x = mu._zscore_per_feature(x)

            source_adj = None if source_adj_all is None else np.asarray(source_adj_all[w], dtype=np.float32)

            if cfg.topology_mode == "none":
                primary_adj = np.eye(num_nodes, dtype=np.float32)
                topo_names: List[str] = []
                topo_stack = None
            else:
                topo_dict = _build_topology_dict(
                    x=x,
                    source_adj=source_adj,
                    fixed_edges=fixed_edges,
                    channel_names=channel_names_local,
                    cfg=cfg,
                )
                topo_names = list(topo_dict.keys())
                topo_stack = np.stack([topo_dict[name] for name in topo_names], axis=0).astype(np.float32)
                primary_adj = topo_stack[0]

            edge_index, edge_weight = dense_to_sparse(torch.tensor(primary_adj, dtype=torch.float32))

            g = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index.long(),
                y=torch.tensor([label], dtype=torch.long),
            )
            g.edge_weight = edge_weight.float()
            g.edge_attr = edge_weight.view(-1, 1).float()

            if attach_dense_adj:
                g.adj = torch.tensor(primary_adj, dtype=torch.float32)

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            # new multi-topology fields
            g.topology_mode = cfg.topology_mode
            g.topology_fusion_mode = cfg.topology_fusion_mode
            g.topology_names = topo_names
            g.topology_adjs = None if topo_stack is None else torch.tensor(topo_stack, dtype=torch.float32)

            graphs.append(g)

    return graphs


# ---------------------------------------------------------------------
# Patched collate
# ---------------------------------------------------------------------
def patched_collate_subject_bags(batch: List[dict]) -> Dict:
    out = _ORIGINAL_COLLATE(batch)

    all_graphs = out["pyg_batch"].to_data_list()
    all_topology_adjs = []
    topology_names_ref = None
    topology_mode_ref = None

    for g in all_graphs:
        if hasattr(g, "topology_adjs") and g.topology_adjs is not None:
            topo = g.topology_adjs
            if not torch.is_tensor(topo):
                topo = torch.tensor(topo, dtype=torch.float32)
            all_topology_adjs.append(topo.float().cpu())

            cur_names = list(getattr(g, "topology_names", []))
            cur_mode = str(getattr(g, "topology_mode", "single"))

            if topology_names_ref is None:
                topology_names_ref = cur_names
                topology_mode_ref = cur_mode
            else:
                if cur_names != topology_names_ref:
                    raise ValueError(f"Inconsistent topology order in batch: {cur_names} vs {topology_names_ref}")
                if cur_mode != topology_mode_ref:
                    raise ValueError(f"Inconsistent topology mode in batch: {cur_mode} vs {topology_mode_ref}")

    if len(all_topology_adjs) == len(all_graphs) and len(all_topology_adjs) > 0:
        out["topology_adjs"] = torch.stack(all_topology_adjs, dim=0)  # [B, K, N, N]
        out["topology_names"] = list(topology_names_ref or [])
        out["topology_mode"] = str(topology_mode_ref or "single")

    return out


# ---------------------------------------------------------------------
# Multi-topology encoder
# ---------------------------------------------------------------------
def _make_branch_encoder(
    *,
    encoder_type: str,
    num_node_features: int,
    num_nodes: int,
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
    edge_mode: str,
):
    enc = encoder_type.lower()

    if enc == "gnn":
        return mu.GNNEncoder(
            in_dim=num_node_features,
            hidden_dim=gnn_hidden_dim,
            emb_dim=graph_emb_dim,
            dropout=dropout,
        )

    if enc == "linkx":
        return mu.RawNodeEdgeMLPEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            node_hidden_dims=node_hidden_dims,
            edge_hidden_dims=edge_hidden_dims,
            branch_emb_dim=branch_emb_dim,
            emb_dim=graph_emb_dim,
            dropout=dropout,
            edge_mode=edge_mode,
            use_upper_triangle=True,
            symmetrize_adj=True,
        )

    if enc == "mlp_node":
        return mu.RawNodeMLPEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            node_hidden_dims=node_hidden_dims,
            proj_dim=branch_emb_dim,
            emb_dim=graph_emb_dim,
            dropout=dropout,
        )

    if enc == "sage":
        if not hasattr(mu, "GraphSAGEEncoder"):
            raise ValueError("GraphSAGEEncoder not found in mil_utils")
        return mu.GraphSAGEEncoder(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=sage_layers,
            dropout=dropout,
            pool=graph_pool,
            jk_mode="last",
        )

    if enc == "gcn2":
        if not hasattr(mu, "GCNIIEncoder"):
            raise ValueError("GCNIIEncoder not found in mil_utils")
        return mu.GCNIIEncoder(
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

    if enc == "h2gcn":
        if not hasattr(mu, "H2GCNLikeEncoder"):
            raise ValueError("H2GCNLikeEncoder not found in mil_utils")
        return mu.H2GCNLikeEncoder(
            num_node_features=num_node_features,
            hidden_dim=gnn_hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=h2gcn_layers,
            dropout=dropout,
            pool=graph_pool,
        )

    raise ValueError(f"Unsupported encoder_type={encoder_type!r}")


# class TopologyFusion(nn.Module):
#     def __init__(self, emb_dim: int, num_topologies: int, fusion_mode: str):
#         super().__init__()
#         self.emb_dim = int(emb_dim)
#         self.num_topologies = int(num_topologies)
#         self.fusion_mode = str(fusion_mode).lower()

#         if self.fusion_mode == "learned_global_weights":
#             self.logits = nn.Parameter(torch.zeros(self.num_topologies))
#         elif self.fusion_mode == "per_graph_attention":
#             hidden = max(self.emb_dim // 2, 16)
#             self.attn = nn.Sequential(
#                 nn.Linear(self.emb_dim, hidden),
#                 nn.Tanh(),
#                 nn.Linear(hidden, 1),
#             )
#         elif self.fusion_mode != "mean":
#             raise ValueError(f"Unsupported fusion_mode={fusion_mode!r}")

#     def forward(self, branch_embs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
#         # branch_embs: [B, K, D]
#         B, K, D = branch_embs.shape
#         if self.fusion_mode == "mean":
#             w = torch.full((B, K), 1.0 / K, device=branch_embs.device, dtype=branch_embs.dtype)
#             z = branch_embs.mean(dim=1)
#         elif self.fusion_mode == "learned_global_weights":
#             wg = torch.softmax(self.logits, dim=0)  # [K]
#             z = torch.sum(branch_embs * wg.view(1, K, 1), dim=1)
#             w = wg.view(1, K).expand(B, K)
#         else:
#             scores = self.attn(branch_embs).squeeze(-1)  # [B, K]
#             w = torch.softmax(scores, dim=1)
#             z = torch.sum(branch_embs * w.unsqueeze(-1), dim=1)

#         return z, {
#             "fusion_mode": self.fusion_mode,
#             "fusion_weights": w,
#         }


class MultiTopologyGraphEncoder(nn.Module):
    def __init__(
        self,
        *,
        topology_names: Sequence[str],
        fusion_mode: str,
        encoder_type: str,
        num_node_features: int,
        num_nodes: int,
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
        edge_mode: str,
    ):
        super().__init__()
        self.topology_names = list(topology_names)
        self.num_nodes = int(num_nodes)

        self.branches = nn.ModuleDict({
            topo: _make_branch_encoder(
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
                edge_mode=edge_mode,
            )
            for topo in self.topology_names
        })
        self.fusion = TopologyFusion(
            emb_dim=graph_emb_dim,
            num_topologies=len(self.topology_names),
            fusion_mode=fusion_mode,
        )

    def _dense_x(self, pyg_batch: Batch) -> torch.Tensor:
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )
        if dense_x.size(1) != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")
        return dense_x

    def _batch_from_dense(self, dense_x: torch.Tensor, dense_adj: torch.Tensor) -> Batch:
        data_list = []
        B = dense_x.shape[0]
        for b in range(B):
            edge_index, edge_weight = dense_to_sparse(dense_adj[b])
            g = Data(
                x=dense_x[b],
                edge_index=edge_index.long(),
            )
            g.edge_weight = edge_weight.float()
            g.edge_attr = edge_weight.view(-1, 1).float()
            g.adj = dense_adj[b]
            data_list.append(g)
        return Batch.from_data_list(data_list).to(dense_x.device)

    def forward(self, batch_dict: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if "topology_adjs" not in batch_dict:
            raise KeyError("batch_dict is missing 'topology_adjs' for multi-topology encoding")

        dense_x = self._dense_x(batch_dict["pyg_batch"])           # [B, N, F]
        topo_adjs = batch_dict["topology_adjs"].to(dense_x.device) # [B, K, N, N]
        topo_names = list(batch_dict.get("topology_names", []))

        if topo_names != self.topology_names:
            raise ValueError(f"Topology name order mismatch: batch={topo_names}, model={self.topology_names}")

        branch_embs = []
        for k, topo_name in enumerate(self.topology_names):
            pyg_k = self._batch_from_dense(dense_x, topo_adjs[:, k])
            emb_k = self.branches[topo_name](pyg_k)  # [B, D]
            branch_embs.append(emb_k)

        branch_embs = torch.stack(branch_embs, dim=1)  # [B, K, D]
        fused, dbg = self.fusion(branch_embs)
        dbg["active_topologies"] = self.topology_names
        return fused, dbg


# ---------------------------------------------------------------------
# Patched SubjectMILClassifier
# Delegates to old classifier for none/single, only uses new path for multi
# ---------------------------------------------------------------------
class PatchedSubjectMILClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",
        num_nodes: Optional[int] = None,
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
        mil_pool_type: str = "gated",
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
    ):
        super().__init__()
        self.cfg = copy.deepcopy(_CURRENT_CFG)

        # Old path: keep exactly old behavior for none/single
        if self.cfg.topology_mode != "multi":
            self.impl = _ORIGINAL_SUBJECT_MIL(
                num_node_features=num_node_features,
                num_classes=num_classes,
                encoder_type=encoder_type,
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
                mil_pool_type=mil_pool_type,
                edge_mode=edge_mode,
                attn_dim=attn_dim,
            )
            self.is_multi = False
            return

        self.is_multi = True
        self.graph_encoder = MultiTopologyGraphEncoder(
            topology_names=self.cfg.topology_names,
            fusion_mode=self.cfg.topology_fusion_mode,
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
            edge_mode=edge_mode,
        )

        if mil_pool_type.lower() == "mean":
            self.mil_pool = mu.MeanMILPool()
        elif mil_pool_type.lower() == "gated":
            self.mil_pool = mu.GatedAttentionMIL(in_dim=graph_emb_dim, attn_dim=attn_dim)
        else:
            raise ValueError(f"Unknown mil_pool_type={mil_pool_type!r}")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_multi:
            return self.impl(batch_dict)

        graph_emb, topo_dbg = self.graph_encoder(batch_dict)
        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        logits = self.classifier(bag_emb)

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "graph_emb": graph_emb,
            "attn_list": attn_list,
            "topology_debug": topo_dbg,
        }


# ---------------------------------------------------------------------
# Install patch before mil_full_std.py is imported
# ---------------------------------------------------------------------
def install_topology_patch(
    *,
    region_to_channels: Optional[Mapping[str, Sequence[str]]] = None,
):
    global _REGION_TO_CHANNELS
    if region_to_channels is not None:
        _REGION_TO_CHANNELS = {str(k): list(v) for k, v in region_to_channels.items()}

    mu.build_graphs_from_payload = patched_build_graphs_from_payload
    mu.collate_subject_bags = patched_collate_subject_bags
    mu.SubjectMILClassifier = PatchedSubjectMILClassifier


# ---------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------
def main():
    install_topology_patch(region_to_channels=DEFAULT_REGION_TO_CHANNELS_MONO)
    runpy.run_module("mil_full_std", run_name="__main__")


if __name__ == "__main__":
    main()



    