# models_gnn.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GraphNorm,
    SAGEConv,
    SAGPooling,
    TopKPooling,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import dense_to_sparse, scatter, softmax, to_dense_batch




ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

# from mil_full_std import (
#     RawNodeEdgeMLPEncoder,
#     RawNodeMLPEncoder,
#     RawNodeAdjCNNEncoder,
#     RawNodeMultiBandCNNEncoder,
#     MultiBandCNNEncoder,
# )
from mil_utils import (
    GNNEncoder,
    GraphSAGEEncoder,
    GCNIIEncoder,
    H2GCNLikeEncoder,
)

# these two are referenced by SubjectMILClassifier in your old stack.
# import them only if they exist in your local mil_utils.py
try:
    from mil_utils import GNNEncoder_GAT, HybridGNNEncoder
except Exception:
    GNNEncoder_GAT = None
    HybridGNNEncoder = None


GraphBackbone = Literal["gcn", "sage", "gatv2", "edge_gated"]
PoolMode = Literal["mean", "max", "add"]
FusionMode = Literal["concat", "gated"]
StageFusionMode = Literal["mean", "concat", "gated"]
GraphBankFusionMode = Literal["static", "summary_gated"]
TopologyRule = Literal["union", "intersection", "vote"]
ReadoutType = Literal[
    "mean",
    "max",
    "add",
    "mean_max_concat",
    "mean_add_concat",
    "attention",
    "gated_attention",
]
NodePoolingType = Literal["none", "topk", "sagpool"]

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeGatedConv(MessagePassing):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 1,
        aggr: str = "add",
        dropout: float = 0.0,
    ) -> None:
        super().__init__(aggr=aggr)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.edge_dim = int(edge_dim)
        self.dropout = float(dropout)

        self.msg_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)

        gate_in_dim = 2 * in_dim + edge_dim
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.Sigmoid(),
        )

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, edge_index, edge_attr=None, edge_weight=None):
        if edge_attr is None and edge_weight is not None:
            edge_attr = edge_weight.view(-1, 1)

        if edge_attr is None:
            edge_attr = x.new_zeros((edge_index.size(1), self.edge_dim))
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.self_lin(x)
        return out

    def message(self, x_i, x_j, edge_attr):
        gate_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        gate = self.gate_mlp(gate_in)
        msg = self.msg_lin(x_j)
        msg = gate * msg
        msg = self.dropout_layer(msg)
        return msg
@dataclass(slots=True)
class GNNModelOutput:
    """
    Standard output container for graph-based EEG models.

    Attributes
    ----------
    logits:
        Classification logits of shape [batch_size, num_classes].
    embedding:
        Main sample-level embedding, usually the graph embedding or fused embedding.
    graph_embedding:
        Graph-branch embedding.
    node_embedding:
        Optional node-feature branch embedding for hybrid models.
    fusion_weights:
        Optional graph-bank fusion weights of shape [batch_size, num_candidates].
    graph_attention_weights:
        Optional graph-branch node attention weights of shape [num_nodes_total].
    node_attention_weights:
        Optional node-branch attention weights of shape [num_nodes_total].
    aux:
        Optional extra debugging information.
    """

    logits: Tensor
    embedding: Tensor
    graph_embedding: Tensor | None = None
    node_embedding: Tensor | None = None
    fusion_weights: Tensor | None = None
    graph_attention_weights: Tensor | None = None
    node_attention_weights: Tensor | None = None
    aux: dict[str, Any] | None = None


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    dropout: float,
    activation: type[nn.Module] = nn.ReLU,
    use_batchnorm: bool = False,
) -> tuple[nn.Sequential, int]:
    """
    Build a simple MLP.

    Returns
    -------
    tuple[nn.Sequential, int]
        The MLP module and its final output dimension.
    """
    if input_dim < 1:
        raise ValueError(f"input_dim must be >= 1, got {input_dim}.")
    if len(hidden_dims) == 0:
        raise ValueError("hidden_dims must contain at least one layer size.")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

    layers: list[nn.Module] = []
    prev = int(input_dim)
    for h in hidden_dims:
        h = int(h)
        if h < 1:
            raise ValueError(f"All hidden dims must be >= 1, got {hidden_dims}.")
        layers.append(nn.Linear(prev, h))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h

    return nn.Sequential(*layers), prev


def _ensure_batch(graph: Data | Batch) -> Batch:
    """Convert a single PyG Data object into a Batch if needed."""
    if isinstance(graph, Batch):
        return graph
    if isinstance(graph, Data):
        return Batch.from_data_list([graph])
    raise TypeError(f"Expected torch_geometric Data or Batch, got {type(graph)}.")


def _extract_edge_weight(pyg_batch: Batch) -> Tensor | None:
    """
    Extract 1D edge weights from a PyG batch if available.
    """
    edge_attr = getattr(pyg_batch, "edge_attr", None)
    if edge_attr is None:
        edge_attr = getattr(pyg_batch, "edge_weight", None)

    if edge_attr is None:
        return None

    if edge_attr.dim() > 1:
        if edge_attr.size(-1) == 1:
            edge_attr = edge_attr.squeeze(-1)
        else:
            edge_attr = edge_attr[:, 0]
    return edge_attr.float()


def _extract_edge_attr_2d(pyg_batch: Batch) -> Tensor | None:
    """
    Extract edge attributes shaped [E, 1] if available.
    """
    edge_attr = getattr(pyg_batch, "edge_attr", None)
    if edge_attr is None:
        edge_weight = getattr(pyg_batch, "edge_weight", None)
        if edge_weight is None:
            return None
        edge_attr = edge_weight.view(-1, 1)

    if edge_attr.dim() == 1:
        edge_attr = edge_attr.view(-1, 1)

    return edge_attr.float()


def _pool_nodes(x: Tensor, batch: Tensor, pool: str = "mean") -> Tensor:
    """
    Thin compatibility wrapper for simple non-learned readouts.

    Supported values
    ----------------
    - "mean"
    - "max"
    - "add"
    - "mean_max_concat"
    - "mean_add_concat"

    For learned readouts such as "attention" or "gated_attention",
    use GraphReadout directly.
    """
    if pool == "mean":
        return global_mean_pool(x, batch)
    if pool == "max":
        return global_max_pool(x, batch)
    if pool == "add":
        return global_add_pool(x, batch)
    if pool == "mean_max_concat":
        return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
    if pool == "mean_add_concat":
        return torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=-1)

    raise ValueError(
        f"Unsupported pool={pool!r} in compatibility wrapper. "
        "Use GraphReadout for attention-based readouts."
    )


def _infer_num_graphs(pyg_batch: Batch) -> int:
    """Infer batch size from pyg_batch.batch."""
    if not hasattr(pyg_batch, "batch"):
        return 1
    return int(pyg_batch.batch.max().item()) + 1 if pyg_batch.batch.numel() > 0 else 1


def _extract_graph_labels(pyg_batch: Batch, num_graphs: int) -> list[Tensor | None]:
    """
    Recover one label tensor per graph for rebuilding batches after adjacency fusion.
    """
    y = getattr(pyg_batch, "y", None)
    if y is None:
        return [None] * num_graphs

    y = y.view(-1)
    if y.numel() == num_graphs:
        return [y[i : i + 1].detach().clone() for i in range(num_graphs)]

    if y.numel() == 1 and num_graphs == 1:
        return [y.detach().clone()]

    if y.numel() > num_graphs:
        step = max(y.numel() // num_graphs, 1)
        out: list[Tensor] = []
        for i in range(num_graphs):
            out.append(y[i * step : i * step + 1].detach().clone())
        return out

    return [None] * num_graphs


def _extract_adj_bank_tensor(
    pyg_batch: Batch,
    *,
    adj_bank: Tensor | None,
    num_graphs: int,
    num_nodes: int,
) -> Tensor:
    """
    Recover adjacency bank as [B, K, N, N].

    Supported inputs
    ----------------
    - explicit adj_bank argument:
      - [K, N, N] for one graph
      - [B, K, N, N]
    - pyg_batch.adj_bank:
      - [K, N, N] for one graph
      - [B*K, N, N] after PyG batching
      - [B, K, N, N]
    """
    source = adj_bank if adj_bank is not None else getattr(pyg_batch, "adj_bank", None)
    if source is None:
        raise ValueError(
            "FusedGraphBankGNN requires adjacency-bank input. "
            "Pass adj_bank explicitly or attach adj_bank to each graph."
        )

    if not torch.is_tensor(source):
        source = torch.as_tensor(source, dtype=torch.float32)
    else:
        source = source.float()

    if source.ndim == 3:
        if num_graphs == 1:
            if source.shape[-2] != num_nodes or source.shape[-1] != num_nodes:
                raise ValueError(
                    f"Expected adj_bank last dims [{num_nodes}, {num_nodes}], got {tuple(source.shape)}."
                )
            return source.unsqueeze(0)

        if source.shape[-2] != num_nodes or source.shape[-1] != num_nodes:
            raise ValueError(
                f"Expected adj_bank last dims [{num_nodes}, {num_nodes}], got {tuple(source.shape)}."
            )
        if source.shape[0] % num_graphs != 0:
            raise ValueError(
                f"Cannot reshape adj_bank with shape {tuple(source.shape)} into "
                f"[{num_graphs}, K, {num_nodes}, {num_nodes}]."
            )
        num_candidates = source.shape[0] // num_graphs
        return source.view(num_graphs, num_candidates, num_nodes, num_nodes)

    if source.ndim == 4:
        if source.shape[0] != num_graphs:
            raise ValueError(
                f"Expected adj_bank batch dimension {num_graphs}, got {tuple(source.shape)}."
            )
        if source.shape[-2] != num_nodes or source.shape[-1] != num_nodes:
            raise ValueError(
                f"Expected adj_bank last dims [{num_nodes}, {num_nodes}], got {tuple(source.shape)}."
            )
        return source

    raise ValueError(
        f"adj_bank must have shape [K,N,N] or [B,K,N,N], got {tuple(source.shape)}."
    )


def _extract_topology_bank_tensor(
    pyg_batch: Batch,
    *,
    topology_bank: Tensor | None,
    num_graphs: int,
    num_nodes: int,
) -> Tensor | None:
    """
    Recover topology bank as [B, K, N, N] if available.
    """
    source = topology_bank if topology_bank is not None else getattr(pyg_batch, "topology_bank", None)
    if source is None:
        return None

    if not torch.is_tensor(source):
        source = torch.as_tensor(source, dtype=torch.float32)
    else:
        source = source.float()

    if source.ndim == 3:
        if num_graphs == 1:
            return source.unsqueeze(0)
        if source.shape[0] % num_graphs != 0:
            raise ValueError(
                f"Cannot reshape topology_bank with shape {tuple(source.shape)} into "
                f"[{num_graphs}, K, {num_nodes}, {num_nodes}]."
            )
        num_candidates = source.shape[0] // num_graphs
        return source.view(num_graphs, num_candidates, num_nodes, num_nodes)

    if source.ndim == 4:
        if source.shape[0] != num_graphs:
            raise ValueError(
                f"Expected topology_bank batch dimension {num_graphs}, got {tuple(source.shape)}."
            )
        return source

    raise ValueError(
        f"topology_bank must have shape [K,N,N] or [B,K,N,N], got {tuple(source.shape)}."
    )


def _dense_batch_to_pyg(
    dense_x: Tensor,
    dense_adj: Tensor,
    mask: Tensor,
    *,
    labels: list[Tensor | None] | None = None,
) -> Batch:
    """
    Convert dense node features and dense adjacency into a PyG Batch.

    Parameters
    ----------
    dense_x:
        Tensor of shape [B, N, F].
    dense_adj:
        Tensor of shape [B, N, N].
    mask:
        Tensor of shape [B, N], where True means a valid node.
    labels:
        Optional list of one label tensor per graph.

    Returns
    -------
    Batch
        A PyG Batch with x, edge_index, edge_weight, edge_attr, and y if provided.
    """
    if dense_x.ndim != 3:
        raise ValueError(f"dense_x must have shape [B,N,F], got {tuple(dense_x.shape)}.")
    if dense_adj.ndim != 3:
        raise ValueError(f"dense_adj must have shape [B,N,N], got {tuple(dense_adj.shape)}.")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape [B,N], got {tuple(mask.shape)}.")
    if dense_x.shape[0] != dense_adj.shape[0] or dense_x.shape[0] != mask.shape[0]:
        raise ValueError("dense_x, dense_adj, and mask must agree on batch dimension.")
    if dense_x.shape[1] != dense_adj.shape[1] or dense_adj.shape[1] != dense_adj.shape[2]:
        raise ValueError("dense_x and dense_adj shapes are inconsistent.")

    batch_list: list[Data] = []
    batch_size = dense_x.shape[0]
    labels = [None] * batch_size if labels is None else labels

    for i in range(batch_size):
        valid_idx = mask[i].nonzero(as_tuple=False).view(-1)
        x_i = dense_x[i, valid_idx]
        adj_i = dense_adj[i][valid_idx][:, valid_idx]

        edge_index, edge_weight = dense_to_sparse(adj_i)
        data = Data(
            x=x_i,
            edge_index=edge_index.long(),
        )
        data.edge_weight = edge_weight.float()
        data.edge_attr = edge_weight.view(-1, 1).float()
        data.adj = adj_i.float()

        y_i = labels[i]
        if y_i is not None:
            data.y = y_i.long()

        batch_list.append(data)

    return Batch.from_data_list(batch_list)


class GraphReadout(nn.Module):
    """
    Reusable graph readout block for graph classification.

    Supported readouts
    ------------------
    - "mean"
    - "max"
    - "add"
    - "mean_max_concat"
    - "mean_add_concat"
    - "attention"
    - "gated_attention"

    Parameters
    ----------
    input_dim:
        Dimension of node embeddings.
    readout_type:
        Readout strategy.
    output_dim:
        Optional output graph-embedding dimension. Useful when concat
        readouts double the raw dimension and you want to project back down.
    hidden_dim:
        Hidden dimension for attention/gated-attention scoring networks.
    dropout:
        Dropout used in the optional projection and attention MLPs.
    return_attention_weights:
        Whether this block should return attention weights by default.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        readout_type: ReadoutType = "mean",
        output_dim: int | None = None,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {input_dim}.")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")

        self.input_dim = int(input_dim)
        self.readout_type = str(readout_type).lower()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.default_return_attention_weights = bool(return_attention_weights)

        if self.readout_type not in {
            "mean",
            "max",
            "add",
            "mean_max_concat",
            "mean_add_concat",
            "attention",
            "gated_attention",
        }:
            raise ValueError(f"Unsupported readout_type={readout_type!r}.")

        if self.readout_type in {"mean", "max", "add", "attention", "gated_attention"}:
            raw_dim = self.input_dim
        else:
            raw_dim = 2 * self.input_dim

        self.raw_output_dim = raw_dim
        self.output_dim = int(output_dim) if output_dim is not None else raw_dim

        if self.readout_type == "attention":
            self.att_mlp = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim, 1),
            )
            self.gate_tanh = None
            self.gate_sigmoid = None
            self.att_score = None
        elif self.readout_type == "gated_attention":
            self.att_mlp = None
            self.gate_tanh = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.gate_sigmoid = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.Sigmoid(),
            )
            self.att_score = nn.Linear(self.hidden_dim, 1)
        else:
            self.att_mlp = None
            self.gate_tanh = None
            self.gate_sigmoid = None
            self.att_score = None

        if self.output_dim != self.raw_output_dim:
            self.proj = nn.Sequential(
                nn.Linear(self.raw_output_dim, self.output_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            )
        else:
            self.proj = nn.Identity()

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        *,
        return_attention_weights: bool | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        x:
            Node embeddings of shape [num_nodes_total, input_dim].
        batch:
            Graph assignment vector of shape [num_nodes_total].
        return_attention_weights:
            Override whether to return node attention weights.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            `(graph_embedding, attention_weights)`
            where `graph_embedding` has shape [num_graphs, output_dim]
            and `attention_weights` has shape [num_nodes_total] for attention
            readouts, otherwise None.
        """
        if x.ndim != 2:
            raise ValueError(f"x must have shape [num_nodes_total, input_dim], got {tuple(x.shape)}.")
        if batch.ndim != 1:
            raise ValueError(f"batch must have shape [num_nodes_total], got {tuple(batch.shape)}.")
        if x.shape[0] != batch.shape[0]:
            raise ValueError("x and batch must have the same first dimension.")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got node embeddings with shape {tuple(x.shape)}."
            )

        want_attn = (
            self.default_return_attention_weights
            if return_attention_weights is None
            else bool(return_attention_weights)
        )

        attention_weights: Tensor | None = None

        if self.readout_type == "mean":
            pooled = global_mean_pool(x, batch)

        elif self.readout_type == "max":
            pooled = global_max_pool(x, batch)

        elif self.readout_type == "add":
            pooled = global_add_pool(x, batch)

        elif self.readout_type == "mean_max_concat":
            pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)

        elif self.readout_type == "mean_add_concat":
            pooled = torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=-1)

        elif self.readout_type == "attention":
            assert self.att_mlp is not None
            scores = self.att_mlp(x).view(-1)
            alpha = softmax(scores, batch)
            pooled = scatter(alpha.unsqueeze(-1) * x, batch, dim=0, reduce="sum")
            attention_weights = alpha if want_attn else None

        elif self.readout_type == "gated_attention":
            assert self.gate_tanh is not None
            assert self.gate_sigmoid is not None
            assert self.att_score is not None
            h = self.gate_tanh(x)
            g = self.gate_sigmoid(x)
            scores = self.att_score(h * g).view(-1)
            alpha = softmax(scores, batch)
            pooled = scatter(alpha.unsqueeze(-1) * x, batch, dim=0, reduce="sum")
            attention_weights = alpha if want_attn else None

        else:
            raise RuntimeError(f"Unhandled readout_type={self.readout_type!r}.")

        pooled = self.proj(pooled)
        return pooled, attention_weights


class NodeFeatureEncoder(nn.Module):
    """
    Non-message-passing node-feature branch for hybrid graph models.

    It first encodes each node independently with an MLP, then applies a
    reusable graph readout.
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dims: Sequence[int] = (128, 64),
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = False,
        readout_type: ReadoutType = "mean",
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()

        self.node_mlp, last_dim = _make_mlp(
            input_dim=int(in_dim),
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )
        self.node_proj = nn.Linear(last_dim, int(emb_dim))
        self.readout = GraphReadout(
            input_dim=int(emb_dim),
            readout_type=readout_type,
            output_dim=int(emb_dim),
            hidden_dim=readout_hidden_dim,
            dropout=readout_dropout,
            return_attention_weights=return_attention_weights,
        )

    def forward(
        self,
        pyg_batch: Data | Batch,
        *,
        return_attention_weights: bool | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Encode node features into a graph-level embedding.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            `(graph_embedding, attention_weights)`.
        """
        batch = _ensure_batch(pyg_batch)
        x = self.node_mlp(batch.x)
        x = self.node_proj(x)
        return self.readout(x, batch.batch, return_attention_weights=return_attention_weights)


class GraphEncoderBlock(nn.Module):
    """
    Shallow practical graph encoder using GCN, GraphSAGE, or GATv2.

    This block cleanly separates:
    1. node encoding / message passing
    2. optional node pooling / graph coarsening
    3. final graph readout
    """

    def __init__(
        self,
        *,
        num_node_features: int,
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        backbone: GraphBackbone = "gcn",
        dropout: float = 0.2,
        gat_heads: int = 4,
        use_edge_weight: bool = True,
        use_batchnorm: bool = True,
        node_pooling_type: NodePoolingType = "none",
        node_pool_ratio: float = 0.8,
        readout_type: ReadoutType = "mean",
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if hidden_dim < 1 or graph_emb_dim < 1:
            raise ValueError("hidden_dim and graph_emb_dim must be >= 1.")
        if gat_heads < 1:
            raise ValueError(f"gat_heads must be >= 1, got {gat_heads}.")
        if not (0.0 < node_pool_ratio <= 1.0):
            raise ValueError(f"node_pool_ratio must be in (0, 1], got {node_pool_ratio}.")

        self.backbone = str(backbone).lower()
        self.dropout = float(dropout)
        self.use_edge_weight = bool(use_edge_weight)
        self.node_pooling_type = str(node_pooling_type).lower()

        if self.backbone not in {"gcn", "sage", "gatv2", "edge_gated"}:
            raise ValueError(
                f"Unsupported backbone={backbone!r}. Use 'gcn', 'sage', 'edge_gated', or 'gatv2'."
            )
        if self.node_pooling_type not in {"none", "topk", "sagpool"}:
            raise ValueError(
                f"Unsupported node_pooling_type={node_pooling_type!r}. "
                "Use 'none', 'topk', or 'sagpool'."
            )

        self.input_proj = nn.Linear(int(num_node_features), int(hidden_dim))

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if self.backbone == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif self.backbone == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
            elif backbone == "edge_gated":
                conv = EdgeGatedConv(
                    in_dim=hidden_dim,
                    out_dim=hidden_dim,
                    edge_dim=1,
                    aggr="add",
                    dropout=dropout,
                )
            else:
                conv = GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=gat_heads,
                    concat=False,
                    edge_dim=1,
                )
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim) if use_batchnorm else nn.Identity())

        if self.node_pooling_type == "topk":
            self.node_pool = TopKPooling(hidden_dim, ratio=node_pool_ratio)
        elif self.node_pooling_type == "sagpool":
            self.node_pool = SAGPooling(hidden_dim, ratio=node_pool_ratio)
        else:
            self.node_pool = None

        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.readout = GraphReadout(
            input_dim=graph_emb_dim,
            readout_type=readout_type,
            output_dim=graph_emb_dim,
            hidden_dim=readout_hidden_dim,
            dropout=readout_dropout,
            return_attention_weights=return_attention_weights,
        )

    def forward(
        self,
        pyg_batch: Data | Batch,
        *,
        return_attention_weights: bool | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Parameters
        ----------
        pyg_batch:
            PyG Data or Batch.
        return_attention_weights:
            Override whether to return node attention weights from the readout.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            `(graph_embedding, attention_weights)`.
        """
        batch = _ensure_batch(pyg_batch)

        x = self.input_proj(batch.x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        edge_weight = _extract_edge_weight(batch) if self.use_edge_weight else None
        edge_attr = _extract_edge_attr_2d(batch)
        edge_index = batch.edge_index
        graph_batch = batch.batch

        for conv, norm in zip(self.convs, self.norms):
            # if self.backbone == "gcn":
            #     x = conv(x, edge_index, edge_weight=edge_weight)
            # elif self.backbone == "sage":
            #     x = conv(x, edge_index)
            # else:
            #     x = conv(x, edge_index, edge_attr=edge_attr)
            if self.backbone == "edge_gated":
                x = conv(x, edge_index, edge_attr=edge_attr, edge_weight=edge_weight)
            elif self.backbone == "gatv2":
                x = conv(x, edge_index, edge_attr=edge_attr if self.use_edge_weight else None)
            elif self.backbone == "gcn":
                x = conv(x, edge_index, edge_weight=edge_weight if self.use_edge_weight else None)
            elif self.backbone == "sage":
                x = conv(x, edge_index)
            if isinstance(norm, GraphNorm):
                x = norm(x, graph_batch)
            else:
                x = norm(x)

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Optional node pooling / coarsening inside the encoder
        pooling_info: dict[str, Tensor] | None = None
        if self.node_pool is not None:
            pooled = self.node_pool(
                x,
                edge_index,
                edge_attr=edge_weight,
                batch=graph_batch,
            )
            x, edge_index, edge_weight, graph_batch, perm, score = pooled
            if edge_weight is not None:
                edge_attr = edge_weight.view(-1, 1)
            else:
                edge_attr = None
            pooling_info = {
                "perm": perm,
                "score": score,
            }

        x = self.node_proj(x)
        graph_emb, attn = self.readout(
            x,
            graph_batch,
            return_attention_weights=return_attention_weights,
        )

        if pooling_info is not None:
            # keep as attribute for possible debugging
            self._last_pooling_info = pooling_info
        else:
            self._last_pooling_info = None

        return graph_emb, attn


class GraphBankFusionBlock(nn.Module):
    """
    Constrained fusion block for a bank of candidate adjacency matrices.

    Fusion is constrained because it only mixes a provided bank of candidates,
    rather than learning a fully dense topology from scratch.
    """

    def __init__(
        self,
        *,
        num_candidates: int,
        num_node_features: int | None = None,
        fusion_mode: GraphBankFusionMode = "static",
        topology_rule: TopologyRule = "union",
        vote_threshold: float = 0.5,
        temperature: float = 1.0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")

        self.num_candidates = int(num_candidates)
        self.fusion_mode = str(fusion_mode).lower()
        self.topology_rule = str(topology_rule).lower()
        self.vote_threshold = float(vote_threshold)
        self.temperature = float(temperature)

        if self.fusion_mode not in {"static", "summary_gated"}:
            raise ValueError(
                f"Unsupported fusion_mode={fusion_mode!r}. Use 'static' or 'summary_gated'."
            )
        if self.topology_rule not in {"union", "intersection", "vote"}:
            raise ValueError(
                f"Unsupported topology_rule={topology_rule!r}. Use 'union', 'intersection', or 'vote'."
            )

        if self.fusion_mode == "static":
            self.fusion_mlp = None
            self.candidate_logits = nn.Parameter(torch.zeros(self.num_candidates))
        else:
            if num_node_features is None or num_node_features < 1:
                raise ValueError(
                    "num_node_features must be provided and >= 1 when fusion_mode='summary_gated'."
                )
            self.candidate_logits = None
            self.fusion_mlp = nn.Sequential(
                nn.Linear(int(num_node_features), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), self.num_candidates),
            )

    def forward(
        self,
        *,
        dense_node_features: Tensor,
        adj_bank: Tensor,
        topology_bank: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        dense_node_features:
            Tensor of shape [B, N, F].
        adj_bank:
            Candidate weighted adjacencies [B, K, N, N].
        topology_bank:
            Optional candidate topology masks [B, K, N, N].
            If omitted, derived from `adj_bank != 0`.
        mask:
            Optional node mask [B, N].

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            `(fused_adj, fused_topology, fusion_weights)`.
        """
        if dense_node_features.ndim != 3:
            raise ValueError(
                f"dense_node_features must have shape [B,N,F], got {tuple(dense_node_features.shape)}."
            )
        if adj_bank.ndim != 4:
            raise ValueError(f"adj_bank must have shape [B,K,N,N], got {tuple(adj_bank.shape)}.")
        if adj_bank.shape[1] != self.num_candidates:
            raise ValueError(
                f"Expected num_candidates={self.num_candidates}, got adj_bank shape {tuple(adj_bank.shape)}."
            )
        if adj_bank.shape[-1] != adj_bank.shape[-2]:
            raise ValueError("adj_bank must be square in the last two dimensions.")
        if dense_node_features.shape[0] != adj_bank.shape[0]:
            raise ValueError("dense_node_features and adj_bank must have the same batch size.")
        if dense_node_features.shape[1] != adj_bank.shape[-1]:
            raise ValueError("dense_node_features and adj_bank must have the same number of nodes.")

        if topology_bank is None:
            topology_bank = (adj_bank.abs() > 0).float()
        else:
            if topology_bank.shape != adj_bank.shape:
                raise ValueError(
                    f"topology_bank shape {tuple(topology_bank.shape)} must match adj_bank shape {tuple(adj_bank.shape)}."
                )
            topology_bank = (topology_bank > 0).float()

        if self.fusion_mode == "static":
            assert self.candidate_logits is not None
            logits = self.candidate_logits.unsqueeze(0).expand(adj_bank.shape[0], -1)
        else:
            assert self.fusion_mlp is not None
            if mask is None:
                summary = dense_node_features.mean(dim=1)
            else:
                masked = dense_node_features * mask.unsqueeze(-1).float()
                denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                summary = masked.sum(dim=1) / denom
            logits = self.fusion_mlp(summary)

        weights = torch.softmax(logits / self.temperature, dim=1)  # [B, K]

        fused_adj = torch.sum(adj_bank * weights[:, :, None, None], dim=1)

        if self.topology_rule == "union":
            fused_topology = topology_bank.max(dim=1).values
        elif self.topology_rule == "intersection":
            fused_topology = topology_bank.min(dim=1).values
        else:
            voted = torch.sum(topology_bank * weights[:, :, None, None], dim=1)
            fused_topology = (voted >= self.vote_threshold).float()

        fused_adj = 0.5 * (fused_adj + fused_adj.transpose(-1, -2))
        fused_topology = ((fused_topology + fused_topology.transpose(-1, -2)) > 0).float()

        eye = torch.eye(fused_adj.shape[-1], device=fused_adj.device, dtype=torch.bool)
        fused_adj = fused_adj.masked_fill(eye.unsqueeze(0), 0.0)
        fused_topology = fused_topology.masked_fill(eye.unsqueeze(0), 0.0)

        fused_adj = fused_adj * fused_topology
        return fused_adj, fused_topology, weights


class GraphFusionHead(nn.Module):
    """
    Fuse a node-feature embedding and a graph embedding.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        graph_dim: int,
        emb_dim: int,
        mode: FusionMode = "concat",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.mode = str(mode).lower()
        if self.mode not in {"concat", "gated"}:
            raise ValueError(f"Unsupported mode={mode!r}. Use 'concat' or 'gated'.")

        in_dim = int(node_dim) + int(graph_dim)

        if self.mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

        self.fuser = nn.Sequential(
            nn.Linear(in_dim, int(emb_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, node_embedding: Tensor, graph_embedding: Tensor) -> Tensor:
        """
        Fuse branch embeddings.
        """
        if node_embedding.ndim != 2 or graph_embedding.ndim != 2:
            raise ValueError("node_embedding and graph_embedding must both be [B, D].")
        if node_embedding.shape[0] != graph_embedding.shape[0]:
            raise ValueError("Both branch embeddings must have the same batch size.")

        x = torch.cat([node_embedding, graph_embedding], dim=1)
        if self.gate is not None:
            x = x * self.gate(x)
        return self.fuser(x)

class StageReadoutFusion(nn.Module):
    def __init__(self, emb_dim: int, mode: str = "concat", dropout: float = 0.0):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.mode = str(mode).lower()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.mode == "concat":
            self.proj = None   # create lazily once num_stages is known
        elif self.mode == "mean":
            self.proj = None
        elif self.mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1),
            )
        else:
            raise ValueError(f"Unsupported stage fusion mode={mode!r}")
        self.concat_proj: nn.Linear | None = None
        self.gate_mlp: nn.Sequential | None = None
    def forward(self, stage_embs: list[torch.Tensor]) -> torch.Tensor:
        if len(stage_embs) == 0:
            raise ValueError("stage_embs is empty")
        if len(stage_embs) == 1:
            return stage_embs[0]
        x = torch.stack(stage_embs, dim=1)   # [B, S, D]
        bsz, num_stages, emb_dim = x.shape

        if emb_dim != self.emb_dim:
            raise ValueError(
                f"Expected emb_dim={self.emb_dim}, got stage embeddings with dim={emb_dim}."
            )

        if self.mode == "mean":
            return x.mean(dim=1)

        if self.mode == "concat":
            flat = x.reshape(bsz, num_stages * emb_dim)   # [B, S*D]
            if self.concat_proj is None:
                self.concat_proj = nn.Linear(num_stages * emb_dim, emb_dim).to(flat.device)
            return self.dropout(self.concat_proj(flat))

        if self.mode == "gated":
            if self.gate_mlp is None:
                self.gate_mlp = nn.Sequential(
                    nn.Linear(emb_dim, emb_dim),
                    nn.ReLU(),
                    nn.Linear(emb_dim, 1),
                ).to(x.device)
            scores = self.gate_mlp(x).squeeze(-1)         # [B, S]
            alpha = torch.softmax(scores, dim=1)          # [B, S]
            fused = (alpha.unsqueeze(-1) * x).sum(dim=1)  # [B, D]
            return fused

        raise RuntimeError("Unreachable")


class ProgressiveGraphEncoderBlock(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int,
        graph_emb_dim: int,
        num_layers: int = 3,
        backbone: str = "gcn",
        dropout: float = 0.0,
        gat_heads: int = 4,
        use_edge_weight: bool = True,
        use_batchnorm: bool = True,
        node_pooling_type: str = "none",
        node_pool_ratio: float = 0.8,
        readout_type: str = "mean_max_concat",
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,
        return_attention_weights: bool = False,
        pool_every_layer: bool = True,
        stage_readout_fusion: str = "concat",
    ):
        
        super().__init__()


        self.num_node_features = int(num_node_features)
        self.hidden_dim = int(hidden_dim)
        self.graph_emb_dim = int(graph_emb_dim)
        self.num_layers = int(num_layers)
        self.backbone = str(backbone).lower()
        self.dropout = float(dropout)
        self.gat_heads = int(gat_heads)
        self.use_edge_weight = bool(use_edge_weight)
        self.use_batchnorm = bool(use_batchnorm)
        self.node_pooling_type = str(node_pooling_type).lower()
        self.node_pool_ratio = float(node_pool_ratio)
        self.readout_type = str(readout_type)
        self.readout_hidden_dim = int(readout_hidden_dim)
        self.readout_dropout = float(readout_dropout)
        self.default_return_attention_weights = bool(return_attention_weights)
        self.pool_every_layer = bool(pool_every_layer)
        self.stage_readout_fusion = str(stage_readout_fusion).lower()

        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if self.backbone not in {"gcn", "sage", "gatv2", "edge_gated"}:
            raise ValueError(
                f"Unsupported backbone={backbone!r}. "
                "Use one of {'gcn', 'sage', 'gatv2', 'edge_gated'}."
            )
        if self.node_pooling_type not in {"none", "topk", "sagpool"}:
            raise ValueError(
                f"Unsupported node_pooling_type={node_pooling_type!r}. "
                "Use one of {'none', 'topk', 'sagpool'}."
            )
        if not (0.0 < self.node_pool_ratio <= 1.0):
            raise ValueError(f"node_pool_ratio must be in (0, 1], got {node_pool_ratio}.")
        if self.stage_readout_fusion not in {"mean", "concat", "gated"}:
            raise ValueError(
                f"Unsupported stage_readout_fusion={stage_readout_fusion!r}. "
                "Use one of {'mean', 'concat', 'gated'}."
            )

        self.input_proj = nn.Linear(self.num_node_features, self.hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.readouts = nn.ModuleList()

        in_dim = self.hidden_dim
        for _ in range(self.num_layers):
            conv = self._build_conv(in_dim, self.hidden_dim)
            self.convs.append(conv)

            if self.use_batchnorm:
                self.norms.append(GraphNorm(self.hidden_dim))
            else:
                self.norms.append(nn.Identity())

            self.readouts.append(
                GraphReadout(
                    input_dim=self.hidden_dim,
                    readout_type=self.readout_type,
                    output_dim=self.graph_emb_dim,
                    hidden_dim=self.readout_hidden_dim,
                    dropout=self.readout_dropout,
                    return_attention_weights=self.default_return_attention_weights,
                )
            )

            if self.node_pooling_type == "none":
                self.pools.append(nn.Identity())
            elif self.node_pooling_type == "topk":
                self.pools.append(TopKPooling(self.hidden_dim, ratio=self.node_pool_ratio))
            elif self.node_pooling_type == "sagpool":
                self.pools.append(SAGPooling(self.hidden_dim, ratio=self.node_pool_ratio))
            else:
                raise RuntimeError(f"Unhandled node_pooling_type={self.node_pooling_type!r}")

            in_dim = self.hidden_dim

        self.stage_fusion = StageReadoutFusion(
            emb_dim=self.graph_emb_dim,
            mode=self.stage_readout_fusion,
            dropout=self.readout_dropout,
        )

    def _build_conv(self, in_dim: int, out_dim: int) -> nn.Module:
        if self.backbone == "gcn":
            return GCNConv(in_dim, out_dim)

        if self.backbone == "sage":
            return SAGEConv(in_dim, out_dim)

        if self.backbone == "gatv2":
            # keep output dim stable across layers
            heads = max(1, self.gat_heads)
            out_per_head = max(1, out_dim // heads)
            return GATv2Conv(
                in_dim,
                out_per_head,
                heads=heads,
                concat=True,
                edge_dim=1 if self.use_edge_weight else None,
            )

        if self.backbone == "edge_gated":
            return EdgeGatedConv(
                in_dim=in_dim,
                out_dim=out_dim,
                edge_dim=1,
                aggr="add",
                dropout=self.dropout,
            )

        raise RuntimeError(f"Unhandled backbone={self.backbone!r}.")

    def _apply_conv(
        self,
        conv: nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        edge_attr_2d: Tensor | None,
    ) -> Tensor:
        if self.backbone == "gcn":
            return conv(x, edge_index, edge_weight=edge_weight if self.use_edge_weight else None)

        if self.backbone == "sage":
            return conv(x, edge_index)

        if self.backbone == "gatv2":
            return conv(
                x,
                edge_index,
                edge_attr=edge_attr_2d if self.use_edge_weight else None,
            )

        if self.backbone == "edge_gated":
            return conv(
                x,
                edge_index,
                edge_attr=edge_attr_2d if self.use_edge_weight else None,
                edge_weight=edge_weight if self.use_edge_weight else None,
            )

        raise RuntimeError(f"Unhandled backbone={self.backbone!r}.")

    def _apply_pool(
        self,
        pool: nn.Module,
        x: Tensor,
        edge_index: Tensor,
        edge_attr_2d: Tensor | None,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        if self.node_pooling_type == "none":
            return x, edge_index, edge_attr_2d, batch

        if self.node_pooling_type in {"topk", "sagpool"}:
            out = pool(x, edge_index, edge_attr=edge_attr_2d, batch=batch)

            # PyG pooling returns:
            # x, edge_index, edge_attr, batch, perm, score
            x_new, edge_index_new, edge_attr_new, batch_new = out[:4]
            return x_new, edge_index_new, edge_attr_new, batch_new

        raise RuntimeError(f"Unhandled node_pooling_type={self.node_pooling_type!r}.")

    def forward(
        self,
        pyg_batch: Data | Batch,
        *,
        return_attention_weights: bool | None = None,
    ) -> tuple[Tensor, dict[str, list[Tensor | None]] | None]:
        """
        Returns
        -------
        tuple[Tensor, dict | None]
            - fused graph embedding: [B, graph_emb_dim]
            - optional attention payload:
                {
                  "stage_readout_attention": [...],
                }
              For non-attention readouts, entries may be None.
        """
        batch_obj = _ensure_batch(pyg_batch)

        x = batch_obj.x
        edge_index = batch_obj.edge_index
        batch = batch_obj.batch

        x = self.input_proj(x)

        edge_weight = _extract_edge_weight(batch_obj)
        edge_attr_2d = _extract_edge_attr_2d(batch_obj)

        want_attn = (
            self.default_return_attention_weights
            if return_attention_weights is None
            else bool(return_attention_weights)
        )

        stage_embs: list[Tensor] = []
        stage_attn: list[Tensor | None] = []

        current_x = x
        current_edge_index = edge_index
        current_batch = batch
        current_edge_weight = edge_weight
        current_edge_attr_2d = edge_attr_2d

        for layer_idx in range(self.num_layers):
            conv = self.convs[layer_idx]
            norm = self.norms[layer_idx]
            readout = self.readouts[layer_idx]
            pool = self.pools[layer_idx]

            current_x = self._apply_conv(
                conv,
                current_x,
                current_edge_index,
                current_edge_weight,
                current_edge_attr_2d,
            )

            if isinstance(norm, GraphNorm):
                current_x = norm(current_x, current_batch)
            else:
                current_x = norm(current_x)

            current_x = F.relu(current_x)
            current_x = F.dropout(current_x, p=self.dropout, training=self.training)

            graph_emb, attn = readout(
                current_x,
                current_batch,
                return_attention_weights=want_attn,
            )
            stage_embs.append(graph_emb)
            stage_attn.append(attn)

            do_pool = (
                self.node_pooling_type != "none"
                and layer_idx < self.num_layers - 1
                and self.pool_every_layer
            )
            if do_pool:
                current_x, current_edge_index, current_edge_attr_2d, current_batch = self._apply_pool(
                    pool,
                    current_x,
                    current_edge_index,
                    current_edge_attr_2d,
                    current_batch,
                )

                # keep scalar edge_weight aligned with pooled edge_attr
                if current_edge_attr_2d is None:
                    current_edge_weight = None
                elif current_edge_attr_2d.dim() == 2 and current_edge_attr_2d.size(-1) == 1:
                    current_edge_weight = current_edge_attr_2d.view(-1)
                else:
                    current_edge_weight = None

        fused_graph_emb = self.stage_fusion(stage_embs)

        if want_attn:
            return fused_graph_emb, {"stage_readout_attention": stage_attn}

        return fused_graph_emb, None


class SimpleFixedGraphGNN(nn.Module):
    """
    Simple shallow GNN baseline on a single provided graph topology.
    """

    def __init__(
        self,
        *,
        num_node_features: int,
        num_classes: int,
        backbone: GraphBackbone = "gcn",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
        use_edge_weight: bool = True,
        use_batchnorm: bool = True,
        node_pooling_type: NodePoolingType = "none",
        node_pool_ratio: float = 0.8,
        readout_type: ReadoutType = "mean",
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,
        return_attention_weights: bool = False,
    ) -> None:
        super().__init__()

        self.graph_encoder = GraphEncoderBlock(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=num_layers,
            backbone=backbone,
            dropout=dropout,
            gat_heads=gat_heads,
            use_edge_weight=use_edge_weight,
            use_batchnorm=use_batchnorm,
            node_pooling_type=node_pooling_type,
            node_pool_ratio=node_pool_ratio,
            readout_type=readout_type,
            readout_hidden_dim=readout_hidden_dim,
            readout_dropout=readout_dropout,
            return_attention_weights=return_attention_weights,
        )
        self.classifier = nn.Linear(int(graph_emb_dim), int(num_classes))
        self.default_return_attention_weights = bool(return_attention_weights)

    def forward(
        self,
        pyg_batch: Data | Batch,
        *,
        return_dict: bool = True,
        return_attention_weights: bool | None = None,
    ) -> GNNModelOutput | Tensor:
        """
        Forward pass.
        """
        want_attn = (
            self.default_return_attention_weights
            if return_attention_weights is None
            else bool(return_attention_weights)
        )
        graph_emb, attn = self.graph_encoder(
            pyg_batch,
            return_attention_weights=want_attn,
        )
        logits = self.classifier(graph_emb)

        if return_dict:
            return GNNModelOutput(
                logits=logits,
                embedding=graph_emb,
                graph_embedding=graph_emb,
                graph_attention_weights=attn,
                aux={"model_family": "simple_fixed_graph_gnn"},
            )
        return logits


class FusedGraphBankGNN(nn.Module):
    """
    GNN using a constrained bank of candidate adjacency matrices.
    """

    def __init__(
        self,
        *,
        num_node_features: int,
        num_classes: int,
        num_nodes: int,
        num_candidates: int,
        backbone: GraphBackbone = "gcn",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        gat_heads: int = 4,
        use_edge_weight: bool = True,
        use_batchnorm: bool = True,
        node_pooling_type: NodePoolingType = "none",
        node_pool_ratio: float = 0.8,
        readout_type: ReadoutType = "mean",
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,
        return_attention_weights: bool = False,
        fusion_mode: GraphBankFusionMode = "static",
        topology_rule: TopologyRule = "union",
        vote_threshold: float = 0.5,
        fusion_temperature: float = 1.0,
        fusion_hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.num_candidates = int(num_candidates)
        self.default_return_attention_weights = bool(return_attention_weights)

        self.bank_fusion = GraphBankFusionBlock(
            num_candidates=num_candidates,
            num_node_features=num_node_features,
            fusion_mode=fusion_mode,
            topology_rule=topology_rule,
            vote_threshold=vote_threshold,
            temperature=fusion_temperature,
            hidden_dim=fusion_hidden_dim,
        )

        self.graph_encoder = GraphEncoderBlock(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=num_layers,
            backbone=backbone,
            dropout=dropout,
            gat_heads=gat_heads,
            use_edge_weight=use_edge_weight,
            use_batchnorm=use_batchnorm,
            node_pooling_type=node_pooling_type,
            node_pool_ratio=node_pool_ratio,
            readout_type=readout_type,
            readout_hidden_dim=readout_hidden_dim,
            readout_dropout=readout_dropout,
            return_attention_weights=return_attention_weights,
        )
        self.classifier = nn.Linear(int(graph_emb_dim), int(num_classes))

    def forward(
        self,
        pyg_batch: Data | Batch,
        *,
        adj_bank: Tensor | None = None,
        topology_bank: Tensor | None = None,
        return_dict: bool = True,
        return_attention_weights: bool | None = None,
    ) -> GNNModelOutput | Tensor:
        """
        Forward pass.
        """
        batch = _ensure_batch(pyg_batch)

        dense_x, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=self.num_nodes)
        num_graphs = dense_x.shape[0]

        bank_adj = _extract_adj_bank_tensor(
            batch,
            adj_bank=adj_bank,
            num_graphs=num_graphs,
            num_nodes=self.num_nodes,
        )
        bank_topology = _extract_topology_bank_tensor(
            batch,
            topology_bank=topology_bank,
            num_graphs=num_graphs,
            num_nodes=self.num_nodes,
        )

        fused_adj, fused_topology, fusion_weights = self.bank_fusion(
            dense_node_features=dense_x,
            adj_bank=bank_adj,
            topology_bank=bank_topology,
            mask=mask,
        )

        labels = _extract_graph_labels(batch, num_graphs)
        fused_batch = _dense_batch_to_pyg(
            dense_x,
            fused_adj,
            mask,
            labels=labels,
        )

        want_attn = (
            self.default_return_attention_weights
            if return_attention_weights is None
            else bool(return_attention_weights)
        )
        graph_emb, attn = self.graph_encoder(
            fused_batch,
            return_attention_weights=want_attn,
        )
        logits = self.classifier(graph_emb)

        if return_dict:
            return GNNModelOutput(
                logits=logits,
                embedding=graph_emb,
                graph_embedding=graph_emb,
                fusion_weights=fusion_weights,
                graph_attention_weights=attn,
                aux={
                    "model_family": "fused_graph_bank_gnn",
                    "fused_topology": fused_topology,
                    "fused_adjacency": fused_adj,
                },
            )
        return logits


class DualBranchGraphModel(nn.Module):
    """
    Hybrid model with:
    - branch A: non-message-passing node-feature branch
    - branch B: true graph branch
    - fusion: dense fusion head
    """

    def __init__(
        self,
        *,
        num_node_features: int,
        num_classes: int,
        num_nodes: int | None = None,
        use_graph_bank: bool = False,
        num_candidates: int | None = None,
        node_hidden_dims: Sequence[int] = (128, 64),
        node_emb_dim: int = 128,
        node_dropout: float = 0.2,
        backbone: GraphBackbone = "gcn",
        hidden_dim: int = 64,
        graph_emb_dim: int = 128,
        num_layers: int = 2,
        graph_dropout: float = 0.2,
        gat_heads: int = 4,
        use_edge_weight: bool = True,
        use_batchnorm: bool = True,
        node_pooling_type: NodePoolingType = "none",
        node_pool_ratio: float = 0.8,
        node_readout_type: ReadoutType = "mean",
        graph_readout_type: ReadoutType = "mean",
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,
        return_attention_weights: bool = False,
        graph_bank_fusion_mode: GraphBankFusionMode = "static",
        topology_rule: TopologyRule = "union",
        vote_threshold: float = 0.5,
        fusion_temperature: float = 1.0,
        graph_bank_hidden_dim: int = 64,
        fusion_mode: FusionMode = "concat",
        fusion_emb_dim: int = 128,
        fusion_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.use_graph_bank = bool(use_graph_bank)
        self.num_nodes = None if num_nodes is None else int(num_nodes)
        self.default_return_attention_weights = bool(return_attention_weights)

        self.node_encoder = NodeFeatureEncoder(
            in_dim=num_node_features,
            hidden_dims=node_hidden_dims,
            emb_dim=node_emb_dim,
            dropout=node_dropout,
            use_batchnorm=use_batchnorm,
            readout_type=node_readout_type,
            readout_hidden_dim=readout_hidden_dim,
            readout_dropout=readout_dropout,
            return_attention_weights=return_attention_weights,
        )

        if self.use_graph_bank:
            if self.num_nodes is None or self.num_nodes < 1:
                raise ValueError("num_nodes must be provided when use_graph_bank=True.")
            if num_candidates is None or int(num_candidates) < 1:
                raise ValueError("num_candidates must be provided and >= 1 when use_graph_bank=True.")

            self.bank_fusion = GraphBankFusionBlock(
                num_candidates=int(num_candidates),
                num_node_features=num_node_features,
                fusion_mode=graph_bank_fusion_mode,
                topology_rule=topology_rule,
                vote_threshold=vote_threshold,
                temperature=fusion_temperature,
                hidden_dim=graph_bank_hidden_dim,
            )
        else:
            self.bank_fusion = None

        self.graph_encoder = GraphEncoderBlock(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            graph_emb_dim=graph_emb_dim,
            num_layers=num_layers,
            backbone=backbone,
            dropout=graph_dropout,
            gat_heads=gat_heads,
            use_edge_weight=use_edge_weight,
            use_batchnorm=use_batchnorm,
            node_pooling_type=node_pooling_type,
            node_pool_ratio=node_pool_ratio,
            readout_type=graph_readout_type,
            readout_hidden_dim=readout_hidden_dim,
            readout_dropout=readout_dropout,
            return_attention_weights=return_attention_weights,
        )

        self.fusion_head = GraphFusionHead(
            node_dim=node_emb_dim,
            graph_dim=graph_emb_dim,
            emb_dim=fusion_emb_dim,
            mode=fusion_mode,
            dropout=fusion_dropout,
        )
        self.classifier = nn.Linear(int(fusion_emb_dim), int(num_classes))

    def forward(
        self,
        pyg_batch: Data | Batch,
        *,
        adj_bank: Tensor | None = None,
        topology_bank: Tensor | None = None,
        return_dict: bool = True,
        return_attention_weights: bool | None = None,
    ) -> GNNModelOutput | Tensor:
        """
        Forward pass.
        """
        batch = _ensure_batch(pyg_batch)
        want_attn = (
            self.default_return_attention_weights
            if return_attention_weights is None
            else bool(return_attention_weights)
        )

        node_emb, node_attn = self.node_encoder(
            batch,
            return_attention_weights=want_attn,
        )
        fusion_weights: Tensor | None = None

        if self.use_graph_bank:
            assert self.bank_fusion is not None
            assert self.num_nodes is not None

            dense_x, mask = to_dense_batch(batch.x, batch.batch, max_num_nodes=self.num_nodes)
            num_graphs = dense_x.shape[0]

            bank_adj = _extract_adj_bank_tensor(
                batch,
                adj_bank=adj_bank,
                num_graphs=num_graphs,
                num_nodes=self.num_nodes,
            )
            bank_topology = _extract_topology_bank_tensor(
                batch,
                topology_bank=topology_bank,
                num_graphs=num_graphs,
                num_nodes=self.num_nodes,
            )

            fused_adj, fused_topology, fusion_weights = self.bank_fusion(
                dense_node_features=dense_x,
                adj_bank=bank_adj,
                topology_bank=bank_topology,
                mask=mask,
            )

            labels = _extract_graph_labels(batch, num_graphs)
            graph_batch = _dense_batch_to_pyg(
                dense_x,
                fused_adj,
                mask,
                labels=labels,
            )
            graph_emb, graph_attn = self.graph_encoder(
                graph_batch,
                return_attention_weights=want_attn,
            )
            aux = {
                "model_family": "dual_branch_graph_model",
                "fused_topology": fused_topology,
                "fused_adjacency": fused_adj,
            }
        else:
            graph_emb, graph_attn = self.graph_encoder(
                batch,
                return_attention_weights=want_attn,
            )
            aux = {
                "model_family": "dual_branch_graph_model",
            }

        fused_emb = self.fusion_head(node_emb, graph_emb)
        logits = self.classifier(fused_emb)

        if return_dict:
            return GNNModelOutput(
                logits=logits,
                embedding=fused_emb,
                graph_embedding=graph_emb,
                node_embedding=node_emb,
                fusion_weights=fusion_weights,
                graph_attention_weights=graph_attn,
                node_attention_weights=node_attn,
                aux=aux,
            )
        return logits



class LegacyEncoderGraphClassifier(nn.Module):
    """
    Reuse the old encoder stack from mil_full_std.py / mil_utils.py,
    but classify a single graph instance at a time.

    This is the right adapter for caueeg_main subject/macro levels.
    """

    def __init__(
        self,
        *,
        num_node_features: int,
        num_classes: int,
        num_nodes: int,
        encoder_type: str,
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",
        gnn_hidden_dim: int = 64,
        sage_layers: int = 2,
        gcn2_layers: int = 8,
        h2gcn_layers: int = 2,
        node_hidden_dims=(256, 128),
        edge_hidden_dims=(128, 64),
        branch_emb_dim: int = 64,
        cnn_channels=(16, 32),
        cnn_num_bands: int = 5,
        edge_mode: str = "topology_weighted",
    ):
        super().__init__()
        enc = encoder_type.lower()
        self.encoder_type = enc

        # if enc == "linkx":
        #     self.encoder = RawNodeEdgeMLPEncoder(
        #         num_nodes=num_nodes,
        #         num_node_features=num_node_features,
        #         node_hidden_dims=node_hidden_dims,
        #         edge_hidden_dims=edge_hidden_dims,
        #         branch_emb_dim=branch_emb_dim,
        #         emb_dim=graph_emb_dim,
        #         dropout=dropout,
        #         edge_mode=edge_mode,
        #         use_upper_triangle=True,
        #         symmetrize_adj=True,
        #     )

        # elif enc == "mlp_node":
        #     self.encoder = RawNodeMLPEncoder(
        #         num_nodes=num_nodes,
        #         num_node_features=num_node_features,
        #         node_hidden_dims=node_hidden_dims,
        #         proj_dim=branch_emb_dim,
        #         emb_dim=graph_emb_dim,
        #         dropout=dropout,
        #     )

        # elif enc == "linkx_cnn":
        #     self.encoder = RawNodeAdjCNNEncoder(
        #         num_nodes=num_nodes,
        #         num_node_features=num_node_features,
        #         node_hidden_dims=node_hidden_dims,
        #         branch_emb_dim=branch_emb_dim,
        #         emb_dim=graph_emb_dim,
        #         cnn_channels=cnn_channels,
        #         dropout=dropout,
        #         symmetrize_adj=True,
        #         zero_diagonal=False,
        #     )

        # elif enc == "linkx_cnn5":
        #     self.encoder = RawNodeMultiBandCNNEncoder(
        #         num_nodes=num_nodes,
        #         num_node_features=num_node_features,
        #         num_bands=cnn_num_bands,
        #         node_hidden_dims=node_hidden_dims,
        #         branch_emb_dim=branch_emb_dim,
        #         emb_dim=graph_emb_dim,
        #         dropout=dropout,
        #         symmetrize_adj=True,
        #         zero_diagonal=False,
        #     )

        # elif enc == "cnn5":
        #     self.encoder = MultiBandCNNEncoder(
        #         num_nodes=num_nodes,
        #         num_node_features=num_node_features,
        #         num_bands=cnn_num_bands,
        #         node_hidden_dims=node_hidden_dims,
        #         branch_emb_dim=branch_emb_dim,
        #         emb_dim=graph_emb_dim,
        #         dropout=dropout,
        #         symmetrize_adj=True,
        #         zero_diagonal=False,
        #     )

        if enc == "gnn":
            self.encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif enc == "sage":
            self.encoder = GraphSAGEEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=sage_layers,
                dropout=dropout,
                pool=graph_pool,
                jk_mode="last",
            )

        elif enc == "gcn2":
            self.encoder = GCNIIEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=gcn2_layers,
                dropout=dropout,
                alpha=0.1,
                theta=0.5,
                shared_weights=True,
                pool=graph_pool,
                use_edge_weight=True,
            )

        elif enc == "h2gcn":
            self.encoder = H2GCNLikeEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=h2gcn_layers,
                dropout=dropout,
                pool=graph_pool,
            )

        elif enc == "gat":
            if GNNEncoder_GAT is None:
                raise ImportError("GNNEncoder_GAT not available in mil_utils.py")
            self.encoder = GNNEncoder_GAT(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                num_layers=3,
                dropout=dropout,
                heads=4,
                edge_dim=1,
                pooling=graph_pool,
            )

        elif enc == "hybrid":
            if HybridGNNEncoder is None:
                raise ImportError("HybridGNNEncoder not available in mil_utils.py")
            self.encoder = HybridGNNEncoder(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                gat_layers=2,
                cheb_layers=2,
                dropout=dropout,
                heads=4,
                edge_dim=1,
                pooling=graph_pool,
            )

        else:
            raise ValueError(f"Unsupported legacy encoder_type={encoder_type!r}")

        self.classifier = nn.Linear(graph_emb_dim, num_classes)

    def forward(self, batch):
        pyg_batch = batch["pyg_batch"] if isinstance(batch, dict) else batch

        if self.encoder_type in {"linkx_cnn5", "cnn5"}:
            conn_stack = getattr(pyg_batch, "conn_stack", None)
            if conn_stack is None and isinstance(batch, dict):
                conn_stack = batch.get("conn_stack", None)
            if conn_stack is None:
                raise ValueError(
                    f"{self.encoder_type} requires multiband conn_stack [B, bands, N, N]"
                )
            emb = self.encoder(pyg_batch, conn_stack)
        else:
            emb = self.encoder(pyg_batch)

        logits = self.classifier(emb)
        return {
            "logits": logits,
            "embedding": emb,
        }
# ---------------------------------------------------------------------
# Example forward-pass usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import torch

    num_nodes = 19
    num_node_features = 14
    num_classes = 3
    num_candidates = 3

    # --------------------------------------------------
    # Build one synthetic fixed graph
    # --------------------------------------------------
    x = torch.randn(num_nodes, num_node_features)
    adj = torch.rand(num_nodes, num_nodes)
    adj = 0.5 * (adj + adj.t())
    adj.fill_diagonal_(0.0)
    edge_index, edge_weight = dense_to_sparse(adj)

    data = Data(
        x=x,
        edge_index=edge_index.long(),
        y=torch.tensor([0], dtype=torch.long),
    )
    data.edge_weight = edge_weight.float()
    data.edge_attr = edge_weight.view(-1, 1).float()
    data.adj = adj

    batch = Batch.from_data_list([data])

    # --------------------------------------------------
    # 1) Simple fixed-graph GNN
    # --------------------------------------------------
    fixed_model = SimpleFixedGraphGNN(
        num_node_features=num_node_features,
        num_classes=num_classes,
        backbone="gcn",
        hidden_dim=64,
        graph_emb_dim=128,
        num_layers=2,
        readout_type="mean_max_concat",
        readout_hidden_dim=64,
        readout_dropout=0.1,
        return_attention_weights=False,
    )
    fixed_out = fixed_model(batch)
    print("SimpleFixedGraphGNN logits:", fixed_out.logits.shape)
    print("SimpleFixedGraphGNN embedding:", fixed_out.embedding.shape)

    # --------------------------------------------------
    # 2) Fused graph-bank GNN
    # --------------------------------------------------
    adj_bank = torch.stack(
        [
            adj,
            (adj > 0.5).float() * adj,
            adj * 0.25,
        ],
        dim=0,
    )  # [K, N, N]

    topology_bank = (adj_bank > 0).float()

    fused_bank_model = FusedGraphBankGNN(
        num_node_features=num_node_features,
        num_classes=num_classes,
        num_nodes=num_nodes,
        num_candidates=num_candidates,
        backbone="sage",
        hidden_dim=64,
        graph_emb_dim=128,
        num_layers=2,
        fusion_mode="summary_gated",
        topology_rule="union",
        readout_type="gated_attention",
        readout_hidden_dim=64,
        return_attention_weights=True,
    )
    fused_out = fused_bank_model(
        batch,
        adj_bank=adj_bank,
        topology_bank=topology_bank,
    )
    print("FusedGraphBankGNN logits:", fused_out.logits.shape)
    print("FusedGraphBankGNN embedding:", fused_out.embedding.shape)
    print(
        "FusedGraphBankGNN fusion weights:",
        fused_out.fusion_weights.shape if fused_out.fusion_weights is not None else None,
    )
    print(
        "FusedGraphBankGNN attention weights:",
        fused_out.graph_attention_weights.shape if fused_out.graph_attention_weights is not None else None,
    )

    # --------------------------------------------------
    # 3) Dual-branch hybrid model on fixed graph
    # --------------------------------------------------
    hybrid_fixed_model = DualBranchGraphModel(
        num_node_features=num_node_features,
        num_classes=num_classes,
        use_graph_bank=False,
        backbone="gcn",
        hidden_dim=64,
        graph_emb_dim=128,
        node_emb_dim=128,
        fusion_emb_dim=128,
        node_readout_type="attention",
        graph_readout_type="mean_max_concat",
        readout_hidden_dim=64,
        return_attention_weights=True,
    )
    hybrid_fixed_out = hybrid_fixed_model(batch)
    print("DualBranchGraphModel (fixed) logits:", hybrid_fixed_out.logits.shape)
    print("DualBranchGraphModel (fixed) fused embedding:", hybrid_fixed_out.embedding.shape)
    print(
        "DualBranchGraphModel (fixed) node attention:",
        hybrid_fixed_out.node_attention_weights.shape if hybrid_fixed_out.node_attention_weights is not None else None,
    )

    # --------------------------------------------------
    # 4) Dual-branch hybrid model on fused graph bank
    # --------------------------------------------------
    hybrid_bank_model = DualBranchGraphModel(
        num_node_features=num_node_features,
        num_classes=num_classes,
        num_nodes=num_nodes,
        use_graph_bank=True,
        num_candidates=num_candidates,
        backbone="gatv2",
        hidden_dim=64,
        graph_emb_dim=128,
        node_emb_dim=128,
        fusion_emb_dim=128,
        graph_bank_fusion_mode="summary_gated",
        topology_rule="vote",
        vote_threshold=0.4,
        node_readout_type="gated_attention",
        graph_readout_type="gated_attention",
        readout_hidden_dim=64,
        return_attention_weights=True,
    )
    hybrid_bank_out = hybrid_bank_model(
        batch,
        adj_bank=adj_bank,
        topology_bank=topology_bank,
    )
    print("DualBranchGraphModel (bank) logits:", hybrid_bank_out.logits.shape)
    print("DualBranchGraphModel (bank) fused embedding:", hybrid_bank_out.embedding.shape)
    print(
        "DualBranchGraphModel (bank) graph embedding:",
        hybrid_bank_out.graph_embedding.shape if hybrid_bank_out.graph_embedding is not None else None,
    )
    print(
        "DualBranchGraphModel (bank) node embedding:",
        hybrid_bank_out.node_embedding.shape if hybrid_bank_out.node_embedding is not None else None,
    )
    print(
        "DualBranchGraphModel (bank) fusion weights:",
        hybrid_bank_out.fusion_weights.shape if hybrid_bank_out.fusion_weights is not None else None,
    )
    print(
        "DualBranchGraphModel (bank) graph attention:",
        hybrid_bank_out.graph_attention_weights.shape if hybrid_bank_out.graph_attention_weights is not None else None,
    )