from __future__ import annotations
from typing import Iterable, Sequence, Optional, Union

from mil_utils import *

from torch_geometric.transforms import GCNNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GATv2Conv, NNConv, GlobalAttention

gcn_norm_transform = GCNNorm(add_self_loops=True)

from torch_geometric.nn.conv.gcn_conv import gcn_norm

def select_label_aligned_greedy_graphs_from_manifest(
    graphs,
    manifest_df: pd.DataFrame,
    *,
    k: int = 10,
    seed: int = 42,
    cluster_col: str = "global_cluster_id",
    label_col: Optional[str] = None,
    only_clean: bool = False,
    clean_col: str = "keep_clean",
    distance_col: str = "global_cluster_distance",
    debug_dir: Optional[str] = None,
):
    """
    Basic label-aligned greedy cluster selection.

    For each subject:
      1) Find clusters containing this subject's segments.
      2) Rank clusters by:
            highest P(subject_label | cluster)
            lowest entropy_norm
            highest number of this subject's segments in that cluster
      3) Select segments from ranked clusters until k segments are collected.

    This is label-aware and should be used for TRAINING only.
    Validation/test should still use all segments.
    """
    from collections import defaultdict

    rng = np.random.default_rng(seed)

    if len(graphs) == 0:
        raise RuntimeError("graphs is empty.")

    if cluster_col not in manifest_df.columns:
        raise KeyError(f"manifest_df missing cluster column: {cluster_col}")

    df = manifest_df.copy()

    # -------------------------
    # normalize key columns
    # -------------------------
    df["subject_id"] = df["subject_id"].astype(str)
    df["segment_id"] = df["segment_id"].astype(int)
    df[cluster_col] = df[cluster_col].astype(int)

    # -------------------------
    # optional clean filtering
    # -------------------------
    if only_clean:
        if clean_col not in df.columns:
            raise KeyError(f"only_clean=True but {clean_col!r} is missing.")
        if df[clean_col].dtype != bool:
            df[clean_col] = (
                df[clean_col]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
            )
        df = df[df[clean_col]].copy()

    # -------------------------
    # graph lookup
    # -------------------------
    graph_lookup = {
        (str(g.subject_id), int(g.segment_id)): g
        for g in graphs
    }

    graph_label_lookup = {
        (str(g.subject_id), int(g.segment_id)): int(g.y.view(-1)[0].item())
        for g in graphs
    }

    df["_key"] = list(zip(df["subject_id"], df["segment_id"]))

    # Keep only manifest rows that exist in current train_graphs.
    df = df[df["_key"].isin(graph_lookup)].copy()

    if len(df) == 0:
        raise RuntimeError("No manifest rows match the provided graphs.")

    # -------------------------
    # get label column
    # -------------------------
    if label_col is None:
        for cand in ["label", "class_label", "y", "true_label", "subject_label"]:
            if cand in df.columns:
                label_col = cand
                break

    if label_col is not None and label_col in df.columns:
        df["_label"] = df[label_col].astype(int)
    else:
        # safest fallback: use labels from graph objects
        df["_label"] = [graph_label_lookup[key] for key in df["_key"]]

    # -------------------------
    # compute cluster class distribution
    # -------------------------
    all_labels = sorted(df["_label"].unique().tolist())

    counts = pd.crosstab(
        df[cluster_col],
        df["_label"],
    ).sort_index()

    counts = counts.reindex(columns=all_labels, fill_value=0)
    print("counts", counts)
    probs = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    print("probs", probs)
    probs_np = probs.to_numpy(dtype=np.float64)
    probs_safe = np.clip(probs_np, 1e-12, 1.0)

    entropy = -(probs_safe * np.log2(probs_safe)).sum(axis=1)
    max_entropy = np.log2(len(all_labels)) if len(all_labels) > 1 else 1.0
    entropy_norm = entropy / max_entropy
    print("entropy", entropy, "entropy_norm", entropy_norm)
    cluster_stats = counts.copy()
    cluster_stats.columns = [f"count_class_{c}" for c in cluster_stats.columns]

    for c in all_labels:
        cluster_stats[f"p_class_{c}"] = probs[c].values

    cluster_stats["num_segments"] = counts.sum(axis=1).values
    cluster_stats["entropy"] = entropy
    cluster_stats["entropy_norm"] = entropy_norm
    cluster_stats = cluster_stats.reset_index()

    cluster_info = {
        int(row[cluster_col]): row.to_dict()
        for _, row in cluster_stats.iterrows()
    }

    # -------------------------
    # greedy selection per subject
    # -------------------------
    selected_keys = []
    debug_rows = []

    for sid, sdf in df.groupby("subject_id", sort=True):
        sdf = sdf.copy()

        # Subject label should be consistent.
        subject_labels = sorted(sdf["_label"].unique().tolist())
        if len(subject_labels) != 1:
            raise ValueError(
                f"Subject {sid} has multiple labels in manifest/graphs: {subject_labels}"
            )

        y = int(subject_labels[0])

        subject_cluster_count = (
            sdf.groupby(cluster_col)
            .size()
            .to_dict()
        )

        subject_clusters = sorted(sdf[cluster_col].unique().tolist())

        def cluster_rank_key(cid):
            info = cluster_info[int(cid)]

            p_y = float(info.get(f"p_class_{y}", 0.0))
            h = float(info["entropy_norm"])
            n_subj = int(subject_cluster_count.get(cid, 0))

            # Sort ascending, so use negative for descending values.
            return (
                -p_y,       # higher P(y | cluster) first
                h,          # lower entropy first
                -n_subj,    # more subject segments first
                int(cid),    # deterministic final tie-break
            )

        ranked_clusters = sorted(subject_clusters, key=cluster_rank_key)

        chosen_rows = []

        for rank, cid in enumerate(ranked_clusters):
            if len(chosen_rows) >= k:
                break

            cdf = sdf[sdf[cluster_col] == cid].copy()

            # Within a selected cluster, choose representative/cleaner segments first if possible.
            if distance_col in cdf.columns:
                cdf = cdf.sort_values(distance_col, ascending=True)
            elif "iforest_score" in cdf.columns:
                cdf = cdf.sort_values("iforest_score", ascending=False)
            else:
                cdf = cdf.sort_values("segment_id", ascending=True)

            for _, row in cdf.iterrows():
                if len(chosen_rows) >= k:
                    break

                chosen_rows.append(row)

                info = cluster_info[int(cid)]
                debug_rows.append({
                    "subject_id": sid,
                    "subject_label": y,
                    "segment_id": int(row["segment_id"]),
                    "cluster_id": int(cid),
                    "cluster_rank_for_subject": int(rank),
                    "p_subject_label_given_cluster": float(info.get(f"p_class_{y}", 0.0)),
                    "entropy_norm": float(info["entropy_norm"]),
                    "subject_segments_in_cluster": int(subject_cluster_count.get(cid, 0)),
                    "selected_order": len(chosen_rows),
                })

        for row in chosen_rows:
            selected_keys.append((str(row["subject_id"]), int(row["segment_id"])))

    selected_graphs = [graph_lookup[key] for key in selected_keys if key in graph_lookup]

    if len(selected_graphs) == 0:
        raise RuntimeError("No graphs selected by label-aligned greedy strategy.")

    # -------------------------
    # save diagnostics
    # -------------------------
    if debug_dir is not None:
        os.makedirs(debug_dir, exist_ok=True)

        cluster_stats.to_csv(
            os.path.join(debug_dir, "label_aligned_cluster_stats.csv"),
            index=False,
        )

        pd.DataFrame(debug_rows).to_csv(
            os.path.join(debug_dir, "label_aligned_selected_segments.csv"),
            index=False,
        )

    print("\n[label_aligned_greedy_k]")
    print(f"selected graphs: {len(selected_graphs)}")
    print(f"selected subjects: {len(set(str(g.subject_id) for g in selected_graphs))}")
    print(f"k per subject target: {k}")
    print(f"only_clean: {only_clean}")

    return selected_graphs
def _extract_edge_weight_1d(pyg_batch):
    edge_weight = getattr(pyg_batch, "edge_weight", None)

    if edge_weight is None:
        edge_weight = getattr(pyg_batch, "edge_attr", None)

    if edge_weight is None:
        return None

    if edge_weight.dim() > 1:
        if edge_weight.size(-1) == 1:
            edge_weight = edge_weight.squeeze(-1)
        else:
            edge_weight = edge_weight[:, 0]

    return edge_weight.float()


def apply_gcn_norm_to_pyg_batch(
    pyg_batch,
    *,
    add_self_loops=True,
    improved=False,
    use_abs_edge_weight=False,
):
    """
    Apply GCN normalization to sparse PyG Batch:
        A_hat = D^{-1/2} (A + I) D^{-1/2}

    Updates:
        pyg_batch.edge_index
        pyg_batch.edge_weight
        pyg_batch.edge_attr
    """
    pyg_batch = pyg_batch.clone()

    edge_weight = _extract_edge_weight_1d(pyg_batch)

    if edge_weight is not None and use_abs_edge_weight:
        edge_weight = edge_weight.abs()

    edge_index_norm, edge_weight_norm = gcn_norm(
        pyg_batch.edge_index,
        edge_weight,
        num_nodes=pyg_batch.num_nodes,
        improved=improved,
        add_self_loops=add_self_loops,
        flow="source_to_target",
        dtype=pyg_batch.x.dtype,
    )

    pyg_batch.edge_index = edge_index_norm
    pyg_batch.edge_weight = edge_weight_norm
    pyg_batch.edge_attr = edge_weight_norm.view(-1, 1)

    return pyg_batch

def gcn_norm_dense_adj(
    adj,
    *,
    add_self_loops=True,
    symmetrize=True,
    eps=1e-8,
    use_abs_degree=False,
):
    """
    Dense GCN normalization.

    Supports:
        [B, N, N]
        [B, K, N, N]
        [B, C, N, N]
    """
    A = adj.float()
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    if symmetrize:
        A = 0.5 * (A + A.transpose(-1, -2))

    n = A.size(-1)
    eye_shape = [1] * (A.dim() - 2) + [n, n]
    eye = torch.eye(n, device=A.device, dtype=A.dtype).view(*eye_shape)

    # remove old diagonal first, then add clean self-loops
    A = A * (1.0 - eye)

    if add_self_loops:
        A = A + eye

    degree_source = A.abs() if use_abs_degree else A
    deg = degree_source.sum(dim=-1).clamp_min(eps)
    deg_inv_sqrt = deg.rsqrt()

    A_norm = deg_inv_sqrt.unsqueeze(-1) * A * deg_inv_sqrt.unsqueeze(-2)
    return A_norm

def apply_gcn_norm_to_batch_dict(
    batch_dict,
    *,
    add_self_loops=True,
    use_abs_edge_weight=False,
    use_abs_degree=False,
):
    """
    Normalize all graph inputs in one place.
    This makes GCN normalization available to all encoders.
    """
    batch_dict = dict(batch_dict)

    if "pyg_batch" in batch_dict and batch_dict["pyg_batch"] is not None:
        batch_dict["pyg_batch"] = apply_gcn_norm_to_pyg_batch(
            batch_dict["pyg_batch"],
            add_self_loops=add_self_loops,
            use_abs_edge_weight=use_abs_edge_weight,
        )

    # For LinkX / dense adjacency encoders
    if "full_adj" in batch_dict and batch_dict["full_adj"] is not None:
        batch_dict["full_adj"] = gcn_norm_dense_adj(
            batch_dict["full_adj"],
            add_self_loops=add_self_loops,
            use_abs_degree=use_abs_degree,
        )

    # For multiband CNN encoder: [G, B, N, N]
    if "conn_stack" in batch_dict and batch_dict["conn_stack"] is not None:
        batch_dict["conn_stack"] = gcn_norm_dense_adj(
            batch_dict["conn_stack"],
            add_self_loops=add_self_loops,
            use_abs_degree=use_abs_degree,
        )

    # For graph-bank encoder: [G, K, N, N]
    if "adj_bank" in batch_dict and batch_dict["adj_bank"] is not None:
        batch_dict["adj_bank"] = gcn_norm_dense_adj(
            batch_dict["adj_bank"],
            add_self_loops=add_self_loops,
            use_abs_degree=use_abs_degree,
        )

    # Do NOT normalize topology_bank.
    # It should stay binary because it is a mask/topology indicator.

    return batch_dict
def dense_adj_to_gcn_norm_graph(x, adj):
    """
    x:   [N, F]
    adj: [N, N]

    returns PyG Data with:
      edge_index
      edge_weight = GCN-normalized edge weights
      edge_attr   = normalized edge weights as edge features for GATv2
    """

    adj = torch.as_tensor(adj, dtype=torch.float32)
    x = torch.as_tensor(x, dtype=torch.float32)

    # Important for EEG connectivity:
    # For Pearson/Spearman, prefer abs() or clamp(min=0)
    adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    adj = 0.5 * (adj + adj.T)
    adj.fill_diagonal_(0.0)

    # safer for signed connectivity
    adj = adj.abs()

    edge_index, edge_weight = dense_to_sparse(adj)

    data = Data(
        x=x,
        edge_index=edge_index.long(),
        edge_weight=edge_weight.float(),
        num_nodes=x.size(0),
    )

    # Apply GCN normalization once
    data = gcn_norm_transform(data)

    # GATv2 uses edge_attr, not edge_weight directly
    data.edge_attr = data.edge_weight.view(-1, 1).float()

    return data

class RawNodeMultiBandCNNEncoder(nn.Module):
    """
    LINKX-style encoder:
      - node branch: MLP over flattened node features
      - connectivity branch: CNN over multiband dense adjacency [G, B, N, N]

    This is the simple version using Conv2d with frequency bands as channels.
    """

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        num_bands: int = 5,
        node_hidden_dims: Sequence[int] = (256, 128),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        # cnn_channels: Sequence[int] = (16, 32, 64),
        dropout: float = 0.2,
        symmetrize_adj: bool = True,
        zero_diagonal: bool = False,
    ):
        super().__init__()

        # if len(cnn_channels) != 3:
        #     raise ValueError("cnn_channels should have length 3, e.g. (16, 32, 64)")

        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.num_bands = num_bands
        self.symmetrize_adj = symmetrize_adj
        self.zero_diagonal = zero_diagonal

        # ----- node branch -----
        node_input_dim = num_nodes * num_node_features
        self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # ----- connectivity CNN branch -----
        # c1, c2, c3 = cnn_channels
        # self.conv1 = nn.Conv2d(in_channels=5, out_channels=32, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(
            in_channels=self.num_bands,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 19 -> 9

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 9 -> 4

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # 4 -> 2

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, branch_emb_dim)

        self.dropout = nn.Dropout(dropout)


        # ----- fusion -----
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _prepare_conn_stack(self, conn_stack: torch.Tensor) -> torch.Tensor:
        """
        conn_stack: [G, B, N, N]
        returns   : [G, B, N, N]
        """
        if conn_stack.ndim != 4:
            raise ValueError(f"Expected conn_stack with shape [G, B, N, N], got {tuple(conn_stack.shape)}")

        g, b, n1, n2 = conn_stack.shape
        if b != self.num_bands:
            raise ValueError(f"Expected num_bands={self.num_bands}, got {b}")
        if n1 != self.num_nodes or n2 != self.num_nodes:
            raise ValueError(
                f"Expected conn_stack shape [G, {self.num_bands}, {self.num_nodes}, {self.num_nodes}], "
                f"got {tuple(conn_stack.shape)}"
            )

        x = conn_stack.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.symmetrize_adj:
            x = 0.5 * (x + x.transpose(-1, -2))

        if self.zero_diagonal:
            eye = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).view(1, 1, self.num_nodes, self.num_nodes)
            x = x * (1.0 - eye)

        return x

    def forward(self, pyg_batch, conn_stack: torch.Tensor):
        # ----- node branch -----
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [G, N, F]

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")

        node_x = dense_x.reshape(dense_x.size(0), -1)  # [G, N*F]
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)              # [G, branch_emb_dim]

        # ----- connectivity branch -----
        x = self._prepare_conn_stack(conn_stack)        # [B, 5, 19, 19]

        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) # [B, 32, 9, 9]
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # [B, 64, 4, 4]
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # [B, 128, 2, 2]

        x = x.flatten(start_dim=1)                      # [B, 512]
        x = self.dropout(F.relu(self.fc1(x)))           # [B, 256]
        cnn_emb = self.dropout(F.relu(self.fc2(x)))     # [B, branch_emb_dim]

        # ----- fuse -----
        fused = torch.cat([node_emb, cnn_emb], dim=1)
        graph_emb = self.fusion(fused)                 # [G, emb_dim]
        return graph_emb



class MultiBandCNNEncoder(nn.Module):
    """
    LINKX-style encoder:
      - node branch: MLP over flattened node features
      - connectivity branch: CNN over multiband dense adjacency [G, B, N, N]

    This is the simple version using Conv2d with frequency bands as channels.
    """

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        num_bands: int = 5,
        node_hidden_dims: Sequence[int] = (256, 128),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        # cnn_channels: Sequence[int] = (16, 32, 64),
        dropout: float = 0.2,
        symmetrize_adj: bool = True,
        zero_diagonal: bool = False,
    ):
        super().__init__()

        # if len(cnn_channels) != 3:
        #     raise ValueError("cnn_channels should have length 3, e.g. (16, 32, 64)")

        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.num_bands = num_bands
        self.symmetrize_adj = symmetrize_adj
        self.zero_diagonal = zero_diagonal

        # ----- node branch -----
        # node_input_dim = num_nodes * num_node_features
        # self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        # self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # ----- connectivity CNN branch -----
        # c1, c2, c3 = cnn_channels
        self.conv1 = nn.Conv2d(in_channels=num_bands, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 19 -> 9

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 9 -> 4

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)   # 4 -> 2

        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, branch_emb_dim)

        self.dropout = nn.Dropout(dropout)


        # ----- fusion -----
        self.fusion = nn.Sequential(
            nn.Linear(branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _prepare_conn_stack(self, conn_stack: torch.Tensor) -> torch.Tensor:
        """
        conn_stack: [G, B, N, N]
        returns   : [G, B, N, N]
        """
        if conn_stack.ndim != 4:
            raise ValueError(f"Expected conn_stack with shape [G, B, N, N], got {tuple(conn_stack.shape)}")

        g, b, n1, n2 = conn_stack.shape
        if b != self.num_bands:
            raise ValueError(f"Expected num_bands={self.num_bands}, got {b}")
        if n1 != self.num_nodes or n2 != self.num_nodes:
            raise ValueError(
                f"Expected conn_stack shape [G, {self.num_bands}, {self.num_nodes}, {self.num_nodes}], "
                f"got {tuple(conn_stack.shape)}"
            )

        x = conn_stack.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.symmetrize_adj:
            x = 0.5 * (x + x.transpose(-1, -2))

        if self.zero_diagonal:
            eye = torch.eye(self.num_nodes, device=x.device, dtype=x.dtype).view(1, 1, self.num_nodes, self.num_nodes)
            x = x * (1.0 - eye)

        return x

    def forward(self, pyg_batch, conn_stack: torch.Tensor):
        # ----- node branch -----
        # dense_x, _ = to_dense_batch(
        #     pyg_batch.x,
        #     pyg_batch.batch,
        #     max_num_nodes=self.num_nodes,
        # )  # [G, N, F]

        # if dense_x.size(1) != self.num_nodes:
        #     raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")

        # node_x = dense_x.reshape(dense_x.size(0), -1)  # [G, N*F]
        # node_h = self.node_mlp(node_x)
        # node_emb = self.node_proj(node_h)              # [G, branch_emb_dim]

        # ----- connectivity branch -----
        x = self._prepare_conn_stack(conn_stack)        # [B, 5, 19, 19]

        x = self.pool1(F.relu(self.bn1(self.conv1(x)))) # [B, 32, 9, 9]
        x = self.pool2(F.relu(self.bn2(self.conv2(x)))) # [B, 64, 4, 4]
        x = self.pool3(F.relu(self.bn3(self.conv3(x)))) # [B, 128, 2, 2]

        x = x.flatten(start_dim=1)                      # [B, 512]
        x = self.dropout(F.relu(self.fc1(x)))           # [B, 256]
        cnn_emb = self.dropout(F.relu(self.fc2(x)))     # [B, branch_emb_dim]

        # ----- fuse -----
        # fused = torch.cat([node_emb, cnn_emb], dim=1)
        graph_emb = self.fusion(cnn_emb)                 # [G, emb_dim]
        return graph_emb


class RawNodeGraphBankGNNEncoder(nn.Module):
    """
    Node-feature branch + shared-GNN graph-bank branch.

    For each segment/graph:
        X:        [N, F]
        adj_bank: [K, N, N]

    For a batch:
        pyg_batch.x contains all node features.
        adj_bank should be [B, K, N, N].

    Forward:
        node_emb = MLP(vec(X))
        z_k = SharedGNN(X, A_k)
        graph_bank_emb = attention_pool({z_k}_{k=1}^K)
        final_emb = fusion(node_emb, graph_bank_emb)

    Returns:
        graph_emb, aux
    """

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        graph_emb_dim: int = 64,
        branch_emb_dim: int = 64,
        node_hidden_dims: Sequence[int] = (128, 64),
        gnn_hidden_dim: int = 64,
        gnn_out_dim: int = 64,
        gnn_layers: int = 2,
        backbone: BackboneType = "gatv2",
        readout: ReadoutType = "mean",
        fusion: FusionType = "gated",
        dropout: float = 0.3,
        gat_heads: int = 4,
        adj_value_mode: AdjValueMode = "abs",
        symmetrize_adj: bool = True,
        zero_diagonal: bool = True,
        use_edge_weight: bool = True,
        use_gcn_norm: bool = False,
        gcn_norm_add_self_loops: bool = True
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.graph_emb_dim = int(graph_emb_dim)
        self.branch_emb_dim = int(branch_emb_dim)

        self.fusion = fusion.lower()
        self.adj_value_mode = adj_value_mode.lower()
        self.symmetrize_adj = bool(symmetrize_adj)
        self.zero_diagonal = bool(zero_diagonal)
        from torch_geometric.transforms import GCNNorm

        # self.gcn_norm_transform = GCNNorm(add_self_loops=True)

        self.use_gcn_norm = bool(use_gcn_norm)
        self.gcn_norm_add_self_loops = bool(gcn_norm_add_self_loops)

        self.gcn_norm_transform = (
            GCNNorm(add_self_loops=gcn_norm_add_self_loops)
            if self.use_gcn_norm
            else None
        )
        # -------------------------
        # Node feature branch
        # -------------------------
        node_input_dim = self.num_nodes * self.num_node_features
        self.node_mlp, node_last_dim = make_mlp(
            node_input_dim,
            node_hidden_dims,
            dropout=dropout,
        )
        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # -------------------------
        # Shared GNN branch
        # -------------------------
        # self.shared_gnn = SharedGraphEncoder(
        #     in_dim=num_node_features,
        #     hidden_dim=gnn_hidden_dim,
        #     out_dim=gnn_out_dim,
        #     num_layers=gnn_layers,
        #     backbone=backbone,
        #     dropout=dropout,
        #     readout=readout,
        #     gat_heads=gat_heads,
        #     use_edge_weight=use_edge_weight,
        # )
        self.shared_gnn = SharedGraphEncoder(
            in_dim=num_node_features,
            hidden_dim=gnn_hidden_dim,
            out_dim=gnn_out_dim,
            num_layers=gnn_layers,
            backbone=backbone,
            dropout=dropout,
            readout=readout,
            gat_heads=gat_heads,
            use_edge_weight=use_edge_weight,
            use_gcn_norm=use_gcn_norm,
            gcn_norm_add_self_loops=gcn_norm_add_self_loops,
        )

        self.graph_proj = nn.Linear(self.shared_gnn.output_dim, branch_emb_dim)

        # -------------------------
        # Attention over K adjacency views
        # -------------------------
        self.view_attention = nn.Sequential(
            nn.Linear(branch_emb_dim, branch_emb_dim),
            nn.Tanh(),
            nn.Linear(branch_emb_dim, 1),
        )

        # -------------------------
        # Node + graph-bank fusion
        # -------------------------
        if self.fusion == "concat":
            self.fusion_mlp = nn.Sequential(
                nn.Linear(2 * branch_emb_dim, graph_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        elif self.fusion == "gated":
            # self.node_to_out = nn.Linear(branch_emb_dim, graph_emb_dim)
            # self.graph_to_out = nn.Linear(branch_emb_dim, graph_emb_dim)

            # self.gate = nn.Sequential(
            #     nn.Linear(2 * branch_emb_dim, graph_emb_dim),
            #     nn.Sigmoid(),
            # )

            self.node_to_out = nn.Linear(branch_emb_dim, graph_emb_dim)
            self.graph_to_delta = nn.Sequential(
                nn.Linear(branch_emb_dim, graph_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(graph_emb_dim, graph_emb_dim),
            )

            # alpha starts very small
            self.graph_residual_scale = nn.Parameter(torch.tensor(-4.0))

        else:
            raise ValueError("fusion must be either 'concat' or 'gated'")

    def _prepare_adj_bank(
        self,
        adj_bank: Tensor,
        batch_size: int,
    ) -> Tensor:
        """
        Accepts:
            [K, N, N] for one graph
            [B, N, N] for single adjacency per graph
            [B*K, N, N] from PyG batching
            [B, K, N, N]

        Returns:
            [B, K, N, N]
        """
        if not torch.is_tensor(adj_bank):
            adj_bank = torch.as_tensor(adj_bank, dtype=torch.float32)

        adj_bank = adj_bank.float()

        if adj_bank.ndim == 4:
            if adj_bank.shape[0] != batch_size:
                raise ValueError(
                    f"Expected adj_bank batch size {batch_size}, got {adj_bank.shape[0]}"
                )
            out = adj_bank

        elif adj_bank.ndim == 3:
            # Case: one graph with K matrices -> [1, K, N, N]
            if batch_size == 1:
                out = adj_bank.unsqueeze(0)

            # Case: one adjacency per graph -> [B, 1, N, N]
            elif adj_bank.shape[0] == batch_size:
                out = adj_bank.unsqueeze(1)

            # Case: flattened [B*K, N, N]
            elif adj_bank.shape[0] % batch_size == 0:
                k = adj_bank.shape[0] // batch_size
                out = adj_bank.view(batch_size, k, self.num_nodes, self.num_nodes)

            else:
                raise ValueError(
                    f"Cannot reshape adj_bank shape {tuple(adj_bank.shape)} "
                    f"into [B, K, N, N] with B={batch_size}"
                )
        else:
            raise ValueError(
                f"adj_bank must have shape [K,N,N], [B,N,N], [B*K,N,N], or [B,K,N,N]. "
                f"Got {tuple(adj_bank.shape)}"
            )

        if out.shape[-2:] != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"Expected adjacency shape [N,N]=[{self.num_nodes},{self.num_nodes}], "
                f"got {tuple(out.shape[-2:])}"
            )

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if self.symmetrize_adj:
            out = 0.5 * (out + out.transpose(-1, -2))

        if self.zero_diagonal:
            eye = torch.eye(
                self.num_nodes,
                device=out.device,
                dtype=out.dtype,
            ).view(1, 1, self.num_nodes, self.num_nodes)
            out = out * (1.0 - eye)

        if self.adj_value_mode == "abs":
            out = out.abs()
        elif self.adj_value_mode == "positive":
            out = torch.clamp(out, min=0.0)
        elif self.adj_value_mode == "binary":
            out = (out.abs() > 1e-8).float()
        elif self.adj_value_mode == "raw":
            pass
        else:
            raise ValueError(f"Unknown adj_value_mode={self.adj_value_mode!r}")

        return out

    def _dense_view_to_pyg_batch(
        self,
        dense_x: Tensor,
        dense_adj: Tensor,
        node_mask: Tensor,
    ) -> Batch:
        """
        Convert one adjacency view into a PyG Batch.

        dense_x:
            [B, N, F]
        dense_adj:
            [B, N, N]
        node_mask:
            [B, N]
        """
        data_list = []

        for b in range(dense_x.shape[0]):
            valid = node_mask[b].bool()

            x_b = dense_x[b, valid]
            adj_b = dense_adj[b][valid][:, valid]

            adj_b = torch.nan_to_num(adj_b, nan=0.0, posinf=0.0, neginf=0.0)
            adj_b = 0.5 * (adj_b + adj_b.T)
            adj_b.fill_diagonal_(0.0)

            # important for signed EEG connectivity
            if self.adj_value_mode == "abs":
                adj_b = adj_b.abs()
            elif self.adj_value_mode == "positive":
                adj_b = torch.clamp(adj_b, min=0.0)
            elif self.adj_value_mode == "binary":
                adj_b = (adj_b.abs() > 1e-8).float()
            elif self.adj_value_mode == "raw":
                pass
            else:
                raise ValueError(f"Unknown adj_value_mode={self.adj_value_mode}")

            edge_index, edge_weight = dense_to_sparse(adj_b)

            g = Data(
                x=x_b,
                edge_index=edge_index.long(),
                edge_weight=edge_weight.float(),
                num_nodes=x_b.size(0),
            )

            # GCN normalization at input stage
            # g = self.gcn_norm_transform(g)

            # Optional GCN normalization
            if self.use_gcn_norm:
                if self.gcn_norm_transform is None:
                    raise RuntimeError("gcn_norm_transform is None but use_gcn_norm=True")
                g = self.gcn_norm_transform(g)

            g.edge_attr = g.edge_weight.view(-1, 1).float()

            data_list.append(g)

        return Batch.from_data_list(data_list)
    def forward(
        self,
        pyg_batch: Batch,
        adj_bank: Optional[Tensor] = None,
    ) -> tuple[Tensor, Dict[str, Any]]:
        """
        Parameters
        ----------
        pyg_batch:
            PyG batch containing node features.
        adj_bank:
            Tensor [B, K, N, N]. If None, tries pyg_batch.adj_bank.

        Returns
        -------
        graph_emb:
            [B, graph_emb_dim]
        aux:
            dictionary with view attention weights and intermediate embeddings.
        """
        dense_x, node_mask = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [B, N, F], [B, N]

        batch_size = dense_x.shape[0]

        if dense_x.shape[1] != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, got {dense_x.shape[1]}"
            )

        if dense_x.shape[2] != self.num_node_features:
            raise ValueError(
                f"Expected num_node_features={self.num_node_features}, got {dense_x.shape[2]}"
            )

        if adj_bank is None:
            adj_bank = getattr(pyg_batch, "adj_bank", None)

        if adj_bank is None:
            raise ValueError(
                "RawNodeGraphBankGNNEncoder needs adj_bank. "
                "Pass adj_bank explicitly or attach g.adj_bank to each graph."
            )

        adj_bank = self._prepare_adj_bank(adj_bank, batch_size=batch_size)
        num_views = adj_bank.shape[1]

        # -------------------------
        # Node branch
        # -------------------------
        node_flat = dense_x.reshape(batch_size, -1)
        node_h = self.node_mlp(node_flat)
        node_emb = self.node_proj(node_h)  # [B, branch_emb_dim]

        # -------------------------
        # Shared GNN over each adjacency view
        # -------------------------
        view_embs = []

        for k in range(num_views):
            view_batch = self._dense_view_to_pyg_batch(
                dense_x=dense_x,
                dense_adj=adj_bank[:, k],
                node_mask=node_mask,
            )
            view_batch = view_batch.to(dense_x.device)

            z_k = self.shared_gnn(view_batch)   # [B, gnn_readout_dim]
            z_k = self.graph_proj(z_k)          # [B, branch_emb_dim]
            view_embs.append(z_k)

        view_embs = torch.stack(view_embs, dim=1)  # [B, K, branch_emb_dim]

        # -------------------------
        # Attention over graph views
        # -------------------------
        view_scores = self.view_attention(view_embs).squeeze(-1)  # [B, K]
        view_alpha = torch.softmax(view_scores, dim=1)            # [B, K]

        graph_bank_emb = torch.sum(
            view_alpha.unsqueeze(-1) * view_embs,
            dim=1,
        )  # [B, branch_emb_dim]

        # -------------------------
        # Fuse node branch and graph-bank branch
        # -------------------------
        if self.fusion == "concat":
            graph_emb = self.fusion_mlp(
                torch.cat([node_emb, graph_bank_emb], dim=-1)
            )

        else:
            # node_out = self.node_to_out(node_emb)
            # graph_out = self.graph_to_out(graph_bank_emb)

            # gate = self.gate(torch.cat([node_emb, graph_bank_emb], dim=-1))
            # graph_emb = gate * graph_out + (1.0 - gate) * node_out
            node_out = self.node_to_out(node_emb)
            graph_delta = self.graph_to_delta(graph_bank_emb)

            alpha = torch.sigmoid(self.graph_residual_scale)  # starts around 0.018
            graph_emb = node_out + alpha * graph_delta
        aux = {
            "view_attention": view_alpha.detach(),
            "view_scores": view_scores.detach(),
            "node_embedding": node_emb.detach(),
            "graph_bank_embedding": graph_bank_emb.detach(),
            "view_embeddings": view_embs.detach(),
        }

        return graph_emb, aux

class RawNodeAdjCNNEncoder(nn.Module):
    """
    LINKX-style encoder:
      - node branch: MLP over flattened node features
      - topology branch: CNN over dense adjacency [B, 1, N, N]

    This is the quick version using one adjacency matrix per graph.
    """

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        node_hidden_dims: Sequence[int] = (256, 128),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        cnn_channels: Sequence[int] = (16, 32),
        dropout: float = 0.2,
        symmetrize_adj: bool = True,
        zero_diagonal: bool = False,
    ):
        super().__init__()

        if len(cnn_channels) != 2:
            raise ValueError("cnn_channels should have length 2, e.g. (16, 32)")

        self.num_nodes = num_nodes
        self.num_node_features = num_node_features
        self.symmetrize_adj = symmetrize_adj
        self.zero_diagonal = zero_diagonal

        # ----- node branch -----
        node_input_dim = num_nodes * num_node_features
        self.node_mlp, node_last_dim = make_mlp(node_input_dim, node_hidden_dims, dropout)
        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        # ----- adjacency CNN branch -----
        c1, c2 = cnn_channels
        self.adj_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.adj_proj = nn.Linear(c2, branch_emb_dim)

        # ----- fusion -----
        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _prepare_adj(self, full_adj: torch.Tensor) -> torch.Tensor:
        """
        full_adj: [B, N, N]
        return : [B, 1, N, N]
        """
        if full_adj.ndim != 3:
            raise ValueError(f"Expected full_adj with shape [B, N, N], got {tuple(full_adj.shape)}")

        if full_adj.size(1) != self.num_nodes or full_adj.size(2) != self.num_nodes:
            raise ValueError(
                f"Expected full_adj shape [B, {self.num_nodes}, {self.num_nodes}], "
                f"got {tuple(full_adj.shape)}"
            )

        adj = full_adj.float()

        if self.symmetrize_adj:
            adj = 0.5 * (adj + adj.transpose(1, 2))

        if self.zero_diagonal:
            eye = torch.eye(self.num_nodes, device=adj.device, dtype=adj.dtype).unsqueeze(0)
            adj = adj * (1.0 - eye)

        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        adj = adj.unsqueeze(1)   # [B, 1, N, N]
        return adj

    def forward(self, pyg_batch, full_adj: torch.Tensor):
        # ----- node branch -----
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [B, N, F]

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}")

        node_x = dense_x.reshape(dense_x.size(0), -1)   # [B, N*F]
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)               # [B, branch_emb_dim]

        # ----- adjacency CNN branch -----
        adj_4d = self._prepare_adj(full_adj)            # [B, 1, N, N]
        adj_h = self.adj_cnn(adj_4d)                    # [B, c2, 1, 1]
        adj_h = adj_h.flatten(1)                        # [B, c2]
        adj_emb = self.adj_proj(adj_h)                  # [B, branch_emb_dim]

        # ----- fuse -----
        fused = torch.cat([node_emb, adj_emb], dim=1)
        graph_emb = self.fusion(fused)                  # [B, emb_dim]
        return graph_emb

def _stable_int_from_string(x: str) -> int:
    """
    Stable integer hash from a string.
    Do NOT use Python's built-in hash(), because it is randomized across runs.
    """
    s = str(x).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def _move_to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors in nested structures to CPU so checkpoints are portable.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj
def _normalize_fixed_edges(
    fixed_edges: Optional[EdgeSpec],
    n_channels: int,
    channel_names: Optional[Sequence[str]] = None,
) -> set:
    """
    Convert fixed_edges into a set of sorted integer node pairs.
    Supports:
      - integer edges: [(0,1), (1,2)]
      - channel-name edges: [("Fp1","F3"), ("F3","C3")]
    """
    if fixed_edges is None:
        return set()

    fixed_pairs = set()
    name_to_idx = None

    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(
                f"channel_names has length {len(channel_names)} but n_channels={n_channels}"
            )
        name_to_idx = {name: i for i, name in enumerate(channel_names)}

    for u, v in fixed_edges:
        if isinstance(u, str) or isinstance(v, str):
            if name_to_idx is None:
                raise ValueError(
                    "fixed_edges contains channel names, but channel_names was not provided."
                )
            if u not in name_to_idx or v not in name_to_idx:
                continue
            i, j = name_to_idx[u], name_to_idx[v]
        else:
            i, j = int(u), int(v)

        if i == j:
            continue
        if not (0 <= i < n_channels and 0 <= j < n_channels):
            raise ValueError(f"Fixed edge {(u, v)} is out of range for {n_channels} nodes.")

        fixed_pairs.add(tuple(sorted((i, j))))

    return fixed_pairs
# =========================================================
# Early stopping based on subject-level validation loss
# =========================================================

class EarlyStopping:
    """
    Early stopping driven by LOWER validation loss.

    Features:
    - warmup via start_epoch
    - min_delta for meaningful improvement
    - keeps top-k checkpoints with lowest val_loss
    - exposes selected checkpoint using:
        1) max val_bal_acc
        2) max val_macro_f1
        3) min val_loss
    - deletes checkpoints that fall out of the top-k set
    """

    def __init__(
        self,
        patience: int,
        start_epoch: int = 0,
        min_delta: float = 0.0,
        top_k: int = 1,
        save_dir: Optional[str] = None,
        verbose: bool = True,
        file_prefix: str = "checkpoint",
    ):
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if start_epoch < 0:
            raise ValueError(f"start_epoch must be >= 0, got {start_epoch}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.patience = int(patience)
        self.start_epoch = int(start_epoch)
        self.min_delta = float(min_delta)
        self.top_k = int(top_k)
        self.save_dir = save_dir
        self.verbose = verbose
        self.file_prefix = file_prefix

        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)

        self.best_val_loss = float("inf")
        self.best_epoch_by_loss = None
        self.counter = 0
        self.should_stop = False
        self.stop_epoch = None

        # sorted by storage rule: lowest val_loss first
        self.checkpoints: List[Dict[str, Any]] = []

    def _improved(self, val_loss: float) -> bool:
        return val_loss < (self.best_val_loss - self.min_delta)

    @staticmethod
    def _storage_sort_key(meta: Dict[str, Any]):
        """
        How checkpoints are ranked for staying in top-k.
        Primary goal: lowest validation loss.
        Ties are broken deterministically.
        """
        return (
            float(meta["val_loss"]),
            -float(meta["val_bal_acc"]),
            -float(meta["val_macro_f1"]),
            int(meta["epoch"]),
        )

    @staticmethod
    def _selection_sort_key(meta: Dict[str, Any]):
        """
        Final checkpoint selection rule requested by user:
          1) max val_bal_acc
          2) max val_macro_f1
          3) min val_loss
        """
        return (
            -float(meta["val_bal_acc"]),
            -float(meta["val_macro_f1"]),
            float(meta["val_loss"]),
            int(meta["epoch"]),
        )

    def _checkpoint_filename(self, epoch: int, val_loss: float) -> str:
        return f"{self.file_prefix}_epoch{epoch:03d}_valloss{val_loss:.6f}.pt"

    def _build_meta(
        self,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "val_bal_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            "path": path,
        }

    def _build_payload(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "epoch": int(epoch),
            "model_state_dict": _move_to_cpu(copy.deepcopy(model.state_dict())),
            "optimizer_state_dict": _move_to_cpu(copy.deepcopy(optimizer.state_dict()))
            if optimizer is not None
            else None,
            # canonical fields
            "val_loss": float(val_loss),
            "val_bal_acc": float(val_bal_acc),
            "val_macro_f1": float(val_macro_f1),
            # legacy-friendly aliases
            "best_val_loss": float(val_loss),
            "best_val_bal_acc": float(val_bal_acc),
            "best_val_macro_f1": float(val_macro_f1),
        }

        if extra_state is not None:
            payload.update(_move_to_cpu(copy.deepcopy(extra_state)))

        return payload

    def _maybe_remove_from_disk(self, meta: Dict[str, Any]) -> None:
        path = meta.get("path", None)
        if path is None:
            return
        if os.path.exists(path):
            try:
                os.remove(path)
                if self.verbose:
                    print(f"Removed checkpoint that fell out of top-{self.top_k}: {path}")
            except OSError:
                if self.verbose:
                    print(f"Warning: could not remove old checkpoint: {path}")

    def _qualifies_for_topk(self, candidate_meta: Dict[str, Any]) -> bool:
        if len(self.checkpoints) < self.top_k:
            return True

        worst_meta = sorted(self.checkpoints, key=self._storage_sort_key)[-1]
        return self._storage_sort_key(candidate_meta) < self._storage_sort_key(worst_meta)

    def _insert_topk(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        candidate_meta = self._build_meta(
            epoch=epoch,
            val_loss=val_loss,
            val_bal_acc=val_bal_acc,
            val_macro_f1=val_macro_f1,
            path=None,
        )

        if not self._qualifies_for_topk(candidate_meta):
            return None

        if self.save_dir is None:
            # metadata only, no file path
            saved_meta = candidate_meta
        else:
            ckpt_path = os.path.join(
                self.save_dir,
                self._checkpoint_filename(epoch=epoch, val_loss=val_loss),
            )
            payload = self._build_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_bal_acc=val_bal_acc,
                val_macro_f1=val_macro_f1,
                extra_state=extra_state,
            )
            torch.save(payload, ckpt_path)
            saved_meta = self._build_meta(
                epoch=epoch,
                val_loss=val_loss,
                val_bal_acc=val_bal_acc,
                val_macro_f1=val_macro_f1,
                path=ckpt_path,
            )

            if self.verbose:
                print(
                    f"Saved top-k checkpoint: epoch={epoch}, "
                    f"val_loss={val_loss:.6f}, "
                    f"val_bal_acc={val_bal_acc:.4f}, "
                    f"val_macro_f1={val_macro_f1:.4f}"
                )

        self.checkpoints.append(saved_meta)
        self.checkpoints = sorted(self.checkpoints, key=self._storage_sort_key)

        while len(self.checkpoints) > self.top_k:
            removed = self.checkpoints.pop(-1)
            self._maybe_remove_from_disk(removed)

        return saved_meta

    def __call__(
        self,
        model,
        optimizer,
        epoch: int,
        val_loss: float,
        val_bal_acc: float,
        val_macro_f1: float,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Returns:
            True if training should stop, else False
        """
        val_loss = float(val_loss)
        val_bal_acc = float(val_bal_acc)
        val_macro_f1 = float(val_macro_f1)
        epoch = int(epoch)

        # Always maintain top-k checkpoints
        self._insert_topk(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            val_bal_acc=val_bal_acc,
            val_macro_f1=val_macro_f1,
            extra_state=extra_state,
        )

        monitor_active = epoch >= self.start_epoch

        if self._improved(val_loss):
            self.best_val_loss = val_loss
            self.best_epoch_by_loss = epoch
            if monitor_active:
                self.counter = 0

            if self.verbose:
                print(
                    f"Validation loss improved at epoch {epoch}: "
                    f"{val_loss:.6f}"
                )
        else:
            if monitor_active:
                self.counter += 1
                if self.verbose:
                    print(
                        f"No val-loss improvement at epoch {epoch}. "
                        f"patience {self.counter}/{self.patience}"
                    )

        if monitor_active and self.counter >= self.patience:
            self.should_stop = True
            self.stop_epoch = epoch
            if self.verbose:
                print(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best val_loss={self.best_val_loss:.6f} "
                    f"(epoch {self.best_epoch_by_loss})."
                )

        return self.should_stop

    def get_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Select final checkpoint from saved top-k by:
          1) highest val_bal_acc
          2) highest val_macro_f1
          3) lower val_loss
        """
        if len(self.checkpoints) == 0:
            return None
        best_meta = sorted(self.checkpoints, key=self._selection_sort_key)[0]
        return copy.deepcopy(best_meta)

    def get_topk_checkpoints(self) -> List[Dict[str, Any]]:
        return copy.deepcopy(self.checkpoints)

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



class SharedGraphEncoder(nn.Module):
    """
    Shared GNN used for each graph view A^(k).

    Input:
        PyG Batch created from one adjacency view.

    Output:
        graph embedding [B, graph_out_dim]
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_layers: int = 2,
        backbone: BackboneType = "gatv2",
        dropout: float = 0.3,
        readout: ReadoutType = "mean",
        gat_heads: int = 4,
        use_edge_weight: bool = True,
        use_gcn_norm: bool = False,
        gcn_norm_add_self_loops: bool = True,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.backbone = backbone.lower()
        self.dropout = float(dropout)
        self.readout = readout
        self.use_edge_weight = bool(use_edge_weight)

        self.use_gcn_norm = bool(use_gcn_norm)
        self.gcn_norm_add_self_loops = bool(gcn_norm_add_self_loops)

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            src_dim = dims[i]
            dst_dim = dims[i + 1]

            if self.backbone == "gcn":
                # conv = GCNConv(src_dim, dst_dim)
                conv = GCNConv(
                    src_dim,
                    dst_dim,
                    normalize=not self.use_gcn_norm,
                    add_self_loops=not self.use_gcn_norm,
                )
            elif self.backbone == "sage":
                conv = SAGEConv(src_dim, dst_dim)
            elif self.backbone == "gatv2":
                conv = GATv2Conv(
                    src_dim,
                    dst_dim,
                    heads=gat_heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=1,
                    add_self_loops=not self.use_gcn_norm,
                    fill_value="mean",
                )
            else:
                raise ValueError(f"Unknown backbone={backbone!r}")

            self.convs.append(conv)
            self.norms.append(GraphNorm(dst_dim))

        if readout == "mean_max":
            self.output_dim = out_dim * 2
        else:
            self.output_dim = out_dim

    def forward(self, batch: Batch) -> Tensor:
        x = batch.x
        edge_index = batch.edge_index
        edge_weight = getattr(batch, "edge_weight", None)
        edge_attr = getattr(batch, "edge_attr", None)

        for conv, norm in zip(self.convs, self.norms):

            if self.backbone == "gcn":
                if self.use_edge_weight and edge_weight is not None:
                    x = conv(x, edge_index, edge_weight=edge_weight)
                else:
                    x = conv(x, edge_index)

            elif self.backbone == "gatv2":
                if self.use_edge_weight and edge_attr is not None:
                    x = conv(x, edge_index, edge_attr=edge_attr)
                else:
                    x = conv(x, edge_index)

            else:
                x = conv(x, edge_index)

            x = norm(x, batch.batch)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.readout == "mean":
            return global_mean_pool(x, batch.batch)
        if self.readout == "max":
            return global_max_pool(x, batch.batch)
        if self.readout == "add":
            return global_add_pool(x, batch.batch)
        if self.readout == "mean_max":
            return torch.cat(
                [
                    global_mean_pool(x, batch.batch),
                    global_max_pool(x, batch.batch),
                ],
                dim=-1,
            )

        raise ValueError(f"Unknown readout={self.readout!r}")


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

class FusedBankLinkXEncoder(nn.Module):
    """
    Approach (1):
      - fuse a bank of candidate adjacency matrices first
      - then feed the fused graph into the usual LINKX encoder

    Reuses:
      - GraphBankFusionBlock from pipeline.gnn
      - RawNodeEdgeMLPEncoder already in this file
      - helper functions from pipeline.gnn for bank extraction / dense->PyG rebuild
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_candidates: int,
        node_hidden_dims=(256, 128),
        edge_hidden_dims=(128, 64),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
        edge_mode: str = "topology_weighted",
        bank_fusion_mode: str = "static",          # "static" | "summary_gated"
        topology_rule: str = "union",              # "union" | "intersection" | "vote"
        vote_threshold: float = 0.5,
        fusion_temperature: float = 1.0,
        fusion_hidden_dim: int = 64,
    ):
        super().__init__()

        # lazy import here to avoid circular import at module load time
        from pipeline.gnn import GraphBankFusionBlock

        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_candidates = int(num_candidates)

        self.bank_fusion = GraphBankFusionBlock(
            num_candidates=self.num_candidates,
            num_node_features=self.num_node_features,
            fusion_mode=bank_fusion_mode,
            topology_rule=topology_rule,
            vote_threshold=vote_threshold,
            temperature=fusion_temperature,
            hidden_dim=fusion_hidden_dim,
        )

        self.linkx = RawNodeEdgeMLPEncoder(
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            node_hidden_dims=node_hidden_dims,
            edge_hidden_dims=edge_hidden_dims,
            branch_emb_dim=branch_emb_dim,
            emb_dim=emb_dim,
            dropout=dropout,
            edge_mode=edge_mode,
            use_upper_triangle=True,
            symmetrize_adj=True,
        )

    def forward(
        self,
        pyg_batch,
        *,
        adj_bank=None,
        topology_bank=None,
    ):
        # lazy imports to avoid circular import
        from pipeline.gnn import (
            _ensure_batch,
            _extract_adj_bank_tensor,
            _extract_topology_bank_tensor,
            _extract_graph_labels,
            _dense_batch_to_pyg,
        )

        batch = _ensure_batch(pyg_batch)

        dense_x, mask = to_dense_batch(
            batch.x,
            batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [B, N, F]

        num_graphs = dense_x.shape[0]

        bank_adj = _extract_adj_bank_tensor(
            batch,
            adj_bank=adj_bank,
            num_graphs=num_graphs,
            num_nodes=self.num_nodes,
        )  # [B, K, N, N]

        bank_topology = _extract_topology_bank_tensor(
            batch,
            topology_bank=topology_bank,
            num_graphs=num_graphs,
            num_nodes=self.num_nodes,
        )  # [B, K, N, N] or None

        fused_adj, fused_topology, fusion_weights = self.bank_fusion(
            dense_node_features=dense_x,
            adj_bank=bank_adj,
            topology_bank=bank_topology,
            mask=mask,
        )  # fused_adj: [B, N, N]

        labels = _extract_graph_labels(batch, num_graphs)

        fused_batch = _dense_batch_to_pyg(
            dense_x,
            fused_adj,
            mask,
            labels=labels,
        )

        graph_emb = self.linkx(fused_batch)

        aux = {
            "fusion_weights": fusion_weights,
            "fused_topology": fused_topology,
            "fused_adjacency": fused_adj,
        }
        return graph_emb, aux        
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


class MultiBranchLinkXEncoder(nn.Module):
    """
    Approach (2):
      - keep each graph candidate separate
      - run one LINKX branch per candidate adjacency
      - fuse candidate embeddings afterward

    Reuses:
      - RawNodeEdgeMLPEncoder
      - helper functions from pipeline.gnn for bank extraction / dense->PyG rebuild
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        num_node_features: int,
        num_candidates: int,
        node_hidden_dims=(256, 128),
        edge_hidden_dims=(128, 64),
        branch_emb_dim: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
        edge_mode: str = "topology_weighted",
        candidate_fusion_mode: str = "concat",   # "mean" | "concat" | "gated"
        fusion_hidden_dim: int = 64,
        fusion_dropout: float = 0.0,
        share_linkx_weights: bool = False,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_candidates = int(num_candidates)
        self.emb_dim = int(emb_dim)
        self.candidate_fusion_mode = str(candidate_fusion_mode).lower()
        self.share_linkx_weights = bool(share_linkx_weights)

        if self.num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}")
        if self.candidate_fusion_mode not in {"mean", "concat", "gated"}:
            raise ValueError(
                f"Unsupported candidate_fusion_mode={candidate_fusion_mode!r}. "
                "Use one of {'mean', 'concat', 'gated'}."
            )

        def _make_branch():
            return RawNodeEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                use_upper_triangle=True,
                symmetrize_adj=True,
            )

        if self.share_linkx_weights:
            self.shared_branch = _make_branch()
            self.branches = None
        else:
            self.shared_branch = None
            self.branches = nn.ModuleList([_make_branch() for _ in range(self.num_candidates)])

        self.fusion_dropout = nn.Dropout(float(fusion_dropout)) if float(fusion_dropout) > 0 else nn.Identity()

        if self.candidate_fusion_mode == "concat":
            self.concat_proj = nn.Linear(self.num_candidates * self.emb_dim, self.emb_dim)
        else:
            self.concat_proj = None

        if self.candidate_fusion_mode == "gated":
            self.gate_mlp = nn.Sequential(
                nn.Linear(self.emb_dim, int(fusion_hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(fusion_hidden_dim), 1),
            )
        else:
            self.gate_mlp = None

    def _get_branch(self, idx: int):
        if self.share_linkx_weights:
            return self.shared_branch
        return self.branches[idx]

    def _fuse_candidate_embeddings(self, cand_embs: list[torch.Tensor]):
        """
        cand_embs: list of [B, D]
        """
        if len(cand_embs) == 0:
            raise ValueError("cand_embs is empty.")
        if len(cand_embs) == 1:
            return cand_embs[0], None

        x = torch.stack(cand_embs, dim=1)   # [B, K, D]

        if self.candidate_fusion_mode == "mean":
            fused = x.mean(dim=1)
            weights = None
            return fused, weights

        if self.candidate_fusion_mode == "concat":
            flat = x.reshape(x.size(0), -1)   # [B, K*D]
            fused = self.concat_proj(flat)
            fused = self.fusion_dropout(fused)
            weights = None
            return fused, weights

        if self.candidate_fusion_mode == "gated":
            scores = self.gate_mlp(x).squeeze(-1)   # [B, K]
            weights = torch.softmax(scores, dim=1)
            fused = (weights.unsqueeze(-1) * x).sum(dim=1)
            fused = self.fusion_dropout(fused)
            return fused, weights

        raise RuntimeError(f"Unhandled candidate_fusion_mode={self.candidate_fusion_mode!r}")

    def forward(
        self,
        pyg_batch,
        *,
        adj_bank=None,
    ):
        # lazy imports to avoid circular imports
        from pipeline.gnn import (
            _ensure_batch,
            _extract_adj_bank_tensor,
            _extract_graph_labels,
            _dense_batch_to_pyg,
        )

        batch = _ensure_batch(pyg_batch)

        dense_x, mask = to_dense_batch(
            batch.x,
            batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [B, N, F]

        num_graphs = dense_x.shape[0]

        bank_adj = _extract_adj_bank_tensor(
            batch,
            adj_bank=adj_bank,
            num_graphs=num_graphs,
            num_nodes=self.num_nodes,
        )  # [B, K, N, N]

        if bank_adj.shape[1] != self.num_candidates:
            raise ValueError(
                f"Expected num_candidates={self.num_candidates}, got bank_adj shape {tuple(bank_adj.shape)}"
            )

        labels = _extract_graph_labels(batch, num_graphs)

        cand_embs = []
        for k in range(self.num_candidates):
            adj_k = bank_adj[:, k]   # [B, N, N]

            pyg_k = _dense_batch_to_pyg(
                dense_x,
                adj_k,
                mask,
                labels=labels,
            )

            branch = self._get_branch(k)
            emb_k = branch(pyg_k)    # [B, D]
            cand_embs.append(emb_k)

        fused_emb, cand_weights = self._fuse_candidate_embeddings(cand_embs)

        aux = {
            "candidate_embeddings": cand_embs,
            "candidate_fusion_weights": cand_weights,
        }
        return fused_emb, aux



class EdgeTokenTransformerEncoder(nn.Module):
    """
    Learn connectivity patterns directly from full weighted adjacency.

    Input:
        conn_stack: [G, B, N, N] or full_adj: [G, N, N]

    It does NOT do message passing over node features.
    It treats edges as tokens:
        edge token = f(weight, node_i, node_j, band)

    Output:
        graph_emb: [G, emb_dim]
    """

    def __init__(
        self,
        num_nodes: int,
        num_bands: int = 5,
        edge_emb_dim: int = 64,
        transformer_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_abs_weight: bool = False,
        use_upper_triangle: bool = True,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.num_bands = int(num_bands)
        self.edge_emb_dim = int(edge_emb_dim)
        self.transformer_dim = int(transformer_dim)
        self.use_abs_weight = bool(use_abs_weight)
        self.use_upper_triangle = bool(use_upper_triangle)

        # Undirected edge list: i < j
        if self.use_upper_triangle:
            edge_i, edge_j = torch.triu_indices(
                self.num_nodes,
                self.num_nodes,
                offset=1,
            )
        else:
            edge_i, edge_j = torch.meshgrid(
                torch.arange(self.num_nodes),
                torch.arange(self.num_nodes),
                indexing="ij",
            )
            edge_i = edge_i.reshape(-1)
            edge_j = edge_j.reshape(-1)
            keep = edge_i != edge_j
            edge_i = edge_i[keep]
            edge_j = edge_j[keep]

        self.register_buffer("edge_i", edge_i.long())
        self.register_buffer("edge_j", edge_j.long())

        # Channel identity embeddings.
        # This lets the model learn Fp1-related, temporal-related, occipital-related patterns.
        self.node_emb = nn.Embedding(self.num_nodes, edge_emb_dim)

        # Band identity embeddings.
        self.band_emb = nn.Embedding(self.num_bands, edge_emb_dim)

        # Edge weight projection.
        # Input has raw weight and abs(weight), useful for signed metrics like Pearson/Spearman.
        self.weight_mlp = nn.Sequential(
            nn.Linear(2, edge_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_emb_dim, edge_emb_dim),
        )

        # Combine weight + node_i + node_j + band.
        self.token_proj = nn.Sequential(
            nn.Linear(4 * edge_emb_dim, transformer_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=4 * transformer_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
        )

        # Attention pooling over edge tokens.
        self.token_attn = nn.Sequential(
            nn.Linear(transformer_dim, transformer_dim // 2),
            nn.Tanh(),
            nn.Linear(transformer_dim // 2, 1),
        )

        self.out_proj = nn.Sequential(
            nn.Linear(transformer_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _prepare_adj(self, conn_or_adj: torch.Tensor) -> torch.Tensor:
        """
        Return adjacency as [G, B, N, N].
        """
        x = conn_or_adj.float()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if x.ndim == 3:
            # [G, N, N] -> [G, 1, N, N]
            x = x.unsqueeze(1)

        if x.ndim != 4:
            raise ValueError(
                f"Expected [G,N,N] or [G,B,N,N], got shape {tuple(x.shape)}"
            )

        G, B, N1, N2 = x.shape

        if N1 != self.num_nodes or N2 != self.num_nodes:
            raise ValueError(
                f"Expected N={self.num_nodes}, got adjacency shape {tuple(x.shape)}"
            )

        if B != self.num_bands:
            if B == 1 and self.num_bands != 1:
                # Allow single-band input if model was accidentally configured as 5 bands?
                # Better to raise to avoid silent mismatch.
                raise ValueError(
                    f"Input has B=1 but model num_bands={self.num_bands}. "
                    "Set num_bands=1 for single adjacency."
                )
            raise ValueError(f"Expected B={self.num_bands}, got B={B}")

        # Symmetrize and remove diagonal.
        x = 0.5 * (x + x.transpose(-1, -2))

        eye = torch.eye(
            self.num_nodes,
            device=x.device,
            dtype=x.dtype,
        ).view(1, 1, self.num_nodes, self.num_nodes)

        x = x * (1.0 - eye)

        if self.use_abs_weight:
            x = x.abs()

        return x

    def forward(self, conn_or_adj: torch.Tensor):
        """
        conn_or_adj:
            [G, B, N, N] for multiband
            or [G, N, N] for single adjacency
        """
        A = self._prepare_adj(conn_or_adj)
        G, B, N, _ = A.shape

        E = self.edge_i.numel()

        # Edge weights: [G, B, E]
        w = A[:, :, self.edge_i, self.edge_j]

        # Edge weight features: [G, B, E, 2]
        w_feat = torch.stack([w, w.abs()], dim=-1)

        # Weight embedding: [G, B, E, D]
        w_emb = self.weight_mlp(w_feat)

        # Node embeddings: [E, D]
        node_i_emb = self.node_emb(self.edge_i)
        node_j_emb = self.node_emb(self.edge_j)

        # Make undirected edge identity stable:
        # edge(i,j) and edge(j,i) should not depend on ordering.
        pair_emb = node_i_emb + node_j_emb

        # Expand to [G, B, E, D]
        pair_emb = pair_emb.view(1, 1, E, self.edge_emb_dim).expand(G, B, E, -1)

        # Band embedding: [B, D] -> [G, B, E, D]
        band_ids = torch.arange(B, device=A.device)
        band_emb = self.band_emb(band_ids)
        band_emb = band_emb.view(1, B, 1, self.edge_emb_dim).expand(G, B, E, -1)

        # Direction/detail embedding:
        # use node_i - node_j also, so the model can distinguish asymmetric channel identities
        # while the adjacency is still symmetrized.
        diff_emb = node_i_emb - node_j_emb
        diff_emb = diff_emb.view(1, 1, E, self.edge_emb_dim).expand(G, B, E, -1)

        token = torch.cat(
            [w_emb, pair_emb, diff_emb, band_emb],
            dim=-1,
        )  # [G, B, E, 4D]

        token = self.token_proj(token)          # [G, B, E, T]
        token = token.reshape(G, B * E, -1)     # [G, B*E, T]

        h = self.transformer(token)             # [G, B*E, T]

        attn_logits = self.token_attn(h).squeeze(-1)  # [G, B*E]
        attn = torch.softmax(attn_logits, dim=1)

        pooled = torch.sum(h * attn.unsqueeze(-1), dim=1)  # [G, T]
        graph_emb = self.out_proj(pooled)                  # [G, emb_dim]

        return graph_emb, attn
class RawNodeEdgeTokenTransformerEncoder(nn.Module):
    """
    Node feature branch + learned edge-token Transformer connectivity branch.

    No message passing.
    No handcrafted graph summaries.
    """

    def __init__(
        self,
        num_nodes: int,
        num_node_features: int,
        num_bands: int = 5,
        node_hidden_dims=(256, 128),
        branch_emb_dim: int = 64,
        edge_transformer_dim: int = 128,
        edge_heads: int = 4,
        edge_layers: int = 2,
        emb_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.num_nodes = int(num_nodes)
        self.num_node_features = int(num_node_features)
        self.num_bands = int(num_bands)

        node_input_dim = num_nodes * num_node_features

        self.node_mlp, node_last_dim = make_mlp(
            node_input_dim,
            node_hidden_dims,
            dropout,
        )

        self.node_proj = nn.Linear(node_last_dim, branch_emb_dim)

        self.edge_encoder = EdgeTokenTransformerEncoder(
            num_nodes=num_nodes,
            num_bands=num_bands,
            edge_emb_dim=branch_emb_dim,
            transformer_dim=edge_transformer_dim,
            num_heads=edge_heads,
            num_layers=edge_layers,
            emb_dim=branch_emb_dim,
            dropout=dropout,
            use_abs_weight=False,
            use_upper_triangle=True,
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * branch_emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, pyg_batch, conn_or_adj):
        dense_x, _ = to_dense_batch(
            pyg_batch.x,
            pyg_batch.batch,
            max_num_nodes=self.num_nodes,
        )  # [G, N, F]

        if dense_x.size(1) != self.num_nodes:
            raise ValueError(
                f"Expected num_nodes={self.num_nodes}, got {dense_x.size(1)}"
            )

        node_x = dense_x.reshape(dense_x.size(0), -1)
        node_h = self.node_mlp(node_x)
        node_emb = self.node_proj(node_h)

        edge_emb, edge_attn = self.edge_encoder(conn_or_adj)

        graph_emb = self.fusion(torch.cat([node_emb, edge_emb], dim=-1))

        return graph_emb, edge_attn
# =========================================================
# Subject-level MIL classifier
# =========================================================
# =========================
# Update SubjectMILClassifier.__init__()
# =========================
class PrototypeClassifier(nn.Module):
    def __init__(self, emb_dim, num_classes):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.01)

    def forward(self, bag_emb):
        z = F.normalize(bag_emb, dim=1)
        p = F.normalize(self.prototypes, dim=1)

        # cosine similarity logits
        logits = torch.matmul(z, p.T) * 10.0
        return logits

class SubjectMILClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",
        num_nodes: Optional[int] = None,

        # shared graph encoder settings
        graph_emb_dim: int = 128,
        dropout: float = 0.2,
        graph_pool: str = "mean",

        # existing GNN settings
        gnn_hidden_dim: int = 64,

        # GraphSAGE settings
        sage_layers: int = 2,

        # GCNII settings
        gcn2_layers: int = 8,
        gcn2_alpha: float = 0.1,
        gcn2_theta: float = 0.5,
        gcn2_shared_weights: bool = True,
        gcn2_use_edge_weight: bool = True,

        # H2GCN settings
        h2gcn_layers: int = 2,

        # raw-MLP params
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        cnn_channels: Sequence[int] = (16, 32),
        # MIL settings
        mil_pool_type: str = "gated",   # "mean" or "gated" or "constrained_weighted_mean"
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
        # cnn_channels: Sequence[int] = (16, 32, 64),
        cnn_num_bands: int = 5,

        num_gnn_layers: int = 2,
        readout_type: str = "mean",
        node_pooling_type: str = "none",
        node_pool_ratio: float = 0.8,
        use_edge_weight: bool = True,
        gat_heads: int = 4,
        readout_hidden_dim: int = 64,
        readout_dropout: float = 0.0,

        graph_backbone: str = "gcn",          # "gcn" | "sage" | "gatv2"
        use_batchnorm: bool = True,
        return_graph_attention_weights: bool = False,
        pool_every_layer: bool = True,
        stage_readout_fusion: str = "concat",

        num_candidates: Optional[int] = None,
        bank_fusion_mode: str = "static",
        bank_topology_rule: str = "union",
        bank_vote_threshold: float = 0.5,
        bank_fusion_temperature: float = 1.0,
        bank_hidden_dim: int = 64,

        candidate_fusion_mode: str = "concat",
        candidate_fusion_hidden_dim: int = 64,
        candidate_fusion_dropout: float = 0.0,
        share_linkx_weights: bool = False,
        use_gcn_norm: bool = False,
        gcn_norm_add_self_loops: bool = True,

        # Prototype-aware MIL settings
        use_prototypes: bool = False,
        num_prototypes: int = 0,
        prototype_emb_dim: int = 16,
        prototype_hidden_dim: int = 64,
        prototype_use_soft: bool = True,
        prototype_use_dist: bool = True,

        gcn_normalize_input: bool = False,
        gcn_norm_abs_weights: bool = False,
        gcn_norm_abs_degree: bool = False,
        use_prototype_classifier: bool = False,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()
        self.use_prototypes = bool(use_prototypes)
        self.num_prototypes = int(num_prototypes)
        self.prototype_use_soft = bool(prototype_use_soft)
        self.prototype_use_dist = bool(prototype_use_dist)
        self.gcn_normalize_input = bool(gcn_normalize_input)
        self.gcn_norm_add_self_loops = bool(gcn_norm_add_self_loops)
        self.gcn_norm_abs_weights = bool(gcn_norm_abs_weights)
        self.gcn_norm_abs_degree = bool(gcn_norm_abs_degree)
        self.use_prototype_classifier = bool(use_prototype_classifier)
        self.prototype_classifier = PrototypeClassifier(graph_emb_dim, num_classes)
        
        if self.use_prototypes:
            if self.num_prototypes <= 0:
                raise ValueError("num_prototypes must be > 0 when use_prototypes=True")

            self.prototype_embedding = nn.Embedding(
                self.num_prototypes,
                prototype_emb_dim,
            )

            proto_input_dim = prototype_emb_dim

            if self.prototype_use_soft:
                proto_input_dim += self.num_prototypes

            if self.prototype_use_dist:
                proto_input_dim += self.num_prototypes

            self.prototype_mlp = nn.Sequential(
                nn.Linear(proto_input_dim, prototype_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(prototype_hidden_dim, prototype_emb_dim),
                nn.ReLU(),
            )

            # Keep graph_emb_dim unchanged, so the existing MIL pool and classifier still work.
            self.prototype_fusion = nn.Sequential(
                nn.Linear(graph_emb_dim + prototype_emb_dim, graph_emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.prototype_embedding = None
            self.prototype_mlp = None
            self.prototype_fusion = None

        self.num_gnn_layers = int(num_gnn_layers)
        self.readout_type = str(readout_type)
        self.node_pooling_type = str(node_pooling_type)
        self.node_pool_ratio = float(node_pool_ratio)
        self.use_edge_weight = bool(use_edge_weight)
        self.gat_heads = int(gat_heads)
        self.readout_hidden_dim = int(readout_hidden_dim)
        self.readout_dropout = float(readout_dropout)

        self.graph_backbone = str(graph_backbone).lower()
        self.use_batchnorm = bool(use_batchnorm)
        self.return_graph_attention_weights = bool(return_graph_attention_weights)


        self.pool_every_layer = bool(pool_every_layer)
        self.stage_readout_fusion = str(stage_readout_fusion).lower()
        if self.encoder_type == "sage":
            self.graph_encoder = GraphSAGEEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=sage_layers,
                dropout=dropout,
                pool=graph_pool,
                jk_mode="last",
            )

        elif self.encoder_type == "linkx_bank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_bank'")
            if num_candidates is None or int(num_candidates) < 1:
                raise ValueError("num_candidates must be provided and >= 1 for encoder_type='linkx_bank'")

            self.graph_encoder = MultiBranchLinkXEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=int(num_candidates),
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                candidate_fusion_mode=candidate_fusion_mode,
                fusion_hidden_dim=candidate_fusion_hidden_dim,
                fusion_dropout=candidate_fusion_dropout,
                share_linkx_weights=share_linkx_weights,
            )
        elif self.encoder_type == "linkx_fused_bank":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_fused_bank'")
            if num_candidates is None or int(num_candidates) < 1:
                raise ValueError("num_candidates must be provided and >= 1 for encoder_type='linkx_fused_bank'")

            self.graph_encoder = FusedBankLinkXEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_candidates=int(num_candidates),
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                edge_mode=edge_mode,
                bank_fusion_mode=bank_fusion_mode,
                topology_rule=bank_topology_rule,
                vote_threshold=bank_vote_threshold,
                fusion_temperature=bank_fusion_temperature,
                fusion_hidden_dim=bank_hidden_dim,
            )
        elif self.encoder_type == "gnn_block":

            from pipeline.gnn import GraphEncoderBlock
            self.graph_encoder = GraphEncoderBlock(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=num_gnn_layers,
                backbone=self.graph_backbone,
                dropout=dropout,
                gat_heads=gat_heads,
                use_edge_weight=use_edge_weight,
                use_batchnorm=use_batchnorm,
                node_pooling_type=node_pooling_type,
                node_pool_ratio=node_pool_ratio,
                readout_type=readout_type,
                readout_hidden_dim=readout_hidden_dim,
                readout_dropout=readout_dropout,
                return_attention_weights=return_graph_attention_weights,
            )

        elif self.encoder_type == "hier_gnn_block":
            from pipeline.gnn import ProgressiveGraphEncoderBlock

            self.graph_encoder = ProgressiveGraphEncoderBlock(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=num_gnn_layers,
                backbone=self.graph_backbone,
                dropout=dropout,
                gat_heads=gat_heads,
                use_edge_weight=use_edge_weight,
                use_batchnorm=use_batchnorm,
                node_pooling_type=node_pooling_type,
                node_pool_ratio=node_pool_ratio,
                readout_type=readout_type,
                readout_hidden_dim=readout_hidden_dim,
                readout_dropout=readout_dropout,
                return_attention_weights=return_graph_attention_weights,
                pool_every_layer=pool_every_layer,
                stage_readout_fusion=stage_readout_fusion,
            )
        elif self.encoder_type == "linkx_cnn":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_cnn'")

            self.graph_encoder = RawNodeAdjCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                cnn_channels=cnn_channels,
                dropout=dropout,
                symmetrize_adj=True,
                zero_diagonal=False,
            )
        elif self.encoder_type in {"linkx_cnn5", "linkx_cnn_bank"}:
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx_cnn5'")

            self.graph_encoder = RawNodeMultiBandCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                num_bands=cnn_num_bands,
                symmetrize_adj=True,
                zero_diagonal=False,
            )

        elif self.encoder_type in {"cnn5", "cnn_bank"}:
            self.graph_encoder = MultiBandCNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                num_bands=cnn_num_bands,
                symmetrize_adj=True,
                zero_diagonal=False,
                )
        elif self.encoder_type == "gcn2":
            self.graph_encoder = GCNIIEncoder(
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

        elif self.encoder_type == "h2gcn":
            self.graph_encoder = H2GCNLikeEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=h2gcn_layers,
                dropout=dropout,
                pool=graph_pool,
            )

        elif self.encoder_type == "gnn":
            self.graph_encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif self.encoder_type == "gat":
            self.graph_encoder = GNNEncoder_GAT(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                num_layers=3,
                dropout=dropout,
                heads=gat_heads,
                edge_dim=1,
                pooling=graph_pool
            )

        elif self.encoder_type == "hybrid":
            self.graph_encoder = HybridGNNEncoder(
                 in_channels=num_node_features, 
                 hidden_channels=gnn_hidden_dim, 
                 emb_dim=graph_emb_dim,
                 gat_layers=num_gnn_layers//2, 
                 cheb_layers=num_gnn_layers//2,
                 dropout=dropout, 
                 heads=gat_heads, 
                 edge_dim=1,
                 pooling=graph_pool)


        elif self.encoder_type == "linkx":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='linkx'")

            self.graph_encoder = RawNodeEdgeMLPEncoder(
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
        elif self.encoder_type == "mlp_node":
            self.graph_encoder = RawNodeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                proj_dim = branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout)

        elif self.encoder_type in ["gnn_bank"]:
            self.graph_encoder = RawNodeGraphBankGNNEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                graph_emb_dim=graph_emb_dim,
                use_gcn_norm=False,
                gcn_norm_add_self_loops=False,
                branch_emb_dim=branch_emb_dim,
                node_hidden_dims=node_hidden_dims,
                gnn_hidden_dim=gnn_hidden_dim,
                gnn_out_dim=gnn_hidden_dim,
                gnn_layers=num_gnn_layers,
                backbone=graph_backbone,        # "gcn", "sage", or "gatv2"
                readout=graph_pool,
                fusion=candidate_fusion_mode,
                dropout=dropout,
                gat_heads=gat_heads,
                adj_value_mode="abs",    # good default for correlation-like adjacency
                symmetrize_adj=True,
                zero_diagonal=True,
                use_edge_weight=True,

            )
        elif self.encoder_type == "edge_token":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='edge_token'")

            self.graph_encoder = RawNodeEdgeTokenTransformerEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                num_bands=cnn_num_bands,
                node_hidden_dims=node_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                edge_transformer_dim=64,
                edge_heads=4,
                edge_layers=2,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type='{encoder_type}'. "
                f"Choose from ['gnn','hybrid', 'gat','linkx', 'cnn5', 'linkx_cnn5', 'mlp_node', 'sage', 'gcn2', 'h2gcn']"
            )

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )

        # elif self.mil_pool_type in ["constrained_weighted_mean", "cwmean"]:
        #     self.mil_pool = ConstrainedWeightedMeanMIL(
        #         in_dim=graph_emb_dim,
        #         attn_dim=attn_dim,
        #         dropout=dropout,
        #         temperature=2.0,
        #         gamma_max=0.6,
        #         min_effective_frac=0.35,
        #         min_entropy=0.75,
        #         lambda_entropy=0.01,
        #         lambda_effective=0.01,
        #         lambda_max_weight=0.01,
        #         segment_dropout=0.0,
        #     )

        else:
            raise ValueError(f"Unknown mil_pool_type='{mil_pool_type}'")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )
    def _get_prototype_tensors(self, batch_dict: Dict):
        """
        Recover prototype tensors from PyG Batch.

        Expected attached graph fields:
            g.proto_id       -> batched as [G]
            g.proto_soft     -> batched as [G, K]
            g.proto_dist_log -> batched as [G, K]
        """
        pyg_batch = batch_dict["pyg_batch"]

        if not hasattr(pyg_batch, "proto_id"):
            raise KeyError(
                "pyg_batch is missing proto_id. "
                "Run attach_segment_prototypes(...) before creating the dataset/dataloader."
            )

        proto_id = pyg_batch.proto_id.view(-1).long()

        proto_soft = getattr(pyg_batch, "proto_soft", None)
        proto_dist_log = getattr(pyg_batch, "proto_dist_log", None)

        if proto_soft is not None:
            proto_soft = proto_soft.float()
            if proto_soft.dim() == 1:
                proto_soft = proto_soft.view(-1, self.num_prototypes)

        if proto_dist_log is not None:
            proto_dist_log = proto_dist_log.float()
            if proto_dist_log.dim() == 1:
                proto_dist_log = proto_dist_log.view(-1, self.num_prototypes)

        return proto_id, proto_soft, proto_dist_log


    def _fuse_prototypes(self, graph_emb: torch.Tensor, batch_dict: Dict) -> torch.Tensor:
        """
        Fuse graph embedding with prototype context.

        graph_emb: [G, graph_emb_dim]
        """
        if not self.use_prototypes:
            return graph_emb

        proto_id, proto_soft, proto_dist_log = self._get_prototype_tensors(batch_dict)

        proto_id = proto_id.to(graph_emb.device)

        parts = [
            self.prototype_embedding(proto_id)
        ]

        if self.prototype_use_soft:
            if proto_soft is None:
                raise KeyError("prototype_use_soft=True, but proto_soft is missing.")
            parts.append(proto_soft.to(graph_emb.device))

        if self.prototype_use_dist:
            if proto_dist_log is None:
                raise KeyError("prototype_use_dist=True, but proto_dist_log is missing.")
            parts.append(proto_dist_log.to(graph_emb.device))

        proto_input = torch.cat(parts, dim=-1)
        proto_context = self.prototype_mlp(proto_input)

        graph_emb = self.prototype_fusion(
            torch.cat([graph_emb, proto_context], dim=-1)
        )

        return graph_emb
    # def _encode_graphs(self, batch_dict):
    #     pyg_batch = batch_dict["pyg_batch"]

    #     if self.encoder_type == "linkx_cnn5":
    #         return self.graph_encoder(pyg_batch, batch_dict["conn_stack"])
    #     elif self.encoder_type == "linkx_cnn":
    #         return self.graph_encoder(pyg_batch, batch_dict["full_adj"])
    #     else:
    #         return self.graph_encoder(pyg_batch)
    def _run_graph_encoder(self, batch_dict):
        pyg_batch = batch_dict["pyg_batch"]

        if self.encoder_type == "linkx_cnn":
            out = self.graph_encoder(pyg_batch, batch_dict["full_adj"])
            return out, None

        elif self.encoder_type in ["linkx_cnn5"]:
            out = self.graph_encoder(pyg_batch, batch_dict["conn_stack"])
            return out, None

        elif self.encoder_type == "linkx_fused_bank":
            out = self.graph_encoder(
                pyg_batch,
                adj_bank=batch_dict.get("adj_bank", None),
                topology_bank=batch_dict.get("topology_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["linkx_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                adj_bank=batch_dict.get("adj_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["cnn5", "cnn_bank", "linkx_cnn_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                conn_stack=batch_dict.get("conn_stack", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None

        elif self.encoder_type in ["gnn_bank", "gatv2_bank", "gcn_bank", "sage_bank"]:
            out = self.graph_encoder(
                batch_dict["pyg_batch"],
                adj_bank=batch_dict.get("adj_bank", None),
            )
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
            return out, None
        elif self.encoder_type == "edge_token":
            if "conn_stack" in batch_dict:
                out = self.graph_encoder(
                    pyg_batch,
                    batch_dict["conn_stack"],
                )
            elif "full_adj" in batch_dict:
                out = self.graph_encoder(
                    pyg_batch,
                    batch_dict["full_adj"],
                )
            else:
                raise KeyError(
                    "edge_token encoder needs batch_dict['conn_stack'] or batch_dict['full_adj']"
                )



        out = self.graph_encoder(pyg_batch)

        if isinstance(out, tuple) and len(out) == 2:
            graph_emb, graph_attn = out
            return graph_emb, graph_attn

        return out, None


    def forward(self, batch_dict: Dict):
        # graph_emb = self._encode_graphs(batch_dict)
        if self.gcn_normalize_input:
            batch_dict = apply_gcn_norm_to_batch_dict(
                batch_dict,
                add_self_loops=self.gcn_norm_add_self_loops,
                use_abs_edge_weight=self.gcn_norm_abs_weights,
                use_abs_degree=self.gcn_norm_abs_degree,
            )
        graph_emb, graph_attn = self._run_graph_encoder(batch_dict)
        
        # NEW: prototype-aware segment embedding
        graph_emb = self._fuse_prototypes(graph_emb, batch_dict)

        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        if self.use_prototype_classifier:
            logits = self.prototype_classifier(bag_emb)
        else:
            logits = self.classifier(bag_emb)

        # return {
        #     "logits": logits,
        #     "bag_emb": bag_emb,
        #     "graph_emb": graph_emb,
        #     "attn_list": attn_list,
        #     "graph_attention_weights": graph_attn,
        # }


        out = {
            "graph_emb": graph_emb,
            "bag_emb": bag_emb,
            "logits": logits,
            "attn_list": attn_list,
        }

        if self.encoder_type == "edge_token":
            out["edge_attn"] = edge_attn

        if graph_attn is not None:
            out["graph_attn"] = graph_attn

            # convenience aliases for common encoder outputs
            if isinstance(graph_attn, dict):
                if "stage_readout_attention" in graph_attn:
                    out["stage_readout_attention"] = graph_attn["stage_readout_attention"]
                if "graph_attention_weights" in graph_attn:
                    out["graph_attention_weights"] = graph_attn["graph_attention_weights"]
                if "fusion_weights" in graph_attn:
                    out["fusion_weights"] = graph_attn["fusion_weights"]
                if "fused_adjacency" in graph_attn:
                    out["fused_adjacency"] = graph_attn["fused_adjacency"]
                if "fused_topology" in graph_attn:
                    out["fused_topology"] = graph_attn["fused_topology"]
                if "candidate_fusion_weights" in graph_attn:
                    out["candidate_fusion_weights"] = graph_attn["candidate_fusion_weights"]
                if "candidate_embeddings" in graph_attn:
                    out["candidate_embeddings"] = graph_attn["candidate_embeddings"]
        
        if self.use_prototypes:
            pyg_batch = batch_dict["pyg_batch"]
            out["proto_id"] = pyg_batch.proto_id.detach().cpu()
            if hasattr(pyg_batch, "proto_soft"):
                out["proto_soft"] = pyg_batch.proto_soft.detach().cpu()

        if hasattr(self.mil_pool, "last_reg_loss") and self.mil_pool.last_reg_loss is not None:
            out["reg_loss"] = self.mil_pool.last_reg_loss

        if hasattr(self.mil_pool, "last_diagnostics"):
            out["pool_diagnostics"] = self.mil_pool.last_diagnostics



        return out

    def forward_with_embeddings(self, batch_dict: Dict):
        """
        Same as forward(...), but explicit name for analysis code.
        """
        return self.forward(batch_dict)

def fit_mil_baseline(
    model,
    train_loader, 
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    save_path=None,
    start_epoch=0,
    min_delta=0.0,
    top_k=10,
    use_center_loss=False,
    center_loss_fn=None,
    verbose=True,
    use_lr_scheduler=True,
    lr_scheduler_metric="val_loss",      # "val_loss", "val_bal_acc", or "val_macro_f1"
    lr_scheduler_factor=0.5,
    lr_scheduler_patience=10,
    lr_scheduler_min_lr=1e-6,
    lr_scheduler_threshold=1e-4,
    lr_scheduler_cooldown=0,
    lr_scheduler_start_epoch=None,       # None => use start_epoch
    use_grad_norm_report=False,
    use_contrastive_loss= False,
    use_soft_targets=False
):
    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    def _get_optimizer_lrs(optimizer):
        """Return current learning rates for all optimizer parameter groups."""
        return [float(group["lr"]) for group in optimizer.param_groups]


    def _infer_plateau_mode(metric_name: str) -> str:
        """
        ReduceLROnPlateau needs:
          - mode="min" for loss/error
          - mode="max" for accuracy/F1/AUC/etc.
        """
        name = str(metric_name).lower()
        if "loss" in name or "error" in name:
            return "min"
        return "max"
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)
        ckpt_prefix = os.path.splitext(os.path.basename(save_path))[0]
    else:
        save_dir = None
        ckpt_prefix = "mil_checkpoint"

    early_stopper = EarlyStopping(
        patience=patience,
        start_epoch=start_epoch,
        min_delta=min_delta,
        top_k=top_k,
        save_dir=save_dir,
        verbose=verbose,
        file_prefix=f"{ckpt_prefix}_topk",
    )

    scheduler = None
    if use_lr_scheduler:
        if lr_scheduler_start_epoch is None:
            lr_scheduler_start_epoch = start_epoch

        scheduler_mode = _infer_plateau_mode(lr_scheduler_metric)

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=float(lr_scheduler_factor),
            patience=int(lr_scheduler_patience),
            threshold=float(lr_scheduler_threshold),
            threshold_mode="abs",
            cooldown=int(lr_scheduler_cooldown),
            min_lr=float(lr_scheduler_min_lr),
        )

        if verbose:
            print(
                "[LR Scheduler] ReduceLROnPlateau enabled | "
                f"metric={lr_scheduler_metric} | "
                f"mode={scheduler_mode} | "
                f"factor={lr_scheduler_factor} | "
                f"patience={lr_scheduler_patience} | "
                f"min_lr={lr_scheduler_min_lr} | "
                f"start_epoch={lr_scheduler_start_epoch}"
            )

    for epoch in range(1, epochs + 1):
        # important for deterministic-but-changing segment sampling
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

        lrs_before = _get_optimizer_lrs(optimizer)
        if use_contrastive_loss:
            train_metrics = train_one_epoch_subject_invariant(
                            model,
                            train_loader,
                            optimizer,
                            criterion,
                            device,
                            lambda_supcon=0.005,
                            supcon_temperature=0.2,
                        )
        else:
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, use_soft_targets, use_grad_norm_report, center_loss_fn, use_center_loss)
        val_metrics = evaluate(model, val_loader, criterion, device)

        val_pred_counts = np.bincount(np.asarray(val_metrics["y_pred"], dtype=int), minlength=3)
        train_pred_counts = np.bincount(np.asarray(train_metrics["y_pred"], dtype=int), minlength=3)

        row = {
            "epoch": int(epoch),

            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["accuracy"]),
            "train_bal_acc": float(train_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),

            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_bal_acc": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),

            "lr": float(lrs_before[0]),
            "all_lrs": lrs_before,
        }

        scheduler_stepped = False
        scheduler_monitor_value = None

        if scheduler is not None and epoch >= int(lr_scheduler_start_epoch):
            if lr_scheduler_metric not in row:
                raise KeyError(
                    f"lr_scheduler_metric={lr_scheduler_metric!r} not found in history row. "
                    f"Available keys: {sorted(row.keys())}"
                )

            scheduler_monitor_value = float(row[lr_scheduler_metric])
            scheduler.step(scheduler_monitor_value)
            scheduler_stepped = True

        lrs_after = _get_optimizer_lrs(optimizer)

        row["lr_after_scheduler"] = float(lrs_after[0])
        row["all_lrs_after_scheduler"] = lrs_after
        row["lr_scheduler_metric"] = lr_scheduler_metric if scheduler is not None else None
        row["lr_scheduler_monitor_value"] = scheduler_monitor_value
        row["lr_scheduler_stepped"] = bool(scheduler_stepped)
        if use_center_loss:
            row["train_center_loss"] = train_metrics.get("center_loss")
            row["train_ce_loss"] = train_metrics.get("ce_loss")

        if use_contrastive_loss:
            row["train_ce_loss"] = train_metrics.get("ce_loss")
            row["train_contrastive_loss"] = train_metrics.get("supcon_loss")
        history.append(row)
        if epoch % 5 == 0:
            print(
                f"Epoch [{epoch:03d}/{epochs}] | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Train Bal Acc: {train_metrics['balanced_accuracy']:.4f} | "
                f"Train F1: {train_metrics['macro_f1']:.4f} || "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f} ||"
                f"LR: {lrs_before[0]:.6g} -> {lrs_after[0]:.6g}"
            )
            print("Train pred counts:", train_pred_counts)
            print("Val pred counts:", val_pred_counts)

        should_stop = early_stopper(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_metrics["loss"],
            val_bal_acc=val_metrics["balanced_accuracy"],
            val_macro_f1=val_metrics["macro_f1"],
            extra_state={
                "history": copy.deepcopy(history),
                "lr_scheduler": {
                    "enabled": bool(use_lr_scheduler),
                    "metric": lr_scheduler_metric,
                    "factor": lr_scheduler_factor,
                    "patience": lr_scheduler_patience,
                    "min_lr": lr_scheduler_min_lr,
                    "threshold": lr_scheduler_threshold,
                    "cooldown": lr_scheduler_cooldown,
                    "start_epoch": lr_scheduler_start_epoch,
                },
            },
        )

        if should_stop:
            break

    best_meta = early_stopper.get_best_checkpoint()
    best_state = None

    if best_meta is not None and best_meta.get("path") is not None:
        best_state = torch.load(best_meta["path"], map_location=device)

        # enrich returned state
        best_state["top_k_checkpoints"] = early_stopper.get_topk_checkpoints()
        best_state["selected_checkpoint"] = copy.deepcopy(best_meta)
        best_state["selected_by"] = [
            "max val_bal_acc",
            "max val_macro_f1",
            "min val_loss",
        ]

        best_state["lr_scheduler_config"] = {
            "enabled": bool(use_lr_scheduler),
            "metric": lr_scheduler_metric,
            "factor": lr_scheduler_factor,
            "patience": lr_scheduler_patience,
            "min_lr": lr_scheduler_min_lr,
            "threshold": lr_scheduler_threshold,
            "cooldown": lr_scheduler_cooldown,
            "start_epoch": lr_scheduler_start_epoch,
        }
        # restore selected weights into current model
        model.load_state_dict(best_state["model_state_dict"])

        # keep old downstream pattern working:
        # write final selected checkpoint back to the original save_path
        if save_path is not None:
            torch.save(best_state, save_path)
            if verbose:
                print(f"Saved final selected checkpoint to: {save_path}")

    elif verbose:
        print("Warning: no checkpoint was selected. Model weights were not restored from disk.")

    final_val_metrics = evaluate(model, val_loader, criterion, device)
    return model, final_val_metrics, history, best_state


def result_already_exists(save_root, check_term, final_csv_name):
    for d in os.listdir(save_root):
        path = os.path.join(save_root, d)
        if os.path.isdir(path) and check_term in os.path.basename(path):
            final_csv_path = os.path.join(path, final_csv_name)
            if os.path.isfile(final_csv_path):
                return True
    return False



import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple
import h5py
import numpy as np


# =========================================================
# Basic H5 helpers
# =========================================================

def _safe_subject_id(subject_id: Any) -> str:
    return str(subject_id).replace("/", "__")


def _decode_str_list(arr) -> List[str]:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def _iter_existing_subject_ids(h5f: h5py.File, subject_ids: Optional[Sequence[str]]) -> List[str]:
    all_ids = list(h5f["subjects"].keys())
    if subject_ids is None:
        return all_ids
    wanted = [_safe_subject_id(sid) for sid in subject_ids]
    return [sid for sid in wanted if sid in all_ids]


def _require_3d_feature_tensor(x: np.ndarray, name: str = "feature tensor") -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [num_windows, num_channels, num_features], got {x.shape}")
    return x


# =========================================================
# H5 payload loader
# =========================================================

def load_h5_payload_for_subjects(
    h5_path: str,
    subject_ids: Optional[Sequence[str]] = None,
    feature_families: Optional[Sequence[str]] = None,
    connectivity_metrics: Optional[Sequence[str]] = None,
    connectivity_band: Optional[int | str] = None,
    load_raw_for_alignment: bool = False,
    load_bad_segment_flag: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Load a subject-indexed payload from the master H5.

    Returns
    -------
    payload : dict
        payload[sid] = {
            "label": int,
            "segment_id": np.ndarray [W],
            "start_sample": np.ndarray [W],
            "end_sample": np.ndarray [W],
            "channel_names": list[str],
            "features": {family: np.ndarray [W, C, F_family]},
            "connectivity": {metric: np.ndarray [W, C, C] or [W, B, C, C]},
            "raw_eeg": None or np.ndarray [W, C, T],
            "bad_segment_flag": optional np.ndarray [W],
            "aligned_raw_eeg": None,
            "aligned_adj": None,
        }
    """
    payload: Dict[str, Dict[str, Any]] = {}

    with h5py.File(h5_path, "r") as h5f:
        for sid in _iter_existing_subject_ids(h5f, subject_ids):
            if sid in {"train_00587", "train_00781", "train_01301"}:
                continue
            grp = h5f[f"subjects/{sid}"]

            # print("feature keys:", list(grp["windows/features"].keys()))
            # print("connectivity keys:", list(grp["windows/connectivity"].keys()))

            entry: Dict[str, Any] = {
                "label": int(grp["metadata"].attrs["label"]),
                "segment_id": grp["windows/raw/segment_id"][:].astype(np.int64),
                "start_sample": grp["windows/raw/start_sample"][:].astype(np.int64),
                "end_sample": grp["windows/raw/end_sample"][:].astype(np.int64),
                "channel_names": _decode_str_list(grp["metadata/channel_names"][:]),
                "features": {},
                "connectivity": {},
                "raw_eeg": None,
                "aligned_raw_eeg": None,
                "aligned_adj": None,
            }
            # print("feature_families", feature_families)
            for family in feature_families or []:
                # print(family)
                x = np.asarray(grp[f"windows/features/{family}"][:], dtype=np.float32)
                entry["features"][family] = x

            for metric in connectivity_metrics or []:
                adj = np.asarray(grp[f"windows/connectivity/{metric}"][:], dtype=np.float32)

                # Optional band slicing for banded connectivity:
                # stored shape is often [W, B, C, C]
                if connectivity_band is not None and adj.ndim == 4:
                    ds = grp[f"windows/connectivity/{metric}"]
                    band_names = list(ds.attrs.get("band_names", []))
                    if isinstance(connectivity_band, str):
                        decoded_band_names = _decode_str_list(band_names)
                        if connectivity_band not in decoded_band_names:
                            raise KeyError(
                                f"Band '{connectivity_band}' not found for metric '{metric}'. "
                                f"Available: {decoded_band_names}"
                            )
                        band_idx = decoded_band_names.index(connectivity_band)
                    else:
                        band_idx = int(connectivity_band)
                    adj = adj[:, band_idx]  # -> [W, C, C]

                entry["connectivity"][metric] = adj

            if load_raw_for_alignment:
                entry["raw_eeg"] = np.asarray(grp["windows/raw/eeg"][:], dtype=np.float32)

            if load_bad_segment_flag:
                qpath = "windows/qc/bad_segment_flag"
                if qpath in grp:
                    entry["bad_segment_flag"] = grp[qpath][:].astype(np.int64)

            payload[sid] = entry

    return payload


# =========================================================
# Feature normalization helpers
# =========================================================

def _safe_std(std: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    std = np.asarray(std, dtype=np.float32)
    return np.where(std < eps, 1.0, std).astype(np.float32)


def subject_wise_feature_zscore(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Subject-wise pooled normalization.

    Input
    -----
    x : np.ndarray [W, C, F]

    Compute one mean/std per feature dimension using all windows and all channels
    of the same subject.

    Output
    ------
    x_norm : np.ndarray [W, C, F]
    stats  : {"mean": [F], "std": [F]}
    """
    x = _require_3d_feature_tensor(x, "x")
    flat = x.reshape(-1, x.shape[-1])  # [W*C, F]

    mean = flat.mean(axis=0, keepdims=True).astype(np.float32)   # [1, F]
    std = _safe_std(flat.std(axis=0, keepdims=True), eps=eps)    # [1, F]

    x_norm = ((flat - mean) / std).reshape(x.shape).astype(np.float32)

    stats = {
        "mean": mean.squeeze(0),   # [F]
        "std": std.squeeze(0),     # [F]
    }
    return x_norm, stats


def channel_wise_subject_feature_zscore(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Channel-wise subject normalization.

    Input
    -----
    x : np.ndarray [W, C, F]

    Compute one mean/std per (channel, feature) using all windows of the same subject.

    Output
    ------
    x_norm : np.ndarray [W, C, F]
    stats  : {"mean": [C, F], "std": [C, F]}
    """
    x = _require_3d_feature_tensor(x, "x")

    mean = x.mean(axis=0).astype(np.float32)   # [C, F]
    std = _safe_std(x.std(axis=0), eps=eps)    # [C, F]

    x_norm = ((x - mean[None, :, :]) / std[None, :, :]).astype(np.float32)

    stats = {
        "mean": mean,   # [C, F]
        "std": std,     # [C, F]
    }
    return x_norm, stats


def normalize_one_feature_tensor(
    x: np.ndarray,
    norm_mode: str,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Dispatch helper for one feature tensor [W, C, F].

    norm_mode:
        - "subject_wise"
        - "channel_wise"
        - "none"
    """
    norm_mode = str(norm_mode).lower()

    if norm_mode == "none":
        x = _require_3d_feature_tensor(x, "x").astype(np.float32, copy=False)
        stats = {
            "mean": np.zeros((x.shape[-1],), dtype=np.float32),
            "std": np.ones((x.shape[-1],), dtype=np.float32),
        }
        return x, stats

    if norm_mode == "subject_wise":
        return subject_wise_feature_zscore(x, eps=eps)

    if norm_mode == "channel_wise":
        return channel_wise_subject_feature_zscore(x, eps=eps)

    raise ValueError(f"Unsupported norm_mode={norm_mode!r}")


# =========================================================
# Payload-level normalization
# =========================================================

def normalize_payload_feature_families(
    payload: Dict[str, Dict[str, Any]],
    feature_families: Sequence[str],
    norm_mode: str,
    *,
    in_place: bool = False,
    eps: float = 1e-8,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    """
    Normalize selected feature families for every subject independently.

    This is leak-safe for subject-wise and channel-wise modes because it uses only
    each subject's own feature tensors.

    Returns
    -------
    payload_norm : same nested structure as input payload
    norm_stats   : norm_stats[sid][family] = {"mean": ..., "std": ...}
    """
    if not in_place:
        payload_norm = copy.deepcopy(payload)
    else:
        payload_norm = payload

    norm_stats: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for sid, subj in payload_norm.items():
        norm_stats[sid] = {}

        if "features" not in subj:
            raise KeyError(f"payload[{sid!r}] is missing 'features'")

        for family in feature_families:
            if family not in subj["features"]:
                raise KeyError(f"payload[{sid!r}]['features'] is missing family {family!r}")

            x = subj["features"][family]  # [W, C, F_family]
            x_norm, stats = normalize_one_feature_tensor(
                x,
                norm_mode=norm_mode,
                eps=eps,
            )

            subj["features"][family] = x_norm
            norm_stats[sid][family] = stats

    return payload_norm, norm_stats


# =========================================================
# Concatenate selected feature families into node features
# =========================================================

def assemble_subject_node_features(
    subject_entry: Dict[str, Any],
    feature_families: Sequence[str],
) -> np.ndarray:
    """
    Concatenate selected families along the last dimension.

    Input:
        subject_entry["features"][family] -> [W, C, F_family]

    Output:
        node_features -> [W, C, F_total]
    """
    feat_list = []
    ref_shape = None

    for family in feature_families:
        if family not in subject_entry["features"]:
            raise KeyError(f"Missing feature family {family!r}")

        x = _require_3d_feature_tensor(subject_entry["features"][family], family)

        if ref_shape is None:
            ref_shape = x.shape[:2]  # (W, C)
        elif x.shape[:2] != ref_shape:
            raise ValueError(
                f"All feature families must share the same [W, C]. "
                f"Got {x.shape[:2]} vs {ref_shape} for family {family!r}"
            )

        feat_list.append(x)

    if len(feat_list) == 0:
        raise ValueError("feature_families is empty")

    return np.concatenate(feat_list, axis=-1).astype(np.float32)


# =========================================================
# Optional helper: remove bad windows consistently
# =========================================================

def filter_payload_bad_windows_in_place(payload: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Drop windows where bad_segment_flag == 1.

    This keeps all stored arrays aligned by window index.
    """
    for sid, subj in payload.items():
        if "bad_segment_flag" not in subj:
            continue

        bad = np.asarray(subj["bad_segment_flag"]).astype(bool)
        keep = ~bad

        subj["segment_id"] = subj["segment_id"][keep]
        subj["start_sample"] = subj["start_sample"][keep]
        subj["end_sample"] = subj["end_sample"][keep]
        subj["bad_segment_flag"] = subj["bad_segment_flag"][keep]

        if subj.get("raw_eeg", None) is not None:
            subj["raw_eeg"] = subj["raw_eeg"][keep]

        if subj.get("aligned_raw_eeg", None) is not None:
            subj["aligned_raw_eeg"] = subj["aligned_raw_eeg"][keep]

        if subj.get("aligned_adj", None) is not None:
            subj["aligned_adj"] = subj["aligned_adj"][keep]

        for family in list(subj["features"].keys()):
            subj["features"][family] = subj["features"][family][keep]

        for metric in list(subj["connectivity"].keys()):
            subj["connectivity"][metric] = subj["connectivity"][metric][keep]

    return payload
if __name__ == "__main__":

    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    class_set ="all3" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # data_paths, labels, sub_id_list = data_paths[:15]+data_paths[40:55]+data_paths[75:], labels[:15]+labels[40:55]+labels[75:], sub_id_list[:15]+sub_id_list[40:55]+sub_id_list[75:]
    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"

    k = 5
    val_ratio = 0.15
    # split_seeds = [15]
    split_seeds = [15, 42, 100]
    batch_size_train=4
    batch_size_val=4
    batch_size_test = 4
    lr=3e-4
    weight_decay=5e-4
    epochs=100
    patience=30

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--all_data_path", type=str, required=True, help="all_data_path")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--base_k", type=int, default=None, required=False, help="base_k")
    parser.add_argument("--dim", type=int,  default=32, required=False, help="dim")
    parser.add_argument("--feature_families", type=str, required=True)   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="pli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="gnn",
        # choices=["gnn", "LINKX", "linkx_cnn", "mlp_node", "sage", "gcn2", "h2gcn"]
        choices=["gnn", "LINKX", "linkx_cnn", "cnn5", "linkx_cnn5", "mlp_node", "sage", "gcn2", "h2gcn"]
        # choices=["gnn", "LINKX", "mlp_node", "sage", "gcn2", "h2gcn"],
    )
    parser.add_argument("--graph_pool", type=str, default="mean", choices=["mean", "max", "add"])
    parser.add_argument("--sage_layers", type=int, default=2)
    parser.add_argument("--gcn2_layers", type=int, default=8)
    parser.add_argument("--gcn2_alpha", type=float, default=0.1)
    parser.add_argument("--gcn2_theta", type=float, default=0.5)
    parser.add_argument("--gcn2_shared_weights", action="store_true")
    parser.add_argument("--gcn2_use_edge_weight", action="store_true")
    parser.add_argument("--h2gcn_layers", type=int, default=2)
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="subject_wise",
        choices=["none", "subject_wise", "channel_wise"],
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="none",
        choices=["none", "ea", "ra"],
        help="Alignment mode: none, Euclidean alignment (ea), or Riemannian alignment (ra).",
    )
    args = parser.parse_args()
    if args.align_mode == "none":
        edge_source = "connectivity"
    else:
        edge_source = "aligned_adj"
    all_data_path = args.all_data_path
    topology = args.topology#"fixed"
    mil_pool_type= args.mil_pool_type #"mean" #"mean"
    edge_mode = args.edge_mode #"topology_binary"
    dim = args.dim
    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    feature_name_list =  args.feature_families.replace(",", "_")
    feature_name_list =  feature_name_list.replace("relative_band_power", "RBP")
    if args.encoder_type in ["LINKX", "mlp_node", "linkx_cnn"]:
        start_epoch=30
    else:
        lr = 1e-3
        # weight_decay = 1e-4
        epochs = 500
        start_epoch = 100
        patience = 50
        # dropout = 0.1
        # dim = 16   # or 32, not larger first
        # graph_emb_dim = dim * 2
        # attn_dim = dim * 2

    gnn_hidden_dim=dim
    graph_emb_dim=dim*2
    attn_dim=dim*2
    dropout=0.3
    node_hidden_dims=(dim*2, dim)
    edge_hidden_dims=(dim*2, dim)
    branch_emb_dim=dim
    base_k=args.base_k
    max_k_per_subject=100
    standardize_features=True

    save_path = os.path.join(root_path,'result_Apr12')
    os.makedirs(save_path,exist_ok = True)
    # all_data_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
    last_part = os.path.basename(all_data_path)
    parts = last_part.split('_')
    
    if "mono" in parts:
        channel_names = config.MONO_CHANNELS
        fixed_pairs = config.MONOFIXEDGES
        channel_name = "mono"
        n_channels = 19
        fixed_edges = _normalize_fixed_edges(fixed_pairs, n_channels, channel_names)

    elif "bi23" in parts:
        channel_names = config.bi23_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi23"

    elif "bi30" in parts:
        channel_names = config.bi30_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi30"


    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{topology}_{feature_families[1]}_{args.connectivity_metric}"
    check_term = f"{args.encoder_type}_{mil_pool_type}_{args.norm_mode}_{channel_name}_{topology}_{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_{args.base_k}_{args.dim}"
    
    # for d in os.listdir(save_path):
    #     path = os.path.join(save_path, d)
    #     if os.path.isdir(path):
    #         # print(path)
    #         last_part = os.path.basename(path)
    #         # if check_term in last_part:
    #         if result_already_exists(save_path, check_term, "overall_summary_test.csv"):
    #             import sys
    #             print(f"Already run: {check_term} skipped!")
    #             sys.exit(0) 

    # folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{channel_name}_{topology}_{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_{args.base_k}_{args.dim}"
    folder_name = f"{timestamp}_{check_term}"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")


    print("File found! Processing...")
    with open(log_path, "w") as f:
        f.write(f"data source {all_data_path}\n")
        f.write(f"k {k}, val_ratio {val_ratio}, split_seeds {split_seeds}\n")
        f.write(f"norm_mode {args.norm_mode}\n")
        f.write(f"note: update - use topology instead of full adj\n")
        f.write(f"\n")

        f.write(f"topology: {topology}, fixed_edges: {fixed_edges}\n")
        f.write(f"feature_families: {args.feature_families}\nconnectivity_metric: {args.connectivity_metric}, connectivity_band: {args.connectivity_band}\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"\n")

        f.write(f"model_name: {args.encoder_type}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta=1e-3, top_k=5 \n")
        f.write(f"batch_size_train {batch_size_train}, batch_size_val {batch_size_val}, batch_size_test {batch_size_test}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"dim {dim} \n gnn_hidden_dim={gnn_hidden_dim} \n graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f"dropout={dropout}\n node_hidden_dims={node_hidden_dims} \n edge_hidden_dims={edge_hidden_dims}\n branch_emb_dim={branch_emb_dim}\n")
    


    result_all = []
    fold_metric_rows = []
    pred_rows = []

    if args.encoder_type =="linkx_cnn5":
        payload = load_h5_payload_for_subjects(
            h5_path=all_data_path,
            subject_ids=sub_id_list,   # load all subjects
            feature_families=feature_families,
            connectivity_metrics=[args.connectivity_metric] if args.connectivity_metric is not None else [],
            connectivity_band=None,
            load_raw_for_alignment=False,
            load_bad_segment_flag=False,
        )
    else:
        payload = load_h5_payload_for_subjects(
            h5_path=all_data_path,
            subject_ids=sub_id_list,   # load all subjects
            feature_families=feature_families,
            connectivity_metrics=[args.connectivity_metric] if args.connectivity_metric is not None else [],
            connectivity_band=args.connectivity_band,
            load_raw_for_alignment=False,
            load_bad_segment_flag=False,
        )
    payload = filter_payload_bad_windows_in_place(payload)

    if args.norm_mode in {"none", "subject_wise", "channel_wise"} and args.align_mode == "none":
        payload, global_norm_stats = normalize_payload_feature_families(
            payload,
            feature_families=feature_families,
            norm_mode=args.norm_mode,
            in_place=True,
        )
    # graphs = build_graphs_from_master_h5(
    #     h5_path=all_data_path,
    #     feature_families=feature_families,
    #     connectivity_metric=args.connectivity_metric,
    #     connectivity_band=args.connectivity_band,
    #     subject_ids=sub_id_list,
    #     standardize_features=standardize_features,
    #     node_feature_mode="selected_features",
    #     connectivity_mode="selected_metric",
    # )
    all_result_rows = []
    for seed in split_seeds:
        set_global_seed(seed)

        print(f"\n========== Split seed: {seed} ==========")
        seed_dir = os.path.join(output_dir, f"seed{seed}")
        os.makedirs(seed_dir,exist_ok = True)
        with open(log_path, "a") as f:
            f.write(f"======================================\n")
            f.write(f"Seed random = {seed}\n")

        all_folds = balanced_kfold_split(sub_id_list, labels, seed, k)
        check_dir = os.path.join(f"{seed_dir}/checkpoints")
        os.makedirs(check_dir,exist_ok=True)
        cv_subject_embeddings = os.path.join(f"{seed_dir}/cv_subject_embeddings")
        os.makedirs(cv_subject_embeddings,exist_ok=True)

        all_fold_data = []
        for i, test_subjects in enumerate(all_folds):
            print(f"\n========== Fold: {i} ==========")
            with open(log_path, "a") as f:
                f.write(f"\n========== Fold: {i} ==========\n")

            tsne_fold = os.path.join(f"{seed_dir}/tsne_fold{i}")
            os.makedirs(tsne_fold,exist_ok=True)
            print(test_subjects)
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]

            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            # train_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in train_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            subject_label_map = dict(zip(train_subjects, train_labels))
       
            new_train_subjects, val_subjects = stratified_split_subjects(
                train_subjects, subject_label_map, val_ratio, seed
            )
   
            print(f"# Train_subjects = {len(new_train_subjects)} | # Validation subjects = {len(val_subjects)}")

            # train_graphs = [g for g in graphs if g.subject_id in new_train_subjects]
            # val_graphs   = [g for g in graphs if g.subject_id in val_subjects]
            # test_graphs  = [g for g in graphs if g.subject_id in set(test_subjects)]

            # train_graphs = build_graphs_from_payload(payload, subject_ids=train_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source =edge_source)
            # val_graphs   = build_graphs_from_payload(payload, subject_ids=val_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source = edge_source)
            # test_graphs  = build_graphs_from_payload(payload, subject_ids=test_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source =edge_source)
            if args.encoder_type =="linkx_cnn5":
                train_graphs = build_graphs_from_payload_multiband(
                    payload=payload,
                    subject_ids=new_train_subjects,
                    feature_families=feature_families,
                    connectivity_metric=args.connectivity_metric,
                    edge_source=edge_source,
                    sparse_adj_reduction="mean",
                )

                val_graphs = build_graphs_from_payload_multiband(
                    payload=payload,
                    subject_ids=val_subjects,
                    feature_families=feature_families,
                    connectivity_metric=args.connectivity_metric,
                    edge_source=edge_source,
                    sparse_adj_reduction="mean",
                )

                test_graphs = build_graphs_from_payload_multiband(
                    payload=payload,
                    subject_ids=test_subjects,
                    feature_families=feature_families,
                    connectivity_metric=args.connectivity_metric,
                    edge_source=edge_source,
                    sparse_adj_reduction="mean",
                )
            else:
                train_graphs = build_graphs_from_payload(
                    payload,
                    subject_ids=new_train_subjects,
                    feature_families=feature_families,
                    connectivity_metric=args.connectivity_metric,
                    edge_source=edge_source,
                    filter_method=args.topology,
                    fixed_edges=fixed_edges,          # from config
                    channel_names=channel_names,      # whatever list you use for this payload
                    undirected=True,
                    standardize_features=True,       # or True if desired
                )

                val_graphs = build_graphs_from_payload(
                    payload,
                    subject_ids=val_subjects,
                    feature_families=feature_families,
                    connectivity_metric=args.connectivity_metric,
                    edge_source=edge_source,
                    filter_method=args.topology,
                    fixed_edges=fixed_edges,          # from config
                    channel_names=channel_names,      # whatever list you use for this payload
                    undirected=True,
                    standardize_features=True,       # or True if desired
                )


                test_graphs = build_graphs_from_payload(
                    payload,
                    subject_ids=test_subjects,
                    feature_families=feature_families,
                    connectivity_metric=args.connectivity_metric,
                    edge_source=edge_source,
                    filter_method=args.topology,
                    fixed_edges=fixed_edges,          # from config
                    channel_names=channel_names,      # whatever list you use for this payload
                    undirected=True,
                    standardize_features=True,       # or True if desired
                )

            train_graphs = attach_summary_features_to_graphs(train_graphs)
            val_graphs   = attach_summary_features_to_graphs(val_graphs)
            test_graphs  = attach_summary_features_to_graphs(test_graphs)

            g = train_graphs[0]
            print(hasattr(g, "edge_attr"))
            print(g.edge_attr[:10] if hasattr(g, "edge_attr") and g.edge_attr is not None else None)
            print(hasattr(g, "edge_weight"))
            print(g.edge_weight[:10] if hasattr(g, "edge_weight") and g.edge_weight is not None else None)


            summary_input_dim = train_graphs[0].summary_feat.numel()
            print("summary_input_dim =", summary_input_dim)

            if base_k is None:
                train_dataset = SubjectBagGraphDataset(
                    train_graphs,
                    max_segments_per_subject=None,   # good default for training memory
                    train=True,
                )

                val_dataset = SubjectBagGraphDataset(
                    val_graphs,
                    max_segments_per_subject=None, # use all segments at validation if memory allows
                    train=False,
                )


                test_dataset = SubjectBagGraphDataset(
                    test_graphs,
                    max_segments_per_subject=None, # use all segments at validation if memory allows
                    train=False,
                )

            else:
                train_dataset = LabelAwareSubjectBagDataset(
                    train_graphs,
                    train=True,
                    base_k=base_k,                      # reference k
                    k_by_label=None,                # auto-compute from class counts
                    target_segments_per_class=None, # defaults to majority_class_subjects * base_k
                    max_k_per_subject=max_k_per_subject,           # optional cap
                    seed=seed,
                    return_segment_ids=True,        # optional, useful for debugging
                )

                val_dataset = LabelAwareSubjectBagDataset(
                    val_graphs,
                    train=False,
                    eval_k_per_subject=None,        # None = use all val segments
                    seed=seed,
                )

                test_dataset = LabelAwareSubjectBagDataset(
                    test_graphs,
                    train=False,
                    eval_k_per_subject=None,        # None = use all test segments
                    seed=seed,
                )

            print("Train subject class counts:", np.bincount(train_dataset.subject_labels, minlength=num_classes))
            print("Val subject class counts:", np.bincount(val_dataset.subject_labels, minlength=num_classes))
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            
            input_model = SubjectMILClassifier(
                num_node_features=train_dataset.num_node_features,
                num_classes=num_classes,
                num_nodes=train_dataset.num_nodes,
                encoder_type=args.encoder_type,
                edge_mode=args.edge_mode,
                graph_emb_dim=dim*2,
                dropout=dropout,
                graph_pool=args.graph_pool,
                gnn_hidden_dim=dim,
                sage_layers=args.sage_layers,
                gcn2_layers=args.gcn2_layers,
                gcn2_alpha=args.gcn2_alpha,
                gcn2_theta=args.gcn2_theta,
                gcn2_shared_weights=args.gcn2_shared_weights,
                gcn2_use_edge_weight=args.gcn2_use_edge_weight,
                h2gcn_layers=args.h2gcn_layers,
                mil_pool_type=mil_pool_type,
                attn_dim=dim*2,
                num_gnn_layers=num_gnn_layers,
                readout_type=readout_type,
                node_pooling_type=node_pooling_type,
                node_pool_ratio=node_pool_ratio,
                use_edge_weight=bool(use_edge_weight),
                gat_heads=gat_heads,
                readout_hidden_dim=readout_hidden_dim,
                readout_dropout=readout_dropout,
                ).to(device)
            class_weights = compute_class_weights_from_subjects(
                subject_labels=train_dataset.subject_labels,
                num_classes=num_classes,
            ).to(device)
            print("class_weights", class_weights)
            # class_weights tensor([1.0000, 0.8148, 1.2941])

            criterion = nn.CrossEntropyLoss()
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(input_model.parameters(), lr=lr, weight_decay=weight_decay)
            if args.encoder_type =="linkx_cnn5":

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size_train,
                    shuffle=True,
                    collate_fn=collate_subject_bags_multiband,
                    num_workers=0,
                    pin_memory=True,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size_val,
                    shuffle=False,
                    collate_fn=collate_subject_bags_multiband,
                    num_workers=0,
                    pin_memory=True,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    collate_fn=collate_subject_bags_multiband,
                    num_workers=0,
                    pin_memory=True,
                )

            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size_train,
                    shuffle=True,
                    collate_fn=collate_subject_bags,
                    num_workers=0,
                    pin_memory=True,
                )

                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size_val,
                    shuffle=False,
                    collate_fn=collate_subject_bags,
                    num_workers=0,
                    pin_memory=True,
                )

                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size_test,
                    shuffle=False,
                    collate_fn=collate_subject_bags,
                    num_workers=0,
                    pin_memory=True,
                )

            model, val_metrics, history, best_state = fit_mil_baseline(
                input_model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs,
                patience,
                save_path=f"{check_dir}/best_mil_model_fold{i}.pt",
                start_epoch=start_epoch,     # warmup: do not count patience before epoch 30
                min_delta=1e-3,     # require at least this much val-loss improvement
                top_k=5,            # keep 5 lowest-loss checkpoints
                verbose=True,
            )

            checkpoint = torch.load(f"{check_dir}/best_mil_model_fold{i}.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Best epoch:", checkpoint["epoch"])
            print("Best val metrics:", checkpoint["best_val_macro_f1"])


            with open(log_path, "a") as f:
                f.write("Final validation metrics:\n")
                f.write(f"Accuracy:           {val_metrics['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Macro-F1:           {val_metrics['macro_f1']:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{val_metrics['conf_matrix']}\n")

            print("\nFinal validation metrics:")
            print(f"Accuracy:           {val_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1:           {val_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(val_metrics["conf_matrix"])


            criterion = nn.CrossEntropyLoss()
            test_metrics = evaluate(model, test_loader, criterion, device)

            with open(log_path, "a") as f:
                f.write("Final test metrics:\n")
                f.write(f"Accuracy:           {test_metrics['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Macro-F1:           {test_metrics['macro_f1']:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{test_metrics['conf_matrix']}\n")


            print("\nFinal test metrics:")
            print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1:           {test_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(test_metrics["conf_matrix"])


            all_result_rows.append({
                "split_seed": seed,
                "fold": i,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            })

            train_subject_rows_f, val_subject_rows_f, test_subject_rows_f = save_fold_subject_embeddings(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    fold_idx=i,
                    save_dir=cv_subject_embeddings
                )

            all_fold_data.append({
                "fold": i,
                "train_rows": train_subject_rows_f,
                "val_rows": val_subject_rows_f,
                "test_rows": test_subject_rows_f,
            })
            # plot_subject_embeddings_tsne(train_subject_rows_f, "subject", "train", tsne_fold, color_by="label", title="Train Subject Embeddings by True Class")
            # plot_subject_embeddings_tsne(val_subject_rows_f, "subject", "val", tsne_fold, color_by="label", title="Validation Subject Embeddings by True Class")
            # plot_subject_embeddings_tsne(test_subject_rows_f, "subject", "test", tsne_fold, color_by="label", title="Test Subject Embeddings by True Class")
            fold_metric_rows.append(metrics_to_row(test_metrics, seed, i, "test"))
            pred_rows.extend(predictions_to_rows(test_metrics, seed, i, "test", num_classes))

            train_seg_rows = collect_segment_embeddings(model, train_loader, device)
            val_seg_rows = collect_segment_embeddings(model, val_loader, device)
            test_seg_rows  = collect_segment_embeddings(model, test_loader, device)
            # plot_subject_embeddings_tsne(train_seg_rows, "segment", "train", tsne_fold, color_by="subject", title="Train Segment Embeddings by True Class")
            # plot_subject_embeddings_tsne(val_seg_rows, "segment", "val", tsne_fold, color_by="subject", title="Validation Segment Embeddings by True Class")
            plot_subject_embeddings_tsne(test_seg_rows, "segment", "test", tsne_fold, color_by="subject", title="Test Segment Embeddings by True Class")

            fingerprint_stats_train = segment_fingerprint_metrics(train_seg_rows)
            fingerprint_stats_test  = segment_fingerprint_metrics(test_seg_rows)

            with open(log_path, "a") as f:
                f.write(f"TRAIN fingerprint stats: {fingerprint_stats_train}\n")
                f.write(f"TEST  fingerprint stats: {fingerprint_stats_test}\n")

        with open(f"{cv_subject_embeddings}/all_fold_subject_rows.pkl", "wb") as f:
            pickle.dump(all_fold_data, f)
        all_fold_data = load_all_fold_data(f"{cv_subject_embeddings}/all_fold_subject_rows.pkl")
        print(len(all_fold_data))
        print(all_fold_data[0].keys())

        aligned_oof_rows = align_oof_test_embeddings_across_folds(
            all_fold_data,
            reference_fold=0
        )

        class_dict = {
            0: "HC",
            1: "AD",
            2: "FTD",
        }
        plot_aligned_subject_embeddings_umap(
            aligned_oof_rows,
            class_names=class_dict,
            title="Out-of-Fold Subject Embeddings",
            annotate_subject_ids=True,
            save_path=f"{seed_dir}/plot_aligned_subject_embeddings_umap.png"
        )
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_path = os.path.join(output_dir, "fold_metrics_all_seeds.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(output_dir, "subject_predictions_all_seeds.csv")
    pred_df.to_csv(pred_path, index=False)

    test_summary_by_split = (
        fold_metrics_df[fold_metrics_df["split"] == "test"]
        .groupby("split_seed")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )
    test_summary_by_split.to_csv(
        os.path.join(output_dir, "test_summary_by_split_seed.csv"),
        index=False
    )

    overall_summary = (
        fold_metrics_df[fold_metrics_df["split"] == "test"][["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    print(overall_summary)
    overall_summary.to_csv(
        os.path.join(output_dir, "overall_summary_test.csv")
    )