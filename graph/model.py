import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv,ChebConv, BatchNorm
from torch_geometric.nn import GINConv, GINEConv, global_add_pool, global_mean_pool, global_max_pool
from torch import Tensor
from einops.layers.torch import Rearrange, Reduce
from torch_geometric.nn import GATv2Conv, NNConv, GlobalAttention
from torch_geometric.utils import dropout_adj


from torch_geometric.data import Batch
from torch_geometric.nn import (
    GATv2Conv,
    ChebConv,
    GraphNorm,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def get_edge_attr_and_weight(data_batch: Batch):
    edge_attr = getattr(data_batch, "edge_attr", None)
    edge_weight = getattr(data_batch, "edge_weight", None)

    # If only edge_weight exists, convert for GATv2
    if edge_attr is None and edge_weight is not None:
        if edge_weight.dim() == 1:
            edge_attr = edge_weight.unsqueeze(-1)   # [E] -> [E, 1]
        else:
            edge_attr = edge_weight

    # If only edge_attr exists, convert for ChebConv if possible
    if edge_weight is None and edge_attr is not None:
        if edge_attr.dim() == 1:
            edge_weight = edge_attr
        elif edge_attr.dim() == 2 and edge_attr.size(-1) == 1:
            edge_weight = edge_attr.view(-1)

    return edge_attr, edge_weight


def pool_graph(x, batch, pooling: str = "mean"):
    pooling = pooling.lower()
    if pooling == "mean":
        return global_mean_pool(x, batch)
    if pooling == "max":
        return global_max_pool(x, batch)
    if pooling == "sum":
        return global_add_pool(x, batch)
    raise ValueError(f"Unknown pooling: {pooling}")


# ---------------------------------------------------
# Blocks
# ---------------------------------------------------
class GATv2Block(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        heads: int = 4,
        edge_dim: int = 1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads when concat=True"
        out_per_head = hidden_dim // heads

        self.conv = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=out_per_head,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=attn_dropout,
        )
        self.norm = GraphNorm(hidden_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = self.norm(x, batch)
        x = F.relu(x)
        return x


class ChebBlock(nn.Module):
    def __init__(self, hidden_dim: int, K: int = 3):
        super().__init__()
        self.conv = ChebConv(hidden_dim, hidden_dim, K=K)
        self.norm = GraphNorm(hidden_dim)

    def forward(self, x, edge_index, batch, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight=edge_weight)
        x = self.norm(x, batch)
        x = F.relu(x)
        return x


# ---------------------------------------------------
# 1) GATv2 encoder for MIL
# ---------------------------------------------------
class GATv2GraphEncoder(nn.Module):
    """
    Segment graph -> graph embedding
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        emb_dim: int = 128,
        num_layers: int = 3,
        heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        pooling: str = "mean",
    ):
        super().__init__()

        self.dropout = dropout
        self.pooling = pooling

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gat_layers = nn.ModuleList([
            GATv2Block(
                hidden_dim=hidden_dim,
                heads=heads,
                edge_dim=edge_dim,
                attn_dropout=attn_dropout,
            )
            for _ in range(num_layers)
        ])

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, data_batch: Batch) -> torch.Tensor:
        x = data_batch.x
        edge_index = data_batch.edge_index
        batch = data_batch.batch

        edge_attr, _ = get_edge_attr_and_weight(data_batch)

        x = self.input_proj(x)

        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for layer in self.gat_layers:
            x = layer(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        graph_emb = pool_graph(x, batch, self.pooling)
        graph_emb = self.graph_proj(graph_emb)
        return graph_emb


# ---------------------------------------------------
# 2) Parallel GATv2 + Cheb hybrid encoder for MIL
# ---------------------------------------------------
class HybridGraphEncoder(nn.Module):
    """
    Parallel spatial (GATv2) + spectral (Cheb) branches
    Segment graph -> graph embedding
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        emb_dim: int = 128,
        gat_layers: int = 2,
        cheb_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 1,
        cheb_K: int = 3,
        dropout: float = 0.2,
        attn_dropout: float = 0.0,
        pooling: str = "mean",
    ):
        super().__init__()

        self.dropout = dropout
        self.pooling = pooling

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gat_stack = nn.ModuleList([
            GATv2Block(
                hidden_dim=hidden_dim,
                heads=heads,
                edge_dim=edge_dim,
                attn_dropout=attn_dropout,
            )
            for _ in range(gat_layers)
        ])

        self.cheb_stack = nn.ModuleList([
            ChebBlock(hidden_dim=hidden_dim, K=cheb_K)
            for _ in range(cheb_layers)
        ])

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, data_batch: Batch) -> torch.Tensor:
        x = data_batch.x
        edge_index = data_batch.edge_index
        batch = data_batch.batch

        edge_attr, edge_weight = get_edge_attr_and_weight(data_batch)

        x0 = self.input_proj(x)

        # Spatial branch: GATv2
        xs = x0
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for layer in self.gat_stack:
            xs = layer(xs, edge_index, batch, edge_attr=edge_attr)
            xs = F.dropout(xs, p=self.dropout, training=self.training)

        xs = pool_graph(xs, batch, self.pooling)

        # Spectral branch: Cheb
        xp = x0
        for layer in self.cheb_stack:
            xp = layer(xp, edge_index, batch, edge_weight=edge_weight)
            xp = F.dropout(xp, p=self.dropout, training=self.training)

        xp = pool_graph(xp, batch, self.pooling)

        # Fusion
        graph_emb = torch.cat([xs, xp], dim=1)
        graph_emb = self.graph_proj(graph_emb)
        return graph_emb

class EdgeAwareGINE(nn.Module):
    def __init__(self, in_dim, hidden=128, num_classes=3, edge_dim=1, num_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        self.node_enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )

        self.edge_enc = nn.Sequential(
            nn.Linear(edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )


        self.edge_norm = nn.LayerNorm(hidden)  
        self.node_norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])

        def gine_mlp():
            return nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
            )

        # self.convs = nn.ModuleList([
        #     GINEConv(gine_mlp(), edge_dim=hidden),
        #     GINEConv(gine_mlp(), edge_dim=hidden),
        # ])
        self.convs = nn.ModuleList([
            GINEConv(gine_mlp(), edge_dim=hidden)
            for _ in range(num_layers)
        ])

        self.cls = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x, edge_index, edge_attr, batch, return_emb=False):
        x = self.node_enc(x)
        e = self.edge_enc(edge_attr)
        e = self.edge_norm(e)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, e)
            x = self.node_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        z = global_mean_pool(x, batch)   # <-- graph embedding

        out = self.cls(z)

        if return_emb:
            return out, z
        return out


# class Gcn_block(nn.Module):
#     def __init__(self, 
#                  hidden_channels=64,
#                  concat=True,
#                  edge_dim=1,
#                  heads=4, 
#                  ):
#         super(Gcn_block, self).__init__()
#         self.conv = GCNConv(hidden_channels, edge_dim=edge_dim)
#         self.bn = BatchNorm(hidden_channels)
#     def forward(self,x,edge_index,batch,edge_attr=None):
#         #print(x.shape)
#         xs = F.relu(self.bn(self.conv(x, edge_index, edge_attr=edge_attr)))
#         return xs
class Gat_block(nn.Module):
    def __init__(self, 
                 hidden_channels=64,
                 concat=True,
                 edge_dim=1,
                 heads=4, 
                 ):
        super(Gat_block, self).__init__()
        self.conv = GATv2Conv(hidden_channels, int(hidden_channels/heads), heads=heads, concat=concat, edge_dim=edge_dim)
        self.bn = BatchNorm(hidden_channels)
    def forward(self,x,edge_index,batch,edge_attr=None):
        #print(x.shape)
        xs = F.relu(self.bn(self.conv(x, edge_index, edge_attr=edge_attr)))
        return xs
class Cheb_block(nn.Module):
    def __init__(self, 
                 hidden_channels=64,
                 K=3,
                 ):
        super(Cheb_block, self).__init__()
        self.conv = ChebConv(hidden_channels, hidden_channels, K=K)
        self.bn = BatchNorm(hidden_channels)
    def forward(self,x,edge_index,batch,edge_attr=None):
        xs = F.relu(self.bn(self.conv(x, edge_index, edge_weight=edge_attr)))
        return xs

class GatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super(GatedAttention, self).__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = 1

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (num_instances, input_dim)
        A_V = self.attention_V(x)  # N x D
        A_U = self.attention_U(x)  # N x D
        
        # Element-wise multiplication (The Gating Mechanism)
        A = self.attention_weights(A_V * A_U) # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        A = F.softmax(A, dim=1)  # Softmax over instances
        
        # Multiply instances by attention weights
        M = torch.mm(A, x)  # K x L
        return M, A

class EEGGNN_GAT_MIL(nn.Module):
    def __init__(self, in_channels=18, hidden_channels=64, num_classes=3, 
                 num_layers=3, dropout=0.3, heads=4, edge_dim=1):
        super(EEGGNN_GAT_MIL, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.input_proj = nn.Linear(in_channels, hidden_channels)

        self.gat_layers = nn.ModuleList([
            Gat_block(hidden_channels=hidden_channels, concat=True, edge_dim=edge_dim, heads=heads)
            for _ in range(num_layers)
        ])
        self.readout_dim = hidden_channels
        self.pos_embedding = nn.Parameter(torch.randn(1, 1000, self.readout_dim)) 

        # MIL Attention Pooling
        # self.attention = nn.Sequential(
        #     nn.Linear(hidden_channels, hidden_channels // 2),
        #     nn.ReLU(),
        #     nn.Linear(hidden_channels // 2, 1)
        # )
        # 3. GATED MIL Attention Pooling
        # Path V (Tanh)
        self.attention_V = nn.Sequential(
            nn.Linear(self.readout_dim, self.readout_dim // 2),
            nn.Tanh()
        )
        # Path U (Sigmoid Gate)
        self.attention_U = nn.Sequential(
            nn.Linear(self.readout_dim, self.readout_dim // 2),
            nn.Sigmoid()
        )
        # Final attention score projection
        self.attention_weights = nn.Linear(self.readout_dim // 2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes)
        )

    def forward(self, bag_of_graphs, return_attention=False):
        # 1. Standard PyG data extraction
        x, edge_index, batch, edge_attr = bag_of_graphs.x, bag_of_graphs.edge_index, bag_of_graphs.batch, bag_of_graphs.edge_attr

        # --- Stage 1: Spatial GNN ---
        # Direct projection instead of deep MLP
        x = self.input_proj(x) 
        
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        for conv in self.gat_layers:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Readout: Shape [num_segments, hidden_channels]
        seg_embeddings = global_mean_pool(x, batch) 
        # Inside forward pass
        # x_mean = global_mean_pool(x, batch)
        # x_max = global_max_pool(x, batch)
        # seg_embeddings = torch.cat([x_mean, x_max], dim=1) # Adjust hidden_channels accordingly

        # --- Stage 2: Temporal & Attention ---
        num_segments = seg_embeddings.size(0)
        positions = self.pos_embedding[:, :num_segments, :].squeeze(0)
        seg_embeddings = seg_embeddings + positions

        # # Calculate Attention Weights
        # attn_weights = self.attention(seg_embeddings) # [num_segments, 1]
        # alpha = F.softmax(attn_weights, dim=0)        # Importance of each segment
        # Gated Attention Mechanism
        A_V = self.attention_V(seg_embeddings)  # [num_segments, hidden//2]
        A_U = self.attention_U(seg_embeddings)  # [num_segments, hidden//2]
        
        # Element-wise multiplication (The Gate)
        attn_weights = self.attention_weights(A_V * A_U) # [num_segments, 1]
        alpha = F.softmax(attn_weights, dim=0)           # Importance of each segment

        # Weighted Sum (Subject Vector)
        subject_vector = torch.sum(alpha * seg_embeddings, dim=0, keepdim=True)

        # --- Stage 3: Classifier ---
        logits = self.classifier(subject_vector)

        if return_attention:
            return logits, alpha
        return logits
        
# class EEGGNN_GAT_MIL(nn.Module):
#     def __init__(self, in_channels=18, hidden_channels=64, num_classes=3, 
#                  num_layers=3, dropout=0.3, heads=4, edge_dim=1):
#         super(EEGGNN_GAT_MIL, self).__init__()
        
#         self.hidden_channels = hidden_channels
#         self.dropout = dropout

#         # 1. NODE ENCODER (Your existing GAT logic)
#         self.input_mlp = nn.Sequential(
#             nn.Linear(in_channels, hidden_channels),
#             nn.BatchNorm1d(hidden_channels),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
#         self.gat_layers = nn.ModuleList([
#             Gat_block(hidden_channels=hidden_channels, concat=True, edge_dim=edge_dim, heads=heads)
#             for _ in range(num_layers)
#         ])

#         # 2. TEMPORAL ENCODER (Optional but recommended for time relations)
#         # Learnable positional encoding to tell the model the order of segments
#         self.pos_embedding = nn.Parameter(torch.randn(1, 500, hidden_channels)) # Support up to 500 segments

#         # 3. GLOBAL CONTEXT ATTENTION (MIL Pooling)
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_channels, hidden_channels // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_channels // 2, 1)
#         )

#         # 4. CLASSIFIER
#         self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
#         self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

#     def forward(self, bag_of_graphs):
#         """
#         bag_of_graphs: A single Batch object containing all segments for ONE subject
#         """
#         x, edge_index, batch, edge_attr = bag_of_graphs.x, bag_of_graphs.edge_index, bag_of_graphs.batch, bag_of_graphs.edge_attr

#         # --- Stage 1: Spatial GNN (Segment Level) ---
#         x = self.input_mlp(x)
#         if edge_attr is not None and edge_attr.dim() == 1:
#             edge_attr = edge_attr.unsqueeze(-1)

#         for conv in self.gat_layers:
#             x = conv(x, edge_index, batch, edge_attr=edge_attr)
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         # Readout: Collapse nodes to get segment embeddings
#         # Shape: [num_segments, hidden_channels]
#         seg_embeddings = global_mean_pool(x, batch) 

#         # --- Stage 2: Temporal & Attention (Subject Level) ---
#         num_segments = seg_embeddings.size(0)
        
#         # Add Positional Encoding (Injects time information)
#         # We take the first 'num_segments' markers
#         positions = self.pos_embedding[:, :num_segments, :].squeeze(0)
#         seg_embeddings = seg_embeddings + positions

#         # Calculate Attention Weights
#         attn_weights = self.attention(seg_embeddings) # [num_segments, 1]
#         alpha = F.softmax(attn_weights, dim=0)        # Normalizes weights to sum to 1

#         # Weighted Sum (The "Subject Vector")
#         # Shape: [1, hidden_channels]
#         subject_vector = torch.sum(alpha * seg_embeddings, dim=0, keepdim=True)

#         # --- Stage 3: Classifier ---
#         out = F.relu(self.fc1(subject_vector))
#         out = F.dropout(out, p=self.dropout, training=self.training)
#         logits = self.fc2(out)

#         return logits
class Hybrid_mlp(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 num_classes=2,
                 gatlayers=2,
                 cheblayers=2,
                 dropout=0.3, 
                 heads=4, 
                 post_mlp_hidden=128,
                 pooling="mean"):
        super(Hybrid_mlp, self).__init__()
        self.pooling = pooling.lower()

        self.dropout = dropout
        self.gatlayers = gatlayers
        self.cheblayers = cheblayers
        # ----- Pre-MLP before GATv2 -----
        self.pre_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            # nn.Linear(mlp_hidden, in_channels),
            # nn.BatchNorm1d(in_channels),
            # nn.ReLU(),
        )
        self.gatv2_list = nn.ModuleList()
        for i in range(self.gatlayers):
            self.gatv2_list.append(Gat_block(hidden_channels,True,1,heads))
        self.cheb_list = nn.ModuleList()
        for i in range(self.cheblayers):
            self.cheb_list.append(Cheb_block(hidden_channels,3+i))
        # # ----- Spatial branch (GATv2) -----
        # # self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1)
        # self.bn_gat1 = BatchNorm(hidden_channels * heads)
        # self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=1)
        # self.bn_gat2 = BatchNorm(hidden_channels)
        
        # ----- Spectral branch (ChebConv) -----
        # self.cheb1 = ChebConv(in_channels, hidden_channels, K=2) #k = 3
        # self.bn_cheb1 = BatchNorm(hidden_channels)
        # self.cheb2 = ChebConv(hidden_channels, hidden_channels, K=3) #k = 4
        # self.bn_cheb2 = BatchNorm(hidden_channels)
        
        # ----- Post-Fusion MLP (dense classifier head) -----
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, post_mlp_hidden),
            nn.BatchNorm1d(post_mlp_hidden),
            nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(post_mlp_hidden, num_classes)
        )
    def apply_pooling(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "sum":
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")



    def forward(self, x, edge_index, batch, edge_attr=None):
        # ----- Pre-MLP -----
        x = self.pre_mlp(x)
        
        # ----- Spatial path (GATv2) -----
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # (num_edges, 1)
        #print(x.shape)
        for i,conv in enumerate(self.gatv2_list):
            x = conv(x,edge_index,edge_attr)
        # xs = global_mean_pool(x, batch)
        xs = self.apply_pooling(x, batch)
        # ----- Spectral path (ChebConv) -----
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        for i,conv in enumerate(self.cheb_list):
            x = conv(x,edge_index,edge_weight)
        # xp = global_mean_pool(x, batch)
        xp = self.apply_pooling(x, batch)
        # ----- Fusion -----
        x_cat = torch.cat([xs, xp], dim=1)
        
        # ----- Post-MLP (final classifier) -----
        out = self.post_mlp(x_cat)
        return out

class Hybrid_RF(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 num_classes=2,
                 gatlayers=2,
                 cheblayers=2,
                 dropout=0.3, 
                 heads=4, 
                 post_mlp_hidden=128,
                 pooling="mean"):
        super(Hybrid_RF, self).__init__()
        self.pooling = pooling.lower()

        self.dropout = dropout
        self.gatlayers = gatlayers
        self.cheblayers = cheblayers

        # ----- Pre-MLP -----
        self.pre_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ----- GAT blocks -----
        self.gatv2_list = nn.ModuleList([
            Gat_block(hidden_channels, True, 1, heads) 
            for _ in range(gatlayers)
        ])

        # ----- Cheb blocks -----
        self.cheb_list = nn.ModuleList([
            Cheb_block(hidden_channels, 3+i)
            for i in range(cheblayers)
        ])

        # ----- Post classifier -----
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, post_mlp_hidden),
            nn.BatchNorm1d(post_mlp_hidden),
            nn.ReLU(),
            nn.Linear(post_mlp_hidden, num_classes)
        )

    def apply_pooling(self, x, batch):
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling == "max":
            return global_max_pool(x, batch)
        elif self.pooling == "sum":
            return global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

    # -------------------------------
    #   UNIFIED FEATURE EXTRACTION
    # -------------------------------
    def _extract_latent(self, x, edge_index, batch, edge_attr=None):
        x = self.pre_mlp(x)
        
        # ----- Spatial path (GATv2) -----
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)  # (num_edges, 1)
        #print(x.shape)
        for i,conv in enumerate(self.gatv2_list):
            x = conv(x,edge_index,edge_attr)
        # xs = global_mean_pool(x, batch)
        xs = self.apply_pooling(x, batch)
        # ----- Spectral path (ChebConv) -----
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None

        for i,conv in enumerate(self.cheb_list):
            x = conv(x,edge_index,edge_weight)
        # xp = global_mean_pool(x, batch)
        xp = self.apply_pooling(x, batch)
        # ----- Fusion -----
        x_cat = torch.cat([xs, xp], dim=1)
        return x_cat
    # -------------------------------
    #       NORMAL FORWARD
    # -------------------------------
    def forward(self, x, edge_index, batch, edge_attr=None):
        x_cat = self._extract_latent(x, edge_index, batch, edge_attr)
        out = self.post_mlp(x_cat)
        return out

    # -------------------------------
    #   FEATURE EXTRACTION (for SVM/RF)
    # -------------------------------
    def extract_features(self, x, edge_index, batch, edge_attr=None):
        """Return only the fused features BEFORE classifier."""
        with torch.no_grad():
            return self._extract_latent(x, edge_index, batch, edge_attr)

# class EEGGNN_GAT(nn.Module):
#     def __init__(self, in_channels=18, hidden_channels=64, num_classes=2, 
#                  dropout=0.3, use_attention=True, heads=4, pooling="mean"):
#         super(EEGGNN_GAT, self).__init__()
        
#         self.use_attention = use_attention
#         self.dropout = dropout
#         self.pooling = pooling.lower()

#         if use_attention:
#             # 3 GAT layers
#             self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
#             gat_out_dim = hidden_channels * heads

#             self.conv2 = GATConv(gat_out_dim, hidden_channels, heads=heads, concat=True)
#             gat_out_dim = hidden_channels * heads

#             self.conv3 = GATConv(gat_out_dim, hidden_channels, heads=1, concat=False)
#             gat_out_dim = hidden_channels
#         else:
#             # If not using attention → fallback to GCN
#             self.conv1 = GCNConv(in_channels, hidden_channels)
#             self.conv2 = GCNConv(hidden_channels, hidden_channels)
#             self.conv3 = GCNConv(hidden_channels, hidden_channels)
#             gat_out_dim = hidden_channels

#         # BatchNorm layers
#         self.bn1 = BatchNorm(hidden_channels * heads if use_attention else hidden_channels)
#         self.bn2 = BatchNorm(hidden_channels * heads if use_attention else hidden_channels)
#         self.bn3 = BatchNorm(hidden_channels)

#         # Deeper classifier (2 dense layers)
#         self.fc1 = nn.Linear(gat_out_dim, gat_out_dim // 2)
#         self.fc2 = nn.Linear(gat_out_dim // 2, num_classes)

#     def forward(self, x, edge_index, batch):
#         # Layer 1
#         x = self.conv1(x, edge_index)
#         x = self.bn1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # Layer 2
#         x = self.conv2(x, edge_index)
#         x = self.bn2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # Layer 3
#         x = self.conv3(x, edge_index)
#         x = self.bn3(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         # Global pooling
#         if self.pooling == "mean":
#             x = global_mean_pool(x, batch)
#         elif self.pooling == "max":
#             x = global_max_pool(x, batch)
#         elif self.pooling == "sum":
#             x = global_add_pool(x, batch)
#         else:
#             raise ValueError(f"Unknown pooling type: {self.pooling}")

#         # Classifier (deeper: 2 dense layers)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         out = self.fc2(x)
#         return out

class EEGGNN_GAT(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 num_classes=2, 
                 num_layers=3,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(EEGGNN_GAT, self).__init__()
        
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
        for i in range(num_layers):
            self.gat_layers.append(Gat_block(
                hidden_channels=hidden_channels,
                concat=True,
                edge_dim=edge_dim,
                heads=heads
            ))

        # ----- Classification head -----
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Input projection
        x = self.input_mlp(x)

        # Ensure edge_attr shape
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # ----- Pass through stacked GAT layers -----
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

        # ----- Classifier -----
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc2(x)

        return out

class EEGGNN_GAT_sanity(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 num_classes=2, 
                 num_layers=3,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(EEGGNN_GAT_sanity, self).__init__()
        
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
        for i in range(num_layers):
            self.gat_layers.append(Gat_block(
                hidden_channels=hidden_channels,
                concat=True,
                edge_dim=edge_dim,
                heads=heads
            ))

        # ----- Classification head -----
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # Input projection
        x = self.input_mlp(x)

        # Ensure edge_attr shape
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # ----- Pass through stacked GAT layers -----
        for conv in self.gat_layers:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # ----- Global pooling -----
        if self.pooling == "mean":
            z = global_mean_pool(x, batch)
        elif self.pooling == "max":
            z = global_max_pool(x, batch)
        elif self.pooling == "sum":
            z = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # ----- Classifier -----
        x = F.relu(self.fc1(z))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.fc2(x)

        return out, z
# class EEGGNN_Hybrid_old(nn.Module):
#     def __init__(self, 
#                  in_channels=18, 
#                  hidden_channels=64, 
#                  num_classes=2,
#                  gat_layers=2, 
#                  cheb_layers=2,
#                  dropout=0.3, 
#                  heads=4, 
#                  edge_dim=1,
#                  pooling="mean"):
#         super(EEGGNN_Hybrid_old, self).__init__()
        
#         self.dropout = dropout
#         self.pooling = pooling.lower()

#         # ----- Spatial branch (GATv2) -----
#         self.gat_input = nn.Linear(in_channels, hidden_channels)
#         self.gat_layers = nn.ModuleList([
#             Gat_block(hidden_channels=hidden_channels, concat=True, edge_dim=edge_dim, heads=heads)
#             for _ in range(gat_layers)
#         ])

#         # ----- Spectral branch (ChebConv) -----
#         self.cheb_input = nn.Linear(in_channels, hidden_channels)
#         self.cheb_layers = nn.ModuleList([
#             Cheb_block(hidden_channels=hidden_channels, K=3 + i)
#             for i in range(cheb_layers)
#         ])

#         # ----- Fusion -----
#         self.classifier = nn.Linear(hidden_channels * 2, num_classes)

#     def apply_pooling(self, x, batch):
#         if self.pooling == "mean":
#             return global_mean_pool(x, batch)
#         elif self.pooling == "max":
#             return global_max_pool(x, batch)
#         elif self.pooling == "sum":
#             return global_add_pool(x, batch)
#         else:
#             raise ValueError(f"Unknown pooling type: {self.pooling}")

#     def forward(self, x, edge_index, batch):
#         # ----- Spatial path (GAT) -----
#         xs = F.relu(self.gat_input(x))
#         for conv in self.gat_layers:
#             xs = conv(xs, edge_index, batch)
#             xs = F.dropout(xs, p=self.dropout, training=self.training)
#         xs = self.apply_pooling(xs, batch)

#         # ----- Spectral path (ChebConv) -----
#         xp = F.relu(self.cheb_input(x))
#         for conv in self.cheb_layers:
#             xp = conv(xp, edge_index, batch)
#             xp = F.dropout(xp, p=self.dropout, training=self.training)
#         xp = self.apply_pooling(xp, batch)

#         # ----- Fusion -----
#         x_cat = torch.cat([xs, xp], dim=1)
#         out = self.classifier(x_cat)
#         return out


# class EEGGNN_Hybrid_old(nn.Module):
    # def __init__(self, in_channels=18, hidden_channels=64, num_classes=2,
    #              dropout=0.3, heads=4):
class EEGGNN_Hybrid_old(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 num_classes=2,
                 gat_layers=2, 
                 cheb_layers=2,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(EEGGNN_Hybrid_old, self).__init__()
        
        self.dropout = dropout
        
        # ----- Spatial branch (GAT) -----
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.bn_gat1 = BatchNorm(hidden_channels * heads)
        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.bn_gat2 = BatchNorm(hidden_channels)
        
        # ----- Spectral branch (Chebyshev Conv) -----
        self.cheb1 = ChebConv(in_channels, hidden_channels, K=3)
        self.bn_cheb1 = BatchNorm(hidden_channels)
        self.cheb2 = ChebConv(hidden_channels, hidden_channels, K=4)
        self.bn_cheb2 = BatchNorm(hidden_channels)
        
        # ----- Fusion -----
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x, edge_index, batch):
        # ----- Spatial path -----
        xs = F.relu(self.bn_gat1(self.gat1(x, edge_index)))
        xs = F.dropout(xs, p=self.dropout, training=self.training)
        xs = F.relu(self.bn_gat2(self.gat2(xs, edge_index)))
        xs = F.dropout(xs, p=self.dropout, training=self.training)
        xs = global_mean_pool(xs, batch)
        
        # ----- Spectral path -----
        xp = F.relu(self.bn_cheb1(self.cheb1(x, edge_index)))
        xp = F.dropout(xp, p=self.dropout, training=self.training)
        xp = F.relu(self.bn_cheb2(self.cheb2(xp, edge_index)))
        xp = F.dropout(xp, p=self.dropout, training=self.training)
        xp = global_mean_pool(xp, batch)
        
        # ----- Fusion -----
        x_cat = torch.cat([xs, xp], dim=1)
        out = self.classifier(x_cat)
        return out



# class EEGGNN_Hybrid_with_weight(nn.Module):
#     def __init__(self, in_channels=18, hidden_channels=64, num_classes=2,
#                  dropout=0.3, heads=4):
#         super(EEGGNN_Hybrid_with_weight, self).__init__()
        
#         self.dropout = dropout
        
#         # ----- Spatial branch (GAT) -----
#         # use edge_dim=1 to handle scalar edge_attr
#         self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, edge_dim=1)
#         self.bn_gat1 = BatchNorm(hidden_channels * heads)
#         self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, edge_dim=1)
#         self.bn_gat2 = BatchNorm(hidden_channels)
        
#         # ----- Spectral branch (Chebyshev Conv) -----
#         self.cheb1 = ChebConv(in_channels, hidden_channels, K=3)
#         self.bn_cheb1 = BatchNorm(hidden_channels)
#         self.cheb2 = ChebConv(hidden_channels, hidden_channels, K=4)
#         self.bn_cheb2 = BatchNorm(hidden_channels)
        
#         # ----- Fusion -----
#         self.classifier = nn.Linear(hidden_channels * 2, num_classes)

#     def forward(self, x, edge_index, batch, edge_attr=None):
#         # ----- Spatial path (GAT) -----
#         if edge_attr is not None and edge_attr.dim() == 1:
#             edge_attr = edge_attr.unsqueeze(-1)  # must be (num_edges, 1)

#         xs = F.relu(self.bn_gat1(self.gat1(x, edge_index, edge_attr=edge_attr)))
#         xs = F.dropout(xs, p=self.dropout, training=self.training)
#         xs = F.relu(self.bn_gat2(self.gat2(xs, edge_index, edge_attr=edge_attr)))
#         xs = F.dropout(xs, p=self.dropout, training=self.training)
#         xs = global_mean_pool(xs, batch)
        
#         # ----- Spectral path (ChebConv) -----
#         if edge_attr is not None:
#             edge_weight = edge_attr.squeeze()
#         else:
#             edge_weight = None

#         xp = F.relu(self.bn_cheb1(self.cheb1(x, edge_index, edge_weight=edge_weight)))
#         xp = F.dropout(xp, p=self.dropout, training=self.training)
#         xp = F.relu(self.bn_cheb2(self.cheb2(xp, edge_index, edge_weight=edge_weight)))
#         xp = F.dropout(xp, p=self.dropout, training=self.training)
#         xp = global_mean_pool(xp, batch)
        
#         # ----- Fusion -----
#         x_cat = torch.cat([xs, xp], dim=1)
#         out = self.classifier(x_cat)
#         return out

class EEGGNN_Hybrid_with_weight(nn.Module):
    def __init__(self, 
                 in_channels=18, 
                 hidden_channels=64, 
                 num_classes=2,
                 gat_layers=2, 
                 cheb_layers=2,
                 dropout=0.3, 
                 heads=4, 
                 edge_dim=1,
                 pooling="mean"):
        super(EEGGNN_Hybrid_with_weight, self).__init__()
        
        self.dropout = dropout
        self.pooling = pooling.lower()

        # ----- Spatial branch (GATv2) -----
        self.gat_input = nn.Linear(in_channels, hidden_channels)
        self.gat_layers = nn.ModuleList([
            Gat_block(hidden_channels=hidden_channels, concat=True, edge_dim=edge_dim, heads=heads)
            for _ in range(gat_layers)
        ])

        # ----- Spectral branch (ChebConv) -----
        self.cheb_input = nn.Linear(in_channels, hidden_channels)
        self.cheb_layers = nn.ModuleList([
            Cheb_block(hidden_channels=hidden_channels, K=3 + i)
            for i in range(cheb_layers)
        ])

        # ----- Fusion -----
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(self, x, edge_index, batch, edge_attr=None):
        if edge_attr is not None and edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        # ----- Spatial path (GATv2) -----
        xs = F.relu(self.gat_input(x))
        for conv in self.gat_layers:
            xs = conv(xs, edge_index, batch, edge_attr=edge_attr)
            xs = F.dropout(xs, p=self.dropout, training=self.training)
        xs = global_mean_pool(xs, batch)

        # ----- Spectral path (ChebConv) -----
        xp = F.relu(self.cheb_input(x))
        edge_weight = edge_attr.squeeze() if edge_attr is not None else None
        for conv in self.cheb_layers:
            xp = conv(xp, edge_index, batch, edge_attr=edge_weight)
            xp = F.dropout(xp, p=self.dropout, training=self.training)
        xp = global_mean_pool(xp, batch)

        # ----- Fusion -----
        x_cat = torch.cat([xs, xp], dim=1)
        out = self.classifier(x_cat)
        return out


        
# https://github.com/erinqhu/EEG-motor-imagery/blob/main/code/main_training_3GCN.ipynb
class GNN_ChebConv(nn.Module):
    def __init__(self, in_channels=18, dim1=32, dim2=64, dim3=128, num_classes=2, dropout=0.3):
        # Init parent
        super(GNN_ChebConv, self).__init__()
        # torch.manual_seed(42)

        # GCN layers
        self.conv1 = ChebConv(in_channels, dim1, K=3)
        self.conv2 = ChebConv(dim1, dim2, K=4)
        self.conv3 = ChebConv(dim2, dim3, K=5)
        self.bn1 = nn.BatchNorm1d(dim1)
        self.bn2 = nn.BatchNorm1d(dim2)
        self.bn3 = nn.BatchNorm1d(dim3)

        # Output layer
        self.dense = nn.Linear(dim3*2, num_classes)

    def forward(self, x, edge_index, batch_index, edge_weight):

        # Conv layers
        # hidden = self.conv1(x, edge_index, edge_weight)

        hidden = self.conv1(x, edge_index, edge_weight, lambda_max=2.0)
        hidden = self.bn1(hidden)
        hidden = F.relu(hidden)
        
        hidden = self.conv2(hidden, edge_index, edge_weight, lambda_max=2.0)
        # hidden = self.conv2(hidden, edge_index, edge_weight)
        hidden = self.bn2(hidden)
        hidden = F.relu(hidden)
        
        hidden = self.conv3(hidden, edge_index, edge_weight, lambda_max=2.0)
        # hidden = self.conv3(hidden, edge_index, edge_weight)
        hidden = self.bn3(hidden)
        hidden = F.relu(hidden)
        
        # Global Pooling (stack different aggregations)
        hidden = torch.cat([global_max_pool(hidden, batch_index), 
                            global_mean_pool(hidden, batch_index)], dim=1)
        
        # Apply a final (linear) classifier.
        out = self.dense(hidden)
        return out
        # return F.log_softmax(hidden, dim=1)

# https://github.com/neerajwagh/eeg-gcnn/blob/master/code_psd_deep_eeg_gcnn/EEGGraphConvNet.py
class EEGGraphConvNet(nn.Module):
    def __init__(self, in_channels, reduced_sensors=False):
        super(EEGGraphConvNet, self).__init__()
        
        # need these for train_model_and_visualize() function
        # self.sfreq = sfreq
        # self.batch_size = batch_size
        self.input_size = 8 if reduced_sensors else 64
        # self.conv1 = GCNConv(6, 16, improved=True, cached=True, normalize=False)
        # self.conv2 = GCNConv(16, 32, improved=True, cached=True, normalize=False)
        # self.conv3 = GCNConv(32, 64, improved=True, cached=True, normalize=False)
        # self.conv4 = GCNConv(64, 50, improved=True, cached=True, normalize=False)
        # self.conv4_bn = BatchNorm(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.conv1 = GCNConv(in_channels, 32, improved=True, cached=True, normalize=False)
        self.conv2 = GCNConv(32, 64, improved=True, cached=True, normalize=False)
        self.conv3 = GCNConv(64, 128, improved=True, cached=True, normalize=False)
        self.conv4 = GCNConv(128, 50, improved=True, cached=True, normalize=False)
        self.conv4_bn = BatchNorm(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.fc_block1 = nn.Linear(50, 30)
        self.fc_block2 = nn.Linear(30, 20)
        self.fc_block3 = nn.Linear(20, 2)

        # Xavier initializations  #init gcn layers
        self.fc_block1.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block2.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
        self.fc_block3.apply(lambda x: nn.init.xavier_normal_(x.weight, gain=1) if type(x) == nn.Linear else None)
    
    def forward(self, x, edge_index, batch, edge_weight, return_graph_embedding=False):
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv2(x, edge_index, edge_weight))

        x = F.leaky_relu(self.conv3(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv4_bn(self.conv4(x, edge_index, edge_weight)))
        out = global_add_pool(x, batch=batch)
        if return_graph_embedding:
            return out

        out = F.leaky_relu(self.fc_block1(out), negative_slope=0.01)
        out = F.dropout(out, p = 0.2, training=self.training)
        out = F.leaky_relu(self.fc_block2(out), negative_slope=0.01)
        out = self.fc_block3(out)

        return out


def mlp(in_dim, hidden_dim, out_dim, num_layers=2, dropout=0.0):
    layers = []
    dims = [in_dim] + [hidden_dim]*(num_layers-1) + [out_dim]
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers += [nn.BatchNorm1d(dims[i+1]),
                       nn.ReLU(inplace=True),
                       nn.Dropout(dropout)]
    return nn.Sequential(*layers)

class EEGGNN_GIN(nn.Module):
    def __init__(self,
                 in_channels=18,
                 hidden_channels=64,
                 num_classes=2,
                 num_layers=5,
                 eps_trainable=True,
                 readout='sum',   # 'sum' | 'mean' | 'max'
                 dropout=0.0,
                 residual=True,
                 mlp_layers=2):
        super().__init__()
        assert readout in ('sum', 'mean', 'max')

        self.readout = readout
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

        # Input projection
        self.in_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

        # GINConv layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GINConv(
                    nn=mlp(hidden_channels, hidden_channels, hidden_channels, num_layers=mlp_layers, dropout=dropout),
                    train_eps=eps_trainable
                )
            )
            self.norms.append(BatchNorm(hidden_channels))

        # Optional per-layer readout before pooling
        self.layer_readout_mlps = nn.ModuleList([
            mlp(hidden_channels, hidden_channels, hidden_channels, num_layers=2, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Classifier
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def _pool(self, x, batch):
        if self.readout == 'sum':
            return global_add_pool(x, batch)
        elif self.readout == 'mean':
            return global_mean_pool(x, batch)
        else:
            return global_max_pool(x, batch)

    def forward(self, x, edge_index, batch):
        x = self.in_proj(x)
        prev = x
        layer_graph_embeds = []

        for k, (conv, bn) in enumerate(zip(self.convs, self.norms)):
            h = conv(prev, edge_index)
            h = bn(h)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)
            if self.residual and h.shape == prev.shape:
                h = h + prev
            prev = h

            # Graph-level readout per layer
            hg = self.layer_readout_mlps[k](h)
            layer_graph_embeds.append(self._pool(hg, batch))

        # Sum readouts across layers
        g = torch.stack(layer_graph_embeds, dim=0).sum(dim=0)

        # Classifier
        out = self.classifier(g)
        return out

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=64, n_channels=19):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 32, (1, 20), (1, 10)),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 40), (1, 20)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(64, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x
class ClassificationHead(nn.Module):
    def __init__(self, num_classes, fc=512):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(fc, 512),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)  # Flatten except batch dimension
        out = self.fc(x)
        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        attn_out, attn_weights = self.self_attn(x, x, x)  # self-attention
        x = self.norm(x + self.dropout(attn_out))         # residual + norm
        pooled = x.mean(dim=1)                            # global mean pooling
        return pooled, attn_weights

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        weights = torch.softmax(self.attn(x), dim=1)      # (batch, seq_len, 1)
        context = torch.sum(weights * x, dim=1)           # weighted sum -> (batch, hidden_dim)
        return context, weights

class EEGLSTM(nn.Module):
    def __init__(self, emb_size=64, n_channels=19, n_classes=2,
                 lstm_hidden=128, num_layers=1, dropout=0.3, pooling="attention"):
        super().__init__()
        self.patch_embed = PatchEmbedding(emb_size, n_channels)
        self.lstm = nn.LSTM(
            emb_size, lstm_hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling

        if pooling == "attention":
            # self.attn = AttentionBlock(lstm_hidden)
            self.attn = SelfAttentionBlock(lstm_hidden, num_heads=4)
            fc_input = lstm_hidden
        elif pooling == "mean":
            fc_input = lstm_hidden
        elif pooling == "both":
            self.attn = AttentionBlock(lstm_hidden)
            fc_input = lstm_hidden * 2
        else:
            raise ValueError("pooling must be 'mean', 'attention', or 'both'")

        self.head = ClassificationHead(n_classes, fc=fc_input)

    def forward(self, x):
        # CNN patch embedding
        x = self.patch_embed(x)    # (batch, seq_len, emb_size)

        # LSTM encoding
        out, _ = self.lstm(x)      # (batch, seq_len, hidden_dim)

        # Pooling
        if self.pooling == "mean":
            pooled = out.mean(dim=1)
        elif self.pooling == "attention":
            pooled, _ = self.attn(out)
        elif self.pooling == "both":
            mean = out.mean(dim=1)
            attn_out, _ = self.attn(out)
            pooled = torch.cat([mean, attn_out], dim=-1)

        pooled = self.dropout(pooled)

        # Classification head
        out = self.head(pooled)
        return out