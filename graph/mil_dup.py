
from lib import *
from model import *
from data_utils import *
from graph_utils import *
from data_preparation import * 
from utils_all import *
from fake_label import *
from copy import deepcopy
import copy
import config
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Sequence
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from hypergraph import build_hypergraphs_from_master_region_topology, EEGHypergraphNet_basic, EEGHypergraphNet
from sklearn.manifold import TSNE
import pickle
from collections import Counter
from torch_geometric.nn import (
    GATConv,
    ChebConv,
    BatchNorm
)
import math
import random
import hashlib
from collections import defaultdict
from torch.utils.data import Dataset

def _stable_int_from_string(x: str) -> int:
    """
    Stable integer hash from a string.
    Do NOT use Python's built-in hash(), because it is randomized across runs.
    """
    s = str(x).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

# =========================================================
# Early stopping based on subject-level validation loss
# =========================================================
import os
import copy
from typing import Any, Dict, List, Optional


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
            if not hasattr(g, "summary_feat"):
                raise AttributeError("Graph is missing summary_feat. Run attach_summary_features_to_graphs(...) first.")
            sf = g.summary_feat
            if torch.is_tensor(sf):
                sf = sf.detach().cpu().numpy()
            all_summary.append(np.asarray(sf, dtype=np.float32))

    pyg_batch = Batch.from_data_list(all_graphs)

    out = {
        "pyg_batch": pyg_batch,
        "summary_x": torch.tensor(np.stack(all_summary, axis=0), dtype=torch.float32),
        "bag_sizes": torch.tensor(bag_sizes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "subject_ids": subject_ids,
    }

    if len(segment_ids_per_subject) > 0:
        out["segment_ids_per_subject"] = segment_ids_per_subject

    return out


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
class SubjectMILClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",   # "gnn" or "mlp_raw"

        # graph shape
        num_nodes: Optional[int] = None,
        edge_mode: Optional[str] = "topology_weighted",
        # GNN params
        gnn_hidden_dim: int = 64,

        # shared graph embedding dim
        graph_emb_dim: int = 128,

        # MIL / classifier
        attn_dim: int = 128,
        dropout: float = 0.2,

        # raw-MLP params
        node_hidden_dims: Sequence[int] = (256, 128),
        edge_hidden_dims: Sequence[int] = (128, 64),
        branch_emb_dim: int = 64,
        mil_pool_type: str = "gated",   # "gated" or "mean"
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()
        self.edge_mode = edge_mode.lower()

        if self.encoder_type == "gnn":
            self.encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif self.encoder_type == "hypergraph":
            self.encoder = EEGHypergraphNet(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                num_classes=3,
                dropout=dropout,
                use_batchnorm=False,
                use_attention=False,
                attention_heads=4,
                attention_mode="edge",
                readout="mean_sum_max",
            )

        elif self.encoder_type == "mlp_raw":
            if num_nodes is None:
                raise ValueError("num_nodes must be provided when encoder_type='mlp_raw'")

            self.encoder = RawNodeEdgeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                edge_hidden_dims=edge_hidden_dims,
                branch_emb_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
                use_upper_triangle=True,
                symmetrize_adj=True,
                edge_mode=edge_mode,
            )
        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type}")

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )
        else:
            raise ValueError(f"Unsupported mil_pool_type={mil_pool_type}")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict):
        graph_emb = self.encoder(batch_dict["pyg_batch"])   # [num_graphs, D]
        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        logits = self.classifier(bag_emb)

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "graph_emb": graph_emb,
            "attn_list": attn_list,
        }
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


# =========================================================
# Subject-level MIL classifier
# =========================================================
class SubjectMILEncoder(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        encoder_type: str = "gnn",   # "gnn" or "mlp_summary" hybrid, gat

        # GNN params
        gnn_hidden_dim: int = 64,

        # shared graph embedding dim
        graph_emb_dim: int = 128,

        # MIL / classifier
        attn_dim: int = 128,
        dropout: float = 0.2,
        readout: str = "mean", #max, sum
        # MLP-summary params
        summary_input_dim: int = None,
        mlp_hidden_dims: Sequence[int] = (128, 64),
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()

        if self.encoder_type == "gnn":
            self.encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif self.encoder_type == "mlp_summary":
            if summary_input_dim is None:
                raise ValueError("summary_input_dim must be provided when encoder_type='mlp_summary'")

            self.encoder = SummaryMLPEncoder(
                input_dim=summary_input_dim,
                hidden_dims=mlp_hidden_dims,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )

        elif self.encoder_type == "gat":
            self.encoder = GNNEncoder_GAT(
                in_channels=num_node_features,
                hidden_channels=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                num_layers=3,
                dropout=dropout,
                heads=4,
                edge_dim=1,
                pooling=readout
            )

        elif self.encoder_type == "hybrid":
            self.encoder = HybridGNNEncoder(
                 in_channels=num_node_features, 
                 hidden_channels=gnn_hidden_dim, 
                 emb_dim=graph_emb_dim,
                 gat_layers=2, 
                 cheb_layers=2,
                 dropout=dropout, 
                 heads=4, 
                 edge_dim=1,
                 pooling=readout)

        else:
            raise ValueError(f"Unsupported encoder_type={encoder_type}")

        # choose one MIL pool
        self.mil_pool = MeanMILPool()
        # self.mil_pool = GatedAttentionMIL(
        #     in_dim=graph_emb_dim,
        #     attn_dim=attn_dim,
        # )

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict):
        if self.encoder_type in ["gnn",'gat','hybrid']:
            graph_emb = self.encoder(batch_dict["pyg_batch"])      # [num_graphs, D]

        elif self.encoder_type == "mlp_summary":
            graph_emb = self.encoder(batch_dict["summary_x"])      # [num_graphs, D]

        else:
            raise ValueError(f"Unsupported encoder_type={self.encoder_type}")

        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])
        logits = self.classifier(bag_emb)

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "graph_emb": graph_emb,
            "attn_list": attn_list,
        }    
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    losses = []
    y_true = []
    y_pred = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad()
        out = model(batch)
        # print(out["graph_emb"].shape) 
        # print(out["bag_emb"].shape)
        # print(out["logits"].shape) 
        logits = out["logits"]
        avg_logits = logits.detach().mean(dim=0).cpu().numpy()
        # print("Avg logits:", avg_logits)
        labels = batch["labels"]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1)

        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_subject_metrics(y_true, y_pred)
    # metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred

    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics



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
):

    best_state = None
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir != "":
            os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        val_pred_counts = np.bincount(np.asarray(val_metrics["y_pred"], dtype=int), minlength=3)
        train_pred_counts = np.bincount(np.asarray(train_metrics["y_pred"], dtype=int), minlength=3)

        history.append({
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_acc": float(train_metrics["accuracy"]),
            "train_bal_acc": float(train_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_acc": float(val_metrics["accuracy"]),
            "val_bal_acc": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        })

        if epoch % 50 == 0:
            print(
                f"Epoch [{epoch:03d}/{epochs}] | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Train Bal Acc: {train_metrics['balanced_accuracy']:.4f} | "
                f"Train F1: {train_metrics['macro_f1']:.4f} || "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
            )
            print("Train pred counts:", train_pred_counts)
            print("Val pred counts:", val_pred_counts)

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = float(val_metrics["macro_f1"])
            best_epoch = epoch
            epochs_no_improve = 0

            # keep best model in memory
            best_state = {
                "epoch": int(epoch),
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "best_val_macro_f1": float(val_metrics["macro_f1"]),
                "best_val_loss": float(val_metrics["loss"]),
                "history": copy.deepcopy(history),
            }

            # save to disk
            if save_path is not None:
                torch.save(best_state, save_path)
                print(f"Saved best model at epoch {epoch} with val macro-F1 = {best_val_f1:.4f}")

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}, best val macro-F1 = {best_val_f1:.4f}")
            break

    # restore best weights from memory
    if best_state is not None:
        model.load_state_dict(best_state["model_state_dict"])

    final_val_metrics = evaluate(model, val_loader, criterion, device)
    return model, final_val_metrics, history, best_state

# def fit_mil_baseline(
#     model,
#     train_loader, 
#     val_loader,
#     optimizer,
#     criterion,
#     device,
#     epochs,
#     patience,
#     save_path=None,
#     start_epoch=0,
#     min_delta=0.0,
#     top_k=10,
#     verbose=True,
# ):
#     best_state = None
#     best_val_f1 = -1.0
#     best_epoch = 0
#     epochs_no_improve = 0
#     history = []

#     if save_path is not None:
#         save_dir = os.path.dirname(save_path)
#         if save_dir != "":
#             os.makedirs(save_dir, exist_ok=True)
#         ckpt_prefix = os.path.splitext(os.path.basename(save_path))[0]
#     else:
#         save_dir = None
#         ckpt_prefix = "mil_checkpoint"

#     early_stopper = EarlyStopping(
#         patience=patience,
#         start_epoch=start_epoch,
#         min_delta=min_delta,
#         top_k=top_k,
#         save_dir=save_dir,
#         verbose=verbose,
#         file_prefix=f"{ckpt_prefix}_topk",
#     )
#     for epoch in range(1, epochs + 1):
#         # important for deterministic-but-changing segment sampling
#         if hasattr(train_loader.dataset, "set_epoch"):
#             train_loader.dataset.set_epoch(epoch - 1)

#         train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
#         val_metrics = evaluate(model, val_loader, criterion, device)

#         val_pred_counts = np.bincount(np.asarray(val_metrics["y_pred"], dtype=int), minlength=3)
#         train_pred_counts = np.bincount(np.asarray(train_metrics["y_pred"], dtype=int), minlength=3)

#         history.append({
#             "epoch": int(epoch),
#             "train_loss": float(train_metrics["loss"]),
#             "train_acc": float(train_metrics["accuracy"]),
#             "train_bal_acc": float(train_metrics["balanced_accuracy"]),
#             "train_macro_f1": float(train_metrics["macro_f1"]),
#             "val_loss": float(val_metrics["loss"]),
#             "val_acc": float(val_metrics["accuracy"]),
#             "val_bal_acc": float(val_metrics["balanced_accuracy"]),
#             "val_macro_f1": float(val_metrics["macro_f1"]),
#         })

#         if epoch % 25 == 0:
#             print(
#                 f"Epoch [{epoch:03d}/{epochs}] | "
#                 f"Train Loss: {train_metrics['loss']:.4f} | "
#                 f"Train Acc: {train_metrics['accuracy']:.4f} | "
#                 f"Train Bal Acc: {train_metrics['balanced_accuracy']:.4f} | "
#                 f"Train F1: {train_metrics['macro_f1']:.4f} || "
#                 f"Val Loss: {val_metrics['loss']:.4f} | "
#                 f"Val Acc: {val_metrics['accuracy']:.4f} | "
#                 f"Val Bal Acc: {val_metrics['balanced_accuracy']:.4f} | "
#                 f"Val F1: {val_metrics['macro_f1']:.4f}"
#             )
#             print("Train pred counts:", train_pred_counts)
#             print("Val pred counts:", val_pred_counts)

#         should_stop = early_stopper(
#             model=model,
#             optimizer=optimizer,
#             epoch=epoch,
#             val_loss=val_metrics["loss"],
#             val_bal_acc=val_metrics["balanced_accuracy"],
#             val_macro_f1=val_metrics["macro_f1"],
#             extra_state={
#                 "history": copy.deepcopy(history),
#             },
#         )

#         if should_stop:
#             break

#     best_meta = early_stopper.get_best_checkpoint()
#     best_state = None

#     if best_meta is not None and best_meta.get("path") is not None:
#         best_state = torch.load(best_meta["path"], map_location=device)

#         # enrich returned state
#         best_state["top_k_checkpoints"] = early_stopper.get_topk_checkpoints()
#         best_state["selected_checkpoint"] = copy.deepcopy(best_meta)
#         best_state["selected_by"] = [
#             "max val_bal_acc",
#             "max val_macro_f1",
#             "min val_loss",
#         ]

#         # restore selected weights into current model
#         model.load_state_dict(best_state["model_state_dict"])

#         # keep old downstream pattern working:
#         # write final selected checkpoint back to the original save_path
#         if save_path is not None:
#             torch.save(best_state, save_path)
#             if verbose:
#                 print(f"Saved final selected checkpoint to: {save_path}")

#     elif verbose:
#         print("Warning: no checkpoint was selected. Model weights were not restored from disk.")

#     final_val_metrics = evaluate(model, val_loader, criterion, device)
#     return model, final_val_metrics, history, best_state

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



# def plot_subject_embeddings_tsne(subject_rows, level, output_dir, color_by="label", title="Subject Embeddings (t-SNE)", class_names=None):
#     X = np.stack([r["embedding"] for r in subject_rows], axis=0)

#     if color_by == "label":
#         c = np.array([r["label"] for r in subject_rows])
#     elif color_by == "pred":
#         c = np.array([r["pred"] for r in subject_rows])
#     else:
#         raise ValueError("color_by must be 'label' or 'pred'")


#     unique_classes = sorted(np.unique(c))

#     if class_names is None:
#         class_names = {cls: f"Class {cls}" for cls in unique_classes}


#     Z2 = TSNE(n_components=2, random_state=42, perplexity=min(10, len(subject_rows)-1)).fit_transform(X)

#     plt.figure(figsize=(6, 5))
#     # sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=c, s=80)

#     for cls in unique_classes:
#         idx = np.where(c == cls)[0]
#         plt.scatter(
#             Z2[idx, 0],
#             Z2[idx, 1],
#             s=80,
#             label=class_names.get(cls, f"Class {cls}")
#         )


#     for i, r in enumerate(subject_rows):
#         short_ids = str(r["subject_id"]).replace('sub-', 's')
#         plt.text(Z2[i, 0], Z2[i, 1], short_ids, fontsize=5)
#     save_path = os.path.join(output_dir, f"{level}_embeddings_tsne.png")
#     plt.title(title)
#     plt.xlabel("t-SNE-1")
#     plt.ylabel("t-SNE-2")
#     # plt.colorbar(sc, label=color_by)
#     plt.legend(title=color_by, loc="best")
#     plt.tight_layout()
#     plt.savefig(save_path)
#     # plt.show()
from sklearn.manifold import TSNE

def plot_subject_embeddings_tsne(
    subject_rows,
    level,
    split,
    output_dir,
    color_by,
    title,
    class_names=None,
    segment_point_size=14,
    subject_point_size=80,
):
    X = np.stack([r["embedding"] for r in subject_rows], axis=0)

    labels = np.array([r["label"] for r in subject_rows])
    subject_ids = [str(r["subject_id"]) for r in subject_rows]

    unique_label_ids = sorted(np.unique(labels))
    if class_names is None:
        class_names = {cls: f"Class {cls}" for cls in unique_label_ids}

    perplexity = min(10, len(subject_rows) - 1)
    if perplexity < 1:
        raise ValueError("Not enough rows for TSNE")

    Z2 = TSNE(
        n_components=2,
        random_state=42,
        perplexity=perplexity
    ).fit_transform(X)

    plt.figure(figsize=(9, 7))

    # ---------------------------------------------------
    # SUBJECT LEVEL: keep same behavior
    # ---------------------------------------------------
    if level == "subject":
        if color_by == "label":
            c = labels
        elif color_by == "pred":
            preds = np.array([r["pred"] for r in subject_rows])
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

        save_name = f"{split}_{level}_embeddings_tsne.png"

    # ---------------------------------------------------
    # SEGMENT LEVEL
    #   - smaller points
    #   - color by subject
    #   - legend includes subject + class
    # ---------------------------------------------------
    elif level == "segment":
        if color_by == "label":
            unique_classes = sorted(np.unique(labels))
            for cls in unique_classes:
                idx = np.where(labels == cls)[0]
                plt.scatter(
                    Z2[idx, 0],
                    Z2[idx, 1],
                    s=segment_point_size,
                    alpha=0.6,
                    label=class_names.get(cls, f"Class {cls}")
                )

        elif color_by == "subject":
            unique_subjects = sorted(set(subject_ids))

            # one color per subject
            cmap = plt.cm.get_cmap("gist_ncar", len(unique_subjects))

            # use one class label per subject
            subject_to_label = {}
            for r in subject_rows:
                sid = str(r["subject_id"])
                if sid not in subject_to_label:
                    subject_to_label[sid] = int(r["label"])

            for k, sid in enumerate(unique_subjects):
                idx = np.where(np.array(subject_ids) == sid)[0]
                cls = subject_to_label[sid]
                class_label_text = class_names.get(cls, f"Class {cls}")
                legend_text = f"{sid} - label {class_label_text}"

                plt.scatter(
                    Z2[idx, 0],
                    Z2[idx, 1],
                    s=segment_point_size,
                    alpha=0.7,
                    color=cmap(k),
                    label=legend_text
                )
        else:
            raise ValueError("For level='segment', color_by must be 'label', 'pred', or 'subject'")

        # no text annotation for every segment
        save_name = f"{split}_{level}_embeddings_tsne_{color_by}.png"

    else:
        raise ValueError("level must be 'subject' or 'segment'")

    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")

    # put legend outside because subject legend can be long
    plt.legend(
        title=color_by,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8
    )

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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import numpy as np

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



if __name__ == "__main__":




    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH

    feature_dim_dict = config.FEATURE_DIM_DICT
    k = 5
    val_ratio = 0.15
    split_seeds = [15, 42, 100]
    batch_size_train=8
    batch_size_val=4
    batch_size_test = 4
    lr=3e-4
    weight_decay=5e-4
    epochs=100
    patience=30
    start_epoch=20
    readout="sum"
    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--data_path", type=str, required=False, help="model_name")
    parser.add_argument("--mil_pool_type", type=str, required=False, help="mil_pool_type")
    parser.add_argument("--edge_mode", type=str, required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, required=False, help="topology")
    parser.add_argument("--base_k", type=int, default=None, required=False, help="base_k")
    parser.add_argument("--dim", type=int,  default=32, required=False, help="dim")
    # parser.add_argument("--model_name", type=str, required=False, help="model_name")
    args = parser.parse_args()
    data_path = args.data_path
    model_name= "mlp"
    class_set ="all3" 
    topology = args.topology#"fixed"
    mil_pool_type= args.mil_pool_type #"mean" #"mean"
    edge_mode = args.edge_mode #"topology_binary"
    dim = args.dim
    gnn_hidden_dim=dim
    graph_emb_dim=dim*2
    attn_dim=dim*2
    dropout=0.3
    # graph_emb_dim=128,
    # attn_dim=128,
    # dropout=0.2,
    node_hidden_dims=(dim*2, dim)
    edge_hidden_dims=(dim*2, dim)
    branch_emb_dim=dim
    base_k=args.base_k
    max_k_per_subject=300
    add_noise = False
    noise_ratio = None
    standardize_features=True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)

    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"
    save_path = os.path.join(root_path,'result_Mar31_MIL-LinkX/earlystoppingvalf1')
    os.makedirs(save_path,exist_ok = True)
    last_part = data_path
    parts = last_part.split('_')



    try:
        node_features = parts[1]
        weight_method = parts[2:]
        _ = get_feature_dim_from_string(feature_dim_dict, node_features)
    except ValueError:
        node_features = parts[0]
        weight_method = parts[1:3]
    
    if "mono" in parts:
        channel_names = config.MONO_CHANNELS
        fixed_edges = config.MONOFIXEDGES
    elif "bi23" in parts:
        channel_names = config.bi23_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
    elif "bi30" in parts:
        channel_names = config.bi30_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    feat, used_features = get_feature_dim_from_string(feature_dim_dict, node_features)
    folder_name = f"{timestamp}_{model_name}_{mil_pool_type}_{last_part}_{topology}_{k}folds"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")
    all_data_path = f"/mnt/data/anphan/AHEAP_data/all_master_graph_data/{data_path}/data_processed/master_graph_data.pt"

    if not os.path.exists(all_data_path):
        raise FileNotFoundError(f"Missing: {all_data_path}")
    if not os.path.exists(all_data_path):
        print(f"Skipping: {all_data_path} not found.")
        sys.exit(1) 

    print("File found! Processing...")
    with open(log_path, "w") as f:
        f.write(f"all_data_path: {all_data_path}\n")
        f.write(f"model_name: {model_name}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"edge_weight = None\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"{topology}, fixed_edges {fixed_edges}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta=1e-3, top_k=5 \n")
        f.write(f"k {k}, val_ratio {val_ratio}, split_seeds {split_seeds}\n")
        f.write(f"batch_size_train {batch_size_train}, batch_size_val {batch_size_val}, batch_size_test {batch_size_test}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        f.write(f"readout {readout}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"dim {dim} \n gnn_hidden_dim={gnn_hidden_dim} \n graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f"dropout={dropout}\n node_hidden_dims={node_hidden_dims} \n edge_hidden_dims={edge_hidden_dims}\n branch_emb_dim={branch_emb_dim}\n")
    


    result_all = []
    fold_metric_rows = []
    pred_rows = []
    graphs = build_graphs_from_master_topology(
        all_data_path,
        subject_ids=None,
        undirected=True,
        filter_method=topology,          # "MST", "fixed", "topk", "reconnect", "combined", "overlap"
        topk=None,
        top_percent=None,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        corruption_mode=None,         # None, "identity", "random", "permute_consistent", "permute_adj_only"
        standardize_features=True,
        label_key="class_id",         # use "segment_label" if later you add fake segment labels
    )


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

            train_graphs = [g for g in graphs if g.subject_id in new_train_subjects]
            val_graphs   = [g for g in graphs if g.subject_id in val_subjects]
            test_graphs  = [g for g in graphs if g.subject_id in set(test_subjects)]

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
            # train_dataset = SubjectBagGraphDataset(
            #     train_graphs,
            #     max_segments_per_subject=None,   # good default for training memory
            #     train=True,
            # )

            # val_dataset = SubjectBagGraphDataset(
            #     val_graphs,
            #     max_segments_per_subject=None, # use all segments at validation if memory allows
            #     train=False,
            # )
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
            if model_name in ["gnn","gat","hybrid"]:
                input_model = SubjectMILClassifier(
                    num_node_features=train_dataset.num_node_features,  # 8
                    num_classes=num_classes,
                    encoder_type=model_name,
                    num_nodes=train_dataset.num_nodes,
                    gnn_hidden_dim=gnn_hidden_dim,
                    graph_emb_dim=graph_emb_dim,
                    attn_dim=attn_dim,
                    dropout=dropout,
                    mil_pool_type=mil_pool_type,
                ).to(device)
            elif model_name == "mlp":
                input_model = SubjectMILClassifier(
                    num_node_features=train_dataset.num_node_features,  # 8
                    num_classes=num_classes,
                    encoder_type="mlp_raw",
                    num_nodes=train_dataset.num_nodes,
                    graph_emb_dim=graph_emb_dim,
                    attn_dim=attn_dim,
                    dropout=dropout,
                    node_hidden_dims=node_hidden_dims,
                    edge_hidden_dims=edge_hidden_dims,
                    branch_emb_dim=64,
                    mil_pool_type=mil_pool_type,
                    edge_mode=edge_mode
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

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size_train,
                shuffle=True,
                collate_fn=collate_subject_bags,
                num_workers=0,
                pin_memory=True,
            )

            batch = next(iter(train_loader))
            print(batch["pyg_batch"])
            print(batch["summary_x"].shape)   # [num_graphs_in_batch, summary_input_dim]
            print(batch["bag_sizes"])
            print(batch["labels"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
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
                # start_epoch=start_epoch,     # warmup: do not count patience before epoch 30
                # min_delta=1e-3,     # require at least this much val-loss improvement
                # top_k=5,            # keep 5 lowest-loss checkpoints
                # verbose=False,
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

            test_dataset = SubjectBagGraphDataset(
                test_graphs,
                max_segments_per_subject=None, # use all segments at validation if memory allows
                train=False,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size_test,
                shuffle=False,
                collate_fn=collate_subject_bags,
                num_workers=0,
                pin_memory=True,
            )
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
            plot_subject_embeddings_tsne(train_subject_rows_f, "subject", "train", tsne_fold, color_by="label", title="Train Subject Embeddings by True Class")
            plot_subject_embeddings_tsne(val_subject_rows_f, "subject", "val", tsne_fold, color_by="label", title="Validation Subject Embeddings by True Class")
            plot_subject_embeddings_tsne(test_subject_rows_f, "subject", "test", tsne_fold, color_by="label", title="Test Subject Embeddings by True Class")
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

            print("TRAIN fingerprint stats:", fingerprint_stats_train)
            print("TEST  fingerprint stats:", fingerprint_stats_test)



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