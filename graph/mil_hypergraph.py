
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
from hypergraph import build_hypergraphs_from_master_region_topology
from sklearn.manifold import TSNE
import pickle
from collections import Counter
from torch_geometric.nn import (
    GATConv,
    ChebConv,
    BatchNorm
)


from torch_geometric.nn import HypergraphConv
from torch_geometric.nn.aggr import AttentionalAggregation

class HypergraphMILSegmentEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        emb_dim: int = 128,
        dropout: float = 0.2,
        use_batchnorm: bool = True,
        use_attention: bool = True,
        attention_heads: int = 4,
        attention_mode: str = "node",   # "node" or "edge"
        readout: str = "mean_max",      # "mean", "max", "sum", "mean_max", "mean_sum_max", "attn"
    ):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention
        self.readout_type = readout
        self.dropout = nn.Dropout(dropout)

        self.conv1 = HypergraphConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            use_attention=use_attention,
            attention_mode=attention_mode,
            heads=attention_heads if use_attention else 1,
            concat=False,
            dropout=dropout if use_attention else 0.0,
        )

        self.conv2 = HypergraphConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            use_attention=use_attention,
            attention_mode=attention_mode,
            heads=attention_heads if use_attention else 1,
            concat=False,
            dropout=dropout if use_attention else 0.0,
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        if readout == "attn":
            gate_hidden = max(hidden_channels // 2, 1)
            self.attn_readout = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_channels, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, 1),
                )
            )
            readout_dim = hidden_channels
        elif readout == "mean_max":
            self.attn_readout = None
            readout_dim = hidden_channels * 2
        elif readout == "mean_sum_max":
            self.attn_readout = None
            readout_dim = hidden_channels * 3
        elif readout in {"mean", "max", "sum"}:
            self.attn_readout = None
            readout_dim = hidden_channels
        else:
            raise ValueError(f"Unsupported readout: {readout}")

        self.proj = nn.Sequential(
            nn.Linear(readout_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

    def _build_hyperedge_attr(self, x, hyperedge_index):
        if hyperedge_index.numel() == 0:
            return x.new_zeros((0, x.size(-1)))

        node_ids = hyperedge_index[0]
        hedge_ids = hyperedge_index[1]
        num_hyperedges = int(hedge_ids.max().item()) + 1

        hedge_attr = x.new_zeros((num_hyperedges, x.size(-1)))
        hedge_attr.index_add_(0, hedge_ids, x[node_ids])

        counts = x.new_zeros(num_hyperedges)
        counts.index_add_(0, hedge_ids, x.new_ones(hedge_ids.size(0)))
        hedge_attr = hedge_attr / counts.clamp_min(1).unsqueeze(-1)
        return hedge_attr

    def _apply_readout(self, x, batch):
        if self.readout_type == "mean":
            return global_mean_pool(x, batch)
        elif self.readout_type == "max":
            return global_max_pool(x, batch)
        elif self.readout_type == "sum":
            return global_add_pool(x, batch)
        elif self.readout_type == "mean_max":
            return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)
        elif self.readout_type == "mean_sum_max":
            return torch.cat([
                global_mean_pool(x, batch),
                global_add_pool(x, batch),
                global_max_pool(x, batch),
            ], dim=-1)
        elif self.readout_type == "attn":
            return self.attn_readout(x, index=batch)
        else:
            raise ValueError(f"Unsupported readout: {self.readout_type}")

    def forward(self, data_batch):
        x = data_batch.x
        hyperedge_index = data_batch.hyperedge_index
        hyperedge_weight = getattr(data_batch, "hyperedge_weight", None)
        batch = data_batch.batch

        hyperedge_attr = self._build_hyperedge_attr(x, hyperedge_index) if self.use_attention else None
        x = self.conv1(
            x,
            hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            hyperedge_attr=hyperedge_attr,
        )
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        hyperedge_attr = self._build_hyperedge_attr(x, hyperedge_index) if self.use_attention else None
        x = self.conv2(
            x,
            hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            hyperedge_attr=hyperedge_attr,
        )
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        g = self._apply_readout(x, batch)
        graph_emb = self.proj(g)
        return graph_emb
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
    """
    Output:
      pyg_batch: Batch of all segment-graphs from all subjects in this minibatch
      summary_x: stacked summary features for all segment-graphs
      bag_sizes: number of segment-graphs per subject
      labels: subject labels
      subject_ids: list[str]
    """
    all_graphs = []
    # all_summary = []
    bag_sizes = []
    labels = []
    subject_ids = []

    for item in batch:
        gs = item["graphs"]
        all_graphs.extend(gs)
        bag_sizes.append(len(gs))
        labels.append(int(item["label"]))
        subject_ids.append(item["subject_id"])

        # for g in gs:
        #     if not hasattr(g, "summary_feat"):
        #         raise AttributeError("Graph is missing summary_feat. Run attach_summary_features_to_graphs(...) first.")
        #     sf = g.summary_feat
        #     if torch.is_tensor(sf):
        #         sf = sf.detach().cpu().numpy()
        #     all_summary.append(np.asarray(sf, dtype=np.float32))

    pyg_batch = Batch.from_data_list(all_graphs)

    return {
        "pyg_batch": pyg_batch,
        # "summary_x": torch.tensor(np.stack(all_summary, axis=0), dtype=torch.float32),
        "bag_sizes": torch.tensor(bag_sizes, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "subject_ids": subject_ids,
    }


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

        self.encoder = HypergraphMILSegmentEncoder(
            in_channels=num_node_features,
            hidden_channels=gnn_hidden_dim,
            emb_dim=graph_emb_dim,
            dropout=dropout,
            use_batchnorm=False,
            use_attention=False,
            attention_heads=4,
            attention_mode="edge",
            readout="mean_sum_max",
        )

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



def plot_subject_embeddings_tsne(subject_rows, output_dir, color_by="label", title="Subject Embeddings (t-SNE)", class_names=None):
    X = np.stack([r["embedding"] for r in subject_rows], axis=0)

    if color_by == "label":
        c = np.array([r["label"] for r in subject_rows])
    elif color_by == "pred":
        c = np.array([r["pred"] for r in subject_rows])
    else:
        raise ValueError("color_by must be 'label' or 'pred'")


    unique_classes = sorted(np.unique(c))

    if class_names is None:
        class_names = {cls: f"Class {cls}" for cls in unique_classes}


    Z2 = TSNE(n_components=2, random_state=42, perplexity=min(10, len(subject_rows)-1)).fit_transform(X)

    plt.figure(figsize=(6, 5))
    # sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=c, s=80)

    for cls in unique_classes:
        idx = np.where(c == cls)[0]
        plt.scatter(
            Z2[idx, 0],
            Z2[idx, 1],
            s=80,
            label=class_names.get(cls, f"Class {cls}")
        )


    for i, r in enumerate(subject_rows):
        plt.text(Z2[i, 0], Z2[i, 1], str(r["subject_id"]), fontsize=8)
    save_path = os.path.join(output_dir, "subject_embeddings_tsne.png")
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    # plt.colorbar(sc, label=color_by)
    plt.legend(title=color_by, loc="best")
    plt.tight_layout()
    plt.savefig(save_path)
    # plt.show()



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
    test_loader,
    device,
    fold_idx,
    save_dir="cv_subject_embeddings"
):
    os.makedirs(save_dir, exist_ok=True)

    train_subject_rows_f = collect_subject_embeddings(model, train_loader, device)
    test_subject_rows_f  = collect_subject_embeddings(model, test_loader, device)

    train_pkl = os.path.join(save_dir, f"fold_{fold_idx}_train_subject_rows.pkl")
    train_csv = os.path.join(save_dir, f"fold_{fold_idx}_train_subject_rows.csv")

    test_pkl = os.path.join(save_dir, f"fold_{fold_idx}_test_subject_rows.pkl")
    test_csv = os.path.join(save_dir, f"fold_{fold_idx}_test_subject_rows.csv")

    save_subject_rows(train_subject_rows_f, train_pkl, train_csv)
    save_subject_rows(test_subject_rows_f, test_pkl, test_csv)

    print(f"Saved fold {fold_idx}:")
    print(" ", train_pkl)
    print(" ", test_pkl)

    return train_subject_rows_f, test_subject_rows_f

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

    epochs=400
    patience=200

    readout="sum"
    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--data_path", type=str, required=False, help="model_name")
    # parser.add_argument("--model_name", type=str, required=False, help="model_name")
    args = parser.parse_args()
    data_path = args.data_path
    model_name= "hypergraph"
    class_set ="all3" 
    topology = "region_hyperedges"
    mil_pool_type="mean" #"mean"
    edge_mode = "mean_abs_adj"
    gnn_hidden_dim=64
    graph_emb_dim=128
    attn_dim=128
    dropout=0.2
    # graph_emb_dim=128,
    # attn_dim=128,
    # dropout=0.2,
    node_hidden_dims=(256, 128)
    edge_hidden_dims=(128, 64)
    branch_emb_dim=64

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
    save_path = os.path.join(root_path,'result_Mar25_MIL-LinkX')
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
    all_data_path = f"/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/{data_path}/data_processed/master_graph_data.pt"

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
        f.write(f"{topology}, fixed_edges {fixed_edges}\n")
        f.write(f"k {k}, val_ratio {val_ratio}, split_seeds {split_seeds}\n")
        f.write(f"batch_size_train {batch_size_train}, batch_size_val {batch_size_val}, batch_size_test {batch_size_test}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        f.write(f"readout {readout}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f" gnn_hidden_dim={gnn_hidden_dim} \n graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f" dropout={dropout}\n node_hidden_dims={node_hidden_dims} \n edge_hidden_dims={edge_hidden_dims}\n branch_emb_dim={branch_emb_dim}\n")
    


    result_all = []
    fold_metric_rows = []
    pred_rows = []
    graphs = build_hypergraphs_from_master_region_topology(
        master_path=all_data_path,
        channel_names=channel_names,
        region_to_channels=config.region_hyperedges,   # or config.region_to_channels
        subject_ids=None,
        label_key="class_id",
        standardize_features=True,
        corruption_mode=None,
        hyperedge_weight_mode="mean_abs_adj",   # or "ones", "mean_adj"
    )
    # graphs = build_graphs_from_master_topology(
    #     all_data_path,
    #     subject_ids=None,
    #     undirected=True,
    #     filter_method=topology,          # "MST", "fixed", "topk", "reconnect", "combined", "overlap"
    #     topk=None,
    #     top_percent=None,
    #     fixed_edges=fixed_edges,
    #     channel_names=channel_names,
    #     corruption_mode=None,         # None, "identity", "random", "permute_consistent", "permute_adj_only"
    #     standardize_features=True,
    #     label_key="class_id",         # use "segment_label" if later you add fake segment labels
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

            train_graphs = [g for g in graphs if g.subject_id in new_train_subjects]
            val_graphs   = [g for g in graphs if g.subject_id in val_subjects]
            test_graphs  = [g for g in graphs if g.subject_id in set(test_subjects)]

            # train_graphs = attach_summary_features_to_graphs(train_graphs)
            # val_graphs   = attach_summary_features_to_graphs(val_graphs)
            # test_graphs  = attach_summary_features_to_graphs(test_graphs)

            g = train_graphs[0]
            print(hasattr(g, "edge_attr"))
            print(g.edge_attr[:10] if hasattr(g, "edge_attr") and g.edge_attr is not None else None)
            print(hasattr(g, "edge_weight"))
            print(g.edge_weight[:10] if hasattr(g, "edge_weight") and g.edge_weight is not None else None)


            # summary_input_dim = train_graphs[0].summary_feat.numel()
            # print("summary_input_dim =", summary_input_dim)
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

            print("Train subject class counts:", np.bincount(train_dataset.subject_labels, minlength=num_classes))
            print("Val subject class counts:", np.bincount(val_dataset.subject_labels, minlength=num_classes))
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            # if model_name in ["gnn","gat","hybrid"]:
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
            # elif model_name == "mlp":
            #     input_model = SubjectMILClassifier(
            #         num_node_features=train_dataset.num_node_features,  # 8
            #         num_classes=num_classes,
            #         encoder_type="mlp_raw",
            #         num_nodes=train_dataset.num_nodes,
            #         graph_emb_dim=graph_emb_dim,
            #         attn_dim=attn_dim,
            #         dropout=dropout,
            #         node_hidden_dims=node_hidden_dims,
            #         edge_hidden_dims=edge_hidden_dims,
            #         branch_emb_dim=64,
            #         mil_pool_type=mil_pool_type,
            #         edge_mode=edge_mode
            #     ).to(device)
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
            # print(batch["summary_x"].shape)   # [num_graphs_in_batch, summary_input_dim]
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
            )

            checkpoint = torch.load(f"{check_dir}/best_mil_model_fold{i}.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            # print("Best epoch:", checkpoint["epoch"])
            # print("Best val metrics:", checkpoint["best_val_macro_f1"])


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

            train_subject_rows_f, test_subject_rows_f = save_fold_subject_embeddings(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    device=device,
                    fold_idx=i,
                    save_dir=cv_subject_embeddings
                )

            all_fold_data.append({
                "fold": i,
                "train_rows": train_subject_rows_f,
                "test_rows": test_subject_rows_f,
            })
            plot_subject_embeddings_tsne(test_subject_rows_f, tsne_fold, color_by="label", title="Test Subject Embeddings by True Class")
            fold_metric_rows.append(metrics_to_row(test_metrics, seed, i, "test"))
            pred_rows.extend(predictions_to_rows(test_metrics, seed, i, "test", num_classes))

            train_seg_rows = collect_segment_embeddings(model, train_loader, device)
            test_seg_rows  = collect_segment_embeddings(model, test_loader, device)

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