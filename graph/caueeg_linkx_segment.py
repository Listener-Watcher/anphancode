from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse

from caueeg_loader_min import load_caueeg_task_datasets
from master_builder import build_master_eeg_dataset
from mil_full_std import RawNodeEdgeMLPEncoder, RawNodeMLPEncoder, load_h5_payload_for_subjects
from mil_utils import GCNIIEncoder, GNNEncoder, GraphSAGEEncoder, H2GCNLikeEncoder
from caueeg.baseline_eval_utils import aggregate_predictions_by_recording, plot_confusion, prediction_df_to_metrics
from utils_all import set_global_seed

from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags, build_graphs_from_payload_multiband, build_graphs_from_payload, collate_subject_bags_multiband

# ---------------------------------------------------------
# CAUEEG channel order: keep only first 19 EEG channels
# ---------------------------------------------------------
CAUEEG_EEG19 = [
    "Fp1", "F3", "C3", "P3", "O1",
    "Fp2", "F4", "C4", "P4", "O2",
    "F7", "T3", "T5", "F8", "T4",
    "T6", "FZ", "CZ", "PZ",
]

SFREQ = 200.0
CROP_LEN = 2000      # 10 sec at 200 Hz
LATENCY = 2000       # skip first 10 sec like CEEDNet
OVERLAP = 0.5
STEP = int(CROP_LEN * (1.0 - OVERLAP))


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
# ---------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------
def segment_recording(
    signal: np.ndarray,
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Convert one CAUEEG recording [21, T] into fixed windows using the same
    10 s / 50% overlap / first-10 s latency convention used in caueeg_linkx_mil.py.
    """
    x = np.asarray(signal, dtype=np.float32)
    x = x[:19]  # drop EKG + photic

    total_len = x.shape[-1]
    starts = list(range(latency, total_len - crop_len + 1, step))
    windows = [x[:, s:s + crop_len].astype(np.float32, copy=False) for s in starts]
    return windows, starts



def make_split_subject_id(serial: str, split_name: str, use_split_prefix: bool = True) -> str:
    serial = str(serial)
    if not use_split_prefix:
        return serial
    return f"{split_name}_{serial}"



def dataset_to_subject_records(
    dataset,
    split_name: str,
    use_split_prefix: bool = True,
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Convert a CAUEEG split into records accepted by build_master_eeg_dataset().
    One recording becomes one subject-like entry in the H5, with all its fixed windows.
    """
    records: List[Dict[str, Any]] = []
    subject_ids: List[str] = []

    for sample in dataset:
        signal = sample["signal"]
        serial = str(sample["serial"])
        subject_id = make_split_subject_id(serial, split_name, use_split_prefix=use_split_prefix)
        label = int(sample["class_label"])
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal, crop_len=crop_len, step=step, latency=latency)
        if len(windows) == 0:
            continue

        rec = {
            "subject_id": subject_id,
            "label": label,
            "class_id": label,
            "sampling_rate": SFREQ,
            "channel_names": CAUEEG_EEG19,
            "windows": windows,
            "start_samples": starts,
            "segment_ids": list(range(len(windows))),
            "recording_info": {
                "serial": serial,
                "age": age,
                "split_name": split_name,
            },
        }
        records.append(rec)
        subject_ids.append(subject_id)

    return records, subject_ids



def payload_to_graphs(
    payload: Dict[str, Dict[str, Any]],
    subject_ids: Sequence[str],
    feature_families: Sequence[str],
    connectivity_metric: str = "pearson",
    connectivity_band: Optional[int] = None,
    standardize_features: bool = True,
) -> List[Data]:
    """
    Turn H5 payload entries into one PyG graph per stored window.
    This is the same graph construction logic used in caueeg_linkx_mil.py,
    but here the graphs are used directly as segment-level training samples.
    """
    graphs: List[Data] = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")

        subj = payload[sid]
        y = int(subj["label"])

        x_all = np.concatenate(
            [np.asarray(subj["features"][fam], dtype=np.float32) for fam in feature_families],
            axis=-1,
        )

        adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)
        if adj_all.ndim == 4:
            if connectivity_band is None:
                raise ValueError("connectivity_band must be set for banded connectivity")
            adj_all = adj_all[:, connectivity_band]

        seg_ids = np.asarray(subj["segment_id"], dtype=np.int64)
        start_samples = np.asarray(subj["start_sample"], dtype=np.int64)

        for w in range(x_all.shape[0]):
            x = x_all[w]
            adj = adj_all[w]

            if standardize_features:
                mu = x.mean(axis=0, keepdims=True)
                sd = x.std(axis=0, keepdims=True)
                x = (x - mu) / (sd + 1e-8)

            adj = np.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
            adj = 0.5 * (adj + adj.T)
            np.fill_diagonal(adj, 0.0)

            edge_index, edge_weight = dense_to_sparse(torch.tensor(adj, dtype=torch.float32))

            g = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=edge_index.long(),
                y=torch.tensor([y], dtype=torch.long),
            )
            g.edge_weight = edge_weight.float()
            g.edge_attr = edge_weight.view(-1, 1).float()
            g.adj = torch.tensor(adj, dtype=torch.float32)
            g.subject_id = str(sid)
            g.serial = str(sid)
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])
            graphs.append(g)

    return graphs


# ---------------------------------------------------------
# Segment datasets / collate
# ---------------------------------------------------------
class GraphSegmentDataset(Dataset):
    def __init__(self, graphs: Sequence[Data]):
        self.graphs = list(graphs)
        if len(self.graphs) == 0:
            raise ValueError("GraphSegmentDataset received an empty graph list")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        return self.graphs[idx]



def select_first_graph_per_subject(graphs: Sequence[Data]) -> List[Data]:
    out: List[Data] = []
    seen = set()
    graphs_sorted = sorted(
        graphs,
        key=lambda g: (str(getattr(g, "subject_id", "")), int(getattr(g, "start_sample", 0))),
    )
    for g in graphs_sorted:
        sid = str(g.subject_id)
        if sid in seen:
            continue
        out.append(g)
        seen.add(sid)
    return out



def collate_graph_segments(batch: Sequence[Data]) -> Dict[str, Any]:
    pyg_batch = Batch.from_data_list(list(batch))
    labels = torch.tensor([int(g.y.view(-1)[0].item()) for g in batch], dtype=torch.long)
    subject_ids = [str(getattr(g, "subject_id", "")) for g in batch]
    segment_ids = [int(getattr(g, "segment_id", -1)) for g in batch]
    start_samples = [int(getattr(g, "start_sample", -1)) for g in batch]

    return {
        "pyg_batch": pyg_batch,
        "labels": labels,
        "subject_ids": subject_ids,
        "serials": subject_ids,
        "segment_ids": segment_ids,
        "start_samples": start_samples,
    }



def move_batch_to_device(batch: Dict[str, Any], device: str | torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif hasattr(v, "to"):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
class SegmentGraphClassifier(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        num_nodes: int,
        encoder_type: str = "linkx",
        graph_emb_dim: int = 64,
        dropout: float = 0.3,
        gnn_hidden_dim: int = 64,
        node_hidden_dims: Sequence[int] = (64, 32),
        edge_hidden_dims: Sequence[int] = (64, 32),
        branch_emb_dim: int = 64,
        edge_mode: str = "topology_weighted",
    ):
        super().__init__()
        encoder_type = encoder_type.lower()
        self.encoder_type = encoder_type

        if encoder_type == "linkx":
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
        elif encoder_type == "mlp_node":
            self.graph_encoder = RawNodeMLPEncoder(
                num_nodes=num_nodes,
                num_node_features=num_node_features,
                node_hidden_dims=node_hidden_dims,
                proj_dim=branch_emb_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )
        elif encoder_type == "gnn":
            self.graph_encoder = GNNEncoder(
                in_dim=num_node_features,
                hidden_dim=gnn_hidden_dim,
                emb_dim=graph_emb_dim,
                dropout=dropout,
            )
        elif encoder_type == "sage":
            self.graph_encoder = GraphSAGEEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=2,
                dropout=dropout,
                pool="mean",
                jk_mode="last",
            )
        elif encoder_type == "gcn2":
            self.graph_encoder = GCNIIEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=8,
                dropout=dropout,
                alpha=0.1,
                theta=0.5,
                shared_weights=True,
                pool="mean",
                use_edge_weight=True,
            )
        elif encoder_type == "h2gcn":
            self.graph_encoder = H2GCNLikeEncoder(
                num_node_features=num_node_features,
                hidden_dim=gnn_hidden_dim,
                graph_emb_dim=graph_emb_dim,
                num_layers=2,
                dropout=dropout,
                pool="mean",
            )
        else:
            raise ValueError(
                f"Unknown encoder_type={encoder_type!r}. "
                f"Choose from ['linkx', 'mlp_node', 'gnn', 'sage', 'gcn2', 'h2gcn']"
            )

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        graph_emb = self.graph_encoder(batch_dict["pyg_batch"])
        logits = self.classifier(graph_emb)
        return {"graph_emb": graph_emb, "logits": logits}


# ---------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------
@dataclass
class EarlyStopper:
    patience: int = 30
    min_delta: float = 0.0
    best_loss: float = np.inf
    best_epoch: int = -1
    bad_epochs: int = 0
    best_state: Optional[Dict[str, Any]] = None

    def step(self, epoch: int, val_loss: float, model: nn.Module) -> bool:
        improved = float(val_loss) < (float(self.best_loss) - float(self.min_delta))
        if improved:
            self.best_loss = float(val_loss)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience



def compute_classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "conf_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
) -> Dict[str, Any]:
    model.train()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch)
        logits = out["logits"]
        labels = batch["labels"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics



def evaluate_segment_loader(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
) -> Dict[str, Any]:
    model.eval()
    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    subject_ids: List[str] = []
    segment_ids: List[int] = []
    start_samples: List[int] = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            out = model(batch)
            logits = out["logits"]
            labels = batch["labels"]
            loss = criterion(logits, labels)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            losses.append(float(loss.item()))
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_prob.extend(probs.detach().cpu().numpy().tolist())
            subject_ids.extend([str(x) for x in batch["subject_ids"]])
            segment_ids.extend([int(x) for x in batch["segment_ids"]])
            start_samples.extend([int(x) for x in batch["start_samples"]])

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["y_prob"] = y_prob
    metrics["subject_ids"] = subject_ids
    metrics["segment_ids"] = segment_ids
    metrics["start_samples"] = start_samples
    return metrics



def fit_segment_baseline(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str | torch.device,
    epochs: int = 100,
    patience: int = 30,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    stopper = EarlyStopper(patience=patience, min_delta=0.0)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_segment_loader(model, val_loader, criterion, device)

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "train_balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "train_macro_f1": float(train_metrics["macro_f1"]),
            "val_loss": float(val_metrics["loss"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
        }
        history.append(row)

        if verbose:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.4f}, "
                f"bal_acc={train_metrics['balanced_accuracy']:.4f}, macro_f1={train_metrics['macro_f1']:.4f} | "
                f"val loss={val_metrics['loss']:.4f}, acc={val_metrics['accuracy']:.4f}, "
                f"bal_acc={val_metrics['balanced_accuracy']:.4f}, macro_f1={val_metrics['macro_f1']:.4f}"
            )

        should_stop = stopper.step(epoch, float(val_metrics["loss"]), model)
        if save_path is not None and stopper.best_epoch == epoch:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_state_dict": stopper.best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": float(stopper.best_loss),
                },
                save_path,
            )

        if should_stop:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_loss={stopper.best_loss:.6f} at epoch {stopper.best_epoch}."
                )
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)

    best_val_metrics = evaluate_segment_loader(model, val_loader, criterion, device)
    best_state = stopper.best_state
    return model, best_val_metrics, history, best_state


# ---------------------------------------------------------
# Prediction tables / saving
# ---------------------------------------------------------
def metrics_to_prediction_df(metrics: Dict[str, Any], num_classes: int) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sid, seg_id, st, y_true, y_pred, probs in zip(
        metrics["subject_ids"],
        metrics["segment_ids"],
        metrics["start_samples"],
        metrics["y_true"],
        metrics["y_pred"],
        metrics["y_prob"],
    ):
        rec: Dict[str, Any] = {
            "subject_id": str(sid),
            "serial": str(sid),
            "segment_id": int(seg_id),
            "start_sample": int(st),
            "true_label": int(y_true),
            "pred_label": int(y_pred),
        }
        for c in range(int(num_classes)):
            rec[f"prob_{c}"] = float(probs[c])
        rows.append(rec)
    return pd.DataFrame(rows)



def save_segment_outputs(
    output_dir: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    multicrop_test_loader: DataLoader,
    criterion: nn.Module,
    device: str | torch.device,
    config: Dict[str, Any],
    history_rows: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)

    num_classes = int(config["out_dims"])
    task = config.get("task", None)
    class_label_to_name = config.get("class_label_to_name", None)
    if isinstance(class_label_to_name, dict):
        class_names = [
            str(class_label_to_name.get(i, class_label_to_name.get(str(i), f"class_{i}")))
            for i in range(num_classes)
        ]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    extra = {
        "file_format": config.get("file_format"),
        "model_name": config.get("model_name", config.get("encoder_type", config.get("model", None))),
        "test_crop_multiple": config.get("test_crop_multiple", None),
        "connectivity_metric": config.get("connectivity_metric", None),
        "connectivity_band": config.get("connectivity_band", None),
    }

    train_metrics = evaluate_segment_loader(model, train_loader, criterion, device)
    val_metrics = evaluate_segment_loader(model, val_loader, criterion, device)
    test_single_metrics = evaluate_segment_loader(model, test_loader, criterion, device)
    test_multi_metrics = evaluate_segment_loader(model, multicrop_test_loader, criterion, device)

    train_pred_df = metrics_to_prediction_df(train_metrics, num_classes=num_classes)
    val_pred_df = metrics_to_prediction_df(val_metrics, num_classes=num_classes)
    test_single_df = metrics_to_prediction_df(test_single_metrics, num_classes=num_classes)
    test_multi_crop_df = metrics_to_prediction_df(test_multi_metrics, num_classes=num_classes)
    test_multi_recording_df = aggregate_predictions_by_recording(test_multi_crop_df, num_classes)

    train_pred_df.to_csv(os.path.join(output_dir, "train_predictions_segment.csv"), index=False)
    val_pred_df.to_csv(os.path.join(output_dir, "val_predictions_segment.csv"), index=False)
    test_single_df.to_csv(os.path.join(output_dir, "test_predictions_singlecrop_segment.csv"), index=False)
    test_multi_crop_df.to_csv(os.path.join(output_dir, "test_predictions_multicrop_crop_level.csv"), index=False)
    test_multi_recording_df.to_csv(os.path.join(output_dir, "test_predictions_multicrop_recording.csv"), index=False)

    summary_rows = [
        prediction_df_to_metrics(train_pred_df, "train_segment", task=task, extra=extra),
        prediction_df_to_metrics(val_pred_df, "val_segment", task=task, extra=extra),
        prediction_df_to_metrics(test_single_df, "test_singlecrop_segment", task=task, extra=extra),
        prediction_df_to_metrics(test_multi_recording_df, "test_multicrop_recording", task=task, extra=extra),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    if history_rows is not None:
        pd.DataFrame(history_rows).to_csv(os.path.join(output_dir, "history.csv"), index=False)

    with open(os.path.join(output_dir, "metrics_test_singlecrop_segment.json"), "w") as f:
        json.dump(summary_rows[2], f, indent=2)
    with open(os.path.join(output_dir, "metrics_test_multicrop_recording.json"), "w") as f:
        json.dump(summary_rows[3], f, indent=2)

    plot_confusion(
        test_single_df,
        class_names=class_names,
        save_path=os.path.join(output_dir, "test_singlecrop_segment_confusion.png"),
        title="Single-Crop Segment-Level Test Confusion",
    )
    plot_confusion(
        test_multi_recording_df,
        class_names=class_names,
        save_path=os.path.join(output_dir, "test_multicrop_recording_confusion.png"),
        title="Multi-Crop Recording-Level Test Confusion",
    )

    print(f"Saved outputs to: {output_dir}")
    print(summary_df)

    return {
        "train_pred_df": train_pred_df,
        "val_pred_df": val_pred_df,
        "test_single_df": test_single_df,
        "test_multi_crop_df": test_multi_crop_df,
        "test_multi_recording_df": test_multi_recording_df,
        "summary_df": summary_df,
    }


# ---------------------------------------------------------
# Main runner
# ---------------------------------------------------------
def run_caueeg_linkx_segment(
    dataset_path: str,
    fixed_edges,          
    channel_names,
    feature_families = ['relative_band_power', 'statistical', 'wavelet_energy'],
    filter_method="fixed",
    task: str = "dementia",
    file_format: str = "feather",
    out_h5: str = "caueeg_master_linkx.h5",
    connectivity_metric: str = "pearson",
    connectivity_band: Optional[int] = None,
    batch_size: int = 64,
    test_batch_size: int = 128,
    epochs: int = 100,
    seed: int =42,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    rebuild_h5: bool = False,
    output_root: str = "graph/results_caueeg_linkx_segment",
    encoder_type: str = "linkx",
    graph_emb_dim: int = 64,
    dropout: float = 0.3,
    gnn_hidden_dim: int = 64,
    node_hidden_dims: Sequence[int] = (64, 32),
    edge_hidden_dims: Sequence[int] = (64, 32),
    branch_emb_dim: int = 64,
    edge_mode: str = "topology_weighted",
    use_split_prefix: bool = True,
    bad_serials: Optional[Sequence[str]] = {"00587", "00781", "01301"},
    crop_len: int = CROP_LEN,
    step: int = STEP,
    latency: int = LATENCY,
) -> Dict[str, Any]:
    os.makedirs(output_root, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_linkxseg_{encoder_type}_{connectivity_metric}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"log.txt")

    if encoder_type in ['linkx_cnn','linkx_cnn5', 'LINKX', 'mlp_node']:
        patience=100
        epochs=200
        # lr=1e-3
        # weight_decay=5e-3

    else:
        patience=100
        epochs=300
        lr=3e-3
        weight_decay=3e-5

    set_global_seed(seed)


    with open(log_path, "w") as f:
        f.write(f"data source {out_h5}, task {task}, file_format {file_format}\n")
        f.write(f"seeds {seed}\n")
        # f.write(f"norm_mode {args.norm_mode}\n")
        # f.write(f"note: update - use topology instead of full adj\n")
        f.write(f"note: bad_ids {bad_serials} \n")

        f.write(f"topology: {filter_method}, fixed_edges: {fixed_edges}, channel_names: {channel_names}\n")
        f.write(f"feature_families: {feature_families}\nconnectivity_metric: {connectivity_metric}, connectivity_band: {connectivity_band}\n")
        f.write(f"\n")

        f.write(f"model_name: {encoder_type}, edge_mode: {edge_mode}\n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        # f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"graph_emb_dim={graph_emb_dim} \n")
        f.write(f"dropout={dropout}\n")
    

    # 1) official split
    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    # 2) convert each recording into subject-like records for H5 building
    train_records, train_ids = dataset_to_subject_records(
        train_set,
        split_name="train",
        use_split_prefix=use_split_prefix,
        crop_len=crop_len,
        step=step,
        latency=latency,
    )
    val_records, val_ids = dataset_to_subject_records(
        val_set,
        split_name="val",
        use_split_prefix=use_split_prefix,
        crop_len=crop_len,
        step=step,
        latency=latency,
    )
    test_records, test_ids = dataset_to_subject_records(
        test_set,
        split_name="test",
        use_split_prefix=use_split_prefix,
        crop_len=crop_len,
        step=step,
        latency=latency,
    )

    if bad_serials is not None:
        bad_serials = {str(x) for x in bad_serials}
        print(bad_serials)
        def _keep_subject(sid: str) -> bool:
            sid = str(sid)
            bare = sid.split("_", 1)[1] if (use_split_prefix and "_" in sid) else sid
            return bare not in bad_serials and sid not in bad_serials

        train_records = [r for r in train_records if _keep_subject(r["subject_id"])]
        val_records = [r for r in val_records if _keep_subject(r["subject_id"])]
        test_records = [r for r in test_records if _keep_subject(r["subject_id"])]
        train_ids = [sid for sid in train_ids if _keep_subject(sid)]
        val_ids = [sid for sid in val_ids if _keep_subject(sid)]
        test_ids = [sid for sid in test_ids if _keep_subject(sid)]

    all_records = train_records + val_records + test_records
    all_ids = train_ids + val_ids + test_ids
    num_classes = len(sorted({r["label"] for r in all_records}))
    feature_families = ["relative_band_power", "hjorth", "statistical"]

    # 3) build or reuse H5
    need_build = rebuild_h5 or (not os.path.isfile(out_h5))
    if need_build:
        print(f"[H5] Building master file: {out_h5}")
        build_master_eeg_dataset(
            subject_records=all_records,
            output_h5_path=out_h5,
            feature_families=feature_families,
            connectivity_metrics=[connectivity_metric],
            overwrite=True,
            skip_bad_segments=False,
            target_sampling_rate=None,
            qc_input_unit="auto",
        )
    else:
        print(f"[H5] Reusing existing master file: {out_h5}")

    # 4) load payload
    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=all_ids,
        feature_families=feature_families,
        connectivity_metrics=[connectivity_metric],
        connectivity_band=connectivity_band,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )
    if encoder_type != "linkx_cnn5":


        train_graphs = build_graphs_from_payload(
            payload, train_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )
        val_graphs = build_graphs_from_payload(
            payload, val_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )
        test_graphs = build_graphs_from_payload(
            payload, test_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )

    else:

        train_graphs = build_graphs_from_payload_multiband(
            payload, train_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )
        val_graphs = build_graphs_from_payload_multiband(
            payload, val_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )
        test_graphs = build_graphs_from_payload_multiband(
            payload, test_ids,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )


    # 6) segment-level datasets / loaders
    train_dataset = GraphSegmentDataset(train_graphs)
    val_dataset = GraphSegmentDataset(val_graphs)
    test_single_dataset = GraphSegmentDataset(select_first_graph_per_subject(test_graphs))
    multicrop_test_dataset = GraphSegmentDataset(test_graphs)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_single_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )
    multicrop_test_loader = DataLoader(
        multicrop_test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_graph_segments,
        num_workers=0,
        pin_memory=True,
    )

    num_node_features = int(train_graphs[0].x.shape[-1])
    num_nodes = int(train_graphs[0].x.shape[0])

    model = SegmentGraphClassifier(
        num_node_features=num_node_features,
        num_classes=num_classes,
        num_nodes=num_nodes,
        encoder_type=encoder_type,
        graph_emb_dim=graph_emb_dim,
        dropout=dropout,
        gnn_hidden_dim=gnn_hidden_dim,
        node_hidden_dims=node_hidden_dims,
        edge_hidden_dims=edge_hidden_dims,
        branch_emb_dim=branch_emb_dim,
        edge_mode=edge_mode,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ckpt_path = os.path.join(run_dir, "best_model.pt")

    model, val_metrics, history, best_state = fit_segment_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        patience=100,
        save_path=ckpt_path,
        verbose=True,
    )

    train_metrics = evaluate_segment_loader(model, train_loader, criterion, device)
    val_metrics = evaluate_segment_loader(model, val_loader, criterion, device)
    test_single_metrics = evaluate_segment_loader(model, test_loader, criterion, device)
    test_multi_metrics = evaluate_segment_loader(model, multicrop_test_loader, criterion, device)

    # config used by save_segment_outputs
    config = dict(config)
    config.update(
        {
            "out_dims": num_classes,
            "encoder_type": encoder_type,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "file_format": file_format,
            "test_crop_multiple": int(sum(g.subject_id == sid for g in test_graphs) for sid in []) if False else None,
        }
    )

    saved = save_segment_outputs(
        output_dir=run_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        multicrop_test_loader=multicrop_test_loader,
        criterion=criterion,
        device=device,
        config=config,
        history_rows=history,
    )


    summary_test = [
        {
            "encoder_type": encoder_type,
            "training_approach": "segment-training",
            "accuracy": float(test_multi_metrics["accuracy"]),
            "balanced_accuracy": float(test_multi_metrics["balanced_accuracy"]),
            "macro_f1": float(test_multi_metrics["macro_f1"]),
            "confusion_matrix": test_multi_metrics["conf_matrix"],
            "feature_families": feature_families,
            "topology": filter_method,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "edge_mode": edge_mode,
            "base_k": "None",
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "start_epoch": "None",
            "lr": lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "graph_emb_dim": graph_emb_dim,
            "attn_dim": "None",
            "seed": seed,            
        },
    ]

    df = pd.DataFrame(summary_test)
    csv_path = os.path.join(run_dir,"summary_test.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved summary metrics: {csv_path}")


    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "multicrop_test_loader": multicrop_test_loader,
        "history": history,
        "best_state": best_state,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_single_metrics": test_single_metrics,
        "test_multi_metrics": test_multi_metrics,
        "run_dir": run_dir,
        **saved,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import config
    # channel_names = config.MONO_CHANNELS
    channel_names = CAUEEG_EEG19
    fixed_pairs = config.MONOFIXEDGES
    channel_name = "mono"
    n_channels = 19
    fixed_edges = _normalize_fixed_edges(fixed_pairs, n_channels, channel_names)
    

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    # parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    # parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--feature_families_str", type=str, default="relative_band_power,statistical")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="coherence")
    parser.add_argument("--connectivity_band", type=int, default=2)
    # parser.add_argument("--base_k", type=int, default=None, required=False, help="base_k")

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="LINKX",
        # choices=["gnn", "LINKX", "linkx_cnn", "mlp_node", "sage", "gcn2", "h2gcn"]
        choices=["gnn", 'hybrid', 'gat', "LINKX", "linkx_cnn", "linkx_cnn5", "mlp_node", "sage", "gcn2", "h2gcn"]
        # choices=["gnn", "LINKX", "mlp_node", "sage", "gcn2", "h2gcn"],
    )
    args = parser.parse_args()
    # feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    for feature_families_str in ["relative_band_power,statistical", "relative_band_power,statistical,hjorth"]:
        feature_families = [x.strip() for x in feature_families_str.split(",") if x.strip()]

        for encoder_type in ["linkx", "mlp_node"]:
            out = run_caueeg_linkx_segment(
                dataset_path="/mnt/data/anphan/CAUEEG/caueeg-dataset",
                fixed_edges=fixed_edges,          
                channel_names=CAUEEG_EEG19,
                task="dementia",
                file_format="feather",
                out_h5="/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5",
                feature_families = feature_families,
                connectivity_metric = args.connectivity_metric,
                connectivity_band = args.connectivity_band,
                # connectivity_band=2,
                batch_size=64,
                test_batch_size=64,
                epochs=200,
                lr=3e-4,
                weight_decay=5e-3,
                device=device,
                rebuild_h5=False,
                output_root="/home/anphan/Documents/EEG_Project/CAUEEG/results_caueeg_linkx_segment",
                encoder_type = encoder_type,
                filter_method = args.topology,
                use_split_prefix=True,
                bad_serials={"00587", "00781", "01301"}
            )

            print("Done. Results saved to:", out["run_dir"])
