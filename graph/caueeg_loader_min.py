import os
import json
from caueeg.caueeg_dataset import CauEegDataset
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Sequence, Any
import torch

from mil_full_std import * #EarlyStopping, RawNodeEdgeMLPEncoder, RawNodeMLPEncoder, RawNodeAdjCNNEncoder, RawNodeMultiBandCNNEncoder
from mil_utils import *
# (
#     train_one_epoch, GraphSAGEEncoder, GCNIIEncoder, H2GCNLikeEncoder, GNNEncoder, MeanMILPool, GatedAttentionMIL
# )
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

def _extract_logits(model_output):
    if torch.is_tensor(model_output):
        return model_output
    if isinstance(model_output, dict):
        for key in ["logits", "output", "outputs", "pred", "prediction"]:
            if key in model_output and torch.is_tensor(model_output[key]):
                return model_output[key]
        raise ValueError(f"Could not find logits in dict keys: {list(model_output.keys())}")
    if isinstance(model_output, (list, tuple)):
        for x in model_output:
            if torch.is_tensor(x):
                return x
        raise ValueError("Could not find tensor logits in tuple/list model output.")
    raise ValueError(f"Unsupported model output type: {type(model_output)}")

def _get_class_names(config, num_classes):
    if "class_label_to_name" in config:
        d = config["class_label_to_name"]
        if isinstance(d, dict):
            names = []
            for i in range(num_classes):
                if i in d:
                    names.append(str(d[i]))
                elif str(i) in d:
                    names.append(str(d[str(i)]))
                else:
                    names.append(f"class_{i}")
            return names
    return [f"class_{i}" for i in range(num_classes)]

# def _safe_divide(a, b):
#     a = np.asarray(a, dtype=np.float64)
#     b = np.asarray(b, dtype=np.float64)
#     out = np.zeros_like(a, dtype=np.float64)
#     mask = b != 0
#     out[mask] = a[mask] / b[mask]
#     return out
def _safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=(b != 0))
def plot_confusion(df, class_names, save_path, title="Normalized Confusion Matrix"):
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()
    num_classes = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = _safe_divide(cm, row_sum)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm_norm, interpolation="nearest")
    plt.colorbar()
    plt.xticks(range(num_classes), class_names, rotation=45)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def prediction_df_to_metrics(df, split_name, task=None, extra=None):
    y_true = df["true_label"].to_numpy()
    y_pred = df["pred_label"].to_numpy()

    row = {
        "split": split_name,
        "loss": np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    if task is not None:
        row["task"] = task
    if extra is not None:
        row.update(extra)
    return row

def collect_prediction_rows(model, loader, preprocess, device, num_classes):
    model.eval()
    rows = []

    with torch.no_grad():
        for sample in loader:
            sample = preprocess(sample)
            # out = model(sample)
            try:
                out = model(sample["signal"], sample["age"])
            except TypeError:
                out = model(sample)
            logits = _extract_logits(out)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()

            y_true = sample["class_label"].detach().cpu().numpy()
            serials = sample["serial"]

            for i, serial in enumerate(serials):
                rec = {
                    "subject_id": str(serial),   # align with LinkX-MIL CSV
                    "serial": str(serial),
                    "true_label": int(y_true[i]),
                    "pred_label": int(np.argmax(probs[i])),
                }
                for c in range(num_classes):
                    rec[f"prob_{c}"] = float(probs[i, c])
                rows.append(rec)

    return pd.DataFrame(rows)

def aggregate_predictions_by_recording(crop_df, num_classes):
    prob_cols = [f"prob_{i}" for i in range(num_classes)]

    agg = {"true_label": ("true_label", "first")}
    for c in prob_cols:
        agg[c] = (c, "mean")

    rec_df = (
        crop_df.groupby("serial", as_index=False)
        .agg(**agg)
        .rename(columns={"serial": "subject_id"})
    )
    rec_df["serial"] = rec_df["subject_id"]
    rec_df["pred_label"] = rec_df[prob_cols].to_numpy().argmax(axis=1)

    # reorder columns
    ordered = ["subject_id", "serial", "true_label", "pred_label"] + prob_cols
    return rec_df[ordered]

def save_baseline_outputs(
    output_dir,
    model,
    train_loader,
    val_loader,
    test_loader,
    multicrop_test_loader,
    preprocess_test,
    device,
    config,
    history_rows=None,
):
    os.makedirs(output_dir, exist_ok=True)

    num_classes = int(config["out_dims"])
    class_names = _get_class_names(config, num_classes)
    task = config.get("task", None)

    extra = {
        "file_format": config.get("file_format"),
        "model_name": config.get("model_name", config.get("model", None)),
        "test_crop_multiple": config.get("test_crop_multiple", 1),
        "input_norm": config.get("input_norm"),
    }

    # single-crop snapshots
    train_pred_df = collect_prediction_rows(model, train_loader, preprocess_test, device, num_classes)
    val_pred_df = collect_prediction_rows(model, val_loader, preprocess_test, device, num_classes)
    test_single_df = collect_prediction_rows(model, test_loader, preprocess_test, device, num_classes)

    # crop-level multicrop predictions
    test_multi_crop_df = collect_prediction_rows(model, multicrop_test_loader, preprocess_test, device, num_classes)

    # EEG-recording-level TTA aggregation
    test_multi_recording_df = aggregate_predictions_by_recording(test_multi_crop_df, num_classes)

    # save prediction CSVs
    train_pred_df.to_csv(os.path.join(output_dir, "train_predictions.csv"), index=False)
    val_pred_df.to_csv(os.path.join(output_dir, "val_predictions.csv"), index=False)
    test_single_df.to_csv(os.path.join(output_dir, "test_predictions_singlecrop.csv"), index=False)
    test_multi_crop_df.to_csv(os.path.join(output_dir, "test_predictions_multicrop_crop_level.csv"), index=False)
    test_multi_recording_df.to_csv(os.path.join(output_dir, "test_predictions_multicrop_recording.csv"), index=False)

    # save summary metrics in LinkX-MIL style
    summary_rows = [
        prediction_df_to_metrics(train_pred_df, "train", task=task, extra=extra),
        prediction_df_to_metrics(val_pred_df, "val", task=task, extra=extra),
        prediction_df_to_metrics(test_single_df, "test_singlecrop", task=task, extra=extra),
        prediction_df_to_metrics(test_multi_recording_df, "test_multicrop_recording", task=task, extra=extra),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    # save optional history
    if history_rows is not None:
        pd.DataFrame(history_rows).to_csv(os.path.join(output_dir, "history.csv"), index=False)

    # save raw metric jsons too
    with open(os.path.join(output_dir, "metrics_test_singlecrop.json"), "w") as f:
        json.dump(summary_rows[2], f, indent=2)
    with open(os.path.join(output_dir, "metrics_test_multicrop_recording.json"), "w") as f:
        json.dump(summary_rows[3], f, indent=2)

    # confusion plots
    plot_confusion(
        test_single_df,
        class_names=class_names,
        save_path=os.path.join(output_dir, "test_singlecrop_confusion.png"),
        title="Single-Crop Test Confusion",
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
def load_caueeg_config(dataset_path: str):
    with open(os.path.join(dataset_path, "annotation.json"), "r") as f:
        annotation = json.load(f)
    return {k: v for k, v in annotation.items() if k != "data"}

def load_caueeg_task_datasets(
    dataset_path: str,
    task: str,
    load_event: bool = True,
    file_format: str = "edf",
    transform=None,
    verbose: bool = False,
):
    task = task.lower()
    if task not in ["abnormal", "dementia", "abnormal-no-overlap", "dementia-no-overlap"]:
        raise ValueError(f"Invalid task: {task}")

    with open(os.path.join(dataset_path, task + ".json"), "r") as f:
        task_dict = json.load(f)

    train_dataset = CauEegDataset(
        dataset_path, task_dict["train_split"],
        load_event=load_event, file_format=file_format, transform=transform
    )
    val_dataset = CauEegDataset(
        dataset_path, task_dict["validation_split"],
        load_event=load_event, file_format=file_format, transform=transform
    )
    test_dataset = CauEegDataset(
        dataset_path, task_dict["test_split"],
        load_event=load_event, file_format=file_format, transform=transform
    )

    config = {k: v for k, v in task_dict.items()
              if k not in ["train_split", "validation_split", "test_split"]}

    return config, train_dataset, val_dataset, test_dataset




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
        mil_pool_type: str = "gated",   # "mean" or "gated"
        edge_mode: str = "topology_weighted",
        attn_dim: int = 128,
        # cnn_channels: Sequence[int] = (16, 32, 64),
        cnn_num_bands: int = 5,
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.mil_pool_type = mil_pool_type.lower()

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
        elif self.encoder_type == "linkx_cnn5":
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

        else:
            raise ValueError(
                f"Unknown encoder_type='{encoder_type}'. "
                f"Choose from ['gnn', 'linkx', 'linkx_cnn5', 'mlp_node', 'sage', 'gcn2', 'h2gcn']"
            )

        if self.mil_pool_type == "mean":
            self.mil_pool = MeanMILPool()
        elif self.mil_pool_type == "gated":
            self.mil_pool = GatedAttentionMIL(
                in_dim=graph_emb_dim,
                attn_dim=attn_dim,
            )
        else:
            raise ValueError(f"Unknown mil_pool_type='{mil_pool_type}'")

        self.classifier = nn.Sequential(
            nn.Linear(graph_emb_dim, graph_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(graph_emb_dim, num_classes),
        )

    def forward(self, batch_dict: Dict):
        if self.encoder_type == "linkx_cnn":
            if "full_adj" not in batch_dict:
                raise KeyError(
                    "batch_dict is missing 'full_adj'. "
                    "Make sure graphs have g.adj attached and collate_subject_bags stacks it."
                )
            graph_emb = self.graph_encoder(
                batch_dict["pyg_batch"],
                batch_dict["full_adj"],
            )
        elif self.encoder_type == "linkx_cnn5":
            if "conn_stack" not in batch_dict:
                raise KeyError(
                    "batch_dict is missing 'conn_stack'. "
                    "Use build_graphs_from_payload_multiband(...) and collate_subject_bags(...)."
                )
            graph_emb = self.graph_encoder(
                batch_dict["pyg_batch"],
                batch_dict["conn_stack"],
            )
        else:
            graph_emb = self.graph_encoder(batch_dict["pyg_batch"])

        bag_emb, attn_list = self.mil_pool(graph_emb, batch_dict["bag_sizes"])

        logits = self.classifier(bag_emb)

        return {
            "logits": logits,
            "bag_emb": bag_emb,
            "graph_emb": graph_emb,
            "attn_list": attn_list,
        }



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
    verbose=True,
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
    for epoch in range(1, epochs + 1):
        # important for deterministic-but-changing segment sampling
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch - 1)

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

        # if epoch % 25 == 0:
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

        should_stop = early_stopper(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_metrics["loss"],
            val_bal_acc=val_metrics["balanced_accuracy"],
            val_macro_f1=val_metrics["macro_f1"],
            extra_state={
                "history": copy.deepcopy(history),
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

def collate_subject_bags(batch: List[dict]) -> Dict:
    all_graphs = []
    all_summary = []
    all_full_adj = []
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

            # full dense adjacency for mlp_raw edge_mode='full_adj'
            if hasattr(g, "adj") and g.adj is not None:
                adj = g.adj
                if torch.is_tensor(adj):
                    adj = adj.detach().cpu()
                else:
                    adj = torch.tensor(adj, dtype=torch.float32)
                all_full_adj.append(adj.float())
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
