from caueeg_loader_min import *
# caueeg_linkx_mil_adapter.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch.utils.data import Dataset
import random
from torch_geometric.utils import dense_to_sparse
# from master_builder import build_master_eeg_dataset

from mil_full_std import RawNodeEdgeMLPEncoder, RawNodeMLPEncoder, RawNodeAdjCNNEncoder, RawNodeMultiBandCNNEncoder
# from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags
from master_builder import FEATURE_REGISTRY, CONNECTIVITY_REGISTRY

from torchvision import transforms
from caueeg.caueeg_script import load_caueeg_task_datasets, calculate_signal_statistics
from caueeg.pipeline import EegRandomCrop, EegDropChannels, EegToTensor, eeg_collate_fn

import json
from datetime import datetime
import pandas as pd
from mil_utils import (
    GraphSAGEEncoder, GCNIIEncoder, H2GCNLikeEncoder, GNNEncoder, MeanMILPool
)

DEFAULT_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45)
    }

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

# ---------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_int(x: str) -> int:
    return int(hashlib.md5(str(x).encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------
# Channel helpers
# ---------------------------------------------------------
def strip_avg_suffix(ch: str) -> str:
    ch = str(ch).strip()
    return ch[:-4] if ch.endswith("-AVG") else ch


def get_caueeg_channel_info(dataset_path: str):
    config = load_caueeg_config(dataset_path)
    signal_header = list(config["signal_header"])

    idx_ekg = signal_header.index("EKG")
    idx_photic = signal_header.index("Photic")
    drop_idx = [idx_ekg, idx_photic]

    eeg19_raw = [ch for ch in signal_header if ch not in ["EKG", "Photic"]]
    eeg19_clean = [strip_avg_suffix(ch) for ch in eeg19_raw]

    return config, signal_header, eeg19_clean, drop_idx

class CauEegLinkxMilOnTheFlyDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        channel_names,
        drop_idx,
        signal_mean,
        signal_std,
        feature_families,
        connectivity_metric,
        connectivity_band=None,
        crop_length=2000,
        latency=2000,
        multiple=8,
        train=True,
        seed=42,
        standardize_features=True,
    ):
        self.raw_dataset = raw_dataset
        self.channel_names = channel_names
        self.drop_idx = list(drop_idx)
        self.signal_mean = signal_mean.astype(np.float32)
        self.signal_std = signal_std.astype(np.float32)
        self.feature_families = list(feature_families)
        self.connectivity_metric = connectivity_metric
        self.connectivity_band = connectivity_band
        self.crop_length = int(crop_length)
        self.latency = int(latency)
        self.multiple = int(multiple)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0
        self.standardize_features = standardize_features

        self.num_nodes = len(self.channel_names)
        self.num_node_features = None

        probe_graph = self._make_probe_graph()
        self.num_node_features = int(probe_graph.x.shape[-1])
        self.num_nodes = int(probe_graph.x.shape[0])

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.raw_dataset)

    def _make_probe_graph(self):
        for idx in range(len(self.raw_dataset)):
            sample = self.raw_dataset[idx]
            signal = np.asarray(sample["signal"], dtype=np.float32)
            signal = np.delete(signal, self.drop_idx, axis=0)

            if signal.shape[-1] <= self.latency + self.crop_length:
                continue

            serial = str(sample["serial"])
            label = int(sample["class_label"])
            start_sample = self.latency

            crop = signal[:, start_sample:start_sample + self.crop_length].astype(np.float32, copy=False)
            crop = self._normalize_crop(crop)

            return self._crop_to_graph(
                crop=crop,
                label=label,
                serial=serial,
                seg_id=0,
                start_sample=start_sample,
            )

        raise RuntimeError("Could not build a probe graph from any sample.")

    def _rng_for_item(self, serial: str):
        base = self.seed + self.epoch * 1000003 + (hash(serial) & 0xffffffff)
        return np.random.default_rng(base)

    def _sample_starts(self, signal_len: int, rng):
        hi = signal_len - self.crop_length
        if hi <= self.latency:
            return []
        return rng.integers(
            low=self.latency,
            high=hi,
            size=self.multiple,
            endpoint=False,
        ).tolist()
    def _normalize_crop(self, crop):
        return ((crop - self.signal_mean.squeeze(0)) / (self.signal_std.squeeze(0) + 1e-8)).astype(np.float32)

    def _crop_to_graph(self, crop, label, serial, seg_id, start_sample):
        feat_list = []
        bands = DEFAULT_BANDS
        sfreq = 200.0

        for fam in self.feature_families:
            values, _ = FEATURE_REGISTRY[fam]["fn"](crop, sfreq, bands)   # [N, F_fam]
            feat_list.append(np.asarray(values, dtype=np.float32))

        x = np.concatenate(feat_list, axis=-1)   # [19, F_total]

        adj, meta = CONNECTIVITY_REGISTRY[self.connectivity_metric]["fn"](crop, sfreq, bands)

        if adj.ndim == 3:
            if self.connectivity_band is None:
                raise ValueError("Need connectivity_band for banded metric.")
            adj = adj[self.connectivity_band]

        if self.standardize_features:
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
            y=torch.tensor([label], dtype=torch.long),
        )
        g.edge_weight = edge_weight.float()
        g.edge_attr = edge_weight.view(-1, 1).float()
        g.adj = torch.tensor(adj, dtype=torch.float32)
        g.subject_id = serial
        g.segment_id = int(seg_id)
        g.start_sample = int(start_sample)
        return g

    def __getitem__(self, idx):
        sample = self.raw_dataset[idx]
        signal = np.asarray(sample["signal"], dtype=np.float32)
        serial = str(sample["serial"])
        label = int(sample["class_label"])

        signal = np.delete(signal, self.drop_idx, axis=0)  # keep EEG 19 only
        rng = self._rng_for_item(serial)
        starts = self._sample_starts(signal.shape[-1], rng)

        graphs = []
        for seg_id, st in enumerate(starts):
            crop = signal[:, st:st + self.crop_length].astype(np.float32, copy=False)
            crop = self._normalize_crop(crop)
            graphs.append(self._crop_to_graph(crop, label, serial, seg_id, st))

        return {
            "subject_id": serial,
            "label": label,
            "graphs": graphs,
        }





def build_stats_transform(crop_length, latency, drop_idx):
    return transforms.Compose([
        EegRandomCrop(
            crop_length=crop_length,
            length_limit=10**7,
            multiple=1,
            latency=latency,
            segment_simulation=False,
            return_timing=False,
        ),
        EegDropChannels(drop_idx),
        EegToTensor(),
    ])
def compute_train_signal_stats(
    dataset_path: str,
    task: str,
    file_format: str,
    crop_length: int,
    latency: int,
    seed: int,
):
    set_global_seed(seed)
    _, _, _, drop_idx = get_caueeg_channel_info(dataset_path)

    stats_transform = build_stats_transform(
        crop_length=crop_length,
        latency=latency,
        drop_idx=drop_idx,
    )

    _, train_set, _, _ = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=stats_transform,
        verbose=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=eeg_collate_fn,
    )

    mean_sum = None
    std_sum = None
    n_count = 0

    for _ in range(5):
        for sample in train_loader:
            signal = sample["signal"]  # [N, C, T]
            std, mean = torch.std_mean(signal, dim=-1, keepdim=True)  # [N, C, 1]

            mean_batch = mean.sum(dim=0, keepdim=True)  # [1, C, 1]
            std_batch = std.sum(dim=0, keepdim=True)    # [1, C, 1]

            if mean_sum is None:
                mean_sum = torch.zeros_like(mean_batch)
                std_sum = torch.zeros_like(std_batch)

            mean_sum += mean_batch
            std_sum += std_batch
            n_count += signal.shape[0]

    signal_mean = mean_sum / n_count   # [1, C, 1]
    signal_std = std_sum / n_count     # [1, C, 1]

    return signal_mean.detach().cpu().numpy(), signal_std.detach().cpu().numpy()
def segment_recording(signal: np.ndarray,
                      crop_len: int = CROP_LEN,
                      step: int = STEP,
                      latency: int = LATENCY):
    """
    signal: [C, T]
    returns:
        windows: list[np.ndarray] each [19, crop_len]
        starts : list[int]
    """
    x = np.asarray(signal, dtype=np.float32)
    x = x[:19]  # drop EKG + photic

    total_len = x.shape[-1]
    starts = list(range(latency, total_len - crop_len + 1, step))

    windows = [x[:, s:s + crop_len].astype(np.float32, copy=False) for s in starts]
    return windows, starts

# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def rows_to_prediction_df(rows, num_classes=None):
    records = []

    for r in rows:
        rec = {
            "subject_id": r["subject_id"],
            "true_label": int(r["label"]),
            "pred_label": int(r["pred"]),
        }

        prob = np.asarray(r["prob"], dtype=np.float32).reshape(-1)
        emb = np.asarray(r["embedding"], dtype=np.float32).reshape(-1)

        if num_classes is None:
            num_classes_local = len(prob)
        else:
            num_classes_local = int(num_classes)

        for i in range(num_classes_local):
            rec[f"prob_{i}"] = float(prob[i])

        # store embedding as one JSON string so CSV stays compact
        rec["embedding_json"] = json.dumps(emb.tolist())
        records.append(rec)

    return pd.DataFrame(records)


def save_predictions_csv(model, loader, device, csv_path, num_classes=None):
    rows = collect_subject_embeddings(model, loader, device)
    df = rows_to_prediction_df(rows, num_classes=num_classes)
    df.to_csv(csv_path, index=False)
    print(f"Saved predictions: {csv_path}")
    return df


def save_summary_metrics_csv(summary_rows, csv_path):
    df = pd.DataFrame(summary_rows)
    df.to_csv(csv_path, index=False)
    print(f"Saved summary metrics: {csv_path}")
    return df


def save_history_csv(history, csv_path):
    df = pd.DataFrame(history)
    df.to_csv(csv_path, index=False)
    print(f"Saved history: {csv_path}")
    return df
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

def _safe_divide(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.divide(
        a,
        b,
        out=np.zeros_like(a, dtype=np.float64),
        where=(b != 0),
    )

def compute_classwise_sens_spec(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    sens = []
    spec = []

    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        sens.append(float(sensitivity))
        spec.append(float(specificity))

    return np.array(sens), np.array(spec), cm


def plot_linkx_mil_baseline_style(metrics, class_names, output_dir, prefix="test"):
    """
    metrics must come from evaluate(...)
    expects:
      metrics["y_true"]
      metrics["y_pred"]
      metrics["y_prob"]
      metrics["conf_matrix"]
    """
    os.makedirs(output_dir, exist_ok=True)

    y_true = np.asarray(metrics["y_true"], dtype=int)
    y_pred = np.asarray(metrics["y_pred"], dtype=int)
    y_prob = np.asarray(metrics["y_prob"], dtype=np.float64)

    num_classes = len(class_names)

    # -----------------------------
    # 1) row-normalized confusion
    # -----------------------------
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
    plt.title("Normalized Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center"
            )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_confusion.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 2) class-wise metrics
    # -----------------------------
    sens, spec, _ = compute_classwise_sens_spec(y_true, y_pred, num_classes)

    x = np.arange(num_classes)
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, sens, width, label="Sensitivity")
    plt.bar(x + width / 2, spec, width, label="Specificity")
    plt.xticks(x, class_names, rotation=45)
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.title("Class-wise Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_classwise_metrics.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # 3) ROC curve
    # -----------------------------
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))

    plt.figure(figsize=(6, 5))

    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, c], y_prob[:, c])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[c]} (AUC={roc_auc:.3f})")

    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, linestyle="--", label=f"micro-average (AUC={auc_micro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle=":")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    num_classes = 3
    class_labels = [0,1,2]
    class_names = ["Healthy", "Dementia", "MCI"]
    
    dataset_path="/mnt/data/anphan/CAUEEG/caueeg-dataset"
    seed=42
    task="dementia"
    file_format="feather"

    feature_families=["relative_band_power", "statistical"] #, "hjorth"

    connectivity_metric="wpli"
    connectivity_band=2
    crop_length=2000
    latency=2000
    batch_size=8
    encoder_type='linkx'
    
    epochs=200
    base_k=10

    edge_mode="topology_weighted"
    mil_pool_type="mean"
    
    lr=1e-3
    weight_decay=5e-3
    graph_emb_dim=64
    dropout=0.3
    attn_dim=64
    patience=50
    start_epoch=50


    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path = "/home/anphan/Documents/EEG_Project/"
    output_root = os.path.join(root_path,'CAUEEG/result_MIL-LinkX')
    os.makedirs(output_root,exist_ok = True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{task}_linkxmil_{connectivity_metric}_k{base_k}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    _, _, eeg19_clean, drop_idx = get_caueeg_channel_info(dataset_path)

    signal_mean, signal_std = compute_train_signal_stats(
        dataset_path=dataset_path,
        task=task,
        file_format=file_format,
        crop_length=crop_length,
        latency=latency,
        seed=seed,
    )

    _, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )


    train_dataset = CauEegLinkxMilOnTheFlyDataset(
        raw_dataset=train_set,
        channel_names=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        crop_length=crop_length,
        latency=latency,
        multiple=base_k,
        train=True,
        seed=seed,
        standardize_features=True,
        )

    val_dataset   = CauEegLinkxMilOnTheFlyDataset(
        raw_dataset=train_set,
        channel_names=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        crop_length=crop_length,
        latency=latency,
        multiple=base_k,
        train=False,
        seed=seed,
        standardize_features=True,
        )

    test_dataset  = CauEegLinkxMilOnTheFlyDataset(
        raw_dataset=train_set,
        channel_names=eeg19_clean,
        drop_idx=drop_idx,
        signal_mean=signal_mean,
        signal_std=signal_std,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        crop_length=crop_length,
        latency=latency,
        multiple=base_k,
        train=False,
        seed=seed,
        standardize_features=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subject_bags,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subject_bags,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_subject_bags,
        num_workers=0,
        pin_memory=True,
    )

    model = SubjectMILClassifier(
        num_node_features=train_dataset.num_node_features,
        num_classes=num_classes,
        num_nodes=train_dataset.num_nodes,
        encoder_type=encoder_type,
        edge_mode=edge_mode,
        graph_emb_dim=graph_emb_dim,
        dropout=dropout,
        mil_pool_type=mil_pool_type,
        attn_dim=attn_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    ckpt_path = os.path.join(run_dir, "best_model.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model, val_metrics, history, best_state = fit_mil_baseline(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=epochs,
        patience=patience,
        save_path=ckpt_path,
        start_epoch=start_epoch,
        min_delta=0.0,
        top_k=5,
        verbose=True,
    )

    # 7) final evaluation
    train_metrics = evaluate(model, train_loader, criterion, device)
    val_metrics = evaluate(model, val_loader, criterion, device)
    test_metrics = evaluate(model, test_loader, criterion, device)
    # after:
    # test_metrics = evaluate(model, test_loader, criterion, device)

    if task == "abnormal":
        class_names = ["normal", "abnormal"]
    elif task == "dementia":
        class_names = ["normal", "mci", "dementia"]
    else:
        # fallback
        num_classes = len(np.unique(test_metrics["y_true"]))
        class_names = [f"class_{i}" for i in range(num_classes)]

    plot_linkx_mil_baseline_style(
        metrics=test_metrics,
        class_names=class_names,
        output_dir=run_dir,
        prefix="test"
    )
    summary_rows = [
        {
            "split": "train",
            "loss": float(train_metrics["loss"]),
            "accuracy": float(train_metrics["accuracy"]),
            "balanced_accuracy": float(train_metrics["balanced_accuracy"]),
            "macro_f1": float(train_metrics["macro_f1"]),
            "task": task,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "base_k": base_k,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "split": "val",
            "loss": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "macro_f1": float(val_metrics["macro_f1"]),
            "task": task,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "base_k": base_k,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
        },
        {
            "split": "test",
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "task": task,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "base_k": base_k,
            "batch_size": batch_size,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]

    save_history_csv(history, os.path.join(run_dir, "history.csv"))
    save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))

    train_pred_df = save_predictions_csv(
        model, train_loader, device,
        os.path.join(run_dir, "train_predictions.csv"),
        num_classes=num_classes,
    )
    val_pred_df = save_predictions_csv(
        model, val_loader, device,
        os.path.join(run_dir, "val_predictions.csv"),
        num_classes=num_classes,
    )
    test_pred_df = save_predictions_csv(
        model, test_loader, device,
        os.path.join(run_dir, "test_predictions.csv"),
        num_classes=num_classes,
    )

    # save_path = "/home/anphan/Downloads/caueeg-ceednet/results_baseline"
    # os.makedirs(save_path,exist_ok = True)
    # output_dir = os.path.join(
    #     save_path,
    #     f"{config['task']}_{config.get('model_name', config.get('model', 'model'))}"
    # )
    # os.makedirs(output_dir,exist_ok = True)

    saved = save_baseline_outputs(
        output_dir=run_dir,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        multicrop_test_loader=multicrop_test_loader,
        preprocess_test=config["preprocess_test"],
        device=config["device"],
        config=config,
        history_rows=history if "history" in locals() else None,
    )


    # return {
    #     "model": model,
    #     "train_loader": train_loader,
    #     "val_loader": val_loader,
    #     "test_loader": test_loader,
    #     "history": history,
    #     "best_state": best_state,
    #     "train_metrics": train_metrics,
    #     "val_metrics": val_metrics,
    #     "test_metrics": test_metrics,
    #     "run_dir": run_dir,
    #     "train_pred_df": train_pred_df,
    #     "val_pred_df": val_pred_df,
    #     "test_pred_df": test_pred_df,
    # }