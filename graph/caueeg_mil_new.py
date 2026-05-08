from caueeg_loader_min import *
from master_builder import build_master_eeg_dataset
from mil_full_std import load_h5_payload_for_subjects, SubjectMILClassifier, fit_mil_baseline
from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags


# caueeg_linkx_mil_adapter.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from master_builder import build_master_eeg_dataset
# from mil_full_std import load_h5_payload_for_subjects, SubjectMILClassifier, fit_mil_baseline
from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags, build_graphs_from_payload_multiband, collate_subject_bags_multiband, build_graphs_from_payload_by_level
from utils_all import set_global_seed

import json
from datetime import datetime
import pandas as pd
from mil_utils import (
    collect_subject_embeddings,
    evaluate,
)

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



def summarize_graph_list(graphs, split_name="train", max_examples=3):
    if len(graphs) == 0:
        print(f"[{split_name}] no graphs")
        return

    levels = [str(getattr(g, "level", "unknown")) for g in graphs]
    subject_ids = [str(getattr(g, "subject_id", "")) for g in graphs]

    print(f"[{split_name}] num_graphs={len(graphs)}")
    print(f"[{split_name}] levels={sorted(set(levels))}")
    print(f"[{split_name}] unique_subjects={len(set(subject_ids))}")

    for i, g in enumerate(graphs[:max_examples]):
        print(
            f"[{split_name}] ex{i}: "
            f"level={getattr(g, 'level', None)}, "
            f"subject_id={getattr(g, 'subject_id', None)}, "
            f"segment_id={getattr(g, 'segment_id', None)}, "
            f"macro_id={getattr(g, 'macro_id', None)}, "
            f"x_shape={tuple(g.x.shape)}, "
            f"adj_shape={tuple(g.adj.shape) if hasattr(g, 'adj') else None}, "
            f"start={getattr(g, 'start_sample', None)}, "
            f"end={getattr(g, 'end_sample', None)}"
        )

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj


def _clean_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if k == "conf_matrix":
            out[k] = _jsonable(v)
        else:
            out[k] = _jsonable(v)
    return out
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


def dataset_to_subject_records(dataset):
    """
    Convert CauEegDataset split into records accepted by build_master_eeg_dataset().
    Use recording serial as MIL bag id.
    """
    records = []
    subject_ids = []

    for sample in dataset:
        signal = sample["signal"]              # [21, T]
        serial = str(sample["serial"])         # use recording id, not patient id
        label = int(sample["class_label"])     # task label
        age = float(sample.get("age", np.nan))

        windows, starts = segment_recording(signal)

        if len(windows) == 0:
            continue

        rec = {
            "subject_id": serial,
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
            },
        }
        records.append(rec)
        subject_ids.append(serial)

    return records, subject_ids


def payload_to_graphs(payload,
                      subject_ids,
                      feature_families,
                      connectivity_metric="pearson",
                      connectivity_band=None,
                      standardize_features=True):
    """
    Minimal CAUEEG adapter:
      node features = concat(RBP, Hjorth, Statistical)
      adjacency     = selected connectivity matrix
    """
    graphs = []

    for sid in subject_ids:
        if sid not in payload:
            raise KeyError(f"Subject {sid!r} not found in payload")
            continue

        subj = payload[sid]
        y = int(subj["label"])

        # [W, N, F_total]
        x_all = np.concatenate(
            [np.asarray(subj["features"][fam], dtype=np.float32) for fam in feature_families],
            axis=-1,
        )

        adj_all = np.asarray(subj["connectivity"][connectivity_metric], dtype=np.float32)

        # if metric is banded, select one band
        if adj_all.ndim == 4:
            if connectivity_band is None:
                raise ValueError("connectivity_band must be set for banded connectivity")
            adj_all = adj_all[:, connectivity_band]

        seg_ids = np.asarray(subj["segment_id"], dtype=np.int64)
        start_samples = np.asarray(subj["start_sample"], dtype=np.int64)

        for w in range(x_all.shape[0]):
            x = x_all[w]          # [19, F]
            adj = adj_all[w]      # [19, 19]

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

            g.subject_id = sid
            g.segment_id = int(seg_ids[w])
            g.start_sample = int(start_samples[w])

            graphs.append(g)

    return graphs

def run_caueeg_linkx_mil(
    dataset_path,
    fixed_edges,          
    channel_names,
    task="abnormal",
    file_format="feather",
    out_h5="caueeg_master_linkx.h5",
    feature_families = ['relative_band_power', 'statistical', 'wavelet_energy'],
    connectivity_metric="pearson",
    connectivity_band=None,
    encoder_type = "linkx_cnn",
    mil_pool_type = "gated",
    filter_method="fixed",
    base_k=8,
    batch_size=8,
    seed=42,
    epochs=100,
    patience=50,
    lr=1e-3,
    weight_decay=1e-4,
    device="cuda",
    rebuild_h5=False,
    output_root="graph/results_caueeg_linkx",
    graph_pool="mean",
    num_gnn_layers=2,
    readout_type="mean",
    node_pooling_type="none",
    node_pool_ratio=0.8,
    use_edge_weight=True,
    gat_heads=4,
    readout_hidden_dim=64,
    readout_dropout=0.0,
    min_delta=1e-3,
    top_k=3,
    start_epoch=50,
    edge_mode="topology_weighted",
    graph_emb_dim=64,
    dropout=0.3,
    attn_dim=64,
    max_k_per_subject = 300,
    bad_ids = {"00587", "00781", "01301"},
    graph_level="segment",
    macro_seconds=300.0,
    macro_reduce_node="mean",
    macro_reduce_adj="mean",
    subject_reduce_node="mean",
    subject_reduce_adj="mean",
    sfreq=200.0,
    graph_backbone="gcn",
    use_batchnorm=True,
    return_graph_attention_weights=0,
    pool_every_layer: bool = True,
    stage_readout_fusion: str = "concat",
):
    run_config = {
        "dataset_path": dataset_path,
        "task": task,
        "file_format": file_format,
        "out_h5": out_h5,
        "feature_families": list(feature_families),
        "connectivity_metric": connectivity_metric,
        "connectivity_band": connectivity_band,
        "encoder_type": encoder_type,
        "mil_pool_type": mil_pool_type,
        "filter_method": filter_method,
        "base_k": int(base_k),
        "batch_size": int(batch_size),
        "device": str(device),
        "rebuild_h5": bool(rebuild_h5),
        "seed": int(seed),
        "epochs": int(epochs),
        "patience": int(patience),
        "start_epoch": int(start_epoch),
        "min_delta": float(min_delta),
        "top_k": int(top_k),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "edge_mode": edge_mode,
        "graph_emb_dim": int(graph_emb_dim),
        "dropout": float(dropout),
        "attn_dim": int(attn_dim),
        "max_k_per_subject": int(max_k_per_subject),

        # stage-1 new graph knobs
        "graph_pool": graph_pool,
        "num_gnn_layers": int(num_gnn_layers),
        "readout_type": readout_type,
        "node_pooling_type": node_pooling_type,
        "node_pool_ratio": float(node_pool_ratio),
        "use_edge_weight": bool(use_edge_weight),
        "gat_heads": int(gat_heads),
        "readout_hidden_dim": int(readout_hidden_dim),
        "readout_dropout": float(readout_dropout),

        "channel_names": list(channel_names) if channel_names is not None else None,
        "fixed_edges": [list(map(int, e)) for e in sorted(list(fixed_edges))] if fixed_edges is not None else None,
        "bad_ids": sorted(list(bad_ids)),

        "graph_level": graph_level,
        "macro_seconds" : macro_seconds,
        "macro_reduce_node": macro_reduce_node,
        "macro_reduce_adj": macro_reduce_adj,
        "subject_reduce_node": subject_reduce_node,
        "subject_reduce_adj": subject_reduce_adj,
        "graph_backbone": graph_backbone,
        "use_batchnorm": bool(use_batchnorm),
        "return_graph_attention_weights": bool(return_graph_attention_weights),
        "graph_encoder_family": "gnn_block" if encoder_type == "gnn_block" else "legacy",
        "pool_every_layer" : pool_every_layer,
        "stage_readout_fusion" : stage_readout_fusion,
    }
    os.makedirs(output_root, exist_ok=True)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{connectivity_metric}_k{base_k}_{encoder_type}_{mil_pool_type}_{filter_method}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    summary_path = os.path.join(run_dir, "summary.json")

    set_global_seed(seed)

    config, train_set, val_set, test_set = load_caueeg_task_datasets(
        dataset_path=dataset_path,
        task=task,
        load_event=False,
        file_format=file_format,
        transform=None,
        verbose=False,
    )

    # 2) convert each recording into subject-like records
    train_records, train_ids = dataset_to_subject_records(train_set)
    val_records, val_ids = dataset_to_subject_records(val_set)
    test_records, test_ids = dataset_to_subject_records(test_set)

    all_records = train_records + val_records + test_records

    train_ids_filter = [sid for sid in train_ids if sid not in bad_ids]
    val_ids_filter   = [sid for sid in val_ids if sid not in bad_ids]
    test_ids_filter = [sid for sid in test_ids if sid not in bad_ids]
    all_ids_filter   = train_ids_filter + val_ids_filter + test_ids_filter
    # all_ids = train_ids + val_ids + test_ids



    num_classes = len(sorted({r["label"] for r in all_records}))

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



    train_ids_suf = ['train_' + item for item in train_ids_filter]
    val_ids_suf = ['val_' + item for item in val_ids_filter]
    test_ids_suf = ['test_' + item for item in test_ids_filter]

    all_ids_suf = train_ids_suf + val_ids_suf + test_ids_suf


    # 4) load payload
    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=all_ids_suf,
        feature_families=feature_families,
        connectivity_metrics=[connectivity_metric],
        connectivity_band=connectivity_band,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    # print(payload.keys())



    train_graphs = build_graphs_from_payload_by_level(
        payload,
        train_ids_suf,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        filter_method=filter_method,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        undirected=True,
        standardize_features=True,
        graph_level=graph_level,
        macro_seconds=macro_seconds,
        sfreq=200.0,
        macro_reduce_node=macro_reduce_node,
        macro_reduce_adj=macro_reduce_adj,
    )

    val_graphs = build_graphs_from_payload_by_level(
        payload,
        val_ids_suf,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        filter_method=filter_method,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        undirected=True,
        standardize_features=True,
        graph_level=graph_level,
        macro_seconds=macro_seconds,
        sfreq=200.0,
        macro_reduce_node=macro_reduce_node,
        macro_reduce_adj=macro_reduce_adj,
    )

    test_graphs = build_graphs_from_payload_by_level(
        payload,
        test_ids_suf,
        feature_families=feature_families,
        connectivity_metric=connectivity_metric,
        connectivity_band=connectivity_band,
        filter_method=filter_method,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        undirected=True,
        standardize_features=True,
        graph_level=graph_level,
        macro_seconds=macro_seconds,
        sfreq=200.0,
        macro_reduce_node=macro_reduce_node,
        macro_reduce_adj=macro_reduce_adj,
    )
    summarize_graph_list(train_graphs, "train")
    summarize_graph_list(val_graphs, "val")
    summarize_graph_list(test_graphs, "test")
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
            base_k=base_k,
            max_k_per_subject=max_k_per_subject,
            seed=seed,
            return_segment_ids=True,
        )
        val_dataset = LabelAwareSubjectBagDataset(
            val_graphs,
            train=False,
            eval_k_per_subject=None,
            seed=seed,
        )
        test_dataset = LabelAwareSubjectBagDataset(
            test_graphs,
            train=False,
            eval_k_per_subject=None,
            seed=seed,
        )





    if encoder_type in ["linkx_cnn5", "cnn5"]:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_subject_bags_multiband,
            num_workers=0,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subject_bags_multiband,
            num_workers=0,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_subject_bags_multiband,
            num_workers=0,
            pin_memory=True,
        )
    else:

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
            shuffle=False,
            collate_fn=collate_subject_bags,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
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
        num_gnn_layers=num_gnn_layers,
        readout_type=readout_type,
        node_pooling_type=node_pooling_type,
        node_pool_ratio=node_pool_ratio,
        use_edge_weight=bool(use_edge_weight),
        gat_heads=gat_heads,
        readout_hidden_dim=readout_hidden_dim,
        readout_dropout=readout_dropout,
        graph_backbone=graph_backbone,
        use_batchnorm=bool(use_batchnorm),
        return_graph_attention_weights=bool(return_graph_attention_weights),
        pool_every_layer=bool(pool_every_layer),
        stage_readout_fusion=stage_readout_fusion,
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
        min_delta=min_delta,
        top_k=top_k,
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
            "confusion_matrix": train_metrics["conf_matrix"],
        },
        {
            "split": "val",
            "loss": float(val_metrics["loss"]),
            "accuracy": float(val_metrics["accuracy"]),
            "balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "macro_f1": float(val_metrics["macro_f1"]),
            "confusion_matrix": val_metrics["conf_matrix"],

        },
        {
            "split": "test",
            "loss": float(test_metrics["loss"]),
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "confusion_matrix": test_metrics["conf_matrix"],

        },
    ]

    summary_test = [
        {
            "encoder_type": encoder_type,
            "training_approach": "MIL-subject",
            "mil_pool_type": mil_pool_type,
            "accuracy": float(test_metrics["accuracy"]),
            "balanced_accuracy": float(test_metrics["balanced_accuracy"]),
            "macro_f1": float(test_metrics["macro_f1"]),
            "confusion_matrix": test_metrics["conf_matrix"],
            "feature_families": feature_families,
            "topology": filter_method,
            "connectivity_metric": connectivity_metric,
            "connectivity_band": connectivity_band,
            "edge_mode": edge_mode,
            "base_k": base_k,
            "batch_size": batch_size,
            "epochs": epochs,
            "patience": patience,
            "start_epoch": start_epoch,
            "lr": lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "graph_emb_dim": graph_emb_dim,
            "attn_dim": attn_dim,
            "seed": seed,            
        },
    ]

    save_history_csv(history, os.path.join(run_dir, "history.csv"))
    save_summary_metrics_csv(summary_rows, os.path.join(run_dir, "summary_metrics.csv"))
    save_summary_metrics_csv(summary_test, os.path.join(run_dir, "summary_test.csv"))

    # train_pred_df = save_predictions_csv(
    #     model, train_loader, device,
    #     os.path.join(run_dir, "train_predictions.csv"),
    #     num_classes=num_classes,
    # )
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

    summary_payload = {
        "run_dir": run_dir,
        "run_name": run_name,
        "timestamp": timestamp,

        "config": run_config,

        "data": {
            "num_train_subjects": len(train_ids_suf),
            "num_val_subjects": len(val_ids_suf),
            "num_test_subjects": len(test_ids_suf),
            "num_train_graphs": len(train_graphs),
            "num_val_graphs": len(val_graphs),
            "num_test_graphs": len(test_graphs),
            "graph_level": graph_level,
            "train_graph_levels": sorted(list({str(getattr(g, "level", "unknown")) for g in train_graphs})),
            "val_graph_levels": sorted(list({str(getattr(g, "level", "unknown")) for g in val_graphs})),
            "test_graph_levels": sorted(list({str(getattr(g, "level", "unknown")) for g in test_graphs})),
            "num_node_features": int(train_dataset.num_node_features),
            "num_nodes": int(train_dataset.num_nodes),
            "num_classes": int(num_classes),
        },

        "best_checkpoint": {
            "epoch": None if best_state is None else best_state.get("epoch"),
            "selected_checkpoint": None if best_state is None else _jsonable(best_state.get("selected_checkpoint")),
            "selected_by": None if best_state is None else _jsonable(best_state.get("selected_by")),
            "top_k_checkpoints": None if best_state is None else _jsonable(best_state.get("top_k_checkpoints")),
        },

        "metrics": {
            "val": _clean_metrics(val_metrics),
            "test": _clean_metrics(test_metrics),
        },

        "history": _jsonable(history),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    return {
        "run_dir": run_dir,
        "summary_path": summary_path,
        "summary": summary_payload,
        "best_state": best_state,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root_path = "/home/anphan/Documents/CAUEEG"


    save_path = os.path.join(root_path,'result_new-arc')
    os.makedirs(save_path,exist_ok = True)


    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    # parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="full", required=False, help="topology")
    parser.add_argument("--feature_families_str", type=str, required=True)   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument("--base_k", type=int, default=10, required=False, help="base_k")
    parser.add_argument("--graph_pool", type=str, default="mean")
    parser.add_argument("--num_gnn_layers", type=int, default=2)
    parser.add_argument("--readout_type", type=str, default="mean")
    parser.add_argument("--node_pooling_type", type=str, default="none")
    parser.add_argument("--node_pool_ratio", type=float, default=0.8)
    parser.add_argument("--use_edge_weight", type=int, default=1)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--readout_hidden_dim", type=int, default=64)
    parser.add_argument("--readout_dropout", type=float, default=0.0)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="gnn",
        # choices=["gnn", "LINKX", "linkx_cnn", "mlp_node", "sage", "gcn2", "h2gcn"]
        choices=["gnn", "hier_gnn_block", "gnn_block", 'hybrid', 'gat', "LINKX", "linkx_cnn", "cnn5","linkx_cnn5", "mlp_node", "sage", "gcn2", "h2gcn"]
        # choices=["gnn", "LINKX", "mlp_node", "sage", "gcn2", "h2gcn"],
    )
    parser.add_argument("--graph_level", type=str, default="segment",
                        choices=["segment", "macro", "subject"])
    parser.add_argument("--macro_seconds", type=float, default=300.0)
    parser.add_argument("--macro_reduce_node", type=str, default="mean",
                        choices=["mean", "median", "max", "min", "std"])
    parser.add_argument("--macro_reduce_adj", type=str, default="mean",
                        choices=["mean", "median", "max", "min", "std"])
    parser.add_argument("--subject_reduce_node", type=str, default="mean",
                        choices=["mean", "median", "max", "min", "std"])
    parser.add_argument("--subject_reduce_adj", type=str, default="mean",
                        choices=["mean", "median", "max", "min", "std"])
    parser.add_argument("--graph_backbone", type=str, default="gcn",
                    choices=["gcn", "sage", "gatv2", "edge_gated"])
    parser.add_argument("--use_batchnorm", type=int, default=1)
    parser.add_argument("--edge_gate_dropout", type=float, default=0.0)
    parser.add_argument("--return_graph_attention_weights", type=int, default=0)
    parser.add_argument("--pool_every_layer", type=int, default=0)
    parser.add_argument("--stage_readout_fusion", type=str, default="concat",
                        choices=["mean", "concat", "gated"])
    args = parser.parse_args()

    import config
    channel_names = CAUEEG_EEG19
    fixed_pairs = config.MONOFIXEDGES
    channel_name = "mono"
    n_channels = 19
    fixed_edges = _normalize_fixed_edges(fixed_pairs, n_channels, channel_names)
    feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]

    out = run_caueeg_linkx_mil(
        dataset_path="/home/anphan/Downloads/caueeg-dataset/",
        fixed_edges=fixed_edges,          
        channel_names=CAUEEG_EEG19,
        task="dementia",
        file_format="edf",
        # out_h5="/mnt/data/anphan/CAUEEG/caueeg_master_linkx.h5",
        out_h5="/home/anphan/Documents/caueeg_randomcrop_master_dementia_seed42.h5",
        feature_families = feature_families,
        connectivity_metric = args.connectivity_metric,
        connectivity_band = args.connectivity_band,
        encoder_type = args.encoder_type,
        mil_pool_type = args.mil_pool_type,
        filter_method = args.topology,
        base_k=args.base_k,
        batch_size=8,
        start_epoch=100,
        epochs=500,
        patience=50,
        lr=1e-3,
        weight_decay=5e-4,
        device=device,
        rebuild_h5=False,
        output_root=save_path,

        graph_pool=args.graph_pool,
        num_gnn_layers=args.num_gnn_layers,
        readout_type=args.readout_type,
        node_pooling_type=args.node_pooling_type,
        node_pool_ratio=args.node_pool_ratio,
        use_edge_weight=args.use_edge_weight,
        gat_heads=args.gat_heads,
        readout_hidden_dim=args.readout_hidden_dim,
        readout_dropout=args.readout_dropout,

        graph_level=args.graph_level,
        macro_seconds=args.macro_seconds,
        macro_reduce_node=args.macro_reduce_node,
        macro_reduce_adj=args.macro_reduce_adj,
        subject_reduce_node=args.subject_reduce_node,
        subject_reduce_adj=args.subject_reduce_adj,


        graph_backbone=args.graph_backbone,
        use_batchnorm=args.use_batchnorm,
        return_graph_attention_weights=args.return_graph_attention_weights,
    )

