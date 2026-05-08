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
from mil_utils import LabelAwareSubjectBagDataset, collate_subject_bags, build_graphs_from_payload_multiband, collate_subject_bags_multiband
from utils_all import set_global_seed

import json
from datetime import datetime
import pandas as pd
from mil_utils import (
    collect_subject_embeddings,
    evaluate,
)
from master_builder import build_master_eeg_dataset, DEFAULT_BANDS, FEATURE_REGISTRY, CONNECTIVITY_REGISTRY

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
def collect_required_connectivity_metrics(
    bank_specs,
    default_connectivity_metric: str,
):
    metrics = {str(default_connectivity_metric)}
    if bank_specs is None:
        return sorted(metrics)

    for spec in bank_specs:
        m = spec.get("connectivity_metric", default_connectivity_metric)
        metrics.add(str(m))
    return sorted(metrics)
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

def build_graph_bank_from_specs(
    payload,
    subject_ids,
    *,
    feature_families,
    default_connectivity_metric,
    default_connectivity_band,
    default_filter_method,
    default_fixed_edges,
    channel_names,
    bank_specs,
    standardize_features=True,
):
    """
    Reuse existing build_graphs_from_payload(...) repeatedly and attach
    a bank [K, N, N] to each graph.

    Each spec can override:
      - name
      - connectivity_metric
      - connectivity_band
      - filter_method
      - fixed_edges
    """
    if bank_specs is None or len(bank_specs) == 0:
        raise ValueError("bank_specs must contain at least one candidate.")

    candidate_names = []
    candidate_graph_lists = []

    for spec_idx, spec in enumerate(bank_specs):
        name = str(spec.get("name", f"cand_{spec_idx}"))
        cand_metric = spec.get("connectivity_metric", default_connectivity_metric)

        # IMPORTANT:
        # do not blindly force one global default band for all candidates
        if "connectivity_band" in spec:
            cand_band = spec["connectivity_band"]
        else:
            cand_band = default_connectivity_band

        cand_filter_method = spec.get("filter_method", default_filter_method)
        cand_fixed_edges = spec.get("fixed_edges", default_fixed_edges)

        gs = build_graphs_from_payload(
            payload,
            subject_ids,
            feature_families=feature_families,
            connectivity_metric=cand_metric,
            connectivity_band=cand_band,
            filter_method=cand_filter_method,
            fixed_edges=cand_fixed_edges,
            channel_names=channel_names,
            undirected=True,
            standardize_features=standardize_features,
        )

        candidate_names.append(name)
        candidate_graph_lists.append(gs)

    base_graphs = candidate_graph_lists[0]

    def _graph_key(g):
        sid = str(getattr(g, "subject_id", ""))
        seg = int(getattr(g, "segment_id", -1))
        start = int(getattr(g, "start_sample", -1))
        return (sid, seg, start)

    # precompute maps once
    candidate_maps = []
    for cand_name, gs in zip(candidate_names, candidate_graph_lists):
        gmap = {}
        for g in gs:
            gmap[_graph_key(g)] = g
        candidate_maps.append(gmap)


    # attach [K, N, N] bank to each base graph
    for g in base_graphs:
        key = _graph_key(g)

        bank_adj = []
        bank_topo = []

        for cand_name, gmap in zip(candidate_names, candidate_maps):
            if key not in gmap:
                raise KeyError(f"Graph key {key} missing in candidate {cand_name!r}.")
            gg = gmap[key]

            if not hasattr(gg, "adj") or gg.adj is None:
                raise ValueError(
                    f"Candidate {cand_name!r} graph for key {key} is missing dense adj. "
                    "Make sure build_graphs_from_payload(..., attach_dense_adj=True)."
                )

            adj = gg.adj
            if torch.is_tensor(adj):
                adj = adj.detach().cpu().float()
            else:
                adj = torch.tensor(adj, dtype=torch.float32)

            topo = (adj != 0).float()

            bank_adj.append(adj)
            bank_topo.append(topo)

        g.adj_bank = torch.stack(bank_adj, dim=0)          # [K, N, N]
        g.topology_bank = torch.stack(bank_topo, dim=0)    # [K, N, N]
        g.topology_names = list(candidate_names)
        # For cnn_bank / linkx_cnn_bank compatibility
        g.conn_stack = g.adj_bank
        g.conn_stack_names = list(candidate_names)

    return base_graphs, candidate_names


BAND_TO_INDEX = {
    "delta": 0,
    "theta": 1,
    "alpha": 2,
    "beta": 3,
    "gamma": 4,
}


def _resolve_band_index(band):
    if isinstance(band, str):
        b = band.lower()
        if b not in BAND_TO_INDEX:
            raise ValueError(f"Unknown band={band!r}. Valid: {list(BAND_TO_INDEX)}")
        return BAND_TO_INDEX[b]
    return int(band)


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
    # epochs=100,
    # lr=1e-3,
    # weight_decay=1e-4,
    device="cuda",
    rebuild_h5=False,
    output_root="graph/results_caueeg_linkx",
    num_candidates: Optional[int] = None,
    bank_specs: Optional[List[Dict[str, Any]]] = None,
    bank_fusion_mode: str = "static",
    bank_topology_rule: str = "union",
    bank_vote_threshold: float = 0.5,
    bank_fusion_temperature: float = 1.0,
    bank_hidden_dim: int = 64,

    candidate_fusion_mode: str = "concat",
    candidate_fusion_hidden_dim: int = 64,
    candidate_fusion_dropout: float = 0.0,
    share_linkx_weights: bool = False,
):
    os.makedirs(output_root, exist_ok=True)

    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{connectivity_metric}_k{base_k}_{encoder_type}_{mil_pool_type}_{filter_method}"
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"log.txt")

    bad_ids = {"00587", "00781", "01301"}
    if encoder_type in ['linkx_cnn','linkx_cnn5', "cnn5", 'LINKX', 'mlp_node']:
        patience=50
        start_epoch=50
        epochs=200
        lr=1e-3
        weight_decay=5e-3

    else:
        patience=200
        start_epoch=100
        epochs=500
        lr=1e-3
        weight_decay=3e-4

    min_delta=1e-3
    top_k=3
    edge_mode="topology_weighted"
    graph_emb_dim=64
    dropout=0.3
    attn_dim=64
    seed=42
    set_global_seed(seed)

    max_k_per_subject = 300
    # feature_families = ['relative_band_power', 'statistical', 'wavelet_energy'] #'hjorth', 


    with open(log_path, "w") as f:
        f.write(f"data source {out_h5}, task {task}, file_format {file_format}\n")
        f.write(f"seeds {seed}\n")
        # f.write(f"norm_mode {args.norm_mode}\n")
        # f.write(f"note: update - use topology instead of full adj\n")
        f.write(f"note: bad_ids {bad_ids} \n")

        f.write(f"topology: {filter_method}, fixed_edges: {fixed_edges}, channel_names: {channel_names}\n")
        f.write(f"feature_families: {feature_families}\nconnectivity_metric: {connectivity_metric}, connectivity_band: {connectivity_band}\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"\n")

        f.write(f"model_name: {encoder_type}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta={min_delta}, top_k={top_k} \n")
        f.write(f"batch_size {batch_size}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        # f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
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

    if encoder_type in {"linkx_fused_bank", "linkx_bank", "cnn_bank", "linkx_cnn_bank"}:
        required_connectivity_metrics = collect_required_connectivity_metrics(
            bank_specs=bank_specs,
            default_connectivity_metric=connectivity_metric,
        )
    else:
        required_connectivity_metrics = [connectivity_metric]



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

    payload_connectivity_band = None if encoder_type in {"linkx_fused_bank", "linkx_bank", "linkx_cnn_bank", "cnn_bank"} else connectivity_band

    # 4) load payload
    payload = load_h5_payload_for_subjects(
        h5_path=out_h5,
        subject_ids=all_ids_suf,
        feature_families=feature_families,
        connectivity_metrics=required_connectivity_metrics,
        connectivity_band=payload_connectivity_band,
        load_raw_for_alignment=False,
        load_bad_segment_flag=False,
    )

    # print(payload.keys())


    # 5) graphs
    # if connectivity_band is not None:
    if encoder_type in {"linkx_fused_bank", "linkx_bank", "linkx_cnn_bank", "cnn_bank"}:
        train_graphs, topology_names = build_graph_bank_from_specs(
            payload,
            train_ids_suf,
            feature_families=feature_families,
            default_connectivity_metric=connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=filter_method,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )
        val_graphs, _ = build_graph_bank_from_specs(
            payload,
            val_ids_suf,
            feature_families=feature_families,
            default_connectivity_metric=connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=filter_method,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )
        test_graphs, _ = build_graph_bank_from_specs(
            payload,
            test_ids_suf,
            feature_families=feature_families,
            default_connectivity_metric=connectivity_metric,
            default_connectivity_band=None,
            default_filter_method=filter_method,
            default_fixed_edges=fixed_edges,
            channel_names=channel_names,
            bank_specs=bank_specs,
            standardize_features=True,
        )
        num_candidates = len(topology_names)

    elif encoder_type not in ["linkx_cnn5", "cnn5"]:
        train_graphs = build_graphs_from_payload(
            payload, train_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )
        val_graphs = build_graphs_from_payload(
            payload, val_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )
        test_graphs = build_graphs_from_payload(
            payload, test_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            connectivity_band=connectivity_band,
            filter_method=filter_method,
            fixed_edges=fixed_edges,          # from config
            channel_names=channel_names,      # whatever list you use for this payload
            undirected=True,
            standardize_features=True,       # or True if desired
        )

    else:

        train_graphs = build_graphs_from_payload_multiband(
            payload, train_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )
        val_graphs = build_graphs_from_payload_multiband(
            payload, val_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )
        test_graphs = build_graphs_from_payload_multiband(
            payload, test_ids_suf,
            feature_families=feature_families,
            connectivity_metric=connectivity_metric,
            # connectivity_band=connectivity_band,
        )

    # 6) MIL bags

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





    if encoder_type in ["linkx_cnn5", "cnn5", "linkx_cnn_bank", "cnn_bank"]:

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
        cnn_num_bands=len(topology_names),
        # graph_pool=graph_pool,
        # gnn_hidden_dim=gnn_hidden_dim,
        # node_hidden_dims=node_hidden_dims,
        # edge_hidden_dims=edge_hidden_dims,
        # branch_emb_dim=branch_emb_dim,

        num_candidates=num_candidates,
        bank_fusion_mode=bank_fusion_mode,
        bank_topology_rule=bank_topology_rule,
        bank_vote_threshold=bank_vote_threshold,
        bank_fusion_temperature=bank_fusion_temperature,
        bank_hidden_dim=bank_hidden_dim,

        candidate_fusion_mode=candidate_fusion_mode,
        candidate_fusion_hidden_dim=candidate_fusion_hidden_dim,
        candidate_fusion_dropout=candidate_fusion_dropout,
        share_linkx_weights=share_linkx_weights,

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

    return {
        "model": model,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "history": history,
        "best_state": best_state,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "run_dir": run_dir,
        # "train_pred_df": train_pred_df,
        "val_pred_df": val_pred_df,
        "test_pred_df": test_pred_df,
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
    root_path = "/home/anphan/Documents/EEG_Project/"


    save_path = os.path.join(root_path,'CAUEEG/result_MIL-LinkX-seed')
    os.makedirs(save_path,exist_ok = True)


    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    # parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--feature_families_str", type=str,  default="relative_band_power,statistical")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="wpli")
    parser.add_argument("--connectivity_band", type=int, default=None)
    parser.add_argument("--base_k", type=int, default=10, required=False, help="base_k")

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="linkx_cnn_bank",
        # choices=["gnn", "LINKX", "linkx_cnn", "mlp_node", "sage", "gcn2", "h2gcn"]
        choices=["gnn", "linkx_cnn_bank", "cnn_bank", 'linkx_bank', 'linkx_fused_bank', 'hybrid', 'gat', "LINKX", "linkx_cnn", "cnn5","linkx_cnn5", "mlp_node", "sage", "gcn2", "h2gcn"]
        # choices=["gnn", "LINKX", "mlp_node", "sage", "gcn2", "h2gcn"],
    )
    args = parser.parse_args()

    import config
    # channel_names = config.MONO_CHANNELS
    channel_names = CAUEEG_EEG19
    fixed_pairs = config.MONOFIXEDGES
    channel_name = "mono"
    n_channels = 19
    fixed_edges = _normalize_fixed_edges(fixed_pairs, n_channels, channel_names)
    feature_families = [x.strip() for x in args.feature_families_str.split(",") if x.strip()]
    
    bank_specs = [
        {"name": "wpli_theta_full", 
        "connectivity_metric": "wpli", 
        "connectivity_band": 1, 
        "filter_method": "full"},
        {"name": "wpli_alpha_full", 
        "connectivity_metric": "wpli", 
        "connectivity_band": 2, 
        "filter_method": "full"},
        {"name": "coherence_alpha_full", 
        "connectivity_metric": "coherence", 
        "connectivity_band": 2, 
        "filter_method": "full"},
        {"name": "coherence_theta_full", 
        "connectivity_metric": "coherence", 
        "connectivity_band": 1, 
        "filter_method": "full"},
    ]


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
        # epochs=200,
        # lr=1e-4,
        # weight_decay=1e-3,
        device=device,
        rebuild_h5=False,
        output_root=save_path,
        bank_specs = bank_specs
    )


    # import h5py
    # import json
    # import numpy as np

    # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_master_linkx.h5"
    # h5_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'



    # with h5py.File(h5_path, "r") as f:
    #     subject_ids = list(f["subjects"].keys())
    #     print("num subjects:", len(subject_ids))
    #     print("first 10 subjects:", subject_ids[:10])

    #     sid = subject_ids[0]
    #     grp = f[f"subjects/{sid}"]

    #     print("\nsubject key:", sid)
    #     print("group keys:", list(grp.keys()))
    #     print("metadata keys:", list(grp["metadata"].keys()))
    #     print("windows keys:", list(grp["windows"].keys()))
    #     print("raw keys:", list(grp["windows/raw"].keys()))

    #     # print("\nmetadata attrs:")
    #     # for k, v in grp["metadata"].attrs.items():
    #     #     print(f"  {k}: {v}")

    #     if "recording_info_json" in grp["metadata"].attrs:
    #         rec_info = json.loads(grp["metadata"].attrs["recording_info_json"])
    #         print("\nrecording_info_json:", rec_info)

    #     eeg = grp["windows/raw/eeg"]
    #     print("\nraw eeg shape:", eeg.shape, "dtype:", eeg.dtype)
    #     print("segment_id shape:", grp["windows/raw/segment_id"].shape)
    #     print("start_sample shape:", grp["windows/raw/start_sample"].shape)
    #     print("end_sample shape:", grp["windows/raw/end_sample"].shape)
    # import h5py
    # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42_backup.h5"

    # # h5_path = "/mnt/data/anphan/CAUEEG/caueeg_randomcrop_master_dementia_seed42.h5"
    # # required_features = ["relative_band_power", "hjorth", "statistical"]
    # # required_connectivity = ["coherence"]   # change if needed
    # required_features = FEATURE_REGISTRY
    # required_connectivity = CONNECTIVITY_REGISTRY
    # missing = []

    # with h5py.File(h5_path, "r") as f:
    #     for sid in f["subjects"].keys():
    #         subj_missing = []

    #         for fam in required_features:
    #             path = f"subjects/{sid}/windows/features/{fam}"
    #             if path not in f:
    #                 subj_missing.append(path)

    #         for metric in required_connectivity:
    #             path = f"subjects/{sid}/windows/connectivity/{metric}"
    #             if path not in f:
    #                 subj_missing.append(path)

    #         if subj_missing:
    #             missing.append((sid, subj_missing))

    # print("num subjects with missing groups:", len(missing))
    # for sid, paths in missing:
    #     print("\nSUBJECT:", sid)
    #     for p in paths:
    #         print("  missing:", p)


    # out_h5 = h5_path

    # def show_h5(name, obj):
    #     if isinstance(obj, h5py.Group):
    #         print(f"[G] {name}")
    #     else:
    #         print(f"[D] {name}  shape={obj.shape} dtype={obj.dtype}")

    # with h5py.File(h5_path, "r") as f:
    #     print("root keys:", list(f.keys()))
    #     print("\n=== full tree ===")
    #     f.visititems(show_h5)

    # with h5py.File(out_h5, "r") as f:
    #     sid = list(f["subjects"].keys())[0]
    #     grp = f["subjects"][sid]

    #     print("subject id:", sid)
    #     print("subject-level keys:", list(grp.keys()))
    #     print("metadata attrs:", dict(grp["metadata"].attrs))

    #     print("raw keys:", list(grp["windows/raw"].keys()))
    #     print("feature keys:", list(grp["windows/features"].keys()))
    #     print("connectivity keys:", list(grp["windows/connectivity"].keys()))

    #     for k in grp["windows/features"].keys():
    #         ds = grp["windows/features"][k]
    #         print("feature", k, ds.shape)

    #     for k in grp["windows/connectivity"].keys():
    #         ds = grp["windows/connectivity"][k]
    #         print("connectivity", k, ds.shape, dict(ds.attrs))

    # import os
    # import pyedflib
    # dataset_path="/mnt/data/anphan/CAUEEG/caueeg-dataset"
    # out_h5 = "/mnt/data/anphan/CAUEEG/caueeg_master_linkx.h5"
    # serial = "01386"
    # edf_path = os.path.join(dataset_path, "signal", "edf", f"{serial}.edf")
    # signals, signal_headers, edf_header = pyedflib.highlevel.read_edf(edf_path)

    # raw_labels = [h["label"] for h in signal_headers]
    # print("EDF labels:", raw_labels)
    # print("n_channels:", len(raw_labels))
    # print("first 19:", raw_labels[:19])
    # print("last 2:", raw_labels[19:])

    # from caueeg_loader_min import load_caueeg_config

    # config = load_caueeg_config(dataset_path)
    # print(config["signal_header"])
    # signal_header = config["signal_header"]
    # eeg19 = [ch for ch in signal_header if ch not in ["EKG", "Photic"]]

    # print("signal_header:", signal_header)
    # print("derived eeg19:", eeg19)
    # print("n eeg channels:", len(eeg19))

    # import pyarrow.feather as feather

    # feather_path = os.path.join(dataset_path, "signal", "feather", f"{serial}.feather")
    # df = feather.read_feather(feather_path)

    # print("Feather columns:", list(df.columns))

    # import h5py

    # with h5py.File(out_h5, "r") as h5f:
    #     sid = list(h5f["subjects"].keys())[0]
    #     ch_names = [
    #         x.decode("utf-8") if isinstance(x, bytes) else str(x)
    #         for x in h5f[f"subjects/{sid}/metadata/channel_names"][:]
    #     ]
    #     print("H5 channel_names:", ch_names)

    # ---------------------------
    # Train bags: 950
    # Sampled segment-instances per epoch: 15200
    # Total sampled segment-instances across training: 3040000
