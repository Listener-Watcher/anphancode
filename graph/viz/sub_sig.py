import torch
import numpy as np
from viz_plots import plot_heatmap, plot_class_mean_connectivity_multiplot
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import ast
import math

from statsmodels.stats.multitest import multipletests
from mne_connectivity.viz import plot_connectivity_circle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from data_utils import *
from data_preparation import *
from utils_all import *

import pandas as pd

def subject_edges_to_df(
    sid,
    mask,
    z,
    Abar,
    ciwidth,
    channel_names=None
):
    rows = []
    N = mask.shape[0]

    for i in range(N):
        for j in range(i+1, N):

            if not mask[i, j]:
                continue

            ni = channel_names[i] if channel_names else i
            nj = channel_names[j] if channel_names else j

            rows.append({
                "subject_id": sid,
                "node_i": ni,
                "node_j": nj,
                "i": i,
                "j": j,
                "z_score": float(z[i, j]),
                "abs_z": float(abs(z[i, j])),
                "mean_adj": float(Abar[i, j]),
                "ci_width": float(ciwidth[i, j])
            })

    return pd.DataFrame(rows)

def get_label_of_subject(by_subj, sid):
    return int(by_subj[sid][0]["class_id"])

def intersect_existing(sids, by_subj):
    return [sid for sid in sids if sid in by_subj]

def subject_edge_mean(by_subj, sid):
    # stack adj over segments: (T, N, N)
    A = torch.stack([d["adj"] for d in by_subj[sid]], dim=0).float()
    return A.mean(dim=0)  # (N, N)

def build_normative_stats(by_subj, control_subject_ids, eps=1e-8):
    subj_means = []
    for sid in control_subject_ids:
        subj_means.append(subject_edge_mean(by_subj, sid))
    M = torch.stack(subj_means, dim=0)  # (S0, N, N)

    mu0 = M.mean(dim=0)                 # (N,N)
    sigma0 = M.std(dim=0, unbiased=True).clamp_min(eps)  # (N,N)
    return mu0, sigma0

def bootstrap_ci_width_edge_mean(by_subj, sid, B=200, ci=(0.025, 0.975), seed=0):
    A = torch.stack([d["adj"] for d in by_subj[sid]], dim=0).float()  # (T,N,N)
    T = A.shape[0]
    g = torch.Generator().manual_seed(seed)

    boot_means = []
    for _ in range(B):
        idx = torch.randint(0, T, (T,), generator=g)
        boot_means.append(A[idx].mean(dim=0))
    boot = torch.stack(boot_means, dim=0)  # (B,N,N)

    lo = boot.quantile(ci[0], dim=0)
    hi = boot.quantile(ci[1], dim=0)
    width = hi - lo                           # (N,N)
    return A.mean(dim=0), width               # (N,N), (N,N)
def stability_mask_from_ciwidth(ciwidth, keep_frac=0.30):
    # only consider upper triangle (i<j)
    N = ciwidth.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    vals = ciwidth[iu[0], iu[1]]
    thr = torch.quantile(vals, keep_frac)   # keep smallest widths
    mask = (ciwidth <= thr)
    mask.fill_diagonal_(False)
    return mask

def normal_cdf(x):
    # Phi(x) using erf
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def zscore_and_p(Abar, mu0, sigma0):
    z = (Abar - mu0) / sigma0
    p = 2.0 * (1.0 - normal_cdf(z.abs()))
    p.fill_diagonal_(1.0)
    return z, p

def bh_fdr_mask(p_matrix, alpha=0.05):
    """
    Benjamini–Hochberg FDR on the upper triangle (i<j), returns NxN bool mask.
    """
    N = p_matrix.shape[0]
    iu = torch.triu_indices(N, N, offset=1)
    p = p_matrix[iu[0], iu[1]]
    m = p.numel()

    p_sorted, _ = torch.sort(p)
    thresh = alpha * torch.arange(1, m+1, device=p.device) / m
    ok = p_sorted <= thresh

    if not ok.any():
        mask = torch.zeros((N, N), dtype=torch.bool, device=p.device)
        return mask

    k = torch.max(torch.where(ok)[0]).item()
    cutoff = p_sorted[k]
    sig = p <= cutoff

    mask = torch.zeros((N, N), dtype=torch.bool, device=p.device)
    mask[iu[0], iu[1]] = sig
    mask = mask | mask.T
    mask.fill_diagonal_(False)
    return mask

def subject_fixed_edge_mask(
    by_subj, sid, mu0, sigma0,
    B=200, keep_frac=0.30, alpha=0.05):

    # 1) subject average + stability (bootstrap CI width)
    Abar, ciwidth = bootstrap_ci_width_edge_mean(by_subj, sid, B=B)
    stable = stability_mask_from_ciwidth(ciwidth, keep_frac=keep_frac)

    # 2) normative z + p
    z, p = zscore_and_p(Abar, mu0, sigma0)

    # 3) FDR significant edges
    fdr_sig = bh_fdr_mask(p, alpha=alpha)

    # 4) final subject graph
    final = stable & fdr_sig
    final = final | final.T
    final.fill_diagonal_(False)

    return final, z, Abar, ciwidth, p

def mask_adj_to_edge_index_attr(adj, mask):
    idx = mask.nonzero(as_tuple=False).T  # (2, E*2)
    edge_attr = adj[idx[0], idx[1]].unsqueeze(-1)  # (E*2, 1)
    return idx, edge_attr


if __name__ == "__main__":

    T=4 #e.g: 2, 4, 6, 8, 10
    overlap= int(0.5*T) #e.g: 0.5, 0.75
    feature_list = ['rbp'] #e.g: [['rbp'], ['rbp', 'hjorth']]
    band_name = None #e.g: None, alpha, beta, theta
    edge_methods = 'plv' #e.g: ['coherence', 'pli', 'corr']
    electrode = 'bi23' #e.g: ['mono', 'bipolar']
    mono_channel = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    if electrode == "mono":
        channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    elif electrode == "bi23":
        channel_names = [
            ("F7", "F3"),
            ("F3", "Fz"),
            ("F4", "Fz"), 
            ("F8", "F4"), 
            ("F3", "C3"),
            ("Fz", "Cz"),
            ("F4", "C4"), 
            ("C3", "T3"),
            ("C3", "Cz"),
            ("C4", "Cz"),
            ("C4", "T4"), 
            ("T3", "T5"),
            ("C3", "P3"),
            ("Cz", "Pz"),
            ("C4", "P4"), 
            ("T4", "T6"), 
            ("P3", "Pz"),
            ("P4", "Pz"),
            ("P4", "O2"), 
            ("P3", "O1"),
            ("T6", "O2"), 
            ("T5", "O1"),
            ("O1", "O2")
        ]
    elif electrode == "bi30":
        channel_names = [
            ("F7", "Fp1"),
            ("F3", "Fp1"),
            ("Fp1", "Fz"), 
            ("F8", "Fp2"), 
            ("F4", "Fp2"),
            ("Fz", "Fp2"),
            ("Fp1", "Fp2"), 
            ("F7", "F3"),
            ("F3", "Fz"),
            ("F4", "Fz"), 
            ("F8", "F4"), 
            ("F3", "C3"),
            ("Fz", "Cz"),
            ("F4", "C4"), 
            ("C3", "T3"),
            ("C3", "Cz"),
            ("C4", "Cz"),
            ("C4", "T4"), 
            ("T3", "T5"),
            ("C3", "P3"),
            ("Cz", "Pz"),
            ("C4", "P4"), 
            ("T4", "T6"), 
            ("P3", "Pz"),
            ("P4", "Pz"),
            ("P4", "O2"), 
            ("P3", "O1"),
            ("T6", "O2"), 
            ("T5", "O1"),
            ("O1", "O2")
        ]

    n_channels = len(channel_names)

    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
        }

    channel_positions_2d = {
        'Fp1': (-0.3, 1.1), 'Fp2': (0.3, 1.1),
        'O1': (-0.3, -1.2), 'O2': (0.3, -1.2),
        'Fz': (0, 0.6),
        'Cz': (0, 0.1),
        'Pz': (0, -0.5),
        'F3': (-0.85, 0.9),'F4': (0.85, 0.9), 
        'P3': (-0.85, -0.9), 'P4': (0.85, -0.9), 
        'T3': (-1.3, 0.0), 'T4': (1.3, 0.0),
        'F7': (-1.15, 0.4), 'F8': (1.15, 0.4),
        'T5': (-1.15, -0.4), 'T6': (1.15, -0.4),
        'C3': (-0.75, 0.15), 'C4': (0.75, 0.15),
        }

    regions = {
        "frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
        "temporal": ["T3", "T4", "T5", "T6"],
        "parietal": ["P3", "P4", "Pz"],
        "occipital": ["O1", "O2"],
        "central": ["C3", "C4", "Cz"]
        }


    dataset = 'aheap'
    class_set = 'all3'
    sfreq=500

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)

    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # data_paths, labels, sub_id_list = data_paths[:2] + data_paths[36:38] + data_paths[-2:], labels[:2] + labels[36:38] + labels[-2:], sub_id_list[:2] +sub_id_list[36:38] + sub_id_list[-2:]
    save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/EDA_stats/significant_edges_subject'
    os.makedirs(save_path,exist_ok = True)
    randomstate_value = 15
    k=10
    CONTROL_LABEL = 0
    all_folds = balanced_kfold_split(sub_id_list, labels, randomstate_value, k)

    if electrode == 'mono':
        formatted_channels = channel_names
    else: 
        formatted_channels = [f"{pair[0]}-{pair[1]}" for pair in channel_names]
        # formatted_channels = bipolar_names

    # #### -----------------------------------------------------
    data_processed_path = f"/home/anphan/Documents/EEG_Project/AHEAP_data/EDA_stats/analysis_Jan2026/bi23_rbp_plv_None/data_processed"
    all_data_path = f"{data_processed_path}/master_graph_data.pt"
    master_data = torch.load(all_data_path)

    by_subj = defaultdict(list)
    for item in master_data:
        sid = item["subject_id"]
        by_subj[sid].append(item)

    # Optional: sort segments by time
    for sid in by_subj:
        by_subj[sid].sort(key=lambda d: d["segment_id"])
    print(by_subj)


    for fold_idx, test_fold in enumerate(all_folds):
        test_subject_ids = all_folds[fold_idx]
        train_subject_ids = [sub_id for sub_id in sub_id_list if sub_id not in test_subject_ids]
        train_control_subject_ids = [
            sid for sid in train_subject_ids
            if get_label_of_subject(by_subj, sid) == CONTROL_LABEL
        ]
        print(f"[Fold {fold_idx}] n_train_controls:", len(train_control_subject_ids))

        # build normative reference from training controls only
        mu0, sigma0 = build_normative_stats(by_subj, train_control_subject_ids)

        # build subject-specific graphs for all subjects in this fold
        fold_subject_ids = train_subject_ids + test_subject_ids

        subject_graphs = {}

        all_rows = []

        for sid in fold_subject_ids:
            mask, z, Abar, ciwidth, p = subject_fixed_edge_mask(
                by_subj, sid, mu0, sigma0,
                B=200, keep_frac=0.30, alpha=0.05
            )

            E = int(mask.sum().item() / 2)
            print(E)
            if E < 20:
                # fallback (relax thresholds)
                mask, z, Abar, ciwidth, p = subject_fixed_edge_mask(
                    by_subj, sid, mu0, sigma0,
                    B=200, keep_frac=0.50, alpha=0.10
                )

            subject_graphs[sid] = {
                "edge_mask": mask.cpu(),
                "Abar": Abar.cpu(),
                "z": z.cpu(),
                "ciwidth": ciwidth.cpu(),
            }

            df_sub = subject_edges_to_df(
                sid,
                mask,
                z,
                Abar,
                ciwidth,
                channel_names=channel_names   # optional
            )

            all_rows.append(df_sub)
        out_path = f"{save_path}/fold{fold_idx}_subject_fixed_graphs.pt"
        torch.save(subject_graphs, out_path)
        print(f"[Fold {fold_idx}] saved:", out_path)
        df_edges = pd.concat(all_rows, ignore_index=True)
        df_edges = df_edges.sort_values("abs_z", ascending=False)
        csv_path = f"{save_path}/fold{fold_idx}_significant_edges.csv"

        df_edges.to_csv(csv_path, index=False)

        print("Saved:", csv_path)


