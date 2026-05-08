import torch
import numpy as np
# from viz_plots import plot_heatmap, plot_class_mean_connectivity_multiplot
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from scipy import stats
import ast
from pathlib import Path
import pandas as pd
from statsmodels.stats.multitest import multipletests
from mne_connectivity.viz import plot_connectivity_circle

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from data_utils import *
from data_preparation import *
from utils_all import *

def build_segment_data_matrices(
    data_window,
    label,
    sfreq,
    bands_dict,
    feature_list,
    node_features_fn,        
    edge_method="coherence",
    band=None
):
    n_nodes = data_window.shape[0]
    adj_matrix = torch.zeros((n_nodes, n_nodes), dtype=torch.float32)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            sig1, sig2 = data_window[i], data_window[j]

            if edge_method == "corr":
                w = compute_corr(sig1, sig2, method="pearson")
            elif edge_method == "spearman":
                w = compute_corr(sig1, sig2, method="spearman")
            elif edge_method == "pli":
                w = compute_pli(sig1, sig2, sfreq, bands_dict, band)
            elif edge_method == "plv":
                w = compute_phase_lag(sig1, sig2)
            elif edge_method == "coherence":
                w = compute_coherence(sig1, sig2, sfreq, bands_dict, band)
            else:
                raise ValueError(f"Unknown edge_method: {edge_method}")

            w = float(w)
            adj_matrix[i, j] = w
            adj_matrix[j, i] = w

    band_node = None
    x, _ = node_features_fn(data_window, sfreq, bands_dict, feature_list, band_node)

    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)

    y = torch.tensor([label], dtype=torch.long)

    return x, adj_matrix, y



def get_class_avg(data_list, c_id):
    subs = set(d['subject_id'] for d in data_list if d['class_id'] == c_id)
    sub_avgs = []
    for s in subs:
        mats = [d['adj'] for d in data_list if d['subject_id'] == s]
        sub_avgs.append(torch.stack(mats).mean(dim=0))
    return torch.stack(sub_avgs).mean(dim=0)


def create_feature_dataframe(data_list, channel_names, feature_names):
    rows = []
    for d in data_list:
        # d['node_features'] is (n_channels, m_features)
        feats = d['node_features'].numpy()
        for ch_idx, ch_name in enumerate(channel_names):
            row = {
                'subject_id': d['subject_id'],
                'class_id': d['class_id'],
                'channel': ch_name,
                'segment_id': d['segment_id']
            }
            # Map each feature value to its name
            for f_idx, f_name in enumerate(feature_names):
                row[f_name] = feats[ch_idx, f_idx]
            rows.append(row)
    return pd.DataFrame(rows)

def save_channel_summaries(data_list, channel_names, feature_names):
    summary_rows = []
    
    # Get unique subjects
    sids = set(d['subject_id'] for d in data_list)
    
    for sid in sids:
        # Get all segments for this subject
        subj_data = [d for d in data_list if d['subject_id'] == sid]
        cid = subj_data[0]['class_id']
        
        # Average features across segments for this subject: (Channels, Features)
        avg_feats = torch.stack([d['node_features'] for d in subj_data]).mean(dim=0).numpy()
        
        for ch_idx, ch_name in enumerate(channel_names):
            row = {'subject_id': sid, 'class_id': cid, 'channel': ch_name}
            for f_idx, f_name in enumerate(feature_names):
                row[f_name] = avg_feats[ch_idx, f_idx]
            summary_rows.append(row)
            
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("channel_feature_summary.csv", index=False)
    return df_summary

def get_expanded_feature_names(feature_list, bands):
    expanded_names = []
    for feat in feature_list:
        if feat == 'rbp':
            # Add a name for each band (e.g., rbp_delta, rbp_alpha...)
            for band_name in bands.keys():
                expanded_names.append(f"rbp_{band_name}")
        elif feat == 'hjorth':
            # Hjorth usually returns 3 values: Activity, Mobility, Complexity
            expanded_names.extend(['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'])
        elif feat == 'energies':
            # Level 5 wavelet decomposition returns 6 energy values 
            # (5 detail levels + 1 approximation level)
            for i in range(1, 6):
                expanded_names.append(f"energy_d{i}")
            expanded_names.append("energy_a5")
        else:
            expanded_names.append(feat)
    return expanded_names


def compute_bipolar_segment(eeg_segment, channel_names, bipolar_pairs):
    """
    eeg_segment: shape [n_channels, n_samples]
    channel_names: list of referential channel names
    returns:
        bipolar_segment: shape [n_bipolar_channels, n_samples]
        bipolar_names: list[str]
    """
    ch_to_idx = {ch: i for i, ch in enumerate(channel_names)}

    bipolar_data = []
    bipolar_names = []

    for ch_a, ch_b in bipolar_pairs:
        if ch_a not in ch_to_idx or ch_b not in ch_to_idx:
            raise ValueError(f"Missing channel for bipolar pair: {ch_a}-{ch_b}")

        x = eeg_segment[ch_to_idx[ch_a]] - eeg_segment[ch_to_idx[ch_b]]
        bipolar_data.append(x)
        bipolar_names.append(f"{ch_a}-{ch_b}")

    bipolar_segment = np.stack(bipolar_data, axis=0).astype(np.float32)
    return bipolar_segment, bipolar_names

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--feature_lists", type=str, required=False, help="Feature Lists")
    # parser.add_argument("--duration", type=int, required=False, help="Window Length")
    # parser.add_argument("--overlap", type=float, required=False, help="overlap ratio")
    parser.add_argument("--edge_methods", type=str, required=False, help="Edge weight methods")
    # parser.add_argument("--band", type=str, required=False, help="Specific Band Name")
    parser.add_argument("--electrode", type=str, required=False, help="Channels setting (19/23/30 channels)")
    
    args = parser.parse_args()

    feature_list = args.feature_lists #e.g: [['rbp'], ['rbp', 'hjorth']]
    edge_methods = args.edge_methods #e.g: ['coherence', 'pli', 'corr']
    electrode = args.electrode #e.g: ['mono', 'bipolar']



    class_set = 'all3'
    band_name = None
    sfreq=500
    T=4
    overlap= 0.5

    import config
    bands = config.BANDS_DICT
    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    fixed_edges = config.MONOFIXEDGES
    bi23_channel_names = config.bi23_channel_names

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)

    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # data_paths, labels, sub_id_list = data_paths[:2] + data_paths[36:38] + data_paths[-2:], labels[:2] + labels[36:38] + labels[-2:], sub_id_list[:2] +sub_id_list[36:38] + sub_id_list[-2:]
    save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_cleaned_feature_data'
    os.makedirs(save_path,exist_ok = True)

    if isinstance(feature_list, str):
        feature_list = ast.literal_eval(feature_list)

    feature_name = ''.join(feature_list)
    actual_feature_names = get_expanded_feature_names(feature_list, bands)
    print(len(actual_feature_names), actual_feature_names)
    folder_path = os.path.join(save_path, f"{electrode}_{feature_name}_{edge_methods}_{band_name}_{T}_{overlap}")
    os.makedirs(folder_path, exist_ok=True)
    all_data = []
    log_path = os.path.join(folder_path, f"log.txt")

    master_dir = Path("/home/anphan/Documents/EEG_Project/AHEAP_data/master_clean_data")

    manifest = pd.read_csv(master_dir / "subject_manifest.csv")
    manifest = manifest[manifest["use_subject"] == 1].copy()

    for _, row in manifest.iterrows():
        subject_id = row["subject_id"]
        pt_path = master_dir / f"{subject_id}.pt"

        obj = torch.load(pt_path, weights_only=False)
        channel_names = obj["channel_names"]

        sid = obj["subject_id"]
        label = obj["class_id"]

        for seg in obj["segments"]:
            seg_idx = seg["segment_id"]
            eeg_segment = seg["eeg_segment"]      # [channels, timepoints]
            start_sample = seg["start_sample"]

            if electrode == 'bi23':
                bipolar_segment, bipolar_names = compute_bipolar_segment(
                    eeg_segment=eeg_segment,
                    channel_names=channel_names,
                    bipolar_pairs=bi23_channel_names
                )
                eeg_window = bipolar_segment
            elif electrode == 'mono':
                eeg_window = eeg_segment


            x, adj_matrix, y = build_segment_data_matrices(
                eeg_window,
                label,
                sfreq,
                bands,
                feature_list,
                node_features,
                edge_methods,
                band_name
            )
            if seg_idx < 1:
                print("x", x.shape)
                print("adj_matrices", adj_matrix.shape)

            
            all_data.append({
                'subject_id': sid,
                'class_id': label,
                'adj': adj_matrix,     # The N x N weight matrix
                'node_features': x,                # The node features
                'segment_id': seg_idx,
                'start_sample': start_sample
            })
    ### -----------------------------------------------------
    torch.save(all_data, f"{folder_path}/master_graph_data.pt")