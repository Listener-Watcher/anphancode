import torch
import numpy as np
from viz_plots import plot_heatmap, plot_class_mean_connectivity_multiplot
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import ast
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
            adj_matrix[i, j] = w
            adj_matrix[j, i] = w
    band_node = None
    x, _ = node_features_fn(data_window, sfreq, bands_dict, feature_list, band_node)
    y = torch.tensor([label], dtype=torch.long)
    return x, adj_matrix, y

def plot_all_subjects_heatmaps(data_list, channel_names, save_folder):
    subject_ids = sorted(list(set(d['subject_id'] for d in data_list)))
    print(f"Found {len(subject_ids)} subjects. Generating heatmaps...")

    for s_id in subject_ids:
        subject_matrices = [d['adj'] for d in data_list if d['subject_id'] == s_id]
        avg_matrix = torch.stack(subject_matrices).mean(dim=0).numpy()
        plt.figure(figsize=(8, 6))
        plt.imshow(matrix, cmap="viridis")
        plt.colorbar()
        plt.xticks(range(len(channel_names)), channel_names, rotation=45)
        plt.yticks(range(len(channel_names)), channel_names)
        class_id = [d['class_id'] for d in data_list if d['subject_id'] == s_id][0]
        plt.title(f"Subject: {s_id} (Class: {class_id})")
        plt.tight_layout()
        plt.savefig(f"{save_folder}/{s_id}_heatmap.png")
        plt.close() 

    print(f"Done! All heatmaps saved to '{save_folder}'")


def plot_connectivity_heatmap(data_list, channel_names, class_names, save_path, title):
    subject_folder = os.path.join(save_path, "subject")
    class_folder = os.path.join(save_path, "class")
    os.makedirs(subject_folder, exist_ok=True)
    os.makedirs(class_folder, exist_ok=True)

    subject_data_cache = {}
    subject_ids = sorted(list(set(d['subject_id'] for d in data_list)))
    
    print(f"Processing {len(subject_ids)} subjects...")

    for s_id in subject_ids:
        s_entries = [d for d in data_list if d['subject_id'] == s_id]
        class_id = s_entries[0]['class_id']
        s_matrices = [d['adj'] for d in s_entries]
        
        avg_matrix = torch.stack(s_matrices).mean(dim=0).numpy()
        subject_data_cache[s_id] = (class_id, avg_matrix)

        plt.figure(figsize=(8, 6))
        im = plt.imshow(avg_matrix, cmap="viridis")
        plt.colorbar(im)
        plt.xticks(range(len(channel_names)), channel_names, rotation=45, ha='right')
        plt.yticks(range(len(channel_names)), channel_names)
        plt.title(f"Subject: {s_id} (Class: {class_names[class_id]})")
        plt.tight_layout()
        plt.savefig(os.path.join(subject_folder, f"{s_id}_heatmap.png"))
        plt.close()

    print("Generating class-level multiplot...")
    class_mean_dict = {}
    unique_classes = np.unique([d['class_id'] for d in data_list])

    for c in unique_classes:
        avgs_for_this_class = [
            torch.from_numpy(val[1]) 
            for val in subject_data_cache.values() if val[0] == c
        ]
        
        if avgs_for_this_class:
            class_mean_dict[c] = torch.stack(avgs_for_this_class).mean(dim=0).numpy()

    plot_class_mean_connectivity_multiplot(
        class_mean=class_mean_dict, 
        class_names=class_names, 
        channel_names=channel_names, 
        folder_name=title, 
        save_path=os.path.join(class_folder, "class_comparison_heatmap.png")
    )
    
    print(f"Done! Plots saved in {save_path}")

def plot_contrast_multiplot(contrast_dict, comparison_names, channel_names, folder_name, save_path):
    keys = sorted(contrast_dict.keys())
    n_plots = len(keys)

    plt.figure(figsize=(6 * n_plots, 6))

    for i, k in enumerate(keys):
        mat = contrast_dict[k]
        ax = plt.subplot(1, n_plots, i + 1)
        
        vmax = np.abs(mat).max()
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_title(comparison_names[i], fontsize=12)
        ax.set_xticks(range(len(channel_names)))
        ax.set_yticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_yticklabels(channel_names)

        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(f"{folder_name} — Class Contrast Comparison", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def get_class_avg(data_list, c_id):
    subs = set(d['subject_id'] for d in data_list if d['class_id'] == c_id)
    sub_avgs = []
    for s in subs:
        mats = [d['adj'] for d in data_list if d['subject_id'] == s]
        sub_avgs.append(torch.stack(mats).mean(dim=0))
    return torch.stack(sub_avgs).mean(dim=0)

def plot_class_contrast(data_list, channel_names, save_path):
    unique_classes = sorted(list(set(d['class_id'] for d in data_list)))
    class_avgs = {c: get_class_avg(data_list, c) for c in unique_classes}
    pairs = [(0, 1), (0, 2), (1, 2)]
    contrast_data = {}
    contrast_labels = []

    for i, (a, b) in enumerate(pairs):
        contrast_data[i] = (class_avgs[a] - class_avgs[b]).numpy()
        contrast_labels.append(f"Class {a} vs {b}")
    plot_contrast_multiplot(
        contrast_dict=contrast_data,
        comparison_names=contrast_labels,
        channel_names=channel_names,
        folder_name="Group Analysis",
        save_path=f"{save_path}/combined_class_contrasts.png"
    )

def save_summary_statistics(data_list, filename="analysis_summaries.pt"):
    summaries = {
        'class_avgs': {},    # Mean Adjacency per class
        'class_contrasts': {}, # Difference matrices
        # 'node_feat_df': None,  # Flattened dataframe for boxplots/stats
    }

    # --- 1. Compute and Store Class Adjacency Averages ---
    unique_classes = sorted(list(set(d['class_id'] for d in data_list)))
    
    for c in unique_classes:
        # Get subject-level averages first to ensure 101-subject balance
        subs = set(d['subject_id'] for d in data_list if d['class_id'] == c)
        sub_mats = []
        for s in subs:
            mats = [d['adj'] for d in data_list if d['subject_id'] == s]
            sub_mats.append(torch.stack(mats).mean(dim=0))
        
        summaries['class_avgs'][c] = torch.stack(sub_mats).mean(dim=0)

    # --- 2. Compute and Store Class Contrasts (0 vs 1, etc.) ---
    import itertools
    for c1, c2 in itertools.combinations(unique_classes, 2):
        summaries['class_contrasts'][f"{c1}_vs_{c2}"] = summaries['class_avgs'][c1] - summaries['class_avgs'][c2]

    torch.save(summaries, filename)
    print(f"Summary statistics saved to {filename}")

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

def plot_boxplot_by_subject(df, sid, feature_name, save_folder):

    subj_df = df[df['subject_id'] == sid]
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=subj_df, x='channel', y=feature_name)
    plt.title(f"Subject {sid}: {feature_name} distribution per Channel")
    plt.xticks(rotation=45)
    plt.savefig(f"{save_folder}/{feature_name}_boxplot_{sid}.png")
    plt.close()

def plot_boxplot_by_class(df, feature_name, save_folder):
    plt.figure(figsize=(12, 5))
    # We use the whole dataframe here
    sns.boxplot(data=df, x='channel', y=feature_name, hue='class_id')
    plt.title(f"All Subjects: {feature_name} distribution per Channel by Class")
    plt.xticks(rotation=45)
    plt.savefig(f"{save_folder}/{feature_name}_boxplot.png")
    plt.close()

def plot_heatmap_by_subject(df, sid, feature_names, save_folder):
    subj_avg = df[df['subject_id'] == sid].groupby('channel')[feature_names].mean()
    subj_avg_transposed = subj_avg.T
    plt.figure(figsize=(10, 4))
    sns.heatmap(subj_avg_transposed, cmap='viridis', annot=False)
    plt.title(f"Feature Heatmap: Subject {sid}")
    plt.xlabel("Channels")
    plt.ylabel("Features")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(f"{save_folder}/heatmaps_feature_{sid}.png")
    plt.close()

def plot_heatmaps_by_class(df, feature_names, save_folder):
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    classes = sorted(df['class_id'].unique())
    
    for i, c_id in enumerate(classes):
        class_avg = df[df['class_id'] == c_id].groupby('channel')[feature_names].mean()
        sns.heatmap(class_avg, ax=axes[i], cmap='viridis', cbar=(i==2))
        axes[i].set_title(f"Class {c_id} Average")
        
    plt.suptitle("Average Node Features per Class")
    plt.savefig(f"{save_folder}/heatmaps_feature.png")
    plt.close()
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
            # expanded_names.extend(['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'])
            expanded_names.extend(['hjorth_mobility', 'hjorth_complexity'])
        elif feat == 'energies':
            # Level 5 wavelet decomposition returns 6 energy values 
            # (5 detail levels + 1 approximation level)
            for i in range(1, 6):
                expanded_names.append(f"energy_d{i}")
            expanded_names.append("energy_a5")
        else:
            expanded_names.append(feat)
    return expanded_names

from scipy import stats

def run_stats_on_wide_df(df_wide, band_name, log_path):
    band_cols = [c for c in df_wide.columns if band_name in c]
    
    df_wide[f'global_{band_name}'] = df_wide[band_cols].mean(axis=1)
    
    groups = [df_wide[df_wide['class_id'] == i][f'global_{band_name}'] for i in [0, 1, 2]]
    f_stat, p_val = stats.f_oneway(*groups)
    
    with open(log_path, "a") as f:
        f.write(f"ANOVA for {band_name}: F={f_stat:.2f}, p={p_val:.4f}\n")


def plot_band_distribution(df, band_name, save_folder):
    plt.figure(figsize=(10, 6))
    
    # We use the long-format dataframe directly
    sns.boxplot(data=df, x='class_id', y=band_name, palette='viridis')
    
    plt.title(f'Distribution of {band_name.upper()} Power across Classes')
    plt.xlabel('Class ID')
    plt.ylabel(f'Log Power ({band_name})')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{save_folder}/{band_name}_band_distribution.png")
    plt.close()

def perform_anova(df, band_name, log_path):
    df_grouped = df.groupby(['subject_id', 'segment_id', 'class_id'])[band_name].mean().reset_index()
    
    # 2. Separate the data by class
    class_0 = df_grouped[df_grouped['class_id'] == 0][band_name]
    class_1 = df_grouped[df_grouped['class_id'] == 1][band_name]
    class_2 = df_grouped[df_grouped['class_id'] == 2][band_name]
    
    # 3. Run One-Way ANOVA
    f_stat, p_val = stats.f_oneway(class_0, class_1, class_2)
    with open(log_path, "a") as f:
        f.write(f"--- Statistical Results for {band_name.upper()} ---\n")
        f.write(f"F-Statistic: {f_stat:.4f}\n")
        f.write(f"p-value: {p_val:.4f}\n")
    
    if p_val < 0.05:
        with open(log_path, "a") as f:
            f.write(f"Result: Significant difference found between classes.\n")
    else:
        with open(log_path, "a") as f:
            f.write(f"Result: No significant difference found.\n")
    
    return f_stat, p_val

def analyze_connectivity_anova(adj_matrices, labels):
    """
    adj_matrices: List or array of shape (N_samples, 19, 19)
    labels: Array of shape (N_samples,)
    """
    n_samples, n_channels, _ = adj_matrices.shape
    # We only take the upper triangle to avoid redundant pairs
    tri_indices = np.triu_indices(n_channels, k=1)
    
    results = []
    
    for i, j in zip(*tri_indices):
        # Extract the specific connection strength for all samples
        connection_series = adj_matrices[:, i, j]
        
        group0 = connection_series[labels == 0]
        group1 = connection_series[labels == 1]
        group2 = connection_series[labels == 2]
        
        f_stat, p_val = stats.f_oneway(group0, group1, group2)
        
        if p_val < 0.05: # Only store significant ones
            results.append({
                'edge': f"{channel_names[i]}-{channel_names[j]}",
                'f_stat': f_stat,
                'p_val': p_val
            })
            
    return pd.DataFrame(results).sort_values('p_val')


def analyze_connectivity_stats(adj_matrices, labels, channel_names):
    """
    adj_matrices: ndarray of shape (N_samples, 19, 19)
    labels: ndarray of shape (N_samples,)
    """
    n_samples, n_ch, _ = adj_matrices.shape
    # Get indices for the upper triangle (excluding diagonal)
    tri_idx = np.triu_indices(n_ch, k=1)
    
    edge_results = []
    
    for i, j in zip(*tri_idx):
        # Extract this specific edge strength for all samples
        edge_data = adj_matrices[:, i, j]
        
        # Split by class
        g0 = edge_data[labels == 0]
        g1 = edge_data[labels == 1]
        g2 = edge_data[labels == 2]
        
        f_stat, p_val = stats.f_oneway(g0, g1, g2)
        
        edge_results.append({
            'edge': f"{channel_names[i]}-{channel_names[j]}",
            'f_stat': f_stat,
            'p_val': p_val,
            'ch_i': i,
            'ch_j': j
        })
    
    df_stats = pd.DataFrame(edge_results)
    
    # IMPORTANT: Multiple Comparison Correction (FDR)
    # Testing 171 edges requires correction to avoid False Positives
    _, p_adj, _, _ = multipletests(df_stats['p_val'], method='fdr_bh')
    df_stats['p_corrected'] = p_adj
    
    return df_stats

def analyze_connectivity_with_fdr(adj_matrices, labels, channel_names):
    """
    adj_matrices: ndarray (N_samples, 19, 19)
    labels: ndarray (N_samples,)
    """
    n_samples, n_ch, _ = adj_matrices.shape
    tri_idx = np.triu_indices(n_ch, k=1) # Upper triangle indices
    
    edge_stats = []
    
    # 1. Loop through every unique electrode pair
    for i, j in zip(*tri_idx):
        edge_data = adj_matrices[:, i, j]
        
        # Split by class (0, 1, 2)
        groups = [edge_data[labels == c] for c in np.unique(labels)]
        f_stat, p_val = stats.f_oneway(*groups)
        
        edge_stats.append({
            'edge': f"{channel_names[i]}-{channel_names[j]}",
            'ch_i': i,
            'ch_j': j,
            'f_stat': f_stat,
            'p_val': p_val
        })
    
    df = pd.DataFrame(edge_stats)
    
    # 2. Apply FDR (Benjamini-Hochberg) Correction
    _, p_adj, _, _ = multipletests(df['p_val'], method='fdr_bh')
    df['p_corrected'] = p_adj
    
    # 3. Filter for truly significant edges
    significant_edges = df[df['p_corrected'] < 0.05]
    return df, significant_edges

def plot_significant_mask(df_stats, n_channels, channel_names, save_folder, level = 'segment'):
    # Initialize an empty matrix
    mask_matrix = np.zeros((n_channels, n_channels))
    
    # Fill only significant edges (using F-statistic as intensity)
    for _, row in df_stats.iterrows():
        if row['p_corrected'] < 0.05:
            mask_matrix[int(row['ch_i']), int(row['ch_j'])] = row['f_stat']
            mask_matrix[int(row['ch_j']), int(row['ch_i'])] = row['f_stat']
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(mask_matrix, xticklabels=channel_names, yticklabels=channel_names, 
                cmap='YlOrRd', annot=False)
    plt.title("Significant Connectivity Hubs (FDR Corrected p < 0.05)")
    plt.savefig(f"{save_folder}/significant_mask_plot_{level}.png")
    plt.close()

def get_top_biomarkers(df_sig, log_path, level = 'segment', top_n=20):
    # Sort by F-statistic descending (highest impact first)
    top_edges = df_sig.sort_values(by='f_stat', ascending=False).head(top_n)
    
    with open(log_path, "a") as f:
        f.write(f"------------------{level}------------------\n")
        f.write(f"--- Top {top_n} Connectivity Biomarkers ---\n")
    for idx, row in top_edges.iterrows():
        with open(log_path, "a") as f:
            f.write(f"Edge {row['edge']}: F={row['f_stat']:.2f}, p-adj={row['p_corrected']:.4e}\n")
    
    return top_edges


def plot_significant_chord(df_sig, channel_names, save_folder):
    n_nodes = len(channel_names)
    # Create an empty connectivity matrix
    con = np.zeros((n_nodes, n_nodes))
    
    # Fill with significant F-statistics
    for _, row in df_sig.iterrows():
        con[row['ch_i'], row['ch_j']] = row['f_stat']
        
    fig, ax = plot_connectivity_circle(
        con, channel_names, n_lines=None, 
        title="Significant Class-Level Connectivity",
        colormap='YlOrRd', facecolor='white', textcolor='black',
        show=False
    )
    fig.savefig(f"{save_folder}/significant_chord.png", dpi=300, bbox_inches='tight')
    print(f"Figure saved to {save_folder}")
    return fig


def plot_top_significant_chord(df_sig, channel_names, save_folder, top_n=20):
    # Take only the top N strongest connections
    df_top = df_sig.sort_values(by='f_stat', ascending=False).head(top_n)
    
    n_nodes = len(channel_names)
    con = np.zeros((n_nodes, n_nodes))
    
    for _, row in df_top.iterrows():
        con[int(row['ch_i']), int(row['ch_j'])] = row['f_stat']
        
    fig, ax = plot_connectivity_circle(
        con, channel_names, n_lines=top_n, 
        title=f"Top {top_n} Class-Level Biomarkers",
        colormap='YlOrRd', facecolor='white', textcolor='black'
    )
    
    fig.savefig(f"{save_folder}/top{top_n}_significant_chord.png", dpi=300, bbox_inches='tight')
        
    return fig


def save_node_feature_stats(df_long, band_names, save_folder):
    for level in ['segment', 'subject']:
        results = []
        
        # Determine grouping
        if level == 'segment':
            # Every segment is a unique data point
            df_grouped = df_long.copy()
        else:
            # Average segments per subject first
            df_grouped = df_long.groupby(['subject_id', 'class_id'])[band_names].mean().reset_index()

        for band in band_names:
            # Split into groups for ANOVA
            groups = [df_grouped[df_grouped['class_id'] == c][band] for c in np.unique(df_long['class_id'])]
            f_stat, p_val = stats.f_oneway(*groups)
            
            results.append({
                'Band': band,
                'F_Statistic': f_stat,
                'p_value': p_val,
                'Significant': p_val < 0.05,
                'N_Samples': len(df_grouped)
            })
        
        # Save CSV
        df_res = pd.DataFrame(results)
        df_res.to_csv(f"{save_folder}/node_feature_anova_{level}.csv", index=False)
        print(f"Stats for {level} level saved (N={len(df_grouped)})")
    # results = []
    # # Use the long-format dataframe from your create_feature_dataframe function
    # for band in band_names:
    #     # Group by subject to get one value per person for valid ANOVA
    #     df_grouped = df_long.groupby(['subject_id', 'class_id'])[band].mean().reset_index()
        
    #     groups = [df_grouped[df_grouped['class_id'] == c][band] for c in [0, 1, 2]]
    #     f_stat, p_val = stats.f_oneway(*groups)
        
    #     results.append({
    #         'Band': band,
    #         'F_Statistic': f_stat,
    #         'p_value': p_val,
    #         'Significant': p_val < 0.05
    #     })
    
    # df_res = pd.DataFrame(results)
    # # df_res.to_csv(f"{save_folder}/node_feature_anova_subject.csv", index=False)
    # print(f"Node feature stats saved!")
    # return df_res

def save_top_edge_comparison(df_sig, adj_matrices, labels, channel_names, save_folder, level = 'segment',  top_n=30):
    # Sort by strongest biomarkers
    top_edges = df_sig.sort_values(by='f_stat', ascending=False).head(top_n)
    comparison_rows = []

    for _, row in top_edges.iterrows():
        i, j = int(row['ch_i']), int(row['ch_j'])
        edge_data = adj_matrices[:, i, j]
        
        comparison_rows.append({
            'Edge': row['edge'],
            'F_Stat': row['f_stat'],
            'Mean_Class_0': np.mean(edge_data[labels == 0]),
            'Mean_Class_1': np.mean(edge_data[labels == 1]),
            'Mean_Class_2': np.mean(edge_data[labels == 2]),
            'p_corrected': row['p_corrected']
        })

    df_comp = pd.DataFrame(comparison_rows)
    df_comp.to_csv(f"{save_folder}/top_biomarker_comparison_{level}.csv", index=False)
    return df_comp
import torch

def subject_standardize_master(master_in, master_out, eps=1e-8):
    all_data = torch.load(master_in)

    # group indices by subject
    subj_to_idx = {}
    for i, e in enumerate(all_data):
        sid = e["subject_id"]
        subj_to_idx.setdefault(sid, []).append(i)

    # standardize per subject
    for sid, idxs in subj_to_idx.items():
        X = torch.stack(
            [all_data[i]["node_features"] for i in idxs],
            dim=0
        )  # [S,19,F]

        mu = X.mean(dim=0)                           # [19,F]
        sd = X.std(dim=0).clamp_min(eps)             # [19,F]

        for i in idxs:
            all_data[i]["node_features"] = (
                all_data[i]["node_features"] - mu
            ) / sd

    torch.save(all_data, master_out)
    print(f"Saved standardized master to: {master_out}")

    return all_data    # ✅ IMPORTANT

if __name__ == "__main__":

    # T=4 #e.g: 2, 4, 6, 8, 10
    # overlap= int(0.5*T) #e.g: 0.5, 0.75
    # feature_list = ['rbp'] #e.g: [['rbp'], ['rbp', 'hjorth']]
    # band_name = None #e.g: None, alpha, beta, theta
    # edge_methods = 'plv' #e.g: ['coherence', 'pli', 'corr']
    # electrode = 'bi23' #e.g: ['mono', 'bipolar']


    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--feature_lists", type=str, required=False, help="Feature Lists")
    parser.add_argument("--duration", type=int, required=False, help="Window Length")
    parser.add_argument("--overlap", type=float, required=False, help="overlap ratio")
    parser.add_argument("--edge_methods", type=str, required=False, help="Edge weight methods")
    parser.add_argument("--band", type=str, required=False, help="Specific Band Name")
    parser.add_argument("--electrode", type=str, required=False, help="Channels setting (19/23/30 channels)")
    
    args = parser.parse_args()
    T=args.duration #e.g: 2, 4, 6, 8, 10
    overlap= int(args.overlap*T) #e.g: 0.5, 0.75
    feature_list = args.feature_lists #e.g: [['rbp'], ['rbp', 'hjorth']]
    band_name = args.band #e.g: None, alpha, beta, theta
    edge_methods = args.edge_methods #e.g: ['coherence', 'pli', 'corr']
    electrode = args.electrode #e.g: ['mono', 'bipolar']


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
    save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data'
    # save_root = f'/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data'
    os.makedirs(save_path,exist_ok = True)
    # os.makedirs(save_root,exist_ok = True)
    randomstate_value = 15
    k=10
    all_folds = balanced_kfold_split(sub_id_list, labels, randomstate_value, k)

    if isinstance(feature_list, str):
        feature_list = ast.literal_eval(feature_list)

    feature_name = ''.join(feature_list)
    actual_feature_names = get_expanded_feature_names(feature_list, bands)
    print(len(actual_feature_names), actual_feature_names)
    # save_dir_sub = os.path.join(save_path, feature_name)
    # os.makedirs(save_dir_sub, exist_ok=True)
    folder_path = os.path.join(save_path, f"{electrode}_{feature_name}_{edge_methods}_{band_name}")
    os.makedirs(folder_path, exist_ok=True)
    data_processed_path = os.path.join(folder_path, "data_processed")
    data_processed_update = os.path.join(folder_path, "data_processed_update")
    os.makedirs(data_processed_update, exist_ok=True)
    class_folder = os.path.join(folder_path, "class")
    os.makedirs(class_folder, exist_ok=True)
    log_path = os.path.join(folder_path, f"log.txt")

    subject_folder = os.path.join(folder_path, "subject")
    os.makedirs(subject_folder, exist_ok=True)


    # Execution
    if electrode == 'mono':
        formatted_channels = channel_names
    else: 
        formatted_channels = [f"{pair[0]}-{pair[1]}" for pair in channel_names]
        # formatted_channels = bipolar_names

    # #### -----------------------------------------------------

    # all_data_path = f"{data_processed_path}/master_graph_data_update.pt"
    # all_data = torch.load(all_data_path)

    all_data = subject_standardize_master(
            master_in=f"{data_processed_path}/master_graph_data_update.pt",
            master_out=f"{data_processed_path}/master_graph_data_update_substd.pt"
        )


    for fold_idx, test_fold in enumerate(all_folds):
        test_subjects = all_folds[fold_idx]
        removed_subject_ids = set(test_subjects)  # your list -> set for fast lookup

        # keep only samples whose subject_id is NOT in the removed list
        all_data_filtered = [d for d in all_data if d["subject_id"] not in removed_subject_ids]

        print("before:", len(all_data), "after:", len(all_data_filtered))

        present_ids = {d["subject_id"] for d in all_data}
        removed_found = sorted(present_ids & removed_subject_ids)
        removed_missing = sorted(removed_subject_ids - present_ids)

        print("removed_found:", len(removed_found))
        print("removed_missing (not in file):", len(removed_missing))


        df = create_feature_dataframe(all_data_filtered, formatted_channels, actual_feature_names)
        # df.to_csv( f"{data_processed_path}/processed_feature_table.csv", index=False)
        # save_node_feature_stats(df, actual_feature_names, data_processed_path)
        ##### -----------------------------------------------------


        print("Generating Class-level plots...")

        df_wide = df.pivot_table(
            index=['subject_id', 'segment_id', 'class_id'], 
            columns='channel', 
            values=actual_feature_names
        )

        df_wide.columns = [f"{feat}_{ch}" for feat, ch in df_wide.columns]
        df_wide = df_wide.reset_index()

        feature_matrix = df_wide.drop(columns=['subject_id', 'segment_id', 'class_id']).values
        labels = df_wide['class_id'].values
        # plot_heatmaps_by_class(df, actual_feature_names, class_folder)

        # for feat in actual_feature_names:
        #     plot_boxplot_by_class(df, feat, class_folder)
        #     plot_band_distribution(df, feat, class_folder)
        #     perform_anova(df, feat, log_path)
        #     run_stats_on_wide_df(df_wide, feat, log_path)

        # print(f"Generating plots for {len(df['subject_id'].unique())} subjects...")
        # for sid in df['subject_id'].unique():
        #     subj_dir = os.path.join(subject_folder, str(sid))
        #     os.makedirs(subj_dir,exist_ok = True)
            
            # plot_heatmap_by_subject(df, sid, actual_feature_names, subj_dir)
            
            # for feat in actual_feature_names:
            #     plot_boxplot_by_subject(df, sid, feat, subj_dir)

        # print("All analysis data and plots have been saved.")

        #### -----------------------------------------------------
        level = 'segment'
        # 1. Prepare your data
        adj_matrices = np.stack([d['adj'].numpy() for d in all_data_filtered])
        print(adj_matrices.shape)

        labels = np.array([d['class_id'] for d in all_data_filtered])

        # 2. Run the function
        df_all, df_sig = analyze_connectivity_with_fdr(adj_matrices, labels, channel_names)
        df_all.to_csv(f"{data_processed_update}/connectivity_full_stats_fdr_Segment_fold{fold_idx}.csv", index=False)
        df_sig.to_csv(f"{data_processed_update}/connectivity_significant_only_Segment_fold{fold_idx}.csv", index=False)
        save_top_edge_comparison(df_sig, adj_matrices, labels, channel_names, data_processed_update, level)

        # get_top_biomarkers(df_sig, log_path, level, top_n=30)
        # plot_significant_mask(df_all, n_channels, channel_names, class_folder, level)
        #### -----------------------------------------------------

        # Sort by the most significant differences (highest F-stat)
        df_sorted = df_sig.sort_values(by='f_stat', ascending=False)
        
        # Save to CSV
        df_sorted.to_csv(f"{data_processed_update}/significant_edges_fold{fold_idx}.csv", index=False)
        # with open(log_path, "a") as f:
        #     f.write(f"Significant biomarkers exported\n")
        #     # f.write(df_sorted[['edge', 'f_stat', 'p_corrected']].head(20))
        #     f.write(
        #     df_sorted[['edge', 'f_stat', 'p_corrected']]
        #     .head(20)
        #     .to_string(index=False)
        #     + "\n"
        #     )
        #### -----------------------------------------------------
        level = 'subject'
        # 1. Group by subject and average their matrices
        subject_matrices = {}
        subject_labels = {}

        for d in all_data_filtered:
            sid = d['subject_id']
            if sid not in subject_matrices:
                subject_matrices[sid] = []
                subject_labels[sid] = d['class_id']
            subject_matrices[sid].append(d['adj'].numpy())

        # 2. Average the matrices for each subject
        avg_adj_list = [np.mean(matrices, axis=0) for matrices in subject_matrices.values()]
        avg_label_list = list(subject_labels.values())

        # 3. Stack for ANOVA
        adj_matrices_subjects = np.stack(avg_adj_list)
        labels_subjects = np.array(avg_label_list)

        # 4. Run ANOVA
        df_all, df_sig = analyze_connectivity_with_fdr(adj_matrices_subjects, labels_subjects, channel_names)
        df_all.to_csv(f"{data_processed_update}/connectivity_full_stats_fdr_Subject_fold{fold_idx}.csv", index=False)
        df_sig.to_csv(f"{data_processed_update}/connectivity_significant_only_Subject_fold{fold_idx}.csv", index=False)
        save_top_edge_comparison(df_sig, adj_matrices_subjects, labels_subjects, channel_names, data_processed_update, level)

        # get_top_biomarkers(df_sig, log_path, level, top_n=30)
        # plot_significant_mask(df_all, n_channels, channel_names, class_folder, level)
        #### -----------------------------------------------------

