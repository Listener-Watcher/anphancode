import torch
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.utils import to_networkx
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from graph_utils import load_subjects
from utils_all import get_class
from data_utils import aheap_get_paths, dryad_get_paths, caueeg_get_paths

def analyze_class_topologies(subject_files, labels):
    """
    subject_files: List of paths to your .pt files
    labels: List of labels corresponding to those files (0, 1, 2)
    """
    class_metrics = {0: [], 1: [], 2: []}
    
    # Define mapping for clarity
    class_map = {0: "Healthy", 1: "Mild", 2: "Severe"}

    for file_path, label in zip(subject_files, labels):
        # Load the bag of graphs (list of Data objects)
        bag = torch.load(file_path, weights_only=False)
        
        # Since all segments in a bag share the SAME edge_index (Backbone),
        # we take the first segment's topology and average the weights.
        first_seg = bag[0]
        num_nodes = first_seg.x.size(0)
        
        # Calculate Average Edge Weights across the whole recording
        all_weights = torch.stack([g.edge_attr for g in bag]) # [segments, edges, 1]
        avg_weights = all_weights.mean(dim=0).squeeze().numpy()
        
        # Convert to NetworkX for Topology Analysis
        edge_index = first_seg.edge_index.numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            # NetworkX metrics usually work best with positive weights (connectivity)
            G.add_edge(u, v, weight=avg_weights[i])

        # --- CALCULATE METRICS ---
        # 1. Global Efficiency: Measures information transfer efficiency
        # We use inverse weight because efficiency relates to the 'shortest path'
        efficiency = nx.global_efficiency(G)
        
        # 2. Average Shortest Path Length (requires a connected graph)
        try:
            path_len = nx.average_shortest_path_length(G, weight=None) # Topological distance
        except nx.NetworkXError:
            path_len = np.nan # If MST logic failed to connect everything

        class_metrics[label].append({
            'Efficiency': efficiency,
            'PathLength': path_len,
            'Density': nx.density(G)
        })

    return class_metrics


def plot_topology_results(metrics, save_path):
    df_list = []
    for cls, results in metrics.items():
        temp_df = pd.DataFrame(results)
        temp_df['Class'] = {0: 'Healthy', 1: 'AD', 2: 'FTD'}[cls]
        df_list.append(temp_df)
    
    df = pd.concat(df_list)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Class', y='Efficiency', data=df, palette='Set2')
    plt.title("Global Efficiency (Communication Speed)")

    plt.subplot(1, 2, 2)
    sns.boxplot(x='Class', y='PathLength', data=df, palette='Set2')
    plt.title("Avg Path Length (Lower is faster)")
    
    plt.tight_layout()
    out_path = os.path.join(save_path, f"topology_results.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    # plt.show()


def analyze_long_range_connectivity(subject_files, labels):
    # Define your electrode groups (Adjust indices based on your specific 19-ch order)
    # Standard 10-20 system indices for 19 channels:
    # channel_names = ['Fp1' 0, 'Fp2' 1, 'F3' 2, 'F4' 3, 'C3' 4, 'C4' 5, 'P3' 6, 'P4' 7, 'O1' 8, 'O2' 9, 'F7' 10, 'F8' 11, 'T3' 12, 'T4' 13, 'T5' 14, 'T6' 15, 'Fz'16, 'Cz'17, 'Pz'18]

    # FRONTAL_NODES = [0, 1, 2, 3, 16]   # e.g., Fp1, Fp2, F3, F4, Fz
    # OCCIPITAL_NODES = [8, 9]        # e.g., O1, O2
    # Assuming 'raw' is your mne object or you have a list of channel names
    # channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", 
    #                  "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"] # Example order
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    FRONTAL_NODES = [i for i, name in enumerate(channel_names) if name.startswith('F')]
    OCCIPITAL_NODES = [i for i, name in enumerate(channel_names) if name.startswith('O')]

    print(f"Correct Frontal Indices: {FRONTAL_NODES}")
    print(f"Correct Occipital Indices: {OCCIPITAL_NODES}")

    results = {0: [], 1: [], 2: []}

    for file_path, label in zip(subject_files, labels):
        bag = torch.load(file_path, weights_only=False)
        # Get the fixed backbone (same for all segments in the bag)
        edge_index = bag[0].edge_index.t().tolist()
        
        # Count unique long-range connections
        long_range_count = 0
        seen_edges = set()
        
        for u, v in edge_index:
            edge = tuple(sorted((int(u), int(v))))
            if edge not in seen_edges:
                # Check if one node is Frontal and the other is Occipital
                # is_frontal = (u in FRONTAL_NODES or v in FRONTAL_NODES)
                # is_occipital = (u in OCCIPITAL_NODES or v in OCCIPITAL_NODES)
                
                # if is_frontal and is_occipital:
                #     long_range_count += 1
                # CORRECTION: Explicitly check for across-region connection
                from_f_to_o = (u in FRONTAL_NODES and v in OCCIPITAL_NODES)
                from_o_to_f = (u in OCCIPITAL_NODES and v in FRONTAL_NODES)
                
                if from_f_to_o or from_o_to_f:
                    long_range_count += 1
                seen_edges.add(edge)
        
        results[label].append(long_range_count)

    return results

# --- Plotting the "Connectivity Gap" ---
def plot_long_range_results(results,save_path):
    data = []
    for cls, counts in results.items():
        name = {0: "Healthy", 1: "AD", 2: "FTD"}[cls]
        for c in counts:
            data.append({"Class": name, "Long-Range Edges": c})
    
    import pandas as pd
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Class", y="Long-Range Edges", data=df, capsize=.1, palette="viridis")
    plt.title("Frontal-to-Occipital Connectivity")
    plt.ylabel("Number of Backbone Edges")
    out_path = os.path.join(save_path, f"Connectivity_Gap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    # plt.show()

if __name__ == "__main__":

    # --- EXECUTION & PLOTTING ---
    root = '/home/anphan/Documents/EEG_Project/AHEAP_data'
    output_path = os.path.join(root, f"combine_minmaxst_graphs_Dec31")
    os.makedirs(output_path,exist_ok = True)
    
    # saved_data_dir = os.path.join(root, f"combine_minmaxst_graphs_Dec31/rbp/rbp_coherence_alpha_mst_commonfix")
    saved_data_dir = '/mnt/data/anphan/AHEAP_data/ALL_fixed_graphs_Jan03/rbp/rbp_coherence_alpha_strong_fixed'
    
    class_set = 'all3'
    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    
    file_paths = []
    label_list = []
    for sid, label in zip(sub_id_list, labels):
        save_path = os.path.join(saved_data_dir, f"{sid}_task-eyesclosed_eeg.pt")

        if not os.path.exists(save_path):
            # print(f"File not found: {save_path}")
            continue
        print(sid, label)
        file_paths.append(save_path)
        label_list.append(label)
    print("file_paths:", len(file_paths))
    print("label_list:", len(label_list))
    metrics = analyze_class_topologies(file_paths, label_list)
    print("metrics:", metrics)
    results = analyze_long_range_connectivity(file_paths, label_list)
    print("results", results)
    
    plot_topology_results(metrics, output_path)
    print("plot_topology_results!")
    plot_long_range_results(results,output_path)
    print("plot_long_range_results!")