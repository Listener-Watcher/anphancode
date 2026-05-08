import argparse
import os
import re
from datetime import datetime
import networkx as nx
import torch

from viz_utils import ensure_dir, save_log
from viz_graphs import *
from stats import *
from viz_plots import *
import sys
# Add project root so we can import graph_utils
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from plot import visualize_eeg_graph_weighted 
from graph_utils import load_subjects
from utils_all import get_class
from data_utils import aheap_get_paths, dryad_get_paths, caueeg_get_paths

def bipolar_positions(bipolar_names, pos_dict):
    new_pos = {}
    for name in bipolar_names:
        A, B = name.split("-")
        x = (pos_dict[A][0] + pos_dict[B][0]) / 2
        y = (pos_dict[A][1] + pos_dict[B][1]) / 2
        new_pos[name] = (x, y)
    return new_pos

import os
import torch
from torch_geometric.data import Batch

def load_subject_data(sid, saved_data_dir, dataset_name):
    # Construct path based on your saving convention
    if dataset_name == 'aheap':
        file_path = os.path.join(saved_data_dir, f"{sid}_task-eyesclosed_eeg.pt")
    else:
        file_path = os.path.join(saved_data_dir, f"{sid}.pt")
    
    if not os.path.exists(file_path):
        return None

    # 1. Read the list of Data objects
    # This matches the 'data_list' output from your build function
    data_list = torch.load(file_path, weights_only=False)

    # 2. Recommended: Pre-convert to a Batch object
    # This makes moving to GPU much faster during training
    bag_as_batch = Batch.from_data_list(data_list)
    
    return bag_as_batch


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    # parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    # args = parser.parse_args()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # dataset = args.dataset.lower()
    class_set = 'all3'
    dataset = 'aheap'
    
    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/EDA_stats/ALL_fixed_graphs_Jan03'
    # save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/EDA_stats/bipolar_update/1128_094935'
    os.makedirs(save_path,exist_ok = True)
    # root = "/mnt/data/anphan/AHEAP_data/significant_graph/mono_rbp_corr_None"
    root = ""
    # root = "/mnt/data/anphan/AHEAP_data/fixed_graph_update_full"
    # root = '/mnt/data/anphan/AHEAP_data/bipolar_23channels'
    # root = '/mnt/data/anphan/AHEAP_data/ALL_fixed_graphs_Jan03'
    # root = '/mnt/data/anphan/AHEAP_data/region_graph/rbp'
    # root = '/mnt/data/anphan/AHEAP_data/duration4_overlap2/rbp'

    # root = "/mnt/data/anphan/AHEAP_data/bipolar_update"
    # root = "/mnt/data/anphan/AHEAP_data"

    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
    # bipolar_pairs = [
    #     ("F8", "F4"), ("F7", "F3"),
    #     ("F4", "C4"), ("F3", "C3"),
    #     ("F4", "Fz"), ("F3", "Fz"),
    #     ("Fz", "Cz"),
    #     ("C4", "P4"), ("C3", "P3"),
    #     ("C4", "T4"), ("C3", "T3"),
    #     ("Cz", "Pz"),
    #     ("T4", "T6"), ("T3", "T5"),
    #     ("P4", "O2"), ("P3", "O1"),
    #     ("T6", "O2"), ("T5", "O1"),
    #     ("O1", "O2")
    # ]


    bipolar_pairs = [
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
    bipolar_names = []

    name_to_idx = {ch: i for i, ch in enumerate(channel_names)}

    for A, B in bipolar_pairs:
        if A not in name_to_idx or B not in name_to_idx:
            continue
        bipolar_names.append(f"{A}-{B}")

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
    bipolar_pos = bipolar_positions(bipolar_names, channel_positions_2d)

    print("============================================")
    print("🔍 Scanning folders for valid .pt datasets...")
    print("============================================")

    # -----------------------------------------------
    # Scan root folder → find folders with 88 .pt files
    # -----------------------------------------------
    valid_folders = []

    for dirpath, dirnames, filenames in os.walk(root):

        # Count .pt files in this specific directory
        pt_files = [f for f in filenames if f.endswith(".pt")]

        # Only keep folders with exactly 88 pt files
        if len(pt_files) == 88:
            print(f"✅ Found valid folder: {dirpath}")
            valid_folders.append(dirpath)

    print("\n==========================================")
    print(f"📦 Total valid folders found: {len(valid_folders)}")
    print("==========================================\n")
    folders = [f for f in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, f))]
    processed_folders = [os.path.basename(f) for f in folders]

    # --------------------------------------------------------
    # Now run EDA for each valid folder
    # --------------------------------------------------------
    # check_path = ['rbp_fixed', 'rbphjorth_fixedgraph_None', 'rbphjorth_fixedgraph_alpha', 'rbphjorth_fixedgraph_delta', 'rbphjorth_fixedgraph_theta',
                  # 'rbp_fixedgraph_None', 'rbp_fixedgraph_delta', 'rbp_fixedgraph_alpha', 'rbp_fixedgraph_theta']
    for saved_subject_dir in valid_folders:
        last_part = os.path.basename(saved_subject_dir)
        # if last_part not in check_path:
        #     continue
        # if last_part in processed_folders:
        #     print("Folder existed! Skip!")
            # continue
        # Parse folder naming convention
        pattern = r"^([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+|[0-9]*\.?[0-9]+)$"
        match = re.match(pattern, last_part)

        if not match:
            print(f"⚠️ Warning: Folder name does not follow pattern, skipping → {last_part}")
            # continue
            node_features = last_part
        else:
            node_features, weight_method, band, filter_method, filter_edge = match.groups()

        # Create output directory
        output_dir = os.path.join(save_path, last_part)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🚀 Running EDA on: {last_part}")
        print(f"   → Output will be saved to: {output_dir}\n")

        folder_name = os.path.basename(saved_subject_dir)

        ensure_dir(output_dir)
        log_path = os.path.join(output_dir, "eda_log.txt")

        save_log("=== Running EDA ===", log_path)
        num_nodes = len(bipolar_names)
        # num_nodes = len(channel_names)

        # ------------------------------------------------
        # Load graphs per subject
        # ------------------------------------------------
        graphs_by_subject = []
        for sid in sub_id_list:
            # 1. Get the list of Data objects for this subject
            graphs = load_subjects([sid], dataset, saved_subject_dir)

            print(f"DEBUG: Loaded {len(graphs)} graphs for subject {sid}")

            # 2. Iterate through the list DIRECTLY
            for i, g in enumerate(graphs): 
                # Now 'g' is correctly a PyG Data object
                print(f"Graph {i}: x={g.x.shape}, edge_index={g.edge_index.shape}")
                
                visualize_eeg_graph_weighted(
                    g,
                    class_names,
                    channel_names, 
                    channel_positions_2d, 
                    output_dir,
                    sid,
                    folder_name = f"{sid}_significant_edges",
                    save = True
                )
                # print(g)
                # print(f"Graph {i}: x={g.x.shape}, edge_index={g.edge_index.shape}, edge_attr={g.edge_attr.shape}")
                # visualize_eeg_graph_weighted(
                #     g[0],
                #     class_names,
                #     channel_names, 
                #     channel_positions_2d, 
                #     output_dir,
                #     sid,
                #     folder_name = f"{sid}_mono_rbp_coherence_None_significant_edges",
                #     save = True)
            # print(f"{channel_names[int(u)]} ↔ {channel_names[int(v)]}")


            graphs_by_subject.append(graphs)


        save_log(f"Loaded {len(graphs_by_subject)} subjects.", log_path)



        # # ------------------------------------------------
        # # 1. Subject-level stats
        # # ------------------------------------------------
        # subject_stats = [compute_subject_edge_stats(g) for g in graphs_by_subject]
        # save_log("Computed subject-level edge stats.", log_path)

        # # ------------------------------------------------
        # # 2. Class-level stats
        # # ------------------------------------------------
        # class_stats = aggregate_class_stats(subject_stats, labels)
        # save_log("Aggregated class-level stats.", log_path)

        # # ------------------------------------------------
        # # 3. Discriminability ranking
        # # ------------------------------------------------
        # ranked = rank_edges_between_classes(class_stats)
        # save_log("Computed discriminability ranking.", log_path)

        # # ------------------------------------------------
        # # 3b. Combined discriminability plot (all class pairs)
        # # ------------------------------------------------
        
        # pairs = list(ranked.items())
        # n_pairs = len(pairs)
        # top_n = 25  # choose number of edges to plot

        # plt.figure(figsize=(6 * n_pairs, 5))

        # for idx, ((c1, c2), edges) in enumerate(pairs):
        #     ax = plt.subplot(1, n_pairs, idx + 1)

        #     # top_n edges
        #     top_edges = edges[:top_n]   # each item = (edge, score)

        #     # extract scores (2-tuple format)
        #     scores = np.array([score for (_, score) in top_edges])
        #     scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        #     # Build graph
        #     G = nx.Graph()
        #     # for node_idx, ch in enumerate(bipolar_names):
        #     for node_idx, ch in enumerate(channel_names):

        #         # G.add_node(node_idx, pos=bipolar_pos[ch])
        #         G.add_node(node_idx, pos=channel_positions_2d[ch])

        #     for ((u, v), score), snorm in zip(top_edges, scores_norm):
        #         G.add_edge(u, v, weight=snorm, color=plt.cm.Reds(snorm))

        #     pos = nx.get_node_attributes(G, "pos")

        #     nx.draw_networkx_nodes(G, pos,
        #         node_size=250,
        #         node_color="lightblue",
        #         edgecolors="black",
        #         ax=ax
        #     )
        #     nx.draw_networkx_labels(G, pos,
        #         # {i: ch for i, ch in enumerate(bipolar_names)},
        #         {i: ch for i, ch in enumerate(channel_names)},
        #         font_size=7,
        #         ax=ax
        #     )

        #     nx.draw_networkx_edges(
        #         G, pos,
        #         width=[1 + 4 * d["weight"] for (_,_,d) in G.edges(data=True)],
        #         edge_color=[d["color"] for (_,_,d) in G.edges(data=True)],
        #         ax=ax
        #     )

        #     ax.set_title(f"{class_names[c1]} vs {class_names[c2]}", fontsize=12)
        #     ax.axis("off")

        # plt.suptitle("Top Discriminative Edges Across Class Pairs", fontsize=15)
        # plt.tight_layout(rect=[0, 0, 1, 0.92])

        # save_file = os.path.join(output_dir, "discriminative_edges_all_pairs.png")
        # plt.savefig(save_file, dpi=300, bbox_inches="tight")
        # plt.close()

        # save_log(f"Saved combined discriminability plot: {save_file}", log_path)


        # # ------------------------------------------------
        # # 4. Class-level mean adjacency (multi-plot heatmap)
        # # ------------------------------------------------
        # class_mean_adj = {}  # collect adjacency matrices per class

        # for sid, sgraphs, label in zip(sub_id_list, graphs_by_subject, labels):
        #     agg = aggregate_subject_graphs(sgraphs, "mean")  # subject-level mean
        #     if label not in class_mean_adj:
        #         class_mean_adj[label] = []
        #     class_mean_adj[label].append(agg)

        # # Convert lists to single mean adjacency per class
        # class_mean_adj_final = {}
        # for c in class_mean_adj:
        #     class_mean_adj_final[c] = sum(class_mean_adj[c]) / len(class_mean_adj[c])

        # # Save one combined heatmap figure
        # heatmap_path = os.path.join(output_dir, "class_mean_connectivity.png")
        # plot_class_mean_connectivity_multiplot(
        #     class_mean_adj_final,
        #     class_names,
        #     channel_names,
        #     # bipolar_names,
        #     folder_name,
        #     heatmap_path
        # )
        # save_log("Saved class mean connectivity heatmaps.", log_path)

        # # ------------------------------------------------
        # # 5. Degree distribution
        # # ------------------------------------------------
        # all_graphs = [g for subj in graphs_by_subject for g in subj]
        # all_labels = [lab for lab, subj in zip(labels, graphs_by_subject) for _ in subj]

        # edge_freq_by_class = {c: {} for c in set(labels)}

        # for graphs, label in zip(graphs_by_subject, labels):
        #     for g in graphs:
        #         edges = g.edge_index.t().cpu().numpy()
        #         for (u, v) in edges:
        #             if u > v:
        #                 u, v = v, u  # make undirected
        #             e = (u, v)
        #             edge_freq_by_class[label][e] = edge_freq_by_class[label].get(e, 0) + 1

        # plot_topk_edge_frequency_barplot(
        #     edge_freq_by_class,
        #     channel_names,
        #     # bipolar_names,
        #     class_names,
        #     output_dir,
        #     top_k=30
        # )


        # nodefeat_dir = os.path.join(output_dir, "node_features")
        # os.makedirs(nodefeat_dir, exist_ok=True)

        # plot_node_features_by_label(
        #     all_graphs,
        #     class_names,
        #     folder_name,
        #     nodefeat_dir,
        #     channel_names
        #     # bipolar_names
        # )

        # save_log("Saved node feature distribution plots.", log_path)




#-------------------------------------------------------------------------------
    # elif dataset == 'dryad':
    #     data_folder = '/mnt/data/anphan/dryad_data/preprocessed_data'
    #     csv_path = '/mnt/data/anphan/dryad_data/preprocessed_data/preprocessed_summary.csv'
    #     data_paths, labels, sub_id_list = dryad_get_paths(csv_path, data_folder)
    #     save_path = '/home/anphan/Documents/EEG_Project/Dryad_data/EDA_stats'
    #     os.makedirs(save_path,exist_ok = True)
    #     dir_path = "/mnt/data/anphan/dryad_data/graph_saved_data/rbphjorth/rbphjorth_dirs.txt"

    # elif dataset == 'caueeg':
    #     json_path = '/home/anphan/Downloads/caueeg-dataset/annotation.json'
    #     data_folder = '/home/anphan/Downloads/caueeg-dataset/processed_data'
    #     data_paths, labels, sub_id_list = caueeg_get_paths(json_path, data_folder)
    #     save_path = '/home/anphan/Documents/EEG_Project/CAUEEG/EDA'
    #     os.makedirs(save_path,exist_ok = True)
    #     save_dir = '/home/anphan/Documents/EEG_Project/AHEAP_data/aggregated_graph'
    #     os.makedirs(save_dir,exist_ok = True)
    #     dir_path = '/home/anphan/Documents/EEG_Project/CAUEEG/graph_data_all/rbphjorth_coherence_None/rbphjorth_coherence_None_dirs.txt'
    #     epochs = 500
    #     iterate = 3

    # else:
    #     print("Wrong dataset! Stop!")

    # with open(dir_path, "r") as f:
    #     saved_subject_dirs = f.read().splitlines()

    # for saved_subject_dir in saved_subject_dirs:
    #     last_part = os.path.basename(saved_subject_dir)
    #     pattern = r"^([a-zA-Z]+)_([a-zA-Z]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+|[0-9]*\.?[0-9]+)$"
    #     match = re.match(pattern, last_part)
    #     # if match:
    #     node_features, weight_method, band, filter_method, filter_edge = match.groups()
    #     output_dir = os.path.join(save_path, last_part)
    #     os.makedirs(output_dir,exist_ok = True)
    #     run_eda(dataset, saved_subject_dir, output_dir, sub_id_list, labels, channel_names)

# Root folder containing all generated graph directories
        # degree_fig = os.path.join(output_dir, "degree_distribution.png")
        # plot_degree_distribution(
        #     all_graphs,
        #     all_labels,
        #     class_names,
        #     folder_name,
        #     degree_fig
        # )
        # save_log("Saved degree distribution plot.", log_path)
        # ------------------------------------------------
        # NEW: Edge frequency distribution
        # ------------------------------------------------
        # freq_by_class = compute_edge_frequency(all_graphs, all_labels, num_nodes)

        # heatmap_path = os.path.join(output_dir, "edge_frequency_heatmaps.png")
        # plot_edge_frequency_heatmaps(freq_by_class, class_names, channel_names, folder_name, heatmap_path)
        # save_log("Saved edge frequency heatmaps.", log_path)

        # topedge_path = os.path.join(output_dir, "top_edges_frequency.png")
        # plot_top_edges(freq_by_class, class_names, channel_names, folder_name, topedge_path)
        # save_log("Saved top-edges frequency plot.", log_path)
        # plot_edge_frequency_barplot(
        #     edge_freq_by_class,
        #     channel_names,
        #     class_names,
        #     output_dir
        # )
        # ------------------------------------------------
        # # 6. Edge count distribution
        # # ------------------------------------------------
        # edgecount_fig = os.path.join(output_dir, "edge_count_distribution.png")
        # plot_edge_count_distribution(
        #     all_graphs,
        #     all_labels,
        #     class_names,
        #     folder_name,
        #     edgecount_fig
        # )
        # save_log("Saved edge count distribution plot.", log_path)

        # ------------------------------------------------
        # 7. Edge stability scatter plot (per class)
        # ------------------------------------------------
        # for c in class_stats:
        #     values = [(v["freq"], v["mu"], v["sd"]) for v in class_stats[c].values()]

        #     stability_fig = os.path.join(output_dir, f"stability_{class_names[c]}.png")
        #     plot_edge_stability(
        #         values,
        #         class_names[c],
        #         folder_name,
        #         stability_fig
        #     )
        #     save_log(f"Saved stability plot for {class_names[c]}.", log_path)

        # ------------------------------------------------
        # 8. Multi-Class Node Feature Plotting
        # ------------------------------------------------
        # save_log("=== EDA Completed ===", log_path)
        # ------------------------------------------------
        # Node feature boxplots
        # ------------------------------------------------
        # nodefeat_box_dir = os.path.join(output_dir, "node_features_boxplot")
        # os.makedirs(nodefeat_box_dir, exist_ok=True)

        # plot_node_features_boxplot(
        #     all_graphs,
        #     class_names,
        #     folder_name,
        #     nodefeat_box_dir,
        #     channel_names
        # )

        # save_log("Saved node feature boxplot plots.", log_path)



        # break