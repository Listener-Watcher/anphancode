import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
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
from graph_utils import load_subjects


def _to_scalar_edge_weight(ew):
    """Convert edge_attr to a scalar float robustly."""
    if ew is None:
        return 0.0
    if hasattr(ew, "detach"):  # torch tensor
        ew = ew.detach().cpu().numpy()
    ew = np.asarray(ew).reshape(-1)
    return float(ew[0]) if ew.size > 0 else 0.0


def bipolar_positions(bipolar_names, pos_dict):
    """
    bipolar_names: list of "A-B" strings
    pos_dict: electrode positions for A and B (e.g., 'F7' -> (x,y))
    """
    new_pos = {}
    for name in bipolar_names:
        A, B = name.split("-")
        if A not in pos_dict or B not in pos_dict:
            raise KeyError(f"Missing position for {A} or {B} in pos_dict.")
        x = (pos_dict[A][0] + pos_dict[B][0]) / 2
        y = (pos_dict[A][1] + pos_dict[B][1]) / 2
        new_pos[name] = (x, y)
    return new_pos


def _normalize_node_names_and_positions(channel_names, electrode_pos_2d):
    """
    Returns:
      node_labels: list[str] length = num_nodes
      pos_by_label: dict[str, (x,y)] for those node_labels
    Supports:
      - channel_names as list[str] (monopolar)
      - channel_names as list[tuple[str,str]] (bipolar)
      - channel_names as list[str] with 'A-B' (bipolar)
    """
    if len(channel_names) == 0:
        return [], {}

    # Case 1: bipolar given as tuples like ("F7","F3")
    if isinstance(channel_names[0], (tuple, list)) and len(channel_names[0]) == 2:
        node_labels = [f"{a}-{b}" for (a, b) in channel_names]
        pos_by_label = bipolar_positions(node_labels, electrode_pos_2d)
        return node_labels, pos_by_label

    # Case 2: strings
    if isinstance(channel_names[0], str):
        # bipolar as "A-B"
        if "-" in channel_names[0]:
            node_labels = list(channel_names)
            pos_by_label = bipolar_positions(node_labels, electrode_pos_2d)
            return node_labels, pos_by_label

        # monopolar as "F3"
        node_labels = list(channel_names)
        # map from label -> pos using electrode_pos_2d
        pos_by_label = {}
        for ch in node_labels:
            if ch not in electrode_pos_2d:
                raise KeyError(f"Missing position for channel '{ch}' in electrode_pos_2d.")
            pos_by_label[ch] = electrode_pos_2d[ch]
        return node_labels, pos_by_label

    raise TypeError("channel_names must be list[str] or list[tuple(str,str)].")


def visualize_eeg_graph_weighted_general(
    data,
    class_names,
    channel_names,
    electrode_positions_2d,
    save_path=None,
    sub_id=None,
    folder_name=None,
    save=False,
    title_prefix="",
    node_size=400,
    font_size=6,
    edge_min_black=0.10,
    dpi=500,
):
    """
    Plots a PyG graph with weighted edges.
    Works for monopolar nodes (19ch) or bipolar nodes (e.g., 23/30 derived channels).

    channel_names:
      - monopolar: ["Fp1","Fp2",...]
      - bipolar (tuple): [("F7","F3"),...]
      - bipolar (str): ["F7-F3",...]
    electrode_positions_2d:
      base electrode positions like your 19-channel dictionary.
    """

    # ---- label setup ----
    label = None
    if hasattr(data, "y") and data.y is not None:
        try:
            label = int(data.y.item())
        except Exception:
            # e.g., multi-element tensor
            label = int(data.y[0].item())

    if label is not None and 0 <= label < len(class_names):
        label_name = class_names[label]
    else:
        label_name = f"Unknown_{label}"

    # ---- normalize names + positions ----
    node_labels, pos_by_label = _normalize_node_names_and_positions(channel_names, electrode_positions_2d)

    # ---- validate node count ----
    num_nodes_expected = len(node_labels)
    num_nodes_in_data = int(data.num_nodes) if hasattr(data, "num_nodes") and data.num_nodes is not None else None
    if num_nodes_in_data is not None and num_nodes_in_data != num_nodes_expected:
        raise ValueError(
            f"Node count mismatch: data.num_nodes={num_nodes_in_data} but len(channel_names)={num_nodes_expected}.\n"
            "Make sure the ordering of channel_names matches node indices in your PyG Data object."
        )

    # ---- build networkx graph ----
    G = to_networkx(data, edge_attrs=["edge_attr"], to_undirected=True)

    # index -> position uses node_labels ordering
    pos = {i: pos_by_label[node_labels[i]] for i in range(num_nodes_expected)}
    label_map = {i: node_labels[i] for i in range(num_nodes_expected)}

    # ---- draw nodes + labels ----
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels=label_map, font_size=font_size)

    # ---- draw edges with normalized weights ----
    if G.number_of_edges() > 0:
        weights = np.array([_to_scalar_edge_weight(attr.get("edge_attr", 0.0)) for _, _, attr in G.edges(data=True)], dtype=float)

        max_w, min_w = float(weights.max()), float(weights.min())
        denom = max_w - min_w
        norm_weights = np.zeros_like(weights) if denom < 1e-12 else (weights - min_w) / denom

        edge_widths = 0.75 + norm_weights * 4.0

        edge_colors = []
        for w in norm_weights:
            if float(w) < edge_min_black:
                edge_colors.append("black")
            else:
                edge_colors.append(plt.cm.Blues(float(w)))

        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors)
    else:
        print("No edges to draw, only visualizing nodes.")

    plt.axis("off")

    # ---- save or show ----
    if save:
        if save_path is None:
            raise ValueError("save_path must be provided when save=True.")

        num_edges = G.number_of_edges()
        title = f"{title_prefix}{sub_id}_{label_name}_{folder_name}".strip("_")
        plt.title(title, fontsize=10)
        plt.text(
            0.5, -0.05,
            f"Number of edges in this graph is {num_edges}",
            ha="center", va="bottom",
            transform=plt.gca().transAxes,
            fontsize=9
        )

        os.makedirs(save_path, exist_ok=True)
        filename = f"{title}.png"
        plt.savefig(os.path.join(save_path, f"{title_prefix}_{folder_name}"), bbox_inches="tight", dpi=dpi)
        plt.close()
        print(f"Saved image: {filename}")
    else:
        plt.show()

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
    # formatted_channels = [f"{pair[0]}-{pair[1]}" for pair in channel_names]

    # bipolar_positions_2d = bipolar_positions(formatted_channels, channel_positions_2d)
    # print(bipolar_positions_2d)
    dataset = 'aheap'
    class_set = 'all3'
    sfreq=500

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)

    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # data_paths, labels, sub_id_list = data_paths[:2] + data_paths[36:38] + data_paths[-2:], labels[:2] + labels[36:38] + labels[-2:], sub_id_list[:2] +sub_id_list[36:38] + sub_id_list[-2:]
    save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/EDA_stats/analysis_Jan2026_goldentest'
    os.makedirs(save_path,exist_ok = True)


    saved_subject_dirs = [
    '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/bi23_rbp_plv_None_25edges',
                            '/mnt/data/anphan/AHEAP_data/goldentest_significant_graph/bi23_rbp_plv_None_35edges',
                            "/mnt/data/anphan/AHEAP_data/segment_significant_graph/bi23_rbp_plv_None",
                            "/mnt/data/anphan/AHEAP_data/significant_graph/bi23_rbp_plv_None"
                            ]
    m=0
    randomstate_value = 15 + m*10
    k = 10
    all_folds = balanced_kfold_split(sub_id_list, labels, randomstate_value, k)

    for saved_subject_dir in saved_subject_dirs:
        last_part = os.path.basename(saved_subject_dir)
        from pathlib import Path
        if last_part == "bi23_rbp_plv_None":
            folder_node = Path(saved_subject_dir).parent.name
            print("folder_node",folder_node)

        for i, test_fold in enumerate(all_folds):
            test_subjects = all_folds[i]
            test_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in test_subjects]
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]
            sub_id = test_subjects[0]
            sub_label = test_labels[0]
            sub_dataset = load_subjects([sub_id], dataset, saved_subject_dir)
            folder_name = f"{last_part}" if last_part != "bi23_rbp_plv_None" else f"{last_part}_{folder_node}"
            visualize_eeg_graph_weighted_general(
                data=sub_dataset[0],
                class_names=class_names,
                channel_names=channel_names,
                electrode_positions_2d=channel_positions_2d,
                save=True,
                save_path=save_path,
                sub_id=sub_id,
                folder_name=folder_name,
                title_prefix=f"Fold{i}",
            )
            if last_part == "bi23_rbp_plv_None":
                break
