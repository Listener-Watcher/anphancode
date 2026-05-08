import argparse
import os
import re
import sys
from datetime import datetime
# Add project root so we can import graph_utils
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from lib import *
from graph_utils import load_subjects
from utils_all import get_class
from data_utils import aheap_get_paths, dryad_get_paths, caueeg_get_paths

def edge_presence_frequency(graphs, num_nodes=None, undirected=True):
    """
    Count edge presence PER GRAPH:
      count_matrix[i,j] = number of graphs where edge (i,j) appears at least once.
    """
    if len(graphs) == 0:
        raise ValueError("No graphs provided.")

    if num_nodes is None:
        # Try to infer from first graph
        num_nodes = int(graphs[0].num_nodes) if hasattr(graphs[0], "num_nodes") else int(graphs[0].x.size(0))

    count_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for g in graphs:
        ei = g.edge_index
        if ei is None or ei.numel() == 0:
            continue

        # Collect unique edges in THIS graph (so multi-edges don't inflate)
        edges = set()
        src = ei[0].detach().cpu().numpy()
        dst = ei[1].detach().cpu().numpy()

        for u, v in zip(src, dst):
            if undirected:
                a, b = (u, v) if u <= v else (v, u)
            else:
                a, b = u, v
            if a == b:
                continue  # skip self-loops if you want
            edges.add((int(a), int(b)))

        for a, b in edges:
            count_matrix[a, b] += 1
            if undirected:
                count_matrix[b, a] += 1  # mirror for display

    return count_matrix


def plot_subject_edge_frequency(
    sid,
    graphs,
    channel_names=None,
    save_path=None,
    top_k=0
):
    # subject label (assumes consistent label across graphs)
    label = None
    if hasattr(graphs[0], "y") and graphs[0].y is not None:
        try:
            label = int(graphs[0].y.item())
        except Exception:
            label = graphs[0].y

    num_graphs = len(graphs)
    num_nodes = int(graphs[0].num_nodes) if hasattr(graphs[0], "num_nodes") else int(graphs[0].x.size(0))

    count_mat = edge_presence_frequency(graphs, num_nodes=num_nodes, undirected=True)
    freq_mat = count_mat / max(num_graphs, 1)  # 0..1

    # ---- Heatmap ----
    plt.figure(figsize=(8, 7))
    plt.imshow(freq_mat, aspect="equal")
    plt.colorbar(label="Edge presence (fraction of graphs)")
    title = f"{sid} | label={label} | graphs={num_graphs}"
    plt.title(title)

    if channel_names is not None and len(channel_names) == num_nodes:
        plt.xticks(range(num_nodes), channel_names, rotation=90)
        plt.yticks(range(num_nodes), channel_names)
    else:
        plt.xticks(range(num_nodes))
        plt.yticks(range(num_nodes))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{sid}_heatmap.png'), dpi=200, bbox_inches="tight")
    plt.close()

    # ---- Optional: Top-K edges bar chart ----
    if top_k and top_k > 0:
        # Only take upper triangle to avoid duplicates (undirected)
        pairs = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                c = count_mat[i, j]
                if c > 0:
                    pairs.append((i, j, c, c / num_graphs))

        pairs.sort(key=lambda x: x[2], reverse=True)
        pairs = pairs[:top_k]

        labels = []
        values = []
        for i, j, c, f in pairs:
            if channel_names is not None and len(channel_names) == num_nodes:
                labels.append(f"{channel_names[i]}–{channel_names[j]}")
            else:
                labels.append(f"{i}-{j}")
            values.append(c)

        plt.figure(figsize=(10, 4))
        plt.bar(range(len(values)), values)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right")
        plt.ylabel("# graphs containing edge")
        plt.title(f"Top {top_k} edges | {sid} | label={label} | graphs={num_graphs} ")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(os.path.join(save_path, f'{sid}_topkedges.png'), dpi=200, bbox_inches="tight")
        plt.close()

    return count_mat, freq_mat



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    # parser.add_argument("--dataset", type=str, required=True, help="Name of dataset")
    parser.add_argument("--dir", type=str, required=False, help="Path to the input directory")
    # parser.add_argument("--model_name", type=str, required=False, help="Name of the model to use")
    # parser.add_argument("--class_set", type=str, required=False, help="Name of the model to use")
    args = parser.parse_args()

    # dataset = args.dataset.lower()
    # model_name = args.model_name
    # class_set = args.class_set
    saved_subject_dir = args.dir
    last_part = os.path.basename(saved_subject_dir)

    # saved_subject_dirs
    dataset = 'aheap'
    class_set = 'all3'
    # model_name = 'GAT'

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    device = torch.device("cuda")
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")


    # if dataset == 'aheap':
    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    save_path = f'/home/anphan/Documents/EEG_Project/AHEAP_data/EDA_stats/subject_edges/{last_part}'
    os.makedirs(save_path,exist_ok = True)
    for sid in sub_id_list:
        # file_name = os.path.join(f'{sid}.png')
        graphs = load_subjects([sid], dataset, saved_subject_dir)
        g = graphs[0]
        # print(f"Graph: x={g.x.shape}, edge_index={g.edge_index.shape}, edge_attr={g.edge_attr.shape}")
        node_dim = g.x.shape[0]
        # print(node_dim)

        if node_dim == 19:
            channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
        elif node_dim == 23:
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
        elif node_dim == 30:
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

        count_mat, freq_mat = plot_subject_edge_frequency(
                                sid, graphs,
                                channel_names=channel_names,  # optional
                                save_path=save_path,
                                top_k=100
                            )
# ---------------- Example usage ----------------
# sid = "sub-001"
