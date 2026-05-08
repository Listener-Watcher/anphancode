import numpy as np
import matplotlib.pyplot as plt
from viz_utils import ensure_dir, save_log
from viz_utils import build_adjacency
import os
from collections import Counter

def plot_heatmap(matrix, channel_names, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(channel_names)), channel_names, rotation=45)
    plt.yticks(range(len(channel_names)), channel_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_edge_frequency_heatmaps(freq_by_class, class_names, channel_names, folder_name, save_path):
    classes = sorted(freq_by_class.keys())
    n = len(classes)

    plt.figure(figsize=(6 * n, 6))

    for idx, c in enumerate(classes):
        ax = plt.subplot(1, n, idx + 1)
        mat = freq_by_class[c]

        im = ax.imshow(mat, cmap="inferno", vmin=0, vmax=1)
        ax.set_title(f"{class_names[c]} (Edge Frequency)")
        ax.set_xticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45)
        ax.set_yticks(range(len(channel_names)))
        ax.set_yticklabels(channel_names)

        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(f"{folder_name} — Edge Frequency Heatmaps (0–1 scale)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()

def plot_class_mean_connectivity_multiplot(class_mean, class_names, channel_names, folder_name, save_path):
    """
    class_mean : dict[class_id] -> adjacency matrix (numpy array)
    class_names : list[str]
    """

    classes = sorted(class_mean.keys())
    n_classes = len(classes)

    plt.figure(figsize=(6 * n_classes, 6))  # dynamic width

    for i, c in enumerate(classes):
        mat = class_mean[c]

        ax = plt.subplot(1, n_classes, i + 1)
        im = ax.imshow(mat, cmap="viridis")

        ax.set_title(class_names[c], fontsize=12)
        ax.set_xticks(range(len(channel_names)))
        ax.set_yticks(range(len(channel_names)))
        ax.set_xticklabels(channel_names, rotation=45, ha='right')
        ax.set_yticklabels(channel_names)

        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.suptitle(f"{folder_name} — Edge Connectivity Heatmaps", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for supertitle
    plt.savefig(save_path)
    plt.close()

# def plot_degree_distribution(graphs, class_labels, channel_names, save_path):
#     from collections import Counter

#     degrees_by_class = {c: [] for c in set(class_labels)}

#     # Compute degrees
#     for g, c in zip(graphs, class_labels):
#         adj = build_adjacency(g)
#         deg = np.sum(adj != 0, axis=1)  # degree per node
#         degrees_by_class[c].extend(deg.tolist())

#     # Degree value range
#     all_degrees = sorted(set([d for degs in degrees_by_class.values() for d in degs]))

#     # Count occurrences per class
#     counts = {c: Counter(degrees_by_class[c]) for c in degrees_by_class}

#     # Plot side-by-side
#     x = np.arange(len(all_degrees))
#     width = 0.25

#     plt.figure(figsize=(10, 6))

#     for i, c in enumerate(sorted(degrees_by_class.keys())):
#         class_counts = [counts[c][d] for d in all_degrees]
#         plt.bar(x + i*width, class_counts, width=width, label=f"Class {c}")

#     plt.xticks(x + width, all_degrees)
#     plt.xlabel("Degree")
#     plt.ylabel("Count")
#     plt.title("Node Degree Distribution (Side-by-side)")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()
def plot_degree_distribution(graphs, class_labels, class_names, folder_name, save_path):

    degrees_by_class = {c: [] for c in set(class_labels)}

    # Compute degrees
    for g, c in zip(graphs, class_labels):
        adj = build_adjacency(g)
        deg = np.sum(adj != 0, axis=1)
        degrees_by_class[c].extend(deg.tolist())

    all_degrees = sorted(set([d for degs in degrees_by_class.values() for d in degs]))
    counts = {c: Counter(degrees_by_class[c]) for c in degrees_by_class}

    x = np.arange(len(all_degrees))
    width = 0.22

    plt.figure(figsize=(10, 6))
    for idx, c in enumerate(sorted(degrees_by_class.keys())):
        class_counts = [counts[c][d] for d in all_degrees]
        plt.bar(x + idx*width, class_counts, width=width,
                label=class_names[c])  # <-- Correct class name

    plt.xticks(x + width, all_degrees)
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Node Degree Distribution")
    plt.suptitle(folder_name, fontsize=13, y=1.02)   # <-- Folder-based super title
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_top_edges(freq_by_class, class_names, channel_names, folder_name, save_path, top_k=20):
    plt.figure(figsize=(10, 6))

    for c in freq_by_class:
        mat = freq_by_class[c]
        tri = np.triu(mat, 1)

        # flatten
        flat = tri.flatten()
        idx = flat.argsort()[-top_k:]

        # Map back to (i, j)
        pairs = [np.unravel_index(i, tri.shape) for i in idx]
        values = [tri[i, j] for (i, j) in pairs]
        labels = [f"{channel_names[i]}-{channel_names[j]}" for (i, j) in pairs]

        plt.plot(labels, values, label=class_names[c], marker='o')

    plt.xticks(rotation=90)
    plt.ylabel("Frequency")
    # plt.title("Top Edges by Frequency")
    plt.suptitle(f"{folder_name} - Top Edges by Frequency", fontsize=14, y=1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# def plot_edge_count_distribution(graphs, class_labels, save_path):
#     edges_by_class = {c: [] for c in set(class_labels)}

#     for g, c in zip(graphs, class_labels):
#         edges_by_class[c].append(g.edge_index.shape[1])

#     plt.figure(figsize=(7, 5))
#     for c in edges_by_class:
#         plt.hist(edges_by_class[c], bins=20, alpha=0.5, label=f"Class {c}")
#     plt.legend()
#     plt.xlabel("Edge count per graph")
#     plt.ylabel("Frequency")
#     plt.title("Graph Edge Count Distribution")
#     plt.savefig(save_path)
#     plt.close()

def plot_edge_count_distribution(graphs, class_labels, class_names, folder_name, save_path):

    edges_by_class = {c: [] for c in set(class_labels)}

    for g, c in zip(graphs, class_labels):
        edges_by_class[c].append(g.edge_index.shape[1])

    plt.figure(figsize=(10, 6))
    for c in sorted(edges_by_class.keys()):
        plt.hist(edges_by_class[c], bins=20, alpha=0.5, label=class_names[c])

    plt.xlabel("Number of edges")
    plt.ylabel("Count")
    plt.title("Edge Count Distribution")
    plt.suptitle(folder_name, fontsize=13, y=1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# def plot_edge_stability(scatter_data, save_path, title):
#     """scatter_data: list of (freq, mu, sd) for a class"""
#     freq, mu, sd = zip(*scatter_data)

#     plt.figure(figsize=(6, 5))
#     plt.scatter(freq, mu, c=sd, cmap="coolwarm", s=40)
#     plt.colorbar(label="SD")
#     plt.xlabel("Frequency")
#     plt.ylabel("Mean Weight")
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

def plot_edge_stability(values, class_name, folder_name, save_path):
    freq, mu, sd = zip(*values)

    plt.figure(figsize=(7, 6))
    plt.scatter(freq, mu, c=sd, cmap="coolwarm", s=40)
    plt.colorbar(label="SD")
    
    plt.xlabel("Frequency")
    plt.ylabel("Mean Weight")
    plt.title(f"Edge Stability — {class_name}")
    plt.suptitle(folder_name, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_node_features_by_label(dataset, class_names, folder_name, output_dir, channel_names):
    """
    dataset: list of PyG Data objects (all graphs)
    class_names: list, e.g., ["Healthy", "AD", "FTD"]
    folder_name: for supertitle
    channel_names: list of 19 EEG channels
    """

    num_nodes, num_features = dataset[0].x.shape
    labels_set = sorted(list(set([int(data.y.item()) for data in dataset])))

    # =============== LOOP THROUGH EACH FEATURE DIMENSION ===============
    for feat_idx in range(num_features):

        # values_by_label[label][node] = list of values
        values_by_label = {label: [[] for _ in range(num_nodes)] for label in labels_set}

        for data in dataset:
            label = int(data.y.item())
            feature_values = data.x[:, feat_idx].cpu().numpy()  # (19,)
            for node_idx in range(num_nodes):
                values_by_label[label][node_idx].append(feature_values[node_idx])

        # =============== Compute per-node mean per class ===============
        means_by_label = {
            label: [np.mean(values_by_label[label][node]) for node in range(num_nodes)]
            for label in labels_set
        }

        # =============== Plotting ===============
        plt.figure(figsize=(12, 5))

        positions = np.arange(num_nodes)
        width = 0.8 / len(labels_set)  # dynamic spacing for any number of classes

        for i, label in enumerate(labels_set):
            plt.bar(
                positions + i * width,
                means_by_label[label],
                width=width,
                alpha=0.8,
                label=class_names[label]
            )

        # X-axis labels
        plt.xticks(
            positions + width * (len(labels_set) - 1) / 2,
            channel_names,
            rotation=45,
            ha='right'
        )

        plt.ylabel(f"Feature {feat_idx} Value")
        plt.title(f"Node Feature {feat_idx} (Mean Across Samples)")
        plt.suptitle(folder_name, fontsize=13, y=1.02)
        plt.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Save
        out_path = os.path.join(output_dir, f"node_feature_{feat_idx}.png")
        plt.savefig(out_path)
        plt.close()

def plot_topk_edge_frequency_barplot(edge_freq_by_class, channel_names, class_names, save_path, top_k=30):
    """
    edge_freq_by_class: dict[class_id][(u,v)] = count
    """

    class_ids = sorted(edge_freq_by_class.keys())
    num_classes = len(class_ids)

    plt.figure(figsize=(16, 12))  # good for 3 subplots stacked

    for idx, c in enumerate(class_ids):

        ax = plt.subplot(num_classes, 1, idx + 1)

        edges = list(edge_freq_by_class[c].keys())
        freqs = [edge_freq_by_class[c][e] for e in edges]

        # Convert to names
        edge_names = [f"{channel_names[u]}–{channel_names[v]}" for (u, v) in edges]

        # Sort descending
        sorted_idx = np.argsort(freqs)[::-1]

        # Select top-K
        sorted_idx = sorted_idx[:top_k]
        edge_names_top = [edge_names[i] for i in sorted_idx]
        freqs_top = [freqs[i] for i in sorted_idx]

        ax.bar(edge_names_top, freqs_top, color="steelblue", alpha=0.85)
        ax.set_title(f"{class_names[c]} — Top {top_k} Edge Frequencies", fontsize=14)
        ax.set_ylabel("Frequency (raw count)")
        ax.set_xticks(range(len(edge_names_top)))
        ax.set_xticklabels(edge_names_top, rotation=45, ha='right', fontsize=8)

    plt.suptitle(f"Top {top_k} Most Frequent Edges Across Classes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    out_path = os.path.join(save_path, f"top{top_k}_edge_frequencies.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved: {out_path}")

def plot_edge_frequency_barplot(edge_freq_by_class, channel_names, class_names, save_path):
    """
    edge_freq_by_class: dict[class_id][(u,v)] = frequency_count
    """

    num_classes = len(edge_freq_by_class)
    class_ids = sorted(edge_freq_by_class.keys())

    plt.figure(figsize=(10 * num_classes, 5))

    for idx, c in enumerate(class_ids):
        ax = plt.subplot(1, num_classes, idx + 1)

        edges = list(edge_freq_by_class[c].keys())
        freqs = [edge_freq_by_class[c][e] for e in edges]

        # Convert (u,v) to "Ch1–Ch2"
        edge_names = [f"{channel_names[u]}–{channel_names[v]}" for (u, v) in edges]

        # Sort edges by frequency (descending)
        sorted_idx = np.argsort(freqs)[::-1]
        edge_names = [edge_names[i] for i in sorted_idx]
        freqs = [freqs[i] for i in sorted_idx]

        ax.bar(edge_names, freqs, color="steelblue", alpha=0.8)
        ax.set_title(f"{class_names[c]} — Edge Frequency", fontsize=14)
        ax.set_ylabel("Frequency (raw count)")
        ax.set_xticks(range(len(edge_names)))
        ax.set_xticklabels(edge_names, rotation=90, fontsize=7)

    plt.suptitle("Edge Frequency (Raw Count) Across Classes", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    plt.savefig(os.path.join(save_path, "edge_frequency_barplot.png"), dpi=300)
    plt.close()





# def plot_node_features_boxplot(dataset, class_names, folder_name, output_dir, channel_names):
#     """
#     dataset: list of PyG Data objects with x = (num_nodes, num_features)
#     class_names: list of class names (len = num_classes)
#     folder_name: used for supertitle
#     output_dir: save directory for boxplots
#     channel_names: list of EEG channel names
#     """

#     num_nodes, num_features = dataset[0].x.shape
#     labels_set = sorted(list(set(int(d.y.item()) for d in dataset)))

#     # Create directories
#     os.makedirs(output_dir, exist_ok=True)

#     # ---- Loop over each feature index ----
#     for feat_idx in range(num_features):

#         # Collect values: dict[label][node] -> list
#         values_by_label = {label: [[] for _ in range(num_nodes)] for label in labels_set}

#         for data in dataset:
#             label = int(data.y.item())
#             feature_values = data.x[:, feat_idx].cpu().numpy()  # shape (19,)
#             for node_idx in range(num_nodes):
#                 values_by_label[label][node_idx].append(feature_values[node_idx])

#         # ---- Prepare data for boxplot ----
#         # We want: one box group per channel, with multiple boxes (one per class)
#         # Format for matplotlib:
#         #   data_for_plot[node] = [list_of_values_for_class0, list_of_values_for_class1, ...]
#         data_for_plot = []
#         for node_idx in range(num_nodes):
#             node_data = [values_by_label[label][node_idx] for label in labels_set]
#             data_for_plot.append(node_data)

#         plt.figure(figsize=(14, 6))

#         # Create grouped boxplot manually
#         # Offsets per class
#         positions = []
#         box_data = []
#         num_classes = len(labels_set)
#         width = 0.8 / num_classes

#         base_positions = np.arange(num_nodes)

#         for class_idx, label in enumerate(labels_set):
#             # values for this class for each node
#             class_node_values = [values_by_label[label][node_idx] for node_idx in range(num_nodes)]
#             class_positions = base_positions + class_idx * width
#             positions.extend(class_positions)
#             box_data.extend(class_node_values)

#         # Actually plot the boxplot
#         plt.boxplot(
#             box_data,
#             positions=positions,
#             widths=width,
#             patch_artist=True,
#             manage_ticks=False
#         )

#         # Color each class differently
#         colors = plt.cm.Set2(np.linspace(0, 1, num_classes))
#         for class_idx in range(num_classes):
#             for b in range(num_nodes):
#                 patch_idx = class_idx * num_nodes + b
#                 plt.gca().artists[patch_idx].set_facecolor(colors[class_idx])

#         # ---- Labeling ----
#         plt.xticks(
#             base_positions + (num_classes - 1) * width / 2,
#             channel_names,
#             rotation=45,
#             ha='right'
#         )

#         plt.title(f"Feature {feat_idx} — Node Feature Distribution (Boxplots)")
#         plt.suptitle(folder_name, fontsize=13, y=1.02)

#         # Legend
#         handles = [
#             plt.Line2D([0], [0], color=colors[i], lw=10)
#             for i in range(num_classes)
#         ]
#         plt.legend(handles, [class_names[label] for label in labels_set], loc="upper right")

#         plt.tight_layout(rect=[0, 0, 1, 0.95])

#         # Save output
#         out_path = os.path.join(output_dir, f"node_feature_{feat_idx}_boxplot.png")
#         plt.savefig(out_path)
#         plt.close()
# def plot_node_features_boxplot(dataset, class_names, channel_names, save_path, method):

#     num_nodes, num_features = dataset[0].x.shape
#     labels_set = sorted(list(set(int(d.y.item()) for d in dataset)))
#     colors = plt.cm.Set2(range(len(labels_set)))

#     for feat_idx in range(num_features):
#         # -----------------------------
#         # Gather all values per class
#         # -----------------------------
#         data_by_class = {label: [[] for _ in range(num_nodes)] for label in labels_set}

#         for data in dataset:
#             label = int(data.y.item())
#             xvals = data.x[:, feat_idx].numpy()
#             for node_idx in range(num_nodes):
#                 data_by_class[label][node_idx].append(xvals[node_idx])

#         # -----------------------------
#         # Create boxplot
#         # -----------------------------
#         plt.figure(figsize=(12, 5))
#         box_data = []

#         for label in labels_set:
#             # flatten node-wise lists into single list per class
#             flattened = [vals for vals in data_by_class[label]]
#             box_data.append(flattened)

#         # matplotlib needs list-of-lists for each class
#         # but here each element is list-of-nodes
#         bp = plt.boxplot(
#             box_data,
#             patch_artist=True
#         )

#         # -----------------------------
#         # Assign colors correctly
#         # -----------------------------
#         for i, box in enumerate(bp['boxes']):
#             box.set_facecolor(colors[i])
#             box.set_alpha(0.5)

#         # -----------------------------
#         # Labels
#         # -----------------------------
#         plt.xticks(
#             np.arange(1, len(labels_set) + 1),
#             [class_names[c] for c in labels_set]
#         )
#         plt.ylabel(f"Feature {feat_idx}")
#         plt.title(f"{method} - Node Feature {feat_idx} by Class")

#         plt.tight_layout()
#         filename = os.path.join(save_path, f"{method}_feat{feat_idx}_node_features_boxplot.png")
#         plt.savefig(filename, dpi=300)
#         plt.close()