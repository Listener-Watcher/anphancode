import numpy as np
from viz_utils import build_adjacency


def compute_subject_edge_stats(graphs):
    """Compute edge frequency, mean, variability for one subject."""
    T = len(graphs)
    store = {}

    for g in graphs:
        adj = build_adjacency(g)
        N = adj.shape[0]

        for i in range(N):
            for j in range(i+1, N):
                w = adj[i, j]
                if w != 0:
                    if (i, j) not in store:
                        store[(i, j)] = []
                    store[(i, j)].append(w)

    stats = {}
    for e, ws in store.items():
        ws = np.array(ws)
        f = len(ws) / T
        mu = ws.mean()
        sd = ws.std()
        stats[e] = {"freq": f, "mu": mu, "sd": sd}

    return stats


def aggregate_class_stats(subject_stats, subject_labels):
    class_dict = {}

    for stat, label in zip(subject_stats, subject_labels):
        if label not in class_dict:
            class_dict[label] = {}

        for e, v in stat.items():
            if e not in class_dict[label]:
                class_dict[label][e] = {"freq": [], "mu": [], "sd": []}
            class_dict[label][e]["freq"].append(v["freq"])
            class_dict[label][e]["mu"].append(v["mu"])
            class_dict[label][e]["sd"].append(v["sd"])

    # compute means
    out = {}
    for c, edges in class_dict.items():
        out[c] = {}
        for e, v in edges.items():
            out[c][e] = {
                "freq": np.mean(v["freq"]),
                "mu": np.mean(v["mu"]),
                "sd": np.mean(v["sd"])
            }
    return out


def rank_edges_between_classes(class_stats):
    """Compute discriminability for each pair of classes."""
    classes = sorted(class_stats.keys())
    results = {}

    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            c1, c2 = classes[i], classes[j]
            edges = set(class_stats[c1].keys()) | set(class_stats[c2].keys())

            ranking = []
            for e in edges:
                if e not in class_stats[c1] or e not in class_stats[c2]:
                    continue

                s1 = class_stats[c1][e]
                s2 = class_stats[c2][e]

                d_mu = abs(s1["mu"] - s2["mu"])
                denom = (s1["sd"] + s2["sd"]) + 1e-8
                effect = d_mu / denom

                freq_sum = s1["freq"] + s2["freq"]
                score = effect * np.log1p(freq_sum)

                ranking.append((e, score))

            ranking.sort(key=lambda x: x[1], reverse=True)
            results[(c1, c2)] = ranking

    return results

def compute_edge_frequency(graphs, labels, num_nodes):
    """
    graphs: list of all graphs
    labels: list of labels (same length as graphs)
    num_nodes: number of EEG channels

    Returns:
        freq_by_class[class_id] = 19x19 matrix of frequencies
    """

    classes = sorted(set(labels))
    freq_by_class = {}

    # Initialize frequency counters
    for c in classes:
        freq_by_class[c] = np.zeros((num_nodes, num_nodes))
    
    counts = {c: 0 for c in classes}

    for g, c in zip(graphs, labels):
        counts[c] += 1
        edges = g.edge_index.t().cpu().numpy()

        # Mark edges as 1 for this graph
        mat = np.zeros((num_nodes, num_nodes))
        for i, j in edges:
            mat[i, j] = 1
            mat[j, i] = 1  # undirected

        freq_by_class[c] += mat

    # Normalize to frequency
    for c in classes:
        freq_by_class[c] /= counts[c]

    return freq_by_class
