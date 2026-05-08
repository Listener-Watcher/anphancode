import numpy as np
from viz_utils import build_adjacency


def aggregate_subject_graphs(graphs, method="mean"):
    """Aggregate window-level graphs for one subject."""
    adjs = [build_adjacency(g) for g in graphs]

    if method == "mean":
        return np.mean(adjs, axis=0)
    else:
        return np.sum(adjs, axis=0)


def compute_edge_list(adj):
    """Return list of ((i,j), weight) for non-zeros."""
    edges = {}
    N = adj.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j] != 0:
                edges[(i, j)] = float(adj[i, j])
    return edges
