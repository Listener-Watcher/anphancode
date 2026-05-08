import os
import torch
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def build_adjacency(data):
    """Convert PyG Data object to dense adjacency matrix."""
    num_nodes = data.num_nodes
    adj = np.zeros((num_nodes, num_nodes), dtype=float)

    edges = data.edge_index.t().cpu().numpy()
    weights = data.edge_attr.squeeze().cpu().numpy()

    for (u, v), w in zip(edges, weights):
        adj[u, v] = w
        adj[v, u] = w

    return adj


def save_log(msg, log_path):
    print(msg)
    with open(log_path, "a") as f:
        f.write(msg + "\n")

