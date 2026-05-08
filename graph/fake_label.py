
from typing import Dict, List, Tuple, Any
from data_preparation import build_graphs_from_master_topology
from lib import *


def assign_subject_fake_labels_to_segments(
    graphs,
    subject_ids=None,
    source="fake_score",
    aggregate_mode="mean",
    num_fake_classes=3,
    top_k_ratio=0.3,
    segment_label_attr="fake_segment_labelB",
    subject_label_attr="fake_subject_label",
    subject_score_attr="fake_subject_score",
    subject_threshold_attr="fake_subject_thresholds",
    update_y=True,
):
    ordered_subject_ids, fake_subject_labels, fake_subject_scores, thresholds = \
        build_fake_subject_labels_from_graphs(
            graphs=graphs,
            subject_ids=subject_ids,
            source=source,
            aggregate_mode=aggregate_mode,
            num_fake_classes=num_fake_classes,
            top_k_ratio=top_k_ratio,
        )

    subject_label_map = {
        sid: int(lbl)
        for sid, lbl in zip(ordered_subject_ids, fake_subject_labels)
    }
    subject_score_map = {
        sid: float(score)
        for sid, score in zip(ordered_subject_ids, fake_subject_scores)
    }

    for g in graphs:
        sid = g.subject_id
        if sid not in subject_label_map:
            continue

        subj_label = subject_label_map[sid]
        subj_score = subject_score_map[sid]

        setattr(g, subject_label_attr, subj_label)
        setattr(g, subject_score_attr, subj_score)
        setattr(g, subject_threshold_attr, thresholds)

        seg_label = torch.tensor([subj_label], dtype=torch.long)
        setattr(g, segment_label_attr, seg_label)

        if update_y:
            g.y = seg_label.clone()

    return graphs, subject_label_map, subject_score_map, thresholds
    
def aggregate_values(values, mode="mean", top_k_ratio=0.3):
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        raise ValueError("Empty value list for aggregation.")

    if mode == "mean":
        return float(values.mean())
    elif mode == "median":
        return float(np.median(values))
    elif mode == "max":
        return float(values.max())
    elif mode == "topk_mean":
        k = max(1, int(np.ceil(len(values) * top_k_ratio)))
        vals_sorted = np.sort(values)[::-1]
        return float(vals_sorted[:k].mean())
    elif mode == "majority":
        # only meaningful if values are binary 0/1
        return float(values.mean())
    else:
        raise ValueError(f"Unknown aggregate mode: {mode}")

def build_fake_subject_labels_from_graphs(
    graphs,
    subject_ids=None,
    source="fake_score",          # "fake_score" or graph attribute name like "y"
    aggregate_mode="mean",        # "mean", "median", "max", "topk_mean", "majority"
    # threshold_quantile=0.5,
    num_fake_classes=3,
    top_k_ratio=0.3,
):
    subj_to_vals = defaultdict(list)

    for g in graphs:
        sid = g.subject_id

        if source == "y":
            # g.y is often shape [1]
            val = float(g.y.view(-1)[0].item())
        else:
            if not hasattr(g, source):
                raise AttributeError(f"Graph is missing attribute '{source}'")
            attr = getattr(g, source)

            if hasattr(attr, "item"):
                val = float(attr.item())
            else:
                val = float(attr)

        subj_to_vals[sid].append(val)

    if subject_ids is None:
        ordered_subject_ids = sorted(subj_to_vals.keys())
    else:
        ordered_subject_ids = [sid for sid in subject_ids if sid in subj_to_vals]

    fake_subject_scores = []
    for sid in ordered_subject_ids:
        score = aggregate_values(
            subj_to_vals[sid],
            mode=aggregate_mode,
            top_k_ratio=top_k_ratio,
        )
        fake_subject_scores.append(score)

    scores_np = np.asarray(fake_subject_scores, dtype=np.float32)
    # threshold = float(np.quantile(fake_subject_scores_np, threshold_quantile))
    # fake_subject_labels = [int(score > threshold) for score in fake_subject_scores]
    if num_fake_classes == 2:
        thresholds = [float(np.quantile(scores_np, 0.5))]
        fake_subject_labels = [int(score > thresholds[0]) for score in fake_subject_scores]

    elif num_fake_classes == 3:
        thresholds = [
            float(np.quantile(scores_np, 1/3)),
            float(np.quantile(scores_np, 2/3)),
        ]
        fake_subject_labels = []
        for score in fake_subject_scores:
            if score < thresholds[0]:
                fake_subject_labels.append(0)
            elif score < thresholds[1]:
                fake_subject_labels.append(1)
            else:
                fake_subject_labels.append(2)
    else:
        raise ValueError("Currently only num_fake_classes=2 or 3 is supported.")

    return ordered_subject_ids, fake_subject_labels, fake_subject_scores, thresholds



def _compute_fake_score_from_graph(g, fake_label_type="nonlinear_edge", motif_k=4):
    sparse_adj = _build_sparse_adj_from_edge_index(
        num_nodes=g.x.shape[0],
        edge_index=g.edge_index,
        edge_weight=g.edge_attr.view(-1),
    )

    if fake_label_type == "nonlinear_edge":
        score = _score_nonlinear_edge_manual(sparse_adj, g.x)
    elif fake_label_type == "dirichlet_contrast":
        score = _score_dirichlet_contrast_manual(sparse_adj, g.x)
    elif fake_label_type == "localized_motif":
        score = _score_localized_motif_manual(sparse_adj, g.x, motif_k=motif_k)
    else:
        raise ValueError(f"Unknown fake_label_type: {fake_label_type}")

    return float(score.item())


def _compute_fake_thresholds(fake_scores, num_fake_classes=3):
    fake_scores_t = torch.tensor(fake_scores, dtype=torch.float32)

    if num_fake_classes == 2:
        return [float(torch.quantile(fake_scores_t, 0.5).item())]
    elif num_fake_classes == 3:
        return [
            float(torch.quantile(fake_scores_t, 1 / 3).item()),
            float(torch.quantile(fake_scores_t, 2 / 3).item()),
        ]
    else:
        raise ValueError("Currently only num_fake_classes=2 or 3 is supported.")


def _score_to_fake_label(score, thresholds, num_fake_classes=3):
    if num_fake_classes == 2:
        return int(score > thresholds[0])

    elif num_fake_classes == 3:
        if score < thresholds[0]:
            return 0
        elif score < thresholds[1]:
            return 1
        else:
            return 2

    else:
        raise ValueError("Currently only num_fake_classes=2 or 3 is supported.")


def build_graphs_from_master_topology_fake_label(
    master_path,
    subject_ids=None,
    undirected=True,
    filter_method="MST",
    fixed_edges=None,
    channel_names=None,
    topk=None,
    top_percent=None,
    corruption_mode=None,
    use_fake_labels=False,
    fake_label_type="nonlinear_edge",
    num_fake_classes=3,
    motif_k=4,
    standardize_features=True,
    label_key="class_id",   # or "segment_label" if present in your pt
):
    """
    Build PyG graphs from full topology, optionally overwrite labels
    using graph-based fake scores computed on the filtered topology.
    """

    # 1) Reuse the existing topology builder
    graphs = build_graphs_from_master_topology(
        master_path=master_path,
        subject_ids=subject_ids,
        undirected=undirected,
        filter_method=filter_method,
        topk=topk,
        top_percent=top_percent,
        fixed_edges=fixed_edges,
        channel_names=channel_names,
        corruption_mode=corruption_mode,
        standardize_features=standardize_features,
        label_key=label_key,
    )

    if len(graphs) == 0:
        return []

    # always keep real label metadata if available in master file
    obj = torch.load(master_path, map_location="cpu")
    if isinstance(obj, dict) and "data" in obj:
        all_data = obj["data"]
    elif isinstance(obj, list):
        all_data = obj
    else:
        raise TypeError("Unsupported .pt format")

    if subject_ids is not None:
        subject_ids = set(subject_ids)
        all_data = [d for d in all_data if d["subject_id"] in subject_ids]

    for g, entry in zip(graphs, all_data):
        if "class_id" in entry:
            g.real_label = int(entry["class_id"])
        if "segment_label" in entry:
            g.segment_label = int(entry["segment_label"])

    # 2) If no fake labels requested, return directly
    if not use_fake_labels:
        return graphs

    # 3) Compute fake scores from the already-built sparse graphs
    fake_scores = [
        _compute_fake_score_from_graph(
            g,
            fake_label_type=fake_label_type,
            motif_k=motif_k,
        )
        for g in graphs
    ]

    thresholds = _compute_fake_thresholds(
        fake_scores=fake_scores,
        num_fake_classes=num_fake_classes,
    )

    # 4) Overwrite graph labels with fake labels
    for g, score in zip(graphs, fake_scores):
        fake_y = _score_to_fake_label(
            score=score,
            thresholds=thresholds,
            num_fake_classes=num_fake_classes,
        )

        g.y = torch.tensor([fake_y], dtype=torch.long)
        g.fake_score = score
        g.fake_thresholds = thresholds
        g.fake_label_type = fake_label_type
        g.num_fake_classes = num_fake_classes
        g.fake_segment_label = int(fake_y)

    return graphs


def _to_2d_features(x: torch.Tensor) -> torch.Tensor:
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    elif x.ndim >= 2:
        x = x.reshape(x.shape[0], -1)
    else:
        raise ValueError(f"node_features must have ndim >= 1, got {x.ndim}")
    return x
    
def _build_sparse_adj_from_edge_index(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Build dense adjacency matrix [N, N] from edge_index and edge_weight.
    """
    adj_sparse = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj_sparse[edge_index[0], edge_index[1]] = edge_weight
    return adj_sparse


def to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    """Convert input to torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.detach().clone().to(dtype)
    return torch.tensor(x, dtype=dtype)


def symmetrize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Make adjacency symmetric and remove NaN/Inf."""
    adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj must be square [N, N], got {tuple(adj.shape)}")
    adj = 0.5 * (adj + adj.T)
    return adj


def preprocess_node_features(x: torch.Tensor) -> torch.Tensor:
    """
    Convert node features into shape [N, F].
    Supports:
      - [N, F]
      - [N]
      - [N, ...] -> flatten trailing dims
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim == 1:
        x = x.unsqueeze(-1)  # [N] -> [N, 1]
    elif x.ndim >= 2:
        n = x.shape[0]
        x = x.reshape(n, -1)  # flatten all feature dims after node dim
    else:
        raise ValueError(f"node_features must have ndim >= 1, got {x.ndim}")

    return x


def _zscore_per_feature(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize each feature across nodes within a segment."""
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)


def normalize_adj_weights(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize adjacency weights to a stable range.
    Keeps zeros at zero.
    """
    adj = adj.clone()
    adj.fill_diagonal_(0.0)
    max_abs = adj.abs().max()
    if max_abs > eps:
        adj = adj / max_abs
    return adj


def _choose_motif_nodes(adj: torch.Tensor, k: int = 4) -> torch.Tensor:
    """
    Deterministically choose motif nodes using weighted degree.
    This avoids randomness and makes the synthetic labels reproducible.
    """
    degree = adj.sum(dim=1)
    k = min(k, adj.shape[0])
    motif_idx = torch.topk(degree, k=k, largest=True).indices
    return motif_idx


def _score_nonlinear_edge_manual(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Nonlinear edge-coupling score:
        sum_ij A_ij * sigmoid(<x_i, x_j> / sqrt(F))
    """
    f = x.shape[1]
    sim = (x @ x.T) / math.sqrt(max(f, 1))
    score_mat = torch.sigmoid(sim)
    score = (adj * score_mat).sum() / (adj.abs().sum() + 1e-8)
    return score


def _score_dirichlet_contrast_manual(adj: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Graph Dirichlet energy contrast.
    If F >= 2:
        use first half of features vs second half of features
    else:
        use full feature set.
    Produces a scalar that depends on spatial smoothness/roughness.
    """
    n, f = x.shape

    def dirichlet_energy(x_sub: torch.Tensor) -> torch.Tensor:
        # pairwise squared distances: ||x_i - x_j||^2
        diff = x_sub.unsqueeze(1) - x_sub.unsqueeze(0)   # [N, N, Fsub]
        dist2 = (diff ** 2).sum(dim=-1)                  # [N, N]
        e = 0.5 * (adj * dist2).sum() / (adj.abs().sum() + 1e-8)
        return e

    if f >= 2:
        mid = f // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        e1 = dirichlet_energy(x1)
        e2 = dirichlet_energy(x2)
        score = e1 - e2
    else:
        score = dirichlet_energy(x)

    return score


def _score_localized_motif_manual(adj: torch.Tensor, x: torch.Tensor, motif_k: int = 4) -> torch.Tensor:
    """
    Localized motif score:
    Focus on a small high-degree node subset and its internal coupling/variance.

    Combines:
      - internal motif edge similarity
      - contrast between motif and non-motif mean activity
    """
    n = adj.shape[0]
    motif_idx = choose_motif_nodes(adj, k=motif_k)

    mask = torch.zeros(n, dtype=torch.bool)
    mask[motif_idx] = True
    non_motif_idx = torch.where(~mask)[0]

    x_m = x[motif_idx]  # [k, F]
    adj_m = adj[motif_idx][:, motif_idx]  # [k, k]

    # internal nonlinear coupling inside motif
    f = x.shape[1]
    sim_m = (x_m @ x_m.T) / math.sqrt(max(f, 1))
    internal_score = (adj_m * torch.tanh(sim_m)).sum() / (adj_m.abs().sum() + 1e-8)

    # contrast motif vs background
    motif_mean = x_m.mean(dim=0)
    if len(non_motif_idx) > 0:
        non_motif_mean = x[non_motif_idx].mean(dim=0)
        contrast = ((motif_mean - non_motif_mean) ** 2).mean()
    else:
        contrast = torch.tensor(0.0, dtype=x.dtype)

    score = internal_score + 0.5 * contrast
    return score


def compute_scores_for_sample(
    sample: Dict[str, Any],
    motif_k: int = 4,
    standardize_features: bool = True,
) -> Dict[str, float]:
    """
    Compute all continuous synthetic scores for one segment.
    """
    adj = to_tensor(sample["adj"], dtype=torch.float32)
    x = to_tensor(sample["node_features"], dtype=torch.float32)

    adj = symmetrize_adj(adj)
    adj = normalize_adj_weights(adj)
    x = preprocess_node_features(x)

    if adj.shape[0] != x.shape[0]:
        raise ValueError(
            f"Mismatch: adj has {adj.shape[0]} nodes but node_features has {x.shape[0]}"
        )

    if standardize_features:
        x = zscore_per_feature(x)

    scores = {
        "score_nonlinear_edge": float(score_nonlinear_edge(adj, x).item()),
        "score_dirichlet_contrast": float(score_dirichlet_contrast(adj, x).item()),
        "score_localized_motif": float(score_localized_motif(adj, x, motif_k=motif_k).item()),
    }
    return scores


def compute_thresholds(
    score_dicts: List[Dict[str, float]],
    quantile: float = 0.5,
) -> Dict[str, float]:
    """
    Compute one threshold per score type.
    Default quantile=0.5 gives roughly balanced fake labels.
    """
    keys = score_dicts[0].keys()
    thresholds = {}

    for key in keys:
        vals = torch.tensor([d[key] for d in score_dicts], dtype=torch.float32)
        thresholds[key] = float(torch.quantile(vals, quantile).item())

    return thresholds



def summarize_fake_labels(data: List[Dict[str, Any]]) -> None:
    """Print class balance for each fake label."""
    fake_keys = [
        "fake_label_nonlinear_edge",
        "fake_label_dirichlet_contrast",
        "fake_label_localized_motif",
    ]

    print("\n===== Fake Label Summary =====")
    for key in fake_keys:
        vals = torch.tensor([int(d[key]) for d in data], dtype=torch.int64)
        n1 = int(vals.sum().item())
        n0 = int((vals == 0).sum().item())
        print(f"{key}: 0 -> {n0}, 1 -> {n1}, positive_rate = {n1 / max(len(vals), 1):.4f}")


# def main():
#     # parser = argparse.ArgumentParser()
#     # parser.add_argument("--input_pt", type=str, required=True, help="Path to input .pt file")
#     # parser.add_argument("--output_pt", type=str, required=True, help="Path to output .pt file")
#     # parser.add_argument(
#     #     "--quantile",
#     #     type=float,
#     #     default=0.5,
#     #     help="Threshold quantile for binarizing scores (0.5 = median / near-balanced)",
#     # )
#     # parser.add_argument(
#     #     "--motif_k",
#     #     type=int,
#     #     default=4,
#     #     help="Number of motif nodes for fake_localized_motif",
#     # )
#     # parser.add_argument(
#     #     "--no_standardize_features",
#     #     action="store_true",
#     #     help="Disable per-segment feature z-scoring across nodes",
#     # )

#     # args = parser.parse_args()

#     # print(f"Loading: {args.input_pt}")
#     input_pt = "/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_coherence_alpha/data_processed/master_graph_data.pt"
#     quantile = 0.5
#     motif_k = 4
#     standardize_features = False
#     data = torch.load(input_pt, map_location="cpu")

#     if not isinstance(data, list):
#         raise TypeError(
#             f"Expected .pt file to contain a list of dicts, got {type(data)}"
#         )

#     if len(data) == 0:
#         raise ValueError("Input dataset is empty.")

#     required_keys = {"subject_id", "class_id", "adj", "node_features", "segment_id", "start_sample"}
#     missing = required_keys - set(data[0].keys())
#     if missing:
#         raise KeyError(f"Missing required keys in first sample: {missing}")

#     updated_data, thresholds = add_fake_labels_to_dataset(
#         data,
#         quantile=quantile,
#         motif_k=motif_k,
#         standardize_features=standardize_features,
#     )
#     output_pt = "/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_coherence_alpha/data_processed/master_graph_data_with_fake_labels.pt"
#     os.makedirs(os.path.dirname(output_pt), exist_ok=True) if os.path.dirname(output_pt) else None

#     save_obj = {
#         "data": updated_data,
#         "fake_label_metadata": {
#             "thresholds": thresholds,
#             "quantile": quantile,
#             "motif_k": motif_k,
#             "standardize_features": standardize_features,
#             "description": {
#                 "fake_label_nonlinear_edge": "Binary label from nonlinear edge-coupling score",
#                 "fake_label_dirichlet_contrast": "Binary label from graph Dirichlet energy contrast score",
#                 "fake_label_localized_motif": "Binary label from localized motif score",
#             },
#         },
#     }

#     torch.save(save_obj, output_pt)
#     print(f"Saved updated dataset to: {output_pt}")
#     print("Thresholds:")
#     for k, v in thresholds.items():
#         print(f"  {k}: {v:.6f}")

#     summarize_fake_labels(updated_data)


# if __name__ == "__main__":
#     main()
