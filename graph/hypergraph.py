from torch_geometric.nn import HypergraphConv
from lib import *
from model import *
from data_utils import *
from graph_utils import *
from data_preparation import * 
from utils_all import *
from fake_label import *
from fake_label import _compute_fake_thresholds, _score_to_fake_label, _zscore_per_feature, _score_nonlinear_edge_manual, _score_dirichlet_contrast_manual, _score_localized_motif_manual
from torch.nn import Linear
from torch_geometric.loader import DataLoader
import torch
import torch.nn.functional as F


# =========================================================
# helpers
# =========================================================
def _infer_num_hyperedges(g):
    if hasattr(g, "region_names"):
        return len(g.region_names)
    if hasattr(g, "hyperedge_weight") and g.hyperedge_weight is not None:
        return int(g.hyperedge_weight.numel())
    if g.hyperedge_index.numel() == 0:
        return 0
    return int(g.hyperedge_index[1].max().item()) + 1


def _get_hyperedge_members(g):
    """
    Return:
        members_list: list of 1D LongTensor, each containing node indices in hyperedge h
    """
    hyperedge_index = g.hyperedge_index.long()
    node_idx = hyperedge_index[0]
    hedge_idx = hyperedge_index[1]

    num_hyperedges = _infer_num_hyperedges(g)
    members_list = []

    for h in range(num_hyperedges):
        members = node_idx[hedge_idx == h]
        if members.numel() > 0:
            members = torch.unique(members)
        members_list.append(members)

    return members_list


def _get_hyperedge_agg_weights(g, device=None):
    """
    Use abs(hyperedge_weight) for score aggregation, because your
    hyperedge_weight may be negative if built from mean_adj.
    """
    H = _infer_num_hyperedges(g)

    if hasattr(g, "hyperedge_weight") and g.hyperedge_weight is not None:
        w = torch.as_tensor(g.hyperedge_weight, dtype=torch.float32, device=device).view(-1)
        if w.numel() < H:
            pad = torch.ones(H - w.numel(), dtype=torch.float32, device=device)
            w = torch.cat([w, pad], dim=0)
        elif w.numel() > H:
            w = w[:H]
        w = w.abs()
    else:
        w = torch.ones(H, dtype=torch.float32, device=device)

    return w


def _weighted_mean(values, weights, eps=1e-8):
    if len(values) == 0:
        return torch.tensor(0.0, dtype=torch.float32, device=weights.device)

    v = torch.stack(values)
    w = torch.stack(weights)
    return (v * w).sum() / (w.sum() + eps)


def _compute_hyperedge_centroids_and_variances(g, eps=1e-8):
    """
    For each hyperedge h:
        centroid_h = mean of node features in h
        var_h      = mean squared distance to centroid

    Returns
    -------
    centroids : list[Tensor[F]]
    variances : list[Tensor[scalar]]
    members_list : list[LongTensor]
    agg_weights : Tensor[H]
    """
    x = torch.as_tensor(g.x, dtype=torch.float32)
    members_list = _get_hyperedge_members(g)
    agg_weights = _get_hyperedge_agg_weights(g, device=x.device)

    centroids = []
    variances = []

    for members in members_list:
        if members.numel() == 0:
            centroids.append(torch.zeros(x.shape[1], dtype=torch.float32, device=x.device))
            variances.append(torch.tensor(0.0, dtype=torch.float32, device=x.device))
            continue

        Xh = x[members]                     # [m, F]
        mu = Xh.mean(dim=0)                 # [F]
        var = ((Xh - mu) ** 2).sum(dim=1).mean()   # scalar

        centroids.append(mu)
        variances.append(var)

    return centroids, variances, members_list, agg_weights


# =========================================================
# score 1: within-hyperedge feature similarity
# =========================================================
def score_within_hyperedge_similarity(g, use_abs=False, eps=1e-8):
    """
    High score if nodes inside each hyperedge have similar features.

    For each hyperedge h:
        score_h = mean pairwise cosine similarity among nodes in h

    Final score = weighted average over hyperedges.
    """
    x = torch.as_tensor(g.x, dtype=torch.float32)
    members_list = _get_hyperedge_members(g)
    agg_weights = _get_hyperedge_agg_weights(g, device=x.device)

    per_h_scores = []
    per_h_weights = []

    for h, members in enumerate(members_list):
        if members.numel() <= 1:
            continue

        Xh = x[members]                                  # [m, F]
        Xh = F.normalize(Xh, p=2, dim=1, eps=eps)
        sim = Xh @ Xh.T                                  # [m, m]

        if use_abs:
            sim = sim.abs()

        tri = torch.triu_indices(sim.shape[0], sim.shape[1], offset=1, device=sim.device)
        vals = sim[tri[0], tri[1]]

        if vals.numel() == 0:
            continue

        per_h_scores.append(vals.mean())
        per_h_weights.append(agg_weights[h])

    return _weighted_mean(per_h_scores, per_h_weights, eps=eps)


# =========================================================
# score 2: hyperedge smoothness
# =========================================================
def score_hyperedge_smoothness(g, eps=1e-8):
    """
    High score if node features vary little inside each hyperedge.

    For each hyperedge h:
        energy_h = mean_i ||x_i - mu_h||^2
        score_h  = 1 / (1 + energy_h)

    Final score = weighted average over hyperedges.
    """
    centroids, variances, members_list, agg_weights = _compute_hyperedge_centroids_and_variances(g, eps=eps)

    per_h_scores = []
    per_h_weights = []

    for h, members in enumerate(members_list):
        if members.numel() == 0:
            continue

        energy_h = variances[h]
        score_h = 1.0 / (1.0 + energy_h)

        per_h_scores.append(score_h)
        per_h_weights.append(agg_weights[h])

    return _weighted_mean(per_h_scores, per_h_weights, eps=eps)


# =========================================================
# score 3: node-to-hyperedge consistency
# =========================================================
def score_node_to_hyperedge_consistency(g, eps=1e-8):
    """
    High score if each node is well aligned with the centroid of its incident hyperedge.

    For each incident pair (node i, hyperedge h):
        consistency(i,h) = cosine(x_i, mu_h)

    Final score = weighted average over all incidents.
    """
    x = torch.as_tensor(g.x, dtype=torch.float32)
    centroids, _, members_list, agg_weights = _compute_hyperedge_centroids_and_variances(g, eps=eps)

    x_norm = F.normalize(x, p=2, dim=1, eps=eps)

    per_inc_scores = []
    per_inc_weights = []

    for h, members in enumerate(members_list):
        if members.numel() == 0:
            continue

        mu = centroids[h].unsqueeze(0)                       # [1, F]
        mu = F.normalize(mu, p=2, dim=1, eps=eps)           # [1, F]
        sims = (x_norm[members] * mu).sum(dim=1)            # [m]

        for s in sims:
            per_inc_scores.append(s)
            per_inc_weights.append(agg_weights[h])

    return _weighted_mean(per_inc_scores, per_inc_weights, eps=eps)


# =========================================================
# score 4: cross-region contrast
# =========================================================
def score_cross_region_contrast(g, log_scale=True, eps=1e-8):
    """
    High score if different hyperedges/regions have distinct centroids,
    relative to their within-hyperedge spread.

    between = mean_{h<k} ||mu_h - mu_k||^2
    within  = weighted mean_h mean_i ||x_i - mu_h||^2

    score = between / (within + eps)
    optionally log(1 + score) for stability
    """
    centroids, variances, members_list, agg_weights = _compute_hyperedge_centroids_and_variances(g, eps=eps)

    valid_idx = [h for h, members in enumerate(members_list) if members.numel() > 0]
    if len(valid_idx) <= 1:
        return torch.tensor(0.0, dtype=torch.float32)

    # between-region distance
    between_vals = []
    between_weights = []

    for i in range(len(valid_idx)):
        h = valid_idx[i]
        for j in range(i + 1, len(valid_idx)):
            k = valid_idx[j]

            d2 = ((centroids[h] - centroids[k]) ** 2).sum()
            w_pair = 0.5 * (agg_weights[h] + agg_weights[k])

            between_vals.append(d2)
            between_weights.append(w_pair)

    between = _weighted_mean(between_vals, between_weights, eps=eps)

    # within-region spread
    within_vals = []
    within_weights = []
    for h in valid_idx:
        within_vals.append(variances[h])
        within_weights.append(agg_weights[h])

    within = _weighted_mean(within_vals, within_weights, eps=eps)

    score = between / (within + eps)
    if log_scale:
        score = torch.log1p(score)

    return score


# =========================================================
# unified dispatcher
# =========================================================
def compute_fake_score_from_hypergraph(
    g,
    fake_label_type="within_hyperedge_similarity",
    motif_k=4,      # unused here, kept for API compatibility
    eps=1e-8,
):
    """
    True hypergraph scores (no graph clique expansion).

    Supported fake_label_type:
      - "within_hyperedge_similarity"
      - "hyperedge_smoothness"
      - "node_to_hyperedge_consistency"
      - "cross_region_contrast"
    """
    if fake_label_type == "within_hyperedge_similarity":
        score = score_within_hyperedge_similarity(g, use_abs=False, eps=eps)

    elif fake_label_type == "hyperedge_smoothness":
        score = score_hyperedge_smoothness(g, eps=eps)

    elif fake_label_type == "node_to_hyperedge_consistency":
        score = score_node_to_hyperedge_consistency(g, eps=eps)

    elif fake_label_type == "cross_region_contrast":
        score = score_cross_region_contrast(g, log_scale=True, eps=eps)

    else:
        raise ValueError(f"Unknown fake_label_type: {fake_label_type}")

    return float(score.item())


# =========================================================
# thresholds + label assignment
# =========================================================
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


def assign_fake_labels_to_hypergraphs(
    hypergraphs,
    fake_label_type="within_hyperedge_similarity",
    num_fake_classes=3,
    update_y=True,
):
    """
    Compute one fake score per hypergraph, threshold into fake segment labels,
    and optionally overwrite g.y so training with batch.y uses the fake labels.
    """
    if len(hypergraphs) == 0:
        return hypergraphs, [], None

    fake_scores = [
        compute_fake_score_from_hypergraph(
            g,
            fake_label_type=fake_label_type,
        )
        for g in hypergraphs
    ]

    thresholds = _compute_fake_thresholds(
        fake_scores=fake_scores,
        num_fake_classes=num_fake_classes,
    )

    for g, score in zip(hypergraphs, fake_scores):
        fake_label = _score_to_fake_label(
            score=score,
            thresholds=thresholds,
            num_fake_classes=num_fake_classes,
        )

        g.fake_score = float(score)
        g.fake_segment_label = int(fake_label)
        g.fake_thresholds = thresholds
        g.fake_label_type = fake_label_type
        g.num_fake_classes = num_fake_classes

        if update_y:
            g.y = torch.tensor([fake_label], dtype=torch.long)

    return hypergraphs, fake_scores, thresholds
class HyperGraphData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "hyperedge_index":
            # row 0 -> node indices
            # row 1 -> hyperedge indices
            return torch.tensor([[self.x.size(0)], [self.hyperedge_weight.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

def build_hypergraphs_from_master_region_topology(
    master_path,
    channel_names,
    region_to_channels,                 # dict: {"frontal": [...], "temporal": [...], ...}
    subject_ids=None,
    label_key="class_id",
    standardize_features=False,
    corruption_mode=None,               # None, "permute_consistent", "permute_adj_only", "identity", "random"
    hyperedge_weight_mode="mean_adj",   # "mean_adj", "mean_abs_adj", "ones"
    keep_empty_hyperedges=False,
):
    def _to_2d_features(x):
        x = torch.as_tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        return x

    def make_identity_adj(n):
        return torch.eye(n, dtype=torch.float32)

    def make_random_adj_like_with_weights(adj, undirected=True):
        n = adj.shape[0]
        out = torch.rand((n, n), dtype=torch.float32)
        if undirected:
            out = (out + out.T) / 2.0
        out.fill_diagonal_(0.0)
        return out

    def permute_graph_consistently(x, adj):
        n = x.shape[0]
        perm = torch.randperm(n)
        x_perm = x[perm]
        adj_perm = adj[perm][:, perm]
        return x_perm, adj_perm, perm

    def permute_adj_only(adj):
        n = adj.shape[0]
        perm = torch.randperm(n)
        adj_perm = adj[perm][:, perm]
        return adj_perm, perm

    # -------------------------------------------------------
    # load master data
    # -------------------------------------------------------
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

    if len(all_data) == 0:
        return []

    n_nodes = len(channel_names)
    name_to_idx = {name: i for i, name in enumerate(channel_names)}

    # -------------------------------------------------------
    # build fixed region-based hyperedge topology once
    # -------------------------------------------------------
    region_names = []
    hyperedge_members = []   # list of lists, each inner list = node indices in that hyperedge

    missing_channels_by_region = {}

    for region_name, ch_list in region_to_channels.items():
        members = []
        missing_here = []

        for ch in ch_list:
            if ch in name_to_idx:
                members.append(name_to_idx[ch])
            else:
                missing_here.append(ch)

        members = sorted(set(members))

        if len(members) == 0:
            if keep_empty_hyperedges:
                region_names.append(region_name)
                hyperedge_members.append([])
            missing_channels_by_region[region_name] = missing_here
            continue

        region_names.append(region_name)
        hyperedge_members.append(members)

        if len(missing_here) > 0:
            missing_channels_by_region[region_name] = missing_here

    if len(region_names) == 0:
        raise ValueError("No valid region hyperedges could be built from region_to_channels and channel_names.")

    # -------------------------------------------------------
    # build hyperedge_index once (topology fixed across samples)
    # -------------------------------------------------------
    incidence_node_idx = []
    incidence_hyperedge_idx = []

    for h_idx, members in enumerate(hyperedge_members):
        for node_idx in members:
            incidence_node_idx.append(node_idx)
            incidence_hyperedge_idx.append(h_idx)

    hyperedge_index = torch.tensor(
        [incidence_node_idx, incidence_hyperedge_idx],
        dtype=torch.long
    )

    # optional metadata: node x region binary matrix
    node_to_region_mask = torch.zeros((n_nodes, len(region_names)), dtype=torch.float32)
    for h_idx, members in enumerate(hyperedge_members):
        for node_idx in members:
            node_to_region_mask[node_idx, h_idx] = 1.0

    # -------------------------------------------------------
    # helper to compute one scalar weight per hyperedge
    # -------------------------------------------------------
    def compute_hyperedge_weights(adj_used):
        weights = []

        for members in hyperedge_members:
            if len(members) <= 1:
                # singleton hyperedge: no within-region pair exists
                weights.append(1.0 if hyperedge_weight_mode != "ones" else 1.0)
                continue

            sub_adj = adj_used[np.ix_(members, members)].clone()
            sub_adj.fill_diagonal_(0.0)

            # use upper-triangle only, avoid double-counting
            triu_idx = torch.triu_indices(sub_adj.shape[0], sub_adj.shape[1], offset=1)
            vals = sub_adj[triu_idx[0], triu_idx[1]]

            if vals.numel() == 0:
                w = 1.0
            elif hyperedge_weight_mode == "mean_adj":
                w = vals.mean().item()
            elif hyperedge_weight_mode == "mean_abs_adj":
                w = vals.abs().mean().item()
            elif hyperedge_weight_mode == "ones":
                w = 1.0
            else:
                raise ValueError(f"Unknown hyperedge_weight_mode: {hyperedge_weight_mode}")

            weights.append(float(w))

        return torch.tensor(weights, dtype=torch.float32)

    # -------------------------------------------------------
    # build graphs
    # -------------------------------------------------------
    graphs = []

    for entry in all_data:
        x = _to_2d_features(entry["node_features"])
        adj_full = torch.as_tensor(entry["adj"], dtype=torch.float32)
        y = torch.tensor([int(entry[label_key])], dtype=torch.long)

        if x.shape[0] != n_nodes:
            raise ValueError(
                f"node_features has {x.shape[0]} nodes but channel_names has {n_nodes}"
            )

        if adj_full.ndim != 2 or adj_full.shape[0] != adj_full.shape[1]:
            raise ValueError(f"adj must be square, got shape {tuple(adj_full.shape)}")

        if adj_full.shape[0] != n_nodes:
            raise ValueError(
                f"adj has {adj_full.shape[0]} nodes but channel_names has {n_nodes}"
            )

        if standardize_features:
            x = _zscore_per_feature(x)

        # -----------------------------------
        # optional corruption on adjacency
        # -----------------------------------
        if corruption_mode == "identity":
            adj_used = make_identity_adj(adj_full.shape[0])

        elif corruption_mode == "random":
            adj_used = make_random_adj_like_with_weights(adj_full, undirected=True)

        elif corruption_mode == "permute_consistent":
            x, adj_used, _ = permute_graph_consistently(x, adj_full)

        elif corruption_mode == "permute_adj_only":
            adj_used, _ = permute_adj_only(adj_full)

        elif corruption_mode is None:
            adj_used = adj_full.clone()

        else:
            raise ValueError(f"Unknown corruption_mode: {corruption_mode}")

        adj_used.fill_diagonal_(0.0)

        hyperedge_weight = compute_hyperedge_weights(adj_used)

        # g = Data(
        #     x=x,
        #     y=y,
        #     hyperedge_index=hyperedge_index.clone(),
        #     hyperedge_weight=hyperedge_weight,
        # )
        g = HyperGraphData(
            x=x,
            y=y,
            hyperedge_index=hyperedge_index.clone().long(),
            hyperedge_weight=hyperedge_weight.float(),
        )
        # metadata
        g.region_names = region_names
        g.node_to_region_mask = node_to_region_mask.clone()
        g.subject_id = entry["subject_id"]
        g.segment_id = entry.get("segment_id", 0)
        g.start_sample = entry.get("start_sample", None)

        graphs.append(g)

    return graphs

# def assign_fake_labels_to_hypergraphs(
#     graphs,
#     fake_label_type="nonlinear_edge",
#     motif_k=4,
#     num_fake_classes=3
#     ):
#     fake_scores = [
#         compute_fake_score_from_hypergraph(
#             g,
#             fake_label_type=fake_label_type,
#             motif_k=motif_k,
#         )
#         for g in graphs
#     ]

#     thresholds = _compute_fake_thresholds(
#         fake_scores=fake_scores,
#         num_fake_classes=num_fake_classes,
#     )

#     # 4) Overwrite graph labels with fake labels
#     for g, score in zip(graphs, fake_scores):
#         fake_y = _score_to_fake_label(
#             score=score,
#             thresholds=thresholds,
#             num_fake_classes=num_fake_classes,
#         )

#         g.y = torch.tensor([fake_y], dtype=torch.long)
#         g.fake_score = score
#         g.fake_segment_label = int(fake_y)

#     return graphs

# def hypergraph_to_clique_adj(num_nodes, hyperedge_index, hyperedge_weight=None):
#     """
#     Convert hypergraph incidence into a dense pairwise adjacency
#     using clique expansion.

#     hyperedge_index: [2, M]
#         row 0 = node indices
#         row 1 = hyperedge indices
#     hyperedge_weight: [num_hyperedges] or None
#     """
#     adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

#     node_idx = hyperedge_index[0]
#     hedge_idx = hyperedge_index[1]

#     num_hyperedges = int(hedge_idx.max().item()) + 1 if hedge_idx.numel() > 0 else 0

#     if hyperedge_weight is None:
#         hyperedge_weight = torch.ones(num_hyperedges, dtype=torch.float32)

#     for h in range(num_hyperedges):
#         members = node_idx[hedge_idx == h]
#         if members.numel() <= 1:
#             continue

#         w = float(hyperedge_weight[h].item())

#         for i in range(len(members)):
#             u = int(members[i])
#             for j in range(i + 1, len(members)):
#                 v = int(members[j])
#                 adj[u, v] += w
#                 adj[v, u] += w

#     adj.fill_diagonal_(0.0)
#     return adj


# def compute_fake_score_from_hypergraph(g, fake_label_type="nonlinear_edge", motif_k=4):
#     adj = hypergraph_to_clique_adj(
#         num_nodes=g.x.shape[0],
#         hyperedge_index=g.hyperedge_index,
#         hyperedge_weight=g.hyperedge_weight,
#     )

#     if fake_label_type == "nonlinear_edge":
#         score = _score_nonlinear_edge_manual(adj, g.x)
#     elif fake_label_type == "dirichlet_contrast":
#         score = _score_dirichlet_contrast_manual(adj, g.x)
#     elif fake_label_type == "localized_motif":
#         score = _score_localized_motif_manual(adj, g.x, motif_k=motif_k)
#     else:
#         raise ValueError(f"Unknown fake_label_type: {fake_label_type}")

#     return float(score.item())


@torch.no_grad()
def eval_subject_level_from_segment_model(model, loader, device, num_classes, agg="mean_prob"):
    """
    Assumes each Data object has:
      - g.subject_id (string or int)
      - g.y is graph label (same for all segments of subject)
    Returns subject-level acc, macro-f1, cm, and per-subject preds.
    """
    model.eval()
    model.to(device)

    subj_probs = defaultdict(list)   # sid -> list of [C] probs
    subj_true  = {}                  # sid -> int label

    for batch in loader:
        batch = batch.to(device)

        # If your model expects (x, edge_index, edge_attr, batch)
        # logits = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)  # [num_graphs, C]


        logits = model(
            x=batch.x,
            hyperedge_index=batch.hyperedge_index,
            batch=batch.batch,
            hyperedge_weight=batch.hyperedge_weight if hasattr(batch, "hyperedge_weight") else None,
        )
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()              # [num_graphs, C]
        y = batch.y.view(-1).detach().cpu().numpy()                              # [num_graphs]

        # batch.subject_id will be a list-like after collation
        # In PyG, non-tensor attributes become a python list in the batch.
        sids = batch.subject_id  # list of subject_ids aligned with graphs in batch

        for i, sid in enumerate(sids):
            subj_probs[sid].append(probs[i])
            # label should be constant per subject; store once
            if sid not in subj_true:
                subj_true[sid] = int(y[i])

    # aggregate
    y_true_sub, y_pred_sub = [], []
    for sid, plist in subj_probs.items():
        P = np.stack(plist, axis=0)  # [nSeg, C]
        if agg == "mean_prob":
            pbar = P.mean(axis=0)
            pred = int(np.argmax(pbar))
        else:
            raise ValueError("agg must be 'mean_prob'")
        y_true_sub.append(subj_true[sid])
        y_pred_sub.append(pred)

    acc = accuracy_score(y_true_sub, y_pred_sub)
    f1  = f1_score(y_true_sub, y_pred_sub, average="macro", zero_division=0)
    cm  = confusion_matrix(y_true_sub, y_pred_sub, labels=list(range(num_classes)))

    return acc, f1, cm, (y_true_sub, y_pred_sub)

@torch.no_grad()
def eval_model(model, loader, device, num_classes):
    model.eval()
    ys, ps = [], []
    losses = []
    for batch in loader:
        batch = batch.to(device)
        # out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

        out = model(
            x=batch.x,
            hyperedge_index=batch.hyperedge_index,
            batch=batch.batch,
            hyperedge_weight=batch.hyperedge_weight if hasattr(batch, "hyperedge_weight") else None,
        )
        loss = F.cross_entropy(out, batch.y.view(-1))
        losses.append(loss.item())
        preds = out.argmax(dim=1).detach().cpu().numpy()
        labels = batch.y.view(-1).detach().cpu().numpy()
        ps.extend(list(preds))
        ys.extend(list(labels))
    if len(ys) == 0:
        return 0.0, 0.0, 0.0, None
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro", zero_division=0) if num_classes > 2 else f1_score(ys, ps, average="binary", zero_division=0)
    cm = confusion_matrix(ys, ps, labels=list(range(num_classes)))
    return float(np.mean(losses)), float(acc), float(f1), cm


def train_segment_level(
    model,
    train_loader,
    val_loader,
    device,
    num_classes,
    lr=3e-4,
    weight_decay=1e-3,
    epochs=200,
    patience=25,
    grad_clip=1.0,
    debug_first_batch=True,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # simple scheduler (optional)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=50)

    best_f1 = -1.0
    best_state = None
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []

        for bi, batch in enumerate(train_loader):
            batch = batch.to(device)

            if debug_first_batch and ep == 1 and bi == 0:
                print_batch_stats("[DEBUG train batch0]", batch)

            opt.zero_grad(set_to_none=True)
            # out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

            out = model(
                x=batch.x,
                hyperedge_index=batch.hyperedge_index,
                batch=batch.batch,
                hyperedge_weight=batch.hyperedge_weight if hasattr(batch, "hyperedge_weight") else None,
            )
            if not safe_isfinite(out):
                raise RuntimeError("Non-finite logits detected. Check edge_attr scaling / data.")
            loss = F.cross_entropy(out, batch.y.view(-1))
            if not safe_isfinite(loss):
                raise RuntimeError("Non-finite loss detected. Check labels / logits / scaling.")

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tr_losses.append(loss.item())

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss, va_acc, va_f1, _ = eval_model(model, val_loader, device, num_classes)
        sched.step(va_loss)
        lr_now = opt.param_groups[0]["lr"]

        print(f"Epoch {ep:03d}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.3f} | val_f1={va_f1:.3f} | lr={lr_now:.2e}")
        if ep == 1 or ep == epochs:
            val_sub_acc, val_sub_f1, val_sub_cm, _ = eval_subject_level_from_segment_model(
                model, val_loader, device, num_classes
            )
            print(f"[VAL-SUB] acc={val_sub_acc:.3f} f1={val_sub_f1:.3f}")
            print(val_sub_cm)

        if va_f1 > best_f1 + 1e-4:
            best_f1 = va_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                val_sub_acc, val_sub_f1, val_sub_cm, _ = eval_subject_level_from_segment_model(
                    model, val_loader, device, num_classes
                )
                print(f"[VAL-SUB] acc={val_sub_acc:.3f} f1={val_sub_f1:.3f}")
                print(val_sub_cm)
                print(f"[EarlyStop] best_val_f1={best_f1:.3f} @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_f1


class EEGHypergraphNet_basic(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.3, backnorm=True):
        super().__init__()

        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, hyperedge_index, batch, hyperedge_weight=None):
        x = self.conv1(x, hyperedge_index, hyperedge_weight=hyperedge_weight)
        if backnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, hyperedge_index, hyperedge_weight=hyperedge_weight)
        if backnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # graph-level output
        x = global_mean_pool(x, batch)
        out = self.classifier(x)
        return out

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.aggr import AttentionalAggregation
class EEGHypergraphNet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_classes,
        dropout=0.3,
        use_batchnorm=True,
        use_attention=True,
        attention_heads=4,
        attention_mode="node",   # "node" or "edge"
        readout="mean_max",      # "mean", "max", "sum", "mean_max", "mean_sum_max", "attn"
    ):
        super().__init__()

        self.use_batchnorm = use_batchnorm
        self.use_attention = use_attention
        self.readout_type = readout
        self.hidden_channels = hidden_channels

        # concat=False keeps output dim = hidden_channels
        self.conv1 = HypergraphConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            use_attention=use_attention,
            attention_mode=attention_mode,
            heads=attention_heads if use_attention else 1,
            concat=False,
            dropout=dropout if use_attention else 0.0,
        )

        self.conv2 = HypergraphConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            use_attention=use_attention,
            attention_mode=attention_mode,
            heads=attention_heads if use_attention else 1,
            concat=False,
            dropout=dropout if use_attention else 0.0,
        )

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout)

        if readout == "attn":
            gate_hidden = max(hidden_channels // 2, 1)
            self.attn_readout = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_channels, gate_hidden),
                    nn.ReLU(),
                    nn.Linear(gate_hidden, 1),
                )
            )
            classifier_in = hidden_channels
        elif readout == "mean_max":
            self.attn_readout = None
            classifier_in = hidden_channels * 2
        elif readout == "mean_sum_max":
            self.attn_readout = None
            classifier_in = hidden_channels * 3
        elif readout in {"mean", "max", "sum"}:
            self.attn_readout = None
            classifier_in = hidden_channels
        else:
            raise ValueError(f"Unsupported readout: {readout}")

        self.classifier = nn.Linear(classifier_in, num_classes)

    def _build_hyperedge_attr(self, x, hyperedge_index):
        """
        Build hyperedge features by averaging node features inside each hyperedge.
        x: [num_nodes, feat_dim]
        hyperedge_index: [2, num_incidence]
            row 0 = node ids
            row 1 = hyperedge ids
        """
        if hyperedge_index.numel() == 0:
            return x.new_zeros((0, x.size(-1)))

        node_ids = hyperedge_index[0]
        hedge_ids = hyperedge_index[1]
        num_hyperedges = int(hedge_ids.max().item()) + 1

        hedge_attr = x.new_zeros((num_hyperedges, x.size(-1)))
        hedge_attr.index_add_(0, hedge_ids, x[node_ids])

        counts = x.new_zeros(num_hyperedges)
        counts.index_add_(0, hedge_ids, x.new_ones(hedge_ids.size(0)))
        hedge_attr = hedge_attr / counts.clamp_min(1).unsqueeze(-1)

        return hedge_attr

    def _apply_readout(self, x, batch):
        if self.readout_type == "mean":
            return global_mean_pool(x, batch)
        elif self.readout_type == "max":
            return global_max_pool(x, batch)
        elif self.readout_type == "sum":
            return global_add_pool(x, batch)
        elif self.readout_type == "mean_max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            return torch.cat([x_mean, x_max], dim=-1)
        elif self.readout_type == "mean_sum_max":
            x_mean = global_mean_pool(x, batch)
            x_sum = global_add_pool(x, batch)
            x_max = global_max_pool(x, batch)
            return torch.cat([x_mean, x_sum, x_max], dim=-1)
        elif self.readout_type == "attn":
            return self.attn_readout(x, index=batch)
        else:
            raise ValueError(f"Unsupported readout: {self.readout_type}")

    def forward(self, x, hyperedge_index, batch, hyperedge_weight=None):
        # Layer 1
        hyperedge_attr = self._build_hyperedge_attr(x, hyperedge_index) if self.use_attention else None
        x = self.conv1(
            x,
            hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            hyperedge_attr=hyperedge_attr,
        )
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Layer 2
        hyperedge_attr = self._build_hyperedge_attr(x, hyperedge_index) if self.use_attention else None
        x = self.conv2(
            x,
            hyperedge_index,
            hyperedge_weight=hyperedge_weight,
            hyperedge_attr=hyperedge_attr,
        )
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Graph-level readout
        g = self._apply_readout(x, batch)
        return g
        # out = self.classifier(g)
        # return out


def train_hypergraph_model(model, train_loader, val_loader, optimizer, criterion, epochs,
 patience_score, device, output_dir, early_stop=True):
    train_losses, val_losses = [], []
    val_accuracies, test_accuracies = [], []
    early_stopper = EarlyStopping(patience=patience_score)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)
    graph_energies = {"train": [], "val": [], "test": []}

    for epoch in range(epochs):
        # ===== TRAIN =====
        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        epoch_energy = []

        for batch in train_loader:

            # batch_data = batch.to(device)

            # print("x.shape =", batch_data.x.shape)
            # print("hyperedge_index.shape =", batch_data.hyperedge_index.shape)
            # print("hyperedge_weight.shape =", batch_data.hyperedge_weight.shape)
            # print("max node idx =", batch_data.hyperedge_index[0].max().item())
            # print("max hyperedge idx =", batch_data.hyperedge_index[1].max().item())
            # print("num nodes =", batch_data.x.size(0))
            # print("num hyperedges =", batch_data.hyperedge_weight.size(0))
            batch = batch.to(device)
            optimizer.zero_grad()
            # out = model(batch.x, batch.edge_index, batch.batch)
            # print("model output...")
            out = model(
                x=batch.x,
                hyperedge_index=batch.hyperedge_index,
                batch=batch.batch,
                hyperedge_weight=batch.hyperedge_weight if hasattr(batch, "hyperedge_weight") else None,
            )
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = out.argmax(dim=1)
            correct += (preds == batch.y).sum().item()
            total_samples += batch.y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        # avg_train_energy = np.mean(epoch_energy)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss_epoch, val_correct, val_total = 0, 0, 0
        val_energy_epoch = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                # out = model(batch.x, batch.edge_index, batch.batch)
                out = model(
                    x=batch.x,
                    hyperedge_index=batch.hyperedge_index,
                    batch=batch.batch,
                    hyperedge_weight=batch.hyperedge_weight if hasattr(batch, "hyperedge_weight") else None,
                )
                loss = criterion(out, batch.y)
                preds = out.argmax(dim=1)

                val_loss_epoch += loss.item()
                val_correct += (preds == batch.y).sum().item()
                val_total += batch.y.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        avg_val_loss = val_loss_epoch / len(val_loader)
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")
        # avg_val_energy = np.mean(val_energy_epoch)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(avg_val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch+1:03d}/{epochs}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {correct/total_samples:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if early_stop and early_stopper(val_f1, model, output_dir, epoch=epoch+1):
            break
    # print(f"Training stopped early at epoch {early_stopper.stop_epoch}.")

    return train_losses, val_losses, val_accuracies



def calculate_metrics_hypergraph(model, test_loader, device, full_metric=False):
    model.to(device)
    
    model.eval()

    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # outputs = model(data.x, data.edge_index, data.batch)

            outputs = model(
                x=data.x,
                hyperedge_index=data.hyperedge_index,
                batch=data.batch,
                hyperedge_weight=data.hyperedge_weight if hasattr(data, "hyperedge_weight") else None,
            )
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            labels = data.y.cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    num_classes = all_probs.shape[1] if all_probs.ndim > 1 else 1
    avg_type = 'binary' if num_classes == 2 else 'macro'

    accuracy = accuracy_score(all_labels, all_preds)

    metrics = {'num_classes': num_classes, 'accuracy': round(accuracy, 4)}

    if full_metric:
        precision = precision_score(all_labels, all_preds, average=avg_type, zero_division=0)
        recall = recall_score(all_labels, all_preds, average=avg_type, zero_division=0)
        f1 = f1_score(all_labels, all_preds, average=avg_type, zero_division=0)

        # --- AUC, FPR, TPR ---
        auc, fpr, tpr = None, None, None
        try:
            if num_classes == 2:
                auc = round(roc_auc_score(all_labels, all_probs[:, 1]), 4)
                fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
            else:
                auc_per_class, fpr_list, tpr_list = [], [], []
                for i in range(num_classes):
                    true_binary = (all_labels == i).astype(int)
                    prob = all_probs[:, i]
                    try:
                        auc_i = roc_auc_score(true_binary, prob)
                        fpr_i, tpr_i, _ = roc_curve(true_binary, prob)
                    except ValueError:
                        auc_i, fpr_i, tpr_i = np.nan, None, None
                    auc_per_class.append(round(auc_i, 4))
                    fpr_list.append(fpr_i)
                    tpr_list.append(tpr_i)
                auc, fpr, tpr = auc_per_class, fpr_list, tpr_list
        except ValueError:
            auc, fpr, tpr = float('nan'), None, None

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

        metrics.update({
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr,
            'confusion_matrix': cm
        })
    return metrics


def get_hypergraph_predictions(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    probabilities = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # outputs = model(data.x, data.edge_index, data.batch)  # Adjust for your model

            outputs = model(
                x=data.x,
                hyperedge_index=data.hyperedge_index,
                batch=data.batch,
                hyperedge_weight=data.hyperedge_weight if hasattr(data, "hyperedge_weight") else None,
            )
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
            pred = outputs.argmax(dim=1)
    return np.array(predictions), np.array(probabilities)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


if __name__ == "__main__":
    import config
    seed = config.SEED
    epochs = config.EPOCHS
    iterate = config.ITERATE
    batchsize = config.BATCHSIZE
    k = config.KFOLD
    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    fixed_edges = config.MONOFIXEDGES
    channel_names = config.MONO_CHANNELS
    feature_dim_dict = config.FEATURE_DIM_DICT
    region_to_channels = config.region_to_channels
    region_hyperedges = config.region_hyperedges
    val_ratio = config.VAL_RATIO
    g = make_torch_generator(seed)
    set_global_seed(seed)



    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--saved_subject_dirs", type=str, required=True, help="Path to the input directory")
    parser.add_argument("--class_set", type=str, default="all3", help="Name of the model to use")
    parser.add_argument("--batchnorm", type=str2bool, default=True, help="backnorm")
    parser.add_argument("--att", type=str2bool, default=True, help="use attention")
    parser.add_argument("--use_fake_label", type=str2bool, default=False, help="use_fake_label")
    parser.add_argument("--patience_score", type=int, default=50, help="patience_score")
    parser.add_argument("--dim", type=int, default=64, help="dim")
    parser.add_argument("--lr", type=float, default=0.01, help="lr")
    parser.add_argument("--drop_out", type=float, default=0.3, help="drop_out")
    parser.add_argument("--hyperedge_weight", type=str, default="mean_abs_adj", help="hyperedge_weight")
    parser.add_argument("--fake_score_method", type=str, default="within_hyperedge_similarity", help="fake_score_method")

    args = parser.parse_args()
    class_set = args.class_set
    saved_subject_dir = args.saved_subject_dirs
    att = args.att
    patience_score = args.patience_score #50
    lr = args.lr #0.0017
    drop_out = args.drop_out #0.1
    # dim = args.dim #256
    batchnorm = args.batchnorm #'sum'
    heads = 4
    # drop_out = 0.3
    dim = 64
    attention_mode="node"
    readout="mean_max"
    use_fake_label = args.use_fake_label
    print(f"--use_fake_label {use_fake_label} --att {att} --batchnorm {batchnorm}")

    if use_fake_label == True:
        fake_label_method = "fake_segment_label"
    else: 
        fake_label_method = "real_label"

    standardize_features=True
    hyperedge_weight =args.hyperedge_weight
    fake_score_method = args.fake_score_method

    model_name = "hypergraph"
    weightdecay = 3e-5 # args. weightdecay #
    # patience_score = 50
    num_classes, class_labels, class_names = get_class(class_set, dataset)
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    save_path = '/home/anphan/Documents/EEG_Project/AHEAP_data/result_hypergraph'
    os.makedirs(save_path,exist_ok = True)
    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    last_part = os.path.basename(saved_subject_dir)
    parts = last_part.split('_')

    try:
        node_features = parts[1]
        weight_method = parts[2:]
        _ = get_feature_dim_from_string(feature_dim_dict, node_features)

    except ValueError:
        node_features = parts[0]
        weight_method = parts[1:3]
    

    feat, used_features = get_feature_dim_from_string(feature_dim_dict, node_features)
    folder_name = f"{timestamp}_{model_name}_{last_part}_{fake_label_method}"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")

    data_processed_path = os.path.join(saved_subject_dir, "data_processed")   
    all_data_path = f"{data_processed_path}/master_graph_data.pt"

    if not os.path.exists(all_data_path):
        raise FileNotFoundError(f"Missing: {all_data_path}")
    if not os.path.exists(all_data_path):
        print(f"Skipping: {all_data_path} not found.")
        sys.exit(1) 
    print("File found! Processing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    with open(log_path, "w") as f:
        f.write(f"{saved_subject_dir}\n")
        f.write(f"Dataset: {dataset} -- num_classes = {num_classes} -- class_labels = {class_labels} -- class_names = {class_names}\n")
        f.write(f"Node Feature(s): {feat} | Number = {used_features}\n")
        f.write(f"Model {model_name}, att {att} \n")
        f.write(f"hyperedge_weight = {hyperedge_weight}, dropout = {drop_out}, hidden_channels = {dim}, attention_mode = {attention_mode}, readout = {readout}\n")
        f.write(f"iterate = {iterate}, batchsize = {batchsize}, epochs = {epochs} \n")
        f.write(f"patience_score = {patience_score}, lr = {lr}, weightdecay= {weightdecay} \n")
        f.write(f"{region_hyperedges}\n")
        f.write(f"validataion split by subject , ratio = {val_ratio}\n")
        f.write(f"fake risk score for hypergraph, {fake_score_method}\n")

    result_all = []
    for m in range(iterate):

        all_folds = balanced_kfold_split(sub_id_list, labels, seed, k)
        cm_sub_soft = []
        hypergraphs = build_hypergraphs_from_master_region_topology(
            master_path=all_data_path,
            channel_names=channel_names,
            region_to_channels=region_hyperedges,
            hyperedge_weight_mode=hyperedge_weight,
            subject_ids=None,
            label_key="class_id",
            standardize_features=True,
            corruption_mode=None,
            # hyperedge_weight_mode="mean_abs_adj",
        )
        if use_fake_label:
            hypergraphs, fake_scores, thresholds = assign_fake_labels_to_hypergraphs(
                            hypergraphs,
                            # fake_label_type="nonlinear_edge",
                            fake_label_type=fake_score_method,
                            update_y=True,
                            num_fake_classes=num_classes
                            )

            hypergraphs, subject_label_map, _, thresholds = assign_subject_fake_labels_to_segments(
                hypergraphs,
                subject_ids=None,
                source="fake_score",
                aggregate_mode="mean",
                num_fake_classes=num_classes,
                top_k_ratio=0.3,
                segment_label_attr="fake_segment_label",
                subject_label_attr="fake_subject_label",
                subject_score_attr="fake_subject_score",
                subject_threshold_attr="fake_subject_thresholds",
                update_y=True,
                )

            labels = [subject_label_map[sid] for sid in sub_id_list]
            print("segment fake thresholds:", fake_scores)
            print("subject fake thresholds:", thresholds)
            print("class counts:", np.bincount(labels, minlength=3))

        for i, test_fold in enumerate(all_folds):
            test_subjects = all_folds[i]
            test_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in test_subjects]
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]

            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            train_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in train_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            subject_label_map = dict(zip(train_subjects, train_labels))
                        
            new_train_subjects, val_subjects = stratified_split_subjects(
                train_subjects, subject_label_map, val_ratio=val_ratio, seed=seed
            )
   
            print(f"# Train_subjects = {len(new_train_subjects)} | # Validation subjects = {len(val_subjects)}")
                        

            train_graphs = [g for g in hypergraphs if g.subject_id in new_train_subjects]
            val_graphs   = [g for g in hypergraphs if g.subject_id in val_subjects]
            test_graphs  = [g for g in hypergraphs if g.subject_id in set(test_subjects)]

            gr = train_graphs[0]
            print(gr.x.shape)
            print(gr.hyperedge_index.shape)
            print(gr.hyperedge_weight.shape)
            print(gr.hyperedge_index[1].max().item())
            train_loader = DataLoader(
                train_graphs,
                batch_size=batchsize,
                shuffle=True,
                drop_last=True,          # ok for train
                num_workers=0,           # simplest reproducible option
                generator=g,
                worker_init_fn=seed_worker,
            )

            val_loader = DataLoader(
                val_graphs,
                batch_size=batchsize,
                shuffle=False,           # important
                drop_last=False,         # important
                num_workers=0,
            )

            test_loader = DataLoader(
                test_graphs,
                batch_size=batchsize,
                shuffle=False,           # important
                drop_last=False,         # important
                num_workers=0,
            )
            with open(log_path, "a") as f:
                f.write(f"------------------------------------------------------------------\n")
                f.write(f"\nIteration {m+1} - Fold {i + 1}/{k}, Test_subjects: {test_subjects}\n")
                f.write(f"Training model {model_name}\n")

            model = EEGHypergraphNet(
                in_channels=feat,
                hidden_channels=64,
                num_classes=num_classes,
                dropout=drop_out,
                use_batchnorm=batchnorm,
                use_attention=att,
                attention_heads=heads,
                attention_mode=attention_mode,
                readout=readout,
            )
            # model = EEGHypergraphNet(
            #     in_channels=feat,
            #     hidden_channels=dim,
            #     num_classes=num_classes,
            #     dropout=drop_out,
            #     backnorm=backnorm
            # )
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weightdecay)
            criterion = torch.nn.CrossEntropyLoss()

            t3 = time.time()
            print("training...")
            train_losses, val_losses, val_accuracies = train_hypergraph_model(model, train_loader, val_loader, optimizer, criterion, epochs, patience_score, device, output_dir, early_stop=True)
            
            t4 = time.time()
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_train_losses.npy"), train_losses)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_val_losses.npy"), val_losses)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_val_accuracies.npy"), val_accuracies)
            with open(log_path, "a") as f:
                f.write(f"Time for Training model = {t4 - t3} seconds\n")
            
            model = load_best_model(model, output_dir, device)
            segment_metrics = calculate_metrics_hypergraph(model, test_loader, device, full_metric=True)

            final_predictions = []
            sub_prob = []
            final_predictions_soft = []

            for sub_id, sub_label in zip(test_subjects, test_labels):
                sub_dataset   = [g for g in hypergraphs if g.subject_id in [sub_id]]

                sub_loader = DataLoader(sub_dataset, batch_size=batchsize, shuffle=False)
                with open(log_path, "a") as f:
                    f.write(f"{sub_id} -- True Label: {sub_label}, Total graphs: {len(sub_dataset)}\n")
                
                graph_preds, graph_prob = get_hypergraph_predictions(model, sub_loader, device)

                class_counts = {cls: int((graph_preds == cls).sum()) for cls in class_labels}
                sub_prediction_majority = np.bincount(graph_preds, minlength=num_classes).argmax()

                avg_prob_all = np.mean(graph_prob, axis=0)  # shape: [num_classes]
                sub_prediction_soft = np.argmax(avg_prob_all)  # argmax handles N>2 automatically

                final_predictions.append(sub_prediction_majority)
                final_predictions_soft.append(sub_prediction_soft)
                sub_prob.append(avg_prob_all)

                avg_prob_str = ", ".join([f"Class {cls} -> {avg_prob_all[cls]:.4f}" for cls in range(num_classes)])
                with open(log_path, "a") as f:
                    f.write(f"- Graph Prediction : {class_counts} --> Majority Voting: {sub_prediction_majority}\n")
                    f.write(f"---- Mean Prob: {avg_prob_str} --> Average Voting: {sub_prediction_soft}\n")
            subject_metrics_soft = calculate_metrics(
                test_labels, 
                final_predictions_soft, 
                class_labels, 
                num_classes, 
                predicted_probabilities=sub_prob
            )

            cm_soft = np.array(subject_metrics_soft["confusion_matrix"])  # convert to numpy array
            cm_sub_soft.append(cm_soft)

            with open(log_path, "a") as f:
                f.write("\nSegment-level results:\n")
                # print(segment_metrics)
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'confusion_matrix']:
                    f.write(f"---{key}: {segment_metrics[key]}\n")

                f.write(f"Subject-level (soft-voting) results:\n")
                for key in ['accuracy', 'precision', 'recall', 'f1_score', 'auc', 'confusion_matrix']:
                    f.write(f"---{key}: {subject_metrics_soft[key]}\n")


            result_all.append((model_name, batchnorm, patience_score, lr, drop_out, dim, seed, m+1, i+1, test_subjects, "segment", segment_metrics["accuracy"], segment_metrics["precision"],\
                            segment_metrics["recall"], segment_metrics["f1_score"], segment_metrics["auc"], segment_metrics["confusion_matrix"]
                            ))

            result_all.append((model_name, batchnorm, patience_score, lr, drop_out, dim, seed, m+1, i+1, test_subjects, "subject (soft)", subject_metrics_soft["accuracy"], subject_metrics_soft["precision"],\
                            subject_metrics_soft["recall"], subject_metrics_soft["f1_score"], subject_metrics_soft["auc"], \
                            subject_metrics_soft["confusion_matrix"]))

            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_seg_fpr.npy"), np.array(segment_metrics["fpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_seg_tpr.npy"), np.array(segment_metrics["tpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_seg_auc.npy"), np.array(segment_metrics["auc"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_sub_fpr.npy"), np.array(subject_metrics_soft["fpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_sub_tpr.npy"), np.array(subject_metrics_soft["tpr"], dtype=object), allow_pickle=True)
            np.save(os.path.join(output_dir, f"iter{m+1}_fold{i+1}_sub_auc.npy"), np.array(subject_metrics_soft["auc"], dtype=object), allow_pickle=True)

        total_cm_sub_soft = np.sum(cm_sub_soft, axis=0)
        with open(log_path, "a") as f:
            f.write(f"Total confusion matrix:\n {total_cm_sub_soft}\n")
        plot_confusion_matrix(
                total_cm_sub_soft,
                class_names=class_names,
                iter_id = m,
                save_path = output_dir,
                show_normed=True
            )
    with open(log_path, "a") as f:
        f.write(f"Model Architecture: {model}\n")
        f.write(f"Number of parameters: {sum(p.numel() for p in model.parameters())}\n")
            
    voting_df = pd.DataFrame(result_all, columns=['Model', 'batchnorm', 'patience_score', 'lr', 'drop_out', 'dim', 'seed', 'Iteration', 'Fold', 'TestSubjects', 'Level', 'Accuracy', 'Precision', 'Recall' ,'F1-score', 'AUC', 'ConfusionMatrix'])
    voting_df.to_csv(os.path.join(output_dir, f"{timestamp}_{last_part}_{model_name}_{class_set}.csv"), index = False)
    print("Saved result in folder:", folder_name)
