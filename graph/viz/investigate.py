import os
import numpy as np
import torch
from collections import defaultdict, Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from lib import *
from model import *
from data_utils import *
from graph_utils import *
from utils_all import *
# ----------------------------
# Edge builders (choose one)
# ----------------------------

def edge_builder_fixed_from_pairs(pairs, num_nodes=19):
    """
    pairs: list of (u,v) undirected edges
    returns edge_index [2, 2E] (both directions)
    """
    src = []
    dst = []
    for u, v in pairs:
        src += [u, v]
        dst += [v, u]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return edge_index

def edge_builder_mst(adj):
    """Per-graph maximum spanning tree from adj [N,N] -> edge_index, edge_attr"""
    A = adj.detach().cpu().numpy()
    N = A.shape[0]
    np.fill_diagonal(A, -np.inf)

    in_tree = np.zeros(N, dtype=bool)
    in_tree[0] = True
    best_w = np.full(N, -np.inf, dtype=float)
    best_parent = np.full(N, -1, dtype=int)
    for v in range(1, N):
        best_w[v] = A[0, v]
        best_parent[v] = 0

    edges = []
    weights = []
    for _ in range(N - 1):
        cand = np.where(~in_tree)[0]
        v = cand[np.argmax(best_w[cand])]
        u = best_parent[v]
        w = best_w[v]
        if u < 0 or not np.isfinite(w):
            u = int(np.where(in_tree)[0][0])
            w = 0.0
        edges.append((u, v))
        weights.append(float(w))
        in_tree[v] = True
        for t in np.where(~in_tree)[0]:
            if A[v, t] > best_w[t]:
                best_w[t] = A[v, t]
                best_parent[t] = v

    src = np.array([u for u, v in edges] + [v for u, v in edges], dtype=np.int64)
    dst = np.array([v for u, v in edges] + [u for u, v in edges], dtype=np.int64)
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    w = np.array(weights, dtype=np.float32)
    edge_attr = torch.tensor(np.concatenate([w, w]).reshape(-1, 1), dtype=torch.float32)
    return edge_index, edge_attr

def edge_builder_from_adj_fixed_edge_index(adj, edge_index):
    """Given per-graph adj and a fixed edge_index, return edge_attr aligned with it."""
    weights = adj[edge_index[0], edge_index[1]]
    return edge_index, weights.unsqueeze(1)


# ----------------------------
# Utilities for diagnostics
# ----------------------------

def _edge_set_undirected(edge_index):
    ei = edge_index.detach().cpu().numpy()
    s = set()
    for u, v in ei.T:
        u, v = int(u), int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        s.add((a, b))
    return s

def _jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)

def _group_graphs_by_subject(graphs):
    subj = defaultdict(list)
    for g in graphs:
        subj[g["subject_id"]].append(g)
    # sort by segment_id if exists
    for sid in subj:
        subj[sid].sort(key=lambda e: e.get("segment_id", 0))
    return subj

def _extract_graph_features(entry, edge_mode, edge_index_fixed=None, fixed_pairs=None):
    """
    Returns:
      x_flat: [19*F]
      x_mean: [F]
      e_vec : [E] (edge_attr flattened)  (may be empty if no edges)
      y     : int
      edge_index used (for topology comparisons)
    """
    x = entry["node_features"].detach().cpu()
    y = int(entry["class_id"])
    adj = entry["adj"]

    # node features
    x_flat = x.reshape(-1).numpy()
    x_mean = x.mean(dim=0).numpy()

    # edges
    if edge_mode == "mst":
        ei, ea = edge_builder_mst(adj)
    elif edge_mode == "fixed_pairs":
        ei = edge_builder_fixed_from_pairs(fixed_pairs, num_nodes=x.size(0))
        ei, ea = edge_builder_from_adj_fixed_edge_index(adj, ei)
    elif edge_mode == "fixed_edge_index":
        ei, ea = edge_builder_from_adj_fixed_edge_index(adj, edge_index_fixed)
    else:
        raise ValueError("edge_mode must be 'mst', 'fixed_pairs', or 'fixed_edge_index'")

    e_vec = ea.view(-1).detach().cpu().numpy() if ea is not None else np.array([], dtype=float)
    return x_flat, x_mean, e_vec, y, ei

def _subject_level_eval(X_seg, y_seg, sid_seg, X_test, y_test, sid_test):
    """
    Train on segment-level features but evaluate at SUBJECT-level by averaging probs over segments.
    """
    clf = LogisticRegression(
        max_iter=8000, n_jobs=-1, solver="saga", multi_class="auto",
        class_weight="balanced"
    )
    clf.fit(X_seg, y_seg)

    # test proba per segment
    proba = clf.predict_proba(X_test)

    # aggregate to subject
    subj_probs = defaultdict(list)
    subj_true = {}
    for p, y, sid in zip(proba, y_test, sid_test):
        subj_probs[sid].append(p)
        subj_true[sid] = y

    y_true_sub = []
    y_pred_sub = []
    for sid, plist in subj_probs.items():
        p_mean = np.mean(np.stack(plist, axis=0), axis=0)
        y_pred = int(np.argmax(p_mean))
        y_true_sub.append(int(subj_true[sid]))
        y_pred_sub.append(y_pred)

    f1 = f1_score(y_true_sub, y_pred_sub, average="macro")
    acc = accuracy_score(y_true_sub, y_pred_sub)
    return f1, acc, len(subj_probs)

def _topology_stability_within_subject(entries, edge_mode, edge_index_fixed=None, fixed_pairs=None, max_pairs_per_subject=30):
    """
    For each subject, compute Jaccard similarity of edge sets between consecutive segments.
    Returns mean, std, and number of comparisons.
    """
    subj = _group_graphs_by_subject(entries)
    scores = []
    for sid, segs in subj.items():
        if len(segs) < 2:
            continue
        # cap comparisons
        count = 0
        prev_set = None
        for e in segs:
            *_, ei = _extract_graph_features(e, edge_mode, edge_index_fixed, fixed_pairs)
            cur_set = _edge_set_undirected(ei)
            if prev_set is not None:
                scores.append(_jaccard(prev_set, cur_set))
                count += 1
                if count >= max_pairs_per_subject:
                    break
            prev_set = cur_set
    if len(scores) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)

def _edge_weight_stability_within_subject(entries, edge_mode, edge_index_fixed=None, fixed_pairs=None, max_pairs_per_subject=30):
    """
    Correlation of edge_attr vectors between consecutive segments within subject.
    Returns mean, std, n comparisons.
    """
    subj = _group_graphs_by_subject(entries)
    corrs = []
    for sid, segs in subj.items():
        if len(segs) < 2:
            continue
        count = 0
        prev_w = None
        for e in segs:
            *_, w, __, = None, None, None  # just to keep structure readable
            x_flat, x_mean, e_vec, y, ei = _extract_graph_features(e, edge_mode, edge_index_fixed, fixed_pairs)
            cur_w = e_vec
            if prev_w is not None and len(cur_w) == len(prev_w) and len(cur_w) > 1:
                c = np.corrcoef(prev_w, cur_w)[0, 1]
                if not np.isnan(c):
                    corrs.append(c)
                count += 1
                if count >= max_pairs_per_subject:
                    break
            prev_w = cur_w
    if len(corrs) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(corrs)), float(np.std(corrs)), len(corrs)


# ----------------------------
# Full report driver
# ----------------------------

def full_gnn_worthiness_report(
    master_path,
    all_folds,
    sub_id_list,
    labels,
    edge_mode="fixed_pairs",
    fixed_pairs=None,           # used if edge_mode == "fixed_pairs"
    edge_index_fixed=None,      # used if edge_mode == "fixed_edge_index"
    folds_to_run=None,          # e.g., [0,1,2] to test quickly
    verbose=True
):
    """
    Returns:
      per_fold: list of dicts
      summary: dict of averaged metrics
    """
    all_data = torch.load(master_path)
    # quick subject->label map from your lists
    subject_label_map = dict(zip(sub_id_list, labels))

    folds = list(range(len(all_folds))) if folds_to_run is None else folds_to_run
    per_fold = []

    for i in folds:
        test_subjects = set(all_folds[i])
        train_subjects = [sid for sid in sub_id_list if sid not in test_subjects]

        # filter entries by subject id
        train_entries = [e for e in all_data if e["subject_id"] in set(train_subjects)]
        test_entries  = [e for e in all_data if e["subject_id"] in test_subjects]

        # subject class counts
        train_counts = Counter(subject_label_map[sid] for sid in train_subjects if sid in subject_label_map)
        test_counts  = Counter(subject_label_map[sid] for sid in test_subjects  if sid in subject_label_map)

        # 1) structure varies? (topology stability)
        topo_mean, topo_std, topo_n = _topology_stability_within_subject(
            train_entries, edge_mode, edge_index_fixed, fixed_pairs
        )

        # 2) edge weights reliable?
        w_mean, w_std, w_n = _edge_weight_stability_within_subject(
            train_entries, edge_mode, edge_index_fixed, fixed_pairs
        )

        # 3) interactions drive labels? (subject-level baselines)
        # Build segment feature matrices
        Xn_flat_tr, Xn_mean_tr, Xe_tr, y_tr, sid_tr = [], [], [], [], []
        Xn_flat_te, Xn_mean_te, Xe_te, y_te, sid_te = [], [], [], [], []

        # Train segments
        for e in train_entries:
            x_flat, x_mean, e_vec, y, ei = _extract_graph_features(e, edge_mode, edge_index_fixed, fixed_pairs)
            Xn_flat_tr.append(x_flat)
            Xn_mean_tr.append(x_mean)
            Xe_tr.append(e_vec)
            y_tr.append(y)
            sid_tr.append(e["subject_id"])

        # Test segments
        for e in test_entries:
            x_flat, x_mean, e_vec, y, ei = _extract_graph_features(e, edge_mode, edge_index_fixed, fixed_pairs)
            Xn_flat_te.append(x_flat)
            Xn_mean_te.append(x_mean)
            Xe_te.append(e_vec)
            y_te.append(y)
            sid_te.append(e["subject_id"])

        Xn_flat_tr = np.asarray(Xn_flat_tr)
        Xn_mean_tr = np.asarray(Xn_mean_tr)
        Xe_tr = np.asarray(Xe_tr)
        y_tr = np.asarray(y_tr)

        Xn_flat_te = np.asarray(Xn_flat_te)
        Xn_mean_te = np.asarray(Xn_mean_te)
        Xe_te = np.asarray(Xe_te)
        y_te = np.asarray(y_te)

        # Baselines:
        # - nodes_flat (topography)
        # - nodes_mean (no topography)
        # - edges_only
        # - combined (nodes_flat + edges)
        f1_nodes_flat, acc_nodes_flat, nsub = _subject_level_eval(Xn_flat_tr, y_tr, sid_tr, Xn_flat_te, y_te, sid_te)
        f1_nodes_mean, acc_nodes_mean, _    = _subject_level_eval(Xn_mean_tr, y_tr, sid_tr, Xn_mean_te, y_te, sid_te)

        # edges-only can fail if Xe has tiny dimension or constant; handle safely
        if Xe_tr.shape[1] > 0 and np.std(Xe_tr) > 1e-12:
            f1_edges, acc_edges, _ = _subject_level_eval(Xe_tr, y_tr, sid_tr, Xe_te, y_te, sid_te)
        else:
            f1_edges, acc_edges = np.nan, np.nan

        X_comb_tr = np.concatenate([Xn_flat_tr, Xe_tr], axis=1) if Xe_tr.shape[1] > 0 else Xn_flat_tr
        X_comb_te = np.concatenate([Xn_flat_te, Xe_te], axis=1) if Xe_te.shape[1] > 0 else Xn_flat_te
        f1_comb, acc_comb, _ = _subject_level_eval(X_comb_tr, y_tr, sid_tr, X_comb_te, y_te, sid_te)

        row = {
            "fold": i,
            "train_subjects": len(train_subjects),
            "test_subjects": len(test_subjects),
            "train_class_counts": dict(train_counts),
            "test_class_counts": dict(test_counts),

            # Condition 1: structure varies meaningfully?
            "topo_jacc_mean_train": topo_mean,
            "topo_jacc_std_train": topo_std,
            "topo_jacc_n_pairs": topo_n,

            # Condition 2: edge weights reliable?
            "edge_corr_mean_train": w_mean,
            "edge_corr_std_train": w_std,
            "edge_corr_n_pairs": w_n,

            # Condition 3: interactions drive labels?
            "F1_nodes_flat": f1_nodes_flat,
            "F1_nodes_mean": f1_nodes_mean,
            "F1_edges_only": f1_edges,
            "F1_combined": f1_comb,
            "ACC_nodes_flat": acc_nodes_flat,
            "ACC_nodes_mean": acc_nodes_mean,
            "ACC_edges_only": acc_edges,
            "ACC_combined": acc_comb,
        }
        per_fold.append(row)

        if verbose:
            print(f"\n=== Fold {i} | edge_mode={edge_mode} ===")
            print("Train subj class counts:", train_counts)
            print("Test  subj class counts:", test_counts)
            print(f"Topology stability (Jaccard, train): mean={topo_mean:.3f} std={topo_std:.3f} (n={topo_n})")
            print(f"Edge weight stability (corr, train): mean={w_mean:.3f} std={w_std:.3f} (n={w_n})")
            print(f"Subject-level F1: nodes_flat={f1_nodes_flat:.3f} | nodes_mean={f1_nodes_mean:.3f} | "
                  f"edges_only={f1_edges:.3f} | combined={f1_comb:.3f}")

    # Summary
    def _avg(key):
        vals = [r[key] for r in per_fold if r[key] == r[key]]  # filter NaNs
        return float(np.mean(vals)) if vals else np.nan

    summary = {
        "master_path": master_path,
        "edge_mode": edge_mode,
        "folds_run": folds,
        "avg_topo_jacc_mean_train": _avg("topo_jacc_mean_train"),
        "avg_edge_corr_mean_train": _avg("edge_corr_mean_train"),
        "avg_F1_nodes_flat": _avg("F1_nodes_flat"),
        "avg_F1_nodes_mean": _avg("F1_nodes_mean"),
        "avg_F1_edges_only": _avg("F1_edges_only"),
        "avg_F1_combined": _avg("F1_combined"),
    }

    if verbose:
        print("\n=== SUMMARY ===")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

    return per_fold, summary


import os
import torch


def safe_full_gnn_worthiness_report(
    master_path,
    all_folds,
    sub_id_list,
    labels,
    edge_mode="fixed_pairs",
    fixed_pairs=None,
    edge_index_fixed=None,
    folds_to_run=None,
    verbose=True
):
    """
    Safe wrapper:
    - Checks file existence
    - Handles corrupted pt
    - Prevents crash
    """

    # ---------- Check existence ----------
    if not os.path.exists(master_path):
        print(f"[WARN] Master file not found: {master_path}")
        return None, None

    # ---------- Try loading ----------
    try:
        all_data = torch.load(master_path, map_location="cpu")
    except Exception as e:
        print(f"[ERROR] Failed to load {master_path}")
        print("Reason:", repr(e))
        return None, None

    # ---------- Check basic structure ----------
    if not isinstance(all_data, list):
        print(f"[ERROR] {master_path} is not a list. Got:", type(all_data))
        return None, None

    if len(all_data) == 0:
        print(f"[WARN] Empty dataset: {master_path}")
        return None, None

    # ---------- Check required keys ----------
    required = {"subject_id", "class_id", "node_features", "adj"}
    bad_entries = 0

    for i, e in enumerate(all_data[:20]):  # only check first 20
        if not isinstance(e, dict):
            bad_entries += 1
            continue

        missing = required - set(e.keys())
        if missing:
            print(f"[ERROR] Entry {i} missing keys: {missing}")
            bad_entries += 1

    if bad_entries > 0:
        print(f"[ERROR] {bad_entries} malformed entries in {master_path}")
        return None, None

    if verbose:
        print(f"[OK] Loaded {len(all_data)} entries from {master_path}")

    # ---------- Run real report ----------
    try:
        return full_gnn_worthiness_report(
            master_path=master_path,
            all_folds=all_folds,
            sub_id_list=sub_id_list,
            labels=labels,
            edge_mode=edge_mode,
            fixed_pairs=fixed_pairs,
            edge_index_fixed=edge_index_fixed,
            folds_to_run=folds_to_run,
            verbose=verbose
        )

    except Exception as e:
        print(f"[ERROR] Analysis failed for {master_path}")
        print("Reason:", repr(e))
        return None, None

if __name__ == "__main__":

    # fixed_pairs = [(0,1),(0,2), ...]  # your chosen fixed edges (undirected)

    # per_fold, summary = full_gnn_worthiness_report(
    #     master_path=".../master_graph_data_update_substd.pt",
    #     all_folds=all_folds,
    #     sub_id_list=sub_id_list,
    #     labels=labels,
    #     edge_mode="fixed_pairs",
    #     fixed_pairs=fixed_pairs,
    #     folds_to_run=[0,1,2],   # quick test first
    #     verbose=True
    # )

    # per_fold, summary = full_gnn_worthiness_report(
    #     master_path=".../master_graph_data_update_substd.pt",
    #     all_folds=all_folds,
    #     sub_id_list=sub_id_list,
    #     labels=labels,
    #     edge_mode="mst",
    #     folds_to_run=[0,1,2],
    #     verbose=True
    # )

    masters = [
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_plv_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_pli_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_coherence_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_corr_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_coherence_alpha',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_plv_alpha',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_pli_alpha',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/bi23_rbphjorth_corr_alpha',

    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_pli_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_plv_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_corr_None',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_corr_alpha',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_plv_alpha',
    '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_pli_alpha'
    ]

    data_dir = '/mnt/data/anphan/derivatives'
    tsv_path = '/home/anphan/Documents/EEG_Project/participants.tsv'
    class_set ="all3" 

    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    all_folds = balanced_kfold_split(sub_id_list, labels, 15, 10)

    # all_summaries = []
    # for mp in masters:
    #     _, summ = full_gnn_worthiness_report(
    #         master_path=os.path.join(mp, "data_processed_update/master_graph_data_update_substd.pt"),
    #         all_folds=all_folds,
    #         sub_id_list=sub_id_list,
    #         labels=labels,
    #         edge_mode="mst",
    #         # fixed_pairs=fixed_pairs,
    #         folds_to_run=[0,1,2],
    #         verbose=False
    #     )
    #     summ["name"] = os.path.basename(os.path.dirname(mp))  # or your own tag
    #     all_summaries.append(summ)

    # for s in all_summaries:
    #     print(s["name"], "F1_edges_only", s["avg_F1_edges_only"], "F1_combined", s["avg_F1_combined"])


    results = []

    for mp in masters:

        print("\nProcessing:", mp)

        per_fold, summary = safe_full_gnn_worthiness_report(
            master_path=os.path.join(mp, "data_processed/master_graph_data_update_substd.pt"),
            all_folds=all_folds,
            sub_id_list=sub_id_list,
            labels=labels,
            edge_mode="mst",
            # fixed_pairs=fixed_pairs,
            folds_to_run=[0,1,2],
            verbose=True
        )

        if summary is None:
            print("[SKIP]", mp)
            continue

        results.append(summary)