import torch
import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection
import argparse

def plot_matrix(mat, title, channel_names, save_name):
    plt.figure(figsize=(8, 6))
    plt.imshow(mat, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(channel_names)), channel_names, rotation=90)
    plt.yticks(range(len(channel_names)), channel_names)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close() 
    # plt.show()
def get_within_between_values(adj, region_indices):
    within_vals = []
    between_vals = []

    region_names = list(region_indices.keys())

    for i, r1 in enumerate(region_names):
        idx1 = region_indices[r1]

        for a in range(len(idx1)):
            for b in range(a + 1, len(idx1)):
                within_vals.append(adj[idx1[a], idx1[b]])

        for j in range(i + 1, len(region_names)):
            r2 = region_names[j]
            idx2 = region_indices[r2]
            for a in idx1:
                for b in idx2:
                    between_vals.append(adj[a, b])

    return np.array(within_vals), np.array(between_vals)

def mean_adj_by_class(records):
    classes = sorted(set(int(r["class_id"]) for r in records))
    out = {}
    for c in classes:
        mats = [r["adj"] for r in records if int(r["class_id"]) == c]
        out[c] = {
            "mean": np.mean(mats, axis=0),
            "std": np.std(mats, axis=0),
            "n_segments": len(mats)
        }
    return out

def region_connectivity_matrix(adj, region_indices, region_names):
    R = len(region_names)
    out = np.zeros((R, R), dtype=float)

    for i, r1 in enumerate(region_names):
        idx1 = region_indices[r1]
        for j, r2 in enumerate(region_names):
            idx2 = region_indices[r2]

            vals = []
            for a in idx1:
                for b in idx2:
                    if a == b:
                        continue
                    vals.append(adj[a, b])

            out[i, j] = np.mean(vals) if len(vals) > 0 else np.nan
    return out

def average_features_by_region(node_features, region_indices, region_names):
    out = []
    for r in region_names:
        idx = region_indices[r]
        out.append(node_features[idx].mean(axis=0))
    return np.stack(out, axis=0)


def mean_node_features_by_class(subject_summaries, class_id):
    mats = [s["mean_node_features"] for s in subject_summaries if s["class_id"] == class_id]
    return np.mean(mats, axis=0)

def channel_feature_similarity(node_feature_matrix):
    return np.corrcoef(node_feature_matrix)

# def region_connectivity_matrix(adj, region_indices, region_names):
#     R = len(region_names)
#     out = np.full((R, R), np.nan, dtype=float)

#     for i, r1 in enumerate(region_names):
#         idx1 = region_indices[r1]
#         for j, r2 in enumerate(region_names):
#             idx2 = region_indices[r2]

#             vals = []
#             for a in idx1:
#                 for b in idx2:
#                     if i == j and a == b:
#                         continue
#                     vals.append(adj[a, b])

#             if len(vals) > 0:
#                 out[i, j] = np.mean(vals)

#     return out

# def average_features_by_region(node_features, region_indices, region_names):
#     region_feats = []
#     for r in region_names:
#         idx = region_indices[r]
#         region_feats.append(node_features[idx].mean(axis=0))
#     return np.stack(region_feats, axis=0)   # [R, F]


def safe_kruskal(groups):
    """
    groups: list of lists/arrays, one per class
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]

    if any(len(g) == 0 for g in groups):
        return np.nan, np.nan

    # if all values identical across all groups, kruskal can fail or be meaningless
    all_vals = np.concatenate(groups)
    if np.allclose(all_vals, all_vals[0]):
        return 0.0, 1.0

    stat, p = kruskal(*groups)
    return stat, p

def pairwise_mannwhitney(values_by_class, class_pairs):
    results = {}
    for c1, c2 in class_pairs:
        x = np.asarray(values_by_class[c1], dtype=float)
        y = np.asarray(values_by_class[c2], dtype=float)

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        if len(x) == 0 or len(y) == 0:
            results[(c1, c2)] = (np.nan, np.nan)
            continue

        if np.allclose(np.concatenate([x, y]), np.concatenate([x, y])[0]):
            results[(c1, c2)] = (0.0, 1.0)
            continue

        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        results[(c1, c2)] = (stat, p)

    return results


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--pt_name", type=str, required=False, help="pt_name")
    
    args = parser.parse_args()
    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/"
    pt_name = args.pt_name
    pt_file = "data_processed/master_graph_data.pt"
    pt_path = os.path.join(root_path, pt_name, pt_file) #"/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_coherence_alpha/data_processed/master_graph_data.pt"
    output_dir = os.path.join(root_path, pt_name, "analysis_EDA") #/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_coherence_alpha/analysis_EDA"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(pt_path):
        # Manually trigger the error to jump to the 'except' block
        raise FileNotFoundError(f"Missing: {pt_path}")
    if not os.path.exists(pt_path):
        print(f"Skipping: {pt_path} not found.")
        # Exit with code 1 to signal the Bash script
        sys.exit(1) 

    # Continue with your logic if the file exists
    print("File found! Processing...")
    data = torch.load(pt_path, map_location="cpu")


    print(type(data))
    print(len(data))
    print(data[0].keys())
    print(data[0]['adj'].shape)
    print(data[0]['node_features'].shape)

    records = []
    for item in data:
        records.append({
            "subject_id": item["subject_id"],
            "class_id": int(item["class_id"]),
            "segment_id": int(item["segment_id"]),
            "adj": item["adj"].cpu().numpy() if torch.is_tensor(item["adj"]) else np.array(item["adj"]),
            "node_features": item["node_features"].cpu().numpy() if torch.is_tensor(item["node_features"]) else np.array(item["node_features"]),
            "start_sample": item["start_sample"]
        })

    N = records[0]["adj"].shape[0]
    F = records[0]["node_features"].shape[1]

    print("num segments:", len(records))
    print("num channels:", N)
    print("num features:", F)


    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

    region_map = {
        "Fp1": "frontal", "Fp2": "frontal",
        "F7": "frontal", "F3": "frontal", "Fz": "frontal", "F4": "frontal", "F8": "frontal",
        "C3": "central", "Cz": "central", "C4": "central",
        "T3": "temporal", "T4": "temporal", "T5": "temporal", "T6": "temporal",
        "P3": "parietal", "Pz": "parietal", "P4": "parietal",
        "O1": "occipital", "O2": "occipital"
    }


    region_indices = defaultdict(list)
    for i, ch in enumerate(channel_names):
        region_indices[region_map[ch]].append(i)

    print(region_indices)
    adj_summary = mean_adj_by_class(records)
    for c in adj_summary:
        plot_matrix(adj_summary[c]["mean"], f"Mean connectivity - Class {c}", channel_names, os.path.join(output_dir, f"adj_summary_channel_{c}.png"))


    plot_matrix(adj_summary[1]["mean"] - adj_summary[0]["mean"], "Class 1 - Class 0", channel_names, os.path.join(output_dir, f"adj_summary_class10.png"))
    plot_matrix(adj_summary[2]["mean"] - adj_summary[0]["mean"], "Class 2 - Class 0", channel_names, os.path.join(output_dir, f"adj_summary_class20.png"))
    plot_matrix(adj_summary[2]["mean"] - adj_summary[1]["mean"], "Class 2 - Class 1", channel_names, os.path.join(output_dir, f"adj_summary_class21.png"))



    subject_group = defaultdict(list)
    for r in records:
        subject_group[r["subject_id"]].append(r)

    subject_summaries = []

    for sid, items in subject_group.items():
        class_id = int(items[0]["class_id"])
        mean_adj = np.mean([x["adj"] for x in items], axis=0)
        mean_node_features = np.mean([x["node_features"] for x in items], axis=0)

        subject_summaries.append({
            "subject_id": sid,
            "class_id": class_id,
            "mean_adj": mean_adj,
            "mean_node_features": mean_node_features
        })


    N = subject_summaries[0]["mean_adj"].shape[0]
    pvals = np.ones((N, N))
    fvals = np.zeros((N, N))

    classes = sorted(set(s["class_id"] for s in subject_summaries))

    for i in range(N):
        for j in range(N):
            groups = [
                [s["mean_adj"][i, j] for s in subject_summaries if s["class_id"] == c]
                for c in classes
            ]
            stat, p = f_oneway(*groups)
            fvals[i, j] = stat
            pvals[i, j] = p


    mask = ~np.eye(N, dtype=bool)
    p_flat = pvals[mask]
    rej, p_corr = fdrcorrection(p_flat, alpha=0.05)

    sig_mat = np.zeros((N, N), dtype=bool)
    sig_mat[mask] = rej

    plot_matrix(sig_mat.astype(float), "Significant channel pairs across 3 classes", channel_names, os.path.join(output_dir, f"sig_mat.png"))

    region_names = list(region_indices.keys())

    for s in subject_summaries:
        s["region_adj"] = region_connectivity_matrix(s["mean_adj"], region_indices, region_names)
    
    R = len(region_names)
    region_pvals = np.ones((R, R))

    for i in range(R):
        for j in range(R):
            groups = [
                [s["region_adj"][i, j] for s in subject_summaries if s["class_id"] == c]
                for c in classes
            ]
            stat, p = kruskal(*groups)   # or f_oneway(*groups)
            region_pvals[i, j] = p
    within_subject = []

    for s in subject_summaries:
        w, b = get_within_between_values(s["mean_adj"], region_indices)
        within_subject.append({
            "subject_id": s["subject_id"],
            "class_id": s["class_id"],
            "within_mean": np.mean(w),
            "between_mean": np.mean(b)
        })

    groups_within = [
        [x["within_mean"] for x in within_subject if x["class_id"] == c]
        for c in classes
    ]

    stat, p = kruskal(*groups_within)   # or f_oneway(*groups_within)
    print("Within-region difference across 3 classes:", stat, p)
    groups_between = [
        [x["between_mean"] for x in within_subject if x["class_id"] == c]
        for c in classes
    ]

    stat, p = kruskal(*groups_between)
    print("Between-region difference across 3 classes:", stat, p)


    for s in subject_summaries:
        s["region_features"] = average_features_by_region(
            s["mean_node_features"], region_indices, region_names
        )
    F = subject_summaries[0]["region_features"].shape[1]

    feature_region_pvals = np.ones((R, F))

    for r in range(R):
        for f in range(F):
            groups = [
                [s["region_features"][r, f] for s in subject_summaries if s["class_id"] == c]
                for c in classes
            ]
            stat, p = kruskal(*groups)   # or f_oneway(*groups)
            feature_region_pvals[r, f] = p

    num_features = subject_summaries[0]["region_features"].shape[1]

    for r_idx, region_name in enumerate(region_names):
        for f_idx in range(num_features):
            rows = []


            for s in subject_summaries:
                rows.append({
                    "subject_id": s["subject_id"],
                    "class_id": s["class_id"],
                    "value": s["region_features"][r_idx, f_idx]
                })

            df = pd.DataFrame(rows)

            data_plot = [df[df["class_id"] == c]["value"].values for c in classes]

            plt.figure(figsize=(6, 4))
            plt.boxplot(data_plot, tick_labels=[f"Class {c}" for c in classes])
            plt.title(f"{region_name} - feature {f_idx}")
            plt.ylabel("Feature value")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir,f"boxplot_{region_name}_feature{f_idx}.png"), dpi=300)
            plt.close() 
    # plt.show()

    for c in classes:
        mean_nf = mean_node_features_by_class(subject_summaries, c)
        sim_nf = channel_feature_similarity(mean_nf)
        plot_matrix(sim_nf, f"Channel similarity from node features - Class {c}", channel_names, os.path.join(output_dir, f"sim_nf_channel_{c}.png"))


    class_pairs = [(classes[i], classes[j]) for i in range(len(classes)) for j in range(i+1, len(classes))]

    region_pair_rows = []

    for i, r1 in enumerate(region_names):
        for j, r2 in enumerate(region_names):
            values_by_class = {}
            for c in classes:
                vals = [
                    s["region_adj"][i, j]
                    for s in subject_summaries
                    if s["class_id"] == c
                ]
                values_by_class[c] = vals

            groups = [values_by_class[c] for c in classes]
            stat, p = safe_kruskal(groups)

            row = {
                "region_1": r1,
                "region_2": r2,
                "kw_stat": stat,
                "kw_p": p,
            }

            for c in classes:
                arr = np.asarray(values_by_class[c], dtype=float)
                row[f"class_{c}_mean"] = np.nanmean(arr)
                row[f"class_{c}_std"] = np.nanstd(arr)

            posthoc = pairwise_mannwhitney(values_by_class, class_pairs)
            for (c1, c2), (st, pp) in posthoc.items():
                row[f"mw_{c1}_vs_{c2}_stat"] = st
                row[f"mw_{c1}_vs_{c2}_p"] = pp

            region_pair_rows.append(row)

    df_region_pairs = pd.DataFrame(region_pair_rows)
    valid_mask = df_region_pairs["kw_p"].notna().values
    rej, p_corr = fdrcorrection(df_region_pairs.loc[valid_mask, "kw_p"].values, alpha=0.05)

    df_region_pairs["kw_p_fdr"] = np.nan
    df_region_pairs["kw_sig_fdr"] = False
    df_region_pairs.loc[valid_mask, "kw_p_fdr"] = p_corr
    df_region_pairs.loc[valid_mask, "kw_sig_fdr"] = rej
    df_region_pairs.to_csv(os.path.join(output_dir, "region_pair_tests.csv"), index=False)
    print(df_region_pairs.sort_values("kw_p_fdr").head(20))
    for c in classes:
        mats = [s["region_adj"] for s in subject_summaries if s["class_id"] == c]
        mean_mat = np.nanmean(mats, axis=0)

        save_path = os.path.join(output_dir, f"region_connectivity_class_{c}.png")
        plot_matrix(mean_mat, f"Region connectivity - Class {c}", region_names, save_path)
    sig_mat = np.zeros((R, R), dtype=float)

    for _, row in df_region_pairs.iterrows():
        i = region_names.index(row["region_1"])
        j = region_names.index(row["region_2"])
        sig_mat[i, j] = 1.0 if bool(row["kw_sig_fdr"]) else 0.0

    plot_matrix(
        sig_mat,
        "Significant region pairs across classes (FDR)",
        region_names,
        os.path.join(output_dir, "region_pair_significance_fdr.png")
    )

    sig_region_pairs = df_region_pairs[df_region_pairs["kw_sig_fdr"]].copy()

    for _, row in sig_region_pairs.iterrows():
        r1, r2 = row["region_1"], row["region_2"]
        i, j = region_names.index(r1), region_names.index(r2)

        plot_data = []
        for c in classes:
            vals = [
                s["region_adj"][i, j]
                for s in subject_summaries
                if s["class_id"] == c
            ]
            plot_data.append(vals)

        plt.figure(figsize=(6, 4))
        plt.boxplot(plot_data, tick_labels=[f"Class {c}" for c in classes])
        plt.title(f"{r1} - {r2}")
        plt.ylabel("Connectivity")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"boxplot_regionpair_{r1}_{r2}.png"), dpi=300)
        plt.close()

    within_region_rows = []

    for i, r in enumerate(region_names):
        values_by_class = {}
        for c in classes:
            vals = [
                s["region_adj"][i, i]
                for s in subject_summaries
                if s["class_id"] == c
            ]
            values_by_class[c] = vals

        groups = [values_by_class[c] for c in classes]
        stat, p = safe_kruskal(groups)

        row = {
            "region": r,
            "kw_stat": stat,
            "kw_p": p
        }

        for c in classes:
            arr = np.asarray(values_by_class[c], dtype=float)
            row[f"class_{c}_mean"] = np.nanmean(arr)
            row[f"class_{c}_std"] = np.nanstd(arr)

        posthoc = pairwise_mannwhitney(values_by_class, class_pairs)
        for (c1, c2), (st, pp) in posthoc.items():
            row[f"mw_{c1}_vs_{c2}_stat"] = st
            row[f"mw_{c1}_vs_{c2}_p"] = pp

        within_region_rows.append(row)

    df_within_regions = pd.DataFrame(within_region_rows)

    rej, p_corr = fdrcorrection(df_within_regions["kw_p"].values, alpha=0.05)
    df_within_regions["kw_p_fdr"] = p_corr
    df_within_regions["kw_sig_fdr"] = rej

    df_within_regions.to_csv(os.path.join(output_dir, "within_region_tests.csv"), index=False)
    print(df_within_regions.sort_values("kw_p_fdr"))

    F = subject_summaries[0]["region_features"].shape[1]
    feature_names = [f"feature_{i}" for i in range(F)]   # replace with real names if you have them

    region_feature_rows = []

    for r_idx, region_name in enumerate(region_names):
        for f_idx, feature_name in enumerate(feature_names):
            values_by_class = {}
            for c in classes:
                vals = [
                    s["region_features"][r_idx, f_idx]
                    for s in subject_summaries
                    if s["class_id"] == c
                ]
                values_by_class[c] = vals

            groups = [values_by_class[c] for c in classes]
            stat, p = safe_kruskal(groups)

            row = {
                "region": region_name,
                "feature_idx": f_idx,
                "feature_name": feature_name,
                "kw_stat": stat,
                "kw_p": p
            }

            for c in classes:
                arr = np.asarray(values_by_class[c], dtype=float)
                row[f"class_{c}_mean"] = np.nanmean(arr)
                row[f"class_{c}_std"] = np.nanstd(arr)

            posthoc = pairwise_mannwhitney(values_by_class, class_pairs)
            for (c1, c2), (st, pp) in posthoc.items():
                row[f"mw_{c1}_vs_{c2}_stat"] = st
                row[f"mw_{c1}_vs_{c2}_p"] = pp

            region_feature_rows.append(row)

    df_region_features = pd.DataFrame(region_feature_rows)
    valid_mask = df_region_features["kw_p"].notna().values
    rej, p_corr = fdrcorrection(df_region_features.loc[valid_mask, "kw_p"].values, alpha=0.05)

    df_region_features["kw_p_fdr"] = np.nan
    df_region_features["kw_sig_fdr"] = False
    df_region_features.loc[valid_mask, "kw_p_fdr"] = p_corr
    df_region_features.loc[valid_mask, "kw_sig_fdr"] = rej

    df_region_features.to_csv(os.path.join(output_dir, "region_feature_tests.csv"), index=False)
    print(df_region_features.sort_values("kw_p_fdr").head(20))
    sig_region_features = df_region_features[df_region_features["kw_sig_fdr"]].copy()

    for _, row in sig_region_features.iterrows():
        region_name = row["region"]
        feature_name = row["feature_name"]
        r_idx = region_names.index(region_name)
        f_idx = int(row["feature_idx"])

        plot_data = []
        for c in classes:
            vals = [
                s["region_features"][r_idx, f_idx]
                for s in subject_summaries
                if s["class_id"] == c
            ]
            plot_data.append(vals)

        plt.figure(figsize=(6, 4))
        plt.boxplot(plot_data, tick_labels=[f"Class {c}" for c in classes])
        plt.title(f"{region_name} - {feature_name}")
        plt.ylabel("Feature value")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"boxplot_regionfeature_{region_name}_{feature_name}.png"),
            dpi=300
        )
        plt.close()
    N = subject_summaries[0]["mean_adj"].shape[0]
    channel_pair_rows = []

    for i in range(N):
        for j in range(i + 1, N):   # upper triangle only
            values_by_class = {}
            for c in classes:
                vals = [
                    s["mean_adj"][i, j]
                    for s in subject_summaries
                    if s["class_id"] == c
                ]
                values_by_class[c] = vals

            groups = [values_by_class[c] for c in classes]
            stat, p = safe_kruskal(groups)

            row = {
                "ch_1_idx": i,
                "ch_2_idx": j,
                "ch_1_name": channel_names[i],
                "ch_2_name": channel_names[j],
                "kw_stat": stat,
                "kw_p": p
            }

            for c in classes:
                arr = np.asarray(values_by_class[c], dtype=float)
                row[f"class_{c}_mean"] = np.nanmean(arr)
                row[f"class_{c}_std"] = np.nanstd(arr)

            posthoc = pairwise_mannwhitney(values_by_class, class_pairs)
            for (c1, c2), (st, pp) in posthoc.items():
                row[f"mw_{c1}_vs_{c2}_stat"] = st
                row[f"mw_{c1}_vs_{c2}_p"] = pp

            channel_pair_rows.append(row)

    df_channel_pairs = pd.DataFrame(channel_pair_rows)
    valid_mask = df_channel_pairs["kw_p"].notna().values
    rej, p_corr = fdrcorrection(df_channel_pairs.loc[valid_mask, "kw_p"].values, alpha=0.05)

    df_channel_pairs["kw_p_fdr"] = np.nan
    df_channel_pairs["kw_sig_fdr"] = False
    df_channel_pairs.loc[valid_mask, "kw_p_fdr"] = p_corr
    df_channel_pairs.loc[valid_mask, "kw_sig_fdr"] = rej

    df_channel_pairs.to_csv(os.path.join(output_dir, "channel_pair_tests.csv"), index=False)
    print(df_channel_pairs.sort_values("kw_p_fdr").head(30))


    hypergraph_hints = {}

    # 1. region hyperedges: regions with strong or significant within-region structure
    sig_within = df_within_regions[df_within_regions["kw_sig_fdr"]]
    hypergraph_hints["candidate_region_hyperedges"] = sig_within["region"].tolist()

    # 2. cross-region hyperedges: significant region pairs
    sig_pairs = df_region_pairs[df_region_pairs["kw_sig_fdr"]]
    hypergraph_hints["candidate_cross_region_hyperedges"] = list(
        zip(sig_pairs["region_1"], sig_pairs["region_2"])
    )

    # 3. important node features: region-feature pairs that differ across classes
    sig_feats = df_region_features[df_region_features["kw_sig_fdr"]]
    hypergraph_hints["important_region_features"] = sig_feats[
        ["region", "feature_idx", "feature_name", "kw_p_fdr"]
    ].to_dict(orient="records")

    # 4. important channel pairs
    sig_ch = df_channel_pairs[df_channel_pairs["kw_sig_fdr"]]
    hypergraph_hints["important_channel_pairs"] = sig_ch[
        ["ch_1_name", "ch_2_name", "kw_p_fdr"]
    ].to_dict(orient="records")

    print(hypergraph_hints)


    pd.DataFrame({
        "candidate_region_hyperedges": pd.Series(hypergraph_hints["candidate_region_hyperedges"])
    }).to_csv(os.path.join(output_dir, "hypergraph_candidate_regions.csv"), index=False)

    pd.DataFrame(
        hypergraph_hints["candidate_cross_region_hyperedges"],
        columns=["region_1", "region_2"]
    ).to_csv(os.path.join(output_dir, "hypergraph_candidate_cross_region_pairs.csv"), index=False)

    pd.DataFrame(
        hypergraph_hints["important_region_features"]
    ).to_csv(os.path.join(output_dir, "hypergraph_important_region_features.csv"), index=False)

    pd.DataFrame(
        hypergraph_hints["important_channel_pairs"]
    ).to_csv(os.path.join(output_dir, "hypergraph_important_channel_pairs.csv"), index=False)
