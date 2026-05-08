import json
from pathlib import Path
import pandas as pd


def get_nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def stringify(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        return json.dumps(list(x), ensure_ascii=False)
    if isinstance(x, dict):
        return json.dumps(x, ensure_ascii=False, sort_keys=True)
    return str(x)


def collect_one_summary(summary_path: Path):
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    spec = s.get("spec", {})
    level = spec.get("level", {})
    topology = spec.get("topology", {})
    edge_weights = spec.get("edge_weights", {})
    conn_tensor = spec.get("connectivity_tensor", {})
    model = spec.get("model", {})
    aggregation = spec.get("aggregation", {})
    train_cfg = spec.get("train", {})

    run_dir = Path(s.get("run_dir", summary_path.parent))
    folder_name = run_dir.name

    row = {
        # identity
        "folder_name": folder_name,
        # "summary_path": str(summary_path),
        # "run_dir": str(run_dir),
        "spec_name": spec.get("name"),
        # "task": spec.get("task"),

        # monitor / selection
        "best_epoch": s.get("best_epoch"),
        "monitor": s.get("monitor"),
        "monitor_value": s.get("monitor_value"),

        # test metrics
        "test_num_samples": get_nested(s, "test_metrics", "num_samples"),
        "test_acc": get_nested(s, "test_metrics", "accuracy"),
        "test_bal_acc": get_nested(s, "test_metrics", "balanced_accuracy"),
        "test_macro_f1": get_nested(s, "test_metrics", "macro_f1"),
        "test_confusion_matrix": get_nested(s, "test_metrics", "confusion_matrix"),
        "test_brier": get_nested(s, "test_metrics", "brier_score"),
        "test_roc_auc_macro_ovr": get_nested(s, "test_metrics", "roc_auc_macro_ovr"),
        "test_pr_auc_macro_ovr": get_nested(s, "test_metrics", "pr_auc_macro_ovr"),

        # val metrics
        "val_num_samples": get_nested(s, "val_metrics", "num_samples"),
        "val_acc": get_nested(s, "val_metrics", "accuracy"),
        "val_bal_acc": get_nested(s, "val_metrics", "balanced_accuracy"),
        "val_macro_f1": get_nested(s, "val_metrics", "macro_f1"),
        "val_confusion_matrix": get_nested(s, "val_metrics", "confusion_matrix"),
        "val_brier": get_nested(s, "val_metrics", "brier_score"),
        "val_roc_auc_macro_ovr": get_nested(s, "val_metrics", "roc_auc_macro_ovr"),
        "val_pr_auc_macro_ovr": get_nested(s, "val_metrics", "pr_auc_macro_ovr"),

        # train metrics
        "train_num_samples": get_nested(s, "train_metrics", "num_samples"),
        "train_acc": get_nested(s, "train_metrics", "accuracy"),
        "train_bal_acc": get_nested(s, "train_metrics", "balanced_accuracy"),
        "train_macro_f1": get_nested(s, "train_metrics", "macro_f1"),
        "train_confusion_matrix": get_nested(s, "train_metrics", "confusion_matrix"),
        # "train_brier": get_nested(s, "train_metrics", "brier_score"),
        # "train_roc_auc_macro_ovr": get_nested(s, "train_metrics", "roc_auc_macro_ovr"),
        # "train_pr_auc_macro_ovr": get_nested(s, "train_metrics", "pr_auc_macro_ovr"),

        # dataset / inputs
        # "dataset_path": spec.get("dataset_path"),
        "h5_path": spec.get("h5_path"),
        "feature_families": stringify(spec.get("feature_families")),
        # "connectivity_metrics_to_load": stringify(spec.get("connectivity_metrics_to_load")),
        "class_names": stringify(s.get("class_names")),
        # training config
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("lr"),
        "weight_decay": train_cfg.get("weight_decay"),
        "epochs": train_cfg.get("epochs"),
        "patience": train_cfg.get("patience"),
        # "monitor_cfg": train_cfg.get("monitor"),
        # "monitor_mode_cfg": train_cfg.get("monitor_mode"),
        "seed": train_cfg.get("seed"),
        # "num_workers": train_cfg.get("num_workers"),

        # experiment axes
        "graph_level": level.get("graph_level"),
        "macro_duration_sec": level.get("macro_duration_sec"),
        "feature_reduce": level.get("feature_reduce"),
        "connectivity_reduce": level.get("connectivity_reduce"),

        "topology_strategy": topology.get("strategy"),
        "topology_metric": topology.get("topology_metric"),
        "topology_band": topology.get("topology_band"),
        "topology_similarity": topology.get("similarity"),
        "topology_kwargs": stringify(topology.get("topology_kwargs")),
        "graph_bank_specs": stringify(topology.get("graph_bank_specs")),
        "fuse_method": topology.get("fuse_method"),
        "fuse_topology_rule": topology.get("fuse_topology_rule"),
        "fuse_vote_threshold": topology.get("fuse_vote_threshold"),
        "primary_candidate": topology.get("primary_candidate"),

        "edge_weight_strategy": edge_weights.get("strategy"),
        "edge_metric": edge_weights.get("edge_metric"),
        "edge_band": edge_weights.get("edge_band"),
        "normalize_mode": edge_weights.get("normalize_mode"),
        "fused_sources": stringify(edge_weights.get("fused_sources")),
        "fused_method_edge": edge_weights.get("fused_method"),

        "connectivity_tensor_metrics": stringify(conn_tensor.get("metrics")),
        "connectivity_tensor_bands": stringify(conn_tensor.get("bands")),

        "model_family": model.get("family"),
        "connectivity_encoder_type": model.get("connectivity_encoder_type"),
        "backbone": model.get("backbone"),
        "hidden_dim": model.get("hidden_dim"),
        "emb_dim": model.get("emb_dim"),
        "dropout": model.get("dropout"),
        "num_layers": model.get("num_layers"),
        "gat_heads": model.get("gat_heads"),
        "use_edge_weight": model.get("use_edge_weight"),
        "use_batchnorm": model.get("use_batchnorm"),
        "graph_readout": model.get("graph_readout"),
        "fusion_mode": model.get("fusion_mode"),
        "graph_bank_fusion_mode": model.get("graph_bank_fusion_mode"),

        "aggregation_strategy": aggregation.get("strategy"),
        "posthoc_eval_vote": aggregation.get("posthoc_eval_vote"),
        "attn_dim": aggregation.get("attn_dim"),
        "train_max_instances_per_subject": aggregation.get("train_max_instances_per_subject"),
        # "eval_max_instances_per_subject": aggregation.get("eval_max_instances_per_subject"),




    }

    # optional file existence checks
    # row["has_best_model_pt"] = (run_dir / "best_model.pt").exists()
    # row["has_history_csv"] = (run_dir / "history.csv").exists()
    # row["has_summary_metrics_csv"] = (run_dir / "summary_metrics.csv").exists()
    # row["has_train_predictions_csv"] = (run_dir / "train_predictions.csv").exists()
    # row["has_val_predictions_csv"] = (run_dir / "val_predictions.csv").exists()
    # row["has_test_predictions_csv"] = (run_dir / "test_predictions.csv").exists()

    return row


def collect_block0_results(results_root: Path):
    rows = []

    # find every summary.json under the results tree
    for summary_path in sorted(results_root.rglob("summary.json")):
        try:
            rows.append(collect_one_summary(summary_path))
        except Exception as e:
            print(f"FAILED to parse {summary_path}: {e}")

    if len(rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # main ranking
    df = df.sort_values(
        by=["val_bal_acc", "val_macro_f1", "test_bal_acc", "val_brier"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    return df

def select_bucket_winners(df, top_k_per_bucket=1):
    bucket_cols = ["graph_level", "model_family"]
    winners = (
        df.sort_values(
            by=["val_bal_acc", "val_macro_f1"], #, "val_loss"
            ascending=[False, False], # True
        )
        .groupby(bucket_cols, as_index=False, group_keys=False)
        .head(top_k_per_bucket)
        .reset_index(drop=True)
    )
    print(winners)
    return winners



if __name__ == "__main__":

    RESULTS_ROOT = Path("/home/anphan/Documents/EEG_Project/results_caueeg")
    # RESULTS_ROOT = Path("/home/anphan/Documents/EEG_Project/CAUEEG/results_pipeline")
    OUT_CSV = RESULTS_ROOT / "leaderboard.csv"
    df = collect_block0_results(RESULTS_ROOT)

    if len(df) == 0:
        print("No summary.json files found.")
    else:
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved leaderboard to: {OUT_CSV}\n")
        select_bucket_winners(df, top_k_per_bucket=1)
        # show_cols = [
        #     "folder_name",
        #     "spec_name",
        #     "graph_level",
        #     "model_family",
        #     "topology_strategy",
        #     "edge_weight_strategy",
        #     "aggregation_strategy",
        #     "graph_readout",
        #     "val_bal_acc",
        #     "val_macro_f1",
        #     "test_bal_acc",
        #     "test_macro_f1",
        # ]
        # print(df[show_cols].to_string(index=False))