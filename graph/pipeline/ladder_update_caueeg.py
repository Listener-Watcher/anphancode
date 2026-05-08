# # default_fixed_edge_pairs_19()
# # default_graph_bank_specs()
# # build_block0_ladder(...)
# # build_legacy_subject_macro_ladder(...)
# # build_graph_bank_ablation_ladder(...)
# # build_stage2_ladder(...)

# from caueeg_main import (
#     CAUEEGExperimentSpec, LevelConfig, TopologyConfig, EdgeWeightConfig,
#     ConnectivityTensorConfig, ModelConfig, AggregationConfig, TrainConfig
# )
from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence
import os
from caueeg_main import (
    CAUEEGExperimentSpec,
    LevelConfig,
    TopologyConfig,
    EdgeWeightConfig,
    ConnectivityTensorConfig,
    ModelConfig,
    AggregationConfig,
    TrainConfig,
    run_caueeg_ladder,
    # default_fixed_edge_pairs_19,
    # default_graph_bank_specs,
)


# ---------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------
def _base_spec(
    *,
    dataset_path: str,
    h5_path: str,
    task: str,
    output_root: str,
) -> CAUEEGExperimentSpec:
    return CAUEEGExperimentSpec(
        name="base",
        task=task,
        dataset_path=dataset_path,
        h5_path=h5_path,
        output_root=output_root,
        feature_families=("relative_band_power", "statistical"),
        connectivity_metrics_to_load=("coherence", "wpli", "plv"),
        connectivity_tensor=ConnectivityTensorConfig(
            metrics=("wpli",),
            bands=(0, 1, 2, 3, 4),   # delta..gamma
        ),
        train=TrainConfig(
            batch_size=8,
            epochs=200,
            patience=80,
            lr=1e-3,
            weight_decay=5e-4,
            seed=42,
            num_workers=0,
        ),
    )


def _level_cfg(level_name: str) -> LevelConfig:
    return LevelConfig(
        graph_level=level_name,
        macro_duration_sec=300.0,
        feature_reduce="mean",
        connectivity_reduce="mean",
    )


def _agg_cfg(level_name: str, strategy: str) -> AggregationConfig:
    if strategy == "none":
        return AggregationConfig(strategy="none")
    if strategy == "mean_mil":
        return AggregationConfig(
            strategy="mean_mil",
            train_max_instances_per_subject=100,
            attn_dim=64,
        )
    if strategy == "gated_attention_mil":
        return AggregationConfig(
            strategy="gated_attention_mil",
            train_max_instances_per_subject=100,
            attn_dim=64,
        )
    if strategy == "subject_fusion":
        return AggregationConfig(
            strategy="subject_fusion",
            attn_dim=64,
        )
    raise ValueError(strategy)


def _train_cfg(level_name: str) -> TrainConfig:
    if level_name == "subject":
        return TrainConfig(
            batch_size=16,
            epochs=100,
            patience=60,
            lr=1e-3,
            weight_decay=5e-3,
            seed=42,
            num_workers=0,
        )
    return TrainConfig(
        batch_size=16,
        epochs=200,
        patience=100,
        lr=1e-3,
        weight_decay=5e-4,
        seed=42,
        num_workers=0,
    )


def _native_graph_topology_bundle(fixed_edges):
    return {
        "fixed": dict(
            topology=TopologyConfig(
                strategy="fixed",
                fixed_edge_pairs=fixed_edges,
            ),
            edge_weights=EdgeWeightConfig(
                strategy="connectivity",
                edge_metric="wpli",
                edge_band=2,  # alpha
            ),
        ),
        "topk": dict(
            topology=TopologyConfig(
                strategy="connectivity",
                topology_metric="wpli",
                topology_band=2,
                topology_kwargs={"mode": "topk", "topk": 4},
            ),
            edge_weights=EdgeWeightConfig(
                strategy="connectivity",
                edge_metric="wpli",
                edge_band=2,
            ),
        ),
        "mst": dict(
            topology=TopologyConfig(
                strategy="connectivity",
                topology_metric="wpli",
                topology_band=2,
                topology_kwargs={"mode": "mst"},
            ),
            edge_weights=EdgeWeightConfig(
                strategy="connectivity",
                edge_metric="wpli",
                edge_band=2,
            ),
        ),
        "graph_bank": dict(
            topology=TopologyConfig(
                strategy="fused_bank",
                graph_bank_specs=default_graph_bank_specs(),
                fuse_method="mean",
                fuse_topology_rule="union",
                primary_candidate=0,
            ),
            edge_weights=EdgeWeightConfig(
                strategy="fused",
                fused_sources=(
                    ("coherence", 1),
                    ("coherence", 2),
                    ("coherence", 3),
                    ("wpli", 2),
                ),
                fused_method="mean",
            ),
        ),
    }


# ---------------------------------------------------------
# 1) Legacy subject + macro
# ---------------------------------------------------------
def build_legacy_subject_macro_ladder(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_caueeg_legacy_subject_macro",
    legacy_encoders: Sequence[str] = (
        # "linkx",
        # "mlp_node",
        # "gnn",
        "gat",
        # "hybrid",
        # "linkx_cnn5",
        # "cnn5",
        # "sage",
        # "gcn2",
        # "h2gcn",
    ),
) -> list[CAUEEGExperimentSpec]:
    base = _base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    specs = []
    fixed_edges = default_fixed_edge_pairs_19()

    topo_bundle = _native_graph_topology_bundle(fixed_edges)
    level_name = "macro"
    # for level_name in ( "macro",): #"subject",
    agg = "none" if level_name == "subject" else "subject_fusion"

    for encoder_type in legacy_encoders:
        topo_key = "graph_bank" if encoder_type in {"linkx_cnn5", "cnn5"} else "fixed"
        topo_kwargs = topo_bundle[topo_key]

        spec = replace(
            base,
            name=f"{level_name}_legacy_{encoder_type}_{agg}",
            level=_level_cfg(level_name),
            topology=topo_kwargs["topology"],
            edge_weights=topo_kwargs["edge_weights"],
            model=ModelConfig(
                family="legacy_encoder",
                encoder_source="legacy",
                encoder_type=encoder_type,
                emb_dim=64,
                hidden_dim=64,
                dropout=0.3,
                graph_readout="mean",
                legacy_graph_pool="mean",
                legacy_num_bands=5,
            ),
            aggregation=_agg_cfg(level_name, agg),
            train=_train_cfg(level_name),
        )
        specs.append(spec)

    return specs


# ---------------------------------------------------------
# 2) Native segment + macro MIL / subject-fusion
# ---------------------------------------------------------
def build_native_mil_segment_macro_readout_pool_ladder(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_caueeg_native_readout_pool",
    model_families: Sequence[str] = (
        "fixed_graph_gnn",
        "fused_graph_bank_gnn",
        "dual_branch_graph",
    ),
    levels: Sequence[str] = ("macro",), #"segment", 
    readouts: Sequence[str] = (
        "mean",
        "mean_max_concat",
        # "attention",
        "gated_attention",
    ),
    node_poolings: Sequence[str] = ("none", "topk", "sagpool"),
    backbones: Sequence[str] = ("gatv2",), # "sage", "gcn"
    topology_keys: Sequence[str] = ("fixed", "topk", "mst", "graph_bank"),
) -> list[CAUEEGExperimentSpec]:
    base = _base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    specs = []
    fixed_edges = default_fixed_edge_pairs_19()
    topo_bundle = _native_graph_topology_bundle(fixed_edges)

    for level_name in levels:
        agg_list = ["mean_mil", "gated_attention_mil"] if level_name == "segment" else ["subject_fusion"]

        for agg in agg_list:
            for fam in model_families:
                for topo_key in topology_keys:
                    if fam == "fused_graph_bank_gnn" and topo_key != "graph_bank":
                        continue
                    if fam != "fused_graph_bank_gnn" and topo_key == "graph_bank":
                        continue

                    topo_kwargs = topo_bundle[topo_key]

                    for backbone in backbones:
                        for readout in readouts:
                            for node_pool in node_poolings:
                                spec = replace(
                                    base,
                                    name=f"{level_name}_{fam}_{topo_key}_{backbone}_{readout}_{node_pool}_{agg}",
                                    level=_level_cfg(level_name),
                                    topology=topo_kwargs["topology"],
                                    edge_weights=topo_kwargs["edge_weights"],
                                    model=ModelConfig(
                                        family=fam,
                                        backbone=backbone,
                                        emb_dim=128,
                                        hidden_dim=64,
                                        dropout=0.2,
                                        graph_readout=readout,
                                        graph_bank_fusion_mode="summary_gated",
                                        fusion_mode="gated",
                                        node_pooling_type=node_pool,   # add this field to ModelConfig
                                        node_pool_ratio=0.8,           # add this field too
                                    ),
                                    aggregation=_agg_cfg(level_name, agg),
                                    train=_train_cfg(level_name),
                                )
                                specs.append(spec)

    return specs


# ---------------------------------------------------------
# 3) Legacy segment + macro with readout/pool alignment
# ---------------------------------------------------------
def build_legacy_mil_segment_macro_readout_pool_ladder(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_caueeg_legacy_readout_pool",
    levels: Sequence[str] = ("macro",), #"segment", 
    legacy_encoders: Sequence[str] = (
        # "mlp_node",
        # "gnn",
        # "sage",
        # "gcn2",
        # "h2gcn",
        "gat",
        # "hybrid",
    ),
    readouts: Sequence[str] = (
        "mean",
        "mean_max_concat",
        "attention",
    ),
    node_poolings: Sequence[str] = ("none", "topk", "sagpool"),
    legacy_graph_pools: Sequence[str] = ("mean", "max", "add"),
    topology_keys: Sequence[str] = ("fixed", "mst"), #"topk",
) -> list[CAUEEGExperimentSpec]:
    """
    This builder is intended for the rewritten legacy path, not the raw old path.

    Practical meaning:
    - gnn/sage/gcn2/h2gcn/gat/hybrid: can use legacy_graph_pool directly
    - mlp_node: should use your LegacyMLPNodeWithGraphReadout or similar rewritten wrapper
    - linkx/linkx_cnn/linkx_cnn5/cnn5 are intentionally excluded here
      because they do not expose native-style node pooling / readout naturally
    """
    base = _base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    specs = []
    fixed_edges = default_fixed_edge_pairs_19()
    topo_bundle = _native_graph_topology_bundle(fixed_edges)

    for level_name in levels:
        agg_list = ["mean_mil", "gated_attention_mil"] if level_name == "segment" else ["subject_fusion"]

        for agg in agg_list:
            for encoder_type in legacy_encoders:
                for topo_key in topology_keys:
                    topo_kwargs = topo_bundle[topo_key]

                    for readout in readouts:
                        for node_pool in node_poolings:
                            for graph_pool in legacy_graph_pools:
                                spec = replace(
                                    base,
                                    name=f"{level_name}_legacyrw_{encoder_type}_{topo_key}_{readout}_{node_pool}_{graph_pool}_{agg}",
                                    level=_level_cfg(level_name),
                                    topology=topo_kwargs["topology"],
                                    edge_weights=topo_kwargs["edge_weights"],
                                    model=ModelConfig(
                                        family="legacy_encoder",
                                        encoder_source="legacy_rewrite",
                                        encoder_type=encoder_type,
                                        emb_dim=64,
                                        hidden_dim=64,
                                        dropout=0.2,
                                        graph_readout=readout,
                                        legacy_graph_pool=graph_pool,
                                        legacy_graph_readout=readout,
                                        legacy_align_native_readout=True,
                                        node_pooling_type=node_pool,
                                        node_pool_ratio=0.8,
                                    ),
                                    aggregation=_agg_cfg(level_name, agg),
                                    train=_train_cfg(level_name),
                                )
                                specs.append(spec)

    return specs




def make_macro_legacy_gat_base_spec(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_macro_legacy_gat_finetune",
) -> CAUEEGExperimentSpec:
    base = _base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    fixed_edges = default_fixed_edge_pairs_19()
    topo_bundle = _native_graph_topology_bundle(fixed_edges)
    topo_kwargs = topo_bundle["fixed"]

    spec = replace(
        base,
        name="macro_legacy_gat_subject_fusion_base",
        level=_level_cfg("macro"),
        topology=topo_kwargs["topology"],
        edge_weights=topo_kwargs["edge_weights"],
        model=ModelConfig(
            family="legacy_encoder",
            encoder_source="legacy",
            encoder_type="gat",
            hidden_dim=64,
            emb_dim=64,
            dropout=0.3,
            graph_readout="mean",
            legacy_graph_pool="mean",
            legacy_num_layers=3,
            legacy_gat_heads=4,
            legacy_edge_dim=1,
        ),
        aggregation=_agg_cfg("macro", "subject_fusion"),
        train=TrainConfig(
            batch_size=8,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=120,
            patience=20,
            monitor="balanced_accuracy",
            monitor_mode="max",
            seed=42,
            num_workers=0,
        ),
    )
    return spec

def build_macro_legacy_gat_stage1_optimizer(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_macro_legacy_gat_stage1_optimizer",
    lrs: Sequence[float] = (3e-4, 1e-3, 3e-3),
    weight_decays: Sequence[float] = (1e-3,),
) -> list[CAUEEGExperimentSpec]:
    base_spec = make_macro_legacy_gat_base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    specs = []
    for lr in lrs:
        for wd in weight_decays:
            spec = replace(base_spec)
            spec.name = f"macro_legacy_gat_sf_opt_lr{lr}_wd{wd}"
            spec.train.lr = lr
            spec.train.weight_decay = wd
            specs.append(spec)

    return specs       


def build_macro_legacy_gat_stage2_modelsize(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_macro_legacy_gat_stage2_modelsize",
    best_lr: float = 1e-3,
    best_weight_decay: float = 1e-4,
    hidden_dims: Sequence[int] = (64, 128),
    emb_dims: Sequence[int] = (64, 128),
    dropouts: Sequence[float] = (0.1, 0.3, 0.5),
) -> list[CAUEEGExperimentSpec]:
    base_spec = make_macro_legacy_gat_base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    base_spec.train.lr = best_lr
    base_spec.train.weight_decay = best_weight_decay

    specs = []
    for hd in hidden_dims:
        for ed in emb_dims:
            for do in dropouts:
                spec = replace(base_spec)
                spec.name = f"macro_legacy_gat_sf_size_hd{hd}_ed{ed}_do{do}"
                spec.model.hidden_dim = hd
                spec.model.emb_dim = ed
                spec.model.dropout = do
                specs.append(spec)

    return specs


def build_macro_legacy_gat_stage3_structure(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_macro_legacy_gat_stage3_structure",
    best_lr: float = 1e-3,
    best_weight_decay: float = 1e-4,
    best_hidden_dim: int = 64,
    best_emb_dim: int = 64,
    best_dropout: float = 0.3,
    num_layers_list: Sequence[int] = (2, 3, 4),
    gat_heads_list: Sequence[int] = (2, 4, 8),
) -> list[CAUEEGExperimentSpec]:
    base_spec = make_macro_legacy_gat_base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    base_spec.train.lr = best_lr
    base_spec.train.weight_decay = best_weight_decay
    base_spec.model.hidden_dim = best_hidden_dim
    base_spec.model.emb_dim = best_emb_dim
    base_spec.model.dropout = best_dropout

    specs = []
    for nl in num_layers_list:
        for nh in gat_heads_list:
            spec = replace(base_spec)
            spec.name = f"macro_legacy_gat_sf_struct_L{nl}_H{nh}"
            spec.model.legacy_num_layers = nl
            spec.model.legacy_gat_heads = nh
            specs.append(spec)

    return specs


def build_macro_legacy_gat_stage4_pool(
    *,
    dataset_path: str,
    h5_path: str,
    task: str = "dementia",
    output_root: str = "./results_macro_legacy_gat_stage4_pool",
    best_lr: float = 1e-3,
    best_weight_decay: float = 1e-4,
    best_hidden_dim: int = 64,
    best_emb_dim: int = 64,
    best_dropout: float = 0.3,
    best_num_layers: int = 3,
    best_gat_heads: int = 4,
    graph_pools: Sequence[str] = ("mean", "max"),
) -> list[CAUEEGExperimentSpec]:
    base_spec = make_macro_legacy_gat_base_spec(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=output_root,
    )

    base_spec.train.lr = best_lr
    base_spec.train.weight_decay = best_weight_decay
    base_spec.model.hidden_dim = best_hidden_dim
    base_spec.model.emb_dim = best_emb_dim
    base_spec.model.dropout = best_dropout
    base_spec.model.legacy_num_layers = best_num_layers
    base_spec.model.legacy_gat_heads = best_gat_heads

    specs = []
    for gp in graph_pools:
        spec = replace(base_spec)
        spec.name = f"macro_legacy_gat_sf_pool_{gp}"
        spec.model.legacy_graph_pool = gp
        specs.append(spec)

    return specs

def default_fixed_edge_pairs_19() -> list[tuple[int, int]]:
    """
    Simple hand-crafted 10-20 neighbor graph for CAUEEG 19 EEG channels.

    Channel order:
    0 Fp1, 1 F3, 2 C3, 3 P3, 4 O1,
    5 Fp2, 6 F4, 7 C4, 8 P4, 9 O2,
    10 F7, 11 T3, 12 T5, 13 F8, 14 T4,
    15 T6, 16 FZ, 17 CZ, 18 PZ
    """
    return [
        (0, 1), (0, 10), (0, 5),
        (5, 6), (5, 13),

        (10, 1), (10, 11),
        (11, 2), (11, 12),
        (12, 3), (12, 4),

        (13, 6), (13, 14),
        (14, 7), (14, 15),
        (15, 8), (15, 9),

        (1, 2), (2, 3), (3, 4),
        (6, 7), (7, 8), (8, 9),

        (1, 16), (6, 16),
        (2, 17), (7, 17),
        (3, 18), (8, 18),

        (16, 17), (17, 18),
    ]


def default_graph_bank_specs() -> list[dict[str, Any]]:
    """
    Stronger graph-bank candidate set.
    Use mostly integer bands to avoid string-band parsing issues.

    Band index map:
      0 delta
      1 theta
      2 alpha
      3 beta
      4 gamma
    """
    return [
        {
            "name": "coh_alpha_topk",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": 2,
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "coh_alpha_mst",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": 2,
            "topology_kwargs": {"mode": "mst"},
        },
        {
            "name": "coh_theta_topk",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "coherence",
            "band": 1,
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "wpli_alpha_topk",
            "topology_mode": "connectivity",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "wpli",
            "band": 2,
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "plv_fixed",
            "topology_mode": "fixed",
            "edge_weight_mode": "connectivity",
            "connectivity_metric": "plv",
            "band": 2,
            # "topology_kwargs": {"mode": "topk", "topk": 4},
        },
        {
            "name": "feature_cosine_topk",
            "topology_mode": "feature_induced",
            "edge_weight_mode": "topology_weight",
            "similarity": "cosine",
            "topology_kwargs": {"mode": "topk", "topk": 4},
        },
    ]



LADDER_REGISTRY = {
    "legacy_subject_macro": build_legacy_subject_macro_ladder,
    "native_mil_segment_macro_readout_pool": build_native_mil_segment_macro_readout_pool_ladder,
    "legacy_mil_segment_macro_readout_pool": build_legacy_mil_segment_macro_readout_pool_ladder,
}


GAT_LADDER_REGISTRY = {
    "macro_legacy_gat_stage1_optimizer": build_macro_legacy_gat_stage1_optimizer,
    "macro_legacy_gat_stage2_modelsize": build_macro_legacy_gat_stage2_modelsize,
    "macro_legacy_gat_stage3_structure": build_macro_legacy_gat_stage3_structure,
    "macro_legacy_gat_stage4_pool": build_macro_legacy_gat_stage4_pool,
}


if __name__ == "__main__":
        
    # print("Available ladders:")
    # for name in LADDER_REGISTRY:
    #     print("-", name)

    dataset_path = "/home/anphan/Downloads/caueeg-dataset/"
    h5_path = "/home/anphan/Documents/caueeg_randomcrop_master_dementia_seed42.h5"
    task = "dementia"
    output_root = "/home/anphan/Documents/CAUEEG/results_pipeline"
    gat_root = os.path.join(output_root,"results_macro_gat_stage1")
    os.makedirs(gat_root, exist_ok = True)

    # builder_name = "native_mil_segment_macro_readout_pool"
    # builder_name = "legacy_mil_segment_macro_readout_pool"
    # # builder_name = "legacy_subject_macro"
    # for builder_name in LADDER_REGISTRY.keys():
    # builder = LADDER_REGISTRY[builder_name]

    # ladder = builder(
    #     dataset_path=dataset_path,
    #     h5_path=h5_path,
    #     task=task,
    #     output_root=output_root,
    # )

    # print(f"\nBuilder: {builder_name}")
    # print(f"Num experiments: {len(ladder)}")
    # for i, spec in enumerate(ladder):
    #     print(f"{i:02d} - {spec.name}")
    #     # run_caueeg_ladder([spec])
# #=============================================================================
    builder_name = "macro_legacy_gat_stage1_optimizer"
    builder = GAT_LADDER_REGISTRY[builder_name]
    ladder = build_macro_legacy_gat_stage1_optimizer(
        dataset_path=dataset_path,
        h5_path=h5_path,
        task=task,
        output_root=gat_root,
    )
    print(f"\nBuilder: {builder_name}")
    print(f"Num experiments: {len(ladder)}")
    for i, spec in enumerate(ladder):
        print(f"{i:02d} - {spec.name}")
        run_caueeg_ladder([spec])

# levels = ["subject", "macro"]
# encoder_types = ["linkx", "sage", "gnn"]
# aggregations = {
#     "subject": ["none"],
#     "macro": ["subject_fusion"],
# }

# import copy

# def build_stage2_caueeg_ladder(default_ladder, leaderboard_df):
#     name_to_spec = {spec.name: spec for spec in default_ladder}
#     winners = select_bucket_winners(leaderboard_df, top_k_per_bucket=3)

#     new_ladder = []

#     for _, row in winners.iterrows():
#         base_spec = copy.deepcopy(name_to_spec[row["spec_name"]])

#         # expand one axis at a time
#         if row["model_family"] in {"fixed_graph_gnn", "dual_branch_graph", "fused_graph_bank_gnn"}:
#             for readout in ["mean", "mean_max_concat", "attention"]:
#                 spec2 = copy.deepcopy(base_spec)
#                 spec2.name = f"{base_spec.name}_readout_{readout}"
#                 spec2.model.graph_readout = readout
#                 new_ladder.append(spec2)

#         if row["graph_level"] in {"segment", "macro"}:
#             for agg in ["mean_mil", "gated_attention_mil"]:
#                 spec2 = copy.deepcopy(base_spec)
#                 spec2.name = f"{base_spec.name}_agg_{agg}"
#                 spec2.aggregation.strategy = agg
#                 new_ladder.append(spec2)

#     return new_ladder
# def build_legacy_subject_macro_ladder(dataset_path, h5_path, task, output_root):
#     base = CAUEEGExperimentSpec(
#         name="base_legacy",
#         task=task,
#         dataset_path=dataset_path,
#         h5_path=h5_path,
#         output_root=output_root,
#     )

#     specs = []

#     for level in ["subject", "macro"]:
#         for encoder_type in ["linkx", "sage", "gnn", "mlp_node", "cnn5", "linkx_cnn5"]:
#             for agg in (["none"] if level == "subject" else ["subject_fusion"]):
#                 spec = replace(
#                     base,
#                     name=f"{level}_legacy_{encoder_type}_{agg}",
#                     level=LevelConfig(
#                         graph_level=level,
#                         macro_duration_sec=300.0,
#                         feature_reduce="mean",
#                         connectivity_reduce="mean",
#                     ),
#                     model=ModelConfig(
#                         family="legacy_encoder",
#                         encoder_source="legacy",
#                         encoder_type=encoder_type,
#                         emb_dim=128,
#                         hidden_dim=64,
#                         dropout=0.2,
#                         legacy_graph_pool="mean",
#                         graph_readout="mean",
#                         legacy_num_bands=5,
#                     ),
#                     aggregation=AggregationConfig(
#                         strategy=agg,
#                         attn_dim=64,
#                         train_max_instances_per_subject=100 if level != "subject" else None,
#                     ),
#                 )
#                 specs.append(spec)

#     return specs

# def build_new_caueeg_ladder(
#     *,
#     dataset_path: str,
#     h5_path: str,
#     task: str = "dementia",
#     output_root: str = "./results_caueeg",
# ) -> list[CAUEEGExperimentSpec]:
#     """
#     Expanded Block-0 ladder.

#     Goals:
#     - keep subject-level baselines
#     - add much stronger segment/macro coverage
#     - test more than just connectivity topk
#     - include fixed / mst / threshold / feature-induced / fused-bank
#     """

#     # integer band ids to avoid string-band parsing issues
#     DELTA = 0
#     THETA = 1
#     ALPHA = 2
#     BETA = 3
#     GAMMA = 4

#     fixed_edges = default_fixed_edge_pairs_19()

#     base = CAUEEGExperimentSpec(
#         name="base",
#         task=task,
#         dataset_path=dataset_path,
#         h5_path=h5_path,
#         output_root=output_root,
#         feature_families=("relative_band_power", "statistical"), #"hjorth", 
#         connectivity_metrics_to_load=("coherence", "wpli", "plv"), # "pearson", ),
#         connectivity_tensor=ConnectivityTensorConfig(
#             metrics=("wpli",),
#             bands=(DELTA, THETA, ALPHA, BETA, GAMMA),
#         ),
#         train=TrainConfig(
#             batch_size=16,
#             epochs=200,
#             patience=30,
#             lr=1e-3,
#             weight_decay=5e-3,
#             seed=42,
#         ),
#     )

#     subject_train = TrainConfig(
#         batch_size=16,
#         epochs=200,
#         patience=30,
#         lr=1e-3,
#         weight_decay=5e-3,
#         seed=42,
#     )

#     segment_train = TrainConfig(
#         batch_size=8,
#         epochs=200,
#         patience=30,
#         lr=1e-3,
#         weight_decay=5e-3,
#         seed=42,
#     )

#     macro_train = TrainConfig(
#         batch_size=8,
#         epochs=200,
#         patience=30,
#         lr=1e-3,
#         weight_decay=5e-3,
#         seed=42,
#     )

#     specs: list[CAUEEGExperimentSpec] = []

#     def add(name: str, **kwargs):
#         specs.append(replace(base, name=name, **kwargs))

#     # ==================================================
#     # Block 1: subject-level dense baselines
#     # ==================================================
#     add(
#         "subject_graph_bank",
#         level=LevelConfig(graph_level="subject"),
#         topology=TopologyConfig(
#             strategy="fused_bank",
#             graph_bank_specs=default_graph_bank_specs(),
#             fuse_method="mean",
#             fuse_topology_rule="union",
#             primary_candidate=0,
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="fused",
#             fused_sources=(
#                 ("coherence", THETA),
#                 ("coherence", ALPHA),
#                 ("coherence", BETA),
#                 ("wpli", ALPHA),
#             ),
#             fused_method="mean",
#         ),
#         model=ModelConfig(
#             family="fused_graph_bank_gnn",
#             backbone="gatv2",
#             graph_readout="attention",
#             emb_dim=64,
#             graph_bank_fusion_mode="summary_gated",
#         ),
#         aggregation=AggregationConfig(
#             strategy="none",
#             # attn_dim=64,
#             # train_max_instances_per_subject=100,
#         ),
#         train=subject_train,
#     )

#     # # ==================================================
#     # # Block 2: subject-level graph baselines
#     # # ==================================================
#     add(
#         "subject_connectivity_topk_gatv2",
#         level=LevelConfig(graph_level="subject"),
#         topology=TopologyConfig(
#             strategy="connectivity",
#             topology_metric="wpli",
#             topology_band=ALPHA,
#             topology_kwargs={"mode": "topk", "topk": 4},
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="connectivity",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#         ),
#         model=ModelConfig(
#             family="fixed_graph_gnn",
#             backbone="gatv2",
#             graph_readout="mean_max",
#             emb_dim=64,
#         ),
#         aggregation=AggregationConfig(strategy="none"),
#         train=subject_train,
#     )

#     add(
#         "subject_connectivity_mst_gatv2",
#         level=LevelConfig(graph_level="subject"),
#         topology=TopologyConfig(
#             strategy="connectivity",
#             topology_metric="wpli",
#             topology_band=ALPHA,
#             topology_kwargs={"mode": "mst"},
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="connectivity",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#         ),
#         model=ModelConfig(
#             family="fixed_graph_gnn",
#             backbone="gatv2",
#             graph_readout="mean_max",
#             emb_dim=64,
#         ),
#         aggregation=AggregationConfig(strategy="none"),
#         train=subject_train,
#     )

#     add(
#         "subject_connectivity_fixed_gatv2",
#         level=LevelConfig(graph_level="subject"),
#         topology=TopologyConfig(
#             strategy="fixed",
#             fixed_edge_pairs=fixed_edges,
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="connectivity",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#         ),
#         model=ModelConfig(
#             family="fixed_graph_gnn",
#             backbone="gatv2",
#             graph_readout="mean_max",
#             emb_dim=64,
#         ),
#         aggregation=AggregationConfig(strategy="none"),
#         train=subject_train,
#     )

#     # ==================================================
#     # Block 3: segment-level dense + MIL
#     # ==================================================
#     add(
#         "segment_node_only_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     add(
#         "segment_node_only_gated_mil",
#         level=LevelConfig(graph_level="segment"),
#         model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
#         aggregation=AggregationConfig(
#             strategy="gated_attention_mil",
#             attn_dim=64,
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     add(
#         "segment_connectivity_only_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         model=ModelConfig(
#             family="connectivity_only",
#             connectivity_encoder_type="cnn",
#             emb_dim=64,
#             dropout=0.2,
#         ),
#         connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     add(
#         "segment_dense_dual_branch_gated_mil",
#         level=LevelConfig(graph_level="segment"),
#         model=ModelConfig(
#             family="dense_dual_branch",
#             connectivity_encoder_type="cnn",
#             emb_dim=64,
#             dropout=0.2,
#         ),
#         connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#         aggregation=AggregationConfig(
#             strategy="gated_attention_mil",
#             attn_dim=64,
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )


#     add(
#         "segment_dense_dual_branch_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         model=ModelConfig(
#             family="dense_dual_branch",
#             connectivity_encoder_type="cnn",
#             emb_dim=64,
#             dropout=0.2,
#         ),
#         connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             attn_dim=64,
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     # ==================================================
#     # Block 4: segment-level graph + MIL
#     # ==================================================
#     add(
#         "segment_connectivity_fixed_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         topology=TopologyConfig(
#             strategy="fixed",
#             fixed_edge_pairs=fixed_edges,
#         ),
#         edge_weights=EdgeWeightConfig(            
#             strategy="connectivity",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#         ),
#         model=ModelConfig(
#             family="fixed_graph_gnn",
#             backbone="gcn",
#             graph_readout="mean",
#             emb_dim=64,
#         ),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     add(
#         "segment_connectivity_topk_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         topology=TopologyConfig(
#             strategy="connectivity",
#             topology_metric="wpli",
#             topology_band=ALPHA,
#             topology_kwargs={"mode": "topk", "topk": 4},
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="connectivity",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#         ),
#         model=ModelConfig(
#             family="fixed_graph_gnn",
#             backbone="gatv2",
#             graph_readout="attention",
#             emb_dim=64,
#         ),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             attn_dim=64,
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     add(
#         "segment_connectivity_mst_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         topology=TopologyConfig(
#             strategy="connectivity",
#             topology_metric="wpli",
#             topology_band=ALPHA,
#             topology_kwargs={"mode": "mst"},
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="connectivity",
#             edge_metric="coherence",
#             edge_band=ALPHA,
#         ),
#         model=ModelConfig(
#             family="fixed_graph_gnn",
#             backbone="gatv2",
#             graph_readout="attention",
#             emb_dim=64,
#         ),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             attn_dim=64,
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )


#     add(
#         "segment_graph_bank_mean_mil",
#         level=LevelConfig(graph_level="segment"),
#         topology=TopologyConfig(
#             strategy="fused_bank",
#             graph_bank_specs=default_graph_bank_specs(),
#             fuse_method="mean",
#             fuse_topology_rule="union",
#             primary_candidate=0,
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="fused",
#             fused_sources=(
#                 ("coherence", THETA),
#                 ("coherence", ALPHA),
#                 ("coherence", BETA),
#                 ("wpli", ALPHA),
#             ),
#             fused_method="mean",
#         ),
#         model=ModelConfig(
#             family="fused_graph_bank_gnn",
#             backbone="gatv2",
#             graph_readout="attention",
#             emb_dim=64,
#             graph_bank_fusion_mode="summary_gated",
#         ),
#         aggregation=AggregationConfig(
#             strategy="mean_mil",
#             attn_dim=64,
#             train_max_instances_per_subject=100,
#         ),
#         train=segment_train,
#     )

#     # ==================================================
#     # Block 5: macro-level dense + subject fusion
#     # ==================================================
#     add(
#         "macro_node_only_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )

#     add(
#         "macro_connectivity_only_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         model=ModelConfig(
#             family="connectivity_only",
#             connectivity_encoder_type="cnn",
#             emb_dim=64,
#             dropout=0.2,
#         ),
#         connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )

#     add(
#         "macro_dense_dual_branch_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         model=ModelConfig(
#             family="dense_dual_branch",
#             connectivity_encoder_type="cnn",
#             emb_dim=64,
#             dropout=0.2,
#         ),
#         connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )

#     # # ==================================================
#     # # Block 6: macro-level graph + subject fusion
#     # # ==================================================

#     add(
#         "macro_connectivity_mst_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         topology=TopologyConfig(
#             strategy="connectivity",
#             topology_metric="wpli",
#             topology_band=ALPHA,
#             topology_kwargs={"mode": "mst"},
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="normalized",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#             normalize_mode="absmax",
#         ),
#         model=ModelConfig(
#             family="dual_branch_graph",
#             backbone="gatv2",
#             graph_readout="mean_max",
#             emb_dim=64,
#             fusion_mode="gated",
#         ),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )

#     add(
#         "macro_connectivity_fixed_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         topology=TopologyConfig(
#             strategy="fixed",
#             fixed_edge_pairs=fixed_edges,
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="normalized",
#             edge_metric="wpli",
#             edge_band=ALPHA,
#             normalize_mode="absmax",
#         ),
#         model=ModelConfig(
#             family="dual_branch_graph",
#             backbone="gatv2",
#             graph_readout="mean_max",
#             emb_dim=64,
#             fusion_mode="gated",
#         ),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )

#     add(
#         "macro_graph_bank_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         topology=TopologyConfig(
#             strategy="fused_bank",
#             graph_bank_specs=default_graph_bank_specs(),
#             fuse_method="mean",
#             fuse_topology_rule="union",
#             primary_candidate=0,
#         ),
#         edge_weights=EdgeWeightConfig(
#             strategy="fused",
#             fused_sources=(
#                 ("coherence", THETA),
#                 ("coherence", ALPHA),
#                 ("coherence", BETA),
#                 ("wpli", ALPHA),
#             ),
#             fused_method="mean",
#         ),
#         model=ModelConfig(
#             family="fused_graph_bank_gnn",
#             backbone="gcn",
#             graph_readout="attention",
#             emb_dim=64,
#             graph_bank_fusion_mode="summary_gated",
#         ),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )


#     add(
#         "subject_legacy_linkx",
#         level=LevelConfig(graph_level="subject"),
#         model=ModelConfig(
#             family="legacy_encoder",
#             encoder_source="legacy",
#             encoder_type="linkx",
#             emb_dim=128,
#             dropout=0.2,
#             legacy_graph_pool="mean",
#             graph_readout="mean",
#         ),
#         aggregation=AggregationConfig(strategy="none"),
#         train=subject_train,
#     )

#     add(
#         "subject_legacy_sage",
#         level=LevelConfig(graph_level="subject"),
#         model=ModelConfig(
#             family="legacy_encoder",
#             encoder_source="legacy",
#             encoder_type="sage",
#             emb_dim=128,
#             hidden_dim=64,
#             dropout=0.2,
#             legacy_graph_pool="mean",
#             graph_readout="mean",
#         ),
#         aggregation=AggregationConfig(strategy="none"),
#         train=subject_train,
#     )

#     add(
#         "macro_legacy_linkx_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         model=ModelConfig(
#             family="legacy_encoder",
#             encoder_source="legacy",
#             encoder_type="linkx",
#             emb_dim=128,
#             dropout=0.2,
#             legacy_graph_pool="mean",
#             graph_readout="mean",
#         ),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )

#     add(
#         "macro_legacy_gnn_subject_fusion",
#         level=LevelConfig(
#             graph_level="macro",
#             macro_duration_sec=300.0,
#             feature_reduce="mean",
#             connectivity_reduce="mean",
#         ),
#         model=ModelConfig(
#             family="legacy_encoder",
#             encoder_source="legacy",
#             encoder_type="gnn",
#             emb_dim=128,
#             hidden_dim=64,
#             dropout=0.2,
#             legacy_graph_pool="mean",
#             graph_readout="mean",
#         ),
#         aggregation=AggregationConfig(strategy="subject_fusion"),
#         train=macro_train,
#     )
#     return specs
# # ---------------------------------------------------------------------
# # Ladder helpers
# # ---------------------------------------------------------------------
# # def default_graph_bank_specs(
# #     *,
# #     metrics: Sequence[str] = ("coherence",),
# #     bands: Sequence[int | str] = ("theta", "alpha", "beta"),
# # ) -> list[dict[str, Any]]:
# #     specs: list[dict[str, Any]] = []
# #     for metric in metrics:
# #         for band in bands:
# #             specs.append(
# #                 {
# #                     "name": f"{metric}_{band}",
# #                     "topology_mode": "connectivity",
# #                     "edge_weight_mode": "connectivity",
# #                     "connectivity_metric": metric,
# #                     "band": band,
# #                     "topology_kwargs": {"mode": "topk", "topk": 4},
# #                 }
# #             )
# #     return specs


# # def build_default_caueeg_ladder(
# #     *,
# #     dataset_path: str,
# #     h5_path: str,
# #     task: str = "dementia",
# #     output_root: str = "./results_caueeg",
# # ) -> list[CAUEEGExperimentSpec]:
# #     """
# #     Curated ladder rather than a naive full Cartesian product.

# #     The blocks are ordered so that simpler, lower-risk comparisons happen first.
# #     """
# #     base = CAUEEGExperimentSpec(
# #         name="base",
# #         task=task,
# #         dataset_path=dataset_path,
# #         h5_path=h5_path,
# #         output_root=output_root,
# #         feature_families=("relative_band_power", "hjorth", "statistical"),
# #         connectivity_metrics_to_load=("coherence",),
# #         connectivity_tensor=ConnectivityTensorConfig(metrics=("coherence",), bands=(2,)),
# #         train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),
# #     )

# #     specs: list[CAUEEGExperimentSpec] = []

# #     # --------------------------------------------------
# #     # Block 1: subject-level dense baselines
# #     # --------------------------------------------------
# #     specs.append(
# #         replace(
# #             base,
# #             name="subject_node_only",
# #             level=LevelConfig(graph_level="subject"),
# #             model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
# #             aggregation=AggregationConfig(strategy="none"),
# #         )
# #     )
# #     specs.append(
# #         replace(
# #             base,
# #             name="subject_connectivity_cnn",
# #             level=LevelConfig(graph_level="subject"),
# #             model=ModelConfig(family="connectivity_only", connectivity_encoder_type="cnn", emb_dim=64, dropout=0.2),
# #             aggregation=AggregationConfig(strategy="none"),
# #             connectivity_tensor=ConnectivityTensorConfig(metrics=("coherence",), bands=(1, 2, 3)),
# #         )
# #     )
# #     specs.append(
# #         replace(
# #             base,
# #             name="subject_dense_dual_branch",
# #             level=LevelConfig(graph_level="subject"),
# #             model=ModelConfig(family="dense_dual_branch", connectivity_encoder_type="cnn", emb_dim=64, dropout=0.2),
# #             aggregation=AggregationConfig(strategy="none"),
# #             connectivity_tensor=ConnectivityTensorConfig(metrics=("coherence",), bands=(1, 2, 3)),
# #         )
# #     )

# #     # --------------------------------------------------
# #     # Block 2: subject-level graph baselines
# #     # --------------------------------------------------
# #     specs.append(
# #         replace(
# #             base,
# #             name="subject_fixed_graph_gnn",
# #             level=LevelConfig(graph_level="subject"),
# #             topology=TopologyConfig(strategy="fixed"),
# #             edge_weights=EdgeWeightConfig(strategy="binary"),
# #             model=ModelConfig(family="fixed_graph_gnn", backbone="gcn", graph_readout="mean_max", emb_dim=64),
# #             aggregation=AggregationConfig(strategy="none"),
# #         )
# #     )
# #     specs.append(
# #         replace(
# #             base,
# #             name="subject_dual_branch_graph",
# #             level=LevelConfig(graph_level="subject"),
# #             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
# #             edge_weights=EdgeWeightConfig(strategy="connectivity", edge_metric="coherence", edge_band=2),
# #             model=ModelConfig(family="dual_branch_graph", backbone="gcn", graph_readout="mean_max", emb_dim=64, fusion_mode="gated"),
# #             aggregation=AggregationConfig(strategy="none"),
# #         )
# #     )

# #     # --------------------------------------------------
# #     # Block 3: segment-level + subject aggregation
# #     # --------------------------------------------------
# #     specs.append(
# #         replace(
# #             base,
# #             name="segment_fixed_graph_mean_mil",
# #             level=LevelConfig(graph_level="segment"),
# #             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
# #             edge_weights=EdgeWeightConfig(strategy="connectivity", edge_metric="coherence", edge_band=2),
# #             model=ModelConfig(family="fixed_graph_gnn", backbone="gcn", graph_readout="mean", emb_dim=64),
# #             aggregation=AggregationConfig(strategy="mean_mil", train_max_instances_per_subject=100, eval_max_instances_per_subject=None),
# #             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),
# #         )
# #     )
# #     specs.append(
# #         replace(
# #             base,
# #             name="segment_fixed_graph_gated_mil",
# #             level=LevelConfig(graph_level="segment"),
# #             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
# #             edge_weights=EdgeWeightConfig(strategy="connectivity", edge_metric="coherence", edge_band=2),
# #             model=ModelConfig(family="fixed_graph_gnn", backbone="gcn", graph_readout="attention", emb_dim=64),
# #             aggregation=AggregationConfig(strategy="gated_attention_mil", attn_dim=64, train_max_instances_per_subject=100),
# #             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),

# #         )
# #     )

# #     # --------------------------------------------------
# #     # Block 4: macro graphs + light subject fusion
# #     # --------------------------------------------------
# #     specs.append(
# #         replace(
# #             base,
# #             name="macro_dual_branch_subject_fusion",
# #             level=LevelConfig(graph_level="macro", macro_duration_sec=300.0, feature_reduce="mean", connectivity_reduce="mean"),
# #             topology=TopologyConfig(strategy="connectivity", topology_metric="coherence", topology_band=2, topology_kwargs={"mode": "topk", "topk": 4}),
# #             edge_weights=EdgeWeightConfig(strategy="normalized", edge_metric="coherence", edge_band=2, normalize_mode="absmax"),
# #             model=ModelConfig(family="dual_branch_graph", backbone="gcn", graph_readout="mean_max", emb_dim=64, fusion_mode="gated"),
# #             aggregation=AggregationConfig(strategy="subject_fusion", train_max_instances_per_subject=None),
# #             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),            
# #         )
# #     )

# #     # --------------------------------------------------
# #     # Block 5: fused graph bank
# #     # --------------------------------------------------
# #     bank_specs = default_graph_bank_specs(metrics=("coherence",), bands=(1, 2, 3))
# #     specs.append(
# #         replace(
# #             base,
# #             name="segment_graph_bank_gnn",
# #             level=LevelConfig(graph_level="segment"),
# #             topology=TopologyConfig(
# #                 strategy="fused_bank",
# #                 graph_bank_specs=bank_specs,
# #                 fuse_method="mean",
# #                 fuse_topology_rule="union",
# #                 primary_candidate=0,
# #             ),
# #             edge_weights=EdgeWeightConfig(strategy="fused", fused_sources=(("coherence", 1), ("coherence", 2), ("coherence", 3)), fused_method="mean"),
# #             model=ModelConfig(family="fused_graph_bank_gnn", backbone="gcn", graph_readout="attention", emb_dim=64, graph_bank_fusion_mode="summary_gated"),
# #             aggregation=AggregationConfig(strategy="gated_attention_mil", attn_dim=64, train_max_instances_per_subject=100),
# #             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),                    
# #         )
# #     )
# #     specs.append(
# #         replace(
# #             base,
# #             name="segment_dual_branch_graph_bank",
# #             level=LevelConfig(graph_level="segment"),
# #             topology=TopologyConfig(
# #                 strategy="fused_bank",
# #                 graph_bank_specs=bank_specs,
# #                 fuse_method="mean",
# #                 fuse_topology_rule="union",
# #                 primary_candidate=0,
# #             ),
# #             edge_weights=EdgeWeightConfig(strategy="fused", fused_sources=(("coherence", 1), ("coherence", 2), ("coherence", 3)), fused_method="mean"),
# #             model=ModelConfig(family="dual_branch_graph", backbone="gcn", graph_readout="attention", emb_dim=64, fusion_mode="gated", graph_bank_fusion_mode="summary_gated"),
# #             aggregation=AggregationConfig(strategy="gated_attention_mil", attn_dim=64, train_max_instances_per_subject=100),
# #             train=TrainConfig(batch_size=8, epochs=60, patience=20, lr=3e-3, weight_decay=1e-3, seed=42),                    

# #         )
# #     )

# #     return specs





# ###### new
#     # ==================================================
#     # Block 1: subject-level dense baselines
#     # ==================================================
#     # add(
#     #     "subject_node_only",
#     #     level=LevelConfig(graph_level="subject"),
#     #     model=ModelConfig(family="node_only", emb_dim=64, dropout=0.2),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     train=subject_train,
#     # )

#     # add(
#     #     "subject_connectivity_cnn",
#     #     level=LevelConfig(graph_level="subject"),
#     #     model=ModelConfig(
#     #         family="connectivity_only",
#     #         connectivity_encoder_type="cnn",
#     #         emb_dim=64,
#     #         dropout=0.2,
#     #     ),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#     #     train=subject_train,
#     # )

#     # add(
#     #     "subject_dense_dual_branch",
#     #     level=LevelConfig(graph_level="subject"),
#     #     model=ModelConfig(
#     #         family="dense_dual_branch",
#     #         connectivity_encoder_type="cnn",
#     #         emb_dim=64,
#     #         dropout=0.2,
#     #     ),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     connectivity_tensor=ConnectivityTensorConfig(metrics=("wpli",), bands=(DELTA, THETA, ALPHA, BETA, GAMMA)),
#     #     train=subject_train,
#     # )

#     # # ==================================================
#     # # Block 2: subject-level graph baselines
#     # # ==================================================
#     # add(
#     #     "subject_fixed_binary_gnn",
#     #     level=LevelConfig(graph_level="subject"),
#     #     topology=TopologyConfig(
#     #         strategy="fixed",
#     #         fixed_edge_pairs=fixed_edges,
#     #     ),
#     #     edge_weights=EdgeWeightConfig(strategy="binary"),
#     #     model=ModelConfig(
#     #         family="fixed_graph_gnn",
#     #         backbone="gcn",
#     #         graph_readout="mean_max",
#     #         emb_dim=64,
#     #     ),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     train=subject_train,
#     # )

#     # add(
#     #     "subject_connectivity_threshold_gnn",
#     #     level=LevelConfig(graph_level="subject"),
#     #     topology=TopologyConfig(
#     #         strategy="connectivity",
#     #         topology_metric="coherence",
#     #         topology_band=ALPHA,
#     #         topology_kwargs={"mode": "threshold", "threshold": 0.30},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(
#     #         strategy="normalized",
#     #         edge_metric="coherence",
#     #         edge_band=ALPHA,
#     #         normalize_mode="absmax",
#     #     ),
#     #     model=ModelConfig(
#     #         family="fixed_graph_gnn",
#     #         backbone="gcn",
#     #         graph_readout="mean_max",
#     #         emb_dim=64,
#     #     ),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     train=subject_train,
#     # )

#     # add(
#     #     "subject_feature_induced_gatv2",
#     #     level=LevelConfig(graph_level="subject"),
#     #     topology=TopologyConfig(
#     #         strategy="feature_induced",
#     #         similarity="cosine",
#     #         topology_kwargs={"mode": "topk", "topk": 4},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(strategy="binary"),
#     #     model=ModelConfig(
#     #         family="fixed_graph_gnn",
#     #         backbone="gatv2",
#     #         graph_readout="mean_max",
#     #         emb_dim=64,
#     #     ),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     train=subject_train,
#     # )

#     # add(
#     #     "subject_dual_branch_graph_topk",
#     #     level=LevelConfig(graph_level="subject"),
#     #     topology=TopologyConfig(
#     #         strategy="connectivity",
#     #         topology_metric="coherence",
#     #         topology_band=ALPHA,
#     #         topology_kwargs={"mode": "topk", "topk": 4},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(
#     #         strategy="connectivity",
#     #         edge_metric="coherence",
#     #         edge_band=ALPHA,
#     #     ),
#     #     model=ModelConfig(
#     #         family="dual_branch_graph",
#     #         backbone="gcn",
#     #         graph_readout="mean_max",
#     #         emb_dim=64,
#     #         fusion_mode="gated",
#     #     ),
#     #     aggregation=AggregationConfig(strategy="none"),
#     #     train=subject_train,
#     # )

#     # ==================================================
#     # Block 4: segment-level graph + MIL
#     # ==================================================

#     # add(
#     #     "segment_connectivity_threshold_gated_mil",
#     #     level=LevelConfig(graph_level="segment"),
#     #     topology=TopologyConfig(
#     #         strategy="connectivity",
#     #         topology_metric="coherence",
#     #         topology_band=ALPHA,
#     #         topology_kwargs={"mode": "threshold", "threshold": 0.30},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(
#     #         strategy="normalized",
#     #         edge_metric="coherence",
#     #         edge_band=ALPHA,
#     #         normalize_mode="absmax",
#     #     ),
#     #     model=ModelConfig(
#     #         family="fixed_graph_gnn",
#     #         backbone="gcn",
#     #         graph_readout="attention",
#     #         emb_dim=64,
#     #     ),
#     #     aggregation=AggregationConfig(
#     #         strategy="gated_attention_mil",
#     #         attn_dim=64,
#     #         train_max_instances_per_subject=100,
#     #     ),
#     #     train=segment_train,
#     # )

#     # add(
#     #     "segment_feature_induced_mean_mil",
#     #     level=LevelConfig(graph_level="segment"),
#     #     topology=TopologyConfig(
#     #         strategy="feature_induced",
#     #         similarity="cosine",
#     #         topology_kwargs={"mode": "topk", "topk": 4},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(strategy="binary"),
#     #     model=ModelConfig(
#     #         family="fixed_graph_gnn",
#     #         backbone="gatv2",
#     #         graph_readout="attention",
#     #         emb_dim=64,
#     #     ),
#     #     aggregation=AggregationConfig(
#     #         strategy="mean_mil",
#     #         attn_dim=64,
#     #         train_max_instances_per_subject=100,
#     #     ),
#     #     train=segment_train,
#     # )

#     # add(
#     #     "segment_dual_branch_graph_topk",
#     #     level=LevelConfig(graph_level="segment"),
#     #     topology=TopologyConfig(
#     #         strategy="connectivity",
#     #         topology_metric="coherence",
#     #         topology_band=ALPHA,
#     #         topology_kwargs={"mode": "topk", "topk": 4},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(
#     #         strategy="connectivity",
#     #         edge_metric="coherence",
#     #         edge_band=ALPHA,
#     #     ),
#     #     model=ModelConfig(
#     #         family="dual_branch_graph",
#     #         backbone="gcn",
#     #         graph_readout="attention",
#     #         emb_dim=64,
#     #         fusion_mode="gated",
#     #     ),
#     #     aggregation=AggregationConfig(
#     #         strategy="gated_attention_mil",
#     #         attn_dim=64,
#     #         train_max_instances_per_subject=100,
#     #     ),
#     #     train=segment_train,
#     # )

#     # # ==================================================
#     # # Block 6: macro-level graph + subject fusion
#     # # ==================================================
#     # add(
#     #     "macro_connectivity_topk_subject_fusion",
#     #     level=LevelConfig(
#     #         graph_level="macro",
#     #         macro_duration_sec=300.0,
#     #         feature_reduce="mean",
#     #         connectivity_reduce="mean",
#     #     ),
#     #     topology=TopologyConfig(
#     #         strategy="connectivity",
#     #         topology_metric="wpli",
#     #         topology_band=ALPHA,
#     #         topology_kwargs={"mode": "topk", "topk": 4},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(
#     #         strategy="normalized",
#     #         edge_metric="wpli",
#     #         edge_band=ALPHA,
#     #         normalize_mode="absmax",
#     #     ),
#     #     model=ModelConfig(
#     #         family="dual_branch_graph",
#     #         backbone="gatv2",
#     #         graph_readout="mean_max",
#     #         emb_dim=64,
#     #         fusion_mode="gated",
#     #     ),
#     #     aggregation=AggregationConfig(strategy="subject_fusion"),
#     #     train=macro_train,
#     # )


#     # add(
#     #     "macro_feature_induced_subject_fusion",
#     #     level=LevelConfig(
#     #         graph_level="macro",
#     #         macro_duration_sec=300.0,
#     #         feature_reduce="mean",
#     #         connectivity_reduce="mean",
#     #     ),
#     #     topology=TopologyConfig(
#     #         strategy="feature_induced",
#     #         similarity="cosine",
#     #         topology_kwargs={"mode": "topk", "topk": 4},
#     #     ),
#     #     edge_weights=EdgeWeightConfig(strategy="binary"),
#     #     model=ModelConfig(
#     #         family="fixed_graph_gnn",
#     #         backbone="gcn",
#     #         graph_readout="mean_max",
#     #         emb_dim=64,
#     #     ),
#     #     aggregation=AggregationConfig(strategy="subject_fusion"),
#     #     train=macro_train,
#     # )