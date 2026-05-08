from mil_utils import *
from mil_connectivity_v2 import (
    SubjectMILClassifierV2,
    collate_subject_bags_v2,
    build_conn_bank_for_segment,
    DEFAULT_CONN_CANDIDATES,
    DEFAULT_REGION_TO_CHANNELS,
)
from mil_full_std import *
import h5py
from master_builder import load_selected_groups, load_connectivity_metric


def attach_conn_bank_to_graphs(
    graphs,
    h5_path: str,
    subject_ids=None,
    channel_names=None,
    region_to_channels=None,
    candidate_specs=None,
):
    """
    Attach g.conn_bank with shape [K, 5, 5] to every graph.

    Each graph is expected to have:
        g.subject_id
        g.segment_id

    HDF5 connectivity usage:
      - wpli / pli / coherence: band-wise in H5, so we load selected bands
      - pearson / spearman: non-band-wise in H5
    """
    if region_to_channels is None:
        region_to_channels = DEFAULT_REGION_TO_CHANNELS
    if candidate_specs is None:
        candidate_specs = DEFAULT_CONN_CANDIDATES

    # ----------------------------
    # 1) Decide which subjects to load
    # ----------------------------
    if subject_ids is None:
        subject_ids = sorted({g.subject_id for g in graphs})

    # ----------------------------
    # 2) Load channel names once
    # ----------------------------
    # load_selected_groups returns channel_names + segment_id + requested connectivity groups
    # We only need one small call for metadata + segment alignment.
    meta_payload = load_selected_groups(
        h5_path,
        subject_ids=subject_ids,
        feature_families=[],
        connectivity_metrics=[],   # metadata only
    )

    if channel_names is None:
        # assume all subjects use same channel order
        first_sid = subject_ids[0]
        channel_names = meta_payload[first_sid]["channel_names"]

    # ----------------------------
    # 3) Load all connectivity arrays you need
    # ----------------------------
    # band-wise metrics -> load already sliced by band
    wpli_theta = load_connectivity_metric(h5_path, "wpli", subject_ids=subject_ids, band="theta")
    wpli_alpha = load_connectivity_metric(h5_path, "wpli", subject_ids=subject_ids, band="alpha")
    wpli_beta  = load_connectivity_metric(h5_path, "wpli", subject_ids=subject_ids, band="beta")

    pli_theta  = load_connectivity_metric(h5_path, "pli", subject_ids=subject_ids, band="theta")
    pli_alpha  = load_connectivity_metric(h5_path, "pli", subject_ids=subject_ids, band="alpha")

    coh_alpha  = load_connectivity_metric(h5_path, "coherence", subject_ids=subject_ids, band="alpha")

    # non-band metrics
    pearson_all  = load_connectivity_metric(h5_path, "pearson", subject_ids=subject_ids, band=None)
    spearman_all = load_connectivity_metric(h5_path, "spearman", subject_ids=subject_ids, band=None)

    # ----------------------------
    # 4) Build segment-id -> row-index map per subject
    # ----------------------------
    segrow_by_subject = {}
    for sid in subject_ids:
        seg_ids = np.asarray(meta_payload[sid]["segment_id"], dtype=np.int64)
        segrow_by_subject[sid] = {int(seg_id): row_idx for row_idx, seg_id in enumerate(seg_ids)}

    # ----------------------------
    # 5) Attach conn_bank to each graph
    # ----------------------------
    for g in graphs:
        sid = g.subject_id
        seg_id = int(getattr(g, "segment_id"))

        if sid not in segrow_by_subject:
            raise KeyError(f"Subject {sid!r} not found in loaded H5 payload.")

        if seg_id not in segrow_by_subject[sid]:
            raise KeyError(
                f"segment_id={seg_id} for subject {sid!r} not found in H5 segment_id list."
            )

        row = segrow_by_subject[sid][seg_id]

        connectivity_sources = {
            "wpli": {
                "theta": wpli_theta[sid]["values"][row],   # [19, 19]
                "alpha": wpli_alpha[sid]["values"][row],   # [19, 19]
                "beta":  wpli_beta[sid]["values"][row],    # [19, 19]
            },
            "pli": {
                "theta": pli_theta[sid]["values"][row],    # [19, 19]
                "alpha": pli_alpha[sid]["values"][row],    # [19, 19]
            },
            "coherence": {
                "alpha": coh_alpha[sid]["values"][row],    # [19, 19]
            },
            "pearson": {
                # pearson is not band-wise in your H5, but your helper accepts fallback usage
                "alpha": pearson_all[sid]["values"][row],  # [19, 19]
            },
            "spearman": {
                # spearman is not band-wise in your H5, but same fallback idea
                "alpha": spearman_all[sid]["values"][row], # [19, 19]
            },
        }

        conn_bank = build_conn_bank_for_segment(
            connectivity_sources=connectivity_sources,
            channel_names=channel_names,
            region_to_channels=region_to_channels,
            candidate_specs=candidate_specs,
        )  # [K, 5, 5]

        g.conn_bank = torch.as_tensor(conn_bank, dtype=torch.float32)

    return graphs

def _stable_int_from_string(x: str) -> int:
    """
    Stable integer hash from a string.
    Do NOT use Python's built-in hash(), because it is randomized across runs.
    """
    s = str(x).encode("utf-8")
    return int(hashlib.md5(s).hexdigest()[:8], 16)

def _move_to_cpu(obj: Any) -> Any:
    """
    Recursively move tensors in nested structures to CPU so checkpoints are portable.
    """
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _move_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_move_to_cpu(v) for v in obj)
    return obj



if __name__ == "__main__":
    candidate_specs = DEFAULT_CONN_CANDIDATES
    region_map = DEFAULT_REGION_TO_CHANNELS
    dataset = config.DATASET
    data_dir = config.DIR_DATA
    tsv_path = config.TSV_PATH
    class_set ="all3" 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    num_classes, class_labels, class_names = get_class(class_set, dataset)
    data_paths, labels, sub_id_list = aheap_get_paths(data_dir, tsv_path, class_set)
    # data_paths, labels, sub_id_list = data_paths[:15]+data_paths[40:55]+data_paths[75:], labels[:15]+labels[40:55]+labels[75:], sub_id_list[:15]+sub_id_list[40:55]+sub_id_list[75:]
    print("data_paths length = ", len(data_paths), "unique label =",len(np.unique(labels)))
    print("-- num_classes =", num_classes, "-- class_labels =", class_labels, "-- class_names =", class_names)
    root_path = "/home/anphan/Documents/EEG_Project/AHEAP_data/"

    k = 5
    val_ratio = 0.15
    # split_seeds = [15]
    split_seeds = [15, 42, 100]
    batch_size_train=8
    batch_size_val=4
    batch_size_test = 4
    lr=3e-4
    weight_decay=5e-4
    epochs=300
    patience=20

    parser = argparse.ArgumentParser(description="Run script with directory path and model name")
    parser.add_argument("--all_data_path", type=str, required=True, help="all_data_path")
    parser.add_argument("--mil_pool_type", type=str, default="mean", required=False, help="mil_pool_type")
    parser.add_argument("--edge_mode", type=str, default="topology_weighted",required=False, help="edge_mode")
    parser.add_argument("--topology", type=str, default="fixed", required=False, help="topology")
    parser.add_argument("--base_k", type=int, default=None, required=False, help="base_k")
    parser.add_argument("--dim", type=int,  default=32, required=False, help="dim")
    parser.add_argument("--feature_families", type=str, default="relative_band_power")   # e.g. "relative_band_power,hjorth"
    parser.add_argument("--connectivity_metric", type=str, default="pli")
    parser.add_argument("--connectivity_band", type=int, default=2)
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="gnn",
        choices=["gnn", "LINKX", "mlp_node", "sage", "gcn2", "h2gcn"],
    )
    parser.add_argument("--graph_pool", type=str, default="mean", choices=["mean", "max", "add"])
    parser.add_argument("--sage_layers", type=int, default=2)
    parser.add_argument("--gcn2_layers", type=int, default=8)
    parser.add_argument("--gcn2_alpha", type=float, default=0.1)
    parser.add_argument("--gcn2_theta", type=float, default=0.5)
    parser.add_argument("--gcn2_shared_weights", action="store_true")
    parser.add_argument("--gcn2_use_edge_weight", action="store_true")
    parser.add_argument("--h2gcn_layers", type=int, default=2)
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="subject_wise",
        choices=["none", "subject_wise", "channel_wise"],
    )
    parser.add_argument(
        "--align_mode",
        type=str,
        default="none",
        choices=["none", "ea", "ra"],
        help="Alignment mode: none, Euclidean alignment (ea), or Riemannian alignment (ra).",
    )
    args = parser.parse_args()
    if args.align_mode == "none":
        edge_source = "connectivity"
    else:
        edge_source = "aligned_adj"
    all_data_path = args.all_data_path
    topology = args.topology#"fixed"
    mil_pool_type= args.mil_pool_type #"mean" #"mean"
    edge_mode = args.edge_mode #"topology_binary"
    dim = args.dim
    feature_families = [x.strip() for x in args.feature_families.split(",") if x.strip()]
    feature_name_list =  args.feature_families.replace(",", "_")
    feature_name_list =  feature_name_list.replace("relative_band_power", "RBP")
    if args.encoder_type in ["LINKX", "mlp_node"]:
        start_epoch=10
    else:
        lr = 1e-4
        weight_decay = 1e-4
        epochs = 120
        start_epoch = 15
        patience = 25
        dropout = 0.1
        dim = 16   # or 32, not larger first
        graph_emb_dim = dim * 2
        attn_dim = dim * 2

    gnn_hidden_dim=dim
    graph_emb_dim=dim*2
    attn_dim=dim*2
    dropout=0.3
    node_hidden_dims=(dim*2, dim)
    edge_hidden_dims=(dim*2, dim)
    branch_emb_dim=dim
    base_k=args.base_k
    max_k_per_subject=300
    standardize_features=True

    save_path = os.path.join(root_path,'result_Apr12_zscoredata_residualconn')
    os.makedirs(save_path,exist_ok = True)
    # all_data_path = '/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5'
    last_part = os.path.basename(all_data_path)
    parts = last_part.split('_')
    
    if "mono" in parts:
        channel_names = config.MONO_CHANNELS
        fixed_edges = config.MONOFIXEDGES
        channel_name = "mono"
    elif "bi23" in parts:
        channel_names = config.bi23_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi23"

    elif "bi30" in parts:
        channel_names = config.bi30_channel_names
        fixed_edges = fixed_edges_share_electrode(channel_names)
        channel_name = "bi30"


    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{topology}_{feature_families[1]}_{args.connectivity_metric}"
    check_term = f"{args.encoder_type}_{mil_pool_type}_{args.norm_mode}_{channel_name}_{topology}_{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_{args.base_k}_{args.dim}"
    
    for d in os.listdir(save_path):
        path = os.path.join(save_path, d)
        if os.path.isdir(path):
            # print(path)
            last_part = os.path.basename(path)
            # if check_term in last_part:
            if result_already_exists(save_path, check_term, "overall_summary_test.csv"):
                import sys
                print(f"Already run: {check_term} skipped!")
                sys.exit(0) 

    # folder_name = f"{timestamp}_{args.encoder_type}_{mil_pool_type}_{channel_name}_{topology}_{feature_name_list}_{args.connectivity_metric}_{args.connectivity_band}_{args.base_k}_{args.dim}"
    folder_name = f"{timestamp}_{check_term}"
    output_dir = os.path.join(save_path, folder_name)
    os.makedirs(output_dir,exist_ok = True)
    log_path = os.path.join(output_dir, f"log.txt")


    print("File found! Processing...")
    with open(log_path, "w") as f:
        f.write(f"data source {all_data_path}\n")
        f.write(f"k {k}, val_ratio {val_ratio}, split_seeds {split_seeds}\n")
        f.write(f"norm_mode {args.norm_mode}\n")
        f.write(f"\n")

        f.write(f"topology: {topology}, fixed_edges: {fixed_edges}\n")
        f.write(f"feature_families: {args.feature_families}\nconnectivity_metric: {args.connectivity_metric}, connectivity_band: {args.connectivity_band}\n")
        f.write(f"sample segments in subject: base_k={base_k}, max_k_per_subject={max_k_per_subject} \n")
        f.write(f"\n")

        f.write(f"model_name: {args.encoder_type}, pooling: {mil_pool_type}, edge_mode: {edge_mode}\n")
        f.write(f"update early stopping method: start_epoch={start_epoch}, min_delta=1e-3, top_k=5 \n")
        f.write(f"batch_size_train {batch_size_train}, batch_size_val {batch_size_val}, batch_size_test {batch_size_test}\n")
        f.write(f"lr {lr}, weight_decay {weight_decay}, epochs {epochs}, patience {patience}\n")
        f.write(f"readout {args.graph_pool}, class_set {class_set}, standardize_features {standardize_features}\n")
        f.write(f"dim {dim} \n gnn_hidden_dim={gnn_hidden_dim} \n graph_emb_dim={graph_emb_dim} \n attn_dim={attn_dim} \n")
        f.write(f"dropout={dropout}\n node_hidden_dims={node_hidden_dims} \n edge_hidden_dims={edge_hidden_dims}\n branch_emb_dim={branch_emb_dim}\n")
    


    result_all = []
    fold_metric_rows = []
    pred_rows = []
    payload = load_h5_payload_for_subjects(
        h5_path=all_data_path,
        subject_ids=sub_id_list,   # load all subjects
        feature_families=feature_families,
        connectivity_metrics=[args.connectivity_metric] if args.connectivity_metric is not None else [],
        connectivity_band=args.connectivity_band,
        load_raw_for_alignment=(args.align_mode != "none"),
        load_bad_segment_flag=True,
    )

    payload = filter_payload_bad_windows_in_place(payload)

    if args.norm_mode in {"none", "subject_wise", "channel_wise"} and args.align_mode == "none":
        payload, global_norm_stats = normalize_payload_feature_families(
            payload,
            feature_families=feature_families,
            norm_mode=args.norm_mode,
            in_place=True,
        )

    # graphs = build_graphs_from_master_h5(
    #     h5_path=all_data_path,
    #     feature_families=feature_families,
    #     connectivity_metric=args.connectivity_metric,
    #     connectivity_band=args.connectivity_band,
    #     subject_ids=sub_id_list,
    #     standardize_features=standardize_features,
    #     node_feature_mode="selected_features",
    #     connectivity_mode="selected_metric",
    # )
    all_result_rows = []
    for seed in split_seeds:
        set_global_seed(seed)

        print(f"\n========== Split seed: {seed} ==========")
        seed_dir = os.path.join(output_dir, f"seed{seed}")
        os.makedirs(seed_dir,exist_ok = True)
        with open(log_path, "a") as f:
            f.write(f"======================================\n")
            f.write(f"Seed random = {seed}\n")

        all_folds = balanced_kfold_split(sub_id_list, labels, seed, k)
        check_dir = os.path.join(f"{seed_dir}/checkpoints")
        os.makedirs(check_dir,exist_ok=True)
        cv_subject_embeddings = os.path.join(f"{seed_dir}/cv_subject_embeddings")
        os.makedirs(cv_subject_embeddings,exist_ok=True)

        all_fold_data = []
        for i, test_subjects in enumerate(all_folds):
            print(f"\n========== Fold: {i} ==========")
            with open(log_path, "a") as f:
                f.write(f"\n========== Fold: {i} ==========\n")

            tsne_fold = os.path.join(f"{seed_dir}/tsne_fold{i}")
            os.makedirs(tsne_fold,exist_ok=True)
            print(test_subjects)
            test_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in test_subjects]

            train_subjects = [sub_id for sub_id in sub_id_list if sub_id not in test_subjects]
            # train_paths = [path for sub_id, path in zip(sub_id_list, data_paths) if sub_id in train_subjects]
            train_labels = [label for sub_id, label in zip(sub_id_list, labels) if sub_id in train_subjects]
            subject_label_map = dict(zip(train_subjects, train_labels))
       
            new_train_subjects, val_subjects = stratified_split_subjects(
                train_subjects, subject_label_map, val_ratio, seed
            )
   
            print(f"# Train_subjects = {len(new_train_subjects)} | # Validation subjects = {len(val_subjects)}")

            # train_graphs = [g for g in graphs if g.subject_id in new_train_subjects]
            # val_graphs   = [g for g in graphs if g.subject_id in val_subjects]
            # test_graphs  = [g for g in graphs if g.subject_id in set(test_subjects)]

            train_graphs = build_graphs_from_payload(payload, subject_ids=new_train_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source =edge_source)
            val_graphs   = build_graphs_from_payload(payload, subject_ids=val_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source = edge_source)
            test_graphs  = build_graphs_from_payload(payload, subject_ids=test_subjects, feature_families=feature_families, connectivity_metric=args.connectivity_metric,edge_source =edge_source)

            train_graphs = attach_summary_features_to_graphs(train_graphs)
            val_graphs   = attach_summary_features_to_graphs(val_graphs)
            test_graphs  = attach_summary_features_to_graphs(test_graphs)

            train_graphs = attach_conn_bank_to_graphs(
                train_graphs,
                h5_path=all_data_path,
                subject_ids=new_train_subjects,
                channel_names=channel_names,
                region_to_channels=region_map,
                candidate_specs=candidate_specs,
                )
            val_graphs   = attach_conn_bank_to_graphs(
                val_graphs, 
                h5_path=all_data_path,
                subject_ids=val_subjects,
                channel_names=channel_names,
                region_to_channels=region_map,
                candidate_specs=candidate_specs,
                )
            test_graphs  = attach_conn_bank_to_graphs(
                test_graphs, 
                h5_path=all_data_path,
                subject_ids=test_subjects,
                channel_names=channel_names,
                region_to_channels=region_map,
                candidate_specs=candidate_specs,
                )

            g = train_graphs[0]
            print(hasattr(g, "edge_attr"))
            print(g.edge_attr[:10] if hasattr(g, "edge_attr") and g.edge_attr is not None else None)
            print(hasattr(g, "edge_weight"))
            print(g.edge_weight[:10] if hasattr(g, "edge_weight") and g.edge_weight is not None else None)


            summary_input_dim = train_graphs[0].summary_feat.numel()
            print("summary_input_dim =", summary_input_dim)

            if base_k is None:
                train_dataset = SubjectBagGraphDataset(
                    train_graphs,
                    max_segments_per_subject=None,   # good default for training memory
                    train=True,
                )

                val_dataset = SubjectBagGraphDataset(
                    val_graphs,
                    max_segments_per_subject=None, # use all segments at validation if memory allows
                    train=False,
                )


                test_dataset = SubjectBagGraphDataset(
                    test_graphs,
                    max_segments_per_subject=None, # use all segments at validation if memory allows
                    train=False,
                )

            else:
                train_dataset = LabelAwareSubjectBagDataset(
                    train_graphs,
                    train=True,
                    base_k=base_k,                      # reference k
                    k_by_label=None,                # auto-compute from class counts
                    target_segments_per_class=None, # defaults to majority_class_subjects * base_k
                    max_k_per_subject=max_k_per_subject,           # optional cap
                    seed=seed,
                    return_segment_ids=True,        # optional, useful for debugging
                )

                val_dataset = LabelAwareSubjectBagDataset(
                    val_graphs,
                    train=False,
                    eval_k_per_subject=None,        # None = use all val segments
                    seed=seed,
                )

                test_dataset = LabelAwareSubjectBagDataset(
                    test_graphs,
                    train=False,
                    eval_k_per_subject=None,        # None = use all test segments
                    seed=seed,
                )

            print("Train subject class counts:", np.bincount(train_dataset.subject_labels, minlength=num_classes))
            print("Val subject class counts:", np.bincount(val_dataset.subject_labels, minlength=num_classes))
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            
            input_model = SubjectMILClassifierV2(
                num_node_features=train_dataset.num_node_features,
                num_classes=num_classes,
                num_nodes=train_dataset.num_nodes,
                encoder_type=args.encoder_type,
                graph_emb_dim=dim * 2,
                dropout=dropout,
                graph_pool=args.graph_pool,
                gnn_hidden_dim=dim,
                sage_layers=args.sage_layers,
                gcn2_layers=args.gcn2_layers,
                gcn2_alpha=args.gcn2_alpha,
                gcn2_theta=args.gcn2_theta,
                gcn2_shared_weights=args.gcn2_shared_weights,
                gcn2_use_edge_weight=args.gcn2_use_edge_weight,
                h2gcn_layers=args.h2gcn_layers,
                attn_dim=dim * 2,
                num_conn_candidates=len(DEFAULT_CONN_CANDIDATES),
                num_regions=5,
                conn_emb_dim=64,
                lambda_mask=1e-3,
            ).to(device)
            class_weights = compute_class_weights_from_subjects(
                subject_labels=train_dataset.subject_labels,
                num_classes=num_classes,
            ).to(device)
            print("class_weights", class_weights)
            # class_weights tensor([1.0000, 0.8148, 1.2941])

            criterion = nn.CrossEntropyLoss()
            # criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.AdamW(input_model.parameters(), lr=lr, weight_decay=weight_decay)

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size_train,
                shuffle=True,
                collate_fn=collate_subject_bags_v2,
                num_workers=0,
                pin_memory=True,
            )

            batch = next(iter(train_loader))
            print(batch["pyg_batch"])
            # print(batch["summary_x"].shape)   # [num_graphs_in_batch, summary_input_dim]
            print(batch["bag_sizes"])
            print(batch["labels"])
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size_val,
                shuffle=False,
                collate_fn=collate_subject_bags_v2,
                num_workers=0,
                pin_memory=True,
            )

            model, val_metrics, history, best_state = fit_mil_baseline(
                input_model,
                train_loader,
                val_loader,
                optimizer,
                criterion,
                device,
                epochs,
                patience,
                save_path=f"{check_dir}/best_mil_model_fold{i}.pt",
                start_epoch=start_epoch,     # warmup: do not count patience before epoch 30
                min_delta=1e-3,     # require at least this much val-loss improvement
                top_k=5,            # keep 5 lowest-loss checkpoints
                verbose=False,
            )

            checkpoint = torch.load(f"{check_dir}/best_mil_model_fold{i}.pt", map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Best epoch:", checkpoint["epoch"])
            print("Best val metrics:", checkpoint["best_val_macro_f1"])


            with open(log_path, "a") as f:
                f.write("Final validation metrics:\n")
                f.write(f"Accuracy:           {val_metrics['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Macro-F1:           {val_metrics['macro_f1']:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{val_metrics['conf_matrix']}\n")

            print("\nFinal validation metrics:")
            print(f"Accuracy:           {val_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {val_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1:           {val_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(val_metrics["conf_matrix"])

            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size_test,
                shuffle=False,
                collate_fn=collate_subject_bags_v2,
                num_workers=0,
                pin_memory=True,
            )

            criterion = nn.CrossEntropyLoss()
            test_metrics = evaluate(model, test_loader, criterion, device)

            with open(log_path, "a") as f:
                f.write("Final test metrics:\n")
                f.write(f"Accuracy:           {test_metrics['accuracy']:.4f}\n")
                f.write(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}\n")
                f.write(f"Macro-F1:           {test_metrics['macro_f1']:.4f}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{test_metrics['conf_matrix']}\n")


            print("\nFinal test metrics:")
            print(f"Accuracy:           {test_metrics['accuracy']:.4f}")
            print(f"Balanced Accuracy:  {test_metrics['balanced_accuracy']:.4f}")
            print(f"Macro-F1:           {test_metrics['macro_f1']:.4f}")
            print("Confusion Matrix:")
            print(test_metrics["conf_matrix"])


            all_result_rows.append({
                "split_seed": seed,
                "fold": i,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
            })

            train_subject_rows_f, val_subject_rows_f, test_subject_rows_f = save_fold_subject_embeddings(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    device=device,
                    fold_idx=i,
                    save_dir=cv_subject_embeddings
                )

            all_fold_data.append({
                "fold": i,
                "train_rows": train_subject_rows_f,
                "val_rows": val_subject_rows_f,
                "test_rows": test_subject_rows_f,
            })
            # plot_subject_embeddings_tsne(train_subject_rows_f, "subject", "train", tsne_fold, color_by="label", title="Train Subject Embeddings by True Class")
            # plot_subject_embeddings_tsne(val_subject_rows_f, "subject", "val", tsne_fold, color_by="label", title="Validation Subject Embeddings by True Class")
            # plot_subject_embeddings_tsne(test_subject_rows_f, "subject", "test", tsne_fold, color_by="label", title="Test Subject Embeddings by True Class")
            fold_metric_rows.append(metrics_to_row(test_metrics, seed, i, "test"))
            pred_rows.extend(predictions_to_rows(test_metrics, seed, i, "test", num_classes))

            train_seg_rows = collect_segment_embeddings(model, train_loader, device)
            val_seg_rows = collect_segment_embeddings(model, val_loader, device)
            test_seg_rows  = collect_segment_embeddings(model, test_loader, device)
            # plot_subject_embeddings_tsne(train_seg_rows, "segment", "train", tsne_fold, color_by="subject", title="Train Segment Embeddings by True Class")
            # plot_subject_embeddings_tsne(val_seg_rows, "segment", "val", tsne_fold, color_by="subject", title="Validation Segment Embeddings by True Class")
            plot_subject_embeddings_tsne(test_seg_rows, "segment", "test", tsne_fold, color_by="subject", title="Test Segment Embeddings by True Class")

            fingerprint_stats_train = segment_fingerprint_metrics(train_seg_rows)
            fingerprint_stats_test  = segment_fingerprint_metrics(test_seg_rows)

            with open(log_path, "a") as f:
                f.write(f"TRAIN fingerprint stats: {fingerprint_stats_train}\n")
                f.write(f"TEST  fingerprint stats: {fingerprint_stats_test}\n")

        with open(f"{cv_subject_embeddings}/all_fold_subject_rows.pkl", "wb") as f:
            pickle.dump(all_fold_data, f)
        all_fold_data = load_all_fold_data(f"{cv_subject_embeddings}/all_fold_subject_rows.pkl")
        print(len(all_fold_data))
        print(all_fold_data[0].keys())

        aligned_oof_rows = align_oof_test_embeddings_across_folds(
            all_fold_data,
            reference_fold=0
        )

        class_dict = {
            0: "HC",
            1: "AD",
            2: "FTD",
        }
        plot_aligned_subject_embeddings_umap(
            aligned_oof_rows,
            class_names=class_dict,
            title="Out-of-Fold Subject Embeddings",
            annotate_subject_ids=True,
            save_path=f"{seed_dir}/plot_aligned_subject_embeddings_umap.png"
        )
    fold_metrics_df = pd.DataFrame(fold_metric_rows)
    fold_metrics_path = os.path.join(output_dir, "fold_metrics_all_seeds.csv")
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(output_dir, "subject_predictions_all_seeds.csv")
    pred_df.to_csv(pred_path, index=False)

    test_summary_by_split = (
        fold_metrics_df[fold_metrics_df["split"] == "test"]
        .groupby("split_seed")[["accuracy", "balanced_accuracy", "macro_f1"]]
        .mean()
        .reset_index()
    )
    test_summary_by_split.to_csv(
        os.path.join(output_dir, "test_summary_by_split_seed.csv"),
        index=False
    )

    overall_summary = (
        fold_metrics_df[fold_metrics_df["split"] == "test"][["accuracy", "balanced_accuracy", "macro_f1"]]
        .agg(["mean", "std"])
    )
    print(overall_summary)
    overall_summary.to_csv(
        os.path.join(output_dir, "overall_summary_test.csv")
    )