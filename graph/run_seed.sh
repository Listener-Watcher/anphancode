segment_selection_strategies=("original_random_k") #"label_aligned_greedy_k" "global_cluster_random_k") # "global_cluster_proportional_random_k" ) 
basek=(10)
encoders=("LINKX" "linkx_bank") #  "mlp_node"  "gat" "cnn5") # "gnn_bank") #  )
binary_pairs=("none") #"ad_hc" "ad_mci" "hc_mci")
h5_paths=(
    "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
)
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/choose_segments_updated.py}"
    # for bag_aug_mode in "${bag_aug_modes[@]}"; do
            # --bag_aug_mode "${bag_aug_mode}"
for h5_path in "${h5_paths[@]}"; do
  for base_k in "${basek[@]}"; do
    for encoder_type in "${encoders[@]}"; do
      for binary_pair in "${binary_pairs[@]}"; do
        for segment_selection_strategy in "${segment_selection_strategies[@]}"; do
          cmd=(
            env CUDA_VISIBLE_DEVICES=GPU-e0e428ea-6d61-746a-65d1-df5a0fbaef9d
            "${PYTHON_BIN}" "${SCRIPT_PATH}"
            --encoder_type "${encoder_type}"
            --segment_selection_strategy "${segment_selection_strategy}"
            --base_k "${base_k}"
            --out_h5 "${h5_path}"
            --binary_pair "${binary_pair}"
            --use_soft_targets
          )


          echo "========================================="
          echo "Command: ${cmd[*]}"
          echo "========================================="

          "${cmd[@]}"
        done
      done
    done
  done
done

wait
echo "All jobs finished."