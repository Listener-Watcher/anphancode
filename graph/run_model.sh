# index, uuid, pci.bus_id, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
# 0, GPU-eb6c0716-025b-941f-3e3d-bab07b5a9539, 00000000:0A:00.0, NVIDIA GeForce RTX 2080 Ti, 58 MiB, 11264 MiB, 8 %
# 1, GPU-e0e428ea-6d61-746a-65d1-df5a0fbaef9d, 00000000:0B:00.0, NVIDIA GeForce RTX 2080 Ti, 9 MiB, 11264 MiB, 0 %
# 2, GPU-c299b4b1-6c49-39d5-e363-3cfee8d1e21c, 00000000:42:00.0, NVIDIA RTX 6000 Ada Generation, 2915 MiB, 49140 MiB, 70 %

segment_selection_strategies=("original_random_k" "all_raw" ) 
feature_family_sets=(
  "relative_band_power,hjorth"
  "relative_band_power,statistical"
  # "relative_band_power,statistical,wavelet_energy"
  # "relative_band_power,statistical,hjorth"
) 
# "original_random_k" "clean_random_k") # 
# "global_cluster_random_k" "global_cluster_proportional_random_k") #
basek=(10) #50
# encoders=('gat' 'hybrid' "linkx_cnn" "gnn_bank" ) #"mlp_node" "LINKX" "linkx_bank" 


# encoders=("linkx_bank" "gnn_bank") # "LINKX" "mlp_node")
#"LINKX" ("mlp_node" "LINKX" "cnn5" "linkx_cnn5")
encoders=("cnn_bank" "linkx_cnn_bank" "cnn5" "linkx_cnn5")
# ("gnn_bank" "edge_token" 'gat' 'hybrid')
# linkx_fused_bank' "sage", "gcn2", "h2gcn"]

candidate_fusion_modes=("concat") # "gated")
mil_pool_types=("mean") # "gated")
levels=("subject") # "macro") #
h5_paths=(
    "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
    # "/home/anphan/Documents/caueeg_sliding_mono_dementia_seed42_overlap50.h5"
)
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/no-overlap.py}"


for h5_path in "${h5_paths[@]}"; do
  for level in "${levels[@]}"; do
    for feature_family_set in "${feature_family_sets[@]}"; do
      for encoder_type in "${encoders[@]}"; do
        for mil_pool_type in "${mil_pool_types[@]}"; do
          for segment_selection_strategy in "${segment_selection_strategies[@]}"; do

            cmd=(
              env CUDA_VISIBLE_DEVICES=GPU-eb6c0716-025b-941f-3e3d-bab07b5a9539
              "${PYTHON_BIN}" "${SCRIPT_PATH}"
              --encoder_type "${encoder_type}"
              --segment_selection_strategy "${segment_selection_strategy}"
              --out_h5 "${h5_path}"
              --mil_pool_type "${mil_pool_type}"
              --level "${level}"
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
done

wait
echo "All jobs finished."