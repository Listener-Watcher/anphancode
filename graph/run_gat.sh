# index, uuid, pci.bus_id, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
# 0, GPU-eb6c0716-025b-941f-3e3d-bab07b5a9539, 00000000:0A:00.0, NVIDIA GeForce RTX 2080 Ti, 58 MiB, 11264 MiB, 8 %
# 1, GPU-e0e428ea-6d61-746a-65d1-df5a0fbaef9d, 00000000:0B:00.0, NVIDIA GeForce RTX 2080 Ti, 9 MiB, 11264 MiB, 0 %
# 2, GPU-c299b4b1-6c49-39d5-e363-3cfee8d1e21c, 00000000:42:00.0, NVIDIA RTX 6000 Ada Generation, 2915 MiB, 49140 MiB, 70 %

#-----------------------------------------------------------------------------------
#!/usr/bin/env bash
# set -euo pipefail

# segment_selection_strategies=("label_aligned_greedy_k" "global_cluster_proportional_random_k" "global_cluster_random_k") 
# basek=(10 25)
# encoders=("LINKX" "linkx_bank") # "gnn_bank" "gat") #  "mlp_node")
# backbones=("gatv2")
# candidate_fusion_modes=("concat") # "gated")
# mil_pool_types=("mean" "gated")
# levels=("segment") # "subject") # "macro") #
# # bag_aug_modes=("none" "mixed_real_pseudo" "cluster_view_ce" "mixed_realmultiview_pseudo") # "same_class_pseudo" "cluster_pseudo" ) # "multiview_consistency")

# h5_paths=(
#     "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
# )
# PYTHON_BIN="${PYTHON_BIN:-python}"
# SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/choose_segments.py}"
# # SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/entropy.py}"
# # SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/caueeg_removenoise_with_levels.py}"
#               # --bag_aug_mode "${bag_aug_mode}"
#     # for bag_aug_mode in "${bag_aug_modes[@]}"; do

# for mil_pool_type in "${mil_pool_types[@]}"; do
#   for encoder_type in "${encoders[@]}"; do
#     for h5_path in "${h5_paths[@]}"; do
#       for base_k in "${basek[@]}"; do
#         for segment_selection_strategy in "${segment_selection_strategies[@]}"; do

#           cmd=(
#             env CUDA_VISIBLE_DEVICES=GPU-eb6c0716-025b-941f-3e3d-bab07b5a9539
#             "${PYTHON_BIN}" "${SCRIPT_PATH}"
#             --encoder_type "${encoder_type}"
#             --segment_selection_strategy "${segment_selection_strategy}"
#             --base_k "${base_k}"
#             --out_h5 "${h5_path}"
#             --mil_pool_type "${mil_pool_type}"
#           )


#           echo "========================================="
#           echo "Command: ${cmd[*]}"
#           echo "========================================="

#           "${cmd[@]}"
#         done
#       done
#     done
#   done
# done

# wait
# echo "All jobs finished."


#=================================================

segment_selection_strategies=("original_random_k") #"all_raw") # "original_random_k" "label_aligned_greedy_k" "global_cluster_proportional_random_k" "global_cluster_random_k") 
basek=(10)
encoders=("linkx_bank") # "LINKX" "mlp_node"  "gat" "cnn5") # "gnn_bank") #  )
backbones=("gatv2")
candidate_fusion_modes=("concat") # "gated")
mil_pool_types=("mean") # "gated")
levels=("subject") # "macro") #
bag_aug_modes=("mixed_real_pseudo" "cluster_view_ce") # "mixed_realmultiview_pseudo") # "same_class_pseudo" "cluster_pseudo" ) # "multiview_consistency")

h5_paths=(
    "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
)
PYTHON_BIN="${PYTHON_BIN:-python}"
# SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/choose_segments.py}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/entropy.py}"
# SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/caueeg_removenoise_with_levels.py}"
              # --bag_aug_mode "${bag_aug_mode}"
    # for bag_aug_mode in "${bag_aug_modes[@]}"; do

for mil_pool_type in "${mil_pool_types[@]}"; do
  for encoder_type in "${encoders[@]}"; do
    for h5_path in "${h5_paths[@]}"; do
      for base_k in "${basek[@]}"; do
        for segment_selection_strategy in "${segment_selection_strategies[@]}"; do

          cmd=(
            env CUDA_VISIBLE_DEVICES=GPU-eb6c0716-025b-941f-3e3d-bab07b5a9539
            "${PYTHON_BIN}" "${SCRIPT_PATH}"
            --encoder_type "${encoder_type}"
            --level "segment"
            --segment_selection_strategy "${segment_selection_strategy}"
            --base_k "${base_k}"
            --out_h5 "${h5_path}"
            --mil_pool_type "${mil_pool_type}"
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