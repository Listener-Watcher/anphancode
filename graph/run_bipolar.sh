#!/bin/bash

# dir_paths=("/mnt/data/anphan/AHEAP_data/bipolar/rbp/rbp_coherence_delta_MST_None" "/mnt/data/anphan/AHEAP_data/bipolar/rbp/rbp_pli_delta_MST_None" "/mnt/data/anphan/AHEAP_data/bipolar/rbphjorth/rbphjorth_coherence_delta_MST_None")
# dir_paths=("/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_plv_None_MST_None" "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_pli_delta_MST_None")
# dir_paths=("/mnt/data/anphan/AHEAP_data/bipolar_update/rbphjorth/rbphjorth_coherence_theta_MST_0.2" 
# "/mnt/data/anphan/AHEAP_data/bipolar_update/rbphjorth/rbphjorth_coherence_alpha_MST_0.2" 
# "/mnt/data/anphan/AHEAP_data/bipolar_update/rbphjorth/rbphjorth_coherence_None_MST_0.2"
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_coherence_delta_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_coherence_alpha_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_corr_None_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_pli_alpha_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_pli_delta_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_pli_None_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_plv_delta_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_plv_alpha_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_plv_None_MST_0.2" 
#   # "/mnt/data/anphan/AHEAP_data/bipolar_update/rbp/rbp_coherence_None_MST_0.2"
# )
dir_paths=(

'bi23_rbphjorth_pli_None'
'mono_rbphjorth_pli_None'
'bi30_rbphjorth_pli_None'

# 'mono_rbphjorth_coherence_alpha'
# 'bi30_rbphjorth_corr_None'
# 'bi23_rbphjorth_plv_None'
# 'mono_rbphjorth_corr_None'
# 'bi30_rbphjorth_pli_alpha'
# 'mono_rbphjorth_corr_alpha'
# 'bi23_rbphjorth_coherence_alpha'
# 'bi23_rbphjorth_pli_alpha'
# 'bi23_rbphjorth_corr_alpha'
# 'bi23_rbphjorth_coherence_None'
# 'mono_rbphjorth_plv_alpha'
# 'mono_rbphjorth_pli_alpha'
# 'bi30_rbphjorth_coherence_alpha'
# 'bi23_rbphjorth_plv_alpha'
# 'bi30_rbphjorth_plv_None'
# 'bi30_rbphjorth_coherence_None'
# 'bi23_rbphjorth_corr_None'
# 'mono_rbphjorth_coherence_None'
# 'mono_rbphjorth_plv_None'
# 'bi30_rbphjorth_corr_alpha'
# 'bi30_rbphjorth_plv_alpha'
)

# model_names=("mlp"  "gnn" ) # "Chebconv" "EEGGraphConvNet")
# model_names=("gat" "hybrid") # "Chebconv" "EEGGraphConvNet")
# class_sets=("all3" "adhc")
mil_pool_types=("mean" "gated")
topologies=("fixed" "MST" "reconnect" "overlap") # "combined" 
edge_modes=("topology_binary" "topology_weighted")
base_ks=(100 150) #50
dims=(32 64 128)
# datasets=("aheap")


# # Nested loop: run each dir_path with all model_names
# for class in "${class_sets[@]}"; do
for dir in "${dir_paths[@]}"; do
  for mil_pool_type in "${mil_pool_types[@]}"; do
    for edge_mode in "${edge_modes[@]}"; do
      for topology in "${topologies[@]}"; do
        for base_k in "${base_ks[@]}"; do
          for dim in "${dims[@]}"; do
    # for model in "${model_names[@]}"; do
            echo "Data-path: $dir | edge_modes: $edge_mode | pool: $mil_pool_type | topology: $topology | base_k: $base_k | dim: $dim"
        # Sequential execution (recommended for single GPU)
            python graph/mil_dup.py --data_path "$dir" --edge_mode "$edge_mode" --mil_pool_type "$mil_pool_type" --topology "$topology" --base_k "$base_k" --dim "$dim"

      # python graph/mil_hypergraph.py --data_path "$dir"
      # python graph/mil_dup.py --model_name "$model"
      # python graph/subject_mil.py --model_name "$model"
          done
        done
      done
    done
  done
done

echo "All runs completed."
