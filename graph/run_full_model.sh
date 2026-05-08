#!/bin/bash

dir_paths=(
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_pli_None'
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_plv_None'
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_corr_None'
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_corr_alpha'
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_plv_alpha'
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_pli_alpha'
'/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_coherence_None'
# '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_graph_data/mono_rbphjorth_coherence_alpha'
  )
# model_names=("GAT" "hybrid")
# readouts=("sum" "max" "mean")
patience_scores=(30 50 100)
# heads=(4 8)
# dims=(64 128 256)
# num_layers=(2 3 4)
lrs=(0.001 0.01)
# drop_outs=(0.1 0.3)
batchnorms=('True' 'False')
fakes=('True' 'False')
atts=('True' 'False')
class_sets=("all3")
hyperedge_weight_mode=("mean_abs_adj" "mean_adj" "ones")
fake_score_methods=("within_hyperedge_similarity" "hyperedge_smoothness" "node_to_hyperedge_consistency" "cross_region_contrast")


for weight in "${hyperedge_weight_mode[@]}"; do
  for dir in "${dir_paths[@]}"; do
    for patience_score in "${patience_scores[@]}"; do
      for batchnorm in "${batchnorms[@]}"; do
        for lr in "${lrs[@]}"; do
          for att in "${atts[@]}"; do
            for fake in "${fakes[@]}"; do
              for fake_score_method in "${fake_score_methods[@]}"; do
                echo "Running for dir: $dir"
                python graph/hypergraph.py \
                --saved_subject_dirs "$dir" \
                --class_set "all3" \
                --hyperedge_weight "$weight"\
                --patience_score "$patience_score" \
                --lr "$lr" \
                --use_fake_label "$fake" \
                --att "$att" \
                --batchnorm "$batchnorm"\
                --fake_score_method "$fake_score_method"
              done
            done
          done
        done
      done
    done
  done
done

echo "All runs completed."

# for model in "${model_names[@]}"; do
#   for dir in "${dir_paths[@]}"; do
#     for readout in "${readouts[@]}"; do
#       for patience_score in "${patience_scores[@]}"; do
#         for head in "${heads[@]}"; do
#           for dim in "${dims[@]}"; do
#             for num_layer in "${num_layers[@]}"; do
#               for drop_out in "${drop_outs[@]}"; do
#                 for lr in "${lrs[@]}"; do
#                   echo "Running for model_name: $model"
#                   python graph/main_graph.py \
#                   --dataset "aheap" \
#                   --saved_subject_dirs "$dir" \
#                   --model_name "$model" \
#                   --class_set "all3" \
#                   --patience_score "$patience_score" \
#                   --lr "$lr" \
#                   --heads "$head" \
#                   --drop_out "$drop_out" \
#                   --dim "$dim" \
#                   --num_layers "$num_layer" \
#                   --readout "$readout"
#                 done
#               done
#             done
#           done
#         done
#       done
#     done
#   done
# done

# echo "All runs completed."


