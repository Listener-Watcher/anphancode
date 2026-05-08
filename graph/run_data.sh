#!/bin/bash
# # Run combinations of features, durations, overlaps, edges, and optional bands

# SCRIPT="graph/feature_extract.py"
# # SCRIPT="graph/bipolar_data.py"
# # SCRIPT="graph/fixed_graph.py"

SCRIPT="graph/master_builder.py"
# # DATASET="aheap"

# # Define feature list combinations (as Python-readable strings)
# FEATURE_LISTS=(
#    "['rbp','hjorth']"
#     # "['rbp','hjorth']" "['hjorth']" "['energies']"
# )

# # Define other parameter lists
DURATIONS=(10.0 20.0)
OVERLAPS=(0.5 0.8)
CHANNELS=("bi23" "mono")

# EDGE_METHODS=('coherence' 'corr' 'plv' 'pli' 'mi')
# # BANDS=("alpha" "none") 
# # "theta" "beta" "delta" "gamma")

# # Loop through combinations
# for feature in "${FEATURE_LISTS[@]}"; do
#   for electrode in "${CHANNELS[@]}"; do

#   # for dur in "${DURATIONS[@]}"; do
#   #   for overlap in "${OVERLAPS[@]}"; do
#     for edge in "${EDGE_METHODS[@]}"; do
#         # for band in "${BANDS[@]}"; do
#         #   if [ "$band" = "none" ]; then
#       echo "Running: feature=$feature, edge=$edge, CHANNELS=$electrode"
#       python "$SCRIPT" \
#         --feature_lists "$feature" \
#         --edge_methods "$edge"\
#         --electrode "$electrode"

#       #     else
#       #       echo "Running: dataset=$DATASET, feature=$feature, edge=$edge, band=$band, dur=$dur, overlap=$overlap"
#       #       python "$SCRIPT" \
#       #         --dataset "$DATASET" \
#       #         --feature_lists "$feature" \
#       #         --duration "$dur" \
#       #         --overlap "$overlap" \
#       #         --edge_methods "$edge" \
#       #         --band "$band"
#       #     fi
#       #   done
#       # done
#     done
#   done
# done

#!/bin/bash

# SCRIPT="/home/anphan/Documents/EEG_Project/graph/pre_mil.py"
# SUMMARY="/home/anphan/Documents/EEG_Project/AHEAP_data/pre_mil_results/all_pre_mil_summary.csv"
# BASE_OUT="/home/anphan/Documents/EEG_Project/AHEAP_data/pre_mil_results"
# mkdir -p "$BASE_OUT"

# for PT in \
#     '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_cleaned_feature_data/mono_rbphjorth_coherence_None_4_0.5/master_graph_data.pt'\
#     '/home/anphan/Documents/EEG_Project/AHEAP_data/all_master_cleaned_feature_data/bi23_rbphjorth_coherence_None_4_0.5/master_graph_data.pt'
# do
#     PARENT_NAME=$(basename "$(dirname "$PT")")
#     SAVE_DIR="${BASE_OUT}/${PARENT_NAME}"

#     python "$SCRIPT" \
#         --pt_path "$PT" \
#         --savepath "$SAVE_DIR" \
#         --summary_csv "$SUMMARY" \
#         --n_splits 10
# done

for dur in "${DURATIONS[@]}"; do
  for overlap in "${OVERLAPS[@]}"; do
    for electrode in "${CHANNELS[@]}"; do
      echo "Running: duration=$dur, overlap=$overlap, CHANNELS=$electrode"
      python "$SCRIPT" \
        --window_sec "$dur" \
        --overlap "$overlap"\
        --montage "$electrode"
    done
  done
done
