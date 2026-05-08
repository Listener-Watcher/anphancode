#!/bin/bash
# Run combinations of features, durations, overlaps, edges, and optional bands

# SCRIPT="graph/viz/golden_test.py"
# SCRIPT="graph/viz/analysis.py"
SCRIPT="graph/viz/full_viz.py"

# Define feature list combinations (as Python-readable strings)
FEATURE_LISTS=(
    "['rbp','hjorth']"
) # 

DURATIONS=(1 8)
OVERLAPS=(0.9 0.5)
EDGE_METHODS=('coherence' 'plv' 'pli') # 'corr') #'mi' 'coherence' 
BANDS=("alpha" "none") # "alpha" "theta" "beta" "delta" "gamma"
CHANNELS=("mono" "bi23") #"bi23" 
# dir_paths=(
# 'mono_rbphjorth_pli_None'
# 'mono_rbphjorth_plv_None'
# 'mono_rbphjorth_corr_None'
# 'mono_rbphjorth_corr_alpha'
# 'mono_rbphjorth_plv_alpha'
# 'mono_rbphjorth_pli_alpha'
# 'mono_rbphjorth_coherence_None'
# 'mono_rbphjorth_coherence_alpha'
# )
# for dir in "${dir_paths[@]}"; do
#   echo "Running: dataset=$dir"
#   python "$SCRIPT" \
#     --pt_name "$dir" 
# done
# # Loop through combinations
for feature in "${FEATURE_LISTS[@]}"; do
  for electrode in "${CHANNELS[@]}"; do
    for dur in "${DURATIONS[@]}"; do
      for overlap in "${OVERLAPS[@]}"; do
        for edge in "${EDGE_METHODS[@]}"; do
          for band in "${BANDS[@]}"; do
            if [ "$band" = "none" ]; then
              echo "Running: dataset=$DATASET, electrode=$electrode, feature=$feature, edge=$edge, dur=$dur, overlap=$overlap"
              python "$SCRIPT" \
                --feature_lists "$feature" \
                --duration "$dur" \
                --overlap "$overlap" \
                --edge_methods "$edge" \
                --electrode "$electrode"
            else
              echo "Running: dataset=$DATASET, feature=$feature, edge=$edge, band=$band, dur=$dur, overlap=$overlap"
              python "$SCRIPT" \
                --feature_lists "$feature" \
                --duration "$dur" \
                --overlap "$overlap" \
                --edge_methods "$edge" \
                --band "$band" \
                --electrode "$electrode"
            fi
          done
        done
      done
    done
  done
done
