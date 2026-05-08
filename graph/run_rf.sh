
segment_selection_strategies=("original_random_k") # "all_raw" "all_clean" "clean_random_k") # 
basek=(10 "none")
h5_paths=(
    "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
    "/home/anphan/Documents/caueeg_sliding_mono_dementia_seed42_overlap50.h5"
    
    # "/home/anphan/Documents/caueeg_merged_sliding_random_trainonly.h5"
)
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/RF.py}"

for h5_path in "${h5_paths[@]}"; do
  for segment_selection_strategy in "${segment_selection_strategies[@]}"; do
    for base_k in "${basek[@]}"; do

      cmd=(
        env CUDA_VISIBLE_DEVICES=2
        "${PYTHON_BIN}" "${SCRIPT_PATH}"
        --segment_selection_strategy "${segment_selection_strategy}"
        --out_h5 "${h5_path}"
        --feature_families_str "relative_band_power,hjorth"

      )

      echo "========================================="
      echo "Command: ${cmd[*]}"
      echo "========================================="

      if [[ "${base_k}" != "none" ]]; then
        cmd+=(--base_k "${base_k}")
      fi

      "${cmd[@]}"
    done
  done
done

wait
echo "All jobs finished."