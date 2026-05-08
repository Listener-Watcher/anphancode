
# index, uuid, pci.bus_id, name, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
# 0, GPU-eb6c0716-025b-941f-3e3d-bab07b5a9539, 00000000:0A:00.0, NVIDIA GeForce RTX 2080 Ti, 58 MiB, 11264 MiB, 8 %
# 1, GPU-e0e428ea-6d61-746a-65d1-df5a0fbaef9d, 00000000:0B:00.0, NVIDIA GeForce RTX 2080 Ti, 9 MiB, 11264 MiB, 0 %
# 2, GPU-c299b4b1-6c49-39d5-e363-3cfee8d1e21c, 00000000:42:00.0, NVIDIA RTX 6000 Ada Generation, 2915 MiB, 49140 MiB, 70 %

training_approaches=("segment_k" "segment_all" )
segment_selection_strategies=("original_random_k") 
encoders=("LINKX" "mlp_node" "linkx_bank")
h5_paths=(
    "/home/anphan/Documents/caueeg_randomcrop_mono_dementia_seed42.h5"
)
feature_families_strs=(
"relative_band_power,hjorth"
# "relative_band_power,statistical"
)
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_PATH="${SCRIPT_PATH:-/home/anphan/Downloads/GNN_2026-main/graph/caueeg_linkx_train_all.py}"

for h5_path in "${h5_paths[@]}"; do
  for training_approach in "${training_approaches[@]}"; do
    for encoder_type in "${encoders[@]}"; do

      cmd=(
        env CUDA_VISIBLE_DEVICES=GPU-e0e428ea-6d61-746a-65d1-df5a0fbaef9d
        "${PYTHON_BIN}" "${SCRIPT_PATH}"
        --training_approach "${training_approach}"
        --encoder_type "${encoder_type}"
        --out_h5 "${h5_path}"
      )

      echo "========================================="
      echo "Command: ${cmd[*]}"
      echo "========================================="

      "${cmd[@]}"

    done
  done
done

wait
echo "All jobs finished."