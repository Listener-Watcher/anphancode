#!/usr/bin/env bash
# set -euo pipefail

# ============================================================
# Sweep NodeInputBuilder modes and GraphReadout modes
#
# Assumes your training entry script supports CLI args like:
#   --node_input_mode
#   --raw_encoder_type
#   --raw_emb_dim
#   --fusion_hidden_dim
#   --fusion_dropout
#   --graph_readout_mode
#   --topk_ratio
#   --topk_min_nodes
#   --readout_attn_hidden_dim
# plus your existing dataset / training args.
#
# Edit the paths below before running.
# ============================================================

PYTHON_BIN="${PYTHON_BIN:-python}"
PY_SCRIPT="${PY_SCRIPT:-/home/anphan/Documents/EEG_Project/graph/mil_full_std_nodeinput.py}"

ALL_DATA_PATH="${ALL_DATA_PATH:-/mnt/data/anphan/AHEAP_data/master_full_data_mono_250hz.h5}"
FEATURE_FAMILIES="${FEATURE_FAMILIES:-relative_band_power,hjorth,statistical}"
CONNECTIVITY_METRIC="${CONNECTIVITY_METRIC:-pli}"
CONNECTIVITY_BAND="${CONNECTIVITY_BAND:-2}"

ENCODER_TYPE="${ENCODER_TYPE:-linkx}"
MIL_POOL_TYPE="${MIL_POOL_TYPE:-mean}"
EDGE_MODE="${EDGE_MODE:-topology_weighted}"
TOPOLOGY="${TOPOLOGY:-fixed}"
GRAPH_POOL="${GRAPH_POOL:-mean}"
NORM_MODE="${NORM_MODE:-none}"
ALIGN_MODE="${ALIGN_MODE:-none}"
BASE_K="${BASE_K:-50}"
DIM="${DIM:-32}"
DROPOUT="${DROPOUT:-0.3}"

RAW_EMB_DIM="${RAW_EMB_DIM:-16}"
FUSION_HIDDEN_DIM="${FUSION_HIDDEN_DIM:-32}"
FUSION_DROPOUT="${FUSION_DROPOUT:-0.1}"
READOUT_ATTN_HIDDEN_DIM="${READOUT_ATTN_HIDDEN_DIM:-64}"
TOPK_RATIO="${TOPK_RATIO:-0.5}"
TOPK_MIN_NODES="${TOPK_MIN_NODES:-4}"

SEEDS="${SEEDS:-15}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p logs

run_one () {
  local node_input_mode="$1"
  local raw_encoder_type="$2"
  local graph_readout_mode="$3"
  local tag="$4"

  echo "============================================================"
  echo "Running: ${tag}"
  echo "  node_input_mode   = ${node_input_mode}"
  echo "  raw_encoder_type  = ${raw_encoder_type}"
  echo "  graph_readout_mode= ${graph_readout_mode}"
  echo "============================================================"

  "${PYTHON_BIN}" "${PY_SCRIPT}" \
    --all_data_path "${ALL_DATA_PATH}" \
    --feature_families "${FEATURE_FAMILIES}" \
    --connectivity_metric "${CONNECTIVITY_METRIC}" \
    --connectivity_band "${CONNECTIVITY_BAND}" \
    --encoder_type "${ENCODER_TYPE}" \
    --mil_pool_type "${MIL_POOL_TYPE}" \
    --edge_mode "${EDGE_MODE}" \
    --topology "${TOPOLOGY}" \
    --graph_pool "${GRAPH_POOL}" \
    --norm_mode "${NORM_MODE}" \
    --align_mode "${ALIGN_MODE}" \
    --base_k "${BASE_K}" \
    --dim "${DIM}" \
    --node_input_mode "${node_input_mode}" \
    --raw_encoder_type "${raw_encoder_type}" \
    --raw_emb_dim "${RAW_EMB_DIM}" \
    --fusion_hidden_dim "${FUSION_HIDDEN_DIM}" \
    --fusion_dropout "${FUSION_DROPOUT}" \
    --graph_readout_mode "${graph_readout_mode}" \
    --topk_ratio "${TOPK_RATIO}" \
    --topk_min_nodes "${TOPK_MIN_NODES}" \
    --readout_attn_hidden_dim "${READOUT_ATTN_HIDDEN_DIM}" \
    --debug_shapes \
    ${EXTRA_ARGS} \
    2>&1 | tee "logs/${tag}.log"
}

# ============================================================
# 1) Minimal fixed experiment set matching the main ablations
# ============================================================
# A. handcrafted_only + mean
# run_one "handcrafted_only" "cnn"      "mean"               "A_handcrafted_only_mean"

# B. raw_only + mean
# run_one "raw_only"         "cnn"      "mean"               "B_raw_only_mean"

# C. handcrafted_plus_raw_concat + mean
# run_one "handcrafted_plus_raw_concat" "cnn" "mean"          "C_hand_plus_raw_concat_mean"

# D. handcrafted_plus_raw_gated + mean
# run_one "handcrafted_plus_raw_gated"  "cnn" "mean"          "D_hand_plus_raw_gated_mean"

# E. handcrafted_plus_raw_gated + attention
# run_one "handcrafted_plus_raw_gated"  "cnn_attn" "attention" "E_hand_plus_raw_gated_attention"

# F. handcrafted_plus_raw_gated + topk_attention_pool
run_one "handcrafted_plus_raw_gated"  "tcn" "topk_attention_pool" "F_hand_plus_raw_gated_topk"

# ============================================================
# 2) Full sweep over all node-input modes x graph-readout modes
#    Uncomment this block if you want the exhaustive matrix.
# ============================================================
# declare -a NODE_INPUT_MODES=(
#   "handcrafted_only"
#   "raw_only"
#   "handcrafted_plus_raw_concat"
#   "handcrafted_plus_raw_gated"
# )
#
# declare -a GRAPH_READOUT_MODES=(
#   "mean"
#   "max"
#   "attention"
#   "topk_attention_pool"
# )
#
# for node_input_mode in "${NODE_INPUT_MODES[@]}"; do
#   case "${node_input_mode}" in
#     handcrafted_only)
#       RAW_ENCODERS=("cnn")
#       ;;
#     raw_only)
#       RAW_ENCODERS=("cnn" "tcn" "cnn_attn")
#       ;;
#     handcrafted_plus_raw_concat|handcrafted_plus_raw_gated)
#       RAW_ENCODERS=("cnn" "tcn" "cnn_attn")
#       ;;
#     *)
#       echo "Unknown node_input_mode=${node_input_mode}" >&2
#       exit 1
#       ;;
#   esac
#
#   for raw_encoder_type in "${RAW_ENCODERS[@]}"; do
#     for graph_readout_mode in "${GRAPH_READOUT_MODES[@]}"; do
#       tag="${node_input_mode}__${raw_encoder_type}__${graph_readout_mode}"
#       run_one "${node_input_mode}" "${raw_encoder_type}" "${graph_readout_mode}" "${tag}"
#     done
#   done
# done

echo "All requested runs finished."
