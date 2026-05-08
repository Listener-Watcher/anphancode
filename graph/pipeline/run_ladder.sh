#!/usr/bin/env bash
set -euo pipefail

# run_ladder.sh
#
# H5-first batch runner for the EEG experiment ladder.
#
# Behavior:
# - AHEAP: loops over split seeds and train seeds.
# - CAUEEG: uses the official fixed train/val/test split, so it only loops over train seeds.
#
# Usage:
#   bash run_ladder.sh configs/ results/
#
# Environment overrides:
#   PYTHON=python
#   MAIN=main.py
#   TRAIN_SEEDS="11 22 33"
#   SPLIT_SEEDS="101 202 303"
#   FILE_GLOB="*.json"
#   NAME_FILTER="block0"
#   EXTRA_ARGS=""

CONFIG_DIR="${1:-./configs}"
RESULTS_DIR="${2:-./results}"

PYTHON_BIN="${PYTHON:-python}"
MAIN_SCRIPT="${MAIN:-main.py}"
TRAIN_SEEDS_STR="${TRAIN_SEEDS:-11 22 33}"
SPLIT_SEEDS_STR="${SPLIT_SEEDS:-101 202 303}"
FILE_GLOB="${FILE_GLOB:-*.json}"
NAME_FILTER="${NAME_FILTER:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${RESULTS_DIR}"

read_dataset_name() {
  local config_path="$1"
  "${PYTHON_BIN}" - "$config_path" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

if path.suffix.lower() in {".yaml", ".yml"}:
    try:
        import yaml
    except Exception:
        print("unknown")
        raise SystemExit(0)
    cfg = yaml.safe_load(text) or {}
else:
    cfg = json.loads(text)

name = (((cfg or {}).get("dataset") or {}).get("name", "unknown"))
print(str(name).strip().lower())
PY
}

shopt -s nullglob
CONFIG_FILES=("${CONFIG_DIR}"/${FILE_GLOB})
shopt -u nullglob

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
  echo "[run_ladder] No config files found under ${CONFIG_DIR} with glob ${FILE_GLOB}" >&2
  exit 1
fi

echo "[run_ladder] config_dir=${CONFIG_DIR}"
echo "[run_ladder] results_dir=${RESULTS_DIR}"
echo "[run_ladder] main=${MAIN_SCRIPT}"
echo "[run_ladder] train_seeds=${TRAIN_SEEDS_STR}"
echo "[run_ladder] split_seeds=${SPLIT_SEEDS_STR}"

for CONFIG_PATH in "${CONFIG_FILES[@]}"; do
  CONFIG_NAME="$(basename "${CONFIG_PATH}")"
  CONFIG_STEM="${CONFIG_NAME%.*}"

  if [ -n "${NAME_FILTER}" ] && [[ "${CONFIG_NAME}" != *"${NAME_FILTER}"* ]]; then
    continue
  fi

  DATASET_NAME="$(read_dataset_name "${CONFIG_PATH}")"
  echo "[run_ladder] config=${CONFIG_NAME} dataset=${DATASET_NAME}"

  if [ "${DATASET_NAME}" = "caueeg" ]; then
    for TRAIN_SEED in ${TRAIN_SEEDS_STR}; do
      RUN_OUT_DIR="${RESULTS_DIR}/${CONFIG_STEM}/official_split/trainseed_${TRAIN_SEED}"
      mkdir -p "${RUN_OUT_DIR}"

      echo "============================================================"
      echo "[run_ladder] config=${CONFIG_NAME} official CAUEEG split train_seed=${TRAIN_SEED}"
      echo "[run_ladder] output=${RUN_OUT_DIR}"
      echo "============================================================"

      # shellcheck disable=SC2086
      "${PYTHON_BIN}" "${MAIN_SCRIPT}" \
        --config "${CONFIG_PATH}" \
        --output-dir "${RUN_OUT_DIR}" \
        --split-seed 0 \
        --train-seed "${TRAIN_SEED}" \
        ${EXTRA_ARGS}
    done
  else
    for SPLIT_SEED in ${SPLIT_SEEDS_STR}; do
      for TRAIN_SEED in ${TRAIN_SEEDS_STR}; do
        RUN_OUT_DIR="${RESULTS_DIR}/${CONFIG_STEM}/splitseed_${SPLIT_SEED}/trainseed_${TRAIN_SEED}"
        mkdir -p "${RUN_OUT_DIR}"

        echo "============================================================"
        echo "[run_ladder] config=${CONFIG_NAME} split_seed=${SPLIT_SEED} train_seed=${TRAIN_SEED}"
        echo "[run_ladder] output=${RUN_OUT_DIR}"
        echo "============================================================"

        # shellcheck disable=SC2086
        "${PYTHON_BIN}" "${MAIN_SCRIPT}" \
          --config "${CONFIG_PATH}" \
          --output-dir "${RUN_OUT_DIR}" \
          --split-seed "${SPLIT_SEED}" \
          --train-seed "${TRAIN_SEED}" \
          ${EXTRA_ARGS}
      done
    done
  fi
done

echo "[run_ladder] Finished all runs."
