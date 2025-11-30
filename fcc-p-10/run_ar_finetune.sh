#!/usr/bin/env bash
# Fine-tune the MACE MP-0b3 model on the Ar dataset prepared under fcc-p-10.
set -euo pipefail

if [ "${CONDA_DEFAULT_ENV:-}" != "mace-env" ]; then
    echo "Please activate the 'mace-env' conda environment before running this script." >&2
    exit 1
fi

: "${CUDA_VISIBLE_DEVICES:=0}"
: "${OMP_NUM_THREADS:=16}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_LOAD_WEIGHTS_ONLY=0

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR="${SCRIPT_DIR}/mace_runs/ar_finetune"
LOG_DIR="${OUT_DIR}/logs"
RESULTS_DIR="${OUT_DIR}/results"
CHECKPOINTS_DIR="${OUT_DIR}/checkpoints"
NEXT_EPOCH_FILE="${OUT_DIR}/next_epoch.txt"
TRAIN_FILE="${SCRIPT_DIR}/ar_train.xyz"
VAL_FILE="${SCRIPT_DIR}/ar_val.xyz"
TEST_FILE="${SCRIPT_DIR}/ar_test.xyz"
FOUNDATION_MODEL="/home/cory/lammps/model-setups/mace_mp0/mace-mp-0b3-medium.model"
AUTO_LOOP=${AUTO_LOOP:-0}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-60}

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${RESULTS_DIR}" "${CHECKPOINTS_DIR}"

if [[ ! -f "${NEXT_EPOCH_FILE}" ]]; then
    echo 1 > "${NEXT_EPOCH_FILE}"
fi

run_training() {
    local target_epoch="$1"
    echo "Launching MACE fine-tuning with outputs in ${OUT_DIR} (target epoch ${target_epoch})"
    mace_run_train \
      --name ar_finetune \
      --model_dir "${OUT_DIR}" \
      --log_dir "${LOG_DIR}" \
      --results_dir "${RESULTS_DIR}" \
      --checkpoints_dir "${CHECKPOINTS_DIR}" \
      --train_file "${TRAIN_FILE}" \
      --valid_file "${VAL_FILE}" \
      --test_file "${TEST_FILE}" \
      --foundation_model "${FOUNDATION_MODEL}" \
      --model MACE \
      --num_channels 256 \
      --max_num_epochs "${target_epoch}" \
      --batch_size ${MACE_BATCH_SIZE:-4} \
      --lr 5e-4 \
      --scheduler ReduceLROnPlateau \
      --ema \
      --ema_decay 0.999 \
      --config_type_weights "{'Ar_300K':1.0,'Ar_900K':1.0,'Ar_1500K':0.5,'Ar_2500K':0.5}" \
      --energy_key energy \
      --forces_key forces \
      --stress_key stress \
      --E0s "{18: 0.0}" \
      --loss weighted \
      --forces_weight 1.0 \
      --energy_weight 0.02 \
      --stress_weight 0.05 \
      --device cuda \
      --restart_latest
}

run_once() {
    local target_epoch="$1"
    if run_training "${target_epoch}"; then
        local next_epoch=$((target_epoch + 1))
        echo "${next_epoch}" > "${NEXT_EPOCH_FILE}"
        echo "Epoch ${target_epoch} complete. Next epoch target: ${next_epoch}"
        return 0
    else
        echo "Training failed at epoch ${target_epoch}. Keeping next epoch target unchanged." >&2
        return 1
    fi
}

if (( AUTO_LOOP )); then
    echo "Auto-loop mode enabled. Press Ctrl+C to stop."
    while true; do
        current_epoch=$(<"${NEXT_EPOCH_FILE}")
        run_once "${current_epoch}" || exit 1
        echo "Sleeping for ${SLEEP_BETWEEN} seconds before next epoch..."
        sleep "${SLEEP_BETWEEN}" || exit 0
    done
else
    current_epoch=$(<"${NEXT_EPOCH_FILE}")
    run_once "${current_epoch}"
fi
