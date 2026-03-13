#!/bin/bash

set -euo pipefail

# Two-stage training script:
# 1. Train the Stage 1 evaluator
# 2. Train the Stage 2 regressor using the Stage 1 checkpoint

KEYPOINTS="cpn_ft_h36m_dbb"
GPU_ID="${GPU_ID:-1}"
WORLD_SIZE="${WORLD_SIZE:-1}"
MASTER_PORT_STAGE1="${MASTER_PORT_STAGE1:-8512}"
MASTER_PORT_STAGE2="${MASTER_PORT_STAGE2:-8513}"
DRY_RUN="${DRY_RUN:-0}"

BASE_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR:-/data/shuoyang67/checkpoint/NewPoseProject/two_stage_mixste}"
STAGE1_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/stage1"
STAGE2_CHECKPOINT_DIR="${BASE_CHECKPOINT_DIR}/stage2"
STAGE1_RESUME="${STAGE1_CHECKPOINT_DIR}/best_epoch.bin"

COMMON_ARGS=(
    "--dataset" "h36m"
    "--keypoints" "${KEYPOINTS}"
    "--number_of_frames" "243"
    "--batch_size" "1024"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--embed_dim" "512"
    "--num_joints" "17"
    "--world_size" "${WORLD_SIZE}"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"
    "--nolog"
)

STAGE1_ARGS=(
    "--train_stage" "1"
    "--epochs" "100"
    "--checkpoint" "${STAGE1_CHECKPOINT_DIR}"
    "--master_port" "${MASTER_PORT_STAGE1}"
)

STAGE2_ARGS=(
    "--train_stage" "2"
    "--epochs" "150"
    "--checkpoint" "${STAGE2_CHECKPOINT_DIR}"
    "--resume" "${STAGE1_RESUME}"
    "--master_port" "${MASTER_PORT_STAGE2}"
    "--wandb"
)

mkdir -p "${STAGE1_CHECKPOINT_DIR}" "${STAGE2_CHECKPOINT_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: CUDA_VISIBLE_DEVICES=${GPU_ID} python main_two_stage.py ${COMMON_ARGS[*]} ${STAGE1_ARGS[*]}"
    touch "${STAGE1_RESUME}"
else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python main_two_stage.py "${COMMON_ARGS[@]}" "${STAGE1_ARGS[@]}"
fi

if [[ ! -f "${STAGE1_RESUME}" ]]; then
    echo "Expected Stage 1 checkpoint not found: ${STAGE1_RESUME}" >&2
    exit 1
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN: CUDA_VISIBLE_DEVICES=${GPU_ID} python main_two_stage.py ${COMMON_ARGS[*]} ${STAGE2_ARGS[*]}"
else
    CUDA_VISIBLE_DEVICES="${GPU_ID}" python main_two_stage.py "${COMMON_ARGS[@]}" "${STAGE2_ARGS[@]}"
fi
