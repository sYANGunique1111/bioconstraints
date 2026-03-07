#!/bin/bash

# =========================================================================
# Ablation Run 1: HOT MixSTE Baseline
# =========================================================================
# Original HoT: single-stage clustering at layer 3 (243→81 tokens),
# recovery via cross-attention only at the end.
# This establishes the baseline for comparison.
# =========================================================================

KEYPOINTS="cpn_ft_h36m_dbb"
MODEL="hot_mixste"

args=(
    # Training settings (matching HoT reference repo)
    "--epochs" "160"
    "--number_of_frames" "243"
    "--batch_size" "972"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"

    # Model selection
    "--model" "$MODEL"

    # Model settings
    "--embed_dim" "512"
    "--num_joints" "17"

    # HOT-specific settings
    "--token_num" "81"
    "--layer_index" "3"

    # Checkpoint
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/ablation_1_hot_baseline"

    # DDP settings
    "--world_size" "2"
    "--master_port" "8500"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"

    # Logging
    "--wandb"
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0,1" python main.py ${args[@]}
