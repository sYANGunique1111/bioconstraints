#!/bin/bash

# =========================================================================
# Ablation Run 2: H2OT with Hierarchical Clustering Only
# =========================================================================
# Multi-stage clustering at blocks 1,2 (243→81→27 tokens).
# Recovery ONLY at the end (no in-loop recovery).
# Isolates the effect of hierarchical clustering vs single-stage.
# =========================================================================

KEYPOINTS="cpn_ft_h36m_dbb"
MODEL="h2ot_mixste"

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

    # H2OT clustering settings
    "--hierarchical_layer_indices" "1,2"
    "--hierarchical_token_nums" "121,81"

    # No hierarchical recovery (end-only, same as baseline)
    # recovery_layer_indices and recovery_token_nums default to empty

    # Checkpoint
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/ablation_2-2_h2ot_hcluster"

    # DDP settings
    "--world_size" "2"
    "--master_port" "8501"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"

    # Logging
    "--wandb"
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0,1" python main.py ${args[@]}
