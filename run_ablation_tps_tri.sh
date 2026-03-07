#!/bin/bash

# =========================================================================
# Ablation Run: TPS + TRI (Sampler Pruning + Interpolation Recovery)
# =========================================================================
# Pruning: TPS (Uniform Sampler) at blocks 1,2 (243→121→81 tokens).
# Recovery: TRI (Linear Interpolation) after regression head.
# Note: TRI is a post-hoc operation, so we disable hierarchical recovery.
# =========================================================================

KEYPOINTS="cpn_ft_h36m_dbb"
MODEL="h2ot_mixste_interp"

args=(
    # Training settings
    "--epochs" "160"
    "--number_of_frames" "243"
    "--batch_size" "972"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"

    # Model selection
    "--model" "$MODEL"

    # TPS pruning + index-aware interpolation recovery
    "--pruning_strategy" "sampler"

    # Model settings
    "--embed_dim" "512"
    "--num_joints" "17"

    # H2OT hierarchical pruning settings
    "--hierarchical_layer_indices" "1,2"
    "--hierarchical_token_nums" "121,81"

    # Disable hierarchical recovery (TRI is post-head)
    # "--recovery_on_hierarchy" 

    # Checkpoint
    "--checkpoint" "checkpoint/ablation_tps_tri"

    # DDP settings
    "--world_size" "2"
    "--master_port" "8501"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"

    # Logging
    "--wandb"
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0,1" python main.py "${args[@]}"
