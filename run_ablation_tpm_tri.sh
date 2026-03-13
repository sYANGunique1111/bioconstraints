#!/bin/bash

# =========================================================================
# Ablation Run: TPMo + TRI+IN (Motion Pruning + Interpolation Recovery with indices)
# =========================================================================
# Pruning: TPMo (Motion-based) at blocks 0,1 (243→121→81 tokens).
# Recovery: TRI+IN (Linear Interpolation with indices) after regression head.
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

    # TPM pruning + index-aware interpolation recovery
    "--pruning_strategy" "motion"

    # Disable use pre-interp output for training loss
    # "--no-use_return_pre_interp"

    # Model settings
    "--embed_dim" "512"
    "--num_joints" "17"

    # H2OT hierarchical pruning settings
    "--hierarchical_layer_indices" "1,2"
    "--hierarchical_token_nums" "121,81"

    # Disable hierarchical recovery (TRI is post-head)
    # "--recovery_on_hierarchy"

    # Checkpoint
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/ablation_tpm_tri_pre_interp_loss"

    # DDP settings
    "--world_size" "1"
    "--master_port" "8500"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"

    # Logging
    "--wandb"
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0" python main.py "${args[@]}"
