#!/bin/bash

# =========================================================================
# Ablation Run 3: Full H2OT (Hierarchical Clustering + Recovery)
# =========================================================================
# Multi-stage clustering at blocks 1,2 (243→81→27 tokens).
# Multi-stage recovery at blocks 5,6 (27→81→243 tokens).
# Full hourglass architecture with symmetric compress/expand.
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
    "--hierarchical_token_nums" "81,27"

    # H2OT hierarchical recovery settings
    "--recovery_on_hierarchy"
    "--recovery_layer_indices" "5,6"
    "--recovery_token_nums" "81,243"

    # Checkpoint
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/ablation_3_h2ot_full"

    # DDP settings
    "--world_size" "2"
    "--master_port" "8502"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"

    # Logging
    "--wandb"
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0,1" python main.py ${args[@]}
