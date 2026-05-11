#!/bin/bash

# Ablation study: Oracle temporal token selection
# Trains HOT-MixSTE where pruning uses GT 2D to pick the best-matching frames.
# Usage: bash main_ablation_oracle.sh

KEYPOINTS="cpn_ft_h36m_dbb"

args=(
    # Training settings
    "--epochs" "120"
    "--number_of_frames" "243"
    "--batch_size" "972"
    "--learning_rate" "0.00006"
    "--lr_decay" "0.976"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"

    # Model (ignored for model creation, but keeps arg parser happy)
    "--model" "hot_mixste"

    # HOT pruning settings (token_num controls how many frames are kept)
    "--token_num" "81"
    "--layer_index" "3"
    "--pruning_strategy" "cluster"
    "--oracle_mode" "chunked"

    # Architecture
    "--embed_dim" "512"
    "--depth" "8"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"

    # Dropout
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.1"

    # Checkpoint
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/ablation_oracle_chunked_token81"

    # DDP
    "--world_size" "1"
    "--master_port" "8502"
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"

    # Other
    "--nolog"
    # "--wandb"
)

export TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_VISIBLE_DEVICES="2" python main_ablation_oracle.py ${args[@]}
