#!/bin/bash

# Training script for NewPoseProject
# Usage: bash main.sh

# Keypoints configuration
KEYPOINTS="cpn_ft_h36m_dbb"
# KEYPOINTS="gt"

# Model selection: "mixste" or "hybrid"
MODEL="hybrid_joint_conv"

args=(
    # Training settings
    "--epochs" "150" 
    "--number_of_frames" "243"
    "--batch_size" "1024"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"
    
    # Model selection
    "--model" "$MODEL"
    
    # Model settings (shared)
    "--embed_dim" "512"
    "--depth" "7"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"
    
    # HybridPoseModel specific
    "--patch_size" "9"
    
    # Dropout settings
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.0"
    
    # Checkpoint settings
    "--checkpoint" "checkpoint/HybridMixSTEWithJointConv-HybridSpatialBlockV-GroupDecoderV2_alpha-patch9-5semantic-group"
    
    # Disable data augmentation (optional)
    # "--no-data-augmentation"
    
    # DDP settings
    "--world_size" "1" 
    "--master_port" "8501" 
    "--master_addr" "127.0.0.1" 
    "--reduce_rank" "0"
    
    # Other options
    "--nolog"
    "--wandb"  # Uncomment to enable W&B logging
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0" python main.py ${args[@]}
