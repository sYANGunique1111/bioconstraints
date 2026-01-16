#!/bin/bash

# Training script for BiomechMixSTE with biomechanical constraints
# Usage: bash main_biomech.sh

# Keypoints configuration
KEYPOINTS="cpn_ft_h36m_dbb"
# KEYPOINTS="gt"

args=(
    # Training settings
    "--epochs" "120" 
    "--number_of_frames" "243"
    "--batch_size" "1024"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"
    
    # Model settings (BiomechMixSTE uses same arch as MixSTE2)
    "--embed_dim" "512"
    "--depth" "8"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"
    
    # Dropout settings
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.0"
    
    # Biomechanical loss weights (balanced for unit scales)
    # Bone/symmetry: MSE in meters (~0.0001) needs high weight to matter
    # Angle: MSE in degrees (~25) needs low weight to not dominate
    "--weight_bone" "100.0"
    "--weight_symmetry" "50.0"
    "--weight_angle" "0.001"
    
    # Checkpoint settings
    "--checkpoint" "checkpoint/biomech_mixste_cpn_h36m"
    
    # DDP settings
    "--world_size" "2" 
    "--master_port" "8501" 
    "--master_addr" "127.0.0.1" 
    "--reduce_rank" "0"
    
    # Other options
    "--nolog"
    "--wandb"  # Uncomment to enable W&B logging
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0,1" python main_biomech.py ${args[@]}
