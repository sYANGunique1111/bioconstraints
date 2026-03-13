#!/bin/bash

# Training script for NewPoseProject
# Usage: bash main.sh

# Keypoints configuration
KEYPOINTS="cpn_ft_h36m_dbb"
# KEYPOINTS="gt"

# Model selection: "mixste", "hot_mixste", "hot_mixste_multi", or "h2ot_mixste"
MODEL="hot_mixste_multi"

args=(
    # Training settings
    "--epochs" "150" 
    "--number_of_frames" "243"
    "--batch_size" "972"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"
    
    # Model selection
    "--model" "$MODEL"
    # Pruning strategy
    "--pruning_strategy" "cluster"
    # HOT/H2OT pruning settings
    "--layer_index" "3"
    "--token_num" "81"
    # Multi-hypothesis HOT settings
    "--num_hypotheses" "5"
    "--symmetry_floor" "1e-3"
    "--joint_angle_floor" "1e-3"
    "--score_eps" "1e-8"
    
    # Model settings (shared)
    "--embed_dim" "512"
    "--depth" "7"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"
    
    # HybridPoseModel specific
    "--patch_size" "3"
    
    # Dropout settings  
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.0"
    
    # Checkpoint settings
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/Thot_mixste_multi_cluster-prune_3layer-81_h5"
    
    # Disable data augmentation (optional)
    # "--no-data-augmentation"
    
    # DDP settings
    "--world_size" "1" 
    "--master_port" "8502" 
    "--master_addr" "127.0.0.1" 
    "--reduce_rank" "0"
    
    # Other options
    "--nolog"
    "--wandb"  # Uncomment to enable W&B logging
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="1" python main.py ${args[@]}
