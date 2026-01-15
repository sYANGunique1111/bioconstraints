#!/bin/bash

# Training script for NewPoseProject - Video-based variant with clip-based centering
# Usage: bash main_video.sh

# Keypoints configuration
KEYPOINTS="cpn_ft_h36m_dbb"
# KEYPOINTS="gt"

# Model selection: "mixste", "hybrid3", "hybrid3_2", "hybrid_mixste", "hybrid_mixste_v2"
MODEL="mixste"

args=(
    # Training settings
    "--epochs" "120" 
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
    "--depth" "8"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"
    
    # HybridPoseModel specific
    "--patch_size" "9"
    
    # Dropout settings
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.0"
    
    # Checkpoint settings - separate directory for video-based variant
    "--checkpoint" "checkpoint/mixste_cpn_h36m_video_clip_based"
    
    # Disable data augmentation (optional)
    # "--no-data-augmentation"
    
    # DDP settings
    "--world_size" "2" 
    "--master_port" "8500" 
    "--master_addr" "127.0.0.1" 
    "--reduce_rank" "0"
    
    # Other options
    "--nolog"
)

export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES="0,1" python main_video.py ${args[@]}
