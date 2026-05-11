#!/bin/bash

# Training script for NewPoseProject
# Usage: bash main.sh

# Keypoints configuration
KEYPOINTS="cpn_ft_h36m_dbb"
# KEYPOINTS="gt"

# Model selection: "mixste", "hybrid_jointwise_mixste", "preserved_hybrid_jointwise_mixste", "hot_mixste", "hot_mixste_multi", "cross_attention" or "h2ot_mixste"
MODEL="hot_mixste_chunked_multistep"

args=(
    # Training settings
    "--epochs" "150" 
    "--number_of_frames" "243"
    "--batch_size" "972"
    "--learning_rate" "0.00004"
    "--lr_decay" "0.99"
    "--dataset" "h36m"
    "--keypoints" "$KEYPOINTS"
    "--loss_type" "mpjpe"
    
    # Model selection
    "--model" "$MODEL"
    # Chunked compression settings
    # "--use_chunk_ortho_loss"
    # "--lambda_chunk_ortho" "1e-3"
    # Pruning strategy
    "--pruning_strategy" "cluster"
    # HOT/H2OT pruning settings
    "--layer_index" "3"
    "--token_num" "27"
    "--hierarchical_layer_indices" "2,3"
    "--hierarchical_token_nums" "81,27"
    # Multi-hypothesis HOT settings
    "--num_hypotheses" "5"
    # "--symmetry_floor" "1e-3"
    # "--joint_angle_floor" "1e-3"
    # "--score_eps" "1e-8"
    
    # Model settings (shared)
    "--embed_dim" "512"
    "--depth" "7"
    "--num_heads" "8"
    "--mlp_ratio" "2"
    "--num_joints" "17"
    
    # HybridPoseModel specific
    "--patch_size" "3"
    # "--use_normalized_graph"
    # "--chunking_scheme" "corner_aligned"
    "--chunking_scheme" "even"
    "--decoder_mode" "cross_attention"
    "--embed_mode" "joint"
    "--use_pairwise_flow"
    # "--norm_mode" "joint"

    # Dropout settings  
    "--drop_rate" "0.0"
    "--attn_drop_rate" "0.0"
    "--drop_path_rate" "0.0"
    
    # Checkpoint settings
    "--checkpoint" "/data/shuoyang67/checkpoint/NewPoseProject/hot_mixste_chunked_multistep_l2-81_l3-27"
    
    # Disable data augmentation (optional)
    # "--no-data-augmentation"
    
    # DDP settings
    "--num_workers" "1"
    "--world_size" "1" 
    "--master_port" "8500" 
    "--master_addr" "127.0.0.1"
    "--reduce_rank" "0"
    
    # Other options
    "--nolog"
    # "--wandb"  # Uncomment to enable W&B logging
)

export TORCH_DISTRIBUTED_DEBUG=INFO
CUDA_VISIBLE_DEVICES="0" python main.py ${args[@]}
