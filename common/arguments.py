"""
Argument parser for 2D-to-3D pose estimation training.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for 2D-to-3D pose estimation')

    # Dataset
    parser.add_argument('-d', '--dataset', default='h36m', type=str, help='dataset (h36m, humaneva)')
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, help='2D detections')
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str)
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str)
    parser.add_argument('-a', '--actions', default='*', type=str, help='actions (* for all)')

    # Training
    parser.add_argument('-e', '--epochs', default=120, type=int)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-lr', '--learning_rate', default=0.00006, type=float)
    parser.add_argument('-lrd', '--lr_decay', default=0.976, type=float)
    parser.add_argument('-f', '--number_of_frames', default=243, type=int)
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false')

    # Model architecture
    parser.add_argument('--model', default='mixste', type=str,
                        help='Model architecture: mixste (MixSTE2), hybrid (HybridPoseModel), or hybrid2 (HybridPoseModel2 with cross-attention)')
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--depth', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=2., type=float)
    parser.add_argument('-nj', '--num_joints', default=17, type=int)
    parser.add_argument('--patch_size', default=9, type=int, help='Temporal patch size for HybridPoseModel')
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--attn_drop_rate', default=0.0, type=float)
    parser.add_argument('--drop_path_rate', default=0.2, type=float)

    # Checkpoint
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str)
    parser.add_argument('-cf', '--checkpoint_frequency', default=10000, type=int)
    parser.add_argument('-r', '--resume', default='', type=str)
    parser.add_argument('--evaluate', default='', type=str)
    parser.add_argument('-mloss', '--min_loss', default=100000, type=float)

    # DDP
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--reduce_rank', default=0, type=int)
    parser.add_argument('--master_port', default='8500', type=str)
    parser.add_argument('--master_addr', default='127.0.0.1', type=str)

    # Other
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--subset', default=1, type=float)
    parser.add_argument('--nolog', action='store_true')
    
    # Biomechanical loss weights (for main_biomech.py)
    parser.add_argument('--weight_bone', default=0.1, type=float, 
                        help='Weight for bone length consistency loss')
    parser.add_argument('--weight_symmetry', default=0.05, type=float,
                        help='Weight for left/right symmetry loss')
    parser.add_argument('--weight_angle', default=0.01, type=float,
                        help='Weight for joint angle limit loss')
    
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')

    parser.set_defaults(data_augmentation=True)
    args = parser.parse_args()

    if args.resume and args.evaluate:
        print('Invalid: --resume and --evaluate cannot be set together')
        exit()

    return args
