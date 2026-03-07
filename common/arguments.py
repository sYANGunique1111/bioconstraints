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
                        help='Model architecture: mixste, hybrid3, hybrid3_2, hybrid_mixste, hybrid_mixste_v2, hybrid_joint_conv, two_stage_grouped, two_stage_patched, h2ot_mixste, h2ot_mixste_interp')
    parser.add_argument('--embed_dim', default=512, type=int)
    parser.add_argument('--depth', default=8, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--mlp_ratio', default=2., type=float)
    parser.add_argument('-nj', '--num_joints', default=17, type=int)
    parser.add_argument('--patch_size', default=9, type=int, help='Temporal patch size for HybridPoseModel')
    parser.add_argument('--drop_rate', default=0.0, type=float)
    parser.add_argument('--attn_drop_rate', default=0.0, type=float)
    parser.add_argument('--drop_path_rate', default=0.2, type=float)
    
    # HOT/H2OT MixSTE arguments
    parser.add_argument('--token_num', default=81, type=int,
                        help='Fallback single-stage clustered token number for HOT/H2OT models')
    parser.add_argument('--layer_index', default=1, type=int,
                        help='Fallback single-stage clustering block index for HOT/H2OT models')
    parser.add_argument('--hierarchical_layer_indices', default='1,2', type=str,
                        help='Comma-separated clustering block indices for H2OT (e.g. "1,2")')
    parser.add_argument('--hierarchical_token_nums', default='81,27', type=str,
                        help='Comma-separated reduced token counts after each clustering stage (e.g. "81,27")')
    parser.add_argument('--recovery_on_hierarchy', action='store_true',
                        help='Enable intermediate recovery stages in H2OT')
    parser.add_argument('--recovery_layer_indices', default='', type=str,
                        help='Comma-separated block indices where recovery is applied (e.g. "4,7")')
    parser.add_argument('--recovery_token_nums', default='', type=str,
                        help='Comma-separated recovered token counts at each recovery stage (e.g. "109,243")')
    parser.add_argument('--pruning_strategy', default='cluster', type=str,
                        help='HOT/H2OT pruning strategy: cluster, learned, motion, sampler')
    parser.add_argument('--recovery_strategy', default='attention', type=str,
                        help='H2OT recovery strategy: attention, interpolation')
    parser.add_argument('--use_return_pre_interp', action=argparse.BooleanOptionalAction, default=True,
                        help='For h2ot_mixste_interp training: use pre-interpolation outputs and kept indices for loss computation')

    # Checkpoint
    parser.add_argument('-c', '--checkpoint', default='/data/shuoyang67/checkpoint/NewPoseProject', type=str)
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
    # Note: Bone/symmetry losses are MSE in meters (~0.0001), while angle loss is MSE in degrees (~25)
    # Weights are adjusted so all losses contribute roughly equally to gradients
    parser.add_argument('--weight_bone', default=100.0, type=float, 
                        help='Weight for bone length consistency loss (high due to meter-scale MSE)')
    parser.add_argument('--weight_symmetry', default=50.0, type=float,
                        help='Weight for left/right symmetry loss')
    parser.add_argument('--weight_angle', default=0.001, type=float,
                        help='Weight for joint angle limit loss (low to avoid dominating MPJPE)')
    
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')

    parser.set_defaults(data_augmentation=True)
    args = parser.parse_args()

    if args.resume and args.evaluate:
        print('Invalid: --resume and --evaluate cannot be set together')
        exit()

    return args
