"""
GT Ablation Study: Compute symmetry and angle losses on processed ground truth poses.

This script computes the symmetry_loss and joint_angle_loss on H36M ground truth poses
to establish baseline metrics for these biomechanical constraints.

Usage:
    python ablation_gt_losses.py
"""

import os
import numpy as np
import torch
from collections import defaultdict

from common.h36m_dataset import Human36mDataset
from common.camera import world_to_camera
from common.loss import (
    symmetry_loss, joint_angle_loss,
    H36M_PARENTS, H36M_LEFT_JOINTS, H36M_RIGHT_JOINTS, DEFAULT_ANGLE_LIMITS
)


def compute_losses_for_poses(poses_3d):
    """
    Compute symmetry and angle losses for a batch of poses.
    
    Args:
        poses_3d: numpy array of shape (N, J, 3) where N is number of frames
        
    Returns:
        sym_losses: list of per-frame symmetry loss values
        angle_losses: list of per-frame angle loss values
    """
    # Convert to torch tensor
    poses = torch.from_numpy(poses_3d).float()
    
    # Add batch dimension: (N, J, 3) -> (N, 1, J, 3) to match expected shape (B, T, J, 3)
    poses = poses.unsqueeze(1)
    
    sym_losses = []
    angle_losses = []
    
    # Process frame by frame to get individual loss values
    for i in range(poses.shape[0]):
        frame_pose = poses[i:i+1]  # (1, 1, J, 3)
        
        # Compute symmetry loss
        sym_loss = symmetry_loss(frame_pose, H36M_PARENTS, H36M_LEFT_JOINTS, H36M_RIGHT_JOINTS)
        sym_losses.append(sym_loss.item())
        
        # Compute angle loss
        angle_loss = joint_angle_loss(frame_pose, H36M_PARENTS, DEFAULT_ANGLE_LIMITS)
        angle_losses.append(angle_loss.item())
    
    return sym_losses, angle_losses


def main():
    print('Loading H36M dataset...')
    dataset_root = '/data/shuoyang67/H36m/annot'
    dataset_path = f'{dataset_root}/data_3d_h36m.npz'
    
    dataset = Human36mDataset(dataset_path)
    
    # All subjects
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    
    # Store losses per action
    action_losses = defaultdict(lambda: {'symmetry': [], 'angle': []})
    all_sym_losses = []
    all_angle_losses = []
    
    print('Processing ground truth poses...')
    
    for subject in subjects:
        print(f'  Processing {subject}...')
        
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            
            if 'positions' not in anim:
                continue
            
            positions = anim['positions']  # World coordinates
            cameras = anim['cameras']
            
            # Process each camera view
            for cam in cameras:
                # Transform to camera coordinates (same as main_biomech.py)
                pos_3d = world_to_camera(positions, R=cam['orientation'], t=cam['translation'])
                
                # Frame-based centering: remove global offset by subtracting root position
                # (same as main_biomech.py line 427)
                pos_3d[:, 1:] -= pos_3d[:, :1]
                
                # Set root to zero (same as main_biomech.py line 229)
                pos_3d[:, 0] = 0
                
                # Compute losses
                sym_losses, angle_losses = compute_losses_for_poses(pos_3d)
                
                # Get base action name (remove camera suffix like " 1", " 2")
                base_action = action.split(' ')[0]
                
                action_losses[base_action]['symmetry'].extend(sym_losses)
                action_losses[base_action]['angle'].extend(angle_losses)
                
                all_sym_losses.extend(sym_losses)
                all_angle_losses.extend(angle_losses)
    
    # Compute statistics and write results
    output_path = 'ablation_gt_losses2.txt'
    
    with open(output_path, 'w') as f:
        f.write('GT Ablation Study: Biomechanical Losses on H36M\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Subjects: {", ".join(subjects)}\n')
        f.write(f'Total frames: {len(all_sym_losses)}\n\n')
        
        f.write('Per-Action Breakdown:\n')
        f.write('-' * 50 + '\n\n')
        
        # Sort actions alphabetically
        for action in sorted(action_losses.keys()):
            losses = action_losses[action]
            
            sym_arr = np.array(losses['symmetry'])
            angle_arr = np.array(losses['angle'])
            
            f.write(f'Action: {action}\n')
            f.write(f'  Frames: {len(sym_arr)}\n')
            f.write(f'  Symmetry Loss:  mean={sym_arr.mean():.8f}, std={sym_arr.std():.8f}\n')
            f.write(f'  Angle Loss:     mean={angle_arr.mean():.8f}, std={angle_arr.std():.8f}\n')
            f.write('\n')
        
        f.write('\n')
        f.write('Overall Dataset:\n')
        f.write('-' * 50 + '\n')
        
        sym_arr = np.array(all_sym_losses)
        angle_arr = np.array(all_angle_losses)
        
        f.write(f'  Symmetry Loss:  mean={sym_arr.mean():.8f}, std={sym_arr.std():.8f}\n')
        f.write(f'  Angle Loss:     mean={angle_arr.mean():.8f}, std={angle_arr.std():.8f}\n')
    
    print(f'\nResults saved to: {output_path}')
    print('\nSummary:')
    print(f'  Total frames: {len(all_sym_losses)}')
    print(f'  Symmetry Loss:  mean={sym_arr.mean():.8f}, std={sym_arr.std():.8f}')
    print(f'  Angle Loss:     mean={angle_arr.mean():.8f}, std={angle_arr.std():.8f}')


if __name__ == '__main__':
    main()
