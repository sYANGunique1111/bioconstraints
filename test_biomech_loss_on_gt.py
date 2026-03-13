import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from common.loss import compute_bone_lengths, compute_joint_angles, bone_length_loss, symmetry_loss, joint_angle_loss
from common.loss import H36M_PARENTS, H36M_LEFT_JOINTS, H36M_RIGHT_JOINTS, DEFAULT_ANGLE_LIMITS

def fetch(keypoints, dataset, subjects, action_filter=None, parse_3d_poses=True):
    """Fetch training/test data from the dataset (identical to main_biomech.py)."""
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []

    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    return out_camera_params, out_poses_3d, out_poses_2d


def main():
    print('Testing Biomechanical Loss on Ground Truth')
    
    # Simulate arguments needed for parsing
    class Args:
        dataset = 'h36m'
        keypoints = 'cpn_ft_h36m_dbb'
        batch_size = 1024
        number_of_frames = 243
        subjects_train = 'S1,S5,S6,S7,S8,S9,S11'
        subjects_test = 'S9,S11'
        actions = '*'
        data_augmentation = False
    
    args = Args()
    
    dataset_root = '/data/shuoyang67/dataset/H36m/annot'
    dataset_path = f'{dataset_root}/data_3d_{args.dataset}.npz'
    
    print('Loading dataset...')
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
    keypoints = np.load(f'{dataset_root}/data_2d_{args.dataset}_{args.keypoints}.npz', allow_pickle=True)

    print('Preparing data (root centering like main_biomech.py)...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # Remove global offset by subtracting root position (same as main_biomech.py)
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    # Validate keypoints match 3D data (from main_biomech.py)
    for subject in dataset.subjects():
        assert subject in keypoints, f'Subject {subject} is missing from 2D detections'
        for action in dataset[subject].keys():
            assert action in keypoints[subject], f'Action {action} of subject {subject} is missing from 2D detections'
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

    subjects_train = args.subjects_train.split(',')
    
    # We test on the training set to get a large representative sample of GT poses
    print('Fetching data...')
    cameras_train, poses_train, poses_train_2d = fetch(keypoints, dataset, subjects_train)

    train_data = ChunkedGenerator(
        args.batch_size // args.number_of_frames, cameras_train, poses_train, poses_train_2d,
        args.number_of_frames, pad=0, causal_shift=0, shuffle=False, augment=False,
        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size // args.number_of_frames,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f'Testing on {train_data.num_frames()} GT frames')

    total_sym_loss = 0
    total_angle_loss = 0
    total_frames = 0
    
    # Store min/max angles observed for debugging
    num_joints = 17
    observed_min = {j: float('inf') for j in DEFAULT_ANGLE_LIMITS.keys()}
    observed_max = {j: float('-inf') for j in DEFAULT_ANGLE_LIMITS.keys()}

    for _, inputs_3d, _ in train_loader:
        inputs_3d = inputs_3d.cuda().float()
        
        # Frame-based centering: set root to zero (exactly as in main_biomech.py:254)
        inputs_3d[:, :, 0] = 0

        # We evaluate the loss assuming 'predicted' is the exact GT
        sym = symmetry_loss(inputs_3d)
        ang = joint_angle_loss(inputs_3d)
        
        # Manually track min/max angles to see what GT ranges actually exist
        angles = compute_joint_angles(inputs_3d, H36M_PARENTS)
        for j in DEFAULT_ANGLE_LIMITS.keys():
            angles_j = angles[..., j]
            observed_min[j] = min(observed_min[j], angles_j.min().item())
            observed_max[j] = max(observed_max[j], angles_j.max().item())

        batch_frames = inputs_3d.shape[0] * inputs_3d.shape[1]
        
        total_sym_loss += sym.item() * batch_frames
        total_angle_loss += ang.item() * batch_frames
        total_frames += batch_frames

    mean_sym_loss = total_sym_loss / total_frames
    mean_angle_loss = total_angle_loss / total_frames

    print(f"\n--- Ground Truth Loss Evaluation ---")
    print(f"Mean Symmetry Loss (L1, meters): {mean_sym_loss:.6f} ({mean_sym_loss*1000:.2f} mm)")
    print(f"Mean Joint Angle Loss (L1 penalty, rad): {mean_angle_loss:.6f}")
    
    print(f"\n--- Joint Angle Analysis ---")
    print("Joint | Allowed Range (rad) | Observed GT Range (rad) | Allowed (deg)      | Observed (deg)")
    print("-" * 90)
    for j, (lim_min, lim_max) in DEFAULT_ANGLE_LIMITS.items():
        obs_min = observed_min[j]
        obs_max = observed_max[j]
        print(f"{j:4d}  | [{lim_min:.2f}, {lim_max:.2f}]      | "
              f"[{obs_min:.2f}, {obs_max:.2f}]           | "
              f"[{np.degrees(lim_min):.0f}°, {np.degrees(lim_max):.0f}°] | "
              f"[{np.degrees(obs_min):.0f}°, {np.degrees(obs_max):.0f}°]")

if __name__ == '__main__':
    main()
