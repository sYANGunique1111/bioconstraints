"""
Main training script for video-based 2D-to-3D pose estimation with clip-based centering.
Unlike main.py which centers each pose at its own pelvis, this version centers each video clip
at the pelvis of the first frame to preserve temporal dependencies.
"""

import os
import sys
import time
import errno
import yaml
import tempfile
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator, UnchunkedGenerator
from common.loss import mpjpe, p_mpjpe
from models.mixste import MixSTE2, HybridMixSTE, HybridMixSTEV2
from models.pose_embedder import HybridPoseModel2, HybridPoseModel3, HybridPoseModel3_2

# Global args (set in main)
args = None

# Global wandb run (for cleanup)
wandb_run = None


def safe_torch_save(obj, path):
    """Atomically save a checkpoint (safe for NFS/networked filesystems)."""
    dir_name = os.path.dirname(path)
    fd, temp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    os.close(fd)
    try:
        torch.save(obj, temp_path)
        shutil.move(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def save_config(args):
    """Save all training arguments to a YAML config file."""
    config_path = os.path.join(args.checkpoint, 'config.yaml')
    config_dict = vars(args).copy()
    
    # Convert any non-serializable types to strings
    for key, value in config_dict.items():
        if not isinstance(value, (int, float, str, bool, list, dict, type(None))):
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
    
    print(f'Config saved to: {config_path}')


def compute_flops(model, args):
    """Compute MACs/FLOPs for the model using ptflops."""
    try:
        from ptflops import get_model_complexity_info
        
        model.eval()
        
        # Define input constructor based on model type
        if args.model == 'hybrid':
            # HybridPoseModel: input shape (B, T, J, 2)
            input_shape = (args.number_of_frames, args.num_joints, 2)
            
            def input_constructor(input_shape):
                return {'x': torch.ones((1,) + input_shape).cuda()}
        else:
            # MixSTE2: input shape (B, T, J, 2)
            input_shape = (args.number_of_frames, args.num_joints, 2)
            
            def input_constructor(input_shape):
                return torch.ones((1,) + input_shape).cuda()
        
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
            input_constructor=input_constructor if args.model == 'hybrid' else None,
            print_per_layer_stat=False,
            verbose=False
        )
        
        return macs
        
    except Exception as e:
        print(f'Warning: Could not compute FLOPs: {e}')
        return None


def fetch(keypoints, dataset, subjects, action_filter=None, subset=1, parse_3d_poses=True):
    """Fetch training/test data from the dataset."""
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


def apply_clip_centering(poses_3d):
    """
    Center each clip at the pelvis of the first frame.
    Args:
        poses_3d: Tensor of shape (B, T, J, 3)
    Returns:
        Centered poses where first frame's pelvis is at origin
    """
    # Subtract pelvis position of first frame from all frames
    # poses_3d[:, :, 0] is pelvis joint (root) for all frames
    # poses_3d[:, 0:1, 0:1] is pelvis position of first frame, shape (B, 1, 1, 3)
    center = poses_3d.shape[1] // 2
    centered_poses = poses_3d - poses_3d[:, center:center+1, 0:1]
    return centered_poses


def apply_frame_centering(poses_3d):
    """
    Center each frame at its own pelvis (old style).
    Args:
        poses_3d: Tensor of shape (B, T, J, 3)
    Returns:
        Centered poses where each frame's pelvis is at origin
    """
    # Center each frame at its own pelvis
    centered_poses = poses_3d.clone()
    centered_poses[:, :, 0] = 0  # Set pelvis to origin
    return centered_poses


def runner(rank, args, train_data, test_data):
    """Main training runner for each DDP process."""
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    lr = args.learning_rate

    # Joint groups configuration
    joint_groups = [
                [1, 2, 3],        # Right Leg
                [4, 5, 6],        # Left Leg  
                [11, 12, 13],     # Left Arm
                [14, 15, 16],     # Right Arm
                [9,10], # Head Nose
                [0,7,8], # Full spine
                [10, 8, 13], # Full shoulders
                [0,1,4] # pelvis right left hip
            ]
    
    # Create model based on selection
    if args.model == 'mixste':
        model_pos = MixSTE2(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=2
        ).cuda()
    elif args.model == 'hybrid3':
        model_pos = HybridPoseModel3(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_channels=2,
            out_channels=3,
            patch_size=args.patch_size,
            hidden_size=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.drop_rate,
            joint_groups=joint_groups
        ).cuda()
    elif args.model == 'hybrid3_2':
        model_pos = HybridPoseModel3_2(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_channels=2,
            out_channels=3,
            patch_size=args.patch_size,
            hidden_size=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            dropout=args.drop_rate,
            joint_groups=joint_groups
        ).cuda()
    elif args.model == 'hybrid_mixste':
        model_pos = HybridMixSTE(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            patch_size=args.patch_size,
            joint_groups=joint_groups,
            decoder_mode=getattr(args, 'decoder_mode', 'overlap_average')
        ).cuda()
    elif args.model == 'hybrid_mixste_v2':
        model_pos = HybridMixSTEV2(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            patch_size=args.patch_size,
            joint_groups=joint_groups,
            decoder_mode=getattr(args, 'decoder_mode', 'overlap_average')
        ).cuda()
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model_pos = DDP(module=model_pos, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    min_loss = args.min_loss
    losses_train = []
    losses_valid_new = []  # New-style loss (clip-based centering)
    losses_valid_old = []  # Old-style loss (frame-based centering)

    epoch = 0
    
    # Setup data loaders
    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size // args.number_of_frames // args.world_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    valid_sampler = DistributedSampler(test_data, num_replicas=args.world_size, rank=rank, shuffle=False, drop_last=True)
    valid_loader = DataLoader(test_data, batch_size=args.batch_size // args.number_of_frames // args.world_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True, sampler=valid_sampler)

    if dist.get_rank() == args.reduce_rank:
        model_params = sum(p.numel() for p in model_pos.parameters())
        model_class_name = model_pos.module.__class__.__name__
        print(f'INFO: Model: {model_class_name} | Trainable parameters: {model_params/1e6:.3f} Million')
        
        # Compute and log MACs/FLOPs
        macs = compute_flops(model_pos.module, args)
        if macs is not None:
            print(f'INFO: Model MACs: {macs/1e9:.3f} GMACs')
            # Save to metrics file
            metrics_path = os.path.join(args.checkpoint, 'metrics.yaml')
            metrics = {
                'model': model_class_name,
                'parameters': model_params,
                'parameters_million': round(model_params / 1e6, 3),
                'macs': int(macs),
                'macs_giga': round(macs / 1e9, 3)
            }
            with open(metrics_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
    
    # Calculate epochs for gradient snapshots (beginning, middle, end) - used by W&B
    gradient_epochs = [0, args.epochs // 2, args.epochs - 1]
    
    # Initialize W&B (rank 0 only)
    if dist.get_rank() == args.reduce_rank and args.wandb:
        import wandb
        run_name = os.path.basename(args.checkpoint)
        wandb.init(
            project="NewPoseProject",
            name=run_name,
            config=vars(args)
        )

    # Training loop
    while epoch < args.epochs:
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        epoch_loss_train = 0
        N = 0

        model_pos.train()
        
        for _, inputs_3d, inputs_2d in train_loader:
            inputs_2d = inputs_2d.cuda().float()
            inputs_3d = inputs_3d.cuda().float()
            
            # Apply clip-based centering: center at first frame's pelvis
            inputs_3d = apply_clip_centering(inputs_3d)

            optimizer.zero_grad()

            # Forward pass
            predicted_3d = model_pos(inputs_2d)
            
            # Compute loss
            loss = mpjpe(predicted_3d, inputs_3d)
            loss.backward()

            epoch_loss_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            optimizer.step()

        losses_train.append(epoch_loss_train / N)
        torch.cuda.empty_cache()

        # Validation
        with torch.no_grad():
            model_pos.eval()
            epoch_loss_valid_new = 0
            epoch_loss_valid_old = 0
            N_valid = 0

            for _, inputs_3d, inputs_2d in valid_loader:
                inputs_3d = inputs_3d.cuda().float()
                inputs_2d = inputs_2d.cuda().float()
                
                # Apply clip-based centering for new-style evaluation
                inputs_3d_clip_centered = apply_clip_centering(inputs_3d)

                predicted_3d = model_pos(inputs_2d)
                
                # New-style loss: compare with clip-centered ground truth
                error_new = mpjpe(predicted_3d, inputs_3d_clip_centered)
                
                # Old-style loss: post-process predictions to frame-centered style
                # First, add back the first frame's pelvis offset to predictions
                # Then center each frame at its own pelvis
                predicted_3d_with_offset = predicted_3d + inputs_3d[:, 0:1, 0:1]
                predicted_3d_frame_centered = apply_frame_centering(predicted_3d_with_offset)
                inputs_3d_frame_centered = apply_frame_centering(inputs_3d)
                error_old = mpjpe(predicted_3d_frame_centered, inputs_3d_frame_centered)
                
                dist.all_reduce(error_new, op=dist.ReduceOp.SUM)
                dist.all_reduce(error_old, op=dist.ReduceOp.SUM)
                
                epoch_loss_valid_new += inputs_3d.shape[0] * inputs_3d.shape[1] * error_new.cpu().item() / args.world_size
                epoch_loss_valid_old += inputs_3d.shape[0] * inputs_3d.shape[1] * error_old.cpu().item() / args.world_size
                N_valid += inputs_3d.shape[0] * inputs_3d.shape[1]

            losses_valid_new.append(epoch_loss_valid_new / N_valid)
            losses_valid_old.append(epoch_loss_valid_old / N_valid)

        elapsed = (time.time() - start_time) / 60
        
        if dist.get_rank() == args.reduce_rank:
            print(f'[{epoch + 1}] time {elapsed:.2f}min lr {lr:.6f} '
                  f'train {losses_train[-1]*1000:.3f}mm '
                  f'valid_new {losses_valid_new[-1]*1000:.3f}mm '
                  f'valid_old {losses_valid_old[-1]*1000:.3f}mm')

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            with open(log_path, mode='a') as f:
                f.write(f'[{epoch + 1}] time {elapsed:.2f} lr {lr:.6f} '
                       f'train {losses_train[-1]*1000:.3f} '
                       f'valid_new {losses_valid_new[-1]*1000:.3f} '
                       f'valid_old {losses_valid_old[-1]*1000:.3f}\n')
            
            # W&B logging
            if args.wandb:
                import wandb
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": losses_train[-1] * 1000,
                    "valid_loss_new": losses_valid_new[-1] * 1000,
                    "valid_loss_old": losses_valid_old[-1] * 1000,
                    "learning_rate": lr,
                    "best_valid_loss": min_loss,
                    "time_per_epoch": elapsed
                }
                
                # Log gradients at 3 points: beginning, middle, end
                if epoch in gradient_epochs:
                    for name, param in model_pos.named_parameters():
                        if param.grad is not None:
                            log_dict[f"gradients/{name}"] = wandb.Histogram(param.grad.cpu().numpy())
                            log_dict[f"params/{name}"] = wandb.Histogram(param.data.cpu().numpy())
                
                wandb.log(log_dict)

        # Learning rate decay
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        dist.barrier()

        # Save checkpoints (using old-style loss for best checkpoint)
        if dist.get_rank() == args.reduce_rank:
            filtered_state_dict = {k.replace("module.", "", 1): v for k, v in model_pos.state_dict().items()}
            
            if epoch % args.checkpoint_frequency == 0:
                chk_path = os.path.join(args.checkpoint, f'epoch_{epoch}.bin')
                print(f'Saving checkpoint to {chk_path}')
                safe_torch_save({
                    'epoch': epoch,
                    'lr': lr,
                    'optimizer': optimizer.state_dict(),
                    'model_pos': filtered_state_dict,
                }, chk_path)

            # Save best checkpoint based on old-style loss for comparison
            if losses_valid_old[-1] * 1000 < min_loss:
                min_loss = losses_valid_old[-1] * 1000
                best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                print(f"Saving best checkpoint (old_loss: {min_loss:.3f}mm, new_loss: {losses_valid_new[-1]*1000:.3f}mm)")
                safe_torch_save({
                    'epoch': epoch,
                    'lr': lr,
                    'optimizer': optimizer.state_dict(),
                    'model_pos': filtered_state_dict,
                }, best_chk_path)
    
    # Finish W&B run
    if dist.get_rank() == args.reduce_rank and args.wandb:
        import wandb
        wandb.finish()


def main(args):
    print('Loading dataset...')
    dataset_root = '/data/shuoyang67/H36m/annot'
    
    # Dataset path configuration
    dataset_path = f'{dataset_root}/data_3d_{args.dataset}.npz'
    
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
        keypoints = np.load(f'{dataset_root}/data_2d_{args.dataset}_{args.keypoints}.npz', allow_pickle=True)
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'
    elif args.dataset.startswith('humaneva'):
        from common.humaneva_dataset import HumanEvaDataset
        dataset = HumanEvaDataset(dataset_path)
        keypoints = np.load(f'{dataset_root}/data_2d_{args.dataset}_{args.keypoints}.npz', allow_pickle=True)
        args.subjects_train = 'Train/S1,Train/S2,Train/S3'
        args.subjects_test = 'Validate/S1,Validate/S2,Validate/S3'
    else:
        raise KeyError(f'Invalid dataset: {args.dataset}')

    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # NOTE: Do NOT remove global offset here - keep raw positions
                    # Clip-based centering will be applied during training/evaluation
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    # Validate keypoints match 3D data
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

    # Normalize 2D keypoints
    # Only process subjects that have camera data in the dataset
    for subject in keypoints.keys():
        if subject not in dataset.cameras():
            continue  # Skip subjects without camera data (e.g., S2, S3, S4)
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    if not args.evaluate:
        print('** Note: reported losses are averaged over all frames.')
        print('** Using clip-based centering: each clip centered at first frame\'s pelvis')
        cameras_train, poses_train, poses_train_2d = fetch(keypoints, dataset, subjects_train, action_filter, subset=args.subset)
        cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test, action_filter)

        train_data = ChunkedGenerator(
            args.batch_size // args.number_of_frames, cameras_train, poses_train, poses_train_2d,
            args.number_of_frames, pad=0, causal_shift=0, shuffle=True, augment=args.data_augmentation,
            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        
        test_data = ChunkedGenerator(
            args.batch_size // args.number_of_frames, cameras_test, poses_test, poses_test_2d,
            args.number_of_frames, pad=0, causal_shift=0, shuffle=False, augment=False)

        print(f'INFO: Training on {train_data.num_frames()} frames')
        print(f'INFO: Testing on {test_data.num_frames()} frames')
        
        mp.spawn(runner, args=(args, train_data, test_data), nprocs=args.world_size, join=True)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    print(f'World size: {args.world_size}')
    print(f'Checkpoint: {args.checkpoint}')

    # Create checkpoint directory
    try:
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(f'Unable to create checkpoint directory: {args.checkpoint}')

    # Save configuration to YAML
    save_config(args)
    
    main(args)
