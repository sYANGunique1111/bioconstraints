"""
Main training script for biomechanical constraint-based 3D pose estimation.
Uses frame-based centering (like main.py) with biomechanical losses.
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
from common.loss import mpjpe, p_mpjpe, biomechanical_loss, H36M_PARENTS, H36M_LEFT_JOINTS, H36M_RIGHT_JOINTS
from models.mixste import BiomechMixSTE

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
        input_shape = (args.number_of_frames, args.num_joints, 2)
        
        macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings=False,
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


def runner(rank, args, train_data, test_data):
    """Main training runner for each DDP process."""
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    lr = args.learning_rate
    
    # Create BiomechMixSTE model
    model_pos = BiomechMixSTE(
        num_frame=args.number_of_frames,
        num_joints=args.num_joints,
        in_chans=2,
        embed_dim_ratio=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=2
    ).cuda()
    
    # Get skeleton info for biomechanical loss
    skeleton_info = model_pos.get_skeleton_info()

    model_pos = DDP(module=model_pos, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    min_loss = args.min_loss
    losses_train = []
    losses_valid = []
    
    # Biomechanical loss tracking
    biomech_losses_train = {'bone_length': [], 'symmetry': [], 'joint_angle': []}

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
        print(f'INFO: Biomech loss weights - bone: {args.weight_bone}, sym: {args.weight_symmetry}, angle: {args.weight_angle}')
        
        # Compute and log MACs/FLOPs
        macs = compute_flops(model_pos.module, args)
        if macs is not None:
            print(f'INFO: Model MACs: {macs/1e9:.3f} GMACs')
            metrics_path = os.path.join(args.checkpoint, 'metrics.yaml')
            metrics = {
                'model': model_class_name,
                'parameters': model_params,
                'parameters_million': round(model_params / 1e6, 3),
                'macs': int(macs),
                'macs_giga': round(macs / 1e9, 3),
                'biomech_weights': {
                    'bone': args.weight_bone,
                    'symmetry': args.weight_symmetry,
                    'angle': args.weight_angle
                }
            }
            with open(metrics_path, 'w') as f:
                yaml.dump(metrics, f, default_flow_style=False)
    
    # Calculate epochs for gradient snapshots
    gradient_epochs = [0, args.epochs // 2, args.epochs - 1]
    
    # Initialize W&B (rank 0 only)
    if dist.get_rank() == args.reduce_rank and args.wandb:
        import wandb
        run_name = os.path.basename(args.checkpoint)
        wandb.init(
            project="NewPoseProject-Biomech",
            name=run_name,
            config=vars(args)
        )

    # Training loop
    while epoch < args.epochs:
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        epoch_loss_train = 0
        epoch_loss_mpjpe = 0
        epoch_biomech = {'bone_length': 0, 'symmetry': 0, 'joint_angle': 0}
        N = 0

        model_pos.train()
        
        for _, inputs_3d, inputs_2d in train_loader:
            inputs_2d = inputs_2d.cuda().float()
            inputs_3d = inputs_3d.cuda().float()
            
            # Frame-based centering: set root to zero (same as main.py)
            inputs_3d[:, :, 0] = 0

            optimizer.zero_grad()

            # Forward pass
            predicted_3d = model_pos(inputs_2d)
            
            # Compute MPJPE loss
            loss_mpjpe = mpjpe(predicted_3d, inputs_3d)
            
            # Compute biomechanical loss
            loss_biomech, loss_dict = biomechanical_loss(
                predicted_3d, inputs_3d,
                parents=skeleton_info['parents'],
                left_joints=skeleton_info['left_joints'],
                right_joints=skeleton_info['right_joints'],
                angle_limits=skeleton_info['angle_limits'],
                weight_bone=args.weight_bone,
                weight_symmetry=args.weight_symmetry,
                weight_angle=args.weight_angle
            )
            
            # Total loss
            loss = loss_mpjpe + loss_biomech
            loss.backward()
            
            # Gradient clipping to prevent explosion from biomechanical loss
            torch.nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1.0)

            batch_size = inputs_3d.shape[0] * inputs_3d.shape[1]
            epoch_loss_train += batch_size * loss.item()
            epoch_loss_mpjpe += batch_size * loss_mpjpe.item()
            epoch_biomech['bone_length'] += batch_size * loss_dict['bone_length']
            epoch_biomech['symmetry'] += batch_size * loss_dict['symmetry']
            epoch_biomech['joint_angle'] += batch_size * loss_dict['joint_angle']
            N += batch_size

            optimizer.step()

        losses_train.append(epoch_loss_train / N)
        biomech_losses_train['bone_length'].append(epoch_biomech['bone_length'] / N)
        biomech_losses_train['symmetry'].append(epoch_biomech['symmetry'] / N)
        biomech_losses_train['joint_angle'].append(epoch_biomech['joint_angle'] / N)
        torch.cuda.empty_cache()

        # Validation
        with torch.no_grad():
            model_pos.eval()
            epoch_loss_valid = 0
            N_valid = 0

            for _, inputs_3d, inputs_2d in valid_loader:
                inputs_3d = inputs_3d.cuda().float()
                inputs_2d = inputs_2d.cuda().float()
                
                # Frame-based centering: set root to zero
                inputs_3d[:, :, 0] = 0

                predicted_3d = model_pos(inputs_2d)
                error = mpjpe(predicted_3d, inputs_3d)
                
                dist.all_reduce(error, op=dist.ReduceOp.SUM)
                epoch_loss_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * error.cpu().item() / args.world_size
                N_valid += inputs_3d.shape[0] * inputs_3d.shape[1]

            losses_valid.append(epoch_loss_valid / N_valid)

        elapsed = (time.time() - start_time) / 60
        
        if dist.get_rank() == args.reduce_rank:
            print(f'[{epoch + 1}] time {elapsed:.2f}min lr {lr:.6f} '
                  f'train {losses_train[-1]*1000:.3f}mm (mpjpe {epoch_loss_mpjpe/N*1000:.3f}) '
                  f'valid {losses_valid[-1]*1000:.3f}mm '
                  f'bone {biomech_losses_train["bone_length"][-1]:.6f} '
                  f'sym {biomech_losses_train["symmetry"][-1]:.6f} '
                  f'angle {biomech_losses_train["joint_angle"][-1]:.4f}')

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            with open(log_path, mode='a') as f:
                f.write(f'[{epoch + 1}] time {elapsed:.2f} lr {lr:.6f} '
                       f'train {losses_train[-1]*1000:.3f} '
                       f'valid {losses_valid[-1]*1000:.3f} '
                       f'bone {biomech_losses_train["bone_length"][-1]:.6f} '
                       f'sym {biomech_losses_train["symmetry"][-1]:.6f} '
                       f'angle {biomech_losses_train["joint_angle"][-1]:.4f}\n')
            
            # W&B logging
            if args.wandb:
                import wandb
                log_dict = {
                    "epoch": epoch + 1,
                    "train_loss": losses_train[-1] * 1000,
                    "train_mpjpe": epoch_loss_mpjpe / N * 1000,
                    "valid_loss": losses_valid[-1] * 1000,
                    "learning_rate": lr,
                    "best_valid_loss": min_loss,
                    "time_per_epoch": elapsed,
                    "biomech/bone_length": biomech_losses_train["bone_length"][-1],
                    "biomech/symmetry": biomech_losses_train["symmetry"][-1],
                    "biomech/joint_angle": biomech_losses_train["joint_angle"][-1],
                }
                
                if epoch in gradient_epochs:
                    for name, param in model_pos.named_parameters():
                        if param.grad is not None:
                            grad_np = param.grad.cpu().numpy()
                            param_np = param.data.cpu().numpy()
                            # Skip if gradients contain NaN (can happen with arccos edge cases)
                            if not np.isnan(grad_np).any():
                                log_dict[f"gradients/{name}"] = wandb.Histogram(grad_np)
                            if not np.isnan(param_np).any():
                                log_dict[f"params/{name}"] = wandb.Histogram(param_np)
                
                wandb.log(log_dict)

        # Learning rate decay
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        dist.barrier()

        # Save checkpoints
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

            # Save best checkpoint
            if losses_valid[-1] * 1000 < min_loss:
                min_loss = losses_valid[-1] * 1000
                best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                print(f"Saving best checkpoint (loss: {min_loss:.3f}mm)")
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
                    # Remove global offset by subtracting root position (same as main.py)
                    pos_3d[:, 1:] -= pos_3d[:, :1]
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
    for subject in keypoints.keys():
        if subject not in dataset.cameras():
            continue
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
        print('** Using biomechanical constraints: bone length, symmetry, joint angles')
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
