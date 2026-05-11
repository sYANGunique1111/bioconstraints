"""
Ablation study: Oracle temporal token selection for HOT-MixSTE.

Trains the OracleSelectionModel which replaces the clustering / learned pruning
step with an oracle that compares each frame's detected 2D pose to the GT 2D
pose (MSE) and keeps the top-K best frames.  This establishes an upper-bound
on how well the model can do when token selection is *perfect*.

Data pipeline difference from main.py:
  - Loads TWO sets of 2D keypoints: detected (e.g. cpn_ft_h36m_dbb) and GT.
  - Uses ChunkedGeneratorWithGT2D which yields (cam, 3d, 2d_det, 2d_gt).
  - Forward pass: model(inputs_2d, gt_2d).
"""

import os
import sys
import time
import errno
import yaml
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
from common.generators import ChunkedGenerator
from common.loss import mpjpe, p_mpjpe
from models.hot.mixste import HOTMixSTEOracle

# Reuse utility helpers from main.py
from main import fetch, save_config, safe_torch_save

# ---------------------------------------------------------------------------
# Dataset: extends ChunkedGenerator to also yield GT 2D keypoints
# ---------------------------------------------------------------------------

class ChunkedGeneratorWithGT2D(ChunkedGenerator):
    """ChunkedGenerator that additionally returns ground-truth 2D keypoints.

    The GT poses are chunked, padded, and (optionally) flipped in exactly the
    same way as the detected 2D poses so that every frame index aligns.

    ``__getitem__`` returns ``(cam, poses_3d, poses_2d_det, poses_2d_gt)``.
    """

    def __init__(self, *args, poses_2d_gt=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert poses_2d_gt is not None, "poses_2d_gt is required"
        assert len(poses_2d_gt) == len(self.poses_2d), (
            f"GT 2D and detected 2D must have the same number of sequences "
            f"({len(poses_2d_gt)} vs {len(self.poses_2d)})"
        )
        # Align GT sequence lengths to detected (chunks are built from detected)
        self.poses_2d_gt = list(poses_2d_gt)
        for i in range(len(self.poses_2d_gt)):
            det_len = self.poses_2d[i].shape[0]
            gt_len = self.poses_2d_gt[i].shape[0]
            if gt_len > det_len:
                self.poses_2d_gt[i] = self.poses_2d_gt[i][:det_len]
            elif gt_len < det_len:
                self.poses_2d_gt[i] = np.pad(
                    self.poses_2d_gt[i],
                    ((0, det_len - gt_len), (0, 0), (0, 0)),
                    'edge',
                )

    def __getitem__(self, idx):
        cam, psd_3d, psd_2d = super().__getitem__(idx)

        # Re-extract chunk metadata (same logic as parent)
        chunk = self.pairs[idx]
        seq_i, start_2d, end_2d, flip = chunk[0], chunk[1], chunk[2], chunk[3]

        seq_2d_gt = self.poses_2d_gt[seq_i]
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d_gt.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d

        if pad_left_2d != 0 or pad_right_2d != 0:
            psd_2d_gt = np.pad(
                seq_2d_gt[low_2d:high_2d],
                ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                'edge',
            )
        else:
            psd_2d_gt = seq_2d_gt[low_2d:high_2d]

        if flip:
            psd_2d_gt = psd_2d_gt.copy()
            psd_2d_gt[:, :, 0] *= -1
            psd_2d_gt[:, self.kps_left + self.kps_right] = \
                psd_2d_gt[:, self.kps_right + self.kps_left]

        return cam, psd_3d, psd_2d, psd_2d_gt


# ---------------------------------------------------------------------------
# Training runner (one per DDP rank)
# ---------------------------------------------------------------------------

def runner(rank, args, train_data, test_data):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=args.world_size, rank=rank,
    )

    lr = args.learning_rate

    # ---- Model ----
    hot_args = type('HotArgs', (), {})()
    hot_args.frames = args.number_of_frames
    hot_args.channel = args.embed_dim
    hot_args.n_joints = args.num_joints
    hot_args.token_num = args.token_num
    hot_args.layer_index = args.layer_index
    hot_args.pruning_strategy = 'cluster'  # forced by OracleSelectionModel anyway
    hot_args.oracle_mode = getattr(args, 'oracle_mode', 'global')
    print("We are using oracle mode: ", args.oracle_mode)
    model_pos = HOTMixSTEOracle(hot_args).cuda()

    model_pos = DDP(module=model_pos, device_ids=[rank])
    optimizer = torch.optim.AdamW(model_pos.parameters(), lr=lr, weight_decay=0.1)

    lr_decay = args.lr_decay
    min_loss = args.min_loss
    losses_train = []
    losses_valid = []
    losses_p_valid = []
    epoch = 0

    # ---- Data loaders ----
    train_sampler = DistributedSampler(
        train_data, num_replicas=args.world_size, rank=rank, shuffle=True,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size // args.number_of_frames // args.world_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, sampler=train_sampler,
    )
    valid_sampler = DistributedSampler(
        test_data, num_replicas=args.world_size, rank=rank,
        shuffle=False, drop_last=True,
    )
    valid_loader = DataLoader(
        test_data,
        batch_size=args.batch_size // args.number_of_frames // args.world_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True, sampler=valid_sampler,
    )

    if dist.get_rank() == args.reduce_rank:
        model_params = sum(p.numel() for p in model_pos.parameters())
        model_class_name = model_pos.module.__class__.__name__
        print(f'INFO: Model: {model_class_name} | '
              f'Trainable parameters: {model_params / 1e6:.3f} Million')

        # Compute MACs/FLOPs
        try:
            from ptflops import get_model_complexity_info

            model_pos.module.eval()
            input_shape = (args.number_of_frames, args.num_joints, 2)

            def input_constructor(input_shape):
                x = torch.ones((1,) + input_shape).cuda()
                gt = torch.ones((1,) + input_shape).cuda()
                return {'x': x, 'gt_2d': gt}

            macs, params = get_model_complexity_info(
                model_pos.module, input_shape,
                input_constructor=input_constructor,
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            if macs is not None:
                print(f'INFO: Model MACs: {macs / 1e9:.3f} GMACs')
                metrics_path = os.path.join(args.checkpoint, 'metrics.yaml')
                metrics = {
                    'model': model_class_name,
                    'parameters': model_params,
                    'parameters_million': round(model_params / 1e6, 3),
                    'macs': int(macs),
                    'macs_giga': round(macs / 1e9, 3),
                }
                with open(metrics_path, 'w') as f:
                    yaml.dump(metrics, f, default_flow_style=False)
        except Exception as e:
            print(f'Warning: Could not compute FLOPs: {e}')

    # ---- W&B (optional) ----
    if dist.get_rank() == args.reduce_rank and args.wandb:
        import wandb
        run_name = os.path.basename(args.checkpoint)
        wandb.init(project="NewPoseProject", name=run_name, config=vars(args))

    # ---- Training loop ----
    while epoch < args.epochs:
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        epoch_loss_train = 0
        N = 0

        model_pos.train()

        for _, inputs_3d, inputs_2d, gt_2d in train_loader:
            inputs_2d = inputs_2d.cuda().float()
            inputs_3d = inputs_3d.cuda().float()
            gt_2d = gt_2d.cuda().float()
            inputs_3d[:, :, 0] = 0  # centre at root

            optimizer.zero_grad()

            predicted_3d = model_pos(inputs_2d, gt_2d)
            loss = mpjpe(predicted_3d, inputs_3d)
            loss.backward()

            epoch_loss_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            optimizer.step()

        losses_train.append(epoch_loss_train / N)
        torch.cuda.empty_cache()

        # ---- Validation ----
        with torch.no_grad():
            model_pos.eval()
            epoch_loss_valid = 0
            epoch_loss_p_mpjpe = 0
            N_valid = 0

            for _, inputs_3d, inputs_2d, gt_2d in valid_loader:
                inputs_3d = inputs_3d.cuda().float()
                inputs_2d = inputs_2d.cuda().float()
                gt_2d = gt_2d.cuda().float()
                inputs_3d[:, :, 0] = 0

                predicted_3d = model_pos(inputs_2d, gt_2d)
                predicted_3d[:, :, 0] = 0

                error = mpjpe(predicted_3d, inputs_3d)
                p_error = torch.tensor(
                    p_mpjpe(predicted_3d.cpu().numpy(), inputs_3d.cpu().numpy())
                ).cuda()

                dist.all_reduce(error, op=dist.ReduceOp.SUM)
                dist.all_reduce(p_error, op=dist.ReduceOp.SUM)

                batch_size = inputs_3d.shape[0] * inputs_3d.shape[1]
                epoch_loss_valid += batch_size * error.cpu().item() / args.world_size
                epoch_loss_p_mpjpe += batch_size * p_error.cpu().item() / args.world_size
                N_valid += batch_size

            losses_valid.append(epoch_loss_valid / N_valid)
            losses_p_valid.append(epoch_loss_p_mpjpe / N_valid)

        elapsed = (time.time() - start_time) / 60

        if dist.get_rank() == args.reduce_rank:
            print(
                f'[{epoch + 1}] time {elapsed:.2f}min lr {lr:.6f} '
                f'train {losses_train[-1] * 1000:.3f}mm '
                f'valid {losses_valid[-1] * 1000:.3f}mm '
                f'PA-MPJPE {losses_p_valid[-1] * 1000:.3f}mm'
            )

            log_path = os.path.join(args.checkpoint, 'training_log.txt')
            with open(log_path, mode='a') as f:
                f.write(
                    f'[{epoch + 1}] time {elapsed:.2f} lr {lr:.6f} '
                    f'train {losses_train[-1] * 1000:.3f} '
                    f'valid {losses_valid[-1] * 1000:.3f} '
                    f'PA-MPJPE {losses_p_valid[-1] * 1000:.3f}\n'
                )

            # W&B logging
            if args.wandb:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": losses_train[-1] * 1000,
                    "valid_loss": losses_valid[-1] * 1000,
                    "valid_p_mpjpe": losses_p_valid[-1] * 1000,
                    "learning_rate": lr,
                    "best_valid_loss": min_loss,
                    "time_per_epoch": elapsed,
                })

        # LR decay
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        dist.barrier()

        # ---- Checkpointing ----
        if dist.get_rank() == args.reduce_rank:
            filtered_state_dict = {
                k.replace("module.", "", 1): v
                for k, v in model_pos.state_dict().items()
            }

            if epoch % args.checkpoint_frequency == 0:
                chk_path = os.path.join(args.checkpoint, f'epoch_{epoch}.bin')
                print(f'Saving checkpoint to {chk_path}')
                safe_torch_save({
                    'epoch': epoch, 'lr': lr,
                    'optimizer': optimizer.state_dict(),
                    'model_pos': filtered_state_dict,
                }, chk_path)

            if losses_valid[-1] * 1000 < min_loss:
                min_loss = losses_valid[-1] * 1000
                best_chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
                print(f"Saving best checkpoint (loss: {min_loss:.3f}mm)")
                safe_torch_save({
                    'epoch': epoch, 'lr': lr,
                    'optimizer': optimizer.state_dict(),
                    'model_pos': filtered_state_dict,
                }, best_chk_path)

    if dist.get_rank() == args.reduce_rank and args.wandb:
        import wandb
        wandb.finish()


# ---------------------------------------------------------------------------
# Data preparation & entry-point
# ---------------------------------------------------------------------------

def main(args):
    print('Loading dataset...')
    dataset_root = '/data/shuoyang67/dataset/H36m/annot'
    dataset_path = f'{dataset_root}/data_3d_{args.dataset}.npz'

    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
        args.subjects_train = 'S1,S5,S6,S7,S8'
        args.subjects_test = 'S9,S11'
    else:
        raise KeyError(f'Oracle ablation only supports h36m (got {args.dataset})')

    # Load BOTH keypoint sets
    keypoints_det = np.load(
        f'{dataset_root}/data_2d_{args.dataset}_{args.keypoints}.npz',
        allow_pickle=True,
    )
    keypoints_gt = np.load(
        f'{dataset_root}/data_2d_{args.dataset}_gt.npz',
        allow_pickle=True,
    )
    print(f'Loaded detected 2D keypoints: {args.keypoints}')
    print(f'Loaded ground-truth 2D keypoints: gt')

    # 3D data preparation (world->camera, root-relative)
    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(
                        anim['positions'], R=cam['orientation'], t=cam['translation'],
                    )
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    # ---- Process DETECTED 2D keypoints ----
    print('Loading detected 2D keypoints...')
    kps_meta_det = keypoints_det['metadata'].item()
    kps_symmetry = kps_meta_det['keypoints_symmetry']
    kps_left, kps_right = list(kps_symmetry[0]), list(kps_symmetry[1])
    joints_left = list(dataset.skeleton().joints_left())
    joints_right = list(dataset.skeleton().joints_right())
    keypoints_det = keypoints_det['positions_2d'].item()

    # ---- Process GT 2D keypoints ----
    print('Loading ground-truth 2D keypoints...')
    keypoints_gt = keypoints_gt['positions_2d'].item()

    # Validate & truncate both keypoint sets to match 3D
    for subject in dataset.subjects():
        assert subject in keypoints_det, f'{subject} missing from detected 2D'
        assert subject in keypoints_gt, f'{subject} missing from GT 2D'
        for action in dataset[subject].keys():
            assert action in keypoints_det[subject], (
                f'{action} of {subject} missing from detected 2D'
            )
            assert action in keypoints_gt[subject], (
                f'{action} of {subject} missing from GT 2D'
            )
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints_det[subject][action])):
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                # Truncate detected
                assert keypoints_det[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints_det[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints_det[subject][action][cam_idx] = \
                        keypoints_det[subject][action][cam_idx][:mocap_length]
                # Truncate GT
                assert keypoints_gt[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints_gt[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints_gt[subject][action][cam_idx] = \
                        keypoints_gt[subject][action][cam_idx][:mocap_length]

            assert len(keypoints_det[subject][action]) == \
                len(dataset[subject][action]['positions_3d'])
            assert len(keypoints_gt[subject][action]) == \
                len(dataset[subject][action]['positions_3d'])

    # Normalize 2D keypoints (both sets, same cameras)
    for subject in keypoints_det.keys():
        if subject not in dataset.cameras():
            continue
        for action in keypoints_det[subject]:
            for cam_idx, kps in enumerate(keypoints_det[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam['res_w'], h=cam['res_h'],
                )
                keypoints_det[subject][action][cam_idx] = kps

    for subject in keypoints_gt.keys():
        if subject not in dataset.cameras():
            continue
        for action in keypoints_gt[subject]:
            for cam_idx, kps in enumerate(keypoints_gt[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam['res_w'], h=cam['res_h'],
                )
                keypoints_gt[subject][action][cam_idx] = kps

    # ---- Fetch ----
    subjects_train = args.subjects_train.split(',')
    subjects_test = args.subjects_test.split(',')

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    print('** Note: reported losses are averaged over all frames.')

    cameras_train, poses_train, poses_train_2d_det = fetch(
        keypoints_det, dataset, subjects_train, action_filter, subset=args.subset,
    )
    _, _, poses_train_2d_gt = fetch(
        keypoints_gt, dataset, subjects_train, action_filter, subset=args.subset,
    )

    cameras_test, poses_test, poses_test_2d_det = fetch(
        keypoints_det, dataset, subjects_test, action_filter,
    )
    _, _, poses_test_2d_gt = fetch(
        keypoints_gt, dataset, subjects_test, action_filter,
    )

    assert len(poses_train_2d_det) == len(poses_train_2d_gt), \
        f"Train 2D count mismatch: {len(poses_train_2d_det)} vs {len(poses_train_2d_gt)}"
    assert len(poses_test_2d_det) == len(poses_test_2d_gt), \
        f"Test 2D count mismatch: {len(poses_test_2d_det)} vs {len(poses_test_2d_gt)}"

    # ---- Generators ----
    train_data = ChunkedGeneratorWithGT2D(
        args.batch_size // args.number_of_frames,
        cameras_train, poses_train, poses_train_2d_det,
        args.number_of_frames, pad=0, causal_shift=0,
        shuffle=True, augment=args.data_augmentation,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right,
        poses_2d_gt=poses_train_2d_gt,
    )

    test_data = ChunkedGeneratorWithGT2D(
        args.batch_size // args.number_of_frames,
        cameras_test, poses_test, poses_test_2d_det,
        args.number_of_frames, pad=0, causal_shift=0,
        shuffle=False, augment=False,
        poses_2d_gt=poses_test_2d_gt,
    )

    print(f'INFO: Training on {train_data.num_frames()} frames')
    print(f'INFO: Testing on {test_data.num_frames()} frames')

    mp.spawn(runner, args=(args, train_data, test_data),
             nprocs=args.world_size, join=True)


if __name__ == '__main__':
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Extract --oracle_mode from argv before parse_args (which rejects unknown args)
    oracle_mode = 'global'
    if '--oracle_mode' in sys.argv:
        idx = sys.argv.index('--oracle_mode')
        oracle_mode = sys.argv.pop(idx + 1)
        sys.argv.pop(idx)

    args = parse_args()
    args.oracle_mode = oracle_mode

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    print(f'=== Ablation: Oracle Token Selection (mode={args.oracle_mode}) ===')
    print(f'World size: {args.world_size}')
    print(f'Checkpoint: {args.checkpoint}')

    try:
        os.makedirs(args.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError(f'Unable to create checkpoint directory: {args.checkpoint}')

    save_config(args)
    main(args)
