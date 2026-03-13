import os
import time
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from common.loss import mpjpe
from common.utils import load_pretrained_weights
from models.hot.mixste.two_stage_mixste import TwoStageMixSTE

def safe_torch_save(obj, path, max_retries=3):
    """Atomically save a checkpoint via /tmp before moving into place."""
    for attempt in range(max_retries):
        fd, temp_path = tempfile.mkstemp(dir="/tmp", suffix=".tmp")
        os.close(fd)
        try:
            torch.save(obj, temp_path)
            shutil.move(temp_path, path)
            return
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

def calculate_pairwise_margin_loss(scores, gt_errors, margin=0.001, eps=1e-6):
    """Computes Margin Ranking Loss over all valid frame pairs."""
    B, F = scores.shape
    pair_mask = ~torch.eye(F, dtype=torch.bool, device=scores.device).unsqueeze(0)

    scores1 = scores.unsqueeze(2).expand(B, F, F)
    scores2 = scores.unsqueeze(1).expand(B, F, F)
    err1 = gt_errors.unsqueeze(2).expand(B, F, F)
    err2 = gt_errors.unsqueeze(1).expand(B, F, F)

    valid = pair_mask & ((err1 - err2).abs() > eps)
    y = torch.where(err1 < err2, torch.ones_like(err1), -torch.ones_like(err1))

    if not valid.any():
        return scores.sum() * 0.0

    loss_fn = nn.MarginRankingLoss(margin=margin)
    return loss_fn(scores1[valid], scores2[valid], y[valid])

def calculate_accuracy_at_k(scores, gt_errors, k):
    """Computes Intersection over K to evaluate sorting performance."""
    _, pred_topk_idx = torch.topk(scores, k, dim=1, largest=True)
    _, gt_topk_idx = torch.topk(gt_errors, k, dim=1, largest=False) # smaller error is better
    
    intersections = 0
    for b in range(scores.shape[0]):
        pred_set = set(pred_topk_idx[b].tolist())
        gt_set = set(gt_topk_idx[b].tolist())
        intersections += len(pred_set.intersection(gt_set))
        
    return intersections, scores.shape[0] * k

def calculate_framewise_2d_errors(inputs_2d, gt_2d):
    """Computes per-frame mean joint L2 error between 2D detections and 2D ground truth."""
    return torch.norm(inputs_2d - gt_2d, dim=-1).mean(dim=-1)

def fetch_2d_sequences(keypoints, subjects, action_filter=None):
    """Matches main.fetch ordering for 2D-only sequences so GT aligns with detector inputs."""
    out_poses_2d = []
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
    return out_poses_2d

class ChunkedGeneratorWithGT2D(Dataset):
    """Wraps ChunkedGenerator to return aligned GT 2D chunks for Stage 1 supervision."""
    def __init__(self, base_dataset, poses_2d_gt):
        if len(base_dataset.poses_2d) != len(poses_2d_gt):
            raise ValueError(
                f"GT 2D sequence count ({len(poses_2d_gt)}) must match detected 2D sequence count ({len(base_dataset.poses_2d)})"
            )
        self.base_dataset = base_dataset
        self.poses_2d_gt = poses_2d_gt

    def __len__(self):
        return len(self.base_dataset)

    def num_frames(self):
        return self.base_dataset.num_frames()

    def batch_num(self):
        return self.base_dataset.batch_num()

    def random_state(self):
        return self.base_dataset.random_state()

    def set_random_state(self, random):
        self.base_dataset.set_random_state(random)

    def augment_enabled(self):
        return self.base_dataset.augment_enabled()

    def __getitem__(self, idx):
        cam, psd_3d, psd_2d = self.base_dataset[idx]
        seq_i, start_2d, end_2d, flip = self.base_dataset.pairs[idx]
        seq_gt_2d = self.poses_2d_gt[seq_i]

        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_gt_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if high_2d <= low_2d:
            anchor_idx = min(max(low_2d, 0), seq_gt_2d.shape[0] - 1)
            psd_gt_2d = np.repeat(seq_gt_2d[anchor_idx:anchor_idx + 1], end_2d - start_2d, axis=0)
        elif pad_left_2d != 0 or pad_right_2d != 0:
            psd_gt_2d = np.pad(
                seq_gt_2d[low_2d:high_2d],
                ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)),
                "edge",
            )
        else:
            psd_gt_2d = seq_gt_2d[low_2d:high_2d]

        if flip:
            psd_gt_2d[:, :, 0] *= -1
            psd_gt_2d[:, self.base_dataset.kps_left + self.base_dataset.kps_right] = \
                psd_gt_2d[:, self.base_dataset.kps_right + self.base_dataset.kps_left]

        return cam, psd_3d, psd_2d, psd_gt_2d

def wrap_datasets_with_gt_2d(train_data, test_data, gt_keypoints, subjects_train, subjects_test, action_filter=None):
    """Builds Stage 1 datasets that carry both detected 2D inputs and aligned GT 2D targets."""
    train_gt = fetch_2d_sequences(gt_keypoints, subjects_train, action_filter)
    test_gt = fetch_2d_sequences(gt_keypoints, subjects_test, action_filter)
    return ChunkedGeneratorWithGT2D(train_data, train_gt), ChunkedGeneratorWithGT2D(test_data, test_gt)

def get_h36m_gt_2d_path(dataset_root):
    return os.path.join(dataset_root, "data_2d_h36m_gt.npz")

def align_detection_and_gt_2d(dataset, keypoints, gt_keypoints):
    """Trim detected 2D, GT 2D, and 3D mocap to the same per-camera frame count."""
    for subject in dataset.subjects():
        assert subject in keypoints, f"Subject {subject} missing from 2D detections"
        assert subject in gt_keypoints, f"Subject {subject} missing from 2D GT"
        for action in dataset[subject].keys():
            assert action in keypoints[subject], f"Action {action} of subject {subject} missing from 2D detections"
            assert action in gt_keypoints[subject], f"Action {action} of subject {subject} missing from 2D GT"
            if "positions_3d" not in dataset[subject][action]:
                continue

            det_views = keypoints[subject][action]
            gt_views = gt_keypoints[subject][action]
            poses_3d = dataset[subject][action]["positions_3d"]
            assert len(det_views) == len(poses_3d), "Detection/3D camera count mismatch"
            assert len(gt_views) == len(poses_3d), "GT/3D camera count mismatch"

            for cam_idx in range(len(det_views)):
                det_len = det_views[cam_idx].shape[0]
                gt_len = gt_views[cam_idx].shape[0]
                mocap_len = poses_3d[cam_idx].shape[0]
                common_length = min(det_len, gt_len, mocap_len)
                if common_length <= 0:
                    raise ValueError(
                        f"Empty aligned sequence for {subject} {action} camera {cam_idx}: "
                        f"det={det_len}, gt={gt_len}, mocap={mocap_len}"
                    )

                keypoints[subject][action][cam_idx] = det_views[cam_idx][:common_length]
                gt_keypoints[subject][action][cam_idx] = gt_views[cam_idx][:common_length]
                poses_3d[cam_idx] = poses_3d[cam_idx][:common_length]

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
                assert len(cams) == len(poses_2d), "Camera count mismatch"
                for cam in cams:
                    if "intrinsic" in cam:
                        out_camera_params.append(cam["intrinsic"])

            if parse_3d_poses and "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d), "Camera count mismatch"
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    return out_camera_params, out_poses_3d, out_poses_2d

def reduce_scalar_sum(value, rank):
    tensor = torch.tensor(float(value), device=torch.device("cuda", rank))
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()

def runner(rank, args, train_data, test_data):
    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # Setup model
    hot_args = type('HotArgs', (), {})()
    hot_args.frames = args.number_of_frames
    hot_args.channel = args.embed_dim
    hot_args.n_joints = args.num_joints
    
    model = TwoStageMixSTE(hot_args).cuda()

    if args.train_stage == 2:
        if not args.resume:
            raise ValueError("Stage 2 training requires --resume to point to a trained Stage 1 checkpoint")
        checkpoint = torch.load(args.resume, map_location="cpu")
        load_pretrained_weights(model, checkpoint.get("model_pos", checkpoint))

    model = DDP(module=model, device_ids=[rank], find_unused_parameters=True)

    if args.train_stage == 2:
        for name, param in model.named_parameters():
            if 'stage1' in name or 'scoring_head' in name:
                param.requires_grad = False
                
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    best_metric = float("inf")
    lr = args.learning_rate

    if dist.get_rank() == args.reduce_rank and args.wandb and args.train_stage == 2:
        import wandb
        run_name = os.path.basename(args.checkpoint)
        wandb.init(
            project="NewPoseProject",
            name=run_name,
            config=vars(args),
        )

    # Setup Loaders (Assume standard ChunkedGenerators from main.py)
    loader_batch_size = args.batch_size // args.number_of_frames // args.world_size
    train_sampler = DistributedSampler(train_data, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=loader_batch_size, sampler=train_sampler)
    
    valid_sampler = DistributedSampler(test_data, num_replicas=args.world_size, rank=rank, shuffle=False)
    valid_loader = DataLoader(test_data, batch_size=loader_batch_size, sampler=valid_sampler)

    epoch = 0
    
    while epoch < args.epochs:
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        model.train()
        if args.train_stage == 2:
            model.module.stage1_STE.eval()
            model.module.stage1_TTE.eval()
            model.module.scoring_head.eval()
        epoch_loss = 0
        N = 0
        
        for batch in train_loader:
            if args.train_stage == 1:
                _, inputs_3d, inputs_2d, inputs_2d_gt = batch
                inputs_2d_gt = inputs_2d_gt.cuda().float()
            else:
                _, inputs_3d, inputs_2d = batch
                inputs_2d_gt = None

            inputs_2d, inputs_3d = inputs_2d.cuda().float(), inputs_3d.cuda().float()
            inputs_3d[:, :, 0] = 0
            
            optimizer.zero_grad()
            
            if args.train_stage == 1:
                gt_errors = calculate_framewise_2d_errors(inputs_2d, inputs_2d_gt)
                scores = model(inputs_2d, stage=1)
                
                loss = calculate_pairwise_margin_loss(scores, gt_errors)
                loss.backward()
                epoch_loss += loss.item() * inputs_2d.shape[0]
                
            elif args.train_stage == 2:
                # Stage 2: Train Regressor with Pruning
                predicted_3d = model(inputs_2d, stage=2, window_size=27, keep_k=9)
                loss = mpjpe(predicted_3d, inputs_3d)
                loss.backward()
                epoch_loss += loss.item() * inputs_3d.shape[0] * inputs_3d.shape[1]
                
            N += inputs_2d.shape[0] if args.train_stage == 1 else inputs_3d.shape[0] * inputs_3d.shape[1]
            optimizer.step()
            
        # Validation Loop
        with torch.no_grad():
            model.eval()
            val_metric = 0
            N_val = 0
            if args.train_stage == 2:
                model.module.stage1_STE.eval()
                model.module.stage1_TTE.eval()
                model.module.scoring_head.eval()
            
            for batch in valid_loader:
                if args.train_stage == 1:
                    _, inputs_3d, inputs_2d, inputs_2d_gt = batch
                    inputs_2d_gt = inputs_2d_gt.cuda().float()
                else:
                    _, inputs_3d, inputs_2d = batch
                    inputs_2d_gt = None

                inputs_2d, inputs_3d = inputs_2d.cuda().float(), inputs_3d.cuda().float()
                inputs_3d[:, :, 0] = 0
                
                if args.train_stage == 1:
                    gt_errors = calculate_framewise_2d_errors(inputs_2d, inputs_2d_gt)
                    scores = model(inputs_2d, stage=1)
                    intersections, total = calculate_accuracy_at_k(scores, gt_errors, k=135)
                    val_metric += intersections
                    N_val += total
                    
                elif args.train_stage == 2:
                    predicted_3d = model(inputs_2d, stage=2, window_size=27, keep_k=15)
                    metric = mpjpe(predicted_3d, inputs_3d)
                    val_metric += metric.item() * inputs_3d.shape[0] * inputs_3d.shape[1]
                    N_val += inputs_3d.shape[0] * inputs_3d.shape[1]

        val_metric = reduce_scalar_sum(val_metric, rank)
        N_val = reduce_scalar_sum(N_val, rank)
        train_loss = reduce_scalar_sum(epoch_loss, rank) / max(reduce_scalar_sum(N, rank), 1.0)
        if args.train_stage == 1:
            val_metric = 1.0 - (val_metric / max(N_val, 1.0))
        else:
            val_metric = val_metric / max(N_val, 1.0)
        elapsed = (time.time() - start_time) / 60.0

        if dist.get_rank() == args.reduce_rank:
            print(f"Stage {args.train_stage} | Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.4f}")

            filtered_state_dict = {k.replace("module.", "", 1): v for k, v in model.state_dict().items()}
            if (epoch + 1) % args.checkpoint_frequency == 0:
                chk_path = os.path.join(args.checkpoint, f"epoch_{epoch + 1}.bin")
                safe_torch_save({
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "model_pos": filtered_state_dict,
                }, chk_path)

            if val_metric < best_metric:
                best_metric = val_metric
                best_chk_path = os.path.join(args.checkpoint, "best_epoch.bin")
                safe_torch_save({
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                    "model_pos": filtered_state_dict,
                }, best_chk_path)

            if args.wandb and args.train_stage == 2:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "stage": args.train_stage,
                    "train_loss": train_loss * 1000,
                    "valid_loss": val_metric * 1000,
                    "learning_rate": lr,
                    "best_valid_loss": best_metric * 1000,
                    "time_per_epoch": elapsed,
                })
            
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        epoch += 1

        dist.barrier()

    if dist.get_rank() == args.reduce_rank and args.wandb and args.train_stage == 2:
        import wandb
        wandb.finish()

    dist.destroy_process_group()

def main(args):
    print("Loading dataset...")
    dataset_root = "/data/shuoyang67/dataset/H36m/annot"
    dataset_path = f"{dataset_root}/data_3d_{args.dataset}.npz"

    if args.dataset == "h36m":
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
        keypoints_file = np.load(f"{dataset_root}/data_2d_{args.dataset}_{args.keypoints}.npz", allow_pickle=True)
        gt_keypoints_file = np.load(get_h36m_gt_2d_path(dataset_root), allow_pickle=True)
        args.subjects_train = "S1,S5,S6,S7,S8"
        args.subjects_test = "S9,S11"
    elif args.dataset.startswith("humaneva"):
        raise KeyError("Two-stage training currently supports h36m only because Stage 1 uses H36M GT 2D labels")
    else:
        raise KeyError(f"Invalid dataset: {args.dataset}")

    print("Preparing data...")
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if "positions" in anim:
                positions_3d = []
                for cam in anim["cameras"]:
                    pos_3d = world_to_camera(anim["positions"], R=cam["orientation"], t=cam["translation"])
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim["positions_3d"] = positions_3d

    print("Loading 2D detections...")
    keypoints_metadata = keypoints_file["metadata"].item()
    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints_file["positions_2d"].item()
    gt_keypoints = gt_keypoints_file["positions_2d"].item()

    align_detection_and_gt_2d(dataset, keypoints, gt_keypoints)

    for subject in keypoints.keys():
        if subject not in dataset.cameras():
            continue
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam["res_w"], h=cam["res_h"])
                keypoints[subject][action][cam_idx] = kps
                gt = gt_keypoints[subject][action][cam_idx]
                gt[..., :2] = normalize_screen_coordinates(gt[..., :2], w=cam["res_w"], h=cam["res_h"])
                gt_keypoints[subject][action][cam_idx] = gt

    subjects_train = args.subjects_train.split(",")
    subjects_test = args.subjects_test.split(",")
    action_filter = None if args.actions == "*" else args.actions.split(",")

    os.makedirs(args.checkpoint, exist_ok=True)
    print("** Note: reported losses are averaged over all frames.")
    cameras_train, poses_train, poses_train_2d = fetch(keypoints, dataset, subjects_train, action_filter, subset=args.subset)
    cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test, action_filter)

    train_data = ChunkedGenerator(
        args.batch_size // args.number_of_frames, cameras_train, poses_train, poses_train_2d,
        args.number_of_frames, pad=0, causal_shift=0, shuffle=True, augment=args.data_augmentation,
        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right,
    )
    test_data = ChunkedGenerator(
        args.batch_size // args.number_of_frames, cameras_test, poses_test, poses_test_2d,
        args.number_of_frames, pad=0, causal_shift=0, shuffle=False, augment=False,
    )

    if args.train_stage == 1:
        train_data, test_data = wrap_datasets_with_gt_2d(
            train_data, test_data, gt_keypoints, subjects_train, subjects_test, action_filter
        )

    print(f"INFO: Training on {train_data.num_frames()} frames")
    print(f"INFO: Testing on {test_data.num_frames()} frames")
    mp.spawn(runner, args=(args, train_data, test_data), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    main(args)
