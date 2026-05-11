"""
Frame-wise ablation evaluation for decoder variants.

This script loads checkpoints using the same dataset/model preparation logic as
main.py, but reports MPJPE and PA-MPJPE per frame index within the input clip
instead of averaging over all frames at once.
"""

import argparse
import csv
import glob
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from main import fetch
from models.efficiency_models import TwoStageGroupedPoseModel, TwoStagePatchedPoseModel
from models.hot.mixste import (
    HOTMixSTE,
    HOTMixSTEMultiHypothesis,
    HOTMixSTEPreservedQuery,
    H2OTMixSTE,
    H2OTMixSTEInterp,
)
from models.mixste import (
    HybridJointWiseMixSTE,
    HybridMixSTE,
    HybridMixSTEV2,
    HybridMixSTEWithJointConv,
    MixSTE2,
)
from models.pose_embedder import HybridPoseModel3, HybridPoseModel3_2


DATASET_ROOT = "/FARM/syangb/data/h36m"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate 2D-to-3D pose checkpoints with frame-wise MPJPE/PA-MPJPE."
    )
    parser.add_argument(
        "--checkpoint_dir",
        action="append",
        default=[],
        help="Checkpoint directory containing config.yaml and best_epoch.bin. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint_glob",
        action="append",
        default=[],
        help="Glob pattern for checkpoint directories. Repeatable.",
    )
    parser.add_argument(
        "--checkpoint_file",
        default="best_epoch.bin",
        help="Checkpoint filename inside each checkpoint directory.",
    )
    parser.add_argument(
        "--decoder_mode_override",
        default="",
        help="Override decoder_mode at model build time. Use carefully if checkpoint architecture differs.",
    )
    parser.add_argument(
        "--align_corners_override",
        choices=["true", "false", "keep"],
        default="keep",
        help="Override decoder.align_corners when available.",
    )
    parser.add_argument(
        "--subjects_test",
        default="S9,S11",
        help="Comma-separated evaluation subjects. Matches main.py default for H36M.",
    )
    parser.add_argument(
        "--actions",
        default="*",
        help="Comma-separated action filter or '*' for all actions.",
    )
    parser.add_argument(
        "--batch_size_override",
        type=int,
        default=0,
        help="Override config batch_size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--output_root",
        default="framewise_ablation",
        help="Directory where CSV/TXT outputs will be written.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Evaluation device, e.g. cuda or cpu.",
    )
    return parser.parse_args()


def as_bool(value):
    return value.lower() == "true"


def resolve_checkpoint_dirs(args):
    checkpoint_dirs = list(args.checkpoint_dir)
    for pattern in args.checkpoint_glob:
        checkpoint_dirs.extend(glob.glob(pattern))

    resolved = []
    seen = set()
    for checkpoint_dir in checkpoint_dirs:
        path = os.path.abspath(checkpoint_dir)
        if path not in seen and os.path.isdir(path):
            resolved.append(path)
            seen.add(path)

    if not resolved:
        raise ValueError("No valid checkpoint directories were provided.")
    return resolved


def load_config(checkpoint_dir):
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)
    return SimpleNamespace(**cfg)


def prepare_dataset(args):
    dataset_path = f"{DATASET_ROOT}/data_3d_{args.dataset}.npz"

    if args.dataset == "h36m":
        from common.h36m_dataset import Human36mDataset

        dataset = Human36mDataset(dataset_path)
        keypoints = np.load(
            f"{DATASET_ROOT}/data_2d_{args.dataset}_{args.keypoints}.npz",
            allow_pickle=True,
        )
    elif args.dataset.startswith("humaneva"):
        from common.humaneva_dataset import HumanEvaDataset

        dataset = HumanEvaDataset(dataset_path)
        keypoints = np.load(
            f"{DATASET_ROOT}/data_2d_{args.dataset}_{args.keypoints}.npz",
            allow_pickle=True,
        )
    else:
        raise KeyError(f"Invalid dataset: {args.dataset}")

    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if "positions" in anim:
                positions_3d = []
                for cam in anim["cameras"]:
                    pos_3d = world_to_camera(
                        anim["positions"], R=cam["orientation"], t=cam["translation"]
                    )
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim["positions_3d"] = positions_3d

    keypoints_metadata = keypoints["metadata"].item()
    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints["positions_2d"].item()

    for subject in dataset.subjects():
        assert subject in keypoints, f"Subject {subject} is missing from 2D detections"
        for action in dataset[subject].keys():
            assert action in keypoints[subject], f"Action {action} of subject {subject} is missing from 2D detections"
            if "positions_3d" not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):
                mocap_length = dataset[subject][action]["positions_3d"][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(dataset[subject][action]["positions_3d"])

    for subject in keypoints.keys():
        if subject not in dataset.cameras():
            continue
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam["res_w"], h=cam["res_h"])
                keypoints[subject][action][cam_idx] = kps

    return dataset, keypoints, kps_left, kps_right, joints_left, joints_right


def build_model(args, device):
    if args.model == "mixste":
        model = MixSTE2(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=2,
        )
    elif args.model == "hybrid3":
        model = HybridPoseModel3(
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
            joint_groups=None,
        )
    elif args.model == "hybrid3_2":
        model = HybridPoseModel3_2(
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
            joint_groups=None,
        )
    elif args.model == "hybrid_mixste":
        model = HybridMixSTE(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            patch_size=args.patch_size,
            joint_groups=None,
            decoder_mode=getattr(args, "decoder_mode", "overlap_average"),
        )
    elif args.model == "hybrid_mixste_v2":
        model = HybridMixSTEV2(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            patch_size=args.patch_size,
            joint_groups=None,
            decoder_mode=getattr(args, "decoder_mode", "overlap_average"),
        )
    elif args.model == "hybrid_joint_conv":
        model = HybridMixSTEWithJointConv(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            patch_size=args.patch_size,
        )
    elif args.model == "hybrid_jointwise_mixste":
        model = HybridJointWiseMixSTE(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_chans=2,
            embed_dim_ratio=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            norm_layer=None,
            patch_size=args.patch_size,
            use_normalized_graph=args.use_normalized_graph,
            decoder_mode=args.decoder_mode,
            embed_mode=args.embed_mode,
            align_corners=True
        )
    elif args.model == "two_stage_grouped":
        model = TwoStageGroupedPoseModel(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_channels=2,
            out_channels=3,
            hidden_size=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            joint_groups=None,
        )
    elif args.model == "two_stage_patched":
        model = TwoStagePatchedPoseModel(
            num_frame=args.number_of_frames,
            num_joints=args.num_joints,
            in_channels=2,
            out_channels=3,
            hidden_size=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            drop_rate=args.drop_rate,
            patch_size=args.patch_size,
            joint_groups=None,
        )
    elif args.model == "hot_mixste":
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = args.number_of_frames
        hot_args.channel = args.embed_dim
        hot_args.n_joints = args.num_joints
        hot_args.token_num = args.token_num
        hot_args.layer_index = args.layer_index
        hot_args.pruning_strategy = args.pruning_strategy
        model = HOTMixSTE(hot_args)
    elif args.model == "hot_mixste_multi":
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = args.number_of_frames
        hot_args.channel = args.embed_dim
        hot_args.n_joints = args.num_joints
        hot_args.token_num = args.token_num
        hot_args.layer_index = args.layer_index
        hot_args.pruning_strategy = args.pruning_strategy
        hot_args.num_hypotheses = args.num_hypotheses
        hot_args.symmetry_floor = args.symmetry_floor
        hot_args.joint_angle_floor = args.joint_angle_floor
        hot_args.score_eps = args.score_eps
        model = HOTMixSTEMultiHypothesis(hot_args)
    elif args.model == "hot_mixste_preserved_query":
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = args.number_of_frames
        hot_args.channel = args.embed_dim
        hot_args.n_joints = args.num_joints
        hot_args.token_num = args.token_num
        hot_args.layer_index = args.layer_index
        hot_args.pruning_strategy = args.pruning_strategy
        model = HOTMixSTEPreservedQuery(hot_args)
    elif args.model == "h2ot_mixste":
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = args.number_of_frames
        hot_args.channel = args.embed_dim
        hot_args.n_joints = args.num_joints
        hot_args.token_num = args.token_num
        hot_args.layer_index = args.layer_index
        hot_args.hierarchical_layer_indices = args.hierarchical_layer_indices
        hot_args.hierarchical_token_nums = args.hierarchical_token_nums
        hot_args.recovery_on_hierarchy = args.recovery_on_hierarchy
        hot_args.recovery_layer_indices = args.recovery_layer_indices
        hot_args.recovery_token_nums = args.recovery_token_nums
        hot_args.pruning_strategy = args.pruning_strategy
        hot_args.recovery_strategy = args.recovery_strategy
        model = H2OTMixSTE(hot_args)
    elif args.model == "h2ot_mixste_interp":
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = args.number_of_frames
        hot_args.channel = args.embed_dim
        hot_args.n_joints = args.num_joints
        hot_args.token_num = args.token_num
        hot_args.layer_index = args.layer_index
        hot_args.hierarchical_layer_indices = args.hierarchical_layer_indices
        hot_args.hierarchical_token_nums = args.hierarchical_token_nums
        hot_args.recovery_on_hierarchy = args.recovery_on_hierarchy
        hot_args.recovery_layer_indices = args.recovery_layer_indices
        hot_args.recovery_token_nums = args.recovery_token_nums
        hot_args.pruning_strategy = args.pruning_strategy
        hot_args.recovery_strategy = "interpolation"
        model = H2OTMixSTEInterp(hot_args)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model.to(device)


def p_mpjpe_per_pose(predicted, target):
    assert predicted.shape == target.shape

    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0 ** 2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0 ** 2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, vt = np.linalg.svd(h)
    v = vt.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / norm_y
    t = mu_x - a * np.matmul(mu_y, r)
    predicted_aligned = a * np.matmul(predicted, r) + t
    return np.linalg.norm(predicted_aligned - target, axis=2).mean(axis=1)


def infer_align_corners(model, args):
    if args.align_corners_override != "keep" and hasattr(model, "decoder") and hasattr(model.decoder, "align_corners"):
        model.decoder.align_corners = as_bool(args.align_corners_override)

    if hasattr(model, "decoder") and hasattr(model.decoder, "align_corners"):
        return model.decoder.align_corners
    return None


def load_checkpoint(model, checkpoint_path, strict):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_pos"] if "model_pos" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, missing, unexpected


def evaluate_checkpoint(model, data_loader, device, num_frames):
    mpjpe_sum = np.zeros(num_frames, dtype=np.float64)
    pampjpe_sum = np.zeros(num_frames, dtype=np.float64)
    count_sum = np.zeros(num_frames, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for _, inputs_3d, inputs_2d in data_loader:
            inputs_2d = inputs_2d.to(device).float()
            inputs_3d = inputs_3d.to(device).float()
            inputs_3d[:, :, 0] = 0

            predicted_3d = model(inputs_2d)
            predicted_3d[:, :, 0] = 0

            frame_errors = torch.norm(predicted_3d - inputs_3d, dim=-1).mean(dim=-1)
            mpjpe_sum += frame_errors.sum(dim=0).cpu().numpy()

            pred_np = predicted_3d.cpu().numpy()
            target_np = inputs_3d.cpu().numpy()
            batch_size, clip_frames = pred_np.shape[:2]
            pampjpe = p_mpjpe_per_pose(
                pred_np.reshape(batch_size * clip_frames, pred_np.shape[2], pred_np.shape[3]),
                target_np.reshape(batch_size * clip_frames, target_np.shape[2], target_np.shape[3]),
            ).reshape(batch_size, clip_frames)
            pampjpe_sum += pampjpe.sum(axis=0)

            count_sum += np.full(num_frames, batch_size, dtype=np.int64)

    return {
        "mpjpe_mm": (mpjpe_sum / count_sum) * 1000.0,
        "pampjpe_mm": (pampjpe_sum / count_sum) * 1000.0,
        "counts": count_sum,
    }


def write_outputs(output_dir, checkpoint_dir, checkpoint_path, runtime_args, eval_args, results, checkpoint_meta):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "framewise_metrics.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "mpjpe_mm", "pampjpe_mm", "num_clips"])
        for frame_idx, (mpjpe_mm, pampjpe_mm, count) in enumerate(
            zip(results["mpjpe_mm"], results["pampjpe_mm"], results["counts"])
        ):
            writer.writerow([frame_idx, f"{mpjpe_mm:.6f}", f"{pampjpe_mm:.6f}", int(count)])

    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as handle:
        handle.write(f"checkpoint_dir: {checkpoint_dir}\n")
        handle.write(f"checkpoint_path: {checkpoint_path}\n")
        handle.write(f"model: {eval_args.model}\n")
        handle.write(f"decoder_mode: {eval_args.decoder_mode}\n")
        handle.write(f"align_corners: {checkpoint_meta['align_corners']}\n")
        handle.write(f"subjects_test: {runtime_args.subjects_test}\n")
        handle.write(f"actions: {runtime_args.actions}\n")
        handle.write(f"batch_size_eval: {checkpoint_meta['batch_size_eval']}\n")
        handle.write(f"checkpoint_epoch: {checkpoint_meta['epoch']}\n")
        handle.write(f"checkpoint_lr: {checkpoint_meta['lr']}\n")
        handle.write(f"strict_load: {checkpoint_meta['strict_load']}\n")
        handle.write(f"missing_keys: {len(checkpoint_meta['missing'])}\n")
        handle.write(f"unexpected_keys: {len(checkpoint_meta['unexpected'])}\n")
        handle.write(f"mean_mpjpe_mm: {results['mpjpe_mm'].mean():.6f}\n")
        handle.write(f"mean_pampjpe_mm: {results['pampjpe_mm'].mean():.6f}\n")
        handle.write(f"best_frame_mpjpe_mm: {results['mpjpe_mm'].min():.6f}\n")
        handle.write(f"worst_frame_mpjpe_mm: {results['mpjpe_mm'].max():.6f}\n")
        handle.write(f"best_frame_pampjpe_mm: {results['pampjpe_mm'].min():.6f}\n")
        handle.write(f"worst_frame_pampjpe_mm: {results['pampjpe_mm'].max():.6f}\n")

        if checkpoint_meta["missing"]:
            handle.write("missing_key_names:\n")
            for key in checkpoint_meta["missing"]:
                handle.write(f"  {key}\n")
        if checkpoint_meta["unexpected"]:
            handle.write("unexpected_key_names:\n")
            for key in checkpoint_meta["unexpected"]:
                handle.write(f"  {key}\n")


def build_eval_args(config_args, runtime_args):
    eval_args = SimpleNamespace(**vars(config_args))
    eval_args.subjects_test = runtime_args.subjects_test or getattr(config_args, "subjects_test", "S9,S11")
    eval_args.actions = runtime_args.actions
    if runtime_args.decoder_mode_override:
        eval_args.decoder_mode = runtime_args.decoder_mode_override
    if runtime_args.batch_size_override > 0:
        eval_args.batch_size = runtime_args.batch_size_override
    return eval_args


def main():
    runtime_args = parse_args()
    checkpoint_dirs = resolve_checkpoint_dirs(runtime_args)
    device = runtime_args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    print("Loading dataset once for all checkpoint evaluations...")
    first_config = load_config(checkpoint_dirs[0])
    eval_args_for_dataset = build_eval_args(first_config, runtime_args)
    dataset, keypoints, kps_left, kps_right, joints_left, joints_right = prepare_dataset(eval_args_for_dataset)

    action_filter = None if runtime_args.actions == "*" else runtime_args.actions.split(",")
    subjects_test = runtime_args.subjects_test.split(",")

    cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test, action_filter)

    for checkpoint_dir in checkpoint_dirs:
        config_args = load_config(checkpoint_dir)
        eval_args = build_eval_args(config_args, runtime_args)
        effective_batch = max(
            1,
            eval_args.batch_size // eval_args.number_of_frames // max(1, getattr(eval_args, "world_size", 1)),
        )

        test_data = ChunkedGenerator(
            effective_batch,
            cameras_test,
            poses_test,
            poses_test_2d,
            eval_args.number_of_frames,
            pad=0,
            causal_shift=0,
            shuffle=False,
            augment=False,
        )
        data_loader = DataLoader(
            test_data,
            batch_size=effective_batch,
            shuffle=False,
            num_workers=runtime_args.num_workers,
            pin_memory=device.startswith("cuda"),
        )

        checkpoint_path = os.path.join(checkpoint_dir, runtime_args.checkpoint_file)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        strict_load = runtime_args.decoder_mode_override == ""
        model = build_model(eval_args, device)
        align_corners = infer_align_corners(model, runtime_args)
        checkpoint, missing, unexpected = load_checkpoint(model, checkpoint_path, strict=strict_load)

        print(
            f"Evaluating {checkpoint_dir} | decoder_mode={eval_args.decoder_mode} "
            f"| align_corners={align_corners}"
        )
        results = evaluate_checkpoint(model, data_loader, device, eval_args.number_of_frames)

        run_label = Path(checkpoint_dir).name
        if runtime_args.decoder_mode_override:
            run_label += f"__decoder-{runtime_args.decoder_mode_override}"
        if runtime_args.align_corners_override != "keep":
            run_label += f"__align-{runtime_args.align_corners_override}"

        output_dir = os.path.join(os.path.abspath(runtime_args.output_root), run_label)
        checkpoint_meta = {
            "align_corners": align_corners,
            "batch_size_eval": effective_batch,
            "epoch": checkpoint.get("epoch", "NA"),
            "lr": checkpoint.get("lr", "NA"),
            "strict_load": strict_load,
            "missing": missing,
            "unexpected": unexpected,
        }
        write_outputs(output_dir, checkpoint_dir, checkpoint_path, runtime_args, eval_args, results, checkpoint_meta)

        print(
            f"Saved frame-wise metrics to {output_dir} | "
            f"mean MPJPE={results['mpjpe_mm'].mean():.3f} mm | "
            f"mean PA-MPJPE={results['pampjpe_mm'].mean():.3f} mm"
        )


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
