"""
Ablation study: Multi-hypothesis scoring with lambda-weighted symmetry + joint angle.

Loads a trained MultiHypothesisModel checkpoint, runs inference on the H36M test set,
and sweeps over (lambda_1, lambda_2) combinations to find the best scoring formula:
    score = lambda_1 * raw_symmetry_penalty + lambda_2 * raw_joint_angle_penalty

The model forward pass is run ONCE per batch to get all hypotheses, then each lambda
combination is evaluated by re-scoring and selecting the best hypothesis.

Usage:
    python ablation_scoring.py --checkpoint checkpoints/Thot_mixste_multi_cluster-prune_3layer-81_h5
"""

import os
import csv
import yaml
import argparse
import itertools
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from common.loss import mpjpe, p_mpjpe

from models.hot.mixste.hot_mixste import (
    MultiHypothesisModel,
    symmetry_penalty_per_frame,
    joint_angle_penalty_per_frame,
)


def build_model_from_config(config):
    """Reconstruct model args from saved config.yaml and instantiate the model."""
    hot_args = type("HotArgs", (), {})()
    hot_args.frames = config["number_of_frames"]
    hot_args.channel = config["embed_dim"]
    hot_args.n_joints = config["num_joints"]
    hot_args.token_num = config["token_num"]
    hot_args.layer_index = config["layer_index"]
    hot_args.pruning_strategy = config["pruning_strategy"]
    hot_args.num_hypotheses = config["num_hypotheses"]
    hot_args.symmetry_floor = config.get("symmetry_floor", 1e-3)
    hot_args.joint_angle_floor = config.get("joint_angle_floor", 1e-3)
    hot_args.score_eps = config.get("score_eps", 1e-8)

    model = MultiHypothesisModel(hot_args)
    return model


def load_test_data(config):
    """Load H36M test data using the same pipeline as main.py."""
    dataset_root = "/data/shuoyang67/dataset/H36m/annot"
    dataset_name = config["dataset"]

    dataset_path = f"{dataset_root}/data_3d_{dataset_name}.npz"

    if dataset_name == "h36m":
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
        keypoints = np.load(
            f"{dataset_root}/data_2d_{dataset_name}_{config['keypoints']}.npz",
            allow_pickle=True,
        )
        subjects_test = "S9,S11"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Prepare 3D data
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

    # Prepare 2D keypoints
    keypoints_metadata = keypoints["metadata"].item()
    keypoints_symmetry = keypoints_metadata["keypoints_symmetry"]
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left = list(dataset.skeleton().joints_left())
    joints_right = list(dataset.skeleton().joints_right())
    keypoints = keypoints["positions_2d"].item()

    for subject in dataset.subjects():
        assert subject in keypoints
        for action in dataset[subject].keys():
            assert action in keypoints[subject]
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

    subjects_test_list = subjects_test.split(",")

    cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test_list)

    test_data = ChunkedGenerator(
        config["batch_size"] // config["number_of_frames"],
        cameras_test, poses_test, poses_test_2d,
        config["number_of_frames"],
        pad=0, causal_shift=0, shuffle=False, augment=False,
    )

    return test_data


def fetch(keypoints, dataset, subjects, action_filter=None):
    """Fetch test data from the dataset."""
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
                assert len(cams) == len(poses_2d)
                for cam in cams:
                    if "intrinsic" in cam:
                        out_camera_params.append(cam["intrinsic"])

            if "positions_3d" in dataset[subject][action]:
                poses_3d = dataset[subject][action]["positions_3d"]
                assert len(poses_3d) == len(poses_2d)
                for i in range(len(poses_3d)):
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    return out_camera_params, out_poses_3d, out_poses_2d


def select_hypothesis_with_lambdas(hypotheses, lambda_sym, lambda_angle):
    """
    Select the best hypothesis using additive raw scoring.
    
    score = lambda_sym * raw_symmetry_penalty + lambda_angle * raw_joint_angle_penalty
    
    Args:
        hypotheses: (B, H, T, J, 3) - all hypothesis predictions
        lambda_sym: weight for symmetry penalty
        lambda_angle: weight for joint angle penalty
        
    Returns:
        best: (B, T, J, 3) - selected best hypothesis per sample
    """
    B, H, T, J, C = hypotheses.shape

    # Reshape to (B*H, T, J, 3) for penalty computation
    flat = hypotheses.reshape(B * H, T, J, C)

    # Compute raw per-frame penalties: (B*H, T)
    sym_penalty = symmetry_penalty_per_frame(flat)       # (B*H, T)
    angle_penalty = joint_angle_penalty_per_frame(flat)   # (B*H, T)

    # Reshape back to (B, H, T)
    sym_penalty = sym_penalty.reshape(B, H, T)
    angle_penalty = angle_penalty.reshape(B, H, T)

    # Aggregate over T to get per-hypothesis scores: (B, H)
    sym_score = sym_penalty.mean(dim=-1)
    angle_score = angle_penalty.mean(dim=-1)

    # Weighted additive scoring
    scores = lambda_sym * sym_score + lambda_angle * angle_score  # (B, H)

    # Select best (lowest score) hypothesis
    best_idx = scores.argmin(dim=1)  # (B,)

    # Gather the best hypothesis
    gather_idx = best_idx[:, None, None, None, None].expand(-1, 1, T, J, C)
    best = hypotheses.gather(dim=1, index=gather_idx).squeeze(1)  # (B, T, J, 3)

    return best


def select_hypothesis_sigmoid(hypotheses, lambda_sym, lambda_angle):
    """
    Select the best hypothesis using sigmoid-transformed quality scores.
    
    1. Compute raw symmetry and joint angle penalties (≥ 0, lower = better).
    2. Transform to quality scores: score = -2 * sigmoid(loss) + 2
       - loss → 0  ⇒ score → 1.0 (good)
       - loss → ∞  ⇒ score → 0.0 (bad)
    3. Combine: final = λ_sym * sym_score + λ_angle * joint_score
    4. Select via argmax (highest quality).
    
    Args:
        hypotheses: (B, H, T, J, 3) - all hypothesis predictions
        lambda_sym: weight for symmetry quality score
        lambda_angle: weight for joint angle quality score
        
    Returns:
        best: (B, T, J, 3) - selected best hypothesis per sample
    """
    B, H, T, J, C = hypotheses.shape

    # Reshape to (B*H, T, J, 3) for penalty computation
    flat = hypotheses.reshape(B * H, T, J, C)

    # Compute raw per-frame penalties: (B*H, T)
    sym_penalty = symmetry_penalty_per_frame(flat)
    angle_penalty = joint_angle_penalty_per_frame(flat)

    # Reshape back to (B, H, T)
    sym_penalty = sym_penalty.reshape(B, H, T)
    angle_penalty = angle_penalty.reshape(B, H, T)

    # Aggregate over T to get per-hypothesis losses: (B, H)
    sym_loss = sym_penalty.mean(dim=-1)
    angle_loss = angle_penalty.mean(dim=-1)

    # Sigmoid transformation: -2 * sigmoid(loss) + 2  →  (0, 1]
    sym_score = -2.0 * torch.sigmoid(sym_loss) + 2.0
    angle_score = -2.0 * torch.sigmoid(angle_loss) + 2.0

    # Weighted additive quality scoring
    scores = lambda_sym * sym_score + lambda_angle * angle_score  # (B, H)

    # Select best (highest quality score) hypothesis
    best_idx = scores.argmax(dim=1)  # (B,)

    # Gather the best hypothesis
    gather_idx = best_idx[:, None, None, None, None].expand(-1, 1, T, J, C)
    best = hypotheses.gather(dim=1, index=gather_idx).squeeze(1)  # (B, T, J, 3)

    return best


def run_ablation(args):
    """Main ablation study runner."""
    # Load config
    checkpoint_dir = args.checkpoint
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model: {config['model']}")
    print(f"Num hypotheses: {config['num_hypotheses']}")

    # Build model
    model = build_model_from_config(config)
    
    # Load weights
    weights_path = os.path.join(checkpoint_dir, "best_epoch.bin")
    checkpoint_data = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint_data["model_pos"])
    print(f"Loaded weights from epoch {checkpoint_data['epoch']}")

    model = model.cuda()
    model.eval()

    # Load test data
    print("Loading test data...")
    test_data = load_test_data(config)
    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    print(f"Test set: {test_data.num_frames()} frames")

    # Lambda grid
    lambda_sym = [0.0, 0.01]
    lambda_joint = [1.0, 3.0, 5.0, 7.0, 10.0]
    lambda_combos = list(itertools.product(lambda_sym, lambda_joint))
    # Skip (0.0, 0.0) since it gives random selection
    lambda_combos = [(l1, l2) for l1, l2 in lambda_combos if not (l1 == 0.0 and l2 == 0.0)]

    print(f"Evaluating {len(lambda_combos)} lambda combinations...")

    # Run inference ONCE, cache all hypotheses
    all_hypotheses = []
    all_gt = []

    print("Running model inference...")
    with torch.no_grad():
        for batch_idx, (_, inputs_3d, inputs_2d) in enumerate(test_loader):
            inputs_2d = inputs_2d.cuda().float()
            inputs_3d = inputs_3d.cuda().float()
            inputs_3d[:, :, 0] = 0  # Center at root

            # Get encoded features, then all hypotheses (bypass selection)
            encoded = model._encode_tokens(inputs_2d)
            hypotheses = model._recover_all_hypotheses(encoded)  # (B, H, T, J, 3)

            all_hypotheses.append(hypotheses.cpu())
            all_gt.append(inputs_3d.cpu())

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(test_loader)}")

    print(f"Inference complete. Cached {len(all_hypotheses)} batches.")

    # ---- Oracle baseline: min MPJPE across hypotheses (using GT) ----
    print("Computing oracle (min-loss) baseline...")
    oracle_total_mpjpe = 0.0
    oracle_total_p_mpjpe = 0.0
    N_oracle = 0

    for hypotheses, gt in zip(all_hypotheses, all_gt):
        hypotheses = hypotheses.cuda()
        gt = gt.cuda()
        B, H, T, J, C = hypotheses.shape

        # Compute per-hypothesis MPJPE: (B, H)
        gt_expanded = gt.unsqueeze(1).expand_as(hypotheses)  # (B, H, T, J, 3)
        per_hyp_error = torch.norm(hypotheses - gt_expanded, dim=-1).mean(dim=(-1, -2))  # (B, H)

        # Select hypothesis with lowest MPJPE per sample
        best_idx = per_hyp_error.argmin(dim=1)  # (B,)
        gather_idx = best_idx[:, None, None, None, None].expand(-1, 1, T, J, C)
        best_pred = hypotheses.gather(dim=1, index=gather_idx).squeeze(1)
        best_pred[:, :, 0] = 0

        batch_size = gt.shape[0] * gt.shape[1]
        error = mpjpe(best_pred, gt).item()
        p_error = p_mpjpe(best_pred.cpu().numpy(), gt.cpu().numpy())

        oracle_total_mpjpe += batch_size * error
        oracle_total_p_mpjpe += batch_size * p_error
        N_oracle += batch_size

    oracle_mpjpe = (oracle_total_mpjpe / N_oracle) * 1000
    oracle_p_mpjpe = (oracle_total_p_mpjpe / N_oracle) * 1000
    print(f"  Oracle (min-loss) => MPJPE: {oracle_mpjpe:.3f}mm, PA-MPJPE: {oracle_p_mpjpe:.3f}mm")

    # Evaluate each lambda combination with BOTH scoring methods
    results = []

    # --- Method 1: Raw additive scoring (argmin) ---
    print("\n--- Raw additive scoring ---")
    for lambda_sym, lambda_angle in lambda_combos:
        total_mpjpe = 0.0
        total_p_mpjpe = 0.0
        N = 0

        for hypotheses, gt in zip(all_hypotheses, all_gt):
            hypotheses = hypotheses.cuda()
            gt = gt.cuda()

            predicted = select_hypothesis_with_lambdas(hypotheses, lambda_sym, lambda_angle)
            predicted[:, :, 0] = 0

            batch_size = gt.shape[0] * gt.shape[1]
            error = mpjpe(predicted, gt).item()
            p_error = p_mpjpe(predicted.cpu().numpy(), gt.cpu().numpy())

            total_mpjpe += batch_size * error
            total_p_mpjpe += batch_size * p_error
            N += batch_size

        avg_mpjpe = (total_mpjpe / N) * 1000
        avg_p_mpjpe = (total_p_mpjpe / N) * 1000

        results.append({
            "method": "raw",
            "lambda_sym": lambda_sym,
            "lambda_angle": lambda_angle,
            "mpjpe_mm": round(avg_mpjpe, 3),
            "pa_mpjpe_mm": round(avg_p_mpjpe, 3),
        })

        print(f"  λ_sym={lambda_sym}, λ_angle={lambda_angle} => MPJPE: {avg_mpjpe:.3f}mm, PA-MPJPE: {avg_p_mpjpe:.3f}mm")

    # --- Method 2: Sigmoid-transformed scoring (argmax) ---
    print("\n--- Sigmoid-transformed scoring ---")
    for lambda_sym, lambda_angle in lambda_combos:
        total_mpjpe = 0.0
        total_p_mpjpe = 0.0
        N = 0

        for hypotheses, gt in zip(all_hypotheses, all_gt):
            hypotheses = hypotheses.cuda()
            gt = gt.cuda()

            predicted = select_hypothesis_sigmoid(hypotheses, lambda_sym, lambda_angle)
            predicted[:, :, 0] = 0

            batch_size = gt.shape[0] * gt.shape[1]
            error = mpjpe(predicted, gt).item()
            p_error = p_mpjpe(predicted.cpu().numpy(), gt.cpu().numpy())

            total_mpjpe += batch_size * error
            total_p_mpjpe += batch_size * p_error
            N += batch_size

        avg_mpjpe = (total_mpjpe / N) * 1000
        avg_p_mpjpe = (total_p_mpjpe / N) * 1000

        results.append({
            "method": "sigmoid",
            "lambda_sym": lambda_sym,
            "lambda_angle": lambda_angle,
            "mpjpe_mm": round(avg_mpjpe, 3),
            "pa_mpjpe_mm": round(avg_p_mpjpe, 3),
        })

        print(f"  λ_sym={lambda_sym}, λ_angle={lambda_angle} => MPJPE: {avg_mpjpe:.3f}mm, PA-MPJPE: {avg_p_mpjpe:.3f}mm")

    # Sort by MPJPE
    results.sort(key=lambda r: r["mpjpe_mm"])

    # Append oracle baseline as a special row
    results.append({
        "method": "oracle",
        "lambda_sym": "-",
        "lambda_angle": "-",
        "mpjpe_mm": round(oracle_mpjpe, 3),
        "pa_mpjpe_mm": round(oracle_p_mpjpe, 3),
    })

    # Save CSV
    csv_path = os.path.join(checkpoint_dir, "ablation_scoring_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "lambda_sym", "lambda_angle", "mpjpe_mm", "pa_mpjpe_mm"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {csv_path}")
    print(f"\nOracle (min-loss, GT-based):")
    print(f"  MPJPE: {oracle_mpjpe:.3f}mm, PA-MPJPE: {oracle_p_mpjpe:.3f}mm")
    print(f"\nBest lambda combination:")
    best = results[0]
    print(f"  [{best['method']}] λ_sym={best['lambda_sym']}, λ_angle={best['lambda_angle']}")
    print(f"  MPJPE: {best['mpjpe_mm']:.3f}mm, PA-MPJPE: {best['pa_mpjpe_mm']:.3f}mm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation study for multi-hypothesis scoring")
    parser.add_argument("--checkpoint", type=str, default="/data/shuoyang67/checkpoint/NewPoseProject/Thot_mixste_multi_cluster-prune_3layer-81_h5", help="Path to checkpoint directory")
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    run_ablation(args)
