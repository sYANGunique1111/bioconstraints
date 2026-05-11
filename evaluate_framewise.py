"""
Frame-wise Evaluation: Compute per-temporal-position MPJPE across the test set.

For a model that takes T-frame sequences, this script computes the average
MPJPE at each frame position t ∈ [0, T-1], revealing which temporal positions
the model finds easier or harder to predict.

Outputs:
  - Console table: MPJPE (mm) at each frame index
  - Text file: detailed results saved next to checkpoint
  - Optional: per-joint breakdown at each frame position

Usage:
    python evaluate_framewise.py --checkpoint /path/to/checkpoint_dir
    python evaluate_framewise.py --checkpoint /path/to/checkpoint_dir --per_joint
    python evaluate_framewise.py --checkpoint /path/to/checkpoint_dir --per_action
"""

import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from common.loss import p_mpjpe


# H36M joint names for per-joint reporting
H36M_JOINT_NAMES = [
    "Pelvis", "R.Hip", "R.Knee", "R.Ankle",
    "L.Hip", "L.Knee", "L.Ankle",
    "Spine", "Thorax", "Neck", "Head",
    "L.Shoulder", "L.Elbow", "L.Wrist",
    "R.Shoulder", "R.Elbow", "R.Wrist",
]


def fetch(keypoints, dataset, subjects, action_filter=None):
    out_poses_3d, out_poses_2d, out_camera_params = [], [], []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                if not any(action.startswith(a) for a in action_filter):
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


def load_dataset_and_keypoints(config):
    dataset_root = "/data/shuoyang67/dataset/H36m/annot"
    dataset_name = config.get("dataset", "h36m")
    dataset_path = f"{dataset_root}/data_3d_{dataset_name}.npz"

    if dataset_name == "h36m":
        from common.h36m_dataset import Human36mDataset
        dataset = Human36mDataset(dataset_path)
        keypoints = np.load(
            f"{dataset_root}/data_2d_{dataset_name}_{config['keypoints']}.npz",
            allow_pickle=True,
        )
        subjects_test = config.get("subjects_test", "S9,S11").split(",")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

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

    action_prefixes_set = set()
    for subject in subjects_test:
        if subject in dataset.subjects():
            for action in dataset[subject].keys():
                action_prefixes_set.add(action.split(' ')[0])
    action_prefixes = sorted(list(action_prefixes_set))

    return dataset, keypoints, subjects_test, action_prefixes


def build_model_from_config(config):
    """Build model from config.yaml — mirrors evaluate.py logic."""
    model_name = config.get("model")
    if model_name == "hot_mixste_chunked":
        from models.hot.mixste import HOTMixSTEChunkedCompression
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = config["number_of_frames"]
        hot_args.channel = config["embed_dim"]
        hot_args.n_joints = config["num_joints"]
        hot_args.token_num = config["token_num"]
        hot_args.layer_index = config["layer_index"]
        hot_args.pruning_strategy = config.get("pruning_strategy", "cluster")
        hot_args.use_chunk_ortho_loss = config.get("use_chunk_ortho_loss") in [True, "True", "true"]
        hot_args.lambda_chunk_ortho = float(config.get("lambda_chunk_ortho", 1e-3))
        hot_args.decoder_mode = config.get("decoder_mode", "cross_attention")
        hot_args.chunking_scheme = config.get("chunking_scheme", "even")
        return HOTMixSTEChunkedCompression(hot_args)
    elif model_name == "hot_mixste_chunked_multistep":
        from models.hot.mixste import HOTMixSTEChunkedCompressionMultiStep
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = config["number_of_frames"]
        hot_args.channel = config["embed_dim"]
        hot_args.n_joints = config["num_joints"]
        hot_args.token_num = config["token_num"]
        hot_args.layer_index = config["layer_index"]
        hot_args.pruning_strategy = config.get("pruning_strategy", "cluster")
        hot_args.use_chunk_ortho_loss = config.get("use_chunk_ortho_loss") in [True, "True", "true"]
        hot_args.lambda_chunk_ortho = float(config.get("lambda_chunk_ortho", 1e-3))
        hot_args.decoder_mode = config.get("decoder_mode", "cross_attention")
        hot_args.chunking_scheme = config.get("chunking_scheme", "even")
        hot_args.use_pairwise_flow = config.get("use_pairwise_flow") in [True, "True", "true"]
        hot_args.hierarchical_layer_indices = config.get("hierarchical_layer_indices", "2,3")
        hot_args.hierarchical_token_nums = config.get("hierarchical_token_nums", "81,27")
        return HOTMixSTEChunkedCompressionMultiStep(hot_args)
    elif model_name == "hot_mixste":
        from models.hot.mixste import HOTMixSTE
        hot_args = type("HotArgs", (), {})()
        hot_args.frames = config["number_of_frames"]
        hot_args.channel = config["embed_dim"]
        hot_args.n_joints = config["num_joints"]
        hot_args.token_num = config["token_num"]
        hot_args.layer_index = config["layer_index"]
        hot_args.pruning_strategy = config.get("pruning_strategy", "cluster")
        return HOTMixSTE(hot_args)
    elif model_name == "mixste":
        from models.mixste import MixSTE2
        return MixSTE2(
            num_frame=config["number_of_frames"],
            num_joints=config["num_joints"],
            in_chans=2,
            embed_dim_ratio=config["embed_dim"],
            depth=config.get("depth", 8),
            num_heads=config.get("num_heads", 8),
            mlp_ratio=2,
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' not implemented in evaluate_framewise.py.")


def compute_framewise_errors(predicted, target):
    """
    Compute per-frame-position MPJPE.

    Args:
        predicted: (B, T, J, 3)
        target:    (B, T, J, 3)

    Returns:
        frame_mpjpe:      (T,) mean over batch & joints, in raw units (meters)
        frame_joint_mpjpe: (T, J) mean over batch only
    """
    # Per-joint Euclidean distance: (B, T, J)
    errors = torch.norm(predicted - target, dim=-1)
    # Mean over batch per (t, j): (T, J)
    frame_joint_mpjpe = errors.mean(dim=0)
    # Mean over joints: (T,)
    frame_mpjpe = frame_joint_mpjpe.mean(dim=-1)
    return frame_mpjpe, frame_joint_mpjpe


def evaluate_framewise(args):
    checkpoint_dir = args.checkpoint
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "dataset" not in config:
        config["dataset"] = "h36m"

    T = config["number_of_frames"]
    print(f"Checkpoint : {checkpoint_dir}")
    print(f"Model      : {config.get('model')}")
    print(f"Seq length : {T}")

    # --- Build & load model ---
    model = build_model_from_config(config)
    weights_path = os.path.join(checkpoint_dir, "best_epoch.bin")
    if not os.path.exists(weights_path):
        bins = [f for f in os.listdir(checkpoint_dir) if f.endswith(".bin")]
        if bins:
            weights_path = os.path.join(checkpoint_dir, bins[0])
        else:
            raise FileNotFoundError("No checkpoint .bin file found.")
    print(f"Weights    : {weights_path}")

    checkpoint_data = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint_data.get("model_pos", checkpoint_data)
    clean_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model = model.cuda()
    model.eval()

    # --- Load data ---
    print("Loading test dataset...")
    dataset, keypoints, subjects_test, action_prefixes = load_dataset_and_keypoints(config)

    J = config.get("num_joints", 17)

    # Accumulators — weighted running mean
    # Global
    global_frame_sum = torch.zeros(T, device="cuda")          # sum of errors per frame
    global_frame_joint_sum = torch.zeros(T, J, device="cuda") # sum of errors per frame per joint
    global_N = 0                                               # total samples (batches * B)

    # Per-action (optional)
    if args.per_action:
        action_frame_sums = {}
        action_Ns = {}

    actions_to_eval = action_prefixes if args.per_action else [None]

    with torch.no_grad():
        for action in actions_to_eval:
            action_filter = [action] if action is not None else None
            cameras_test, poses_test, poses_test_2d = fetch(
                keypoints, dataset, subjects_test, action_filter=action_filter
            )
            if poses_test is None:
                continue

            test_data = ChunkedGenerator(
                config["batch_size"] // config["number_of_frames"],
                cameras_test, poses_test, poses_test_2d,
                config["number_of_frames"],
                pad=0, causal_shift=0, shuffle=False, augment=False,
            )
            bs = args.batch_size if args.batch_size else max(1, config["batch_size"] // config["number_of_frames"])
            test_loader = DataLoader(
                test_data, batch_size=bs, shuffle=False,
                num_workers=config.get("num_workers", 4), pin_memory=True,
            )

            if args.per_action and action is not None:
                action_frame_sums[action] = torch.zeros(T, device="cuda")
                action_Ns[action] = 0

            for _, inputs_3d, inputs_2d in test_loader:
                inputs_2d = inputs_2d.cuda().float()
                inputs_3d = inputs_3d.cuda().float()
                inputs_3d[:, :, 0] = 0

                predicted_3d = model(inputs_2d)
                predicted_3d[:, :, 0] = 0

                B = inputs_3d.shape[0]
                frame_mpjpe, frame_joint_mpjpe = compute_framewise_errors(predicted_3d, inputs_3d)

                global_frame_sum += frame_mpjpe * B
                global_frame_joint_sum += frame_joint_mpjpe * B
                global_N += B

                if args.per_action and action is not None:
                    action_frame_sums[action] += frame_mpjpe * B
                    action_Ns[action] += B

    # --- Compute final averages ---
    avg_frame_mpjpe = (global_frame_sum / global_N).cpu().numpy() * 1000  # mm
    avg_frame_joint_mpjpe = (global_frame_joint_sum / global_N).cpu().numpy() * 1000  # mm

    # --- Print results ---
    print("\n" + "=" * 60)
    print("Frame-wise MPJPE (mm)")
    print("=" * 60)
    print(f"{'Frame':>6} | {'MPJPE (mm)':>10} | {'Δ from mean':>11}")
    print("-" * 35)
    overall_mean = avg_frame_mpjpe.mean()
    for t in range(T):
        delta = avg_frame_mpjpe[t] - overall_mean
        sign = "+" if delta >= 0 else ""
        print(f"{t:>6} | {avg_frame_mpjpe[t]:>10.3f} | {sign}{delta:>10.3f}")
    print("-" * 35)
    print(f"{'Mean':>6} | {overall_mean:>10.3f} |")
    print(f"{'Std':>6} | {avg_frame_mpjpe.std():>10.3f} |")
    print(f"{'Min':>6} | {avg_frame_mpjpe.min():>10.3f} | frame {avg_frame_mpjpe.argmin()}")
    print(f"{'Max':>6} | {avg_frame_mpjpe.max():>10.3f} | frame {avg_frame_mpjpe.argmax()}")

    # --- Per-joint breakdown ---
    if args.per_joint:
        print("\n" + "=" * 60)
        print("Per-Joint Frame-wise MPJPE (mm)")
        print("=" * 60)
        header = f"{'Frame':>6} | " + " | ".join(f"{name:>8}" for name in H36M_JOINT_NAMES[:J])
        print(header)
        print("-" * len(header))
        for t in range(T):
            row = f"{t:>6} | " + " | ".join(f"{avg_frame_joint_mpjpe[t, j]:>8.2f}" for j in range(J))
            print(row)

    # --- Per-action breakdown ---
    if args.per_action:
        print("\n" + "=" * 60)
        print("Per-Action Frame-wise MPJPE (mm) — selected frames")
        print("=" * 60)
        # Show first, middle, last, min, max frames
        mid = T // 2
        print(f"{'Action':<20} | {'f=0':>8} | {'f=mid':>8} | {'f=last':>8} | {'f=best':>8} | {'f=worst':>8} | {'mean':>8}")
        print("-" * 90)
        for action in sorted(action_frame_sums.keys()):
            if action_Ns[action] == 0:
                continue
            a_mpjpe = (action_frame_sums[action] / action_Ns[action]).cpu().numpy() * 1000
            print(f"{action:<20} | {a_mpjpe[0]:>8.2f} | {a_mpjpe[mid]:>8.2f} | {a_mpjpe[-1]:>8.2f} | "
                  f"{a_mpjpe.min():>8.2f} | {a_mpjpe.max():>8.2f} | {a_mpjpe.mean():>8.2f}")

    # --- Save to file ---
    output_path = os.path.join(checkpoint_dir, "framewise_mpjpe.txt")
    with open(output_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_dir}\n")
        f.write(f"Model: {config.get('model')}\n")
        f.write(f"Sequence length: {T}\n")
        f.write(f"Total test samples: {global_N}\n\n")

        f.write("Frame-wise MPJPE (mm)\n")
        f.write("=" * 40 + "\n")
        for t in range(T):
            f.write(f"frame {t:>4}: {avg_frame_mpjpe[t]:.4f} mm\n")
        f.write(f"\nMean: {overall_mean:.4f} mm\n")
        f.write(f"Std:  {avg_frame_mpjpe.std():.4f} mm\n")
        f.write(f"Best frame:  {avg_frame_mpjpe.argmin()} ({avg_frame_mpjpe.min():.4f} mm)\n")
        f.write(f"Worst frame: {avg_frame_mpjpe.argmax()} ({avg_frame_mpjpe.max():.4f} mm)\n")

        if args.per_joint:
            f.write("\n\nPer-Joint Frame-wise MPJPE (mm)\n")
            f.write("=" * 40 + "\n")
            for t in range(T):
                joints_str = ", ".join(f"{H36M_JOINT_NAMES[j]}={avg_frame_joint_mpjpe[t,j]:.2f}" for j in range(J))
                f.write(f"frame {t:>4}: {joints_str}\n")

        if args.per_action:
            f.write("\n\nPer-Action Frame-wise MPJPE (mm)\n")
            f.write("=" * 40 + "\n")
            for action in sorted(action_frame_sums.keys()):
                if action_Ns[action] == 0:
                    continue
                a_mpjpe = (action_frame_sums[action] / action_Ns[action]).cpu().numpy() * 1000
                f.write(f"\n{action}:\n")
                for t in range(T):
                    f.write(f"  frame {t:>4}: {a_mpjpe[t]:.4f} mm\n")

    # Also save raw numpy arrays for plotting
    np_output_path = os.path.join(checkpoint_dir, "framewise_mpjpe.npz")
    save_dict = {
        "frame_mpjpe": avg_frame_mpjpe,
        "frame_joint_mpjpe": avg_frame_joint_mpjpe,
        "joint_names": np.array(H36M_JOINT_NAMES[:J]),
    }
    if args.per_action:
        for action in action_frame_sums:
            if action_Ns[action] > 0:
                save_dict[f"action_{action}"] = (action_frame_sums[action] / action_Ns[action]).cpu().numpy() * 1000
    np.savez(np_output_path, **save_dict)

    print(f"\nResults saved to: {output_path}")
    print(f"Raw data saved to: {np_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame-wise MPJPE evaluation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=None, help="DataLoader batch size (optional)")
    parser.add_argument("--per_joint", action="store_true", help="Also report per-joint errors at each frame")
    parser.add_argument("--per_action", action="store_true", help="Also report per-action frame-wise errors")
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluate_framewise(args)
