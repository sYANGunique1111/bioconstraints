"""
Evaluation script for Pose Estimation Checkpoints, supporting action-wise breakdown.

Usage:
    python evaluate.py --checkpoint /path/to/checkpoint_dir
"""

import os
import yaml
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from common.loss import mpjpe, p_mpjpe


def fetch(keypoints, dataset, subjects, action_filter=None):
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
        raise ValueError(f"Unsupported dataset: {dataset_name} in this fallback evaluate script.")

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
                
    # Extract unique action prefixes
    action_prefixes_set = set()
    for subject in subjects_test:
        if subject in dataset.subjects():
            for action in dataset[subject].keys():
                action_prefixes_set.add(action.split(' ')[0])
    action_prefixes = sorted(list(action_prefixes_set))

    return dataset, keypoints, subjects_test, action_prefixes


def build_model_from_config(config):
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
        hot_args.use_pairwise_flow = config.get("use_pairwise_flow") in [True, "True", "true"]
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
    else:
        raise NotImplementedError(f"Model '{model_name}' is not yet implemented in evaluate.py.")


def evaluate(args):
    checkpoint_dir = args.checkpoint
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if 'dataset' not in config:
        config['dataset'] = 'h36m'
    
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model: {config.get('model')}")
    
    model = build_model_from_config(config)
    
    weights_path = os.path.join(checkpoint_dir, "best_epoch.bin")
    if not os.path.exists(weights_path):
        print(f"Warning: {weights_path} not found. Trying to find any .bin file...")
        bins = [f for f in os.listdir(checkpoint_dir) if f.endswith('.bin')]
        if bins:
            weights_path = os.path.join(checkpoint_dir, bins[0])
        else:
            raise FileNotFoundError("No checkpoint .bin file found.")
            
    print(f"Loading weights from: {weights_path}")
    checkpoint_data = torch.load(weights_path, map_location="cpu")
    
    state_dict = checkpoint_data.get("model_pos", checkpoint_data)
    clean_state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    
    model = model.cuda()
    model.eval()

    print("Loading test dataset and keypoints...")
    dataset, keypoints, subjects_test, action_prefixes = load_dataset_and_keypoints(config)
    
    # Store per-action metrics
    action_mpjpes = []
    action_p_mpjpes = []
    
    print("\nStarting action-wise evaluation...")
    print("-" * 60)
    print(f"{'Action':<20} | {'MPJPE (mm)':<15} | {'PA-MPJPE (mm)':<15}")
    print("-" * 60)

    with torch.no_grad():
        for action in action_prefixes:
            cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test, action_filter=[action])
            if poses_test is None:
                continue
                
            test_data = ChunkedGenerator(
                config["batch_size"] // config["number_of_frames"],
                cameras_test, poses_test, poses_test_2d,
                config["number_of_frames"],
                pad=0, causal_shift=0, shuffle=False, augment=False,
            )
            
            test_loader = DataLoader(
                test_data,
                batch_size=args.batch_size if args.batch_size else max(1, config["batch_size"] // config["number_of_frames"]),
                shuffle=False,
                num_workers=config.get("num_workers", 4),
                pin_memory=True,
            )
            
            total_mpjpe = 0.0
            total_p_mpjpe = 0.0
            N = 0
            
            for batch_idx, (_, inputs_3d, inputs_2d) in enumerate(test_loader):
                inputs_2d = inputs_2d.cuda().float()
                inputs_3d = inputs_3d.cuda().float()
                inputs_3d[:, :, 0] = 0

                predicted_3d = model(inputs_2d)
                predicted_3d[:, :, 0] = 0
                
                error = mpjpe(predicted_3d, inputs_3d).item()
                p_error = p_mpjpe(predicted_3d.cpu().numpy(), inputs_3d.cpu().numpy())

                batch_size = inputs_3d.shape[0] * inputs_3d.shape[1]
                total_mpjpe += error * batch_size
                total_p_mpjpe += p_error * batch_size
                N += batch_size
                
            if N > 0:
                avg_mpjpe = (total_mpjpe / N) * 1000
                avg_p_mpjpe = (total_p_mpjpe / N) * 1000
                
                action_mpjpes.append(avg_mpjpe)
                action_p_mpjpes.append(avg_p_mpjpe)
                
                print(f"{action:<20} | {avg_mpjpe:<15.3f} | {avg_p_mpjpe:<15.3f}")

    print("-" * 60)
    
    # Calculate unweighted averages across all action categories
    overall_mpjpe = np.mean(action_mpjpes)
    overall_p_mpjpe = np.mean(action_p_mpjpes)
    
    print("\n" + "=" * 50)
    print("Action-wise Unweighted Average Results (matching DiffCardGCN evaluation):")
    print(f"  MPJPE:    {overall_mpjpe:.3f} mm")
    print(f"  PA-MPJPE: {overall_p_mpjpe:.3f} mm")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Checkpoint with Action-wise breakdowns")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for dataloader (optional)")
    args = parser.parse_args()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluate(args)
