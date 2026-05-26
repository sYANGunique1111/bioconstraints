"""
Evaluation script for Pose Estimation Checkpoints, supporting action-wise breakdown.

Usage:
    python evaluate.py --checkpoint /path/to/checkpoint_dir
"""

import os
import yaml
import argparse
import numpy as np
import gc

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


def resolve_dataset_root(explicit_root=None):
    if explicit_root:
        return explicit_root

    env_root = os.environ.get("H36M_DATASET_ROOT")
    if env_root:
        return env_root

    candidates = [
        "/FARM/syangb/data/h36m",
        "/data/shuoyang67/dataset/H36m/annot",
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not resolve dataset root. Pass --dataset_root or set H36M_DATASET_ROOT."
    )


def load_dataset_and_keypoints(config, dataset_root=None):
    dataset_root = resolve_dataset_root(dataset_root)
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


def evaluate_checkpoint(checkpoint_dir, batch_size=None, sequence_chunk_mode=None, dataset_root=None, action_wise=True):
    model = None
    checkpoint_data = None
    dataset = None
    keypoints = None
    result = None
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yaml at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if 'dataset' not in config:
        config['dataset'] = 'h36m'

    effective_chunk_mode = sequence_chunk_mode or config.get("sequence_chunk_mode", "center_pad")
    
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Model: {config.get('model')}")
    print(f"Sequence chunk mode: {effective_chunk_mode}")
    
    try:
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
        del clean_state_dict
        del state_dict
        del checkpoint_data
        checkpoint_data = None
        
        model = model.cuda()
        model.eval()

        print("Loading test dataset and keypoints...")
        dataset, keypoints, subjects_test, action_prefixes = load_dataset_and_keypoints(config, dataset_root=dataset_root)
        
        action_results = []
        if action_wise:
            eval_groups = action_prefixes
            print("\nStarting action-wise evaluation...")
            print("-" * 60)
            print(f"{'Action':<20} | {'MPJPE (mm)':<15} | {'PA-MPJPE (mm)':<15}")
            print("-" * 60)
        else:
            eval_groups = [None]
            print("\nStarting aggregate evaluation on all test actions...")

        with torch.inference_mode():
            total_eval_mpjpe = 0.0
            total_eval_p_mpjpe = 0.0
            total_eval_frames = 0

            for action in eval_groups:
                action_filter = [action] if action_wise else None
                cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test, action_filter=action_filter)
                if poses_test is None:
                    continue
                    
                test_data = ChunkedGenerator(
                    config["batch_size"] // config["number_of_frames"],
                    cameras_test, poses_test, poses_test_2d,
                    config["number_of_frames"],
                    pad=0, causal_shift=0, shuffle=False, augment=False,
                    chunk_mode=effective_chunk_mode,
                )
                
                test_loader = DataLoader(
                    test_data,
                    batch_size=batch_size if batch_size else max(1, config["batch_size"] // config["number_of_frames"]),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                )
                
                total_mpjpe = 0.0
                total_p_mpjpe = 0.0
                N = 0
                
                for _, inputs_3d, inputs_2d in test_loader:
                    inputs_2d = inputs_2d.cuda(non_blocking=True).float()
                    inputs_3d = inputs_3d.cuda(non_blocking=True).float()
                    inputs_3d[:, :, 0] = 0

                    predicted_3d = model(inputs_2d)
                    predicted_3d[:, :, 0] = 0
                    
                    error = mpjpe(predicted_3d, inputs_3d).item()
                    p_error = p_mpjpe(predicted_3d.cpu().numpy(), inputs_3d.cpu().numpy())

                    frame_count = inputs_3d.shape[0] * inputs_3d.shape[1]
                    total_mpjpe += error * frame_count
                    total_p_mpjpe += p_error * frame_count
                    N += frame_count

                    del predicted_3d
                    del inputs_2d
                    del inputs_3d
                    
                del test_loader
                del test_data
                del cameras_test
                del poses_test
                del poses_test_2d
                
                if N > 0:
                    avg_mpjpe = (total_mpjpe / N) * 1000
                    avg_p_mpjpe = (total_p_mpjpe / N) * 1000

                    total_eval_mpjpe += total_mpjpe
                    total_eval_p_mpjpe += total_p_mpjpe
                    total_eval_frames += N
                    
                    if action_wise:
                        action_results.append(
                            {
                                "action": action,
                                "mpjpe_mm": avg_mpjpe,
                                "pa_mpjpe_mm": avg_p_mpjpe,
                            }
                        )
                        print(f"{action:<20} | {avg_mpjpe:<15.3f} | {avg_p_mpjpe:<15.3f}")

        if action_wise:
            print("-" * 60)

        overall_mpjpe = (total_eval_mpjpe / total_eval_frames) * 1000
        overall_p_mpjpe = (total_eval_p_mpjpe / total_eval_frames) * 1000
        
        print("\n" + "=" * 50)
        if action_wise:
            print("Overall frame-weighted results across all evaluated action chunks:")
        else:
            print("Aggregate results across all test actions:")
        print(f"  MPJPE:    {overall_mpjpe:.3f} mm")
        print(f"  PA-MPJPE: {overall_p_mpjpe:.3f} mm")
        print("=" * 50 + "\n")

        result = {
            "checkpoint": checkpoint_dir,
            "model": config.get("model"),
            "sequence_chunk_mode": effective_chunk_mode,
            "action_wise": action_wise,
            "mpjpe_mm": float(overall_mpjpe),
            "pa_mpjpe_mm": float(overall_p_mpjpe),
            "actions": action_results,
        }
        return result
    finally:
        if model is not None:
            model.cpu()
        del model
        del checkpoint_data
        del dataset
        del keypoints
        del result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(args):
    return evaluate_checkpoint(
        checkpoint_dir=args.checkpoint,
        batch_size=args.batch_size,
        sequence_chunk_mode=args.sequence_chunk_mode,
        dataset_root=args.dataset_root,
        action_wise=not args.aggregate_only,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Checkpoint with Action-wise breakdowns")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for dataloader (optional)")
    parser.add_argument(
        "--sequence_chunk_mode",
        type=str,
        default=None,
        choices=["drop", "center_pad", "tail_pad", "stride"],
        help="Override the sequence chunking mode used at evaluation time",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional H36M dataset root; defaults to H36M_DATASET_ROOT or known local paths",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip per-action breakdown and evaluate all test actions together",
    )
    args = parser.parse_args()
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    evaluate(args)
