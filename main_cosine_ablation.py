import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
import torch.nn.functional as F

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, world_to_camera
from common.generators import ChunkedGenerator
from main import fetch
from models.hot.mixste import HOTMixSTEChunkedCompression

# Monkey-patch the ortho loss to also compute mean absolute cosine similarity
original_ortho_loss = HOTMixSTEChunkedCompression._compute_chunk_orthogonality_loss

def augmented_ortho_loss(self, x):
    if x.shape[1] <= 1:
        self.latest_chunk_ortho_loss_abs = x.new_zeros(())
        return original_ortho_loss(self, x)

    tokens = rearrange(x, 'b f n c -> b n f c')
    tokens = F.normalize(tokens, p=2, dim=-1, eps=1e-6)
    sim = torch.matmul(tokens, tokens.transpose(-1, -2))

    diag_mask = torch.eye(sim.shape[-1], device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(diag_mask.view(1, 1, sim.shape[-1], sim.shape[-1]), 0.0)

    denom = x.shape[0] * x.shape[1] * x.shape[2] * max(x.shape[1] - 1, 1)
    sq_loss = sim.square().sum() / denom
    abs_loss = sim.abs().sum() / denom
    
    self.latest_chunk_ortho_loss_abs = abs_loss
    return sq_loss

HOTMixSTEChunkedCompression._compute_chunk_orthogonality_loss = augmented_ortho_loss

def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    
    config_path = os.path.join(args.checkpoint, "config.yaml")
    if os.path.exists(config_path):
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            # Override command-line defaults with the saved checkpoint config
            for k, v in config.items():
                if hasattr(args, k):
                    setattr(args, k, v)
                    
    print(f'Using layer_index={args.layer_index}, token_num={args.token_num}')
    
    print('Loading dataset...')
    dataset_root = '/data/shuoyang67/dataset/H36m/annot'
    dataset_path = f'{dataset_root}/data_3d_{args.dataset}.npz'
    
    from common.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
    keypoints = np.load(f'{dataset_root}/data_2d_{args.dataset}_{args.keypoints}.npz', allow_pickle=True)
    args.subjects_test = 'S9,S11'

    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

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

    for subject in keypoints.keys():
        if subject not in dataset.cameras():
            continue
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_test = args.subjects_test.split(',')
    cameras_test, poses_test, poses_test_2d = fetch(keypoints, dataset, subjects_test, None)

    test_data = ChunkedGenerator(
        args.batch_size // args.number_of_frames, cameras_test, poses_test, poses_test_2d,
        args.number_of_frames, pad=0, causal_shift=0, shuffle=False, augment=False)

    print(f'INFO: Testing on {test_data.num_frames()} frames')
    
    valid_loader = DataLoader(test_data, batch_size=args.batch_size // args.number_of_frames,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)

    hot_args = type('HotArgs', (), {})()
    hot_args.frames = args.number_of_frames
    hot_args.channel = args.embed_dim
    hot_args.n_joints = args.num_joints
    hot_args.token_num = args.token_num
    hot_args.layer_index = int(args.layer_index)
    hot_args.pruning_strategy = args.pruning_strategy
    # Force ortho loss calculation
    hot_args.use_chunk_ortho_loss = True
    hot_args.lambda_chunk_ortho = 1.0

    model_pos = HOTMixSTEChunkedCompression(hot_args).cuda()
    
    chk_path = os.path.join(args.checkpoint, 'best_epoch.bin')
    print(f'Loading checkpoint {chk_path}')
    checkpoint = torch.load(chk_path, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'], strict=False)
    
    model_pos.eval()
    
    total_sq_sim_eval = 0
    total_abs_sim_eval = 0
    N_valid = 0

    with torch.no_grad():
        for _, inputs_3d, inputs_2d in valid_loader:
            inputs_2d = inputs_2d.cuda().float()
            inputs_3d = inputs_3d.cuda().float()
            
            # Forward pass
            predicted_3d = model_pos(inputs_2d)
            
            sq_sim = model_pos.latest_chunk_ortho_loss
            abs_sim = model_pos.latest_chunk_ortho_loss_abs
            
            # batch_size = inputs_3d.shape[0] * inputs_3d.shape[1]
            batch_size = inputs_3d.shape[0]
            total_sq_sim_eval += batch_size * sq_sim.item()
            total_abs_sim_eval += batch_size * abs_sim.item()
            N_valid += batch_size

    avg_sq_sim = total_sq_sim_eval / N_valid
    avg_abs_sim = total_abs_sim_eval / N_valid
    
    print(f'Average Squared Cosine Similarity: {avg_sq_sim:.6f}')
    print(f'Average Absolute Cosine Similarity: {avg_abs_sim:.6f}')
    
    out_file = os.path.join(args.checkpoint, 'cosine_sim_results.txt')
    with open(out_file, 'w') as f:
        f.write(f'Average Squared Cosine Similarity: {avg_sq_sim:.6f}\n')
        f.write(f'Average Absolute Cosine Similarity: {avg_abs_sim:.6f}\n')
        
    print(f'Results logged to {out_file}')

if __name__ == '__main__':
    main()
