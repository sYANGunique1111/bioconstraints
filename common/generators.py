# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from itertools import zip_longest
from torch.utils.data import Dataset
import numpy as np

     
class ChunkedGenerator(Dataset):
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        if cameras is not None:
            assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = []  # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.cameras = cameras
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return len(self.pairs) * self.chunk_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        chunk = self.pairs[idx]
        seq_i, start_3d, end_3d, flip = chunk[0], chunk[1], chunk[2], chunk[3]
        start_2d = start_3d
        end_2d = end_3d

        # 2D poses
        seq_2d = self.poses_2d[seq_i]
        low_2d = max(start_2d, 0)
        high_2d = min(end_2d, seq_2d.shape[0])
        pad_left_2d = low_2d - start_2d
        pad_right_2d = end_2d - high_2d
        if pad_left_2d != 0 or pad_right_2d != 0:
            psd_2d = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
        else:
            psd_2d = seq_2d[low_2d:high_2d]

        if flip:
            # Flip 2D keypoints
            psd_2d[:, :, 0] *= -1
            psd_2d[:, self.kps_left + self.kps_right] = psd_2d[:, self.kps_right + self.kps_left]

        # 3D poses
        if self.poses_3d is not None:
            seq_3d = self.poses_3d[seq_i]
            low_3d = max(start_3d, 0)
            high_3d = min(end_3d, seq_3d.shape[0])
            pad_left_3d = low_3d - start_3d
            pad_right_3d = end_3d - high_3d
            if pad_left_3d != 0 or pad_right_3d != 0:
                psd_3d = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
            else:
                psd_3d = seq_3d[low_3d:high_3d]

            if flip:
                # Flip 3D joints
                psd_3d[:, :, 0] *= -1
                psd_3d[:, self.joints_left + self.joints_right] = psd_3d[:, self.joints_right + self.joints_left]
            
        # Cameras
        if self.cameras is not None:
            cam = self.cameras[seq_i]
            if flip:
                # Flip horizontal distortion coefficients
                cam[2] *= -1
                cam[7] *= -1
        else:
            cam = np.empty_like(psd_3d)
        return cam, psd_3d, psd_2d
       

    def batch_num(self):
        return self.num_batches
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment


class UnchunkedGenerator(Dataset):
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
    
    def __len__(self):
        return len(self.poses_2d)
    
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def __getitem__(self, index):
        batch_cam = self.cameras[index] if len(self.cameras) > 0 else None
        batch_3d = self.poses_3d[index] if len(self.poses_3d) > 0 else None
        batch_2d = self.poses_2d[index]
        
        if self.augment:
            # Append flipped version
            if batch_cam is not None:
                batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                batch_cam[1, 2] *= -1
                batch_cam[1, 7] *= -1
            
            if batch_3d is not None:
                batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                batch_3d[1, :, :, 0] *= -1
                batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

            batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
            batch_2d[1, :, :, 0] *= -1
            batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]
            
        return batch_cam, batch_3d, batch_2d


if __name__ == '__main__':
    pass
