"""
Test script to verify data loader functionality.
"""

import sys
import numpy as np

# Add project root to path
sys.path.insert(0, '/users/shuoyang67/projects/NewPoseProject')

from common.generators import ChunkedGenerator, UnchunkedGenerator


def test_chunked_generator():
    """Test ChunkedGenerator functionality."""
    print("Testing ChunkedGenerator...")
    
    # Create dummy data
    num_sequences = 5
    poses_2d = [np.random.randn(100, 17, 2) for _ in range(num_sequences)]
    poses_3d = [np.random.randn(100, 17, 3) for _ in range(num_sequences)]
    cameras = None  # Optional
    
    batch_size = 4
    chunk_length = 27
    
    generator = ChunkedGenerator(
        batch_size=batch_size,
        cameras=cameras,
        poses_3d=poses_3d,
        poses_2d=poses_2d,
        chunk_length=chunk_length,
        shuffle=True,
        augment=False
    )
    
    print(f"  Number of samples: {len(generator)}")
    print(f"  Number of frames: {generator.num_frames()}")
    
    # Test __getitem__
    cam, pose_3d, pose_2d = generator[0]
    print(f"  Sample 3D shape: {pose_3d.shape}")
    print(f"  Sample 2D shape: {pose_2d.shape}")
    
    assert pose_3d.shape == (chunk_length, 17, 3), f"Unexpected 3D shape: {pose_3d.shape}"
    assert pose_2d.shape == (chunk_length, 17, 2), f"Unexpected 2D shape: {pose_2d.shape}"
    print("  ChunkedGenerator test passed!")


def test_unchunked_generator():
    """Test UnchunkedGenerator functionality."""
    print("\nTesting UnchunkedGenerator...")
    
    # Create dummy data with varying sequence lengths
    poses_2d = [
        np.random.randn(50, 17, 2),
        np.random.randn(100, 17, 2),
        np.random.randn(75, 17, 2),
    ]
    poses_3d = [
        np.random.randn(50, 17, 3),
        np.random.randn(100, 17, 3),
        np.random.randn(75, 17, 3),
    ]
    cameras = None
    
    generator = UnchunkedGenerator(
        cameras=cameras,
        poses_3d=poses_3d,
        poses_2d=poses_2d,
        augment=False
    )
    
    print(f"  Number of sequences: {len(generator)}")
    print(f"  Total frames: {generator.num_frames()}")
    
    # Test __getitem__
    cam, pose_3d, pose_2d = generator[0]
    print(f"  First sequence 3D shape: {pose_3d.shape}")
    print(f"  First sequence 2D shape: {pose_2d.shape}")
    
    assert pose_3d.shape[0] == 50, f"Unexpected first sequence length: {pose_3d.shape[0]}"
    print("  UnchunkedGenerator test passed!")


def test_augmentation():
    """Test data augmentation."""
    print("\nTesting augmentation...")
    
    poses_2d = [np.random.randn(100, 17, 2) for _ in range(3)]
    poses_3d = [np.random.randn(100, 17, 3) for _ in range(3)]
    
    kps_left = [4, 5, 6, 11, 12, 13]
    kps_right = [1, 2, 3, 14, 15, 16]
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]
    
    generator_no_aug = ChunkedGenerator(
        batch_size=4, cameras=None, poses_3d=poses_3d, poses_2d=poses_2d,
        chunk_length=27, augment=False,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right
    )
    
    generator_with_aug = ChunkedGenerator(
        batch_size=4, cameras=None, poses_3d=poses_3d, poses_2d=poses_2d,
        chunk_length=27, augment=True,
        kps_left=kps_left, kps_right=kps_right,
        joints_left=joints_left, joints_right=joints_right
    )
    
    print(f"  Without augmentation: {len(generator_no_aug)} samples")
    print(f"  With augmentation: {len(generator_with_aug)} samples")
    
    assert len(generator_with_aug) == 2 * len(generator_no_aug), "Augmentation should double the samples"
    print("  Augmentation test passed!")


if __name__ == '__main__':
    print("=" * 50)
    print("Data Loader Tests")
    print("=" * 50)
    
    try:
        test_chunked_generator()
        test_unchunked_generator()
        test_augmentation()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise
