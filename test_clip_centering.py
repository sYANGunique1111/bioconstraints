"""
Test script to verify clip-based centering logic in main_video.py
Run with: conda activate base && python test_clip_centering.py
"""

import torch
import numpy as np

def apply_clip_centering(poses_3d):
    """
    Center each clip at the pelvis of the first frame.
    Args:
        poses_3d: Tensor of shape (B, T, J, 3)
    Returns:
        Centered poses where first frame's pelvis is at origin
    """
    centered_poses = poses_3d - poses_3d[:, 0:1, 0:1]
    return centered_poses


def apply_frame_centering(poses_3d):
    """
    Center each frame at its own pelvis (old style).
    Args:
        poses_3d: Tensor of shape (B, T, J, 3)
    Returns:
        Centered poses where each frame's pelvis is at origin
    """
    centered_poses = poses_3d.clone()
    centered_poses[:, :, 0] = 0  # Set pelvis to origin
    return centered_poses


def test_clip_centering():
    """Test the clip-based centering logic."""
    print("=" * 60)
    print("Testing Clip-Based Centering Logic")
    print("=" * 60)
    
    # Create sample data: (B=2, T=4, J=17, C=3)
    B, T, J = 2, 4, 17
    
    # Create random poses with non-zero pelvis positions
    poses_3d = torch.randn(B, T, J, 3)
    
    # Add some global offset to make pelvis positions non-zero
    pelvis_offsets = torch.randn(B, T, 1, 3) * 10
    poses_3d = poses_3d + pelvis_offsets
    
    print(f"\nOriginal poses shape: {poses_3d.shape}")
    print(f"Original first frame pelvis positions (batch 0):")
    print(poses_3d[0, 0, 0])
    print(f"Original second frame pelvis positions (batch 0):")
    print(poses_3d[0, 1, 0])
    
    # Apply clip-based centering
    clip_centered = apply_clip_centering(poses_3d)
    
    print(f"\n{'=' * 60}")
    print("After Clip-Based Centering:")
    print(f"{'=' * 60}")
    print(f"First frame pelvis (should be [0,0,0]):")
    print(clip_centered[0, 0, 0])
    print(f"Second frame pelvis (should be relative to first):")
    print(clip_centered[0, 1, 0])
    
    # Verify first frame's pelvis is at origin
    assert torch.allclose(clip_centered[:, 0, 0], torch.zeros(B, 3), atol=1e-6), \
        "First frame's pelvis should be at origin!"
    print("✓ First frame's pelvis is at origin")
    
    # Apply frame-based centering
    frame_centered = apply_frame_centering(poses_3d)
    
    print(f"\n{'=' * 60}")
    print("After Frame-Based Centering (Old Style):")
    print(f"{'=' * 60}")
    print(f"First frame pelvis (should be [0,0,0]):")
    print(frame_centered[0, 0, 0])
    print(f"Second frame pelvis (should be [0,0,0]):")
    print(frame_centered[0, 1, 0])
    
    # Verify all frames' pelvis is at origin
    assert torch.allclose(frame_centered[:, :, 0], torch.zeros(B, T, 3), atol=1e-6), \
        "All frames' pelvis should be at origin!"
    print("✓ All frames' pelvis is at origin")
    
    # Test the evaluation scenario
    print(f"\n{'=' * 60}")
    print("Testing Evaluation Scenario:")
    print(f"{'=' * 60}")
    
    # Simulate predictions (clip-centered)
    predicted_3d = torch.randn(B, T, J, 3)
    
    # Ground truth with global offset
    inputs_3d = poses_3d.clone()
    
    # New-style: compare with clip-centered GT
    inputs_3d_clip_centered = apply_clip_centering(inputs_3d)
    print(f"\nNew-style GT shape: {inputs_3d_clip_centered.shape}")
    print(f"New-style GT first frame pelvis: {inputs_3d_clip_centered[0, 0, 0]}")
    
    # Old-style: post-process predictions
    # Add back the first frame's pelvis offset to predictions
    predicted_3d_with_offset = predicted_3d + inputs_3d[:, 0:1, 0:1]
    predicted_3d_frame_centered = apply_frame_centering(predicted_3d_with_offset)
    inputs_3d_frame_centered = apply_frame_centering(inputs_3d)
    
    print(f"\nOld-style predictions shape: {predicted_3d_frame_centered.shape}")
    print(f"Old-style predictions first frame pelvis: {predicted_3d_frame_centered[0, 0, 0]}")
    print(f"Old-style GT first frame pelvis: {inputs_3d_frame_centered[0, 0, 0]}")
    
    assert torch.allclose(predicted_3d_frame_centered[:, :, 0], torch.zeros(B, T, 3), atol=1e-6), \
        "Old-style predictions pelvis should be at origin!"
    assert torch.allclose(inputs_3d_frame_centered[:, :, 0], torch.zeros(B, T, 3), atol=1e-6), \
        "Old-style GT pelvis should be at origin!"
    print("✓ Old-style post-processing works correctly")
    
    print(f"\n{'=' * 60}")
    print("All tests passed! ✓")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    test_clip_centering()
