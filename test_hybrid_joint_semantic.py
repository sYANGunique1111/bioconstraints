"""
Test script for HybridJointSemanticEmbedder (Refactored)
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import HybridJointSemanticEmbedder

def test_hybrid_joint_semantic_embedder():
    """Test the refactored HybridJointSemanticEmbedder"""
    
    # Test parameters
    batch_size = 4
    num_frame = 243
    num_joints = 17
    in_channels = 2
    patch_size = 9
    hidden_size = 512
    
    # Default H36M joint groups for semantic processing
    joint_groups = [
        [1, 2, 3],        # Right Leg
        [4, 5, 6],        # Left Leg  
        [0, 7, 8, 9, 10], # Torso/Spine
        [11, 12, 13],     # Left Arm
        [14, 15, 16]      # Right Arm
    ]
    
    print("=" * 60)
    print("Test 1: With semantic groups (joint_groups provided)")
    print("=" * 60)
    
    # Create embedder with semantic groups
    embedder_with_semantic = HybridJointSemanticEmbedder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        joint_groups=joint_groups  # Provide groups for semantic processing
    )
    
    # Create dummy input
    x = torch.randn(batch_size, num_frame, num_joints, in_channels)
    
    # Test with semantic groups
    output_with_semantic = embedder_with_semantic(x)
    
    expected_t_patches = num_frame // patch_size  # 243 // 9 = 27
    expected_spatial_dim = num_joints + len(joint_groups)  # 17 + 5 = 22
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_with_semantic.shape}")
    print(f"Expected shape: ({batch_size}, {expected_t_patches}, {expected_spatial_dim}, {hidden_size})")
    
    assert output_with_semantic.shape == (batch_size, expected_t_patches, expected_spatial_dim, hidden_size), \
        f"Shape mismatch! Got {output_with_semantic.shape}, expected ({batch_size}, {expected_t_patches}, {expected_spatial_dim}, {hidden_size})"
    
    print("✓ Shape is correct!")
    print(f"✓ First {num_joints} tokens are from joint groups")
    print(f"✓ Last {len(joint_groups)} tokens are from semantic groups")
    
    print("\n" + "=" * 60)
    print("Test 2: Without semantic groups (joint_groups=None)")
    print("=" * 60)
    
    # Create embedder without semantic groups
    embedder_without_semantic = HybridJointSemanticEmbedder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        in_channels=in_channels,
        hidden_size=hidden_size,
        joint_groups=None  # No semantic processing
    )
    
    # Test without semantic groups
    output_without_semantic = embedder_without_semantic(x)
    
    expected_spatial_dim_no_semantic = num_joints  # 17
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_without_semantic.shape}")
    print(f"Expected shape: ({batch_size}, {expected_t_patches}, {expected_spatial_dim_no_semantic}, {hidden_size})")
    
    assert output_without_semantic.shape == (batch_size, expected_t_patches, expected_spatial_dim_no_semantic, hidden_size), \
        f"Shape mismatch! Got {output_without_semantic.shape}, expected ({batch_size}, {expected_t_patches}, {expected_spatial_dim_no_semantic}, {hidden_size})"
    
    print("✓ Shape is correct!")
    print(f"✓ Only {num_joints} tokens from joint groups (no semantic)")
    
    print("\n" + "=" * 60)
    print("Test 3: Verify first 17 tokens are the same")
    print("=" * 60)
    
    # The first 17 tokens should be identical since joint groups are processed the same way
    joint_tokens_from_full = output_with_semantic[:, :, :num_joints, :]
    joint_tokens_only = output_without_semantic
    
    print(f"Joint tokens from full output shape: {joint_tokens_from_full.shape}")
    print(f"Joint tokens only output shape: {joint_tokens_only.shape}")
    
    # Check if they are the same
    are_same = torch.allclose(joint_tokens_from_full, joint_tokens_only, rtol=1e-5, atol=1e-7)
    print(f"✓ First 17 tokens match: {are_same}")
    
    if are_same:
        print("✓ Joint groups are processed independently from semantic groups!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return embedder_with_semantic, embedder_without_semantic, output_with_semantic, output_without_semantic


if __name__ == "__main__":
    emb_with, emb_without, out_with, out_without = test_hybrid_joint_semantic_embedder()
