"""
Test script for updated HybridMixSTEWithJointConv with HybridSpatialBlock
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import HybridMixSTEWithJointConv, HybridSpatialBlock

def test_hybrid_spatial_block():
    """Test the HybridSpatialBlock"""
    print("="* 60)
    print("Test 1: HybridSpatialBlock")
    print("=" * 60)
    
    batch_size = 4
    time_patches = 27
    num_joints = 17
    num_groups = 5
    hidden_size = 512
    
    # Create block
    block = HybridSpatialBlock(
        dim=hidden_size,
        num_joints=num_joints,
        num_groups=num_groups,
        num_heads=8,
        mlp_ratio=2.
    )
    
    # Create input (joint + semantic tokens)
    P = num_joints + num_groups  # 22
    x = torch.randn(batch_size, time_patches, P, hidden_size)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = block(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {x.shape}")
    
    assert output.shape == x.shape, f"Shape mismatch! Got {output.shape}, expected {x.shape}"
    
    print("✓ HybridSpatialBlock test passed!\n")
    
    return block

def test_hybrid_mixste_with_joint_conv():
    """Test the updated HybridMixSTEWithJointConv model"""
    
    print("=" * 60)
    print("Test 2: Updated HybridMixSTEWithJointConv")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    num_frame = 243
    num_joints = 17
    in_channels = 2
    patch_size = 9
    hidden_size = 512
    depth = 8
    num_heads = 8
    
    # Create model (with default semantic groups)
    model = HybridMixSTEWithJointConv(
        num_frame=num_frame,
        num_joints=num_joints,
        in_chans=in_channels,
        embed_dim_ratio=hidden_size,
        depth=depth,
        num_heads=num_heads,
        patch_size=patch_size
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params / 1e6:.3f} Million parameters")
    print(f"Number of semantic groups: {model.num_groups}")
    
    # Create dummy input
    x = torch.randn(batch_size, num_frame, num_joints, in_channels)
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    expected_output_shape = (batch_size, num_frame, num_joints, 3)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {expected_output_shape}")
    
    # Verify shape
    assert output.shape == expected_output_shape, \
        f"Shape mismatch! Got {output.shape}, expected {expected_output_shape}"
    
    print("\n✓ Shape is correct!")
    print("✓ Model forward pass successful!")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return model, output


if __name__ == "__main__":
    # Test 1: HybridSpatialBlock
    block = test_hybrid_spatial_block()
    
    # Test 2: Complete model
    model, output = test_hybrid_mixste_with_joint_conv()
