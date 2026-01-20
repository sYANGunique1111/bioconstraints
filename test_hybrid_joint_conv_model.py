"""
Test script for HybridMixSTEWithJointConv
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import HybridMixSTEWithJointConv

def test_hybrid_mixste_with_joint_conv():
    """Test the new HybridMixSTEWithJointConv model"""
    
    # Test parameters
    batch_size = 4
    num_frame = 243
    num_joints = 17
    in_channels = 2
    patch_size = 9
    hidden_size = 512
    depth = 8
    num_heads = 8
    
    print("=" * 60)
    print("Test: HybridMixSTEWithJointConv Model")
    print("=" * 60)
    
    # Create model
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
    
    # Test assertion for spatial dimension
    print("\n" + "=" * 60)
    print("Test: Spatial dimension assertion (should fail for >17)")
    print("=" * 60)
    
    try:
        # Manually create wrong input for embedder (this will be caught in ST_forward)
        wrong_tokens = torch.randn(batch_size, 27, 22, hidden_size)  # 22 > 17
        model.ST_forward(wrong_tokens)
        print("✗ Assertion did not trigger (this is wrong!)")
    except AssertionError as e:
        print(f"✓ Assertion triggered correctly: {e}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    
    return model, output


if __name__ == "__main__":
    model, output = test_hybrid_mixste_with_joint_conv()
