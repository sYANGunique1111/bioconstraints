"""
Test script to verify MixSTE2 model instantiation and forward pass.
"""

import sys
import torch

# Add project root to path
sys.path.insert(0, '/users/shuoyang67/projects/NewPoseProject')

from models.mixste import MixSTE2


def test_model_instantiation():
    """Test that the model can be instantiated with default parameters."""
    print("Testing model instantiation...")
    
    model = MixSTE2(
        num_frame=243,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=512,
        depth=8,
        num_heads=8,
        mlp_ratio=2.
    )
    
    print(f"  Model created successfully")
    print(f"  Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


def test_forward_pass():
    """Test that the model can perform a forward pass."""
    print("\nTesting forward pass...")
    
    # Create model
    model = MixSTE2(
        num_frame=81,  # Smaller for testing
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=256,  # Smaller for testing
        depth=4,
        num_heads=8,
        mlp_ratio=2.
    )
    model.eval()
    
    # Create dummy input: (batch, frames, joints, channels)
    batch_size = 4
    num_frames = 81
    num_joints = 17
    in_channels = 2
    
    x = torch.randn(batch_size, num_frames, num_joints, in_channels)
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    expected_shape = (batch_size, num_frames, num_joints, 3)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print("  Forward pass successful!")


def test_cuda_forward():
    """Test forward pass on CUDA if available."""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping CUDA test")
        return
    
    print("\nTesting CUDA forward pass...")
    
    model = MixSTE2(
        num_frame=81,
        num_joints=17,
        in_chans=2,
        embed_dim_ratio=256,
        depth=4,
        num_heads=8,
        mlp_ratio=2.
    ).cuda()
    model.eval()
    
    x = torch.randn(2, 81, 17, 2).cuda()
    
    with torch.no_grad():
        output = model(x)
    
    print(f"  Input device: {x.device}")
    print(f"  Output device: {output.device}")
    print(f"  Output shape: {output.shape}")
    print("  CUDA forward pass successful!")


if __name__ == '__main__':
    print("=" * 50)
    print("MixSTE2 Model Tests")
    print("=" * 50)
    
    try:
        test_model_instantiation()
        test_forward_pass()
        test_cuda_forward()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise
