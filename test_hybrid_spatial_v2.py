"""
Test script for HybridSpatialBlockV2 (Cross-Attention variant)
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import HybridSpatialBlock, HybridSpatialBlockV2

def test_hybrid_spatial_blocks():
    """Compare HybridSpatialBlock (MHSA) vs HybridSpatialBlockV2 (MHCA)"""
    
    batch_size = 4
    time_patches = 27
    num_joints = 17
    num_groups = 5
    hidden_size = 512
    
    P = num_joints + num_groups  # 22
    x = torch.randn(batch_size, time_patches, P, hidden_size)
    
    print("=" * 70)
    print("Test 1: HybridSpatialBlock (MHSA - Self-Attention)")
    print("=" * 70)
    
    # Create  MHSA block
    block_v1 = HybridSpatialBlock(
        dim=hidden_size,
        num_joints=num_joints,
        num_groups=num_groups,
        num_heads=8,
        mlp_ratio=2.
    )
    
    # Count parameters
    v1_params = sum(p.numel() for p in block_v1.parameters())
    print(f"Block V1 parameters: {v1_params:,}")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output_v1 = block_v1(x)
    
    print(f"Output shape: {output_v1.shape}")
    assert output_v1.shape == x.shape
    print("✓ HybridSpatialBlock (MHSA) test passed!\n")
    
    print("=" * 70)
    print("Test 2: HybridSpatialBlockV2 (MHCA - Cross-Attention)")
    print("=" * 70)
    
    # Create MHCA block
    block_v2 = HybridSpatialBlockV2(
        dim=hidden_size,
        num_joints=num_joints,
        num_groups=num_groups,
        num_heads=8,
        mlp_ratio=2.
    )
    
    # Count parameters
    v2_params = sum(p.numel() for p in block_v2.parameters())
    print(f"Block V2 parameters: {v2_params:,}")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output_v2 = block_v2(x)
    
    print(f"Output shape: {output_v2.shape}")
    assert output_v2.shape == x.shape
    print("✓ HybridSpatialBlockV2 (MHCA) test passed!\n")
    
    print("=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"V1 (MHSA) parameters: {v1_params:,}")
    print(f"V2 (MHCA) parameters: {v2_params:,}")
    print(f"Difference: {v2_params - v1_params:,} ({(v2_params/v1_params - 1)*100:.2f}% increase)")
    
    print("\nKey Difference:")
    print("  V1 (MHSA): All tokens attend to all tokens")
    print("  V2 (MHCA): Joint tokens (Q) attend to semantic tokens (K,V)")
    print("             Joints get updated by semantic context")
    print("             Semantics processed independently through MLP")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    
    return block_v1, block_v2, output_v1, output_v2


if __name__ == "__main__":
    block_v1, block_v2, out_v1, out_v2 = test_hybrid_spatial_blocks()
