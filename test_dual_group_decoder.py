"""
Test script for DualGroupDecoder
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import DualGroupDecoder, SimpleJointDecoder

def test_dual_group_decoder():
    """Test the DualGroupDecoder"""
    
    batch_size = 4
    num_frame = 243
    num_joints = 17
    patch_size = 9
    hidden_size = 512
    out_channels = 3
    
    # Default joint groups
    joint_groups = [
        [1, 2, 3],        # Right Leg
        [4, 5, 6],        # Left Leg  
        [0, 7, 8, 9, 10], # Torso/Spine
        [11, 12, 13],     # Left Arm
        [14, 15, 16]      # Right Arm
    ]
    num_groups = len(joint_groups)
    
    T_patches = num_frame // patch_size  # 27
    P = num_joints + num_groups  # 22
    
    print("=" * 70)
    print("Test 1: DualGroupDecoder")
    print("=" * 70)
    
    # Create decoder
    decoder = DualGroupDecoder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        hidden_size=hidden_size,
        out_channels=out_channels,
        joint_groups=joint_groups,
        joint_weight=0.5,
        semantic_weight=0.5
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder parameters: {total_params:,}")
    
    # Create input tokens (B, T_patches, J + num_groups, hidden_size)
    tokens = torch.randn(batch_size, T_patches, P, hidden_size)
    print(f"Input tokens shape: {tokens.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = decoder(tokens)
    
    expected_shape = (batch_size, num_frame, num_joints, out_channels)
    print(f"Output shape: {output.shape}")
    print(f"Expected shape: {expected_shape}")
    
    assert output.shape == expected_shape, \
        f"Shape mismatch! Got {output.shape}, expected {expected_shape}"
    
    print("\n✓ DualGroupDecoder test passed!")
    
    # Check that all joints are covered
    print("\nJoint Coverage Analysis:")
    all_joints = set()
    for i, group in enumerate(joint_groups):
        all_joints.update(group)
        print(f"  Group {i}: joints {group}")
    
    missing_joints = set(range(num_joints)) - all_joints
    if missing_joints:
        print(f"  WARNING: Joints {missing_joints} not covered by any semantic group!")
    else:
        print(f"  ✓ All {num_joints} joints covered by semantic groups")
    
    # Check for overlaps
    overlap_count = {j: 0 for j in range(num_joints)}
    for group in joint_groups:
        for j in group:
            overlap_count[j] += 1
    
    overlapping_joints = {j: count for j, count in overlap_count.items() if count > 1}
    if overlapping_joints:
        print(f"\n  Overlapping joints (predicted by multiple semantic groups):")
        for j, count in overlapping_joints.items():
            print(f"    Joint {j}: {count} semantic groups")
        print(f"    → These will be averaged within semantic predictions first")
    else:
        print(f"\n  ✓ No overlapping joints in semantic groups")
    
    print("\n" + "=" * 70)
    print("Test 2: Compare with SimpleJointDecoder")
    print("=" * 70)
    
    # Create simple decoder for comparison
    simple_decoder = SimpleJointDecoder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        hidden_size=hidden_size,
        out_channels=out_channels
    )
    
    simple_params = sum(p.numel() for p in simple_decoder.parameters())
    
    # Simple decoder only uses joint tokens
    joint_tokens = tokens[:, :, :num_joints, :]
    with torch.no_grad():
        simple_output = simple_decoder(joint_tokens)
    
    print(f"SimpleJointDecoder parameters: {simple_params:,}")
    print(f"DualGroupDecoder parameters: {total_params:,}")
    print(f"Parameter increase: {total_params - simple_params:,} ({(total_params/simple_params - 1)*100:.2f}%)")
    
    print("\nKey Differences:")
    print("  SimpleJointDecoder: Uses only 17 joint tokens")
    print("  DualGroupDecoder: Uses 17 joint + 5 semantic tokens")
    print("  DualGroupDecoder: Weighted combination (default 0.5 each)")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    
    return decoder, output


if __name__ == "__main__":
    decoder, output = test_dual_group_decoder()
