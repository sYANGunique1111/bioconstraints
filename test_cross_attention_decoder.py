"""
Test script for CrossAttentionDecoder
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import CrossAttentionDecoder, DualGroupDecoder, SimpleJointDecoder

def test_cross_attention_decoder():
    """Test and compare CrossAttentionDecoder with other decoders"""
    
    batch_size = 4
    num_frame = 243
    num_joints = 17
    patch_size = 9
    hidden_size = 512
    num_groups = 5
    out_channels = 3
    
    T_patches = num_frame // patch_size  # 27
    P = num_joints + num_groups  # 22
    
    # Create input tokens
    tokens = torch.randn(batch_size, T_patches, P, hidden_size)
    joint_tokens_only = tokens[:, :, :num_joints, :]
    
    print("=" * 70)
    print("Decoder Comparison")
    print("=" * 70)
    
    decoders = {}
    
    # 1. SimpleJointDecoder
    print("\n1. SimpleJointDecoder:")
    simple_decoder = SimpleJointDecoder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        hidden_size=hidden_size,
        out_channels=out_channels
    )
    simple_params = sum(p.numel() for p in simple_decoder.parameters())
    with torch.no_grad():
        simple_out = simple_decoder(joint_tokens_only)
    print(f"  Parameters: {simple_params:,}")
    print(f"  Input: (B, T_patches, 17, C)")
    print(f"  Output: {simple_out.shape}")
    decoders['Simple'] = (simple_decoder, simple_params)
    
    # 2. DualGroupDecoder
    print("\n2. DualGroupDecoder:")
    dual_decoder = DualGroupDecoder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        hidden_size=hidden_size,
        out_channels=out_channels
        # joint_groups will use defaults
    )
    dual_params = sum(p.numel() for p in dual_decoder.parameters())
    with torch.no_grad():
        dual_out = dual_decoder(tokens)
    print(f"  Parameters: {dual_params:,}")
    print(f"  Input: (B, T_patches, 22, C)")
    print(f"  Output: {dual_out.shape}")
    decoders['Dual'] = (dual_decoder, dual_params)
    
    # 3. CrossAttentionDecoder
    print("\n3. CrossAttentionDecoder:")
    cross_decoder = CrossAttentionDecoder(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        hidden_size=hidden_size,
        out_channels=out_channels,
        num_groups=num_groups,
        num_heads=8
    )
    cross_params = sum(p.numel() for p in cross_decoder.parameters())
    with torch.no_grad():
        cross_out = cross_decoder(tokens)
    print(f"  Parameters: {cross_params:,}")
    print(f"  Input: (B, T_patches, 22, C)")
    print(f"  Output: {cross_out.shape}")
    decoders['Cross'] = (cross_decoder, cross_params)
    
    # Verify shapes
    expected_shape = (batch_size, num_frame, num_joints, out_channels)
    assert simple_out.shape == expected_shape
    assert dual_out.shape == expected_shape
    assert cross_out.shape == expected_shape
    
    print("\n" + "=" * 70)
    print("Parameter Comparison")
    print("=" * 70)
    print(f"SimpleJointDecoder:     {simple_params:8,} (baseline)")
    print(f"DualGroupDecoder:       {dual_params:8,} (+{dual_params - simple_params:,}, {(dual_params/simple_params - 1)*100:+.1f}%)")
    print(f"CrossAttentionDecoder:  {cross_params:8,} (+{cross_params - simple_params:,}, {(cross_params/simple_params - 1)*100:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("Architecture Comparison")
    print("=" * 70)
    print("SimpleJointDecoder:")
    print("  - Uses: 17 joint tokens only")
    print("  - Method: Direct prediction head")
    
    print("\nDualGroupDecoder:")
    print("  - Uses: 17 joint + 5 semantic tokens")
    print("  - Method: Separate heads + weighted averaging")
    print("  - Issue: Many parameters, potentially unstable")
    
    print("\nCrossAttentionDecoder:")
    print("  - Uses: 17 joint + 5 semantic tokens")
    print("  - Method: Cross-attention → single head")
    print("  - Benefit: Simpler, fewer parameters than Dual")
    
    print("\n" + "=" * 70)
    print(f"✓ All decoders produce correct output shape: {expected_shape}")
    print("=" * 70)
    
    return decoders


if __name__ == "__main__":
    decoders = test_cross_attention_decoder()
