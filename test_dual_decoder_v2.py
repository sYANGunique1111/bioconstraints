"""
Test script for DualGroupDecoderV2
"""
import sys
sys.path.append('/users/shuoyang67/projects/NewPoseProject')

import torch
from models.mixste import DualGroupDecoderV2, DualGroupDecoder

def test_dual_group_decoder_v2():
    """Test DualGroupDecoderV2 and compare with V1"""
    
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
    
    print("=" * 70)
    print("DualGroupDecoderV2 Test")
    print("=" * 70)
    
    # V2: Simplified (flattened tokens, learnable weight)
    print("\nDualGroupDecoderV2:")
    decoder_v2 = DualGroupDecoderV2(
        num_frame=num_frame,
        num_joints=num_joints,
        patch_size=patch_size,
        hidden_size=hidden_size,
        out_channels=out_channels,
        num_groups=num_groups,
        init_weight=0.5
    )
    v2_params = sum(p.numel() for p in decoder_v2.parameters())
    print(f"  Parameters: {v2_params:,}")
    print(f"  Initial alpha (raw): {decoder_v2.alpha.item():.4f}")
    print(f"  Initial alpha (sigmoid): {torch.sigmoid(decoder_v2.alpha).item():.4f}")
    
    with torch.no_grad():
        output_v2 = decoder_v2(tokens)
    print(f"  Input shape: {tokens.shape}")
    print(f"  Output shape: {output_v2.shape}")
    
    # Verify shapes
    expected_shape = (batch_size, num_frame, num_joints, out_channels)
    assert output_v2.shape == expected_shape
    
    print("\n✓ Output shape is correct!")
    
    print("\n" + "=" * 70)
    print("Architecture Details")
    print("=" * 70)
    print("V2 Design:")
    print("  - Flatten joint tokens: (B, 27, 17, 512) → (B, 27, 8704)")
    print("  - Flatten semantic tokens: (B, 27, 5, 512) → (B, 27, 2560)")
    print("  - Joint head: 8704 → 459 (all 17 joints)")
    print("  - Semantic head: 2560 → 459 (all 17 joints)")
    print("  - Combination: Learnable α ∈ (0, 1) via sigmoid")
    print(f"\n  Joint head params: {sum(p.numel() for p in decoder_v2.joint_head.parameters()):,}")
    print(f"  Semantic head params: {sum(p.numel() for p in decoder_v2.semantic_head.parameters()):,}")
    print(f"  Alpha param: 1")
    print(f"  Total: {v2_params:,}")
    
    print("\n" + "=" * 70)
    print("Learnable Weight Details")
    print("=" * 70)
    print(f"alpha (raw parameter): {decoder_v2.alpha.item():.4f}")
    print(f"alpha (after sigmoid): {torch.sigmoid(decoder_v2.alpha).item():.4f}")
    print(f"\nCombination formula:")
    print(f"  final = sigmoid(α) * joint_pred + (1 - sigmoid(α)) * semantic_pred")
    print(f"  Currently: {torch.sigmoid(decoder_v2.alpha).item():.2f} * joint + {(1-torch.sigmoid(decoder_v2.alpha)).item():.2f} * semantic")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)
    
    return decoder_v2, output_v2


if __name__ == "__main__":
    v2, out2 = test_dual_group_decoder_v2()
