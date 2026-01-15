"""
Test script to verify loss function computations.
"""

import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, '/users/shuoyang67/projects/NewPoseProject')

from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error_train


def test_mpjpe():
    """Test MPJPE computation."""
    print("Testing MPJPE...")
    
    # Create identical tensors - error should be 0
    pred = torch.randn(4, 81, 17, 3)
    target = pred.clone()
    
    error = mpjpe(pred, target)
    assert error.item() == 0, f"Expected 0 error for identical tensors, got {error.item()}"
    print(f"  Zero error test passed: {error.item():.6f}")
    
    # Create tensors with known difference
    pred = torch.zeros(2, 10, 17, 3)
    target = torch.ones(2, 10, 17, 3)
    
    error = mpjpe(pred, target)
    expected = np.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
    assert abs(error.item() - expected) < 1e-5, f"Expected {expected}, got {error.item()}"
    print(f"  Known error test passed: {error.item():.6f} (expected {expected:.6f})")


def test_p_mpjpe():
    """Test P-MPJPE computation."""
    print("\nTesting P-MPJPE...")
    
    # Create identical poses - error should be 0 after alignment
    pred = np.random.randn(10, 17, 3)
    target = pred.copy()
    
    error = p_mpjpe(pred, target)
    assert error < 1e-10, f"Expected near-zero error for identical poses, got {error}"
    print(f"  Zero error test passed: {error:.10f}")
    
    # Create scaled poses - P-MPJPE should align them
    scale = 2.0
    pred_scaled = pred * scale
    
    error = p_mpjpe(pred_scaled, target)
    assert error < 1e-10, f"Expected near-zero error after alignment, got {error}"
    print(f"  Scale invariance test passed: {error:.10f}")


def test_velocity_error():
    """Test velocity error computation."""
    print("\nTesting Velocity Error...")
    
    # Create tensors with zero velocity
    pred = torch.ones(4, 81, 17, 3)
    target = torch.ones(4, 81, 17, 3)
    
    error = mean_velocity_error_train(pred, target, axis=1)
    assert error.item() == 0, f"Expected 0 for constant poses, got {error.item()}"
    print(f"  Zero velocity test passed: {error.item():.6f}")


if __name__ == '__main__':
    print("=" * 50)
    print("Loss Function Tests")
    print("=" * 50)
    
    try:
        test_mpjpe()
        test_p_mpjpe()
        test_velocity_error()
        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        raise
