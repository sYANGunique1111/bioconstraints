import torch
import numpy as np
import sys
import os

# Add current directory to path to import common
sys.path.append(os.getcwd())

from common.loss import joint_angle_loss, compute_joint_angles, H36M_PARENTS, H36M_CHILDREN, DEFAULT_ANGLE_LIMITS

def test_joint_angle_logic():
    print("=== Testing Joint Angle Logic ===")
    
    # Create a synthetic pose for a single limb (Right Leg)
    # 0: Pelvis, 1: RHip, 2: RKnee, 3: RAnkle
    # H36M indices: 0, 1, 2, 3
    # Parents: -1, 0, 1, 2
    
    # Case 1: Perfectly straight limb (along Z axis)
    # Pelvis at (0,0,0), Hip at (0,0,1), Knee at (0,0,2), Ankle at (0,0,3)
    pose_straight = torch.zeros((1, 1, 17, 3))
    pose_straight[0, 0, 0] = torch.tensor([0, 0, 0.0])
    pose_straight[0, 0, 1] = torch.tensor([0, 0, 1.0])
    pose_straight[0, 0, 2] = torch.tensor([0, 0, 2.0])
    pose_straight[0, 0, 3] = torch.tensor([0, 0, 3.0])
    
    angles_straight = compute_joint_angles(pose_straight, H36M_PARENTS)
    knee_angle_straight = angles_straight[0, 0, 2].item()
    print(f"Straight limb knee angle (rad): {knee_angle_straight:.4f} (Expected: ~{np.pi:.4f})")
    print(f"Straight limb knee angle (deg): {np.degrees(knee_angle_straight):.2f} (Expected: 180.00)")

    # Case 2: 90 degree bent limb (L-shape)
    # Thigh: (0,0,0) to (0,0,1)
    # Shin: (0,0,1) to (1,0,1)
    pose_bent = torch.zeros((1, 1, 17, 3))
    pose_bent[0, 0, 1] = torch.tensor([0, 0, 0.0]) # Hip
    pose_bent[0, 0, 2] = torch.tensor([0, 0, 1.0]) # Knee
    pose_bent[0, 0, 3] = torch.tensor([1, 0, 1.0]) # Ankle
    
    angles_bent = compute_joint_angles(pose_bent, H36M_PARENTS)
    knee_angle_bent = angles_bent[0, 0, 2].item()
    print(f"90deg bent limb knee angle (rad): {knee_angle_bent:.4f} (Expected: ~{np.pi/2:.4f})")
    print(f"90deg bent limb knee angle (deg): {np.degrees(knee_angle_bent):.2f} (Expected: 90.00)")

    # Case 3: Impossible bend (Acute angle, 30 degrees)
    # Vectors: (0,0,1) and (-cos(30), 0, -sin(30)) relative to knee
    # This would be a very sharp bend.
    angle_rad = np.radians(30)
    pose_broken = torch.zeros((1, 1, 17, 3))
    pose_broken[0, 0, 1] = torch.tensor([0, 0, 0.0]) # Hip
    pose_broken[0, 0, 2] = torch.tensor([0, 0, 1.0]) # Knee
    pose_broken[0, 0, 3] = torch.tensor([np.sin(angle_rad), 0, 1.0 - np.cos(angle_rad)]) # Acute angle
    
    angles_broken = compute_joint_angles(pose_broken, H36M_PARENTS)
    knee_angle_broken = angles_broken[0, 0, 2].item()
    print(f"Sharp bend knee angle (deg): {np.degrees(knee_angle_broken):.2f}")

def test_loss_function():
    print("\n=== Testing Joint Angle Loss ===")
    
    # Default limit for knee (index 2) is (0.26 rad, pi rad)
    # 0.26 rad is approx 15 degrees.
    
    # Test 1: Straight limb (Should be 0 loss if pi is the max)
    pose_straight = torch.zeros((1, 1, 17, 3))
    pose_straight[0, 0, 1] = torch.tensor([0, 0, 0.0])
    pose_straight[0, 0, 2] = torch.tensor([0, 0, 1.0])
    pose_straight[0, 0, 3] = torch.tensor([0, 0, 2.0])
    
    loss_straight = joint_angle_loss(pose_straight)
    print(f"Loss for straight limb: {loss_straight.item():.6f}")

    # Test 2: Invalid bend (5 degrees, limit is 15 degrees)
    angle_invalid = np.radians(5)
    pose_invalid = torch.zeros((1, 1, 17, 3))
    pose_invalid[0, 0, 1] = torch.tensor([0, 0, 0.0]) # Hip
    pose_invalid[0, 0, 2] = torch.tensor([0, 0, 1.0]) # Knee
    # Vector 1: (0,0,1)
    # Vector 2: opposite direction but slightly offset by 5 deg
    # anatomical angle 5 deg means vectors are almost opposite
    pose_invalid[0, 0, 3] = torch.tensor([np.sin(angle_invalid), 0, 1.0 - np.cos(angle_invalid)])
    
    actual_angle = compute_joint_angles(pose_invalid, H36M_PARENTS)[0, 0, 2].item()
    print(f"Invalid angle: {np.degrees(actual_angle):.2f} deg (Limit min: {np.degrees(0.26):.2f} deg)")
    
    loss_invalid = joint_angle_loss(pose_invalid)
    print(f"Loss for invalid bend: {loss_invalid.item():.6f}")

    # Test 3: Multiple violations
    # Set knee and elbow to bad angles
    pose_multi = pose_invalid.clone()
    # Left elbow (12): limit (0.15, pi)
    pose_multi[0, 0, 11] = torch.tensor([0, 0, 0.0]) # Shoulder
    pose_multi[0, 0, 12] = torch.tensor([0, 0, 1.0]) # Elbow
    pose_multi[0, 0, 13] = torch.tensor([0.01, 0, 1.0 - 0.9999]) # Very sharp bend
    
    loss_multi = joint_angle_loss(pose_multi)
    print(f"Loss for multiple violations: {loss_multi.item():.6f}")

if __name__ == "__main__":
    test_joint_angle_logic()
    test_loss_function()
