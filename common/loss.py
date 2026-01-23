import torch
import numpy as np


def mpjpe(predicted, target, return_joints_err=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    if not return_joints_err:
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    else:
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        # errors: [B, T, N]
        from einops import rearrange
        errors = rearrange(errors, 'B T N -> N (B T)')
        errors = torch.mean(errors, dim=-1).cpu().numpy().reshape(-1) * 1000
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1)), errors


def mpjpe_diffusion(predicted, target, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    For diffusion model outputs with multiple hypotheses.
    """
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        errors = rearrange(errors, 'b t h f n  -> t h b f n', ).reshape(t, h, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors


def mpjpe_diffusion_all_min(predicted, target, mean_pos=False):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    Returns the minimum error across all hypotheses.
    """
    if not mean_pos:
        t = predicted.shape[1]
        h = predicted.shape[2]
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t, h, 1, 1, 1)
        errors = torch.norm(predicted - target, dim=len(target.shape)-1)
        from einops import rearrange
        errors = rearrange(errors, 'b t h f n  -> t h b f n', )
        min_errors = torch.min(errors, dim=1, keepdim=False).values
        min_errors = min_errors.reshape(t, -1)
        min_errors = torch.mean(min_errors, dim=-1, keepdim=False)
        return min_errors
    else:
        t = predicted.shape[1]
        h = predicted.shape[2]
        mean_pose = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t, 1, 1, 1)
        errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
        from einops import rearrange
        errors = rearrange(errors, 'b t f n  -> t b f n', )
        errors = errors.reshape(t, -1)
        errors = torch.mean(errors, dim=-1, keepdim=False)
        return errors


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape
    
    if len(predicted.shape) == 4:
        # Reshape (B, T, J, C) -> (B*T, J, C) for frame-wise alignment
        predicted = predicted.reshape(-1, predicted.shape[2], predicted.shape[3])
        target = target.reshape(-1, target.shape[2], target.shape[3])

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a*np.matmul(muY, R)  # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))


def p_mpjpe_diffusion(predicted, target, mean_pos=False):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    For diffusion model outputs with multiple hypotheses.
    """
    b_sz, t_sz, h_sz, f_sz, j_sz, c_sz = predicted.shape
    if not mean_pos:
        target = target.unsqueeze(1).unsqueeze(1).repeat(1, t_sz, h_sz, 1, 1, 1)
    else:
        predicted = torch.mean(predicted, dim=2, keepdim=False)
        target = target.unsqueeze(1).repeat(1, t_sz, 1, 1, 1)

    predicted = predicted.cpu().numpy().reshape(-1, j_sz, c_sz)
    target = target.cpu().numpy().reshape(-1, j_sz, c_sz)

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    if not mean_pos:
        target = target.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, h_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        errors = errors.transpose(1, 2, 0, 3, 4).reshape(t_sz, h_sz, -1)  # t, h, b, f, n
        errors = np.mean(errors, axis=2, keepdims=False)
        min_errors = np.min(errors, axis=1, keepdims=False)
        return min_errors
    else:
        target = target.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        predicted_aligned = predicted_aligned.reshape(b_sz, t_sz, f_sz, j_sz, c_sz)
        errors = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)
        errors = errors.transpose(1, 0, 2, 3)
        errors = errors.reshape(t_sz, -1)
        errors = np.mean(errors, axis=1, keepdims=False)
        return errors


def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted from:
    https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
    """
    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)


def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    assert axis == 1
    velocity_predicted = predicted[:, 1:, :, :] - predicted[:, :-1, :, :]
    velocity_target = target[:, 1:, :, :] - target[:, :-1, :, :]

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))


def mean_velocity_error(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape

    velocity_predicted = np.diff(predicted, axis=axis)
    velocity_target = np.diff(target, axis=axis)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))


# =============================================================================
# Biomechanical Loss Functions
# Based on 6D rotation representation from Zhou et al. CVPR 2019:
# "On the Continuity of Rotation Representations in Neural Networks"
# =============================================================================

# Default H36M 17-joint skeleton parents
# Joint indices: 0=Pelvis, 1-3=RightLeg, 4-6=LeftLeg, 7-8=Spine/Thorax, 
#                9-10=Neck/Head, 11-13=LeftArm, 14-16=RightArm
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# Children for each joint (derived from parents)
# Used for computing joint angles correctly
H36M_CHILDREN = [
    [1, 4, 7],  # 0: Pelvis -> RHip, LHip, Spine
    [2],        # 1: RHip -> RKnee
    [3],        # 2: RKnee -> RAnkle
    [],         # 3: RAnkle (leaf)
    [5],        # 4: LHip -> LKnee
    [6],        # 5: LKnee -> LAnkle
    [],         # 6: LAnkle (leaf)
    [8],        # 7: Spine -> Thorax
    [9, 11, 14],# 8: Thorax -> Neck, LShoulder, RShoulder
    [10],       # 9: Neck -> Head
    [],         # 10: Head (leaf)
    [12],       # 11: LShoulder -> LElbow
    [13],       # 12: LElbow -> LWrist
    [],         # 13: LWrist (leaf)
    [15],       # 14: RShoulder -> RElbow
    [16],       # 15: RElbow -> RWrist
    [],         # 16: RWrist (leaf)
]

# Left/Right joint pairs for symmetry (child joints)
H36M_LEFT_JOINTS = [4, 5, 6, 11, 12, 13]   # Left leg (hip, knee, ankle) + Left arm (shoulder, elbow, wrist)
H36M_RIGHT_JOINTS = [1, 2, 3, 14, 15, 16]  # Right leg + Right arm

# Default joint angle limits in RADIANS [min, max]
# Now correctly indexed: angle AT joint j is the angle formed by bones meeting at j
# For joint j: angle between (parent->j) and (j->child)
# Convention: π = straight limb, smaller = more bent/flexed
# Limits adjusted based on GT analysis to be more permissive
DEFAULT_ANGLE_LIMITS = {
    # Knees: allow deep flexion for sitting actions
    # Range: ~15° (0.26 rad) to 180° (π rad)
    2: (0.26, np.pi),   # Right knee
    5: (0.26, np.pi),   # Left knee
    # Elbows: allow full flexion
    # Range: ~15° (0.26 rad) to 180° (π rad)  
    12: (0.26, np.pi),  # Left elbow
    15: (0.26, np.pi),  # Right elbow
    # Hips: very wide range for sitting/squatting
    # Range: ~10° (0.17 rad) to 180° (π rad)
    1: (0.17, np.pi),   # Right hip
    4: (0.17, np.pi),   # Left hip
    # Shoulders: very flexible
    # Range: ~15° (0.26 rad) to 180° (π rad)
    11: (0.26, np.pi),  # Left shoulder
    14: (0.26, np.pi),  # Right shoulder
    # Spine joints: limited flexion
    # Range: ~90° (π/2 rad) to 180° (π rad)
    7: (np.pi / 2, np.pi),  # Spine (Pelvis->Spine->Thorax)
    8: (np.pi / 3, np.pi),  # Thorax (~60° min)
    9: (np.pi / 3, np.pi),  # Neck (~60° min)
}


def rotation_6d_to_matrix(rot_6d):
    """
    Convert 6D rotation representation to rotation matrix via Gram-Schmidt orthonormalization.
    Reference: Zhou et al. CVPR 2019 "On the Continuity of Rotation Representations in Neural Networks"
    
    Args:
        rot_6d: Tensor of shape (..., 6) representing 6D rotation
        
    Returns:
        Rotation matrix of shape (..., 3, 3)
    """
    # Split into two 3D vectors
    a1 = rot_6d[..., :3]
    a2 = rot_6d[..., 3:6]
    
    # Gram-Schmidt orthonormalization
    b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
    
    # b2 = a2 - (b1 · a2) * b1
    dot = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = a2 - dot * b1
    b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
    
    # b3 = b1 × b2
    b3 = torch.cross(b1, b2, dim=-1)
    
    # Stack into rotation matrix
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(mat):
    """
    Extract 6D rotation representation from rotation matrix.
    
    Args:
        mat: Rotation matrix of shape (..., 3, 3)
        
    Returns:
        6D rotation of shape (..., 6)
    """
    return mat[..., :2].flatten(start_dim=-2)


def compute_bone_vectors(poses, parents):
    """
    Compute bone vectors from 3D poses.
    
    Args:
        poses: 3D poses of shape (B, T, J, 3) or (B, J, 3)
        parents: List of parent indices for each joint
        
    Returns:
        bone_vectors: Bone vectors of shape (..., J, 3), where bone_vectors[..., j, :] 
                      is the vector from parent[j] to j. For root joint, it's zero.
    """
    parents = torch.tensor(parents, device=poses.device, dtype=torch.long)
    
    # Get parent positions
    parent_positions = poses[..., parents, :]
    
    # Handle root joint (parent = -1) - set parent position to same as root
    root_mask = parents == -1
    parent_positions[..., root_mask, :] = poses[..., root_mask, :]
    
    # Bone vector = child position - parent position
    bone_vectors = poses - parent_positions
    
    return bone_vectors


def compute_bone_lengths(poses, parents):
    """
    Compute bone lengths from 3D poses.
    
    Args:
        poses: 3D poses of shape (B, T, J, 3) or (B, J, 3)
        parents: List of parent indices for each joint
        
    Returns:
        bone_lengths: Bone lengths of shape (..., J)
    """
    bone_vectors = compute_bone_vectors(poses, parents)
    bone_lengths = torch.norm(bone_vectors, dim=-1)
    return bone_lengths


def compute_joint_angles(poses, parents, children=None):
    """
    Compute joint angles - the angle formed AT each joint.
    
    For joint j (non-leaf, non-root):
    - Incoming bone: parent(j) -> j
    - Outgoing bone: j -> child(j) (uses first child if multiple)
    - Angle AT joint j = angle between incoming and outgoing bones
    
    Example for knee (joint 2):
    - Incoming: Hip -> Knee (thigh)
    - Outgoing: Knee -> Ankle (shin)
    - Angle: the knee flexion angle
    
    Args:
        poses: 3D poses of shape (B, T, J, 3) or (B, J, 3)
        parents: List of parent indices for each joint
        children: List of children lists for each joint (default: H36M_CHILDREN)
        
    Returns:
        angles: Joint angles in RADIANS of shape (..., J)
                Leaf joints and root have angle = π (undefined/straight)
    """
    if children is None:
        children = H36M_CHILDREN
    
    bone_vectors = compute_bone_vectors(poses, parents)
    num_joints = len(parents)
    
    # Initialize angles to π (straight/undefined for leaves and root)
    angles_rad = torch.full(poses.shape[:-1], np.pi, device=poses.device, dtype=poses.dtype)
    
    for j in range(num_joints):
        # Skip root (no incoming bone) and leaf joints (no outgoing bone)
        if parents[j] == -1 or len(children[j]) == 0:
            continue
        
        # Incoming bone: parent -> j
        incoming = bone_vectors[..., j, :]  # shape: (..., 3)
        
        # Outgoing bone: j -> first child
        # Use first child if joint has multiple children
        first_child = children[j][0]
        outgoing = bone_vectors[..., first_child, :]  # shape: (..., 3)
        
        # Normalize
        incoming_norm = incoming / (torch.norm(incoming, dim=-1, keepdim=True) + 1e-8)
        outgoing_norm = outgoing / (torch.norm(outgoing, dim=-1, keepdim=True) + 1e-8)
        
        # Vector angle between incoming and outgoing
        # When bones are parallel (straight limb): dot=1, arccos=0
        # When bones are opposite (fully bent): dot=-1, arccos=π
        dot_product = torch.sum(incoming_norm * outgoing_norm, dim=-1)
        
        # CRITICAL: Use epsilon to avoid infinite gradient at arccos boundaries
        # d/dx arccos(x) = -1/sqrt(1-x²) → ∞ as x → ±1
        # Straight limbs (common state) push dot_product to 1.0, causing explosion
        eps = 1e-6
        dot_product = torch.clamp(dot_product, -1.0 + eps, 1.0 - eps)
        
        vector_angle_rad = torch.acos(dot_product)
        
        # Convert to anatomical convention:
        # Straight limb (parallel vectors, 0 rad vector angle) = π anatomical
        # Fully bent (opposite vectors, π rad vector angle) = 0 anatomical
        angles_rad[..., j] = np.pi - vector_angle_rad
    
    return angles_rad


def bone_length_loss(predicted, target, parents=None):
    """
    Compute bone length consistency loss using L1 (absolute difference).
    Penalizes differences in bone lengths between predicted and ground truth poses.
    
    Args:
        predicted: Predicted 3D poses of shape (B, T, J, 3)
        target: Ground truth 3D poses of shape (B, T, J, 3)
        parents: List of parent indices (default: H36M_PARENTS)
        
    Returns:
        Scalar loss value (mean absolute error of bone lengths)
    """
    if parents is None:
        parents = H36M_PARENTS
    
    pred_lengths = compute_bone_lengths(predicted, parents)
    gt_lengths = compute_bone_lengths(target, parents)
    
    # Skip root joint (bone length is 0), use L1 loss
    loss = torch.mean(torch.abs(pred_lengths[..., 1:] - gt_lengths[..., 1:]))
    
    return loss


def symmetry_loss(predicted, parents=None, left_joints=None, right_joints=None):
    """
    Compute symmetry loss using L1 (absolute difference).
    Penalizes differences in bone lengths between left and right limbs.
    
    Args:
        predicted: Predicted 3D poses of shape (B, T, J, 3)
        parents: List of parent indices (default: H36M_PARENTS)
        left_joints: List of left-side joint indices (default: H36M_LEFT_JOINTS)
        right_joints: List of right-side joint indices (default: H36M_RIGHT_JOINTS)
        
    Returns:
        Scalar loss value (mean absolute error between left/right bone lengths)
    """
    if parents is None:
        parents = H36M_PARENTS
    if left_joints is None:
        left_joints = H36M_LEFT_JOINTS
    if right_joints is None:
        right_joints = H36M_RIGHT_JOINTS
    
    pred_lengths = compute_bone_lengths(predicted, parents)
    
    # Get left and right bone lengths
    left_lengths = pred_lengths[..., left_joints]
    right_lengths = pred_lengths[..., right_joints]
    
    # Symmetry loss: left and right should have same lengths, use L1 loss
    loss = torch.mean(torch.abs(left_lengths - right_lengths))
    
    return loss


def joint_angle_loss(predicted, parents=None, angle_limits=None):
    """
    Compute joint angle limit loss using L1 penalty in radians.
    Penalizes joint angles outside anatomically plausible ranges.
    
    Args:
        predicted: Predicted 3D poses of shape (B, T, J, 3)
        parents: List of parent indices (default: H36M_PARENTS)
        angle_limits: Dict mapping joint index to (min_angle, max_angle) in RADIANS
                      (default: DEFAULT_ANGLE_LIMITS)
        
    Returns:
        Scalar loss value (penalty for angles outside limits, in radians)
    """
    if parents is None:
        parents = H36M_PARENTS
    if angle_limits is None:
        angle_limits = DEFAULT_ANGLE_LIMITS
    
    angles = compute_joint_angles(predicted, parents)
    
    total_loss = torch.tensor(0.0, device=predicted.device)
    num_constrained = 0
    
    for joint_idx, (min_angle, max_angle) in angle_limits.items():
        joint_angles = angles[..., joint_idx]
        
        # Penalty for angles below minimum (over-flexion)
        below_min = torch.relu(min_angle - joint_angles)
        
        # Penalty for angles above maximum
        above_max = torch.relu(joint_angles - max_angle)
        
        # L1 penalty (in radians, comparable scale to MPJPE in meters)
        total_loss = total_loss + torch.mean(below_min + above_max)
        num_constrained += 1
    
    if num_constrained > 0:
        total_loss = total_loss / num_constrained
    
    return total_loss


def biomechanical_loss(predicted, target, parents=None, left_joints=None, right_joints=None,
                       angle_limits=None, weight_bone=0.1, weight_symmetry=0.05, weight_angle=0.01):
    """
    Compute combined biomechanical loss.
    
    Args:
        predicted: Predicted 3D poses of shape (B, T, J, 3)
        target: Ground truth 3D poses of shape (B, T, J, 3)
        parents: List of parent indices (default: H36M_PARENTS)
        left_joints: List of left-side joint indices
        right_joints: List of right-side joint indices
        angle_limits: Dict mapping joint index to (min_angle, max_angle)
        weight_bone: Weight for bone length loss
        weight_symmetry: Weight for symmetry loss
        weight_angle: Weight for joint angle loss
        
    Returns:
        total_loss: Combined weighted biomechanical loss
        loss_dict: Dictionary with individual loss components
    """
    if parents is None:
        parents = H36M_PARENTS
    
    # Compute individual losses
    loss_bone = bone_length_loss(predicted, target, parents)
    loss_sym = symmetry_loss(predicted, parents, left_joints, right_joints)
    loss_angle = joint_angle_loss(predicted, parents, angle_limits)
    
    # Weighted combination
    total_loss = (weight_bone * loss_bone + 
                  weight_symmetry * loss_sym + 
                  weight_angle * loss_angle)
    
    loss_dict = {
        'bone_length': loss_bone.item(),
        'symmetry': loss_sym.item(),
        'joint_angle': loss_angle.item(),
        'biomech_total': total_loss.item()
    }
    
    return total_loss, loss_dict
