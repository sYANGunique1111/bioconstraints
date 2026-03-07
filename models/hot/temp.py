import torch

def interpolate_pose_batch(pruned_poses, kept_indices, target_seq_len):
    """
    Recovers original sequence length from pruned pose data.
    
    Args:
        pruned_poses: Tensor of shape (B, F_prime, J, C)
        kept_indices: Tensor of shape (B, F_prime) representing the frame indices kept.
                      MUST be sorted in ascending order.
        target_seq_len: Int (F), the original total number of frames.
        
    Returns:
        Tensor of shape (B, F, J, C)
    """
    if pruned_poses.ndim != 4:
        raise ValueError(f"pruned_poses must have shape (B, F', J, C), got {tuple(pruned_poses.shape)}")
    if kept_indices.ndim != 2:
        raise ValueError(f"kept_indices must have shape (B, F'), got {tuple(kept_indices.shape)}")

    B, F_prime, J, C = pruned_poses.shape
    if kept_indices.shape[0] != B or kept_indices.shape[1] != F_prime:
        raise ValueError(
            f"kept_indices shape {tuple(kept_indices.shape)} does not match (B, F') = ({B}, {F_prime})"
        )
    if F_prime < 1:
        raise ValueError("F' must be >= 1")
    if target_seq_len < 1:
        raise ValueError("target_seq_len must be >= 1")
    
    # 1. Flatten J and C to treat them as independent features
    # Shape becomes (B, F_prime, D) where D = J*C
    y_old = pruned_poses.view(B, F_prime, -1)
    
    # 2. Prepare x_old (indices)
    # Ensure indices are floats for calculation, shape (B, F_prime)
    x_old = kept_indices.to(device=pruned_poses.device, dtype=torch.float32)
    if not torch.all(x_old[:, 1:] >= x_old[:, :-1]):
        raise ValueError("kept_indices must be sorted in ascending order per batch")
    
    # 3. Prepare x_new (target indices 0, 1, ..., F-1)
    # Shape (1, F) -> Broadcasts to (B, F) later
    x_new = torch.arange(target_seq_len, device=pruned_poses.device, dtype=torch.float32)
    x_new = x_new.unsqueeze(0).expand(B, -1).contiguous()

    # If only one frame survives pruning, best possible reconstruction is constant replication.
    if F_prime == 1:
        y_new = y_old[:, :1, :].expand(B, target_seq_len, y_old.shape[-1])
        return y_new.view(B, target_seq_len, J, C)
    
    # --- Interpolation Core (Batched) ---
    
    # Find neighbors for x_new in x_old
    # result shape: (B, F)
    # We rely on x_old being sorted.
    indices = torch.searchsorted(x_old, x_new, side='right')
    
    # Clamp indices to handle boundary extrapolation (constantly holds first/last frame)
    indices = torch.clamp(indices, 1, F_prime - 1)
    
    # Gather left/right neighbors
    # helper to expand indices for gathering on the feature dim
    def gather_on_time(target_tensor, idx_tensor):
        # target: (B, F', D), idx: (B, F) -> output: (B, F, D)
        # We need to expand idx to match D dimension
        D_dim = target_tensor.shape[-1]
        expanded_idx = idx_tensor.unsqueeze(-1).expand(-1, -1, D_dim)
        return torch.gather(target_tensor, 1, expanded_idx)

    idx_left = indices - 1
    idx_right = indices

    # Get the coordinate values (time) for neighbors
    # x_old is (B, F'), we gather to get (B, F)
    x_l = torch.gather(x_old, 1, idx_left)
    x_r = torch.gather(x_old, 1, idx_right)
    
    # Get the pose values for neighbors
    # y_old is (B, F', D)
    y_l = gather_on_time(y_old, idx_left)
    y_r = gather_on_time(y_old, idx_right)

    # Calculate linear weights
    # shape (B, F)
    eps = 1e-6
    weights = (x_new - x_l) / (x_r - x_l + eps)
    
    # Expand weights for the feature dimension for multiplication
    weights = weights.unsqueeze(-1) 

    # Linear Interpolation: y = y_l + w * (y_r - y_l)
    y_new = y_l + weights * (y_r - y_l)
    
    # 4. Reshape back to (B, F, J, C)
    return y_new.view(B, target_seq_len, J, C)

if __name__ == "__main__":
    # --- Usage Example ---
    B, F, J, C = 2, 10, 15, 3  # Original shape
    F_prime = 4                # Pruned length

    # Create dummy pruned data
    pruned_data = torch.randn(B, F_prime, J, C)

    # Create dummy indices (Sorted!)
    # Batch 0 kept frames: [0, 3, 6, 9]
    # Batch 1 kept frames: [0, 2, 5, 9]
    indices = torch.tensor([
        [0, 3, 6, 9],
        [0, 2, 5, 9]
    ])

    recovered = interpolate_pose_batch(pruned_data, indices, target_seq_len=F)
    print(f"Output shape: {recovered.shape}")  # torch.Size([2, 10, 15, 3])
