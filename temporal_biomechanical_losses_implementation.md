# Temporal Biomechanical Losses for 3D Human Pose Estimation

## Purpose

Implement two new temporal biomechanical losses for 3D human pose estimation:

1. **Temporal average symmetry loss**
2. **Top-k supervised joint acceleration loss**

The existing temporal limb-length consistency loss has already been implemented and should **not** be reimplemented here.

The frequency-domain temporal loss idea has been discarded because it may be too restrictive for actions that naturally involve frequent joint-angle changes.

---

## Input Tensor Convention

Predicted 3D pose:

\[
\hat{X} \in \mathbb{R}^{B \times T \times J \times C}
\]

Ground-truth 3D pose:

\[
X^{GT} \in \mathbb{R}^{B \times T \times J \times C}
\]

where:

- \(B\): batch size
- \(T\): temporal length
- \(J\): number of joints
- \(C = 3\): 3D coordinates

Expected tensor shape in code:

```python
pred.shape == (B, T, J, 3)
target.shape == (B, T, J, 3)
```

Use the same coordinate representation as the existing training pipeline. Do not add an extra absolute/root-relative conversion inside these losses unless the project already does so elsewhere.

---

# 1. Temporal Average Symmetry Loss

## Goal

Enforce left-right skeletal symmetry at the temporal sequence level.

This loss should compare the **temporal average bone lengths** of corresponding left/right bones. It should not compare left/right bones independently at every frame.

Small asymmetry is allowed using a margin threshold:

\[
\tau = 0.0005 \text{ meters}
\]

If the project stores poses in millimeters, this is equivalent to:

\[
\tau = 0.5 \text{ mm}
\]

---

## Required Inputs

The implementation needs a list of symmetric bone pairs.

Each bone is represented by two joint indices:

```python
bone = (joint_start, joint_end)
```

A symmetric bone pair is represented as:

```python
((left_start, left_end), (right_start, right_end))
```

Example structure:

```python
symmetric_bone_pairs = [
    ((left_shoulder, left_elbow), (right_shoulder, right_elbow)),
    ((left_elbow, left_wrist), (right_elbow, right_wrist)),
    ((left_hip, left_knee), (right_hip, right_knee)),
    ((left_knee, left_ankle), (right_knee, right_ankle)),
]
```

The exact joint indices should follow the dataset/model skeleton definition.

---

## Mathematical Definition

For each symmetric bone pair \(k\), compute the left and right bone lengths at every frame:

\[
\hat{l}_{b,t,L,k}
=
\left\|
\hat{X}_{b,t,j^L_1}
-
\hat{X}_{b,t,j^L_2}
\right\|_2
\]

\[
\hat{l}_{b,t,R,k}
=
\left\|
\hat{X}_{b,t,j^R_1}
-
\hat{X}_{b,t,j^R_2}
\right\|_2
\]

Then compute temporal average bone lengths over the whole sequence:

\[
\bar{l}_{b,L,k}
=
\frac{1}{T}
\sum_{t=1}^{T}
\hat{l}_{b,t,L,k}
\]

\[
\bar{l}_{b,R,k}
=
\frac{1}{T}
\sum_{t=1}^{T}
\hat{l}_{b,t,R,k}
\]

Apply the margin threshold:

\[
L_{sym}
=
\frac{1}{BK}
\sum_{b=1}^{B}
\sum_{k=1}^{K}
\max
\left(
0,
\left|
\bar{l}_{b,L,k}
-
\bar{l}_{b,R,k}
\right|
-
\tau
\right)
\]

where:

- \(K\): number of symmetric bone pairs
- \(\tau = 0.0005\) meters by default

---

## Implementation Notes

- The loss is computed only from `pred`.
- It does not require `target`.
- The temporal average is computed over dimension `T`.
- The margin threshold should be configurable.
- The output should be a scalar tensor.
- If poses are stored in millimeters, set `tau=0.5` instead of `0.0005`.

---

## PyTorch-Style Pseudocode

```python
import torch


def temporal_average_symmetry_loss(
    pred: torch.Tensor,
    symmetric_bone_pairs: list,
    tau: float = 0.0005,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        pred:
            Tensor of shape (B, T, J, 3).
        symmetric_bone_pairs:
            List of symmetric bone pairs.
            Each item has format:
                ((left_start, left_end), (right_start, right_end))
        tau:
            Margin threshold. Default assumes coordinates are in meters.
            Use tau=0.5 if coordinates are in millimeters.
        eps:
            Small value for numerical stability.

    Returns:
        Scalar tensor.
    """

    assert pred.ndim == 4, "pred should have shape (B, T, J, 3)"
    assert pred.shape[-1] == 3, "last dimension should be 3D coordinates"

    losses = []

    for left_bone, right_bone in symmetric_bone_pairs:
        l_start, l_end = left_bone
        r_start, r_end = right_bone

        # Shape: (B, T, 3)
        left_vec = pred[:, :, l_start, :] - pred[:, :, l_end, :]
        right_vec = pred[:, :, r_start, :] - pred[:, :, r_end, :]

        # Shape: (B, T)
        left_len = torch.sqrt(torch.sum(left_vec ** 2, dim=-1) + eps)
        right_len = torch.sqrt(torch.sum(right_vec ** 2, dim=-1) + eps)

        # Shape: (B,)
        left_mean = left_len.mean(dim=1)
        right_mean = right_len.mean(dim=1)

        # Shape: (B,)
        diff = torch.abs(left_mean - right_mean)

        # Margin penalty
        pair_loss = torch.relu(diff - tau)

        losses.append(pair_loss)

    if len(losses) == 0:
        return pred.new_tensor(0.0)

    # Shape: (B, K)
    losses = torch.stack(losses, dim=1)

    return losses.mean()
```

---

# 2. Top-k Supervised Joint Acceleration Loss

## Goal

Implement a supervised temporal loss based on joint acceleration.

This is **not** an acceleration smoothing loss.

It should not force acceleration to be small. Instead, it should compare predicted joint acceleration with ground-truth joint acceleration at the most dynamically active predicted joint-time locations.

The loss should:

1. compute predicted joint acceleration
2. compute ground-truth joint acceleration
3. compute predicted acceleration magnitude
4. select top-k largest predicted acceleration entries per sample
5. compare predicted and GT acceleration only at those selected entries
6. average over \(B \times k\)

---

## Mathematical Definition

Predicted joint acceleration:

\[
\Delta^2 \hat{X}_{b,t,j}
=
\hat{X}_{b,t+2,j}
-
2\hat{X}_{b,t+1,j}
+
\hat{X}_{b,t,j}
\]

Ground-truth joint acceleration:

\[
\Delta^2 X^{GT}_{b,t,j}
=
X^{GT}_{b,t+2,j}
-
2X^{GT}_{b,t+1,j}
+
X^{GT}_{b,t,j}
\]

for:

\[
t = 1, \dots, T-2
\]

In zero-indexed code, this corresponds to:

```python
acc = x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
```

The acceleration tensor has shape:

```python
(B, T - 2, J, 3)
```

Compute predicted acceleration magnitude:

\[
a_{b,t,j}
=
\left\|
\Delta^2 \hat{X}_{b,t,j}
\right\|_2
\]

Flatten the temporal and joint dimensions:

\[
a_b \in \mathbb{R}^{(T-2)J}
\]

Select the top-k entries independently for each batch sample:

\[
\mathcal{K}_b = TopK(a_b)
\]

Then compare predicted acceleration with GT acceleration at the selected joint-time entries:

\[
L_{acc}
=
\frac{1}{Bk}
\sum_{b=1}^{B}
\sum_{(t,j) \in \mathcal{K}_b}
\rho
\left(
\Delta^2 \hat{X}_{b,t,j}
-
\Delta^2 X^{GT}_{b,t,j}
\right)
\]

The selected entries correspond to specific \((t, j)\) pairs. Therefore, there is **no additional averaging over all joints** after top-k selection.

---

## Top-k Selection Rule

Top-k should be selected over the flattened temporal-joint dimension:

```python
(T - 2) * J
```

per sample.

That means each selected item corresponds to one specific joint at one specific temporal acceleration position.

Do **not** select top-k frames and then average over all joints. That would be a different loss.

---

## Coordinate Representation

Use the same coordinate representation as the main pose supervision.

- If the model is trained on root-relative poses, compute acceleration on root-relative poses.
- If the model is trained on camera-space poses, compute acceleration on camera-space poses.
- Do not perform additional root-centering inside this loss unless the rest of the pipeline already does so.

---

## Recommended Error Function

Keep the acceleration error function configurable.

Common options:

### Option A: L2 vector norm

\[
\rho(v) = \|v\|_2
\]

This produces one scalar error per selected joint-time entry.

### Option B: SmoothL1 over coordinates

Apply SmoothL1 to the selected acceleration vectors and average over coordinates.

This may be more robust to outliers.

The implementation should follow the loss style already used in the project. If no preference exists, SmoothL1 is a safe default.

---

## Implementation Notes

- This loss requires both `pred` and `target`.
- Both tensors must have shape `(B, T, J, 3)`.
- Need `T >= 3`.
- Top-k is selected using predicted acceleration magnitude, not GT magnitude and not acceleration error.
- Top-k is selected independently per batch sample.
- Clamp `k` so it does not exceed `(T - 2) * J`.
- If `k <= 0`, return zero loss.
- The output should be a scalar tensor.

---

## PyTorch-Style Pseudocode

```python
import torch
import torch.nn.functional as F


def topk_joint_acceleration_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int,
    loss_type: str = "smooth_l1",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Args:
        pred:
            Predicted pose tensor of shape (B, T, J, 3).
        target:
            Ground-truth pose tensor of shape (B, T, J, 3).
        k:
            Number of top acceleration entries selected per batch sample.
            Top-k is selected over flattened (T - 2) * J entries.
        loss_type:
            One of {"smooth_l1", "l1", "l2_norm", "mse"}.
        eps:
            Small value for numerical stability.

    Returns:
        Scalar tensor.
    """

    assert pred.ndim == 4, "pred should have shape (B, T, J, 3)"
    assert target.ndim == 4, "target should have shape (B, T, J, 3)"
    assert pred.shape == target.shape, "pred and target should have the same shape"
    assert pred.shape[-1] == 3, "last dimension should be 3D coordinates"

    B, T, J, C = pred.shape

    if T < 3 or k <= 0:
        return pred.new_tensor(0.0)

    # Shape: (B, T - 2, J, 3)
    pred_acc = pred[:, 2:, :, :] - 2.0 * pred[:, 1:-1, :, :] + pred[:, :-2, :, :]
    target_acc = target[:, 2:, :, :] - 2.0 * target[:, 1:-1, :, :] + target[:, :-2, :, :]

    # Predicted acceleration magnitude.
    # Shape: (B, T - 2, J)
    pred_acc_mag = torch.sqrt(torch.sum(pred_acc ** 2, dim=-1) + eps)

    # Flatten temporal and joint dimensions.
    # Shape: (B, (T - 2) * J)
    pred_acc_mag_flat = pred_acc_mag.reshape(B, -1)

    num_entries = pred_acc_mag_flat.shape[1]
    k = min(k, num_entries)

    # Shape: (B, k)
    _, topk_idx = torch.topk(pred_acc_mag_flat, k=k, dim=1, largest=True, sorted=False)

    # Flatten acceleration tensors to match topk indices.
    # Shape: (B, (T - 2) * J, 3)
    pred_acc_flat = pred_acc.reshape(B, num_entries, C)
    target_acc_flat = target_acc.reshape(B, num_entries, C)

    # Expand indices for gathering 3D vectors.
    # Shape: (B, k, 3)
    gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, C)

    # Shape: (B, k, 3)
    selected_pred_acc = torch.gather(pred_acc_flat, dim=1, index=gather_idx)
    selected_target_acc = torch.gather(target_acc_flat, dim=1, index=gather_idx)

    diff = selected_pred_acc - selected_target_acc

    if loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(selected_pred_acc, selected_target_acc, reduction="mean")

    elif loss_type == "l1":
        loss = torch.abs(diff).mean()

    elif loss_type == "mse":
        loss = torch.mean(diff ** 2)

    elif loss_type == "l2_norm":
        # Shape: (B, k)
        loss = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps).mean()

    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")

    return loss
```

---

# Combined Usage Example

These two losses can be implemented as independent functions and weighted externally by the training code.

Example:

```python
loss_sym = temporal_average_symmetry_loss(
    pred=pred_3d,
    symmetric_bone_pairs=symmetric_bone_pairs,
    tau=0.0005,
)

loss_acc = topk_joint_acceleration_loss(
    pred=pred_3d,
    target=target_3d,
    k=topk_acc_k,
    loss_type="smooth_l1",
)

loss = base_loss + lambda_sym * loss_sym + lambda_acc * loss_acc
```

The values of `lambda_sym`, `lambda_acc`, and `topk_acc_k` should be treated as hyperparameters.

---

# Final Implementation Checklist

## Temporal Average Symmetry Loss

- [ ] Accept predicted 3D pose tensor with shape `(B, T, J, 3)`.
- [ ] Accept symmetric bone pair list.
- [ ] Compute left/right bone lengths at every frame.
- [ ] Average each bone length over the temporal dimension.
- [ ] Compare temporal average left/right lengths.
- [ ] Apply margin threshold `tau = 0.0005 m` by default.
- [ ] Return scalar loss.

## Top-k Supervised Joint Acceleration Loss

- [ ] Accept predicted and target 3D pose tensors with shape `(B, T, J, 3)`.
- [ ] Compute second-order temporal acceleration for both prediction and GT.
- [ ] Compute predicted acceleration magnitude.
- [ ] Flatten `(T - 2, J)` into one joint-time dimension.
- [ ] Select top-k entries per batch sample using predicted acceleration magnitude.
- [ ] Gather corresponding predicted and GT acceleration vectors.
- [ ] Compare selected acceleration vectors.
- [ ] Average only over selected `B * k` entries, not over all joints.
- [ ] Return scalar loss.

---

# Key Design Decisions

1. The already implemented temporal limb-length loss should not be touched.
2. Symmetry is enforced using temporal average bone lengths, not frame-level bone lengths.
3. Symmetry uses a margin threshold of `0.0005 m`.
4. Joint acceleration loss is supervised, not a smoothing prior.
5. Top-k acceleration entries are selected from predicted acceleration magnitude.
6. Top-k selection is over flattened joint-time entries: `(T - 2) * J`.
7. The acceleration loss averages only over selected entries.
8. The loss should use the same coordinate representation as the main pose loss.
9. No frequency-domain temporal loss should be implemented.
