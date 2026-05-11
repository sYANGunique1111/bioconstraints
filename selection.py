x.shape == (B, H, T, J, C)
selected_posewise.shape      == (B, H, T, J, C)   # greedy path for each initial hypothesis
selected_jointwise.shape     == (B, H, T, J, C)   # joint-wise greedy path for each initial hypothesis
selected_viterbi.shape       == (B, T, J, C)      # one globally optimal path per batch
path_posewise.shape          == (B, H, T)
path_jointwise.shape         == (B, H, T, J)
path_viterbi.shape           == (B, T)

import torch


def pairwise_pose_cost(prev, curr, reduction="mean", metric="l2"):
    """
    Compute pairwise pose-level transition costs.

    Args:
        prev: (B, H, J, C), hypotheses at frame t-1
        curr: (B, H, J, C), hypotheses at frame t
        reduction: "mean" or "sum" over joints
        metric: "l2" or "l2_squared"

    Returns:
        cost: (B, H_prev, H_next)
    """
    # (B, H_prev, H_next, J, C)
    diff = prev[:, :, None, :, :] - curr[:, None, :, :, :]

    if metric == "l2":
        joint_cost = torch.norm(diff, dim=-1)  # (B, H_prev, H_next, J)
    elif metric == "l2_squared":
        joint_cost = (diff ** 2).sum(dim=-1)   # (B, H_prev, H_next, J)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    if reduction == "mean":
        return joint_cost.mean(dim=-1)
    elif reduction == "sum":
        return joint_cost.sum(dim=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def pairwise_joint_cost(prev, curr, metric="l2"):
    """
    Compute pairwise joint-level transition costs.

    Args:
        prev: (B, H, J, C), hypotheses at frame t-1
        curr: (B, H, J, C), hypotheses at frame t
        metric: "l2" or "l2_squared"

    Returns:
        cost: (B, H_prev, H_next, J)
    """
    # (B, H_prev, H_next, J, C)
    diff = prev[:, :, None, :, :] - curr[:, None, :, :, :]

    if metric == "l2":
        return torch.norm(diff, dim=-1)         # (B, H_prev, H_next, J)
    elif metric == "l2_squared":
        return (diff ** 2).sum(dim=-1)          # (B, H_prev, H_next, J)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def greedy_posewise_selection(x, reduction="mean", metric="l2"):
    """
    Greedy pose-wise hypothesis selection.

    For each initial hypothesis at frame 0, greedily extends one full-body path.

    Args:
        x: (B, H, T, J, C)

    Returns:
        selected: (B, H, T, J, C)
            selected[b, h0] is the greedy pose sequence starting from hypothesis h0.
        paths: (B, H, T)
            hypothesis index selected at each frame.
    """
    assert x.ndim == 5, "Expected x with shape (B, H, T, J, C)"
    B, H, T, J, C = x.shape
    device = x.device

    paths = torch.empty(B, H, T, dtype=torch.long, device=device)

    # One path for each possible initial hypothesis.
    paths[:, :, 0] = torch.arange(H, device=device)[None, :]

    batch_idx = torch.arange(B, device=device)[:, None]

    for t in range(1, T):
        prev = x[:, :, t - 1]  # (B, H, J, C)
        curr = x[:, :, t]      # (B, H, J, C)

        cost = pairwise_pose_cost(
            prev,
            curr,
            reduction=reduction,
            metric=metric,
        )  # (B, H_prev, H_next)

        prev_h = paths[:, :, t - 1]  # (B, H_start)

        # For each current path, get costs from its previous hypothesis to all next hypotheses.
        # selected_cost[b, h_start, h_next]
        selected_cost = cost[batch_idx, prev_h, :]  # (B, H_start, H_next)

        next_h = selected_cost.argmin(dim=-1)       # (B, H_start)
        paths[:, :, t] = next_h

    selected = gather_pose_paths(x, paths)          # (B, H, T, J, C)

    return selected, paths


def greedy_jointwise_selection(x, metric="l2"):
    """
    Greedy joint-wise hypothesis selection.

    For each initial hypothesis at frame 0, each joint independently chooses
    its next hypothesis at every frame.

    This can create hybrid poses where different joints come from different
    diffusion hypotheses.

    Args:
        x: (B, H, T, J, C)

    Returns:
        selected: (B, H, T, J, C)
            selected[b, h0] is a hybrid sequence starting from hypothesis h0.
        paths: (B, H, T, J)
            paths[b, h0, t, j] is the selected hypothesis for joint j at frame t.
    """
    assert x.ndim == 5, "Expected x with shape (B, H, T, J, C)"
    B, H, T, J, C = x.shape
    device = x.device

    paths = torch.empty(B, H, T, J, dtype=torch.long, device=device)

    # At frame 0, every joint starts from the same initial full-body hypothesis.
    init_h = torch.arange(H, device=device)[None, :, None].expand(B, H, J)
    paths[:, :, 0, :] = init_h

    batch_idx = torch.arange(B, device=device)[:, None, None]
    joint_idx = torch.arange(J, device=device)[None, None, :]

    for t in range(1, T):
        prev = x[:, :, t - 1]  # (B, H, J, C)
        curr = x[:, :, t]      # (B, H, J, C)

        cost = pairwise_joint_cost(prev, curr, metric=metric)
        # cost: (B, H_prev, H_next, J)

        prev_h = paths[:, :, t - 1, :]  # (B, H_start, J)

        # Move joint dimension before H_next for easier indexing:
        # cost_by_joint: (B, H_prev, J, H_next)
        cost_by_joint = cost.permute(0, 1, 3, 2)

        # selected_cost: (B, H_start, J, H_next)
        selected_cost = cost_by_joint[batch_idx, prev_h, joint_idx, :]

        next_h = selected_cost.argmin(dim=-1)  # (B, H_start, J)
        paths[:, :, t, :] = next_h

    selected = gather_joint_paths(x, paths)    # (B, H, T, J, C)

    return selected, paths


def viterbi_posewise_selection(x, reduction="mean", metric="l2"):
    """
    Viterbi-style global pose-wise hypothesis selection.

    Finds one globally optimal hypothesis path per batch sequence under
    pairwise temporal transition cost.

    Args:
        x: (B, H, T, J, C)

    Returns:
        selected: (B, T, J, C)
            globally selected pose sequence.
        path: (B, T)
            selected hypothesis index at each frame.
        dp: (B, T, H)
            accumulated minimum cost ending at each hypothesis.
    """
    assert x.ndim == 5, "Expected x with shape (B, H, T, J, C)"
    B, H, T, J, C = x.shape
    device = x.device
    dtype = x.dtype

    dp = torch.zeros(B, T, H, dtype=dtype, device=device)
    backptr = torch.empty(B, T, H, dtype=torch.long, device=device)

    # No previous hypothesis at t=0.
    backptr[:, 0, :] = -1

    for t in range(1, T):
        prev = x[:, :, t - 1]  # (B, H, J, C)
        curr = x[:, :, t]      # (B, H, J, C)

        trans_cost = pairwise_pose_cost(
            prev,
            curr,
            reduction=reduction,
            metric=metric,
        )  # (B, H_prev, H_next)

        total_cost = dp[:, t - 1, :, None] + trans_cost
        # total_cost: (B, H_prev, H_next)

        best_cost, best_prev = total_cost.min(dim=1)
        # best_cost: (B, H_next)
        # best_prev: (B, H_next)

        dp[:, t, :] = best_cost
        backptr[:, t, :] = best_prev

    # Pick best final hypothesis.
    last_h = dp[:, T - 1, :].argmin(dim=-1)  # (B,)

    path = torch.empty(B, T, dtype=torch.long, device=device)
    path[:, T - 1] = last_h

    batch_idx = torch.arange(B, device=device)

    # Backtrack.
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = backptr[batch_idx, t, path[:, t]]

    selected = gather_single_path(x, path)  # (B, T, J, C)

    return selected, path, dp


def gather_pose_paths(x, paths):
    """
    Gather full-body paths.

    Args:
        x: (B, H, T, J, C)
        paths: (B, P, T), where P is number of paths, usually H

    Returns:
        selected: (B, P, T, J, C)
    """
    B, H, T, J, C = x.shape
    _, P, _ = paths.shape

    x_t = x.permute(0, 2, 1, 3, 4)  # (B, T, H, J, C)

    batch_idx = torch.arange(B, device=x.device)[:, None, None]
    time_idx = torch.arange(T, device=x.device)[None, None, :]

    selected = x_t[batch_idx, time_idx, paths]  # (B, P, T, J, C)

    return selected


def gather_joint_paths(x, paths):
    """
    Gather joint-wise hybrid paths.

    Args:
        x: (B, H, T, J, C)
        paths: (B, P, T, J)

    Returns:
        selected: (B, P, T, J, C)
    """
    B, H, T, J, C = x.shape
    _, P, _, _ = paths.shape

    x_t = x.permute(0, 2, 3, 1, 4)  # (B, T, J, H, C)

    batch_idx = torch.arange(B, device=x.device)[:, None, None, None]
    time_idx = torch.arange(T, device=x.device)[None, None, :, None]
    joint_idx = torch.arange(J, device=x.device)[None, None, None, :]

    selected = x_t[batch_idx, time_idx, joint_idx, paths]
    # (B, P, T, J, C)

    return selected


def gather_single_path(x, path):
    """
    Gather one full-body path per batch.

    Args:
        x: (B, H, T, J, C)
        path: (B, T)

    Returns:
        selected: (B, T, J, C)
    """
    B, H, T, J, C = x.shape

    x_t = x.permute(0, 2, 1, 3, 4)  # (B, T, H, J, C)

    batch_idx = torch.arange(B, device=x.device)[:, None]
    time_idx = torch.arange(T, device=x.device)[None, :]

    selected = x_t[batch_idx, time_idx, path]  # (B, T, J, C)

    return selected