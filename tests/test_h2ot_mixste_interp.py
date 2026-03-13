from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hot.mixste.h2ot_mixste import H2OTMixSTEInterp, interpolate_pose_batch_with_indices


def _build_args(pruning_strategy="cluster"):
    return SimpleNamespace(
        channel=32,
        frames=27,
        n_joints=17,
        token_num=9,
        layer_index=1,
        hierarchical_layer_indices=[1, 2],
        hierarchical_token_nums=[9, 3],
        recovery_on_hierarchy=False,
        recovery_layer_indices=[],
        recovery_token_nums=[],
        pruning_strategy=pruning_strategy,
        recovery_strategy="interpolation",
    )


def _assert_strictly_increasing(indices):
    assert torch.all(indices[:, 1:] > indices[:, :-1])


def _run_and_check(pruning_strategy):
    args = _build_args(pruning_strategy=pruning_strategy)
    model = H2OTMixSTEInterp(args).eval()
    x = torch.randn(2, args.frames, args.n_joints, 2)

    with torch.no_grad():
        y, meta = model(x, return_metadata=True)

    kept = meta["final_kept_indices"]
    assert y.shape == (2, args.frames, args.n_joints, 3)
    assert kept.shape == (2, args.hierarchical_token_nums[-1])
    _assert_strictly_increasing(kept)
    assert torch.all(kept >= 0)
    assert torch.all(kept < args.frames)

    assert len(meta["stage_indices"]) == len(args.hierarchical_layer_indices)


def test_h2ot_mixste_interp_cluster_indices_sorted_and_recovered():
    _run_and_check("cluster")


def test_h2ot_mixste_interp_motion_indices_sorted_and_recovered():
    _run_and_check("motion")


def test_h2ot_mixste_interp_sampler_indices_sorted_and_recovered():
    _run_and_check("sampler")


def test_h2ot_mixste_interp_learned_indices_sorted_and_recovered():
    _run_and_check("learned")


def test_h2ot_mixste_interp_return_pre_interp_contract():
    args = _build_args(pruning_strategy="sampler")
    model = H2OTMixSTEInterp(args).eval()
    x = torch.randn(2, args.frames, args.n_joints, 2)

    with torch.no_grad():
        recovered, pre_interp, kept = model(x, return_pre_interp=True)

    assert recovered.shape == (2, args.frames, args.n_joints, 3)
    assert pre_interp.shape == (2, args.hierarchical_token_nums[-1], args.n_joints, 3)
    assert kept.shape == (2, args.hierarchical_token_nums[-1])
    _assert_strictly_increasing(kept)


def test_h2ot_mixste_interp_gt_gather_matches_pre_interp_shape():
    args = _build_args(pruning_strategy="motion")
    model = H2OTMixSTEInterp(args).eval()
    x = torch.randn(1, args.frames, args.n_joints, 2)
    gt = torch.randn(1, args.frames, args.n_joints, 3)

    with torch.no_grad():
        _, pre_interp, kept = model(x, return_pre_interp=True)

    gathered_gt = torch.gather(
        gt,
        1,
        kept[:, :, None, None].expand(-1, -1, gt.shape[2], gt.shape[3]),
    )
    assert gathered_gt.shape == pre_interp.shape


def test_interpolate_pose_batch_with_indices_hits_kept_knots():
    batch_size, frames, joints, channels = 1, 8, 2, 3
    kept_indices = torch.tensor([[0, 3, 7]], dtype=torch.long)
    pruned = torch.randn(batch_size, 3, joints, channels)

    recovered = interpolate_pose_batch_with_indices(pruned, kept_indices, frames)
    sampled = torch.gather(
        recovered,
        1,
        kept_indices[:, :, None, None].expand(batch_size, 3, joints, channels),
    )
    assert torch.allclose(sampled, pruned, atol=1e-4, rtol=1e-4)


def test_interpolate_pose_batch_with_indices_unsorted_raises():
    pruned = torch.randn(1, 3, 2, 3)
    unsorted = torch.tensor([[0, 7, 3]], dtype=torch.long)
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_pose_batch_with_indices(pruned, unsorted, 8)
