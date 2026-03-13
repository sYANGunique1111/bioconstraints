from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hot.mixste.h2ot_mixste import H2OTMixSTE


def _build_args(
    recovery_on_hierarchy=False, 
    recovery_layer_indices=None, 
    recovery_token_nums=None,
    pruning_strategy="cluster",
    recovery_strategy="attention"
):
    return SimpleNamespace(
        channel=64,
        frames=243,
        n_joints=17,
        token_num=81,
        layer_index=1,
        hierarchical_layer_indices=[1, 2],
        hierarchical_token_nums=[81, 27],
        recovery_on_hierarchy=recovery_on_hierarchy,
        recovery_layer_indices=recovery_layer_indices if recovery_layer_indices is not None else [],
        recovery_token_nums=recovery_token_nums if recovery_token_nums is not None else [],
        pruning_strategy=pruning_strategy,
        recovery_strategy=recovery_strategy,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_default_final_recovery_cuda():
    """Test default cluster+attention strategy (backward compatibility)."""
    args = _build_args(recovery_on_hierarchy=False)
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, args.frames, args.n_joints, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_hierarchical_recovery_cuda():
    """Test hierarchical recovery with cluster+attention."""
    args = _build_args(
        recovery_on_hierarchy=True,
        recovery_layer_indices=[4, 7],
        recovery_token_nums=[109, 243],
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, args.frames, args.n_joints, 3)


def test_h2ot_mixste_invalid_reduction_raises():
    """Test that invalid token reduction raises error."""
    args = _build_args()
    args.hierarchical_token_nums = [300, 27]
    with pytest.raises(ValueError):
        H2OTMixSTE(args)


# NEW TESTS FOR TPMo, TPS, TRI strategies

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_tps_pruning_cuda():
    """Test TPS (Token Pruning Sampler) pruning strategy."""
    args = _build_args(
        pruning_strategy="sampler",
        recovery_strategy="attention",  # Can use TRA with TPS
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, args.frames, args.n_joints, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_tpmo_pruning_cuda():
    """Test TPMo (Token Pruning Motion) pruning strategy."""
    args = _build_args(
        pruning_strategy="motion",
        recovery_strategy="attention",  # Can use TRA with TPMo
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)  # input_2d defaults to x internally

    assert y.shape == (2, args.frames, args.n_joints, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_tri_recovery_cuda():
    """Test TRI (Token Recovering Interpolation) recovery strategy."""
    args = _build_args(
        pruning_strategy="sampler",  # TPS is required for TRI (ordered input)
        recovery_strategy="interpolation",
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, args.frames, args.n_joints, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_tps_tri_combination_cuda():
    """Test recommended TPS+TRI combination (paper Table 2: best efficiency)."""
    args = _build_args(
        pruning_strategy="sampler",
        recovery_strategy="interpolation",
        recovery_on_hierarchy=False,
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, args.frames, args.n_joints, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_tpmo_tri_combination_cuda():
    """Test TPMo+TRI combination (with sorted indices for TRI compatibility)."""
    args = _build_args(
        pruning_strategy="motion",
        recovery_strategy="interpolation",
        recovery_on_hierarchy=False,
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x)

    assert y.shape == (2, args.frames, args.n_joints, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this test.")
def test_h2ot_mixste_tpmo_with_external_input_cuda():
    """Test TPMo with externally provided input_2d."""
    args = _build_args(
        pruning_strategy="motion",
        recovery_strategy="attention",
    )
    model = H2OTMixSTE(args).cuda().eval()
    x = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")
    input_2d = torch.randn(2, args.frames, args.n_joints, 2, device="cuda")

    with torch.no_grad():
        y = model(x, input_2d=input_2d)

    assert y.shape == (2, args.frames, args.n_joints, 3)


def test_h2ot_mixste_invalid_pruning_strategy_raises():
    """Test that invalid pruning_strategy raises error."""
    args = _build_args(pruning_strategy="invalid")
    with pytest.raises(ValueError, match="pruning_strategy must be one of"):
        H2OTMixSTE(args)


def test_h2ot_mixste_invalid_recovery_strategy_raises():
    """Test that invalid recovery_strategy raises error."""
    args = _build_args(recovery_strategy="invalid")
    with pytest.raises(ValueError, match="recovery_strategy must be one of"):
        H2OTMixSTE(args)


# CPU tests (faster for CI)

def test_h2ot_mixste_cluster_attention_cpu():
    """Test cluster+attention on CPU (backward compatibility)."""
    args = _build_args(pruning_strategy="cluster", recovery_strategy="attention")
    model = H2OTMixSTE(args).eval()
    x = torch.randn(1, args.frames, args.n_joints, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, args.frames, args.n_joints, 3)


def test_h2ot_mixste_tps_tri_cpu():
    """Test TPS+TRI on CPU (recommended H2OT combination)."""
    args = _build_args(pruning_strategy="sampler", recovery_strategy="interpolation")
    model = H2OTMixSTE(args).eval()
    x = torch.randn(1, args.frames, args.n_joints, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, args.frames, args.n_joints, 3)


def test_h2ot_mixste_learned_attention_cpu():
    """Test learned selector pruning with attention recovery on CPU."""
    args = _build_args(pruning_strategy="learned", recovery_strategy="attention")
    model = H2OTMixSTE(args).eval()
    x = torch.randn(1, args.frames, args.n_joints, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, args.frames, args.n_joints, 3)


def test_h2ot_mixste_tpmo_tri_cpu():
    """Test TPMo+TRI on CPU (with sorted indices)."""
    args = _build_args(pruning_strategy="motion", recovery_strategy="interpolation")
    model = H2OTMixSTE(args).eval()
    x = torch.randn(1, args.frames, args.n_joints, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, args.frames, args.n_joints, 3)
