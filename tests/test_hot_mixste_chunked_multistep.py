from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.hot.mixste.hot_mixste import ChunkCompressMultiStepModel, ChunkedCompressionModel


def _build_args(**overrides):
    base = dict(
        channel=32,
        frames=243,
        n_joints=17,
        token_num=81,
        layer_index=1,
        hierarchical_layer_indices=[1, 2],
        hierarchical_token_nums=[162, 81],
        pruning_strategy="cluster",
        use_chunk_ortho_loss=False,
        lambda_chunk_ortho=0.0,
        decoder_mode="cross_attention",
        chunking_scheme="center81_two_step",
        use_pairwise_flow=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_center81_two_step_stage0_layout():
    chunk_lengths = ChunkCompressMultiStepModel._build_center81_two_step_chunk_lengths(243, 162, 0)

    assert len(chunk_lengths) == 162
    assert sum(chunk_lengths) == 243
    assert chunk_lengths == ([2] * 41 + [1] * 81 + [2] * 40)


def test_center81_two_step_model_stage_configs():
    model = ChunkCompressMultiStepModel(_build_args()).eval()

    stage0 = model.stage_configs[0]
    stage1 = model.stage_configs[1]

    assert stage0["input_frames"] == 243
    assert stage0["token_num"] == 162
    assert stage0["use_uneven_chunking"] is True
    assert stage0["chunk_lengths"] == ([2] * 41 + [1] * 81 + [2] * 40)

    assert stage1["input_frames"] == 162
    assert stage1["token_num"] == 81
    assert stage1["use_uneven_chunking"] is False
    assert stage1["chunk_lengths"] is None


def test_center81_two_step_stage0_preserves_singleton_center_tokens():
    model = ChunkCompressMultiStepModel(_build_args()).eval()
    x = torch.arange(243 * model.channel, dtype=torch.float32).reshape(1, 243, 1, model.channel)

    with torch.no_grad():
        compressed = model._chunk_compress_stage(x, 0)

    assert compressed.shape == (1, 162, 1, model.channel)
    expected = x[:, 82:163]
    actual = compressed[:, 41:122]
    assert torch.equal(actual, expected)


def test_center81_two_step_forward_cpu():
    model = ChunkCompressMultiStepModel(_build_args()).eval()
    x = torch.randn(1, 243, 17, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 243, 17, 3)


def test_single_stage_one_step_interp_cubic_forward_cpu():
    model = ChunkedCompressionModel(
        _build_args(
            token_num=81,
            layer_index=1,
            decoder_mode="one_step_interp_cubic",
            chunking_scheme="even",
        )
    ).eval()
    x = torch.randn(1, 243, 17, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 243, 17, 3)


def test_feature_interp_attn_identifier_ids_match_interpolation_grid():
    identifier_ids = ChunkedCompressionModel._build_interp_identifier_ids(
        token_num=81,
        recover_num=243,
    )
    anchor_indices = torch.nonzero(identifier_ids[0] == 0, as_tuple=False).flatten()
    expected = (
        (torch.arange(81, dtype=torch.float32) * (243 - 1) / (81 - 1))
        .round()
        .to(torch.long)
    )

    assert identifier_ids.shape == (1, 243)
    assert int((identifier_ids == 0).sum().item()) == 81
    assert int((identifier_ids == 1).sum().item()) == 162
    assert torch.equal(anchor_indices, expected)


def test_single_stage_feature_interp_attn_registers_identifier_embedding():
    model = ChunkedCompressionModel(
        _build_args(
            token_num=81,
            layer_index=1,
            decoder_mode="feature_interp_attn",
            chunking_scheme="even",
            channel=8,
        )
    ).eval()

    assert model.feature_interp_attn_identifier_embed.num_embeddings == 2
    assert model.feature_interp_attn_identifier_embed.embedding_dim == 8
    assert model.feature_interp_attn_identifier_ids.shape == (1, 243)
    assert int((model.feature_interp_attn_identifier_ids == 0).sum().item()) == 81


def test_single_stage_ut_insert_anchor_indices_even_chunk_centers():
    anchor_indices = ChunkedCompressionModel._build_ut_anchor_indices(
        recover_num=243,
        token_num=81,
        chunk_size=3,
        anchor_mode="chunk_center",
    )

    expected = torch.arange(81, dtype=torch.long) * 3 + 1
    assert torch.equal(anchor_indices, expected)


def test_single_stage_ut_insert_dense_sequence_uses_ut_off_anchor():
    model = ChunkedCompressionModel(
        _build_args(
            token_num=81,
            layer_index=1,
            decoder_mode="ut_insert_attention",
            chunking_scheme="even",
            channel=8,
        )
    ).eval()
    x = torch.arange(81 * 8, dtype=torch.float32).view(1, 81, 8)

    dense = model._build_ut_dense_sequence(x)
    anchor_indices = model.ut_anchor_indices_tensor

    expected_pt = x + model.decoder_ut_pos_embed[:, anchor_indices, :].to(dtype=x.dtype)
    assert torch.allclose(dense[:, anchor_indices, :], expected_pt)

    non_anchor_mask = torch.ones(model.recover_num, dtype=torch.bool)
    non_anchor_mask[anchor_indices] = False
    expected_ut = (
        model.decoder_ut_token.expand(1, model.recover_num, x.shape[-1])[:, non_anchor_mask, :]
        + model.decoder_ut_pos_embed[:, non_anchor_mask, :].to(dtype=x.dtype)
    )
    assert torch.allclose(dense[:, non_anchor_mask, :], expected_ut)


def test_single_stage_ut_insert_attention_forward_cpu():
    model = ChunkedCompressionModel(
        _build_args(
            token_num=81,
            layer_index=1,
            decoder_mode="ut_insert_attention",
            chunking_scheme="even",
        )
    ).eval()
    x = torch.randn(1, 243, 17, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 243, 17, 3)


def test_single_stage_feature_interp_attn_forward_cpu():
    model = ChunkedCompressionModel(
        _build_args(
            token_num=81,
            layer_index=1,
            decoder_mode="feature_interp_attn",
            chunking_scheme="even",
        )
    ).eval()
    x = torch.randn(1, 243, 17, 2)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 243, 17, 3)


def test_center81_two_step_rejects_incompatible_hierarchy():
    args = _build_args(hierarchical_token_nums=[121, 81])

    with pytest.raises(ValueError, match="stage 0 requires token_num=162"):
        ChunkCompressMultiStepModel(args)


def test_single_stage_chunked_rejects_center81_two_step_scheme():
    args = _build_args()

    with pytest.raises(ValueError, match="Unknown chunking_scheme: center81_two_step"):
        ChunkedCompressionModel(args)


def test_single_stage_ut_insert_rejects_corner_aligned_chunking():
    args = _build_args(
        token_num=81,
        layer_index=1,
        decoder_mode="ut_insert_attention",
        chunking_scheme="corner_aligned",
    )

    with pytest.raises(ValueError, match="supports chunking_scheme='even' only"):
        ChunkedCompressionModel(args)
