import torch
import pytest

from src.models import FlashCRA, TransformerBackbone


def test_flashcra_zeroes_cls_phase_when_rotary_mask_disabled():
    attn = FlashCRA(embed_dim=16, num_heads=4, attn_dropout=0.0)
    conditioning = torch.randn(2, 5, 16)
    rotary_mask = torch.tensor(
        [
            [False, True, True, False, False],
            [False, True, False, False, False],
        ],
        dtype=torch.bool,
    )
    phases = attn._phases_from_p(conditioning, rotary_mask=rotary_mask)
    assert torch.allclose(phases[:, 0], torch.zeros_like(phases[:, 0]))


def test_flashcra_fa4_failure_falls_back_to_fa2(monkeypatch):
    attn = FlashCRA(embed_dim=16, num_heads=4, attn_dropout=0.0)
    attn._fa4_available = True
    attn._fa4_checked = True
    attn._fa4_supported = True
    attn._fa4_runtime_disabled = False

    expected = torch.randn(2, 5, 4, 4)

    def _raise(*args, **kwargs):
        raise RuntimeError("fa4 fail")

    monkeypatch.setattr(attn, "_run_fa4_dense", _raise)
    monkeypatch.setattr(attn, "_run_fa2_dense", lambda qkv: expected)

    out = attn._run_dense_attention(torch.randn(2, 5, 3, 4, 4))
    assert out is expected
    assert attn._fa4_runtime_disabled


def test_backbone_requires_cuda_for_attention():
    model = TransformerBackbone(
        embed_dim=32,
        num_layers=2,
        num_heads=4,
        mlp_hidden_dim=64,
        attn_dropout=0.0,
        mlp_dropout=0.0,
    )
    x = torch.randn(2, 6, 32)
    p = torch.randn(2, 6, 32)
    padding = torch.tensor(
        [
            [False, False, False, False, True, True],
            [False, False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    rotary_mask = ~padding
    rotary_mask[:, 0] = False

    with pytest.raises(RuntimeError, match="requires CUDA"):
        model(x, p, key_padding_mask=padding, rotary_mask=rotary_mask)
