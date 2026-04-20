import torch

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


def test_backbone_runs_on_cpu_with_padding_and_rotary_mask():
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

    out = model(x, p, key_padding_mask=padding, rotary_mask=rotary_mask)
    assert out.shape == x.shape
    assert torch.all(out[padding] == 0)
