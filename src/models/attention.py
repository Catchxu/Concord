from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_

    FLASH_ATTN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in CPU-only smoke tests.
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_qkvpacked_func = None
    pad_input = None
    unpad_input = None
    apply_rotary_emb_qkv_ = None
    FLASH_ATTN_AVAILABLE = False


class RandomProjection(nn.Module):
    """
    Fixed orthogonal random projection inspired by FAVOR+ / Performer.

    This module projects `(B, L, D)` embeddings to `(B, L, M)` using a
    non-trainable orthogonal random matrix. By default, `M = D / 2`.
    """

    def __init__(
        self,
        embed_dim: int,
        projected_dim: int | None = None,
        preserve_variance: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if projected_dim is None:
            if embed_dim % 2 != 0:
                raise ValueError(
                    f"embed_dim must be even when projected_dim is omitted, got {embed_dim}."
                )
            projected_dim = embed_dim // 2
        if projected_dim <= 0:
            raise ValueError(f"projected_dim must be positive, got {projected_dim}.")
        if projected_dim > embed_dim:
            raise ValueError(
                f"projected_dim ({projected_dim}) cannot exceed embed_dim ({embed_dim})."
            )

        self.embed_dim = embed_dim
        self.projected_dim = projected_dim
        self.preserve_variance = preserve_variance

        self.register_buffer("projection", self._sample_projection())

    def _sample_projection(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        gaussian = torch.randn(
            self.embed_dim,
            self.projected_dim,
            device=device,
            dtype=dtype,
        )
        q, r = torch.linalg.qr(gaussian, mode="reduced")
        signs = torch.sign(torch.diagonal(r))
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        q = q * signs.unsqueeze(0)
        if self.preserve_variance:
            q = q * (self.embed_dim / self.projected_dim) ** 0.5
        return q.contiguous()

    @torch.no_grad()
    def resample_projection(self) -> None:
        self.projection.copy_(
            self._sample_projection(
                device=self.projection.device,
                dtype=self.projection.dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, L, D), got {tuple(x.shape)}")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {x.shape[-1]}."
            )
        return x @ self.projection


class FlashCRA(nn.Module):
    """
    Collaborative rotary attention with FlashAttention and SDPA fallback.

    Shape convention:
    - `x`: tokens being updated
    - `p`: conditioning tokens projected into collaborative rotary phases
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        qkv_bias: bool = True,
        out_bias: bool = True,
        rotary_interleaved: bool = False,
        phase_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"Collaborative rotary attention requires even head_dim, got {self.head_dim}."
            )

        self.attn_dropout = attn_dropout
        self.softmax_scale = self.head_dim ** -0.5
        self.rotary_interleaved = rotary_interleaved
        self.phase_scale = phase_scale

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=out_bias)
        self.position_projector = RandomProjection(
            embed_dim=embed_dim,
            projected_dim=embed_dim // 2,
        )

    def _phases_from_p(
        self,
        p: torch.Tensor,
        rotary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        phases = self.position_projector(p)
        phases = phases.reshape(
            p.shape[0],
            p.shape[1],
            self.num_heads,
            self.head_dim // 2,
        )
        if self.phase_scale != 1.0:
            phases = phases * self.phase_scale
        if rotary_mask is not None:
            if rotary_mask.shape != (p.shape[0], p.shape[1]):
                raise ValueError(
                    f"rotary_mask must have shape {(p.shape[0], p.shape[1])}, "
                    f"got {tuple(rotary_mask.shape)}"
                )
            phases = phases * rotary_mask.to(dtype=phases.dtype).unsqueeze(-1).unsqueeze(-1)
        return phases

    def _apply_collaborative_rotary_torch(
        self,
        qkv: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        cos = phases.cos().to(dtype=qkv.dtype)
        sin = phases.sin().to(dtype=qkv.dtype)

        q = qkv[:, :, 0]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        if self.rotary_interleaved:
            q_rot = torch.empty_like(q)
            k_rot = torch.empty_like(k)
            q_even = q[..., ::2]
            q_odd = q[..., 1::2]
            k_even = k[..., ::2]
            k_odd = k[..., 1::2]

            q_rot[..., ::2] = q_even * cos - q_odd * sin
            q_rot[..., 1::2] = q_even * sin + q_odd * cos
            k_rot[..., ::2] = k_even * cos - k_odd * sin
            k_rot[..., 1::2] = k_even * sin + k_odd * cos
            return torch.stack((q_rot, k_rot, v), dim=2)

        half_dim = self.head_dim // 2
        q_first, q_second = q[..., :half_dim], q[..., half_dim:]
        k_first, k_second = k[..., :half_dim], k[..., half_dim:]

        q_rot = torch.cat(
            (
                q_first * cos - q_second * sin,
                q_first * sin + q_second * cos,
            ),
            dim=-1,
        )
        k_rot = torch.cat(
            (
                k_first * cos - k_second * sin,
                k_first * sin + k_second * cos,
            ),
            dim=-1,
        )
        return torch.stack((q_rot, k_rot, v), dim=2)

    def _apply_collaborative_rotary(
        self,
        qkv: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not FLASH_ATTN_AVAILABLE
            or apply_rotary_emb_qkv_ is None
            or not qkv.is_cuda
            or phases.requires_grad
        ):
            return self._apply_collaborative_rotary_torch(qkv, phases)

        qkv = qkv.contiguous()
        cos = phases.cos().to(dtype=qkv.dtype)
        sin = phases.sin().to(dtype=qkv.dtype)

        for batch_idx in range(qkv.shape[0]):
            for head_idx in range(self.num_heads):
                apply_rotary_emb_qkv_(
                    qkv[batch_idx : batch_idx + 1, :, :, head_idx : head_idx + 1, :],
                    cos[batch_idx, :, head_idx],
                    sin[batch_idx, :, head_idx],
                    interleaved=self.rotary_interleaved,
                )

        return qkv

    def _forward_sdpa(
        self,
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = qkv[:, :, 0].transpose(1, 2)  # (B, H, L, Dh)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        attn_mask = None
        if key_padding_mask is not None:
            valid_keys = (~key_padding_mask).unsqueeze(1).unsqueeze(2)
            attn_mask = valid_keys.expand(-1, self.num_heads, q.shape[-2], -1)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            scale=self.softmax_scale,
        )
        out = out.transpose(1, 2).reshape(qkv.shape[0], qkv.shape[1], self.embed_dim)
        out = self.out_proj(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out

    def _forward_flash(
        self,
        qkv: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout if self.training else 0.0
        batch_size, seqlen = qkv.shape[0], qkv.shape[1]

        if key_padding_mask is None:
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
            )
            out = out.reshape(batch_size, seqlen, self.embed_dim)
            return self.out_proj(out)

        attention_mask = ~key_padding_mask
        unpad_outputs = unpad_input(qkv, attention_mask)
        if len(unpad_outputs) == 4:
            qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_outputs
        elif len(unpad_outputs) == 5:
            qkv_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_outputs
        else:
            raise ValueError(
                f"Unexpected number of outputs from unpad_input: {len(unpad_outputs)}"
            )

        if qkv_unpad.shape[0] == 0:
            out = qkv.new_zeros(batch_size, seqlen, self.embed_dim)
            out = self.out_proj(out)
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
            return out

        out_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=self.softmax_scale,
        )
        out = pad_input(out_unpad, indices, batch_size, seqlen)
        out = out.reshape(batch_size, seqlen, self.embed_dim)
        out = self.out_proj(out)
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        rotary_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: `(B, L, D)` tokens being updated.
            p: `(B, L, D)` collaborative conditioning tokens.
            key_padding_mask: `(B, L)` bool, where True denotes padding.
            rotary_mask: `(B, L)` bool, where True enables collaborative rotation.
        """

        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, L, D), got {tuple(x.shape)}")

        batch_size, seqlen, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {embed_dim}."
            )
        if p.shape != (batch_size, seqlen, self.embed_dim):
            raise ValueError(
                f"p must have shape {(batch_size, seqlen, self.embed_dim)}, "
                f"got {tuple(p.shape)}"
            )
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, seqlen):
                raise ValueError(
                    "key_padding_mask must have shape "
                    f"{(batch_size, seqlen)}, got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)
        if rotary_mask is not None and rotary_mask.dtype != torch.bool:
            rotary_mask = rotary_mask.to(torch.bool)

        qkv = self.qkv_proj(x).reshape(
            batch_size,
            seqlen,
            3,
            self.num_heads,
            self.head_dim,
        )
        phases = self._phases_from_p(p, rotary_mask=rotary_mask)
        qkv = self._apply_collaborative_rotary(qkv, phases)

        can_use_flash = (
            FLASH_ATTN_AVAILABLE
            and flash_attn_qkvpacked_func is not None
            and x.is_cuda
            and qkv.dtype in {torch.float16, torch.bfloat16}
        )
        if can_use_flash:
            return self._forward_flash(qkv, key_padding_mask=key_padding_mask)
        return self._forward_sdpa(qkv, key_padding_mask=key_padding_mask)
