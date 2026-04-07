import torch
import torch.nn as nn

from flash_attn import flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.layers.rotary import apply_rotary_emb_qkv_


class RandomProjection(nn.Module):
    """
    Fixed orthogonal random projection inspired by FAVOR+ / Performer.

    This module projects `(B, L, D)` embeddings to `(B, L, M)` using a
    non-trainable orthogonal random matrix. By default, `M = D / 2`.

    Args:
        embed_dim: input embedding dimension `D`
        projected_dim: output dimension `M`, defaults to `D // 2`
        preserve_variance: scale by `sqrt(D / M)` to keep feature magnitudes stable
    """

    def __init__(
        self,
        embed_dim: int,
        projected_dim: int | None = None,
        preserve_variance: bool = True,
    ):
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
            raise ValueError(
                f"projected_dim must be positive, got {projected_dim}."
            )
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

        # Fix the sign ambiguity in QR so resampling behaves consistently.
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
    FlashAttention-backed collaborative rotary attention.

    Shape convention:
    - `L`: full sequence length

    Args:
        embed_dim: model dimension
        num_heads: number of attention heads
        attn_dropout: attention dropout probability
        qkv_bias: bias for fused qkv projection
        out_bias: bias for output projection
        rotary_interleaved: False = GPT-NeoX style, True = GPT-J style
        phase_scale: scales the projected phase before cos / sin
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
    ):
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

    def _phases_from_p(self, p: torch.Tensor) -> torch.Tensor:
        phases = self.position_projector(p)
        phases = phases.reshape(
            p.shape[0],
            p.shape[1],
            self.num_heads,
            self.head_dim // 2,
        )
        if self.phase_scale != 1.0:
            phases = phases * self.phase_scale
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
        # flash_attn's rotary kernels do not propagate gradients into cos / sin,
        # so we only use the fused path when the projected phase is gradient-free.
        if not qkv.is_cuda or phases.requires_grad:
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

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            p: (B, L, D), collaborative positional embedding
            key_padding_mask: (B, L) bool, where True means padded position

        Returns:
            out: (B, L, D)
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, L, D), got {tuple(x.shape)}")

        B, L, D = x.shape
        if D != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {D}."
            )
        if p.shape != (B, L, self.embed_dim):
            raise ValueError(
                f"p must have shape {(B, L, self.embed_dim)}, got {tuple(p.shape)}"
            )

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    f"key_padding_mask must have shape {(B, L)}, got {tuple(key_padding_mask.shape)}"
                )
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(torch.bool)

        dropout_p = self.attn_dropout if self.training else 0.0

        # (B, L, 3D) -> (B, L, 3, H, Dh)
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        phases = self._phases_from_p(p)
        qkv = self._apply_collaborative_rotary(qkv, phases)

        # ------------------------------------------------------------
        # Path 1: no padding -> packed QKV FlashAttention
        # ------------------------------------------------------------
        if key_padding_mask is None:
            out = flash_attn_qkvpacked_func(
                qkv,
                dropout_p=dropout_p,
                softmax_scale=self.softmax_scale,
            )
            out = out.reshape(B, L, self.embed_dim)
            out = self.out_proj(out)
            return out

        # ------------------------------------------------------------
        # Path 2: padding present -> unpad -> varlen packed QKV -> repad
        # key_padding_mask: True = padded
        # attention mask expected by unpad_input: True = valid
        # ------------------------------------------------------------
        attention_mask = ~key_padding_mask  # (B, L), True = valid

        # qkv_unpad: (nnz, 3, H, Dh)
        # indices: positions of valid tokens in flattened (B, L)
        # cu_seqlens: cumulative lengths, shape (B + 1,)
        # max_seqlen: int
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
            out = x.new_zeros(B, L, self.embed_dim)
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

        # out_unpad: (nnz, H, Dh) -> repad to (B, L, H, Dh)
        out = pad_input(out_unpad, indices, B, L)

        out = out.reshape(B, L, self.embed_dim)
        out = self.out_proj(out)
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out




if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the attention smoke test.")

    device = torch.device("cuda")
    torch.manual_seed(42)

    model = FlashCRA(
        embed_dim=64,
        num_heads=8,
        attn_dropout=0.0,
    ).to(device)
    model = model.half()
    model.eval()

    x = torch.randn(2, 16, 64, device=device, dtype=torch.float16)
    p = torch.randn(2, 16, 64, device=device, dtype=torch.float16)
    key_padding_mask = torch.zeros(2, 16, device=device, dtype=torch.bool)
    key_padding_mask[1, 12:] = True

    with torch.no_grad():
        out_no_padding = model(x, p, key_padding_mask=None)
        out_with_padding = model(x, p, key_padding_mask=key_padding_mask)
        projected = model.position_projector(p)

    print("device:", out_no_padding.device)
    print("dtype:", out_no_padding.dtype)
    print("no_padding_shape:", tuple(out_no_padding.shape))
    print("with_padding_shape:", tuple(out_with_padding.shape))
    print("masked_tail_norm:", out_with_padding[1, 12:].abs().sum().item())
    print("projected_shape:", tuple(projected.shape))
