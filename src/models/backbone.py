import torch
import torch.nn as nn
from time import perf_counter

from .attention import FlashCRA


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.up_proj(x)
        out = gate * value
        out = self.down_proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with FlashAttention and SwiGLU MLP.

    Shape convention:
    - `L`: full sequence length
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        norm_eps: float = 1e-6,
        rotary_interleaved: bool = False,
        phase_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(embed_dim, eps=norm_eps)
        self.attn = FlashCRA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            rotary_interleaved=rotary_interleaved,
            phase_scale=phase_scale,
        )
        self.mlp_norm = nn.RMSNorm(embed_dim, eps=norm_eps)
        self.mlp = SwiGLUMLP(
            embed_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            bias=mlp_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn.forward(
            self.attn_norm(x),
            p,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.mlp(self.mlp_norm(x))

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x


class TransformerBackbone(nn.Module):
    """
    Transformer backbone built from FlashAttention blocks.

    Shape convention:
    - `L`: full sequence length
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        attn_dropout: float = 0.1,
        mlp_dropout: float = 0.1,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        norm_eps: float = 1e-6,
        rotary_interleaved: bool = False,
        phase_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}.")

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim or (4 * embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    attn_dropout=attn_dropout,
                    mlp_dropout=mlp_dropout,
                    qkv_bias=qkv_bias,
                    out_bias=out_bias,
                    mlp_bias=mlp_bias,
                    norm_eps=norm_eps,
                    rotary_interleaved=rotary_interleaved,
                    phase_scale=phase_scale
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`x` must have shape (B, L, D), got {tuple(x.shape)}.")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {x.shape[-1]}."
            )

        for layer in self.layers:
            x = layer(x, p, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x


def _make_key_padding_mask(
    batch_size: int,
    seqlen: int,
    device: torch.device,
    min_valid_tokens: int = 1,
) -> torch.Tensor:
    if seqlen <= 0:
        raise ValueError(f"`seqlen` must be positive, got {seqlen}.")

    valid_lengths = torch.randint(
        low=min_valid_tokens,
        high=seqlen + 1,
        size=(batch_size,),
        device=device,
    )
    positions = torch.arange(seqlen, device=device).unsqueeze(0)
    return positions >= valid_lengths.unsqueeze(1)


@torch.no_grad()
def _benchmark_backbone_runtime(
    model: TransformerBackbone,
    batch_size: int,
    seqlen: int,
    device: torch.device,
    dtype: torch.dtype,
    warmup_steps: int = 10,
    benchmark_steps: int = 10,
) -> tuple[float, float]:
    x = torch.randn(batch_size, seqlen, model.embed_dim, device=device, dtype=dtype)
    p = torch.randn(batch_size, seqlen, model.embed_dim, device=device, dtype=dtype)
    key_padding_mask = _make_key_padding_mask(
        batch_size=batch_size,
        seqlen=seqlen,
        device=device,
    )

    for _ in range(warmup_steps):
        model(x, p, key_padding_mask=key_padding_mask)

    torch.cuda.synchronize(device)
    start = perf_counter()
    for _ in range(benchmark_steps):
        model(x, p, key_padding_mask=key_padding_mask)
    torch.cuda.synchronize(device)
    elapsed_s = perf_counter() - start

    avg_s = elapsed_s / benchmark_steps
    tokens_per_s = (batch_size * seqlen * benchmark_steps) / elapsed_s
    return avg_s, tokens_per_s


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the backbone runtime benchmark.")

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(42)

    model = TransformerBackbone(
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_hidden_dim=3072,
        attn_dropout=0.1,
        mlp_dropout=0.1,
    ).to(device)
    model = model.to(dtype=dtype)
    model.eval()

    num_parameters = sum(param.numel() for param in model.parameters())
    num_trainable_parameters = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    default_cells = 32
    default_genes = 1024
    cell_sweep = [16, 32, 64, 128, 256]
    gene_sweep = [512, 1024, 2048, 4096, 8192]

    print(f"parameters={num_parameters}")
    print(f"trainable_parameters={num_trainable_parameters}")
    print(f"default_cells={default_cells} default_genes={default_genes}")

    print("runtime_benchmark_s_vary_cells:")
    for cells in cell_sweep:
        avg_s, tokens_per_s = _benchmark_backbone_runtime(
            model=model,
            batch_size=cells,
            seqlen=default_genes,
            device=device,
            dtype=dtype,
        )
        print(
            f"cells={cells:4d} genes={default_genes:5d} avg_s={avg_s:8.4f} tokens_per_s={tokens_per_s:12.1f}"
        )

    print("runtime_benchmark_s_vary_genes:")
    for genes in gene_sweep:
        avg_s, tokens_per_s = _benchmark_backbone_runtime(
            model=model,
            batch_size=default_cells,
            seqlen=genes,
            device=device,
            dtype=dtype,
        )
        print(
            f"cells={default_cells:4d} genes={genes:5d} avg_s={avg_s:8.4f} tokens_per_s={tokens_per_s:12.1f}"
        )
