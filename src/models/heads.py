from __future__ import annotations

import torch
import torch.nn as nn


class ExpressionReconstructionHead(nn.Module):
    """Decode contextualized expression embeddings back to scalar values."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states).squeeze(-1)


class MaskedGenePredictionHead(nn.Module):
    """Predict masked gene identities from contextual gene embeddings."""

    def __init__(self, embed_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class CellClassificationHead(nn.Module):
    """Cell-level classifier operating on pooled Concord cell embeddings."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, num_classes)

    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(cell_embeddings))


class GeneTaskHead(nn.Module):
    """Generic token-level prediction head for downstream gene tasks."""

    def __init__(self, embed_dim: int, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, gene_embeddings: torch.Tensor) -> torch.Tensor:
        return self.proj(gene_embeddings)
