from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricCellContrastiveLoss(nn.Module):
    """Symmetric InfoNCE loss between `[e-cls]` and `[g-cls]` cell embeddings."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(
        self,
        expression_cls: torch.Tensor,
        gene_cls: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if expression_cls.shape != gene_cls.shape:
            raise ValueError("Contrastive inputs must have matching shapes.")

        expression_norm = F.normalize(expression_cls, dim=-1, eps=1e-6)
        gene_norm = F.normalize(gene_cls, dim=-1, eps=1e-6)

        logits = expression_norm @ gene_norm.transpose(0, 1)
        logits = logits / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_e = F.cross_entropy(logits, labels)
        loss_g = F.cross_entropy(logits.transpose(0, 1), labels)
        loss = 0.5 * (loss_e + loss_g)
        metrics = {
            "contrastive_loss": float(loss.detach().item()),
            "contrastive_temperature": float(self.temperature),
        }
        return loss, metrics
