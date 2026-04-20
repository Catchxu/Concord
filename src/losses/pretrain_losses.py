from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import SymmetricCellContrastiveLoss


@dataclass
class LossBundle:
    loss: torch.Tensor
    metrics: dict[str, float]


class ExpressionReconstructionLoss(nn.Module):
    """Masked MSE loss for reconstructing expression values."""

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if predictions.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("Expression predictions, targets, and mask must share shape.")
        if not mask.any():
            zero = predictions.sum() * 0.0
            return zero, {"expression_reconstruction_loss": 0.0}
        loss = F.mse_loss(predictions[mask], targets[mask])
        return loss, {"expression_reconstruction_loss": float(loss.detach().item())}


class MaskedGenePredictionLoss(nn.Module):
    """Cross-entropy loss for masked gene identity prediction."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return loss, {"gene_prediction_loss": float(loss.detach().item())}


class PretrainingLossManager(nn.Module):
    """Composable Concord pretraining losses with phase-aware defaults."""

    def __init__(
        self,
        *,
        expression_weight: float = 1.0,
        gene_weight: float = 1.0,
        contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
        enable_expression: bool = True,
        enable_gene: bool = True,
        enable_contrastive: bool = True,
    ) -> None:
        super().__init__()
        self.expression_weight = expression_weight
        self.gene_weight = gene_weight
        self.contrastive_weight = contrastive_weight

        self.enable_expression = enable_expression
        self.enable_gene = enable_gene
        self.enable_contrastive = enable_contrastive

        self.expression_loss = ExpressionReconstructionLoss()
        self.gene_loss = MaskedGenePredictionLoss()
        self.contrastive_loss = SymmetricCellContrastiveLoss(temperature=contrastive_temperature)

    def forward(
        self,
        *,
        phase: str,
        pretrain_output: Any,
        head_outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> LossBundle:
        normalized_phase = phase.lower()
        metrics: dict[str, float] = {}
        total_loss = torch.zeros((), device=next(iter(head_outputs.values())).device if head_outputs else batch["gene_ids"].device)

        if normalized_phase in {"expression", "expression_reconstruction", "joint"} and self.enable_expression:
            expression_logits = head_outputs["expression_reconstruction"]
            expression_loss, expression_metrics = self.expression_loss(
                expression_logits,
                batch["expression_targets"],
                batch["expression_mask"],
            )
            total_loss = total_loss + (self.expression_weight * expression_loss)
            metrics.update(expression_metrics)

        if normalized_phase in {"gene", "gene_prediction", "joint"} and self.enable_gene:
            gene_logits = head_outputs["gene_prediction"]
            gene_loss, gene_metrics = self.gene_loss(
                gene_logits,
                batch["gene_targets"],
            )
            total_loss = total_loss + (self.gene_weight * gene_loss)
            metrics.update(gene_metrics)

        if normalized_phase in {"contrastive", "cell_contrastive", "joint"} and self.enable_contrastive:
            contrastive_loss, contrastive_metrics = self.contrastive_loss(
                pretrain_output.expression_mode.cls_embedding,
                pretrain_output.gene_mode.cls_embedding,
            )
            total_loss = total_loss + (self.contrastive_weight * contrastive_loss)
            metrics.update(contrastive_metrics)

        metrics["total_loss"] = float(total_loss.detach().item())
        return LossBundle(loss=total_loss, metrics=metrics)
