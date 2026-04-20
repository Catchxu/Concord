from __future__ import annotations

import contextlib
import math

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.losses import PretrainingLossManager
from src.models import (
    CellClassificationHead,
    ConcordModel,
    ExpressionReconstructionHead,
    GeneTaskHead,
    MaskedGenePredictionHead,
)

from .checkpointing import CheckpointManager
from .distributed import (
    RuntimeContext,
    autocast_context,
    move_batch_to_device,
    reduce_scalar_dict,
)


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(config.get("name", "adamw")).lower()
    if name != "adamw":
        raise ValueError(f"Unsupported optimizer {config.get('name')}.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 5e-5)),
        betas=tuple(config.get("betas", [0.9, 0.999])),
        eps=float(config.get("eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    total_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    warmup_ratio = float(config.get("warmup_ratio", 0.03))
    min_lr_ratio = float(config.get("min_lr_ratio", 0.0))
    schedule = str(config.get("name", "linear_warmup_decay")).lower()
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / max(warmup_steps, 1)
        progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if schedule == "constant":
            return 1.0
        if schedule in {"linear", "linear_warmup_decay"}:
            return max(min_lr_ratio, 1.0 - progress)
        if schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        raise ValueError(f"Unsupported scheduler {schedule}.")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


class PretrainingSystem(nn.Module):
    """Thin training wrapper so FSDP can wrap a standard forward path."""

    def __init__(
        self,
        concord_model: ConcordModel,
        *,
        expression_head: ExpressionReconstructionHead,
        gene_head: MaskedGenePredictionHead,
        loss_manager: PretrainingLossManager,
    ) -> None:
        super().__init__()
        self.concord_model = concord_model
        self.expression_head = expression_head
        self.gene_head = gene_head
        self.loss_manager = loss_manager

    def forward(self, batch: dict[str, torch.Tensor], *, phase: str) -> dict[str, Any]:
        pretrain_output = self.concord_model.forward_pretrain(batch, phase=phase)
        head_outputs: dict[str, torch.Tensor] = {}

        normalized_phase = phase.lower()
        if normalized_phase in {"expression", "expression_reconstruction", "joint"}:
            head_outputs["expression_reconstruction"] = self.expression_head(
                pretrain_output.expression_mode.sequence_embeddings
            )
        if normalized_phase in {"gene", "gene_prediction", "joint"}:
            head_outputs["gene_prediction"] = self.gene_head(
                pretrain_output.gene_mode.sequence_embeddings
            )

        loss_bundle = self.loss_manager(
            phase=phase,
            pretrain_output=pretrain_output,
            head_outputs=head_outputs,
            batch=batch,
        )
        return {
            "loss": loss_bundle.loss,
            "metrics": loss_bundle.metrics,
            "pretrain_output": pretrain_output,
            "head_outputs": head_outputs,
        }


class CellFineTuneSystem(nn.Module):
    """Cell-level fine-tuning wrapper using Concord expression mode."""

    def __init__(
        self,
        concord_model: ConcordModel,
        head: CellClassificationHead,
        *,
        pool: str = "cls",
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.concord_model = concord_model
        self.head = head
        self.pool = pool
        self.freeze_backbone = freeze_backbone

    def _pool_embeddings(self, sequence_output: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pool == "cls":
            return sequence_output[:, 0]
        if self.pool == "mean":
            valid_mask = torch.ones(
                sequence_output.shape[:2],
                device=sequence_output.device,
                dtype=torch.bool,
            )
            if padding_mask is not None:
                valid_mask = ~padding_mask
            valid_mask[:, 0] = True
            weights = valid_mask.unsqueeze(-1).to(dtype=sequence_output.dtype)
            return (sequence_output * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        raise ValueError(f"Unsupported pooling mode {self.pool}.")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        context_manager = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with context_manager:
            mode_output = self.concord_model.forward_expression_mode(batch)
        pooled = self._pool_embeddings(
            mode_output.sequence_embeddings,
            mode_output.key_padding_mask,
        )
        logits = self.head(pooled)
        labels = batch["cell_labels"]
        loss = F.cross_entropy(logits, labels)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()
        return {
            "loss": loss,
            "metrics": {
                "supervised_loss": float(loss.detach().item()),
                "accuracy": float(accuracy.detach().item()),
            },
            "logits": logits,
        }


class GeneFineTuneSystem(nn.Module):
    """Token-level gene task wrapper using Concord gene mode."""

    def __init__(
        self,
        concord_model: ConcordModel,
        head: GeneTaskHead,
        *,
        ignore_index: int = -100,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.concord_model = concord_model
        self.head = head
        self.ignore_index = ignore_index
        self.freeze_backbone = freeze_backbone

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, Any]:
        context_manager = torch.no_grad() if self.freeze_backbone else contextlib.nullcontext()
        with context_manager:
            mode_output = self.concord_model.forward_gene_mode(batch)
        logits = self.head(mode_output.sequence_embeddings)
        labels = batch["gene_labels"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        return {
            "loss": loss,
            "metrics": {"supervised_loss": float(loss.detach().item())},
            "logits": logits,
        }


class Trainer:
    """Shared training loop with gradient accumulation and checkpoint support."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        runtime: RuntimeContext,
        checkpoint_manager: CheckpointManager | None,
        logger: Any,
        precision: str = "fp32",
        grad_accum_steps: int = 1,
        grad_clip_norm: float | None = None,
        log_every_steps: int = 10,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.runtime = runtime
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.precision = precision
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip_norm = grad_clip_norm
        self.log_every_steps = log_every_steps

    def _grad_sync_context(self, should_sync: bool):
        if should_sync or not isinstance(self.model, FSDP):
            return contextlib.nullcontext()
        return self.model.no_sync()

    def _clip_grad_norm(self) -> float | None:
        if self.grad_clip_norm is None:
            return None
        if isinstance(self.model, FSDP):
            grad_norm = self.model.clip_grad_norm_(self.grad_clip_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        if torch.is_tensor(grad_norm):
            return float(grad_norm.detach().item())
        return float(grad_norm)

    def train_epochs(
        self,
        train_loader: torch.utils.data.DataLoader,
        *,
        epochs: int,
        trainer_state: dict[str, Any] | None = None,
        save_every_epochs: int = 1,
        resolved_config: dict[str, Any] | None = None,
        forward_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if trainer_state is None:
            trainer_state = {"epoch": 0, "global_step": 0}
        if resolved_config is None:
            resolved_config = {}
        if forward_kwargs is None:
            forward_kwargs = {}

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(int(trainer_state.get("epoch", 0)), epochs):
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

            epoch_metrics: dict[str, float] = {}
            for step_idx, batch in enumerate(train_loader):
                batch = move_batch_to_device(batch, self.runtime.device)
                should_sync = ((step_idx + 1) % self.grad_accum_steps) == 0
                with self._grad_sync_context(should_sync):
                    with autocast_context(self.runtime, self.precision):
                        output = self.model(batch, **forward_kwargs)
                        loss = output["loss"] / self.grad_accum_steps
                    loss.backward()

                if should_sync:
                    grad_norm = self._clip_grad_norm()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    trainer_state["global_step"] = int(trainer_state.get("global_step", 0)) + 1

                    metrics = dict(output["metrics"])
                    if grad_norm is not None:
                        metrics["grad_norm"] = grad_norm
                    epoch_metrics = metrics

                    if trainer_state["global_step"] % self.log_every_steps == 0:
                        reduced = reduce_scalar_dict(metrics, self.runtime)
                        if self.runtime.is_main:
                            metric_str = ", ".join(f"{key}={value:.4f}" for key, value in sorted(reduced.items()))
                            self.logger.info(
                                "epoch=%s step=%s %s",
                                epoch + 1,
                                trainer_state["global_step"],
                                metric_str,
                            )

            trainer_state["epoch"] = epoch + 1
            if (
                self.checkpoint_manager is not None
                and save_every_epochs > 0
                and (epoch + 1) % save_every_epochs == 0
            ):
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    trainer_state=trainer_state,
                    resolved_config=resolved_config,
                )
            if epoch_metrics:
                reduced = reduce_scalar_dict(epoch_metrics, self.runtime)
                if self.runtime.is_main:
                    metric_str = ", ".join(f"{key}={value:.4f}" for key, value in sorted(reduced.items()))
                    self.logger.info("epoch=%s complete %s", epoch + 1, metric_str)

        return trainer_state
