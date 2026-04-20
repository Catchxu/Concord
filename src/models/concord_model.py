from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import TransformerBackbone
from .tokenizers import ExpressionTokenizer, GeneTokenizer


@dataclass
class ConcordModeOutput:
    sequence_embeddings: torch.Tensor
    cls_embedding: torch.Tensor
    token_embeddings: torch.Tensor
    key_padding_mask: torch.Tensor | None
    rotary_mask: torch.Tensor | None
    input_ids: torch.Tensor


@dataclass
class ConcordPretrainOutput:
    phase: str
    expression_mode: ConcordModeOutput | None
    gene_mode: ConcordModeOutput | None
    expression_mask: torch.Tensor | None
    gene_mask: torch.Tensor | None
    expression_targets: torch.Tensor | None
    gene_targets: torch.Tensor | None


class ConcordModel(nn.Module):
    """Dual-token Concord encoder with one shared Transformer backbone."""

    def __init__(
        self,
        gene_tokenizer: GeneTokenizer,
        expression_tokenizer: ExpressionTokenizer,
        backbone: TransformerBackbone,
        *,
        normalize_condition_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.gene_tokenizer = gene_tokenizer
        self.expression_tokenizer = expression_tokenizer
        self.backbone = backbone
        self.normalize_condition_tokens = normalize_condition_tokens

        if gene_tokenizer.embed_dim != expression_tokenizer.embed_dim:
            raise ValueError("Gene and expression tokenizers must share the same embed_dim.")
        if backbone.embed_dim != gene_tokenizer.embed_dim:
            raise ValueError("Backbone embed_dim must match tokenizer embed_dim.")

    def _coerce_batch(
        self,
        batch_or_tensors: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if batch_or_tensors is None:
            batch = {}
        elif isinstance(batch_or_tensors, dict):
            batch = dict(batch_or_tensors)
        else:
            raise TypeError("ConcordModel expects a batch dict or keyword tensor arguments.")
        batch.update(kwargs)
        required = {"gene_ids", "expression_values"}
        missing = required - set(batch.keys())
        if missing:
            raise KeyError(f"Missing required Concord batch keys: {sorted(missing)}")
        return batch

    def _prepare_condition(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_condition_tokens:
            return hidden_states
        return F.normalize(hidden_states, dim=-1, eps=1e-6)

    def forward_expression_mode(
        self,
        batch_or_tensors: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConcordModeOutput:
        batch = self._coerce_batch(batch_or_tensors, **kwargs)
        key_padding_mask = batch.get("key_padding_mask")
        rotary_mask = batch.get("rotary_mask")

        gene_tokens = self.gene_tokenizer(batch["gene_ids"])
        expression_tokens = self.expression_tokenizer(
            batch["expression_values"],
            expression_bin_ids=batch.get("expression_bin_ids"),
            key_padding_mask=key_padding_mask,
            expression_mask=batch.get("expression_mask"),
        )
        contextual = self.backbone(
            x=expression_tokens.embeddings,
            p=self._prepare_condition(gene_tokens.embeddings),
            key_padding_mask=key_padding_mask,
            rotary_mask=rotary_mask,
        )
        return ConcordModeOutput(
            sequence_embeddings=contextual,
            cls_embedding=contextual[:, 0],
            token_embeddings=contextual[:, 1:],
            key_padding_mask=key_padding_mask,
            rotary_mask=rotary_mask,
            input_ids=expression_tokens.input_ids,
        )

    def forward_gene_mode(
        self,
        batch_or_tensors: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ConcordModeOutput:
        batch = self._coerce_batch(batch_or_tensors, **kwargs)
        key_padding_mask = batch.get("key_padding_mask")
        rotary_mask = batch.get("rotary_mask")

        gene_tokens = self.gene_tokenizer(
            batch["gene_ids"],
            gene_mask=batch.get("gene_mask"),
        )
        expression_tokens = self.expression_tokenizer(
            batch["expression_values"],
            expression_bin_ids=batch.get("expression_bin_ids"),
            key_padding_mask=key_padding_mask,
        )
        contextual = self.backbone(
            x=gene_tokens.embeddings,
            p=self._prepare_condition(expression_tokens.embeddings),
            key_padding_mask=key_padding_mask,
            rotary_mask=rotary_mask,
        )
        return ConcordModeOutput(
            sequence_embeddings=contextual,
            cls_embedding=contextual[:, 0],
            token_embeddings=contextual[:, 1:],
            key_padding_mask=key_padding_mask,
            rotary_mask=rotary_mask,
            input_ids=gene_tokens.input_ids,
        )

    def forward_pretrain(
        self,
        batch: dict[str, Any],
        *,
        phase: str,
    ) -> ConcordPretrainOutput:
        normalized_phase = phase.lower()
        if normalized_phase in {"expression", "expression_reconstruction"}:
            expression_mode = self.forward_expression_mode(batch)
            gene_mode = None
        elif normalized_phase in {"gene", "gene_prediction"}:
            expression_mode = None
            gene_mode = self.forward_gene_mode(batch)
        elif normalized_phase in {"contrastive", "cell_contrastive"}:
            expression_mode = self.forward_expression_mode(batch)
            gene_mode = self.forward_gene_mode(batch)
        else:
            raise ValueError(f"Unsupported pretraining phase: {phase}")

        return ConcordPretrainOutput(
            phase=normalized_phase,
            expression_mode=expression_mode,
            gene_mode=gene_mode,
            expression_mask=batch.get("expression_mask"),
            gene_mask=batch.get("gene_mask"),
            expression_targets=batch.get("expression_targets"),
            gene_targets=batch.get("gene_targets"),
        )
