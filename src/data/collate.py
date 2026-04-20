from __future__ import annotations

from typing import Any

import torch

from .masking import sample_token_mask


class ConcordCollator:
    """Collate raw single-cell rows into Concord-ready batches."""

    def __init__(
        self,
        gene_tokenizer: Any,
        expression_tokenizer: Any,
        *,
        max_tokens: int,
        expression_mask_ratio: float = 0.15,
        gene_mask_ratio: float = 0.15,
        seed: int = 0,
    ) -> None:
        if max_tokens <= 1:
            raise ValueError("max_tokens must be at least 2 to accommodate CLS.")
        self.gene_tokenizer = gene_tokenizer
        self.expression_tokenizer = expression_tokenizer
        self.max_tokens = max_tokens
        self.sequence_length = max_tokens - 1
        self.expression_mask_ratio = expression_mask_ratio
        self.gene_mask_ratio = gene_mask_ratio
        self.generator = torch.Generator().manual_seed(seed)

    def _select_tokens(
        self,
        expression: torch.Tensor,
        gene_token_ids: torch.Tensor,
        gene_labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        nonzero_indices = torch.nonzero(expression > 0, as_tuple=False).flatten()
        if nonzero_indices.numel() == 0:
            return (
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.float32),
                None if gene_labels is None else torch.empty(0, dtype=torch.long),
            )

        nonzero_values = expression[nonzero_indices]
        order = torch.argsort(nonzero_values, descending=True)
        selected = nonzero_indices[order[: self.sequence_length]]
        selected_labels = None if gene_labels is None else gene_labels[selected]
        return gene_token_ids[selected], expression[selected], selected_labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        if batch_size == 0:
            raise ValueError("batch must contain at least one sample.")

        gene_ids = torch.full(
            (batch_size, self.max_tokens),
            fill_value=self.gene_tokenizer.pad_token_id,
            dtype=torch.long,
        )
        expression_values = torch.zeros((batch_size, self.max_tokens), dtype=torch.float32)
        key_padding_mask = torch.ones((batch_size, self.max_tokens), dtype=torch.bool)
        key_padding_mask[:, 0] = False
        rotary_mask = torch.zeros((batch_size, self.max_tokens), dtype=torch.bool)

        sample_ids: list[str] = []
        cell_labels: list[torch.Tensor] = []
        gene_labels = torch.full((batch_size, self.max_tokens), fill_value=-100, dtype=torch.long)

        for batch_idx, sample in enumerate(batch):
            selected_gene_ids, selected_expression, selected_gene_labels = self._select_tokens(
                sample["expression"],
                sample["gene_token_ids"],
                sample.get("gene_labels"),
            )
            num_selected = selected_gene_ids.numel()

            gene_ids[batch_idx, 0] = self.gene_tokenizer.cls_token_id
            if num_selected > 0:
                gene_ids[batch_idx, 1 : 1 + num_selected] = selected_gene_ids
                expression_values[batch_idx, 1 : 1 + num_selected] = selected_expression
                key_padding_mask[batch_idx, 1 : 1 + num_selected] = False
                rotary_mask[batch_idx, 1 : 1 + num_selected] = True
                if selected_gene_labels is not None:
                    gene_labels[batch_idx, 1 : 1 + num_selected] = selected_gene_labels

            sample_ids.append(str(sample["sample_id"]))
            if "cell_label" in sample:
                cell_labels.append(torch.as_tensor(sample["cell_label"], dtype=torch.long))

        expression_bin_ids = self.expression_tokenizer.build_input_ids(
            expression_values=expression_values,
            key_padding_mask=key_padding_mask,
        )
        expression_mask = sample_token_mask(
            rotary_mask,
            self.expression_mask_ratio,
            generator=self.generator,
        )
        gene_mask = sample_token_mask(
            rotary_mask,
            self.gene_mask_ratio,
            generator=self.generator,
        )
        gene_targets = torch.where(
            gene_mask,
            gene_ids,
            torch.full_like(gene_ids, fill_value=-100),
        )

        collated = {
            "gene_ids": gene_ids,
            "expression_values": expression_values,
            "expression_bin_ids": expression_bin_ids,
            "key_padding_mask": key_padding_mask,
            "rotary_mask": rotary_mask,
            "expression_mask": expression_mask,
            "gene_mask": gene_mask,
            "expression_targets": expression_values.clone(),
            "gene_targets": gene_targets,
            "sample_ids": sample_ids,
        }
        if cell_labels:
            collated["cell_labels"] = torch.stack(cell_labels, dim=0)
        if any("gene_labels" in sample for sample in batch):
            collated["gene_labels"] = gene_labels
        return collated
