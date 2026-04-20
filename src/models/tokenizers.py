from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.data.vocab import GeneVocab


@dataclass
class TokenizerOutput:
    input_ids: torch.LongTensor
    embeddings: torch.Tensor


class GeneTokenizer(nn.Module):
    """Gene tokenizer with optional Gene2Vec initialization."""

    def __init__(
        self,
        gene_vocab: GeneVocab,
        embed_dim: int,
        *,
        initializer_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.gene_vocab = gene_vocab
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(gene_vocab.vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initializer_scale)
        self.embedding.weight.data[gene_vocab.pad_token_id].zero_()

        if gene_vocab.gene2vec is not None:
            self.initialize_from_gene2vec(gene_vocab.gene2vec)

    @property
    def pad_token_id(self) -> int:
        return self.gene_vocab.pad_token_id

    @property
    def cls_token_id(self) -> int:
        return self.gene_vocab.cls_token_id

    @property
    def mask_token_id(self) -> int:
        return self.gene_vocab.mask_token_id

    def initialize_from_gene2vec(
        self,
        gene2vec: dict[str, list[float]] | torch.Tensor,
    ) -> None:
        """Initialize overlapping genes from Gene2Vec vectors when provided."""

        if torch.is_tensor(gene2vec):
            if gene2vec.shape != (self.gene_vocab.num_genes, self.embed_dim):
                raise ValueError(
                    "Tensor Gene2Vec initialization must have shape "
                    f"({self.gene_vocab.num_genes}, {self.embed_dim}), got {tuple(gene2vec.shape)}."
                )
            self.embedding.weight.data[self.gene_vocab.gene_offset :] = gene2vec.to(
                device=self.embedding.weight.device,
                dtype=self.embedding.weight.dtype,
            )
            return

        matched = 0
        for gene_name, vector in gene2vec.items():
            if gene_name not in self.gene_vocab.gene_names:
                continue
            tensor = torch.as_tensor(vector, dtype=self.embedding.weight.dtype)
            if tensor.numel() != self.embed_dim:
                continue
            token_id = self.gene_vocab.token_id_from_gene_name(gene_name)
            self.embedding.weight.data[token_id] = tensor
            matched += 1
        if matched == 0:
            raise ValueError("No overlapping Gene2Vec entries matched the gene vocabulary.")

    def forward(
        self,
        gene_ids: torch.LongTensor,
        *,
        gene_mask: torch.Tensor | None = None,
    ) -> TokenizerOutput:
        input_ids = gene_ids.clone()
        if gene_mask is not None:
            if gene_mask.shape != gene_ids.shape:
                raise ValueError("gene_mask must match gene_ids shape.")
            input_ids = torch.where(
                gene_mask,
                torch.full_like(input_ids, self.mask_token_id),
                input_ids,
            )
        embeddings = self.embedding(input_ids)
        return TokenizerOutput(input_ids=input_ids, embeddings=embeddings)


class ExpressionTokenizer(nn.Module):
    """Discretize non-zero expression values into learnable codebook embeddings."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_bins: int = 64,
        max_log1p_value: float = 10.0,
        initializer_scale: float = 0.02,
    ) -> None:
        super().__init__()
        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}.")

        self.embed_dim = embed_dim
        self.num_bins = num_bins
        self.pad_token_id = 0
        self.cls_token_id = 1
        self.mask_token_id = 2
        self.bin_offset = 3
        self.vocab_size = num_bins + self.bin_offset
        self.max_log1p_value = max_log1p_value

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=initializer_scale)
        self.embedding.weight.data[self.pad_token_id].zero_()
        self.register_buffer(
            "bin_edges",
            torch.linspace(0.0, max_log1p_value, num_bins + 1),
            persistent=False,
        )

    def build_input_ids(
        self,
        expression_values: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.LongTensor:
        if expression_values.shape != key_padding_mask.shape:
            raise ValueError("expression_values and key_padding_mask must have matching shape.")

        input_ids = torch.full_like(expression_values, fill_value=self.pad_token_id, dtype=torch.long)
        input_ids[:, 0] = self.cls_token_id
        valid_mask = ~key_padding_mask
        valid_mask[:, 0] = False
        if valid_mask.any():
            log_values = torch.log1p(expression_values.clamp_min(0.0))
            clipped = log_values.clamp(min=0.0, max=self.max_log1p_value)
            bucketized = torch.bucketize(clipped[valid_mask], self.bin_edges[1:-1], right=False)
            input_ids[valid_mask] = bucketized + self.bin_offset
        return input_ids

    def forward(
        self,
        expression_values: torch.Tensor,
        *,
        expression_bin_ids: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        expression_mask: torch.Tensor | None = None,
    ) -> TokenizerOutput:
        if expression_bin_ids is None:
            if key_padding_mask is None:
                raise ValueError(
                    "key_padding_mask is required when expression_bin_ids are not provided."
                )
            expression_bin_ids = self.build_input_ids(expression_values, key_padding_mask)

        input_ids = expression_bin_ids.clone()
        if expression_mask is not None:
            if expression_mask.shape != input_ids.shape:
                raise ValueError("expression_mask must match expression input shape.")
            input_ids = torch.where(
                expression_mask,
                torch.full_like(input_ids, self.mask_token_id),
                input_ids,
            )
        embeddings = self.embedding(input_ids)
        return TokenizerOutput(input_ids=input_ids, embeddings=embeddings)
