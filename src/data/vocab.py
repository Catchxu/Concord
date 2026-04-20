from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


@dataclass
class GeneVocab:
    """Vocabulary metadata for Concord gene tokens."""

    gene_names: tuple[str, ...]
    gene_label_ids: dict[str, int] | None = None
    pad_token: str = "[pad]"
    cls_token: str = "[g-cls]"
    mask_token: str = "[g-mask]"
    gene2vec: dict[str, list[float]] | None = None
    _gene_to_index: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.gene_names:
            raise ValueError("GeneVocab requires at least one gene name.")
        self._gene_to_index = {name: idx for idx, name in enumerate(self.gene_names)}
        if len(self._gene_to_index) != len(self.gene_names):
            raise ValueError("GeneVocab gene_names must be unique.")

    @classmethod
    def from_gene_names(
        cls,
        gene_names: Iterable[str],
        gene_label_ids: dict[str, int] | None = None,
        gene2vec: dict[str, list[float]] | None = None,
    ) -> "GeneVocab":
        normalized = tuple(str(name) for name in gene_names)
        return cls(
            gene_names=normalized,
            gene_label_ids=gene_label_ids,
            gene2vec=gene2vec,
        )

    @property
    def gene_offset(self) -> int:
        return 3

    @property
    def pad_token_id(self) -> int:
        return 0

    @property
    def cls_token_id(self) -> int:
        return 1

    @property
    def mask_token_id(self) -> int:
        return 2

    @property
    def num_genes(self) -> int:
        return len(self.gene_names)

    @property
    def vocab_size(self) -> int:
        return self.num_genes + self.gene_offset

    @property
    def num_gene_labels(self) -> int:
        if self.gene_label_ids is None:
            return self.vocab_size
        return max(self.gene_label_ids.values()) + 1

    def gene_index(self, gene_name: str) -> int:
        return self._gene_to_index[gene_name]

    def token_id_from_gene_name(self, gene_name: str) -> int:
        return self.gene_index(gene_name) + self.gene_offset

    def encode_gene_names(self, gene_names: Iterable[str]) -> torch.LongTensor:
        return torch.tensor(
            [self.token_id_from_gene_name(name) for name in gene_names],
            dtype=torch.long,
        )

    def encode_gene_indices(self, indices: torch.Tensor | list[int]) -> torch.LongTensor:
        tensor = torch.as_tensor(indices, dtype=torch.long)
        return tensor + self.gene_offset

    def decode_token_ids(self, token_ids: torch.Tensor | list[int]) -> list[str]:
        decoded: list[str] = []
        for token_id in torch.as_tensor(token_ids, dtype=torch.long).tolist():
            if token_id < self.gene_offset:
                decoded.append({0: self.pad_token, 1: self.cls_token, 2: self.mask_token}[token_id])
            else:
                decoded.append(self.gene_names[token_id - self.gene_offset])
        return decoded

    def gene_prediction_target(self, gene_token_ids: torch.Tensor) -> torch.Tensor:
        if self.gene_label_ids is None:
            return gene_token_ids.clone()
        targets = torch.full_like(gene_token_ids, fill_value=-100)
        for gene_name, label_id in self.gene_label_ids.items():
            token_id = self.token_id_from_gene_name(gene_name)
            targets = torch.where(
                gene_token_ids == token_id,
                torch.full_like(targets, fill_value=label_id),
                targets,
            )
        return targets
