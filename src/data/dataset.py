from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    import anndata as ad
except ImportError:  # pragma: no cover - optional dependency.
    ad = None

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover - optional dependency.
    sp = None

from torch.utils.data import Dataset

from .vocab import GeneVocab


@dataclass
class SyntheticDatasetConfig:
    num_cells: int = 16
    num_genes: int = 64
    num_cell_types: int = 3
    seed: int = 0
    density: float = 0.2


class SingleCellMatrixDataset(Dataset):
    """Dataset wrapper over dense, sparse, or AnnData single-cell matrices."""

    def __init__(
        self,
        matrix: Any,
        gene_vocab: GeneVocab,
        *,
        gene_names: list[str] | None = None,
        cell_labels: torch.Tensor | np.ndarray | list[int] | None = None,
        gene_targets_matrix: Any | None = None,
        sample_ids: list[str] | None = None,
        cell_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        self.matrix = matrix
        self.gene_vocab = gene_vocab
        self.cell_labels = None if cell_labels is None else torch.as_tensor(cell_labels, dtype=torch.long)
        self.gene_targets_matrix = gene_targets_matrix
        self.sample_ids = sample_ids
        self.cell_metadata = cell_metadata

        self._matrix_backend = self._detect_backend(matrix)
        self._num_cells, self._num_genes = self._infer_shape(matrix)

        resolved_gene_names = gene_names
        if self._matrix_backend == "anndata":
            resolved_gene_names = list(matrix.var_names if gene_names is None else gene_names)
        if resolved_gene_names is None:
            resolved_gene_names = list(gene_vocab.gene_names)
        if len(resolved_gene_names) != self._num_genes:
            raise ValueError(
                f"Expected {self._num_genes} gene names, got {len(resolved_gene_names)}."
            )

        self.feature_gene_names = list(map(str, resolved_gene_names))
        self.feature_gene_token_ids = torch.tensor(
            [gene_vocab.token_id_from_gene_name(name) for name in self.feature_gene_names],
            dtype=torch.long,
        )

    @staticmethod
    def _detect_backend(matrix: Any) -> str:
        if torch.is_tensor(matrix):
            return "torch"
        if isinstance(matrix, np.ndarray):
            return "numpy"
        if sp is not None and sp.issparse(matrix):
            return "scipy_sparse"
        if ad is not None and isinstance(matrix, ad.AnnData):
            return "anndata"
        raise TypeError(
            "matrix must be a torch.Tensor, numpy.ndarray, scipy sparse matrix, or AnnData."
        )

    @staticmethod
    def _infer_shape(matrix: Any) -> tuple[int, int]:
        if hasattr(matrix, "shape") and len(matrix.shape) == 2:
            return int(matrix.shape[0]), int(matrix.shape[1])
        raise ValueError(f"matrix must be 2D, got shape={getattr(matrix, 'shape', None)}")

    def __len__(self) -> int:
        return self._num_cells

    def _row_to_tensor(self, index: int) -> torch.Tensor:
        if self._matrix_backend == "torch":
            row = self.matrix[index]
            return row.detach().clone().to(dtype=torch.float32)
        if self._matrix_backend == "numpy":
            return torch.as_tensor(self.matrix[index], dtype=torch.float32)
        if self._matrix_backend == "scipy_sparse":
            row = self.matrix.getrow(index).toarray().reshape(-1)
            return torch.as_tensor(row, dtype=torch.float32)
        if self._matrix_backend == "anndata":
            row = self.matrix.X[index]
            if sp is not None and sp.issparse(row):
                row = row.toarray()
            return torch.as_tensor(np.asarray(row).reshape(-1), dtype=torch.float32)
        raise RuntimeError(f"Unsupported backend {self._matrix_backend}")

    def _targets_row_to_tensor(self, index: int) -> torch.Tensor:
        source = self.gene_targets_matrix
        if torch.is_tensor(source):
            return source[index].detach().clone().to(dtype=torch.long)
        if isinstance(source, np.ndarray):
            return torch.as_tensor(source[index], dtype=torch.long)
        if sp is not None and sp.issparse(source):
            row = source.getrow(index).toarray().reshape(-1)
            return torch.as_tensor(row, dtype=torch.long)
        if ad is not None and isinstance(source, ad.AnnData):
            row = source.X[index]
            if sp is not None and sp.issparse(row):
                row = row.toarray()
            return torch.as_tensor(np.asarray(row).reshape(-1), dtype=torch.long)
        raise RuntimeError("Unsupported gene_targets_matrix backend.")

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            "expression": self._row_to_tensor(index),
            "gene_token_ids": self.feature_gene_token_ids,
            "sample_id": str(index) if self.sample_ids is None else self.sample_ids[index],
        }
        if self.cell_labels is not None:
            item["cell_label"] = self.cell_labels[index]
        if self.gene_targets_matrix is not None:
            item["gene_labels"] = self._targets_row_to_tensor(index)
        if self.cell_metadata is not None:
            item["cell_metadata"] = self.cell_metadata[index]
        return item


def build_synthetic_dataset(
    gene_vocab: GeneVocab,
    config: SyntheticDatasetConfig | None = None,
) -> SingleCellMatrixDataset:
    """Create a tiny sparse-ish synthetic dataset for smoke testing."""

    cfg = SyntheticDatasetConfig() if config is None else config
    if cfg.num_genes != gene_vocab.num_genes:
        raise ValueError(
            f"Synthetic dataset num_genes ({cfg.num_genes}) must match gene vocab ({gene_vocab.num_genes})."
        )

    generator = torch.Generator().manual_seed(cfg.seed)
    mask = torch.rand(cfg.num_cells, cfg.num_genes, generator=generator) < cfg.density
    values = torch.poisson(torch.full((cfg.num_cells, cfg.num_genes), 2.0))
    matrix = torch.where(mask, values, torch.zeros_like(values)).to(torch.float32)
    cell_labels = torch.randint(
        low=0,
        high=cfg.num_cell_types,
        size=(cfg.num_cells,),
        generator=generator,
    )
    return SingleCellMatrixDataset(
        matrix=matrix,
        gene_vocab=gene_vocab,
        gene_names=list(gene_vocab.gene_names),
        cell_labels=cell_labels,
        sample_ids=[f"cell-{idx}" for idx in range(cfg.num_cells)],
    )
