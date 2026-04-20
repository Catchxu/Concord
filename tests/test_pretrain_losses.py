import torch

from src.losses import PretrainingLossManager
from src.models.concord_model import ConcordModeOutput, ConcordPretrainOutput


def test_pretraining_loss_manager_expression_and_gene_paths():
    manager = PretrainingLossManager()
    batch = {
        "expression_targets": torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
        "expression_mask": torch.tensor([[False, True, False]], dtype=torch.bool),
        "gene_targets": torch.tensor([[-100, 4, -100]], dtype=torch.long),
        "gene_ids": torch.tensor([[1, 4, 0]], dtype=torch.long),
    }
    dummy_mode = ConcordModeOutput(
        sequence_embeddings=torch.randn(1, 3, 8),
        cls_embedding=torch.randn(1, 8),
        token_embeddings=torch.randn(1, 2, 8),
        key_padding_mask=None,
        rotary_mask=None,
        input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long),
    )

    expr_bundle = manager(
        phase="expression_reconstruction",
        pretrain_output=ConcordPretrainOutput(
            phase="expression_reconstruction",
            expression_mode=dummy_mode,
            gene_mode=None,
            expression_mask=batch["expression_mask"],
            gene_mask=None,
            expression_targets=batch["expression_targets"],
            gene_targets=batch["gene_targets"],
        ),
        head_outputs={"expression_reconstruction": torch.tensor([[0.0, 1.5, 2.0]])},
        batch=batch,
    )
    assert expr_bundle.loss.item() >= 0
    assert "expression_reconstruction_loss" in expr_bundle.metrics

    gene_bundle = manager(
        phase="gene_prediction",
        pretrain_output=ConcordPretrainOutput(
            phase="gene_prediction",
            expression_mode=None,
            gene_mode=dummy_mode,
            expression_mask=None,
            gene_mask=torch.tensor([[False, True, False]], dtype=torch.bool),
            expression_targets=batch["expression_targets"],
            gene_targets=batch["gene_targets"],
        ),
        head_outputs={"gene_prediction": torch.randn(1, 3, 8)},
        batch=batch,
    )
    assert gene_bundle.loss.item() >= 0
    assert "gene_prediction_loss" in gene_bundle.metrics


def test_pretraining_loss_manager_contrastive_path():
    manager = PretrainingLossManager()
    mode_a = ConcordModeOutput(
        sequence_embeddings=torch.randn(4, 3, 8),
        cls_embedding=torch.randn(4, 8),
        token_embeddings=torch.randn(4, 2, 8),
        key_padding_mask=None,
        rotary_mask=None,
        input_ids=torch.zeros(4, 3, dtype=torch.long),
    )
    mode_b = ConcordModeOutput(
        sequence_embeddings=torch.randn(4, 3, 8),
        cls_embedding=torch.randn(4, 8),
        token_embeddings=torch.randn(4, 2, 8),
        key_padding_mask=None,
        rotary_mask=None,
        input_ids=torch.zeros(4, 3, dtype=torch.long),
    )
    batch = {
        "gene_ids": torch.zeros(4, 3, dtype=torch.long),
        "expression_targets": torch.zeros(4, 3),
        "expression_mask": torch.zeros(4, 3, dtype=torch.bool),
        "gene_targets": torch.full((4, 3), -100, dtype=torch.long),
    }
    bundle = manager(
        phase="contrastive",
        pretrain_output=ConcordPretrainOutput(
            phase="contrastive",
            expression_mode=mode_a,
            gene_mode=mode_b,
            expression_mask=None,
            gene_mask=None,
            expression_targets=None,
            gene_targets=None,
        ),
        head_outputs={},
        batch=batch,
    )
    assert bundle.loss.item() >= 0
    assert "contrastive_loss" in bundle.metrics
