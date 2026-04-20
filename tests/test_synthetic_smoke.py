import logging

import torch

from src.data import ConcordCollator, GeneVocab, SyntheticDatasetConfig, build_synthetic_dataset
from src.losses import PretrainingLossManager
from src.models import (
    CellClassificationHead,
    ConcordModel,
    ExpressionReconstructionHead,
    ExpressionTokenizer,
    GeneTokenizer,
    MaskedGenePredictionHead,
    TransformerBackbone,
)
from src.train.checkpointing import CheckpointManager
from src.train.distributed import RuntimeContext, build_dataloader
from src.train.trainer import CellFineTuneSystem, PretrainingSystem, Trainer, build_optimizer, build_scheduler


def test_tiny_end_to_end_synthetic_run(tmp_path):
    runtime = RuntimeContext(
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
        distributed=False,
        is_main=True,
    )
    gene_vocab = GeneVocab.from_gene_names([f"gene_{idx}" for idx in range(24)])
    dataset = build_synthetic_dataset(
        gene_vocab,
        config=SyntheticDatasetConfig(num_cells=12, num_genes=24, num_cell_types=3, seed=3, density=0.3),
    )

    gene_tokenizer = GeneTokenizer(gene_vocab=gene_vocab, embed_dim=32)
    expression_tokenizer = ExpressionTokenizer(embed_dim=32, num_bins=8)
    collator = ConcordCollator(
        gene_tokenizer,
        expression_tokenizer,
        max_tokens=12,
        expression_mask_ratio=0.2,
        gene_mask_ratio=0.2,
        seed=3,
    )
    loader = build_dataloader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
        runtime=runtime,
    )

    concord_model = ConcordModel(
        gene_tokenizer=gene_tokenizer,
        expression_tokenizer=expression_tokenizer,
        backbone=TransformerBackbone(
            embed_dim=32,
            num_layers=2,
            num_heads=4,
            mlp_hidden_dim=64,
            attn_dropout=0.0,
            mlp_dropout=0.0,
            phase_scale=0.7853981633974483,
        ),
    )
    pretrain_system = PretrainingSystem(
        concord_model=concord_model,
        expression_head=ExpressionReconstructionHead(embed_dim=32),
        gene_head=MaskedGenePredictionHead(embed_dim=32, vocab_size=gene_vocab.vocab_size),
        loss_manager=PretrainingLossManager(),
    )
    optimizer = build_optimizer(pretrain_system, {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0})
    scheduler = build_scheduler(optimizer, {"name": "linear_warmup_decay", "warmup_ratio": 0.0}, total_steps=4)
    logger = logging.getLogger("concord-test-pretrain")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    checkpoint_manager = CheckpointManager(tmp_path / "checkpoints", runtime=runtime)
    trainer = Trainer(
        model=pretrain_system,
        optimizer=optimizer,
        scheduler=scheduler,
        runtime=runtime,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        precision="fp32",
        grad_accum_steps=1,
        grad_clip_norm=1.0,
        log_every_steps=100,
    )
    state = trainer.train_epochs(
        loader,
        epochs=1,
        trainer_state={"epoch": 0, "global_step": 0},
        save_every_epochs=1,
        resolved_config={"smoke": True},
        forward_kwargs={"phase": "expression_reconstruction"},
    )
    assert state["global_step"] > 0

    finetune_system = CellFineTuneSystem(
        concord_model=concord_model,
        head=CellClassificationHead(embed_dim=32, num_classes=3),
    )
    finetune_optimizer = build_optimizer(
        finetune_system,
        {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
    )
    finetune_scheduler = build_scheduler(
        finetune_optimizer,
        {"name": "linear_warmup_decay", "warmup_ratio": 0.0},
        total_steps=4,
    )
    finetune_trainer = Trainer(
        model=finetune_system,
        optimizer=finetune_optimizer,
        scheduler=finetune_scheduler,
        runtime=runtime,
        checkpoint_manager=None,
        logger=logger,
        precision="fp32",
        grad_accum_steps=1,
        grad_clip_norm=1.0,
        log_every_steps=100,
    )
    finetune_state = finetune_trainer.train_epochs(
        loader,
        epochs=1,
        trainer_state={"epoch": 0, "global_step": 0},
        forward_kwargs={},
    )
    assert finetune_state["global_step"] > 0
