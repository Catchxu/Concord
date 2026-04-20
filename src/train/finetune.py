from __future__ import annotations

import argparse
import math

from pathlib import Path
from typing import Any

import torch

from src.data import ConcordCollator, GeneVocab, SyntheticDatasetConfig, build_synthetic_dataset
from src.models import (
    CellClassificationHead,
    ConcordModel,
    ExpressionTokenizer,
    GeneTaskHead,
    GeneTokenizer,
    TransformerBackbone,
    TransformerBlock,
)

from .checkpointing import CheckpointManager
from .config import load_config, save_config
from .distributed import (
    build_dataloader,
    cleanup_runtime,
    initialize_runtime,
    maybe_apply_activation_checkpointing,
    maybe_wrap_fsdp,
    setup_logger,
)
from .trainer import (
    CellFineTuneSystem,
    GeneFineTuneSystem,
    Trainer,
    build_optimizer,
    build_scheduler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Concord.")
    parser.add_argument("--config", default="configs/finetune_cta.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _build_gene_vocab(data_cfg: dict[str, Any]) -> GeneVocab:
    gene_names = data_cfg.get("gene_names")
    if gene_names is None:
        num_genes = int(data_cfg.get("synthetic", {}).get("num_genes", 64))
        gene_names = [f"gene_{idx}" for idx in range(num_genes)]
    return GeneVocab.from_gene_names(gene_names)


def _build_concord_model(config: dict[str, Any], gene_vocab: GeneVocab) -> ConcordModel:
    model_cfg = config["model"]
    gene_tokenizer = GeneTokenizer(gene_vocab, embed_dim=int(model_cfg["embed_dim"]))
    expression_tokenizer = ExpressionTokenizer(
        embed_dim=int(model_cfg["embed_dim"]),
        num_bins=int(model_cfg.get("expression_tokenizer", {}).get("num_bins", 64)),
        max_log1p_value=float(model_cfg.get("expression_tokenizer", {}).get("max_log1p_value", 10.0)),
    )
    backbone = TransformerBackbone(
        embed_dim=int(model_cfg["embed_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_hidden_dim=int(model_cfg.get("mlp_hidden_dim", 4 * int(model_cfg["embed_dim"]))),
        attn_dropout=float(model_cfg.get("attn_dropout", 0.0)),
        mlp_dropout=float(model_cfg.get("mlp_dropout", 0.0)),
        phase_scale=float(model_cfg.get("phase_scale", math.pi / 4)),
        rotary_interleaved=bool(model_cfg.get("rotary_interleaved", False)),
    )
    return ConcordModel(gene_tokenizer, expression_tokenizer, backbone)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = initialize_runtime(seed=int(config["runtime"].get("seed", 0)))

    run_dir = Path(config["trainer"].get("output_dir", "outputs/finetune")).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("concord.finetune", run_dir / "train.log", runtime)
    save_config(run_dir / "config_resolved.yaml", config)

    try:
        gene_vocab = _build_gene_vocab(config["data"])
        dataset = build_synthetic_dataset(
            gene_vocab,
            config=SyntheticDatasetConfig(**config["data"].get("synthetic", {})),
        )
        concord_model = _build_concord_model(config, gene_vocab)
        collator = ConcordCollator(
            concord_model.gene_tokenizer,
            concord_model.expression_tokenizer,
            max_tokens=int(config["data"]["max_tokens"]),
            expression_mask_ratio=float(config["data"].get("expression_mask_ratio", 0.0)),
            gene_mask_ratio=float(config["data"].get("gene_mask_ratio", 0.0)),
            seed=int(config["runtime"].get("seed", 0)),
        )
        train_loader = build_dataloader(
            dataset,
            batch_size=int(config["data"]["batch_size"]),
            shuffle=True,
            collate_fn=collator,
            runtime=runtime,
            num_workers=int(config["data"].get("num_workers", 0)),
        )

        task_type = str(config["task"]["type"]).lower()
        if task_type == "cell_classification":
            system: torch.nn.Module = CellFineTuneSystem(
                concord_model=concord_model,
                head=CellClassificationHead(
                    embed_dim=int(config["model"]["embed_dim"]),
                    num_classes=int(config["task"]["num_classes"]),
                    dropout=float(config["task"].get("dropout", 0.1)),
                ),
                pool=str(config["task"].get("pool", "cls")),
                freeze_backbone=bool(config["task"].get("freeze_backbone", False)),
            )
        elif task_type == "gene_task":
            system = GeneFineTuneSystem(
                concord_model=concord_model,
                head=GeneTaskHead(
                    embed_dim=int(config["model"]["embed_dim"]),
                    output_dim=int(config["task"]["output_dim"]),
                ),
                freeze_backbone=bool(config["task"].get("freeze_backbone", False)),
            )
        else:
            raise ValueError(f"Unsupported fine-tuning task type {task_type}.")

        maybe_apply_activation_checkpointing(
            system,
            enabled=bool(config["runtime"].get("activation_checkpointing", False)),
            target_classes=(TransformerBlock,),
        )
        system = maybe_wrap_fsdp(
            system,
            runtime=runtime,
            enabled=bool(config["runtime"].get("fsdp", {}).get("enabled", runtime.distributed)),
            precision=str(config["runtime"].get("precision", "fp32")),
            target_classes=(TransformerBlock,),
            sharding_strategy=str(config["runtime"].get("fsdp", {}).get("sharding_strategy", "FULL_SHARD")),
        )

        total_steps = max(1, len(train_loader) * int(config["trainer"]["epochs"]))
        optimizer = build_optimizer(system, config["optimizer"])
        scheduler = build_scheduler(optimizer, config["scheduler"], total_steps=total_steps)
        checkpoint_manager = CheckpointManager(run_dir / "checkpoints", runtime=runtime)

        if args.resume is not None:
            checkpoint_manager.load(
                model=system,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=args.resume,
            )

        trainer = Trainer(
            model=system,
            optimizer=optimizer,
            scheduler=scheduler,
            runtime=runtime,
            checkpoint_manager=checkpoint_manager,
            logger=logger,
            precision=str(config["runtime"].get("precision", "fp32")),
            grad_accum_steps=int(config["trainer"].get("gradient_accumulation_steps", 1)),
            grad_clip_norm=config["trainer"].get("grad_clip_norm"),
            log_every_steps=int(config["trainer"].get("log_every_steps", 10)),
        )
        trainer.train_epochs(
            train_loader,
            epochs=int(config["trainer"]["epochs"]),
            trainer_state={"epoch": 0, "global_step": 0},
            save_every_epochs=int(config["trainer"].get("save_every_epochs", 1)),
            resolved_config=config,
        )
        logger.info("Fine-tuning completed.")
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()
