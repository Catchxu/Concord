from __future__ import annotations

import argparse
import math

from pathlib import Path
from typing import Any

import torch

from src.data import ConcordCollator, GeneVocab, SyntheticDatasetConfig, build_synthetic_dataset
from src.losses import PretrainingLossManager
from src.models import (
    ConcordModel,
    ExpressionReconstructionHead,
    ExpressionTokenizer,
    GeneTokenizer,
    MaskedGenePredictionHead,
    TransformerBackbone,
    TransformerBlock,
)

from .checkpointing import CheckpointManager
from .config import load_config, save_config
from .distributed import (
    barrier,
    build_dataloader,
    cleanup_runtime,
    initialize_runtime,
    maybe_apply_activation_checkpointing,
    maybe_wrap_fsdp,
    setup_logger,
)
from .trainer import PretrainingSystem, Trainer, build_optimizer, build_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain Concord.")
    parser.add_argument("--config", default="configs/pretrain.yaml")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def _build_gene_vocab(data_cfg: dict[str, Any]) -> GeneVocab:
    gene_names = data_cfg.get("gene_names")
    if gene_names is None:
        num_genes = int(data_cfg.get("synthetic", {}).get("num_genes", 64))
        gene_names = [f"gene_{idx}" for idx in range(num_genes)]
    return GeneVocab.from_gene_names(gene_names)


def _build_dataset(config: dict[str, Any], gene_vocab: GeneVocab):
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("name", "synthetic")).lower()
    if dataset_name != "synthetic":
        raise ValueError("This initial Concord implementation ships a synthetic CLI dataset path.")
    synthetic_cfg = SyntheticDatasetConfig(**data_cfg.get("synthetic", {}))
    return build_synthetic_dataset(gene_vocab, config=synthetic_cfg)


def _build_concord_model(config: dict[str, Any], gene_vocab: GeneVocab) -> ConcordModel:
    model_cfg = config["model"]
    expression_cfg = model_cfg.get("expression_tokenizer", {})
    gene_tokenizer = GeneTokenizer(
        gene_vocab=gene_vocab,
        embed_dim=int(model_cfg["embed_dim"]),
    )
    expression_tokenizer = ExpressionTokenizer(
        embed_dim=int(model_cfg["embed_dim"]),
        num_bins=int(expression_cfg.get("num_bins", 64)),
        max_log1p_value=float(expression_cfg.get("max_log1p_value", 10.0)),
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
    return ConcordModel(
        gene_tokenizer=gene_tokenizer,
        expression_tokenizer=expression_tokenizer,
        backbone=backbone,
        normalize_condition_tokens=bool(model_cfg.get("normalize_condition_tokens", True)),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = initialize_runtime(seed=int(config["runtime"].get("seed", 0)))

    run_dir = Path(config["trainer"].get("output_dir", "outputs/pretrain")).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("concord.pretrain", run_dir / "train.log", runtime)
    save_config(run_dir / "config_resolved.yaml", config)

    try:
        gene_vocab = _build_gene_vocab(config["data"])
        dataset = _build_dataset(config, gene_vocab)
        concord_model = _build_concord_model(config, gene_vocab)

        collator = ConcordCollator(
            concord_model.gene_tokenizer,
            concord_model.expression_tokenizer,
            max_tokens=int(config["data"]["max_tokens"]),
            expression_mask_ratio=float(config["pretrain"].get("expression_mask_ratio", 0.15)),
            gene_mask_ratio=float(config["pretrain"].get("gene_mask_ratio", 0.15)),
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

        system = PretrainingSystem(
            concord_model=concord_model,
            expression_head=ExpressionReconstructionHead(embed_dim=int(config["model"]["embed_dim"])),
            gene_head=MaskedGenePredictionHead(
                embed_dim=int(config["model"]["embed_dim"]),
                vocab_size=gene_vocab.vocab_size,
            ),
            loss_manager=PretrainingLossManager(
                expression_weight=float(config["pretrain"].get("expression_weight", 1.0)),
                gene_weight=float(config["pretrain"].get("gene_weight", 1.0)),
                contrastive_weight=float(config["pretrain"].get("contrastive_weight", 1.0)),
                contrastive_temperature=float(config["pretrain"].get("contrastive_temperature", 0.1)),
            ),
        )

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
            use_orig_params=bool(config["runtime"].get("fsdp", {}).get("use_orig_params", True)),
            limit_all_gathers=bool(config["runtime"].get("fsdp", {}).get("limit_all_gathers", True)),
            sync_module_states=bool(config["runtime"].get("fsdp", {}).get("sync_module_states", True)),
        )

        total_steps = max(
            1,
            len(train_loader)
            * int(config["trainer"]["epochs_per_phase"])
            * len(config["pretrain"]["phase_order"]),
        )
        optimizer = build_optimizer(system, config["optimizer"])
        scheduler = build_scheduler(optimizer, config["scheduler"], total_steps=total_steps)
        checkpoint_manager = CheckpointManager(run_dir / "checkpoints", runtime=runtime)

        trainer_state = None
        if args.resume is not None:
            loaded_state = checkpoint_manager.load(
                model=system,
                optimizer=optimizer,
                scheduler=scheduler,
                checkpoint_dir=args.resume,
            )
            if loaded_state is not None:
                trainer_state = loaded_state.get("trainer_state")

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

        if trainer_state is None:
            trainer_state = {"phase_index": 0, "global_step": 0}

        phase_order = list(config["pretrain"]["phase_order"])
        for phase_index, phase in enumerate(phase_order):
            if phase_index < int(trainer_state.get("phase_index", 0)):
                continue
            logger.info("Starting pretraining phase: %s", phase)
            phase_state = {
                "epoch": int(trainer_state.get("epoch", 0)) if phase_index == int(trainer_state.get("phase_index", 0)) else 0,
                "global_step": int(trainer_state.get("global_step", 0)),
                "phase_index": phase_index,
            }
            phase_state = trainer.train_epochs(
                train_loader,
                epochs=int(config["trainer"]["epochs_per_phase"]),
                trainer_state=phase_state,
                save_every_epochs=int(config["trainer"].get("save_every_epochs", 1)),
                resolved_config=config,
                forward_kwargs={"phase": phase},
            )
            trainer_state.update(phase_state)
            trainer_state["phase_index"] = phase_index + 1
            trainer_state["epoch"] = 0
            barrier()

        logger.info("Pretraining completed. Final global_step=%s", trainer_state.get("global_step", 0))
    finally:
        cleanup_runtime()


if __name__ == "__main__":
    main()
