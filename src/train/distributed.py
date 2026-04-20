from __future__ import annotations

import contextlib
import logging
import os
import random

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.distributed as dist

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    MixedPrecision,
    ShardingStrategy,
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler


@dataclass
class RuntimeContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    distributed: bool
    is_main: bool


def seed_everything(seed: int, *, rank: int = 0) -> None:
    seed_value = int(seed) + int(rank)
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def initialize_runtime(seed: int = 0) -> RuntimeContext:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    distributed = world_size > 1

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    runtime = RuntimeContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
        is_main=(rank == 0),
    )
    seed_everything(seed, rank=runtime.rank)
    return runtime


def cleanup_runtime() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def reduce_scalar_dict(metrics: dict[str, float], runtime: RuntimeContext) -> dict[str, float]:
    if not runtime.distributed or not metrics:
        return metrics
    keys = list(metrics.keys())
    values = torch.tensor(
        [float(metrics[key]) for key in keys],
        device=runtime.device,
        dtype=torch.float64,
    )
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= runtime.world_size
    return {key: float(values[idx].item()) for idx, key in enumerate(keys)}


def build_dataloader(
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    collate_fn: Any,
    runtime: RuntimeContext,
    num_workers: int = 0,
    drop_last: bool = False,
) -> DataLoader:
    sampler = None
    if runtime.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            num_replicas=runtime.world_size,
            rank=runtime.rank,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=(runtime.device.type == "cuda"),
    )


def setup_logger(name: str, log_path: str | Path, runtime: RuntimeContext) -> logging.Logger:
    log_path = Path(log_path).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if runtime.is_main:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        logger.addHandler(logging.NullHandler())
    return logger


def autocast_context(runtime: RuntimeContext, precision: str) -> contextlib.AbstractContextManager[None]:
    normalized = str(precision).lower()
    if runtime.device.type != "cuda" or normalized in {"fp32", "float32"}:
        return contextlib.nullcontext()
    if normalized in {"bf16", "bfloat16"}:
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized in {"fp16", "float16"}:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported autocast precision {precision}.")


def _resolve_fsdp_mixed_precision(precision: str) -> MixedPrecision | None:
    normalized = str(precision).lower()
    if normalized in {"bf16", "bfloat16"}:
        dtype = torch.bfloat16
    elif normalized in {"fp16", "float16"}:
        dtype = torch.float16
    elif normalized in {"fp32", "float32"}:
        return None
    else:
        raise ValueError(f"Unsupported FSDP precision {precision}.")
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def maybe_apply_activation_checkpointing(
    module: torch.nn.Module,
    *,
    enabled: bool,
    target_classes: Iterable[type[torch.nn.Module]],
) -> None:
    if not enabled:
        return
    target_classes = tuple(target_classes)
    if not target_classes:
        return

    wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    apply_activation_checkpointing(
        module,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda submodule: isinstance(submodule, target_classes),
    )


def maybe_wrap_fsdp(
    module: torch.nn.Module,
    *,
    runtime: RuntimeContext,
    enabled: bool,
    precision: str,
    target_classes: Iterable[type[torch.nn.Module]],
    sharding_strategy: str = "FULL_SHARD",
    use_orig_params: bool = True,
    limit_all_gathers: bool = True,
    sync_module_states: bool = True,
) -> torch.nn.Module:
    module = module.to(runtime.device)
    if not enabled or not runtime.distributed or runtime.device.type != "cuda":
        return module

    target_classes = tuple(target_classes)
    auto_wrap_policy = None
    if target_classes:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=set(target_classes),
        )

    mixed_precision = _resolve_fsdp_mixed_precision(precision)
    fsdp_sharding = getattr(ShardingStrategy, str(sharding_strategy).upper())
    return FSDP(
        module,
        device_id=runtime.device,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=fsdp_sharding,
        use_orig_params=use_orig_params,
        limit_all_gathers=limit_all_gathers,
        sync_module_states=sync_module_states,
    )
