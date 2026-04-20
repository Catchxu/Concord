from __future__ import annotations

import json
import warnings

from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp

from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from .distributed import RuntimeContext, barrier


class CheckpointManager:
    """Save and restore Concord training state with FSDP-aware model checkpoints."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        runtime: RuntimeContext,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.runtime = runtime
        self.manifest_path = self.root_dir / "manifest.json"

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {}
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        if not self.runtime.is_main:
            return
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def latest_checkpoint(self) -> Path | None:
        manifest = self._read_manifest()
        latest = manifest.get("latest_checkpoint")
        if latest is None:
            return None
        return Path(latest)

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        trainer_state: dict[str, Any],
        resolved_config: dict[str, Any],
    ) -> Path:
        global_step = int(trainer_state.get("global_step", 0))
        checkpoint_dir = self.root_dir / f"step_{global_step:08d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(model, FSDP):
            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated\.",
                    category=FutureWarning,
                )
                with FSDP.state_dict_type(
                    model,
                    StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=state_cfg,
                ):
                    model_state = model.state_dict()
            dcp.save({"model": model_state}, checkpoint_id=checkpoint_dir / "model")
        elif self.runtime.is_main:
            torch.save(model.state_dict(), checkpoint_dir / "model.pt")

        if self.runtime.is_main:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "trainer_state": trainer_state,
                    "resolved_config": resolved_config,
                },
                checkpoint_dir / "training_state.pt",
            )
            self._write_manifest(
                {
                    "latest_checkpoint": str(checkpoint_dir),
                    "global_step": global_step,
                    "trainer_state": trainer_state,
                }
            )

        barrier()
        return checkpoint_dir

    def load(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, Any] | None:
        resolved_dir = Path(checkpoint_dir).expanduser().resolve() if checkpoint_dir else self.latest_checkpoint()
        if resolved_dir is None:
            return None

        if isinstance(model, FSDP):
            state_cfg = ShardedStateDictConfig(offload_to_cpu=True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated\.",
                    category=FutureWarning,
                )
                with FSDP.state_dict_type(
                    model,
                    StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=state_cfg,
                ):
                    state = {"model": model.state_dict()}
                    dcp.load(state, checkpoint_id=resolved_dir / "model")
                    model.load_state_dict(state["model"])
        else:
            payload = torch.load(
                resolved_dir / "model.pt",
                map_location="cpu",
                weights_only=False,
            )
            model.load_state_dict(payload)

        training_state_path = resolved_dir / "training_state.pt"
        if not training_state_path.exists():
            return None
        training_state = torch.load(
            training_state_path,
            map_location="cpu",
            weights_only=False,
        )
        if optimizer is not None:
            optimizer.load_state_dict(training_state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(training_state["scheduler"])
        return training_state
