import torch

from src.train.checkpointing import CheckpointManager
from src.train.distributed import RuntimeContext


def test_checkpoint_manager_roundtrip(tmp_path):
    runtime = RuntimeContext(
        rank=0,
        world_size=1,
        local_rank=0,
        device=torch.device("cpu"),
        distributed=False,
        is_main=True,
    )
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    manager = CheckpointManager(tmp_path, runtime=runtime)
    original_weight = model.weight.detach().clone()
    manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state={"global_step": 3, "epoch": 1},
        resolved_config={"example": True},
    )

    with torch.no_grad():
        model.weight.add_(10.0)
    manager.load(model=model, optimizer=optimizer, scheduler=scheduler)
    assert torch.allclose(model.weight, original_weight)
