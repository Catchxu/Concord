# Concord: Building Consensus Representations for Single Cells with Collaborative Random Projection

Concord is a dual-token single-cell model. Each cell is represented with:

- gene tokens for gene identity
- expression tokens for expression value

Both token streams use one shared Transformer backbone. In expression mode, gene tokens condition expression updates. In gene mode, expression tokens condition gene updates.

<br/>
<div align=center>
<img src="/docs/framework.png" width="70%">
</div>
<br/>

## Framework

The framework figure shows the full training story:

- `a`: dual tokenization plus a shared CRA backbone
- `b`: masked gene prediction in gene mode
- `c`: masked expression reconstruction in expression mode
- `d`: `[g-cls]` and `[e-cls]` aggregation with cell-level contrastive learning

This repo keeps the original CRA/backbone implementation as the core and builds the missing training and data stack around it.

## What Is In This Repo

- `src/models/attention.py`, `src/models/backbone.py`
  Collaborative rotary attention and the shared Transformer backbone.
- `src/models/tokenizers.py`
  Gene and expression tokenizers, including CLS handling and expression binning.
- `src/models/concord_model.py`
  The dual-mode `ConcordModel` wrapper with:
  `forward_expression_mode(...)`, `forward_gene_mode(...)`, and `forward_pretrain(...)`.
- `src/models/heads.py`
  Pretraining heads and downstream task heads.
- `src/data/`
  Gene vocabulary, dataset wrappers, masking, and collate logic.
- `src/losses/`
  Expression reconstruction, masked gene prediction, and cell contrastive losses.
- `src/train/`
  Config loading, distributed helpers, FSDP wrapping, checkpointing, pretraining, and fine-tuning.
- `configs/`
  Shared defaults plus pretraining and fine-tuning configs.
- `tests/`
  Smoke tests for the backbone, tokenizers, losses, checkpoints, and a tiny synthetic run.

## Important Defaults

- The shared backbone stays close to the existing implementation.
- `SwiGLUMLP` is preserved instead of replacing it with a different MLP block.
- CLS tokens do not receive collaborative rotary phase transforms.
- Attention execution follows the `scCAFM` style:
  prefer FA4/CuTe kernels when supported, otherwise fall back to FA2.
- No non-FlashAttention execution path is enabled for the Concord attention stack.
- Pretraining defaults to three sequential phases:
  1. expression reconstruction
  2. gene prediction
  3. cell contrastive alignment
- Masked gene prediction currently defaults to gene identity prediction, because the paper mentions gene categories but does not specify a concrete category source in the provided spec.

## Quick Start

Install:

```bash
python -m pip install -e .
```

Install with tests:

```bash
python -m pip install -e .[dev]
```

Optional extras:

- `.[singlecell]` for `anndata` and sparse single-cell inputs
- `.[flash]` for FlashAttention

FlashAttention-backed model execution requires CUDA.

## Run

Pretraining:

```bash
python -m src.train.pretrain --config configs/pretrain.yaml
```

Cell-level fine-tuning:

```bash
python -m src.train.finetune --config configs/finetune_cta.yaml
```

Gene-level fine-tuning:

```bash
python -m src.train.finetune --config configs/finetune_gene.yaml
```

The shipped configs use paper-style model defaults such as `embed_dim=768`, `num_layers=12`, `num_heads=12`, and `max_tokens=2048`. The default CLI dataset path is synthetic so the repo can run without a large external corpus, but actual model execution still requires CUDA because the attention stack is FlashAttention-only.

## Batch Format

The collator standardizes training batches around:

- `gene_ids`
- `expression_values`
- `expression_bin_ids`
- `key_padding_mask`
- `rotary_mask`
- `expression_mask`
- `gene_mask`
- `expression_targets`
- `gene_targets`

Fine-tuning may additionally use `cell_labels` or `gene_labels`.

## Checkpoints

Checkpoints are written under the run directory, for example:

```text
outputs/pretrain/checkpoints/
```

The checkpoint manager stores:

- model state
- optimizer state
- scheduler state
- trainer state
- a manifest pointing to the latest checkpoint

When FSDP is active, model state is saved through an FSDP-aware sharded checkpoint path.

## Tests

Run the smoke suite with:

```bash
pytest
```

Current smoke coverage includes:

- backbone shape and masking behavior
- tokenizer determinism
- modular pretraining losses
- checkpoint save/load
- tiny synthetic pretrain and fine-tune flows on CUDA
