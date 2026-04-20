# Concord

Concord is a dual-token single-cell foundation model that keeps gene identity tokens and expression-value tokens separate while updating them with one shared Transformer backbone and collaborative rotary attention.

This repo preserves the existing CRA/backbone implementation in `src/models/attention.py` and `src/models/backbone.py`, then builds the surrounding production scaffolding around it:

- dual-mode `ConcordModel`
- gene and expression tokenizers
- modular pretraining losses and heads
- dense/sparse/AnnData-ready dataset interfaces
- FSDP-first training helpers with CPU-safe fallbacks
- checkpointing and resume support
- smoke tests on tiny synthetic data

## What Is Implemented

- `src/models/attention.py`
  - existing `RandomProjection` and `FlashCRA`
  - added optional `rotary_mask` so `[g-cls]` and `[e-cls]` do not receive collaborative rotary phases
  - added PyTorch SDPA fallback for CPU or non-FlashAttention execution
- `src/models/backbone.py`
  - existing `SwiGLUMLP`, `TransformerBlock`, and `TransformerBackbone`
  - threaded `rotary_mask` through the shared backbone API
- `src/models/tokenizers.py`
  - `GeneTokenizer` with optional Gene2Vec initialization hook
  - `ExpressionTokenizer` with configurable fixed-bin `log1p` discretization
- `src/models/concord_model.py`
  - `forward_expression_mode(...)`
  - `forward_gene_mode(...)`
  - `forward_pretrain(..., phase=...)`
- `src/models/heads.py`
  - expression reconstruction head
  - masked gene prediction head
  - cell-level and gene-level downstream heads
- `src/data/`
  - `GeneVocab`
  - dataset wrappers for dense arrays, torch tensors, scipy sparse matrices, and AnnData
  - collator that builds Concord batches with CLS-aware rotary and masking tensors
- `src/losses/`
  - expression reconstruction loss
  - masked gene prediction loss
  - symmetric cell contrastive loss
- `src/train/`
  - config loading with includes
  - distributed/FSDP helpers
  - checkpoint manager
  - shared trainer
  - pretraining and fine-tuning entrypoints

## Important Defaults

- The backbone keeps `SwiGLUMLP` rather than switching to GEGLU. This is a deliberate preservation choice to stay compatible with the existing implementation base.
- Masked gene prediction defaults to gene-identity prediction because the paper names gene categories but does not specify a concrete label source in the provided spec.
- Default pretraining follows sequential phases:
  1. expression reconstruction
  2. gene prediction
  3. cell contrastive alignment
- Sequence construction uses non-zero genes only, sorted by descending expression and truncated to `max_tokens - 1`, then prepends a CLS token.

## Install

Minimal install:

```bash
python -m pip install -e .
```

For tests:

```bash
python -m pip install -e .[dev]
```

For AnnData or scipy sparse inputs:

```bash
python -m pip install -e .[singlecell]
```

FlashAttention is optional because the repo now has a CPU-safe SDPA fallback:

```bash
python -m pip install -e .[flash]
```

## Running

Pretraining with the shipped config:

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

The shipped configs use paper-style model defaults (`embed_dim=768`, `num_layers=12`, `num_heads=12`, `max_tokens=2048`) but point at a tiny synthetic dataset so the repo is runnable without an external corpus.

## Batch Contract

The collator standardizes Concord batches around these keys:

- `gene_ids`
- `expression_values`
- `expression_bin_ids`
- `key_padding_mask`
- `rotary_mask`
- `expression_mask`
- `gene_mask`
- `expression_targets`
- `gene_targets`

Cell-level fine-tuning additionally uses `cell_labels`. Gene-level fine-tuning can use `gene_labels` when provided by the dataset.

## Checkpoints

Checkpoints are written under the run directory, for example:

```text
outputs/pretrain/checkpoints/
```

The manager writes:

- a manifest with the latest checkpoint path
- model state
- optimizer state
- scheduler state
- trainer state for resume

When FSDP wraps the training system, model weights are saved through an FSDP-aware sharded state path.

## Smoke Tests

Run the test suite with:

```bash
pytest
```

The smoke tests cover:

- backbone and CPU fallback behavior
- tokenizer behavior
- modular pretraining losses
- checkpoint save/load roundtrips
- a tiny synthetic pretrain + fine-tune flow
