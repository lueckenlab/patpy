---
name: patpy.tl supervised methods
description: Donor-level supervised prediction (multi-instance learning) — MixMIL, PaSCient, PULSAR.
---

# patpy.tl — supervised donor-level prediction

These methods directly predict a donor-level label (`disease_status`, `severity`, …) from per-cell features, without producing a distance matrix as the primary output. All inherit from `SupervisedSampleMethod`.

## Symbols

```python
patpy.tl.MixMIL(sample_key, label_keys, tasks, ...)     # trainable    | pip install mixmil           | layer="X_pca",  CPU OK
patpy.tl.PaSCient(sample_key, label_keys, tasks, ...)   # trainable    | git+pascient                 | layer="X_pca",  GPU recommended
patpy.tl.PULSAR(sample_key, label_keys, tasks, ...)     # ZERO-SHOT    | git+snap-stanford/PULSAR     | layer="X_uce" (~1280 dims), GPU default
```

> **Three classes, two paradigms.**
> `MixMIL` and `PaSCient` *train* a model on your donors using their per-cell PCA features.
> `PULSAR` is **zero-shot**: it loads a pretrained HuggingFace transformer (`KuanP/pulsar-pbmc` by default) and *infers* on top of UCE foundation-model embeddings. `label_keys` is used only to **score** PULSAR's representation, never to fit it.
> Pick PULSAR only if you already have UCE-style cell embeddings in `adata.obsm["X_uce"]`. With `X_pca` or scVI/scArches outputs, use `MixMIL` instead.

Common signature for the trainable two (`MixMIL`, `PaSCient`):

```python
ModelClass(
    sample_key: str,
    label_keys: list[str] | str,           # donor-level columns in adata.obs
    tasks: list[Literal["classification", "regression", "ranking"]] | str,
    cell_group_key: str | None = None,
    layer: str = "X_pca",                  # per-cell features (in adata.obsm or adata.layers)
    likelihood: "binomial" | "categorical" = "binomial",   # MixMIL only
    n_trials: int = 2,                     # MixMIL only
    n_epochs: int = 2000,                  # MixMIL only
    seed: int = 42,
)
```

Full signature for `PULSAR`:

```python
patpy.tl.PULSAR(
    sample_key: str,
    label_keys: list[str],
    tasks: list[Literal["classification", "regression", "ranking"]],
    cell_group_key: str | None = None,
    layer: str = "X_uce",                  # NOT X_pca; PULSAR expects ~1280-dim UCE embeddings
    pretrained_model: str = "KuanP/pulsar-pbmc",   # HF model id (downloaded on first use, ~hundreds of MB)
    sample_cell_num: int = 1024,           # cells subsampled per donor at inference
    batch_size: int = 10,
    device: str = "cuda",                  # set "cpu" if no GPU; inference will be slow
    resample_num: int = 1,
    seed: int = 67,
)
```

Same protocol for all three:

```python
model = MethodClass(...)
model.prepare_anndata(adata)         # validates donor-level labels are constant per donor (and trains, for MixMIL/PaSCient)
sample_scores = model.get_sample_importance()   # per-sample score / prediction
cell_importance = model.get_cell_importance()   # per-cell attribution (attention-based methods)
```

## When to use which

- **`MixMIL`** — attention-based MIL (Engelmann et al. 2024). Single-label classification or regression on a donor-level target. `likelihood="binomial"` for binary targets, `"categorical"` for multi-class. Fast on PCA features. **Default starting point if you have `X_pca`.**
- **`PaSCient`** — multi-task donor-level model (Genentech). Heavier; supports multi-label training. Same input requirements as `MixMIL`.
- **`PULSAR`** — zero-shot foundation-model classifier (Pang et al.). Only works on UCE embeddings; downloads a HuggingFace model on first use; defaults to GPU. Use when you already have `obsm["X_uce"]` and want zero-shot scoring without training.

If the user doesn't know which to pick: **start with `MixMIL` on `layer="X_pca"`**, single label, `n_epochs=2000`, `seed=0`. It is the fastest, lightest-dependency, and works on the embeddings most CellxGene atlases ship with (`X_pca`, `X_rpca`, scVI latent, …).

## Minimal example (MixMIL)

```python
import patpy

# adata.obs has donor_id, disease_status (constant per donor, two classes)
# adata.obsm has X_pca

model = patpy.tl.MixMIL(
    sample_key="donor_id",
    label_keys=["disease_status"],
    tasks=["classification"],
    layer="X_pca",
    likelihood="binomial",
    n_trials=2,
    n_epochs=2000,
    seed=0,
)
model.prepare_anndata(adata)
sample_scores = model.get_sample_importance()
cell_importance = model.get_cell_importance()

# Evaluate
y_true = patpy.pp.extract_metadata(adata, "donor_id", ["disease_status"])["disease_status"]
y_pred = sample_scores.loc[y_true.index]
result = patpy.tl.evaluate_prediction(y_true, y_pred, task="classification")
```

## Common pitfalls

- **Donor-level labels must be constant per donor.** Every cell of a given donor must carry the same `label_keys` value. `prepare_anndata` validates this and raises if violated.
- **`MixMIL` ignores extra labels.** If `label_keys` has more than one entry, only the first is used. A warning is raised.
- **`label_keys` and `tasks` must be the same length.** Pass single strings (auto-wrapped to one-element lists) or matched lists. Mismatched lengths raise `ValueError`.
- **`layer="X_pca"` is read from `adata.obsm`** by default — not from `adata.layers`. The default name is the scanpy convention. If you have raw counts in `adata.layers["counts"]` and want to use those, set `layer="counts"`.
- **`MixMIL` install is heavy.** `pip install mixmil` pulls `torch` and `torch-scatter`. Source-building `torch-scatter` against PyTorch 2.x can take >10 min on a single CPU core; prefer the prebuilt wheel index, e.g. `pip install --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-<X.Y.Z>+cpu.html` after `pip install torch`.
- **PULSAR's default is GPU and UCE.** With `device="cuda"` on a CPU-only host you'll fail at first forward pass; set `device="cpu"` explicitly. With `layer="X_pca"` (or any non-UCE embedding) the pretrained encoder either errors on shape or produces meaningless scores — pre-compute UCE embeddings and store them in `adata.obsm["X_uce"]` first.
- **PULSAR downloads a HuggingFace model on first use.** Hundreds of MB to `~/.cache/huggingface/`. Make sure the host has internet, or pre-download with `huggingface-cli download KuanP/pulsar-pbmc`.
- **`PaSCient` is not on PyPI**: install with `pip install git+https://github.com/genentech/pascient.git@main` and the `patpy[pascient]` extra for Hydra/Lightning deps.
- **`PULSAR` is not on PyPI either**: `pip install git+https://github.com/snap-stanford/PULSAR.git@main`.
- **Train/test split is not handled by these classes.** `MixMIL`/`PaSCient` train on all donors; `PULSAR` is zero-shot, so the question doesn't apply, but its score is still optimistic if you don't separately hold out donors. For honest evaluation, hold out donors before `prepare_anndata`, or use cross-validation externally.

## Related skills

- Score predictions with calibrated F1 / Spearman → [../evaluation/SKILL.md](../evaluation/SKILL.md) (`evaluate_prediction`).
- For unsupervised representation (no donor labels) → [../sample_representation/SKILL.md](../sample_representation/SKILL.md).
