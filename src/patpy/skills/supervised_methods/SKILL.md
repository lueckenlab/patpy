---
name: patpy.tl supervised methods
description: Donor-level supervised prediction (multi-instance learning) — MixMIL, PaSCient, PULSAR.
---

# patpy.tl — supervised donor-level prediction

These methods directly predict a donor-level label (`disease_status`, `severity`, …) from per-cell features, without producing a distance matrix as the primary output. All inherit from `SupervisedSampleMethod`.

## Symbols

```python
patpy.tl.MixMIL(sample_key, label_keys, tasks, ...)     # patpy[mixmil] extra
patpy.tl.PaSCient(sample_key, label_keys, tasks, ...)   # patpy[pascient] extra
patpy.tl.PULSAR(sample_key, label_keys, tasks, ...)     # install PULSAR from git
```

Common signature:

```python
SupervisedSampleMethod(
    sample_key: str,
    label_keys: list[str] | str,           # donor-level columns in adata.obs
    tasks: list[Literal["classification", "regression", "ranking"]] | str,
    cell_group_key: str | None = None,
    layer: str = "X_pca",                  # per-cell features (in adata.obsm or adata.layers)
    seed: int = 42,
)
```

Same protocol for all three:

```python
model = MethodClass(...)
model.prepare_anndata(adata)         # validates donor-level labels are constant per donor
# training is method-specific (often inside prepare_anndata or a fit() call)
sample_scores = model.get_sample_importance()   # per-sample prediction / score
cell_importance = model.get_cell_importance()   # per-cell attribution (for attention-based methods)
```

## When to use which

- **`MixMIL`** — attention-based MIL (Engelmann et al. 2024). Single-label classification or regression on a donor-level target. `likelihood="binomial"` for binary targets, `"categorical"` for multi-class. Fast on PCA features.
- **`PaSCient`** — multi-task donor-level model (Genentech). Heavier; supports multi-label training.
- **`PULSAR`** — graph-based donor-level model.

If the user doesn't know which to pick: start with `MixMIL` on `layer="X_pca"`, single label, `n_epochs=2000`, `seed=0`. It's the fastest and least dependency-heavy.

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
- **`PaSCient` is not on PyPI**: install with `pip install git+https://github.com/genentech/pascient.git@main` and the `patpy[pascient]` extra for Hydra/Lightning deps.
- **Train/test split is not handled by these classes.** They train on all donors. For honest evaluation, hold out donors before `prepare_anndata`, or use cross-validation externally.

## Related skills

- Score predictions with calibrated F1 / Spearman → [../evaluation/SKILL.md](../evaluation/SKILL.md) (`evaluate_prediction`).
- For unsupervised representation (no donor labels) → [../sample_representation/SKILL.md](../sample_representation/SKILL.md).
