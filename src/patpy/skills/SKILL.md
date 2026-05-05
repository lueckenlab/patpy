---
name: patpy
description: Patient/sample-level representation learning from single-cell data. Use for: building a sample×sample distance matrix from an AnnData of cells, evaluating sample representations against donor metadata, and supervised donor-level prediction.
---

# patpy — agent skill index

`patpy` operates on `anndata.AnnData` objects where each row is a single cell and each donor / patient / sample contributes many cells. The package exposes three top-level submodules — `pp` (preprocessing), `tl` (tools), `pl` (plotting) — plus a `datasets` module for synthetic data and example fixtures.

## When to use which skill

| User goal | Skill |
|---|---|
| Find / download a public single-cell dataset by disease, tissue, or assay (CellxGene Discover) | [cellxgene/SKILL.md](cellxgene/SKILL.md) |
| Compute QC, filter samples / cell groups, prepare AnnData | [preprocessing/SKILL.md](preprocessing/SKILL.md) |
| Get a sample×sample distance matrix from cells (unsupervised) | [sample_representation/SKILL.md](sample_representation/SKILL.md) |
| Predict donor-level labels from per-cell features (supervised) | [supervised_methods/SKILL.md](supervised_methods/SKILL.md) |
| Score a representation against metadata (kNN, silhouette, distance test, persistence) | [evaluation/SKILL.md](evaluation/SKILL.md) |
| Volcano / heatmap plots for correlation or covariate-association results | [plotting/SKILL.md](plotting/SKILL.md) |
| Synthetic data with controlled abundance / expression perturbations | [datasets/SKILL.md](datasets/SKILL.md) |

## Mental model — the central object

Most `tl` workflows return a **square distance matrix `(n_samples, n_samples)`** indexed in the order of `method.samples`. Everything in `evaluation.md` consumes a distance matrix plus a per-sample target. Don't confuse the *cell-level* AnnData (input) with the *sample-level* AnnData (`method.to_adata()`, output).

## Cross-cutting pitfalls

- **Count vs. normalized data.** Several methods assume a specific layer (`layer="X_pca"` for `Pseudobulk` / `MixMIL`, raw counts for VAE-style methods like `MrVI`/`SCPoli`). Always check the method's `__init__` for the `layer` default. Use `patpy.pp.is_count_data(adata.X)` to verify.
- **Sample key vs. cell-group key.** `sample_key` is the donor / patient identifier in `adata.obs`; `cell_group_key` is the cell type or cluster column. They are **not** interchangeable.
- **NaN distances.** Methods that aggregate per cell-type (e.g. `GroupedPseudobulk`) can produce NaN entries if a sample is missing a cell type. Use `patpy.pp.fill_nan_distances` before downstream evaluation.
- **Optional extras.** Methods like `MrVI`, `SCPoli`, `PILOT`, `WassersteinTSNE`, `GloScope`, `PaSCient`, `MixMIL`, `PULSAR` depend on extras (`pip install patpy[mrvi]`, `[scpoli]`, etc.). Importing the symbol is cheap; instantiation may raise `ImportError` if the extra is missing — surface that to the user, do not silently substitute another method.
- **Random seed.** Most methods accept `seed=`. Use it. Several baselines (`RandomVector`, training-based methods) are otherwise non-reproducible.

## Public API surface (authoritative)

```python
import patpy
patpy.pp  # preprocessing
patpy.tl  # tools (representation, evaluation, supervised, condition comparison)
patpy.pl  # plotting
patpy.datasets  # synthetic data + COVID-19 hallmarks helper
```

Symbols *not* listed in the per-area skill files are private. Do not call functions whose name starts with `_`.
