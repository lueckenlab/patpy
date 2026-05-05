---
name: patpy.tl sample representation
description: Unsupervised methods that turn a cell-level AnnData into a sample×sample distance matrix and an optional sample embedding.
---

# patpy.tl — sample representation methods

All methods inherit from `SampleRepresentationMethod` and follow the same protocol:

```python
method = MethodClass(sample_key=..., cell_group_key=..., layer=..., seed=...)
method.prepare_anndata(adata)
distances = method.calculate_distance_matrix()       # (n_samples, n_samples), order = method.samples
sample_adata = method.to_adata(metadata=meta_df)     # sample-level AnnData with .obsm embeddings
```

> **`cell_group_key` is positional, not optional.** Most classes here (`Pseudobulk`, `GroupedPseudobulk`, `CellGroupComposition`, …) require both `sample_key` and `cell_group_key` to be passed even though `cell_group_key=None` is meaningful (= "no grouping; one pseudobulk per sample"). Calling `Pseudobulk(sample_key="donor")` raises `TypeError: missing 1 required positional argument: 'cell_group_key'`. Pass `cell_group_key=None` explicitly when you don't want grouping.

## Method catalog

| Class | What it does | `layer` default | Required extra |
|---|---|---|---|
| `Pseudobulk` | Mean per-sample expression vector → pdist | `"X_pca"` | — |
| `GroupedPseudobulk` | Per-(sample, cell-group) pseudobulk; distance = avg over groups | `"X_pca"` | — |
| `CellGroupComposition` | Cell-type fraction vector (optional CLR) → pdist | `None` | — |
| `RandomVector` | Random Gaussian embedding (negative-control baseline) | n/a | — |
| `MrVI` | scVI-tools sample-aware VAE | raw counts | `patpy[mrvi]` |
| `SCPoli` | scArches conditional VAE | raw counts | `patpy[scpoli]` |
| `PILOT` | PILOT optimal-transport based | n/a | `patpy[pilot]` |
| `WassersteinTSNE` | Wasserstein t-SNE over per-sample distributions | `"X_pca"` | `patpy[wassersteintsne]` |
| `GloScope`, `GloScope_py` | Density-based (R / Python) | n/a | `patpy[gloscope-py-cpu]` |
| `DiffusionEarthMoverDistance` | Diffusion EMD | n/a | `patpy[diffusionemd]` |
| `MOFA` | Multi-Omics Factor Analysis | varies | varies |

## Choosing a method

- **Always include** `Pseudobulk` and `RandomVector` as baselines. If a fancy method does not beat `Pseudobulk` on the chosen evaluation, the fancy method is not adding signal on this dataset.
- **`CellGroupComposition`** is the right choice when the biology of interest is cell-type *abundance* shifts, not expression shifts.
- **`GroupedPseudobulk`** captures expression shifts conditioned on cell-type identity. Requires every sample to have every cell group; combine with `pp.filter_small_cell_groups`.
- **`MrVI` / `SCPoli`** require raw counts and are slow to train. Use them when sample-level batch effects are large and need to be modeled jointly with biology.

## Minimal example

```python
import patpy

# adata has obs["donor_id"], obs["cell_type"], obsm["X_pca"]
method = patpy.tl.Pseudobulk(
    sample_key="donor_id",
    cell_group_key="cell_type",   # required; pass None for un-grouped pseudobulk
    layer="X_pca",
    seed=0,
)
method.prepare_anndata(adata)
D = method.calculate_distance_matrix()           # ndarray, shape = (n_samples, n_samples)
samples = method.samples                          # row/col order of D

# Score the representation against a metadata column
meta = patpy.pp.extract_metadata(adata, sample_key="donor_id", columns=["disease"])
result = method.evaluate_representation(target="disease", method="knn", metadata=meta, n_neighbors=3, task="classification")
# result["score"] is calibrated F1 in [0, 1]; 0 = random, 1 = perfect.
```

## Common pitfalls

- **Order matters.** `D[i, j]` is the distance between `method.samples[i]` and `method.samples[j]`. Never assume the order matches `adata.obs[sample_key].unique()` — go through `method.samples`.
- **`force=True`.** Distances are cached in `adata.uns[method.DISTANCES_UNS_KEY]`. If you change parameters and re-call without `force=True`, you'll get the stale cached matrix. Symptoms: results don't change when you change `aggregate=` or `dist=`.
- **Layer mismatch.** `Pseudobulk` defaults to `layer="X_pca"`. If your AnnData has no `obsm["X_pca"]`, this raises. Either run PCA first (`scanpy.tl.pca`) or pass `layer=None` to use `.X`.
- **`MrVI`, `SCPoli`** need raw counts. Passing log-normalized data silently produces bad embeddings.
- **`GroupedPseudobulk` + missing cell types** → NaN entries in `D`. Run `patpy.pp.fill_nan_distances(D)` or filter with `pp.filter_small_cell_groups` upstream.
- **`RandomVector` is non-deterministic across calls** unless you fix `np.random.seed` (the constructor accepts `seed=` but the implementation calls `np.random.normal` directly). For reproducible baselines, set `np.random.seed(seed)` before `calculate_distance_matrix`.

## Related skills

- Score the resulting `D` → [../evaluation/SKILL.md](../evaluation/SKILL.md).
- Predict donor labels directly from cells (no distance matrix needed) → [../supervised_methods/SKILL.md](../supervised_methods/SKILL.md).
- Filter / prepare the input AnnData → [../preprocessing/SKILL.md](../preprocessing/SKILL.md).
