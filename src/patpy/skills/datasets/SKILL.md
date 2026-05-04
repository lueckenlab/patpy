---
name: patpy.datasets
description: Synthetic single-cell data with controllable abundance / expression perturbations and a COVID-19 hallmarks helper.
---

# patpy.datasets

```python
patpy.datasets.simulate_data(adata, cell_type_key, layer=None,
                             abundance_perturbation=None, gene_perturbation=None,
                             perturbation_strength=1.0, expression_noise_scale=0.05,
                             dropout_rate=0.7) -> AnnData
patpy.datasets.process_adata(adata, verbose=False) -> AnnData
patpy.datasets.covid_19_hallmarks() -> dict[str, ...]
```

## When to use

- **`simulate_data`** — generate a perturbed copy of a real `AnnData` with known ground-truth shifts in cell-type abundance and/or per-gene expression. Useful for benchmarking representation methods against a known signal.
- **`process_adata`** — convenience pipeline: QC metrics → normalize_total → log1p → scale → PCA(30) → neighbors → UMAP. Same as a typical scanpy preprocessing recipe.
- **`covid_19_hallmarks`** — returns a dictionary of known COVID-19 hallmark genes/cell-types based on the COMBAT study, useful as ground-truth annotation for evaluation.

## Minimal example

```python
import patpy
import numpy as np

np.random.seed(0)

# Perturb monocytes: 2× abundance, 1.5× IFI27 expression
sim = patpy.datasets.simulate_data(
    adata,
    cell_type_key="cell_type",
    abundance_perturbation={"Monocyte": 2.0},
    gene_perturbation={"Monocyte": {"IFI27": 1.5}},
    perturbation_strength=1.0,
    expression_noise_scale=0.05,
    dropout_rate=0.7,
)
sim = patpy.datasets.process_adata(sim)
```

## Common pitfalls

- **Reproducibility.** `simulate_data` uses `np.random` directly. Set `np.random.seed(...)` *before* calling — there is no `seed=` parameter.
- **`perturbation_strength` is a [0, 1] scaler.** A value of `1.0` applies the fold change as specified in the perturbation dict. `0.0` means no perturbation. Values outside `[0, 1]` are not validated; behavior is undefined.
- **`dropout_rate=0.7` is the default.** This re-introduces 70% zeros after expression bootstrapping. If you want to keep all simulated counts, pass `dropout_rate=0`.
- **`gene_perturbation` is a nested dict** keyed `{cell_type: {gene_name: fold_change}}` — *not* flat. Easy to get wrong.
- **`process_adata` mutates** with scanpy in place but also returns the object. The PCA dim is fixed at 30; if you need a different value, run scanpy directly.
- **`covid_19_hallmarks()` is descriptive, not a dataset loader.** It returns hallmark annotations, not an AnnData.

## Related skills

- Feed the simulated data to a method → [../sample_representation/SKILL.md](../sample_representation/SKILL.md) or [../supervised_methods/SKILL.md](../supervised_methods/SKILL.md).
- Score the recovery of the planted signal → [../evaluation/SKILL.md](../evaluation/SKILL.md).
