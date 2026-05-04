---
name: patpy.pp
description: Preprocessing for sample-level analysis — QC, sample/cell-group filtering, compositional metrics, metadata extraction, count-data checks.
---

# patpy.pp — preprocessing

Operate on a cell-level `AnnData`. Output is either a filtered `AnnData` or a sample-indexed `pd.DataFrame`.

## Symbols

```python
patpy.pp.calculate_cell_qc_metrics(adata, sample_key, cell_qc_vars, agg_function=np.median) -> pd.DataFrame
patpy.pp.calculate_compositional_metrics(adata, sample_key, composition_keys, normalize_to=100) -> pd.DataFrame
patpy.pp.calculate_n_cells_per_sample(adata, sample_key) -> pd.DataFrame  # column: "n_cells"
patpy.pp.extract_metadata(adata, sample_key, columns, samples=None) -> pd.DataFrame  # sample-indexed
patpy.pp.filter_small_samples(adata, sample_key, sample_size_threshold=300) -> AnnData
patpy.pp.filter_small_cell_groups(adata, sample_key, cell_group_key, cluster_size_threshold=5) -> AnnData
patpy.pp.subsample(adata, obs_category_col, min_samples_per_category, fraction=None, n_obs=None) -> AnnData
patpy.pp.is_count_data(matrix, window_size=10000) -> bool  # checks the top-left window only
patpy.pp.fill_nan_distances(distances, n_max_distances=5) -> np.ndarray
patpy.pp.prepare_data_for_phemd(adata, sample_col, n_top_var_genes=100)
patpy.pp.convert_cell_types_to_phemd_format(adata, cell_type_col, sample_col, output_dir, ...)
patpy.pp.get_helical_embedding(...)  # requires patpy[helical] extra
```

## When to use

- Before any `tl` representation method: filter out samples / cell groups too small to support the method (especially `GroupedPseudobulk`, `WassersteinTSNE`, `GloScope`).
- To build sample-level tables for downstream regression / plotting (`extract_metadata`, `calculate_n_cells_per_sample`, `calculate_compositional_metrics`).
- To check that you're feeding the right matrix to a count-based method (`is_count_data`).

## Minimal example

```python
import patpy

adata = adata[adata.obs["pct_counts_mt"] < 20].copy()
adata = patpy.pp.filter_small_samples(adata, sample_key="donor_id", sample_size_threshold=200)
adata = patpy.pp.filter_small_cell_groups(adata, sample_key="donor_id", cell_group_key="cell_type", cluster_size_threshold=10)

qc = patpy.pp.calculate_cell_qc_metrics(adata, sample_key="donor_id", cell_qc_vars=["n_genes_by_counts", "total_counts"])
comp = patpy.pp.calculate_compositional_metrics(adata, sample_key="donor_id", composition_keys=["cell_type"])
meta = patpy.pp.extract_metadata(adata, sample_key="donor_id", columns=["disease", "age"])
```

## Common pitfalls

- `filter_small_samples` and `filter_small_cell_groups` print the number of removed entries and **return a new copy**. Reassign or you'll silently keep the unfiltered object.
- `calculate_compositional_metrics` returns *fractions × normalize_to* (default 100, i.e. percentages — not 0–1 proportions).
- `is_count_data` only inspects the `[:window_size, :window_size]` window. For very small AnnDatas this is the whole matrix, but for huge sparse matrices a leading dense block of integers can mask non-integer entries elsewhere. Treat the result as a heuristic.
- `extract_metadata` warns and de-duplicates if a sample has multiple distinct values for a column. Make sure metadata is genuinely sample-level before calling it.
- `subsample` requires exactly one of `fraction` or `n_obs`. Setting both, or neither, raises.

## Related skills

- After preprocessing → [../sample_representation/SKILL.md](../sample_representation/SKILL.md) or [../supervised_methods/SKILL.md](../supervised_methods/SKILL.md).
- Compositional / QC tables feed into [../evaluation/SKILL.md](../evaluation/SKILL.md) (e.g. as `target` for `evaluate_representation`).
