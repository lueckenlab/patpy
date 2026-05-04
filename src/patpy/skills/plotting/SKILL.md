---
name: patpy.pl
description: Plotting helpers for correlation volcanos and embedding–covariate association heatmaps.
---

# patpy.pl — plotting

Two functions, both consume DataFrames produced elsewhere in `patpy.tl` (or by user code with the matching column schema).

## Symbols

```python
patpy.pl.correlation_volcano(correlation_df, x="correlation", y="-log_p_value_adj",
                             color_by="cell_type", top_n=10, figsize=(12, 8),
                             x_jitter_strength=0, y_jitter_strength=2) -> (Figure, Axes)
patpy.pl.embedding_covariate_heatmap(assoc_df, *, covariate_col="covariate", pc_col="PC",
                                     value_col="-log10p", p_thresh=0.05, cmap="Reds",
                                     figsize=None, title="...", return_fig=False) -> Figure | None
```

## When to use

- **`correlation_volcano`** — plot per-gene correlation vs. (signed) significance from `patpy.tl.correlate_cell_type_expression` / `correlate_composition`. Expects columns `correlation`, `-log_p_value_adj`, `gene_name`, plus the `color_by` column.
- **`embedding_covariate_heatmap`** — companion plot for `patpy.tl.associate_embedding_with_covariates`. Expects `covariate`, `PC`, `-log10p`, optionally `p_value`. Annotates cells with `***` / `**` / `*` / `·` significance stars.

## Minimal example

```python
import patpy

# correlation_volcano
corr_df = patpy.tl.correlate_cell_type_expression(meta_adata=..., expression_adata=..., target=..., ...)
fig, ax = patpy.pl.correlation_volcano(corr_df, top_n=15)

# embedding_covariate_heatmap
assoc = patpy.tl.associate_embedding_with_covariates(pdata, ["Source", "Sex"], obsm_key="X_pca")
fig = patpy.pl.embedding_covariate_heatmap(assoc, return_fig=True)
```

## Common pitfalls

- **Column-name contracts are strict.** `correlation_volcano` will `KeyError` if the input doesn't have the expected `correlation` / `-log_p_value_adj` / `gene_name` columns. If you computed correlations yourself, rename columns to match before calling.
- **`embedding_covariate_heatmap` infers PC ordering from the column prefix.** It strips digits from the first label and sorts numerically. If your component labels don't follow `<prefix><int>` (e.g. `PC1`, `PC2`), the sort falls back to lexicographic — check the resulting axis before publishing.
- **`return_fig=False` calls `plt.show()`** and returns `None`. Pass `return_fig=True` to get the `Figure` for further customization or saving.
- **`correlation_volcano` jitters labels stochastically.** Set `np.random.seed(...)` before calling if you need reproducible label positions.

## Related skills

- Generate the inputs to these plots → [../evaluation/SKILL.md](../evaluation/SKILL.md) (correlation / association functions live in `patpy.tl`).
