from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def _store_importance(adata: sc.AnnData, model, label: str, normalized: bool) -> str:
    """Write per-cell importance scores into ``adata.obs`` and return the column name."""
    att_col = f"{label}_importance"
    imp_df = model.get_cell_importance(label=label, normalized=normalized)
    # imp_df is indexed by model.adata.obs_names; reindex to align with passed adata
    adata.obs[att_col] = imp_df[att_col].reindex(adata.obs_names)
    return att_col


def plot_attention_umap(
    adata: sc.AnnData,
    model,
    label: str,
    *,
    cell_type_key: str,
    umap_key: str = "X_umap",
    sample_key: str | None = None,
    normalized: bool = False,
    n_cols: int = 2,
    figsize_per_panel: tuple[float, float] = (5.0, 4.5),
    palette: str = "magma",
    size: float = 2.0,
    alpha: float = 0.6,
    title_prefix: str = "",
):
    """UMAP coloured by attention weight, split by class.

    For each unique class value of *label*, only cells from donors belonging
    to that class are shown; the colour encodes the attention weight learned
    by *model* for each cell.

    Parameters
    ----------
    adata
        AnnData containing cells.  Must have ``umap_key`` in ``.obsm`` and
        ``cell_type_key`` in ``.obs``.
    model
        A fitted :class:`~patpy.tl.supervised.SupervisedSampleMethod` that
        implements :meth:`get_cell_importance`.
    label
        Which label's attention weights to visualise.
    cell_type_key
        Column in ``adata.obs`` with cell-type annotations.
    umap_key
        Key in ``adata.obsm`` containing UMAP coordinates (shape ``(n, 2)``).
    sample_key
        Column with donor IDs.  Defaults to ``model.sample_key``.
    normalized
        Passed to :meth:`get_cell_importance`.
        ``False`` (default) uses raw pre-softmax scores; ``True`` uses
        post-softmax weights that sum to 1 per bag.
    n_cols
        Number of columns in the subplot grid.
    figsize_per_panel
        Width and height of each class panel.
    palette
        Matplotlib colormap name for the attention weight.
    size
        Scatter point size.
    alpha
        Scatter point transparency.
    title_prefix
        Optional prefix added to each panel title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    sample_key = sample_key or model.sample_key

    if umap_key not in adata.obsm:
        raise ValueError(f"umap_key='{umap_key}' not found in adata.obsm.")
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key='{cell_type_key}' not found in adata.obs.")
    if sample_key not in adata.obs.columns:
        raise ValueError(f"sample_key='{sample_key}' not found in adata.obs.")
    if label not in adata.obs.columns:
        raise ValueError(f"label='{label}' not found in adata.obs.")

    att_col = _store_importance(adata, model, label, normalized)

    # sc.pl.umap requires the embedding to be at obsm["X_umap"]; alias if needed
    _orig_umap = adata.obsm.get("X_umap")
    if umap_key != "X_umap":
        adata.obsm["X_umap"] = adata.obsm[umap_key]

    try:
        classes = sorted(adata.obs[label].dropna().unique())
        n_rows = int(np.ceil(len(classes) / n_cols))
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
            squeeze=False,
        )

        for idx, cls in enumerate(classes):
            donor_ids = (
                adata.obs.groupby(sample_key)[label].first()
                .pipe(lambda s: s[s == cls].index)
            )
            adata_cls = adata[adata.obs[sample_key].isin(donor_ids)].copy()

            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            sc.pl.umap(
                adata_cls, color=att_col, ax=ax,
                color_map=palette, size=size, alpha=alpha,
                title=f"{title_prefix}{label} = {cls}",
                colorbar_loc="right", show=False,
            )

        for idx in range(len(classes), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(f"Attention weights — {label}", fontsize=13, y=1.01)
        fig.tight_layout()
    finally:
        # Restore obsm to its original state
        if umap_key != "X_umap":
            if _orig_umap is not None:
                adata.obsm["X_umap"] = _orig_umap
            elif "X_umap" in adata.obsm:
                del adata.obsm["X_umap"]

    return fig


def plot_attention_by_cell_type(
    adata: sc.AnnData,
    model,
    label: str,
    *,
    cell_type_key: str,
    sample_key: str | None = None,
    normalized: bool = False,
    n_top_cell_types: int = 20,
    figsize: tuple[float, float] | None = None,
    palette: str | list | None = None,
):
    """Violin plot of attention weights by cell type.

    Shows how attention is distributed across cell types.  Cell types are
    ranked by mean attention weight and the top ``n_top_cell_types`` are shown.

    Parameters
    ----------
    adata
        AnnData with ``cell_type_key`` in ``.obs``.
    model
        Fitted model with :meth:`get_cell_importance`.
    label
        Label whose attention weights to visualise.
    cell_type_key
        Column in ``adata.obs`` with cell-type annotations.
    sample_key
        Column with donor IDs.
    normalized
        Passed to :meth:`get_cell_importance`.
    n_top_cell_types
        How many cell types to show (ranked by mean attention).
    figsize
        Figure size.  Defaults to ``(10, 0.5 * n_top_cell_types)``.
    palette
        Colour palette passed to :func:`scanpy.pl.violin`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    sample_key = sample_key or model.sample_key

    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key='{cell_type_key}' not found in adata.obs.")
    if label not in adata.obs.columns:
        raise ValueError(f"label='{label}' not found in adata.obs.")

    att_col = _store_importance(adata, model, label, normalized)
    weight_label = "Attention weight (softmax)" if normalized else "Attention score (pre-softmax)"

    top_ct = (
        adata.obs.groupby(cell_type_key)[att_col].mean()
        .nlargest(n_top_cell_types)
        .index.tolist()
    )
    adata_top = adata[adata.obs[cell_type_key].isin(top_ct)].copy()
    # Reorder cell_type_key as a categorical so sc.pl.violin respects the ranking
    adata_top.obs[cell_type_key] = pd.Categorical(
        adata_top.obs[cell_type_key], categories=top_ct, ordered=True
    )

    fh = figsize or (10, max(4, 0.5 * n_top_cell_types))
    fig, ax = plt.subplots(figsize=fh)

    sc.pl.violin(
        adata_top,
        keys=[att_col],
        groupby=cell_type_key,
        rotation=90,
        palette=palette or "Set2",
        ax=ax,
        show=False,
    )
    ax.set_xlabel("Cell type", fontsize=10)
    ax.set_ylabel(weight_label, fontsize=10)
    ax.set_title(f"Attention by cell type — {label}", fontsize=12)
    fig.tight_layout()
    return fig


def plot_attention_celltype_heatmap(
    adata: sc.AnnData,
    model,
    label: str,
    *,
    cell_type_key: str,
    sample_key: str | None = None,
    normalized: bool = False,
    n_top_cell_types: int = 20,
    figsize: tuple[float, float] | None = None,
    cmap: str = "RdYlBu_r",
):
    """Heatmap of mean attention weight per (cell type × class).

    Parameters
    ----------
    adata, model, label, cell_type_key, sample_key
        See :func:`plot_attention_by_cell_type`.
    n_top_cell_types
        Number of cell types shown (ranked by variance across classes).
    figsize
        Figure size.
    cmap
        Matplotlib colormap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sample_key = sample_key or model.sample_key

    att_col = _store_importance(adata, model, label, normalized)
    weight_label = "Mean attention (softmax)" if normalized else "Mean attention score (pre-softmax)"

    pivot = (
        adata.obs.groupby([cell_type_key, label])[att_col]
        .mean()
        .unstack(label)
    )

    top_ct = pivot.var(axis=1).nlargest(n_top_cell_types).index
    pivot = pivot.loc[top_ct]

    fh = figsize or (max(5, 1.5 * pivot.shape[1]), max(6, 0.35 * n_top_cell_types))
    fig, ax = plt.subplots(figsize=fh)
    sns.heatmap(
        pivot,
        cmap=cmap,
        ax=ax,
        linewidths=0.3,
        cbar_kws={"label": weight_label},
    )
    ax.set_title(f"Mean attention weight per cell type — {label}", fontsize=12)
    ax.set_ylabel("Cell type", fontsize=10)
    ax.set_xlabel(label, fontsize=10)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig
