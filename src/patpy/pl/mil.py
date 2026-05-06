"""Attention-weight visualizations for MIL models.

Functions
---------
plot_attention_umap
    Scatter UMAP coloured by per-cell attention weight, one panel per class.
plot_attention_by_cell_type
    Violin / dot plot of attention weights aggregated by cell type.
"""
from __future__ import annotations

import inspect
import logging
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)


def _get_cell_importance(model, label: str, normalized: bool) -> pd.DataFrame:
    """Call model.get_cell_importance(), handling models that lack `normalized`."""
    sig = inspect.signature(model.get_cell_importance)
    if "normalized" in sig.parameters:
        return model.get_cell_importance(label=label, normalized=normalized)
    # Model (e.g. MixMIL) always returns softmax-normalised weights
    if not normalized:
        warnings.warn(
            f"{type(model).__name__}.get_cell_importance() does not support "
            "raw pre-softmax scores — returning softmax-normalised weights instead.",
            stacklevel=3,
        )
    return model.get_cell_importance(label=label)


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
    normalized
        Passed to :meth:`~patpy.tl.mil_models.TorchMILWrapper.get_cell_importance`.
        ``False`` (default) uses raw pre-softmax scores; ``True`` uses
        post-softmax weights that sum to 1 per bag.

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

    att_df = _get_cell_importance(model, label=label, normalized=normalized)
    att_col = f"{label}_importance"
    weight_label = "Attention weight (softmax)" if normalized else "Attention score (pre-softmax)"

    # Merge attention weights into a working frame
    frame = pd.DataFrame(
        {
            "umap_0": adata.obsm[umap_key][:, 0],
            "umap_1": adata.obsm[umap_key][:, 1],
            "cell_type": adata.obs[cell_type_key].values,
            sample_key: adata.obs[sample_key].values,
            label: adata.obs[label].values,
        },
        index=adata.obs_names,
    )
    # Attach attention weights (NaN for cells not covered by the model)
    frame[att_col] = att_df[att_col] if att_col in att_df.columns else np.nan

    classes = sorted(frame[label].dropna().unique())
    n_rows = int(np.ceil(len(classes) / n_cols))
    fig_w = figsize_per_panel[0] * n_cols
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    vmin = frame[att_col].quantile(0.02)
    vmax = frame[att_col].quantile(0.98)

    for idx, cls in enumerate(classes):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]

        # Background: all cells in grey
        ax.scatter(
            frame["umap_0"], frame["umap_1"],
            c="lightgrey", s=size * 0.5, alpha=0.3, rasterized=True
        )

        # Foreground: cells from donors of this class, coloured by attention
        donor_ids_in_class = (
            frame.groupby(sample_key)[label].first()
            .pipe(lambda s: s[s == cls].index)
        )
        mask = frame[sample_key].isin(donor_ids_in_class) & frame[att_col].notna()
        sub = frame[mask]

        sc_plot = ax.scatter(
            sub["umap_0"], sub["umap_1"],
            c=sub[att_col],
            cmap=palette,
            vmin=vmin, vmax=vmax,
            s=size, alpha=alpha,
            rasterized=True,
        )
        plt.colorbar(sc_plot, ax=ax, label=weight_label, fraction=0.04)
        title = f"{title_prefix}{label} = {cls}"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=8)

    # Hide unused panels
    for idx in range(len(classes), n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Attention weights — {label}", fontsize=13, y=1.01)
    fig.tight_layout()
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
    """Violin/dot plot of attention weights aggregated by cell type, per class.

    Shows how attention is distributed across cell types in each class of
    the prediction label.  Cell types are ranked by mean attention weight in
    the first class and the top ``n_top_cell_types`` are displayed.

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
    n_top_cell_types
        How many cell types to show (ranked by mean attention).
    figsize
        Figure size.  Defaults to ``(10, 0.5 * n_top_cell_types)``.
    palette
        Colour palette passed to seaborn.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sample_key = sample_key or model.sample_key

    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key='{cell_type_key}' not found in adata.obs.")
    if label not in adata.obs.columns:
        raise ValueError(f"label='{label}' not found in adata.obs.")

    att_df = _get_cell_importance(model, label=label, normalized=normalized)
    att_col = f"{label}_importance"
    weight_label = "Attention weight (softmax)" if normalized else "Attention score (pre-softmax)"

    frame = pd.DataFrame(
        {
            "cell_type": adata.obs[cell_type_key].values,
            sample_key: adata.obs[sample_key].values,
            label: adata.obs[label].values,
            att_col: att_df[att_col].values,
        },
        index=adata.obs_names,
    )

    top_ct = (
        frame.groupby("cell_type")[att_col].mean()
        .nlargest(n_top_cell_types)
        .index.tolist()
    )
    frame = frame[frame["cell_type"].isin(top_ct)]

    fh = figsize or (10, max(4, 0.5 * n_top_cell_types))
    fig, ax = plt.subplots(figsize=fh)

    sns.violinplot(
        data=frame,
        y="cell_type",
        x=att_col,
        hue=label,
        orient="h",
        scale="width",
        inner="quartile",
        palette=palette or "Set2",
        order=top_ct,
        ax=ax,
        cut=0,
        density_norm="width",
    )
    ax.set_xlabel(weight_label, fontsize=10)
    ax.set_ylabel("Cell type", fontsize=10)
    ax.set_title(f"Attention by cell type — {label}", fontsize=12)
    ax.tick_params(labelsize=9)
    ax.legend(title=label, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
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

    att_df = _get_cell_importance(model, label=label, normalized=normalized)
    att_col = f"{label}_importance"
    weight_label = "Mean attention (softmax)" if normalized else "Mean attention score (pre-softmax)"

    frame = pd.DataFrame(
        {
            "cell_type": adata.obs[cell_type_key].values,
            sample_key: adata.obs[sample_key].values,
            label: adata.obs[label].values,
            att_col: att_df[att_col].values,
        },
        index=adata.obs_names,
    )

    pivot = (
        frame.groupby(["cell_type", label])[att_col]
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
