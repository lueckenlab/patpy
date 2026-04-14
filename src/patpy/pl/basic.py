from __future__ import annotations
 
from typing import TYPE_CHECKING
 
import matplotlib.pyplot as plt
import numpy as np
 
if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
 

def correlation_volcano(
    correlation_df,
    x="correlation",
    y="-log_p_value_adj",
    color_by="cell_type",
    top_n=10,
    figsize=(12, 8),
    x_jitter_strength=0,
    y_jitter_strength=2,
):
    """
    Create a volcano plot from a correlation dataframe.

    Parameters
    ----------
    correlation_df : pandas.DataFrame
        DataFrame containing correlation data.
    x : str, optional
        Column name for x-axis. Default is 'correlation'.
    y : str, optional
        Column name for y-axis. Default is '-log_p_value_adj'.
    color_by : str, optional
        Column name to color points by. Default is 'cell_type'.
    top_n : int, optional
        Number of top genes to label. Default is 10.
    figsize : tuple of int, optional
        Figure size (width, height) in inches. Default is (12, 8).
    x_jitter_strength : float, optional
        Strength of jitter in x direction for gene labels. Default is 0 (no X jitter)
    y_jitter_strength : float, optional
        Strength of jitter in y direction for gene labels. Default is 0.1.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create scatter plot
    scatter = ax.scatter(
        correlation_df[x],
        correlation_df[y],
        c=correlation_df[color_by].astype("category").cat.codes,
        alpha=0.6,
        cmap="tab20",
    )

    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_by)
    cbar.set_ticks(range(len(correlation_df[color_by].unique())))
    cbar.set_ticklabels(correlation_df[color_by].unique())

    # Set labels and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Volcano Plot of Gene Correlations")

    # Add gene labels for top n genes with jitter and lines
    top_genes = correlation_df.nlargest(top_n, y)
    for _, row in top_genes.iterrows():
        x_jitter = row[x] + np.random.normal(0, x_jitter_strength)
        y_jitter = row[y] + np.random.normal(0, y_jitter_strength)
        ax.annotate(
            row["gene_name"],
            (x_jitter, y_jitter),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=8,
            # arrowprops=dict(arrowstyle="-", color="darkgrey")
        )
        ax.plot([row[x], x_jitter], [row[y], y_jitter], color="darkgrey", linewidth=0.5)

    # Add a horizontal line at significance level
    ax.axhline(y=-np.log(0.05), color="r", linestyle="--", linewidth=1)

    _, ymax = plt.ylim()
    plt.ylim(0, ymax)

    plt.tight_layout()
    return fig, ax

 
def embedding_covariate_heatmap(
    assoc_df: pd.DataFrame,
    *,
    covariate_col: str = "covariate",
    pc_col: str = "PC",
    value_col: str = "-log10p",
    p_thresh: float = 0.05,
    cmap: str = "Reds",
    figsize: tuple[int, int] | None = None,
    title: str = "Embedding association with covariates (-log10 p)",
    return_fig: bool = False,
) -> Figure | None:
    """Heatmap of embedding–covariate association statistics.

    Companion plot for :func:`patpy.tl.associate_embedding_with_covariates`. Each
    cell shows the ``-log10(p-value)`` from a one-way ANOVA or Kruskal-Wallis
    test of that component against that covariate. Cells are annotated with
    significance stars (``***`` p < 0.001, ``**`` p < 0.01, ``*`` p < 0.05,
    ``.`` p < 0.10).

    Parameters
    ----------
    assoc_df : pd.DataFrame
        Output of :func:`patpy.tl.associate_embedding_with_covariates`.
    covariate_col : str, default ``"covariate"``
        Column in ``assoc_df`` identifying the covariate (y-axis).
    pc_col : str, default ``"PC"``
        Column in ``assoc_df`` identifying the embedding component (x-axis).
    value_col : str, default ``"-log10p"``
        Column to use as heatmap values.
    p_thresh : float, default ``0.05``
        Significance threshold used to derive star annotations. Cells with
        p-value below this threshold are annotated.
    cmap : str, default ``"Reds"``
        Matplotlib/seaborn colormap name.
    figsize : tuple[int, int], optional
        Figure size. Defaults to ``(n_components + 2, n_covariates + 1)``.
    title : str
        Plot title.
    return_fig : bool, default ``False``
        If ``True``, return the figure instead of calling ``plt.show()``.

    Returns
    -------
    Figure | None
        The figure if ``return_fig=True``, otherwise ``None``.

    Examples
    --------
    >>> assoc = patpy.tl.associate_embedding_with_covariates(
    ...     pdata, ["Source", "Sex"], obsm_key="X_pca"
    ... )
    >>> patpy.pl.embedding_covariate_heatmap(assoc)
    """
    import seaborn as sns

    pivot = assoc_df.pivot(index=covariate_col, columns=pc_col, values=value_col)

    # Sort columns numerically (PC1, PC2, … PC10, not PC1, PC10, PC2)
    try:
        col_label = pivot.columns[0]
        prefix = "".join(c for c in col_label if not c.isdigit())
        pc_order = sorted(pivot.columns, key=lambda s: int(s.replace(prefix, "")))
    except (ValueError, IndexError):
        pc_order = list(pivot.columns)
    pivot = pivot[pc_order]

    # Build matching p-value pivot for star annotations
    if "p_value" in assoc_df.columns:
        p_pivot = assoc_df.pivot(
            index=covariate_col, columns=pc_col, values="p_value"
        )[pc_order]
    else:
        # Reconstruct p-values from -log10p
        p_pivot = 10 ** (-pivot)

    def _stars(p: float) -> str:
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.10:
            return "."
        return ""

    annot = p_pivot.applymap(_stars)

    n_covariates, n_pcs = pivot.shape
    if figsize is None:
        figsize = (max(6, n_pcs + 2), max(2, n_covariates + 1))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        pivot,
        annot=annot,
        fmt="",
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": value_col},
        linewidths=0.5,
        linecolor="lightgrey",
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Star legend as figure subtitle
    fig.text(
        0.5, -0.02,
        "Significance: *** p<0.001   ** p<0.01   * p<0.05   . p<0.10",
        ha="center",
        va="top",
        fontsize=9,
        color="grey",
        style="italic",
    )

    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()
    return None