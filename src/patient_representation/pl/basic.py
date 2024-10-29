import matplotlib.pyplot as plt
import numpy as np


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
