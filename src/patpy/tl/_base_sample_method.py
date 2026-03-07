from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from patpy.pp import extract_metadata, fill_nan_distances


def _create_colormap(df: pd.DataFrame, col: str, palette: str = "Spectral") -> pd.Series:
    """Map unique values of *col* to colours from *palette*."""
    unique_values = df[col].unique()
    colors = sns.color_palette(palette, n_colors=len(unique_values))
    color_map = dict(zip(unique_values, colors, strict=False))
    return df[col].map(color_map)


class BaseSampleMethod:
    """Base class for SampleRepresentationMethod and SupervisedSampleMethod class.

    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` containing sample (donor) identifiers.
    cell_group_key : str or None
        Column in ``adata.obs`` containing cell-type / cell-group labels.
        May be ``None`` when grouping is not required.
    layer : str or None, default ``None``
        Feature source.  ``None`` or ``"X"`` → ``adata.X``;
        any other string → first checked in ``adata.obsm``, then
        ``adata.layers``.
    seed : int, default 67
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_key: str,
        cell_group_key: str | None,
        layer: str | None = None,
        seed: int = 67,
    ) -> None:
        self.sample_key = sample_key
        self.cell_group_key = cell_group_key
        self.layer = layer
        self.seed = seed

        self.adata: sc.AnnData | None = None
        self.samples: np.ndarray | None = None
        self.cell_groups: np.ndarray | None = None
        self.embeddings: dict[str, np.ndarray] = {}

    def prepare_anndata(self, adata: sc.AnnData) -> None:
        """Store *adata* and populate :attr:`samples` / :attr:`cell_groups`.

        Subclasses must call ``super().prepare_anndata(adata)`` first,
        then perform method-specific initialisation (model training, etc.).

        Parameters
        ----------
        adata
            Single-cell AnnData.  Must contain :attr:`sample_key` in ``.obs``.
        """
        if self.sample_key not in adata.obs.columns:
            raise ValueError(f"sample_key='{self.sample_key}' not found in adata.obs.")

        self.adata = adata
        self.samples = adata.obs[self.sample_key].unique()

        if self.cell_group_key is not None and self.cell_group_key in adata.obs.columns:
            self.cell_groups = adata.obs[self.cell_group_key].unique()

    def _get_data(self) -> np.ndarray:
        """Return the feature matrix from the slot specified by :attr:`layer`."""
        self._check_fitted()

        if self.layer is None or self.layer == "X":
            warnings.warn("Using data from adata.X", stacklevel=2)
            return self.adata.X

        if self.layer in self.adata.obsm:
            warnings.warn(f"Using data from adata.obsm['{self.layer}']", stacklevel=2)
            return self.adata.obsm[self.layer]

        if self.layer in self.adata.layers:
            warnings.warn(f"Using data from adata.layers['{self.layer}']", stacklevel=2)
            return self.adata.layers[self.layer]

        raise ValueError(
            f"layer='{self.layer}' not found in adata.obsm or adata.layers. Please make sure it is specified correctly."
        )

    def _move_layer_to_X(self) -> sc.AnnData:
        """Return a copy of :attr:`adata` with :attr:`layer` moved to ``.X``.

        Some models require features in ``adata.X``.  This helper avoids
        mutating the user's AnnData in place.
        """
        if self.layer in ("X", None):
            # The data is already in correct slot
            return self.adata

        # getting only those layers with the same shape of the new X matrix from adata.layers[self.layer] to be copied in the new anndata below
        filtered_layers = {
            key: np.copy(layer)
            for key, layer in self.adata.layers.items()
            if key != self.layer and layer.shape == self.adata.layers.get(self.layer, np.empty(0)).shape
        }
        # Copy everything except from .var* to new adata, with correct layer in X
        new_adata = sc.AnnData(
            X=self._get_data(),
            obs=self.adata.obs,
            obsm=self.adata.obsm,
            layers=filtered_layers,
            uns=self.adata.uns,
            obsp=self.adata.obsp,
        )
        new_adata.obsm["X_old"] = self.adata.X
        return new_adata

    def _extract_metadata(self, columns: list[str]) -> pd.DataFrame:
        """Return a DataFrame with *columns* aligned to :attr:`samples`."""
        return extract_metadata(self.adata, self.sample_key, columns, samples=self.samples)

    def _check_fitted(self) -> None:
        """Raise :class:`RuntimeError` if :meth:`prepare_anndata` has not been called."""
        if self.adata is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call prepare_anndata() first.")

    def embed(
        self,
        distances: np.ndarray,
        method: str = "UMAP",
        n_jobs: int = -1,
        verbose: bool = False,
    ) -> np.ndarray:
        """Embed *distances* into 2-D coordinates.

        Parameters
        ----------
        distances
            Square distance matrix of shape ``(n_samples, n_samples)``.
        method
            One of ``"MDS"``, ``"TSNE"``, ``"UMAP"``.
        n_jobs
            Number of parallel threads (``-1`` = all).
        verbose
            Print progress information.

        Returns
        -------
        coordinates : np.ndarray
            Array of shape ``(n_samples, 2)``.
        """
        distances = fill_nan_distances(distances)

        if method == "MDS":
            from sklearn.manifold import MDS

            coords = MDS(
                n_components=2,
                dissimilarity="precomputed",
                verbose=verbose,
                n_jobs=n_jobs,
                random_state=self.seed,
            ).fit_transform(distances)

        elif method == "TSNE":
            from openTSNE import TSNE

            coords = TSNE(
                n_components=2,
                metric="precomputed",
                neighbors="exact",
                n_jobs=n_jobs,
                random_state=self.seed,
                verbose=verbose,
                initialization="spectral",
            ).fit(distances)

        elif method == "UMAP":
            from umap import UMAP

            coords = UMAP(
                n_components=2,
                metric="precomputed",
                random_state=self.seed,
                verbose=verbose,
                n_jobs=n_jobs,
            ).fit_transform(distances)

        else:
            raise ValueError(f"Method '{method}' is not supported. Choose one of ['MDS', 'TSNE', 'UMAP'].")

        self.embeddings[method] = coords
        return coords

    def plot_clustermap(
        self,
        distances: np.ndarray,
        metadata_cols: list[str] | None = None,
        figsize: tuple[int, int] = (10, 12),
    ):
        """Plot a hierarchically-clustered heat-map of *distances*.

        Parameters
        ----------
        distances
            Square distance matrix.
        metadata_cols
            Optional list of ``.obs`` columns to annotate the heat-map.
        figsize
            Figure size passed to :func:`seaborn.clustermap`.

        Returns
        -------
        seaborn.matrix.ClusterGrid
        """
        import scipy.cluster.hierarchy as hc
        import scipy.spatial as sp

        linkage = hc.linkage(sp.distance.squareform(distances), method="average")

        if not metadata_cols:
            return sns.clustermap(distances, row_linkage=linkage, col_linkage=linkage)

        metadata = self._extract_metadata(columns=metadata_cols)
        annotation_colors = pd.DataFrame({col: _create_colormap(metadata, col) for col in metadata_cols})

        return sns.clustermap(
            pd.DataFrame(distances, index=annotation_colors.index, columns=annotation_colors.index),
            col_colors=annotation_colors,
            figsize=figsize,
            row_linkage=linkage,
            col_linkage=linkage,
        )

    def plot_embedding(
        self,
        distances: np.ndarray,
        method: str = "UMAP",
        metadata_cols: list[str] | None = None,
        continuous_palette: str = "viridis",
        categorical_palette: str = "tab10",
        na_color: str = "lightgray",
        axes=None,
    ):
        """Plot a 2-D embedding of *distances*, optionally coloured by metadata.

        Parameters
        ----------
        distances
            Square distance matrix used to compute (or look up) the embedding.
        method
            Embedding method.  One of ``"MDS"``, ``"TSNE"``, ``"UMAP"``.
        metadata_cols
            Columns from ``.obs`` used for colouring.
        continuous_palette, categorical_palette
            Seaborn palette names for continuous / categorical metadata.
        na_color
            Colour used for samples with missing metadata values.
        axes
            Existing matplotlib Axes (or array of Axes) to plot into.

        Returns
        -------
        matplotlib Axes or array of Axes
        """
        import matplotlib.pyplot as plt

        if method not in self.embeddings:
            self.embed(distances, method=method)

        embedding_df = pd.DataFrame(
            self.embeddings[method],
            columns=[f"{method}_0", f"{method}_1"],
            index=self.samples,
        )

        if metadata_cols is None:
            if axes is None:
                axes = sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1")
            else:
                sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1", ax=axes)
            return axes

        metadata_df = self._extract_metadata(columns=metadata_cols)
        embedding_df = pd.concat([embedding_df, metadata_df], axis=1)

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(metadata_cols), sharey=True, figsize=(len(metadata_cols) * 5, 5))

        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes

        for i, col in enumerate(metadata_cols):
            n_unique = len(np.unique(metadata_df[col].dropna()))
            palette = continuous_palette if n_unique > 5 else categorical_palette
            ax = axes_flat[i] if len(metadata_cols) > 1 else axes_flat

            sns.scatterplot(
                embedding_df[metadata_df[col].isna()],
                x=f"{method}_0",
                y=f"{method}_1",
                ax=ax,
                color=na_color,
            )
            sns.scatterplot(
                embedding_df,
                x=f"{method}_0",
                y=f"{method}_1",
                hue=col,
                ax=ax,
                palette=palette,
            )

        return axes
