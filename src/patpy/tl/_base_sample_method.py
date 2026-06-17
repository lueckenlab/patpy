from __future__ import annotations

import warnings
from typing import Literal

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

        self.sample_representation = None
        self.test_sample_labels: list | None = None
        self._fitted: bool = False

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
        self._check_adata_loaded()

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

        # getting only those layers with the same shape of the new X matrix from adata.layers[self.layer] to be copied in the new anndata below.
        # Newer anndata exposes ``X`` as ``layers[None]``; skip that key so we don't
        # re-inject it as a layer and clash with the explicit ``X`` passed below.
        filtered_layers = {
            key: np.copy(layer)
            for key, layer in self.adata.layers.items()
            if key is not None
            and key != self.layer
            and layer.shape == self.adata.layers.get(self.layer, np.empty(0)).shape
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

    def _check_adata_loaded(self) -> None:
        """Raise :class:`RuntimeError` if :meth:`prepare_anndata` has not been called."""
        if self.adata is None:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call prepare_anndata() first.")

    def _check_fitted(self) -> None:
        """Raise :class:`RuntimeError` if :meth:`prepare_anndata` has not completed successfully."""
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} is not fitted. Call prepare_anndata() first.")

    def calculate_distance_matrix(self):
        """Compute a sample-by-sample distance matrix. Subclasses must override."""
        self._check_fitted()
        raise NotImplementedError(f"{type(self).__name__} must implement calculate_distance_matrix().")

    def embed(
        self,
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
        distances = self.calculate_distance_matrix()
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
        method: str = "UMAP",
        metadata_cols: list[str] | None = None,
        continuous_palette: str = "viridis",
        categorical_palette: str = "tab10",
        na_color: str = "lightgray",
        axes=None,
        use_uns_colors: bool = True,
        color_key_suffix: str = "_colors",
        show_legend: bool = True,
    ):
        """Plot a 2-D embedding of *distances*, optionally coloured by metadata.

        Parameters
        ----------
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
        use_uns_colors
            If ``True``, look for colors in ``adata.uns[f'{col}{color_key_suffix}']``
            and use them if available (similar to scanpy).
        color_key_suffix
            Suffix for the color key in ``adata.uns``. Default is ``"_colors"``.
            For example, with suffix ``"_colors"``, will look for ``adata.uns['cell_type_colors']``.
        show_legend
            If ``True``, display the legend. If ``False``, hide it.

        Returns
        -------
        matplotlib Axes or array of Axes
        """
        import matplotlib.pyplot as plt

        if method not in self.embeddings:
            self.embed(method=method)

        embedding_df = pd.DataFrame(
            self.embeddings[method],
            columns=[f"{method}_0", f"{method}_1"],
            index=self.samples,
        )

        if metadata_cols is None:
            if axes is None:
                axes = sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1", legend=show_legend)
            else:
                sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1", ax=axes, legend=show_legend)
            return axes

        metadata_df = self._extract_metadata(columns=metadata_cols)
        embedding_df = pd.concat([embedding_df, metadata_df], axis=1)

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(metadata_cols), sharey=True, figsize=(len(metadata_cols) * 5, 5))

        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else axes

        for i, col in enumerate(metadata_cols):
            n_unique = len(np.unique(metadata_df[col].dropna()))

            # Try to get colors from adata.uns
            palette = None
            if use_uns_colors:
                color_key = f"{col}{color_key_suffix}"
                if color_key in self.adata.uns:
                    unique_vals = pd.unique(metadata_df[col].dropna())
                    colors = self.adata.uns[color_key]
                    # Create a mapping from values to colors
                    palette = dict(zip(unique_vals, colors, strict=False))

            # Fall back to default palette if not found in adata.uns
            if palette is None:
                palette = continuous_palette if n_unique > 5 else categorical_palette

            ax = axes_flat[i] if len(metadata_cols) > 1 else axes_flat

            sns.scatterplot(
                embedding_df[metadata_df[col].isna()],
                x=f"{method}_0",
                y=f"{method}_1",
                ax=ax,
                color=na_color,
                legend=False,
            )
            sns.scatterplot(
                embedding_df,
                x=f"{method}_0",
                y=f"{method}_1",
                hue=col,
                ax=ax,
                palette=palette,
                legend=show_legend,
            )

        return axes

    def _get_sample_representation_frame(self) -> pd.DataFrame:
        """Return the sample representation as a DataFrame indexed by sample.

        Works for every method flavour: representation methods set
        :attr:`sample_representation` directly (sometimes as a plain ndarray),
        while supervised methods may only expose it lazily through
        :meth:`get_sample_representations`. An ndarray is wrapped using
        :attr:`samples` as the index.
        """
        if hasattr(self, "get_sample_representations"):
            # Supervised methods recompute the embedding from the current adata;
            # always call it fresh so a cached representation from a previously
            # loaded cohort is never reused (it would mis-align with the labels).
            rep = self.get_sample_representations()
        else:
            rep = self.sample_representation
            if rep is None and hasattr(self, "calculate_distance_matrix"):
                # Representation methods (e.g. Pseudobulk) populate
                # sample_representation lazily when the distance matrix is computed.
                self.calculate_distance_matrix()
                rep = self.sample_representation
        if rep is None:
            raise RuntimeError(
                f"{type(self).__name__} has no sample representation. Call prepare_anndata "
                "(and, for supervised methods, get_sample_representations) before fitting a probe."
            )
        if not isinstance(rep, pd.DataFrame):
            rep = pd.DataFrame(np.asarray(rep), index=self.samples)
        return rep

    def fit_linear_probe(
        self,
        target: str,
        task: Literal["classification", "regression"] = "classification",
        test_size: float = 0.2,
        random_state: int = 42,
        test_sample_labels: list | None = None,
        store: bool = False,
    ) -> dict:
        """Fit a linear probe on top of sample embeddings.

        The probe is a plain sklearn model (:class:`~sklearn.linear_model.Ridge`
        for regression, balanced :class:`~sklearn.linear_model.LogisticRegression`
        for classification) trained on the method's sample representation. This
        works for any method that produces a per-sample embedding, including
        supervised methods whose native head solves a *different* task (e.g.
        training a regression probe on top of a classification model such as
        :class:`~patpy.tl.MixMIL`).

        Parameters
        ----------
        target
            Column in ``self.adata.obs`` to predict.
        task
            ``"classification"`` or ``"regression"``.
        test_size
            Fraction of donors held out for evaluation when
            ``test_sample_labels`` is not provided.
        random_state
            Random seed for the train/test split (used only when
            ``test_sample_labels`` is not provided).
        test_sample_labels
            Explicit list of sample labels (index values of
            :attr:`sample_representation`) to use as the test set.
            When provided, ``test_size`` and ``random_state`` are ignored.
            Pass an empty list to train the probe on *all* samples — useful
            when fitting a probe that will be applied to a different cohort; the
            returned metrics are then computed on the train set (see
            ``evaluated_on`` below). When ``None``, a random split is performed
            and the chosen test labels are stored in :attr:`test_sample_labels`
            for reproducibility.
        store
            When ``True`` (supervised methods only), register the fitted probe
            so that :meth:`predict` can reuse it on the current (or a swapped-in)
            cohort. The probe is saved in ``self._probes[target]`` and *target*
            is added to ``self.label_keys`` / ``self.tasks`` if not already
            present. This is how a regression head is attached to a
            classification-only model.

        Returns
        -------
        dict
            Keys: ``"model"``, ``"test_sample_labels"``, ``"evaluated_on"``,
            ``"{target}_test"``, ``"{target}_pred"``.

            For classification: additionally ``"accuracy"`` and ``"f1"``.
            For regression: additionally ``"r2"``, ``"pearson"``,
            ``"spearman"`` and ``"mae"``.

            ``evaluated_on`` is ``"test"`` when a non-empty test set is used and
            ``"train"`` when the probe was trained on all samples; in the latter
            case the metrics and ``"{target}_test"``/``"{target}_pred"`` describe
            the train set.

        Examples
        --------
        >>> result = model.fit_linear_probe(target="age", task="regression")
        >>> print(f"Pearson r = {result['pearson']:.3f}")  # doctest: +SKIP

        Attach a regression head to a classification model and predict:

        >>> model.fit_linear_probe("age", task="regression", store=True)  # doctest: +SKIP
        >>> ages = model.predict("age")  # doctest: +SKIP
        """
        # Note: we intentionally do not call _check_fitted() here. Representation
        # methods never flip the _fitted flag (their embedding is computed lazily),
        # yet they have a perfectly good sample representation to probe. Supervised
        # methods that are not trained will instead raise when their representation
        # is requested below.
        self._check_adata_loaded()

        if target not in self.adata.obs.columns:
            raise ValueError(f"target='{target}' not found in adata.obs.")
        if store and not hasattr(self, "_probes"):
            raise AttributeError(
                "store=True is only supported for supervised methods that maintain a `_probes` registry."
            )

        rep = self._get_sample_representation_frame()
        all_labels = rep.index

        # Extract target values from adata.obs (works for all sample methods)
        target_values = self._extract_metadata(columns=[target])

        if test_sample_labels is not None:
            test_set = set(test_sample_labels)
            test_idx = [lbl for lbl in all_labels if lbl in test_set]
            train_idx = [lbl for lbl in all_labels if lbl not in test_set]
        else:
            from sklearn.model_selection import train_test_split

            y_all = target_values.loc[all_labels, target].values
            train_idx, test_idx = train_test_split(
                list(all_labels),
                test_size=test_size,
                random_state=random_state,
                stratify=(y_all if task == "classification" else None),
            )

        self.test_sample_labels = test_idx
        X_train = rep.loc[train_idx].values
        y_train = target_values.loc[train_idx, target].values

        # Evaluate on the held-out test set when one is given; otherwise fall back to
        # the train set so the returned metrics describe the fitted probe instead of
        # being empty. ``evaluated_on`` records which set the metrics refer to.
        if len(test_idx) > 0:
            eval_X = rep.loc[test_idx].values
            eval_y = target_values.loc[test_idx, target].values
            evaluated_on = "test"
        else:
            eval_X = X_train
            eval_y = y_train
            evaluated_on = "train"

        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, f1_score

            model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(eval_X)
            result = {
                "model": model,
                "test_sample_labels": test_idx,
                "evaluated_on": evaluated_on,
                f"{target}_test": eval_y,
                f"{target}_pred": y_pred,
                "accuracy": accuracy_score(eval_y, y_pred),
                "f1": f1_score(eval_y, y_pred, average="weighted", zero_division=0),
            }
        elif task == "regression":
            from sklearn.linear_model import Ridge

            from patpy.tl.evaluation import evaluate_regression

            model = Ridge(alpha=0.1)
            model.fit(X_train, y_train)
            y_pred = model.predict(eval_X)
            metrics = evaluate_regression(eval_y, y_pred)
            result = {
                "model": model,
                "test_sample_labels": test_idx,
                "evaluated_on": evaluated_on,
                f"{target}_test": eval_y,
                f"{target}_pred": y_pred,
                "r2": metrics["r2"],
                "pearson": metrics["pearson"],
                "spearman": metrics["spearman"],
                "mae": metrics["mae"],
            }
        else:
            raise ValueError(f"task must be 'classification' or 'regression', got '{task}'.")

        if store:
            self._probes[target] = result["model"]
            if hasattr(self, "label_keys") and target not in self.label_keys:
                self.label_keys.append(target)
                self.tasks.append(task)

        return result
