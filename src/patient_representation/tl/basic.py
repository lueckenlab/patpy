import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns

from patient_representation.pp import (
    extract_metadata,
    fill_nan_distances,
    filter_small_cell_types,
    filter_small_samples,
    is_count_data,
    subsample,
)
from patient_representation.tl._types import _EVALUATION_METHODS


def valid_aggregate(aggregate: str):
    """Returns a valid aggregation function or raises an error if invalid"""
    valid_aggregates = {"mean": np.mean, "median": np.median, "sum": np.sum}
    if aggregate not in valid_aggregates:
        raise ValueError(f"Aggregation function '{aggregate}' is not supported")
    return valid_aggregates[aggregate]


def valid_distance_metric(dist: str):
    """Returns if the distance metric is valid or raises an error"""
    valid_dists = {"euclidean", "cosine", "cityblock"}
    if dist not in valid_dists:
        raise ValueError(f"Distance metric '{dist}' is not supported")
    return dist


def make_matrix_symmetric(matrix):
    """Make a matrix symmetric by averaging it with its transpose.

    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse.spmatrix
        The input matrix to be made symmetric.
    matrix_name : str, optional
        Name of the matrix for the warning message, by default "Matrix".

    Returns
    -------
    np.ndarray or scipy.sparse.spmatrix
        Symmetric matrix.
    """
    import warnings

    import numpy as np
    import scipy.sparse

    is_sparse = scipy.sparse.issparse(matrix)

    def is_symmetric(mat):
        if is_sparse:
            diff = mat - mat.T
            return np.allclose(diff.data, 0)
        else:
            return np.allclose(mat, mat.T)

    def symmetrize(mat):
        if is_sparse:
            return (mat + mat.T).multiply(0.5)
        else:
            return (mat + mat.T) * 0.5

    if is_symmetric(matrix):
        return matrix
    else:
        warnings.warn(
            "Data matrix is not symmetric. Fixing by symmetrizing.",
            stacklevel=2,
        )
        return symmetrize(matrix)


def create_colormap(df, col, palette="Spectral"):
    """Create a color map for the unique values of the column `col` of data frame `df`"""
    unique_values = df[col].unique()

    colors = sns.color_palette(palette, n_colors=len(unique_values))
    color_map = dict(zip(unique_values, colors))
    return df[col].map(color_map)


def describe_metadata(metadata: pd.DataFrame) -> None:
    """Prints the basic information about the metadata and tries to guess column types

    Parameters
    ----------
    metadata : pd.DataFrame
        File with metadata for the samples. Or any pandas data frame you want to describe
    """
    from pandas.api.types import is_numeric_dtype

    n = metadata.shape[0]

    numeric_cols = []
    categorical_cols = []

    for col in metadata.columns:
        n_missing = metadata[col].isna().sum()
        n_unique = len(metadata[col].unique())

        if is_numeric_dtype(metadata[col]) and n_unique > 10:
            numeric_cols.append(col)
        elif n_unique > 1 and n_unique < n // 2:
            categorical_cols.append(col)

        print("Column", col)
        print("Type:", metadata[col].dtype)
        print("Number of missing values:", n_missing, f"({round(100 * n_missing / n, 2)}%)")
        print("Number of unique values:", n_unique)

        if n_unique < 50:
            print("Unique values:", metadata[col].unique())

        print("-" * 25)
        print()

    print("Possibly, numerical columns:", numeric_cols)
    print("Possibly, categorical columns:", categorical_cols)


def phemd(data, labels, n_clusters=8, random_state=42, n_jobs=-1):
    """Compute the PhEMD between distributions. As specified in Chen et al. 2019.

    Source: https://github.com/atong01/MultiscaleEMD/blob/main/comparison/phemd.py

    Args:
        data: 2-D array N x F points by features.
        labels: 2-D array N x M points by distributions.

    Returns
    -------
        distance_matrix: 2-D M x M array with each cell representing the
        distance between each distribution of points.
    """
    import ot
    import phate
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances

    phate_op = phate.PHATE(random_state=random_state, n_jobs=n_jobs)
    phate_op.fit(data)
    cluster_op = KMeans(n_clusters, random_state=random_state)
    cluster_ids = cluster_op.fit_predict(phate_op.diff_potential)
    cluster_centers = np.array(
        [
            np.average(
                data[(cluster_ids == c)],
                axis=0,
                weights=labels[cluster_ids == c].sum(axis=1),
            )
            for c in range(n_clusters)
        ]
    )
    # Compute the cluster histograms C x M
    cluster_counts = np.array([labels[(cluster_ids == c)].sum(axis=0) for c in range(n_clusters)])
    cluster_dists = np.ascontiguousarray(pairwise_distances(cluster_centers, metric="euclidean"))

    N, M = labels.shape
    assert data.shape[0] == N
    dists = np.empty((M, M))
    for i in range(M):
        for j in range(i, M):
            weights_a = np.ascontiguousarray(cluster_counts[:, i])
            weights_b = np.ascontiguousarray(cluster_counts[:, j])
            dists[i, j] = dists[j, i] = ot.emd2(weights_a, weights_b, cluster_dists)
    return dists


def calculate_average_without_nans(array, axis=0, return_sample_sizes=True, default_value=0):
    """Calculate average across `axis` in `array`. Consider only numbers, drop NAs

        If all values along the axis are NaN, fill with a default value

    Note that sample size can be different for each value in the resulting array

    Parameters
    ----------
    array : np.ndarray
        Array to calculate average for
    axis : int = 0
        Axis to calculate average across
    return_sample_sizes : bool = True
        If True, return number of NAs for each value in the resulting array

    Returns
    -------
    averages : np.ndarray
        Average across `axis` in `array`

    Examples
    --------
    >>> arr = np.array([
            np.ones(shape=(2, 2)),
            np.ones(shape=(2, 2)) * 3,
            [[5, np.nan],
             [5, np.nan]],
        ])  # arr now contains 3 2x2 matrices
    >>> arr[0, 1, 1] = np.nan
    >>> arr[0, 1, 0] = np.nan
    >>> arr[1, 1, :] = np.nan
    >>> arr  # One layer contains 0 nans, another 1, the next on 2, and the last one 4 (all)
    array([[[ 1.,  1.],
        [nan, nan]],

       [[ 3.,  3.],
        [nan, nan]],

       [[ 5., nan],
        [ 5., nan]]])

    >>> averages, sample_sizes = calculate_average_without_nans(arr, axis=0)
    >>> averages
    array([[ 3.,  2.],
           [ 5., nan]])
    >>> sample_sizes
    array([[3, 2],
           [1, 0]])
    """
    not_empty_values = ~np.isnan(array)
    sample_sizes = not_empty_values.sum(axis=axis)

    # Fill NaNs with the mean of non-NaN values
    mean_values = np.nanmean(array, axis=axis, keepdims=True)

    # Replace remaining NaNs with default_value
    mean_values = np.where(np.isnan(mean_values), default_value, mean_values)

    array_filled = np.where(not_empty_values, array, mean_values)

    averages = np.mean(array_filled, axis=axis)

    if return_sample_sizes:
        return averages, sample_sizes

    return averages


class PatientsRepresentationMethod:
    """Base class for patient representation methods"""

    DISTANCES_UNS_KEY = "X_method-name_distances"

    def _get_data(self):
        """Extract data from correct layer specified by `self.layer`"""
        if self.adata is None:
            raise RuntimeError("adata is not yet set. Please, run prepare_anndata() method first")

        if self.layer is None or self.layer == "X":
            # Assuming, data is stored in .X
            warnings.warn("Using data from adata.X", stacklevel=1)
            return self.adata.X

        elif self.adata.obsm and self.layer in self.adata.obsm:
            warnings.warn(f"Using data from key {self.layer} of adata.obsm", stacklevel=1)
            return self.adata.obsm[self.layer]

        elif self.adata.layers and self.layer in self.adata.layers:
            warnings.warn(f"Using data from key {self.layer} of adata.layers", stacklevel=1)
            return self.adata.layers[self.layer]

        else:
            raise ValueError(f"Cannot find layer {self.layer} in adata. Please make sure it is specified correctly")

    def _move_layer_to_X(self) -> sc.AnnData:
        """Some models require data to be stored in `adata.X`. This method moves `self.layer` to `.X`"""
        if self.layer == "X" or self.layer is None:
            # The data is already in correct slot
            return self.adata

        # getting only those layers with the same shape of the new X mat from adata.obsm[self.layer] to be copied in the new anndata below
        filtered_layers = {
            key: np.copy(layer)
            for key, layer in self.adata.layers.items()
            if key != self.layer and layer.shape == self.adata.obsm[self.layer].shape
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

    def _extract_metadata(self, columns) -> pd.DataFrame:
        """Return dataframe with requested `columns` in the correct rows order"""
        return extract_metadata(self.adata, self.sample_key, columns, samples=self.samples)

    def __init__(self, sample_key, cells_type_key, layer=None, seed=67):
        """Initialize the model

        Parameters
        ----------
        sample_key : str
            Column in .obs containing sample IDs
        cells_type_key : str
            Column in .obs containing cell types
        layer : Optional[str] = None
            What to use as data in a model. If None or "X", `adata.X` is used. Otherwise, the corresponding key from `adata.obsm` will be used
        seed : int = 67
            Number to initialize pseudorandom generator
        """
        self.sample_key = sample_key
        self.cells_type_key = cells_type_key
        self.layer = layer
        self.seed = seed

        self.adata = None
        self.samples = None
        self.cell_types = None
        self.embeddings = {}
        self.samples_adata = None

    # fit-like method: save data and process it
    def prepare_anndata(self, adata, sample_size_threshold: int = 1, cluster_size_threshold: int = 0):
        """Prepare adata for the analysis, filter cell types and samples with too few observations

        Parameters
        ----------
        sample_size_threshold : int = 300
            Minimum sample size to be considered. Samples with fewer number of cells
            are filtered out
        cluster_size_threshold : int = 5
            Minimum cell type size per sample to be considered. Cell types with fewer
            number of cells at least in 1 sample are filtered out

        """
        self.adata = adata

        # Filter samples with too few cells
        self.adata = filter_small_samples(
            adata=self.adata, sample_key=self.sample_key, sample_size_threshold=sample_size_threshold
        )
        self.samples = self.adata.obs[self.sample_key].unique()

        # Filter cell types with too few cells
        self.adata = filter_small_cell_types(
            adata=self.adata,
            sample_key=self.sample_key,
            cells_type_key=self.cells_type_key,
            cluster_size_threshold=cluster_size_threshold,
        )
        self.cell_types = self.adata.obs[self.cells_type_key].unique()

    def calculate_distance_matrix(self, force: bool = False):
        """Transform-like method: returns samples distances matrix"""
        if self.DISTANCES_UNS_KEY in self.adata.uns and not force:
            return self.adata.uns[self.DISTANCES_UNS_KEY]

    def plot_clustermap(self, metadata_cols=None, figsize=(10, 12), *args, **kwargs):
        """Plot a clusterized heatmap of distances"""
        import scipy.cluster.hierarchy as hc
        import scipy.spatial as sp

        distances = self.calculate_distance_matrix(*args, **kwargs)
        linkage = hc.linkage(sp.distance.squareform(distances), method="average")

        if not metadata_cols:
            return sns.clustermap(distances, row_linkage=linkage, col_linkage=linkage)

        metadata = self._extract_metadata(columns=metadata_cols)

        annotation_colors = {}

        for col in metadata_cols:
            annotation_colors[col] = create_colormap(metadata, col)

        annotation_colors = pd.DataFrame(annotation_colors)

        return sns.clustermap(
            pd.DataFrame(distances, index=annotation_colors.index, columns=annotation_colors.index),
            col_colors=annotation_colors,
            figsize=figsize,
        )

    def embed(self, method="UMAP", n_jobs: int = -1, verbose: bool = False):
        """Convert distances to embedding of the samples

        Parameters
        ----------
        method : str = "TSNE
            Method to use for embedding. Currently, "TSNE" and "MDS" are supported
        n_jobs : int = 1
            Number of threads to use for computation. Use -1 to run on all processors
        verbose : bool = False
            If True, print logging information during the computation

        Returns
        -------
        coordinates : array-like
            Coordinates of samples in the embedding space. 2D for TSNE and MDS
        """
        distances = self.adata.uns[self.DISTANCES_UNS_KEY]
        distances = fill_nan_distances(distances)

        if method == "MDS":
            from sklearn.manifold import MDS

            mds = MDS(
                n_components=2, dissimilarity="precomputed", verbose=verbose, n_jobs=n_jobs, random_state=self.seed
            )
            coordinates = mds.fit_transform(distances)
        elif method == "TSNE":
            from openTSNE import TSNE

            tsne = TSNE(
                n_components=2,
                metric="precomputed",
                neighbors="exact",
                n_jobs=n_jobs,
                random_state=self.seed,
                verbose=verbose,
                initialization="spectral",  # pca doesn't work with precomputed distances
            )
            coordinates = tsne.fit(distances)
        elif method == "UMAP":
            from umap import UMAP

            umap = UMAP(n_components=2, metric="precomputed", random_state=self.seed, verbose=verbose, n_jobs=n_jobs)
            coordinates = umap.fit_transform(distances)

        else:
            raise ValueError(f'Method {method} is not supported, please use one of ["MDS", "TSNE", "UMAP"]')

        self.embeddings[method] = coordinates
        return coordinates

    def to_adata(self, metadata: pd.DataFrame = None, *args, **kwargs):
        """Convert samples data to AnnData object

        Parameters
        ----------
        metadata : Optional[pd.DataFrame] = None
            Metadata about samples to be added to .obs of AnnData object. Should contain samples in index
        *args, **kwargs
            Additional arguments to pass to calculate_distance_matrix method

        Returns
        -------
        samples_adata : AnnData
            AnnData object with samples data
        """
        if (
            self.patient_representations is not None
            and self.patient_representations.ndim == 2
            and self.patient_representations.shape[0] == len(self.samples)
        ):
            representation = self.patient_representations
        else:
            representation = np.array(self.embed())

        self.samples_adata = sc.AnnData(
            X=representation,
            obs=metadata.loc[self.samples] if metadata is not None else None,
            obsm={self.DISTANCES_UNS_KEY: self.calculate_distance_matrix(*args, **kwargs)},
        )

        # Move samples embeddings to .obsm
        for method, embedding in self.embeddings.items():
            self.samples_adata.obsm["X_" + method.lower()] = embedding

        return self.samples_adata

    def plot_embedding(
        self,
        method="UMAP",
        metadata_cols=None,
        continuous_palette="viridis",
        categorical_palette="tab10",
        na_color="lightgray",
    ):
        """Plot embedding of samples colored by `metadata_cols`"""
        import matplotlib.pyplot as plt

        if method not in self.embeddings:
            self.embed(method=method)

        embedding_df = pd.DataFrame(self.embeddings[method], columns=[f"{method}_0", f"{method}_1"], index=self.samples)

        if metadata_cols is None:
            # Simply plot the embedding
            axes = sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1")
        else:
            # Colorize samples by metadata
            metadata_df = self._extract_metadata(columns=metadata_cols)
            embedding_df = pd.concat([embedding_df, metadata_df], axis=1)

            _, axes = plt.subplots(nrows=1, ncols=len(metadata_cols), sharey=True, figsize=(len(metadata_cols) * 5, 5))

            for i, col in enumerate(metadata_cols):
                n_unique_values = len(np.unique(metadata_df[col]))
                if n_unique_values > 5:
                    palette = continuous_palette
                else:
                    palette = categorical_palette

                # If there is only 1 metadata column, axes is not subscriptable
                ax = axes[i] if len(metadata_cols) > 1 else axes

                # Plot points with missing values in metadata
                sns.scatterplot(
                    embedding_df[metadata_df[col].isna()],
                    x=f"{method}_0",
                    y=f"{method}_1",
                    ax=ax,
                    color=na_color,
                )
                # Plot points with known metadata
                sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1", hue=col, ax=ax, palette=palette)

        return axes

    def evaluate_representation(
        self,
        target,
        method: _EVALUATION_METHODS = "knn",
        metadata=None,
        num_donors_subset=None,
        proportion_donors_subset=None,
        **parameters,
    ):
        """Evaluate representation of `target` for the given distance matrix

        Parameters
        ----------
        target : "str"
            A patient covariate to evaluate representation for
        method : Literal["knn", "distances", "proportions", "silhouette"]
            Method to use for evaluation:
            - knn: predict values of `target` using K-nearest neighbors and evaluate the prediction
            - distances: test if distances between samples are significantly different from the null distribution
            - proportions: test if distribution of `target` differs between groups (e.g. clusters)
            - silhouette: calculate silhouette score for the given distances
        num_donors_subset : int, optional
            Absolute number of donors to include in the evaluation.
        proportion_donors_subset : float, optional
            Proportion of donors to include in the evaluation.
        parameters : dict
            Parameters for the evaluation method. The following parameters are used:
            - knn:
                - n_neighbors: number of neighbors to use for prediction
                - task: type of prediction task. One of "classification", "regression", "ranking". See documentation of `predict_knn` for more information
            - distances:
                - control_level: value of `target` that should be used as a control group
                - normalization_type: type of normalization to use. One of "total", "shift", "var". See documentation of `test_distances_significance` for more information
                - n_bootstraps: number of bootstrap iterations to use
                - trimmed_fraction: fraction of the most extreme values to remove from the distribution
                - compare_by_difference: if True, normalization is defined as difference (as in the original paper). Otherwise, it is defined as a ratio
            - proportions:
                - groups: groups (e.g. cluster numbers) of the observations

        Returns
        -------
        result : dict
            Result of evaluation with the following keys:
            - score: a number evaluating the representation. The higher the better
            - metric: name of the metric used for evaluation
            - n_unique: number of unique values in `target`
            - n_observations: number of observations used for evaluation. Can be different for different targets, even within one dataset (because of NAs)
            - method: name of the method used for evaluation
            There are other optional keys depending on the method used for evaluation.
        """
        from patient_representation.tl import evaluate_representation

        if metadata is None:
            metadata = self._extract_metadata([target])

        return evaluate_representation(
            self.calculate_distance_matrix(),
            metadata[target],
            method,
            num_donors_subset=num_donors_subset,
            proportion_donors_subset=proportion_donors_subset,
            **parameters,
        )

    def predict_metadata(self, target, metadata=None, n_neighbors: int = 3, task="classification"):
        """Predict classes from metadata column `target` for samples using K-Nearest Neighbors classifier

        Parameters
        ----------
        target : str
            Column name from `adata.obs`, which will be used for classification
        metadata : Optional[pd.DataFrame] = None
            Table with metadata about samples. Index should contain samples. If None, `adata.obs` is used
        n_neighbors : int = 3
            Number of neighbors to use for classification
        task : str = "classification"

        Returns
        -------
        y_true : array-like
            True values of `target` from metadata for samples with known values
        y_predicted : array-like
            Predicted values of `target` for samples with known values
        """
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        if metadata is None:
            metadata = self._extract_metadata([target])

        y_true = metadata[target]
        is_class_known = y_true.notna()
        distances = self.calculate_distance_matrix()
        distances = distances[is_class_known][:, is_class_known]  # Drop samples with unknown target

        # Diagonal contains 0s forcing using the same sample for prediction
        # This gives the perfect prediction even for random target (super weird)
        # Filling diagonal with large value removes this leakage
        np.fill_diagonal(distances, distances.max())

        if task == "classification":
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed", weights="distance")
        elif task == "regression":
            knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric="precomputed", weights="distance")
        else:
            raise ValueError(f'task {task} is not supported, please set one of ["classification", "regression"]')

        knn.fit(distances, y_true[is_class_known])

        return y_true[is_class_known], knn.predict(distances)

    def plot_metadata_distribution(
        self,
        metadata_columns: list[str],
        tasks: list[str],
        method: _EVALUATION_METHODS = "knn",
        embedding: str = "UMAP",
        metadata=None,
        metric_threshold=0.4,
    ):
        """Predict metadata columns, and plot embeddings colorised by metadata values

        Parameters
        ----------
        metadata_columns : list
            List of metadata columns to show
        tasks : list
            Tasks for each metadata column (classification, ranking or regression). Can be one string for all columns.
        method : Literal["knn", "distances", "proportions", "silhouette"]
            Method to use for evaluation. See documentation of `evaluate_representation` for more information
        embedding : str = "UMAP"
            Embedding to use for plotting
        metric_threshold : float = 0.3
            Results with lower values than this metric will not be displayed
        """
        if isinstance(tasks, str):
            tasks = [tasks] * len(metadata_columns)

        result_cols = ("feature", "score", "metric", "n_unique", "n_observations", "method")
        results = []

        for col, task in zip(metadata_columns, tasks):
            result = self.evaluate_representation(target=col, method=method, metadata=metadata, task=task)
            results.append(
                (col, result["score"], result["metric"], result["n_unique"], result["n_observations"], result["method"])
            )

        results = pd.DataFrame(results, index=metadata_columns, columns=result_cols)
        results = results.sort_values("score", ascending=False)

        # Plot results from the best to the worst
        for _, row in results.iterrows():
            if row["score"] < metric_threshold:
                break

            col = row["feature"]
            ax = self.plot_embedding(metadata_cols=[col], method=embedding)
            ax.set_title(f'{col}: {round(row["score"], 4)}')
            ax.legend(loc=(1.05, 0))

        return results


class MrVI(PatientsRepresentationMethod):
    """Deep generative modeling for quantifying sample-level heterogeneity in single-cell omics.

    Source: https://www.biorxiv.org/content/10.1101/2022.10.04.510898v2
    """

    DISTANCES_UNS_KEY = "X_mrvi_distances"

    def __init__(
        self,
        sample_key: str,
        cells_type_key: str,
        batch_key: str = None,
        layer=None,
        seed=67,
        max_epochs=400,
        **model_params,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.model = None
        self.model_params = model_params
        self.patient_representations = None
        self.max_epochs = max_epochs
        self.batch_key = batch_key

    def prepare_anndata(self, adata, sample_size_threshold: int = 1, cluster_size_threshold: int = 0):
        """Train MrVI model

        Parameters
        ----------
        adata : AnnData object with raw counts in .X

        Sets
        ----
        model : MrVI model
        """
        from scvi.external import MRVI

        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        assert is_count_data(self._get_data()), "`layer` must contain count data with integer numbers"

        layer = None if self.layer == "X" else self.layer
        MRVI.setup_anndata(self.adata, sample_key=self.sample_key, layer=layer, batch_key=self.batch_key)

        self.model = MRVI(self.adata, **self.model_params)
        self.model.train(max_epochs=self.max_epochs)

        self.samples = self.model.sample_order

    def calculate_distance_matrix(
        self,
        groupby=None,
        keep_cell=True,
        calculate_representations=False,
        batch_size: int = 32,
        mc_samples: int = 10,
        force: bool = False,
    ):
        """Return sample by sample distances matrix

        Parameters
        ----------
        calculate_representations : bool = False
            If True, calculate representations of samples and cells, otherwise only return distances matrix
        batch_size : int = 1000
            Number of cells in batch when calculating matrix of distances between samples
        mc_samples : int = 10
            Number of Monte Carlo samples to use for computing the local sample representation.
        force : bool = False
            If True, recalculate distances

        Sets
        ----
        adata.obsm["X_mrvi_z"] - latent representation from the layer Z of MrVI
        adata.obsm["X_mrvi_u"] - latent representation from the layer U of MrVI
        adata.uns["X_mrvi_distances"] – matrix of distances between samples according to MrVI representation

        Returns
        -------
        Matrix of distances between samples
        """
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None and not force:
            return distances

        # Make sure that batch size is between 1 and number of cells
        batch_size = int(np.clip(batch_size, 1, len(self.adata)))

        if calculate_representations:
            if "X_mrvi_z" not in self.adata.obsm or force:
                print("Calculating cells representation from layer Z")
                self.adata.obsm["X_mrvi_z"] = self.model.get_latent_representation(give_z=True)
            if "X_mrvi_u" not in self.adata.obsm or force:
                print("Calculating cells representation from layer U")
                self.adata.obsm["X_mrvi_u"] = self.model.get_latent_representation(give_z=False)

            print("Calculating cells representations")
            # This is a tensor of shape (n_cells, n_samples, n_latent_variables)
            cell_sample_representations = self.model.get_local_sample_representation(batch_size=batch_size)

            self.patient_representations = np.zeros(shape=(len(self.samples), cell_sample_representations.shape[2]))

            print("Calculating samples representations")
            # For a patient representation we will take centroid of cells of this sample
            for i, sample in enumerate(self.samples):
                sample_mask = self.adata.obs[self.sample_key] == sample
                self.patient_representations[i] = cell_sample_representations[sample_mask, i].mean(axis=0)

            # Here, we obtain distances between samples in a different way
            # MrVI calculates sample-sample distances per cell and then aggregates them (see below)
            # Here, we first aggregate cells and then calculate sample-sample distances. Note that it produces different results
            print(
                f"Using aggregated cell representation approach, distances are stored in self.adata.uns[{self.DISTANCES_UNS_KEY}_cell_based"
            )
            distances = scipy.spatial.distance.pdist(self.patient_representations)
            distances = scipy.spatial.distance.squareform(distances)
            self.adata.uns[self.DISTANCES_UNS_KEY + "_cell_based"] = distances

        print("Calculating distance matrix between samples")

        # Calculate distances in MrVI recommended way with counterfactuals
        distances = self.model.get_local_sample_distances(
            groupby=groupby, keep_cell=keep_cell, batch_size=batch_size, mc_samples=mc_samples
        )

        distances_to_average = distances["cell" if groupby is None else groupby].values
        avg_distances, sample_sizes = calculate_average_without_nans(distances_to_average, axis=0)

        self.adata.uns["mrvi_parameters"] = {
            "batch_size": batch_size,
            "sample_sizes": sample_sizes,
        }

        self.adata.uns[self.DISTANCES_UNS_KEY] = avg_distances

        return self.adata.uns[self.DISTANCES_UNS_KEY]


class WassersteinTSNE(PatientsRepresentationMethod):
    """Method based on the matrix of pairwise Wasserstein distances between units.

    Source: https://arxiv.org/abs/2205.07531
    """

    DISTANCES_UNS_KEY = "X_wasserstein_distances"

    def __init__(self, sample_key, cells_type_key, replicate_key, layer="X_scvi", seed=67):
        """Create Wasserstein distances embedding between samples

        Parameters
        ----------
        sample_key : str
            Key in .obs that specifies the samples between which distances are calculated.
            This corresponds to "unit" in the original WassersteinTSNE paper
        replicate_key : str
            Key in .obs that specifies some kind of replicate for the observations of a sample.
            Could be cell types. Corresponds to "sample" in the original WassersteinTSNE paper
        layer : Optional[str]
            Key in .obsm where the data is stored. We recommend using scVI or scANVI embedding
        seed : int = 67
            Number to initialize pseudorandom generator
        """
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.replicate_key = replicate_key

        self.model = None
        self.distances_model = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 1, cluster_size_threshold: int = 0):
        """Set up Gaussian Wasserstein Distance model"""
        import WassersteinTSNE as WT

        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        data = pd.DataFrame(self._get_data())
        data.set_index([self.adata.obs[self.sample_key], self.adata.obs[self.replicate_key]], inplace=True)

        self.model = WT.Dataset2Gaussians(data)
        self.distances_model = WT.GaussianWassersteinDistance(self.model)

    def calculate_distance_matrix(self, covariance_weight=0.5, force: bool = False):
        r"""Return sample by sample distances matrix

        Parameters
        ----------
        covariance_weight : float = 0.5
            Float between 0 and 1, which indicates how much the distance between covariances
            influences the distances. Corresponds to a parameter $\\lambda$ in original paper,
            and to papameter `w` in the WassersteinTSNE package
        force : bool = False
            If True, recalculate distances

        Returns
        -------
        Matrix of distances between samples
        """
        is_correct_key_in_uns = (
            "wasserstein_covariance_weight" in self.adata.uns
            and self.adata.uns["wasserstein_covariance_weight"] == covariance_weight
        )
        is_recalculated = force or not is_correct_key_in_uns

        if self.DISTANCES_UNS_KEY in self.adata.uns:
            if is_recalculated:
                warnings.warn(f"Rewriting uns key {self.DISTANCES_UNS_KEY}", stacklevel=1)
            else:
                return self.adata.uns[self.DISTANCES_UNS_KEY]

        distances = self.distances_model.matrix(covariance_weight).values
        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["wasserstein_covariance_weight"] = covariance_weight

        return self.adata.uns[self.DISTANCES_UNS_KEY]

    def plot_clustermap(self, covariance_weight=0.5):
        """Plot clusterized heatmap of samples"""
        return super().clustermap(covariance_weight=covariance_weight)


class PILOT(PatientsRepresentationMethod):
    """Optimal transport based method to compute the Wasserstein distance between two single single-cell experiments.

    Source: https://www.biorxiv.org/content/10.1101/2022.12.16.520739v1
    """

    DISTANCES_UNS_KEY = "X_pilot_distances"

    def __init__(
        self,
        sample_key,
        cells_type_key,
        patient_state_col,
        dataset_name="pilot_dataset",
        layer="X_pca",
        seed=67,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.patient_state_col = patient_state_col
        self.dataset_name = dataset_name

        self.results_dir = None
        self.pc = None
        self.annotation = None
        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False, **pilot_parameters):
        """Calculate matrix of distances between samples

        Parameters
        ----------
        force : bool = False
            If True, recalculate distances
        pilot_parameters : dict
            Parameters to pass to pilot.tl.wasserstein_distance. Possible keys and default values are:
            - metric = 'cosine'
            - regulizer = 0.2
            - normalization = True
            - regularized = 'unreg'
            - reg = 0.1
            - res = 0.01
            - steper = 0.01
            For parameters description, refer to the PILOT documentation

        Returns
        -------
        Matrix of distances between samples
        """
        import pilotpy as pt

        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        # This runs all the calculations and adds several keys to .uns
        pt.tl.wasserstein_distance(
            self.adata,
            clusters_col=self.cells_type_key,
            sample_col=self.sample_key,
            status=self.patient_state_col,
            emb_matrix=self.layer,
            data_type="scRNA",
            **pilot_parameters,
        )

        # Matrix of cell type proportions for each sample
        self.patient_representations = (
            pd.DataFrame(self.adata.uns["proportions"], index=self.cell_types).T.loc[self.samples].to_numpy()
        )

        distances = self.adata.uns["EMD_df"].loc[self.samples, self.samples].to_numpy()
        distances = make_matrix_symmetric(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["pilot_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            **pilot_parameters,
        }
        return distances


class TotalPseudobulk(PatientsRepresentationMethod):
    """A simple baseline, which represents patients as pseudobulk of their gene expression"""

    DISTANCES_UNS_KEY = "X_pseudobulk_distances"

    def __init__(self, sample_key, cells_type_key, layer="X_pca", seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False, aggregate="mean", dist="euclidean"):
        """Calculate distances between pseudobulk representations of samples"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        aggregation_func = valid_aggregate(aggregate)
        distance_metric = valid_distance_metric(dist)

        data = self._get_data()

        self.patient_representations = np.zeros(shape=(len(self.samples), data.shape[1]))

        for i, sample in enumerate(self.samples):
            sample_cells = data[self.adata.obs[self.sample_key] == sample, :]
            self.patient_representations[i] = aggregation_func(sample_cells, axis=0)

        distances = scipy.spatial.distance.pdist(self.patient_representations, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["bulk_parameters"] = {
            "sample_key": self.sample_key,
            "aggregate": aggregate,
            "distance_type": distance_metric,
        }

        return distances


class CellTypePseudobulk(PatientsRepresentationMethod):
    """Baseline, where distances between patients are average distances between their cell type pseudobulks"""

    DISTANCES_UNS_KEY = "X_ct_pseudobulk_distances"

    def __init__(self, sample_key, cells_type_key, layer="X_pca", seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False, aggregate="mean", dist="euclidean"):
        """Calculate distances between patients as average distance between per cell-type pseudobulks"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        aggregation_func = valid_aggregate(aggregate)
        distance_metric = valid_distance_metric(dist)

        data = self._get_data()

        # List of matrices with embedding centroids for samples for each cell type
        self.patient_representations = np.zeros(shape=(len(self.cell_types), len(self.samples), data.shape[1]))
        for i, cell_type in enumerate(self.cell_types):
            for j, sample in enumerate(self.samples):
                cells_data = data[
                    (self.adata.obs[self.sample_key] == sample) & (self.adata.obs[self.cells_type_key] == cell_type)
                ]
                if cells_data.size == 0:
                    self.patient_representations[i, j] = np.nan
                else:
                    self.patient_representations[i, j] = aggregation_func(cells_data, axis=0)

        # Matrix of distances between samples for each cell type
        distances = np.zeros(shape=(len(self.cell_types), len(self.samples), len(self.samples)))

        for i, cell_type_embeddings in enumerate(self.patient_representations):
            samples_distances = scipy.spatial.distance.pdist(cell_type_embeddings, metric=distance_metric)
            distances[i] = scipy.spatial.distance.squareform(samples_distances)

        avg_distances, sample_sizes = calculate_average_without_nans(distances, axis=0)

        self.adata.uns[self.DISTANCES_UNS_KEY] = avg_distances
        self.adata.uns["celltypebulk_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "aggregate": aggregate,
            "distance_type": distance_metric,
            "sample_sizes": sample_sizes,
        }

        return avg_distances


class RandomVector(PatientsRepresentationMethod):
    """A dummy baseline, which represents patients as random embeddings"""

    DISTANCES_UNS_KEY = "X_random_vector_distances"

    def __init__(self, sample_key, cells_type_key, latent_dim: int = 30, seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.latent_dim = latent_dim
        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between patients represented as random vectors"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        self.patient_representations = np.random.normal(size=(len(self.samples), self.latent_dim))

        distances = scipy.spatial.distance.pdist(self.patient_representations)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["random_vec_parameters"] = {
            "sample_key": self.sample_key,
        }

        return distances


class CellTypesComposition(PatientsRepresentationMethod):
    """A simple baseline, which represents patients as composition of their cell types"""

    DISTANCES_UNS_KEY = "X_celltype_composition"

    def __init__(self, sample_key, cells_type_key, layer=None, seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False, dist="euclidean"):
        """Calculate distances between patients represented as cell type composition vectors"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distance_metric = valid_distance_metric(dist)

        # Calculate proportions of the cell types for each sample
        self.patient_representations = pd.crosstab(
            self.adata.obs[self.sample_key], self.adata.obs[self.cells_type_key], normalize="index"
        )
        self.patient_representations = self.patient_representations.loc[self.samples]

        distances = scipy.spatial.distance.pdist(self.patient_representations.values, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["composition_parameters"] = {"sample_key": self.sample_key, "distance_type": distance_metric}

        return distances


class SCellBOW(PatientsRepresentationMethod):
    """NLP based approach from https://www.biorxiv.org/content/10.1101/2022.12.28.522060v1.full.pdf"""

    DISTANCES_UNS_KEY = "X_scellbow"

    def __init__(
        self,
        sample_key,
        cells_type_key,
        model_dir="scellbow_model",
        n_worker=1,
        latent_dim: int = 300,
        n_iter: int = 20,
        layer=None,
        seed=67,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.model_dir = model_dir
        self.n_worker = n_worker
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.patient_representations = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 1, cluster_size_threshold: int = 0):
        """Pretrain SCellBOW model"""
        import SCellBOW as sb

        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        self.adata = self._move_layer_to_X()

        sb.SCellBOW_pretrain(
            self.adata, save_dir=self.model_dir, vec_size=self.latent_dim, n_worker=self.n_worker, iter=self.n_iter
        )

        self.adata = sb.SCellBOW_cluster(self.adata, self.model_dir).run()

    def calculate_distance_matrix(self, force: bool = False, average="mean"):
        """Calculate distances between patients"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        if average == "mean":
            func = np.mean
        elif average == "median":
            func = np.median
        else:
            raise ValueError(f"Averaging function {average} is not supported")

        # X_embbed contains 50 components PCA of SCellBOW cell embeddings
        cell_representations = self.adata.obsm["X_embed"]
        self.patient_representations = np.zeros(shape=(len(self.samples), cell_representations.shape[1]))

        for i, sample in enumerate(self.samples):
            sample_cells = cell_representations[self.adata.obs[self.sample_key] == sample, :]
            # Aggregate representations of cells for each sample
            self.patient_representations[i] = func(sample_cells, axis=0)

        distances = scipy.spatial.distance.pdist(self.patient_representations)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["scellbow_parameters"] = {
            "sample_key": self.sample_key,
            "distance_type": "euclidean",
            "latent_dim": self.latent_dim,
            "n_iter": self.n_iter,
        }

        return distances


class SCPoli(PatientsRepresentationMethod):
    """A semi-supervised conditional deep generative model from https://www.biorxiv.org/content/10.1101/2022.11.28.517803v1"""

    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    DISTANCES_UNS_KEY = "X_scpoli"

    def __init__(
        self,
        sample_key,
        cells_type_key,
        latent_dim=3,
        layer=None,
        seed=67,
        n_epochs: int = 50,
        pretraining_epochs: int = 40,
        eta: float = 5,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.latent_dim = latent_dim
        self.model = None
        self.patient_representation = None
        self.n_epochs = n_epochs
        self.pretraining_epochs = pretraining_epochs
        self.eta = eta

    def prepare_anndata(
        self, adata, sample_size_threshold: int = 1, cluster_size_threshold: int = 0, optimize_adata=True
    ):
        """Set up scPoli model"""
        from scarches.models.scpoli import scPoli

        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        self.adata = self._move_layer_to_X()

        if optimize_adata:
            self.adata = sc.AnnData(
                X=self.adata.X,
                obs=self.adata.obs[[self.sample_key, self.cells_type_key]],
                var=pd.DataFrame(index=self.adata.var_names),
            )

        assert is_count_data(self.adata.X), "`layer` must contain count data with integer numbers"

        self.model = scPoli(
            adata=self.adata,
            condition_keys=self.sample_key,
            cell_type_keys=self.cells_type_key,
            embedding_dims=self.latent_dim,
        )

        self.model.train(
            n_epochs=self.n_epochs,
            pretraining_epochs=self.pretraining_epochs,
            early_stopping_kwargs=self.early_stopping_kwargs,
            eta=self.eta,
        )

        self.patient_representation = self.model.get_conditional_embeddings().X

    def calculate_distance_matrix(self, force: bool = False, dist="euclidean"):
        """Calculate distances between scPoli sample embeddings"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distance_metric = valid_distance_metric(dist)

        distances = scipy.spatial.distance.pdist(self.patient_representation, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["scpoli_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "distance_type": distance_metric,
            "latent_dim": self.latent_dim,
            "n_epochs": self.n_epochs,
            "pretraining_epochs": self.pretraining_epochs,
            "eta": self.eta,
        }

        return distances


class PhEMD(PatientsRepresentationMethod):
    """Phenotypic Earth Mover's Distance. Source: https://pubmed.ncbi.nlm.nih.gov/31932777/

    Python implementation source: https://github.com/atong01/MultiscaleEMD/blob/main/comparison/phemd.py
    """

    DISTANCES_UNS_KEY = "X_phemd"

    def __init__(self, sample_key, cells_type_key, layer=None, n_clusters: int = 8, seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.n_clusters = n_clusters
        self.encoded_labels = None

    def prepare_anndata(
        self,
        adata,
        sample_size_threshold: int = 1,
        cluster_size_threshold: int = 0,
        subset_fraction: float = None,
        subset_n_obs: int = None,
        subset_min_obs_per_sample: int = 500,
    ):
        """Prepare anndata for PhEMD calculation. As computation is very slow, using subset of cells is recommended

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        sample_size_threshold : int = 1
            Minimum number of cells in a sample
        cluster_size_threshold : int = 0
            Minimum number of cells in a cluster
        subset_fraction : float = None
            Fraction of cells from each sample to use for PhEMD calculation
        subset_n_obs : int = None
            Number of cells from each sample to use for PhEMD calculation. Ignored if `subset_fraction` is set
        subset_min_obs_per_sample : int = 500
            Minimum number of cells per sample to use for PhEMD calculation
        """
        super().prepare_anndata(adata, sample_size_threshold, cluster_size_threshold)

        if subset_fraction is not None or subset_n_obs is not None:
            self.adata = subsample(
                self.adata,
                obs_category_col=self.cells_type_key,
                fraction=subset_fraction,
                n_obs=subset_n_obs,
                min_obs_per_category=subset_min_obs_per_sample,
            )

        # Convert labels to a format required by phemd implementation
        # The labels will be one-hot encoded and divided by the number of samples
        sc_labels_df = pd.get_dummies(self.adata.obs[self.sample_key])
        self.samples = sc_labels_df.columns
        self.encoded_labels = sc_labels_df.to_numpy()
        self.encoded_labels = self.encoded_labels / self.encoded_labels.sum(axis=0)

    def calculate_distance_matrix(self, force: bool = False, n_jobs=-1):
        """Calculate distances between samples"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distances = phemd(
            self._get_data(), self.encoded_labels, n_clusters=self.n_clusters, random_state=self.seed, n_jobs=n_jobs
        )

        self.adata.uns["phemd_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "n_clusters": self.n_clusters,
        }
        self.adata.uns[self.DISTANCES_UNS_KEY] = distances

        return distances


class DiffusionEarthMoverDistance(PatientsRepresentationMethod):
    """Diffusion Earth Mover's Distance. Source: https://arxiv.org/pdf/2102.12833"""

    DISTANCES_UNS_KEY = "X_diffusion_emd"

    def __init__(self, sample_key, cells_type_key, layer=None, seed=67, n_neighbors: int = 15, n_scales: int = 6):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)

        self.n_neighbors = n_neighbors
        self.n_scales = n_scales
        self.labels = None
        self.model = None
        self.patient_representations = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 1, cluster_size_threshold: int = 0):
        """Prepare anndata, calculate neighbors and convert labels to distributions as required by DiffusionEMD"""
        from DiffusionEMD import DiffusionCheb

        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        # Encode labels as one-hot and normalize them per sample
        samples_encoding = pd.get_dummies(self.adata.obs[self.sample_key])
        labels = samples_encoding.to_numpy().astype(int)
        self.labels = labels / labels.sum(axis=0)

        # Make sure that the order is correct
        self.samples = samples_encoding.columns

        sc.pp.neighbors(self.adata, use_rep=self.layer, method="gauss", n_neighbors=self.n_neighbors)

        self.adata.obsp["connectivities"] = make_matrix_symmetric(self.adata.obsp["connectivities"])

        self.model = DiffusionCheb(n_scales=self.n_scales)

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between samples"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        # Embeddings where the L1 distance approximates the Earth Mover's Distance
        self.patient_representations = self.model.fit_transform(self.adata.obsp["connectivities"], self.labels)
        distances = scipy.spatial.distance.pdist(self.patient_representations, metric="cityblock")
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["diffusion_emd_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "n_neighbors": self.n_neighbors,
            "n_scales": self.n_scales,
        }

        return self.adata.uns[self.DISTANCES_UNS_KEY]


class MOFA(PatientsRepresentationMethod):
    """Patient representation using MOFA2 model, treating patients as samples with optional cell type views."""

    DISTANCES_UNS_KEY = "X_mofa_distances"

    def __init__(
        self,
        sample_key,
        cells_type_key,
        layer=None,
        seed=67,
        n_factors=10,
        aggregate_cell_types: bool = False,
        **mofa_params,
    ):
        """
        Initialize the MOFA2Method class.

        Parameters
        ----------
        sample_key : str
            Column in .obs containing sample (patient) IDs.
        cells_type_key : str
            Column in .obs containing cell type information.
        layer : Optional[str] = None
            Layer in AnnData to use for gene expression data. If None, uses .X.
        seed : int = 67
            Random seed for reproducibility.
        n_factors : int = 10
            Number of latent factors to learn.
        aggregate_cell_types : bool = False
            If True, treat each cell type as a separate view. If False, aggregate gene expression across all cell types into a single view.
        mofa_params : dict
            Additional parameters for MOFA2.
        """
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, layer=layer, seed=seed)
        self.n_factors = n_factors
        self.aggregate_cell_types = aggregate_cell_types
        self.mofa_params = mofa_params
        self.model = None
        self.patient_representations = None
        self.views = None  # List of views (cell types) or single view
        self.cell_types = None

    def prepare_anndata(self, adata, sample_size_threshold=1, cluster_size_threshold=0):
        """
        Prepare AnnData for MOFA2, optionally treating cell types as separate views.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix.
        sample_size_threshold : int = 1
            Minimum number of cells per sample to retain.
        cluster_size_threshold : int = 0
            Minimum number of cells per cell type within a sample to retain.
        """
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        data = self._get_data()
        if scipy.sparse.issparse(data):
            data = data.toarray()

        genes = self.adata.var_names.astype(str)

        patient_ids = self.adata.obs[self.sample_key].astype(str).values

        # Create DataFrame from gene expression data
        data_df = pd.DataFrame(data, columns=genes)
        data_df["patient_id"] = patient_ids

        if self.aggregate_cell_types:
            cell_types = self.adata.obs[self.cells_type_key].astype(str).values
            data_df["cell_type"] = cell_types
            unique_patients = data_df["patient_id"].unique()
            self.samples = unique_patients.tolist()
            unique_cell_types = data_df["cell_type"].unique()
            self.cell_types = unique_cell_types.tolist()

            # Aggregate gene expression by patient and cell type using mean
            aggregated_data = data_df.groupby(["patient_id", "cell_type"]).mean()

            # Initialize list to store views
            views = []
            for cell_type in unique_cell_types:
                # Check if cell type exists in aggregated data
                if cell_type in aggregated_data.index.get_level_values("cell_type"):
                    # Extract data for the current cell type
                    cell_type_data = aggregated_data.xs(cell_type, level="cell_type")
                    # Reindex to include all patients, filling missing with zeros
                    cell_type_data = cell_type_data.reindex(unique_patients, fill_value=0)
                    # Convert to NumPy array (shape: n_patients x n_genes)
                    cell_type_matrix = cell_type_data.values
                    views.append(cell_type_matrix)
                else:
                    print(f"Cell type {cell_type} not found in aggregated data.")

            self.views = [[view_matrix] for view_matrix in views]  # List of NumPy arrays, one per cell type
        else:
            # Aggregate gene expression across all cell types for each patient using mean
            aggregated_data = data_df.groupby("patient_id").mean()
            self.samples = aggregated_data.index.tolist()
            data_matrix = aggregated_data.values  # Shape: (n_patients, n_genes)
            self.views = [[data_matrix]]  # Single view with one group

    def calculate_distance_matrix(self, force=False):
        """
        Calculate distances between patients using MOFA2 latent factors.

        Parameters
        ----------
        force : bool = False
            If True, recalculate the distance matrix even if it exists.

        Returns
        -------
        distances : np.ndarray
            Matrix of distances between patients.
        """
        from mofapy2.run.entry_point import entry_point

        distances = super().calculate_distance_matrix(force=force)
        if distances is not None:
            return distances

        ent = entry_point()

        ent.set_data_options(
            scale_groups=False,
            scale_views=True,
            center_groups=False,
        )

        if self.aggregate_cell_types:
            views_names = self.cell_types
        else:
            views_names = ["gene_expression"]

        # Set data matrix for MOFA2
        ent.set_data_matrix(
            data=self.views, samples_names=[self.samples], views_names=views_names, groups_names=["group1"]
        )

        # Set model options
        ent.set_model_options(
            factors=self.n_factors,
            ard_factors=True,
            ard_weights=True,
            spikeslab_weights=True,
        )

        # Set training options
        ent.set_train_options(seed=self.seed, verbose=True, **self.mofa_params)

        # Build and run the MOFA2 model
        ent.build()
        ent.run()

        # Retrieve the trained model
        self.model = ent.model

        expectations = self.model.getExpectations()

        factors_expectation = expectations["Z"]  # Dictionary with keys 'E' and 'V'
        factors_matrix = factors_expectation["E"]  # Shape: (n_patients, n_factors)

        self.patient_representations = factors_matrix

        distances = scipy.spatial.distance.pdist(self.patient_representations, metric="euclidean")
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["mofa_parameters"] = {
            "sample_key": self.sample_key,
            "n_factors": self.n_factors,
            "aggregate_cell_types": self.aggregate_cell_types,
            **self.mofa_params,
        }

        return distances
