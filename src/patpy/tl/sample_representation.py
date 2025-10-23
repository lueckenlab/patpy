import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from patpy.pp import (
    extract_metadata,
    fill_nan_distances,
    filter_small_samples,
    is_count_data,
    subsample,
)
from patpy.tl._types import _EVALUATION_METHODS

VALID_AGGREGATES = {"mean": np.mean, "median": np.median, "sum": np.sum}

VALID_DISTANCES = {"euclidean", "cosine", "cityblock"}


def valid_aggregate(aggregate: str) -> Callable[[np.ndarray, ...], np.ndarray]:
    """Returns a valid aggregation function or raises an error if invalid

    Parameters
    ----------
    aggregate : str
        Name of aggregation function to use. One of: "mean", "median", "sum"

    Returns
    -------
    Callable
        Numpy aggregation function

    Raises
    ------
    ValueError
        If aggregation function is not supported
    """
    if aggregate not in VALID_AGGREGATES:
        raise ValueError(f"Aggregation function '{aggregate}' is not supported")
    return VALID_AGGREGATES[aggregate]


def valid_distance_metric(dist: str):
    """Returns if the distance metric is valid or raises an error"""
    if dist not in VALID_DISTANCES:
        raise ValueError(f"Distance metric '{dist}' is not supported")
    return dist


def make_matrix_symmetric(matrix):
    """Make a matrix symmetric by averaging it with its transpose.

    Parameters
    ----------
    matrix : np.ndarray or scipy.sparse.spmatrix
        The input matrix to be made symmetric.

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
    color_map = dict(zip(unique_values, colors, strict=False))
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


def correlate_composition(meta_adata, expression_adata, sample_key, cell_type_key, target, method="spearman"):
    """
    Correlate cell type composition with a target variable.

    Parameters
    ----------
    meta_adata : AnnData
        AnnData object containing metadata for each sample.
    expression_adata : AnnData
        AnnData object containing gene expression data.
    sample_key : str
        Key in expression_adata.obs for sample identifiers.
    cell_type_key : str
        Key in expression_adata.obs for cell type annotations.
    target : str
        Key in meta_adata.obs for the target variable to correlate with.
    method : str, optional
        Correlation method to use. Either "spearman" (default) or "pearson".

    Returns
    -------
    pd.DataFrame
        DataFrame containing correlation results for each cell type. It contains the following columns:
        - "correlation": Correlation coefficient between cell type proportion and target variable
        - "p_value": Raw p-value for the correlation
        - "p_value_adj": Adjusted p-value after Benjamini-Hochberg correction
        - "-log_p_value_adj": negative logarithm of adjusted p-value
    """
    # Select the correlation function
    if method == "spearman":
        correlation_fun = spearmanr
    elif method == "pearson":
        correlation_fun = pearsonr
    else:
        raise ValueError('Method must be either "spearman" or "pearson"')

    # Calculate cell type composition using patpy tool
    composition = CellGroupComposition(sample_key, cell_type_key)
    composition.prepare_anndata(expression_adata)
    _ = (
        composition.calculate_distance_matrix()
    )  # We don't need distance matrix but this method calculates cell type proportions as well

    cell_type_fractions = composition.sample_representation
    cell_type_fractions = cell_type_fractions.loc[
        meta_adata.obs_names
    ]  # make sure that the order is the same as in input data

    cell_type_corrs = {}

    for cell_type in cell_type_fractions.columns:
        correlation, p_value = correlation_fun(meta_adata.obs[target], cell_type_fractions[cell_type])
        cell_type_corrs[cell_type] = {"correlation": correlation, "p_value": p_value}

    cell_type_corrs = pd.DataFrame(cell_type_corrs).T

    # Perform Benjamini-Hochberg correction
    cell_type_corrs["p_value_adj"] = multipletests(cell_type_corrs["p_value"], method="fdr_bh")[1]

    cell_type_corrs["-log_p_value_adj"] = -np.log(cell_type_corrs["p_value_adj"])

    # Sort the DataFrame by adjusted p-value
    cell_type_corrs = cell_type_corrs.sort_values(["p_value_adj", "correlation"], ascending=[True, False])

    return cell_type_corrs


def correlate_cell_type_expression(
    meta_adata,
    expression_adata,
    sample_key,
    cell_type_key,
    target,
    layer="X",
    min_sample_size=50,
    method="spearman",
    keep_pseudobulks_in_data=True,
):
    """
    Calculate correlation between gene expression and a target variable for each cell type.

    Parameters
    ----------
    meta_adata : AnnData
        AnnData object containing metadata and target variable.
    expression_adata : AnnData
        AnnData object containing gene expression data.
    sample_key : str
        Key in adata.obs for sample information.
    cell_type_key : str
        Key in adata.obs for cell type information.
    target : str
        Column name in meta_adata.obs containing the target variable.
    layer : str
        slot in .obsm or .layers of `expression_adata` to use for getting pseudobulks. Default is "X" to use .X
    min_sample_size : int, optional
        Minimum number of cells required for a sample to be included. Default is 50.
    method : str, optional
        Correlation method to use. Either "spearman" or "pearson". Default is "spearman".
    keep_pseudobulks_in_data : bool, optional
        If True (default), keep cell type pseudobulks in the meta_adata. They will be stored in .obsm slot
        with the name <cell_type>_pseudobulk

    Returns
    -------
    pd.DataFrame
        DataFrame containing correlation results for each cell type and gene. It contains the following columns:
        - cell_type: The cell type
        - gene_name: The gene name
        - correlation: Correlation coefficient between gene expression and target variable
        - p_value: Raw p-value for the correlation
        - n_observations: Number of observations used for the correlation
        - "-log_p_value_adj": negative logarithm of adjusted p-value
    """
    # Select the correlation function
    if method == "spearman":
        correlation_fun = spearmanr
    elif method == "pearson":
        correlation_fun = pearsonr
    else:
        raise ValueError('Method must be either "spearman" or "pearson"')

    if min_sample_size is not None and min_sample_size > 0:
        expression_adata = filter_small_samples(expression_adata, sample_key, min_sample_size)

    cell_type_pseudobulk = GroupedPseudobulk(sample_key, cell_type_key, layer=layer)
    cell_type_pseudobulk.prepare_anndata(expression_adata)
    _ = cell_type_pseudobulk.calculate_distance_matrix()

    expression_correlations = []

    for i, cell_type in enumerate(cell_type_pseudobulk.cell_groups):
        pseudobulks = cell_type_pseudobulk.sample_representation[i]

        if keep_pseudobulks_in_data:
            meta_adata.obsm[f"{cell_type}_pseudobulk"] = pd.DataFrame(
                pseudobulks, index=cell_type_pseudobulk.samples, columns=expression_adata.var_names
            ).loc[meta_adata.obs_names]

        # Get the target values for the samples. Always make sure that the order is the same!
        target_values = meta_adata.obs.loc[cell_type_pseudobulk.samples, target].values

        # Calculate correlation for each gene
        for gene_idx, gene_name in enumerate(
            expression_adata.var_names
        ):  # TODO: potential bug when a layer with different n features is used
            gene_expression = pseudobulks[:, gene_idx]
            correlation, p_value = correlation_fun(gene_expression, target_values, nan_policy="omit")

            # Save the sample size. Nans appear when a cell type in a sample doesn't have any cells
            n_observations = (~np.isnan(gene_expression)).sum()
            expression_correlations.append((cell_type, gene_name, correlation, p_value, n_observations))

    # Convert the results to a DataFrame
    expression_correlation_df = pd.DataFrame(
        expression_correlations, columns=["cell_type", "gene_name", "correlation", "p_value", "n_observations"]
    )

    expression_correlation_df = expression_correlation_df[expression_correlation_df["correlation"].notna()]

    # Calculate adjusted p-values
    expression_correlation_df["p_value_adj"] = multipletests(expression_correlation_df["p_value"], method="fdr_bh")[1]

    expression_correlation_df["-log_p_value_adj"] = -np.log(expression_correlation_df["p_value_adj"])

    # Sort the DataFrame by p-value (ascending) and correlation (descending)
    expression_correlation_df = expression_correlation_df.sort_values(
        ["p_value_adj", "correlation"], ascending=[True, False]
    )

    return expression_correlation_df


class SampleRepresentationMethod:
    """Base class for sample representation methods"""

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

        # getting only those layers with the same shape of the new X matrix from adata.layers[self.layer] to be copied in the new anndata below
        filtered_layers = {
            key: np.copy(layer)
            for key, layer in self.adata.layers.items()
            if key != self.layer and layer.shape == self.adata.layers[self.layer].shape
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

    def __init__(self, sample_key, cell_group_key, layer=None, seed=67):
        """Initialize the model

        Parameters
        ----------
        sample_key : str
            Column in .obs containing sample IDs
        cell_group_key : str
            Column in .obs containing cell group key (for example, cell type)
        layer : Optional[str] = None
            What to use as data in a model. If None or "X", `adata.X` is used. Otherwise, the corresponding key from `adata.obsm` will be used
        seed : int = 67
            Number to initialize pseudorandom generator
        """
        self.sample_key = sample_key
        self.cell_group_key = cell_group_key
        self.layer = layer
        self.seed = seed

        self.adata = None
        self.samples = None
        self.cell_groups = None
        self.embeddings = {}
        self.samples_adata = None

    # fit-like method: save data and process it
    def prepare_anndata(self, adata):
        """fit-like method: prepare adata for the analysis"""
        self.adata = adata

        self.samples = self.adata.obs[self.sample_key].unique()

        if self.cell_group_key is not None and self.cell_group_key in self.adata.obs:
            self.cell_groups = self.adata.obs[self.cell_group_key].unique()

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
            self.sample_representation is not None
            and self.sample_representation.ndim == 2
            and self.sample_representation.shape[0] == len(self.samples)
        ):
            representation = self.sample_representation
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
        axes=None,
    ):
        """Plot embedding of samples colored by `metadata_cols`"""
        import matplotlib.pyplot as plt

        if method not in self.embeddings:
            self.embed(method=method)

        embedding_df = pd.DataFrame(self.embeddings[method], columns=[f"{method}_0", f"{method}_1"], index=self.samples)

        if metadata_cols is None:
            # Simply plot the embedding
            if axes is None:
                axes = sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1")
            else:
                sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1", ax=axes)
        else:
            # Colorize samples by metadata
            metadata_df = self._extract_metadata(columns=metadata_cols)
            embedding_df = pd.concat([embedding_df, metadata_df], axis=1)

            if axes is None:
                _, axes = plt.subplots(
                    nrows=1, ncols=len(metadata_cols), sharey=True, figsize=(len(metadata_cols) * 5, 5)
                )
            else:
                axes = axes.flatten() if isinstance(axes, np.ndarray) else axes

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
            A sample-level covariate to evaluate representation for
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
        from patpy.tl.evaluation import evaluate_representation

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

        for col, task in zip(metadata_columns, tasks, strict=False):
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
            ax.set_title(f"{col}: {round(row['score'], 4)}")
            ax.legend(loc=(1.05, 0))

        return results

    def _get_pseudobulk(
        self,
        aggregation: str,
        fill_value,
        aggregate_cell_types=True,
        sample_key=None,
        cell_group_key=None,
        samples=None,
        cell_groups=None,
    ):
        """
        Generate pseudobulk data by aggregating gene expression data per patient and optionally per cell type.

        Parameters
        ----------
        aggregation : str
            Name of the aggregation function to use (e.g., 'mean', 'median', 'sum').
        fill_value : float
            Value to use for missing data (e.g., np.nan for CellTypePseudobulk and MOFA).
        aggregate_cell_types : bool
            If True, aggregate by both sample and cell type. If False, aggregate only by sample.
        sample_key : str, optional
            Key in `adata.obs` for sample (patient) IDs. Defaults to `self.sample_key`.
        cell_group_key : str, optional
            Key in `adata.obs` for cell groups. Defaults to `self.cell_group_key`.
        samples : list, optional
            List of sample IDs. Defaults to `self.samples`.
        cell_groups : list, optional
            List of cell groups. Defaults to `self.cell_groups`.

        Returns
        -------
        numpy.ndarray or list of numpy.ndarray
            Pseudobulk data, either as a 3D array (for each cell type) or 2D array (for each patient).
        """
        aggregation_func = valid_aggregate(aggregation)

        sample_key = sample_key or self.sample_key
        cell_group_key = cell_group_key or self.cell_group_key
        samples = samples or self.samples
        cell_groups = cell_groups or self.cell_groups

        data = self._get_data()

        if aggregate_cell_types:
            pseudobulk_data = np.zeros(shape=(len(cell_groups), len(samples), data.shape[1]))

            for i, cell_group in enumerate(cell_groups):
                for j, sample in enumerate(samples):
                    cells_data = data[
                        (self.adata.obs[sample_key] == sample) & (self.adata.obs[cell_group_key] == cell_group)
                    ]

                    if cells_data.size == 0:
                        pseudobulk_data[i, j] = fill_value
                    else:
                        pseudobulk_data[i, j] = aggregation_func(cells_data, axis=0)

            return pseudobulk_data
        else:
            pseudobulk_data = np.zeros(shape=(len(samples), data.shape[1]))

            for j, sample in enumerate(samples):
                cells_data = data[self.adata.obs[sample_key] == sample]

                if cells_data.size == 0:
                    pseudobulk_data[j] = fill_value
                else:
                    pseudobulk_data[j] = aggregation_func(cells_data, axis=0)

            return pseudobulk_data


class MrVI(SampleRepresentationMethod):
    """Deep generative modeling for quantifying sample-level heterogeneity in single-cell omics.

    Source: https://www.biorxiv.org/content/10.1101/2022.10.04.510898v2
    """

    DISTANCES_UNS_KEY = "X_mrvi_distances"

    def __init__(
        self,
        sample_key: str,
        cell_group_key: str,
        batch_key: str = None,
        layer=None,
        seed=67,
        max_epochs=400,
        **model_params,
    ):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.model = None
        self.model_params = model_params
        self.sample_representation = None
        self.max_epochs = max_epochs
        self.batch_key = batch_key

    def prepare_anndata(self, adata):
        """Train MrVI model

        Parameters
        ----------
        adata : AnnData object with raw counts in .X

        Sets
        ----
        model : MrVI model
        """
        from scvi.external import MRVI

        super().prepare_anndata(adata=adata)

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

            self.sample_representation = np.zeros(shape=(len(self.samples), cell_sample_representations.shape[2]))

            print("Calculating samples representations")
            # For a sample representation we will take centroid of cells of this sample
            for i, sample in enumerate(self.samples):
                sample_mask = self.adata.obs[self.sample_key] == sample
                self.sample_representation[i] = cell_sample_representations[sample_mask, i].mean(axis=0)

            # Here, we obtain distances between samples in a different way
            # MrVI calculates sample-sample distances per cell and then aggregates them (see below)
            # Here, we first aggregate cells and then calculate sample-sample distances. Note that it produces different results
            print(
                f"Using aggregated cell representation approach, distances are stored in self.adata.uns[{self.DISTANCES_UNS_KEY}_cell_based"
            )
            distances = scipy.spatial.distance.pdist(self.sample_representation)
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


class WassersteinTSNE(SampleRepresentationMethod):
    """Method based on the matrix of pairwise Wasserstein distances between units.

    Source: https://arxiv.org/abs/2205.07531
    """

    DISTANCES_UNS_KEY = "X_wasserstein_distances"

    def __init__(self, sample_key, cell_group_key, replicate_key, layer="X_scvi", seed=67):
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
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.replicate_key = replicate_key

        self.model = None
        self.distances_model = None

    def prepare_anndata(self, adata):
        """Set up Gaussian Wasserstein Distance model"""
        import WassersteinTSNE as WT

        super().prepare_anndata(adata=adata)

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


class PILOT(SampleRepresentationMethod):
    """Optimal transport based method to compute the Wasserstein distance between two single single-cell experiments.

    Source: https://www.biorxiv.org/content/10.1101/2022.12.16.520739v1
    """

    DISTANCES_UNS_KEY = "X_pilot_distances"

    def __init__(
        self,
        sample_key,
        cell_group_key,
        sample_state_col,
        dataset_name="pilot_dataset",
        layer="X_pca",
        seed=67,
    ):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.sample_state_col = sample_state_col
        self.dataset_name = dataset_name

        self.results_dir = None
        self.pc = None
        self.annotation = None
        self.sample_representation = None

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
            clusters_col=self.cell_group_key,
            sample_col=self.sample_key,
            status=self.sample_state_col,
            emb_matrix=self.layer,
            data_type="scRNA",
            **pilot_parameters,
        )

        # Matrix of cell group proportions for each sample
        self.sample_representation = (
            pd.DataFrame(self.adata.uns["proportions"], index=self.cell_groups).T.loc[self.samples].to_numpy()
        )

        distances = self.adata.uns["EMD_df"].loc[self.samples, self.samples].to_numpy()
        distances = make_matrix_symmetric(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["pilot_parameters"] = {
            "sample_key": self.sample_key,
            "cell_group_key": self.cell_group_key,
            **pilot_parameters,
        }
        return distances


class Pseudobulk(SampleRepresentationMethod):
    """A simple baseline, which represents samples as pseudobulk of their gene expression"""

    DISTANCES_UNS_KEY = "X_pseudobulk_distances"

    def __init__(self, sample_key, cell_group_key, layer="X_pca", seed=67):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.sample_representation = None

    def calculate_distance_matrix(self, force: bool = False, aggregate="mean", dist="euclidean"):
        """Calculate distances between pseudobulk representations of samples"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        aggregation_func = valid_aggregate(aggregate)
        distance_metric = valid_distance_metric(dist)

        data = self._get_data()

        self.sample_representation = np.zeros(shape=(len(self.samples), data.shape[1]))

        for i, sample in enumerate(self.samples):
            sample_cells = data[self.adata.obs[self.sample_key] == sample, :]
            self.sample_representation[i] = aggregation_func(sample_cells, axis=0)

        distances = scipy.spatial.distance.pdist(self.sample_representation, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["bulk_parameters"] = {
            "sample_key": self.sample_key,
            "aggregate": aggregate,
            "distance_type": distance_metric,
        }

        return distances


class GroupedPseudobulk(SampleRepresentationMethod):
    """Baseline, where distances between samples are average distances between their cell group pseudobulks"""

    DISTANCES_UNS_KEY = "X_ct_pseudobulk_distances"

    def __init__(self, sample_key, cell_group_key, layer="X_pca", seed=67):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.sample_representation = None

    def calculate_distance_matrix(self, force: bool = False, aggregate="mean", dist="euclidean"):
        """Calculate distances between samples as average distance between per cell-type pseudobulks"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distance_metric = valid_distance_metric(dist)

        self.sample_representation = self._get_pseudobulk(
            aggregation=aggregate, fill_value=np.nan, aggregate_cell_types=True
        )

        # Matrix of distances between samples for each cell group
        distances = np.zeros(shape=(len(self.cell_groups), len(self.samples), len(self.samples)))

        for i, cell_group_embeddings in enumerate(self.sample_representation):
            samples_distances = scipy.spatial.distance.pdist(cell_group_embeddings, metric=distance_metric)
            distances[i] = scipy.spatial.distance.squareform(samples_distances)

        avg_distances, sample_sizes = calculate_average_without_nans(distances, axis=0)

        self.adata.uns[self.DISTANCES_UNS_KEY] = avg_distances
        self.adata.uns["celltypebulk_parameters"] = {
            "sample_key": self.sample_key,
            "cell_group_key": self.cell_group_key,
            "aggregate": aggregate,
            "distance_type": distance_metric,
            "sample_sizes": sample_sizes,
        }

        return avg_distances


class RandomVector(SampleRepresentationMethod):
    """A dummy baseline, which represents samples as random embeddings"""

    DISTANCES_UNS_KEY = "X_random_vector_distances"

    def __init__(self, sample_key, cell_group_key, latent_dim: int = 30, seed=67):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, seed=seed)

        self.latent_dim = latent_dim
        self.sample_representation = None

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between samples represented as random vectors"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        self.sample_representation = np.random.normal(size=(len(self.samples), self.latent_dim))

        distances = scipy.spatial.distance.pdist(self.sample_representation)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["random_vec_parameters"] = {
            "sample_key": self.sample_key,
        }

        return distances


class CellGroupComposition(SampleRepresentationMethod):
    """A simple baseline, which represents samples as composition of their cell groups (for example, cell type fractions)"""

    DISTANCES_UNS_KEY = "X_composition"

    def __init__(self, sample_key, cell_group_key, layer=None, seed=67):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.sample_representation = None

    def calculate_distance_matrix(self, force: bool = False, dist="euclidean"):
        """Calculate distances between samples represented as cell group composition vectors"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distance_metric = valid_distance_metric(dist)

        # Calculate proportions of the cell groups for each sample
        self.sample_representation = pd.crosstab(
            self.adata.obs[self.sample_key], self.adata.obs[self.cell_group_key], normalize="index"
        )
        self.sample_representation = self.sample_representation.loc[self.samples]

        distances = scipy.spatial.distance.pdist(self.sample_representation.values, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["composition_parameters"] = {"sample_key": self.sample_key, "distance_type": distance_metric}

        return distances


class SCPoli(SampleRepresentationMethod):
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
        cell_group_key,
        latent_dim=3,
        layer=None,
        seed=67,
        n_epochs: int = 50,
        pretraining_epochs: int = 40,
        eta: float = 5,
    ):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.latent_dim = latent_dim
        self.model = None
        self.sample_representation = None
        self.n_epochs = n_epochs
        self.pretraining_epochs = pretraining_epochs
        self.eta = eta

    def prepare_anndata(self, adata, optimize_adata=True):
        """Set up scPoli model"""
        from scarches.models.scpoli import scPoli

        super().prepare_anndata(adata=adata)

        self.adata = self._move_layer_to_X()

        if optimize_adata:
            self.adata = sc.AnnData(
                X=self.adata.X,
                obs=self.adata.obs[[self.sample_key, self.cell_group_key]],
                var=pd.DataFrame(index=self.adata.var_names),
            )

        assert is_count_data(self.adata.X), "`layer` must contain count data with integer numbers"

        self.model = scPoli(
            adata=self.adata,
            condition_keys=self.sample_key,
            cell_type_keys=self.cell_group_key,
            embedding_dims=self.latent_dim,
        )

        self.model.train(
            n_epochs=self.n_epochs,
            pretraining_epochs=self.pretraining_epochs,
            early_stopping_kwargs=self.early_stopping_kwargs,
            eta=self.eta,
        )

        self.sample_representation = self.model.get_conditional_embeddings().X

    def calculate_distance_matrix(self, force: bool = False, dist="euclidean"):
        """Calculate distances between scPoli sample embeddings"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distance_metric = valid_distance_metric(dist)

        distances = scipy.spatial.distance.pdist(self.sample_representation, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["scpoli_parameters"] = {
            "sample_key": self.sample_key,
            "cell_group_key": self.cell_group_key,
            "distance_type": distance_metric,
            "latent_dim": self.latent_dim,
            "n_epochs": self.n_epochs,
            "pretraining_epochs": self.pretraining_epochs,
            "eta": self.eta,
        }

        return distances


class PhEMD(SampleRepresentationMethod):
    """Phenotypic Earth Mover's Distance. Source: https://pubmed.ncbi.nlm.nih.gov/31932777/

    Python implementation source: https://github.com/atong01/MultiscaleEMD/blob/main/comparison/phemd.py
    """

    DISTANCES_UNS_KEY = "X_phemd"

    def __init__(self, sample_key, cell_group_key, layer=None, n_clusters: int = 8, seed=67):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.n_clusters = n_clusters
        self.encoded_labels = None

    def prepare_anndata(
        self,
        adata,
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
        subset_fraction : float = None
            Fraction of cells from each sample to use for PhEMD calculation
        subset_n_obs : int = None
            Number of cells from each sample to use for PhEMD calculation. Ignored if `subset_fraction` is set
        subset_min_obs_per_sample : int = 500
            Minimum number of cells per sample to use for PhEMD calculation
        """
        super().prepare_anndata(adata=adata)

        if subset_fraction is not None or subset_n_obs is not None:
            self.adata = subsample(
                self.adata,
                obs_category_col=self.cell_group_key,
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
            "cell_group_key": self.cell_group_key,
            "n_clusters": self.n_clusters,
        }
        self.adata.uns[self.DISTANCES_UNS_KEY] = distances

        return distances


class DiffusionEarthMoverDistance(SampleRepresentationMethod):
    """Diffusion Earth Mover's Distance. Source: https://arxiv.org/pdf/2102.12833"""

    DISTANCES_UNS_KEY = "X_diffusion_emd"

    def __init__(self, sample_key, cell_group_key, layer=None, seed=67, n_neighbors: int = 15, n_scales: int = 6):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)

        self.n_neighbors = n_neighbors
        self.n_scales = n_scales
        self.labels = None
        self.model = None
        self.sample_representation = None

    def prepare_anndata(self, adata):
        """Prepare anndata, calculate neighbors and convert labels to distributions as required by DiffusionEMD"""
        from DiffusionEMD import DiffusionCheb

        super().prepare_anndata(adata=adata)

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
        self.sample_representation = self.model.fit_transform(self.adata.obsp["connectivities"], self.labels)
        distances = scipy.spatial.distance.pdist(self.sample_representation, metric="cityblock")
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["diffusion_emd_parameters"] = {
            "sample_key": self.sample_key,
            "cell_group_key": self.cell_group_key,
            "n_neighbors": self.n_neighbors,
            "n_scales": self.n_scales,
        }

        return self.adata.uns[self.DISTANCES_UNS_KEY]


class MOFA(SampleRepresentationMethod):
    """
    Patient representation using MOFA2 model, treating patients as samples with optional cell type views.

    Parameters
    ----------
    sample_key : str
        Column in `.obs` containing sample (patient) IDs.
    cell_group_key : str
        Column in `.obs` containing cell type information.
    layer : Optional[str], default: None
        Layer in AnnData to use for gene expression data. If None, uses `.X`.
    seed : int, default: 67
        Random seed for reproducibility.
    n_factors : int, default: 10
        Number of latent factors to learn.
    aggregate_cell_types : bool, default: True
        If True, treat each cell type as a separate view.
        If False, aggregate gene expression across all cell types into a single view.
    aggregation_mode: str, default: "mean"
        Name of the aggregation function to use (e.g., 'mean', 'median', 'sum')
    scale_views : bool, optional
        Scale each view to unit variance.
    scale_groups : bool, default: False
        Scale each group to unit variance.
    center_groups : bool, default: True
        Center each group.
    use_float32 : bool, default: False
        Use 32-bit floating point precision.
    ard_factors : bool, default: False
        Use Automatic Relevance Determination (ARD) prior on factors.
    ard_weights : bool, default: True
        Use ARD prior on weights.
    spikeslab_weights : bool, default: True
        Use spike-and-slab prior on weights.
    spikeslab_factors : bool, default: False
        Use spike-and-slab prior on factors.
    iterations : int, default: 1000
        Maximum number of training iterations.
    convergence_mode : {'fast', 'medium', 'slow'}, default: 'fast'
        Convergence speed mode.
    startELBO : int, default: 1
        Iteration number to start computing the Evidence Lower Bound (ELBO).
    freqELBO : int, default: 1
        Frequency of ELBO computation after `startELBO`.
    gpu_mode : bool, default: False
        Use GPU for training.
    gpu_device : Optional[int], default: None
        GPU device ID to use.
    verbose : bool, default: False
        Verbose output during training.
    quiet : bool, default: False
        Suppress training output.
    outfile : Optional[str], default: None
        Path to save the trained model.
    save_interrupted : bool, default: False
        Save the model if training is interrupted.

    """

    DISTANCES_UNS_KEY = "X_mofa_distances"

    def __init__(
        self,
        sample_key: str,
        cell_group_key: str,
        layer: str | None = None,
        seed: int = 67,
        n_factors: int = 10,
        aggregate_cell_types: bool = True,
        aggregation_mode: str = "mean",
        scale_views: bool = False,
        scale_groups: bool = False,
        center_groups: bool = True,
        use_float32: bool = False,
        ard_factors: bool = False,
        ard_weights: bool = True,
        spikeslab_weights: bool = True,
        spikeslab_factors: bool = False,
        iterations: int = 1000,
        convergence_mode: str = "fast",
        startELBO: int = 1,
        freqELBO: int = 1,
        gpu_mode: bool = False,
        gpu_device: int | None = None,
        verbose: bool = False,
        quiet: bool = False,
        outfile: str | None = None,
        save_interrupted: bool = False,
    ):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)
        self.n_factors = n_factors
        self.aggregate_cell_types = aggregate_cell_types
        self.model = None
        self.sample_representation = None
        self.views = None  # List of views (cell types) or single view
        self.views_names = None
        self.aggregation_mode = aggregation_mode

        self.data_options = {
            "scale_views": scale_views,
            "scale_groups": scale_groups,
            "center_groups": center_groups,
            "use_float32": use_float32,
        }

        self.model_options = {
            "factors": self.n_factors,
            "ard_factors": ard_factors,
            "ard_weights": ard_weights,
            "spikeslab_weights": spikeslab_weights,
            "spikeslab_factors": spikeslab_factors,
        }

        self.train_options = {
            "iter": iterations,
            "convergence_mode": convergence_mode,
            "startELBO": startELBO,
            "freqELBO": freqELBO,
            "gpu_mode": gpu_mode,
            "gpu_device": gpu_device,
            "seed": self.seed,
            "verbose": verbose,
            "quiet": quiet,
            "outfile": outfile,
            "save_interrupted": save_interrupted,
        }

    def prepare_anndata(self, adata):
        """
        Prepare AnnData for MOFA2, optionally treating cell types as separate views.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix
        """
        from mofapy2.run.entry_point import entry_point

        super().prepare_anndata(adata=adata)

        if self.aggregate_cell_types:
            # Aggregate by BOTH sample and cell type
            pseudobulk_data = self._get_pseudobulk(
                aggregation=self.aggregation_mode, fill_value=np.nan, aggregate_cell_types=True
            )
            self.views = [[view_matrix] for view_matrix in pseudobulk_data]  # -> multiple  celltype view appraoch
            self.views_names = self.cell_groups
        else:
            # Aggregate ONLY by patient
            pseudobulk_data = self._get_pseudobulk(
                aggregation=self.aggregation_mode, fill_value=np.nan, aggregate_cell_types=False
            )
            self.views = [[pseudobulk_data]]  # -> single view appraoch
            self.views_names = ["aggregated_gene_expression"]

        ent = entry_point()

        ent.set_data_options(**self.data_options)

        ent.set_data_matrix(
            data=self.views,
            samples_names=[self.samples],
            views_names=self.views_names,
            groups_names=[
                "group1"
            ],  # All patients are considered as a single group; no group-specific modeling is needed
        )

        ent.set_model_options(**self.model_options)

        ent.set_train_options(**self.train_options)

        ent.build()
        ent.run()

        self.model = ent.model

    def calculate_distance_matrix(self, force=False, store_weights=False, dist="euclidean"):
        """
        Calculate distances between patients using MOFA2 latent factors.

        Parameters
        ----------
        force : bool = False
            If True, recalculate the distance matrix even if it exists.
        store_weights : bool, default: False
            If True, store the weights (relation of factors to genes) in `self.adata.uns`.

        Returns
        -------
        distances : np.ndarray
            Matrix of distances between patients.
        """
        distances = super().calculate_distance_matrix(force=force)
        if distances is not None:
            return distances

        distance_metric = valid_distance_metric(dist)

        # get factors expectation (latent representations of samples)
        self.sample_representation = self.model.nodes["Z"].getExpectation()  # Shape: (n_patients, n_factors)

        # store weights (relation of factors to genes)
        if store_weights:
            weights = self.model.nodes["W"].getExpectation()
            # weights is a list with one matrix per view
            if self.aggregate_cell_types:
                mofa_weights = {view_name: weights[i] for i, view_name in enumerate(self.views_names)}
            else:
                mofa_weights = weights[0]

        distances = scipy.spatial.distance.pdist(self.sample_representation, metric=distance_metric)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["mofa_parameters"] = {
            "sample_key": self.sample_key,
            "n_factors": self.n_factors,
            "aggregate_cell_types": self.aggregate_cell_types,
            **self.data_options,
            **self.model_options,
            **self.train_options,
        }
        if store_weights:
            self.adata.uns["mofa_parameters"]["weights"] = mofa_weights

        return distances


class GloScope(SampleRepresentationMethod):
    """A class that loads a file to R using rpy2 and follows the same interface as other SampleRepresentation methods"""

    DISTANCES_UNS_KEY = "X_gloscope_distances"

    def __init__(
        self, sample_key, cell_group_key=None, layer=None, seed=67, dist_mat="KL", dens="KNN", k=25, n_workers=1
    ):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)
        self.dist_mat = dist_mat
        self.dens = dens
        self.k = k
        self.sample_representation = None
        self.n_workers = n_workers

    def prepare_anndata(self, adata):
        """Prepare anndata for GloScope calculation"""
        from rpy2 import robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.packages import importr

        with (robjects.default_converter + numpy2ri.converter + pandas2ri.converter).context():
            super().prepare_anndata(adata=adata)

        # Load the R packages
        robjects.r("library(GloScope)")
        importr("BiocParallel")

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between samples represented as GloScope embeddings"""
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri, pandas2ri
        from rpy2.robjects.vectors import StrVector

        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        embedding_df = pd.DataFrame(self._get_data(), index=self.adata.obs_names)

        with (robjects.default_converter + numpy2ri.converter + pandas2ri.converter).context():
            # Assign embedding and sample IDs to R environment
            robjects.globalenv["embedding_df"] = pandas2ri.py2rpy(embedding_df)
            robjects.globalenv["sample_ids"] = StrVector(self.adata.obs[self.sample_key].values)

            print("Calculating GloScope distance matrix")

            # Call GloScope function in R
            robjects.r(
                f"""
            dist_matrix <- gloscope(
                embedding_df,
                sample_ids,
                dens = '{self.dens}',
                dist_mat = '{self.dist_mat}',
                k = {self.k},
                BPPARAM = BiocParallel::MulticoreParam(workers = {self.n_workers}, RNGseed = {self.seed})
            )
            """
            )

            # Retrieve the distance matrix from R environment
            distances = robjects.r("as.data.frame(dist_matrix)")

        # Sometimes, gloscope produces small negative distances
        # According to developers, they can be treated as zeros: https://github.com/epurdom/GloScope/issues/3
        distances[distances < 0] = 0

        self.sample_representation = distances
        self.samples = list(self.sample_representation.index)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances.to_numpy()

        return distances.to_numpy()


class GloScope_py(SampleRepresentationMethod):
    """GloScope implementation in Python for CPU and GPU

    Source publication: https://doi.org/10.1186/s13059-024-03398-1
    """

    def __init__(self, sample_key, cell_group_key=None, layer="X_pca", seed=67, k=25, use_gpu=False, n_components=None):
        super().__init__(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer, seed=seed)
        self.k = k
        self.use_gpu = use_gpu
        self.n_components = n_components

        if self.use_gpu:
            self.DISTANCES_UNS_KEY = "X_gloscope_cuml_distances"
        else:
            self.DISTANCES_UNS_KEY = "X_gloscope_pynndescent_distances"

    @staticmethod
    def kl_divergence(r_i, r_j, m_i, m_j, d) -> float:
        """
        Calculates KL(H_i || H_j) (Kullback-Leibler divergence) based on pre-calculated kNN distances.

        The formula is taken from the paper "Visualizing scRNA-Seq data at population scale with GloScope"
        (https://doi.org/10.1186/s13059-024-03398-1).

        Parameters
        ----------
        r_i : np.ndarray
            k-nearest neighbor distances of samples in H_i from points in H_i itself.
        r_j : np.ndarray
            k-nearest neighbor distances of samples in H_i from points in H_j.
        m_i : int
            Number of samples in H_i.
        m_j : int
            Number of samples in H_j.
        d : int
            Dimensionality of the data.

        Returns
        -------
        float
            The KL divergence KL(H_i || H_j).
        """
        # Logarithm of the ratio of the kNN distances
        log_ratios = np.log(r_j / r_i)

        # Gloscope formula for KL divergence
        kl = (d / m_i) * np.sum(log_ratios) + np.log(m_j / (m_i - 1))

        return kl

    def calculate_distance_matrix_pynndescent(self):
        """
        Calculates the symmetric Kullback-Leibler divergence using approximate kNN distances.

        Parameters
        ----------
        self.adata : AnnData
            The AnnData object stored in the class instance.
        self.sample_key : str
            Column in `.obs` containing sample (patient) IDs.
        self.k: int, default: 25
            Number of nearest neighbors for k-nearest neighbor calculation.
        self.layer : str, default: X_pca
            Key in `.obsm` for the embeddings.
        self.n_components : int, default: None
            Number of embedding components that should be kept.

        Returns
        -------
        pd.DataFrame
            The symmetric Kullback-Leibler distance matrix (samples x samples).
        """
        from itertools import combinations_with_replacement

        import pynndescent

        data = self._get_data()

        # Subset the data if n_components is set
        if self.n_components is not None:
            data = data[:, : self.n_components]

        # Prepare the embedding (one embedding per sample)
        embedding_dict = {s: np.asarray(data[self.adata.obs[self.sample_key] == s]) for s in self.samples}

        # Precompute kNN index for each sample and kNN distances for each samplle within its own sample
        #   --> Index can be used multiple times, which helps with the runtime
        index_dict = {}
        knn_dict = {}

        for sample, embedding in embedding_dict.items():
            index = pynndescent.NNDescent(embedding, n_neighbors=self.k, random_state=42)
            _, dist = index.query(embedding, k=self.k)

            index_dict[sample] = index
            knn_dict[sample] = dist[:, -1]

        # Empty DataFrame for the result
        distances = pd.DataFrame(index=self.samples, columns=self.samples, dtype=float)
        d = data.shape[1]  # Dimensionality of the embedding (needed for KL)

        # Iterate through all sample pairs (e.g., 'AB' -> 'AA', 'AB', 'BB')
        #   --> use combinations_with_replacement(), so only 'AB' and not 'AB', 'BA' is included
        for s_i, s_j in combinations_with_replacement(self.samples, r=2):
            # When s_i == s_j then the distance is zero (diagonal of matrix)
            if s_i == s_j:
                distances.loc[s_i, s_j] = 0
                continue

            data_i = embedding_dict[s_i]  # Get embedding for s_i
            data_j = embedding_dict[s_j]  # Get embedding for s_j

            # Get kNN distances of S_i in S_j (use precomputed index of s_j)
            _, dist_ij = index_dict[s_j].query(data_i, k=self.k)

            # Get kNN distances of S_j in S_i (use precomputed index of s_i)
            _, dist_ji = index_dict[s_i].query(data_j, k=self.k)

            # Get numbers of samples
            m_i = embedding_dict[s_i].shape[0]
            m_j = embedding_dict[s_j].shape[0]

            # Calculate Kullback-Leibler divergences
            kl_ij = GloScope_py.kl_divergence(knn_dict[s_i], dist_ij[:, -1], m_i, m_j, d)
            kl_ji = GloScope_py.kl_divergence(knn_dict[s_j], dist_ji[:, -1], m_j, m_i, d)

            # Sum up the two divergences to get a distance
            kl_sym = kl_ij + kl_ji

            # Save distance in matrix
            #   --> to [i,j] and [j,i] as the matrix is symmetric
            distances.loc[s_i, s_j] = kl_sym
            distances.loc[s_j, s_i] = kl_sym

        return distances

    def calculate_distance_matrix_cuml(self):
        """
        Calculates symmetric Kullback-Leibler divergence using RAPIDS cuML NearestNeighbors on GPU.

        Parameters
        ----------
        self.adata : AnnData
            The AnnData object stored in the class instance.
        self.sample_key : str
            Column in `.obs` containing sample (patient) IDs.
        self.k: int, default: 25
            Number of nearest neighbors for k-nearest neighbor calculation.
        self.layer : str, default: X_pca
            Key in `.obsm` for the embeddings.
        self.n_components : int, default: None
            Number of embedding components that should be kept.

        Returns
        -------
        pd.DataFrame
            The symmetric Kullback-Leibler distance matrix (samples x samples).
        """
        from itertools import combinations_with_replacement

        import cupy as cp
        from cuml.neighbors import NearestNeighbors

        data = self._get_data()

        # Subset the data if n_components is set
        if self.n_components is not None:
            data = data[:, : self.n_components]

        # Prepare the embedding (one embedding per sample)
        # --> convert into cupy arrays
        embedding_dict = {g: cp.asarray(data[self.adata.obs[self.sample_key] == g]) for g in self.samples}

        # Self kNN distances for each sample (r in KL)
        knn_self_dists = {}

        for sample, X in embedding_dict.items():
            nn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
            nn.fit(X)
            dists, _ = nn.kneighbors(X)
            knn_self_dists[sample] = cp.asnumpy(dists[:, -1])  # Convert back to numpy array

        # Empty DataFrame for the result
        distances = pd.DataFrame(index=self.samples, columns=self.samples, dtype=float)
        d = data.shape[1]  # Dimensionality of the embedding (needed for KL)

        # Iterate through all sample pairs (e.g., 'AB' -> 'AA', 'AB', 'BB')
        #   --> use combinations_with_replacement(), so only 'AB' and not 'AB', 'BA' is included
        for s_i, s_j in combinations_with_replacement(self.samples, r=2):
            # When s_i == s_j then the distance is zero (diagonal of matrix)
            if s_i == s_j:
                distances.loc[s_i, s_j] = 0
                continue

            data_i = embedding_dict[s_i]  # Get embedding for s_i
            data_j = embedding_dict[s_j]  # Get embedding for s_i

            # Get kNN distances of S_i in S_j
            nn_j = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
            nn_j.fit(data_j)
            dists_ij, _ = nn_j.kneighbors(data_i)
            dists_ij = cp.asnumpy(dists_ij[:, -1])

            # Get kNN distances of S_j in S_i
            nn_i = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
            nn_i.fit(data_i)
            dists_ji, _ = nn_i.kneighbors(data_j)
            dists_ji = cp.asnumpy(dists_ji[:, -1])

            # Get numbers of samples
            m_i = data_i.shape[0]
            m_j = data_j.shape[0]

            # Calculate Kullback-Leibler divergences
            kl_ij = GloScope_py.kl_divergence(knn_self_dists[s_i], dists_ij, m_i, m_j, d)
            kl_ji = GloScope_py.kl_divergence(knn_self_dists[s_j], dists_ji, m_j, m_i, d)

            # Sum up the two divergences to get a distance
            kl_sym = kl_ij + kl_ji

            # Save distance in matrix
            #   --> to [i,j] and [j,i] as the matrix is symmetric
            distances.loc[s_i, s_j] = kl_sym
            distances.loc[s_j, s_i] = kl_sym

        return distances

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate symmetric Kullback-Leibler divergence between samples using GloScope approach"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        # If use_gpu=True use cuML (else PyNNDescent) for the matrix calculation
        if self.use_gpu:
            distances = self.calculate_distance_matrix_cuml()
        else:
            distances = self.calculate_distance_matrix_pynndescent()

        self.sample_representation = distances
        self.samples = list(self.sample_representation.index)

        # The calculation methods return a DataFrame
        # --> convert to a numpy array for saving/returning
        self.adata.uns[self.DISTANCES_UNS_KEY] = distances.to_numpy()

        return distances.to_numpy()
