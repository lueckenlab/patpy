import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats import trim_mean

from patpy.tl._types import _EVALUATION_METHODS, _NORMALIZATION_TYPES, _PREDICTION_TASKS


def _upper_diagonal(matrix):
    """Return upper diagonal of the matrix excluding the diagonal itself"""
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _get_normalized_distances(
    distances, conditions, control_level, normalization_type: _NORMALIZATION_TYPES, compare_by_difference: bool = True
):
    """Calculate distances between samples normalized in respect to the other group

    Based on Petukhov et al (2022): https://www.biorxiv.org/content/10.1101/2022.03.15.484475v1.full.pdf

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    conditions : array-like
        Vector with the same length as `distances` containing a categorical variable
    control_level
        Value of `conditions` that should be used as a control group
    normalization_type : Literal["total", "shift", "var"]
        Type of normalization to use. In the text below, "case" means enything that is not a control group.
        - total: normalize distances between control and case groups to the median of within-control group distances
        - shift: normalize distances between control and case groups to the average of within-control and within-case group median distances
        - var: normalize distances within case group to the median of within-control group distances
    compare_by_difference : bool = True
        If True, normalization is defined as difference (as in the original paper). Otherwise, it is defined as a ratio

    Returns
    -------
    normalized_distances : array-like
        Vector of normalized distances between samples
    """
    is_control = conditions == control_level
    is_case = ~is_control

    between_distances = distances[is_control][:, is_case].flatten()
    d_control = _upper_diagonal(distances[is_control][:, is_control])
    d_case = _upper_diagonal(distances[is_case][:, is_case])

    if normalization_type == "total":
        comparison_group = between_distances
        compare_to = np.median(d_control)

    elif normalization_type == "shift":
        comparison_group = between_distances
        compare_to = 0.5 * (np.median(d_case) + np.median(d_control))

    elif normalization_type == "var":
        comparison_group = d_case
        compare_to = np.median(d_control)

    else:
        raise ValueError("Wrong normalization_type, please choose one of ('total', 'shift', 'var')")

    if compare_by_difference:
        return comparison_group - compare_to
    else:
        return comparison_group / compare_to


def _get_null_distances_distribution(
    distances,
    conditions,
    control_level,
    normalization_type: _NORMALIZATION_TYPES,
    n_bootstraps: int = 1000,
    trimmed_fraction: float = 0.2,
    compare_by_difference: bool = True,
):
    """Calculate null distribution of average normalized distances between samples

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    conditions : array-like
        Vector with the same length as `distances` containing a categorical variable
    control_level
        Value of `conditions` that should be used as a control group
    normalization_type : Literal["total", "shift", "var"]
        Type of normalization to use. For explanation, see the documetation of `_get_normalized_distances`
    n_bootstraps : int = 1000
        Number of bootstrap iterations to use
    trimmed_fraction : float = 0.2
        Fraction of the most extreme values to remove from the distribution
    compare_by_difference : bool = True
        If True, normalization is defined as difference (as in the original paper). Otherwise, it is defined as a ratio

    Returns
    -------
    statistics : array-like
        Vector of statistics for each bootstrap iteration
    """
    statistics = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        norm_distances = _get_normalized_distances(
            distances,
            np.random.permutation(conditions),
            control_level,
            normalization_type,
            compare_by_difference=compare_by_difference,
        )
        statistics[i] = trim_mean(norm_distances, trimmed_fraction)

    return statistics


def _identity_up_to_suffix(name, names) -> list:
    """Returns true if name is identical to any of the names in names, ignoring the suffixes"""
    cropped_name = name[: name.rfind("_")]
    return [other_name[: other_name.rfind("_")] == cropped_name for other_name in names]


def test_distances_significance(
    distances,
    conditions,
    control_level,
    normalization_type: _NORMALIZATION_TYPES,
    n_bootstraps: int = 1000,
    trimmed_fraction: float = 0.2,
    compare_by_difference: bool = True,
):
    """Test if distances are significantly different from the null distribution

    Based on Petukhov et al (2022): https://www.biorxiv.org/content/10.1101/2022.03.15.484475v1.full.pdf

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    conditions : array-like
        Vector with the same length as `distances` containing a categorical variable
    control_level
        Value of `conditions` that should be used as a control group
    normalization_type : Literal["total", "shift", "var"]
        Type of normalization to use. In the text below, "case" means enything that is not a control group.
        - total: normalize distances between control and case groups to the median of within-control group distances
        - shift: normalize distances between control and case groups to the average of within-control and within-case group median distances
        - var: normalize distances within case group to the median of within-control group distances
    n_bootstraps : int = 1000
        Number of bootstrap iterations to use
    trimmed_fraction : float = 0.2
        Fraction of the most extreme values to remove from the distribution
    compare_by_difference : bool = True
        If True, normalization is defined as difference (as in the original paper). Otherwise, it is defined as a ratio
    """
    normalized_distances = _get_normalized_distances(
        distances, conditions, control_level, normalization_type, compare_by_difference
    )
    real_statistic = trim_mean(normalized_distances, trimmed_fraction)

    null_distributed_statistics = _get_null_distances_distribution(
        distances, conditions, control_level, normalization_type, n_bootstraps, trimmed_fraction, compare_by_difference
    )

    p_value = (null_distributed_statistics >= real_statistic).sum() / n_bootstraps

    normalized_distances -= np.median(null_distributed_statistics)

    return normalized_distances, real_statistic, p_value


def persistence_evaluation(
    distances, conditions, max_feature_difference, n_neighbors=10, order="sublevel", infinity_value="max"
):
    r"""Calculate the number, total lifetime and persistence pairs of the connected components in the kNN graph

    Computation is performed while stepwise filtering through the graph starting from the lowest
    value. In practice, if the vertex feature values correspond to e.g. disease severity,
    this function can be used to evaluate how connected components in the kNN graph change as
    the disease severity increases. This evaluates the connectivity of
    the sample representation with respect to the feature.

    Parameters
    ----------
    distances : square matrix, crs_matrix
        Matrix of distances between samples.
    conditions : array-like, numerical
        Numerical vector with the values of a feature for each sample.
    max_feature_difference : float
        Maximum difference in the feature values allowed between connected nodes.
    n_neighbors : int = 7
        Number of neighbors to use for constructing the kNN graph.
    order : str = "sublevel" or "superlevel"
        The order of the filtration. Either "subevel" to filter from the lowest to
        the highest value or "superlevel" to filter from the highest to the lowest value.
    infinity_value : str = "max" or float
        The maximium filtration value. It should be larger or equal to the maximum
        value of the condition. By default it is equal to the maximum value of the feature.
        If set to a float, it uses the specified value. Higher values increase the death coordinate
        and thus the lifetime of any components that remain after all edges have been included.

    Returns
    -------
    result : dict
        Result of the evaluation with the following keys:
        - n_components: int
            The number of connected components detected during the filtration. The lower the number,
            the better i.e. more connected the representation w.r.t. the condition.
        - total_lifetime: float
            The total lifetime of the connected components computed as \\sum_{i=2}^{n_components} (d_i - b_i).
            We disregard the lifetime of the first connected component as it is always equal to the
            difference between the maximum and minimum filtration value. The lower the total_lifetime,
            the better i.e. more connected the representation w.r.t to the condition.
        - persistence_pairs: list
            Persistence pairs of the form [[b_1, d_1], [b_2, d_2], ..., [b_N, d_N]] where
            b_i denotes the birth value at and d_i is the death value of a connected component
            and N is the number of connected components detected during the filtration.

    References
    ----------
    [1] Boissonnat and Maria (2024): https://gudhi.inria.fr/python/latest/simplex_tree_ref.html
    [2] Limbeck and Rieck (2024): https://arxiv.org/abs/2409.03575v1
    [3] Rieck et al. (2017): https://ieeexplore.ieee.org/document/8017588
    """
    import anndata
    from persistence import calculate_persistent_homology, connectivities_to_edge_list

    adata = anndata.AnnData(X=np.zeros((distances.shape[0], 1)))
    adata.obsm["distances"] = distances
    adata.obs["conditions"] = conditions
    conditions = adata.obs["conditions"].values

    ### Calculate a kNN graph from the distances
    sc.pp.neighbors(adata, use_rep="distances", n_neighbors=n_neighbors, metric="precomputed")

    ### Let's remove all edges between nodes that have a feature difference greater than the max_feature_difference
    features = np.tile(adata.obs["conditions"].values, (adata.n_obs, 1))
    edges_to_remove = np.abs(features - features.T) > max_feature_difference
    adata.obsp["connectivities"][edges_to_remove] = 0

    ### Convert the connectivities to an edge list
    connectivities = adata.obsp["connectivities"]
    edge_list = connectivities_to_edge_list(connectivities, mutal_nbhs=False)

    ### Calculate persistent homology and return the persistence pairs corresponding to connected components
    persistence_pairs = calculate_persistent_homology(
        conditions, edge_list, k=2, order=order, infinity_value=infinity_value, min_persistence=0.0
    )
    persistence_pairs = persistence_pairs[0][1:]

    ### Calculate the total lifetime of the connected components
    total_lifetime = 0
    for [b, d] in persistence_pairs:
        lt = d - b
        total_lifetime += lt

    ### Calculate the number of connected components
    clusters = len(persistence_pairs)

    return {"n_components": clusters, "total_lifetime": total_lifetime, "persistence_pairs": persistence_pairs}


def predict_knn(distances, y_true, n_neighbors: int = 3, task: _PREDICTION_TASKS = "classification"):
    """Predict values of `y_true` using K-nearest neighbors

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    y_true : array-like
        Vector with the same length as `distances` containing values for prediction
    n_neighbors : int = 3
        Number of neighbors to use for prediction
    task : Literal["classification", "regression", "ranking"]
        Type of prediction task:
        - classification: predict class labels
        - regression: predict continuous values
        - ranking: predict ranks of the values. Currently, formulated as a regression task

    Returns
    -------
    y_predicted : array-like
        Predicted values of `target` for samples with known values of `y_true`
    """
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

    # Diagonal contains 0s forcing using the same sample for prediction
    # This gives the perfect prediction even for random target (super weird)
    # Filling diagonal with large value removes this leakage
    np.fill_diagonal(distances, distances.max())

    if task == "classification":
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="precomputed", weights="distance")
    elif task == "regression" or task == "ranking":
        knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric="precomputed", weights="distance")
    else:
        raise ValueError(f'task {task} is not supported, please set one of ["classification", "regression", "ranking"]')

    knn.fit(distances, y_true)

    return knn.predict(distances)


def evaluate_prediction(y_true, y_pred, task, **parameters):
    """Evaluate how well `y_pred` predicts `y_true`

    Parameters
    ----------
    y_true : array-like
        Vector with the values of a feature
    y_pred : array-like
        Vector with the predicted values of a feature
    task : Literal["classification", "regression", "ranking"]
        Type of prediction task. See documentation of `predict_knn` for more information

    Returns
    -------
    result : dict
        Result of evaluation with the following keys:
        - score: score of the prediction
        - metric: name of the metric used for evaluation. The following metrics are currently used:
            - f1_macro_calibrated: F1 score for classification task. Calibrated to have value 0 for random prediction and 1 for perfect prediction
            - spearman_r: Spearman correlation for regression and ranking tasks
    """
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score

    if task == "classification":
        score = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        metric = "f1_macro_calibrated"

        n_classes = len(np.unique(y_true))

        if n_classes == 1:
            score = 0
        else:
            # Calibrate the metric. Expected value is 1 / n_classes (e.g. 1/2 for a binary classification)
            # With this calibration score==0 means that the prediction is as good as random
            # score==1 would mean the perfect prediction
            # Note that score can be less than 0 in this case => prediction is worse than random. In this case, it is clipped to 0
            score = (score - 1 / n_classes) / (1 - 1 / n_classes)

    elif task == "regression" or task == "ranking":
        score = spearmanr(y_true, y_pred).statistic
        metric = "spearman_r"

    else:
        raise ValueError(f"{task} is not valid task")

    # For calibrated F1, negative score means worse than random prediction. It doesn't matter to us if it is
    # as good as random or worse, so we clip it to 0.
    # For Spearman correlation, negative score means that nearest neighbors often have values of covariate
    # from the other end of the distribution. It is not very meaningful for us at all, and it is not
    # obvious, what is the minimum possible value in this case. -1 is barely possible, because
    # if the neighbors of a sample have "opposite" values, score for these neighbors would be positive.
    # So we clip it to 0 as well.
    if score < -0.5:
        warnings.warn(f"Score has a big negative value: {score}, which is usually not expected.", stacklevel=1)
    score = np.clip(score, 0, 1)

    return {"score": score, "metric": metric}


def test_proportions(target, groups):
    """Run statistical test to check if distribution of `target` differs between `groups`

    Parameters
    ----------
    target : array-like
        Categories of the observations
    groups : array-like
        Groups (e.g. cluster numbers) of the observations

    Returns
    -------
    result : dict
        Result of statistical test with the following keys
        - score: chi-square statistic
        - p_value: p-value of the test
        - dof: number of the degrees of freedom for the statistical test
    """
    from scipy.stats import chi2_contingency

    contingency_table = pd.crosstab(target, groups)
    score, p_value, dof, _ = chi2_contingency(contingency_table)

    return {"score": score, "p_value": p_value, "dof": dof, "metric": "chi2"}


def _filter_missing(distances, target):
    """Leave only observations for which value of `target` is not missing"""
    not_empty_values = target.notna()
    distances = distances[not_empty_values][:, not_empty_values]

    return distances, target[not_empty_values]


def _select_random_subset(distances, target, num_donors_subset=None, proportion_donors_subset=None):
    """
    Select a random subset of donors from the distances matrix based on a specified number or proportion of donors.

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    target : array-like
        Vector with the values of a feature for each sample
    num_donors_subset : int, optional
        Absolute number of donors to include in the evaluation.
    proportion_donors_subset : float, optional
        Proportion of donors to include in the evaluation.

    Returns
    -------
    distances_subset : square matrix
        Distances among the randomly selected donors.
    target_subset: array-like
        Targets associated with the randomly selected donors.

    Raises
    ------
    - ValueError: If neither `num_donors_subset` nor proportion_donors_subset is specified.
    - ValueError: If `num_donors_subset` is not between 2 and the total number of donors.
    - ValueError: If `proportion_donors_subset` is not a valid proportion (i.e., not between 0 and 1).
    """
    n_donors = distances.shape[0]
    if num_donors_subset is not None:
        if not (2 <= num_donors_subset <= n_donors):
            raise ValueError("num_donors_subset must be between 2 and the maximum number of donors.")
        subset_size = num_donors_subset
    elif proportion_donors_subset is not None:
        if not (0 < proportion_donors_subset <= 1):
            raise ValueError("prop_donors_subset must be a proportion between 0 and 1.")
        subset_size = int(n_donors * proportion_donors_subset)
    else:
        raise ValueError("Either num_donors_subset or prop_donors_subset must be specified.")

    selected_indices = np.random.choice(n_donors, subset_size, replace=False)
    distances_subset = distances[selected_indices, :][:, selected_indices]
    target_subset = target[selected_indices]
    return distances_subset, target_subset


def evaluate_representation(
    distances,
    target,
    method: _EVALUATION_METHODS = "knn",
    num_donors_subset=None,
    proportion_donors_subset=None,
    **parameters,
):
    """Evaluate representation of `target` for the given distance matrix

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    target : array-like
        Vector with the values of a feature for each sample
    method : Literal["knn", "distances", "proportions", "silhouette"]
        Method to use for evaluation:
        - knn: predict values of `target` using K-nearest neighbors and evaluate the prediction
        - distances: test if distances between samples are significantly different from the null distribution
        - proportions: test if distribution of `target` differs between groups (e.g. clusters)
        - silhouette: calculate silhouette score for the given distances
        - persistence: calculate the persistence of connected components in filtration of a kNN graph based on the values of `target`
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
        - persistence:
            - max_feature_difference: maximum difference in the feature values allowed between connected nodes
            - n_neighbors: number of neighbors to use for constructing the kNN graph

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
    if num_donors_subset is not None or proportion_donors_subset is not None:
        distances, target = _select_random_subset(distances, target, num_donors_subset, proportion_donors_subset)

    distances, target = _filter_missing(distances, target)

    if method == "knn":
        y_pred = predict_knn(distances, y_true=target, **parameters)
        result = evaluate_prediction(target, y_pred, **parameters)

    elif method == "distances":
        _, score, p_value = test_distances_significance(distances, conditions=target, **parameters)
        result = {"score": score, "p_value": p_value, "metric": "distances", **parameters}

    elif method == "proportions":
        if "groups" not in parameters:
            raise ValueError('Please, add "groups" key (for example, with clusters) in the parameters')

        result = test_proportions(target, parameters["groups"])

    elif method == "silhouette":
        from sklearn.metrics import silhouette_score

        score = silhouette_score(distances, labels=target, metric="precomputed")

        result = {"score": score, "metric": "silhouette"}

    elif method == "persistence":
        if "max_feature_difference" not in parameters:
            raise ValueError('Please, add "max_feature_difference" key in the parameters')

        result_ph = persistence_evaluation(distances, target, **parameters)

        ## The lower the total_lifetime, the better the representation
        result = {"score": result_ph["total_lifetime"], "metric": "total_lifetime"}

    result["n_unique"] = len(np.unique(target))
    result["n_observations"] = len(target)  # Without missing values this number can change between features
    result["method"] = method

    return result


def _get_col_from_adata(adata, col) -> pd.Series:
    """Extract a column from .obs or .X of the annotated object"""
    if col in adata.obs.columns:
        return adata.obs[col]
    else:
        return pd.Series(adata[:, col].X.toarray().flatten(), index=adata.obs_names)


def trajectory_correlation(
    meta_adata, root_sample, trajectory_variable, representations=None, inverse_trajectory=False, force=False
):
    """Compute the correlation between the trajectory variable and the diffusion pseudotime for each representation

    Parameters
    ----------
    meta_adata: AnnData
        The annotated data object with sample metadata
    root_sample: str
        The root sample to use for the diffusion pseudotime. It must be a presumable start of the trajectory.
        For example, the healthiest patient if trajectory is the disease severity or the youngest patient if trajectory is the age.
    trajectory_variable: str
        The covariate in `meta_adata` containing the trajectory information. Must contain numbers or ordered categories.
    representations: list of str, optional
        The representations to compute the correlation with. If None, all representations in meta_adata.uns["sample_representations"] will be used.
    inverse_trajectory: bool, optional
        Set to True if start of the trajectory is the highest value of the variable.
    force: bool, optional
        If True, the diffusion pseudotime will be recomputed even if it already exists.

    Returns
    -------
    pd.DataFrame
        A dataframe with the correlation between the trajectory variable and the diffusion pseudotime for each representation.

    Sets
    ----
    meta_adata.uns["iroot"]
        The index of the root sample in `meta_adata.obs_names`.
    meta_adata.obs[f"{representation}_dpt_pseudotime"]
        The diffusion pseudotime for each representation.
    meta_adata.obs[f"{representation}_dpt_pseudotime"]
        The diffusion pseudotime for each representation.
    meta_adata.obsm[f"X_{representation}_diffmap"]
        The diffusion map for each representation.
    """
    import ehrapy as ep

    if representations is None:
        representations = meta_adata.uns["sample_representations"]

    meta_adata.uns["iroot"] = np.flatnonzero(meta_adata.obs_names == root_sample)[0]

    trajectory_correlations = []

    for representation in representations:
        try:
            if not force and f"{representation}_dpt_pseudotime" in meta_adata.obs.columns:
                print(f"Diffmap for {representation} already computed, skipping")

            else:
                print(f"Computing diffmap for {representation}")
                ep.tl.diffmap(meta_adata, neighbors_key=f"{representation}_neighbors")
                meta_adata.obsm[f"X_{representation}_diffmap"] = meta_adata.obsm["X_diffmap"]
                ep.tl.dpt(meta_adata, neighbors_key=f"{representation}_neighbors")
                meta_adata.obs.rename(columns={"dpt_pseudotime": f"{representation}_dpt_pseudotime"}, inplace=True)

        except (KeyError, ValueError, RuntimeError) as e:
            print(f"Error computing diffmap for {representation}: {e}")
            meta_adata.obs[f"{representation}_dpt_pseudotime"] = np.zeros(len(meta_adata.obs))
            continue

        target = _get_col_from_adata(meta_adata, trajectory_variable)

        corr, _ = stats.spearmanr(target, meta_adata.obs[f"{representation}_dpt_pseudotime"], nan_policy="omit")

        if inverse_trajectory:
            corr = -corr
        trajectory_correlations.append(corr)

    trajectory_metric_df = pd.DataFrame(trajectory_correlations, index=representations, columns=["correlation"])

    return trajectory_metric_df.sort_values("correlation", ascending=False)


def knn_prediction_score(
    meta_adata, benchmark_schema: dict, representations=None, n_neighbors=3, reverse_technical_score=True
):
    """Compute the KNN prediction score for each representation and covariate type

    Parameters
    ----------
    meta_adata: AnnData
        The annotated data object with sample metadata
    benchmark_schema: dict
        The benchmark schema to use. Must have the following structure:
        - Keys must be "relevant", "technical", and "contextual".
        - Values for must be a dictionary with keys being the covariate names.
        - Values for the second layer must be a string being the task type: "classification", "regression" or "ranking".
        For example:
        {
            "relevant": {
                "Disease_severity": "ranking",
                "Swab_result": "classification",
                "forced_expiratory_volume_in_1s": "regression",
            },
            "technical": {
                "Site": "classification",
                "Batch": "classification",
                "n_cells": "regression",
            },
            "contextual": {
                "Age": "regression",
            },
        }
    representations: list of str, optional
        The representations to compute the score for. If None, all representations in meta_adata.uns["sample_representations"] will be used.
    n_neighbors: int, optional
        The number of neighbors to use for the KNN prediction.
    reverse_technical_score: bool, optional
        If True, the technical scores will be reversed to interpret them as batch effect removal.

    Returns
    -------
    pd.DataFrame
        A dataframe with the KNN prediction score for each representation and covariate type. Columns are:
        - "representation": the representation name
        - "covariate": the covariate name
        - "covariate_type": the covariate type
        - "metric": the metric used to compute the score
        - "score": the KNN prediction score
    """
    if representations is None:
        representations = meta_adata.uns["sample_representations"]

    results = []

    for representation in representations:
        for covariate_type in benchmark_schema:
            for col in benchmark_schema[covariate_type]:
                task = benchmark_schema[covariate_type][col]
                try:
                    distances = meta_adata.obsm[f"{representation}_distances"]

                    if isinstance(distances, pd.DataFrame):
                        distances = distances.loc[meta_adata.obs_names][meta_adata.obs_names].values

                    result = evaluate_representation(
                        distances=distances, target=meta_adata.obs[col], method="knn", task=task, n_neighbors=3
                    )
                except (KeyError, ValueError, RuntimeError) as e:
                    print("Representation:", representation)
                    print("Covariate:", col)
                    print("Task:", task)
                    print("Error:", e)
                    print()
                    raise (e)
                    continue

                result["representation"] = representation
                result["covariate"] = col
                result["covariate_type"] = covariate_type

                # Inverse technical score to interpret them as batch effect removal
                if covariate_type == "technical":
                    result["score"] = 1 - result["score"]

                if result["metric"] == "spearman_r":
                    result["score"] = abs(result["score"])

                results.append(result)

    return pd.DataFrame(results)


def replicate_robustness(distances_df: pd.DataFrame, replicate_identity_function=_identity_up_to_suffix) -> float:
    """Compute the replicate robustness metric, which checks how close the repliicate samples are to each other

    Parameters
    ----------
    distances_df: pd.DataFrame
        The distances between samples. Must be a data frame with samples names in index and columns.
    replicate_identity_function: Callable
        A function that takes names of two samples and returns True if they are replicates

    """
    replicate_indexes = np.zeros(distances_df.shape[0])

    for i, col in enumerate(distances_df.columns):
        distances_df[col] = distances_df[col].astype(float)
        dists_to_others = distances_df[col].sort_values(ascending=True)[1:]
        replicate_idx = np.where(replicate_identity_function(col, dists_to_others.index))[0].item()
        replicate_indexes[i] = replicate_idx

    return 1 - np.mean(replicate_indexes)
