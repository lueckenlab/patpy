import numpy as np
import pandas as pd
from scipy.stats import trim_mean

from patient_representation.tl._types import _EVALUATION_METHODS, _NORMALIZATION_TYPES, _PREDICTION_TASKS


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

    if normalization_type == "total":
        comparison_group = between_distances
        compare_to = np.median(d_control)

    elif normalization_type == "shift":
        d_case = _upper_diagonal(distances[is_case][:, is_case])
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

    pvalue = (null_distributed_statistics >= real_statistic).sum() / n_bootstraps

    normalized_distances -= np.median(null_distributed_statistics)

    return normalized_distances, real_statistic, pvalue


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
            # Note that score can be less than 0 in this case => prediction is worse than random
            score = (score - 1 / n_classes) / (1 - 1 / n_classes)

    elif task == "regression" or task == "ranking":
        score = spearmanr(y_true, y_pred).statistic
        metric = "spearman_r"

    else:
        raise ValueError(f"{task} is not valid task")

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


def evaluate_representation(distances, target, method: _EVALUATION_METHODS = "knn", **parameters):
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
    distances, target = _filter_missing(distances, target)

    if method == "knn":
        y_pred = predict_knn(distances, y_true=target, **parameters)
        result = evaluate_prediction(target, y_pred, **parameters)

    elif method == "distances":
        _, score, p_value = test_distances_significance(distances, conditions=target, **parameters)
        result = {"score": score, "pvalue": p_value, "metric": "distances", **parameters}

    elif method == "proportions":
        if "groups" not in parameters:
            raise ValueError('Please, add "groups" key (for example, with clusters) in the parameters')

        result = test_proportions(target, parameters["groups"])

    elif method == "silhouette":
        from sklearn.metrics import silhouette_score

        score = silhouette_score(distances, labels=target, metric="precomputed")

        result = {"score": score, "metric": "silhouette"}

    result["n_unique"] = len(np.unique(target))
    result["n_observations"] = len(target)  # Without missing values this number can change between features
    result["method"] = method

    return result
