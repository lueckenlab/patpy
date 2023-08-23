import numpy as np
import pandas as pd
from scipy.stats import trim_mean


def _upper_diagonal(matrix):
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _get_normalized_distances(distances, conditions, control_level, normalization_type, compare_by_difference=True):
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
    normalization_type,
    n_bootstraps=1000,
    trimmed_fraction=0.2,
    compare_by_difference=True,
):
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
    normalization_type,
    n_bootstraps=1000,
    trimmed_fraction=0.2,
    compare_by_difference=True,
):
    """Test if distances are significantly different from the null distribution"""
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


def predict_knn(distances, y_true, n_neighbors: int = 3, task="classification"):
    """Predict values of `y_true` using K-nearest neighbors

    Parameters
    ----------
    distances : square matrix
        Matrix of distances between samples
    y_true : array-like
        Vector with the same length as `distances` containing values for prediction
    n_neighbors : int = 3
        Number of neighbors to use for classification
    task : str = "classification"

    Returns
    -------
    y_predicted : array-like
        Predicted values of `target` for samples with known values
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


def evaluate_prediction(y_true, y_pred, task):
    """Evaluate how well `y_pred` predicts `y_true`"""
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


def evaluate_representation(distances, target, method="knn", **parameters):
    """Evaluate representation of `target` for the given distance matrix"""
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

    return result
