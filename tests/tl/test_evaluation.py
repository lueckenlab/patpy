import numpy as np
import pandas as pd

from patient_representation.tl.evaluation import (
    _filter_missing,
    _get_normalized_distances,
    _get_null_distances_distribution,
    _select_random_subset,
    distances_significance_test,
    evaluate_prediction,
    evaluate_representation,
    predict_knn,
    statistical_test_proportions,
)


def test_get_normalized_distances(toy_distances):
    distances, conditions = toy_distances
    control_level = "control"

    normalized_distances = _get_normalized_distances(distances, conditions, control_level, normalization_type="total")

    assert isinstance(normalized_distances, np.ndarray), "Normalized distances should be an array."
    assert (
        normalized_distances.shape[0] == 4
    ), "There should be 4 normalized distances, corresponding to between-group distances."
    assert not np.isnan(normalized_distances).any(), "There should be no NaN values in normalized distances."


def test_get_null_distances_distribution(toy_distances):
    distances, conditions = toy_distances
    control_level = "control"

    null_distribution = _get_null_distances_distribution(
        distances, conditions, control_level, normalization_type="total", n_bootstraps=10, trimmed_fraction=0.1
    )

    assert isinstance(null_distribution, np.ndarray), "Null distribution should be an array."
    assert (
        null_distribution.shape[0] == 10
    ), "Null distribution should have 10 values, corresponding to the number of bootstraps."
    assert not np.isnan(null_distribution).any(), "There should be no NaN values in the null distribution."


def test_distances_significance_test(toy_distances):
    distances, conditions = toy_distances
    control_level = "control"

    normalized_distances, real_statistic, p_value = distances_significance_test(
        distances, conditions, control_level, n_bootstraps=10, trimmed_fraction=0.1, normalization_type="total"
    )

    assert isinstance(normalized_distances, np.ndarray), "Normalized distances should be an array."
    assert isinstance(real_statistic, float), "Real statistic should be a float."
    assert isinstance(p_value, float), "P-value should be a float."
    assert 0 <= p_value <= 1, "P-value should be between 0 and 1."


def test_predict_knn(toy_distances):
    distances, _ = toy_distances
    y_true = np.array([0, 0, 1, 1])

    y_pred = predict_knn(distances, y_true, n_neighbors=2, task="classification")

    assert isinstance(y_pred, np.ndarray), "Predicted values should be an array."
    assert y_pred.shape == y_true.shape, "Predicted values should have the same shape as true values."
    assert set(y_pred).issubset(set(y_true)), "Predicted values should be a subset of true values."


def test_evaluate_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    result = evaluate_prediction(y_true, y_pred, task="classification")

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "score" in result, "Result should contain a score."
    assert "metric" in result, "Result should contain a metric."
    assert result["metric"] == "f1_macro_calibrated", "Metric should be 'f1_macro_calibrated' for classification task."
    assert 0 <= result["score"] <= 1, "Score should be between 0 and 1."


def test_statistical_test_proportions():
    target = pd.Series(["A", "A", "B", "B"])
    groups = pd.Series([1, 1, 2, 2])

    result = statistical_test_proportions(target, groups)

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "score" in result, "Result should contain a score."
    assert "p_value" in result, "Result should contain a p-value."
    assert "dof" in result, "Result should contain degrees of freedom."
    assert result["metric"] == "chi2", "Metric should be 'chi2'."


def test_filter_missing():
    distances = np.array([[0, 1, 2], [1, 0, np.nan], [2, np.nan, 0]])
    target = pd.Series([1, np.nan, 3])

    filtered_distances, filtered_target = _filter_missing(distances, target)

    assert filtered_distances.shape == (
        2,
        2,
    ), "Filtered distances should have shape (2, 2) after removing missing values."
    assert filtered_target.shape[0] == 2, "Filtered target should have 2 values after removing missing values."


def test_select_random_subset():
    distances = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    target = np.array([1, 2, 3])

    distances_subset, target_subset = _select_random_subset(distances, target, num_donors_subset=2)

    assert distances_subset.shape == (2, 2), "Distances subset should have shape (2, 2) when selecting 2 donors."
    assert target_subset.shape[0] == 2, "Target subset should have 2 values when selecting 2 donors."


def test_evaluate_representation(toy_distances):
    distances, conditions = toy_distances

    result = evaluate_representation(distances, target=conditions, method="knn", n_neighbors=2, task="classification")

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "score" in result, "Result should contain a score."
    assert "metric" in result, "Result should contain a metric."
    assert "n_unique" in result, "Result should contain the number of unique values in target."
    assert "n_observations" in result, "Result should contain the number of observations."
    assert "method" in result, "Result should contain the method used for evaluation."
