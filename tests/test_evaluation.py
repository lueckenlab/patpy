import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform

from patpy.tl import evaluation as evaluation_module
from patpy.tl.evaluation import (
    _filter_missing,
    _get_normalized_distances,
    _get_null_distances_distribution,
    _select_random_subset,
    evaluate_prediction,
    evaluate_representation,
    knn_prediction_score,
    predict_knn,
)


# Validate normalization of between-group distances with chosen control group.
def test_get_normalized_distances(toy_distances):
    distances, conditions = toy_distances
    control_level = "control"

    normalized = _get_normalized_distances(distances, conditions, control_level, normalization_type="total")

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape[0] == 4
    assert not np.isnan(normalized).any()


# Ensure null distribution bootstrap produces the expected shape and finite values.
def test_get_null_distances_distribution(toy_distances):
    distances, conditions = toy_distances
    control_level = "control"

    null_dist = _get_null_distances_distribution(
        distances, conditions, control_level, normalization_type="total", n_bootstraps=10, trimmed_fraction=0.1
    )

    assert isinstance(null_dist, np.ndarray)
    assert null_dist.shape[0] == 10
    assert not np.isnan(null_dist).any()


# Verify k-NN prediction shape and label domain for classification.
def test_predict_knn(toy_distances):
    distances, _ = toy_distances
    y_true = np.array([0, 0, 1, 1])

    y_pred = predict_knn(distances, y_true, n_neighbors=2, task="classification")

    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_true.shape
    assert set(y_pred).issubset(set(y_true))


# Validate evaluation wrapper returns calibrated F1 for classification.
def test_evaluate_prediction():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0])

    result = evaluate_prediction(y_true, y_pred, task="classification")

    assert isinstance(result, dict)
    assert "score" in result and "metric" in result
    assert result["metric"] == "f1_macro_calibrated"
    assert 0 <= result["score"] <= 1


# Ensure missing targets prune distances and labels consistently.
def test_filter_missing():
    distances = np.array([[0, 1, 2], [1, 0, np.nan], [2, np.nan, 0]], dtype=float)
    target = pd.Series([1, np.nan, 3])

    filtered_distances, filtered_target = _filter_missing(distances, target)

    assert filtered_distances.shape == (2, 2)
    assert filtered_target.shape[0] == 2
    assert not np.isnan(filtered_distances).any()


# Confirm random subset selection respects requested donor count.
def test_select_random_subset():
    distances = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=float)
    target = np.array([1, 2, 3])

    distances_subset, target_subset = _select_random_subset(distances, target, num_donors_subset=2)

    assert distances_subset.shape == (2, 2)
    assert target_subset.shape[0] == 2


# Validate full evaluate_representation pipeline for k-NN classification.
def test_evaluate_representation(toy_distances):
    distances, conditions = toy_distances

    result = evaluate_representation(distances, target=conditions, method="knn", n_neighbors=2, task="classification")

    assert isinstance(result, dict)
    for key in ("score", "metric", "n_unique", "n_observations", "method"):
        assert key in result


def _make_meta_adata_for_knn():
    """Two same-class neighbors at distance 1.0 and three opposite-class neighbors at 1.001.

    Distance-weighted k-NN with k=3 picks two same-class + one opposite (predicts correct
    class), while k=5 picks two same-class + three opposite (predicts the wrong class).
    The diagonal is set to a large value so each sample's self-distance is never among
    the k nearest, regardless of tie-breaking.
    """
    n = 6
    obs = pd.DataFrame(
        {
            "label": pd.Categorical(["A", "A", "A", "B", "B", "B"]),
            "site": pd.Categorical(["A", "A", "A", "B", "B", "B"]),
        },
        index=[f"s{i}" for i in range(n)],
    )
    adata = AnnData(X=np.zeros((n, 1), dtype=float), obs=obs)

    distances = np.full((n, n), 1.001)
    distances[:3, :3] = 1.0
    distances[3:, 3:] = 1.0
    np.fill_diagonal(distances, 10.0)

    adata.obsm["rep_distances"] = distances
    adata.uns["sample_representations"] = ["rep"]
    return adata


# Ensure n_neighbors is forwarded to the underlying k-NN, not hard-coded.
def test_knn_prediction_score_uses_n_neighbors():
    adata = _make_meta_adata_for_knn()
    schema = {"relevant": {"label": "classification"}}

    res_3 = knn_prediction_score(adata, schema, n_neighbors=3)
    res_5 = knn_prediction_score(adata, schema, n_neighbors=5)

    score_3 = res_3.loc[res_3["covariate"] == "label", "score"].iloc[0]
    score_5 = res_5.loc[res_5["covariate"] == "label", "score"].iloc[0]

    assert score_3 == pytest.approx(1.0)
    assert score_5 == pytest.approx(0.0)


# Ensure reverse_technical_score only inverts technical scores when set to True.
def test_knn_prediction_score_reverse_technical_score():
    adata = _make_meta_adata_for_knn()
    schema = {
        "relevant": {"label": "classification"},
        "technical": {"site": "classification"},
    }

    res_reversed = knn_prediction_score(adata, schema, n_neighbors=3, reverse_technical_score=True)
    res_raw = knn_prediction_score(adata, schema, n_neighbors=3, reverse_technical_score=False)

    technical_reversed = res_reversed.loc[res_reversed["covariate_type"] == "technical", "score"].iloc[0]
    technical_raw = res_raw.loc[res_raw["covariate_type"] == "technical", "score"].iloc[0]
    relevant_reversed = res_reversed.loc[res_reversed["covariate_type"] == "relevant", "score"].iloc[0]
    relevant_raw = res_raw.loc[res_raw["covariate_type"] == "relevant", "score"].iloc[0]

    assert technical_reversed == pytest.approx(1 - technical_raw)
    assert relevant_reversed == pytest.approx(relevant_raw)
