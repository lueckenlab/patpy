import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from patpy.tl.evaluation import (
    _filter_missing,
    _get_normalized_distances,
    _get_null_distances_distribution,
    _permanova_ss_w,
    _select_random_subset,
    evaluate_prediction,
    evaluate_representation,
    knn_prediction_score,
    permanova_pseudo_f_statistic,
    predict_knn,
    replicate_robustness,
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


# Using the same Pseudo-F tests as vegan reference examples (Anderson 2001).
def test_permanova_pseudo_f_reference_no_ties():
    d = np.array([[0, 1, 5, 4], [1, 0, 3, 2], [5, 3, 0, 3], [4, 2, 3, 0]], dtype=float)
    f = permanova_pseudo_f_statistic(d, np.array([0, 0, 1, 1]))
    assert f == pytest.approx(4.4)


def test_permanova_pseudo_f_reference_ties():
    d = np.array([[0, 1, 1, 4], [1, 0, 3, 2], [1, 3, 0, 3], [4, 2, 3, 0]], dtype=float)
    f = permanova_pseudo_f_statistic(d, np.array([0, 0, 1, 1]))
    assert f == pytest.approx(2.0)


def _permanova_ss_w_double_loop(distance_sq: np.ndarray, grouping: np.ndarray) -> float:
    """Reference implementation for vectorization regression test (Anderson / adonis2 S_W).
    Included to make sure the vectorized implementation is equivalent."""
    n = distance_sq.shape[0]
    counts = np.bincount(grouping)
    s_w = 0.0
    for i in range(n):
        g = grouping[i]
        ng = counts[g]
        row_sum = 0.0
        for j in range(i + 1, n):
            if grouping[j] == g:
                row_sum += distance_sq[i, j]
        s_w += row_sum / ng
    return s_w


def test_permanova_ss_w_vectorized_matches_double_loop():
    rng = np.random.default_rng(0)
    for n in (6, 9, 14):
        for _ in range(8):
            d = rng.random((n, n))
            d = (d + d.T) / 2
            np.fill_diagonal(d, 0)
            d_sq = d * d
            grouping = rng.integers(0, 4, size=n, dtype=int)
            if len(np.unique(grouping)) < 2:
                grouping[0] = (grouping[0] + 1) % 4
            vec = _permanova_ss_w(d_sq, grouping)
            ref = _permanova_ss_w_double_loop(d_sq, grouping)
            assert vec == pytest.approx(ref)


def test_permanova_non_contiguous_group_codes():
    d = np.array([[0, 1, 5, 4], [1, 0, 3, 2], [5, 3, 0, 3], [4, 2, 3, 0]], dtype=float)
    f1 = permanova_pseudo_f_statistic(d, np.array([0, 0, 1, 1]))
    f2 = permanova_pseudo_f_statistic(d, np.array([10, 10, 20, 20]))
    assert f1 == pytest.approx(f2)


# Validate full evaluate_representation pipeline for PERMANOVA.
def test_evaluate_representation_permanova(toy_distances):
    distances, conditions = toy_distances

    result = evaluate_representation(distances, target=conditions, method="permanova", permutations=199, random_state=0)

    assert isinstance(result, dict)
    assert result["method"] == "permanova"
    assert result["metric"] == "permanova_pseudo_f"
    assert "p_value" in result
    assert np.isfinite(result["score"])
    for key in ("score", "metric", "n_unique", "n_observations", "method"):
        assert key in result


def test_evaluate_representation_permanova_rejects_continuous_numeric():
    n = 40
    rng = np.random.default_rng(0)
    d = rng.random((n, n))
    d = (d + d.T) / 2
    np.fill_diagonal(d, 0)
    target = pd.Series(np.linspace(0, 1, n))

    with pytest.raises(ValueError, match="categorical"):
        evaluate_representation(d, target=target, method="permanova", permutations=9, random_state=0)


def test_evaluate_representation_unknown_method(toy_distances):
    distances, conditions = toy_distances

    with pytest.raises(ValueError, match="Unknown evaluation method"):
        evaluate_representation(distances, target=conditions, method="not_a_real_method")  # type: ignore[arg-type]


def test_evaluate_representation_knn(toy_distances):
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


REPLICATE_NAMES = [
    "donor0_a",
    "donor0_b",
    "donor1_a",
    "donor1_b",
    "donor2_a",
    "donor2_b",
    "donor3_a",
    "donor3_b",
]


# Replicate robustness should be 1 when each sample's replicate is its nearest neighbour.
def test_replicate_robustness_perfect():
    # Replicate pairs sit at 1; all other donor pairs at 8. Each sample's nearest
    # non-self neighbour is its own replicate.
    dist = np.array(
        [
            [0, 1, 8, 8, 8, 8, 8, 8],
            [1, 0, 8, 8, 8, 8, 8, 8],
            [8, 8, 0, 1, 8, 8, 8, 8],
            [8, 8, 1, 0, 8, 8, 8, 8],
            [8, 8, 8, 8, 0, 1, 8, 8],
            [8, 8, 8, 8, 1, 0, 8, 8],
            [8, 8, 8, 8, 8, 8, 0, 1],
            [8, 8, 8, 8, 8, 8, 1, 0],
        ],
        dtype=float,
    )
    distances_df = pd.DataFrame(dist, index=REPLICATE_NAMES, columns=REPLICATE_NAMES)

    score = replicate_robustness(distances_df)

    assert score == pytest.approx(1.0)


# Replicate robustness should be 0 when each sample's replicate is its farthest neighbour.
def test_replicate_robustness_worst():
    # Replicate pairs sit at 8; all other donor pairs at 1. Each sample's own
    # replicate is the farthest non-self neighbour.
    dist = np.array(
        [
            [0, 8, 1, 1, 1, 1, 1, 1],
            [8, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 8, 1, 1, 1, 1],
            [1, 1, 8, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 8, 1, 1],
            [1, 1, 1, 1, 8, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 8],
            [1, 1, 1, 1, 1, 1, 8, 0],
        ],
        dtype=float,
    )
    distances_df = pd.DataFrame(dist, index=REPLICATE_NAMES, columns=REPLICATE_NAMES)

    score = replicate_robustness(distances_df)

    assert score == pytest.approx(0.0)


# Replicate robustness should fall strictly between 0 and 1 for a mixed configuration.
def test_replicate_robustness_intermediate():
    # Donors 0, 1 have tight replicate pairs (distance 1, nearest non-self neighbour).
    # Donors 2, 3 have loose replicate pairs (distance 8, farthest non-self neighbour).
    # All other donor pairs sit at the background distance 4.
    dist = np.array(
        [
            [0, 1, 4, 4, 4, 4, 4, 4],
            [1, 0, 4, 4, 4, 4, 4, 4],
            [4, 4, 0, 1, 4, 4, 4, 4],
            [4, 4, 1, 0, 4, 4, 4, 4],
            [4, 4, 4, 4, 0, 8, 4, 4],
            [4, 4, 4, 4, 8, 0, 4, 4],
            [4, 4, 4, 4, 4, 4, 0, 8],
            [4, 4, 4, 4, 4, 4, 8, 0],
        ],
        dtype=float,
    )
    distances_df = pd.DataFrame(dist, index=REPLICATE_NAMES, columns=REPLICATE_NAMES)

    score = replicate_robustness(distances_df)

    assert 0.0 < score < 1.0
    # Half the samples have replicate at rank 0; the other half at rank n-2.
    # Mean rank = (n-2)/2; normalised score = 1 - 0.5 = 0.5.
    assert score == pytest.approx(0.5)
