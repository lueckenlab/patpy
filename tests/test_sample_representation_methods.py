import numpy as np
import pytest

from patpy.tl.sample_representation import (
    CellGroupComposition,
    GroupedPseudobulk,
    Pseudobulk,
    RandomVector,
    calculate_average_without_nans,
)

SAMPLE_KEY = "sample_id"
CELL_KEY = "cell_type"

LIGHTWEIGHT_METHODS = [
    (Pseudobulk, {"layer": "X"}),
    (GroupedPseudobulk, {"layer": "X"}),
    (RandomVector, {}),
    (CellGroupComposition, {}),
]


# Verifies that every lightweight sample representation computes and stores a symmetric distance matrix.
@pytest.mark.parametrize("method_cls, kwargs", LIGHTWEIGHT_METHODS)
def test_distance_matrix_is_computed_and_cached(method_cls, kwargs, synthetic_adata):
    adata = synthetic_adata.copy()
    n_samples = adata.obs[SAMPLE_KEY].nunique()

    method = method_cls(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, **kwargs)
    method.prepare_anndata(adata)
    distances = method.calculate_distance_matrix(force=True)

    assert isinstance(distances, np.ndarray)
    assert distances.shape == (n_samples, n_samples)
    assert np.allclose(distances, distances.T)
    assert method.DISTANCES_UNS_KEY in adata.uns
    assert np.array_equal(adata.uns[method.DISTANCES_UNS_KEY], distances)


# Ensures all methods respect cached distances unless explicitly forced to recompute.
@pytest.mark.parametrize("method_cls, kwargs", LIGHTWEIGHT_METHODS)
def test_distance_matrix_uses_cache_when_present(method_cls, kwargs, synthetic_adata):
    adata = synthetic_adata.copy()

    method = method_cls(sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, **kwargs)
    method.prepare_anndata(adata)
    baseline = method.calculate_distance_matrix(force=True)

    cached = np.full_like(baseline, fill_value=-1.0)
    adata.uns[method.DISTANCES_UNS_KEY] = cached

    distances = method.calculate_distance_matrix()

    assert np.array_equal(distances, cached)
    assert adata.uns[method.DISTANCES_UNS_KEY] is cached


# Validates averaging logic with and without NaNs, including default fill behavior.
def test_calculate_average_without_nans(integer_matrix, float_matrix_with_nans, nan_heavy_matrix):
    averages, sample_sizes = calculate_average_without_nans(integer_matrix, axis=0)
    expected_averages = np.array([4, 5, 6])
    expected_sample_sizes = np.array([3, 3, 3])
    assert np.allclose(averages, expected_averages)
    assert np.array_equal(sample_sizes, expected_sample_sizes)

    averages, sample_sizes = calculate_average_without_nans(float_matrix_with_nans, axis=0)
    expected_averages = np.array([2.5, 5, 7.5])
    expected_sample_sizes = np.array([2, 2, 2])
    assert np.allclose(averages, expected_averages)
    assert np.array_equal(sample_sizes, expected_sample_sizes)

    averages, sample_sizes = calculate_average_without_nans(nan_heavy_matrix, axis=0, default_value=0)
    expected_averages = np.array([0, 5, 6])
    expected_sample_sizes = np.array([0, 2, 1])
    assert np.allclose(averages, expected_averages)
    assert np.array_equal(sample_sizes, expected_sample_sizes)

    averages = calculate_average_without_nans(nan_heavy_matrix, axis=0, return_sample_sizes=False)
    assert np.allclose(averages, expected_averages)
