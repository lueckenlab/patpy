import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from datasets.synthetic import bootstrap_genes
from patpy.tl.sample_representation import (
    CellGroupComposition,
    GroupedPseudobulk,
    Pseudobulk,
    RandomVector,
)


@pytest.fixture(scope="session")
def synthetic_adata():
    """Build a small but structured AnnData object using synthetic bootstrapping utilities."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 20

    base_cell = rng.poisson(lam=6, size=n_genes) + 1
    cells = [
        bootstrap_genes(base_cell + rng.integers(0, 3, size=n_genes), noise_scale=0.05) for _ in range(n_cells)
    ]

    sample_pattern = np.repeat([f"sample_{i}" for i in range(6)], repeats=n_cells // 6)
    cell_type_pattern = np.tile(
        ["ct_a", "ct_b", "ct_c", "ct_a", "ct_b", "ct_c", "ct_a", "ct_b", "ct_c", "ct_a"],
        reps=6,
    )

    adata = AnnData(
        np.vstack(cells),
        obs=pd.DataFrame(
            {
                "sample_id": sample_pattern,
                "cell_type": cell_type_pattern,
            }
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    return adata


@pytest.fixture(autouse=True)
def reset_numpy_seed():
    """Keep numpy RNG deterministic so distance-based tests are reproducible."""
    np.random.seed(0)


LIGHTWEIGHT_METHODS = [
    (Pseudobulk, {"layer": "X"}),
    (GroupedPseudobulk, {"layer": "X"}),
    (RandomVector, {}),
    (CellGroupComposition, {"layer": "X"}),
]


# Verifies that every lightweight sample representation computes and stores a symmetric distance matrix.
@pytest.mark.parametrize("method_cls, kwargs", LIGHTWEIGHT_METHODS)
def test_distance_matrix_is_computed_and_cached(method_cls, kwargs, synthetic_adata):
    adata = synthetic_adata.copy()
    n_samples = adata.obs["sample_id"].nunique()

    method = method_cls(sample_key="sample_id", cell_group_key="cell_type", **kwargs)
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

    method = method_cls(sample_key="sample_id", cell_group_key="cell_type", **kwargs)
    method.prepare_anndata(adata)
    baseline = method.calculate_distance_matrix(force=True)

    cached = np.full_like(baseline, fill_value=-1.0)
    adata.uns[method.DISTANCES_UNS_KEY] = cached

    distances = method.calculate_distance_matrix()

    assert np.array_equal(distances, cached)
    assert adata.uns[method.DISTANCES_UNS_KEY] is cached
