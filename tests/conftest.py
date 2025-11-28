import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform

from datasets.synthetic import bootstrap_genes


@pytest.fixture(scope="session")
def synthetic_adata():
    """Structured AnnData with multiple samples and cell types for tests."""
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

    return AnnData(
        np.vstack(cells),
        obs=pd.DataFrame(
            {
                "sample_id": sample_pattern,
                "cell_type": cell_type_pattern,
            }
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )


@pytest.fixture(autouse=True)
def reset_numpy_seed():
    """Ensure deterministic RNG across tests."""
    np.random.seed(0)


@pytest.fixture
def integer_matrix():
    """Dense integer matrix without missing values."""
    return np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
    )


@pytest.fixture
def float_matrix_with_nans():
    """Matrix with scattered NaNs to validate averaging with missing entries."""
    return np.array(
        [
            [1, 2, np.nan],
            [4, np.nan, 6],
            [np.nan, 8, 9],
        ]
    )


@pytest.fixture
def nan_heavy_matrix():
    """Matrix containing entire NaN columns to test default fill handling."""
    return np.array(
        [
            [np.nan, 2, np.nan],
            [np.nan, np.nan, 6],
            [np.nan, 8, np.nan],
        ]
    )


@pytest.fixture
def toy_distances():
    """Creates a toy distance matrix and conditions array for evaluation tests."""
    points = np.array([[0, 1], [1, 1], [2, 2], [3, 3]])
    distances = squareform(pdist(points))
    conditions = pd.Series(["control", "control", "case", "case"])
    return distances, conditions
