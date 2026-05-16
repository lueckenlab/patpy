import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.spatial.distance import pdist, squareform

from patpy.datasets.synthetic import bootstrap_genes


def pytest_addoption(parser):
    parser.addoption(
        "--run-datasets",
        action="store_true",
        default=False,
        help="Run tests marked `dataset` (download real datasets; slow).",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-datasets"):
        return
    skip_dataset = pytest.mark.skip(reason="needs --run-datasets to run")
    for item in items:
        if "dataset" in item.keywords:
            item.add_marker(skip_dataset)


@pytest.fixture(scope="session")
def synthetic_adata():
    """Structured AnnData with multiple samples and cell types for tests."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 60, 20

    base_cell = rng.poisson(lam=6, size=n_genes) + 1
    cells = [bootstrap_genes(base_cell + rng.integers(0, 3, size=n_genes), noise_scale=0.05) for _ in range(n_cells)]

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


@pytest.fixture(scope="session")
def pbmc3k_adata(request, tmp_path_factory):
    """Preprocessed PBMC3k dataset with randomly assigned sample labels.

    Provides real single-cell data with X_pca embedding and louvain cell-type
    annotations, suitable for methods that require biological structure in the
    data (e.g. DiffusionEMD, GloScope, WassersteinTSNE, PILOT, MOFA).

    Under pytest-xdist each worker runs its own session fixture, so without a
    cross-process lock multiple workers race on scanpy's shared
    ``pbmc3k_processed.h5ad`` download and one ends up reading a partially
    written file (``OSError: ... truncated file``). The filelock pattern below
    is the one recommended in the pytest-xdist docs.
    """
    import scanpy as sc

    def _load() -> AnnData:
        adata = sc.datasets.pbmc3k_processed()
        rng = np.random.default_rng(0)
        n_samples = 4
        adata.obs["sample_id"] = rng.choice([f"sample_{i}" for i in range(n_samples)], size=adata.n_obs).astype(str)
        return adata

    # Detect xdist via its ``worker_id`` fixture. Without xdist (or its plugin
    # not installed), no cross-worker race is possible, so just load directly.
    try:
        worker_id = request.getfixturevalue("worker_id")
    except pytest.FixtureLookupError:
        return _load()
    if worker_id == "master":
        return _load()

    from filelock import FileLock

    lock_path = tmp_path_factory.getbasetemp().parent / "pbmc3k_processed.lock"
    with FileLock(str(lock_path)):
        return _load()


@pytest.fixture
def toy_distances():
    """Creates a toy distance matrix and conditions array for evaluation tests."""
    points = np.array([[0, 1], [1, 1], [2, 2], [3, 3]])
    distances = squareform(pdist(points))
    conditions = pd.Series(["control", "control", "case", "case"])
    return distances, conditions
