import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from scipy.spatial.distance import pdist, squareform


@pytest.fixture
def toy_adata():
    """
    Creates a toy AnnData object for testing.
    """
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    var = pd.DataFrame(index=["gene1", "gene2", "gene3"])
    obs = pd.DataFrame({"sample": ["sample1", "sample2", "sample1"], "cell_type": ["typeA", "typeB", "typeA"]})
    adata = sc.AnnData(X=data, obs=obs, var=var)
    adata.var["variances"] = [0.1, 0.5, 0.9]
    return adata


@pytest.fixture
def toy_distances():
    """Creates a toy distance matrix and conditions array for testing."""
    distances = squareform(pdist(np.array([[0, 1], [1, 1], [2, 2], [3, 3]])))
    conditions = pd.Series(["control", "control", "case", "case"])
    # conditions = np.array(["control", "control", "case", "case"])
    return distances, conditions


@pytest.fixture
def toy_dataframe():
    return pd.DataFrame(
        {
            "sample_key": ["sample1", "sample2", "sample3", "sample4"],
            "cell_group_key": ["cell_type1", "cell_type2", "cell_type3", "cell_type4"],
            "numeric_col": [1.5, 2.3, 3.1, 4.2],
            "missing_col": [1.0, np.nan, 3.5, np.nan],
        }
    )
