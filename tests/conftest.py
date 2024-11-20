import numpy as np
import pandas as pd
import pytest
import scanpy as sc


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
