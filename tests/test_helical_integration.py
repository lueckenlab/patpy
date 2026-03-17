"""Integration tests for get_helical_embedding using real Helical model classes.

Run with:
    pytest tests/test_helical_integration.py -m helical -v

All tests are skipped automatically when the `helical` package is not installed.
They load real model weights and are therefore slow; keep them out of the default
CI run by selecting only the `helical` marker when needed.
"""

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

from patpy.pp.basic import get_helical_embedding

# ---------------------------------------------------------------------------
# Skip the entire module when helical is not installed
# ---------------------------------------------------------------------------
helical = pytest.importorskip("helical", reason="helical package not installed")

pytestmark = pytest.mark.helical

# ---------------------------------------------------------------------------
# Minimal fixture
# ---------------------------------------------------------------------------

# A handful of real human gene symbols / Ensembl IDs that sit in every
# Helical model vocabulary.  Using a tiny set keeps memory and I/O low.
_GENE_SYMBOLS = [
    "GAPDH", "ACTB", "TP53", "MYC", "BRCA1",
    "TNF", "IL6", "CD3E", "CD8A", "PTPRC",
]
_ENSEMBL_IDS = [
    "ENSG00000111640",  # GAPDH
    "ENSG00000075624",  # ACTB
    "ENSG00000141510",  # TP53
    "ENSG00000136997",  # MYC
    "ENSG00000012048",  # BRCA1
    "ENSG00000232810",  # TNF
    "ENSG00000136244",  # IL6
    "ENSG00000198851",  # CD3E
    "ENSG00000153563",  # CD8A
    "ENSG00000081237",  # PTPRC
]


@pytest.fixture(scope="module")
def minimal_adata_symbols():
    """Five cells × 10 common human genes (HGNC symbols as var_names)."""
    rng = np.random.default_rng(42)
    X = rng.poisson(lam=5, size=(5, len(_GENE_SYMBOLS))).astype("float32")
    return AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(5)]),
        var=pd.DataFrame(index=_GENE_SYMBOLS),
    )


@pytest.fixture(scope="module")
def minimal_adata_ensembl():
    """Five cells × 10 common human genes (Ensembl IDs as var_names)."""
    rng = np.random.default_rng(42)
    X = rng.poisson(lam=5, size=(5, len(_ENSEMBL_IDS))).astype("float32")
    return AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(5)]),
        var=pd.DataFrame(index=_ENSEMBL_IDS),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_embedding(result: sc.AnnData, obsm_key: str, n_cells: int) -> None:
    assert obsm_key in result.obsm, f"{obsm_key!r} not found in adata.obsm"
    emb = result.obsm[obsm_key]
    assert emb.ndim == 2, "Embedding must be 2-D"
    assert emb.shape[0] == n_cells, f"Expected {n_cells} rows, got {emb.shape[0]}"
    assert emb.shape[1] > 0, "Embedding dimension must be positive"
    assert not np.isnan(emb).all(), "Embedding is all NaN"


# ---------------------------------------------------------------------------
# Per-model tests
# ---------------------------------------------------------------------------

def test_scgpt_real(minimal_adata_symbols):
    adata = minimal_adata_symbols.copy()
    result = get_helical_embedding(adata, model="scgpt", batch_size=2, device="cpu")
    _assert_embedding(result, "X_scgpt", adata.n_obs)


def test_geneformer_real(minimal_adata_symbols):
    adata = minimal_adata_symbols.copy()
    result = get_helical_embedding(
        adata,
        model="geneformer",
        batch_size=2,
        device="cpu",
        model_name="gf-6L-30M-i2048",  # smallest variant
    )
    _assert_embedding(result, "X_geneformer", adata.n_obs)


def test_uce_real(minimal_adata_symbols):
    adata = minimal_adata_symbols.copy()
    result = get_helical_embedding(
        adata,
        model="uce",
        batch_size=2,
        device="cpu",
        model_name="4layer_model",  # smallest variant
    )
    _assert_embedding(result, "X_uce", adata.n_obs)


def test_uce_real_dense_input(minimal_adata_symbols):
    """UCE must not crash when adata.X is a dense numpy array."""
    import scipy.sparse as sp

    adata = minimal_adata_symbols.copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    result = get_helical_embedding(
        adata,
        model="uce",
        batch_size=2,
        device="cpu",
        model_name="4layer_model",
    )
    _assert_embedding(result, "X_uce", adata.n_obs)


def test_transcriptformer_real(minimal_adata_ensembl):
    adata = minimal_adata_ensembl.copy()
    result = get_helical_embedding(
        adata,
        model="transcriptformer",
        batch_size=2,
        model_name="tf_exemplar",  # smallest variant
        gene_col_name="index",
    )
    emb = result.obsm["X_transcriptformer"]
    _assert_embedding(result, "X_transcriptformer", adata.n_obs)
    assert emb.dtype == np.float32, f"Expected float32, got {emb.dtype}"
