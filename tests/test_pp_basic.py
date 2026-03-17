from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from patpy.pp.basic import (
    _to_numpy,
    calculate_cell_qc_metrics,
    calculate_compositional_metrics,
    calculate_n_cells_per_sample,
    convert_cell_types_to_phemd_format,
    extract_metadata,
    fill_nan_distances,
    filter_small_cell_groups,
    filter_small_samples,
    get_helical_embedding,
    is_count_data,
    prepare_data_for_phemd,
    subsample,
)

SAMPLE_KEY = "sample_id"
CELL_KEY = "cell_type"


# Verify PhEMD preparation returns expected components and ordering.
def test_prepare_data_for_phemd_shapes(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.var["variances"] = np.linspace(0, 1, adata.n_vars)

    expression_data, all_genes, selected_genes, sample_names = prepare_data_for_phemd(
        adata, sample_col=SAMPLE_KEY, n_top_var_genes=5
    )

    assert expression_data.shape == adata.X.shape
    assert list(all_genes) == list(adata.var_names)
    assert len(selected_genes) == 5
    assert list(sample_names) == list(adata.obs[SAMPLE_KEY])


# Ensure PhEMD conversion writes all required tables per cell type.
def test_convert_cell_types_to_phemd_format(tmp_path, synthetic_adata):
    adata = synthetic_adata.copy()
    adata.var["variances"] = np.linspace(0, 1, adata.n_vars)

    convert_cell_types_to_phemd_format(
        adata, cell_type_col=CELL_KEY, sample_col=SAMPLE_KEY, output_dir=tmp_path, n_top_var_genes=5
    )

    for cell_type in adata.obs[CELL_KEY].unique():
        cell_dir = tmp_path / cell_type
        assert cell_dir.exists()
        expression = pd.read_csv(cell_dir / "expression.csv", header=None)
        all_genes = pd.read_csv(cell_dir / "all_genes.csv", header=None)
        selected = pd.read_csv(cell_dir / "selected_genes.csv", header=None)
        samples = pd.read_csv(cell_dir / "samples.csv", header=None)

        assert not expression.empty
        assert len(all_genes) == adata.n_vars
        assert len(selected) == 5
        assert len(samples) == len(expression)


# Check compositional metrics aggregation per sample and category.
def test_calculate_compositional_metrics(synthetic_adata):
    result = calculate_compositional_metrics(
        synthetic_adata, sample_key=SAMPLE_KEY, composition_keys=[CELL_KEY], normalize_to=100
    )

    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == set(synthetic_adata.obs[SAMPLE_KEY].unique())
    assert all(col.startswith("cell_type_") for col in result.columns)
    for sample in synthetic_adata.obs[SAMPLE_KEY].unique():
        assert np.isclose(result.loc[sample].sum(), 100)


# Validate QC metric aggregation and column naming.
def test_calculate_cell_qc_metrics(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.obs["QC_ngenes"] = np.linspace(10, 70, adata.n_obs, dtype=int)
    adata.obs["QC_total_UMI"] = np.linspace(100, 400, adata.n_obs, dtype=int)

    result = calculate_cell_qc_metrics(
        adata,
        sample_key=SAMPLE_KEY,
        cell_qc_vars=["QC_ngenes", "QC_total_UMI"],
        agg_function=np.median,
    )

    assert "median_QC_ngenes" in result.columns
    assert "median_QC_total_UMI" in result.columns
    assert not result.isna().any().any()
    assert set(result.index) == set(adata.obs[SAMPLE_KEY])


# Confirm cell counts per sample are tallied correctly.
def test_calculate_n_cells_per_sample(synthetic_adata):
    result = calculate_n_cells_per_sample(synthetic_adata, sample_key=SAMPLE_KEY)

    assert set(result.index) == set(synthetic_adata.obs[SAMPLE_KEY])
    assert result["n_cells"].sum() == synthetic_adata.n_obs


# Ensure samples below size threshold are removed.
def test_filter_small_samples(synthetic_adata):
    adata = synthetic_adata.copy()
    small_sample = adata.obs[SAMPLE_KEY].unique()[0]
    small_mask = adata.obs[SAMPLE_KEY] == small_sample
    indices = np.flatnonzero(small_mask.to_numpy())
    keep_mask = np.ones(adata.n_obs, dtype=bool)
    keep_mask[small_mask.to_numpy()] = False
    keep_mask[indices[:2]] = True
    adata = adata[keep_mask].copy()

    filtered = filter_small_samples(adata, sample_key=SAMPLE_KEY, sample_size_threshold=3)

    assert small_sample not in filtered.obs[SAMPLE_KEY].values
    assert filtered.obs[SAMPLE_KEY].nunique() == adata.obs[SAMPLE_KEY].nunique() - 1


# Ensure undersized or absent cell groups are removed.
def test_filter_small_cell_groups(synthetic_adata):
    # Pass-through: threshold below the minimum group size — all cell types kept.
    filtered = filter_small_cell_groups(
        synthetic_adata, sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, cluster_size_threshold=2
    )
    assert set(filtered.obs[CELL_KEY].unique()) == set(synthetic_adata.obs[CELL_KEY].unique())

    # Filtering: add a rare cell type that exists only in one sample with one cell,
    # which is below the threshold for all other samples (0 < threshold).
    adata = synthetic_adata.copy()
    rare_row = adata[:1].copy()
    rare_row.obs[CELL_KEY] = "rare_ct"
    from anndata import concat

    adata_with_rare = concat([adata, rare_row])
    filtered = filter_small_cell_groups(
        adata_with_rare, sample_key=SAMPLE_KEY, cell_group_key=CELL_KEY, cluster_size_threshold=2
    )
    assert "rare_ct" not in filtered.obs[CELL_KEY].values
    assert set(filtered.obs[CELL_KEY].unique()) == set(synthetic_adata.obs[CELL_KEY].unique())


# Check subsampling respects per-category minimums and fraction sizing.
def test_subsample(synthetic_adata):
    subsampled = subsample(synthetic_adata, obs_category_col=CELL_KEY, min_samples_per_category=1, fraction=0.5)

    assert subsampled.shape[0] <= synthetic_adata.shape[0]
    assert all(ct in subsampled.obs[CELL_KEY].values for ct in synthetic_adata.obs[CELL_KEY].unique())


# Verify metadata extraction preserves order and handles duplicate sample key column.
def test_extract_metadata_with_sample_column(synthetic_adata):
    adata = synthetic_adata.copy()
    donor_condition = {sample: f"group_{i}" for i, sample in enumerate(adata.obs[SAMPLE_KEY].unique())}
    adata.obs["donor_condition"] = adata.obs[SAMPLE_KEY].map(donor_condition)

    metadata = extract_metadata(adata, sample_key=SAMPLE_KEY, columns=[SAMPLE_KEY, "donor_condition"])

    assert list(metadata.index) == list(adata.obs[SAMPLE_KEY].unique())
    assert SAMPLE_KEY in metadata.columns
    assert "donor_condition" in metadata.columns


# Validate integer-only detection for count matrices.
def test_is_count_data(integer_matrix):
    assert is_count_data(integer_matrix)

    non_count_matrix = np.array([[1.1, 2.2], [3.3, 4.4]])
    assert not is_count_data(non_count_matrix)


# ── get_helical_embedding ──────────────────────────────────────────────────────


def _make_mock_model(embedding_dim: int, n_cells: int):
    """Return a mock helical model that produces a fixed numpy embedding array."""
    mock_model = MagicMock()
    mock_model.process_data.return_value = MagicMock()
    mock_model.get_embeddings.return_value = np.ones((n_cells, embedding_dim), dtype="float32")
    return mock_model


@pytest.mark.parametrize(
    "model_name, obsm_key, config_cls_path, model_cls_path",
    [
        (
            "scgpt",
            "X_scgpt",
            "helical.models.scgpt.scGPTConfig",
            "helical.models.scgpt.scGPT",
        ),
        (
            "geneformer",
            "X_geneformer",
            "helical.models.geneformer.GeneformerConfig",
            "helical.models.geneformer.Geneformer",
        ),
        (
            "uce",
            "X_uce",
            "helical.models.uce.UCEConfig",
            "helical.models.uce.UCE",
        ),
    ],
)
def test_get_helical_embedding_stores_obsm(synthetic_adata, model_name, obsm_key, config_cls_path, model_cls_path):
    """Each supported model stores embeddings under the correct adata.obsm key."""
    pytest.importorskip("helical", reason="helical package not installed")
    adata = synthetic_adata.copy()
    embedding_dim = 16

    mock_model = _make_mock_model(embedding_dim, adata.n_obs)
    mock_config_cls = MagicMock(return_value=MagicMock())
    mock_model_cls = MagicMock(return_value=mock_model)

    with patch(config_cls_path, mock_config_cls), patch(model_cls_path, mock_model_cls):
        result = get_helical_embedding(adata, model=model_name, batch_size=4, device="cpu")

    assert obsm_key in result.obsm
    assert result.obsm[obsm_key].shape == (adata.n_obs, embedding_dim)


def test_get_helical_embedding_transcriptformer(synthetic_adata):
    """TranscriptFormer embeddings are converted to float32 numpy arrays."""
    pytest.importorskip("helical", reason="helical package not installed")
    import torch

    adata = synthetic_adata.copy()
    embedding_dim = 32
    torch_embeddings = torch.ones((adata.n_obs, embedding_dim))

    mock_model = MagicMock()
    mock_model.process_data.return_value = MagicMock()
    mock_model.get_embeddings.return_value = torch_embeddings

    with (
        patch("helical.models.transcriptformer.model.TranscriptFormer", return_value=mock_model),
        patch(
            "helical.models.transcriptformer.transcriptformer_config.TranscriptFormerConfig",
            return_value=MagicMock(),
        ),
    ):
        result = get_helical_embedding(adata, model="transcriptformer", batch_size=4)

    assert "X_transcriptformer" in result.obsm
    emb = result.obsm["X_transcriptformer"]
    assert emb.shape == (adata.n_obs, embedding_dim)
    assert emb.dtype == np.float32


def test_get_helical_embedding_invalid_model(synthetic_adata):
    """An unrecognized model name raises ValueError."""
    pytest.importorskip("helical", reason="helical package not installed")
    with pytest.raises(ValueError, match="Unrecognized model"):
        get_helical_embedding(synthetic_adata, model="unknown_model")


def test_get_helical_embedding_case_insensitive(synthetic_adata):
    """Model name matching is case-insensitive."""
    pytest.importorskip("helical", reason="helical package not installed")
    adata = synthetic_adata.copy()
    embedding_dim = 8

    mock_model = _make_mock_model(embedding_dim, adata.n_obs)

    with (
        patch("helical.models.scgpt.scGPTConfig", return_value=MagicMock()),
        patch("helical.models.scgpt.scGPT", return_value=mock_model),
    ):
        result = get_helical_embedding(adata, model="ScGPT", batch_size=4, device="cpu")

    assert "X_scgpt" in result.obsm


def test_to_numpy_converts_tensor():
    """_to_numpy converts a torch.Tensor to a numpy array."""
    torch = pytest.importorskip("torch")
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    arr = _to_numpy(t)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, t.numpy())


def test_to_numpy_passthrough_list():
    """_to_numpy wraps plain lists in a numpy array."""
    pytest.importorskip(
        "helical", reason="helical package not installed"
    )  # This test requires torch, so pass it if there is no helical
    arr = _to_numpy([[1, 2], [3, 4]])
    assert isinstance(arr, np.ndarray)


# Confirm NaN distances are filled symmetrically with max-distance scaling.
def test_fill_nan_distances():
    distances = np.array([[0, np.nan, 3], [np.nan, 0, 2], [3, 2, 0]], dtype=float)
    filled = fill_nan_distances(distances, n_max_distances=1)

    assert not np.isnan(filled).any()
    assert filled.shape == distances.shape
    assert filled[0, 1] == filled[1, 0]
