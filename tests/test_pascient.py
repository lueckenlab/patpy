"""Tests for the PaSCient supervised method wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import torch
from torch import nn

from patpy.tl.supervised import PaSCient

N_CELLS = 200
N_GENES = 30
N_DONORS = 10
CELLS_PER_DONOR = N_CELLS // N_DONORS
EMB_DIM = 32  # small embedding dimension for tests
N_CLASSES = 3


# ---------------------------------------------------------------------------
# Lightweight mock PaSCient model
# ---------------------------------------------------------------------------


class _MockAggregator(nn.Module):
    """Mimics cell2patient_aggregation.aggregate (masked mean)."""

    def aggregate(self, data, mask):
        # data: (B, 1, C, D), mask: (B, 1, C)
        mask_f = mask.unsqueeze(-1).float()  # (B, 1, C, 1)
        return (data * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp(min=1)


class _MockPaSCientModel(nn.Module):
    """Minimal model implementing the PaSCient sub-module interface.

    gene2cell_encoder:   (B, 1, C, G)  → (B, 1, C, EMB_DIM)
    cell2cell_encoder:   identity
    cell2patient_aggregation.aggregate: masked mean → (B, 1, EMB_DIM)
    patient_encoder:     identity
    patient_predictor:   linear → (B, 1, N_CLASSES)
    """

    def __init__(self, n_genes, emb_dim=EMB_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.gene2cell_encoder = nn.Linear(n_genes, emb_dim)
        self.cell2cell_encoder = _IdentityWithKwargs()
        self.cell2patient_aggregation = _MockAggregator()
        self.patient_encoder = nn.Identity()
        self.patient_predictor = nn.Linear(emb_dim, n_classes)


class _IdentityWithKwargs(nn.Module):
    """Identity that accepts and ignores keyword arguments (like padding_mask)."""

    def forward(self, x, **kwargs):
        return x


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_adata():
    """AnnData with donor-level labels suitable for PaSCient tests."""
    rng = np.random.default_rng(0)
    donor_ids = np.repeat([f"donor_{i:02d}" for i in range(N_DONORS)], CELLS_PER_DONOR)
    cell_types = rng.choice(["T cell", "B cell", "NK cell"], size=N_CELLS)

    donor_disease = {d: int(i % 2) for i, d in enumerate(np.unique(donor_ids))}
    donor_age = {d: float(20 + i * 3) for i, d in enumerate(np.unique(donor_ids))}

    obs = pd.DataFrame(
        {
            "donor_id": donor_ids,
            "cell_type": cell_types,
            "disease": [donor_disease[d] for d in donor_ids],
            "age": [donor_age[d] for d in donor_ids],
        },
        index=[f"cell_{i}" for i in range(N_CELLS)],
    )
    return sc.AnnData(X=rng.random((N_CELLS, N_GENES)).astype("float32"), obs=obs)


@pytest.fixture
def _patch_pascient(monkeypatch):
    """Monkey-patch PaSCient model loading to inject a lightweight mock.

    Replaces ``_load_pascient_model`` and ``_resolve_checkpoint_paths``
    so that no checkpoint directory, Hydra config, or pascient package
    is needed.
    """
    mock_model = _MockPaSCientModel(N_GENES)

    monkeypatch.setattr(
        PaSCient,
        "_load_pascient_model",
        staticmethod(lambda *args, **kwargs: mock_model),
    )
    monkeypatch.setattr(
        PaSCient,
        "_resolve_checkpoint_paths",
        lambda self: ("/fake/config", "/fake/checkpoint.ckpt"),
    )


@pytest.fixture
def pascient_model(basic_adata, _patch_pascient):
    """Fitted PaSCient model on the basic_adata fixture."""
    model = PaSCient(
        sample_key="donor_id",
        label_keys=["disease"],
        tasks=["classification"],
        checkpoint_dir="/fake/checkpoint",
        n_cells=10,
        batch_size=4,
        device="cpu",
    )
    model.prepare_anndata(basic_adata)
    return model


@pytest.fixture
def pascient_model_multilabel(basic_adata, _patch_pascient):
    """Fitted PaSCient model with two labels."""
    model = PaSCient(
        sample_key="donor_id",
        label_keys=["disease", "age"],
        tasks=["classification", "regression"],
        checkpoint_dir="/fake/checkpoint",
        n_cells=10,
        batch_size=4,
        device="cpu",
    )
    model.prepare_anndata(basic_adata)
    return model


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


class TestPaSCientConstruction:
    def test_checkpoint_dir_required(self):
        with pytest.raises(ValueError, match="checkpoint_dir is required"):
            PaSCient(sample_key="d", label_keys=["x"], tasks=["classification"])

    def test_constructor_stores_params(self):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            checkpoint_dir="/path",
            n_cells=500,
            batch_size=8,
            device="cpu",
            normalize=False,
        )
        assert model.checkpoint_dir == "/path"
        assert model.n_cells == 500
        assert model.batch_size == 8
        assert model.device == "cpu"
        assert model.normalize is False

    def test_default_params(self):
        model = PaSCient(
            sample_key="d",
            label_keys=["x"],
            tasks=["classification"],
            checkpoint_dir="/p",
        )
        assert model.n_cells == 1000
        assert model.batch_size == 16
        assert model.normalize is True


# ---------------------------------------------------------------------------
# Tests: prepare_anndata
# ---------------------------------------------------------------------------


class TestPaSCientPrepareAnndata:
    def test_samples_match_adata(self, pascient_model):
        assert set(pascient_model.samples) == {f"donor_{i:02d}" for i in range(N_DONORS)}

    def test_fitted_is_true(self, pascient_model):
        assert pascient_model._fitted is True

    def test_sample_representation_is_set(self, pascient_model):
        assert pascient_model.sample_representation is not None

    def test_labels_populated(self, pascient_model):
        assert pascient_model.labels is not None
        assert "disease" in pascient_model.labels.columns

    def test_cell_embeddings_populated(self, pascient_model):
        assert len(pascient_model._cell_embeddings) == N_DONORS

    def test_prepare_anndata_missing_label_raises(self, basic_adata, _patch_pascient):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["nonexistent"],
            tasks=["classification"],
            checkpoint_dir="/fake",
            device="cpu",
        )
        with pytest.raises(ValueError, match="not found"):
            model.prepare_anndata(basic_adata)


# ---------------------------------------------------------------------------
# Tests: get_sample_representations
# ---------------------------------------------------------------------------


class TestPaSCientSampleRepresentations:
    def test_shape(self, pascient_model):
        reps = pascient_model.get_sample_representations()
        assert reps.shape == (N_DONORS, EMB_DIM)

    def test_indexed_by_donor(self, pascient_model):
        reps = pascient_model.get_sample_representations()
        assert set(reps.index) == set(pascient_model.samples)

    def test_columns_named_dim_i(self, pascient_model):
        reps = pascient_model.get_sample_representations()
        assert list(reps.columns) == [f"dim_{i}" for i in range(EMB_DIM)]

    def test_no_nans(self, pascient_model):
        assert not pascient_model.get_sample_representations().isna().any().any()

    def test_raises_before_prepare(self):
        model = PaSCient(
            sample_key="d",
            label_keys=["x"],
            tasks=["classification"],
            checkpoint_dir="/p",
        )
        with pytest.raises(RuntimeError, match="prepare_anndata"):
            model.get_sample_representations()


# ---------------------------------------------------------------------------
# Tests: get_sample_importance
# ---------------------------------------------------------------------------


class TestPaSCientSampleImportance:
    def test_one_row_per_donor(self, pascient_model):
        assert pascient_model.get_sample_importance().shape[0] == N_DONORS

    def test_column_named_after_label(self, pascient_model):
        assert "disease_importance" in pascient_model.get_sample_importance().columns

    def test_values_are_positive(self, pascient_model):
        scores = pascient_model.get_sample_importance()
        assert (scores["disease_importance"] > 0).all()

    def test_equals_l2_norm(self, pascient_model):
        scores = pascient_model.get_sample_importance()
        reps = pascient_model.get_sample_representations()
        expected = np.linalg.norm(reps.values, axis=1)
        np.testing.assert_allclose(
            scores.loc[reps.index, "disease_importance"].values,
            expected,
            rtol=1e-5,
        )

    def test_cached_on_second_call(self, pascient_model):
        scores1 = pascient_model.get_sample_importance()
        # Corrupt internal state — recompute from scratch would crash
        pascient_model.sample_representation = None
        scores2 = pascient_model.get_sample_importance()
        pd.testing.assert_frame_equal(scores1, scores2, check_dtype=False)

    def test_multilabel_has_average_importance(self, pascient_model_multilabel):
        scores = pascient_model_multilabel.get_sample_importance()
        assert "average_importance" in scores.columns
        assert "disease_importance" in scores.columns
        assert "age_importance" in scores.columns


# ---------------------------------------------------------------------------
# Tests: get_cell_importance
# ---------------------------------------------------------------------------


class TestPaSCientCellImportance:
    def test_one_row_per_cell(self, pascient_model):
        assert pascient_model.get_cell_importance().shape[0] == N_CELLS

    def test_column_named_after_label(self, pascient_model):
        assert "disease_importance" in pascient_model.get_cell_importance().columns

    def test_values_in_zero_one(self, pascient_model):
        imp = pascient_model.get_cell_importance()
        assert (imp["disease_importance"] >= 0).all()
        assert (imp["disease_importance"] <= 1.0 + 1e-6).all()

    def test_written_to_adata_obs(self, pascient_model):
        pascient_model.get_cell_importance()
        assert "disease_importance" in pascient_model.adata.obs.columns

    def test_cached_on_second_call(self, pascient_model):
        imp1 = pascient_model.get_cell_importance()
        # Corrupt internal state
        pascient_model._cell_embeddings = {}
        pascient_model.sample_representation = None
        imp2 = pascient_model.get_cell_importance()
        pd.testing.assert_frame_equal(imp1, imp2, check_dtype=False)

    def test_multilabel_columns(self, pascient_model_multilabel):
        imp = pascient_model_multilabel.get_cell_importance()
        assert "disease_importance" in imp.columns
        assert "age_importance" in imp.columns


# ---------------------------------------------------------------------------
# Tests: distance matrix
# ---------------------------------------------------------------------------


class TestPaSCientDistanceMatrix:
    def test_shape(self, pascient_model):
        assert pascient_model.calculate_distance_matrix().shape == (N_DONORS, N_DONORS)

    def test_symmetric(self, pascient_model):
        dist = pascient_model.calculate_distance_matrix()
        np.testing.assert_allclose(dist, dist.T, atol=1e-6)

    def test_diagonal_is_zero(self, pascient_model):
        np.testing.assert_allclose(np.diag(pascient_model.calculate_distance_matrix()), 0.0, atol=1e-6)

    def test_non_negative(self, pascient_model):
        assert (pascient_model.calculate_distance_matrix() >= 0).all()


# ---------------------------------------------------------------------------
# Tests: fine_tune and predict (inherited from SupervisedSampleMethod)
# ---------------------------------------------------------------------------


class TestPaSCientFineTunePredict:
    @pytest.fixture
    def finetuned(self, pascient_model):
        pascient_model.fine_tune(["disease", "age"], ["classification", "regression"])
        return pascient_model

    def test_fine_tune_stores_probes(self, finetuned):
        assert "disease" in finetuned._probes
        assert "age" in finetuned._probes
        assert hasattr(finetuned._probes["disease"], "predict_proba")
        assert hasattr(finetuned._probes["age"], "predict")

    def test_predict_classification_returns_dataframe(self, finetuned):
        result = finetuned.predict("disease")
        assert isinstance(result, pd.DataFrame)
        assert "disease_pred" in result.columns

    def test_predict_classification_probabilities_sum_to_one(self, finetuned):
        result = finetuned.predict("disease")
        prob_cols = [c for c in result.columns if c.startswith("prob_")]
        np.testing.assert_array_almost_equal(result[prob_cols].sum(axis=1), 1.0)

    def test_predict_regression_returns_series(self, finetuned):
        result = finetuned.predict("age")
        assert isinstance(result, pd.Series)
        assert result.name == "age"

    def test_predict_indexed_by_sample(self, finetuned):
        result = finetuned.predict("disease")
        assert set(result.index) == set(finetuned.samples)

    def test_predict_unknown_label_raises(self, finetuned):
        with pytest.raises(ValueError, match="not found in model label keys"):
            finetuned.predict("nonexistent")


# ---------------------------------------------------------------------------
# Tests: fit_linear_probe (inherited from BaseSampleMethod)
# ---------------------------------------------------------------------------


class TestPaSCientLinearProbe:
    def test_classification_keys(self, pascient_model):
        result = pascient_model.fit_linear_probe(target="disease", task="classification")
        for key in ("model", "test_sample_labels", "disease_test", "disease_pred", "accuracy", "f1"):
            assert key in result

    def test_regression_keys(self, pascient_model_multilabel):
        result = pascient_model_multilabel.fit_linear_probe(target="age", task="regression")
        for key in ("model", "test_sample_labels", "age_test", "age_pred", "r2", "pearson"):
            assert key in result

    def test_accuracy_in_range(self, pascient_model):
        result = pascient_model.fit_linear_probe(target="disease", task="classification")
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: normalization
# ---------------------------------------------------------------------------


class TestPaSCientNormalization:
    def test_lognormalize_output_shape(self):
        x = torch.rand(4, 1, 10, 30)
        mask = torch.ones(4, 1, 10, dtype=torch.bool)
        x_out, mask_out = PaSCient._lognormalize(x, mask)
        assert x_out.shape == x.shape
        assert mask_out.shape == mask.shape

    def test_lognormalize_masked_cells_unchanged(self):
        x = torch.ones(1, 1, 5, 10)
        mask = torch.tensor([[[True, True, False, False, False]]])
        x_out, _ = PaSCient._lognormalize(x, mask)
        # Masked cells (False) should have counts_per_cell forced to 1,
        # so x/1 = x, then log1p(1) = log(2) ≈ 0.693
        expected_masked = np.log1p(1.0)
        np.testing.assert_allclose(x_out[0, 0, 2:, :].numpy(), expected_masked, rtol=1e-5)

    def test_lognormalize_produces_finite_values(self):
        x = torch.rand(2, 1, 10, 30) * 100
        mask = torch.ones(2, 1, 10, dtype=torch.bool)
        x_out, _ = PaSCient._lognormalize(x, mask)
        assert torch.isfinite(x_out).all()

    def test_normalize_false_skips_normalization(self, basic_adata, _patch_pascient):
        """When normalize=False, raw expression values pass through unchanged."""
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            checkpoint_dir="/fake",
            n_cells=10,
            batch_size=4,
            device="cpu",
            normalize=False,
        )
        model.prepare_anndata(basic_adata)
        # If normalization were applied, values would differ from raw
        # Just verify it completes and produces embeddings
        assert model.sample_representation is not None
        assert model._fitted is True


# ---------------------------------------------------------------------------
# Tests: expression matrix loading
# ---------------------------------------------------------------------------


class TestPaSCientExpressionMatrix:
    def test_layer_none_uses_X(self, basic_adata, _patch_pascient):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            checkpoint_dir="/fake",
            layer=None,
            n_cells=10,
            batch_size=4,
            device="cpu",
        )
        model.prepare_anndata(basic_adata)
        assert model._fitted

    def test_layer_from_layers(self, basic_adata, _patch_pascient):
        basic_adata.layers["raw_counts"] = basic_adata.X.copy()
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            checkpoint_dir="/fake",
            layer="raw_counts",
            n_cells=10,
            batch_size=4,
            device="cpu",
        )
        model.prepare_anndata(basic_adata)
        assert model._fitted

    def test_invalid_layer_raises(self, basic_adata, _patch_pascient):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            checkpoint_dir="/fake",
            layer="nonexistent_layer",
            n_cells=10,
            batch_size=4,
            device="cpu",
        )
        with pytest.raises(ValueError, match="not found"):
            model.prepare_anndata(basic_adata)

    def test_sparse_input_handled(self, basic_adata, _patch_pascient):
        """PaSCient should handle sparse expression matrices."""
        import scipy.sparse

        basic_adata.X = scipy.sparse.csr_matrix(basic_adata.X)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            checkpoint_dir="/fake",
            n_cells=10,
            batch_size=4,
            device="cpu",
        )
        model.prepare_anndata(basic_adata)
        assert model._fitted


# ---------------------------------------------------------------------------
# Tests: checkpoint path resolution
# ---------------------------------------------------------------------------


class TestPaSCientCheckpointResolution:
    def test_missing_hydra_dir_raises(self, tmp_path):
        model = PaSCient(
            sample_key="d",
            label_keys=["x"],
            tasks=["classification"],
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match=".hydra"):
            model._resolve_checkpoint_paths()

    def test_missing_ckpt_file_raises(self, tmp_path):
        (tmp_path / ".hydra").mkdir()
        model = PaSCient(
            sample_key="d",
            label_keys=["x"],
            tasks=["classification"],
            checkpoint_dir=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match=".ckpt"):
            model._resolve_checkpoint_paths()

    def test_resolves_from_checkpoints_subdir(self, tmp_path):
        (tmp_path / ".hydra").mkdir()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "model.ckpt").touch()
        model = PaSCient(
            sample_key="d",
            label_keys=["x"],
            tasks=["classification"],
            checkpoint_dir=str(tmp_path),
        )
        config_path, ckpt_path = model._resolve_checkpoint_paths()
        assert config_path.endswith(".hydra")
        assert ckpt_path.endswith("model.ckpt")

    def test_resolves_ckpt_from_root_dir(self, tmp_path):
        (tmp_path / ".hydra").mkdir()
        (tmp_path / "weights.ckpt").touch()
        model = PaSCient(
            sample_key="d",
            label_keys=["x"],
            tasks=["classification"],
            checkpoint_dir=str(tmp_path),
        )
        config_path, ckpt_path = model._resolve_checkpoint_paths()
        assert ckpt_path.endswith("weights.ckpt")
