"""Tests for the PaSCient supervised method wrapper.

All tests are skipped when ``torch`` is not installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

_has_torch = importlib.util.find_spec("torch") is not None

# Skip the entire module at collection time when torch is absent.
# pytest.importorskip raises Skipped during collection, preventing
# ImportErrors from torch/nn class definitions below.
torch = pytest.importorskip("torch", reason="torch not installed")
nn = torch.nn

from patpy.tl.supervised import PaSCient  # noqa: E402

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


class _MockLossConfig:
    """Minimal loss config matching SamplePredictor expectations."""

    def __init__(self, labels):
        self.sample_prediction_loss = _MockSamplePredLoss(labels)


class _MockSamplePredLoss:
    def __init__(self, labels):
        self.weight = 1.0
        self.labels = labels


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
        self.losses = _MockLossConfig(["mock_label"])
        self.prediction_labels = ["mock_label"]
        self.sample_prediction_loss_func = nn.CrossEntropyLoss()


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
    def test_no_checkpoint_and_no_train_raises(self, basic_adata):
        model = PaSCient(sample_key="donor_id", label_keys=["disease"], tasks=["classification"], device="cpu")
        with pytest.raises(ValueError, match="checkpoint_dir or.*train=True"):
            model.prepare_anndata(basic_adata)

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
        assert model.n_cells == 1500
        assert model.batch_size == 16
        assert model.normalize is True
        assert model.n_epochs == 4
        assert model.lr == 1e-4
        assert model.weight_decay == 1e-4
        assert model.latent_dim == 1024
        assert model.patient_emb_dim == 512
        assert model.seed == 12345


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


_has_captum = _has_torch and importlib.util.find_spec("captum") is not None


class TestPaSCientCellImportance:
    def test_one_row_per_cell(self, pascient_model):
        assert pascient_model.get_cell_importance().shape[0] == N_CELLS

    def test_column_named_after_label(self, pascient_model):
        assert "disease_importance" in pascient_model.get_cell_importance().columns

    def test_values_non_negative(self, pascient_model):
        imp = pascient_model.get_cell_importance()
        assert (imp["disease_importance"] >= 0).all()

    def test_written_to_adata_obs(self, pascient_model):
        pascient_model.get_cell_importance()
        assert "disease_importance" in pascient_model.adata.obs.columns

    def test_cached_on_second_call(self, pascient_model):
        imp1 = pascient_model.get_cell_importance()
        # Corrupt internal state
        pascient_model._cell_embeddings = {}
        pascient_model.sample_representation = None
        pascient_model._pascient_model = None
        imp2 = pascient_model.get_cell_importance()
        pd.testing.assert_frame_equal(imp1, imp2, check_dtype=False)

    def test_multilabel_columns(self, pascient_model_multilabel):
        imp = pascient_model_multilabel.get_cell_importance()
        assert "disease_importance" in imp.columns
        assert "age_importance" in imp.columns

    @pytest.mark.skipif(not _has_captum, reason="captum not installed")
    def test_ig_produces_scores(self, pascient_model):
        """IG-based importance should produce non-negative scores."""
        scores = pascient_model._cell_importance_ig(target=0)
        assert len(scores) == N_CELLS
        assert (scores >= 0).all()

    def test_cosine_fallback_produces_scores(self, pascient_model):
        """Cosine fallback should produce values in [0, 1]."""
        scores = pascient_model._cell_importance_cosine()
        assert len(scores) == N_CELLS
        assert (scores >= 0).all()
        assert (scores <= 1.0 + 1e-6).all()


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


# ---------------------------------------------------------------------------
# Tests: end-to-end training
# ---------------------------------------------------------------------------


def _mock_train(self, adata, *, label_key=None, task=None):
    """Lightweight training stand-in that builds a mock model and runs one forward pass.

    After training, moves the model to CPU to simulate Lightning Trainer
    teardown behaviour (which moves modules off the accelerator device).
    Downstream methods (``_extract_embeddings``, ``_predict_native``) are
    responsible for moving the model back to ``self.device`` before inference.
    """
    expression = self._get_expression_matrix()
    n_genes = expression.shape[1]
    label_key = label_key or self.label_keys[0]
    task = task or self.tasks[0]
    label_vals = self.labels[label_key].values

    if task == "classification":
        classes = sorted(np.unique(label_vals))
        n_classes = len(classes)
        self._class_names = list(classes)
    else:
        n_classes = 1
        self._class_names = None

    self._trained_label = label_key

    if self._pascient_model is None:
        self._pascient_model = _MockPaSCientModel(n_genes, n_classes=n_classes)

    model = self._pascient_model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
    x = torch.randn(1, 1, self.n_cells, n_genes)
    pad = torch.ones(1, 1, self.n_cells, dtype=torch.bool)
    preds = self._forward_model(x, pad)[2].squeeze(1)
    loss = torch.nn.functional.cross_entropy(preds, torch.zeros(1, dtype=torch.long))
    loss.backward()
    optimizer.step()
    # Simulate Lightning Trainer teardown: move model to CPU
    model.to("cpu")
    model.eval()


@pytest.fixture
def _patch_build_model(monkeypatch):
    """Patch _build_model and _train to avoid importing pascient/lightning."""
    monkeypatch.setattr(
        PaSCient,
        "_build_model",
        lambda self, n_genes, n_classes: _MockPaSCientModel(n_genes, n_classes=n_classes),
    )
    monkeypatch.setattr(PaSCient, "_train", _mock_train)


class TestPaSCientTraining:
    def test_train_from_scratch(self, basic_adata, _patch_build_model):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=2,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        assert model._fitted
        assert model.sample_representation is not None
        assert model.sample_representation.shape[0] == N_DONORS

    def test_train_regression(self, basic_adata, _patch_build_model):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["age"],
            tasks=["regression"],
            n_cells=10,
            batch_size=4,
            n_epochs=2,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        assert model._fitted

    def test_train_then_predict(self, basic_adata, _patch_build_model):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=2,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("disease", "classification")
        preds = model.predict("disease")
        assert isinstance(preds, pd.DataFrame)
        assert "disease_pred" in preds.columns

    def test_train_produces_embeddings(self, basic_adata, _patch_build_model):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=2,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        reps = model.get_sample_representations()
        assert reps.shape == (N_DONORS, EMB_DIM)
        assert not reps.isna().any().any()


# ---------------------------------------------------------------------------
# Tests: device consistency after training
# ---------------------------------------------------------------------------


def _mock_train_moves_to_cpu(self, adata, *, label_key=None, task=None):
    """Like _mock_train — simulates Lightning teardown moving model to CPU.
    Callers (``_extract_embeddings``, ``_predict_native``) must handle the move back."""
    expression = self._get_expression_matrix()
    n_genes = expression.shape[1]
    label_key = label_key or self.label_keys[0]
    task = task or self.tasks[0]
    label_vals = self.labels[label_key].values

    if task == "classification":
        classes = sorted(np.unique(label_vals))
        n_classes = len(classes)
        self._class_names = list(classes)
    else:
        n_classes = 1
        self._class_names = None

    self._trained_label = label_key

    if self._pascient_model is None:
        self._pascient_model = _MockPaSCientModel(n_genes, n_classes=n_classes)

    model = self._pascient_model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
    x = torch.randn(1, 1, self.n_cells, n_genes)
    pad = torch.ones(1, 1, self.n_cells, dtype=torch.bool)
    preds = self._forward_model(x, pad)[2].squeeze(1)
    loss = torch.nn.functional.cross_entropy(preds, torch.zeros(1, dtype=torch.long))
    loss.backward()
    optimizer.step()
    model.to("cpu")
    model.eval()


class TestPaSCientDeviceConsistency:
    """Verify that model parameters stay on ``self.device`` after every operation."""

    @pytest.fixture
    def _patch_build_model_cpu(self, monkeypatch):
        monkeypatch.setattr(
            PaSCient,
            "_build_model",
            lambda self, n_genes, n_classes: _MockPaSCientModel(n_genes, n_classes=n_classes),
        )
        monkeypatch.setattr(PaSCient, "_train", _mock_train_moves_to_cpu)

    def _assert_model_on_device(self, model):
        expected = torch.device(model.device)
        for name, param in model._pascient_model.named_parameters():
            assert param.device == expected, f"Parameter {name} is on {param.device}, expected {expected}"

    def test_model_on_device_after_prepare_anndata(self, basic_adata, _patch_build_model_cpu):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        self._assert_model_on_device(model)

    def test_model_on_device_after_fine_tune_same_label(self, basic_adata, _patch_build_model_cpu):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("disease", "classification")
        self._assert_model_on_device(model)

    def test_model_on_device_after_fine_tune_new_label(self, basic_adata, _patch_build_model_cpu):
        basic_adata.obs["age"] = np.tile(np.arange(20, 20 + N_DONORS, dtype=float), CELLS_PER_DONOR)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            patient_emb_dim=EMB_DIM,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("age", tasks="regression")
        self._assert_model_on_device(model)

    def test_extract_embeddings_after_train(self, basic_adata, _patch_build_model_cpu):
        """_extract_embeddings should succeed even when _train moved model to CPU."""
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        # Re-extract should not raise a device mismatch error
        model._extract_embeddings(basic_adata)
        assert model.sample_representation is not None

    def test_forward_model_consistent_device(self, basic_adata, _patch_build_model_cpu):
        """_forward_model should work with tensors on self.device."""
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        x = torch.randn(1, 1, 10, N_GENES, device=model.device)
        pad = torch.ones(1, 1, 10, dtype=torch.bool, device=model.device)
        patient_emb, cell_emb, preds = model._forward_model(x, pad)
        assert patient_emb.device == torch.device(model.device)


# ---------------------------------------------------------------------------
# Tests: predict without fine_tune for training label
# ---------------------------------------------------------------------------


class TestPaSCientPredictNative:
    """Verify that predict() works for the training label without fine_tune()."""

    def test_predict_training_label_without_fine_tune(self, basic_adata, _patch_build_model):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        preds = model.predict("disease")
        assert isinstance(preds, pd.DataFrame)
        assert "disease_pred" in preds.columns
        assert set(preds.index) == set(model.samples)

    def test_predict_training_label_probabilities_sum_to_one(self, basic_adata, _patch_build_model):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        preds = model.predict("disease")
        prob_cols = [c for c in preds.columns if c.startswith("prob_")]
        np.testing.assert_array_almost_equal(preds[prob_cols].sum(axis=1), 1.0)

    def test_predict_new_label_requires_fine_tune(self, basic_adata, _patch_build_model):
        """predict() for a label not used in training should require fine_tune."""
        basic_adata.obs["age"] = np.tile(np.arange(20, 20 + N_DONORS, dtype=float), CELLS_PER_DONOR)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        with pytest.raises((RuntimeError, ValueError)):
            model.predict("age")

    def test_fine_tune_new_label_then_predict(self, basic_adata, _patch_build_model):
        """After fine-tuning for a new label, the native head predicts it."""
        basic_adata.obs["age"] = np.tile(np.arange(20, 20 + N_DONORS, dtype=float), CELLS_PER_DONOR)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            patient_emb_dim=EMB_DIM,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("age", tasks="regression")
        # Fine-tuned label should work via native head
        age_preds = model.predict("age")
        assert isinstance(age_preds, pd.Series)

    def test_original_label_needs_probe_after_new_fine_tune(self, basic_adata, _patch_build_model):
        """After fine-tuning a new label, original label needs sklearn probe."""
        basic_adata.obs["age"] = np.tile(np.arange(20, 20 + N_DONORS, dtype=float), CELLS_PER_DONOR)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            patient_emb_dim=EMB_DIM,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("age", tasks="regression")
        # Original label no longer has native head — needs sklearn probe
        with pytest.raises(RuntimeError, match="fine_tune"):
            model.predict("disease")
        # But sklearn fine-tune + predict works
        model.fine_tune("disease", tasks="classification")
        preds = model.predict("disease")
        assert "disease_pred" in preds.columns


# ---------------------------------------------------------------------------
# Integration tests: real pascient components (no mocks)
# ---------------------------------------------------------------------------

_has_pascient = importlib.util.find_spec("pascient") is not None
_skip_no_pascient = pytest.mark.skipif(not _has_pascient, reason="pascient not installed")

SMALL_LATENT = 32  # keep integration tests fast
SMALL_EMB = 16


@_skip_no_pascient
class TestPaSCientIntegration:
    """Tests that run real pascient model components end-to-end.

    Skipped when the ``pascient`` package is not installed.  Uses tiny
    dimensions (latent_dim=32, patient_emb_dim=16) so the tests stay
    fast.
    """

    def test_build_model_returns_sample_predictor(self, basic_adata):
        from pascient.model.sample_predictor import SamplePredictor

        from patpy.tl.supervised import SupervisedSampleMethod

        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        # Only populate labels without running full prepare_anndata
        # (which requires checkpoint_dir or train=True)
        SupervisedSampleMethod.prepare_anndata(model, basic_adata)
        predictor = model._build_model(n_genes=N_GENES, n_classes=2)
        assert isinstance(predictor, SamplePredictor)

    def test_train_from_scratch_real(self, basic_adata):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        assert model._fitted
        assert model.sample_representation is not None
        assert model.sample_representation.shape == (N_DONORS, SMALL_EMB)

    def test_sample_importance_real(self, basic_adata):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        scores = model.get_sample_importance()
        assert scores.shape[0] == N_DONORS
        assert "disease_importance" in scores.columns

    def test_cell_importance_real(self, basic_adata):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        imp = model.get_cell_importance()
        assert imp.shape[0] == N_CELLS
        assert (imp["disease_importance"] >= 0).all()

    def test_fine_tune_and_predict_real(self, basic_adata):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("disease", "classification")
        preds = model.predict("disease")
        assert isinstance(preds, pd.DataFrame)
        assert "disease_pred" in preds.columns
        assert set(preds.index) == set(model.samples)

    def test_distance_matrix_real(self, basic_adata):
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        dist = model.calculate_distance_matrix()
        assert dist.shape == (N_DONORS, N_DONORS)
        np.testing.assert_allclose(dist, dist.T, atol=1e-6)

    def test_model_on_device_after_train_real(self, basic_adata):
        """After real Lightning training, model should remain on self.device."""
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        expected = torch.device(model.device)
        for name, param in model._pascient_model.named_parameters():
            assert param.device == expected, f"{name} on {param.device}, expected {expected}"

    def test_model_on_device_after_fine_tune_new_label_real(self, basic_adata):
        """After fine-tuning with a new label, model should stay on self.device."""
        basic_adata.obs["age"] = np.tile(np.arange(20, 20 + N_DONORS, dtype=float), CELLS_PER_DONOR)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("age", tasks="regression")
        expected = torch.device(model.device)
        for name, param in model._pascient_model.named_parameters():
            assert param.device == expected, f"{name} on {param.device}, expected {expected}"

    def test_predict_training_label_without_fine_tune_real(self, basic_adata):
        """Native predict() should work without calling fine_tune() first."""
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        preds = model.predict("disease")
        assert isinstance(preds, pd.DataFrame)
        assert "disease_pred" in preds.columns
        prob_cols = [c for c in preds.columns if c.startswith("prob_")]
        np.testing.assert_array_almost_equal(preds[prob_cols].sum(axis=1), 1.0)

    def test_fine_tune_new_label_then_predict_real(self, basic_adata):
        """After fine-tuning a new label, native head predicts it."""
        basic_adata.obs["age"] = np.tile(np.arange(20, 20 + N_DONORS, dtype=float), CELLS_PER_DONOR)
        model = PaSCient(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            n_cells=10,
            batch_size=4,
            n_epochs=1,
            latent_dim=SMALL_LATENT,
            patient_emb_dim=SMALL_EMB,
            device="cpu",
        )
        model.prepare_anndata(basic_adata, train=True)
        model.fine_tune("age", tasks="regression")
        age_preds = model.predict("age")
        assert isinstance(age_preds, pd.Series)
        # Original label requires sklearn probe after new fine-tune
        with pytest.raises(RuntimeError, match="fine_tune"):
            model.predict("disease")
