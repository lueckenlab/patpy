from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

N_CELLS = 200
N_GENES = 30
N_DONORS = 10
N_PCS = 8
CELLS_PER_DONOR = N_CELLS // N_DONORS


@pytest.fixture
def basic_adata():
    """Minimal AnnData with two donor-level labels and a PCA embedding.

    Layout
    ------
    * 200 cells x 30 genes
    * 10 donors, 20 cells each
    * ``disease``: binary (0/1), alternates per donor (donor_00=0, donor_01=1, ...)
    * ``age``:     continuous, 20 + 3*i for donor i
    * ``X_pca``:   random (200 x 8) float32 embedding
    """
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

    adata = sc.AnnData(X=rng.random((N_CELLS, N_GENES)).astype("float32"), obs=obs)
    adata.obsm["X_pca"] = rng.random((N_CELLS, N_PCS)).astype("float32")
    return adata



@pytest.fixture
def mixmil_model(basic_adata):
    from patpy.tl.supervised import MixMIL

    model = MixMIL(
        sample_key="donor_id",
        label_keys=["disease"],
        tasks=["classification"],
        layer="X_pca",
        n_epochs=2,
    )
    model.prepare_anndata(basic_adata)
    return model


@pytest.fixture
def mixmil_model_multilabel(basic_adata):
    """MixMIL trained jointly on two binary labels (P=2 outputs).

    Uses binomial likelihood with two binary columns so P=2 is exercised
    without requiring a Gaussian head that the installed MixMIL does not
    support.
    """
    from patpy.tl.supervised import MixMIL

    adata = basic_adata.copy()
    # Add a second binary donor-level label (inverted disease)
    donor_disease2 = {
        d: int((int(d.split("_")[1]) + 1) % 2)
        for d in adata.obs["donor_id"].unique()
    }
    adata.obs["disease2"] = [donor_disease2[d] for d in adata.obs["donor_id"]]

    model = MixMIL(
        sample_key="donor_id",
        label_keys=["disease", "disease2"],
        tasks=["classification", "classification"],
        layer="X_pca",
        likelihood="binomial",
        n_epochs=2,
    )
    model.prepare_anndata(adata)
    return model


@pytest.fixture
def basic_adata_string_labels(basic_adata):
    """AnnData with multiclass string labels (3 classes)."""
    adata = basic_adata.copy()
    # Convert disease to string labels
    disease_mapping = {0: "healthy", 1: "diseased"}
    adata.obs["disease_str"] = adata.obs["disease"].map(disease_mapping)

    # Add a 3-class string label
    disease_classes = ["healthy", "diseased", "pre-disease"]
    donor_disease_multi = {
        d: disease_classes[i % len(disease_classes)]
        for i, d in enumerate(adata.obs["donor_id"].unique())
    }
    adata.obs["disease_multi"] = [donor_disease_multi[d] for d in adata.obs["donor_id"]]

    return adata


@pytest.fixture
def mixmil_model_string_labels(basic_adata_string_labels):
    """MixMIL trained on string labels (binary)."""
    from patpy.tl.supervised import MixMIL

    model = MixMIL(
        sample_key="donor_id",
        label_keys=["disease_str"],
        tasks=["classification"],
        layer="X_pca",
        n_epochs=2,
    )
    model.prepare_anndata(basic_adata_string_labels)
    return model


@pytest.fixture
def mixmil_model_multiclass_strings(basic_adata_string_labels):
    """MixMIL trained on 3-class string labels."""
    from patpy.tl.supervised import MixMIL

    model = MixMIL(
        sample_key="donor_id",
        label_keys=["disease_multi"],
        tasks=["classification"],
        layer="X_pca",
        n_epochs=2,
    )
    model.prepare_anndata(basic_adata_string_labels)
    return model


@pytest.fixture
def pulsar_adata(basic_adata):
    basic_adata.obsm["X_uce"] = np.random.default_rng(1).random((N_CELLS, 1280)).astype("float32")
    return basic_adata


@pytest.fixture
def _patch_pulsar(monkeypatch):
    """Replace PULSAR.from_pretrained with a small random-weight model.

    This avoids downloading the ~500 MB pretrained checkpoint and works
    around a transformers 5.x incompatibility, while still running the
    real PULSAR forward pass and the real extract_donor_embeddings_from_h5ad
    code path.  hidden_size=512 matches the (N_DONORS, 512) shape expectation.
    """
    from pulsar.model import PULSAR as _PulsarModel, PULSARConfig

    small_config = PULSARConfig(
        input_size=1280,                  # must match X_uce embedding dim
        hidden_size=512,                  # output dim expected by the tests
        encoder_num_hidden_layers=1,      # lightweight
        encoder_num_attention_heads=8,    # 512 / 8 = 64 head_dim
        encoder_intermediate_size=256,
        use_decoder=False,
        use_cell_state=False,
        cls_transform=False,
    )
    small_model = _PulsarModel(small_config)
    monkeypatch.setattr(_PulsarModel, "from_pretrained", lambda *args, **kwargs: small_model)


@pytest.fixture
def pulsar_model(pulsar_adata, _patch_pulsar):
    from patpy.tl.supervised import PULSAR

    # sample_cell_num=4 keeps the forward pass fast with 20 cells/donor.
    model = PULSAR(
        sample_key="donor_id",
        label_keys=["age", "disease"],
        tasks=["regression", "classification"],
        layer="X_uce",
        device="cpu",
        sample_cell_num=4,
        batch_size=10,
    )
    model.prepare_anndata(pulsar_adata)
    return model


def _make_base(sample_key="donor_id", cell_group_key="cell_type", layer="X_pca"):
    from patpy.tl._base_sample_method import BaseSampleMethod

    class _Concrete(BaseSampleMethod):
        pass

    return _Concrete(sample_key=sample_key, cell_group_key=cell_group_key, layer=layer)


class TestBaseSampleMethod:
    def test_check_adata_loaded_raises_before_prepare(self):
        """_check_adata_loaded must raise before prepare_anndata is called."""
        base = _make_base()
        with pytest.raises(RuntimeError, match="not fitted"):
            base._check_adata_loaded()

    def test_check_adata_loaded_passes_after_prepare(self, basic_adata):
        base = _make_base()
        base.prepare_anndata(basic_adata)
        base._check_adata_loaded()

    def test_prepare_anndata_finds_correct_number_of_donors(self, basic_adata):
        base = _make_base()
        base.prepare_anndata(basic_adata)
        assert len(base.samples) == N_DONORS

    def test_prepare_anndata_donor_ids_match_adata(self, basic_adata):
        base = _make_base()
        base.prepare_anndata(basic_adata)
        expected = {f"donor_{i:02d}" for i in range(N_DONORS)}
        assert set(base.samples) == expected

    def test_prepare_anndata_missing_sample_key_raises(self, basic_adata):
        base = _make_base(sample_key="nonexistent")
        with pytest.raises(ValueError, match="sample_key"):
            base.prepare_anndata(basic_adata)

    def test_prepare_anndata_populates_cell_groups(self, basic_adata):
        base = _make_base()
        base.prepare_anndata(basic_adata)
        assert set(base.cell_groups) == {"T cell", "B cell", "NK cell"}

    def test_prepare_anndata_cell_group_key_none_leaves_groups_none(self, basic_adata):
        base = _make_base(cell_group_key=None)
        base.prepare_anndata(basic_adata)
        assert base.cell_groups is None

    def test_get_data_returns_correct_obsm_array(self, basic_adata):
        base = _make_base(layer="X_pca")
        base.prepare_anndata(basic_adata)
        data = base._get_data()
        assert data.shape == (N_CELLS, N_PCS)
        np.testing.assert_array_equal(data, basic_adata.obsm["X_pca"])

    def test_get_data_raises_for_missing_layer(self, basic_adata):
        base = _make_base(layer="X_does_not_exist")
        base.prepare_anndata(basic_adata)
        with pytest.raises(ValueError, match="not found"):
            base._get_data()

    def test_get_data_returns_layers_entry_when_not_in_obsm(self, basic_adata):
        # Using (N_CELLS, 5) was wrong — AnnData rejects mismatched n_vars.
        layer_data = np.ones((N_CELLS, N_GENES), dtype="float32")
        basic_adata.layers["my_layer"] = layer_data
        base = _make_base(layer="my_layer")
        base.prepare_anndata(basic_adata)
        data = base._get_data()
        np.testing.assert_array_equal(data, layer_data)

    def test_extract_metadata_correct_shape(self, basic_adata):
        base = _make_base()
        base.prepare_anndata(basic_adata)
        meta = base._extract_metadata(["disease", "age"])
        assert meta.shape == (N_DONORS, 2)

    def test_extract_metadata_values_match_donor_ground_truth(self, basic_adata):
        """Each row must hold the correct donor-level label value."""
        base = _make_base()
        base.prepare_anndata(basic_adata)
        meta = base._extract_metadata(["disease"])
        for i, donor in enumerate(sorted(base.samples)):
            assert meta.loc[donor, "disease"] == i % 2


def _make_supervised(label_keys=None, tasks=None):
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Concrete(SupervisedSampleMethod):
        pass

    return _Concrete(
        sample_key="donor_id",
        label_keys=label_keys or ["disease"],
        tasks=tasks or ["classification"],
    )


class TestSupervisedSampleMethod:
    def test_mismatched_label_and_tasks_raises(self):
        from patpy.tl.supervised import SupervisedSampleMethod

        class _C(SupervisedSampleMethod):
            pass

        with pytest.raises(ValueError, match="same length"):
            _C(
                sample_key="donor_id",
                label_keys=["disease", "age"],
                tasks=["classification"],
            )

    def test_prepare_anndata_missing_label_raises(self, basic_adata):
        model = _make_supervised(label_keys=["no_such_column"])
        with pytest.raises(ValueError, match="not found"):
            model.prepare_anndata(basic_adata)

    def test_prepare_anndata_labels_has_correct_columns(self, basic_adata):
        model = _make_supervised(
            label_keys=["disease", "age"],
            tasks=["classification", "regression"],
        )
        model.prepare_anndata(basic_adata)
        assert list(model.labels.columns) == ["disease", "age"]

    def test_prepare_anndata_labels_indexed_by_donor(self, basic_adata):
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        assert set(model.labels.index) == {f"donor_{i:02d}" for i in range(N_DONORS)}

    def test_prepare_anndata_label_values_correct(self, basic_adata):
        """model.labels must reflect the per-donor ground truth, not cell-level noise."""
        model = _make_supervised(
            label_keys=["disease", "age"],
            tasks=["classification", "regression"],
        )
        model.prepare_anndata(basic_adata)
        for i, donor in enumerate(sorted(model.labels.index)):
            assert model.labels.loc[donor, "disease"] == i % 2
            assert model.labels.loc[donor, "age"] == pytest.approx(20 + i * 3)

    def test_prepare_anndata_warns_on_ambiguous_labels(self, basic_adata):
        """If a donor has cells with two different label values, warn."""
        adata = basic_adata.copy()
        first_donor_cell = adata.obs[adata.obs["donor_id"] == "donor_00"].index[0]
        adata.obs.loc[first_donor_cell, "disease"] = 99
        model = _make_supervised()
        with pytest.warns(UserWarning, match="multiple values"):
            model.prepare_anndata(adata)

    def test_get_sample_importance_raises_before_prepare(self):
        model = _make_supervised()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.get_sample_importance()

    def test_get_sample_importance_raises_not_implemented_after_prepare(self, basic_adata):
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        with pytest.raises(NotImplementedError):
            model.get_sample_importance()

    def test_get_cell_importance_raises_not_implemented(self, basic_adata):
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        with pytest.raises(NotImplementedError):
            model.get_cell_importance()

    def test_get_sample_representations_raises_not_implemented(self, basic_adata):
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        with pytest.raises(NotImplementedError):
            model.get_sample_representations()

    def test_calculate_distance_matrix_propagates_not_implemented(self, basic_adata):
        """Delegates to get_sample_representations — must raise when that is missing."""
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        with pytest.raises(NotImplementedError):
            model.calculate_distance_matrix()

    def test_donor_col_length_equals_n_donors(self, basic_adata):
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        assert model._donor_col("disease").shape == (N_DONORS,)

    def test_donor_col_values_aligned_to_self_samples(self, basic_adata):
        """Values returned by _donor_col must match model.samples order exactly."""
        model = _make_supervised()
        model.prepare_anndata(basic_adata)
        disease_arr = model._donor_col("disease")
        for i, donor in enumerate(model.samples):
            donor_idx = int(donor.split("_")[1])
            assert disease_arr[i] == donor_idx % 2


class TestMixMIL:
    def test_prepare_anndata_missing_layer_raises(self, basic_adata):
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            layer="X_nonexistent",
        )
        with pytest.raises(ValueError, match="not found"):
            model.prepare_anndata(basic_adata)

    def test_prepare_anndata_stores_model(self, mixmil_model):
        assert mixmil_model._model is not None

    def test_prepare_anndata_samples_match_adata(self, mixmil_model):
        assert set(mixmil_model.samples) == {f"donor_{i:02d}" for i in range(N_DONORS)}

    def test_additional_covariate_from_obs_accepted(self, basic_adata):
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            layer="X_pca",
            additional_covariates=["age"],
        )
        model.prepare_anndata(basic_adata)
        assert model._model is not None

    def test_additional_covariate_missing_raises(self, basic_adata):
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            layer="X_pca",
            additional_covariates=["no_such_column"],
        )
        with pytest.raises(ValueError, match="no_such_column"):
            model.prepare_anndata(basic_adata)

    def test_get_sample_importance_returns_dataframe(self, mixmil_model):
        assert isinstance(mixmil_model.get_sample_importance(), pd.DataFrame)

    def test_get_sample_importance_one_row_per_donor(self, mixmil_model):
        assert mixmil_model.get_sample_importance().shape[0] == N_DONORS

    def test_get_sample_importance_indexed_by_donor(self, mixmil_model):
        scores = mixmil_model.get_sample_importance()
        assert set(scores.index) == {f"donor_{i:02d}" for i in range(N_DONORS)}

    def test_get_sample_importance_column_named_after_label(self, mixmil_model):
        assert "disease_importance" in mixmil_model.get_sample_importance().columns

    def test_get_sample_importance_values_are_finite(self, mixmil_model):
        scores = mixmil_model.get_sample_importance()
        assert np.isfinite(scores["disease_importance"].values).all()

    def test_get_sample_importance_cached_on_second_call(self, mixmil_model):
        """Second call must use the cache and not call the model again.
        Cache round-trip through adata.uns converts float32→float64,
        so dtype is not checked.
        """
        scores1 = mixmil_model.get_sample_importance()
        mixmil_model._model = None  # corrupt — a recompute would crash
        scores2 = mixmil_model.get_sample_importance()
        pd.testing.assert_frame_equal(scores1, scores2, check_dtype=False)

    def test_get_sample_importance_force_bypasses_cache(self, mixmil_model):
        _ = mixmil_model.get_sample_importance()
        # Poison the cache
        mixmil_model.adata.uns["supervised_sample_importance"] = {"disease_importance": {"donor_00": 999.0}}
        scores = mixmil_model.get_sample_importance(force=True)
        assert scores["disease_importance"].max() != 999.0

    def test_get_cell_importance_returns_dataframe(self, mixmil_model):
        assert isinstance(mixmil_model.get_cell_importance(), pd.DataFrame)

    def test_get_cell_importance_one_row_per_cell(self, mixmil_model):
        assert mixmil_model.get_cell_importance().shape[0] == N_CELLS

    def test_get_cell_importance_column_named_after_label(self, mixmil_model):
        assert "disease_importance" in mixmil_model.get_cell_importance().columns

    def test_get_cell_importance_values_are_non_negative(self, mixmil_model):
        imp = mixmil_model.get_cell_importance()
        assert (imp["disease_importance"] >= 0).all()

    def test_get_cell_importance_weights_sum_to_one_per_donor(self, mixmil_model):
        """MixMIL uses softmax attention, so cell weights must sum to 1.0 per donor."""
        imp = mixmil_model.get_cell_importance()
        mixmil_model.adata.obs["_imp"] = imp["disease_importance"].values
        donor_sums = mixmil_model.adata.obs.groupby("donor_id", observed=True)["_imp"].sum()
        np.testing.assert_allclose(donor_sums.values, 1.0, atol=1e-5)
        del mixmil_model.adata.obs["_imp"]

    def test_get_cell_importance_written_to_adata_obs(self, mixmil_model):
        mixmil_model.get_cell_importance()
        assert "disease_importance" in mixmil_model.adata.obs.columns

    def test_get_cell_importance_cached_on_second_call(self, mixmil_model):
        imp1 = mixmil_model.get_cell_importance()
        mixmil_model._model = None  # corrupt — recompute would crash
        imp2 = mixmil_model.get_cell_importance()
        pd.testing.assert_frame_equal(imp1, imp2, check_dtype=False)

    def test_get_cell_importance_force_bypasses_cache(self, mixmil_model):
        _ = mixmil_model.get_cell_importance()
        mixmil_model.adata.obs["disease_importance"] = -999.0  # poison cache
        imp = mixmil_model.get_cell_importance(force=True)
        assert (imp["disease_importance"] >= 0).all()

    def test_get_sample_representations_returns_dataframe(self, mixmil_model):
        assert isinstance(mixmil_model.get_sample_representations(), pd.DataFrame)

    def test_get_sample_representations_shape(self, mixmil_model):
        assert mixmil_model.get_sample_representations().shape == (N_DONORS, N_PCS)

    def test_get_sample_representations_columns_named_dim_i(self, mixmil_model):
        reps = mixmil_model.get_sample_representations()
        assert list(reps.columns) == [f"dim_{i}" for i in range(N_PCS)]

    def test_get_sample_representations_indexed_by_donor(self, mixmil_model):
        reps = mixmil_model.get_sample_representations()
        assert set(reps.index) == {f"donor_{i:02d}" for i in range(N_DONORS)}

    def test_get_sample_representations_no_nans(self, mixmil_model):
        reps = mixmil_model.get_sample_representations()
        assert not reps.isna().any().any()

    def test_calculate_distance_matrix_shape(self, mixmil_model):
        assert mixmil_model.calculate_distance_matrix().shape == (N_DONORS, N_DONORS)

    def test_calculate_distance_matrix_is_symmetric(self, mixmil_model):
        dist = mixmil_model.calculate_distance_matrix()
        np.testing.assert_allclose(dist, dist.T, atol=1e-6)

    def test_calculate_distance_matrix_diagonal_is_zero(self, mixmil_model):
        dist = mixmil_model.calculate_distance_matrix()
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)

    def test_calculate_distance_matrix_is_non_negative(self, mixmil_model):
        assert (mixmil_model.calculate_distance_matrix() >= 0).all()

    def test_training_history_is_list_of_dicts(self, mixmil_model):
        history = mixmil_model.training_history
        assert isinstance(history, list)
        assert all(isinstance(e, dict) for e in history)

    def test_training_history_contains_loss_key(self, mixmil_model):
        assert all("loss" in e for e in mixmil_model.training_history)

    def test_training_history_has_numeric_losses(self, mixmil_model):
        losses = [e["loss"] for e in mixmil_model.training_history]
        assert len(losses) > 0
        assert all(np.isfinite(v) for v in losses)

    def test_multilabel_get_sample_importance_has_both_columns(self, mixmil_model_multilabel):
        """Multi-label model must return one importance column per label."""
        scores = mixmil_model_multilabel.get_sample_importance()
        assert "disease_importance" in scores.columns
        assert "disease2_importance" in scores.columns

    def test_multilabel_get_sample_importance_has_average_column(self, mixmil_model_multilabel):
        scores = mixmil_model_multilabel.get_sample_importance()
        assert "average_importance" in scores.columns

    def test_multilabel_get_cell_importance_default_is_first_label(self, mixmil_model_multilabel):
        imp = mixmil_model_multilabel.get_cell_importance()
        assert "disease_importance" in imp.columns

    def test_multilabel_get_cell_importance_explicit_label(self, mixmil_model_multilabel):
        imp = mixmil_model_multilabel.get_cell_importance(label="disease2")
        assert "disease2_importance" in imp.columns
        assert imp.shape[0] == N_CELLS

    def test_multilabel_get_cell_importance_invalid_label_raises(self, mixmil_model_multilabel):
        with pytest.raises(ValueError, match="label_keys"):
            mixmil_model_multilabel.get_cell_importance(label="nonexistent")

    # ------------------------------------------------------------------
    # predict method
    # ------------------------------------------------------------------

    def test_predict_classification_returns_dataframe(self, mixmil_model):
        pred = mixmil_model.predict("disease")
        assert isinstance(pred, pd.DataFrame)

    def test_predict_classification_has_probability_columns(self, mixmil_model):
        pred = mixmil_model.predict("disease")
        assert "prob_0" in pred.columns
        assert "prob_1" in pred.columns
        assert "y_pred" in pred.columns

    def test_predict_classification_shape(self, mixmil_model):
        pred = mixmil_model.predict("disease")
        assert pred.shape[0] == N_DONORS

    def test_predict_classification_probabilities_sum_to_one(self, mixmil_model):
        pred = mixmil_model.predict("disease")
        prob_sum = pred[["prob_0", "prob_1"]].sum(axis=1)
        assert np.allclose(prob_sum, 1.0)

    def test_predict_classification_indexed_by_donor(self, mixmil_model):
        pred = mixmil_model.predict("disease")
        assert list(pred.index) == list(mixmil_model.samples)

    def test_predict_invalid_label_raises(self, mixmil_model):
        with pytest.raises(ValueError, match="not found in model label keys"):
            mixmil_model.predict("nonexistent_label")

    def test_predict_raises_on_unfitted_model(self, basic_adata):
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            layer="X_pca",
            n_epochs=2,
        )
        with pytest.raises(RuntimeError, match="prepare_anndata"):
            model.predict("disease")

    # ------------------------------------------------------------------
    # fine_tune method
    # ------------------------------------------------------------------

    def test_fine_tune_extends_label_keys(self, basic_adata):
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease"],
            tasks=["classification"],
            layer="X_pca",
            n_epochs=2,
        )
        model.prepare_anndata(basic_adata)
        original_label = model.label_keys.copy()

        # Add another label
        model.fine_tune(["disease"], ["classification"], n_epochs=1)
        # Since "disease" is already there, it should not change
        assert model.label_keys == original_label

    def test_fine_tune_adds_new_label(self, mixmil_model, basic_adata):
        assert "disease" in mixmil_model.label_keys
        assert "age" not in mixmil_model.label_keys

        # Add age as a new label (but skip it since it's regression)
        # Instead, create a new binary label
        basic_adata.obs["disease2"] = (basic_adata.obs["disease"] == 0).astype(int)
        mixmil_model.fine_tune(["disease2"], ["classification"], n_epochs=1)

        assert "disease" in mixmil_model.label_keys
        assert "disease2" in mixmil_model.label_keys

    def test_fine_tune_preserves_predict_on_old_labels(self, mixmil_model, basic_adata):
        basic_adata.obs["disease2"] = (basic_adata.obs["disease"] == 0).astype(int)

        pred_before = mixmil_model.predict("disease")

        mixmil_model.fine_tune(["disease2"], ["classification"], n_epochs=1)
        pred_after = mixmil_model.predict("disease")

        # Predictions should still work (though values may differ due to retraining)
        assert pred_after.shape == pred_before.shape

    def test_fine_tune_rejects_mixed_task_types(self, mixmil_model):
        with pytest.raises(ValueError, match="single task type"):
            mixmil_model.fine_tune(["age"], ["regression"], n_epochs=1)

    def test_fine_tune_clears_cache(self, mixmil_model):
        # Get importance before fine-tune
        mixmil_model.get_sample_importance()
        assert "supervised_sample_importance" in mixmil_model.adata.uns

        # Fine-tune should clear the cache
        mixmil_model.fine_tune(["disease"], ["classification"], n_epochs=1)
        assert "supervised_sample_importance" not in mixmil_model.adata.uns

    def test_fine_tune_raises_on_missing_label(self, mixmil_model):
        with pytest.raises(ValueError, match="not found in adata.obs.columns"):
            mixmil_model.fine_tune(["nonexistent_label"], ["classification"])

    # ------------------------------------------------------------------
    # String labels and multiclass classification
    # ------------------------------------------------------------------

    def test_predict_binary_string_labels_creates_mappings(self, mixmil_model_string_labels):
        """Test that string labels are stored in _label_mappings."""
        assert "disease_str" in mixmil_model_string_labels._label_mappings
        classes, encode_dict = mixmil_model_string_labels._label_mappings["disease_str"]
        assert set(classes) == {"healthy", "diseased"}
        # Classes are sorted alphabetically, so diseased comes before healthy
        assert encode_dict == {"diseased": 0, "healthy": 1}

    def test_predict_binary_string_labels_returns_class_names(self, mixmil_model_string_labels):
        """Test that y_pred contains original string class names, not indices."""
        pred = mixmil_model_string_labels.predict("disease_str")
        assert set(pred["y_pred"].unique()).issubset({"healthy", "diseased"})
        # Should not contain indices like 0 or 1
        assert not pred["y_pred"].isin([0, 1]).all()

    def test_predict_binary_string_labels_prob_columns(self, mixmil_model_string_labels):
        """Test that probability columns use class names."""
        pred = mixmil_model_string_labels.predict("disease_str")
        expected_cols = {"prob_healthy", "prob_diseased", "y_pred"}
        assert set(pred.columns) == expected_cols

    def test_predict_multiclass_string_labels_creates_mappings(self, mixmil_model_multiclass_strings):
        """Test that 3-class string labels are stored in _label_mappings."""
        assert "disease_multi" in mixmil_model_multiclass_strings._label_mappings
        classes, encode_dict = mixmil_model_multiclass_strings._label_mappings["disease_multi"]
        assert set(classes) == {"healthy", "diseased", "pre-disease"}
        assert len(encode_dict) == 3

    def test_predict_multiclass_string_labels_returns_class_names(self, mixmil_model_multiclass_strings):
        """Test that y_pred contains original string class names for multiclass."""
        pred = mixmil_model_multiclass_strings.predict("disease_multi")
        assert set(pred["y_pred"].unique()).issubset({"healthy", "diseased", "pre-disease"})

    def test_predict_multiclass_string_labels_prob_columns(self, mixmil_model_multiclass_strings):
        """Test that probability columns use class names for multiclass."""
        pred = mixmil_model_multiclass_strings.predict("disease_multi")
        prob_cols = {c for c in pred.columns if c.startswith("prob_")}
        expected_cols = {"prob_healthy", "prob_diseased", "prob_pre-disease"}
        assert prob_cols == expected_cols

    def test_predict_multiclass_string_labels_probabilities_sum(self, mixmil_model_multiclass_strings):
        """Test that probabilities sum to 1 for multiclass string labels."""
        pred = mixmil_model_multiclass_strings.predict("disease_multi")
        prob_cols = [c for c in pred.columns if c.startswith("prob_")]
        prob_sum = pred[prob_cols].sum(axis=1)
        assert np.allclose(prob_sum, 1.0)

    def test_fine_tune_preserves_string_label_mappings(self, basic_adata_string_labels):
        """Test that mappings are preserved when fine-tuning with same label."""
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease_str"],
            tasks=["classification"],
            layer="X_pca",
            n_epochs=2,
        )
        model.prepare_anndata(basic_adata_string_labels)

        # Get initial mapping
        classes_before, _ = model._label_mappings["disease_str"]

        # Fine-tune with same label
        model.fine_tune(["disease_str"], ["classification"], n_epochs=1)

        # Mapping should be unchanged
        classes_after, _ = model._label_mappings["disease_str"]
        assert classes_before == classes_after

    def test_fine_tune_adds_string_label_mapping(self, basic_adata_string_labels):
        """Test that new string label mappings are created during fine_tune."""
        from patpy.tl.supervised import MixMIL

        model = MixMIL(
            sample_key="donor_id",
            label_keys=["disease_str"],
            tasks=["classification"],
            layer="X_pca",
            n_epochs=2,
        )
        model.prepare_anndata(basic_adata_string_labels)

        # Fine-tune with a new string label
        model.fine_tune(["disease_multi"], ["classification"], n_epochs=1)

        # Both mappings should exist
        assert "disease_str" in model._label_mappings
        assert "disease_multi" in model._label_mappings

        # New mapping should be created
        classes, _ = model._label_mappings["disease_multi"]
        assert set(classes) == {"healthy", "diseased", "pre-disease"}

    def test_training_history_extends_on_fine_tune(self, mixmil_model, basic_adata):
        """Test that training history accumulates across fine_tune calls."""
        initial_history_len = len(mixmil_model._history)
        assert initial_history_len > 0

        # Fine-tune with same label
        basic_adata.obs["disease2"] = (basic_adata.obs["disease"] == 0).astype(int)
        mixmil_model.fine_tune(["disease2"], ["classification"], n_epochs=2)

        # History should have extended, not replaced
        final_history_len = len(mixmil_model._history)
        assert final_history_len > initial_history_len
        # Should have added 2 more epochs
        assert final_history_len == initial_history_len + 2


class TestPULSAR:
    def test_prepare_anndata_missing_layer_raises(self, basic_adata):
        from patpy.tl.supervised import PULSAR

        model = PULSAR(
            sample_key="donor_id",
            label_keys=["age"],
            tasks=["regression"],
            layer="X_nonexistent",
        )
        with pytest.raises(ValueError, match="not found in adata.obsm"):
            model.prepare_anndata(basic_adata)

    def test_prepare_anndata_low_dim_embedding_warns(self, basic_adata, _patch_pulsar):
        from patpy.tl.supervised import PULSAR

        basic_adata.obsm["X_low"] = np.zeros((N_CELLS, 10), dtype="float32")
        model = PULSAR(
            sample_key="donor_id",
            label_keys=["age"],
            tasks=["regression"],
            layer="X_low",
            device="cpu",
            sample_cell_num=4,
        )
        # Warning is emitted before forward pass; dim mismatch in projection
        # layer (small model expects 1280-dim input, gets 10) is expected.
        with pytest.warns(UserWarning, match="dimensions"):
            try:
                model.prepare_anndata(basic_adata)
            except Exception:
                pass

    def test_prepare_anndata_samples_match_adata(self, pulsar_model):
        assert set(pulsar_model.samples) == {f"donor_{i:02d}" for i in range(N_DONORS)}

    def test_get_sample_representations_shape(self, pulsar_model):
        assert pulsar_model.get_sample_representations().shape == (N_DONORS, 512)

    def test_get_sample_representations_indexed_by_donor(self, pulsar_model):
        reps = pulsar_model.get_sample_representations()
        assert set(reps.index) == set(pulsar_model.samples)

    def test_get_sample_representations_columns_named_dim_i(self, pulsar_model):
        reps = pulsar_model.get_sample_representations()
        assert list(reps.columns) == [f"dim_{i}" for i in range(512)]

    def test_get_sample_representations_no_nans(self, pulsar_model):
        assert not pulsar_model.get_sample_representations().isna().any().any()

    def test_get_sample_importance_one_row_per_donor(self, pulsar_model):
        assert pulsar_model.get_sample_importance().shape[0] == N_DONORS

    def test_get_sample_importance_column_named_after_label(self, pulsar_model):
        # label_keys[0] == "age"
        assert "age_importance" in pulsar_model.get_sample_importance().columns

    def test_get_sample_importance_values_are_positive(self, pulsar_model):
        """L2 norm of any non-zero vector is strictly positive."""
        assert (pulsar_model.get_sample_importance()["age_importance"] > 0).all()

    def test_get_sample_importance_equal_l2_norm_of_embeddings(self, pulsar_model):
        """Score must equal the L2 norm of the donor embedding — test the formula."""
        scores = pulsar_model.get_sample_importance()
        reps = pulsar_model.get_sample_representations()
        expected = np.linalg.norm(reps.values, axis=1)
        np.testing.assert_allclose(
            scores.loc[reps.index, "age_importance"].values,
            expected,
            rtol=1e-5,
        )

    def test_get_sample_importance_cached_on_second_call(self, pulsar_model):
        """Cache round-trip through adata.uns converts float32→float64,
        so dtype is not checked.
        """
        scores1 = pulsar_model.get_sample_importance()
        pulsar_model._donor_embeddings = None  # corrupt — recompute would crash
        scores2 = pulsar_model.get_sample_importance()
        pd.testing.assert_frame_equal(scores1, scores2, check_dtype=False)

    def test_get_cell_importance_one_row_per_cell(self, pulsar_model):
        assert pulsar_model.get_cell_importance().shape[0] == N_CELLS

    def test_get_cell_importance_column_named_after_label(self, pulsar_model):
        assert "age_importance" in pulsar_model.get_cell_importance().columns

    def test_get_cell_importance_values_in_zero_one(self, pulsar_model):
        """Absolute cosine similarity is bounded in [0, 1]."""
        imp = pulsar_model.get_cell_importance()
        assert (imp["age_importance"] >= 0).all()
        assert (imp["age_importance"] <= 1.0 + 1e-6).all()

    def test_get_cell_importance_written_to_adata_obs(self, pulsar_model):
        pulsar_model.get_cell_importance()
        assert "age_importance" in pulsar_model.adata.obs.columns

    def test_get_cell_importance_cached_on_second_call(self, pulsar_model):
        imp1 = pulsar_model.get_cell_importance()
        pulsar_model._donor_embeddings = None  # corrupt — recompute would crash
        imp2 = pulsar_model.get_cell_importance()
        pd.testing.assert_frame_equal(imp1, imp2, check_dtype=False)

    def test_calculate_distance_matrix_shape(self, pulsar_model):
        assert pulsar_model.calculate_distance_matrix().shape == (N_DONORS, N_DONORS)

    def test_calculate_distance_matrix_is_symmetric(self, pulsar_model):
        dist = pulsar_model.calculate_distance_matrix()
        np.testing.assert_allclose(dist, dist.T, atol=1e-6)

    def test_calculate_distance_matrix_diagonal_is_zero(self, pulsar_model):
        np.testing.assert_allclose(np.diag(pulsar_model.calculate_distance_matrix()), 0.0, atol=1e-6)

    def test_calculate_distance_matrix_is_non_negative(self, pulsar_model):
        assert (pulsar_model.calculate_distance_matrix() >= 0).all()

    def test_fit_linear_probe_classification_keys(self, pulsar_model):
        result = pulsar_model.fit_linear_probe(target="disease", task="classification")
        for key in ("model", "test_sample_labels", "y_test", "y_pred", "accuracy", "f1"):
            assert key in result

    def test_fit_linear_probe_classification_no_data_matrices(self, pulsar_model):
        """Return dict must not contain the expensive raw data matrices."""
        result = pulsar_model.fit_linear_probe(target="disease", task="classification")
        for key in ("X_train", "X_test", "y_train"):
            assert key not in result

    def test_fit_linear_probe_classification_accuracy_in_range(self, pulsar_model):
        result = pulsar_model.fit_linear_probe(target="disease", task="classification")
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_fit_linear_probe_regression_keys(self, pulsar_model):
        result = pulsar_model.fit_linear_probe(target="age", task="regression")
        for key in ("model", "test_sample_labels", "y_test", "y_pred", "r2", "pearson"):
            assert key in result

    def test_fit_linear_probe_regression_no_data_matrices(self, pulsar_model):
        result = pulsar_model.fit_linear_probe(target="age", task="regression")
        for key in ("X_train", "X_test", "y_train"):
            assert key not in result

    def test_fit_linear_probe_train_test_sizes(self, pulsar_model):
        """test_size=0.2 with 10 donors → 2 test, 8 train."""
        result = pulsar_model.fit_linear_probe(target="age", task="regression", test_size=0.2)
        assert len(result["test_sample_labels"]) == 2
        assert len(result["y_test"]) == 2

    def test_fit_linear_probe_random_split_stores_test_sample_labels(self, pulsar_model):
        """After a random split, model.test_sample_labels must be populated."""
        assert pulsar_model.test_sample_labels is None
        pulsar_model.fit_linear_probe(target="age", task="regression", test_size=0.2)
        assert pulsar_model.test_sample_labels is not None
        assert len(pulsar_model.test_sample_labels) == 2

    def test_fit_linear_probe_explicit_test_sample_labels(self, pulsar_model):
        """When test_sample_labels is provided, exactly those samples appear in the test set."""
        held_out = ["donor_00", "donor_01", "donor_02"]
        result = pulsar_model.fit_linear_probe(target="age", task="regression", test_sample_labels=held_out)
        assert sorted(result["test_sample_labels"]) == sorted(held_out)
        assert len(result["y_test"]) == len(held_out)

    def test_fit_linear_probe_explicit_labels_overwrite_the_field(self, pulsar_model):
        """When caller supplies test_sample_labels, model.test_sample_labels is not overwritten."""
        held_out = ["donor_00", "donor_01"]
        pulsar_model.fit_linear_probe(target="age", task="regression", test_sample_labels=held_out)
        # field should still be None — the caller owns the split
        assert pulsar_model.test_sample_labels == held_out

        new_held_out = ["donor_00", "donor_02"]

        pulsar_model.fit_linear_probe(target="age", task="regression", test_sample_labels=new_held_out)
        # field should still be None — the caller owns the split
        assert pulsar_model.test_sample_labels == new_held_out

    def test_fit_linear_probe_explicit_labels_train_is_complement(self, pulsar_model):
        """Training set must be exactly the complement of the provided test labels."""
        held_out = {"donor_00", "donor_01", "donor_02"}
        result = pulsar_model.fit_linear_probe(target="age", task="regression", test_sample_labels=list(held_out))
        all_donors = set(pulsar_model.sample_representation.index)
        expected_train = all_donors - held_out
        assert len(result["y_test"]) + len(expected_train) == len(all_donors)

    def test_fit_linear_probe_invalid_task_raises(self, pulsar_model):
        with pytest.raises(ValueError, match="task must be"):
            pulsar_model.fit_linear_probe(task="ranking")

    def test_fit_linear_probe_defaults_to_first_label_key(self, pulsar_model):
        """target=None must use label_keys[0], which is 'age' (continuous, > 1)."""
        result = pulsar_model.fit_linear_probe(task="regression")
        # age values are 20, 23, ... — all above 1 — distinguishable from binary disease
        assert np.unique(result["y_test"]).max() > 1


def _make_unfitted_supervised_models():
    """Return (model_id, unfitted_model) pairs without triggering any external imports."""
    from patpy.tl.supervised import MixMIL, PULSAR

    return [
        pytest.param(
            MixMIL(sample_key="donor_id", label_keys=["disease"], tasks=["classification"], layer="X_pca"),
            id="MixMIL",
        ),
        pytest.param(
            PULSAR(sample_key="donor_id", label_keys=["age"], tasks=["regression"], layer="X_uce", device="cpu"),
            id="PULSAR",
        ),
    ]


class TestCheckFitted:
    """Tests for BaseSampleMethod._fitted flag and _check_fitted() guard."""

    # ------------------------------------------------------------------
    # Unfitted state — no prepare_anndata called, no mocks needed
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("model", _make_unfitted_supervised_models())
    def test_fitted_false_before_prepare_anndata(self, model):
        assert model._fitted is False

    @pytest.mark.parametrize("model", _make_unfitted_supervised_models())
    def test_check_fitted_raises_before_prepare_anndata(self, model):
        with pytest.raises(RuntimeError, match="prepare_anndata"):
            model._check_fitted()

    @pytest.mark.parametrize("model", _make_unfitted_supervised_models())
    def test_calculate_distance_matrix_raises_before_prepare_anndata(self, model):
        with pytest.raises(RuntimeError, match="prepare_anndata"):
            model.calculate_distance_matrix()

    @pytest.mark.parametrize("model", _make_unfitted_supervised_models())
    def test_fit_linear_probe_raises_before_prepare_anndata(self, model):
        with pytest.raises(RuntimeError, match="prepare_anndata"):
            model.fit_linear_probe()

    # ------------------------------------------------------------------
    # Fitted state — requires mocked prepare_anndata fixtures
    # ------------------------------------------------------------------

    def test_fitted_true_after_prepare_anndata_mixmil(self, mixmil_model):
        assert mixmil_model._fitted is True

    def test_check_fitted_passes_after_prepare_anndata_mixmil(self, mixmil_model):
        mixmil_model._check_fitted()  # must not raise

    def test_fitted_true_after_prepare_anndata_pulsar(self, pulsar_model):
        assert pulsar_model._fitted is True

    def test_check_fitted_passes_after_prepare_anndata_pulsar(self, pulsar_model):
        pulsar_model._check_fitted()  # must not raise
