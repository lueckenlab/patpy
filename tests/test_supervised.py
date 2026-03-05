from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

SAMPLE_KEY = "donor_id"
LABEL_KEY = "disease"
CELL_KEY = "cell_type"

N_DONORS = 6
N_CELLS_PER_DONOR = 20
N_CELLS = N_DONORS * N_CELLS_PER_DONOR
N_GENES = 50
N_PCA_DIMS = 10
N_UCE_DIMS = 1280
PULSAR_HIDDEN = 16


@pytest.fixture
def supervised_adata():
    """Minimal AnnData suitable for MixMIL.

    Contains:
    - integer count matrix in ``.X``
    - ``donor_id`` and ``disease`` columns in ``.obs``
    - ``cell_type`` column in ``.obs``
    - ``X_pca`` in ``.obsm`` (10-dim)
    """
    rng = np.random.default_rng(42)

    donor_ids = np.repeat([f"D{i}" for i in range(N_DONORS)], N_CELLS_PER_DONOR)

    labels = np.repeat([0] * (N_DONORS // 2) + [1] * (N_DONORS // 2), N_CELLS_PER_DONOR)
    cell_types = np.tile(["B", "T", "NK", "Mono"], N_CELLS // 4)

    adata = AnnData(
        X=rng.negative_binomial(5, 0.5, size=(N_CELLS, N_GENES)).astype(np.float32),
        obs=pd.DataFrame(
            {SAMPLE_KEY: donor_ids, LABEL_KEY: labels, CELL_KEY: cell_types},
            index=[f"cell_{i}" for i in range(N_CELLS)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)]),
    )
    adata.obsm["X_pca"] = rng.normal(size=(N_CELLS, N_PCA_DIMS)).astype(np.float32)
    return adata


@pytest.fixture
def pulsar_adata():
    """Minimal AnnData suitable for PULSAR.

    Extends ``supervised_adata`` with ``X_uce`` in ``.obsm``
    (1280-dim, matching the default UCE / PULSAR input size).
    """
    rng = np.random.default_rng(0)

    donor_ids = np.repeat([f"D{i}" for i in range(N_DONORS)], N_CELLS_PER_DONOR)
    labels = np.repeat(["control"] * (N_DONORS // 2) + ["disease"] * (N_DONORS // 2), N_CELLS_PER_DONOR)
    cell_types = np.tile(["B", "T", "NK", "Mono"], N_CELLS // 4)

    adata = AnnData(
        X=rng.negative_binomial(5, 0.5, size=(N_CELLS, N_GENES)).astype(np.float32),
        obs=pd.DataFrame(
            {SAMPLE_KEY: donor_ids, LABEL_KEY: labels, CELL_KEY: cell_types},
            index=[f"cell_{i}" for i in range(N_CELLS)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(N_GENES)]),
    )
    adata.obsm["X_uce"] = rng.normal(size=(N_CELLS, N_UCE_DIMS)).astype(np.float32)
    return adata


def _assert_sample_scores(scores, n_donors):
    """Assert that get_sample_scores() returns a well-formed DataFrame."""
    assert isinstance(scores, pd.DataFrame)
    assert len(scores) == n_donors
    assert "score" in scores.columns
    assert scores["score"].notna().all()


def _assert_cell_scores(cell_scores, n_cells, obs_names):
    """Assert that get_cell_scores() returns a well-formed DataFrame."""
    assert isinstance(cell_scores, pd.DataFrame)
    assert len(cell_scores) == n_cells
    assert "score" in cell_scores.columns
    assert cell_scores["score"].notna().all()
    assert list(cell_scores.index) == list(obs_names)


def _assert_sample_embeddings(embeddings, n_donors, min_dims=1):
    """Assert that get_sample_embeddings() returns a well-formed DataFrame."""
    assert isinstance(embeddings, pd.DataFrame)
    assert len(embeddings) == n_donors
    assert embeddings.shape[1] >= min_dims
    assert embeddings.notna().all().all()

    assert all(c.startswith("dim_") for c in embeddings.columns)


def test_base_raises_before_fit(supervised_adata):
    """All output methods raise RuntimeError when called before prepare_anndata."""
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Dummy(SupervisedSampleMethod):
        def get_sample_scores(self):
            self._check_fitted()

        def get_cell_scores(self):
            self._check_fitted()

    dummy = _Dummy(sample_key=SAMPLE_KEY, label_key=LABEL_KEY)
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        dummy.get_sample_scores()
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        dummy.get_cell_scores()


def test_base_raises_on_missing_sample_key(supervised_adata):
    """prepare_anndata raises ValueError when sample_key is absent from obs."""
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Dummy(SupervisedSampleMethod):
        def get_sample_scores(self): ...
        def get_cell_scores(self): ...

    dummy = _Dummy(sample_key="nonexistent", label_key=LABEL_KEY)
    with pytest.raises(ValueError, match="sample_key"):
        dummy.prepare_anndata(supervised_adata.copy())


def test_base_raises_on_missing_label_key(supervised_adata):
    """prepare_anndata raises ValueError when label_key is absent from obs."""
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Dummy(SupervisedSampleMethod):
        def get_sample_scores(self): ...
        def get_cell_scores(self): ...

    dummy = _Dummy(sample_key=SAMPLE_KEY, label_key="nonexistent")
    with pytest.raises(ValueError, match="label_key"):
        dummy.prepare_anndata(supervised_adata.copy())


def test_base_warns_on_ambiguous_labels(supervised_adata):
    """prepare_anndata warns when a donor has multiple different label values."""
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Dummy(SupervisedSampleMethod):
        def get_sample_scores(self): ...
        def get_cell_scores(self): ...

    adata = supervised_adata.copy()

    adata.obs.loc[adata.obs[SAMPLE_KEY] == "D0", LABEL_KEY] = [99, 0] + [0] * (N_CELLS_PER_DONOR - 2)

    dummy = _Dummy(sample_key=SAMPLE_KEY, label_key=LABEL_KEY)
    with pytest.warns(UserWarning, match="multiple values"):
        dummy.prepare_anndata(adata)


def test_base_get_sample_embeddings_raises_by_default(supervised_adata):
    """Base get_sample_embeddings() raises NotImplementedError when not overridden."""
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Dummy(SupervisedSampleMethod):
        def prepare_anndata(self, adata):
            super().prepare_anndata(adata)

        def get_sample_scores(self): ...
        def get_cell_scores(self): ...

    dummy = _Dummy(sample_key=SAMPLE_KEY, label_key=LABEL_KEY)
    dummy.prepare_anndata(supervised_adata.copy())
    with pytest.raises(NotImplementedError):
        dummy.get_sample_embeddings()


def test_base_get_layer_data_raises_for_unknown_layer(supervised_adata):
    """_get_layer_data raises ValueError when the requested layer does not exist."""
    from patpy.tl.supervised import SupervisedSampleMethod

    class _Dummy(SupervisedSampleMethod):
        def get_sample_scores(self): ...
        def get_cell_scores(self): ...

    dummy = _Dummy(sample_key=SAMPLE_KEY, label_key=LABEL_KEY, layer="nonexistent_layer")
    dummy.prepare_anndata(supervised_adata.copy())
    with pytest.raises(ValueError, match="not found"):
        dummy._get_layer_data()


def test_mixmil_sample_scores_shape_and_columns(supervised_adata):
    """get_sample_scores returns one row per donor with a 'score' column."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    n_donors = adata.obs[SAMPLE_KEY].nunique()

    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
    )
    model.prepare_anndata(adata)
    scores = model.get_sample_scores()

    _assert_sample_scores(scores, n_donors)


def test_mixmil_cell_scores_shape_and_index(supervised_adata):
    """get_cell_scores returns one row per cell, indexed by obs_names."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()

    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
    )
    model.prepare_anndata(adata)
    cell_scores = model.get_cell_scores()

    _assert_cell_scores(cell_scores, adata.n_obs, adata.obs_names)


def test_mixmil_cell_scores_sum_to_one_per_donor(supervised_adata):
    """Attention weights within each donor bag are softmax-normalised (sum ≈ 1)."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
    )
    model.prepare_anndata(adata)
    cell_scores = model.get_cell_scores()

    for donor in adata.obs[SAMPLE_KEY].unique():
        mask = adata.obs[SAMPLE_KEY] == donor
        donor_sum = cell_scores.loc[mask, "score"].sum()
        assert np.isclose(donor_sum, 1.0, atol=1e-4), (
            f"Attention weights for donor {donor} sum to {donor_sum}, expected 1."
        )


def test_mixmil_sample_embeddings_shape(supervised_adata):
    """get_sample_embeddings returns (n_donors, Q) DataFrame with dim_ columns."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    n_donors = adata.obs[SAMPLE_KEY].nunique()

    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
    )
    model.prepare_anndata(adata)
    embeddings = model.get_sample_embeddings()

    _assert_sample_embeddings(embeddings, n_donors, min_dims=N_PCA_DIMS)


def test_mixmil_donor_index_matches_across_outputs(supervised_adata):
    """sample_scores, sample_embeddings, and cell_scores share consistent donor IDs."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
    )
    model.prepare_anndata(adata)

    scores = model.get_sample_scores()
    embeddings = model.get_sample_embeddings()

    assert set(scores.index) == set(embeddings.index)
    assert set(scores.index) == set(adata.obs[SAMPLE_KEY].unique())


def test_mixmil_raises_before_fit(supervised_adata):
    """All output methods raise RuntimeError when called before prepare_anndata."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    model = MixMIL(sample_key=SAMPLE_KEY, label_key=LABEL_KEY)
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        model.get_sample_scores()
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        model.get_cell_scores()
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        model.get_sample_embeddings()


def test_mixmil_fallback_pca_when_layer_missing(supervised_adata):
    """MixMIL computes PCA and warns when the requested obsm layer is absent."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()

    del adata.obsm["X_pca"]

    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
        n_pcs=5,
    )
    with pytest.warns(UserWarning, match="Computing PCA"):
        model.prepare_anndata(adata)

    scores = model.get_sample_scores()
    assert len(scores) == N_DONORS


def test_mixmil_warns_and_skips_missing_sex_column(supervised_adata):
    """encode_sex=True warns gracefully when 'sex' is absent from obs."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=True,
        encode_age=False,
    )
    with pytest.warns(UserWarning, match="sex"):
        model.prepare_anndata(adata)

    scores = model.get_sample_scores()
    assert len(scores) == N_DONORS


def test_mixmil_warns_and_skips_missing_age_column(supervised_adata):
    """encode_age=True warns gracefully when 'age' is absent from obs."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=True,
    )
    with pytest.warns(UserWarning, match="age"):
        model.prepare_anndata(adata)

    scores = model.get_sample_scores()
    assert len(scores) == N_DONORS


def test_mixmil_age_covariate_is_encoded(supervised_adata):
    """encode_age=True successfully uses a numeric age column from obs."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    rng = np.random.default_rng(1)
    adata = supervised_adata.copy()

    donor_ages = {f"D{i}": 30 + i * 5 for i in range(N_DONORS)}
    adata.obs["age"] = adata.obs[SAMPLE_KEY].map(donor_ages).astype(float)

    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=True,
    )
    model.prepare_anndata(adata)
    scores = model.get_sample_scores()
    assert len(scores) == N_DONORS


def test_mixmil_additional_covariate_raises_for_unknown_key(supervised_adata):
    """Providing an unknown additional_covariate raises ValueError."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
        additional_covariates=["nonexistent_covariate"],
    )
    with pytest.raises(ValueError, match="nonexistent_covariate"):
        model.prepare_anndata(adata)


def test_mixmil_training_history_is_populated(supervised_adata):
    """training_history is a non-empty list after prepare_anndata."""
    pytest.importorskip("mixmil")
    from patpy.tl.supervised import MixMIL

    adata = supervised_adata.copy()
    model = MixMIL(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_pca",
        n_epochs=2,
        encode_sex=False,
        encode_age=False,
    )
    model.prepare_anndata(adata)

    assert model.training_history is not None
    assert len(model.training_history) > 0


@pytest.fixture
def mock_pulsar_model(monkeypatch, pulsar_adata):
    """Monkeypatch pulsar imports so tests run without downloading model weights.

    Injects a tiny PyTorch model with the same interface as the real PULSAR
    (from_pretrained / forward returning list with CLS at [:, 0, :]).
    """
    torch = pytest.importorskip("torch")

    hidden = PULSAR_HIDDEN
    input_size = N_UCE_DIMS

    class _TinyProjector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(input_size, hidden)

        def forward(self, x):
            return self.proj(x)

    class _TinyModel(torch.nn.Module):
        hidden_size = hidden

        def __init__(self):
            super().__init__()
            self.proj = _TinyProjector()

        @classmethod
        def from_pretrained(cls, name_or_path):
            return cls()

        def eval(self):
            return self

        def to(self, *args, **kwargs):
            return self

        def forward(self, cell_embeddings, *args, **kwargs):
            proj = self.proj(cell_embeddings)

            batch, seq, h = proj.shape
            cls = torch.zeros(batch, 1, h)
            out = torch.cat([cls, proj], dim=1)
            return [out]

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    tiny_model = _TinyModel()

    import types

    pulsar_model_mod = types.ModuleType("pulsar.model")
    pulsar_model_mod.PULSAR = _TinyModel
    monkeypatch.setitem(__import__("sys").modules, "pulsar", types.ModuleType("pulsar"))
    monkeypatch.setitem(__import__("sys").modules, "pulsar.model", pulsar_model_mod)

    def _fake_extract(
        adata,
        model,
        label_name=None,
        donor_id_key="donor_id",
        embedding_key="X_uce",
        device="cpu",
        sample_cell_num=8,
        resample_num=1,
        batch_size=4,
        seed=0,
        **kwargs,
    ):
        rng = np.random.default_rng(seed)
        result = {}
        for donor in adata.obs[donor_id_key].unique():
            emb = rng.normal(size=(hidden,)).astype(np.float32)
            result[donor] = {"embedding": [emb], "label": None}
        return result

    pulsar_utils_mod = types.ModuleType("pulsar.utils")
    pulsar_utils_mod.extract_donor_embeddings_from_h5ad = _fake_extract
    monkeypatch.setitem(__import__("sys").modules, "pulsar.utils", pulsar_utils_mod)

    return tiny_model


def test_pulsar_sample_embeddings_shape(pulsar_adata, mock_pulsar_model):
    """get_sample_embeddings returns (n_donors, hidden_dim) DataFrame."""
    from patpy.tl.supervised import PULSAR

    adata = pulsar_adata.copy()
    n_donors = adata.obs[SAMPLE_KEY].nunique()

    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)
    embeddings = model.get_sample_embeddings()

    _assert_sample_embeddings(embeddings, n_donors, min_dims=1)


def test_pulsar_sample_scores_shape_and_columns(pulsar_adata, mock_pulsar_model):
    """get_sample_scores returns (n_donors,) DataFrame with 'score' column."""
    from patpy.tl.supervised import PULSAR

    adata = pulsar_adata.copy()
    n_donors = adata.obs[SAMPLE_KEY].nunique()

    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)
    scores = model.get_sample_scores()

    _assert_sample_scores(scores, n_donors)

    assert (scores["score"] >= 0).all()


def test_pulsar_cell_scores_shape_and_index(pulsar_adata, mock_pulsar_model):
    """get_cell_scores returns (n_cells,) DataFrame indexed by obs_names."""
    from patpy.tl.supervised import PULSAR

    adata = pulsar_adata.copy()

    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)
    cell_scores = model.get_cell_scores()

    _assert_cell_scores(cell_scores, adata.n_obs, adata.obs_names)


def test_pulsar_cell_scores_are_bounded(pulsar_adata, mock_pulsar_model):
    """Cell scores are cosine-similarity-like values in [0, 1]."""
    from patpy.tl.supervised import PULSAR

    adata = pulsar_adata.copy()
    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)
    cell_scores = model.get_cell_scores()

    assert (cell_scores["score"] >= 0).all()
    assert (cell_scores["score"] <= 1 + 1e-6).all()


def test_pulsar_raises_before_fit(pulsar_adata):
    """All output methods raise RuntimeError when called before prepare_anndata."""
    pytest.importorskip("torch")  # PULSAR needs torch even before fitting
    from patpy.tl.supervised import PULSAR

    model = PULSAR(sample_key=SAMPLE_KEY, label_key=LABEL_KEY)
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        model.get_sample_scores()
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        model.get_cell_scores()
    with pytest.raises(RuntimeError, match="prepare_anndata"):
        model.get_sample_embeddings()


def test_pulsar_raises_when_layer_missing(pulsar_adata, mock_pulsar_model):
    """prepare_anndata raises ValueError when the UCE layer is absent from obsm."""
    from patpy.tl.supervised import PULSAR

    adata = pulsar_adata.copy()
    del adata.obsm["X_uce"]

    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
    )
    with pytest.raises(ValueError, match="X_uce"):
        model.prepare_anndata(adata)


def test_pulsar_donor_index_matches_across_outputs(pulsar_adata, mock_pulsar_model):
    """sample_scores, sample_embeddings, and cell_scores share consistent donor IDs."""
    from patpy.tl.supervised import PULSAR

    adata = pulsar_adata.copy()
    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)

    scores = model.get_sample_scores()
    embeddings = model.get_sample_embeddings()

    assert set(scores.index) == set(embeddings.index)
    assert set(scores.index) == set(adata.obs[SAMPLE_KEY].unique())


def test_pulsar_fit_linear_probe_classification(pulsar_adata, mock_pulsar_model):
    """fit_linear_probe returns accuracy and f1 for a classification task."""
    from patpy.tl.supervised import PULSAR

    pytest.importorskip("sklearn")

    adata = pulsar_adata.copy()
    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)
    result = model.fit_linear_probe(task="classification", test_size=0.4)

    assert "accuracy" in result
    assert "f1" in result
    assert "model" in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_pulsar_fit_linear_probe_regression(pulsar_adata, mock_pulsar_model):
    """fit_linear_probe returns r2 and pearson for a regression task."""
    from patpy.tl.supervised import PULSAR

    pytest.importorskip("sklearn")

    adata = pulsar_adata.copy()

    rng = np.random.default_rng(7)
    donor_ages = {d: float(30 + i * 5) for i, d in enumerate(adata.obs[SAMPLE_KEY].unique())}
    adata.obs["age"] = adata.obs[SAMPLE_KEY].map(donor_ages)

    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key="age",
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)
    result = model.fit_linear_probe(task="regression", test_size=0.4)

    assert "r2" in result
    assert "pearson" in result
    assert "model" in result


def test_pulsar_fit_linear_probe_invalid_task_raises(pulsar_adata, mock_pulsar_model):
    """fit_linear_probe raises ValueError for an unsupported task type."""
    from patpy.tl.supervised import PULSAR

    pytest.importorskip("sklearn")

    adata = pulsar_adata.copy()
    model = PULSAR(
        sample_key=SAMPLE_KEY,
        label_key=LABEL_KEY,
        layer="X_uce",
        pretrained_model="mock/pulsar",
        device="cpu",
        sample_cell_num=8,
        batch_size=4,
    )
    model.prepare_anndata(adata)

    with pytest.raises(ValueError, match="task"):
        model.fit_linear_probe(task="ranking")
