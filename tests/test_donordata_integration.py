import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from patpy.datasets import DatasetInfo, as_donordata
from patpy.tl import CellGroupComposition, GroupedPseudobulk, MixMIL, PaSCient, Pseudobulk
from patpy.tl.evaluation import knn_prediction_score, trajectory_correlation
from patpy.tl.sample_representation import (
    SampleRepresentationMethod,
    correlate_cell_type_expression,
    correlate_composition,
)
from patpy.tl.supervised import SupervisedSampleMethod

donordata = pytest.importorskip("donordata")

N_DONORS = 10
CELLS_PER_DONOR = 20
N_GENES = 15


@pytest.fixture
def donor_table():
    return pd.DataFrame(
        {
            "disease": np.arange(N_DONORS) % 2,
            "status": pd.Categorical(np.where(np.arange(N_DONORS) % 2, "case", "control")),
            "age": 20.0 + 3 * np.arange(N_DONORS),
            "site": np.repeat(["A", "B"], N_DONORS // 2),
        },
        index=pd.Index([f"donor_{i:02d}" for i in range(N_DONORS)], name="donor_id"),
    )


@pytest.fixture
def cells():
    rng = np.random.default_rng(0)
    donors = np.repeat([f"donor_{i:02d}" for i in range(N_DONORS)], CELLS_PER_DONOR)
    order = rng.permutation(donors.size)
    obs = pd.DataFrame(
        {
            "donor_id": donors[order],
            "cell_type": rng.choice(["T", "B", "NK"], donors.size),
        },
        index=[f"cell_{i}" for i in range(donors.size)],
    )
    obs["batch"] = obs["donor_id"].map({f"donor_{i:02d}": f"b{i % 3}" for i in range(N_DONORS)})
    adata = sc.AnnData(X=rng.poisson(3, size=(donors.size, N_GENES)).astype("float32"), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(donors.size, 6)).astype("float32")
    return adata


@pytest.fixture
def dd(cells, donor_table):
    dd = donordata.DonorData(D=donor_table, C=cells, donor_id="donor_id")
    dd.obsm["gPCs"] = pd.DataFrame(
        np.arange(N_DONORS * 2, dtype="float32").reshape(N_DONORS, 2), index=dd.obs_names, columns=["pc1", "pc2"]
    )
    return dd


def _make_supervised(**kwargs):
    class _Concrete(SupervisedSampleMethod):
        pass

    defaults = {"sample_key": "donor_id", "label_keys": ["disease"], "tasks": ["classification"]}
    defaults.update(kwargs)
    return _Concrete(**defaults)


def test_labels_come_from_donor_table(dd, donor_table):
    model = _make_supervised(label_keys=["disease", "age"], tasks=["classification", "regression"])
    model.prepare_anndata(dd)
    assert model.adata is dd.C
    assert model.donor_data is dd
    assert "age" not in dd.C.obs.columns
    expected = donor_table.loc[[str(s) for s in model.samples], ["disease", "age"]]
    np.testing.assert_array_equal(model.labels.to_numpy(), expected.to_numpy())
    assert list(model.labels.index) == list(model.samples)


def test_metadata_mixes_donor_table_and_cells(dd):
    model = _make_supervised()
    model.prepare_anndata(dd)
    metadata = model._extract_metadata(["batch", "site", "status"])
    assert list(metadata.columns) == ["batch", "site", "status"]
    assert metadata.loc["donor_03", "batch"] == "b0"
    assert metadata.loc["donor_07", "site"] == "B"
    assert metadata.loc["donor_01", "status"] == "case"


def test_sample_key_follows_donor_id(dd):
    model = _make_supervised(sample_key="patient")
    with pytest.warns(UserWarning, match="donor_id"):
        model.prepare_anndata(dd)
    assert model.sample_key == "donor_id"


def test_missing_label_raises(dd):
    model = _make_supervised(label_keys=["unknown"])
    with pytest.raises(ValueError, match="unknown"):
        model.prepare_anndata(dd)


def test_linear_probe_on_donor_level_target(dd):
    method = CellGroupComposition(sample_key="donor_id", cell_group_key="cell_type")
    method.prepare_anndata(dd)
    result = method.fit_linear_probe(target="age", task="regression", test_sample_labels=[])
    assert result["evaluated_on"] == "train"
    assert len(result["age_pred"]) == N_DONORS


def test_to_donordata_stores_aligned_results(dd):
    method = CellGroupComposition(sample_key="donor_id", cell_group_key="cell_type")
    method.prepare_anndata(dd)
    distances = method.calculate_distance_matrix()
    method.embed(method="MDS")
    result = method.to_donordata("composition")
    assert result is dd
    assert dd.obsm["composition"].shape[0] == N_DONORS
    assert dd.obsm["X_mds_composition"].shape == (N_DONORS, 2)
    assert dd.uns["sample_representations"] == ["composition"]
    positions = dd.obs_names.get_indexer([str(s) for s in method.samples])
    stored = dd.obsp["composition_distances"]
    np.testing.assert_allclose(stored[np.ix_(positions, positions)], distances)

    subset = dd.filter_donors("site == 'B'")
    keep = dd.obs_names.get_indexer(subset.obs_names)
    np.testing.assert_allclose(subset.obsp["composition_distances"], stored[np.ix_(keep, keep)])
    pd.testing.assert_frame_equal(subset.obsm["composition"], dd.obsm["composition"].loc[subset.obs_names])


def test_to_donordata_from_plain_anndata(cells, donor_table):
    flat = cells.copy()
    flat.obs = flat.obs.join(donor_table, on="donor_id")
    method = CellGroupComposition(sample_key="donor_id", cell_group_key="cell_type")
    method.prepare_anndata(flat)
    result = method.to_donordata()
    assert result.G_type == "table"
    assert {"disease", "age", "site", "batch"} <= set(result.obs.columns)
    assert "CellGroupComposition_distances" in result.obsp
    assert method.donor_data is result


def test_mixmil_covariates_from_donor_table_and_obsm(dd):
    pytest.importorskip("torch")
    model = MixMIL(
        sample_key="donor_id",
        label_keys=["status"],
        tasks=["classification"],
        additional_covariates=["age", "gPCs"],
    )
    model.prepare_anndata(dd, train=False)
    model._build_label_mappings()
    Xs, F, Y = model._build_tensors()
    assert len(Xs) == N_DONORS
    assert F.shape == (N_DONORS, 4)
    rows = dd.obs_names.get_indexer([str(s) for s in model.samples])
    np.testing.assert_allclose(F[:, 1].numpy(), dd.obs["age"].to_numpy()[rows])
    np.testing.assert_allclose(F[:, 2:].numpy(), dd.obsm["gPCs"].to_numpy()[rows])
    expected_y = (dd.obs["status"].to_numpy()[rows] == "control").astype("float32")
    np.testing.assert_allclose(Y[:, 0].numpy(), expected_y)
    for donor, bag in zip(model.samples, Xs, strict=True):
        assert bag.shape[0] == (dd.C.obs["donor_id"] == donor).sum()


@dataclass
class _Batch:
    x: object
    padded_mask: object
    sample_metadata: dict
    cell_metadata: dict
    view_names: list


@pytest.mark.parametrize("n_cells", [8, 32])
def test_pascient_training_bags_come_from_mil_dataset(dd, n_cells):
    torch = pytest.importorskip("torch")
    model = PaSCient(
        sample_key="donor_id",
        label_keys=["status"],
        tasks=["classification"],
        n_cells=n_cells,
        batch_size=4,
        val_fraction=0.2,
        device="cpu",
    )
    SupervisedSampleMethod.prepare_anndata(model, dd)
    y_map = {d: int(v == "case") for d, v in zip(model.labels.index, model.labels["status"], strict=True)}
    train_dl, val_dl = model._training_loaders(model.adata, "status", y_map, "classification", _Batch)
    assert len(train_dl.dataset) + len(val_dl.dataset) == N_DONORS
    assert len(val_dl.dataset) == 2

    batch = next(iter(train_dl))
    assert batch.x.shape == (4, 1, n_cells, N_GENES)
    assert batch.padded_mask.shape == (4, 1, n_cells)
    assert batch.sample_metadata["status"].dtype == torch.long
    assert torch.isfinite(batch.x).all()
    assert torch.all(batch.x[~batch.padded_mask] == 0)
    real_cells = batch.padded_mask.sum(dim=-1).squeeze(1)
    assert torch.all(real_cells == min(n_cells, CELLS_PER_DONOR))


def test_pascient_regression_targets_are_float(dd):
    torch = pytest.importorskip("torch")
    model = PaSCient(
        sample_key="donor_id", label_keys=["age"], tasks=["regression"], n_cells=5, batch_size=10, device="cpu"
    )
    SupervisedSampleMethod.prepare_anndata(model, dd)
    y_map = dict(zip(model.labels.index, model.labels["age"].to_numpy(), strict=True))
    train_dl, _ = model._training_loaders(model.adata, "age", y_map, "regression", _Batch)
    batch = next(iter(train_dl))
    assert batch.sample_metadata["age"].dtype == torch.float32
    assert set(batch.sample_metadata["age"].numpy()) <= set(model.labels["age"].to_numpy().astype("float32"))


def test_pascient_train_feeds_mil_batches_to_trainer(dd, monkeypatch):
    torch = pytest.importorskip("torch")
    seen = []

    class _Trainer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, model, train_dataloaders, val_dataloaders):
            for loader in (train_dataloaders, val_dataloaders):
                seen.extend(loader)

    lightning = types.ModuleType("lightning")
    lightning.Trainer = _Trainer
    structures = types.ModuleType("pascient.data.data_structures")
    structures.SampleBatch = _Batch
    monkeypatch.setitem(sys.modules, "lightning", lightning)
    monkeypatch.setitem(sys.modules, "pascient", types.ModuleType("pascient"))
    monkeypatch.setitem(sys.modules, "pascient.data", types.ModuleType("pascient.data"))
    monkeypatch.setitem(sys.modules, "pascient.data.data_structures", structures)
    monkeypatch.setattr(
        PaSCient,
        "_build_model",
        lambda self, n_genes, n_classes, class_counts=None, task=None: torch.nn.Linear(n_genes, n_classes),
    )

    model = PaSCient(
        sample_key="donor_id", label_keys=["status"], tasks=["classification"], n_cells=12, batch_size=3, device="cpu"
    )
    SupervisedSampleMethod.prepare_anndata(model, dd)
    model._train(model.adata)
    assert sum(batch.x.shape[0] for batch in seen) == N_DONORS
    assert all(batch.x.shape[1:] == (1, 12, N_GENES) for batch in seen)
    assert model._class_names == ["case", "control"]
    labels = torch.cat([batch.sample_metadata["status"] for batch in seen])
    assert sorted(labels.tolist()) == sorted((dd.obs["status"] == "control").astype(int).tolist())


def test_to_donordata_into_given_object(dd, cells):
    method = Pseudobulk(sample_key="donor_id", cell_group_key="cell_type", layer="X_pca")
    method.prepare_anndata(cells[cells.obs["cell_type"] != "NK"].copy())
    method.to_donordata("pb_no_nk", donor_data=dd)
    assert "pb_no_nk_distances" in dd.obsp
    assert dd.uns["sample_representations"] == ["pb_no_nk"]
    assert method.donor_data is dd


def test_benchmark_on_pseudobulk_meta_adata(dd):
    dd.D = dd.pseudobulk(func="mean")
    meta_adata = dd.D
    for key, method in {
        "composition": CellGroupComposition(sample_key="donor_id", cell_group_key="cell_type"),
        "pseudobulk": Pseudobulk(sample_key="donor_id", cell_group_key="cell_type", layer="X_pca"),
    }.items():
        method.prepare_anndata(dd)
        method.calculate_distance_matrix()
        method.to_donordata(key)
    assert {"composition_distances", "pseudobulk_distances"} <= set(meta_adata.obsp)
    assert meta_adata.obs_names.tolist() == dd.obs_names.tolist()

    schema = {"relevant": {"status": "classification"}, "technical": {"site": "classification"}}
    scores = knn_prediction_score(meta_adata, schema, representations=dd.uns["sample_representations"], n_neighbors=3)
    assert set(scores["representation"]) == {"composition", "pseudobulk"}
    assert scores["score"].between(-1, 1).all()
    from_dd = knn_prediction_score(dd, schema, representations=dd.uns["sample_representations"], n_neighbors=3)
    pd.testing.assert_frame_equal(scores, from_dd)

    pytest.importorskip("ehrapy")
    trajectory = trajectory_correlation(
        meta_adata, root_sample="donor_00", trajectory_variable="age", representations=["composition", "pseudobulk"]
    )
    assert set(trajectory.index) == {"composition", "pseudobulk"}
    assert "composition_neighbors" in meta_adata.uns
    assert "composition_dpt_pseudotime" in dd.obs.columns


def test_donor_covariates_reach_libraries_that_read_cells(dd):
    class _NeedsStatus(SampleRepresentationMethod):
        def calculate_distance_matrix(self, force=False):
            self._ensure_cell_columns(["status"])
            return pd.crosstab(self.adata.obs[self.sample_key], self.adata.obs["status"]).to_numpy()

    method = _NeedsStatus(sample_key="donor_id", cell_group_key="cell_type")
    method.prepare_anndata(dd)
    assert "status" not in dd.C.obs.columns
    method.calculate_distance_matrix()
    assert method.adata is dd.C
    assert (dd.C.obs["status"].to_numpy() == dd.obs["status"].to_numpy()[dd.donor_codes]).all()
    with pytest.raises(ValueError, match="not found"):
        method._ensure_cell_columns(["unknown"])


def test_cell_group_resolved_distances_are_stored(dd):
    method = GroupedPseudobulk(sample_key="donor_id", cell_group_key="cell_type", layer="X_pca")
    method.prepare_anndata(dd)
    method.calculate_distance_matrix()
    assert set(method.group_distances) == {"T", "B", "NK"}
    method.to_donordata("grouped")
    for group in ("T", "B", "NK"):
        assert dd.obsp[f"grouped_distances_{group}"].shape == (N_DONORS, N_DONORS)
    assert dd.obsm["grouped"].shape == (N_DONORS, 3 * 6)
    subset = dd.filter_donors("site == 'A'")
    assert subset.obsp["grouped_distances_B"].shape == (subset.n_donors, subset.n_donors)


def test_to_adata_takes_metadata_from_donor_table(dd):
    method = CellGroupComposition(sample_key="donor_id", cell_group_key="cell_type")
    method.prepare_anndata(dd)
    samples = method.to_adata()
    assert {"disease", "age", "site"} <= set(samples.obs.columns)
    assert samples.obs["age"].tolist() == dd.obs.loc[[str(s) for s in method.samples], "age"].tolist()


def test_correlation_helpers_accept_donordata(dd):
    composition = correlate_composition(dd, cell_type_key="cell_type", target="age")
    assert set(composition.index) == {"T", "B", "NK"}
    expression = correlate_cell_type_expression(dd, cell_type_key="cell_type", target="age", min_sample_size=5)
    assert set(expression["cell_type"]) <= {"T", "B", "NK"}
    assert "B_pseudobulk" in dd.obsm


def test_dataset_as_donordata(cells, donor_table):
    flat = cells.copy()
    flat.obs = flat.obs.join(donor_table, on="donor_id")
    info = DatasetInfo(
        n_samples=N_DONORS,
        n_cells=flat.n_obs,
        n_features=flat.n_vars,
        sample_key="donor_id",
        cell_type_key="cell_type",
        sample_metadata_columns=["disease", "age", "site", "missing_column"],
    )
    dd = as_donordata(flat, info)
    assert set(dd.obs.columns) == {"disease", "age", "site"}
    assert "age" in dd.C.obs.columns
    meta = dd.pseudobulk(func="mean")
    assert as_donordata(flat, info, meta_adata=meta).D_type == "anndata"


def test_pilot_reads_status_from_donor_table(dd):
    pytest.importorskip("pilotpy")
    from patpy.tl import PILOT

    method = PILOT(sample_key="donor_id", cell_group_key="cell_type", layer="X_pca", sample_state_col="status")
    method.prepare_anndata(dd)
    distances = method.calculate_distance_matrix()
    assert distances.shape == (N_DONORS, N_DONORS)
    assert "status" in dd.C.obs.columns


@pytest.fixture
def dd3(donor_table):
    rng = np.random.default_rng(1)
    samples = pd.DataFrame(
        {
            "donor_id": np.repeat(donor_table.index.to_numpy(), 2),
            "visit": np.tile([0, 1], N_DONORS),
            "severity": rng.choice(["mild", "severe"], 2 * N_DONORS),
        },
        index=pd.Index([f"{d}_v{v}" for d in donor_table.index for v in (0, 1)], name="sample_id"),
    )
    cells_per_sample = CELLS_PER_DONOR // 2
    sample_of_cell = np.repeat(samples.index.to_numpy(), cells_per_sample)
    obs = pd.DataFrame(
        {"sample_id": sample_of_cell, "cell_type": rng.choice(["T", "B", "NK"], sample_of_cell.size)},
        index=[f"cell_{i}" for i in range(sample_of_cell.size)],
    )
    adata = sc.AnnData(X=rng.poisson(3, size=(sample_of_cell.size, N_GENES)).astype("float32"), obs=obs)
    adata.obsm["X_pca"] = rng.normal(size=(adata.n_obs, 6)).astype("float32")
    return donordata.DonorData(D=donor_table, S=samples, C=adata, donor_id="donor_id", sample_id="sample_id")


def test_samples_are_the_units_with_a_sample_level(dd3):
    method = CellGroupComposition(sample_key="sample_id", cell_group_key="cell_type")
    method.prepare_anndata(dd3)
    assert method.unit_level == "sample"
    assert len(method.samples) == dd3.n_samples
    metadata = method._extract_metadata(["severity", "age", "visit"])
    names = [str(s) for s in method.samples]
    np.testing.assert_array_equal(metadata["severity"], dd3.S.loc[names, "severity"])
    np.testing.assert_array_equal(metadata["age"], dd3.get_df("age", level="sample").loc[names, "age"])
    method.to_donordata("composition")
    samples = dd3.levels["sample"]
    assert samples.obsm["composition"].shape == (dd3.n_samples, 3)
    assert samples.obsp["composition_distances"].shape == (dd3.n_samples, dd3.n_samples)
    assert "composition" not in dd3.obsm
    kept = dd3.filter_samples("visit == 0")
    assert kept.levels["sample"].obsp["composition_distances"].shape == (N_DONORS, N_DONORS)


def test_donor_key_pools_the_samples_of_a_donor(dd3):
    assert "donor_id" not in dd3.C.obs.columns
    method = Pseudobulk(sample_key="donor_id", cell_group_key="cell_type", layer="X_pca")
    method.prepare_anndata(dd3)
    assert method.unit_level == "donor"
    assert len(method.samples) == N_DONORS
    assert (dd3.C.obs["donor_id"].astype(str).to_numpy() == dd3.obs_names[dd3.donor_codes].to_numpy()).all()
    method.to_donordata("pseudobulk")
    assert dd3.obsp["pseudobulk_distances"].shape == (N_DONORS, N_DONORS)


def test_other_sample_key_defaults_to_the_samples(dd3):
    method = CellGroupComposition(sample_key="patient", cell_group_key="cell_type")
    with pytest.warns(UserWarning, match="sample identifier 'sample_id'"):
        method.prepare_anndata(dd3)
    assert method.sample_key == "sample_id"


def test_sample_and_donor_covariates_reach_the_cells(dd3):
    method = CellGroupComposition(sample_key="sample_id", cell_group_key="cell_type")
    method.prepare_anndata(dd3)
    method._ensure_cell_columns(["severity", "site"])
    np.testing.assert_array_equal(dd3.C.obs["severity"], dd3.S["severity"].to_numpy()[dd3.sample_codes])
    np.testing.assert_array_equal(dd3.C.obs["site"], dd3.obs["site"].to_numpy()[dd3.donor_codes])
    assert method._has_metadata("visit")
    assert not method._has_metadata("unknown")


def test_scores_are_computed_per_sample(dd3):
    method = CellGroupComposition(sample_key="sample_id", cell_group_key="cell_type")
    method.prepare_anndata(dd3)
    method.to_donordata("composition")
    schema = {"relevant": {"severity": "classification"}, "technical": {"site": "classification"}}
    scores = knn_prediction_score(dd3, schema, n_neighbors=3)
    assert set(scores["covariate"]) == {"severity", "site"}
    assert (scores["n_observations"] == dd3.n_samples).all()


def test_correlation_helpers_use_the_samples(dd3):
    composition = correlate_composition(dd3, cell_type_key="cell_type", target="age")
    assert set(composition.index) == {"T", "B", "NK"}
    correlate_cell_type_expression(dd3, cell_type_key="cell_type", target="visit", min_sample_size=5)
    assert dd3.levels["sample"].obsm["B_pseudobulk"].shape == (dd3.n_samples, N_GENES)
