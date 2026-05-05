import json

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from scipy.sparse import csr_matrix

from patpy.mcp import tools as mcp_tools


def _make_test_adata():
    obs = pd.DataFrame(
        {
            "donor_id": [
                "donor_a",
                "donor_a",
                "donor_a",
                "donor_b",
                "donor_b",
                "donor_c",
            ],
            "cell_type": [
                "T",
                "T",
                "B",
                "T",
                "B",
                "T",
            ],
            "disease": [
                "case",
                "case",
                "case",
                "control",
                "control",
                "case",
            ],
            "age_group": [
                "adult",
                "adult",
                "adult",
                "adult",
                "adult",
                "senior",
            ],
        },
        index=[f"cell_{idx}" for idx in range(6)],
    )
    var = pd.DataFrame(index=["gene_1", "gene_2", "gene_3"])
    adata = AnnData(
        X=np.array(
            [
                [8.0, 1.0, 0.0],
                [7.5, 1.5, 0.0],
                [2.0, 6.0, 1.0],
                [1.0, 8.0, 0.5],
                [0.0, 7.5, 1.5],
                [5.0, 2.0, 3.0],
            ]
        ),
        obs=obs,
        var=var,
    )
    adata.obsm["X_pca"] = np.array(
        [
            [2.0, 0.0],
            [1.8, 0.1],
            [0.2, 1.0],
            [0.0, 2.0],
            [-0.1, 1.9],
            [1.0, 0.8],
        ]
    )
    adata.obsp["distances"] = csr_matrix(np.ones((adata.n_obs, adata.n_obs)) - np.eye(adata.n_obs))
    return adata


def _write_test_adata(tmp_path):
    adata_path = tmp_path / "test_adata.h5ad"
    _make_test_adata().write_h5ad(adata_path)
    return adata_path


def test_dataset_summary_reports_sample_and_cell_group_counts(tmp_path):
    adata_path = _write_test_adata(tmp_path)

    result = mcp_tools.dataset_summary(
        adata_path=str(adata_path),
        sample_key="donor_id",
        cell_group_key="cell_type",
        output_dir=str(tmp_path / "summary"),
    )

    assert result["summary"]["n_samples"] == 3
    assert result["summary"]["n_cell_groups"] == 2
    assert "sample_counts" in result["artifacts"]
    assert "manifest" in result["artifacts"]


def test_preprocess_dataset_writes_filtered_outputs(tmp_path):
    adata_path = _write_test_adata(tmp_path)

    result = mcp_tools.preprocess_dataset(
        adata_path=str(adata_path),
        output_dir=str(tmp_path / "preprocessed"),
        sample_key="donor_id",
        cell_group_key="cell_type",
        sample_size_threshold=2,
        metadata_columns=["disease"],
        composition_keys=["cell_type"],
    )

    filtered_adata = read_h5ad(result["artifacts"]["filtered_adata"])
    assert filtered_adata.obs["donor_id"].nunique() == 2
    assert result["summary"]["removed_samples"] == ["donor_c"]
    assert "metadata" in result["artifacts"]
    assert "compositional_metrics" in result["artifacts"]


def test_build_and_evaluate_representation(tmp_path):
    adata_path = _write_test_adata(tmp_path)

    representation = mcp_tools.build_representation(
        adata_path=str(adata_path),
        output_dir=str(tmp_path / "representation"),
        sample_key="donor_id",
        cell_group_key="cell_type",
        method="pseudobulk",
        layer="X_pca",
        metadata_columns=["disease"],
    )
    distances = np.load(representation["artifacts"]["distances"])
    assert distances.shape == (3, 3)
    assert "sample_adata" in representation["artifacts"]

    evaluation = mcp_tools.evaluate_representation(
        output_dir=str(tmp_path / "evaluation"),
        representation_manifest_path=representation["artifacts"]["manifest"],
        adata_path=str(adata_path),
        sample_key="donor_id",
        target_column="disease",
        method="knn",
        n_neighbors=1,
    )

    assert evaluation["summary"]["evaluation_result"]["method"] == "knn"
    assert "report" in evaluation["artifacts"]


def test_generate_plot_writes_svg_and_png(tmp_path):
    assoc = pd.DataFrame(
        {
            "covariate": ["disease", "disease", "age_group", "age_group"],
            "PC": ["PC1", "PC2", "PC1", "PC2"],
            "p_value": [0.01, 0.2, 0.03, 0.9],
            "-log10p": [2.0, 0.7, 1.5, 0.05],
        }
    )
    assoc_path = tmp_path / "assoc.csv"
    assoc.to_csv(assoc_path, index=False)

    result = mcp_tools.generate_plot(
        table_path=str(assoc_path),
        output_dir=str(tmp_path / "plots"),
        plot_type="embedding_covariate_heatmap",
    )

    assert result["artifacts"]["svg"].endswith(".svg")
    assert result["artifacts"]["png"].endswith(".png")


def test_simulate_dataset_writes_h5ad(tmp_path):
    adata_path = _write_test_adata(tmp_path)

    result = mcp_tools.simulate_dataset(
        adata_path=str(adata_path),
        output_dir=str(tmp_path / "simulated"),
        cell_group_key="cell_type",
        abundance_perturbation={"B": 2.0},
        dropout_rate=0.0,
        expression_noise_scale=0.0,
        seed=0,
    )

    simulated = read_h5ad(result["artifacts"]["simulated_adata"])
    assert simulated.n_obs > 0
    assert result["summary"]["processed_output"] is False


def test_run_supervised_prediction_writes_predictions_and_evaluation(tmp_path, monkeypatch):
    adata_path = _write_test_adata(tmp_path)

    class FakeModel:
        def __init__(self, sample_key, label_keys, tasks, cell_group_key=None, layer="X_pca", seed=67):
            self.sample_key = sample_key
            self.label_keys = label_keys
            self.tasks = tasks
            self.seed = seed
            self.samples = None

        def prepare_anndata(self, adata):
            self.samples = adata.obs[self.sample_key].unique()

        def predict(self, label):
            return pd.DataFrame(
                {
                    "prob_case": [0.9, 0.2, 0.8],
                    "prob_control": [0.1, 0.8, 0.2],
                    f"{label}_pred": ["case", "control", "case"],
                },
                index=self.samples,
            )

        def get_sample_importance(self):
            return pd.DataFrame({f"{self.label_keys[0]}_importance": [0.9, 0.1, 0.8]}, index=self.samples)

    monkeypatch.setitem(
        mcp_tools.SUPERVISED_MODEL_SPECS,
        "mixmil",
        mcp_tools.SupervisedModelSpec("FakeModel"),
    )
    monkeypatch.setattr(mcp_tools.tl, "FakeModel", FakeModel, raising=False)

    result = mcp_tools.run_supervised_prediction(
        adata_path=str(adata_path),
        output_dir=str(tmp_path / "supervised"),
        sample_key="donor_id",
        label_keys=["disease"],
        tasks=["classification"],
        model="mixmil",
    )

    evaluation = json.loads((tmp_path / "supervised" / "prediction_evaluation.json").read_text(encoding="utf-8"))
    assert "prediction_disease" in result["artifacts"]
    assert evaluation["disease"]["metric"] == "f1_macro_calibrated"
