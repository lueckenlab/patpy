import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from patpy.tl import characterize_clusters, cluster_covariate_enrichment

CLUSTER_KEY = "leiden"


@pytest.fixture
def clustered_adata():
    """Three clusters of samples where cluster L0 is set apart by one pathway and by disease.

    Cluster L0: high `pathway_up` in monocytes, high `frac_mono`, all "lupus", all from study_A.
    Clusters L1 and L2: background values, mixed disease, both studies.
    `pathway_flat` and `sex` carry no cluster signal at all.
    """
    rng = np.random.default_rng(0)
    n_per_cluster = 20
    clusters = np.repeat(["L0", "L1", "L2"], n_per_cluster)
    n_samples = len(clusters)
    is_l0 = clusters == "L0"

    obs = pd.DataFrame(
        {
            CLUSTER_KEY: clusters,
            "age": rng.normal(50, 10, n_samples),
            "disease": np.where(is_l0, "lupus", rng.choice(["healthy", "colitis"], n_samples)),
            "study": np.where(is_l0, "study_A", rng.choice(["study_A", "study_B"], n_samples)),
            "sex": rng.choice(["female", "male"], n_samples),
        },
        index=[f"sample_{i}" for i in range(n_samples)],
    )

    composition = pd.DataFrame(
        {
            "frac_mono": rng.normal(0.2, 0.02, n_samples) + is_l0 * 0.15,
            "frac_t_cell": rng.normal(0.5, 0.05, n_samples),
        },
        index=obs.index,
    )
    pathways = pd.DataFrame(
        {
            "pathway_up": rng.normal(0.0, 0.1, n_samples) + is_l0 * 1.0,
            "pathway_flat": rng.normal(0.0, 0.1, n_samples),
        },
        index=obs.index,
    )

    adata = AnnData(np.zeros((n_samples, 1), dtype=np.float32), obs=obs)
    adata.obsm["cell_type_composition"] = composition
    adata.obsm["gene_set_scores_Mono"] = pathways
    return adata


def test_characterize_clusters_finds_the_enriched_features(clustered_adata):
    summary = characterize_clusters(
        clustered_adata,
        cluster_key=CLUSTER_KEY,
        obs_keys=["age"],
        obsm_keys=["cell_type_composition", "gene_set_scores_Mono"],
    )

    assert set(summary["cluster"]) == {"L0", "L1", "L2"}
    assert set(summary["feature_set"]) == {"obs", "cell_type_composition", "gene_set_scores_Mono"}
    # Every (cluster, feature) pair is tested: 3 clusters x (1 obs + 2 + 2) features
    assert len(summary) == 3 * 5

    significant_in_l0 = summary.query("cluster == 'L0' and p_adjusted < 0.05")
    assert set(significant_in_l0["feature"]) == {"pathway_up", "frac_mono"}

    pathway_up = significant_in_l0.query("feature == 'pathway_up'").iloc[0]
    assert pathway_up["cohens_d"] > 1
    assert pathway_up["auroc"] > 0.9
    assert pathway_up["mean_in"] > pathway_up["mean_out"]
    assert pathway_up["n_in"] == 20
    assert pathway_up["n_out"] == 40

    # Features with no cluster signal stay non-significant
    flat = summary.query("feature in ['pathway_flat', 'age'] and p_adjusted < 0.05")
    assert flat.empty


def test_characterize_clusters_defaults_to_numeric_obs_columns(clustered_adata):
    summary = characterize_clusters(clustered_adata, cluster_key=CLUSTER_KEY)

    # `age` is the only numeric obs column; the categorical ones are ignored
    assert set(summary["feature"]) == {"age"}


def test_characterize_clusters_skips_small_clusters(clustered_adata):
    clustered_adata.obs.loc["sample_0", CLUSTER_KEY] = "tiny"

    summary = characterize_clusters(clustered_adata, cluster_key=CLUSTER_KEY, obs_keys=["age"])

    assert "tiny" not in set(summary["cluster"])


def test_characterize_clusters_tolerates_missing_values(clustered_adata):
    clustered_adata.obs.loc[clustered_adata.obs_names[:5], "age"] = np.nan

    summary = characterize_clusters(clustered_adata, cluster_key=CLUSTER_KEY, obs_keys=["age"])

    # The 5 samples with a missing age drop out of the test, whichever cluster they sit in
    assert (summary["n_in"] + summary["n_out"] == clustered_adata.n_obs - 5).all()
    assert summary["p_value"].notna().all()


def test_characterize_clusters_rejects_unknown_keys(clustered_adata):
    with pytest.raises(ValueError, match="not found in adata.obs"):
        characterize_clusters(clustered_adata, cluster_key="absent")

    with pytest.raises(ValueError, match="not found in adata.obsm"):
        characterize_clusters(clustered_adata, cluster_key=CLUSTER_KEY, obsm_keys=["absent"])

    with pytest.raises(ValueError, match="test must be one of"):
        characterize_clusters(clustered_adata, cluster_key=CLUSTER_KEY, obs_keys=["age"], test="wilcoxon")


def test_cluster_covariate_enrichment_ranks_disease_above_sex(clustered_adata):
    enrichment, association = cluster_covariate_enrichment(
        clustered_adata, cluster_key=CLUSTER_KEY, covariates=["disease", "study", "sex"]
    )

    association = association.set_index("covariate")
    assert association.loc["disease", "cramers_v"] > association.loc["sex", "cramers_v"]
    assert association.loc["disease", "p_value"] < 0.05
    assert association.loc["sex", "p_value"] > 0.05

    lupus_in_l0 = enrichment.query("covariate == 'disease' and cluster == 'L0' and level == 'lupus'").iloc[0]
    assert lupus_in_l0["frac_in_cluster"] == 1.0
    assert lupus_in_l0["frac_outside"] == 0.0
    assert lupus_in_l0["log2_odds_ratio"] > 0
    assert lupus_in_l0["p_adjusted"] < 0.05


def test_cluster_covariate_enrichment_handles_missing_values(clustered_adata):
    clustered_adata.obs["therapy_response"] = pd.Series(
        ["R"] * 10 + [np.nan] * 50, index=clustered_adata.obs_names, dtype=object
    )

    dropped, _ = cluster_covariate_enrichment(
        clustered_adata, cluster_key=CLUSTER_KEY, covariates=["therapy_response"], dropna=True
    )
    kept, _ = cluster_covariate_enrichment(
        clustered_adata, cluster_key=CLUSTER_KEY, covariates=["therapy_response"], dropna=False
    )

    assert set(dropped["level"]) == {"R"}
    assert set(kept["level"]) == {"R", "missing"}


def test_cluster_covariate_enrichment_rejects_unknown_covariates(clustered_adata):
    with pytest.raises(ValueError, match="Covariates not found in adata.obs"):
        cluster_covariate_enrichment(clustered_adata, cluster_key=CLUSTER_KEY, covariates=["absent"])
