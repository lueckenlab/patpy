import json

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from patpy.datasets._gene_sets import _parse_gmt, download_gene_sets
from patpy.pp import score_gene_sets

SAMPLE_KEY = "sample_id"
CELL_TYPE_KEY = "cell_type"

PATHWAY = "PATHWAY_UP"
OTHER_PATHWAY = "PATHWAY_FLAT"
GENE_SETS = {
    PATHWAY: [f"gene_{i}" for i in range(5)],
    OTHER_PATHWAY: [f"gene_{i}" for i in range(50, 55)],
    "PATHWAY_ABSENT": ["not_measured_1", "not_measured_2"],
}


@pytest.fixture
def scoring_adata():
    """Cells of two cell types, where sample_1 monocytes over-express the PATHWAY_UP genes."""
    rng = np.random.default_rng(0)
    n_cells, n_genes = 400, 600

    obs = pd.DataFrame(
        {
            SAMPLE_KEY: np.repeat(["sample_0", "sample_1", "sample_2", "sample_3"], n_cells // 4),
            CELL_TYPE_KEY: np.tile(["Mono", "T"], n_cells // 2),
        }
    )
    X = rng.lognormal(mean=0.5, sigma=0.3, size=(n_cells, n_genes))

    boosted = (obs[SAMPLE_KEY] == "sample_1").values & (obs[CELL_TYPE_KEY] == "Mono").values
    X[np.ix_(boosted, np.arange(5))] += 5

    return AnnData(
        X,
        obs=obs,
        var=pd.DataFrame(
            {"symbol": [f"gene_{i}" for i in range(n_genes)]},
            index=[f"ENSG{i:05d}" for i in range(n_genes)],
        ),
    )


# Scores are returned as a samples x pathways frame covering every sample.
def test_scores_shape_and_index(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]

    scores = score_gene_sets(
        adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY, OTHER_PATHWAY],
        gene_sets=GENE_SETS,
    )

    assert isinstance(scores, pd.DataFrame)
    assert list(scores.columns) == [PATHWAY, OTHER_PATHWAY]
    assert list(scores.index) == ["sample_0", "sample_1", "sample_2", "sample_3"]
    assert scores.notna().all().all()


# The sample whose monocytes over-express the gene set gets the highest score for it.
def test_scores_detect_upregulation(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]

    scores = score_gene_sets(
        adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY, OTHER_PATHWAY],
        gene_sets=GENE_SETS,
    )

    assert scores[PATHWAY].idxmax() == "sample_1"
    assert scores.loc["sample_1", PATHWAY] > scores.drop(index="sample_1")[PATHWAY].max() + 1

    # The unrelated gene set does not single out sample_1
    assert scores[OTHER_PATHWAY].idxmax() != "sample_1"


# The signal is specific to the cell type that carries it.
def test_scores_are_cell_type_specific(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]

    monocytes = score_gene_sets(
        adata, sample_key=SAMPLE_KEY, cell_type_key=CELL_TYPE_KEY, cell_type="Mono", gene_sets=GENE_SETS
    )
    t_cells = score_gene_sets(
        adata, sample_key=SAMPLE_KEY, cell_type_key=CELL_TYPE_KEY, cell_type="T", gene_sets=GENE_SETS
    )

    assert monocytes.loc["sample_1", PATHWAY] > t_cells.loc["sample_1", PATHWAY] + 1


# Gene symbols are looked up in a .var column when var_names are not symbols.
def test_gene_symbols_column(scoring_adata):
    with_symbols = score_gene_sets(
        scoring_adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY],
        gene_sets=GENE_SETS,
        gene_symbols="symbol",
    )
    assert with_symbols[PATHWAY].idxmax() == "sample_1"

    # Without the mapping, the Ensembl var_names match no gene of the set
    with pytest.warns(UserWarning, match="No genes of pathway"):
        without_symbols = score_gene_sets(
            scoring_adata,
            sample_key=SAMPLE_KEY,
            cell_type_key=CELL_TYPE_KEY,
            cell_type="Mono",
            pathways=[PATHWAY],
            gene_sets=GENE_SETS,
        )
    assert without_symbols[PATHWAY].isna().all()


# Scoring a layer instead of .X uses that layer's values.
def test_layer(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]
    # Shift every cell by one sample block: the boosted sample_1 monocytes land on sample_2
    adata.layers["shifted"] = np.roll(adata.X, shift=100, axis=0)

    from_x = score_gene_sets(
        adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY],
        gene_sets=GENE_SETS,
    )
    from_layer = score_gene_sets(
        adata,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY],
        gene_sets=GENE_SETS,
        layer="shifted",
    )

    assert from_layer[PATHWAY].idxmax() == "sample_2"  # the boosted cells moved to sample_2
    assert not np.allclose(from_x[PATHWAY], from_layer[PATHWAY])


# A pathway with no measured genes is reported as NaN rather than failing.
def test_absent_pathway_is_nan(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]

    with pytest.warns(UserWarning, match="No genes of pathway 'PATHWAY_ABSENT'"):
        scores = score_gene_sets(
            adata,
            sample_key=SAMPLE_KEY,
            cell_type_key=CELL_TYPE_KEY,
            cell_type="Mono",
            pathways=[PATHWAY, "PATHWAY_ABSENT"],
            gene_sets=GENE_SETS,
        )

    assert scores["PATHWAY_ABSENT"].isna().all()
    assert scores[PATHWAY].notna().all()


# Samples of the cohort that have no cells of the cell type score NaN.
def test_samples_without_cell_type_are_nan(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]
    monocytes = adata[adata.obs[CELL_TYPE_KEY] == "Mono"].copy()
    monocytes = monocytes[monocytes.obs[SAMPLE_KEY] != "sample_3"].copy()

    all_samples = list(adata.obs[SAMPLE_KEY].unique())
    scores = score_gene_sets(
        monocytes,
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY],
        gene_sets=GENE_SETS,
        samples=all_samples,
    )

    assert list(scores.index) == all_samples
    assert scores.loc["sample_3", PATHWAY] != scores.loc["sample_3", PATHWAY]  # NaN
    assert scores.drop(index="sample_3").notna().all().all()


# A cell type absent from the data yields an all-NaN frame with a warning.
def test_missing_cell_type(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]

    with pytest.warns(UserWarning, match="No cells of cell type 'B'"):
        scores = score_gene_sets(
            adata,
            sample_key=SAMPLE_KEY,
            cell_type_key=CELL_TYPE_KEY,
            cell_type="B",
            pathways=[PATHWAY],
            gene_sets=GENE_SETS,
        )

    assert scores.shape == (4, 1)
    assert scores.isna().all().all()


# Median aggregation is supported and differs from the mean.
def test_median_aggregation(scoring_adata):
    adata = scoring_adata.copy()
    adata.var_names = adata.var["symbol"]

    kwargs = dict(
        sample_key=SAMPLE_KEY,
        cell_type_key=CELL_TYPE_KEY,
        cell_type="Mono",
        pathways=[PATHWAY],
        gene_sets=GENE_SETS,
    )
    mean_scores = score_gene_sets(adata, aggregate="mean", **kwargs)
    median_scores = score_gene_sets(adata, aggregate="median", **kwargs)

    assert median_scores[PATHWAY].idxmax() == "sample_1"
    assert not np.allclose(mean_scores[PATHWAY], median_scores[PATHWAY])


# Bad arguments raise informative errors.
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_key": "missing"}, "sample_key"),
        ({"sample_key": SAMPLE_KEY, "cell_type_key": "missing", "cell_type": "Mono"}, "cell_type_key"),
        ({"sample_key": SAMPLE_KEY, "cell_type": "Mono"}, "together"),
        ({"sample_key": SAMPLE_KEY, "aggregate": "sum"}, "aggregate"),
        ({"sample_key": SAMPLE_KEY, "pathways": ["NOT_A_PATHWAY"]}, "not found in gene_sets"),
    ],
)
def test_invalid_arguments(scoring_adata, kwargs, match):
    with pytest.raises(ValueError, match=match):
        score_gene_sets(scoring_adata, gene_sets=GENE_SETS, **kwargs)


# GMT parsing keeps the gene set name and genes, and drops malformed lines.
def test_parse_gmt():
    gmt = "SET_A\thttp://a\tGENE1\tGENE2\nSET_B\tdescription\tGENE3\t\nEMPTY_SET\tdescription\n"

    gene_sets = _parse_gmt(gmt)

    assert gene_sets == {"SET_A": ["GENE1", "GENE2"], "SET_B": ["GENE3"]}


# Cached collections are read back from disk instead of re-downloaded.
def test_download_gene_sets_uses_cache(tmp_path):
    (tmp_path / "btm.json").write_text(json.dumps({"module": ["GENE1"]}))

    gene_sets = download_gene_sets(collections=("btm",), cache_dir=tmp_path)

    assert gene_sets == {"btm": {"module": ["GENE1"]}}


# Flattening prefixes gene-set names with their collection.
def test_download_gene_sets_flatten(tmp_path):
    (tmp_path / "btm.json").write_text(json.dumps({"module": ["GENE1"]}))
    (tmp_path / "hallmark_immune.json").write_text(json.dumps({"HALLMARK_HYPOXIA": ["GENE2"]}))

    gene_sets = download_gene_sets(collections=("btm", "hallmark_immune"), cache_dir=tmp_path, flatten=True)

    assert gene_sets == {"BTM__module": ["GENE1"], "H__HALLMARK_HYPOXIA": ["GENE2"]}


# An unknown collection is rejected before any download happens.
def test_download_gene_sets_unknown_collection(tmp_path):
    with pytest.raises(ValueError, match="Unknown collection"):
        download_gene_sets(collections=("not_a_collection",), cache_dir=tmp_path)
