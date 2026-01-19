import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from patpy.tl.sample_representation import CellGroupComposition


@pytest.fixture
def simple_adata():
    """Create a minimal AnnData for testing with varying cell type composition per sample."""
    n_genes = 50
    np.random.seed(42)

    # Create samples with DIFFERENT cell type compositions
    obs_data = []
    cell_types = ["T_cell", "B_cell", "Monocyte", "NK_cell", "Dendritic"]
    for sample_idx, sample in enumerate(["S1", "S2", "S3", "S4"]):
        for i, ct in enumerate(cell_types):
            # Different counts per cell type per sample to ensure varying compositions
            n_cells = 5 + sample_idx * 2 + i * 3
            obs_data.extend([(sample, ct)] * n_cells)

    obs = pd.DataFrame(obs_data, columns=["sample", "cell_type"])
    n_cells = len(obs)
    X = np.random.poisson(5, size=(n_cells, n_genes))

    adata = sc.AnnData(X=X.astype(float), obs=obs)
    return adata


@pytest.fixture
def categorical_adata():
    """Create AnnData with categorical columns including unused categories."""
    n_cells = 100
    n_genes = 50

    np.random.seed(42)
    X = np.random.poisson(5, size=(n_cells, n_genes))

    obs = pd.DataFrame(
        {
            "sample": pd.Categorical(
                np.repeat(["S1", "S2", "S3", "S4"], 25),
                categories=["S1", "S2", "S3", "S4", "S5_unused"],  # S5 is unused
            ),
            "cell_type": pd.Categorical(
                np.tile(["T_cell", "B_cell", "Monocyte", "NK_cell", "Dendritic"], 20),
                categories=["T_cell", "B_cell", "Monocyte", "NK_cell", "Dendritic", "Unused_type"],
            ),
        }
    )

    adata = sc.AnnData(X=X.astype(float), obs=obs)
    return adata


class TestCellGroupCompositionInitialization:
    def test_default_parameters(self):
        comp = CellGroupComposition()
        assert comp.sample_key == "sample"
        assert comp.cell_group_key == "type"
        assert comp.apply_clr is False
        assert comp.pseudocount == 1
        assert comp.sample_representation is None

    def test_custom_parameters(self):
        comp = CellGroupComposition(sample_key="donor", cell_group_key="celltype", apply_clr=True, pseudocount=0.5)
        assert comp.sample_key == "donor"
        assert comp.cell_group_key == "celltype"
        assert comp.apply_clr is True
        assert comp.pseudocount == 0.5


class TestCLRTransformation:
    """Test CLR transformation when apply_clr=True (SETA-style)."""

    def test_clr_row_sums_near_zero(self, simple_adata):
        """CLR transformation property: rows should sum to approximately 0."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        row_sums = comp.sample_representation.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 0, decimal=10)

    def test_clr_output_shape(self, simple_adata):
        """CLR output should have shape (n_samples, n_cell_types)."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        n_samples = simple_adata.obs["sample"].nunique()
        n_cell_types = simple_adata.obs["cell_type"].nunique()
        assert comp.sample_representation.shape == (n_samples, n_cell_types)

    def test_clr_values_after_distance_matrix(self, simple_adata):
        """CLR should be computed and stored in sample_representation."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        assert comp.sample_representation is not None
        n_samples = simple_adata.obs["sample"].nunique()
        n_cell_types = simple_adata.obs["cell_type"].nunique()
        assert comp.sample_representation.shape == (n_samples, n_cell_types)

    def test_clr_handles_unused_categories(self, categorical_adata):
        """Unused categories should not appear in CLR output."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(categorical_adata)
        comp.calculate_distance_matrix()

        # Should only have 4 samples and 5 cell types (not the unused ones)
        assert comp.sample_representation.shape == (4, 5)
        assert "S5_unused" not in comp.sample_representation.index
        assert "Unused_type" not in comp.sample_representation.columns

    def test_clr_with_custom_pseudocount(self, simple_adata):
        """Different pseudocounts should produce different results."""
        comp1 = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True, pseudocount=1)
        comp1.prepare_anndata(simple_adata)
        comp1.calculate_distance_matrix()
        clr_1 = comp1.sample_representation

        # Need to clear cache to allow recalculation with different params
        simple_adata.uns.pop("X_composition", None)
        simple_adata.uns.pop("composition_parameters", None)

        comp2 = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True, pseudocount=0.5)
        comp2.prepare_anndata(simple_adata)
        comp2.calculate_distance_matrix()
        clr_2 = comp2.sample_representation

        # Results should differ
        assert not np.allclose(clr_1.values, clr_2.values)

        # But both should still have row sums near zero
        np.testing.assert_array_almost_equal(clr_1.sum(axis=1), 0, decimal=10)
        np.testing.assert_array_almost_equal(clr_2.sum(axis=1), 0, decimal=10)


class TestProportions:
    """Test standard proportions when apply_clr=False."""

    def test_proportions_row_sums_to_one(self, simple_adata):
        """Proportions should sum to 1 for each sample."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=False)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        row_sums = comp.sample_representation.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, 1, decimal=10)

    def test_proportions_nonnegative(self, simple_adata):
        """Proportions should be non-negative."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=False)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        assert (comp.sample_representation.values >= 0).all()


class TestDistanceMatrix:
    def test_distance_matrix_shape(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix()

        n_samples = simple_adata.obs["sample"].nunique()
        assert distances.shape == (n_samples, n_samples)

    def test_distance_matrix_symmetric(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix()

        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_distance_matrix_diagonal_zero(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix()

        np.testing.assert_array_almost_equal(np.diag(distances), 0)

    def test_distance_matrix_nonnegative(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix()

        assert (distances >= 0).all()

    def test_sample_representation_set(self, simple_adata):
        """After calculating distances, sample_representation should be set."""
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        assert comp.sample_representation is not None
        assert isinstance(comp.sample_representation, pd.DataFrame)


class TestCaching:
    def test_distances_cached_in_uns(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix()

        assert "X_composition" in simple_adata.uns
        assert "composition_parameters" in simple_adata.uns

    def test_parameters_stored_correctly(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix(dist="euclidean")

        params = simple_adata.uns["composition_parameters"]
        assert params["sample_key"] == "sample"
        assert params["cell_group_key"] == "cell_type"
        assert params["distance_type"] == "euclidean"
        assert params["apply_clr"] is True
        assert params["pseudocount"] == 1

    def test_parameters_stored_without_clr(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=False)
        comp.prepare_anndata(simple_adata)
        comp.calculate_distance_matrix(dist="euclidean")

        params = simple_adata.uns["composition_parameters"]
        assert params["apply_clr"] is False
        assert params["pseudocount"] is None

    def test_force_recalculates(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)

        distances1 = comp.calculate_distance_matrix()
        # Modify cached value
        simple_adata.uns["X_composition"] = np.zeros_like(distances1)

        # Without force, should return cached (zeros)
        distances_cached = comp.calculate_distance_matrix(force=False)
        assert (distances_cached == 0).all()

        # With force, should recalculate
        distances_forced = comp.calculate_distance_matrix(force=True)
        np.testing.assert_array_almost_equal(distances_forced, distances1)


class TestDistanceMetrics:
    def test_euclidean_distance(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix(dist="euclidean")
        assert distances is not None

    def test_cosine_distance(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix(dist="cosine", force=True)
        assert distances is not None

    def test_cityblock_distance(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)
        distances = comp.calculate_distance_matrix(dist="cityblock", force=True)
        assert distances is not None

    def test_different_metrics_give_different_results(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)

        dist_euclidean = comp.calculate_distance_matrix(dist="euclidean", force=True)
        dist_cosine = comp.calculate_distance_matrix(dist="cosine", force=True)
        dist_cityblock = comp.calculate_distance_matrix(dist="cityblock", force=True)

        # All should be different
        assert not np.allclose(dist_euclidean, dist_cosine)
        assert not np.allclose(dist_euclidean, dist_cityblock)
        assert not np.allclose(dist_cosine, dist_cityblock)

    def test_invalid_distance_raises(self, simple_adata):
        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(simple_adata)

        with pytest.raises(ValueError):
            comp.calculate_distance_matrix(dist="invalid_metric", force=True)


class TestEdgeCases:
    def test_single_cell_type(self):
        """CellGroupComposition with CLR should work with only one cell type."""
        n_cells = 100
        np.random.seed(42)
        X = np.random.poisson(5, size=(n_cells, 50))
        obs = pd.DataFrame(
            {
                "sample": np.repeat(["S1", "S2", "S3", "S4"], 25),
                "cell_type": ["T_cell"] * n_cells,  # Only one cell type
            }
        )
        adata = sc.AnnData(X=X.astype(float), obs=obs)

        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(adata)
        distances = comp.calculate_distance_matrix()

        # With only one cell type, all samples have the same composition
        # So all distances should be 0
        np.testing.assert_array_almost_equal(distances, 0)

    def test_two_samples(self):
        """CellGroupComposition with CLR should work with only two samples."""
        n_cells = 50
        np.random.seed(42)
        X = np.random.poisson(5, size=(n_cells, 50))
        obs = pd.DataFrame(
            {
                "sample": np.repeat(["S1", "S2"], 25),
                "cell_type": np.tile(["T_cell", "B_cell", "Monocyte", "NK_cell", "Dendritic"], 10),
            }
        )
        adata = sc.AnnData(X=X.astype(float), obs=obs)

        comp = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp.prepare_anndata(adata)
        distances = comp.calculate_distance_matrix()

        assert distances.shape == (2, 2)
        assert distances[0, 0] == 0
        assert distances[1, 1] == 0
        assert distances[0, 1] == distances[1, 0]  # Symmetric


class TestCLRvsProportion:
    """Test that CLR and proportions give different but valid results."""

    def test_clr_vs_proportions_different(self, simple_adata):
        """CLR and proportions should give different distance matrices."""
        comp_clr = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=True)
        comp_clr.prepare_anndata(simple_adata)
        dist_clr = comp_clr.calculate_distance_matrix()

        # Reset the adata uns
        simple_adata.uns.pop("X_composition", None)

        comp_prop = CellGroupComposition(sample_key="sample", cell_group_key="cell_type", apply_clr=False)
        comp_prop.prepare_anndata(simple_adata)
        dist_prop = comp_prop.calculate_distance_matrix()

        # Should be different
        assert not np.allclose(dist_clr, dist_prop)

        # But both should be valid distance matrices
        np.testing.assert_array_almost_equal(dist_clr, dist_clr.T)
        np.testing.assert_array_almost_equal(dist_prop, dist_prop.T)
        np.testing.assert_array_almost_equal(np.diag(dist_clr), 0)
        np.testing.assert_array_almost_equal(np.diag(dist_prop), 0)
