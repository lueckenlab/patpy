import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mrvi
import numpy as np
import pandas as pd
import PILOT as pt
import SCellBOW as sb
import scipy
import seaborn as sns
import torch
import WassersteinTSNE as WT
from scarches.models.scpoli import scPoli


def prepare_data_for_phemd(adata, sample_col, n_top_var_genes: int = 100):
    """Convert the expression data to the input format of PhEMD

    Returns
    -------
    all_expression_data : list
        Expression data of G genes for the cells of each of S samples
    all_genes : list[str]
        Names of the genes for the expression data
    selected_genes : list[str]
        Subset of genes to use
    samples_names : list[str]
        List of S names for the samples
    """
    top_variance = adata.var["variances"].sort_values(ascending=False)[:n_top_var_genes]
    selected_genes = top_variance.index

    expression_data = adata.X.toarray()
    samples_names = adata.obs[sample_col]

    return expression_data, adata.var_names, selected_genes, samples_names


def convert_cell_types_to_phemd_format(
    adata, cell_type_col, sample_col, output_dir="./", cell_types=None, n_top_var_genes=100
):
    """Converts `adata` to the tables required by PhEMD and saves them to `output_dir`"""
    if cell_types is None:
        cell_types = adata.obs[cell_type_col].unique()

    for cell_type in cell_types:
        cell_type_adata = adata[adata.obs[cell_type_col] == cell_type]
        all_expression_data, all_genes, selected_genes, samples_names = prepare_data_for_phemd(
            cell_type_adata, sample_col, n_top_var_genes
        )

        cell_type_dir = Path(output_dir) / cell_type
        cell_type_dir.mkdir(exist_ok=True)

        pd.DataFrame(all_expression_data).to_csv(cell_type_dir / "expression.csv", index=False, header=False)
        pd.DataFrame(all_genes).to_csv(cell_type_dir / "all_genes.csv", index=False, header=False)
        pd.DataFrame(selected_genes).to_csv(cell_type_dir / "selected_genes.csv", index=False, header=False)
        pd.DataFrame(samples_names).to_csv(cell_type_dir / "samples.csv", index=False, header=False)


def create_colormap(df, col, palette="Spectral"):
    """Create a color map for the unique values of the column `col` of data frame `df`"""
    unique_values = df[col].unique()

    colors = sns.color_palette(palette, n_colors=len(unique_values))
    color_map = dict(zip(unique_values, colors))
    return df[col].map(color_map)


class PatientsRepresentationMethod:
    """Base class for patient representation methods"""

    DISTANCES_UNS_KEY = "X_method-name_distances"

    @staticmethod
    def filter_small_samples(adata, sample_key, sample_size_threshold: int = 300) -> set:
        """Leave only samples with not less than `sample_size_threshold` cells"""
        sample_size_counts = adata.obs[sample_key].value_counts()
        small_samples = sample_size_counts[sample_size_counts < sample_size_threshold].index
        filtered_samples = set(adata.obs[sample_key]) - set(small_samples)
        print(len(small_samples), "samples removed:", ", ".join(small_samples))

        return filtered_samples

    @staticmethod
    def filter_small_cell_types(adata, sample_key, cells_type_key, cluster_size_threshold: int = 5) -> set:
        """Leave only cell types with not less than `cluster_size_threshold` cells"""
        cells_counts = adata.obs[[sample_key, cells_type_key]].value_counts().reset_index(name="count")

        # This step does not filter cell types with 0 counts
        small_cell_types = cells_counts.loc[cells_counts["count"] < cluster_size_threshold, cells_type_key].unique()
        small_cell_types = set(small_cell_types)

        if cluster_size_threshold > 0:
            # Add cell types with 0 counts in some samples
            for sample in adata.obs[sample_key].unique():
                for cell_type in adata.obs[cells_type_key].unique():
                    sample_cells = adata[(adata.obs[sample_key] == sample) & (adata.obs[cells_type_key] == cell_type)]
                    if not sample_cells:
                        small_cell_types.add(cell_type)

        filtered_cell_types = set(adata.obs[cells_type_key]) - set(small_cell_types)
        print(len(small_cell_types), "cell types removed:", ", ".join(small_cell_types))

        return filtered_cell_types

    def _extract_metadata(self, columns) -> pd.DataFrame:
        """Return dataframe with requested `columns` in the correct rows order"""
        metadata = self.adata.obs[[self.sample_key, *columns]].drop_duplicates()
        metadata = metadata.set_index(self.sample_key)
        return metadata.loc[self.samples]

    def __init__(self, sample_key, cells_type_key, seed=67):
        self.sample_key = sample_key
        self.cells_type_key = cells_type_key
        self.seed = seed

        self.adata = None
        self.samples = None
        self.cell_types = None
        self.embeddings = {}

    # fit-like method: save data and process it
    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Prepare adata for the analysis, filter cell types and samples with too few observations

        Parameters
        ----------
        sample_size_threshold : int = 300
            Minimum sample size to be considered. Samples with fewer number of cells
            are filtered out
        cluster_size_threshold : int = 5
            Minimum cell type size per sample to be considered. Cell types with fewer
            number of cells at least in 1 sample are filtered out

        """
        self.adata = adata

        # Filter samples with too few cells
        filtered_samples = self.filter_small_samples(
            adata=self.adata, sample_key=self.sample_key, sample_size_threshold=sample_size_threshold
        )
        self.samples = list(filtered_samples)
        self.adata = self.adata[self.adata.obs[self.sample_key].isin(filtered_samples)].copy()

        filtered_cell_types = self.filter_small_cell_types(
            adata=self.adata,
            sample_key=self.sample_key,
            cells_type_key=self.cells_type_key,
            cluster_size_threshold=cluster_size_threshold,
        )
        self.cell_types = list(filtered_cell_types)
        self.adata = self.adata[self.adata.obs[self.sample_key].isin(filtered_samples)].copy()
        self.adata = self.adata[self.adata.obs[self.cells_type_key].isin(filtered_cell_types)].copy()

    def calculate_distance_matrix(self, force: bool = False):
        """Transform-like method: returns samples distances matrix"""
        if self.DISTANCES_UNS_KEY in self.adata.uns and not force:
            return self.adata.uns[self.DISTANCES_UNS_KEY]

    def plot_clustermap(self, metadata_cols=None, figsize=(10, 12), *args, **kwargs):
        """Plot a clusterized heatmap of distances"""
        import scipy.cluster.hierarchy as hc
        import scipy.spatial as sp

        distances = self.calculate_distance_matrix(*args, **kwargs)
        linkage = hc.linkage(sp.distance.squareform(distances), method="average")

        if not metadata_cols:
            return sns.clustermap(distances, row_linkage=linkage, col_linkage=linkage)

        metadata = self._extract_metadata(columns=metadata_cols)

        annotation_colors = {}

        for col in metadata_cols:
            annotation_colors[col] = create_colormap(metadata, col)

        annotation_colors = pd.DataFrame(annotation_colors)

        return sns.clustermap(
            pd.DataFrame(distances, index=annotation_colors.index, columns=annotation_colors.index),
            col_colors=annotation_colors,
            figsize=figsize,
        )

    def embed(self, method="MDS"):
        """Calculate embedding of samples with the given `method`"""
        if method == "MDS":
            from sklearn.manifold import MDS

            mds = MDS(n_components=2, dissimilarity="precomputed")
            coordinates = mds.fit_transform(self.adata.uns[self.DISTANCES_UNS_KEY])

        self.embeddings[method] = coordinates
        return coordinates

    def plot_embedding(self, method="MDS", metadata_cols=None):
        """Plot embedding of samples colored by `metadata_cols`"""
        if method not in self.embeddings:
            self.embed(method=method)

        embedding_df = pd.DataFrame(self.embeddings[method], columns=[f"{method}_0", f"{method}_1"], index=self.samples)

        if metadata_cols is None:
            # Simply plot the embedding
            axes = sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1")
        else:
            # Colorize samples by metadata
            metadata_df = self._extract_metadata(columns=metadata_cols)
            embedding_df = pd.concat([embedding_df, metadata_df], axis=1)

            _, axes = plt.subplots(nrows=1, ncols=len(metadata_cols), sharey=True, figsize=(len(metadata_cols) * 5, 5))

            for i, col in enumerate(metadata_cols):
                sns.scatterplot(embedding_df, x=f"{method}_0", y=f"{method}_1", hue=col, ax=axes[i])

        return axes


class MrVI(PatientsRepresentationMethod):
    """Deep generative modeling for quantifying sample-level heterogeneity in single-cell omics.

    Source: https://www.biorxiv.org/content/10.1101/2022.10.04.510898v1
    """

    DISTANCES_UNS_KEY = "X_mrvi_distances"

    def __init__(self, sample_key: str, cells_type_key: str, categorical_nuisance_keys: list, seed=67, **model_params):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.categorical_nuisance_keys = categorical_nuisance_keys

        self.model = None
        self.model_params = model_params

    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Train MrVI model

        Parameters
        ----------
        adata : AnnData object with raw counts in .X

        Sets
        ----
        model : MrVI model
        """
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        mrvi.MrVI.setup_anndata(
            self.adata, sample_key=self.sample_key, categorical_nuisance_keys=self.categorical_nuisance_keys
        )

        self.model = mrvi.MrVI(self.adata, **self.model_params)
        self.model.train()

    def calculate_distance_matrix(self, cells_mask=None, force: bool = False):
        """Return sample by sample distances matrix

        Parameters
        ----------
        cells_mask : Iterable[bool] with the size identical to the number of cells
            Boolean vector which indicates what cells to take for the calculation of the distances matrix.
            Could for example indicate cells of a particular cell type
        force : bool = False
            If True, recalculate distances

        Sets
        ----
        adata.obsm["X_mrvi_z"] - latent representation from the layer Z of MrVI
        adata.obsm["X_mrvi_u"] - latent representation from the layer U of MrVI
        adata.uns["X_mrvi_distances"] – matrix of distances between samples according to MrVI representation

        Returns
        -------
        Matrix of distances between samples
        """
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        if "X_mrvi_z" not in self.adata.obsm or force:
            self.adata.obsm["X_mrvi_z"] = self.model.get_latent_representation(give_z=True)
        if "X_mrvi_u" not in self.adata.obsm or force:
            self.adata.obsm["X_mrvi_u"] = self.model.get_latent_representation(give_z=False)

        self.adata.uns[self.DISTANCES_UNS_KEY] = self.model.get_local_sample_representation(return_distances=True)

        if cells_mask is None:
            sample_sample_distances = self.adata.uns[self.DISTANCES_UNS_KEY].mean(axis=0)

        else:
            distance_matrices_subset = self.adata.uns[self.DISTANCES_UNS_KEY][cells_mask, :, :]
            sample_sample_distances = distance_matrices_subset.mean(axis=0)

        return sample_sample_distances


class WassersteinTSNE(PatientsRepresentationMethod):
    """Method based on the matrix of pairwise Wasserstein distances between units.

    Source: https://arxiv.org/abs/2205.07531
    """

    DISTANCES_UNS_KEY = "X_wasserstein_distances"

    @staticmethod
    def _set_diagonal_to_zeros(matrix):
        """Set diagonal elements of the `matrix` to zeros

        Sometimes, these elements don't equal 0 for some reason. TODO: check why

        Parameters
        ----------
        matrix : Uniion[pandas.DataFrame, numpy.array]
            Square matrix of distances

        Returns
        -------
        Matrix where the diagonal elements are zeros
        """
        if isinstance(matrix, pd.DataFrame):
            for sample in matrix.index:
                matrix.loc[sample, sample] = 0
            return matrix

        elif isinstance(matrix, np.ndarray):
            for i in range(matrix.shape[0]):
                matrix[i, i] = 0
            return matrix

        else:
            raise TypeError(f"Type {type(matrix)} is not supported")

    def __init__(self, sample_key, cells_type_key, replicate_key, seed=67):
        """Create Wasserstein distances embedding between samples

        Parameters
        ----------
        sample_key : str
            Key in .obs that specifies the samples between which distances are calculated.
            This corresponds to "unit" in the original WassersteinTSNE paper
        replicate_key : str
            Key in .obs that specifies some kind of replicate for the observations of a sample.
            Could be cell types. Corresponds to "sample" in the original WassersteinTSNE paper
        """
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.replicate_key = replicate_key

        self.data = None
        self.model = None
        self.distances_model = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Set up Gaussian Wasserstein Distance model"""
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        self.data = pd.DataFrame(adata.X)
        self.data.set_index([adata.obs[self.sample_key], adata.obs[self.replicate_key]], inplace=True)

        self.model = WT.Dataset2Gaussians(self.data)
        self.distances_model = WT.GaussianWassersteinDistance(self.model)

    def calculate_distance_matrix(self, covariance_weight=0.5, force: bool = False):
        r"""Return sample by sample distances matrix

        Parameters
        ----------
        covariance_weight : float = 0.5
            Float between 0 and 1, which indicates how much the distance between covariances
            influences the distances. Corresponds to a parameter $\\lambda$ in original paper,
            and to papameter `w` in the WassersteinTSNE package
        force : bool = False
            If True, recalculate distances

        Returns
        -------
        Matrix of distances between samples
        """
        is_correct_key_in_uns = (
            "wasserstein_covariance_weight" in self.adata.uns
            and self.adata.uns["wasserstein_covariance_weight"] == covariance_weight
        )
        is_recalculated = force or not is_correct_key_in_uns

        if self.DISTANCES_UNS_KEY in self.adata.uns:
            if is_recalculated:
                warnings.warn(f"Rewriting uns key {self.DISTANCES_UNS_KEY}", stacklevel=1)
            else:
                return self.adata.uns[self.DISTANCES_UNS_KEY]

        distances = self.distances_model.matrix(covariance_weight)
        distances = self._set_diagonal_to_zeros(distances)
        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["wasserstein_covariance_weight"] = covariance_weight

        return self.adata.uns[self.DISTANCES_UNS_KEY]

    def plot_clustermap(self, covariance_weight=0.5):
        """Plot clusterized heatmap of samples"""
        return super().clustermap(covariance_weight=covariance_weight)


class CloudPred(PatientsRepresentationMethod):
    """End-to-end differentiable learning algorithm which is coupled with a biologically informed mixture of cell types model.

    Source: https://arxiv.org/abs/2110.07069
    """

    DISTANCES_UNS_KEY = "X_cloudpred_distances"

    def _reduce_dims(self, X):
        """Transform data in the same way as during training"""
        pc = np.load(f"{self.data_dir}/{self.patient_state_col}/pc_{self.transform}_{self.seed}_{self.dims}_5.npz")[
            "pc"
        ]
        pc = pc[:, : self.dims]

        ### Project onto principal components ###
        mu = scipy.sparse.vstack([x[0] for x in X]).mean(axis=0)

        return [(x[0].dot(pc) - np.matmul(mu, pc), *x[1:]) for x in X]

    def _get_model_predictions(self, n_mixtures) -> pd.DataFrame:
        """Read the whole dataset in Cloudpred format, and get the output of Mixture model for it

        Parameters
        ----------
        n_mixtures : int
            Number of Gaussian mixtures in the model

        Returns
        -------
        Data frame with the following columns:
        - gaussian_0, gaussian_1, ... – probabilities from i_th gaussian for each of the `n_mixtures`
        - cell_type
        - outcome : the patient state
        - patient : index of the patient
        """
        col_names = [f"gaussian_{i}" for i in range(n_mixtures)]

        # Read data. For each state it contains a list of tuples with 3 elements
        # 1. Counts matrix
        # 2. Patient state
        # 3. Cell types vector
        # Don't ask me, I did not suggest this format
        X = np.load(self.data_dir / self.patient_state_col / "Xall.pkl", allow_pickle=True)
        # Merge classes 0 and 1 into one array
        X = [*X[0], *X[1]]
        X = self._reduce_dims(X)

        probs_df = None

        for idx, patient_data in enumerate(X):
            cells, outcome, cell_types = patient_data

            for cell_type in np.unique(cell_types):
                cell_type_mask = cell_types == cell_type
                with torch.no_grad():
                    cells_probs = self.model.mixture.forward(cells[cell_type_mask]).numpy()

                cells_probs = np.array(cells_probs)

                patient_df = pd.DataFrame(cells_probs.reshape(-1, n_mixtures), columns=col_names)
                patient_df["cell_type"] = cell_type
                patient_df["outcome"] = outcome
                patient_df["patient"] = idx

                if probs_df is None:
                    probs_df = patient_df
                else:
                    probs_df = pd.concat([probs_df, patient_df])

        return probs_df

    def __init__(
        self,
        sample_key,
        cells_type_key,
        patient_state_col,
        data_dir="./",
        seed=67,
        test_size=0.25,
        valid_size=0.25,
        dims=100,
        pc=50,
        transform="log",
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.patient_state_col = patient_state_col
        self.data_dir = Path(data_dir)
        self.test_size = test_size
        self.valid_size = valid_size
        self.dims = dims
        self.pc = pc
        self.transform = transform

        self.model = None
        self.probs_df = None
        self.patient_representations = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Save files in a format readable for CloudPred

        Note: you need a 2 times size of your `adata` additional disk storage, because
        authors of CloudPred duplicate the data on the disk twice

        Parameters
        ----------
        adata : AnnData object with normalized counts in .X
        """
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        # Save data for each state to the separate file as required by cloudpred
        for state in self.adata.obs[self.patient_state_col].unique():
            state_dir_path = self.data_dir / self.patient_state_col / str(state)
            state_dir_path.mkdir(parents=True, exist_ok=True)

        for sample in self.samples:
            state = self.adata.obs.loc[self.adata.obs[self.sample_key] == sample, self.patient_state_col].values[0]

            X = self.adata.X[self.adata.obs[self.sample_key] == sample, :].transpose()

            state_dir_path = self.data_dir / self.patient_state_col / str(state)
            scipy.sparse.save_npz(state_dir_path / f"{sample}.npz", X)
            np.save(
                state_dir_path / f"ct_{sample}.npy",
                self.adata.obs[self.adata.obs[self.sample_key] == sample][self.cells_type_key].values,
            )

    def calculate_distance_matrix(self, centers=5, force: bool = False):
        """Run the cloudpred model and convert putput to distances"""
        from cloudpred.main import run_pipeline

        distances = super().calculate_distance_matrix(force=force)

        if distances is not None and self.adata.uns["cloudpred_parameters"]["centers"] == centers:
            return distances

        run_pipeline(
            data_dir=f"{self.data_dir / self.patient_state_col}/",
            centers=[centers],
            dims=self.dims,
            pc=self.pc,
            seed=self.seed,
            transform=self.transform,
            valid_size=self.valid_size,
            test_size=self.test_size,
        )

        self.model = torch.load(self.data_dir / self.patient_state_col / "model.pt")
        self.probs_df = self._get_model_predictions(n_mixtures=centers)
        self.patient_representations = (
            self.probs_df.drop(["cell_type", "outcome"], axis=1).groupby("patient").agg(np.mean)
        )

        distances = scipy.spatial.distance.pdist(self.patient_representations.values)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["cloudpred_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "patient_state_col": self.patient_state_col,
            "centers": centers,
            "transform": self.transform,
        }

        return distances


class PILOT(PatientsRepresentationMethod):
    """Optimal transport based method to compute the Wasserstein distance between two single single-cell experiments.

    Source: https://www.biorxiv.org/content/10.1101/2022.12.16.520739v1
    """

    DISTANCES_UNS_KEY = "X_pilot_distances"

    @staticmethod
    def _make_matrix_symmetric(matrix):
        """Distances matrix returned by pilot.wasserstein_d is slightly not symmetric. This method fixes it"""
        if (matrix == matrix.T).all():
            return matrix

        unsymmetry = matrix - matrix.T
        warnings.warn(f"Distances matrix is not symmetric. Highest deviation: {unsymmetry.max()}. Fixing", stacklevel=1)

        return (matrix + matrix.T) / 2

    def __init__(
        self,
        sample_key,
        cells_type_key,
        patient_state_col,
        dataset_name="pilot_dataset",
        embedding_matrix="X_pca",
        seed=67,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.patient_state_col = patient_state_col
        self.dataset_name = dataset_name
        self.embedding_matrix = embedding_matrix

        self.results_dir = None
        self.pc = None
        self.annotation = None
        self.patient_representations = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Set up PILOT model"""
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        self.pc, self.annotation, self.results_dir = pt.extract_data_anno_scRNA_from_h5ad(
            self.adata,
            emb_matrix=self.embedding_matrix,
            clusters_col=self.cells_type_key,
            sample_col=self.sample_key,
            status=self.patient_state_col,
            name_dataset=self.dataset_name,
        )

    def calculate_distance_matrix(self, c_reg: float = 10, force: bool = False):
        """Calculate matrix of distances between samples"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        cluster_representations = pt.Cluster_Representations(
            self.annotation, regulizer=c_reg, regularization=(c_reg != 0)
        )

        self.patient_representations = np.array(list(cluster_representations.values()))

        cell_types_distances = pt.cost_matrix(self.annotation, self.pc, self.results_dir, cell_col=0)

        distances = pt.wasserstein_d(
            cluster_representations,
            cell_types_distances / cell_types_distances.max(),
            regularized="unreg",
            path=self.results_dir,
        )

        distances = self._make_matrix_symmetric(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["pilot_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "patient_state_col": self.patient_state_col,
            "c_reg": c_reg,
        }

        return distances


class TotalPseudobulk(PatientsRepresentationMethod):
    """A simple baseline, which represents patients as pseudobulk of their gene expression"""

    DISTANCES_UNS_KEY = "X_pseudobulk_distances"

    def __init__(self, sample_key, cells_type_key, embedding_matrix="X_pca", seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.embedding_matrix = embedding_matrix
        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False, average="mean"):
        """Calculate distances between pseudobulk representations of samples"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        self.patient_representations = np.zeros(
            shape=(len(self.samples), self.adata.obsm[self.embedding_matrix].shape[1])
        )

        if average == "mean":
            func = np.mean
        elif average == "median":
            func = np.median
        else:
            raise ValueError(f"Averaging function {average} is not supported")

        for i, sample in enumerate(self.samples):
            sample_cells = self.adata[self.adata.obs[self.sample_key] == sample, :]
            self.patient_representations[i] = func(sample_cells.obsm[self.embedding_matrix], axis=0)

        distances = scipy.spatial.distance.pdist(self.patient_representations)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["bulk_parameters"] = {"sample_key": self.sample_key, "average": average}

        return distances


class CellTypePseudobulk(PatientsRepresentationMethod):
    """Baseline, where distances between patients are average distances between their cell type pseudobulks"""

    DISTANCES_UNS_KEY = "X_cellsbulk_distances"

    def __init__(self, sample_key, cells_type_key, embedding_matrix="X_pca", seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.embedding_matrix = embedding_matrix
        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False, average="mean"):
        """Calculate distances between patients as average distance between per cell-type pseudobulks"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        if average == "mean":
            func = np.mean
        elif average == "median":
            func = np.median
        else:
            raise ValueError(f"Averaging function {average} is not supported")

        # List of matrices with embedding centroids for samples for each cell type
        self.patient_representations = np.zeros(
            shape=(len(self.cell_types), len(self.samples), self.adata.obsm[self.embedding_matrix].shape[1])
        )

        for i, cell_type in enumerate(self.cell_types):
            for j, sample in enumerate(self.samples):
                cells = self.adata[
                    (self.adata.obs[self.sample_key] == sample) & (self.adata.obs[self.cells_type_key] == cell_type)
                ]
                self.patient_representations[i, j] = func(cells.obsm[self.embedding_matrix], axis=0)

        # Matrix of distances between samples for each cell type
        distances = np.zeros(shape=(len(self.cell_types), len(self.samples), len(self.samples)))

        for i, cell_type_embeddings in enumerate(self.patient_representations):
            samples_distances = scipy.spatial.distance.pdist(cell_type_embeddings)
            distances[i] = scipy.spatial.distance.squareform(samples_distances)

        avg_distances = distances.mean(axis=0)

        self.adata.uns[self.DISTANCES_UNS_KEY] = avg_distances
        self.adata.uns["celltypebulk_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "average": average,
        }

        return avg_distances


class CellTypesComposition(PatientsRepresentationMethod):
    """A simple baseline, which represents patients as composition of their cell types"""

    DISTANCES_UNS_KEY = "X_celltype_composition"

    def __init__(self, sample_key, cells_type_key, seed=67):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.patient_representations = None

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between patients represented as cell type composition vectors"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        # Calculate proportions of the cell types for each sample
        self.patient_representations = pd.crosstab(
            self.adata.obs[self.sample_key], self.adata.obs[self.cells_type_key], normalize="index"
        )

        distances = scipy.spatial.distance.pdist(self.patient_representations.values)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["composition_parameters"] = {"sample_key": self.sample_key, "distance_type": "euclidean"}

        return distances


class SCellBOW(PatientsRepresentationMethod):
    """NLP based approach from https://www.biorxiv.org/content/10.1101/2022.12.28.522060v1.full.pdf"""

    DISTANCES_UNS_KEY = "X_scellbow"

    def __init__(
        self,
        sample_key,
        cells_type_key,
        model_dir="scellbow_model",
        n_worker=1,
        latent_dim: int = 300,
        n_iter: int = 20,
        seed=67,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.model_dir = model_dir
        self.n_worker = n_worker
        self.latent_dim = latent_dim
        self.n_iter = n_iter
        self.patient_representations = None

    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Pretrain SCellBOW model"""
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        sb.SCellBOW_pretrain(
            self.adata, save_dir=self.model_dir, vec_size=self.latent_dim, n_worker=self.n_worker, iter=self.n_iter
        )

        self.adata = sb.SCellBOW_cluster(self.adata, self.model_dir).run()

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between patients"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        self.patient_representations = self.adata.obsm["X_embed"]

        distances = scipy.spatial.distance.pdist(self.patient_representations.values)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["scellbow_parameters"] = {
            "sample_key": self.sample_key,
            "distance_type": "euclidean",
            "latent_dim": self.latent_dim,
            "n_iter": self.n_iter,
        }

        return distances


class SCPoli(PatientsRepresentationMethod):
    """A semi-supervised conditional deep generative model from https://www.biorxiv.org/content/10.1101/2022.11.28.517803v1"""

    early_stopping_kwargs = {
        "early_stopping_metric": "val_prototype_loss",
        "mode": "min",
        "threshold": 0,
        "patience": 20,
        "reduce_lr": True,
        "lr_patience": 13,
        "lr_factor": 0.1,
    }

    DISTANCES_UNS_KEY = "X_scpoli"

    def __init__(
        self,
        sample_key,
        cells_type_key,
        latent_dim=3,
        seed=67,
        n_epochs: int = 50,
        pretraining_epochs: int = 40,
        eta: float = 5,
    ):
        super().__init__(sample_key=sample_key, cells_type_key=cells_type_key, seed=seed)

        self.latent_dim = latent_dim
        self.model = None
        self.patient_representation = None
        self.n_epochs = n_epochs
        self.pretraining_epochs = pretraining_epochs
        self.eta = eta

    def prepare_anndata(self, adata, sample_size_threshold: int = 300, cluster_size_threshold: int = 5):
        """Set up scPoli model"""
        super().prepare_anndata(
            adata=adata, sample_size_threshold=sample_size_threshold, cluster_size_threshold=cluster_size_threshold
        )

        self.model = scPoli(
            adata=self.adata,
            condition_key=self.sample_key,
            cell_type_keys=self.cells_type_key,
            embedding_dim=self.latent_dim,
        )

        self.model.train(
            n_epochs=self.n_epochs,
            pretraining_epochs=self.pretraining_epochs,
            early_stopping_kwargs=self.early_stopping_kwargs,
            eta=self.eta,
        )

        self.patient_representation = self.model.get_conditional_embeddings().X

    def calculate_distance_matrix(self, force: bool = False):
        """Calculate distances between scPoli sample embeddings"""
        distances = super().calculate_distance_matrix(force=force)

        if distances is not None:
            return distances

        distances = scipy.spatial.distance.pdist(self.patient_representation)
        distances = scipy.spatial.distance.squareform(distances)

        self.adata.uns[self.DISTANCES_UNS_KEY] = distances
        self.adata.uns["scpoli_parameters"] = {
            "sample_key": self.sample_key,
            "cells_type_key": self.cells_type_key,
            "distance_type": "euclidean",
            "latent_dim": self.latent_dim,
            "n_epochs": self.n_epochs,
            "pretraining_epochs": self.pretraining_epochs,
            "eta": self.eta,
        }

        return distances
