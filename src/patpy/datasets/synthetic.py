import warnings

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse


def bootstrap_genes(cell, proportions=None, noise_scale: float = 0.05):
    """Simulate a similar cell by sampling genes with proportions given by `proportions` and adding noise to them

    Parameters
    ----------
    cell : np.ndarray or scipy.sparse.csr_matrix
        Expression of genes in a cell to bootstrap
    proportions : np.ndarray, default None
        Proportions of genes to sample. If None, use the original proportions
    noise_scale : float, default 0.05
        Scale of the noise added to the proportions. 0.05 means 5% deviation from the original proportions

    Returns
    -------
    np.ndarray
        Bootstraped expression of genes in a cell
    """
    if issparse(cell):
        # float64 cast is required because otherwise, the sum of proportions is > 1 later due to numeric precision
        cell = cell.toarray().astype(np.float64)

    total_counts = cell.sum()

    if proportions is None:
        proportions = (cell / total_counts).flatten().astype(np.float64)

    if noise_scale > 0:
        noise = np.random.uniform(1 - noise_scale, 1 + noise_scale, size=proportions.shape)
        proportions *= noise
        proportions /= proportions.sum()  # Renormalise

    return np.random.multinomial(total_counts, proportions)


def simulate_cells(adata, layer, cell_type_key, cell_type_counts: pd.Series):
    """
    Generate a new annotated data object by simulating cells similar to the ones in `adata`

    The simulation works as following:
    1. Sample random cells from `adata` with number in each cell type given by `cell_type_counts`
    2. For each cell, select a random neighbor and interpolate between the cell and its neighbor

    An interpolation is required to make new cells less similar to the original ones.
    Otherwise, the nearest neighbor of a new cell is always the original cell, even with noise and dropout.

    Parameters
    ----------
    adata : AnnData
        Annotated object with single-cell RNA-seq data
    layer : str
        Layer to use for sampling
    cell_type_key : str
        Key in `adata.obs` that stores cell type labels
    cell_type_counts : pd.Series
        Counts of cells for each cell type. Can be obtained from `adata.obs[cell_type_key].value_counts()`

    Returns
    -------
    AnnData
        Annotated object with simulated cells
    """
    cells = []

    data = adata.layers[layer] if layer != "X" else adata.X
    barcodes = []

    for cell_type, count in zip(cell_type_counts.index, cell_type_counts, strict=False):
        cell_type_barcodes = adata[adata.obs[cell_type_key] == cell_type].obs_names
        random_cells = np.random.choice(cell_type_barcodes, size=count)

        for barcode in random_cells:
            # Interpolate between cell and its random neighbor
            cell_idx = adata.obs_names.get_loc(barcode)
            random_neighbor_idx = _get_random_neighbor(adata, cell_idx)
            neighbor_cell = data[random_neighbor_idx].toarray().ravel()
            interpolation_factor = np.random.random()

            cell = data[cell_idx].toarray().ravel()
            new_cell = cell * (1 - interpolation_factor) + neighbor_cell * interpolation_factor
            cells.append(new_cell)

        barcodes.extend(random_cells)

    return sc.AnnData(np.array(cells), var=adata.var, obs=adata.obs.loc[barcodes])


def perturb_genes(
    adata, cell_type_key, gene_perturbation: dict[dict[str, str]] = None, layer=None, perturbation_strength: float = 1.0
):
    """
    Perturb gene expression in `adata` for given cell types and genes

    Parameters
    ----------
    adata : AnnData
        Annotated object with single-cell RNA-seq data
    cell_type_key : str
        Key in `adata.obs` that stores cell type labels
    gene_perturbation : dict[dict[str, str]]
        Perturbation of gene expression. Keys are cell types, values are dictionaries with genes and fold changes for perturbation scale.
        E.g. value 2 means that gene expression will be doubled (if `perturbation_strength` is 1)
    layer : str, default None
        Layer to use for perturbation. If None, use `adata.X`
    perturbation_strength : float from 0 to 1, default 1.0
        Strength of the perturbation where 0 means not perturbed, and 1 means perturbed at maximim scale (defined in the corresponding dictionaries)
    """
    for cell_type, genes_perturbed in gene_perturbation.items():
        cell_type_indices = adata.obs[cell_type_key] == cell_type

        for gene, fold_change in genes_perturbed.items():
            if gene not in adata.var_names:
                warnings.warn(f"Gene {gene} not found in adata.var_names", stacklevel=1)
                continue

            if layer is None or layer == "X":
                adata.X[cell_type_indices, adata.var_names == gene] *= (
                    1 - perturbation_strength
                ) + perturbation_strength * fold_change
            else:
                adata.layers[layer][cell_type_indices, adata.var_names == gene] *= (
                    1 - perturbation_strength
                ) + perturbation_strength * fold_change

    return adata


def perturb_cell_type_abundance(
    cell_type_counts, abundance_perturbation: dict[str, float] = None, perturbation_strength: float = 1.0
):
    """
    Perturb cell type abundance in `adata` for given cell types

    Parameters
    ----------
    cell_type_counts : pd.Series
        Counts of cells for each cell type. Can be obtained from `adata.obs[cell_type_key].value_counts()`
    abundance_perturbation : dict[str, float]
        Perturbation of cell type abundance. Keys are cell types, values are fold changes.
        E.g. value 2 means that cell type abundance will be doubled (if `perturbation_strength` is 1)
    perturbation_strength : float from 0 to 1, default 1.0
        Strength of the perturbation where 0 means not perturbed, and 1 means perturbed at maximim scale (defined in the corresponding dictionaries)
    """
    cell_type_counts = cell_type_counts.astype(float)  # Prevent warning due to float32 cast

    for cell_type, fold_change in abundance_perturbation.items():
        cell_type_counts[cell_type] *= (1 - perturbation_strength) + perturbation_strength * fold_change

    return cell_type_counts.astype(int)


def _get_random_neighbor(adata, idx):
    """Get a random neighbor of a cell at index `idx` in `adata`"""
    distances = adata.obsp["distances"][idx].toarray()[0]
    neighbor_indices = np.nonzero(distances)[0]
    neighbor_indices = neighbor_indices[neighbor_indices > 0]  # Not sample the same cell
    random_neighbor_idx = np.random.choice(neighbor_indices)
    return random_neighbor_idx


def simulate_data(
    adata,
    cell_type_key,
    layer=None,
    abundance_perturbation: dict[str, float] = None,
    gene_perturbation: dict[dict[str, str]] = None,
    perturbation_strength: float = 1.0,
    expression_noise_scale: float = 0.05,
    dropout_rate: float = 0.7,
):
    """
    Simulate data with perturbation of cell type abundance and/or gene expression

    Simulation works as follows:
    - Bootstrapping cells with given cell type counts
    - Bootstrapping genes for each cell adding multiplicative noise
    - Perturbing gene expression for given cell types and genes
    - Adding dropout to the expression

    Parameters
    ----------
    adata : AnnData
        Annotated object with single-cell RNA-seq data. Must have `cell_type_key` in `adata.obs` and `distances` in `adata.obsp`
    cell_type_key : str
        Key in `adata.obs` that stores cell type labels
    layer : str, default "X"
        Layer to use for perturbation. If None, use `adata.X`
    abundance_perturbation : dict[str, float], default None
        Perturbation of cell type abundance. Keys are cell types, values are fold changes .
        E.g. value 2 means that cell type abundance will be doubled (if `perturbation_strength` is 1)
    gene_perturbation : dict[dict[str, float]], default None
        Perturbation of gene expression. Keys are cell types, values are dictionaries with genes and their perturbation scales defined as fold changes.
        E.g. value 2 means that gene expression will be doubled (if `perturbation_strength` is 1)
    perturbation_strength : float from 0 to 1
        Strength of the perturbation where 0 means not perturbed, and 1 means perturbed at maximim scale (defined in the corresponding dictionaries)
    expression_noise_scale : float from 0 to 1, default 0.05
        Scale of the noise added to the expression of each gene. 0.05 means 5% deviation from the original expression
    dropout_rate : float from 0 to 1, default 0.7
        Rate of the dropout applied to the expression of each gene

    Returns
    -------
    adata : AnnData
        Annotated object with simulated perturbed single-cell RNA-seq data
    """
    if layer is None:
        layer = "X"

    cell_type_counts = adata.obs[cell_type_key].value_counts()

    if abundance_perturbation is not None:
        cell_type_counts = perturb_cell_type_abundance(
            cell_type_counts=cell_type_counts,
            abundance_perturbation=abundance_perturbation,
            perturbation_strength=perturbation_strength,
        )

    bootstrapped_adata = simulate_cells(
        adata, layer=layer, cell_type_key=cell_type_key, cell_type_counts=cell_type_counts
    )

    new_cells = np.zeros(shape=bootstrapped_adata.shape)

    for i, cell in enumerate(bootstrapped_adata.X):
        new_cells[i] = bootstrap_genes(cell, noise_scale=expression_noise_scale)

    bootstrapped_adata.X = new_cells

    if gene_perturbation is not None:
        bootstrapped_adata = perturb_genes(
            adata=bootstrapped_adata,
            cell_type_key=cell_type_key,
            gene_perturbation=gene_perturbation,
            layer="X",
            perturbation_strength=perturbation_strength,
        )

    if dropout_rate:
        # We could estimate sparsity for each gene from the original data i.e.:
        # sparsity_level = np.array((adata.X == 0).mean(axis=0)).ravel()
        # dropout_mask = np.random.binomial(1, 1-sparsity_level, size=(bootstrapped_adata.n_obs, adata.n_vars)) == 0
        # bootstrapped_adata.X[dropout_mask] = 0
        # However, original data *already* has dropout with quite high level (>90%), so appying it again would leave too few genes. Let's put a fixed smaller rate instead.

        dropout_mask = np.random.binomial(1, dropout_rate, size=(bootstrapped_adata.n_obs, adata.n_vars)) == 1
        bootstrapped_adata.X[dropout_mask] = 0

    return bootstrapped_adata


def process_adata(adata, verbose: bool = False):
    """
    Process `adata` by calculating QC metrics, normalizing, scaling, PCA, building neighbors graph and UMAP

    Parameters
    ----------
    adata : AnnData
        Annotated object with single-cell RNA-seq data
    verbose : bool, default False
        Whether to print progress

    Returns
    -------
    adata : AnnData
        Annotated object with processed single-cell RNA-seq data
    """
    if verbose:
        print("Calculating QC metrics...")
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    if verbose:
        print("Normalizing...")
    sc.pp.normalize_total(adata)

    if verbose:
        print("Log transforming...")
    sc.pp.log1p(adata)

    if verbose:
        print("Scaling...")
    sc.pp.scale(adata)

    if verbose:
        print("PCA...")
    sc.tl.pca(adata, n_comps=30)

    if verbose:
        print("Building neighbors graph...")
    sc.pp.neighbors(adata, n_neighbors=50, n_pcs=30)

    if verbose:
        print("UMAP...")
    sc.tl.umap(adata)

    return adata


def covid_19_hallmarks():
    """Return some of the hallmarks of COVID-19 based on COMBAT study: https://pmc.ncbi.nlm.nih.gov/articles/PMC8776501

    Returns
    -------
    abundance_hallmarks : dict[str, float]
        Cell type abundance hallmarks of COVID-19. Keys are cell types and values are fold changes of cell type proportions
    expression_hallmarks : dict[str, dict[str, float]]
        Expression hallmarks of severe COVID-19. Keys are cell types, values are genes and their DEG effect size corrected for dropout
    """
    # Cell types that change in proportions. It is actually interconnected (e.g. increasing one cell type proportion decreases others), but
    # it is not particularly important for a simple model
    abundance_hallmarks = {
        "FCGR3A+ Monocytes": 0.4,  # Approx 6 / 16 based on Fig. S4
        "Dendritic cells": 0.2,  # Approx 0.8 / 2.5 for pDC based on Fig. S4, 0.5 / 4 for cDC
        "CD14+ Monocytes": 2,  # Approximated from Figure 3B
        "CD8 T cells": 0.5,  # Approximated from Figure 3B
        "NK cells": 4,
    }

    # Because of dropout, estimated FCs will be smaller than "true", so let's account for this.
    # We will multiply overexpressed genes by a coefficient, and leave lowly expressed as is
    dropout_bias_fix_coef = 1 / 0.3

    # Gene expression changes
    expression_hallmarks = {
        "CD4 T cells": {
            "IFITM1": 0.5,  # From Supp table 3
            "ID3": 8 * dropout_bias_fix_coef,  # From Supp table 3
            "TUBA1B": 0.55,  # From Supp table 3
        },
        "CD8 T cells": {
            "MT-CO1": 1.65 * dropout_bias_fix_coef,  # From Supp table 3
            "GNLY": 0.25,  # From Supp table 3
        },
        "NK cells": {
            "S1PR1": 4.76 * dropout_bias_fix_coef,  # From Supp table 3
        },
        "B cells": {
            "TXNIP": 0.42,  # From Supp table 3
        },
        "CD14+ Monocytes": {
            "S100A8": 2.3 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "HLA-DQA1": 0.125,  # Based on Supp Fig. 4 F
            "CLU": 2**6.05 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "FAM20A": 2**5.95 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "PIM3": 2**2 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "CYP19A1": 2**6.1 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "PIM1": 2**3.8 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "CDKN2D": 2 * dropout_bias_fix_coef,  # Based on Supp Fig. 4 F
            "CDIP1": 2**-1.8,  # Based on Supp Fig. 4 F
            "CACNA2D3": 2**-4,  # Based on Supp Fig. 4 F
            # "ADGRE3": 2 ** -3.2,  # Based on Supp Fig. 4 F, not found in the PBMC3k data
            "ZNF217": 2**-0.6,  # Based on Supp Fig. 4 F
            "ZNF703": 2**-3,  # Based on Supp Fig. 4 F
            "ADAMTS5": 2**-5,  # Based on Supp Fig. 4 F
            "HLA-DQA2": 2**-4.8,  # Based on Supp Fig. 4 F
        },
    }

    return abundance_hallmarks, expression_hallmarks
