import numpy as np
import pandas as pd


def calculate_compositional_metrics(adata, sample_key, composition_keys, normalize_to: int = 100) -> pd.DataFrame:
    """
    Calculate compositional metrics for the given AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object
    sample_key : str
        Key for the sample information in `adata.obs`
    composition_keys : list[str]
        List of columns from `adata.obs` representing the composition categories (e.g. cell type)
    normalize_to : int = 100
        Value to which the compositional metrics will be normalized. Default is 100

    Returns
    -------
    compositional_metrics : pandas.DataFrame
        DataFrame containing compositional metrics. Rows are samples, and columns
        are categories from each of `composition_keys`. Values are fractions of
        categories in samples

    Examples
    --------
    >>> example = sc.AnnData(
            X=np.random.normal(size=(4, 2)),
            obs=pd.DataFrame(
                {"sample": ["a", "a", "b", "b"],
                 "cell_type": ["A", "B", "A", "A"]})
            )
    >>> calculate_compositional_metrics(example, sample_key="sample", composition_keys=["cell_type"])
    cell_type  cell_type_A  cell_type_B
    sample
    a                 50.0         50.0
    b                100.0          0.0
    """
    compositional_metrics = []

    for col in composition_keys:
        # Create table of counts of cells in each sample per category
        col_proportions = pd.crosstab(
            index=adata.obs[sample_key], columns=adata.obs[col], normalize="index"  # Sum by sample equals to 1
        )

        # Add name of the original column to new columns
        # E.g. if there were a column "cell_type" with categories "B" and "T"
        # In the resulting data frame there will be columns "cell_type_B" and "cell_type_T"
        new_col_names = {category: f"{col}_{category}" for category in col_proportions.columns}
        col_proportions = col_proportions.rename(columns=new_col_names)
        col_proportions *= normalize_to

        compositional_metrics.append(col_proportions)

    compositional_metrics = pd.concat(compositional_metrics, axis=1)

    return compositional_metrics


def calculate_cell_qc_metrics(adata, sample_key, cell_qc_vars, agg_function=np.median) -> pd.DataFrame:
    """
    Calculate agregated cell quality control metrics for the given AnnData object

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    sample_key : str
        Key for the sample information in `adata.obs`
    cell_qc_vars: list[str]
        List of column keys representing the cell QC variables. For example, number of genes per cell
    agg_function: Callable = numpy.median
        Aggregation function to use for aggregating cell QC metrics. Default is numpy.median

    Returns
    -------
    cells_qc_aggregated : pandas.DataFrame
        DataFrame with samples in rows and aggregated QC metrics in columns

    Examples
    --------
    >>> calculate_cell_qc_metrics(adata, sample_key="scRNASeq_sample_ID", cell_qc_vars=["QC_ngenes", "QC_total_UMI"])
                        median_QC_ngenes  median_QC_total_UMI
    scRNASeq_sample_ID
    G05061-Ja005E-PBCa            1112.0               3150.0
    G05064-Ja005E-PBCa             982.5               2955.0
    """
    new_col_names = {col_name: agg_function.__name__ + "_" + col_name for col_name in cell_qc_vars}

    metadata = adata.obs[[sample_key, *cell_qc_vars]].groupby(by=sample_key)

    cells_qc_aggregated = metadata.aggregate(agg_function)
    cells_qc_aggregated = cells_qc_aggregated.rename(columns=new_col_names)

    return cells_qc_aggregated


def calculate_n_cells_per_sample(adata, sample_key) -> pd.DataFrame:
    """
    Calculate the number of cells per sample in the given AnnData object.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    sample_key : str
        Key for the sample information in `adata.obs`

    Returns
    -------
    cell_counts : pandas.DataFrame
        DataFrame containing the number of cells per sample in the columns "n_cells"
    """
    cell_counts = pd.DataFrame(adata.obs[sample_key].value_counts())
    cell_counts.columns = ["n_cells"]
    return cell_counts


def filter_small_samples(adata, sample_key, sample_size_threshold: int = 300):
    """Leave only samples with not less than `sample_size_threshold` cells"""
    sample_size_counts = adata.obs[sample_key].value_counts()
    small_samples = sample_size_counts[sample_size_counts < sample_size_threshold].index
    filtered_samples = set(adata.obs[sample_key]) - set(small_samples)
    print(len(small_samples), "samples removed:", ", ".join(small_samples))

    adata = adata[adata.obs[sample_key].isin(filtered_samples)].copy()

    return adata


def filter_small_cell_types(adata, sample_key, cells_type_key, cluster_size_threshold: int = 5):
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

    adata = adata[adata.obs[cells_type_key].isin(filtered_cell_types)].copy()

    return adata
