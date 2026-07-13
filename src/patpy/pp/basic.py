import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def prepare_data_for_phemd(adata, sample_col, n_top_var_genes: int = 100):
    """Convert the expression data to the input format of PhEMD (R implementation)

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

    if sp.issparse(adata.X):
        expression_data = adata.X.toarray()
    else:
        expression_data = adata.X
    samples_names = adata.obs[sample_col]

    return expression_data, adata.var_names, selected_genes, samples_names


def convert_cell_types_to_phemd_format(
    adata, cell_type_col, sample_col, output_dir="./", cell_types=None, n_top_var_genes=100
):
    """Converts `adata` to the tables required by PhEMD (R implementation) and saves them to `output_dir`"""
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
            index=adata.obs[sample_key],
            columns=adata.obs[col],
            normalize="index",  # Sum by sample equals to 1
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


def extract_metadata(adata: sc.AnnData, sample_key: str, columns: list, samples: list = None) -> pd.DataFrame:
    """
    Return a dataframe with requested `columns` in the correct rows order.

    Parameters
    ----------
    - adata (sc.AnnData): The AnnData object containing the data.
    - sample_key (str): The key identifying the sample in the observation metadata.
    - columns (list): A list of column names to extract from the observation metadata.

    Returns
    -------
    - pd.DataFrame: A dataframe with the requested columns, indexed by the sample key.
    """
    if samples is None:
        samples = adata.obs[sample_key].unique()

    metadata = adata.obs[[sample_key, *columns]].drop_duplicates()

    # Check if the sample key is also in columns to avoid reindexing errors
    need_to_rename_sample_key = sample_key in columns

    # To avoid error, we rename column with sample key, reindex dataframe, and then rename sample column back
    if need_to_rename_sample_key:
        # Rename the first column with sample key to sample_key_dupl
        metadata.columns = [sample_key + "_dupl"] + list(metadata.columns[1:])

    metadata = metadata.set_index(sample_key)

    if need_to_rename_sample_key:
        metadata.rename(columns={sample_key + "_dupl": sample_key}, inplace=True)

    if (metadata.index.value_counts() > 1).any():
        warnings.warn(
            "Metadata contains multiple values for the same sample, taking only the first occurence", stacklevel=2
        )
        metadata = metadata[~metadata.index.duplicated(keep="first")]

    return metadata.loc[samples]


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


def filter_small_cell_groups(adata, sample_key, cell_group_key, cluster_size_threshold: int = 5):
    """Leave only cell groups with not less than `cluster_size_threshold` cells"""
    cells_counts = adata.obs[[sample_key, cell_group_key]].value_counts().reset_index(name="count")

    # This step does not filter cell types with 0 counts
    small_cell_types = cells_counts.loc[cells_counts["count"] < cluster_size_threshold, cell_group_key].unique()
    small_cell_types = set(small_cell_types)

    if cluster_size_threshold > 0:
        # Add cell types with 0 counts in some samples
        for sample in adata.obs[sample_key].unique():
            for cell_type in adata.obs[cell_group_key].unique():
                sample_cells = adata[(adata.obs[sample_key] == sample) & (adata.obs[cell_group_key] == cell_type)]
                if not sample_cells:
                    small_cell_types.add(cell_type)

    filtered_cell_types = set(adata.obs[cell_group_key]) - set(small_cell_types)
    print(len(small_cell_types), "cell types removed:", ", ".join(small_cell_types))

    adata = adata[adata.obs[cell_group_key].isin(filtered_cell_types)].copy()

    return adata


def subsample(adata, obs_category_col: str, min_samples_per_category: int, fraction=None, n_obs=None):
    """Subsample cells from each category in `obs_category_col` to have at least `min_samples_per_category` cells.

    Parameters
    ----------
    adata : AnnData
        Annotated data object containing cells.
    obs_category_col : str
        Name of the column in `adata.obs` containing categories to subsample.
    min_samples_per_category : int
        Minimum number of cells per category.
    fraction : float or None
        Fraction of cells to take from each category. If `None`, `n_obs` must be set.
    n_obs : int or None
        Number of cells to take from each category. If `None`, `fraction` must be set.

    Returns
    -------
    AnnData
        Subsampled AnnData object.
    """
    subsample_idxs = []

    assert fraction is None or 0 < fraction <= 1, "`fraction` must be a number between 0 and 1"
    if fraction is None:
        assert n_obs is not None and int(n_obs), "`n_obs` must be an integer number or `fraction` must be set"

    for level in adata.obs[obs_category_col].unique():
        level_cells = adata.obs[obs_category_col] == level
        obs_per_level = sum(level_cells)
        level_idxs = np.where(level_cells)[0]

        if obs_per_level <= min_samples_per_category:
            # Take all cells from this level
            subsample_idxs.extend(level_idxs)
        else:
            if fraction is not None:
                n_cells = int(fraction * obs_per_level)
            else:
                n_cells = int(n_obs)

            selected_cells_idxs = np.random.choice(level_idxs, size=n_cells, replace=False)
            subsample_idxs.extend(selected_cells_idxs)

    return adata[subsample_idxs]


def is_count_data(matrix, window_size=10000) -> bool:
    """Ensure that `matrix` only contains integers"""
    from scipy.sparse import issparse

    if issparse(matrix):
        return np.all(matrix[:window_size, :window_size].data % 1 == 0)

    return np.all(matrix[:window_size, :window_size] % 1 == 0)


def fill_nan_distances(distances, n_max_distances=5):
    """Fill NaN values in `distances` with maximum distance multiplied by `n_max_distances`"""
    distances = distances.copy()
    nans = np.isnan(distances)
    max_distance = distances[~nans].max()

    distances[nans] = n_max_distances * max_distance

    return distances


def _map_genes_to_var_names(adata: sc.AnnData, genes: list[str], gene_symbols: str | None) -> list[str]:
    """Translate gene symbols into `adata.var_names`, dropping genes absent from `adata`.

    When `gene_symbols` is `None`, `var_names` are assumed to be symbols already.
    """
    if gene_symbols is None:
        available = set(adata.var_names)
        return [gene for gene in genes if gene in available]

    symbol_to_var_name = pd.Series(adata.var_names, index=adata.var[gene_symbols].astype(str))
    symbol_to_var_name = symbol_to_var_name[~symbol_to_var_name.index.duplicated(keep="first")]

    present = [gene for gene in genes if gene in symbol_to_var_name.index]
    return list(symbol_to_var_name.loc[present])


def score_gene_sets(
    adata: sc.AnnData,
    sample_key: str,
    cell_type_key: str | None = None,
    cell_type: str | None = None,
    pathways: list[str] | None = None,
    gene_sets: dict[str, list[str]] | None = None,
    layer: str | None = None,
    gene_symbols: str | None = None,
    samples: list | None = None,
    aggregate: str = "mean",
    ctrl_size: int = 50,
    seed: int = 0,
) -> pd.DataFrame:
    """Score gene sets in the cells of one cell type and aggregate the scores per sample.

    Cells are scored with :func:`scanpy.tl.score_genes` (mean expression of the gene set
    minus the mean expression of a reference set matched on expression level), and the
    per-cell scores are then aggregated within each sample.

    The expression data must be normalised and log-transformed, as
    :func:`scanpy.tl.score_genes` expects.

    Parameters
    ----------
    adata : sc.AnnData
        Cell-level annotated data object.
    sample_key : str
        Column in `adata.obs` identifying samples.
    cell_type_key : str | None = None
        Column in `adata.obs` with cell-type annotations. Must be given together with
        `cell_type`; when both are `None`, all cells are scored.
    cell_type : str | None = None
        The cell type to score. Only cells of this type contribute to the scores.
    pathways : list[str] | None = None
        Names of the gene sets to score. Must be keys of `gene_sets`. Defaults to every
        gene set in `gene_sets`.
    gene_sets : dict[str, list[str]] | None = None
        Mapping `{pathway: [gene_symbols]}`. Defaults to the immune subset of the MSigDB
        Hallmark collection, downloaded with :func:`patpy.datasets.download_gene_sets`.
    layer : str | None = None
        Layer of `adata` holding the normalised log-expression. `None` means `.X`.
    gene_symbols : str | None = None
        Column in `adata.var` holding gene symbols. Use it when `var_names` are not
        symbols (e.g. Ensembl IDs). When `None`, `var_names` are assumed to be symbols.
    samples : list | None = None
        Samples making up the index of the returned frame. Defaults to the unique samples
        of `adata`. Samples without cells of `cell_type` get `NaN` scores — pass the full
        cohort here when `adata` only holds the cells of one cell type.
    aggregate : str = "mean"
        How per-cell scores are aggregated within a sample: `"mean"` or `"median"`.
    ctrl_size : int = 50
        Size of the reference gene set, passed to :func:`scanpy.tl.score_genes`.
    seed : int = 0
        Random seed for the reference gene sampling.

    Returns
    -------
    scores : pd.DataFrame
        Samples × pathways gene-set scores. Pathways whose genes are all absent from
        `adata` are returned as `NaN` columns.

    Examples
    --------
    >>> scores = score_gene_sets(
    ...     adata,
    ...     sample_key="sampleID",
    ...     cell_type_key="Level1",
    ...     cell_type="Mono",
    ...     pathways=["HALLMARK_INTERFERON_ALPHA_RESPONSE"],
    ... )  # doctest: +SKIP
    """
    if aggregate not in ("mean", "median"):
        raise ValueError(f"aggregate must be 'mean' or 'median', got {aggregate!r}.")
    if sample_key not in adata.obs.columns:
        raise ValueError(f"sample_key='{sample_key}' not found in adata.obs.")
    if (cell_type_key is None) != (cell_type is None):
        raise ValueError("`cell_type_key` and `cell_type` must be given together.")
    if cell_type_key is not None and cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key='{cell_type_key}' not found in adata.obs.")

    if gene_sets is None:
        from patpy.datasets import download_gene_sets

        gene_sets = download_gene_sets(collections=("hallmark_immune",))["hallmark_immune"]

    if pathways is None:
        pathways = list(gene_sets)

    unknown_pathways = [pathway for pathway in pathways if pathway not in gene_sets]
    if unknown_pathways:
        raise ValueError(f"Pathways not found in gene_sets: {unknown_pathways}")

    if samples is None:
        samples = list(adata.obs[sample_key].unique())

    if cell_type is not None:
        cells_of_cell_type = (adata.obs[cell_type_key] == cell_type).values

        if not cells_of_cell_type.any():
            warnings.warn(f"No cells of cell type '{cell_type}' found, returning NaN scores.", stacklevel=2)
            return pd.DataFrame(np.nan, index=pd.Index(samples, name=sample_key), columns=pathways)

        adata = adata[cells_of_cell_type]

    # score_genes writes into .obs, which a view does not allow
    adata = adata.copy()

    cell_scores = {}

    for pathway in pathways:
        genes = _map_genes_to_var_names(adata, gene_sets[pathway], gene_symbols)

        if not genes:
            warnings.warn(f"No genes of pathway '{pathway}' found in adata, its score is NaN.", stacklevel=2)
            cell_scores[pathway] = np.full(adata.n_obs, np.nan)
            continue

        sc.tl.score_genes(
            adata,
            gene_list=genes,
            score_name=pathway,
            layer=layer,
            ctrl_size=ctrl_size,
            random_state=seed,
            use_raw=False,
        )
        cell_scores[pathway] = adata.obs[pathway].values

    cell_scores = pd.DataFrame(cell_scores, index=adata.obs_names)
    cell_scores[sample_key] = adata.obs[sample_key].values

    scores = cell_scores.groupby(sample_key, observed=True).agg(aggregate)
    scores.index = scores.index.astype(str)

    return scores.reindex([str(sample) for sample in samples])


def _feature_names(adata: sc.AnnData, layer: str) -> list[str]:
    """Return feature names for the slot `layer` points to.

    Gene names are used for `.X` and `.layers`; `.obsm` matrices have no
    feature names, so positional names `{layer}_{i}` are generated.
    """
    if layer in (None, "X") or layer in adata.layers:
        return list(adata.var_names)

    if layer in adata.obsm:
        n_features = adata.obsm[layer].shape[1]
        return [f"{layer}_{i}" for i in range(n_features)]

    raise ValueError(f"layer='{layer}' not found in adata.obsm or adata.layers.")


def aggregate_sample_info(
    adata: sc.AnnData,
    sample_key: str,
    cell_type_key: str | None = None,
    layer: str | None = None,
    metadata_cols: list[str] | None = None,
    cell_type_pseudobulk: bool = False,
    aggregate: str = "mean",
    fill_value: float = np.nan,
) -> sc.AnnData:
    """Aggregate a cell-level AnnData into a sample-level ("meta") AnnData.

    Every piece of the output is optional and controlled by a key. With only
    `sample_key` given, the result is an empty AnnData with one observation per
    unique sample and no features and no metadata.

    Parameters
    ----------
    adata : sc.AnnData
        Cell-level annotated data object.
    sample_key : str
        Column in `adata.obs` identifying samples. Observations of the returned
        object are the unique values of this column, in order of appearance.
    cell_type_key : str | None = None
        Column in `adata.obs` with cell-type (cell-group) annotations. When given,
        cell-type composition is stored in `.obsm["cell_type_composition"]`.
    layer : str | None = None
        Source of the expression data to aggregate into `.X`. Looked up first in
        `adata.obsm`, then in `adata.layers`; pass `"X"` to aggregate `adata.X`.
        When `None`, no expression is aggregated and `.X` has zero features.
    metadata_cols : list[str] | None = None
        Columns of `adata.obs` to carry over to `.obs` of the returned object.
    cell_type_pseudobulk : bool = False
        If `True`, add one layer per cell type, named `{cell_type}_pseudobulk`,
        holding the pseudobulk of that cell type only. Samples without cells of a
        given cell type get `fill_value`. Requires `layer` and `cell_type_key`.
    aggregate : str = "mean"
        Aggregation function used for pseudobulk. One of `"mean"`, `"median"`, `"sum"`.
    fill_value : float = np.nan
        Value used for sample × cell-type combinations without cells.

    Returns
    -------
    meta_adata : sc.AnnData
        Sample-level object with `n_obs` equal to the number of unique samples.
        `.X` holds the pseudobulk of `layer` (empty if `layer` is `None`),
        `.obsm["cell_type_composition"]` the cell-type fractions,
        `.layers["{cell_type}_pseudobulk"]` the per-cell-type pseudobulks, and
        `.obs` the requested sample metadata.

    Examples
    --------
    >>> meta_adata = aggregate_sample_info(
    ...     adata,
    ...     sample_key="sample_id",
    ...     cell_type_key="cell_type",
    ...     layer="X",
    ...     metadata_cols=["disease", "sex"],
    ...     cell_type_pseudobulk=True,
    ... )  # doctest: +SKIP
    """
    # Imported here because patpy.tl imports patpy.pp at module level
    from patpy.tl.sample_representation import CellGroupComposition, Pseudobulk

    if sample_key not in adata.obs.columns:
        raise ValueError(f"sample_key='{sample_key}' not found in adata.obs.")
    if cell_type_key is not None and cell_type_key not in adata.obs.columns:
        raise ValueError(f"cell_type_key='{cell_type_key}' not found in adata.obs.")
    if cell_type_pseudobulk and (layer is None or cell_type_key is None):
        raise ValueError("cell_type_pseudobulk=True requires both `layer` and `cell_type_key` to be set.")

    samples = adata.obs[sample_key].unique()
    sample_names = [str(sample) for sample in samples]

    if metadata_cols:
        obs = extract_metadata(adata, sample_key=sample_key, columns=list(metadata_cols), samples=samples)
        obs.index = pd.Index(sample_names, name=sample_key)
    else:
        obs = pd.DataFrame(index=pd.Index(sample_names, name=sample_key))

    if layer is None:
        meta_adata = sc.AnnData(X=np.zeros((len(samples), 0)), obs=obs)
    else:
        pseudobulk = Pseudobulk(sample_key=sample_key, cell_group_key=cell_type_key, layer=layer)
        pseudobulk.prepare_anndata(adata)

        sample_pseudobulk = pseudobulk._get_pseudobulk(
            aggregation=aggregate, fill_value=fill_value, aggregate_cell_types=False
        )
        var = pd.DataFrame(index=pd.Index(_feature_names(adata, layer), name="feature"))
        meta_adata = sc.AnnData(X=sample_pseudobulk, obs=obs, var=var)

        if cell_type_pseudobulk:
            cell_type_pseudobulks = pseudobulk._get_pseudobulk(
                aggregation=aggregate, fill_value=fill_value, aggregate_cell_types=True
            )
            for cell_type, cell_type_data in zip(pseudobulk.cell_groups, cell_type_pseudobulks, strict=True):
                meta_adata.layers[f"{cell_type}_pseudobulk"] = cell_type_data

    if cell_type_key is not None:
        composition_method = CellGroupComposition(sample_key=sample_key, cell_group_key=cell_type_key)
        composition_method.prepare_anndata(adata)
        composition_method.calculate_distance_matrix()

        # Cell-type fractions have one column per cell type, so they cannot live in
        # .layers (which must match the shape of .X); .obsm is the fitting slot
        composition = composition_method.sample_representation.copy()
        composition.index = pd.Index(sample_names, name=sample_key)
        composition.columns = [str(cell_type) for cell_type in composition.columns]
        meta_adata.obsm["cell_type_composition"] = composition

    meta_adata.uns["aggregate_sample_info"] = {
        "sample_key": sample_key,
        "cell_type_key": cell_type_key,
        "layer": layer,
        "aggregate": aggregate,
    }

    return meta_adata


def _to_numpy(x):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)
    except (TypeError, RuntimeError) as err:
        raise TypeError(f"Failed to convert to numpy array: {err}") from err


def get_helical_embedding(
    adata: sc.AnnData,
    model: str,
    batch_size: int = 24,
    device: str = "cuda",
    **kwargs,
) -> sc.AnnData:
    """
    Compute and store cell embeddings from a Helical model in adata.obsm.

    Parameters
    ----------
    adata : AnnData
    model : str
        Which Helical model to use. Must be one of:
          - "scgpt"
              scGPT docs: https://helical.readthedocs.io/en/latest/model_cards/scgpt/
          - "geneformer"
              Geneformer docs: https://helical.readthedocs.io/en/latest/model_cards/geneformer/
          - "uce"
              UCE docs: https://helical.readthedocs.io/en/latest/model_cards/uce/
          - "transcriptformer"
              TranscriptFormer docs: https://helical.readthedocs.io/en/latest/model_cards/transcriptformer/
    batch_size : int, optional
        Batch size for inference. Defaults to 64.
    device : str, optional
        Device for PyTorch inference: "cpu" or "cuda". Defaults to "cpu".
    **kwargs :
        All remaining keyword arguments are passed directly into the chosen model’s Config constructor.
        Below is a summary of each model’s Config‐class attributes (names and defaults) that can be overridden:

        — scGPTConfig kwargs (from https://helical.readthedocs.io/en/latest/configs/scgpt_config/) :
            • pad_token (str): padding token. Default `"<pad>"`.
            • fast_transformer (bool): use fast transformer. Default `True`.
            • nlayers (int): number of layers. Default `12`.
            • nheads (int): number of attention heads. Default `8`.
            • embsize (int): embedding dimension. Default `512`.
            • d_hid (int): hidden layer dimension. Default `512`.
            • dropout (float): dropout rate. Default `0.2`.
            • n_layers_cls (int): classification head layers. Default `3`.
            • mask_value (int): mask token value. Default `-1`.
            • pad_value (int): padding token value. Default `-2`.
            • world_size (int): distributed world size. Default `8`.
            • accelerator (bool): whether to use accelerator. Default `False`.
            • use_fast_transformer (bool): alias for fast_transformer. Default `False`.

        — GeneformerConfig kwargs (from https://helical.readthedocs.io/en/latest/configs/geneformer_config/) :
            • model_name ({'gf-6L-30M-i2048','gf-12L-30M-i2048','gf-12L-95M-i4096',
                        'gf-20L-95M-i4096','gf-12L-95M-i4096-CLcancer'}):
            model variant. Default `"gf-12L-30M-i2048"`.
            • emb_layer (int): which layer to extract. Default `-1`.
            • emb_mode ({'cls','cell','gene'}): embedding mode. Default `"cell"`.
            • accelerator (bool): use accelerator. Default `False`.
            • nproc (int): processes for data prep. Default `1`.
            • custom_attr_name_dict (dict): map new obs attrs. Default `None`.

        — UCEConfig kwargs (from https://helical.readthedocs.io/en/latest/configs/uce_config/) :
            • model_name ({'33l_8ep_1024t_1280','4layer_model'}): model variant.
            Default `"4layer_model"`.
            • species ({'human','mouse','frog','zebrafish','mouse_lemur','pig',
                        'macaca_fascicularis','macaca_mulatta'}): data species.
            Default `"human"`.
            • gene_embedding_model ({'ESM2'}): gene embedding source. Default `'ESM2'`.
            • pad_length (int): sequence padding length. Default `1536`.
            • pad_token_idx (int): pad token index. Default `0`.
            • chrom_token_left_idx (int): left chromosome token. Default `1`.
            • chrom_token_right_idx (int): right chromosome token. Default `2`.
            • cls_token_idx (int): CLS token index. Default `3`.
            • CHROM_TOKEN_OFFSET (int): offset constant. Default `143574`.
            • sample_size (int): per-sample patch size. Default `1024`.
            • CXG (bool): use CXG format. Default `True`.
            • output_dim (int): final embedding dim. Default `1280`.
            • d_hid (int): hidden dim. Default `5120`.
            • token_dim (int): token embedding dim. Default `5120`.
            • multi_gpu (bool): multi-GPU inference. Default `False`.
            • accelerator (bool): use accelerator. Default `False`.

        — TranscriptFormerConfig kwargs (from https://helical.readthedocs.io/en/latest/configs/transcriptformer/) :
            • model_name ({'tf_sapiens','tf_metazoa','tf_exemplar'}):
            model variant. Default `"tf_metazoa"`.
            • emb_mode ({'gene','cell'}): The mode to use for the embeddings. Default `"cell"`.
            • output_keys (List[{'gene_llh','llh'}]): The keys to output. Default `["gene_llh"]`.
            • obs_keys (List[str]): obs columns to attach. Default `["all"]`.
            • data_files (List[str]): AnnData file paths. Default `[None]`.
            • output_path (str): where to save results. Default `"./inference_results"`.
            • load_checkpoint (str): Path to model weights file (automatically set by inference.py). Default `None`.
            • pretrained_embedding (str): Path to pretrained embeddings for out-of-distribution species. Default `None`.
            • precision (str): numerical precision. Default `"16-mixed"`.
            • gene_col_name (str): Column name in AnnData.var containing gene names which will be mapped to ensembl ids. If index is set, .var_names will be used. Default `"ensembl_id"`.
            • clip_counts (int): max count clipping. Default `30`.
            • filter_to_vocabs (bool): Whether to filter genes to only those in the vocabulary Default `True`.
            • filter_outliers (float): Standard deviation threshold for filtering outlier cells (0.0 = no filtering). Default `0.0`.
            • normalize_to_scale (float): Scale factor for count normalization (0 = no normalization). Default `0`.
            • sort_genes (bool): Whether to sort the genes. Default `False`.
            • randomize_genes (bool): Whether to randomize the genes. Default `False`.
            • min_expressed_genes (int): min genes per cell. Default `0`.

    Returns
    -------
    AnnData
        The same AnnData, but with a new .obsm key `"X_<model>"`. For example,
        if `model="scgpt"`, embeddings are stored in `adata.obsm["X_scgpt"]`.
        Shape is (n_cells, model_hidden_dim).

    Raises
    ------
    ImportError
        If the chosen Helical submodule isn’t installed (e.g. Helical wasn’t installed with `[scgpt]`).
    ValueError
        If `model` is not one of the four supported names.
    """
    model_lower = model.lower()

    if model_lower == "scgpt":
        from helical.models.scgpt import scGPT, scGPTConfig

        config = scGPTConfig(
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
        scgpt_model = scGPT(configurer=config)

        data_for_scgpt = scgpt_model.process_data(adata)

        embeddings = scgpt_model.get_embeddings(data_for_scgpt)

        adata.obsm["X_scgpt"] = embeddings
        return adata

    elif model_lower == "geneformer":
        from helical.models.geneformer import Geneformer, GeneformerConfig

        config = GeneformerConfig(
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
        gf_model = Geneformer(configurer=config)

        data_for_gf = gf_model.process_data(adata)

        embeddings = gf_model.get_embeddings(data_for_gf)

        adata.obsm["X_geneformer"] = embeddings
        adata.var_names = adata.var_names.astype(str)

        return adata

    elif model_lower == "uce":
        from helical.models.uce import UCE, UCEConfig

        adata.var_names = adata.var_names.astype(str)
        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X)

        config = UCEConfig(
            batch_size=batch_size,
            device=device,
            **kwargs,
        )
        uce_model = UCE(configurer=config)

        data_for_uce = uce_model.process_data(adata)
        embeddings = uce_model.get_embeddings(data_for_uce)

        adata.obsm["X_uce"] = embeddings
        return adata

    elif model_lower == "transcriptformer":
        from helical.models.transcriptformer.model import TranscriptFormer
        from helical.models.transcriptformer.transcriptformer_config import TranscriptFormerConfig

        config = TranscriptFormerConfig(
            batch_size=batch_size,
            **kwargs,
        )
        tf_model = TranscriptFormer(configurer=config)

        data_for_tf = tf_model.process_data([adata])
        embeddings = tf_model.get_embeddings(data_for_tf)

        adata.obsm["X_transcriptformer"] = _to_numpy(embeddings).astype("float32", copy=False)
        return adata

    else:
        raise ValueError(
            f"Unrecognized model '{model}'. Please choose one of: 'scgpt', 'geneformer', 'uce', or 'transcriptformer'."
        )
