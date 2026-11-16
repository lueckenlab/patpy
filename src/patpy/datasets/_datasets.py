from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import scanpy as sc
from anndata import AnnData

from patpy._settings import settings
from patpy.datasets._dataloader import _download

DatasetKind = Literal["raw", "processed"]
InflammationAtlasSplit = Literal["main", "external", "validation"]


@dataclass(frozen=True)
class DatasetInfo:
    """Standard schema describing a patpy dataset.

    Bundled alongside the :class:`~anndata.AnnData` returned by the dataset
    loaders so downstream code (and tests) does not have to copy-paste this
    information from elsewhere.

    Attributes
    ----------
    n_samples
        Number of unique samples (donors / patients) in the dataset.
    n_cells
        Total number of cells (``adata.n_obs``).
    n_features
        Total number of features / genes (``adata.n_vars``).
    sample_key
        Column in ``adata.obs`` identifying the biological sample.
    cell_type_key
        Column in ``adata.obs`` containing cell-type annotations to use by default.
    sample_metadata_columns
        Columns in ``adata.obs`` that are constant within a sample and capture
        sample-level (donor-level) metadata such as condition, outcome, or batch.
    """

    n_samples: int
    n_cells: int
    n_features: int
    sample_key: str
    cell_type_key: str
    sample_metadata_columns: list[str] = field(default_factory=list)


_COMBAT_INFO = DatasetInfo(
    n_samples=138,
    n_cells=783677,
    n_features=3000,
    sample_key="scRNASeq_sample_ID",
    cell_type_key="Annotation_major_subset",
    sample_metadata_columns=["Source", "Outcome", "Death28", "Institute", "Pool_ID"],
)

_HLCA_INFO = DatasetInfo(
    n_samples=339,
    n_cells=1687127,
    n_features=3000,
    sample_key="donor_id",
    cell_type_key="cell_type",
    sample_metadata_columns=[
        "suspension_type",
        "BMI",
        "age_or_mean_of_age_range",
        "age_range",
        "anatomical_region_ccf_score",
        "cause_of_death",
        "core_or_extension",
        "fresh_or_frozen",
        "lung_condition",
        "sequencing_platform",
        "smoking_status",
        "subject_type",
        "assay",
        "disease",
        "sex",
        "tissue",
        "self_reported_ethnicity",
        "development_stage",
    ],
)

_ONEK1K_INFO = DatasetInfo(
    n_samples=981,
    n_cells=1248980,
    n_features=3000,
    sample_key="donor_id",
    cell_type_key="cell_type",
    sample_metadata_columns=["pool_number", "age", "sex"],
)

_STEPHENSON_INFO = DatasetInfo(
    n_samples=131,
    n_cells=639482,
    n_features=3000,
    sample_key="sample_id",
    cell_type_key="cell_type",
    sample_metadata_columns=[
        "Resample",
        "Collection_Day",
        "Swab_result",
        "Status",
        "Smoker",
        "Status_on_day_collection",
        "Status_on_day_collection_summary",
        "Days_from_onset",
        "Site",
        "Worst_Clinical_Status",
        "Outcome",
        "disease",
        "sex",
        "development_stage",
    ],
)

_TICATLAS_INFO = DatasetInfo(
    n_samples=123,
    n_cells=267547,
    n_features=3000,
    sample_key="patient",
    cell_type_key="lv1_annot",
    sample_metadata_columns=["patient", "gender", "subtype", "source"],
)

_COMBAT_STEPHENSON_INFO = DatasetInfo(
    n_samples=251,
    n_cells=1399435,
    n_features=1856,
    sample_key="sample_id",
    cell_type_key="cell_coarse_aligned",
    sample_metadata_columns=[
        "dataset",
        "Site",
        "SARSCoV2PCR",
        "Outcome",
        "TimeSinceOnset",
        "split",
        "Status",
        "severity_ord",
        "severity_idx",
        "outcome_bin",
        "Status_encoded",
    ],
)

# Per-split metadata for the inflammation atlas. All three splits are
# latents-only (n_features=0; the 30-dim scANVI embedding lives in
# obsm['X_scANVI_atlas']). Main carries ground-truth Level1/Level2 annotations;
# external and validation carry model-predicted Level1pred/Level2pred labels.
_INFLAMMATION_ATLAS_SAMPLE_METADATA_COLUMNS = [
    "studyID",
    "chemistry",
    "technology",
    "disease",
    "sex",
    "binned_age",
    "disease_group",
    "therapy_response",
    "smoking_status",
]

_INFLAMMATION_ATLAS_INFOS: dict[InflammationAtlasSplit, DatasetInfo] = {
    "main": DatasetInfo(
        n_samples=817,
        n_cells=4918140,
        n_features=0,
        sample_key="sampleID",
        cell_type_key="Level1",
        sample_metadata_columns=_INFLAMMATION_ATLAS_SAMPLE_METADATA_COLUMNS,
    ),
    "external": DatasetInfo(
        n_samples=86,
        n_cells=572872,
        n_features=0,
        sample_key="sampleID",
        cell_type_key="Level1pred",
        sample_metadata_columns=_INFLAMMATION_ATLAS_SAMPLE_METADATA_COLUMNS,
    ),
    "validation": DatasetInfo(
        n_samples=144,
        n_cells=849922,
        n_features=0,
        sample_key="sampleID",
        cell_type_key="Level1pred",
        sample_metadata_columns=_INFLAMMATION_ATLAS_SAMPLE_METADATA_COLUMNS,
    ),
}

# Registry of Figshare URLs for each dataset / kind. ``None`` means "not yet
# uploaded"; the loader will raise NotImplementedError for those combinations.
_DATASET_URLS: dict[str, dict[DatasetKind, str | None]] = {
    "combat": {
        "processed": "https://ndownloader.figshare.com/files/64217586",
        "raw": None,
    },
    "hlca": {
        "processed": "https://ndownloader.figshare.com/files/64225983",
        "raw": None,
    },
    "onek1k": {
        "processed": "https://ndownloader.figshare.com/files/64225884",
        "raw": None,
    },
    "stephenson": {
        "processed": "https://ndownloader.figshare.com/files/64226109",
        "raw": None,
    },
    "ticatlas": {
        "processed": "https://ndownloader.figshare.com/files/64226097",
        "raw": None,
    },
    "combat_stephenson": {
        "processed": "https://ndownloader.figshare.com/files/64486770",
        "raw": None,
    },
    "inflammation_atlas_main": {
        "processed": "https://ndownloader.figshare.com/files/64486761",
        "raw": None,
    },
    "inflammation_atlas_external": {
        "processed": "https://ndownloader.figshare.com/files/64486752",
        "raw": None,
    },
    "inflammation_atlas_validation": {
        "processed": "https://ndownloader.figshare.com/files/64486758",
        "raw": None,
    },
}

# Zipped sample-metadata AnnData for the COMBAT cohort (~4 MB).
_COMBAT_META_URL = "https://ndownloader.figshare.com/files/64291092"
_COMBAT_META_FILENAME = "combat_meta_adata.h5ad"


def _load_dataset(url: str, output_file_name: str, overwrite: bool) -> AnnData:
    """Shared download / cache / load logic.

    Downloads ``url`` as a zip, extracts it into ``settings.datasetdir``, and
    reads the resulting ``output_file_name`` (which must be the name of the
    ``.h5ad`` file inside the zip) as an :class:`~anndata.AnnData`.
    """
    output_file_path = settings.datasetdir / output_file_name
    if not Path(output_file_path).exists() or overwrite:
        zip_name = f"{output_file_name}.zip"
        _download(
            url=url,
            output_file_name=zip_name,
            output_path=settings.datasetdir,
            is_zip=True,
        )
        (settings.datasetdir / zip_name).unlink(missing_ok=True)
    return sc.read_h5ad(output_file_path)


def _resolve_dataset_url(name: str, kind: DatasetKind) -> str:
    """Return the URL for ``name``/``kind`` or raise a clear error."""
    if kind not in ("raw", "processed"):
        raise ValueError(f"kind must be 'raw' or 'processed', got {kind!r}.")
    url = _DATASET_URLS[name][kind]
    if url is None:
        raise NotImplementedError(f"The {kind!r} version of the {name!r} dataset is not available yet.")
    return url


def _load_named_dataset(name: str, kind: DatasetKind, overwrite: bool) -> AnnData:
    """Resolve the URL for ``name``/``kind`` and download the dataset."""
    url = _resolve_dataset_url(name, kind)
    return _load_dataset(
        url=url,
        output_file_name=f"{name}_{kind}.h5ad",
        overwrite=overwrite,
    )


def _load_combat_metadata(overwrite: bool) -> AnnData:
    """Download and return the COMBAT sample-metadata AnnData (~4 MB zipped)."""
    return _load_dataset(
        url=_COMBAT_META_URL,
        output_file_name=_COMBAT_META_FILENAME,
        overwrite=overwrite,
    )


def combat(
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    load_metadata: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, ...]:
    """COvid-19 Multi-omics Blood ATlas (COMBAT) dataset.

    The processed version was prepared with the standard scanpy pipeline; cells
    annotated as "nan" were removed; PCA, scVI, scANVI, and scPoli dimensionality
    reduction were applied. The dataset contains 783,677 cells and 3,000
    features. The processed download is approximately 1.5 GB compressed and
    ~5 GB unzipped.

    Parameters
    ----------
    kind
        Either ``"processed"`` (default) or ``"raw"``. Currently only
        ``"processed"`` is available; ``"raw"`` raises :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    load_metadata
        If ``True``, also download the ~4 MB sample-metadata
        :class:`~anndata.AnnData` and return it as an extra element.
    return_dataset_info
        If ``True``, append a :class:`DatasetInfo` describing the dataset's
        standard schema (sample / cell-type keys, cell counts, etc.) to the
        return value.

    Returns
    -------
        By default the :class:`~anndata.AnnData` object alone. When
        ``load_metadata`` and/or ``return_dataset_info`` are set, a tuple is
        returned in this fixed order: ``(adata, meta_adata, info)`` (omitting
        any element that was not requested).

    References
    ----------
        Ahern, D. J., Ai, Z., Ainsworth, M., Allan, C., Allcock, A., Angus, B., ... & Salio, M. (2022). A blood atlas of COVID-19 defines hallmarks of disease severity and specificity. Cell, 185(5), 916-938. https://doi.org/10.1016/j.cell.2022.01.012.
        COvid-19 Multi-omics Blood ATlas (COMBAT) Consortium. (2021). A blood atlas of COVID-19 defines hallmarks of disease severity and specificity: Associated data (1.0.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6120249

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.combat()
        >>> adata, meta_adata = patpy.datasets.combat(load_metadata=True)
        >>> adata, info = patpy.datasets.combat(return_dataset_info=True)
        >>> adata, meta_adata, info = patpy.datasets.combat(load_metadata=True, return_dataset_info=True)
    """
    adata = _load_named_dataset("combat", kind=kind, overwrite=overwrite)
    extras: list[AnnData | DatasetInfo] = []
    if load_metadata:
        extras.append(_load_combat_metadata(overwrite=overwrite))
    if return_dataset_info:
        extras.append(_COMBAT_INFO)
    if not extras:
        return adata
    return (adata, *extras)


def hlca(
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, DatasetInfo]:
    """Human Lung Cell Atlas (HLCA) dataset.

    The processed version was prepared with the standard scanpy pipeline; cells
    annotated as "nan" were removed; PCA, scVI, scANVI, and scPoli dimensionality
    reduction were applied. The dataset contains 1,687,127 cells and 3,000
    features. The processed download is approximately 3 GB compressed and
    ~6.5 GB unzipped.

    Parameters
    ----------
    kind
        Either ``"processed"`` (default) or ``"raw"``. Currently only
        ``"processed"`` is available; ``"raw"`` raises :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    return_dataset_info
        If ``True``, return a tuple ``(adata, DatasetInfo)`` instead of just ``adata``.

    References
    ----------
        Sikkema, L., Ramírez-Suástegui, C., Strobl, D. C., Gillett, T. E., Zappia, L., Madissoon, E., ... & Theis, F. J. (2023). An integrated cell atlas of the lung in health and disease. Nature medicine, 29(6), 1563-1577. https://doi.org/10.1038/s41591-023-02327-2

    Returns
    -------
        :class:`~anndata.AnnData` object of scRNA-seq profiles, optionally paired
        with a :class:`DatasetInfo` describing the dataset's standard schema.

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.hlca()
        >>> adata, info = patpy.datasets.hlca(return_dataset_info=True)
    """
    adata = _load_named_dataset("hlca", kind=kind, overwrite=overwrite)
    if return_dataset_info:
        return adata, _HLCA_INFO
    return adata


def onek1k(
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, DatasetInfo]:
    """OneK1K dataset.

    The processed version was prepared with the standard scanpy pipeline; cells
    annotated as "nan" were removed; PCA, scVI, scANVI, and scPoli dimensionality
    reduction were applied. The dataset contains 1,248,980 cells and 3,000
    features. The processed download is approximately 2.5 GB compressed and
    ~4 GB unzipped.

    Parameters
    ----------
    kind
        Either ``"processed"`` (default) or ``"raw"``. Currently only
        ``"processed"`` is available; ``"raw"`` raises :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    return_dataset_info
        If ``True``, return a tuple ``(adata, DatasetInfo)`` instead of just ``adata``.

    References
    ----------
        Yazar, S., Alquicira-Hernandez, J., Wing, K., Senabouth, A., Gordon, M. G., Andersen, S., ... & Powell, J. E. (2022). Single-cell eQTL mapping identifies cell type–specific genetic control of autoimmune disease. Science, 376(6589), eabf3041. https://doi.org/10.1126/science.abf3041
        https://onek1k.org/

    Returns
    -------
        :class:`~anndata.AnnData` object of scRNA-seq profiles, optionally paired
        with a :class:`DatasetInfo` describing the dataset's standard schema.

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.onek1k()
        >>> adata, info = patpy.datasets.onek1k(return_dataset_info=True)
    """
    adata = _load_named_dataset("onek1k", kind=kind, overwrite=overwrite)
    if return_dataset_info:
        return adata, _ONEK1K_INFO
    return adata


def stephenson(
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, DatasetInfo]:
    """Multi-omics immune response in COVID-19 (Stephenson) dataset.

    The processed version was prepared with the standard scanpy pipeline; cells
    annotated as "nan" were removed; PCA, scVI, scANVI, and scPoli dimensionality
    reduction were applied. The dataset contains 639,482 cells and 3,000
    features. The processed download is approximately 1.5 GB compressed and
    ~4.5 GB unzipped.

    Parameters
    ----------
    kind
        Either ``"processed"`` (default) or ``"raw"``. Currently only
        ``"processed"`` is available; ``"raw"`` raises :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    return_dataset_info
        If ``True``, return a tuple ``(adata, DatasetInfo)`` instead of just ``adata``.

    References
    ----------
        Stephenson, E., Reynolds, G., Botting, R. A., Calero-Nieto, F. J., Morgan, M. D., Tuong, Z. K., ... & Haniffa, M. (2021). Single-cell multi-omics analysis of the immune response in COVID-19. Nature medicine, 27(5), 904-916. https://doi.org/10.1038/s41591-021-01329-2

    Returns
    -------
        :class:`~anndata.AnnData` object of scRNA-seq profiles, optionally paired
        with a :class:`DatasetInfo` describing the dataset's standard schema.

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.stephenson()
        >>> adata, info = patpy.datasets.stephenson(return_dataset_info=True)
    """
    adata = _load_named_dataset("stephenson", kind=kind, overwrite=overwrite)
    if return_dataset_info:
        return adata, _STEPHENSON_INFO
    return adata


def ticatlas(
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, DatasetInfo]:
    """Tumor Immune Cell Atlas (TICAtlas) dataset.

    The processed version was prepared with the standard scanpy pipeline; cells
    annotated as "nan" were removed; PCA, scVI, scANVI, and scPoli dimensionality
    reduction were applied. The dataset contains 267,547 cells and 3,000
    features. The processed download is approximately 0.5 GB compressed and
    ~1.8 GB unzipped.

    Parameters
    ----------
    kind
        Either ``"processed"`` (default) or ``"raw"``. Currently only
        ``"processed"`` is available; ``"raw"`` raises :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    return_dataset_info
        If ``True``, return a tuple ``(adata, DatasetInfo)`` instead of just ``adata``.

    References
    ----------
        Nieto, P., Elosua-Bayes, M., Trincado, J. L., Marchese, D., Massoni-Badosa, R., Salvany, M., ... & Heyn, H. (2021). A single-cell tumor immune atlas for precision oncology. Genome research, 31(10), 1913-1926. https://doi.org/10.1101/gr.273300.120

    Returns
    -------
        :class:`~anndata.AnnData` object of scRNA-seq profiles, optionally paired
        with a :class:`DatasetInfo` describing the dataset's standard schema.

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.ticatlas()
        >>> adata, info = patpy.datasets.ticatlas(return_dataset_info=True)
    """
    adata = _load_named_dataset("ticatlas", kind=kind, overwrite=overwrite)
    if return_dataset_info:
        return adata, _TICATLAS_INFO
    return adata


def combat_stephenson(
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, DatasetInfo]:
    """Harmonized COMBAT + Stephenson COVID-19 PBMC dataset.

    Cross-study harmonization of the COMBAT and Stephenson COVID-19 PBMC cohorts
    sharing a unified cell-type annotation (``cell_coarse_aligned``) and joint
    PCA/scVI/scANVI embeddings (in ``adata.obsm``). The published file
    contains 1,399,435 cells across 251 samples (2 source datasets), with 1,856
    genes retained after harmonization. Raw counts are kept under
    ``adata.layers['X_raw_counts']``. Sample-level metadata (Site, Status,
    severity, outcome, etc.) is available in ``adata.obs`` directly.

    Parameters
    ----------
    kind
        Either ``"processed"`` (default) or ``"raw"``. The processed dataset is
        available via Figshare; ``"raw"`` currently raises
        :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    return_dataset_info
        If ``True``, return a tuple ``(adata, DatasetInfo)`` instead of just ``adata``.

    References
    ----------
        Ahern, D. J., et al. (2022). A blood atlas of COVID-19 defines hallmarks of disease severity and specificity. Cell, 185(5), 916-938. https://doi.org/10.1016/j.cell.2022.01.012
        Stephenson, E., et al. (2021). Single-cell multi-omics analysis of the immune response in COVID-19. Nature medicine, 27(5), 904-916. https://doi.org/10.1038/s41591-021-01329-2

    Returns
    -------
        :class:`~anndata.AnnData` object of scRNA-seq profiles, optionally paired
        with a :class:`DatasetInfo` describing the dataset's standard schema.

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.combat_stephenson()
        >>> adata, info = patpy.datasets.combat_stephenson(return_dataset_info=True)
    """
    adata = _load_named_dataset("combat_stephenson", kind=kind, overwrite=overwrite)
    if return_dataset_info:
        return adata, _COMBAT_STEPHENSON_INFO
    return adata


def inflammation_atlas(
    split: InflammationAtlasSplit = "main",
    kind: DatasetKind = "processed",
    overwrite: bool = False,
    return_dataset_info: bool = False,
) -> AnnData | tuple[AnnData, DatasetInfo]:
    """Cross-study inflammation atlas (scANVI latents).

    The published "processed" version is a scANVI-embedded latents-only
    :class:`~anndata.AnnData` with the 30-dimensional scANVI embedding under
    ``adata.obsm['X_scANVI_atlas']``. The file has no gene-expression matrix
    for now; use the embedding directly for
    sample-representation workflows. Sample-level metadata (``studyID``,
    ``chemistry``, ``disease``, ``sex``, ``binned_age``) is available in
    ``adata.obs`` directly.

    Three splits are exposed via the ``split`` argument:

    * ``"main"`` (default) — atlas training data, 4,918,140 cells / 817 samples,
      with ground-truth ``Level1`` / ``Level2`` cell-type annotations.
    * ``"external"`` — held-out studies, 572,872 cells /
      86 samples, with model-predicted ``Level1pred`` / ``Level2pred`` labels.
    * ``"validation"`` — held-out samples, 849,922 cells /
      144 samples, with model-predicted ``Level1pred`` / ``Level2pred`` labels.

    Parameters
    ----------
    split
        Which split to load: ``"main"``, ``"external"``, or ``"validation"``.
        Defaults to ``"main"``.
    kind
        Either ``"processed"`` (default) or ``"raw"``. The processed splits are
        available via Figshare; ``"raw"`` currently raises
        :class:`NotImplementedError`.
    overwrite
        If ``True``, re-download the dataset even when a cached copy exists.
    return_dataset_info
        If ``True``, return a tuple ``(adata, DatasetInfo)`` instead of just
        ``adata``. The :class:`DatasetInfo` is split-specific: ``cell_type_key``
        is ``"Level1"`` for ``main`` and ``"Level1pred"`` for the other two
        splits. ``n_features`` is reported as ``0`` because the file is
        latents-only.


    References
    ----------
       Jiménez-Gracia, L., Maspero, D., Aguilar-Fernández, S. et al. Interpretable inflammation landscape of circulating immune cells. Nat Med 32, 633–644 (2026). https://doi.org/10.1038/s41591-025-04126-3


    Returns
    -------
        :class:`~anndata.AnnData` object with scANVI latents under
        ``obsm['X_scANVI_atlas']``, optionally paired with a :class:`DatasetInfo`
        describing the requested split's standard schema.

    Examples
    --------
        >>> import patpy
        >>> adata = patpy.datasets.inflammation_atlas()
        >>> latents = adata.obsm["X_scANVI_atlas"]
        >>> adata, info = patpy.datasets.inflammation_atlas(return_dataset_info=True)
        >>> ext = patpy.datasets.inflammation_atlas(split="external")
        >>> val = patpy.datasets.inflammation_atlas(split="validation")
    """
    if split not in _INFLAMMATION_ATLAS_INFOS:
        raise ValueError(f"split must be one of {tuple(_INFLAMMATION_ATLAS_INFOS)}, got {split!r}.")
    adata = _load_named_dataset(f"inflammation_atlas_{split}", kind=kind, overwrite=overwrite)
    if return_dataset_info:
        return adata, _INFLAMMATION_ATLAS_INFOS[split]
    return adata
