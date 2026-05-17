"""Load + clean the aging / OneK1K AnnData for the benchmark.

Three transformations are common across the methods and are factored out here:

1. Read the dataset (local h5ad for the AIFI aging cohort, ``patpy.datasets``
   loader for OneK1K).
2. Clean the age column. The aging cohort stores age as a category with
   ``"89+"`` for donors aged ≥89; we map ``89+ → 89`` so the column is
   numeric and ``ranking``/``regression`` evaluations can use it.
3. Cap cells per donor (only for the slow methods) using a deterministic
   subsample tied to ``seed``. Returns a *view* of the original AnnData.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import patpy

from configs import DatasetConfig, AGING_H5AD


def _read_aging() -> ad.AnnData:
    return ad.read_h5ad(AGING_H5AD)


def _read_aging_backed() -> ad.AnnData:
    return ad.read_h5ad(AGING_H5AD, backed="r")


def _read_onek1k() -> ad.AnnData:
    a, _ = patpy.datasets.onek1k(return_dataset_info=True)
    return a


def load_dataset(cfg: DatasetConfig, backed: bool = False) -> ad.AnnData:
    if cfg.loader_kind == "local_h5ad":
        adata = _read_aging_backed() if backed else _read_aging()
    elif cfg.loader_kind == "patpy":
        loader = getattr(patpy.datasets, cfg.loader_arg)
        adata, _ = loader(return_dataset_info=True)
    else:
        raise ValueError(f"unknown loader_kind {cfg.loader_kind!r}")
    return _add_clean_age(adata, cfg)


def _add_clean_age(adata: ad.AnnData, cfg: DatasetConfig, old_threshold: int = 65) -> ad.AnnData:
    """Add ``age`` (numeric, ``89+ → 89``) + ``age_group`` (``"old"`` if age≥threshold).

    The binary group is used by the supervised methods (PaSCient / MixMIL) since
    patpy's wrappers around them currently route a ``"regression"`` task through
    a categorical / binomial head that crashes on continuous ages. The notebook
    still scores everyone on continuous age via KNN on the donor embedding.
    """
    raw = adata.obs[cfg.age_col]
    if pd.api.types.is_numeric_dtype(raw):
        adata.obs["age"] = raw.astype(np.float64)
    else:
        cleaned = raw.astype(str).str.replace("+", "", regex=False)
        adata.obs["age"] = pd.to_numeric(cleaned, errors="coerce").astype(np.float64)
    adata.obs["age_group"] = np.where(adata.obs["age"] >= old_threshold, "old", "young")
    return adata


def subsample_cells_per_donor(
    adata: ad.AnnData,
    sample_key: str,
    max_cells: int | None,
    seed: int = 0,
) -> ad.AnnData:
    """Cap each donor at ``max_cells`` (random with given seed)."""
    if max_cells is None:
        return adata
    rng = np.random.default_rng(seed)
    keep_idx: list[np.ndarray] = []
    obs_index = np.arange(adata.n_obs)
    grouped = pd.Series(adata.obs[sample_key].values).groupby(adata.obs[sample_key].values, observed=True).indices
    for _donor, idx in grouped.items():
        if len(idx) <= max_cells:
            keep_idx.append(obs_index[idx])
        else:
            chosen = rng.choice(idx, size=max_cells, replace=False)
            keep_idx.append(obs_index[chosen])
    keep = np.sort(np.concatenate(keep_idx))
    return adata[keep].copy()


def smoke_subset(
    adata: ad.AnnData,
    sample_key: str,
    n_donors: int = 20,
    max_cells_per_donor: int = 200,
    seed: int = 0,
) -> ad.AnnData:
    """Pick ``n_donors`` random donors, each capped at ``max_cells_per_donor``."""
    rng = np.random.default_rng(seed)
    donors = pd.Series(adata.obs[sample_key].astype(str).values).unique()
    donors = donors[rng.permutation(len(donors))[:n_donors]]
    keep = adata.obs[sample_key].astype(str).isin(donors).values
    sub = adata[keep].copy()
    return subsample_cells_per_donor(sub, sample_key, max_cells_per_donor, seed=seed)


def smoke_load(cfg, n_donors: int = 20, max_cells_per_donor: int = 200, seed: int = 0):
    """Fast smoke loader: backed read + slice-before-materialise for the aging cohort.

    Avoids materialising the full 5 GB AnnData when we only need a few thousand
    cells. For the OneK1K loader the patpy helper already eagerly reads — we
    accept that cost and subset afterwards.
    """
    if cfg.loader_kind == "local_h5ad":
        a = _read_aging_backed()
        a = _add_clean_age(a, cfg)
        rng = np.random.default_rng(seed)
        donors = pd.unique(a.obs[cfg.sample_key].astype(str))
        donors = donors[rng.permutation(len(donors))[:n_donors]]
        donor_set = set(donors.tolist())
        cell_mask = a.obs[cfg.sample_key].astype(str).isin(donor_set).values
        sub = a[cell_mask].to_memory()
        a.file.close()
        return subsample_cells_per_donor(sub, cfg.sample_key, max_cells_per_donor, seed=seed)
    return smoke_subset(load_dataset(cfg), cfg.sample_key, n_donors, max_cells_per_donor, seed)
