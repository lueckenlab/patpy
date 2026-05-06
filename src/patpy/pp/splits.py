"""Sample-level train/validation/test splitting for MIL benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc


def make_sample_splits(
    adata: sc.AnnData,
    sample_key: str,
    label_key: str,
    *,
    test_size: float = 0.2,
    val_size: float = 0.15,
    covariate_keys: list[str] | None = None,
    n_splits: int = 1,
    seed: int = 42,
    split_col: str = "split",
) -> sc.AnnData:
    """Assign train/val/test partition labels to samples in ``adata.obs``.

    Splitting is at the **sample** level — all cells from a given sample land
    in the same partition.  The primary label (``label_key``) drives
    stratification; optional ``covariate_keys`` are folded into a composite
    key so that class balance *and* covariate distributions are preserved
    across partitions.

    Parameters
    ----------
    adata
        Single-cell AnnData.  Must contain ``sample_key`` and ``label_key``
        in ``.obs``.
    sample_key
        Column in ``adata.obs`` with sample / donor identifiers.
    label_key
        Primary column to stratify by (e.g. ``"disease"`` or ``"age_bin"``).
        Continuous values are automatically quartile-binned.
    test_size
        Fraction of samples held out as the test set.
    val_size
        Fraction of *all* samples held out for validation (computed before the
        test split, so the effective fraction of remaining samples is
        ``val_size / (1 - test_size)``).
    covariate_keys
        Additional ``adata.obs`` columns to include in the composite
        stratification key (e.g. ``["sex", "study"]``).
    n_splits
        Number of independent train/val/test splits.  When > 1, columns
        ``"{split_col}_0"``, ``"{split_col}_1"``, … are written.
    seed
        Base random seed; split ``i`` uses ``seed + i``.
    split_col
        Base name for the output column(s) in ``adata.obs``.

    Returns
    -------
    sc.AnnData
        The input ``adata`` with split column(s) added to ``.obs`` in-place.

    Examples
    --------
    >>> adata = make_sample_splits(adata, sample_key="donor_id", label_key="disease")
    >>> train = adata[adata.obs["split"] == "train"]
    >>> test  = adata[adata.obs["split"] == "test"]
    """
    if sample_key not in adata.obs.columns:
        raise ValueError(f"sample_key='{sample_key}' not found in adata.obs.")
    if label_key not in adata.obs.columns:
        raise ValueError(f"label_key='{label_key}' not found in adata.obs.")

    extra_cols = covariate_keys or []
    for col in extra_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"covariate_keys entry '{col}' not found in adata.obs.")

    # One row per sample
    meta = adata.obs[[sample_key, label_key, *extra_cols]].copy()
    meta = meta.groupby(sample_key, sort=False).first().reset_index()

    strat = _composite_strat_key(meta, label_key, extra_cols)

    col_names = (
        [f"{split_col}_{i}" for i in range(n_splits)] if n_splits > 1 else [split_col]
    )

    for i, col in enumerate(col_names):
        assignment = _one_split(meta[sample_key].values, strat.values, test_size, val_size, seed + i)
        adata.obs[col] = adata.obs[sample_key].map(assignment).values

    return adata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _composite_strat_key(
    meta: pd.DataFrame,
    label_key: str,
    covariate_keys: list[str],
) -> pd.Series:
    """Return a composite string column for stratified splitting."""

    def _discretize(s: pd.Series) -> pd.Series:
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            return pd.qcut(s, q=4, labels=False, duplicates="drop").astype(str)
        return s.astype(str)

    parts = [_discretize(meta[label_key])]
    for cov in covariate_keys:
        parts.append(_discretize(meta[cov]))

    if len(parts) == 1:
        return parts[0]
    return parts[0].str.cat(parts[1:], sep="|")


def _one_split(
    samples: np.ndarray,
    strat: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, str]:
    """Return {sample_id: 'train'|'val'|'test'} for one random split.

    Falls back to a non-stratified shuffle when there are too few samples to
    satisfy the stratification constraint.
    """
    from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

    rng = np.random.default_rng(seed)

    def _split(X, y, frac):
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
            a, b = next(sss.split(X, y))
        except ValueError:
            ss = ShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
            a, b = next(ss.split(X))
        return a, b

    # Step 1 — carve out test set
    train_val_idx, test_idx = _split(samples, strat, test_size)

    # Step 2 — carve out val from the remainder
    adjusted_val = min(val_size / (1.0 - test_size), 0.9)
    tv_samples = samples[train_val_idx]
    tv_strat = strat[train_val_idx]
    train_local, val_local = _split(tv_samples, tv_strat, adjusted_val)

    assignment: dict[str, str] = {}
    for idx in test_idx:
        assignment[samples[idx]] = "test"
    for local in train_local:
        assignment[tv_samples[local]] = "train"
    for local in val_local:
        assignment[tv_samples[local]] = "val"

    return assignment