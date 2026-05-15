from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def build_condition_combinations(
    adata: ad.AnnData,
    condition_cols: list[str],
    sep: str = "_",
) -> pd.DataFrame:
    """Build a DataFrame of all observed combinations of condition columns.

    Returns one row per unique observed combination, not the Cartesian
    product of all levels.

    Parameters
    ----------
    adata : AnnData
        Annotated data object whose ``.obs`` contains the condition columns.
    condition_cols : list[str]
        Column names in ``adata.obs`` to combine.
    sep : str, default ``"_"``
        Separator used when joining levels into a combined label.

    Returns
    -------
    pd.DataFrame
        Columns ``condition_cols`` plus a ``"label"`` column containing the
        joined label for each combination.

    Examples
    --------
    >>> combos = build_condition_combinations(adata, ["timepoint", "disease_subtype"])
    >>> combos
       timepoint disease_subtype          label
    0        T1               A           T1_A
    1        T1               B           T1_B
    2        T2               A           T2_A
    """
    if not condition_cols:
        raise ValueError("condition_cols must contain at least one column name.")
    for col in condition_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs.")
    combos = adata.obs[condition_cols].drop_duplicates().reset_index(drop=True)
    combos["label"] = combos[condition_cols].astype(str).agg(sep.join, axis=1)
    return combos


def build_all_pairwise_contrasts(
    adata: ad.AnnData,
    condition_cols: list[str],
    sep: str = "_",
) -> list[dict]:
    """Generate all pairwise contrasts across observed condition combinations.

    Parameters
    ----------
    adata : AnnData
        Annotated data object.
    condition_cols : list[str]
        Columns in ``adata.obs`` defining the conditions to combine.
    sep : str, default ``"_"``
        Separator for joining condition levels.

    Returns
    -------
    list[dict]
        List of contrast dicts, each with keys ``"group"``, ``"baseline"``,
        and ``"label"``.

    Examples
    --------
    >>> contrasts = build_all_pairwise_contrasts(adata, ["timepoint", "disease_subtype"])
    >>> contrasts[0]
    {'group': 'T1_A', 'baseline': 'T1_B', 'label': 'T1_A_vs_T1_B'}
    """
    combos = build_condition_combinations(adata, condition_cols, sep=sep)
    labels = combos["label"].tolist()
    contrasts = []
    for i, group in enumerate(labels):
        for baseline in labels[i + 1 :]:
            contrasts.append({"group": group, "baseline": baseline, "label": f"{group}_vs_{baseline}"})
    return contrasts


def filter_adata_to_conditions(
    adata: ad.AnnData,
    condition_col: str,
    groups: list[str],
) -> ad.AnnData:
    """Subset ``adata`` to cells belonging to specific condition groups.

    Parameters
    ----------
    adata : AnnData
    condition_col : str
        Column in ``adata.obs`` to filter on.
    groups : list[str]
        Values in ``condition_col`` to keep.

    Returns
    -------
    AnnData
        Subset of ``adata`` containing only cells in the specified groups.
    """
    mask = adata.obs[condition_col].isin(groups)
    return adata[mask].copy()
