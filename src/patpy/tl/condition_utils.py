from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def build_condition_combinations(
    adata: ad.AnnData,
    condition_cols: list[str],
    sep: str = "_",
) -> pd.DataFrame:
    """Build a DataFrame of all observed combinations of condition columns.

    Returns one row per unique observed combination, not the Cartesian  product of all levels.

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
       timepoint disease_subtype  label
    0        T1               A   T1_A
    1        T1               B   T1_B
    2        T2               A   T2_A
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
            contrasts.append(
                {
                    "group": group,
                    "baseline": baseline,
                    "label": f"{group}_vs_{baseline}",
                }
            )
    return contrasts


def add_combined_condition_column(
    adata: ad.AnnData,
    condition_cols: list[str],
    new_col: str = "condition_combined",
    sep: str = "_",
) -> ad.AnnData:
    """Add a concatenated condition label column to ``adata.obs`` in-place.

    Parameters
    ----------
    adata : AnnData
        Annotated data object (modified in place).
    condition_cols : list[str]
        Columns in ``adata.obs`` to combine.
    new_col : str, default ``"condition_combined"``
        Name of the new column.
    sep : str, default ``"_"``
        Separator used when joining condition levels.

    Returns
    -------
    AnnData
        The input ``adata`` with the new column added.

    Examples
    --------
    >>> adata = add_combined_condition_column(adata, ["timepoint", "disease_subtype"], new_col="tp_disease")
    >>> adata.obs["tp_disease"].unique()
    array(['T1_A', 'T1_B', 'T2_A', 'T2_B'], dtype=object)
    """
    for col in condition_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Column '{col}' not found in adata.obs.")
    adata.obs[new_col] = adata.obs[condition_cols].astype(str).agg(sep.join, axis=1)
    return adata


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


def _iter_contrasts(
    model_cls: Any,
    adata: ad.AnnData,
    condition_cols: list[str],
    *,
    combined_col: str,
    sep: str,
    subset_contrasts: list[dict] | None,
    kwargs: dict,
) -> pd.DataFrame:
    """Internal: run model_cls.compare_groups for every pairwise contrast."""
    adata = add_combined_condition_column(adata, condition_cols, new_col=combined_col, sep=sep)
    all_contrasts = subset_contrasts or build_all_pairwise_contrasts(adata, condition_cols, sep=sep)

    results = []
    for spec in all_contrasts:
        group = spec["group"]
        baseline = spec["baseline"]
        label = spec["label"]

        sub = filter_adata_to_conditions(adata, combined_col, [group, baseline])
        if sub.n_obs == 0:
            warnings.warn(
                f"No cells found for contrast '{label}'; skipping.",
                UserWarning,
                stacklevel=3,
            )
            continue

        try:
            res = model_cls.compare_groups(
                sub,
                column=combined_col,
                baseline=baseline,
                groups_to_compare=group,
                **kwargs,
            )
            res["contrast"] = label
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Contrast '{label}' failed: {exc}",
                UserWarning,
                stacklevel=3,
            )

    if not results:
        raise RuntimeError(
            "All contrasts failed or returned no results. Check your data, design, and condition columns."
        )

    return pd.concat(results, ignore_index=True)


def run_condition_combinations(
    model_cls: Any,
    adata: ad.AnnData,
    condition_cols: list[str],
    *,
    combined_col: str = "condition_combined",
    sep: str = "_",
    subset_contrasts: list[dict] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Run any pertpy differential method across all pairwise condition contrasts.

    Given multiple overlapping condition axes (e.g. ``["timepoint",
    "disease_subtype"]``), this function:

    1. Creates a compound label column (e.g. ``"T1_CancerA"``) in
       ``adata.obs``.
    2. Enumerates all *observed* pairwise contrasts.
    3. Calls ``model_cls.compare_groups(sub, column=combined_col, ...)`` for
       each pair.
    4. Concatenates all results, adding a ``"contrast"`` column.

    ``model_cls`` can be **any pertpy class** that exposes a
    ``compare_groups(adata, column, baseline, groups_to_compare, **kwargs)``
    classmethod — currently ``PyDESeq2``, ``EdgeR``, and ``Milo``.

    Parameters
    ----------
    model_cls : pertpy class
        Any pertpy differential method class with a ``compare_groups``
        classmethod.  E.g. ``pt.tl.PyDESeq2``, ``pt.tl.EdgeR``,
        ``pt.tl.Milo``.
    adata : AnnData
        Input AnnData (pseudobulked for DE, single-cell for DA).
    condition_cols : list[str]
        Columns in ``adata.obs`` whose combinations define the condition
        space, e.g. ``["timepoint", "disease_subtype"]``.
    combined_col : str, default ``"condition_combined"``
        Name for the new compound column added to ``adata.obs``.
    sep : str, default ``"_"``
        Separator for joining condition levels into labels.
    subset_contrasts : list[dict], optional
        Run only these contrasts instead of all pairwise ones.  Each entry
        must be a dict with keys ``"group"``, ``"baseline"``, ``"label"``
        (as returned by :func:`build_all_pairwise_contrasts`).
    **kwargs
        Forwarded verbatim to ``model_cls.compare_groups()``.  Common
        options: ``layer``, ``paired_by``, ``mask``, ``fit_kwargs``,
        ``test_kwargs``.

    Returns
    -------
    pd.DataFrame
        Tidy results from all contrasts, with a ``"contrast"`` column
        identifying each pairwise comparison.

    Raises
    ------
    RuntimeError
        If every contrast fails or returns no cells.

    Examples
    --------
    Differential expression with PyDESeq2 across timepoint x subtype:

    >>> import pertpy as pt
    >>> import patpy.tl.differential as ptd
    >>>
    >>> res = ptd.run_condition_combinations(
    ...     pt.tl.PyDESeq2,
    ...     pdata,
    ...     condition_cols=["timepoint", "disease_subtype"],
    ...     layer="counts",
    ... )
    >>> res.head()

    Differential expression with EdgeR, paired by patient:

    >>> res = ptd.run_condition_combinations(
    ...     pt.tl.EdgeR,
    ...     pdata,
    ...     condition_cols=["timepoint", "disease_subtype"],
    ...     layer="counts",
    ...     paired_by="patient_id",
    ... )

    Run only a specific subset of contrasts:

    >>> contrasts = ptd.build_all_pairwise_contrasts(pdata, ["timepoint", "subtype"])
    >>> res = ptd.run_condition_combinations(
    ...     pt.tl.PyDESeq2,
    ...     pdata,
    ...     condition_cols=["timepoint", "subtype"],
    ...     subset_contrasts=contrasts[:3],
    ... )
    """
    return _iter_contrasts(
        model_cls,
        adata,
        condition_cols,
        combined_col=combined_col,
        sep=sep,
        subset_contrasts=subset_contrasts,
        kwargs=kwargs,
    )


class ConditionComparison:
    """Reusable wrapper for running a pertpy method across condition combinations.

    Stores the model class and default keyword arguments so you can call
    :meth:`run` on multiple datasets without repeating yourself.

    Parameters
    ----------
    model_cls : pertpy class
        Any pertpy differential method class with a ``compare_groups``
        classmethod.  E.g. ``pt.tl.PyDESeq2``, ``pt.tl.EdgeR``,
        ``pt.tl.Milo``.
    **default_kwargs
        Default keyword arguments forwarded to ``model_cls.compare_groups()``
        on every call.  Can be overridden per-call in :meth:`run`.

    Examples
    --------
    >>> import pertpy as pt
    >>> import patpy.tl.differential as ptd
    >>>
    >>> cc = ptd.ConditionComparison(pt.tl.EdgeR, layer="counts", paired_by="patient_id")
    >>>
    >>> res_tp_subtype = cc.run(pdata, condition_cols=["timepoint", "disease_subtype"])
    >>> res_tp_only = cc.run(pdata, condition_cols=["timepoint"])
    """

    def __init__(self, model_cls: Any, **default_kwargs: Any) -> None:
        self.model_cls = model_cls
        self.default_kwargs = default_kwargs

    def run(
        self,
        adata: ad.AnnData,
        condition_cols: list[str],
        *,
        combined_col: str = "condition_combined",
        sep: str = "_",
        subset_contrasts: list[dict] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Run the method across all pairwise contrasts of *condition_cols*.

        Parameters
        ----------
        adata : AnnData
        condition_cols : list[str]
            Columns whose combinations define the condition space.
        combined_col : str, default ``"condition_combined"``
            Name for the compound column added to ``adata.obs``.
        sep : str, default ``"_"``
            Separator for joining condition levels.
        subset_contrasts : list[dict], optional
            Restrict to these contrasts only (see
            :func:`build_all_pairwise_contrasts`).
        **kwargs
            Override or extend the default kwargs stored on this instance.

        Returns
        -------
        pd.DataFrame
            Tidy results with a ``"contrast"`` column.
        """
        merged_kwargs = {**self.default_kwargs, **kwargs}
        return _iter_contrasts(
            self.model_cls,
            adata,
            condition_cols,
            combined_col=combined_col,
            sep=sep,
            subset_contrasts=subset_contrasts,
            kwargs=merged_kwargs,
        )

    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self.default_kwargs.items())
        return f"ConditionComparison({self.model_cls.__name__}, {kw})"
