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

    Returns one row per unique observed combination, not the Cartesian product of all levels.

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
       timepoint disease_subtype label
    0        T1               A  T1_A
    1        T1               B  T1_B
    2        T2               A  T2_A
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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Internal: run model_cls.compare_groups for every pairwise contrast.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        - Concatenated tidy results with a ``"contrast"`` column.
        - Dict mapping each contrast label to a model instance constructed
          with the same subset AnnData and design, so callers can access
          pertpy's plotting methods (``plot_volcano``, ``plot_fold_change``,
          ``plot_multicomparison_fc``, ``plot_paired``).
    """
    adata.obs[combined_col] = adata.obs[condition_cols].astype(str).agg(sep.join, axis=1)
    all_contrasts = subset_contrasts or build_all_pairwise_contrasts(adata, condition_cols, sep=sep)

    results = []
    models: dict[str, Any] = {}

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

            # Store a model instance as a method carrier for pertpy's plotting API.
            # compare_groups constructs and fits the model internally; we construct
            # a second lightweight instance here solely so the user can call
            # instance.plot_volcano(res_df), instance.plot_fold_change(res_df), etc.
            # The design "~{combined_col}" matches what compare_groups uses internally.
            try:
                models[label] = model_cls(
                    sub,
                    design=f"~{combined_col}",
                    layer=kwargs.get("layer", None),
                )
            except Exception:  # noqa: BLE001
                # If construction fails (model may require extra args), skip
                # silently — results are still returned regardless.
                pass

        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Contrast '{label}' failed: {exc}",
                UserWarning,
                stacklevel=3,
            )

    if not results:
        raise RuntimeError(
            "All contrasts failed or returned no results. "
            "Check your data, design, and condition columns."
        )

    return pd.concat(results, ignore_index=True), models


class ConditionComparison:
    """Reusable wrapper for running a pertpy method across condition combinations.

    Stores the model class and default keyword arguments so you can call
    :meth:`run` on multiple datasets or condition axes without repeating yourself.

    After :meth:`run` completes, fitted model instances are available via
    :attr:`models_` and :meth:`get_model`, giving full access to pertpy's
    plotting API via the named wrappers :meth:`plot_volcano`,
    :meth:`plot_fold_change`, and :meth:`plot_multicomparison_fc`, or via the
    internal :meth:`_plot` dispatcher for any other pertpy plot method.

    There are two ways to use this class:

    - :meth:`run` (instance method): create the object once with your model
      and shared settings, then call ``run()`` as many times as needed on
      different datasets or condition axes. The stored defaults are merged into
      every call automatically, and model instances are stored on ``self``.

    - :meth:`run_once` (classmethod): skip creating an instance altogether.
      Pass everything in a single call and get back a ``(results, models)``
      tuple. Use this when you only need one result but still want access to
      the model instances for plotting.

    Parameters
    ----------
    model_cls : pertpy class
        Any pertpy differential method class with a ``compare_groups``
        classmethod. E.g. ``pt.tl.PyDESeq2``, ``pt.tl.EdgeR``,
        ``pt.tl.Milo``.
    **default_kwargs
        Default keyword arguments forwarded to ``model_cls.compare_groups()``
        on every :meth:`run` call. Can be overridden per-call.

    Attributes
    ----------
    models_ : dict[str, Any]
        Populated after :meth:`run`. Maps each contrast label to a model
        instance constructed with the same subset AnnData and design formula
        used internally by ``compare_groups``. Use :meth:`get_model` for
        safe access with a clear error if the contrast is not found.

    Examples
    --------
    Run and then use pertpy's plotting methods via the named wrappers:

    >>> import pertpy as pt
    >>> import patpy.tl.condition_comparison as ptc
    >>>
    >>> cc = ptc.ConditionComparison(pt.tl.PyDESeq2, layer="counts")
    >>> res = cc.run(pdata, condition_cols=["Source"])
    >>>
    >>> # Named plot wrappers — no need to know pertpy's internal method names
    >>> cc.plot_volcano("COVID_SEV_vs_HV", results_df=res)
    >>> cc.plot_fold_change("COVID_SEV_vs_HV", results_df=res, n_top_vars=20)
    >>> cc.plot_multicomparison_fc("COVID_SEV_vs_HV", results_df=res)
    >>>
    >>> # Or retrieve the model instance directly for full control
    >>> model = cc.get_model("COVID_SEV_vs_HV")
    >>> model.plot_volcano(res[res["contrast"] == "COVID_SEV_vs_HV"])

    Reuse model and settings across multiple condition axes:

    >>> cc = ptc.ConditionComparison(pt.tl.EdgeR, layer="counts", paired_by="patient_id")
    >>> res_source = cc.run(pdata, condition_cols=["Source"])
    >>> res_sex    = cc.run(pdata, condition_cols=["Sex"])

    One-off analysis with model access:

    >>> res, models = ptc.ConditionComparison.run_once(
    ...     pt.tl.PyDESeq2, pdata, condition_cols=["Source"], layer="counts"
    ... )
    >>> models["COVID_SEV_vs_HV"].plot_volcano(res[res["contrast"] == "COVID_SEV_vs_HV"])
    """

    def __init__(self, model_cls: Any, **default_kwargs: Any) -> None:
        self.model_cls = model_cls
        self.default_kwargs = default_kwargs
        self.models_: dict[str, Any] = {}

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
        """Run the stored method across all pairwise contrasts of ``condition_cols``.

        After this call, :attr:`models_` is populated with a fitted model
        instance per contrast, giving access to pertpy's full plotting API
        via :meth:`plot_volcano`, :meth:`plot_fold_change`,
        :meth:`plot_multicomparison_fc`, or :meth:`get_model`.

        Parameters
        ----------
        adata : AnnData
            Input AnnData (pseudobulked for DE, single-cell for DA).
        condition_cols : list[str]
            Columns in ``adata.obs`` whose combinations define the condition
            space, e.g. ``["Source", "Sex"]``.
        combined_col : str, default ``"condition_combined"``
            Name for the new compound column added to ``adata.obs``.
        sep : str, default ``"_"``
            Separator for joining condition levels into labels.
        subset_contrasts : list[dict], optional
            Restrict to these contrasts only (see
            :func:`build_all_pairwise_contrasts`).
        **kwargs
            Override or extend the default kwargs stored on this instance.
            For example, if the instance was created with ``layer="counts"``,
            passing ``layer="lognorm"`` here overrides it for this call only.

        Returns
        -------
        pd.DataFrame
            Tidy results from all contrasts, with a ``"contrast"`` column
            identifying each pairwise comparison.

        Examples
        --------
        >>> cc = ConditionComparison(pt.tl.PyDESeq2, layer="counts")
        >>> res = cc.run(pdata, condition_cols=["Source"])
        >>>
        >>> # Named plot wrappers
        >>> cc.plot_volcano("COVID_SEV_vs_HV", results_df=res)
        >>> cc.plot_fold_change("COVID_SEV_vs_HV", results_df=res, n_top_vars=20)
        """
        merged_kwargs = {**self.default_kwargs, **kwargs}
        results, models = _iter_contrasts(
            self.model_cls,
            adata,
            condition_cols,
            combined_col=combined_col,
            sep=sep,
            subset_contrasts=subset_contrasts,
            kwargs=merged_kwargs,
        )
        self.models_.update(models)
        return results

    def get_model(self, contrast: str) -> Any:
        """Return the model instance for a specific contrast.

        The instance is constructed with the same subset AnnData and design
        formula used by ``compare_groups`` for that contrast. Use it to call
        any pertpy instance plotting method directly, or use the convenience
        wrappers :meth:`plot_volcano`, :meth:`plot_fold_change`, and
        :meth:`plot_multicomparison_fc`.

        Parameters
        ----------
        contrast : str
            Contrast label, e.g. ``"COVID_SEV_vs_HV"``. Must match a key in
            :attr:`models_`. Call ``list(cc.models_)`` to see available labels.

        Returns
        -------
        pertpy model instance

        Raises
        ------
        KeyError
            If ``contrast`` is not found in :attr:`models_`. This can happen
            if the model constructor failed silently during :meth:`run` — check
            that the model class accepts ``(adata, design, layer)`` arguments.
        """
        if contrast not in self.models_:
            available = list(self.models_)
            raise KeyError(
                f"No model instance found for contrast '{contrast}'. "
                f"Available contrasts: {available}. "
                "Note: model construction is attempted on a best-effort basis; "
                "some model classes may not support the default constructor signature."
            )
        return self.models_[contrast]

    # ------------------------------------------------------------------
    # Named plot wrappers
    # ------------------------------------------------------------------

    def plot_volcano(
        self,
        contrast: str,
        *,
        results_df: pd.DataFrame,
        **plot_kwargs: Any,
    ) -> Any:
        """Call pertpy's ``plot_volcano`` for a specific contrast.

        Retrieves the stored model instance for ``contrast``, filters
        ``results_df`` to that contrast, and calls ``model.plot_volcano``.

        Parameters
        ----------
        contrast : str
            Contrast label identifying the model instance and used to filter
            ``results_df``.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **plot_kwargs
            Additional keyword arguments forwarded to pertpy's
            ``plot_volcano``.

        Returns
        -------
        Whatever pertpy's ``plot_volcano`` returns (usually ``None`` or a
        Figure).

        Examples
        --------
        >>> cc = ConditionComparison(pt.tl.PyDESeq2, layer="counts")
        >>> res = cc.run(pdata, condition_cols=["Source", "Sex"])
        >>> cc.plot_volcano("COVID_SEV_0_vs_HV_0", results_df=res)
        """
        return self._plot("plot_volcano", contrast=contrast, results_df=results_df, **plot_kwargs)

    def plot_fold_change(
        self,
        contrast: str,
        *,
        results_df: pd.DataFrame,
        **plot_kwargs: Any,
    ) -> Any:
        """Call pertpy's ``plot_fold_change`` for a specific contrast.

        Retrieves the stored model instance for ``contrast``, filters
        ``results_df`` to that contrast, and calls ``model.plot_fold_change``.

        Parameters
        ----------
        contrast : str
            Contrast label identifying the model instance and used to filter
            ``results_df``.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **plot_kwargs
            Additional keyword arguments forwarded to pertpy's
            ``plot_fold_change``.

        Returns
        -------
        Whatever pertpy's ``plot_fold_change`` returns (usually ``None`` or a
        Figure).

        Examples
        --------
        >>> cc = ConditionComparison(pt.tl.PyDESeq2, layer="counts")
        >>> res = cc.run(pdata, condition_cols=["Source", "Sex"])
        >>> cc.plot_fold_change("COVID_SEV_0_vs_HV_0", results_df=res, n_top_vars=20)
        """
        return self._plot("plot_fold_change", contrast=contrast, results_df=results_df, **plot_kwargs)

    def plot_multicomparison_fc(
        self,
        contrast: str,
        *,
        results_df: pd.DataFrame,
        **plot_kwargs: Any,
    ) -> Any:
        """Call pertpy's ``plot_multicomparison_fc`` using a stored model instance.

        Unlike :meth:`plot_volcano` and :meth:`plot_fold_change`, this method
        passes the **full** ``results_df`` unfiltered, since
        ``plot_multicomparison_fc`` is designed to visualise multiple contrasts
        side by side. The ``contrast`` argument is used only to select which
        model instance to call the method on.

        Parameters
        ----------
        contrast : str
            Contrast label used to look up the model instance (any contrast
            in :attr:`models_` is valid). The full ``results_df`` is passed
            to the method regardless.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **plot_kwargs
            Additional keyword arguments forwarded to pertpy's
            ``plot_multicomparison_fc``.

        Returns
        -------
        Whatever pertpy's ``plot_multicomparison_fc`` returns.

        Examples
        --------
        >>> cc = ConditionComparison(pt.tl.PyDESeq2, layer="counts")
        >>> res = cc.run(pdata, condition_cols=["Source", "Sex"])
        >>> cc.plot_multicomparison_fc(list(cc.models_)[0], results_df=res)
        """
        model = self.get_model(contrast)
        plot_fn = getattr(model, "plot_multicomparison_fc", None)
        if plot_fn is None:
            raise AttributeError(
                f"Model class '{type(model).__name__}' has no method "
                f"'plot_multicomparison_fc'."
            )
        return plot_fn(results_df, **plot_kwargs)

    # ------------------------------------------------------------------
    # Internal generic dispatcher (escape hatch for advanced users)
    # ------------------------------------------------------------------

    def _plot(self, method: str, *, contrast: str, results_df: pd.DataFrame, **plot_kwargs: Any) -> Any:
        """Call an arbitrary pertpy plot method for a specific contrast.

        This is the internal dispatcher used by the named wrappers
        (:meth:`plot_volcano`, :meth:`plot_fold_change`). Prefer those over
        calling this directly. For ``plot_multicomparison_fc``, use
        :meth:`plot_multicomparison_fc` instead, as the full DataFrame is
        passed unfiltered in that case.

        Parameters
        ----------
        method : str
            Name of the pertpy plotting method, e.g. ``"plot_volcano"``,
            ``"plot_fold_change"``, ``"plot_paired"``.
        contrast : str
            Contrast label used to retrieve the model instance and to filter
            ``results_df``.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`. Will be filtered
            to ``contrast`` before being passed to the plot method.
        **plot_kwargs
            Additional keyword arguments forwarded to the plot method.

        Returns
        -------
        Whatever the pertpy plot method returns (usually ``None`` or a Figure).
        """
        model = self.get_model(contrast)
        plot_fn = getattr(model, method, None)
        if plot_fn is None:
            raise AttributeError(
                f"Model class '{type(model).__name__}' has no method '{method}'. "
                f"Available plot methods: {[m for m in dir(model) if m.startswith('plot_')]}"
            )
        contrast_df = results_df[results_df["contrast"] == contrast]
        return plot_fn(contrast_df, **plot_kwargs)

    @classmethod
    def run_once(
        cls,
        model_cls: Any,
        adata: ad.AnnData,
        condition_cols: list[str],
        *,
        combined_col: str = "condition_combined",
        sep: str = "_",
        subset_contrasts: list[dict] | None = None,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Run a pertpy method across all pairwise contrasts in a single call.

        Returns both the tidy results DataFrame and a dict of fitted model
        instances (one per contrast), so you can immediately call pertpy's
        plotting methods without creating a :class:`ConditionComparison` first.

        This classmethod is a convenience shorthand for one-off analyses.
        It is exactly equivalent to creating a :class:`ConditionComparison`,
        calling :meth:`run`, and accessing :attr:`models_`:

        .. code-block:: python

            # run_once — everything in one call
            res, models = ConditionComparison.run_once(
                pt.tl.PyDESeq2, pdata, ["Source"], layer="counts"
            )
            models["COVID_SEV_vs_HV"].plot_volcano(res[res["contrast"] == "COVID_SEV_vs_HV"])

            # Equivalent using an instance (preferred when running more than once)
            cc = ConditionComparison(pt.tl.PyDESeq2, layer="counts")
            res = cc.run(pdata, ["Source"])
            cc.plot_volcano("COVID_SEV_vs_HV", results_df=res)

        Parameters
        ----------
        model_cls : pertpy class
            Any pertpy differential method class with a ``compare_groups``
            classmethod. E.g. ``pt.tl.PyDESeq2``, ``pt.tl.EdgeR``,
            ``pt.tl.Milo``.
        adata : AnnData
            Input AnnData (pseudobulked for DE, single-cell for DA).
        condition_cols : list[str]
            Columns in ``adata.obs`` whose combinations define the condition
            space, e.g. ``["Source"]``.
        combined_col : str, default ``"condition_combined"``
            Name for the new compound column added to ``adata.obs``.
        sep : str, default ``"_"``
            Separator for joining condition levels into labels.
        subset_contrasts : list[dict], optional
            Restrict to these contrasts only (see
            :func:`build_all_pairwise_contrasts`).
        **kwargs
            Forwarded verbatim to ``model_cls.compare_groups()``. Common
            options: ``layer``, ``paired_by``, ``mask``, ``fit_kwargs``,
            ``test_kwargs``.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            - Tidy results from all contrasts, with a ``"contrast"`` column.
            - Dict mapping each contrast label to its model instance.

        Raises
        ------
        RuntimeError
            If every contrast fails or returns no cells.

        Examples
        --------
        >>> import pertpy as pt
        >>> import patpy.tl.condition_comparison as ptc
        >>>
        >>> res, models = ptc.ConditionComparison.run_once(
        ...     pt.tl.PyDESeq2,
        ...     pdata,
        ...     condition_cols=["Source"],
        ...     layer="counts",
        ... )
        >>> res.head()
        >>>
        >>> # Use pertpy's plotting API on any contrast
        >>> contrast = "COVID_SEV_vs_HV"
        >>> models[contrast].plot_volcano(res[res["contrast"] == contrast])
        >>> models[contrast].plot_fold_change(res[res["contrast"] == contrast])
        """
        cc = cls(model_cls, **kwargs)
        results = cc.run(
            adata,
            condition_cols,
            combined_col=combined_col,
            sep=sep,
            subset_contrasts=subset_contrasts,
        )
        return results, cc.models_

    def __repr__(self) -> str:
        n_models = len(self.models_)
        kw = ", ".join(f"{k}={v!r}" for k, v in self.default_kwargs.items())
        base = f"ConditionComparison({self.model_cls.__name__}, {kw})"
        if n_models:
            return f"{base} [{n_models} model(s): {list(self.models_)}]"
        return base