from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd

from patpy.tl._differential_utils import (
    build_all_pairwise_contrasts,
    build_condition_combinations,
    filter_adata_to_conditions,
)

if TYPE_CHECKING:
    import anndata as ad

__all__ = [
    "build_condition_combinations",
    "build_all_pairwise_contrasts",
    "filter_adata_to_conditions",
    "ConditionComparison",
]


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
    """Run `model_cls.compare_groups` for every pairwise contrast.

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

            # Store a model instance as a method carrier for pertpy's plotting
            # API.  compare_groups constructs and fits the model internally; we
            # construct a second lightweight instance here solely so the caller
            # can invoke plot_volcano(res_df), plot_fold_change(res_df), etc.
            # The design "~{combined_col}" matches what compare_groups uses.
            try:
                models[label] = model_cls(
                    sub,
                    design=f"~{combined_col}",
                    layer=kwargs.get("layer", None),
                )
            except Exception:  # noqa: BLE001
                pass

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

    return pd.concat(results, ignore_index=True), models


class ConditionComparison:
    """Run any pertpy DE method across all pairwise condition combinations.

    Fits one **independent model per contrast**, each on the subset of
    samples belonging to the two groups being compared.  Uses
    ``model_cls.compare_groups()`` internally and therefore works with any
    pertpy differential method (PyDESeq2, EdgeR, Milo, …).

    When to use this vs :class:`~patpy.tl.factorial_comparison.FactorialDE`
    -------------------------------------------------------------------------
    * Use ``ConditionComparison`` when condition combinations may be missing
      from the data, when you want to quickly screen many condition axes, or
      when you need a method beyond EdgeR / PyDESeq2 (e.g. Milo for
      differential abundance).
    * Use :class:`~patpy.tl.factorial_comparison.FactorialDE` when all
      condition combinations are present and you want (a) more powerful
      pairwise tests via shared dispersion estimation across all groups, or
      (b) a formal interaction test (e.g. "is the COVID-19 response
      sex-dimorphic?").

    Parameters
    ----------
    model_cls : pertpy differential method class
        Any class with a ``compare_groups`` classmethod, e.g.
        ``pt.tl.PyDESeq2``, ``pt.tl.EdgeR``, ``pt.tl.Milo``.
    **default_kwargs
        Default keyword arguments forwarded to ``model_cls.compare_groups()``
        on every :meth:`run` call.  Can be overridden per-call.

    Attributes
    ----------
    models_ : dict[str, Any]
        Populated after :meth:`run`.  Maps each contrast label to a model
        instance constructed with the same subset AnnData and design formula
        used internally by ``compare_groups``.  Use :meth:`get_model` for
        safe access with a clear error if the contrast is not found.

    Examples
    --------
    Run and then use pertpy's plotting methods via the named wrappers:

    >>> import pertpy as pt
    >>> import patpy.tl.condition_comparison as ptc
    >>>
    >>> cc = ptc.ConditionComparison(pt.tl.PyDESeq2)
    >>> res = cc.run(pdata, condition_cols=["Source", "Sex"])
    >>>
    >>> # How many genes pass FDR < 0.05 per contrast?
    >>> res[res["adj_p_value"] < 0.05].groupby("contrast")["variable"].count()
    >>>
    >>> # Named plot wrappers
    >>> cc.plot_volcano("COVID_SEV_female_vs_HV_female", results_df=res)
    >>> cc.plot_fold_change("COVID_SEV_female_vs_HV_female", results_df=res, n_top_vars=20)
    >>> cc.plot_multicomparison_fc("COVID_SEV_female_vs_HV_female", results_df=res)
    >>>
    >>> # Or retrieve the model instance directly for full control
    >>> model = cc.get_model("COVID_SEV_female_vs_HV_female")
    >>> model.plot_volcano(res[res["contrast"] == "COVID_SEV_female_vs_HV_female"])

    Reuse across multiple condition axes:

    >>> cc = ptc.ConditionComparison(pt.tl.EdgeR)
    >>> res_source_sex = cc.run(pdata, condition_cols=["Source", "Sex"])
    >>> res_source = cc.run(pdata, condition_cols=["Source"])

    One-off analysis:

    >>> res, models = ptc.ConditionComparison.run_once(pt.tl.PyDESeq2, pdata, condition_cols=["Source", "Sex"])
    >>> models["COVID_SEV_female_vs_HV_female"].plot_volcano(res[res["contrast"] == "COVID_SEV_female_vs_HV_female"])

    Restrict to biologically motivated contrasts only:

    >>> all_contrasts = ptc.build_all_pairwise_contrasts(pdata, ["Source", "Sex"])
    >>> vs_hv = [c for c in all_contrasts if "HV" in c["group"] or "HV" in c["baseline"]]
    >>> res_vs_hv, _ = ptc.ConditionComparison.run_once(
    ...     pt.tl.PyDESeq2,
    ...     pdata,
    ...     condition_cols=["Source", "Sex"],
    ...     subset_contrasts=vs_hv,
    ... )
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
            Restrict to these contrasts only.  Each dict must have keys
            ``"group"``, ``"baseline"``, ``"label"``.  Use
            :func:`build_all_pairwise_contrasts` to generate the full list
            and then filter it.
        **kwargs
            Override or extend the default kwargs stored on this instance.

        Returns
        -------
        pd.DataFrame
            Tidy results from all contrasts, with a ``"contrast"`` column
            identifying each pairwise comparison.
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
        formula used by ``compare_groups`` for that contrast.  Use it to call
        any pertpy instance plotting method directly, or use the convenience
        wrappers :meth:`plot_volcano`, :meth:`plot_fold_change`, and
        :meth:`plot_multicomparison_fc`.

        Parameters
        ----------
        contrast : str
            Contrast label, e.g. ``"COVID_SEV_female_vs_HV_female"``.
            Call ``list(cc.models_)`` to see available labels.

        Returns
        -------
        pertpy model instance

        Raises
        ------
        KeyError
            If ``contrast`` is not found in :attr:`models_`.
        """
        if contrast not in self.models_:
            raise KeyError(
                f"No model instance found for contrast '{contrast}'. "
                f"Available contrasts: {list(self.models_)}. "
                "Note: model construction is attempted on a best-effort basis; "
                "some model classes may not support the default constructor signature."
            )
        return self.models_[contrast]

    def plot_volcano(
        self,
        contrast: str,
        *,
        results_df: pd.DataFrame,
        **plot_kwargs: Any,
    ) -> Any:
        """Call pertpy's ``plot_volcano`` for a specific contrast.

        Parameters
        ----------
        contrast : str
            Contrast label identifying the model instance and used to filter ``results_df``.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **plot_kwargs
            Additional keyword arguments forwarded to pertpy's ``plot_volcano``.

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

        Parameters
        ----------
        contrast : str
            Contrast label identifying the model instance and used to filter ``results_df``.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **plot_kwargs
            Additional keyword arguments forwarded to pertpy's ``plot_fold_change``.

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
        side by side.  The ``contrast`` argument selects which model instance
        to call the method on.

        Parameters
        ----------
        contrast : str
            Any contrast label in :attr:`models_`.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **plot_kwargs
            Additional keyword arguments forwarded to pertpy's ``plot_multicomparison_fc``.

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
            raise AttributeError(f"Model class '{type(model).__name__}' has no method 'plot_multicomparison_fc'.")
        return plot_fn(results_df, **plot_kwargs)

    def _plot(
        self,
        method: str,
        *,
        contrast: str,
        results_df: pd.DataFrame,
        **plot_kwargs: Any,
    ) -> Any:
        """Internal dispatcher — call an arbitrary pertpy plot method.

        Prefer the named wrappers (:meth:`plot_volcano`,
        :meth:`plot_fold_change`) over calling this directly.  For
        ``plot_multicomparison_fc`` use :meth:`plot_multicomparison_fc`
        instead, as the full DataFrame is passed unfiltered in that case.

        Parameters
        ----------
        method : str
            Pertpy plotting method name, e.g. ``"plot_volcano"``,
            ``"plot_fold_change"``, ``"plot_paired"``.
        contrast : str
        results_df : pd.DataFrame
        **plot_kwargs
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

        Convenience shorthand for one-off analyses.  Equivalent to creating a
        :class:`ConditionComparison`, calling :meth:`run`, and accessing
        :attr:`models_`.

        Parameters
        ----------
        model_cls : pertpy differential method class
        adata : AnnData
        condition_cols : list[str]
        combined_col : str, default ``"condition_combined"``
        sep : str, default ``"_"``
        subset_contrasts : list[dict], optional
        **kwargs
            Forwarded to ``model_cls.compare_groups()``.  Common options:
            ``layer``, ``paired_by``, ``mask``, ``fit_kwargs``,
            ``test_kwargs``.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, Any]]
            - Tidy results with a ``"contrast"`` column.
            - Dict mapping each contrast label to its model instance.

        Raises
        ------
        RuntimeError
            If every contrast fails or returns no cells.

        Examples
        --------
        >>> res, models = ptc.ConditionComparison.run_once(pt.tl.PyDESeq2, pdata, condition_cols=["Source", "Sex"])
        >>> res[res["adj_p_value"] < 0.05].groupby("contrast")["variable"].count()
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
