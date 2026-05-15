from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import anndata as ad


def _make_and_fit(model_cls: Any, adata: ad.AnnData, design_formula: str, layer: str | None) -> Any:
    """Construct a pertpy model and call ``.fit()`` exactly once.

    pertpy's ``EdgeR.fit()`` assigns ``self.fit = <R ListVector>``, clobbering
    the Python method.  We call ``fit()`` immediately after construction and
    never call it again.  ``layer`` is passed to the constructor (where pertpy
    expects it); no extra arguments are forwarded to ``fit()``.
    """
    kw = {"layer": layer} if layer is not None else {}
    model = model_cls(adata, design=design_formula, **kw)
    model.fit()
    return model


def _edger_coef_key(coef_names: list[str], group_col: str, level: str) -> str | None:
    """Resolve a group level to its R design-matrix column name.

    R's ``model.matrix`` uses bracket notation for no-intercept designs::

        condition_group[COVID_SEV_female]

    while pandas uses plain concatenation::

        condition_groupCOVID_SEV_female

    Both are tried.
    """
    for candidate in (f"{group_col}[{level}]", f"{group_col}{level}"):
        if candidate in coef_names:
            return candidate
    return None


def _edger_group(
    model_cls: Any,
    adata: ad.AnnData,
    group_col: str,
    contrasts: list[dict],
    layer: str | None,
) -> tuple[pd.DataFrame, Any]:
    """Single EdgeR model with ``~ 0 + group``; extract every pairwise contrast.

    ``~ 0 + group`` (no intercept) gives one coefficient per group level,
    estimated jointly from **all samples** in a single GLM.  Any pairwise
    comparison is a ``[+1, -1, 0, …]`` contrast vector. This
    follows the edgeR User's Guide §3.2.

    Contrast vectors are built from the design matrix column names, which use
    R bracket notation; :func:`_edger_coef_key` handles both naming conventions.
    """
    model = _make_and_fit(model_cls, adata, f"~ 0 + {group_col}", layer)
    coef_names = list(model.design.columns)
    results = []
    for spec in contrasts:
        gk = _edger_coef_key(coef_names, group_col, spec["group"])
        bk = _edger_coef_key(coef_names, group_col, spec["baseline"])
        if gk is None or bk is None:
            warnings.warn(
                f"Contrast '{spec['label']}': could not map "
                f"'{spec['group']}' or '{spec['baseline']}' to a design column. "
                f"Available columns: {coef_names}. Skipping.",
                UserWarning,
                stacklevel=4,
            )
            continue
        vec = [0.0] * len(coef_names)
        vec[coef_names.index(gk)] = 1.0
        vec[coef_names.index(bk)] = -1.0
        try:
            res = model._test_single_contrast(vec)
            res["contrast"] = spec["label"]
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Contrast '{spec['label']}' failed: {exc}",
                UserWarning,
                stacklevel=4,
            )
    if not results:
        raise RuntimeError(f"All EdgeR group contrasts failed. Design matrix columns found: {coef_names}")
    return pd.concat(results, ignore_index=True), model


def _edger_interaction(
    model_cls: Any,
    adata: ad.AnnData,
    condition_cols: list[str],
    ref_levels: dict[str, str] | None,
    layer: str | None,
) -> tuple[pd.DataFrame, Any]:
    """Single EdgeR model with ``~ A + B + A:B``; test every coefficient.

    The interaction coefficient ``A:B`` tests whether the effect of A differs
    between levels of B — the formal sex-dimorphism test.  Main-effect
    coefficients give the effect of each factor at the reference level of the
    other.

    Reference levels are set via ``ref_levels`` so that coefficients are
    immediately interpretable (e.g. COVID_SEV vs HV, male vs female).  This
    follows the edgeR User's Guide §3.3.

    Each coefficient is tested independently by passing a unit contrast vector
    ``[0, …, 0, 1, 0, …, 0]`` with the 1 at the coefficient's position.
    """
    col_a, col_b = condition_cols
    if ref_levels:
        import pandas as _pd

        for col, ref in ref_levels.items():
            if col in adata.obs.columns:
                cats = [ref] + [c for c in adata.obs[col].unique() if c != ref]
                adata.obs[col] = _pd.Categorical(adata.obs[col], categories=cats)
    model = _make_and_fit(model_cls, adata, f"~ {col_a} + {col_b} + {col_a}:{col_b}", layer)
    coef_names = list(model.design.columns)
    results = []
    for i, coef in enumerate(coef_names):
        vec = [0.0] * len(coef_names)
        vec[i] = 1.0
        try:
            res = model._test_single_contrast(vec)
            res["contrast"] = coef
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Coefficient '{coef}' failed: {exc}", UserWarning, stacklevel=4)
    if not results:
        raise RuntimeError("All EdgeR interaction coefficient tests failed.")
    return pd.concat(results, ignore_index=True), model


def _pydeseq2_coef_names(model: Any) -> list[str]:
    """Return coefficient names from a fitted PyDESeq2 model."""
    return list(model.dds.varm["LFC"].columns)


def _pydeseq2_parse_level(coef: str, group_col: str) -> str:
    """Strip the group-column prefix from a treatment-coded coefficient name.

    PyDESeq2 / R treatment coding produces names like::

        condition_group[T.COVID_SEV_female]  # bracket notation
        condition_groupCOVID_SEV_female  # plain concatenation

    Both are handled, returning the bare level string.
    """
    s = coef.replace(f"{group_col}[T.", "").rstrip("]")
    if s.startswith(group_col):
        s = s[len(group_col) :]
    return s


def _pydeseq2_group(
    model_cls: Any,
    adata: ad.AnnData,
    group_col: str,
    contrasts: list[dict],
    layer: str | None,
    **deseq2_kwargs: Any,
) -> tuple[pd.DataFrame, Any]:
    """Single PyDESeq2 model with ``~ group``; extract every pairwise contrast.

    PyDESeq2 requires an intercept, so we use ``~ group`` (treatment coding)
    rather than ``~ 0 + group``.  The alphabetically first level becomes the
    reference, absorbed into the intercept; all other levels get an explicit
    coefficient.

    Any pairwise comparison is built as a numeric contrast vector:
    ``+1`` at the numerator's coefficient index, ``-1`` at the denominator's
    (or just ``+1`` / ``-1`` alone when one group is the reference level,
    which has no coefficient).

    This is the DESeq2 "combined factor" approach from the DESeq2 vignette
    §"Interactions", fitting a single model on all samples and extracting
    any comparison via ``contrast`` vectors.
    """
    model = _make_and_fit(model_cls, adata, f"~ {group_col}", layer)
    coef_names = _pydeseq2_coef_names(model)

    level_to_coef: dict[str, str] = {
        _pydeseq2_parse_level(c, group_col): c for c in coef_names if group_col in c and c != "Intercept"
    }
    all_levels = sorted(adata.obs[group_col].unique().tolist())
    reference = next((lv for lv in all_levels if lv not in level_to_coef), None)

    results = []
    for spec in contrasts:
        group, baseline, label = spec["group"], spec["baseline"], spec["label"]
        try:
            vec = np.zeros(len(coef_names))
            if group != reference:
                if group not in level_to_coef:
                    raise KeyError(f"Level '{group}' not in coefficient map {level_to_coef}")
                vec[coef_names.index(level_to_coef[group])] += 1.0
            if baseline != reference:
                if baseline not in level_to_coef:
                    raise KeyError(f"Level '{baseline}' not in coefficient map {level_to_coef}")
                vec[coef_names.index(level_to_coef[baseline])] -= 1.0
            kw = {"quiet": True, **deseq2_kwargs}
            res = model._test_single_contrast(vec.tolist(), **kw)
            res["contrast"] = label
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Contrast '{label}' failed: {exc}", UserWarning, stacklevel=4)
    if not results:
        raise RuntimeError(
            f"All PyDESeq2 group contrasts failed. "
            f"Coefficients: {coef_names}, level map: {level_to_coef}, reference: {reference}"
        )
    return pd.concat(results, ignore_index=True), model


def _pydeseq2_interaction(
    model_cls: Any,
    adata: ad.AnnData,
    condition_cols: list[str],
    ref_levels: dict[str, str] | None,
    layer: str | None,
    **deseq2_kwargs: Any,
) -> tuple[pd.DataFrame, Any]:
    """Single PyDESeq2 model with ``~ A + B + A:B``; test every coefficient.

    Follows DESeq2 vignette Example 2: multi-factor design with an interaction
    term.  Each non-intercept coefficient is tested independently via a numeric
    contrast vector with a single ``1.0`` entry at that coefficient's position.

    The interaction coefficient ``A[T.x]:B[T.y]`` tests whether the effect of
    A (at level x vs its reference) differs when B is at level y vs its
    reference — the formal interaction test.
    """
    col_a, col_b = condition_cols
    if ref_levels:
        import pandas as _pd

        for col, ref in ref_levels.items():
            if col in adata.obs.columns:
                cats = [ref] + [c for c in adata.obs[col].unique() if c != ref]
                adata.obs[col] = _pd.Categorical(adata.obs[col], categories=cats)
    model = _make_and_fit(model_cls, adata, f"~ {col_a} + {col_b} + {col_a}:{col_b}", layer)
    coef_names = _pydeseq2_coef_names(model)
    results = []
    for i, coef in enumerate(coef_names):
        if coef == "Intercept":
            continue
        vec = np.zeros(len(coef_names))
        vec[i] = 1.0
        try:
            kw = {"quiet": True, **deseq2_kwargs}
            res = model._test_single_contrast(vec.tolist(), **kw)
            res["contrast"] = coef
            results.append(res)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Coefficient '{coef}' failed: {exc}", UserWarning, stacklevel=4)
    if not results:
        raise RuntimeError("All PyDESeq2 interaction coefficient tests failed.")
    return pd.concat(results, ignore_index=True), model


class FactorialDE:
    """Fit a single DE model on all samples and extract multiple contrasts.

    Unlike :class:`~patpy.tl.condition_comparison.ConditionComparison`, which
    fits one independent model per contrast on a data subset, ``FactorialDE``
    fits **one model on all donors simultaneously**.  This has two statistical
    consequences:

    1. **Shared dispersion / size-factor estimation.**  The mean-dispersion
       trend (EdgeR) or size factors (PyDESeq2) are estimated from all groups
       jointly, giving more stable and powerful tests — especially for lowly
       expressed genes where per-subset estimates are noisy.

    2. **Formal interaction test.**  With ``encoding="interaction"``, the
       design ``~ A + B + A:B`` includes an interaction coefficient that
       directly and formally tests whether the effect of A differs between
       levels of B (e.g. "is the COVID-19 response sex-dimorphic?").  This
       question cannot be answered by pairwise comparisons alone.

    **Limitation:** requires all condition combinations to be present in the
    data.  If any combination is missing, use
    :class:`~patpy.tl.condition_comparison.ConditionComparison` instead.

    Two encodings
    -------------
    ``"group"``
        Design ``~ 0 + condition_group`` (EdgeR) or ``~ condition_group``
        (PyDESeq2).  One coefficient per condition combination.  Any pairwise
        contrast is a linear combination of two coefficients from the single
        fitted model.  Produces the same contrasts as ``ConditionComparison``
        but with shared dispersion.

    ``"interaction"``
        Design ``~ A + B + A:B``.  The ``A:B`` coefficient tests whether the
        effect of A is the same across levels of B.  Main-effect coefficients
        give the effect of each factor at the reference level of the other.
        Set ``ref_levels`` to control which level is the reference so
        coefficients are immediately interpretable.

    Parameters
    ----------
    model_cls : ``pt.tl.EdgeR`` or ``pt.tl.PyDESeq2``
    layer : str, optional
        Layer in adata to use as counts.

    Attributes
    ----------
    model_ : pertpy model instance
        Populated after :meth:`run`.  The single fitted model shared across
        all contrasts.

    Examples
    --------
    Group encoding — shared dispersion, pairwise contrasts:

    >>> fc = ptf.FactorialDE(pt.tl.EdgeR)
    >>> res = fc.run(pdata, condition_cols=["Source", "Sex"], encoding="group")
    >>> res[res["adj_p_value"] < 0.05].groupby("contrast")["variable"].count()
    >>> fc.plot_volcano("COVID_SEV_female_vs_HV_female", results_df=res)

    Interaction encoding — formal test of sex-dimorphic COVID-19 response:

    >>> fc2 = ptf.FactorialDE(pt.tl.EdgeR)
    >>> res2 = fc2.run(
    ...     pdata,
    ...     condition_cols=["Source", "Sex"],
    ...     encoding="interaction",
    ...     ref_levels={"Source": "HV", "Sex": "female"},
    ... )
    >>> # Interaction term: is the COVID-19 effect different in males?
    >>> res2[res2["contrast"].str.contains(":")].query("adj_p_value < 0.05").head(20)

    One-shot convenience:

    >>> res, model = ptf.FactorialDE.run_once(
    ...     pt.tl.PyDESeq2,
    ...     pdata,
    ...     condition_cols=["Source", "Sex"],
    ...     encoding="interaction",
    ...     ref_levels={"Source": "HV", "Sex": "female"},
    ... )
    """

    def __init__(self, model_cls: Any, layer: str | None = None) -> None:
        self.model_cls = model_cls
        self.layer = layer
        self.model_: Any = None

    def run(
        self,
        adata: ad.AnnData,
        condition_cols: list[str],
        *,
        encoding: str = "group",
        group_col: str = "condition_group",
        sep: str = "_",
        ref_levels: dict[str, str] | None = None,
        contrasts: list[dict] | None = None,
        **deseq2_kwargs: Any,
    ) -> pd.DataFrame:
        """Fit the model and extract all contrasts / coefficients.

        Parameters
        ----------
        adata : AnnData
            Pseudobulked input (one row per donor).
        condition_cols : list[str]
            Columns in ``adata.obs`` defining the factorial conditions,
            e.g. ``["Source", "Sex"]``.
        encoding : {"group", "interaction"}, default ``"group"``
            Which design to use (see class docstring).
        group_col : str, default ``"condition_group"``
            Name for the combined column added to ``adata.obs`` when
            ``encoding="group"``.
        sep : str, default ``"_"``
            Separator for combining condition levels into labels.
        ref_levels : dict[str, str], optional
            Reference level per condition column, e.g.
            ``{"Source": "HV", "Sex": "female"}``.  Only used when
            ``encoding="interaction"``.  Sets which level is absorbed into
            the intercept, making main-effect coefficients interpretable as
            "effect at the reference level of the other factor".
        contrasts : list[dict], optional
            Explicit subset of contrasts to test (``encoding="group"`` only).
            Each dict must have keys ``"group"``, ``"baseline"``, ``"label"``.
            If ``None``, all pairwise contrasts are tested automatically.

        Returns
        -------
        pd.DataFrame
            Tidy DE results with a ``"contrast"`` column.  For
            ``encoding="group"`` the contrast column contains labels like
            ``"COVID_SEV_female_vs_HV_female"``.  For
            ``encoding="interaction"`` it contains the coefficient names from
            the design matrix (e.g. ``"Source[T.COVID_SEV]"``,
            ``"Source[T.COVID_SEV]:Sex[T.male]"``).
        """
        adata = adata.copy()
        name = self.model_cls.__name__

        if encoding == "group":
            adata.obs[group_col] = adata.obs[condition_cols].astype(str).agg(sep.join, axis=1)
            if contrasts is None:
                levels = sorted(adata.obs[group_col].unique().tolist())
                contrasts = [
                    {"group": g, "baseline": b, "label": f"{g}_vs_{b}"}
                    for i, g in enumerate(levels)
                    for b in levels[i + 1 :]
                ]
            if name == "EdgeR":
                results, model = _edger_group(self.model_cls, adata, group_col, contrasts, self.layer)
            elif name == "PyDESeq2":
                results, model = _pydeseq2_group(
                    self.model_cls, adata, group_col, contrasts, self.layer, **deseq2_kwargs
                )
            else:
                raise ValueError(f"Unsupported model '{name}'. Use pt.tl.EdgeR or pt.tl.PyDESeq2.")

        elif encoding == "interaction":
            if len(condition_cols) != 2:
                raise ValueError("encoding='interaction' requires exactly 2 condition_cols.")
            if name == "EdgeR":
                results, model = _edger_interaction(self.model_cls, adata, condition_cols, ref_levels, self.layer)
            elif name == "PyDESeq2":
                results, model = _pydeseq2_interaction(
                    self.model_cls, adata, condition_cols, ref_levels, self.layer, **deseq2_kwargs
                )
            else:
                raise ValueError(f"Unsupported model '{name}'. Use pt.tl.EdgeR or pt.tl.PyDESeq2.")

        else:
            raise ValueError(f"encoding must be 'group' or 'interaction', got '{encoding}'.")

        self.model_ = model
        return results

    def get_model(self) -> Any:
        """Return the single fitted model instance.

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called yet.
        """
        if self.model_ is None:
            raise RuntimeError("No model fitted yet. Call run() first.")
        return self.model_

    def plot_volcano(
        self,
        contrast: str,
        *,
        results_df: pd.DataFrame,
        **kw: Any,
    ) -> Any:
        """Call pertpy's ``plot_volcano`` for a specific contrast.

        Parameters
        ----------
        contrast : str
            Contrast label to filter ``results_df`` on and to identify which
            result to plot.
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **kw
            Forwarded to pertpy's ``plot_volcano``.
        """
        return self.get_model().plot_volcano(results_df[results_df["contrast"] == contrast], **kw)

    def plot_fold_change(
        self,
        contrast: str,
        *,
        results_df: pd.DataFrame,
        **kw: Any,
    ) -> Any:
        """Call pertpy's ``plot_fold_change`` for a specific contrast.

        Parameters
        ----------
        contrast : str
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **kw
            Forwarded to pertpy's ``plot_fold_change``.
        """
        return self.get_model().plot_fold_change(results_df[results_df["contrast"] == contrast], **kw)

    def plot_multicomparison_fc(
        self,
        *,
        results_df: pd.DataFrame,
        **kw: Any,
    ) -> Any:
        """Call pertpy's ``plot_multicomparison_fc`` on the full results.

        Passes the complete ``results_df`` unfiltered, since
        ``plot_multicomparison_fc`` is designed to visualise multiple
        contrasts side by side.

        Parameters
        ----------
        results_df : pd.DataFrame
            Full results DataFrame returned by :meth:`run`.
        **kw
            Forwarded to pertpy's ``plot_multicomparison_fc``.
        """
        return self.get_model().plot_multicomparison_fc(results_df, **kw)

    @classmethod
    def run_once(
        cls,
        model_cls: Any,
        adata: ad.AnnData,
        condition_cols: list[str],
        *,
        encoding: str = "group",
        group_col: str = "condition_group",
        sep: str = "_",
        ref_levels: dict[str, str] | None = None,
        contrasts: list[dict] | None = None,
        layer: str | None = None,
        **deseq2_kwargs: Any,
    ) -> tuple[pd.DataFrame, Any]:
        """One-shot convenience wrapper — returns ``(results_df, model)``.

        Equivalent to creating a :class:`FactorialDE` instance, calling
        :meth:`run`, and accessing :attr:`model_`.

        Parameters
        ----------
        model_cls : ``pt.tl.EdgeR`` or ``pt.tl.PyDESeq2``
        adata : AnnData
        condition_cols : list[str]
        encoding : {"group", "interaction"}, default ``"group"``
        group_col : str, default ``"condition_group"``
        sep : str, default ``"_"``
        ref_levels : dict[str, str], optional
        contrasts : list[dict], optional
        layer : str, optional

        Returns
        -------
        tuple[pd.DataFrame, Any]
            Tidy results and the single fitted model instance.

        Examples
        --------
        >>> res, model = ptf.FactorialDE.run_once(
        ...     pt.tl.EdgeR,
        ...     pdata,
        ...     condition_cols=["Source", "Sex"],
        ...     encoding="interaction",
        ...     ref_levels={"Source": "HV", "Sex": "female"},
        ... )
        >>> res[res["contrast"].str.contains(":")].query("adj_p_value < 0.05").head()
        """
        fde = cls(model_cls, layer=layer)
        results = fde.run(
            adata,
            condition_cols,
            encoding=encoding,
            group_col=group_col,
            sep=sep,
            ref_levels=ref_levels,
            contrasts=contrasts,
            **deseq2_kwargs,
        )
        return results, fde.model_

    def __repr__(self) -> str:
        fitted = "fitted" if self.model_ is not None else "not fitted"
        return f"FactorialDE({self.model_cls.__name__}, layer={self.layer!r}) [{fitted}]"
