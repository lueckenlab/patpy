"""MIL benchmark runner for patient-level prediction.

Run multiple MIL models on stratified train/val/test splits and collect
metrics into a tidy DataFrame for comparison.
"""
from __future__ import annotations

import logging
import traceback
import warnings

import numpy as np
import pandas as pd
import scanpy as sc

from patpy.tl._types import _PREDICTION_TASKS
from patpy.tl.supervised import SupervisedSampleMethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _eval_classification(
    y_true: np.ndarray,
    preds: pd.DataFrame,
    label: str,
) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    y_pred = preds[f"{label}_pred"].values
    prob_cols = [c for c in preds.columns if c.startswith("prob_")]
    proba = preds[prob_cols].values

    results: dict[str, float] = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    try:
        if len(prob_cols) == 2:
            results["auroc"] = roc_auc_score(y_true, proba[:, 1])
            results["aupr"] = average_precision_score(y_true, proba[:, 1])
        elif len(prob_cols) > 2:
            results["auroc"] = roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro"
            )
    except ValueError as exc:
        logger.debug("AUROC/AUPR skipped: %s", exc)

    return results


def _eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _sample_labels(adata: sc.AnnData, sample_key: str, label_key: str) -> pd.Series:
    """Return one label value per sample, indexed by sample ID."""
    return adata.obs[[sample_key, label_key]].groupby(sample_key).first()[label_key]


# ---------------------------------------------------------------------------
# MILBenchmark
# ---------------------------------------------------------------------------


class MILBenchmark:
    """Run multiple MIL models on stratified train/val/test splits.

    Each model is trained on the training partition and evaluated on both
    the validation and test partitions.  All models must implement
    :meth:`~patpy.tl.supervised.SupervisedSampleMethod.predict_on_adata`
    (inductive prediction on held-out donors).

    Parameters
    ----------
    models : dict[str, SupervisedSampleMethod]
        Mapping of model name → configured (but not yet fitted) model instance.
    sample_key : str
        Column in ``adata.obs`` with donor identifiers.
    label_keys : list[str]
        Donor-level labels to evaluate.
    tasks : list[str]
        Prediction task per label (``"classification"`` or ``"regression"``).
    split_col : str, default ``"split"``
        Column in ``adata.obs`` carrying ``"train"``/``"val"``/``"test"``
        partition labels.  When absent, :func:`~patpy.pp.make_sample_splits`
        is called automatically using *label_keys[0]* for stratification.
    test_size : float, default 0.2
        Fraction held out as test (used when ``split_col`` is absent).
    val_size : float, default 0.15
        Fraction of non-test samples used for validation.
    stratify_by : list[str] or None
        Additional covariate columns for balanced splitting.
    n_splits : int, default 1
        Number of independent random splits.  When > 1, split columns
        ``"{split_col}_0"``, ``"{split_col}_1"``, … are used/created.
    seed : int, default 42

    Examples
    --------
    >>> from patpy.tl.mil_models import ABMIL, TransMIL
    >>> bench = MILBenchmark(
    ...     models={
    ...         "ABMIL":    ABMIL(sample_key="donor_id", label_keys=["disease"], tasks=["classification"]),
    ...         "TransMIL": TransMIL(sample_key="donor_id", label_keys=["disease"], tasks=["classification"]),
    ...     },
    ...     sample_key="donor_id",
    ...     label_keys=["disease"],
    ...     tasks=["classification"],
    ...     n_splits=3,
    ... )
    >>> results = bench.run(adata)
    >>> results.groupby(["model", "metric"])["value"].mean()
    """

    def __init__(
        self,
        models: dict[str, SupervisedSampleMethod],
        sample_key: str,
        label_keys: list[str],
        tasks: list[_PREDICTION_TASKS],
        *,
        split_col: str = "split",
        test_size: float = 0.2,
        val_size: float = 0.15,
        stratify_by: list[str] | None = None,
        n_splits: int = 1,
        seed: int = 42,
    ) -> None:
        if len(label_keys) != len(tasks):
            raise ValueError("label_keys and tasks must have the same length.")
        self.models = models
        self.sample_key = sample_key
        self.label_keys = label_keys
        self.tasks = tasks
        self.split_col = split_col
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_by = stratify_by or []
        self.n_splits = n_splits
        self.seed = seed

        self._results: list[dict] = []

    def run(self, adata: sc.AnnData) -> pd.DataFrame:
        """Fit all models across all splits and return a tidy results DataFrame.

        Parameters
        ----------
        adata
            Single-cell AnnData.  Split columns are written to ``.obs``
            when they do not already exist.

        Returns
        -------
        pd.DataFrame
            Columns: ``model``, ``split_idx``, ``eval_split``, ``label``,
            ``metric``, ``value``.
        """
        self._results = []

        # Ensure splits exist
        adata = self._ensure_splits(adata)

        col_names = (
            [f"{self.split_col}_{i}" for i in range(self.n_splits)]
            if self.n_splits > 1
            else [self.split_col]
        )

        for split_i, col in enumerate(col_names):
            logger.info("Running split %d / %d  (column='%s')", split_i + 1, self.n_splits, col)
            self._run_one_split(adata, col, split_i)

        return pd.DataFrame(self._results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_splits(self, adata: sc.AnnData) -> sc.AnnData:
        """Create split columns in adata.obs when they are missing."""
        from patpy.pp.splits import make_sample_splits

        col_names = (
            [f"{self.split_col}_{i}" for i in range(self.n_splits)]
            if self.n_splits > 1
            else [self.split_col]
        )

        missing = [c for c in col_names if c not in adata.obs.columns]
        if not missing:
            return adata

        logger.info(
            "Split column(s) %s not found. Calling make_sample_splits().", missing
        )
        make_sample_splits(
            adata,
            sample_key=self.sample_key,
            label_key=self.label_keys[0],
            test_size=self.test_size,
            val_size=self.val_size,
            covariate_keys=self.stratify_by or None,
            n_splits=self.n_splits,
            seed=self.seed,
            split_col=self.split_col,
        )
        return adata

    def _run_one_split(
        self, adata: sc.AnnData, col: str, split_i: int
    ) -> None:
        train_mask = adata.obs[col] == "train"
        train_adata = adata[train_mask].copy()

        if train_adata.n_obs == 0:
            warnings.warn(f"No training cells found in split column '{col}'.", stacklevel=2)
            return

        for model_name, model in self.models.items():
            logger.info("  Model: %s", model_name)

            # Pre-populate label mappings from the full adata so val/test classes
            # not present in this training split don't cause index-out-of-bounds.
            model._label_mappings = {}
            for lk in model.label_keys:
                col = adata.obs.groupby(model.sample_key)[lk].first().dropna()
                if col.dtype.kind not in ("f", "i", "u"):
                    classes = sorted(col.unique())
                    model._label_mappings[lk] = (classes, {c: i for i, c in enumerate(classes)})

            # Check inductive-prediction support
            try:
                model.prepare_anndata(train_adata)
            except Exception as exc:
                logger.error(
                    "Model '%s' failed on split %d:\n%s",
                    model_name, split_i, traceback.format_exc(),
                )
                print(f"[benchmark] ERROR — {model_name} split {split_i}: {exc}")
                continue

            for eval_split in ("val", "test"):
                split_mask = adata.obs[col] == eval_split
                eval_adata = adata[split_mask]

                if eval_adata.n_obs == 0:
                    continue

                for label, task in zip(self.label_keys, self.tasks):
                    try:
                        preds = model.predict_on_adata(eval_adata, label)
                    except NotImplementedError:
                        warnings.warn(
                            f"Model '{model_name}' does not implement predict_on_adata(). "
                            "Skipping held-out evaluation.",
                            stacklevel=2,
                        )
                        break

                    y_true_series = _sample_labels(eval_adata, self.sample_key, label)
                    # Align y_true to the prediction index
                    y_true = y_true_series.reindex(preds.index).values

                    metrics = self._compute_metrics(y_true, preds, label, task)
                    for metric_name, value in metrics.items():
                        self._results.append(
                            {
                                "model": model_name,
                                "split_idx": split_i,
                                "eval_split": eval_split,
                                "label": label,
                                "metric": metric_name,
                                "value": value,
                            }
                        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        preds: pd.Series | pd.DataFrame,
        label: str,
        task: str,
    ) -> dict[str, float]:
        if task == "classification":
            if not isinstance(preds, pd.DataFrame):
                warnings.warn(
                    f"Expected DataFrame for classification label '{label}'; got Series.",
                    stacklevel=3,
                )
                return {}
            return _eval_classification(y_true, preds, label)
        else:
            y_pred = preds.values if isinstance(preds, pd.Series) else preds.iloc[:, 0].values
            return _eval_regression(y_true.astype(float), y_pred.astype(float))

    def summary(self) -> pd.DataFrame:
        """Return mean ± std across splits, aggregated per model × label × metric.

        Returns
        -------
        pd.DataFrame
            Indexed by ``(model, label, metric)`` with ``mean`` and ``std`` columns.
        """
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        df = pd.DataFrame(self._results)
        return (
            df.groupby(["model", "eval_split", "label", "metric"])["value"]
            .agg(["mean", "std"])
            .round(4)
        )


# ---------------------------------------------------------------------------
# RepresentationBenchmark
# ---------------------------------------------------------------------------


class RepresentationBenchmark:
    """Benchmark methods that produce sample-level representations with a supervised linear head.

    Supports two families of methods:

    * **Unsupervised** (``SampleRepresentationMethod`` subclasses — Pseudobulk,
      CellGroupComposition, GroupedPseudobulk, …): run on the *full* AnnData
      (transductive), then split the resulting ``sample_representation`` into
      train / val / test by the split column.

    * **Supervised-as-feature-extractor** (``SupervisedSampleMethod`` subclasses
      that implement :meth:`get_sample_representations` — MixMIL, PULSAR, …):
      fit on the training partition only, then apply the frozen model to val/test
      to obtain their representations.

    In both cases a ``LogisticRegression`` (classification) or ``Ridge``
    (regression) probe is fitted on train representations and evaluated on
    val / test using the same metrics as :class:`MILBenchmark`.

    Parameters
    ----------
    models : dict[str, object]
        ``{name: model_instance}`` mapping.  Models must not yet be fitted.
    sample_key, label_keys, tasks, n_splits, split_col, test_size, val_size,
    stratify_by, seed
        Same semantics as :class:`MILBenchmark`.
    """

    def __init__(
        self,
        models: dict[str, object],
        sample_key: str,
        label_keys: list[str],
        tasks: list[_PREDICTION_TASKS],
        n_splits: int = 3,
        split_col: str = "split",
        test_size: float = 0.2,
        val_size: float = 0.15,
        stratify_by: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.models = models
        self.sample_key = sample_key
        self.label_keys = [label_keys] if isinstance(label_keys, str) else label_keys
        self.tasks = [tasks] if isinstance(tasks, str) else tasks
        self.n_splits = n_splits
        self.split_col = split_col
        self.test_size = test_size
        self.val_size = val_size
        self.stratify_by = stratify_by
        self.seed = seed
        self._results: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, adata: sc.AnnData) -> pd.DataFrame:
        """Fit all models across all splits and return a tidy results DataFrame."""
        self._results = []
        adata = self._ensure_splits(adata)
        col_names = (
            [f"{self.split_col}_{i}" for i in range(self.n_splits)]
            if self.n_splits > 1
            else [self.split_col]
        )
        for split_i, col in enumerate(col_names):
            logger.info("RepBenchmark split %d/%d (col='%s')", split_i + 1, self.n_splits, col)
            self._run_one_split(adata, col, split_i)
        return pd.DataFrame(self._results)

    def summary(self) -> pd.DataFrame:
        """Mean ± std across splits per model × label × metric."""
        if not self._results:
            raise RuntimeError("No results yet. Call run() first.")
        df = pd.DataFrame(self._results)
        return (
            df.groupby(["model", "eval_split", "label", "metric"])["value"]
            .agg(["mean", "std"])
            .round(4)
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_splits(self, adata: sc.AnnData) -> sc.AnnData:
        from patpy.pp.splits import make_sample_splits

        col_names = (
            [f"{self.split_col}_{i}" for i in range(self.n_splits)]
            if self.n_splits > 1
            else [self.split_col]
        )
        missing = [c for c in col_names if c not in adata.obs.columns]
        if not missing:
            return adata
        make_sample_splits(
            adata,
            sample_key=self.sample_key,
            label_key=self.label_keys[0],
            test_size=self.test_size,
            val_size=self.val_size,
            covariate_keys=self.stratify_by or None,
            n_splits=self.n_splits,
            seed=self.seed,
            split_col=self.split_col,
        )
        return adata

    def _run_one_split(self, adata: sc.AnnData, col: str, split_i: int) -> None:
        from sklearn.linear_model import LogisticRegression, Ridge

        sample_split = adata.obs.groupby(self.sample_key)[col].first()
        train_ids = sample_split[sample_split == "train"].index.tolist()
        val_ids   = sample_split[sample_split == "val"].index.tolist()
        test_ids  = sample_split[sample_split == "test"].index.tolist()

        train_adata = adata[adata.obs[col] == "train"].copy()
        if train_adata.n_obs == 0:
            warnings.warn(f"No training cells in split column '{col}'.", stacklevel=2)
            return

        sample_label_map = {
            lk: adata.obs.groupby(self.sample_key)[lk].first().dropna()
            for lk in self.label_keys
        }

        for model_name, model in self.models.items():
            try:
                train_rep, eval_reps = self._get_representations(
                    model, adata, train_adata, train_ids, val_ids, test_ids, col
                )
            except Exception:
                logger.error(
                    "Model '%s' failed getting representations on split %d:\n%s",
                    model_name, split_i, traceback.format_exc(),
                )
                print(f"[rep_benchmark] ERROR — {model_name} split {split_i}: representation failed")
                continue

            for label, task in zip(self.label_keys, self.tasks):
                y_all = sample_label_map[label]

                y_train = y_all.reindex(train_rep.index).dropna()
                X_train = train_rep.loc[y_train.index].values

                if len(np.unique(y_train)) < 2:
                    continue

                if task == "classification":
                    probe = LogisticRegression(max_iter=1000, class_weight="balanced")
                else:
                    probe = Ridge(alpha=0.1)

                try:
                    probe.fit(X_train, y_train.values)
                except Exception:
                    logger.error("Probe fit failed for %s/%s split %d:\n%s",
                                 model_name, label, split_i, traceback.format_exc())
                    continue

                for eval_split, eval_rep in eval_reps.items():
                    y_eval = y_all.reindex(eval_rep.index).dropna()
                    if y_eval.empty:
                        continue
                    X_eval = eval_rep.loc[y_eval.index].values

                    try:
                        preds = self._probe_predict(probe, X_eval, task, label, y_eval.index)
                        metrics = self._compute_metrics(y_eval.values, preds, label, task)
                    except Exception:
                        logger.error("Eval failed for %s/%s/%s split %d:\n%s",
                                     model_name, label, eval_split, split_i, traceback.format_exc())
                        continue

                    for metric_name, value in metrics.items():
                        self._results.append({
                            "model": model_name,
                            "split_idx": split_i,
                            "eval_split": eval_split,
                            "label": label,
                            "metric": metric_name,
                            "value": value,
                        })

    def _get_representations(
        self,
        model,
        full_adata: sc.AnnData,
        train_adata: sc.AnnData,
        train_ids: list,
        val_ids: list,
        test_ids: list,
        split_col: str,
    ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        """Return (train_rep_df, {"val": val_rep_df, "test": test_rep_df}).

        Both DataFrames are indexed by sample ID.
        """
        from patpy.tl.sample_representation import SampleRepresentationMethod

        if isinstance(model, SampleRepresentationMethod):
            # Transductive: run on full adata, no label leakage (unsupervised)
            model.prepare_anndata(full_adata)
            model.calculate_distance_matrix(force=True)
            rep = model.sample_representation
            if rep is None:
                raise RuntimeError(f"{type(model).__name__}.sample_representation is None after calculate_distance_matrix().")
            # Normalise to DataFrame indexed by sample ID
            if isinstance(rep, pd.DataFrame):
                rep_df = rep
            elif rep.ndim == 3:
                # GroupedPseudobulk: (n_cell_groups, n_samples, n_features) → mean over cell groups
                rep_df = pd.DataFrame(
                    np.nanmean(rep, axis=0),
                    index=model.samples,
                )
            else:
                rep_df = pd.DataFrame(rep, index=model.samples)

            train_rep = rep_df.loc[[s for s in train_ids if s in rep_df.index]]
            val_rep   = rep_df.loc[[s for s in val_ids   if s in rep_df.index]]
            test_rep  = rep_df.loc[[s for s in test_ids  if s in rep_df.index]]

        else:
            # Inductive: fit on train, apply frozen model to val/test
            model.prepare_anndata(train_adata)
            train_rep = _as_rep_df(model.get_sample_representations())

            val_adata  = full_adata[full_adata.obs[split_col] == "val"].copy()
            test_adata = full_adata[full_adata.obs[split_col] == "test"].copy()

            # Swap adata to run trained model on eval data without retraining
            SupervisedSampleMethod.prepare_anndata(model, val_adata)
            val_rep = _as_rep_df(model.get_sample_representations())

            SupervisedSampleMethod.prepare_anndata(model, test_adata)
            test_rep = _as_rep_df(model.get_sample_representations())

        return train_rep, {"val": val_rep, "test": test_rep}

    @staticmethod
    def _probe_predict(
        probe,
        X: np.ndarray,
        task: str,
        label: str,
        index: pd.Index,
    ) -> pd.Series | pd.DataFrame:
        if task == "classification":
            proba = probe.predict_proba(X)
            classes = probe.classes_
            result = pd.DataFrame(
                {f"prob_{c}": proba[:, i] for i, c in enumerate(classes)},
                index=index,
            )
            result[f"{label}_pred"] = probe.predict(X)
            return result
        else:
            return pd.Series(probe.predict(X), index=index, name=label)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        preds: pd.Series | pd.DataFrame,
        label: str,
        task: str,
    ) -> dict[str, float]:
        if len(np.unique(y_true)) < 2:
            return {}
        if task == "classification":
            if not isinstance(preds, pd.DataFrame) or f"{label}_pred" not in preds.columns:
                return {}
            return _eval_classification(y_true, preds, label)
        else:
            y_pred = preds.values if isinstance(preds, pd.Series) else preds.iloc[:, 0].values
            return _eval_regression(y_true.astype(float), y_pred.astype(float))


def _as_rep_df(rep) -> pd.DataFrame:
    """Ensure a representation is a plain DataFrame with sample IDs as index."""
    if isinstance(rep, pd.DataFrame):
        return rep
    return pd.DataFrame(rep)


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def run_mil_benchmark(
    adata: sc.AnnData,
    sample_key: str,
    label_keys: list[str],
    tasks: list[_PREDICTION_TASKS],
    *,
    models: dict[str, SupervisedSampleMethod] | None = None,
    layer: str = "X_pca",
    split_col: str = "split",
    test_size: float = 0.2,
    val_size: float = 0.15,
    stratify_by: list[str] | None = None,
    n_splits: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a MIL benchmark on *adata* and return a tidy results DataFrame.

    A convenience wrapper around :class:`MILBenchmark`.  When *models* is
    ``None``, a default suite of torchmil models (ABMIL, TransMIL, DSMIL) is
    used together with MixMIL.

    Parameters
    ----------
    adata
        Single-cell AnnData.
    sample_key
        Column in ``adata.obs`` with donor identifiers.
    label_keys
        Donor-level labels to evaluate (e.g. ``["disease", "age"]``).
    tasks
        Prediction task per label.
    models
        Named model instances.  When ``None``, a default suite is used.
    layer
        Feature key in ``adata.obsm`` (used for default models).
    split_col
        Column for train/val/test labels.
    test_size, val_size
        Fractions for held-out sets.
    stratify_by
        Extra covariate columns for balanced splits.
    n_splits
        Number of independent random splits to average over.
    seed
        Base random seed.

    Returns
    -------
    pd.DataFrame
        Tidy results table.  See :meth:`MILBenchmark.run`.

    Examples
    --------
    >>> results = run_mil_benchmark(
    ...     adata,
    ...     sample_key="donor_id",
    ...     label_keys=["disease"],
    ...     tasks=["classification"],
    ...     n_splits=5,
    ... )
    >>> results.groupby(["model", "metric"])["value"].mean().unstack()
    """
    if models is None:
        models = _default_models(sample_key, label_keys, tasks, layer)

    bench = MILBenchmark(
        models=models,
        sample_key=sample_key,
        label_keys=label_keys,
        tasks=tasks,
        split_col=split_col,
        test_size=test_size,
        val_size=val_size,
        stratify_by=stratify_by,
        n_splits=n_splits,
        seed=seed,
    )
    return bench.run(adata)


def _default_models(
    sample_key: str,
    label_keys: list[str],
    tasks: list[str],
    layer: str,
) -> dict[str, SupervisedSampleMethod]:
    """Build a default model suite for quick benchmarking."""
    from patpy.tl.mil_models import ABMIL, DSMIL, TransMIL
    from patpy.tl.supervised import MixMIL

    common = {
        "sample_key": sample_key,
        "label_keys": label_keys,
        "tasks": tasks,
        "layer": layer,
    }
    suite: dict[str, SupervisedSampleMethod] = {
        "MixMIL": MixMIL(**common),
        "ABMIL": ABMIL(**common),
    }
    # TransMIL and DSMIL are added when torchmil is available
    try:
        _get_torchmil_class_soft("TransMIL")
        suite["TransMIL"] = TransMIL(**common)
        suite["DSMIL"] = DSMIL(**common)
    except ImportError:
        logger.debug("torchmil not available — skipping TransMIL and DSMIL in default suite.")
    return suite


def _get_torchmil_class_soft(name: str):
    try:
        import torchmil.models as _tm

        return getattr(_tm, name)
    except ImportError as e:
        raise ImportError from e
