from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc

from patpy.tl._base_sample_method import BaseSampleMethod
from patpy.tl._types import _PREDICTION_TASKS

logger = logging.getLogger(__name__)


class SupervisedSampleMethod(BaseSampleMethod):
    """Base class for supervised sample-level representation methods.

    Subclasses must implement :meth:`prepare_anndata`,
    :meth:`get_sample_importance`, and :meth:`get_cell_importance`.
    :meth:`get_sample_representation` is optional. It can be overwritten when the model
    produces a meaningful latent donor vector.

    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` with donor / sample identifiers.
    label_keys : list[str]
        Columns in ``adata.obs`` with donor-level supervision targets
        (e.g. ``["disease_status", "age"]``).  Every cell belonging to a
        given donor must carry the same value for each label key.
        When a method does not support multi-label training, pass a
        single-element list; a warning is raised if more than one label is
        provided in that case.
    tasks : list[_PREDICTION_TASKS]
        Prediction task for each entry in *label_keys*.  Must be the same
        length as *label_keys*.  Valid values: ``"classification"``,
        ``"regression"``, ``"ranking"``.
    cell_group_key : str or None, optional
        Column in ``adata.obs`` with cell-type annotations.  Used by
        plotting helpers and subclasses that stratify by cell type.
    layer : str, default ``"X_pca"``
        Key in ``adata.obsm`` (or ``adata.layers``) to use as per-cell
        features.  If the key is absent when :meth:`prepare_anndata` is
        called, subclasses should compute an appropriate fallback (e.g. PCA).
    seed : int, default 42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        seed: int = 42,
    ):
        super().__init__(
            sample_key=sample_key,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )

        # Convert strings to lists
        label_keys = [label_keys] if isinstance(label_keys, str) else label_keys
        tasks = [tasks] if isinstance(tasks, str) else tasks

        if len(label_keys) != len(tasks):
            raise ValueError(
                f"label_keys (len={len(label_keys)}) and tasks (len={len(tasks)}) must have the same length."
            )

        self.label_keys = list(label_keys)
        self.tasks = list(tasks)
        self.labels: pd.DataFrame | None = None
        self._probes: dict[str, object] = {}  # label → fitted sklearn probe model
        self._label_mappings: dict[str, tuple[list, dict]] = {}  # label_key → (classes, encode_dict)

    def prepare_anndata(self, adata: sc.AnnData) -> None:
        """Validate *adata*, populate :attr:`samples` and :attr:`labels`.

        Subclasses must call ``super().prepare_anndata(adata)`` before
        performing method-specific training.

        Parameters
        ----------
        adata
            Single-cell AnnData.  Must contain :attr:`sample_key` and all
            entries of :attr:`label_keys` in ``.obs``.

        Sets
        ----
        self.labels : pd.DataFrame
            Shape ``(n_donors, n_label_keys)``, indexed by donor ID.  One column per
            label key.
        """
        super().prepare_anndata(adata)

        missing = [k for k in self.label_keys if k not in adata.obs.columns]
        if missing:
            raise ValueError(f"label_keys {missing} not found in adata.obs.")

        self.labels = self._extract_metadata(self.label_keys)

    def predict(self, label: str) -> pd.Series | pd.DataFrame:
        """Predict `label` for every sample using sklearn probes.

        This default implementation uses probes fitted by :meth:`fine_tune`.
        Subclasses with native prediction heads (e.g. MixMIL) should override
        both :meth:`predict` and :meth:`fine_tune`.

        Parameters
        ----------
        label: str
            Sample label to predict, such as "disease" or "age".
            Must be in `self.label_keys` either from initialisation or by
            calling a fine-tuning method prior to running prediction.

        Returns
        -------
        pd.Series or pd.DataFrame
            Predictions indexed by donor ID. Format depends on the task:
            - classification: DataFrame with probability columns + "{label}_pred" (argmax)
            - regression/ranking: Series of predicted values
        """
        self._check_fitted()
        if label not in self.label_keys:
            raise ValueError(
                f"`label='{label}'` is not found in model label keys. Please train the model to predict the label first"
            )

        if label not in self._probes:
            raise RuntimeError(f"No probe fitted for label '{label}'. Call fine_tune() first.")

        task = self.tasks[self.label_keys.index(label)]
        rep = self.get_sample_representations()
        X = rep.values
        probe = self._probes[label]

        if task == "classification":
            proba = probe.predict_proba(X)  # (n_donors, n_classes)
            classes = probe.classes_
            # Create probability columns for each class
            prob_cols = {f"prob_{c}": proba[:, i] for i, c in enumerate(classes)}
            result = pd.DataFrame(prob_cols, index=self.samples)
            # Add predicted class column
            y_pred = probe.predict(X)
            result[f"{label}_pred"] = y_pred
            return result
        else:
            return pd.Series(probe.predict(X), index=self.samples, name=label)

    def _prepare_fine_tune(
        self,
        labels: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
    ) -> tuple[list[str], list[_PREDICTION_TASKS]]:
        """Validate and register new labels/tasks for fine-tuning.

        Performs string-to-list conversion, adata check, length validation,
        label existence check, extending label_keys/tasks, re-extracting
        metadata, and clearing stale caches.

        Returns
        -------
        labels : list[str]
            Normalised list of label names.
        tasks : list[_PREDICTION_TASKS]
            Normalised list of task types.
        """
        labels = [labels] if isinstance(labels, str) else labels
        tasks = [tasks] if isinstance(tasks, str) else tasks

        self._check_adata_loaded()

        if len(labels) != len(tasks):
            raise ValueError(f"labels (len={len(labels)}) and tasks (len={len(tasks)}) must have the same length.")

        for label in labels:
            if label not in self.adata.obs.columns:
                raise ValueError(f"label '{label}' not found in adata.obs.columns.")

        for label, task in zip(labels, tasks, strict=True):
            if label not in self.label_keys:
                self.label_keys.append(label)
                self.tasks.append(task)

        self.labels = self._extract_metadata(self.label_keys)
        self.adata.uns.pop("supervised_sample_importance", None)

        return labels, tasks

    def fine_tune(self, labels: list[str] | str, tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS, **kwargs):
        """Fine-tune / continue training the model on new or existing labels.

        The default implementation fits sklearn linear probes on top of
        :meth:`get_sample_representations`.  Subclasses with native training
        (e.g. MixMIL) should override this method and call
        :meth:`_prepare_fine_tune` for shared validation and bookkeeping.

        Parameters
        ----------
        labels : list[str] or str
            Labels to train/extend the model for, e.g., ["disease", "age"] or "disease".
        tasks : list[_PREDICTION_TASKS] or _PREDICTION_TASKS
            Corresponding prediction task for each label. Must match the length
            of `labels`. One of ["classification", "regression", "ranking"].
        **kwargs
            Additional training parameters (e.g., n_epochs, lr) passed to subclass.

        Raises
        ------
        NotImplementedError
            Subclasses with non-embedding-based predictions must override this method.
        RuntimeError
            If called before :meth:`prepare_anndata`.
        """
        from sklearn.linear_model import LogisticRegression, Ridge

        labels, tasks = self._prepare_fine_tune(labels, tasks)

        try:
            rep = self.get_sample_representations()  # requires subclass to implement
        except NotImplementedError as e:
            raise NotImplementedError(
                f"{type(self).__name__} does not provide sample representations. "
                "fine_tune() requires get_sample_representations() to be implemented."
            ) from e

        X = rep.values

        for label, task in zip(labels, tasks, strict=True):
            y = self.labels.loc[self.samples, label].values
            if task == "classification":
                probe = LogisticRegression(max_iter=1000, class_weight="balanced")
            else:  # regression / ranking
                probe = Ridge(alpha=0.1)
            probe.fit(X, y)
            self._probes[label] = probe

    def get_sample_importance(self, force: bool = False) -> pd.DataFrame:
        """Return per-donor prediction importances / posterior means.

        Results are cached in ``adata.obsm["supervised_sample_importance"]``
        after the first call.

        Parameters
        ----------
        force
            Recompute even if cached results exist.

        Returns
        -------
        pd.DataFrame
            Indexed by donor ID.  One column per label key, named
            ``"<label_key>_score"``.  An additional ``"average_importance"``
            column is added when more than one label is present.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        RuntimeError
            If called before :meth:`prepare_anndata`.
        """
        self._check_adata_loaded()

        cache_key = "supervised_sample_importance"
        if not force and self.adata is not None and cache_key in self.adata.uns:
            return pd.DataFrame(self.adata.uns[cache_key], index=self.samples)

        raise NotImplementedError(f"{type(self).__name__} must implement get_sample_importance().")

    def get_cell_importance(self, force: bool = False) -> pd.DataFrame:
        """Return per-cell attention or contribution scores.

        Results are cached in ``adata.obs`` under
        ``"<label_key>_importance"`` columns after the first call.

        Parameters
        ----------
        force
            Recompute even if cached results exist.

        Returns
        -------
        pd.DataFrame
            Indexed by ``adata.obs_names``.  One column per label key,
            named ``"<label_key>_importance"``.  An additional
            ``"average_importance"`` column is added when more than one
            label is present.  Higher values indicate stronger contribution
            to the donor-level prediction.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        RuntimeError
            If called before :meth:`prepare_anndata`.
        """
        self._check_adata_loaded()

        importance_cols = [f"{k}_importance" for k in self.label_keys]
        if not force and all(c in self.adata.obs.columns for c in importance_cols):
            return self.adata.obs[importance_cols]

        raise NotImplementedError(f"{type(self).__name__} must implement get_cell_importance().")

    def get_sample_representations(self) -> pd.DataFrame:
        """Return latent donor-level embeddings (optional).

        Not all supervised methods produce a meaningful latent vector per
        donor.  Override this method when they do.

        Returns
        -------
        pd.DataFrame
            Indexed by donor ID with embedding dimensions as columns
            (``"dim_0"``, ``"dim_1"``, …).

        Raises
        ------
        NotImplementedError
            Default implementation — override in subclasses that provide
            donor embeddings.
        """
        raise NotImplementedError(f"{type(self).__name__} does not provide sample representations / embeddings.")

    def calculate_distance_matrix(self, dist: str = "euclidean") -> np.ndarray:
        """Compute a donor x donor distance matrix from sample representations.

        Requires :meth:`get_sample_representations` to be implemented.

        Parameters
        ----------
        dist
            Distance metric passed to ``scipy.spatial.distance.pdist``.

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape ``(n_donors, n_donors)``.
        """
        import scipy.spatial.distance

        representations = self.get_sample_representations()
        distances = scipy.spatial.distance.pdist(representations.values, metric=dist)
        return scipy.spatial.distance.squareform(distances)

    def _donor_col(self, col: str) -> np.ndarray:
        """Return a per-donor array for *col*, aligned to :attr:`samples`."""
        self._check_adata_loaded()
        return self._extract_metadata(columns=[col]).loc[self.samples].values.ravel()

    def _build_label_mappings(self) -> None:
        """Build and store mappings for string labels.

        For each label key whose donor-level column is non-numeric (strings,
        objects), a mapping from unique sorted class names to integer indices is
        stored in :attr:`_label_mappings`.  Already-mapped labels are skipped.
        """
        for label_key in self.label_keys:
            if label_key in self._label_mappings:
                continue

            col = self._donor_col(label_key)
            if col.dtype.kind not in ("f", "i", "u"):  # not float/int/uint
                classes = sorted(np.unique(col))
                encode_dict = {c: i for i, c in enumerate(classes)}
                self._label_mappings[label_key] = (classes, encode_dict)


class MixMIL(SupervisedSampleMethod):
    """Attention-based multi-instance mixed model for donor phenotype prediction by Engelmann et al. 2024 (https://arxiv.org/abs/2311.02455).

    MixMIL frames donor-level prediction as a multi-instance learning problem.
    Each donor is a "bag" of cells; the model learns which cells are most
    informative for the prediction target via an attention mechanism.

    .. note::
        MixMIL supports a **single label** per training run.  When
        *label_keys* contains more than one entry, a warning is raised and
        only the first entry is used for training.

    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` with donor / sample identifiers.
    label_keys : list[str]
        Donor-level target columns.  All labels are used jointly during training.
    tasks : list[_PREDICTION_TASKS]
        Prediction task for each label key.
    cell_group_key : str or None, optional
        Column in ``adata.obs`` with cell-type annotations.
    layer : str, default ``"X_pca"``
        Key in ``adata.obsm`` to use as per-cell bag features.
    likelihood : {"binomial", "categorical"} or None, default ``"binomial"``
        Output likelihood for the MixMIL model.
    n_trials : int or None, default 2
        Number of binomial trials.  Only used when
        ``likelihood="binomial"``.
    n_epochs : int, default 2000
        Training epochs.
    batch_size : int, default 64
        Mini-batch size.
    lr : float, default 1e-3
        Adam learning rate.
    additional_covariates : list[str] or None, optional
        Extra covariate columns from ``adata.obs`` (numeric) or keys from
        ``adata.obsm`` to include as fixed effects.
    dtype : str, default ``"float32"``
        Floating-point precision for all tensors.
    seed : int, default 67
        Random seed.

    Examples
    --------
    >>> model = MixMIL(
    ...     sample_key="donor_id",
    ...     label_keys=["disease_status"],
    ...     tasks=["classification"],
    ...     layer="X_pca",
    ... )
    >>> model.prepare_anndata(adata)
    >>> scores = model.get_sample_importance()
    >>> importance = model.get_cell_importance()
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str],
        tasks: list[_PREDICTION_TASKS],
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        likelihood: Literal["binomial", "categorical"] | None = "binomial",
        n_trials: int | None = 2,
        n_epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        additional_covariates: list[str] | None = None,
        dtype: str = "float32",
        seed: int = 67,
    ) -> None:
        super().__init__(
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )

        self.likelihood = likelihood
        self.n_trials = n_trials
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.additional_covariates = additional_covariates or []
        self.dtype = dtype

        self._model = None
        self._history: list | None = None
        self._label_dim_slices: dict[str, slice] = {}  # label_key → slice into Y/logits columns

    def prepare_anndata(self, adata: sc.AnnData, train: bool = True, **kwargs) -> None:
        """Train MixMIL on *adata*.

        Parameters
        ----------
        adata
            Single-cell AnnData.  Must contain :attr:`sample_key` and
            :attr:`label_keys` in ``.obs``, and per-cell features under
            :attr:`layer` in ``.obsm``.
        train: bool = True
            If True, train the model on loaded data for tasks and labels set at initialisation
        """
        super().prepare_anndata(adata)

        if self.layer not in adata.obsm and self.layer not in adata.layers and self.layer not in ("X", None):
            raise ValueError(
                f"layer='{self.layer}' not found in adata.obsm or adata.layers. "
                "Please compute the required embedding before calling prepare_anndata()."
            )

        if train:
            self.fine_tune(self.label_keys, self.tasks, **kwargs)

    def fine_tune(
        self,
        labels: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        n_epochs: int | None = None,
        lr: float | None = None,
        **kwargs,
    ) -> None:
        """Fine-tune or continue training MixMIL on new or existing labels.

        The model is extended to predict the given labels (which may be a superset of existing labels).
        Existing learned parameters are preserved via warm-start when the output dimension changes.

        Parameters
        ----------
        labels : list[str] or str
            Labels to train/extend the model for, e.g., ["disease", "age"] or "disease".
        tasks : list[_PREDICTION_TASKS] or _PREDICTION_TASKS
            Corresponding prediction task for each label. Must match the length of `labels`.
            One of ["classification", "regression", "ranking"] or a single task string.
        n_epochs : int, optional
            Number of training epochs. Defaults to `self.n_epochs` if not provided.
        lr : float, optional
            Learning rate. Defaults to `self.lr` if not provided.
        **kwargs
            Additional arguments (unused; for API consistency).

        Raises
        ------
        ValueError
            If adding regression/ranking to a model trained only on classification
            (MixMIL currently supports a single task type).
        """
        import torch

        try:
            from mixmil import MixMIL as _MixMIL
        except ImportError as e:
            raise ImportError("mixmil is required. Install with: pip install mixmil") from e

        tasks = [tasks] if isinstance(tasks, str) else tasks
        all_tasks = self.tasks + tasks
        if len(set(all_tasks)) > 1:
            raise ValueError(
                f"MixMIL supports a single task type per model. "
                f"Cannot mix {set(all_tasks)}. "
                f"Current tasks: {self.tasks}. New tasks: {tasks}."
            )

        # Store old P and old labels/tasks for weight transfer
        P_old = len(self.label_keys) if self._model is not None else None
        old_label_keys = self.label_keys.copy()

        labels, tasks = self._prepare_fine_tune(labels, tasks)

        # Build mappings for string labels
        self._build_label_mappings()

        # Clear stale cell-importance caches
        importance_cols = [f"{k}_importance" for k in self.label_keys]
        for col in importance_cols:
            if col in self.adata.obs.columns:
                del self.adata.obs[col]

        # Rebuild tensors with updated label_keys
        Xs, F, Y = self._build_tensors()
        P_new = Y.shape[1]

        # Check if we're fine-tuning with the exact same labels/tasks (no new labels added)
        new_labels_added = (set(labels) > set(old_label_keys)) or (P_old != P_new)

        # If fine-tuning with same labels/tasks, continue training existing model
        if new_labels_added or self._model is None:
            # Initialize new model (either first time or adding new labels)
            if self.likelihood is not None:
                new_model = _MixMIL.init_with_mean_model(
                    Xs,
                    F,
                    Y,
                    likelihood=self.likelihood,
                    n_trials=self.n_trials,
                )
            else:
                Q = Xs[0].shape[1]
                K = F.shape[1]
                new_model = _MixMIL(Q=Q, K=K, P=P_new)

            # Transfer weights from old model if it exists and P changed
            if self._model is not None and P_old != P_new:
                p = min(P_old, P_new)
                with torch.no_grad():
                    new_model.posterior.q_mu[:, :p] = self._model.posterior.q_mu[:, :p]
                    new_model.alpha[:, :p] = self._model.alpha[:, :p]
                    new_model.log_sigma_u[:, :p] = self._model.log_sigma_u[:, :p]
                    new_model.log_sigma_z[:, :p] = self._model.log_sigma_z[:, :p]

            self._model = new_model

        n_epochs_use = n_epochs if n_epochs is not None else self.n_epochs
        lr_use = lr if lr is not None else self.lr

        new_history = self._model.train(
            Xs,
            F,
            Y,
            n_epochs=n_epochs_use,
            batch_size=self.batch_size,
            lr=lr_use,
        )

        if self._history is None:
            self._history = new_history
        else:
            self._history.extend(new_history)
        self._fitted = True

    def predict(self, label: str) -> pd.Series | pd.DataFrame:
        """Predict the given label for all samples.

        Parameters
        ----------
        label : str
            The label to predict. Must be in `self.label_keys`.

        Returns
        -------
        pd.Series or pd.DataFrame
            Predictions indexed by donor ID. Format depends on the task:
            - classification: DataFrame with probability columns (one per class) + "{label}_pred" (argmax)
            - regression/ranking: Series of predicted values

        Examples
        --------
        >>> preds = model.predict("disease")  # classification → DataFrame
        >>> ages = model.predict("age")  # regression → Series
        """
        import torch

        self._check_fitted()
        if label not in self.label_keys:
            raise ValueError(f"`label='{label}'` is not found in model label keys.")

        task = self.tasks[self.label_keys.index(label)]

        # Build tensors for inference
        Xs, F, _ = self._build_tensors()

        with torch.no_grad():
            logits = self._model.predict(Xs)  # (n_donors, P)

        logits_np = logits.numpy()

        # Extract logits for this label using precomputed dimension slices
        label_logits = logits_np[:, self._label_dim_slices[label]]

        if task == "classification":
            # Apply softmax to get probabilities
            import scipy.special

            if label_logits.shape[1] == 1:
                # Binary: apply sigmoid to single logit
                proba_positive = scipy.special.expit(label_logits.ravel())
                proba = np.column_stack([1 - proba_positive, proba_positive])
            else:
                # Multiclass: apply softmax
                proba = scipy.special.softmax(label_logits, axis=1)

            # Get class labels
            if label in self._label_mappings:
                classes, _ = self._label_mappings[label]
                classes_list = classes
            else:
                classes_list = list(range(proba.shape[1]))

            # Create DataFrame with probability columns and predicted class
            prob_cols = {f"prob_{c}": proba[:, i] for i, c in enumerate(classes_list)}
            result = pd.DataFrame(prob_cols, index=self.samples)

            # Add predictions
            y_pred_indices = proba.argmax(axis=1)
            if label in self._label_mappings:
                result[f"{label}_pred"] = [classes_list[idx] for idx in y_pred_indices]
            else:
                result[f"{label}_pred"] = y_pred_indices

            return result
        else:
            # Regression or ranking
            raw = label_logits[:, 0] if label_logits.shape[1] > 0 else label_logits.ravel()
            return pd.Series(raw, index=self.samples, name=label)

    def get_sample_importance(self, force: bool = False) -> pd.DataFrame:
        """Per-donor posterior predictions from MixMIL.

        Parameters
        ----------
        force
            Recompute even if cached results exist in ``adata.uns``.

        Returns
        -------
        pd.DataFrame
            Indexed by donor ID.  One column per label key named
            ``"<label_key>_importance"``.  An additional
            ``"average_importance"`` column is added for multi-label models.

        Examples
        --------
        >>> scores = model.get_sample_importance()
        """
        self._check_adata_loaded()

        cache_key = "supervised_sample_importance"
        if not force and cache_key in self.adata.uns:
            return pd.DataFrame(self.adata.uns[cache_key], index=self.samples)

        import torch

        Xs, F, Y = self._build_tensors()

        with torch.no_grad():
            preds = self._model.predict(Xs).numpy()  # (n_donors, P)

        preds = preds.reshape(len(self.samples), -1)  # ensure 2-D

        if preds.shape[1] == 1:
            sample_importance = pd.DataFrame(
                {f"{self.label_keys[0]}_importance": preds.ravel()},
                index=self.samples,
            )
        else:
            sample_importance = pd.DataFrame(
                {f"{key}_importance": preds[:, p] for p, key in enumerate(self.label_keys)},
                index=self.samples,
            )
            sample_importance.insert(0, "average_importance", preds.mean(axis=1))

        self.adata.uns[cache_key] = sample_importance.to_dict()
        return sample_importance

    def get_cell_importance(self, label: str | None = None, force: bool = False) -> pd.DataFrame:
        """Per-cell attention weights from the trained MixMIL model.

        Weights within each donor's bag are softmax-normalised and therefore
        sum to 1 per donor.  For multi-label models, weights are computed
        separately per output; use *label* to select which output to return.

        Parameters
        ----------
        label
            Which label's attention weights to return.  Defaults to
            ``label_keys[0]``.  Must be one of :attr:`label_keys`.
        force
            Recompute even if cached results exist in ``adata.obs``.

        Returns
        -------
        pd.DataFrame
            Indexed by ``adata.obs_names``.  Column
            ``"<label>_importance"``.

        Examples
        --------
        >>> importance = model.get_cell_importance()
        >>> importance = model.get_cell_importance(label="disease")
        >>> adata.obs["importance"] = importance["disease_importance"].values
        >>> sc.pl.umap(adata, color="importance")
        """
        self._check_adata_loaded()

        if label is None:
            label = self.label_keys[0]
        elif label not in self.label_keys:
            raise ValueError(f"label='{label}' is not in label_keys={self.label_keys}.")

        label_idx = self.label_keys.index(label)
        importance_col = f"{label}_importance"

        if not force and importance_col in self.adata.obs.columns:
            return self.adata.obs[[importance_col]]

        import torch

        Xs, _, unsort_idx = self._build_bags()

        with torch.no_grad():
            w, _ = self._model.get_weights(Xs)

        # w is a list of (n_cells_i, P) tensors; concatenate and unsort
        w_cat = torch.cat(w, dim=0).numpy()  # (n_cells, P)
        w_cat = w_cat[unsort_idx]

        # Extract the column corresponding to the requested label
        if w_cat.ndim == 1 or w_cat.shape[1] == 1:
            cell_weights = w_cat.ravel()
        else:
            cell_weights = w_cat[:, label_idx]

        cell_importances = pd.DataFrame(
            {importance_col: cell_weights},
            index=self.adata.obs_names,
        )
        self.adata.obs[importance_col] = cell_weights
        return cell_importances

    def get_sample_representations(self) -> pd.DataFrame:
        """Per-donor latent embeddings (attention-weighted mean of cell features).

        The embedding is computed as the attention-weighted mean of the
        per-cell bag features — analogous to the ``qu_mu`` (random-effect
        mean) in MixMIL notation.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_donors, Q)``, indexed by donor ID.
            Columns are named ``"dim_0"``, ``"dim_1"``, …

        Examples
        --------
        >>> representations = model.get_sample_representations()
        >>> distances = model.calculate_distance_matrix()
        """
        self._check_adata_loaded()
        import torch

        Xs, _, _ = self._build_bags()

        with torch.no_grad():
            w, _ = self._model.get_weights(Xs)

        embeddings = []
        for bag_X, bag_w in zip(Xs, w, strict=True):
            # bag_w: (n_cells, P) — average over outputs to get a scalar weight per cell
            w_scalar = bag_w.mean(dim=1, keepdim=True) if bag_w.ndim > 1 else bag_w.unsqueeze(1)
            emb = (w_scalar * bag_X).sum(dim=0).numpy()
            embeddings.append(emb)

        embeddings_arr = np.stack(embeddings)
        cols = [f"dim_{i}" for i in range(embeddings_arr.shape[1])]
        self.sample_representation = pd.DataFrame(embeddings_arr, index=self.samples, columns=cols)

        return self.sample_representation

    @property
    def training_history(self) -> list | None:
        """List of per-step loss dicts returned by MixMIL training."""
        return self._history

    def _build_bags(self):
        """Sort cells by donor and return per-donor bag tensors.

        Returns
        -------
        Xs : list[torch.Tensor]
            One tensor per donor containing that donor's cell embeddings.
        categories : np.ndarray
            Donor IDs in sorted (alphabetical) order, aligned to *Xs*.
        unsort_idx : np.ndarray
            Index array to reverse the sort (restore original cell order).
        """
        import torch

        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        unsort_idx = np.argsort(sort_idx)
        X_sorted = self._get_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]

        cat = pd.Categorical(sorted_donors)
        indptr = np.concatenate([[0], np.bincount(cat.codes).cumsum()])

        Xs = [torch.from_numpy(X_sorted[s:e]) for s, e in zip(indptr[:-1], indptr[1:], strict=True)]
        return Xs, np.array(cat.categories), unsort_idx

    def _build_tensors(self):
        """Build ``Xs``, ``F``, ``Y`` tensors aligned to :attr:`samples`.

        Returns
        -------
        Xs : list[torch.Tensor]
            Per-donor bags of cell embeddings.
        F : torch.Tensor
            Fixed-effect covariate matrix of shape ``(n_donors, K)``.
        Y : torch.Tensor
            Target variable of shape ``(n_donors, P)`` where P is the number of label keys.
        """
        import torch

        dtype_torch = getattr(torch, self.dtype)

        Xs, categories, _ = self._build_bags()
        # Preserve the donor order implied by pd.Categorical (alphabetical)
        # and align self.samples to it so scores/importances line up
        self.samples = categories

        n_donors = len(self.samples)
        covariate_list = [torch.ones((n_donors, 1), dtype=dtype_torch)]

        for cov in self.additional_covariates:
            if cov in self.adata.obs.columns:
                vals = self._donor_col(cov).astype("float32")
                covariate_list.append(torch.tensor(vals, dtype=dtype_torch).unsqueeze(1))
            elif cov in self.adata.obsm:
                # obsm covariate — must already be donor-level (n_donors × d)
                vals = self.adata.obsm[cov].astype("float32")
                covariate_list.append(torch.from_numpy(vals))
            else:
                raise ValueError(f"additional_covariates entry '{cov}' not found in adata.obs or adata.obsm.")

        F = torch.cat(covariate_list, dim=1)

        # Stack all label columns into (n_donors, P)
        # Encode string labels to integers using mappings
        # For multiclass labels (classification task with >2 classes), use one-hot encoding
        y_cols_list = []
        dim_offset = 0
        for key in self.label_keys:
            col = self._donor_col(key)
            task = self.tasks[self.label_keys.index(key)]

            # Recode strings to integers if needed
            if key in self._label_mappings:
                _, encode_dict = self._label_mappings[key]
                col_encoded = np.array([encode_dict[val] for val in col], dtype="int32")
            else:
                col_encoded = col.astype("int32")

            # Check if multiclass classification (>2 unique values in classification task)
            n_classes = len(np.unique(col_encoded))
            if task == "classification" and n_classes > 2:
                one_hot = np.eye(n_classes, dtype="float32")[col_encoded]
                y_cols_list.extend([one_hot[:, i] for i in range(n_classes)])
                n_dims = n_classes
            else:
                y_cols_list.append(col_encoded.astype("float32"))
                n_dims = 1

            self._label_dim_slices[key] = slice(dim_offset, dim_offset + n_dims)
            dim_offset += n_dims

        y_cols = np.column_stack(y_cols_list)
        Y = torch.tensor(y_cols, dtype=torch.float32)  # (n_donors, P)

        return Xs, F, Y


class PULSAR(SupervisedSampleMethod):
    """Donor-level representation via the PULSAR foundation model by Pang et al. 2025 (https://www.biorxiv.org/content/10.1101/2025.11.24.685470v1).

    PULSAR processes a bag of cell embeddings (e.g. UCE) through a
    transformer encoder and returns a CLS token as the donor embedding.
    This wrapper loads a pre-trained PULSAR model (from HuggingFace or a
    local path) and extracts donor-level representations following the
    patpy ``SupervisedSampleMethod`` interface.

    PULSAR is used **zero-shot** — it does not fine-tune on
    ``label_key``.  The label is used only for evaluation helpers inherited
    from ``SupervisedSampleMethod``.


    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` with donor / sample identifiers.
    label_keys : list[str]
        Donor-level label columns used for evaluation only — PULSAR is
        applied zero-shot.
    tasks : list[_PREDICTION_TASKS]
        Prediction task per label key (used only for evaluation).
    cell_group_key : str or None, optional
        Column in ``adata.obs`` with cell-type annotations.
    layer : str, default ``"X_uce"``
        Key in ``adata.obsm`` containing per-cell embeddings.  PULSAR
        expects high-dimensional embeddings such as UCE (~1280 dims).
    pretrained_model : str, default ``"KuanP/pulsar-pbmc"``
        HuggingFace model ID or local path.
    sample_cell_num : int, default 1024
        Cells to sample per donor during embedding extraction.
    batch_size : int, default 10
        Donors processed per forward pass.
    device : str, default ``"cuda"``
        Inference device.
    resample_num : int, default 1
        Stochastic resampling passes per donor (results are averaged).
    seed : int, default 67
        Random seed.

    Examples
    --------
    >>> model = PULSAR(
    ...     sample_key="donor_id",
    ...     label_keys=["age"],
    ...     tasks=["regression"],
    ...     layer="X_uce",
    ... )
    >>> model.prepare_anndata(adata)
    >>> embeddings = model.get_sample_representations()  # (n_donors, 512)
    >>> result = model.fit_linear_probe(task="regression")
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str],
        tasks: list[_PREDICTION_TASKS],
        cell_group_key: str | None = None,
        layer: str = "X_uce",
        pretrained_model: str = "KuanP/pulsar-pbmc",
        sample_cell_num: int = 1024,
        batch_size: int = 10,
        device: str = "cuda",
        resample_num: int = 1,
        seed: int = 67,
    ) -> None:
        super().__init__(
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )
        self.pretrained_model = pretrained_model
        self.sample_cell_num = sample_cell_num
        self.batch_size = batch_size
        self.device = device
        self.resample_num = resample_num

        self._pulsar_model = None
        self.sample_representation: pd.DataFrame | None = None

    def prepare_anndata(self, adata: sc.AnnData) -> None:
        """Load PULSAR model and extract donor-level CLS embeddings.

        Parameters
        ----------
        adata
            Must contain :attr:`sample_key` and all :attr:`label_keys` in
            ``.obs``, and per-cell embeddings under :attr:`layer` in
            ``.obsm``.
        """
        import torch

        try:
            from pulsar.model import PULSAR as _PulsarModel
            from pulsar.utils import extract_donor_embeddings_from_h5ad
        except ImportError as e:
            raise ImportError("pulsar is required. Install from: https://github.com/snap-stanford/PULSAR") from e

        super().prepare_anndata(adata)

        if self.layer not in adata.obsm:
            raise ValueError(
                f"layer='{self.layer}' not found in adata.obsm. "
                "PULSAR requires pre-computed high-dimensional cell embeddings "
                f"(e.g. UCE) in adata.obsm['{self.layer}']."
            )

        emb_dim = adata.obsm[self.layer].shape[1]
        if emb_dim < 256:
            warnings.warn(
                f"adata.obsm['{self.layer}'] has only {emb_dim} dimensions. "
                "PULSAR expects high-dimensional embeddings such as UCE (~1280 dims). "
                "Low-dimensional inputs like PCA will produce unreliable donor embeddings.",
                UserWarning,
                stacklevel=2,
            )

        # transformers >=5.x requires `all_tied_weights_keys` on PreTrainedModel subclasses;
        # PULSAR predates this requirement and has no tied weights, so patch if missing.
        if not hasattr(_PulsarModel, "all_tied_weights_keys"):
            _PulsarModel.all_tied_weights_keys = {}

        self._pulsar_model = _PulsarModel.from_pretrained(self.pretrained_model)

        self._pulsar_model.eval()

        try:
            self._pulsar_model = self._pulsar_model.to(self.device).to(torch.bfloat16)
        except RuntimeError as e:
            warnings.warn(
                f"Could not move model to device='{self.device}': {e}. Falling back to CPU.",
                stacklevel=2,
            )

            self.device = "cpu"
            self._pulsar_model = self._pulsar_model.to("cpu").to(torch.bfloat16)

        donor_embedding_collection = extract_donor_embeddings_from_h5ad(
            adata,
            model=self._pulsar_model,
            label_name=self.label_keys[0],
            donor_id_key=self.sample_key,
            embedding_key=self.layer,
            device=self.device,
            sample_cell_num=self.sample_cell_num,
            resample_num=self.resample_num,
            batch_size=self.batch_size,
            seed=self.seed,
        )

        donor_ids, embeddings = [], []
        for donor_id, data in donor_embedding_collection.items():
            donor_ids.append(donor_id)
            embeddings.append(np.mean(data["embedding"], axis=0))

        embeddings_arr = np.stack(embeddings)
        cols = [f"dim_{i}" for i in range(embeddings_arr.shape[1])]
        self.sample_representation = pd.DataFrame(embeddings_arr, index=donor_ids, columns=cols)
        self.samples = np.array(donor_ids)
        self._fitted = True

    def get_sample_representations(self) -> pd.DataFrame:
        """Per-donor CLS embeddings from PULSAR.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_donors, hidden_dim)`` (typically 512 for
            ``pulsar-pbmc``), indexed by donor ID.
            Columns are named ``"dim_0"``, ``"dim_1"``, …

        Examples
        --------
        >>> embeddings = model.get_sample_representations()
        >>> distances = model.calculate_distance_matrix()
        """
        self._check_adata_loaded()
        return self.sample_representation

    def get_sample_importance(self, force: bool = False) -> pd.DataFrame:
        """Per-donor scores derived from the L2 norm of CLS embeddings.

        Because PULSAR is zero-shot (no label-specific output head), the L2
        norm of the CLS embedding is exposed as a scalar score.  For
        downstream prediction tasks, use :meth:`get_sample_representations`
        and :meth:`fit_linear_probe`.

        Parameters
        ----------
        force
            Recompute even if cached results exist.

        Returns
        -------
        pd.DataFrame
            Indexed by donor ID.  Column ``"<label_key>_score"`` (the L2
            norm of the CLS embedding).

        Examples
        --------
        >>> scores = model.get_sample_importance()
        """
        self._check_adata_loaded()

        cache_key = "supervised_sample_importance"
        if not force and cache_key in self.adata.uns:
            return pd.DataFrame(self.adata.uns[cache_key], index=self.samples)

        norms = np.linalg.norm(self.sample_representation.values, axis=1)
        label = self.label_keys[0]
        sample_importance = pd.DataFrame(
            {f"{label}_importance": norms},
            index=self.sample_representation.index,
        )

        self.adata.uns[cache_key] = sample_importance.to_dict()

        return sample_importance

    def get_cell_importance(self, force: bool = False) -> pd.DataFrame:
        """Per-cell contribution scores as absolute cosine similarity to donor CLS.

        PULSAR does not expose explicit attention weights.  As a proxy, the
        absolute cosine similarity between each cell's raw embedding and the
        donor CLS vector is used.

        Parameters
        ----------
        force
            Recompute even if cached results exist in ``adata.obs``.

        Returns
        -------
        pd.DataFrame
            Indexed by ``adata.obs_names``.  Column
            ``"<label_key>_importance"``.

        Examples
        --------
        >>> importance = model.get_cell_importance()
        >>> adata.obs["importance"] = importance.values
        >>> sc.pl.umap(adata, color="importance")
        """
        self._check_adata_loaded()

        label = self.label_keys[0]
        importance_col = f"{label}_importance"

        if not force and importance_col in self.adata.obs.columns:
            return self.adata.obs[[importance_col]]

        cell_embeddings = self.adata.obsm[self.layer]
        donor_col = self.adata.obs[self.sample_key].values
        scores = np.zeros(len(self.adata))

        for donor_id in self.samples:
            mask = donor_col == donor_id
            if not mask.any() or donor_id not in self.sample_representation.index:
                continue

            cls_vec = self.sample_representation.loc[donor_id].values
            cells = cell_embeddings[mask]
            d = min(cells.shape[1], cls_vec.shape[0])

            dot = np.abs(cells[:, :d] @ cls_vec[:d])
            cell_norms = np.linalg.norm(cells[:, :d], axis=1) + 1e-8
            cls_norm = np.linalg.norm(cls_vec[:d]) + 1e-8
            scores[mask] = dot / (cell_norms * cls_norm)

        cell_importances = pd.DataFrame(
            {importance_col: scores},
            index=self.adata.obs_names,
        )
        self.adata.obs[importance_col] = scores
        return cell_importances


class PaSCient(SupervisedSampleMethod):
    """Patient-level representation via PaSCient by De Brouwer et al. 2025 (https://doi.org/10.1016/j.cels.2025.101195).

    PaSCient learns multi-cellular patient representations from single-cell
    transcriptomics data using a hierarchical encoder that processes gene
    expression through gene-to-cell, cell-to-cell, and cell-to-patient
    aggregation stages to produce a fixed-size patient embedding.

    This wrapper can either load a pre-trained PaSCient model from a
    checkpoint directory **or** train one from scratch on the provided
    AnnData.

    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` with donor / sample identifiers.
    label_keys : list[str]
        Donor-level target columns.  Used for training (when
        ``train=True``) and for evaluation.
    tasks : list[_PREDICTION_TASKS]
        Prediction task for each label key.
    cell_group_key : str or None, optional
        Column in ``adata.obs`` with cell-type annotations.
    layer : str or None, default ``None``
        Key in ``adata.layers`` containing raw counts or log-normalised
        expression.  If ``None``, ``adata.X`` is used.  PaSCient
        expects log-normalised expression; set ``normalize=True``
        (the default) to apply log-normalisation automatically.
    checkpoint_dir : str or None, optional
        Path to a PaSCient checkpoint directory.  Must contain
        ``.hydra/config.yaml`` and a checkpoint file under
        ``checkpoints/``.  When ``None``, a model is trained from
        scratch via :meth:`prepare_anndata` with ``train=True``.
    n_cells : int, default 1500
        Number of cells to sample per donor during training and
        inference.  Donors with fewer cells are zero-padded; donors
        with more are randomly subsampled.
    batch_size : int, default 16
        Donors processed per forward pass.
    device : str, default ``"cuda"``
        Device for training and inference.
    normalize : bool, default True
        Whether to apply log-normalisation (total-count normalisation
        with ``target_sum=1e4`` followed by ``log1p``) to the
        expression data.  Set to ``False`` if the data is already
        log-normalised.
    n_epochs : int, default 4
        Training epochs (only used when ``train=True``).
    lr : float, default 1e-4
        Learning rate (only used when ``train=True``).
    weight_decay : float, default 1e-4
        Weight decay (only used when ``train=True``).
    val_fraction : float, default 0.1
        Fraction of donors held out for validation during training.
    latent_dim : int, default 1024
        Cell embedding dimension (only used when training from scratch).
    patient_emb_dim : int, default 512
        Patient embedding dimension (only used when training from
        scratch).
    seed : int, default 12345
        Random seed.

    Examples
    --------
    Train from scratch:

    >>> model = PaSCient(
    ...     sample_key="donor_id",
    ...     label_keys=["disease"],
    ...     tasks=["classification"],
    ... )
    >>> model.prepare_anndata(adata, train=True)

    Load from checkpoint:

    >>> model = PaSCient(
    ...     sample_key="donor_id",
    ...     label_keys=["disease"],
    ...     tasks=["classification"],
    ...     checkpoint_dir="/path/to/pascient/checkpoint",
    ... )
    >>> model.prepare_anndata(adata)
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str],
        tasks: list[_PREDICTION_TASKS],
        cell_group_key: str | None = None,
        layer: str | None = None,
        checkpoint_dir: str | None = None,
        n_cells: int = 1500,
        batch_size: int = 16,
        device: str = "cuda",
        normalize: bool = True,
        n_epochs: int = 4,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.1,
        latent_dim: int = 1024,
        patient_emb_dim: int = 512,
        seed: int = 12345,
    ) -> None:
        super().__init__(
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )

        self.checkpoint_dir = checkpoint_dir
        self.n_cells = n_cells
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.latent_dim = latent_dim
        self.patient_emb_dim = patient_emb_dim

        self._pascient_model = None
        self._cell_embeddings: dict[str, np.ndarray] = {}  # donor_id → (n_cells, emb_dim)
        self.sample_representation: pd.DataFrame | None = None
        self._trained_label: str | None = None  # label_key used to train the native head
        self._class_names: list | None = None  # class names for classification decode

    @staticmethod
    def _load_pascient_model(config_path: str, checkpoint_path: str):
        """Load a PaSCient model from Hydra config and PyTorch checkpoint.

        Parameters
        ----------
        config_path
            Path to the ``.hydra/`` config directory inside the
            checkpoint directory.
        checkpoint_path
            Path to the ``.ckpt`` file.

        Returns
        -------
        torch.nn.Module
            PaSCient model in eval mode with loaded weights.
        """
        import sys

        import torch

        try:
            import pascient
        except ImportError as e:
            raise ImportError("pascient is required. Install with: pip install patpy[pascient]") from e

        # PaSCient was originally packaged as "cellm"; Hydra configs may
        # reference the old name, so register it as an alias.
        sys.modules.setdefault("cellm", pascient)

        try:
            import hydra
            from hydra import compose, initialize_config_dir
        except ImportError as e:
            raise ImportError(
                "hydra-core is required for PaSCient model loading. Install with: pip install patpy[pascient]"
            ) from e

        with initialize_config_dir(version_base=None, config_dir=config_path, job_name="patpy_pascient"):
            cfg = compose(
                config_name="config.yaml",
                return_hydra_config=True,
                overrides=[
                    "data.multiprocessing_context=null",
                    "data.batch_size=1",
                ],
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = hydra.utils.instantiate(cfg.get("metrics"))
        model = hydra.utils.instantiate(cfg.model, metrics=metrics)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        return model

    def _resolve_checkpoint_paths(self) -> tuple[str, str]:
        """Resolve config and checkpoint file paths from :attr:`checkpoint_dir`.

        Returns
        -------
        config_path : str
            Absolute path to the ``.hydra`` config directory.
        checkpoint_path : str
            Absolute path to the ``.ckpt`` checkpoint file.
        """
        import glob
        import os

        config_path = os.path.join(self.checkpoint_dir, ".hydra")
        if not os.path.isdir(config_path):
            raise FileNotFoundError(
                f"Hydra config directory not found at '{config_path}'. "
                "checkpoint_dir must contain a '.hydra/' subdirectory."
            )
        config_path = os.path.abspath(config_path)

        ckpt_dir = os.path.join(self.checkpoint_dir, "checkpoints")
        if os.path.isdir(ckpt_dir):
            ckpt_files = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
        else:
            ckpt_files = glob.glob(os.path.join(self.checkpoint_dir, "*.ckpt"))

        if not ckpt_files:
            raise FileNotFoundError(
                f"No .ckpt checkpoint file found in '{self.checkpoint_dir}' or '{self.checkpoint_dir}/checkpoints/'."
            )
        checkpoint_path = os.path.abspath(ckpt_files[0])

        return config_path, checkpoint_path

    def prepare_anndata(self, adata: sc.AnnData, train: bool = False) -> None:
        """Load or train PaSCient model and extract donor-level embeddings.

        Parameters
        ----------
        adata
            Single-cell AnnData.  Must contain :attr:`sample_key` and all
            :attr:`label_keys` in ``.obs``.
        train
            If ``True``, train the model on *adata*.  When
            ``checkpoint_dir`` is ``None`` a fresh model is built;
            otherwise the pretrained model is loaded first and then
            fine-tuned.
        """
        super().prepare_anndata(adata)

        if self.checkpoint_dir is not None:
            config_path, checkpoint_path = self._resolve_checkpoint_paths()
            self._pascient_model = self._load_pascient_model(config_path, checkpoint_path)
        elif not train:
            raise ValueError("Either provide checkpoint_dir or set train=True to train from scratch.")

        if train:
            self._train(adata)

        try:
            self._pascient_model = self._pascient_model.to(self.device)
        except RuntimeError as e:
            warnings.warn(
                f"Could not move model to device='{self.device}': {e}. Falling back to CPU.",
                stacklevel=2,
            )
            self.device = "cpu"
            self._pascient_model = self._pascient_model.to("cpu")

        # Extract embeddings for all donors
        self._extract_embeddings(adata)
        self._fitted = True

    def _build_model(self, n_genes: int, n_classes: int):
        """Build a fresh PaSCient model from pascient component classes.

        Parameters
        ----------
        n_genes
            Number of input genes.
        n_classes
            Number of prediction outputs.

        Returns
        -------
        pascient.model.sample_predictor.SamplePredictor
            A ``SamplePredictor`` instance with the default PaSCient
            architecture.
        """
        from functools import partial
        from types import SimpleNamespace

        import torch
        import torch.nn as nn

        try:
            from pascient.components.aggregators import NonLinearAttnAggregator
            from pascient.components.basic_models import BasicMLP
            from pascient.components.cell_to_cell import CellToCellIdentity
            from pascient.components.cell_to_output import CellToOutputNone
            from pascient.components.masking import DummyMasking
            from pascient.model.sample_predictor import SamplePredictor
        except ImportError as e:
            raise ImportError("pascient is required. Install with: pip install patpy[pascient]") from e

        latent_dim = self.latent_dim  # 1024
        emb_dim = self.patient_emb_dim  # 512

        from pascient.components.losses import CrossEntropyLossViews

        # Minimal losses config expected by SamplePredictor.init_losses()
        losses = SimpleNamespace(
            sample_prediction_loss=SimpleNamespace(
                weight=1.0,
                loss_fn=partial(CrossEntropyLossViews),
                labels=self.label_keys[:1],
            ),
        )

        return SamplePredictor(
            num_genes=n_genes,
            optimizer=partial(torch.optim.Adam, lr=self.lr, weight_decay=self.weight_decay),
            scheduler=None,
            masking_strategy=DummyMasking(),
            # Linear gene→cell encoder: n_hidden_layers=-1 is a pascient
            # BasicMLP convention meaning "single Linear(input_dim, hidden_dim),
            # no activation, no output projection".
            gene2cell_encoder=BasicMLP(n_genes, hidden_dim=latent_dim, output_dim=latent_dim, n_hidden_layers=-1),
            cell2cell_encoder=CellToCellIdentity(),
            cell2patient_aggregation=NonLinearAttnAggregator(
                attention_model=BasicMLP(
                    latent_dim, hidden_dim=latent_dim, output_dim=1, n_hidden_layers=0, activation_cls=nn.Tanh
                )
            ),
            cell2output=CellToOutputNone(),
            patient_encoder=BasicMLP(
                latent_dim,
                hidden_dim=emb_dim,
                output_dim=emb_dim,
                n_hidden_layers=0,
                activation_cls=nn.PReLU,
                activation_out_cls=nn.PReLU,
            ),
            cell_decoder=None,
            # Linear predictor head (n_hidden_layers=-1 → single linear layer)
            patient_predictor=BasicMLP(emb_dim, hidden_dim=n_classes, output_dim=n_classes, n_hidden_layers=-1),
            metrics=[],
            losses=losses,
            cell_contrastive_strategy="classic",
        )

    def _train(self, adata: sc.AnnData) -> None:
        """Train PaSCient using a Lightning Trainer following the upstream pipeline.

        Builds a :class:`~pascient.model.sample_predictor.SamplePredictor`
        (or reuses an existing one loaded from a checkpoint) and trains it
        with ``lightning.Trainer.fit`` on a DataModule that wraps *adata*
        into PaSCient's ``SampleBatch`` format.

        Parameters
        ----------
        adata
            AnnData with donor labels in ``.obs``.
        """
        import torch
        from torch.utils.data import DataLoader, Dataset, random_split

        try:
            import lightning as L
            from pascient.data.data_structures import SampleBatch
        except ImportError as e:
            raise ImportError(
                "pascient and lightning are required for training. Install with: pip install patpy[pascient]"
            ) from e

        expression = self._get_expression_matrix()
        n_genes = expression.shape[1]
        donor_col = adata.obs[self.sample_key].values

        # Encode labels to integer class indices for classification
        label_key = self.label_keys[0]
        task = self.tasks[0]
        label_vals = self.labels[label_key].values

        if task == "classification":
            classes = sorted(np.unique(label_vals))
            n_classes = len(classes)
            class_to_int = {c: i for i, c in enumerate(classes)}
            y_map = {d: class_to_int[v] for d, v in zip(self.labels.index, label_vals, strict=True)}
            self._class_names = list(classes)
        else:
            n_classes = 1
            y_map = {d: float(v) for d, v in zip(self.labels.index, label_vals, strict=True)}
            self._class_names = None

        self._trained_label = label_key

        if self._pascient_model is None:
            self._pascient_model = self._build_model(n_genes, n_classes)

        # -- Dataset that produces SampleBatch per donor ----------------

        normalize = self.normalize
        lognorm = self._lognormalize  # staticmethod ref
        subset_or_pad = self._subset_or_pad_cells

        class _DS(Dataset):
            def __init__(self_, donors):
                self_.donors = list(donors)

            def __len__(self_):
                return len(self_.donors)

            def __getitem__(self_, idx):
                donor_id = self_.donors[idx]
                mask = donor_col == donor_id
                x, pad, _ = subset_or_pad(expression[mask])

                x_t = torch.tensor(x[np.newaxis], dtype=torch.float32)
                pad_t = torch.tensor(pad[np.newaxis], dtype=torch.bool)

                if normalize:
                    x_t, pad_t = lognorm(x_t, pad_t)

                return SampleBatch(
                    x=x_t,
                    padded_mask=pad_t,
                    sample_metadata={label_key: torch.tensor(y_map[donor_id])},
                    cell_metadata={},
                    view_names=["view_0"],
                )

        def _collate(batch):
            return SampleBatch(
                x=torch.stack([b.x for b in batch]),
                padded_mask=torch.stack([b.padded_mask for b in batch]),
                sample_metadata={
                    k: torch.stack([b.sample_metadata[k] for b in batch]) for k in batch[0].sample_metadata
                },
                cell_metadata={},
                view_names=batch[0].view_names,
            )

        full_ds = _DS(self.samples)
        n_val = max(1, int(len(full_ds) * self.val_fraction))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(self.seed))

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=_collate)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=_collate)

        # -- Train via Lightning Trainer --------------------------------

        accelerator = "gpu" if self.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        trainer = L.Trainer(
            max_epochs=self.n_epochs,
            accelerator=accelerator,
            devices=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=True,
        )
        trainer.fit(self._pascient_model, train_dataloaders=train_dl, val_dataloaders=val_dl)
        self._pascient_model.eval()

    def _get_expression_matrix(self) -> np.ndarray:
        """Return expression data as a dense float32 numpy array.

        Delegates to :meth:`_get_data` from the base class, then
        densifies and casts to float32.

        Returns
        -------
        np.ndarray
            Dense expression matrix of shape ``(n_cells, n_genes)``.
        """
        import scipy.sparse

        X = self._get_data()
        if scipy.sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def _subset_or_pad_cells(
        self,
        donor_expression: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Subsample or zero-pad a donor's cells to :attr:`n_cells`.

        Parameters
        ----------
        donor_expression
            Expression matrix for one donor, shape ``(n_donor_cells, n_genes)``.
        rng
            Random generator for subsampling.  When ``None``,
            ``np.random`` is used.

        Returns
        -------
        sampled : np.ndarray
            ``(n_cells, n_genes)`` expression matrix.
        pad_mask : np.ndarray
            ``(n_cells,)`` boolean mask (``True`` for real cells).
        indices : np.ndarray
            Indices into *donor_expression* for each real cell.
        """
        n_donor_cells, n_genes = donor_expression.shape
        choice = rng.choice if rng is not None else np.random.choice

        if n_donor_cells >= self.n_cells:
            idx = choice(n_donor_cells, size=self.n_cells, replace=False)
            sampled = donor_expression[idx]
            pad_mask = np.ones(self.n_cells, dtype=bool)
        else:
            sampled = np.zeros((self.n_cells, n_genes), dtype=np.float32)
            sampled[:n_donor_cells] = donor_expression
            pad_mask = np.zeros(self.n_cells, dtype=bool)
            pad_mask[:n_donor_cells] = True
            idx = np.arange(n_donor_cells)

        return sampled, pad_mask, idx

    @staticmethod
    def _lognormalize(x, padding_mask, target_sum: float = 1e4):
        """Log-normalise a batch tensor following the PaSCient convention.

        Parameters
        ----------
        x : torch.Tensor
            Raw count tensor of shape ``(batch, 1, cells, genes)``.
        padding_mask : torch.Tensor
            Boolean mask of shape ``(batch, 1, cells)``.  ``True`` for
            observed cells.
        target_sum : float
            Total-count normalisation target (default ``1e4``).

        Returns
        -------
        x_norm : torch.Tensor
            Log-normalised tensor, same shape as *x*.
        padding_mask : torch.Tensor
            Unchanged mask (returned for convenience).
        """
        counts_per_cell = x.sum(dim=-1) + 1e-8  # (batch, 1, cells)
        counts_per_cell = counts_per_cell / target_sum
        counts_per_cell[~padding_mask] = 1
        x_norm = x / counts_per_cell.unsqueeze(-1)
        x_norm = x_norm.log1p()
        return x_norm, padding_mask

    def _forward_model(self, x, padding_mask):
        """Run the PaSCient encoder pipeline.

        Parameters
        ----------
        x : torch.Tensor
            Expression tensor ``(batch, 1, cells, genes)``.
        padding_mask : torch.Tensor
            Boolean mask ``(batch, 1, cells)``.

        Returns
        -------
        patient_embeddings : torch.Tensor
            ``(batch, 1, emb_dim)``
        cell_embeddings : torch.Tensor
            ``(batch, 1, cells, cell_emb_dim)``
        patient_predictions : torch.Tensor
            ``(batch, 1, n_classes)``
        """
        model = self._pascient_model
        cell_embds = model.gene2cell_encoder(x)
        cell_cross_embds = model.cell2cell_encoder(cell_embds, padding_mask=padding_mask)
        patient_embds = model.cell2patient_aggregation.aggregate(data=cell_cross_embds, mask=padding_mask)
        patient_embds_enc = model.patient_encoder(patient_embds)
        patient_preds = model.patient_predictor(patient_embds_enc)
        return patient_embds_enc, cell_cross_embds, patient_preds

    def _extract_embeddings(self, adata: sc.AnnData) -> None:
        """Process all donors through PaSCient and store embeddings.

        Parameters
        ----------
        adata
            Source AnnData.
        """
        import torch

        rng = np.random.default_rng(self.seed)
        donor_col = adata.obs[self.sample_key].values
        expression = self._get_expression_matrix()
        expression.shape[1]

        donor_ids = []
        all_sample_embeddings = []

        donor_list = list(self.samples)
        for batch_start in range(0, len(donor_list), self.batch_size):
            batch_donors = donor_list[batch_start : batch_start + self.batch_size]
            batch_x = []
            batch_pad = []
            batch_donor_ids = []

            for donor_id in batch_donors:
                cell_mask = donor_col == donor_id
                donor_expr = expression[cell_mask]

                if donor_expr.shape[0] == 0:
                    logger.warning("Donor '%s' has no cells, skipping.", donor_id)
                    continue

                sampled, pad_mask, _ = self._subset_or_pad_cells(donor_expr, rng)

                # PaSCient expects shape (1, n_cells, n_genes) per view
                batch_x.append(sampled[np.newaxis, :, :])
                batch_pad.append(pad_mask[np.newaxis, :])
                batch_donor_ids.append(donor_id)

            if not batch_x:
                continue

            x_tensor = torch.tensor(np.stack(batch_x), dtype=torch.float32, device=self.device)
            pad_tensor = torch.tensor(np.stack(batch_pad), dtype=torch.bool, device=self.device)

            # Apply log-normalisation on the tensor if requested
            if self.normalize:
                x_tensor, pad_tensor = self._lognormalize(x_tensor, pad_tensor)

            with torch.no_grad():
                patient_embds, cell_embds, _preds = self._forward_model(x_tensor, pad_tensor)

            # patient_embds: (batch, 1, emb_dim) → (batch, emb_dim)
            sample_emb = patient_embds.squeeze(1).cpu().numpy()
            all_sample_embeddings.append(sample_emb)

            # cell_embds: (batch, 1, n_cells, cell_emb_dim)
            cell_emb_np = cell_embds.squeeze(1).cpu().numpy()
            for i, did in enumerate(batch_donor_ids):
                self._cell_embeddings[did] = cell_emb_np[i]

            donor_ids.extend(batch_donor_ids)

        if not all_sample_embeddings:
            raise RuntimeError("No donor embeddings could be extracted. Check that adata contains valid samples.")

        embeddings_arr = np.concatenate(all_sample_embeddings, axis=0)
        cols = [f"dim_{i}" for i in range(embeddings_arr.shape[1])]
        self.sample_representation = pd.DataFrame(embeddings_arr, index=donor_ids, columns=cols)
        self.samples = np.array(donor_ids)

    def get_sample_representations(self) -> pd.DataFrame:
        """Per-donor embeddings from PaSCient.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_donors, emb_dim)`` (typically 512), indexed by
            donor ID.  Columns are named ``"dim_0"``, ``"dim_1"``, ...

        Examples
        --------
        >>> embeddings = model.get_sample_representations()
        >>> distances = model.calculate_distance_matrix()
        """
        self._check_adata_loaded()
        return self.sample_representation

    def fine_tune(
        self,
        labels: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        **kwargs,
    ) -> None:
        """Fine-tune the PaSCient model on new or existing labels.

        If the label matches the one used for initial training and
        ``pascient`` is installed, the ``SamplePredictor`` is trained
        for further epochs via ``lightning.Trainer.fit``.  Otherwise,
        sklearn linear probes are fitted on top of the frozen patient
        embeddings (inherited behaviour).

        Parameters
        ----------
        labels : list[str] or str
            Labels to train for.
        tasks : list[_PREDICTION_TASKS] or _PREDICTION_TASKS
            Corresponding prediction task per label.
        **kwargs
            Extra arguments forwarded to the training loop (e.g.
            ``n_epochs``).
        """
        labels_list = [labels] if isinstance(labels, str) else list(labels)
        tasks_list = [tasks] if isinstance(tasks, str) else list(tasks)

        # If the request matches the native head label, continue
        # training the SamplePredictor end-to-end.
        if len(labels_list) == 1 and labels_list[0] == self._trained_label and self._pascient_model is not None:
            n_epochs = kwargs.get("n_epochs", self.n_epochs)
            old_epochs = self.n_epochs
            self.n_epochs = n_epochs
            try:
                self._train(self.adata)
            finally:
                self.n_epochs = old_epochs
            # Re-extract embeddings after continued training
            self._extract_embeddings(self.adata)
            return

        # For a different label, swap the prediction head and train
        # the full model if pascient is available.
        if self._pascient_model is not None:
            try:
                from pascient.components.basic_models import BasicMLP
            except ImportError:
                BasicMLP = None

            if BasicMLP is not None and len(labels_list) == 1:
                labels_list, tasks_list = self._prepare_fine_tune(labels_list, tasks_list)
                label_key = labels_list[0]
                task = tasks_list[0]
                label_vals = self.labels[label_key].values

                if task == "classification":
                    classes = sorted(np.unique(label_vals))
                    n_out = len(classes)
                    self._class_names = list(classes)
                else:
                    n_out = 1
                    self._class_names = None

                emb_dim = self.patient_emb_dim
                self._pascient_model.patient_predictor = BasicMLP(
                    emb_dim, hidden_dim=n_out, output_dim=n_out, n_hidden_layers=-1
                )
                self._trained_label = label_key

                n_epochs = kwargs.get("n_epochs", self.n_epochs)
                old_epochs = self.n_epochs
                self.n_epochs = n_epochs
                try:
                    self._train(self.adata)
                finally:
                    self.n_epochs = old_epochs
                self._extract_embeddings(self.adata)
                return

        # Fallback: sklearn probes on frozen embeddings
        super().fine_tune(labels, tasks, **kwargs)

    def predict(self, label: str) -> pd.Series | pd.DataFrame:
        """Predict *label* for every donor.

        When *label* matches the label used to train the native
        ``SamplePredictor`` head, the model's own predictions are
        returned.  Otherwise, the inherited sklearn-probe prediction
        is used (requires :meth:`fine_tune` to have been called for
        that label first).

        Parameters
        ----------
        label : str
            Donor-level label to predict.

        Returns
        -------
        pd.Series or pd.DataFrame
            Classification: DataFrame with ``prob_<class>`` columns
            and ``<label>_pred``.  Regression: Series of values.
        """
        self._check_fitted()
        if label not in self.label_keys:
            raise ValueError(f"`label='{label}'` is not found in model label keys.")

        # Use the native prediction head when available
        if label == self._trained_label and self._pascient_model is not None:
            return self._predict_native(label)

        # Fallback to sklearn probes
        return super().predict(label)

    def _predict_native(self, label: str) -> pd.Series | pd.DataFrame:
        """Run the SamplePredictor's native head on all donors."""
        import torch

        task = self.tasks[self.label_keys.index(label)]
        expression = self._get_expression_matrix()
        expression.shape[1]
        donor_col = self.adata.obs[self.sample_key].values
        rng = np.random.default_rng(self.seed)

        all_preds = []
        donor_ids = []

        for batch_start in range(0, len(self.samples), self.batch_size):
            batch_donors = list(self.samples[batch_start : batch_start + self.batch_size])
            batch_x, batch_pad = [], []

            for donor_id in batch_donors:
                cell_mask = donor_col == donor_id
                donor_expr = expression[cell_mask]
                if donor_expr.shape[0] == 0:
                    continue

                sampled, pad, _ = self._subset_or_pad_cells(donor_expr, rng)
                batch_x.append(sampled[np.newaxis, :, :])
                batch_pad.append(pad[np.newaxis, :])
                donor_ids.append(donor_id)

            if not batch_x:
                continue

            x_t = torch.tensor(np.stack(batch_x), dtype=torch.float32, device=self.device)
            pad_t = torch.tensor(np.stack(batch_pad), dtype=torch.bool, device=self.device)
            if self.normalize:
                x_t, pad_t = self._lognormalize(x_t, pad_t)

            with torch.no_grad():
                preds = self._forward_model(x_t, pad_t)[2]  # (batch, 1, n_out)

            all_preds.append(preds.squeeze(1).cpu().numpy())

        preds_arr = np.concatenate(all_preds, axis=0)  # (n_donors, n_out)

        if task == "classification":
            import scipy.special

            proba = scipy.special.softmax(preds_arr, axis=1)
            classes = self._class_names or list(range(proba.shape[1]))
            result = pd.DataFrame(
                {f"prob_{c}": proba[:, i] for i, c in enumerate(classes)},
                index=donor_ids,
            )
            result[f"{label}_pred"] = [classes[i] for i in proba.argmax(axis=1)]
            return result
        else:
            return pd.Series(preds_arr.ravel(), index=donor_ids, name=label)

    def get_sample_importance(self, force: bool = False) -> pd.DataFrame:
        """Per-donor importance derived from the L2 norm of patient embeddings.

        PaSCient does not expose per-label importance directly.  The L2
        norm of the patient embedding is used as a scalar proxy.  For
        label-specific prediction, use :meth:`fine_tune` and
        :meth:`predict`.

        Parameters
        ----------
        force
            Recompute even if cached results exist.

        Returns
        -------
        pd.DataFrame
            Indexed by donor ID.  One column per label key named
            ``"<label_key>_importance"``.

        Examples
        --------
        >>> scores = model.get_sample_importance()
        """
        self._check_adata_loaded()

        cache_key = "supervised_sample_importance"
        if not force and cache_key in self.adata.uns:
            return pd.DataFrame(self.adata.uns[cache_key], index=self.samples)

        norms = np.linalg.norm(self.sample_representation.values, axis=1)

        if len(self.label_keys) == 1:
            sample_importance = pd.DataFrame(
                {f"{self.label_keys[0]}_importance": norms},
                index=self.samples,
            )
        else:
            sample_importance = pd.DataFrame(
                {f"{k}_importance": norms for k in self.label_keys},
                index=self.samples,
            )
            sample_importance.insert(0, "average_importance", norms)

        self.adata.uns[cache_key] = sample_importance.to_dict()
        return sample_importance

    def get_cell_importance(self, force: bool = False, target: int = 0) -> pd.DataFrame:
        """Per-cell importance via Integrated Gradients.

        Computes Integrated Gradients (IG) attributions from ``captum``
        following the approach in the PaSCient paper (De Brouwer et al.
        2025, https://doi.org/10.1016/j.cels.2025.101195).  For each donor, the IG attributions at the gene level
        are computed with respect to the model's prediction for the
        specified *target* class, using a zero baseline.  The per-cell
        importance score is the L2 norm of the per-gene attribution
        vector for that cell.

        When ``captum`` is not installed or the model is unavailable,
        cosine similarity between each cell's expression and the
        donor patient embedding is used as a proxy.

        Parameters
        ----------
        force
            Recompute even if cached results exist in ``adata.obs``.
        target
            Index of the prediction target (class index) for which IG
            attributions are computed.  Defaults to 0.

        Returns
        -------
        pd.DataFrame
            Indexed by ``adata.obs_names``.  One column per label key
            named ``"<label_key>_importance"``.

        Examples
        --------
        >>> importance = model.get_cell_importance()
        >>> adata.obs["importance"] = importance.values
        >>> sc.pl.umap(adata, color="importance")
        """
        self._check_adata_loaded()

        importance_cols = [f"{k}_importance" for k in self.label_keys]
        if not force and all(c in self.adata.obs.columns for c in importance_cols):
            return self.adata.obs[importance_cols]

        # Try IG-based importance; fall back to cosine similarity.
        try:
            scores = self._cell_importance_ig(target=target)
        except (ImportError, RuntimeError):
            logger.info(
                "Integrated Gradients not available; falling back to cosine similarity "
                "(install captum for gradient-based cell importance)."
            )
            scores = self._cell_importance_cosine()

        result = pd.DataFrame(index=self.adata.obs_names)
        for col in importance_cols:
            result[col] = scores
            self.adata.obs[col] = scores

        return result

    def _cell_importance_ig(self, target: int = 0) -> np.ndarray:
        """Compute per-cell importance using Integrated Gradients.

        Parameters
        ----------
        target
            Class index for the IG attribution target.

        Returns
        -------
        np.ndarray
            Per-cell importance scores of length ``n_cells``.
        """
        import torch
        from captum.attr import IntegratedGradients

        if self._pascient_model is None:
            raise RuntimeError("PaSCient model not loaded.")

        model = self._pascient_model

        class _IGForwardModel(torch.nn.Module):
            """Wrapper that returns softmax predictions for IG attribution."""

            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model

            def forward(self, x, padding_mask):
                cell_embds = self.base_model.gene2cell_encoder(x)
                cell_cross_embds = self.base_model.cell2cell_encoder(cell_embds, padding_mask=padding_mask)
                patient_embds = self.base_model.cell2patient_aggregation.aggregate(
                    data=cell_cross_embds, mask=padding_mask
                )
                patient_embds = self.base_model.patient_encoder(patient_embds)
                patient_preds = self.base_model.patient_predictor(patient_embds)
                return torch.softmax(patient_preds[:, 0], dim=-1)

        ig_model = _IGForwardModel(model)
        ig = IntegratedGradients(ig_model)

        rng = np.random.default_rng(self.seed)
        donor_col = self.adata.obs[self.sample_key].values
        expression = self._get_expression_matrix()
        expression.shape[1]
        scores = np.zeros(len(self.adata))

        for donor_id in self.samples:
            cell_mask = donor_col == donor_id
            if not cell_mask.any():
                continue

            donor_expr = expression[cell_mask]
            if donor_expr.shape[0] == 0:
                continue

            sampled, pad_mask, cell_indices = self._subset_or_pad_cells(donor_expr, rng)

            # Shape: (1, 1, n_cells, n_genes)
            x_tensor = torch.tensor(sampled[np.newaxis, np.newaxis, :, :], dtype=torch.float32, device=self.device)
            pad_tensor = torch.tensor(pad_mask[np.newaxis, np.newaxis, :], dtype=torch.bool, device=self.device)

            if self.normalize:
                x_tensor, pad_tensor = self._lognormalize(x_tensor, pad_tensor)

            x_tensor.requires_grad = True
            baseline = torch.zeros_like(x_tensor)

            attr, _ = ig.attribute(
                x_tensor,
                baselines=baseline,
                additional_forward_args=pad_tensor,
                target=target,
                return_convergence_delta=True,
            )

            # attr: (1, 1, n_cells, n_genes)
            attr_np = attr[0, 0].detach().cpu().numpy()  # (n_cells, n_genes)

            # Per-cell importance = L2 norm of the gene-level attributions
            n_real = min(len(cell_indices), self.n_cells)
            cell_scores = np.linalg.norm(attr_np[:n_real], axis=1)

            # Map scores back to the donor's cells in adata
            donor_positions = np.where(cell_mask)[0]
            for j in range(n_real):
                scores[donor_positions[cell_indices[j]]] = cell_scores[j]

        return scores

    def _cell_importance_cosine(self) -> np.ndarray:
        """Compute per-cell importance via cosine similarity with patient embedding.

        Returns
        -------
        np.ndarray
            Per-cell importance scores of length ``n_cells``.
        """
        donor_col = self.adata.obs[self.sample_key].values
        scores = np.zeros(len(self.adata))

        for donor_id in self.samples:
            cell_mask = donor_col == donor_id
            if not cell_mask.any() or donor_id not in self.sample_representation.index:
                continue

            patient_vec = self.sample_representation.loc[donor_id].values
            n_real_cells = cell_mask.sum()

            if donor_id in self._cell_embeddings and self._cell_embeddings[donor_id].shape[0] >= n_real_cells:
                cell_emb = self._cell_embeddings[donor_id][:n_real_cells]
                d = min(cell_emb.shape[1], patient_vec.shape[0])
                dot = np.abs(cell_emb[:, :d] @ patient_vec[:d])
                cell_norms = np.linalg.norm(cell_emb[:, :d], axis=1) + 1e-8
                patient_norm = np.linalg.norm(patient_vec[:d]) + 1e-8
                scores[cell_mask] = dot / (cell_norms * patient_norm)
            else:
                cell_data = self._get_data()[cell_mask]
                if hasattr(cell_data, "toarray"):
                    cell_data = cell_data.toarray()
                d = min(cell_data.shape[1], patient_vec.shape[0])
                dot = np.abs(cell_data[:, :d] @ patient_vec[:d])
                cell_norms = np.linalg.norm(cell_data[:, :d], axis=1) + 1e-8
                patient_norm = np.linalg.norm(patient_vec[:d]) + 1e-8
                scores[cell_mask] = dot / (cell_norms * patient_norm)

        return scores
