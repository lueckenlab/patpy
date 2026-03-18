from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import scanpy as sc

from patpy.tl._base_sample_method import BaseSampleMethod
from patpy.tl._types import _PREDICTION_TASKS


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
        """Predict `label` for every sample in `self.samples`.

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

    def fine_tune(self, labels: list[str] | str, tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS, **kwargs):
        """Fine-tune / continue training the model on new or existing labels.

        The model is extended to predict the given labels (which may be a superset
        of existing labels). Existing learned parameters are preserved via warm-start.

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

        # Convert strings to lists
        labels = [labels] if isinstance(labels, str) else labels
        tasks = [tasks] if isinstance(tasks, str) else tasks

        self._check_adata_loaded()

        if len(labels) != len(tasks):
            raise ValueError(f"labels (len={len(labels)}) and tasks (len={len(tasks)}) must have the same length.")

        for label in labels:
            if label not in self.adata.obs.columns:
                raise ValueError(f"label '{label}' not found in adata.obs.columns.")

        # Extend label_keys / tasks
        for label, task in zip(labels, tasks, strict=True):
            if label not in self.label_keys:
                self.label_keys.append(label)
                self.tasks.append(task)

        self.labels = self._extract_metadata(self.label_keys)
        self.adata.uns.pop("supervised_sample_importance", None)

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
        self._label_mappings: dict[str, tuple[list, dict]] = {}  # label_key → (classes, encode_dict)

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

        # Convert strings to lists
        labels = [labels] if isinstance(labels, str) else labels
        tasks = [tasks] if isinstance(tasks, str) else tasks

        self._check_adata_loaded()

        if len(labels) != len(tasks):
            raise ValueError(f"labels (len={len(labels)}) and tasks (len={len(tasks)}) must have the same length.")

        for label in labels:
            if label not in self.adata.obs.columns:
                raise ValueError(f"label '{label}' not found in adata.obs.columns.")

        # Validate task compatibility: all tasks should be the same (MixMIL limitation)
        all_tasks = self.tasks + [t for l, t in zip(labels, tasks, strict=False) if l not in self.label_keys]
        if len(set(all_tasks)) > 1:
            raise ValueError(
                f"MixMIL supports a single task type per model. "
                f"Cannot mix {set(all_tasks)}. "
                f"Current tasks: {self.tasks}. New tasks: {tasks}."
            )

        # Store old P and old labels/tasks for weight transfer
        P_old = len(self.label_keys) if self._model is not None else None
        old_label_keys = self.label_keys.copy()

        # Extend label_keys and tasks with new ones
        for label, task in zip(labels, tasks, strict=True):
            if label not in self.label_keys:
                self.label_keys.append(label)
                self.tasks.append(task)

        # Build mappings for string labels
        self._build_label_mappings()

        # Re-extract metadata
        self.labels = self._extract_metadata(self.label_keys)

        # Clear stale caches
        self.adata.uns.pop("supervised_sample_importance", None)
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

        # Calculate the dimension range for this label based on multiclass info
        if not hasattr(self, "_multiclass_info"):
            # Fallback for models without multiclass info
            label_idx = self.label_keys.index(label)
            dim_start = label_idx
            dim_end = label_idx + 1
            n_classes = 2 if logits_np.shape[1] == 1 else 2  # binary
        else:
            # Calculate cumulative dimensions
            dim_start = 0
            for k in self.label_keys:
                if k == label:
                    break
                dim_start += self._multiclass_info[k][0]
            n_classes = self._multiclass_info[label][0]
            dim_end = dim_start + n_classes

        # Extract logits for this label
        label_logits = logits_np[:, dim_start:dim_end]

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
            raw = label_logits[:, 0].numpy() if label_logits.shape[1] > 0 else label_logits.ravel().numpy()
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

        # Sort cells by donor to build bags in the same order used during training.
        # Sorting is required because MixMIL processes one contiguous tensor per donor.
        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        unsort_idx = np.argsort(sort_idx)
        X_sorted = self._get_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]

        iidx = pd.Categorical(sorted_donors).codes
        indptr = np.concatenate([[0], np.bincount(iidx).cumsum()])

        Xs = [torch.from_numpy(X_sorted[s:e]) for s, e in zip(indptr[:-1], indptr[1:], strict=True)]

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

        # Sort cells by donor (same ordering as in training)
        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        X_sorted = self._get_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]

        iidx = pd.Categorical(sorted_donors).codes
        indptr = np.concatenate([[0], np.bincount(iidx).cumsum()])

        Xs = [torch.from_numpy(X_sorted[s:e]) for s, e in zip(indptr[:-1], indptr[1:], strict=True)]

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

    def _build_label_mappings(self) -> None:
        """Build and store mappings for string labels. Called during fine_tune."""
        for label_key in self.label_keys:
            if label_key in self._label_mappings:
                # Already mapped (from previous fine_tune call)
                continue

            col = self._donor_col(label_key)
            # Check if column contains non-numeric data (strings, objects)
            if col.dtype.kind not in ("f", "i", "u"):  # not float/int/uint
                # Create mapping: unique values (sorted) → indices
                classes = sorted(np.unique(col))
                encode_dict = {c: i for i, c in enumerate(classes)}
                self._label_mappings[label_key] = (classes, encode_dict)

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

        # Sort cells by donor so bags are contiguous in memory
        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        X_sorted = self._get_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]

        cat = pd.Categorical(sorted_donors)
        indptr = np.concatenate([[0], np.bincount(cat.codes).cumsum()])

        # Preserve the donor order implied by pd.Categorical (alphabetical)
        # and align self.samples to it so scores/importances line up
        self.samples = np.array(cat.categories)

        Xs = [torch.from_numpy(X_sorted[s:e]) for s, e in zip(indptr[:-1], indptr[1:], strict=True)]

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
                # Use one-hot encoding for multiclass
                one_hot = np.eye(n_classes, dtype="float32")[col_encoded]
                y_cols_list.extend([one_hot[:, i] for i in range(n_classes)])
                # Store multiclass info: this label occupies n_classes dimensions
                if not hasattr(self, "_multiclass_info"):
                    self._multiclass_info = {}
                self._multiclass_info[key] = (n_classes, key in self._label_mappings)
            else:
                # Binary/regression: single dimension
                y_cols_list.append(col_encoded.astype("float32"))
                if not hasattr(self, "_multiclass_info"):
                    self._multiclass_info = {}
                self._multiclass_info[key] = (1, key in self._label_mappings)

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
