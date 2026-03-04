from __future__ import annotations

import warnings
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import scanpy as sc

class SupervisedSampleMethod:
    """Base class for supervised sample-level representation methods.

    Subclasses must implement :meth:`prepare_anndata`,
    :meth:`get_sample_scores`, and :meth:`get_cell_scores`.
    :meth:`get_sample_embeddings` is optional — override it when the model
    produces a meaningful latent donor vector.

    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` with donor / sample identifiers.
    label_key : str
        Column in ``adata.obs`` with the donor-level supervision target
        (e.g. disease status, treatment response).  Every cell of a given
        donor must carry the same value.
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
        label_key: str,
        cell_group_key: Optional[str] = None,
        layer: str = "X_pca",
        seed: int = 42,
    ):
        self.sample_key = sample_key
        self.label_key = label_key
        self.cell_group_key = cell_group_key
        self.layer = layer
        self.seed = seed

        self.adata: Optional[sc.AnnData] = None
        self.samples: Optional[np.ndarray] = None
        self.labels: Optional[pd.Series] = None
        self.cell_groups: Optional[np.ndarray] = None

    def prepare_anndata(self, adata: sc.AnnData) -> None:
        """Store ``adata``, populate ``self.samples`` and ``self.labels``.

        Subclasses must call ``super().prepare_anndata(adata)`` first,
        then perform model-specific training.

        Parameters
        ----------
        adata : AnnData
            Single-cell AnnData.  Must contain ``sample_key`` and
            ``label_key`` columns in ``.obs``.
        """
        self.adata = adata

        if self.sample_key not in adata.obs.columns:
            raise ValueError(
                f"sample_key='{self.sample_key}' not found in adata.obs."
            )
        if self.label_key not in adata.obs.columns:
            raise ValueError(
                f"label_key='{self.label_key}' not found in adata.obs."
            )

        self.samples = adata.obs[self.sample_key].unique()

        donor_labels = (
            adata.obs[[self.sample_key, self.label_key]]
            .drop_duplicates(self.sample_key)
            .set_index(self.sample_key)
            .loc[self.samples, self.label_key]
        )
        n_per_donor = adata.obs.groupby(self.sample_key)[self.label_key].nunique()
        ambiguous = n_per_donor[n_per_donor > 1]
        if len(ambiguous) > 0:
            warnings.warn(
                f"label_key='{self.label_key}' has multiple values for donors: "
                f"{ambiguous.index.tolist()}.  Using the first occurrence per donor.",
                stacklevel=2,
            )
        self.labels = donor_labels

        if self.cell_group_key is not None and self.cell_group_key in adata.obs.columns:
            self.cell_groups = adata.obs[self.cell_group_key].unique()

    def get_sample_scores(self) -> pd.DataFrame:
        """Return per-donor prediction scores / posterior means.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by donor ID.  Must contain at least a
            ``"score"`` column; subclasses may add further columns
            (e.g. uncertainty, per-output scores for multi-output models).

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        RuntimeError
            If called before :meth:`prepare_anndata`.
        """
        self._check_fitted()
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_sample_scores()."
        )

    def get_cell_scores(self) -> pd.DataFrame:
        """Return per-cell attention or contribution scores.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``adata.obs_names``.  Must contain at
            least a ``"score"`` column.  Higher values indicate stronger
            contribution to the donor-level prediction.

        Raises
        ------
        NotImplementedError
            Subclasses must override this method.
        RuntimeError
            If called before :meth:`prepare_anndata`.
        """
        self._check_fitted()
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_cell_scores()."
        )

    def get_sample_embeddings(self) -> pd.DataFrame:
        """Return latent donor-level embeddings (optional).

        Not all supervised methods produce a meaningful latent vector per
        donor.  Override this method when they do.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by donor ID with embedding dimensions as
            columns (``"dim_0"``, ``"dim_1"``, …).

        Raises
        ------
        NotImplementedError
            Default implementation — override in subclasses that provide
            donor embeddings.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide sample embeddings."
        )

    def calculate_distance_matrix(self, dist: str = "euclidean") -> np.ndarray:
        """Compute a donor x donor distance matrix from sample embeddings.

        Requires :meth:`get_sample_embeddings` to be implemented.

        Parameters
        ----------
        dist : str, default ``"euclidean"``
            Distance metric passed to ``scipy.spatial.distance.pdist``.

        Returns
        -------
        np.ndarray
            Symmetric distance matrix of shape ``(n_donors, n_donors)``.
        """
        import scipy.spatial.distance

        embeddings = self.get_sample_embeddings()  # raises if not implemented
        distances = scipy.spatial.distance.pdist(embeddings.values, metric=dist)
        return scipy.spatial.distance.squareform(distances)

    def _check_fitted(self) -> None:
        if self.adata is None:
            raise RuntimeError(
                "Model is not yet fitted. Call prepare_anndata() first."
            )

    def _get_layer_data(self) -> np.ndarray:
        """Return the cell-level feature matrix from ``self.layer``."""
        self._check_fitted()
        if self.layer in self.adata.obsm:
            return self.adata.obsm[self.layer]
        if self.layer in self.adata.layers:
            return self.adata.layers[self.layer]
        if self.layer in ("X", None):
            return self.adata.X
        raise ValueError(
            f"layer='{self.layer}' not found in adata.obsm or adata.layers."
        )

    def plot_sample_scores(
        self,
        group_by: Optional[str] = None,
        ax=None,
    ):
        """Plot per-donor scores, optionally grouped by a metadata column.

        Parameters
        ----------
        group_by : str, optional
            Column in ``adata.obs`` to use for grouping / coloring.
            Defaults to ``label_key``.
        ax : matplotlib Axes, optional
            Axes to plot into.  A new figure is created if ``None``.

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        group_by = group_by or self.label_key
        scores = self.get_sample_scores()

        if group_by in self.adata.obs.columns:
            meta = (
                self.adata.obs[[self.sample_key, group_by]]
                .drop_duplicates(self.sample_key)
                .set_index(self.sample_key)
            )
            scores = scores.join(meta)

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        if group_by in scores.columns:
            sns.stripplot(data=scores, x=group_by, y="score", ax=ax, jitter=True)
            sns.boxplot(
                data=scores,
                x=group_by,
                y="score",
                ax=ax,
                fill=False,
                linewidth=1.5,
            )
        else:
            sns.histplot(scores["score"], ax=ax)

        ax.set_title(f"{type(self).__name__} — sample scores")
        return ax

    def plot_cell_scores(
        self,
        basis: str = "X_umap",
        ax=None,
        **scatter_kwargs,
    ):
        """Plot per-cell scores onto a pre-computed embedding.

        Parameters
        ----------
        basis : str, default ``"X_umap"``
            Key in ``adata.obsm`` for the 2-D embedding coordinates.
        ax : matplotlib Axes, optional
        **scatter_kwargs
            Additional keyword arguments passed to ``sc.pl.embedding``.

        Returns
        -------
        matplotlib Axes
        """
        self._check_fitted()
        cell_scores = self.get_cell_scores()
        self.adata.obs["_score"] = cell_scores["score"].values

        ax = sc.pl.embedding(
            self.adata,
            basis=basis,
            color="_score",
            show=False,
            ax=ax,
            **scatter_kwargs,
        )
        del self.adata.obs["_score"]
        return ax


class MixMIL(SupervisedSampleMethod):
    """Attention-based multi-instance mixed model for donor phenotype prediction by Engelmann et al. 2024 (https://arxiv.org/abs/2311.02455).

    Parameters
    ----------
    sample_key : str
        Column in ``adata.obs`` with donor / sample identifiers.
    label_key : str
        Column in ``adata.obs`` with the donor-level target variable
        (e.g. disease status).  Must be numeric or integer-coded.
    cell_group_key : str or None, optional
        Column in ``adata.obs`` with cell-type annotations (used by
        inherited plotting helpers only).
    layer : str, default ``"X_pca"``
        Key in ``adata.obsm`` to use as per-cell bag features.  PCA is
        computed automatically when the key is absent.
    n_pcs : int, default 50
        Number of PCA components computed when ``layer`` is not present.
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
    encode_sex : bool, default True
        Include ``"sex"`` from ``adata.obs`` as a fixed-effect covariate.
        A warning is issued and sex is skipped when the column is absent.
    encode_age : bool, default True
        Include z-normalised ``"age"`` from ``adata.obs`` as a fixed-effect
        covariate.  A warning is issued when the column is absent.
    additional_covariates : list[str], optional
        Extra covariate columns from ``adata.obs`` (numeric) or keys from
        ``adata.obsm`` to include as fixed effects.
    dtype : str, default ``"float32"``
        Floating-point precision for all tensors.
    seed : int, default 67
        Random seed (patpy convention).
    """

    def __init__(
        self,
        sample_key: str,
        label_key: str,
        cell_group_key: Optional[str] = None,
        layer: str = "X_pca",
        n_pcs: int = 50,
        likelihood: Optional[Literal["binomial", "categorical"]] = "binomial",
        n_trials: Optional[int] = 2,
        n_epochs: int = 2000,
        batch_size: int = 64,
        lr: float = 1e-3,
        encode_sex: bool = True,
        encode_age: bool = True,
        additional_covariates: Optional[List[str]] = None,
        dtype: str = "float32",
        seed: int = 67,
    ):
        super().__init__(
            sample_key=sample_key,
            label_key=label_key,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )
        self.n_pcs = n_pcs
        self.likelihood = likelihood
        self.n_trials = n_trials
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.encode_sex = encode_sex
        self.encode_age = encode_age
        self.additional_covariates = additional_covariates or []
        self.dtype = dtype

        self._model = None
        self._history = None

        self._donor_order: Optional[np.ndarray] = None

    def prepare_anndata(self, adata) -> None:
        """Train MixMIL on ``adata``.

        Calls ``super().prepare_anndata()`` to validate keys, then builds
        the per-donor bag tensors ``Xs``, the fixed-effect matrix ``F``,
        and the target ``Y``, and trains the MixMIL model.

        Parameters
        ----------
        adata : AnnData
            Single-cell AnnData.  Must contain ``sample_key`` and
            ``label_key`` in ``.obs``.
        """
        try:
            from mixmil import MixMIL as _MixMIL
        except ImportError as e:
            raise ImportError(
                "mixmil is required. Install with:  pip install mixmil"
            ) from e

        super().prepare_anndata(adata)
        self._ensure_layer()

        Xs, F, Y, donor_order = self._build_tensors()
        self._donor_order = donor_order

        if self.likelihood is not None:
            model = _MixMIL.init_with_mean_model(
                Xs, F, Y,
                likelihood=self.likelihood,
                n_trials=self.n_trials,
            )
        else:
            Q = Xs[0].shape[1]
            K = F.shape[1]
            model = _MixMIL(Q=Q, K=K)

        self._history = model.train(
            Xs, F, Y,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
        )
        self._model = model

    def get_sample_scores(self) -> pd.DataFrame:
        """Per-donor posterior predictions.

        Returns
        -------
        pd.DataFrame
            Indexed by donor ID.  Columns:

            * ``"score"`` - posterior mean prediction (averaged over outputs
              for multi-output models).
            * ``"score_<p>"`` - individual output columns when ``P > 1``.
        """
        self._check_fitted()
        import torch

        Xs, F, Y, _ = self._build_tensors()

        with torch.no_grad():
            preds = self._model.predict(Xs).numpy()  # (n_donors, P)

        if preds.ndim == 1 or preds.shape[1] == 1:
            scores = pd.DataFrame(
                {"score": preds.ravel()},
                index=self._donor_order,
            )
        else:
            scores = pd.DataFrame(
                {f"score_{p}": preds[:, p] for p in range(preds.shape[1])},
                index=self._donor_order,
            )
            scores.insert(0, "score", preds.mean(axis=1))

        return scores

    def get_cell_scores(self) -> pd.DataFrame:
        """Per-cell attention weights from the trained MixMIL model.

        The attention weight of a cell reflects how strongly it drives
        the donor-level prediction.  Weights within each donor's bag are
        softmax-normalised and therefore sum to 1 per donor.

        Returns
        -------
        pd.DataFrame
            Indexed by ``adata.obs_names``.  Columns:

            * ``"score"`` – mean attention weight across output dimensions.
            * ``"score_<p>"`` – per-output weights when ``P > 1``.
        """
        self._check_fitted()
        import torch

        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        unsort_idx = np.argsort(sort_idx)

        X_sorted = self._get_layer_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]
        iidx = pd.Categorical(sorted_donors).codes
        indptr = np.concatenate([[0], np.bincount(iidx).cumsum()])

        Xs = [
            torch.from_numpy(X_sorted[s:e])
            for s, e in zip(indptr[:-1], indptr[1:], strict=True)
        ]

        with torch.no_grad():
            w, _ = self._model.get_weights(Xs)

        w_cat = torch.cat(w, dim=0).numpy()

        w_cat = w_cat[unsort_idx]

        if w_cat.ndim == 1 or w_cat.shape[1] == 1:
            df = pd.DataFrame(
                {"score": w_cat.ravel()},
                index=self.adata.obs_names,
            )
        else:
            df = pd.DataFrame(
                {f"score_{p}": w_cat[:, p] for p in range(w_cat.shape[1])},
                index=self.adata.obs_names,
            )
            df.insert(0, "score", w_cat.mean(axis=1))

        return df

    def get_sample_embeddings(self) -> pd.DataFrame:
        """Per-donor latent embeddings (weighted-mean cell representations).

        The embedding is computed as the attention-weighted mean of the
        per-cell bag features in the learned latent space — analogous to
        the ``qu_mu`` (random-effect mean) in MixMIL notation.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_donors, Q)``, indexed by donor ID.
            Columns are named ``"dim_0"``, ``"dim_1"``, …
        """
        self._check_fitted()
        import torch

        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        X_sorted = self._get_layer_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]
        iidx = pd.Categorical(sorted_donors).codes
        indptr = np.concatenate([[0], np.bincount(iidx).cumsum()])

        Xs = [
            torch.from_numpy(X_sorted[s:e])
            for s, e in zip(indptr[:-1], indptr[1:], strict=True)
        ]

        with torch.no_grad():
            w, _ = self._model.get_weights(Xs) 

        embeddings = []
        for bag_X, bag_w in zip(Xs, w, strict=True):
            w_scalar = bag_w.mean(dim=1, keepdim=True) 
            emb = (w_scalar * bag_X).sum(dim=0).numpy()  
            embeddings.append(emb)

        embeddings_arr = np.stack(embeddings)
        cols = [f"dim_{i}" for i in range(embeddings_arr.shape[1])]
        return pd.DataFrame(embeddings_arr, index=self._donor_order, columns=cols)

    @property
    def training_history(self) -> Optional[list]:
        """List of per-step loss dicts returned by MixMIL training."""
        return self._history

    def plot_training_loss(self, ax=None):
        """Plot training loss curve.

        Parameters
        ----------
        ax : matplotlib Axes, optional

        Returns
        -------
        matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if self._history is None:
            raise RuntimeError("Call prepare_anndata() first.")

        history_df = pd.DataFrame(self._history)
        epoch_loss = history_df.groupby("epoch")["loss"].mean()

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3))

        ax.plot(epoch_loss.index, epoch_loss.values)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("MixMIL training loss")
        return ax

    def _ensure_layer(self) -> None:
        """Compute PCA if the requested obsm layer is not yet present."""
        import scanpy as sc

        if self.layer not in self.adata.obsm:
            warnings.warn(
                f"Key '{self.layer}' not found in adata.obsm. "
                f"Computing PCA with n_comps={self.n_pcs}.",
                stacklevel=3,
            )
            sc.pp.pca(self.adata, n_comps=self.n_pcs)
            if self.layer != "X_pca":
                self.adata.obsm[self.layer] = self.adata.obsm["X_pca"]

    def _build_tensors(self):
        """Build ``Xs``, ``F``, ``Y`` tensors and return donor order.

        Returns
        -------
        Xs : list[Tensor]   per-donor cell-embedding bags
        F  : Tensor         fixed-effect covariates (n_donors x K)
        Y  : Tensor         target variable          (n_donors x 1)
        donor_order : ndarray   donor IDs in the order matching Xs/F/Y
        """
        import torch

        dtype_torch = getattr(torch, self.dtype)

        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        X_sorted = self._get_layer_data()[sort_idx].astype(self.dtype)
        sorted_donors = self.adata.obs[self.sample_key].values[sort_idx]

        cat = pd.Categorical(sorted_donors)
        iidx = cat.codes
        donor_order = np.array(cat.categories)
        indptr = np.concatenate([[0], np.bincount(iidx).cumsum()])

        Xs = [
            torch.from_numpy(X_sorted[s:e])
            for s, e in zip(indptr[:-1], indptr[1:], strict=True)
        ]
        n_donors = len(donor_order)

        covariate_list = [torch.ones((n_donors, 1), dtype=dtype_torch)]

        def _donor_col(col: str) -> np.ndarray:
            return (
                self.adata.obs[[self.sample_key, col]]
                .drop_duplicates(self.sample_key)
                .set_index(self.sample_key)
                .loc[donor_order, col]
                .values
            )

        if self.encode_sex:
            if "sex" in self.adata.obs.columns:
                codes = pd.Categorical(_donor_col("sex")).codes.astype(self.dtype)
                covariate_list.append(
                    torch.tensor(codes, dtype=dtype_torch).unsqueeze(1)
                )
            else:
                warnings.warn(
                    "encode_sex=True but 'sex' not found in adata.obs; skipping.",
                    stacklevel=3,
                )

        if self.encode_age:
            if "age" in self.adata.obs.columns:
                age = _donor_col("age").astype(self.dtype)
                age_t = torch.tensor(age, dtype=dtype_torch).unsqueeze(1)
                mean, std = age_t.mean(), age_t.std()
                if std > 0 and not (
                    torch.isclose(mean, torch.zeros(1), atol=1e-2)
                    and torch.isclose(std, torch.ones(1), atol=1e-2)
                ):
                    age_t = (age_t - mean) / std
                covariate_list.append(age_t)
            else:
                warnings.warn(
                    "encode_age=True but 'age' not found in adata.obs; skipping.",
                    stacklevel=3,
                )

        for cov in self.additional_covariates:
            if cov in self.adata.obs.columns:
                vals = _donor_col(cov).astype("float32")
                covariate_list.append(
                    torch.tensor(vals, dtype=dtype_torch).unsqueeze(1)
                )
            elif cov in self.adata.obsm:
                # obsm covariate — must already be donor-level (n_donors x d)
                vals = self.adata.obsm[cov].astype("float32")
                covariate_list.append(torch.from_numpy(vals))
            else:
                raise ValueError(
                    f"additional_covariates entry '{cov}' not found in "
                    "adata.obs or adata.obsm."
                )

        F = torch.cat(covariate_list, dim=1)

        y_vals = _donor_col(self.label_key).astype("float32")
        Y = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)

        return Xs, F, Y, donor_order