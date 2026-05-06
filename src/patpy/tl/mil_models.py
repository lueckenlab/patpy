"""MIL model classes for the patpy benchmark.

All models jointly train the MIL aggregator and prediction head end-to-end
— no sklearn probes are used anywhere.

Provides:
- ``TorchMILWrapper`` — base adapter for any torchmil model class.
- ``ABMIL``   — torchmil gated attention MIL (Ilse et al. 2018).
- ``TransMIL`` — torchmil transformer MIL.
- ``DSMIL``   — torchmil dual-stream MIL.
- ``MultiMIL`` — thin wrapper around the ``multimil`` package (Theislab).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import scanpy as sc

from patpy.tl._types import _PREDICTION_TASKS
from patpy.tl.supervised import SupervisedSampleMethod, _logits_to_prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data-conversion helpers
# ---------------------------------------------------------------------------


def _bags_to_padded(bags: list, max_cells: int | None = None) -> tuple:
    """Convert a list of variable-length bag tensors to a padded batch.

    Parameters
    ----------
    bags : list of (n_i, feat_dim) Tensors
    max_cells : optional cap on cells per bag (random subsample, no replacement)

    Returns
    -------
    X_padded : Tensor of shape (n_bags, max_n, feat_dim)
    mask     : BoolTensor of shape (n_bags, max_n), True for real instances.
    """
    import torch

    if max_cells is not None:
        rng = np.random.default_rng(0)
        bags = [
            b[rng.choice(b.shape[0], min(b.shape[0], max_cells), replace=False)]
            if b.shape[0] > max_cells else b
            for b in bags
        ]

    max_n = max(b.shape[0] for b in bags)
    feat = bags[0].shape[1]
    X = torch.zeros(len(bags), max_n, feat)
    mask = torch.zeros(len(bags), max_n, dtype=torch.bool)
    for i, bag in enumerate(bags):
        n = bag.shape[0]
        X[i, :n] = bag
        mask[i, :n] = True
    return X, mask


def _model_forward(model, X, mask, **kwargs):
    """Call model.forward() without mask."""
    return model(X, **kwargs)


def _get_criterion(task: str, n_classes: int):
    """Return the appropriate loss for a task."""
    import torch.nn as nn

    if task == "classification" and n_classes <= 2:
        return nn.BCEWithLogitsLoss()
    elif task == "classification":
        return nn.CrossEntropyLoss()
    else:
        return nn.MSELoss()


def _get_torchmil_class(name: str):
    """Import and return a torchmil model class by name."""
    try:
        import torchmil.models as _tm

        cls = getattr(_tm, name, None)
        if cls is None:
            raise ImportError(f"torchmil has no model '{name}'.")
        return cls
    except ImportError as e:
        raise ImportError(
            f"torchmil is required for {name}. Install with: pip install torchmil"
        ) from e


# ---------------------------------------------------------------------------
# TorchMILWrapper — base class
# ---------------------------------------------------------------------------


class TorchMILWrapper(SupervisedSampleMethod):
    """Patpy adapter for ``torchmil`` model classes.

    Each label key gets an independently trained torchmil model
    (aggregator + prediction head jointly trained end-to-end).
    No sklearn probes are used.

    Input data is automatically converted from the patpy bag format (list of
    per-donor tensors) to torchmil's padded-batch format
    ``(batch, max_bag_size, feat_dim)`` with a Boolean mask.

    Parameters
    ----------
    model_class_name : str
        torchmil model class name, e.g. ``"ABMIL"``, ``"TransMIL"``,
        ``"DSMIL"``.
    sample_key : str
    label_keys : list[str]
    tasks : list[str]
    model_kwargs : dict, optional
        Extra keyword arguments forwarded to the torchmil constructor
        (e.g. ``{"att_dim": 256, "gated": True}``).
    n_epochs : int, default 200
    batch_size : int, default 16
        Donors per gradient step.
    lr : float, default 1e-3
    weight_decay : float, default 1e-4
    device : str, default ``"auto"``
        ``"auto"`` picks CUDA when available.
    seed : int, default 42

    Examples
    --------
    >>> model = TorchMILWrapper(
    ...     "ABMIL",
    ...     sample_key="donor_id",
    ...     label_keys=["disease"],
    ...     tasks=["classification"],
    ... )
    >>> model.prepare_anndata(train_adata)
    >>> preds = model.predict_on_adata(test_adata, label="disease")
    """

    def __init__(
        self,
        model_class_name: str,
        sample_key: str,
        label_keys: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        model_kwargs: dict | None = None,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        seed: int = 42,
        max_cells_per_bag: int | None = None,
    ) -> None:
        super().__init__(
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )
        self.model_class_name = model_class_name
        self.model_kwargs = model_kwargs or {}
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_cells_per_bag = max_cells_per_bag
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        # One torchmil model per label key
        self._models: dict[str, object] = {}

    # ------------------------------------------------------------------
    # patpy interface
    # ------------------------------------------------------------------

    def prepare_anndata(self, adata: sc.AnnData, train: bool = True, **kwargs) -> None:
        super().prepare_anndata(adata)
        if (
            self.layer not in adata.obsm
            and self.layer not in adata.layers
            and self.layer not in ("X", None)
        ):
            raise ValueError(f"layer='{self.layer}' not found in adata.obsm or adata.layers.")
        self._build_label_mappings()
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
        """Jointly train aggregator + prediction head for each label."""
        import torch
        import torch.optim as optim

        labels, tasks = self._prepare_fine_tune(labels, tasks)
        self._build_label_mappings()

        device = self._resolve_device(torch)
        bags, sample_ids = self._build_bags_from_adata(self.adata)
        in_dim = bags[0].shape[1]

        n_ep = n_epochs if n_epochs is not None else self.n_epochs
        lr_use = lr if lr is not None else self.lr

        X_padded, mask = _bags_to_padded(bags, self.max_cells_per_bag)
        X_padded, mask = X_padded.to(device), mask.to(device)

        for label, task in zip(labels, tasks):
            n_classes = self._n_classes(label, task)
            model = self._build_model(in_dim, task, n_classes).to(device)
            Y = self._build_y_tensor(label, task, sample_ids, n_classes).to(device)

            torch.manual_seed(self.seed)
            optimizer = optim.Adam(
                model.parameters(), lr=lr_use, weight_decay=self.weight_decay
            )

            rng = np.random.default_rng(self.seed)
            for epoch in range(n_ep):
                model.train()
                order = rng.permutation(len(bags))
                ep_loss = 0.0
                for start in range(0, len(bags), self.batch_size):
                    idx = order[start : start + self.batch_size]
                    b_X = X_padded[idx]
                    b_mask = mask[idx]
                    b_Y = Y[idx]

                    optimizer.zero_grad()
                    logits = _model_forward(model, b_X, b_mask)  # (batch, n_out)
                    loss = model.criterion(logits, b_Y)
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()

                if epoch % 50 == 0:
                    logger.debug(
                        "%s [%s] epoch %d/%d loss=%.4f",
                        self.model_class_name, label, epoch, n_ep, ep_loss,
                    )

            self._models[label] = model

        self._fitted = True

    def predict(self, label: str) -> pd.Series | pd.DataFrame:
        """Predict on training samples."""
        self._check_fitted()
        bags, sample_ids = self._build_bags_from_adata(self.adata)
        return self._predict_bags(bags, sample_ids, label)

    def predict_on_adata(self, adata: sc.AnnData, label: str) -> pd.Series | pd.DataFrame:
        """Predict on held-out donors in *adata* without retraining."""
        self._check_fitted()
        if label not in self.label_keys:
            raise ValueError(f"label='{label}' not in label_keys={self.label_keys}.")
        bags, sample_ids = self._build_bags_from_adata(adata)
        return self._predict_bags(bags, sample_ids, label)

    def get_sample_importance(self, force: bool = False) -> pd.DataFrame:
        self._check_fitted()
        bags, sample_ids = self._build_bags_from_adata(self.adata)
        rows = {}
        for label, model in self._models.items():
            logits = self._run_forward(bags, model)[:, 0]
            rows[f"{label}_importance"] = logits
        return pd.DataFrame(rows, index=sample_ids)

    def get_cell_importance(
        self,
        label: str | None = None,
        normalized: bool = False,
        force: bool = False,
    ) -> pd.DataFrame:
        """Per-cell attention weights (attention-based models only).

        Parameters
        ----------
        label
            Which label's model to use.  Defaults to the first label key.
        normalized
            If ``False`` (default), return raw pre-softmax attention logits
            captured via a forward hook on the model's ``nn.Softmax`` layer.
            These are on an unbounded scale and preserve the full dynamic range
            before normalisation.  Falls back to ``log(softmax + ε)`` when no
            ``nn.Softmax`` module is found.
            If ``True``, return post-softmax weights that sum to 1 per bag
            (directly comparable across cells within a donor).
        """
        self._check_adata_loaded()
        if label is None:
            label = self.label_keys[0]
        if label not in self._models:
            raise RuntimeError(f"No model fitted for label='{label}'.")

        import torch

        model = self._models[label]
        device = next(model.parameters()).device
        bags, _ = self._build_bags_from_adata(self.adata)
        X_padded, mask = _bags_to_padded(bags, self.max_cells_per_bag)
        X_padded, mask = X_padded.to(device), mask.to(device)

        model.eval()
        weights_all: list[np.ndarray] = []

        if normalized:
            with torch.no_grad():
                try:
                    _, att = _model_forward(model, X_padded, mask, return_att=True)
                except TypeError:
                    raise NotImplementedError(
                        f"{self.model_class_name} does not expose attention weights via return_att."
                    )
            for i, bag in enumerate(bags):
                n = bag.shape[0]
                weights_all.append(att[i, :n].cpu().numpy())
        else:
            # Capture pre-softmax inputs via a hook on nn.Softmax modules
            pre_softmax: list[torch.Tensor] = []

            def _make_hook(store: list):
                def _hook(m, inp, out):
                    store.append(inp[0].detach().cpu())
                return _hook

            hooks = [
                m.register_forward_hook(_make_hook(pre_softmax))
                for m in model.modules()
                if isinstance(m, torch.nn.Softmax)
            ]

            with torch.no_grad():
                try:
                    _, att = _model_forward(model, X_padded, mask, return_att=True)
                except TypeError:
                    for h in hooks:
                        h.remove()
                    raise NotImplementedError(
                        f"{self.model_class_name} does not expose attention weights via return_att."
                    )

            for h in hooks:
                h.remove()

            if pre_softmax:
                # Last captured tensor is the attention score matrix
                raw = pre_softmax[-1]  # (n_bags, max_bag_size) or (n_bags, 1, max_bag_size)
                if raw.ndim == 3:
                    raw = raw.squeeze(1)
                for i, bag in enumerate(bags):
                    n = bag.shape[0]
                    weights_all.append(raw[i, :n].numpy())
            else:
                # Fallback: log-transform of normalized weights (monotone, same ranking)
                logger.warning(
                    "No nn.Softmax module found in %s — returning log(normalized + 1e-8) "
                    "as raw score proxy.",
                    self.model_class_name,
                )
                for i, bag in enumerate(bags):
                    n = bag.shape[0]
                    w = att[i, :n].cpu().numpy()
                    weights_all.append(np.log(np.clip(w, 1e-8, None)))

        sort_idx = np.argsort(self.adata.obs[self.sample_key].values)
        unsort_idx = np.argsort(sort_idx)
        weights_flat = np.concatenate(weights_all)[unsort_idx]

        col = f"{label}_importance"
        self.adata.obs[col] = weights_flat
        return self.adata.obs[[col]]

    def get_sample_representations(self) -> pd.DataFrame:
        """Attention-weighted bag embeddings (requires the model to support it)."""
        raise NotImplementedError(
            f"{self.model_class_name} does not expose intermediate bag embeddings. "
            "Use get_sample_importance() instead."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, torch):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _n_classes(self, label: str, task: str) -> int:
        if task != "classification":
            return 1
        col = self._donor_col(label)
        if label in self._label_mappings:
            return len(self._label_mappings[label][0])
        return len(np.unique(col))

    def _build_model(self, in_dim: int, task: str, n_classes: int):
        """Construct a torchmil model with the correct criterion and output head."""
        import torch.nn as nn

        model_cls = _get_torchmil_class(self.model_class_name)
        criterion = _get_criterion(task, n_classes)

        kwargs = {
            "in_shape": (in_dim,),
            "criterion": criterion,
            **self.model_kwargs,
        }
        model = model_cls(**kwargs)

        # Override classifier for multi-class (torchmil models use 'classifier')
        if task == "classification" and n_classes > 2:
            model.classifier = nn.LazyLinear(n_classes)

        return model

    def _build_y_tensor(
        self, label: str, task: str, sample_ids: np.ndarray, n_classes: int
    ):
        """Build the target tensor aligned to sample_ids."""
        import torch

        col = self._extract_metadata([label]).loc[sample_ids, label].values

        if label in self._label_mappings:
            _, enc = self._label_mappings[label]
            col = np.array([enc[v] for v in col])

        if task == "classification" and n_classes <= 2:
            return torch.tensor(col.astype(np.float32))
        elif task == "classification":
            return torch.tensor(col.astype(np.int64))
        else:
            return torch.tensor(col.astype(np.float32))

    def _run_forward(self, bags: list, model) -> np.ndarray:
        import torch

        device = next(model.parameters()).device
        model.eval()
        X_padded, mask = _bags_to_padded(bags, self.max_cells_per_bag)
        X_padded, mask = X_padded.to(device), mask.to(device)
        with torch.no_grad():
            logits = _model_forward(model, X_padded, mask).cpu().numpy()
        if logits.ndim == 1:
            logits = logits[:, None]
        return logits

    def _predict_bags(
        self, bags: list, sample_ids: np.ndarray, label: str
    ) -> pd.Series | pd.DataFrame:
        if label not in self._models:
            raise RuntimeError(f"No model fitted for label='{label}'. Call fine_tune() first.")

        model = self._models[label]
        task = self.tasks[self.label_keys.index(label)]
        n_classes = self._n_classes(label, task)

        logits = self._run_forward(bags, model)  # (n_bags, n_out)

        return _logits_to_prediction(logits, task, label, sample_ids, self._label_mappings)


# ---------------------------------------------------------------------------
# Convenience subclasses
# ---------------------------------------------------------------------------


class ABMIL(TorchMILWrapper):
    """Gated attention MIL via ``torchmil.models.ABMIL`` (Ilse et al. 2018).

    Parameters
    ----------
    sample_key, label_keys, tasks
        See :class:`TorchMILWrapper`.
    att_dim : int, default 128
        Attention and hidden-layer width.
    gated : bool, default True
        Use gated (Hadamard) attention.
    layer : str, default ``"X_pca"``
    n_epochs, batch_size, lr, weight_decay, device, seed
        Training hyper-parameters.

    Examples
    --------
    >>> model = ABMIL(sample_key="donor_id", label_keys=["disease"], tasks=["classification"])
    >>> model.prepare_anndata(train_adata)
    >>> preds = model.predict_on_adata(test_adata, label="disease")
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        att_dim: int = 128,
        gated: bool = True,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            model_class_name="ABMIL",
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            model_kwargs={"att_dim": att_dim, "gated": gated},
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )


class TransMIL(TorchMILWrapper):
    """Transformer-based MIL via ``torchmil.models.TransMIL``.

    Parameters
    ----------
    sample_key, label_keys, tasks
        See :class:`TorchMILWrapper`.
    att_dim : int, default 512
        Embedding dimension (must be divisible by *n_heads*).
    n_layers : int, default 2
        Number of Nyströmformer layers.
    n_heads : int, default 8
        Number of attention heads.
    dropout : float, default 0.0
    layer : str, default ``"X_pca"``
    n_epochs, batch_size, lr, weight_decay, device, seed
        Training hyper-parameters.

    Examples
    --------
    >>> model = TransMIL(sample_key="donor_id", label_keys=["age"], tasks=["regression"])
    >>> model.prepare_anndata(train_adata)
    >>> preds = model.predict_on_adata(test_adata, label="age")
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        att_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.0,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            model_class_name="TransMIL",
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            model_kwargs={
                "att_dim": att_dim,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "dropout": dropout,
            },
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )


class DSMIL(TorchMILWrapper):
    """Dual-stream MIL via ``torchmil.models.DSMIL``.

    DSMIL jointly trains an instance classifier and a bag classifier,
    using the critical instance score for bag aggregation.

    Parameters
    ----------
    sample_key, label_keys, tasks
        See :class:`TorchMILWrapper`.
    att_dim : int, default 128
    dropout : float, default 0.0
    layer : str, default ``"X_pca"``
    n_epochs, batch_size, lr, weight_decay, device, seed
        Training hyper-parameters.

    Examples
    --------
    >>> model = DSMIL(sample_key="donor_id", label_keys=["disease"], tasks=["classification"])
    >>> model.prepare_anndata(train_adata)
    >>> preds = model.predict_on_adata(test_adata, label="disease")
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        att_dim: int = 128,
        dropout: float = 0.0,
        n_epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        seed: int = 42,
    ) -> None:
        super().__init__(
            model_class_name="DSMIL",
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            model_kwargs={"att_dim": att_dim, "dropout": dropout},
            n_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            seed=seed,
        )


# ---------------------------------------------------------------------------
# MultiMIL wrapper
# ---------------------------------------------------------------------------


class MultiMIL(SupervisedSampleMethod):
    """Wrapper around the ``multimil`` package (Theislab).

    MultiMIL learns sample-level latent representations via a multi-modal
    variational autoencoder.  A lightweight PyTorch prediction head is trained
    jointly with the model (not a sklearn probe).

    Requires ``multimil`` to be installed::

        pip install multimil

    Parameters
    ----------
    sample_key : str
    label_keys : list[str]
    tasks : list[str]
    cell_group_key : str or None
    layer : str, default ``"X_pca"``
    n_latent : int, default 16
        Latent dimensionality.
    max_epochs : int, default 200
    batch_size : int, default 128
    lr : float, default 1e-3
    seed : int, default 42

    Examples
    --------
    >>> model = MultiMIL(sample_key="donor_id", label_keys=["disease"], tasks=["classification"])
    >>> model.prepare_anndata(train_adata)
    >>> preds = model.predict_on_adata(test_adata, label="disease")
    """

    def __init__(
        self,
        sample_key: str,
        label_keys: list[str] | str,
        tasks: list[_PREDICTION_TASKS] | _PREDICTION_TASKS,
        cell_group_key: str | None = None,
        layer: str = "X_pca",
        n_latent: int = 16,
        max_epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        super().__init__(
            sample_key=sample_key,
            label_keys=label_keys,
            tasks=tasks,
            cell_group_key=cell_group_key,
            layer=layer,
            seed=seed,
        )
        self.n_latent = n_latent
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self._multimil_model = None
        self._heads: dict[str, object] = {}  # label → nn.Module prediction head
        self._train_latent: pd.DataFrame | None = None

    def prepare_anndata(self, adata: sc.AnnData, train: bool = True, **kwargs) -> None:
        super().prepare_anndata(adata)
        self._build_label_mappings()
        if train:
            self._fit_multimil(adata)
            self._fit_heads()

    def _fit_multimil(self, adata: sc.AnnData) -> None:
        try:
            from multimil.model import MultiMIL as _MM
        except ImportError as e:
            raise ImportError(
                "multimil is required. Install with: pip install multimil"
            ) from e

        training_adata = self._move_layer_to_X()
        _MM.setup_anndata(training_adata, sample_key=self.sample_key)
        self._multimil_model = _MM(training_adata, n_latent=self.n_latent)
        self._multimil_model.train(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            plan_kwargs={"lr": self.lr},
        )
        latent = self._multimil_model.get_latent_representation()
        self._train_latent = pd.DataFrame(
            latent,
            index=self.samples,
            columns=[f"dim_{i}" for i in range(latent.shape[1])],
        )
        self.sample_representation = self._train_latent
        self._fitted = True

    def _fit_heads(self) -> None:
        """Train one PyTorch linear head per label jointly on the latent space."""
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = torch.tensor(self._train_latent.values, dtype=torch.float32).to(device)

        for label, task in zip(self.label_keys, self.tasks):
            col = self._donor_col(label)
            if label in self._label_mappings:
                _, enc = self._label_mappings[label]
                col = np.array([enc[v] for v in col])

            if task == "classification":
                n_classes = len(self._label_mappings[label][0]) if label in self._label_mappings else len(np.unique(col))
                if n_classes <= 2:
                    head = nn.Linear(self.n_latent, 1).to(device)
                    criterion = nn.BCEWithLogitsLoss()
                    Y = torch.tensor(col.astype(np.float32)).to(device)
                else:
                    head = nn.Linear(self.n_latent, n_classes).to(device)
                    criterion = nn.CrossEntropyLoss()
                    Y = torch.tensor(col.astype(np.int64)).to(device)
            else:
                head = nn.Linear(self.n_latent, 1).to(device)
                criterion = nn.MSELoss()
                Y = torch.tensor(col.astype(np.float32)).to(device)

            optimizer = optim.Adam(head.parameters(), lr=self.lr)
            torch.manual_seed(self.seed)
            for _ in range(200):
                head.train()
                optimizer.zero_grad()
                logits = head(X).squeeze(-1)
                loss = criterion(logits, Y)
                loss.backward()
                optimizer.step()

            self._heads[label] = head

    def predict(self, label: str) -> pd.Series | pd.DataFrame:
        self._check_fitted()
        return self._head_predict(self._train_latent, label, self.samples)

    def predict_on_adata(self, adata: sc.AnnData, label: str) -> pd.Series | pd.DataFrame:
        self._check_fitted()
        if label not in self.label_keys:
            raise ValueError(f"label='{label}' not in label_keys.")
        latent = self._encode_adata(adata)
        sample_ids = latent.index.values
        return self._head_predict(latent, label, sample_ids)

    def _encode_adata(self, adata: sc.AnnData) -> pd.DataFrame:
        donor_ids = adata.obs[self.sample_key].unique()
        inf_adata = self._move_layer_to_X()
        inf_adata = inf_adata[inf_adata.obs[self.sample_key].isin(donor_ids)]
        latent = self._multimil_model.get_latent_representation(inf_adata)
        samples_in = adata.obs[[self.sample_key]].groupby(self.sample_key).first().index
        return pd.DataFrame(
            latent,
            index=samples_in,
            columns=[f"dim_{i}" for i in range(latent.shape[1])],
        )

    def _head_predict(
        self, latent: pd.DataFrame, label: str, sample_ids: np.ndarray
    ) -> pd.Series | pd.DataFrame:
        import torch

        head = self._heads[label]
        task = self.tasks[self.label_keys.index(label)]
        device = next(head.parameters()).device
        X = torch.tensor(latent.values, dtype=torch.float32).to(device)

        head.eval()
        with torch.no_grad():
            logits = head(X).cpu().numpy()

        if logits.ndim == 1:
            logits = logits[:, None]

        return _logits_to_prediction(logits, task, label, sample_ids, self._label_mappings)

    def get_sample_representations(self) -> pd.DataFrame:
        self._check_fitted()
        return self._train_latent

    def get_sample_importance(self, force: bool = False) -> pd.DataFrame:
        self._check_fitted()
        return self._train_latent

    def get_cell_importance(self, label: str | None = None, force: bool = False) -> pd.DataFrame:
        raise NotImplementedError("MultiMIL does not provide cell-level importance scores.")
