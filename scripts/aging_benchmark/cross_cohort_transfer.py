"""Train models on AIFI, predict OneK1K ages — generalisation experiment.

Steps:

1. Find shared genes between AIFI HVGs and OneK1K HVGs (intersection of var_names).
2. Recompute PCA on AIFI restricted to those shared genes so the loadings
   live on a feature space both cohorts have. Apply the same projection
   to OneK1K → both datasets now share a 50-PC representation.
3. Train PaSCient (regression on age) and SampleCLR (contrastive +
   regression on age) on AIFI using that shared representation.
4. Use the trained models to predict OneK1K ages. Cell sampling uses
   ``n_cells=200`` because OneK1K donors average ~1300 cells (vs ~16K
   for AIFI) so a bigger bag would pad heavily.

Outputs (``data/aging_benchmark/cross_cohort_transfer/``):

    pca_aifi_shared.npz         loadings + variance + shared gene names
    aifi_embeddings.npz         AIFI sample embeddings (pascient, sampleclr_ft)
    onek1k_embeddings.npz       OneK1K sample embeddings (same methods)
    predictions.csv             per-(dataset, donor): true vs predicted age
    transfer_scores.csv         dataset x method: R², Spearman, MAE
    runtime.json
"""

from __future__ import annotations

import gc
import json
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

import patpy

AIFI_PATH = Path("/ictstr01/groups/luckylab/workspace/vladimir.shitov/aifi_data/imm_of_aging/imm-of-aging_pp.h5ad")
OUT_DIR = Path("/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial/data/aging_benchmark/cross_cohort_transfer")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PCS = 50
N_CELLS_PER_DONOR = 200      # Small so OneK1K donors aren't padded heavily
PASCIENT_EPOCHS = 30
SAMPLECLR_EPOCHS = 40


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_aifi() -> ad.AnnData:
    a = ad.read_h5ad(AIFI_PATH)
    a.obs["age"] = pd.to_numeric(a.obs["sample.subjectAgeAtDraw"].astype(str).str.replace("+", "")).astype(np.float32)
    a.obs["age_group"] = np.where(a.obs["age"] >= 65, "old", "young")
    a.obs["donor"] = a.obs["subject.subjectGuid"].astype(str)
    return a


def load_onek1k() -> ad.AnnData:
    a, _ = patpy.datasets.onek1k(return_dataset_info=True)
    a.obs["donor"] = a.obs["donor_id"].astype(str)
    a.obs["age_group"] = np.where(a.obs["age"] >= 65, "old", "young")
    return a


def cap_cells_per_donor(adata: ad.AnnData, max_cells: int, seed: int = SEED) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    keep: list[np.ndarray] = []
    obs_idx = np.arange(adata.n_obs)
    grouped = pd.Series(adata.obs["donor"].values).groupby(adata.obs["donor"].values, observed=True).indices
    for _donor, idx in grouped.items():
        if len(idx) <= max_cells:
            keep.append(obs_idx[idx])
        else:
            keep.append(obs_idx[rng.choice(idx, size=max_cells, replace=False)])
    return adata[np.sort(np.concatenate(keep))].copy()


def fit_shared_pca(aifi: ad.AnnData, shared_genes: list[str], n_pcs: int = N_PCS) -> tuple[PCA, np.ndarray, np.ndarray]:
    """Recompute PCA on AIFI restricted to shared genes."""
    idx = aifi.var_names.get_indexer(shared_genes)
    X = aifi.X[:, idx]
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    log(f"fitting PCA on AIFI[:, shared]: shape={X.shape}  (will use IncrementalPCA-like sklearn PCA(svd_solver='auto'))")
    # Sample a subset for fitting if memory is a problem
    fit_sample = X
    if X.shape[0] > 300_000:
        rng = np.random.default_rng(SEED)
        pick = rng.choice(X.shape[0], 300_000, replace=False)
        fit_sample = X[pick]
        log(f"  subsampled to {fit_sample.shape[0]:,} cells for PCA fit")
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=SEED)
    pca.fit(fit_sample)
    log(f"  PCA fit done; cumulative variance = {pca.explained_variance_ratio_.sum():.3f}")
    return pca, X, idx


def project(X: np.ndarray | "scipy.sparse.spmatrix", pca: PCA) -> np.ndarray:
    if hasattr(X, "toarray"):
        X = X.toarray()
    return pca.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)


def score_age(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 3:
        return {"r2": np.nan, "spearman": np.nan, "mae": np.nan, "n": int(mask.sum())}
    y_true, y_pred = y_true[mask], y_pred[mask]
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if not np.isnan(rho) else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n": int(mask.sum()),
    }


def run_pascient_transfer(aifi: ad.AnnData, onek1k: ad.AnnData) -> tuple[dict, np.ndarray, np.ndarray, pd.Series, pd.Series]:
    """Train on AIFI, predict on OneK1K."""
    pa = patpy.tl.PaSCient(
        sample_key="donor", label_keys=["age"], tasks=["regression"],
        cell_group_key=None, layer="X_pca_shared",
        n_cells=N_CELLS_PER_DONOR,
        batch_size=16 if torch.cuda.is_available() else 8,
        n_epochs=PASCIENT_EPOCHS,
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize=False,
        seed=SEED,
    )
    log("PaSCient: prepare_anndata(aifi, train=True)")
    pa.prepare_anndata(aifi, train=True)
    aifi_emb = pa.get_sample_representations()
    aifi_pred = pa.predict("age")

    # Inference on OneK1K — repoint the wrapper to a new adata and re-extract.
    log("PaSCient: pointing at OneK1K for inference")
    pa.adata = onek1k
    pa.samples = pd.unique(onek1k.obs["donor"]).tolist()
    pa._cell_embeddings = {}
    pa._extract_embeddings(onek1k)
    onek1k_emb = pa.get_sample_representations()
    onek1k_pred = pa.predict("age")

    return aifi_emb.values, onek1k_emb.values, aifi_pred, onek1k_pred


def run_sampleclr_transfer(aifi: ad.AnnData, onek1k: ad.AnnData) -> tuple[np.ndarray, np.ndarray]:
    """Train SampleCLR on AIFI (SSL + FT on age), embed OneK1K with the trained nets."""
    from sampleclr import ContrastiveModel
    from sampleclr.utils import get_sample_representations_from_adata

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    aifi_donors = list(pd.unique(aifi.obs["donor"]))
    onek1k_donors = list(pd.unique(onek1k.obs["donor"]))

    model = ContrastiveModel(
        adata=aifi, sample_key="donor", layer="X_pca_shared",
        tasks={"regression": ["age"]},
        batch_size=8,
        num_epochs_stage1=SAMPLECLR_EPOCHS, num_epochs_stage2=SAMPLECLR_EPOCHS,
        num_warmup_epochs_stage1=max(1, SAMPLECLR_EPOCHS // 5),
        num_warmup_epochs_stage2=max(1, SAMPLECLR_EPOCHS // 5),
        early_stopping_patience=8,
        use_batch_aware_sampler=True,
        batch_sampler_batch_col="batch_id",
        verbose=False, seed=SEED,
    )
    log("SampleCLR: pretrain")
    model.pretrain()
    log("SampleCLR: fine_tune")
    model.fine_tune(val_metric="loss")

    aifi_emb = get_sample_representations_from_adata(
        projector=model.projector, aggregator=model.aggregator,
        adata=aifi, sample_key="donor", layer="X_pca_shared",
        meta_obs_names=aifi_donors, subset_size=N_CELLS_PER_DONOR, device=dev,
    )
    onek1k_emb = get_sample_representations_from_adata(
        projector=model.projector, aggregator=model.aggregator,
        adata=onek1k, sample_key="donor", layer="X_pca_shared",
        meta_obs_names=onek1k_donors, subset_size=N_CELLS_PER_DONOR, device=dev,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(aifi_emb), np.asarray(onek1k_emb), aifi_donors, onek1k_donors


def knn_age_predict(train_emb, train_age, test_emb, k=5):
    """Predict ages on test_emb via KNN regression from (train_emb, train_age)."""
    d = squareform(pdist(np.vstack([test_emb, train_emb]), metric="euclidean"))[:len(test_emb), len(test_emb):]
    k = min(k, train_emb.shape[0])
    nn = np.argsort(d, axis=1)[:, :k]
    return np.asarray(train_age)[nn].mean(axis=1)


def main() -> int:
    runtime = {"status": "running", "patpy": patpy.__version__}
    runtime_path = OUT_DIR / "runtime.json"
    runtime_path.write_text(json.dumps(runtime, indent=2))
    t0 = time.time()

    try:
        log("=== Load AIFI ===")
        aifi = load_aifi()
        log(f"AIFI: {aifi.n_obs:,} cells x {aifi.n_vars} genes, {aifi.obs['donor'].nunique()} donors")

        log("=== Load OneK1K ===")
        onek1k = load_onek1k()
        log(f"OneK1K: {onek1k.n_obs:,} cells x {onek1k.n_vars} genes, {onek1k.obs['donor'].nunique()} donors")

        # 1. Shared genes
        shared = sorted(set(aifi.var_names.astype(str)) & set(onek1k.var_names.astype(str)))
        log(f"shared HVGs: {len(shared)}  (AIFI has {aifi.n_vars}, OneK1K has {onek1k.n_vars})")

        # 2. Fit PCA on AIFI restricted to shared genes
        pca, aifi_X_shared, aifi_idx = fit_shared_pca(aifi, shared, n_pcs=N_PCS)

        # 3. Project AIFI + OneK1K into shared PC space
        log("=== Projecting AIFI + OneK1K into shared PCA space ===")
        aifi.obsm["X_pca_shared"] = pca.transform(aifi_X_shared).astype(np.float32)
        del aifi_X_shared; gc.collect()
        onek1k_idx = onek1k.var_names.get_indexer(shared)
        # OneK1K is sparse — densify in chunks to avoid blowing up memory.
        log(f"  densifying OneK1K[:, shared] of shape ({onek1k.n_obs}, {len(shared)})")
        X_one = onek1k.X[:, onek1k_idx]
        if hasattr(X_one, "toarray"):
            X_one = X_one.toarray()
        X_one = np.asarray(X_one, dtype=np.float32)
        onek1k.obsm["X_pca_shared"] = pca.transform(X_one).astype(np.float32)
        del X_one; gc.collect()
        log(f"  AIFI obsm[X_pca_shared]: {aifi.obsm['X_pca_shared'].shape}")
        log(f"  OneK1K obsm[X_pca_shared]: {onek1k.obsm['X_pca_shared'].shape}")

        # Save PCA artifacts
        np.savez(
            OUT_DIR / "pca_aifi_shared.npz",
            components=pca.components_, mean=pca.mean_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            shared_genes=np.asarray(shared, dtype=object),
        )

        # 4. Cap cells per donor for both cohorts (training + inference)
        log(f"=== Capping cells/donor at {N_CELLS_PER_DONOR} ===")
        aifi_sub = cap_cells_per_donor(aifi, N_CELLS_PER_DONOR, seed=SEED)
        log(f"AIFI capped: {aifi_sub.n_obs:,} cells")
        # OneK1K: many donors are below N_CELLS_PER_DONOR already; cap anyway and let
        # PaSCient's internal padding handle the short donors.
        onek1k_sub = cap_cells_per_donor(onek1k, N_CELLS_PER_DONOR, seed=SEED)
        log(f"OneK1K capped: {onek1k_sub.n_obs:,} cells (median per donor = "
            f"{int(onek1k_sub.obs['donor'].value_counts().median())})")

        # 5. Build donor-level age targets
        aifi_meta = aifi_sub.obs.groupby("donor", observed=True)["age"].first()
        onek1k_meta = onek1k_sub.obs.groupby("donor", observed=True)["age"].first().astype(np.float32)

        rows = []
        all_aifi_emb: dict[str, np.ndarray] = {}
        all_onek1k_emb: dict[str, np.ndarray] = {}
        all_predictions: list[dict] = []

        # --- PaSCient ---
        try:
            t1 = time.time()
            aifi_pa, one_pa, aifi_pred_pa, one_pred_pa = run_pascient_transfer(aifi_sub, onek1k_sub)
            all_aifi_emb["pascient"] = aifi_pa
            all_onek1k_emb["pascient"] = one_pa
            runtime["t_pascient_sec"] = round(time.time() - t1, 1)
            # In-cohort score (AIFI)
            aifi_s = score_age(aifi_meta.reindex(aifi_pred_pa.index).values,
                                aifi_pred_pa.values.astype(np.float64))
            # Transfer score (OneK1K from AIFI-trained model)
            one_s = score_age(onek1k_meta.reindex(one_pred_pa.index).values,
                                one_pred_pa.values.astype(np.float64))
            rows.append({"method": "pascient", "set": "aifi_train", **aifi_s})
            rows.append({"method": "pascient", "set": "onek1k_transfer", **one_s})
            log(f"PaSCient AIFI train R²={aifi_s['r2']:.3f}  OneK1K transfer R²={one_s['r2']:.3f}")
            for donor, p in aifi_pred_pa.items():
                all_predictions.append({"dataset": "aifi", "method": "pascient", "donor": donor,
                                         "y_true": float(aifi_meta.get(donor, np.nan)), "y_pred": float(p)})
            for donor, p in one_pred_pa.items():
                all_predictions.append({"dataset": "onek1k", "method": "pascient", "donor": donor,
                                         "y_true": float(onek1k_meta.get(donor, np.nan)), "y_pred": float(p)})
        except Exception as e:
            log(f"PaSCient failed: {type(e).__name__}: {e}")
            runtime.setdefault("errors", {})["pascient"] = f"{type(e).__name__}: {e}"

        # --- SampleCLR ---
        try:
            t1 = time.time()
            aifi_sc, one_sc, aifi_donors, one_donors = run_sampleclr_transfer(aifi_sub, onek1k_sub)
            all_aifi_emb["sampleclr_ft"] = aifi_sc
            all_onek1k_emb["sampleclr_ft"] = one_sc
            runtime["t_sampleclr_sec"] = round(time.time() - t1, 1)
            # SampleCLR doesn't have a native age-predict head; use KNN regression from
            # AIFI embeddings (train) onto OneK1K embeddings (test).
            y_aifi = aifi_meta.reindex(aifi_donors).values.astype(np.float32)
            y_one = onek1k_meta.reindex(one_donors).values.astype(np.float32)
            aifi_pred_sc = knn_age_predict(aifi_sc, y_aifi, aifi_sc, k=5)   # in-sample (loose)
            one_pred_sc = knn_age_predict(aifi_sc, y_aifi, one_sc, k=5)     # transfer
            aifi_s = score_age(y_aifi, aifi_pred_sc)
            one_s = score_age(y_one, one_pred_sc)
            rows.append({"method": "sampleclr_ft", "set": "aifi_train", **aifi_s})
            rows.append({"method": "sampleclr_ft", "set": "onek1k_transfer", **one_s})
            log(f"SampleCLR AIFI in-sample R²={aifi_s['r2']:.3f}  OneK1K transfer R²={one_s['r2']:.3f}")
            for donor, p in zip(aifi_donors, aifi_pred_sc):
                all_predictions.append({"dataset": "aifi", "method": "sampleclr_ft", "donor": donor,
                                         "y_true": float(aifi_meta.get(donor, np.nan)), "y_pred": float(p)})
            for donor, p in zip(one_donors, one_pred_sc):
                all_predictions.append({"dataset": "onek1k", "method": "sampleclr_ft", "donor": donor,
                                         "y_true": float(onek1k_meta.get(donor, np.nan)), "y_pred": float(p)})
        except Exception as e:
            log(f"SampleCLR failed: {type(e).__name__}: {e}")
            runtime.setdefault("errors", {})["sampleclr"] = f"{type(e).__name__}: {e}"

        # --- save ---
        np.savez(OUT_DIR / "aifi_embeddings.npz", **all_aifi_emb)
        np.savez(OUT_DIR / "onek1k_embeddings.npz", **all_onek1k_emb)
        pd.DataFrame(all_predictions).to_csv(OUT_DIR / "predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(OUT_DIR / "transfer_scores.csv", index=False)
        runtime["status"] = "ok"
        runtime["n_shared_genes"] = int(len(shared))
        log("=== summary ===")
        log(pd.DataFrame(rows).to_string(index=False))
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        runtime["status"] = "failed"
        runtime["error"] = f"{type(e).__name__}: {e}"
        runtime["traceback"] = traceback.format_exc()
    finally:
        runtime["t_total_sec"] = round(time.time() - t0, 1)
        runtime_path.write_text(json.dumps(runtime, indent=2, default=str))

    return 0 if runtime.get("status") == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
