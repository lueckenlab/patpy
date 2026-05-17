"""Run one ``(dataset, method)`` cell of the age-prediction benchmark.

Usage:

    python run_method.py --dataset aging --method pseudobulk
    python run_method.py --dataset onek1k --method sampleclr --smoke

Reads its config from ``configs.py``. Saves into
``data/aging_benchmark/<dataset>/<method>/``:

    embedding.npy      (n_samples, dim)
    distance.npy       (n_samples, n_samples)
    samples.npy        ordered donor ids
    meta.parquet       donor-level metadata (incl. cleaned ``age``)
    knn_scores.csv     KNN score for age + every covariate in cfg.schema
    runtime.json       wall time, peak RSS, status, n_donors, n_cells

The supervised methods (PaSCient, MixMIL, SampleCLR) are fit on
80% of donors (deterministic seed). For an apples-to-apples score, the
KNN evaluation restricts the K-NN reference set to train donors and
predicts held-out test donors' ``age``. The unsupervised methods use the
full embedding for the K-NN evaluation too, with the same train/test
split — see ``score_age_held_out``.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import patpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import CONFIGS, METHODS, OUT_ROOT, DatasetConfig, MethodConfig  # noqa: E402
from data_loader import load_dataset, smoke_load, subsample_cells_per_donor  # noqa: E402


SEED = 42
TEST_FRACTION = 0.2
N_NEIGHBORS = 5


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] RSS={rss_gb():.1f}GB {msg}", flush=True)


def donor_split(donors: list[str], seed: int = SEED, test_fraction: float = TEST_FRACTION):
    """Stable 80/20 donor split. Returns (train_donors, test_donors)."""
    rng = np.random.default_rng(seed)
    donors = np.asarray(donors)
    perm = rng.permutation(len(donors))
    n_test = max(1, int(round(len(donors) * test_fraction)))
    test = set(donors[perm[:n_test]].tolist())
    train = [d for d in donors if d not in test]
    test = [d for d in donors if d in test]
    return train, test


def aggregate_donor_metadata(adata, cfg: DatasetConfig) -> pd.DataFrame:
    """One row per donor: ``age`` + ``age_group`` + every column in ``cfg.sample_cols``."""
    keep_cols = ["age", "age_group"] + cfg.sample_cols
    if cfg.age_col != "age":
        keep_cols.append(cfg.age_col)
    seen, ordered = set(), []
    for c in keep_cols:
        if c not in seen and c in adata.obs.columns:
            seen.add(c)
            ordered.append(c)
    df = adata.obs[[cfg.sample_key] + ordered].copy()
    df[cfg.sample_key] = df[cfg.sample_key].astype(str)
    meta = df.groupby(cfg.sample_key, observed=True).first()
    return meta


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


def run_pseudobulk(adata, cfg: DatasetConfig, train_donors, **_):
    from patpy.tl import Pseudobulk

    pb = Pseudobulk(sample_key=cfg.sample_key, cell_group_key=cfg.cell_type_key, layer=cfg.layer, seed=SEED)
    pb.prepare_anndata(adata)
    dist = pb.calculate_distance_matrix()
    emb = pd.DataFrame(pb.sample_representation, index=pb.samples)
    return _to_arrays(emb, dist)


def run_composition(adata, cfg: DatasetConfig, train_donors, **_):
    from patpy.tl import CellGroupComposition

    comp = CellGroupComposition(
        sample_key=cfg.sample_key, cell_group_key=cfg.cell_type_key, apply_clr=True, seed=SEED,
    )
    comp.prepare_anndata(adata)
    dist = comp.calculate_distance_matrix()
    rep = comp.sample_representation
    emb = rep if isinstance(rep, pd.DataFrame) else pd.DataFrame(rep, index=comp.samples)
    return _to_arrays(emb, dist)


def run_gloscope(adata, cfg: DatasetConfig, train_donors, **_):
    from patpy.tl import GloScope_py

    gs = GloScope_py(sample_key=cfg.sample_key, cell_group_key=cfg.cell_type_key, layer=cfg.layer, seed=SEED, k=25)
    gs.prepare_anndata(adata)
    dist = gs.calculate_distance_matrix()
    # GloScope is distance-only; synthesise an embedding by classical MDS on distances.
    if isinstance(dist, pd.DataFrame):
        emb = _mds_from_distance(dist.values)
        emb.index = dist.index
    else:
        emb = _mds_from_distance(dist)
        emb.index = pd.Index(gs.samples)
    return emb, dist


def run_pascient(adata, cfg: DatasetConfig, train_donors, **_):
    """PaSCient on continuous ``age`` (regression).

    Relies on the local patpy patch that swaps PaSCient's prediction loss
    to MSELoss when ``tasks=["regression"]``. The trained donor embedding
    is used downstream for KNN regression on held-out donors.
    """
    import torch
    from patpy.tl import PaSCient

    n_per_donor = max(1, adata.n_obs // max(1, adata.obs[cfg.sample_key].nunique()))
    pa = PaSCient(
        sample_key=cfg.sample_key,
        label_keys=["age"],
        tasks=["regression"],
        cell_group_key=cfg.cell_type_key,
        layer=cfg.layer,
        n_cells=min(500, n_per_donor),
        batch_size=8,
        n_epochs=2 if _is_smoke(adata) else 6,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=SEED,
    )
    pa.prepare_anndata(adata, train=True)
    rep = pa.get_sample_representations()
    return _to_arrays(rep)


def run_mixmil(adata, cfg: DatasetConfig, train_donors, **_):
    """MixMIL on binary ``age_group``.

    Upstream `mixmil` only offers binomial / categorical likelihoods (no Gaussian),
    so continuous-age regression isn't supported. We train the supervised head on
    ``age_group = age >= 65`` and let the learned donor embedding carry the
    underlying continuous age structure for downstream KNN regression.
    """
    from patpy.tl import MixMIL

    mm = MixMIL(
        sample_key=cfg.sample_key,
        label_keys=["age_group"],
        tasks=["classification"],
        cell_group_key=cfg.cell_type_key,
        layer=cfg.layer,
        likelihood="binomial",
        n_trials=2,
        n_epochs=200 if _is_smoke(adata) else 1500,
        batch_size=8,
        seed=SEED,
    )
    mm.prepare_anndata(adata, train=True)
    rep = mm.get_sample_representations()
    return _to_arrays(rep)


def run_sampleclr(adata, cfg: DatasetConfig, train_donors, batch_aware: bool = True, **_):
    import torch
    from sampleclr import ContrastiveModel
    from sampleclr.utils import get_sample_representations_from_adata

    smoke = _is_smoke(adata)
    sample_order = _donor_order(adata, cfg)
    kwargs = dict(
        adata=adata,
        sample_key=cfg.sample_key,
        layer=cfg.layer,
        tasks={"regression": ["age"]},
        batch_size=8,
        num_epochs_stage1=10 if smoke else 100,
        num_epochs_stage2=10 if smoke else 100,
        num_warmup_epochs_stage1=2 if smoke else 10,
        num_warmup_epochs_stage2=2 if smoke else 10,
        early_stopping_patience=3 if smoke else 10,
        seed=SEED,
    )
    if batch_aware:
        kwargs["use_batch_aware_sampler"] = True
        kwargs["batch_sampler_batch_col"] = cfg.batch_key
    model = ContrastiveModel(**kwargs)
    model.pretrain()
    model.fine_tune(val_metric="loss")
    rep = get_sample_representations_from_adata(
        projector=model.projector,
        aggregator=model.aggregator,
        adata=adata,
        sample_key=cfg.sample_key,
        layer=cfg.layer,
        meta_obs_names=sample_order,
        subset_size=500,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    emb = pd.DataFrame(np.asarray(rep), index=sample_order)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return _to_arrays(emb)


RUNNERS = {
    "pseudobulk": run_pseudobulk,
    "composition": run_composition,
    "gloscope": run_gloscope,
    "pascient": run_pascient,
    "mixmil": run_mixmil,
    "sampleclr": run_sampleclr,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_smoke(adata) -> bool:
    return adata.n_obs <= 50_000


def _donor_order(adata, cfg: DatasetConfig) -> list[str]:
    return list(pd.unique(adata.obs[cfg.sample_key].astype(str)))


def _to_arrays(emb, dist=None):
    """Normalise (embedding, distance) into (DataFrame, ndarray)."""
    if not isinstance(emb, pd.DataFrame):
        emb = pd.DataFrame(np.asarray(emb))
    if dist is None:
        dist = squareform(pdist(emb.values, metric="euclidean")).astype(np.float32)
    elif isinstance(dist, pd.DataFrame):
        dist = dist.values.astype(np.float32)
    else:
        dist = np.asarray(dist, dtype=np.float32)
    return emb, dist


def _mds_from_distance(dist, n_components: int = 16):
    """Classical MDS from a square distance matrix."""
    if isinstance(dist, pd.DataFrame):
        idx, dist = dist.index, dist.values
    else:
        idx = pd.RangeIndex(dist.shape[0])
    n = dist.shape[0]
    D2 = dist.astype(np.float64) ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    w, v = np.linalg.eigh(B)
    order = np.argsort(-w)[: max(1, min(n_components, n - 1))]
    w = np.clip(w[order], 0, None)
    v = v[:, order]
    emb_values = v * np.sqrt(w)
    return pd.DataFrame(emb_values, index=idx)


def score_age_held_out(distances, age_per_donor, train_donors, test_donors, donor_order):
    """Predict held-out donors' ages by KNN, return R² + Spearman + MAE."""
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score, mean_absolute_error

    idx = {d: i for i, d in enumerate(donor_order)}
    train_i = np.array([idx[d] for d in train_donors if d in idx])
    test_i = np.array([idx[d] for d in test_donors if d in idx])
    if len(test_i) == 0:
        return {"r2": np.nan, "spearman": np.nan, "mae": np.nan, "n_test": 0}

    age_arr = pd.Series(age_per_donor).reindex(donor_order).values.astype(float)
    valid_train = train_i[~np.isnan(age_arr[train_i])]
    valid_test = test_i[~np.isnan(age_arr[test_i])]
    if len(valid_train) == 0 or len(valid_test) == 0:
        return {"r2": np.nan, "spearman": np.nan, "mae": np.nan, "n_test": int(len(valid_test))}

    D = distances[np.ix_(valid_test, valid_train)]
    k = min(N_NEIGHBORS, len(valid_train))
    nn = np.argsort(D, axis=1)[:, :k]
    y_pred = age_arr[valid_train][nn].mean(axis=1)
    y_true = age_arr[valid_test]
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(rho) if not np.isnan(rho) else np.nan,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "n_test": int(len(valid_test)),
    }


def score_covariate(distances, target_per_donor, donor_order, task):
    """Score a covariate using leave-one-out-style KNN over all donors.

    Used for technical covariates (batch_id, pool_id) — supervised methods
    don't see them during training, and we want a baseline "is this covariate
    leaked in the embedding".
    """
    from patpy.tl.evaluation import evaluate_representation

    target = pd.Series(target_per_donor).reindex(donor_order).reset_index(drop=True)
    try:
        result = evaluate_representation(
            distances=distances,
            target=target,
            method="knn",
            task=task,
            n_neighbors=N_NEIGHBORS,
        )
        return {"score": float(result.get("score", np.nan)), "metric": result.get("metric")}
    except Exception as e:
        return {"score": np.nan, "metric": None, "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument("--smoke", action="store_true", help="Run on 20 donors x 200 cells subset.")
    parser.add_argument("--smoke-donors", type=int, default=20)
    parser.add_argument("--smoke-cells", type=int, default=200)
    parser.add_argument("--out-suffix", default="", help="Append to output dir (e.g. '_smoke').")
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    mcfg = METHODS[args.method]

    tag = f"{args.dataset}/{args.method}"
    suffix = args.out_suffix or ("_smoke" if args.smoke else "")
    out_dir = OUT_ROOT / f"{args.dataset}{suffix}" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / "runtime.json"

    log(f"=== {tag}  smoke={args.smoke} -> {out_dir} ===")
    runtime = {
        "dataset": args.dataset,
        "method": args.method,
        "smoke": args.smoke,
        "patpy": patpy.__version__,
        "status": "running",
    }
    runtime_path.write_text(json.dumps(runtime, indent=2))

    try:
        t_load = time.time()
        if args.smoke:
            adata = smoke_load(cfg, args.smoke_donors, args.smoke_cells, seed=SEED)
            log(f"smoke subset: n_obs={adata.n_obs:,} n_donors={adata.obs[cfg.sample_key].nunique()}")
        else:
            adata = load_dataset(cfg)
            log(f"loaded {args.dataset}: n_obs={adata.n_obs:,} n_vars={adata.n_vars}")
            adata = patpy.pp.filter_small_samples(adata, sample_key=cfg.sample_key, sample_size_threshold=200)
            if mcfg.cap_cells_per_donor is not None:
                adata = subsample_cells_per_donor(adata, cfg.sample_key, mcfg.cap_cells_per_donor, seed=SEED)
                log(f"capped to {mcfg.cap_cells_per_donor} cells/donor: n_obs={adata.n_obs:,}")
        runtime["t_load_sec"] = round(time.time() - t_load, 1)

        donor_order = _donor_order(adata, cfg)
        train_donors, test_donors = donor_split(donor_order, seed=SEED)
        log(f"donor split: n_train={len(train_donors)} n_test={len(test_donors)}")

        t_fit = time.time()
        emb, dist = RUNNERS[args.method](
            adata, cfg, train_donors=train_donors, batch_aware=mcfg.batch_aware,
        )
        runtime["t_fit_sec"] = round(time.time() - t_fit, 1)
        runtime["n_donors"] = adata.obs[cfg.sample_key].nunique()
        runtime["n_cells"] = int(adata.n_obs)
        runtime["embedding_shape"] = list(emb.shape)
        log(f"fit complete: emb={emb.shape} dist={dist.shape}  elapsed={runtime['t_fit_sec']}s")

        # Align embedding + distance to a single canonical donor_order.
        emb.index = emb.index.astype(str)
        if set(emb.index) >= set(donor_order):
            old_order = list(emb.index)
            emb = emb.reindex(donor_order)
            if dist.shape[0] == len(old_order):
                idx_map = {d: i for i, d in enumerate(old_order)}
                pos = np.array([idx_map[d] for d in donor_order])
                dist = dist[np.ix_(pos, pos)]
        else:
            log("WARN: emb.index doesn't match expected donor_order — using emb.index as canonical.")
            donor_order = list(emb.index)

        # Save the artifacts
        np.save(out_dir / "embedding.npy", emb.values.astype(np.float32))
        np.save(out_dir / "distance.npy", dist.astype(np.float32))
        np.save(out_dir / "samples.npy", np.asarray(donor_order, dtype=object))

        meta = aggregate_donor_metadata(adata, cfg).reindex(donor_order)
        meta.to_parquet(out_dir / "meta.parquet")

        # Score age (held-out donors) + every covariate in schema (all donors)
        age_per_donor = meta["age"]
        score_age = score_age_held_out(dist, age_per_donor, train_donors, test_donors, donor_order)
        log(f"age held-out: R²={score_age['r2']:.3f} Spearman={score_age['spearman']:.3f} MAE={score_age['mae']:.2f}")
        rows = [
            {"covariate": "age", "covariate_type": "relevant", "task": "regression", **score_age},
        ]
        for covariate_type, ctype_dict in cfg.schema.items():
            for col, task in ctype_dict.items():
                if col == "age":
                    continue
                if col not in meta.columns:
                    continue
                # technical/categorical → use full KNN score (no train/test split needed)
                s = score_covariate(dist, meta[col], donor_order, task=task)
                rows.append({"covariate": col, "covariate_type": covariate_type, "task": task, **s})
        knn_df = pd.DataFrame(rows)
        knn_df.to_csv(out_dir / "knn_scores.csv", index=False)

        runtime["status"] = "ok"
        runtime["score_age_r2"] = score_age["r2"]
        runtime["score_age_spearman"] = score_age["spearman"]
    except Exception as e:
        log(f"FAILED: {type(e).__name__}: {e}")
        runtime["status"] = "failed"
        runtime["error"] = f"{type(e).__name__}: {e}"
        runtime["traceback"] = traceback.format_exc()
    finally:
        runtime["t_total_sec"] = round(time.time() - t_load if "t_load" in dir() else 0.0, 1)
        runtime["rss_gb_peak"] = round(rss_gb(), 1)
        runtime_path.write_text(json.dumps(runtime, indent=2, default=str))

    return 0 if runtime.get("status") == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
