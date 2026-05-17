"""SampleCLR run that snapshots the aggregator at SSL and at FT and dumps attention.

Usage:

    python run_sampleclr_attention.py --dataset aging
    python run_sampleclr_attention.py --dataset onek1k

Writes into ``data/aging_benchmark/<dataset>/sampleclr_attention/``:

    embedding_ssl.npy            (n_donors, output_dim)
    embedding_ft.npy             (n_donors, output_dim)
    distance_ssl.npy / .npy      (n_donors, n_donors)
    distance_ft.npy / .npy
    samples.npy                  ordered donor IDs
    meta.parquet                 donor-level metadata (incl. ``age``, ``age_group``)
    knn_scores.csv               held-out age KNN for SSL + FT
    attention_ssl.parquet        per-cell: donor, cell_type, head_0…head_{H-1}, obs_idx
    attention_ft.parquet
    runtime.json                 status + timings

500 cells per donor are sampled with a fixed seed; the same cell selection is
used for SSL and FT so the attention deltas are comparable per-cell.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import CONFIGS, OUT_ROOT, DatasetConfig  # noqa: E402
from data_loader import load_dataset, smoke_load, subsample_cells_per_donor  # noqa: E402
from run_method import N_NEIGHBORS, SEED, TEST_FRACTION, donor_split, log  # noqa: E402


SUBSET_SIZE = 500   # cells per donor for inference + attention


def aggregator_with_weights(aggregator, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Call aggregator and force it to return attention weights.

    SampleCLR's MultiHeadAggregationNetwork has a ``return_weights`` kwarg
    on its ``forward()``. We override the (default-False) instance flag for
    the duration of the call.
    """
    output = aggregator(x, return_weights=True)
    if not isinstance(output, tuple):
        raise RuntimeError("aggregator did not return weights — wrong type?")
    return output


def infer_and_attention(
    projector,
    aggregator,
    adata: ad.AnnData,
    cfg: DatasetConfig,
    donor_order: list[str],
    cell_subset_idx: dict[str, np.ndarray],
    device: str,
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Run inference on the cached cell selection.

    Parameters
    ----------
    cell_subset_idx
        ``{donor_id: ndarray of obs indices used for that donor}``. Built once
        from the SSL pass and re-used for FT so per-cell attention is
        directly comparable.

    Returns
    -------
    embedding : (n_donors, output_dim)
    attention_rows : list of dicts (one per cell) with donor, head weights, cell_type, obs_idx
    distance : (n_donors, n_donors)
    """
    projector_was_training = projector.training
    aggregator_was_training = aggregator.training
    projector.eval()
    aggregator.eval()

    output_dim = projector.output_dim
    n_donors = len(donor_order)
    embedding = np.zeros((n_donors, output_dim), dtype=np.float32)
    attention_rows: list[dict] = []

    obs_donor = adata.obs[cfg.sample_key].astype(str).values
    obs_celltype = adata.obs[cfg.cell_type_key].astype(str).values

    with torch.no_grad():
        for i, donor in enumerate(donor_order):
            idx = cell_subset_idx[donor]
            x_np = adata.obsm[cfg.layer][idx].astype(np.float32)
            x = torch.tensor(x_np).unsqueeze(0).to(device)

            agg, weights = aggregator_with_weights(aggregator, x)
            rep = projector(agg)
            embedding[i] = rep.squeeze(0).cpu().numpy()

            w = weights.squeeze(0).cpu().numpy()  # (N, H)
            cell_types = obs_celltype[idx]
            for j in range(w.shape[0]):
                row = {"donor": donor, "cell_type": cell_types[j], "obs_idx": int(idx[j])}
                for h in range(w.shape[1]):
                    row[f"head_{h}"] = float(w[j, h])
                attention_rows.append(row)

    projector.train(projector_was_training)
    aggregator.train(aggregator_was_training)

    distance = squareform(pdist(embedding, metric="euclidean")).astype(np.float32)
    return embedding, attention_rows, distance


def score_age_held_out(distances, age_per_donor, train_donors, test_donors, donor_order):
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


def build_cell_subset(adata: ad.AnnData, cfg: DatasetConfig, donor_order: list[str]) -> dict[str, np.ndarray]:
    """Pick ``SUBSET_SIZE`` random cell indices per donor (deterministic seed)."""
    rng = np.random.default_rng(SEED)
    obs_donor = adata.obs[cfg.sample_key].astype(str).values
    pos = np.arange(adata.n_obs)
    out: dict[str, np.ndarray] = {}
    for donor in donor_order:
        cell_mask = obs_donor == donor
        donor_pos = pos[cell_mask]
        if len(donor_pos) >= SUBSET_SIZE:
            chosen = rng.choice(donor_pos, size=SUBSET_SIZE, replace=False)
        else:
            chosen = rng.choice(donor_pos, size=SUBSET_SIZE, replace=True)
        out[donor] = np.sort(chosen)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-donors", type=int, default=20)
    parser.add_argument("--smoke-cells", type=int, default=200)
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]

    suffix = "_smoke" if args.smoke else ""
    out_dir = OUT_ROOT / f"{args.dataset}{suffix}" / "sampleclr_attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / "runtime.json"

    log(f"=== {args.dataset}/sampleclr_attention smoke={args.smoke} -> {out_dir} ===")
    runtime = {
        "dataset": args.dataset,
        "method": "sampleclr_attention",
        "smoke": args.smoke,
        "status": "running",
    }
    runtime_path.write_text(json.dumps(runtime, indent=2))
    t0 = time.time()

    try:
        if args.smoke:
            adata = smoke_load(cfg, args.smoke_donors, args.smoke_cells, seed=SEED)
        else:
            import patpy
            adata = load_dataset(cfg)
            adata = patpy.pp.filter_small_samples(adata, sample_key=cfg.sample_key, sample_size_threshold=200)
            adata = subsample_cells_per_donor(adata, cfg.sample_key, SUBSET_SIZE, seed=SEED)
        log(f"adata: {adata.n_obs:,} cells, {adata.obs[cfg.sample_key].nunique()} donors")
        runtime["t_load_sec"] = round(time.time() - t0, 1)

        donor_order = list(pd.unique(adata.obs[cfg.sample_key].astype(str)))
        train_donors, test_donors = donor_split(donor_order, seed=SEED)

        # Build cell selection once so SSL and FT see the same cells per donor
        cell_subset_idx = build_cell_subset(adata, cfg, donor_order)

        # --- SampleCLR with batch-aware sampler, regression on age ---
        from sampleclr import ContrastiveModel

        smoke = args.smoke
        kwargs = dict(
            adata=adata,
            sample_key=cfg.sample_key,
            layer=cfg.layer,
            tasks={"regression": ["age"]},
            batch_size=8,
            num_epochs_stage1=10 if smoke else 40,
            num_epochs_stage2=10 if smoke else 40,
            num_warmup_epochs_stage1=2 if smoke else 5,
            num_warmup_epochs_stage2=2 if smoke else 5,
            early_stopping_patience=3 if smoke else 8,
            use_batch_aware_sampler=True,
            batch_sampler_batch_col=cfg.batch_key,
            seed=SEED,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        t_train = time.time()
        model = ContrastiveModel(**kwargs)

        log("--- pretrain (SSL) ---")
        model.pretrain()
        runtime["t_ssl_sec"] = round(time.time() - t_train, 1)
        log("--- SSL inference + attention ---")
        emb_ssl, att_ssl_rows, dist_ssl = infer_and_attention(
            model.projector, model.aggregator, adata, cfg, donor_order, cell_subset_idx, device,
        )
        log(f"SSL: emb={emb_ssl.shape} attention_rows={len(att_ssl_rows)}")

        t_ft = time.time()
        log("--- fine_tune (supervised on age) ---")
        model.fine_tune(val_metric="loss")
        runtime["t_ft_sec"] = round(time.time() - t_ft, 1)
        log("--- FT inference + attention ---")
        emb_ft, att_ft_rows, dist_ft = infer_and_attention(
            model.projector, model.aggregator, adata, cfg, donor_order, cell_subset_idx, device,
        )
        log(f"FT:  emb={emb_ft.shape}  attention_rows={len(att_ft_rows)}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- save ---
        np.save(out_dir / "embedding_ssl.npy", emb_ssl)
        np.save(out_dir / "embedding_ft.npy", emb_ft)
        np.save(out_dir / "distance_ssl.npy", dist_ssl)
        np.save(out_dir / "distance_ft.npy", dist_ft)
        np.save(out_dir / "samples.npy", np.asarray(donor_order, dtype=object))

        # Donor-level metadata
        meta_cols = ["age", "age_group", cfg.age_col] + cfg.sample_cols
        seen, ordered = set(), []
        for c in meta_cols:
            if c in adata.obs.columns and c not in seen:
                seen.add(c)
                ordered.append(c)
        df_obs = adata.obs[[cfg.sample_key] + ordered].copy()
        df_obs[cfg.sample_key] = df_obs[cfg.sample_key].astype(str)
        meta = df_obs.groupby(cfg.sample_key, observed=True).first().reindex(donor_order)
        meta.to_parquet(out_dir / "meta.parquet")

        # Score age held-out for SSL and FT
        age_per_donor = meta["age"]
        score_ssl = score_age_held_out(dist_ssl, age_per_donor, train_donors, test_donors, donor_order)
        score_ft = score_age_held_out(dist_ft, age_per_donor, train_donors, test_donors, donor_order)
        log(f"SSL age held-out: R²={score_ssl['r2']:.3f} Spearman={score_ssl['spearman']:.3f}")
        log(f"FT  age held-out: R²={score_ft['r2']:.3f} Spearman={score_ft['spearman']:.3f}")
        knn = pd.DataFrame([
            {"covariate": "age", "stage": "ssl", **score_ssl},
            {"covariate": "age", "stage": "ft",  **score_ft},
        ])
        knn.to_csv(out_dir / "knn_scores.csv", index=False)

        # Save attention as parquet (small per-cell rows)
        pd.DataFrame(att_ssl_rows).to_parquet(out_dir / "attention_ssl.parquet")
        pd.DataFrame(att_ft_rows).to_parquet(out_dir / "attention_ft.parquet")

        runtime.update({
            "status": "ok",
            "n_donors": int(len(donor_order)),
            "n_cells_per_donor": int(SUBSET_SIZE),
            "n_heads": int(att_ft_rows[0]["head_0"] is not None and (len(att_ft_rows[0]) - 3)),
            "score_age_r2_ssl": score_ssl["r2"],
            "score_age_r2_ft": score_ft["r2"],
            "score_age_spearman_ssl": score_ssl["spearman"],
            "score_age_spearman_ft":  score_ft["spearman"],
        })
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
