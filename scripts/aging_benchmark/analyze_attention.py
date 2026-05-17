"""Turn SampleCLR per-cell attention into biological readouts.

Two outputs per dataset:

1. ``celltype_head_age_corr.parquet`` — for every (cell_type, head) compute
   the per-donor mean attention on cells of that type, then Pearson its
   correlation with donor ``age``. One row per (cell_type, head). Two
   blocks (SSL, FT) joined.

2. ``gene_attention_corr_<stage>.parquet`` — for the top (cell_type, head)
   hits, Pearson correlation between per-cell attention in that head and
   each gene's log-normalised expression in that cell-type's cells.

The data is reused from the existing ``data/aging_benchmark/<dataset>/
sampleclr_attention/`` directory produced by ``run_sampleclr_attention.py``.

Usage:

    python analyze_attention.py --dataset aging
    python analyze_attention.py --dataset onek1k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import CONFIGS, OUT_ROOT, AGING_H5AD  # noqa: E402


TOP_HITS = 12          # number of (cell_type, head) hits to dig into
N_TOP_GENES = 30       # genes per hit to keep in the output


def celltype_head_age_correlation(att: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per (cell_type, head): correlate per-donor mean attention with age."""
    head_cols = [c for c in att.columns if c.startswith("head_")]
    out_rows: list[dict] = []

    for cell_type, sub in att.groupby("cell_type", observed=True):
        donor_means = sub.groupby("donor", observed=True)[head_cols].mean()
        donor_means = donor_means.join(meta["age"], how="inner").dropna(subset=["age"])
        if len(donor_means) < 5:
            continue
        age = donor_means.pop("age").values.astype(float)
        n_cells = int(len(sub))
        for h in head_cols:
            v = donor_means[h].values.astype(float)
            if np.std(v) < 1e-8:
                r = np.nan
            else:
                r = float(np.corrcoef(age, v)[0, 1])
            out_rows.append({
                "cell_type": cell_type,
                "head": h,
                "r": r,
                "n_cells": n_cells,
                "n_donors": int(len(donor_means)),
            })
    return pd.DataFrame(out_rows).sort_values("r", key=lambda s: s.abs(), ascending=False)


def gene_attention_correlation(
    att: pd.DataFrame,
    adata: ad.AnnData,
    hits: pd.DataFrame,
    n_top_genes: int = N_TOP_GENES,
) -> pd.DataFrame:
    """For each (cell_type, head) hit: top genes correlated with attention.

    Uses the cells of that cell_type that were sampled by SampleCLR (the rows
    of ``att``). The expression matrix is taken from ``adata.X`` indexed by
    ``att.obs_idx``.
    """
    out_rows: list[dict] = []
    gene_names = np.asarray(adata.var_names)
    is_sparse = hasattr(adata.X, "toarray")

    for _, hit in hits.iterrows():
        sub = att[att["cell_type"] == hit["cell_type"]]
        if len(sub) < 30:
            continue
        obs_idx = sub["obs_idx"].values
        att_vec = sub[hit["head"]].values.astype(np.float32)

        # Dense per-cell expression (n_cells x n_genes)
        X = adata.X[obs_idx]
        if is_sparse:
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Pearson per gene: corr(att, gene). Vectorised.
        att_centered = att_vec - att_vec.mean()
        att_std = att_vec.std() + 1e-12
        X_centered = X - X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0) + 1e-12
        rs = (att_centered @ X_centered) / (len(att_vec) * att_std * X_std)

        # Top |r|
        top_idx = np.argsort(-np.abs(rs))[:n_top_genes]
        for j in top_idx:
            out_rows.append({
                "cell_type": hit["cell_type"],
                "head": hit["head"],
                "head_age_r": float(hit["r"]),
                "gene": gene_names[j],
                "r": float(rs[j]),
                "n_cells": int(len(att_vec)),
            })
    return pd.DataFrame(out_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(CONFIGS))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    suffix = "_smoke" if args.smoke else ""
    in_dir = OUT_ROOT / f"{args.dataset}{suffix}" / "sampleclr_attention"
    if not (in_dir / "attention_ft.parquet").exists():
        raise SystemExit(f"missing {in_dir / 'attention_ft.parquet'} — run run_sampleclr_attention.py first")
    out_dir = in_dir

    meta = pd.read_parquet(in_dir / "meta.parquet")
    att_ssl = pd.read_parquet(in_dir / "attention_ssl.parquet")
    att_ft = pd.read_parquet(in_dir / "attention_ft.parquet")
    print(f"SSL attention rows: {len(att_ssl):,}  FT attention rows: {len(att_ft):,}")
    print(f"head columns: {[c for c in att_ft.columns if c.startswith('head_')]}")

    # --- 1) (cell_type, head) vs age correlation, SSL + FT ---
    ssl_corr = celltype_head_age_correlation(att_ssl, meta).assign(stage="ssl")
    ft_corr = celltype_head_age_correlation(att_ft, meta).assign(stage="ft")
    corr = pd.concat([ssl_corr, ft_corr], ignore_index=True)
    corr.to_parquet(out_dir / "celltype_head_age_corr.parquet")
    print(f"\nSaved {out_dir / 'celltype_head_age_corr.parquet'}  rows={len(corr)}")
    print("Top (cell_type, head) age correlations after FT:")
    print(ft_corr.head(TOP_HITS).to_string(index=False))

    # --- 2) For top FT hits, gene-attention correlation ---
    if args.dataset == "aging":
        # Load aging adata fully for expression access
        adata = ad.read_h5ad(AGING_H5AD)
    else:
        import patpy
        adata = getattr(patpy.datasets, cfg.loader_arg)(return_dataset_info=False)
    # Restrict to the cells SampleCLR actually saw to keep memory small.
    all_obs_idx = pd.unique(np.concatenate([att_ssl["obs_idx"].values, att_ft["obs_idx"].values]))
    all_obs_idx.sort()
    adata = adata[all_obs_idx].copy()
    # Re-map ``obs_idx`` to the new (subsetted) row positions
    remap = {orig: new for new, orig in enumerate(all_obs_idx.tolist())}
    att_ssl_r = att_ssl.assign(obs_idx=att_ssl["obs_idx"].map(remap))
    att_ft_r  = att_ft.assign(obs_idx=att_ft["obs_idx"].map(remap))
    print(f"adata subset for gene correlation: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    top_ft = ft_corr.head(TOP_HITS).copy()
    gene_corr_ft = gene_attention_correlation(att_ft_r, adata, top_ft, n_top_genes=N_TOP_GENES)
    gene_corr_ft.to_parquet(out_dir / "gene_attention_corr_ft.parquet")
    print(f"Saved {out_dir / 'gene_attention_corr_ft.parquet'}  rows={len(gene_corr_ft)}")

    # Also for SSL, for the same (cell_type, head) hits we picked from FT —
    # this is what shows what fine-tuning *moved*.
    gene_corr_ssl = gene_attention_correlation(att_ssl_r, adata, top_ft, n_top_genes=N_TOP_GENES)
    gene_corr_ssl.to_parquet(out_dir / "gene_attention_corr_ssl.parquet")
    print(f"Saved {out_dir / 'gene_attention_corr_ssl.parquet'}  rows={len(gene_corr_ssl)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
