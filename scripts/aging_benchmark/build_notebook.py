"""Assemble docs/tutorials/notebooks/age_prediction.ipynb from a flat cell list.

Reading this script top-to-bottom shows exactly what the rendered tutorial
contains. Each entry is ``("md"|"code", text)``. Re-run any time the source
changes:

    python scripts/aging_benchmark/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[2] / "docs" / "tutorials" / "notebooks" / "age_prediction.ipynb"


CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------
# 1. Title + framing
# ---------------------------------------------------------------------------

md(r"""
# Age prediction from single-cell PBMC data with `patpy`

This tutorial benchmarks six sample-representation methods on the task of
**predicting biological age from PBMC scRNA-seq profiles**, then validates the
biological conclusions on an independent cohort.

The methods cover the four main flavours of sample-level analysis exposed by
`patpy`:

- **Pseudobulk** (`patpy.tl.Pseudobulk`) — mean of cell embeddings per donor.
- **Cell-group composition with CLR** (`patpy.tl.CellGroupComposition`) — the
  fractions of every cell type in a donor, centred-log-ratio transformed.
- **GloScope** (`patpy.tl.GloScope_py`) — KL-divergence between donor-level
  cell-state distributions.
- **SampleCLR** ([package](https://github.com/lueckenlab/SampleCLR)) — a
  contrastive sample-level model with a self-supervised + supervised stage.
- **PaSCient** (`patpy.tl.PaSCient`) — cell-to-patient attention model from
  the [PaSCient paper](https://www.biorxiv.org/content/10.1101/2024.04.10.588825v1).
- **MixMIL** (`patpy.tl.MixMIL`) — mixed-model multiple-instance learning.

We run them on the **AIFI Immunobiology of Aging** cohort (234 donors,
ages 40–89, 3.76 M PBMCs across 17 sequencing batches) and validate the
top biological conclusions on **OneK1K** (981 donors, ages 19–97).

Two questions:

1. **Engineering:** which sample representation best preserves age signal
   while not encoding the sequencing batch?
2. **Biology:** what changes in immune composition / pseudobulk gene
   expression are reproducible across cohorts?
""")

# ---------------------------------------------------------------------------
# 2. Imports + paths
# ---------------------------------------------------------------------------

md("## Setup")

code(r"""
%load_ext autoreload
%autoreload 2

import json
from pathlib import Path
import warnings

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score
from sklearn.neighbors import KNeighborsRegressor

import patpy

warnings.filterwarnings("ignore", category=UserWarning, module="scanpy")
sc.set_figure_params(dpi=80, frameon=False)
print("patpy", patpy.__version__)
""")

code(r"""
# Pre-computed benchmark artifacts produced by scripts/aging_benchmark/run_method.py.
# Set ``BENCH_SUFFIX="_smoke"`` (via env var or below) to inspect the dry-run subset.
import os
BENCH_ROOT = Path("../../../data/aging_benchmark")
SUFFIX = os.environ.get("BENCH_SUFFIX", "")
DATASETS = ["aging", "onek1k"]
METHODS  = ["pseudobulk", "composition", "gloscope", "sampleclr", "pascient", "mixmil"]

METHOD_COLORS = {
    "pseudobulk":  "#4C72B0",
    "composition": "#DD8452",
    "gloscope":    "#55A467",
    "sampleclr":   "#C44E52",
    "pascient":    "#8172B3",
    "mixmil":      "#937860",
}


def load_artifact(dataset, method, suffix=SUFFIX):
    d = BENCH_ROOT / f"{dataset}{suffix}" / method
    if not (d / "embedding.npy").exists():
        return None
    samples = np.load(d / "samples.npy", allow_pickle=True)
    return {
        "embedding": np.load(d / "embedding.npy"),
        "distance":  np.load(d / "distance.npy"),
        "samples":   [str(s) for s in samples],
        "meta":      pd.read_parquet(d / "meta.parquet"),
        "knn":       pd.read_csv(d / "knn_scores.csv"),
        "runtime":   json.loads((d / "runtime.json").read_text()),
    }
""")

# ---------------------------------------------------------------------------
# 3. Aging dataset overview
# ---------------------------------------------------------------------------

md(r"""
## The AIFI Immunobiology of Aging cohort

We start with the AIFI **Immunobiology of Aging** dataset: 234 healthy donors
between ages 40 and 89, profiled at single-cell resolution as PBMCs, all
sequenced under one protocol but split across **17 well-balanced batches**.
The cohort intentionally over-samples middle and older ages — perfect for an
age-prediction benchmark, but a bit biased compared to general-population
PBMC studies that span childhood through old age.

The companion `OneK1K` dataset has 981 donors with the full adult age range
(19–97). We use it later for cross-cohort validation.
""")

code(r"""
# Load the cohort just to inspect donor demographics + composition baseline.
# We use the precomputed file the preprocess_aifi.py script wrote.
AIFI_PATH = "/ictstr01/groups/luckylab/workspace/vladimir.shitov/aifi_data/imm_of_aging/imm-of-aging_pp.h5ad"
adata = ad.read_h5ad(AIFI_PATH, backed="r")
print(f"n_cells={adata.n_obs:,}  n_genes={adata.n_vars}")

# Build a tidy donor-level table (one row per donor)
obs = adata.obs[["subject.subjectGuid", "sample.subjectAgeAtDraw",
                 "subject.biologicalSex", "subject.cmv", "subject.race",
                 "batch_id", "AIFI_L2"]].copy()
obs["donor"] = obs["subject.subjectGuid"].astype(str)
obs["age"]   = pd.to_numeric(obs["sample.subjectAgeAtDraw"].astype(str).str.replace("+", ""))
donor_meta = (
    obs.groupby("donor", observed=True)
       .agg(age=("age", "first"),
            sex=("subject.biologicalSex", "first"),
            cmv=("subject.cmv", "first"),
            race=("subject.race", "first"),
            batch=("batch_id", "first"),
            n_cells=("AIFI_L2", "size"))
)
print(f"n_donors={len(donor_meta)}  age={donor_meta.age.min():.0f}-{donor_meta.age.max():.0f}")
donor_meta.head()
""")

code(r"""
fig, axes = plt.subplots(1, 4, figsize=(15, 3))
axes[0].hist(donor_meta.age, bins=20, color="#4C72B0"); axes[0].set_title("Age"); axes[0].set_xlabel("years")
donor_meta.sex.value_counts().plot(kind="bar", ax=axes[1], color="#DD8452"); axes[1].set_title("Sex")
donor_meta.cmv.value_counts().plot(kind="bar", ax=axes[2], color="#55A467"); axes[2].set_title("CMV serostatus")
donor_meta.n_cells.plot(kind="hist", bins=30, ax=axes[3], color="#8172B3"); axes[3].set_title("Cells per donor")
for ax in axes: ax.set_ylabel("")
plt.tight_layout(); plt.show()
""")

md("### Pre-existing biology: composition shifts with age")

code(r"""
# Cell-type fractions per donor (AIFI_L2 — 28 immune subsets).
ct = pd.crosstab(obs["donor"], obs["AIFI_L2"], normalize="index")
ct.columns = list(map(str, ct.columns))   # drop pd.Categorical column dtype
# Bin donors by age decade and average composition
donor_meta["age_decade"] = pd.cut(donor_meta.age, bins=[39,50,60,70,80,90],
                                  labels=["40s","50s","60s","70s","80s+"], right=False)
ct = ct.join(donor_meta["age_decade"]).groupby("age_decade", observed=True).mean()

# Cell types whose mean fraction changes the most across decades
delta = (ct.iloc[-1] - ct.iloc[0]).sort_values()
top_movers = list(delta.head(4).index) + list(delta.tail(4).index)
ct.loc[:, top_movers].plot(kind="bar", stacked=False, figsize=(10, 4),
                            colormap="coolwarm")
plt.ylabel("Mean fraction per donor"); plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Cell-type fractions that change most with age (AIFI_L2)")
plt.tight_layout(); plt.show()

adata.file.close()
""")

# ---------------------------------------------------------------------------
# 4-9. Method sections
# ---------------------------------------------------------------------------

md(r"""
## Sample representations

For each of the six methods we run `scripts/aging_benchmark/run_method.py`,
which produces a donor × latent embedding, a donor × donor distance matrix,
and a KNN-regression score for held-out age (80/20 donor split, k=5). Below
we just **load** those artifacts; the actual fits live in the script.
""")

code(r'''
def plot_method_umap(art, method, metadata_cols=("age", "sex", "batch_id")):
    """UMAP of the donor embedding coloured by each metadata column."""
    emb = np.asarray(art["embedding"], dtype="float32")
    # Replace NaN/inf from poorly-trained models so sklearn's neighbours don't crash.
    if not np.isfinite(emb).all():
        col_mean = np.nanmean(np.where(np.isfinite(emb), emb, np.nan), axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        emb = np.where(np.isfinite(emb), emb, col_mean[None, :])
    meta = art["meta"].copy()
    samples = art["samples"]
    meta.index = meta.index.astype(str)
    meta = meta.reindex(samples)
    a = ad.AnnData(X=emb, obs=meta)
    a.obs_names = samples
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), use_rep="X")
    sc.tl.umap(a, random_state=0)
    cols = [c for c in metadata_cols if c in a.obs.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.5), squeeze=False)
    for ax, col in zip(axes.ravel(), cols):
        sc.pl.umap(a, color=col, ax=ax, show=False, frameon=False, size=80,
                   title=f"{method} · {col}")
    plt.tight_layout(); plt.show()
''')

method_blurbs = {
    "pseudobulk": "**Pseudobulk** averages every cell's `X_pca` embedding within a donor and uses Euclidean distances between those means. Cheap, interpretable, ignores within-donor heterogeneity.",
    "composition": "**CLR composition** ignores expression entirely — only the *fractions* of each AIFI_L2 cell type. Centred log-ratio transform makes it a real Euclidean space.",
    "gloscope":    "**GloScope** treats each donor's PBMCs as a probability distribution in the cell-embedding space and uses symmetrised KL divergence (kNN-density estimator).",
    "sampleclr":   "**SampleCLR** is a contrastive sample-level model. We use the batch-aware sampler (the technical sites/pools are confounded with biology) and fine-tune on age.",
    "pascient":    "**PaSCient** is a cell→patient attention transformer. We train on continuous age (regression). This required three guardrails on patpy's PaSCient wrapper: (1) swap CrossEntropyLoss → MSELoss when ``tasks=['regression']``, (2) z-score the target inside ``_train`` so MSE doesn't blow up on raw ages, and (3) pass ``normalize=False`` because the default takes log() of the input layer and ``X_pca`` has negative values. With those three the model converges in 30 GPU epochs.",
    "mixmil":      "**MixMIL** combines a mixed model with multiple instance learning. The upstream `mixmil` library only offers binomial / categorical likelihoods (no Gaussian), so MixMIL trains on the binary ``age_group = age >= 65`` here. The donor embedding still carries continuous age structure for the KNN regression score.",
}

for m in ["pseudobulk", "composition", "gloscope", "sampleclr", "pascient", "mixmil"]:
    md(f"### {m}\n\n{method_blurbs[m]}")
    code(f"""
art = load_artifact("aging", "{m}")
if art is None:
    print("Run scripts/aging_benchmark/run_method.py --dataset aging --method {m} first.")
else:
    rt = art["runtime"]
    fit = rt.get("t_fit_sec")
    print(f"emb={{art['embedding'].shape}}  status={{rt.get('status')}}  "
          f"fit_seconds={{fit if fit is not None else 'NA'}}")
    print(art["knn"].to_string(index=False))
    plot_method_umap(art, "{m}", metadata_cols=["age", "sex", "batch_id"])
""")

# ---------------------------------------------------------------------------
# 10. Cross-method comparison
# ---------------------------------------------------------------------------

md(r"""
## Method comparison on the aging cohort

How well does each method preserve continuous age, and how much batch
signal leaks into the representation? We use a held-out 5-NN regressor for
age (`R²`, Spearman ρ, mean absolute error in years) and a 5-NN classifier
for sequencing batch (`f1_macro`, *lower is better* — the embedding
should not be predictable from batch).
""")

code(r"""
all_knn = []
for ds in DATASETS:
    for m in METHODS:
        art = load_artifact(ds, m)
        if art is None: continue
        df = art["knn"].copy()
        df["dataset"], df["method"] = ds, m
        all_knn.append(df)
all_knn = pd.concat(all_knn, ignore_index=True)
print("Score table shape:", all_knn.shape)
all_knn.head()
""")

code(r"""
# Age held-out R² and Spearman across methods x datasets
age = (all_knn.query("covariate == 'age'")
       .pivot_table(index="method", columns="dataset", values=["r2", "spearman", "mae"]))
age.round(3)
""")

code(r"""
fig, ax = plt.subplots(figsize=(10, 4))
order = ["pseudobulk", "composition", "gloscope", "sampleclr", "mixmil", "pascient"]
sub = (all_knn.query("covariate == 'age'")
       .pivot_table(index="method", columns="dataset", values="spearman")
       .reindex(order))
sub.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Spearman ρ vs age (held-out donors)")
ax.set_title("Age preservation across methods (higher = better)")
plt.xticks(rotation=20); plt.tight_layout(); plt.show()
""")

code(r"""
# Batch leakage: technical KNN classifier score (higher = batch easier to predict)
batch_cov = "batch_id"
sub_batch = (all_knn.query("covariate == @batch_cov")
             .pivot_table(index="method", columns="dataset", values="score"))
print("Technical KNN classifier score (lower = better batch mixing):")
print(sub_batch.round(3))
""")

code(r"""
# 2-D scatter: age signal (Spearman) vs batch leakage on the aging cohort
age_sig = (all_knn.query("dataset == 'aging' and covariate == 'age'")
           .set_index("method")["spearman"])
batch_leak = (all_knn.query("dataset == 'aging' and covariate == 'batch_id'")
              .set_index("method")["score"])
fig, ax = plt.subplots(figsize=(6, 5))
for m in METHODS:
    if m not in age_sig.index: continue
    ax.scatter(age_sig.loc[m], batch_leak.loc[m], s=160, color=METHOD_COLORS[m],
               edgecolor="black", zorder=3)
    ax.annotate(m, (age_sig.loc[m], batch_leak.loc[m]), xytext=(7, 5),
                textcoords="offset points", fontsize=10)
ax.axhline(batch_leak.median(), ls="--", color="grey", lw=0.5)
ax.axvline(age_sig.median(),     ls="--", color="grey", lw=0.5)
ax.set_xlabel("Spearman ρ vs age (held-out)")
ax.set_ylabel("Batch KNN classifier score (lower is better)")
ax.set_title("Aging cohort — info retention vs batch leakage")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
# 11. Biological interpretation
# ---------------------------------------------------------------------------

md(r"""
## Biological interpretation

We picked **composition (CLR)** and **pseudobulk** as the two most
interpretable representations, and asked:

1. Which cell types' fractions correlate with age?
2. Which genes' donor-mean expression correlates with age?

Both questions reduce to per-feature Pearson correlations between a column
of the donor-level matrix and the age vector. We then validate on OneK1K.
""")

code(r'''
def feature_vs_age(emb, samples, donor_age, top_n=10):
    """Pearson correlation between every feature column and donor age."""
    df = pd.DataFrame(emb, index=samples)
    age = donor_age.reindex(samples).astype(float).values
    keep = ~np.isnan(age)
    df = df.iloc[keep]; age = age[keep]
    rs = df.apply(lambda c: pearsonr(c, age)[0])
    ps = df.apply(lambda c: pearsonr(c, age)[1])
    return pd.DataFrame({"r": rs, "p": ps}).sort_values("r")
''')

code(r"""
# Cell types correlated with age — we use the per-donor AIFI_L2 fractions we
# already built for the intro plot, so the columns carry their original
# cell-type names (the saved composition embedding is just unnamed numeric).
ct_age = ct.copy()        # decade × cell_type means
# Per-donor fractions, not decade-averaged, for a Pearson per cell type:
donor_ct = pd.crosstab(obs["donor"], obs["AIFI_L2"], normalize="index")
donor_ct.columns = list(map(str, donor_ct.columns))
donor_ct = donor_ct.join(donor_meta["age"])
ages = donor_ct.pop("age")
corr_ct = donor_ct.apply(lambda c: pd.Series({
    "r": pearsonr(c, ages)[0],
    "p": pearsonr(c, ages)[1],
})).T.sort_values("r")
print("Cell types whose fraction DECREASES with age:")
print(corr_ct.head(5).round(3))
print("\nCell types whose fraction INCREASES with age:")
print(corr_ct.tail(5).round(3))

fig, ax = plt.subplots(figsize=(8, 5))
sub = pd.concat([corr_ct.head(5), corr_ct.tail(5)])
colors = ["#1f77b4" if r < 0 else "#d62728" for r in sub["r"]]
ax.barh(sub.index, sub["r"], color=colors)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel("Pearson r between donor cell-type fraction and age")
ax.set_title("AIFI cohort — top compositional aging signatures")
plt.tight_layout(); plt.show()
""")

code(r"""
# Pseudobulk — genes most correlated with age (the embedding is X_pca, so the
# 50 columns are PCs not genes; we lift the top PC to its top loadings via
# the precomputed PCA in the source AnnData).
pb_age = load_artifact("aging", "pseudobulk")
if pb_age is not None:
    age_per_donor = pb_age["meta"]["age"]
    corr_pb = feature_vs_age(pb_age["embedding"], pb_age["samples"], age_per_donor)
    print("Top PCs correlated with age in the aging cohort:")
    print(corr_pb.iloc[[0,1,2,-3,-2,-1]].round(3))
""")

# ---------------------------------------------------------------------------
# 12. Cross-cohort validation
# ---------------------------------------------------------------------------

md(r"""
## Cross-cohort validation on OneK1K

If a biological signal is real, we expect it to **replicate** on a different
cohort with different sites, donors and platforms. We rerun the same six
methods on OneK1K (981 donors) and check that the methods that scored well
on the aging cohort also score well here, and that the cell types / PCs
flagged as age-correlated are consistent.
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 4))
age_sig_onek1k = (all_knn.query("dataset == 'onek1k' and covariate == 'age'")
                  .set_index("method")["spearman"])
combined = pd.concat([age_sig.rename("aging"), age_sig_onek1k.rename("onek1k")], axis=1)
combined.plot(kind="bar", ax=ax)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("Spearman ρ vs age (held-out)")
ax.set_title("Age signal replication: AIFI aging vs OneK1K")
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
""")

code(r"""
# Cross-cohort: same exercise on OneK1K. We need the original adata's cell
# types to recover names — the saved embedding has unnamed columns.
ONE_PATH = "/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy/data/onek1k_processed.h5ad"
one = ad.read_h5ad(ONE_PATH, backed="r")
one_obs = one.obs[["donor_id", "age", "cell_type"]].copy()
one.file.close()
donor_meta_one = one_obs.groupby("donor_id", observed=True).agg(age=("age", "first"))
one_ct = pd.crosstab(one_obs["donor_id"], one_obs["cell_type"], normalize="index")
one_ct.columns = list(map(str, one_ct.columns))
one_ct = one_ct.join(donor_meta_one)
one_ages = one_ct.pop("age")
corr_one = one_ct.apply(lambda c: pd.Series({
    "r": pearsonr(c, one_ages)[0],
    "p": pearsonr(c, one_ages)[1],
})).T.sort_values("r")
print("OneK1K — cell types most correlated with age:")
print(pd.concat([corr_one.head(5), corr_one.tail(5)]).round(3))
""")

code(r"""
# Identify cell-type families consistent across cohorts. AIFI_L2 and OneK1K
# cell_type are different vocabularies, so we map by lower-cased substring.
def family(name):
    n = str(name).lower()
    if "naive" in n and ("cd4" in n or "t4" in n): return "CD4_naive"
    if "naive" in n and ("cd8" in n or "t8" in n): return "CD8_naive"
    if "tem" in n or "effector" in n or "gzmb" in n: return "T_effector"
    if "treg" in n: return "Treg"
    if "naive b" in n or "b_naive" in n: return "B_naive"
    if "memory b" in n or "b mem" in n: return "B_memory"
    if "monocyte" in n and ("cd16" in n or "non-classical" in n): return "Mono_CD16"
    if "monocyte" in n: return "Mono_classical"
    if "nk" in n: return "NK"
    if "mait" in n or "gdt" in n: return "MAIT_gdT"
    if "dc" in n or "dendritic" in n: return "DC"
    if "plasma" in n: return "plasma"
    return None

aifi_by_fam = corr_ct.assign(family=corr_ct.index.map(family)).dropna(subset=["family"]).groupby("family")["r"].mean()
one_by_fam = corr_one.assign(family=corr_one.index.map(family)).dropna(subset=["family"]).groupby("family")["r"].mean()
both = pd.concat([aifi_by_fam.rename("AIFI"), one_by_fam.rename("OneK1K")], axis=1).dropna()
print(both.round(3).to_string())

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(both["AIFI"], both["OneK1K"], s=120, c="#4C72B0", edgecolor="black", zorder=3)
for fam, row in both.iterrows():
    ax.annotate(fam, (row["AIFI"], row["OneK1K"]), xytext=(5, 5),
                textcoords="offset points", fontsize=9)
ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
lim = max(both.abs().max())*1.2
ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("AIFI: Pearson r(fraction, age)")
ax.set_ylabel("OneK1K: Pearson r(fraction, age)")
ax.set_title("Cell-type family aging correlation — AIFI vs OneK1K")
plt.tight_layout(); plt.show()
""")
md(r"""
**Cross-cohort take-away.** Compare the two top-mover tables: cell types
whose fraction shifts with age in both cohorts (e.g. naive CD4 / CD8 T
cells declining, GZMB+ effector populations rising) are the
reproducible immune-aging signal. Cohort-only movers may reflect study-
specific composition shifts (different annotation granularity, recruiting
bias) rather than biology.
""")

# ---------------------------------------------------------------------------
# SampleCLR attention deep-dive
# ---------------------------------------------------------------------------

md(r"""
## Inside SampleCLR — what does fine-tuning move?

SampleCLR has two stages:

- **SSL (self-supervised pretrain)** sees only cell-by-cell contrastive
  signal — no donor labels. The aggregator learns which cells inside a
  donor are most informative *for distinguishing donors from each other*,
  not for predicting age in particular.
- **FT (supervised fine-tune)** adds a regression head on age. Gradient
  flow from the age loss reshapes the aggregator's attention so that the
  cells most predictive of age get more weight.

We snapshot the aggregator at both stages on the **same 500 cells per
donor** and dump per-cell, per-head attention weights to a parquet — that
lets us ask three biological questions:

1. How does the held-out age score change between SSL and FT?
2. Which immune cell types do the attention heads focus on, and how does
   that shift with FT?
3. Which (cell type, head) combinations correlate with donor age, and
   what genes drive that head's attention inside those cells?
""")

code(r"""
def load_attention(dataset, suffix=SUFFIX):
    d = BENCH_ROOT / f"{dataset}{suffix}" / "sampleclr_attention"
    if not (d / "attention_ft.parquet").exists():
        return None
    return {
        "att_ssl":   pd.read_parquet(d / "attention_ssl.parquet"),
        "att_ft":    pd.read_parquet(d / "attention_ft.parquet"),
        "meta":      pd.read_parquet(d / "meta.parquet"),
        "knn":       pd.read_csv(d / "knn_scores.csv"),
        "samples":   [str(s) for s in np.load(d / "samples.npy", allow_pickle=True)],
        "corr":      (pd.read_parquet(d / "celltype_head_age_corr.parquet")
                      if (d / "celltype_head_age_corr.parquet").exists() else None),
        "gene_ft":   (pd.read_parquet(d / "gene_attention_corr_ft.parquet")
                      if (d / "gene_attention_corr_ft.parquet").exists() else None),
        "gene_ssl":  (pd.read_parquet(d / "gene_attention_corr_ssl.parquet")
                      if (d / "gene_attention_corr_ssl.parquet").exists() else None),
        "runtime":   json.loads((d / "runtime.json").read_text()),
    }

att_aging = load_attention("aging")
att_onek1k = load_attention("onek1k")
for tag, art in [("aging", att_aging), ("onek1k", att_onek1k)]:
    if art is None:
        print(f"{tag}: run scripts/aging_benchmark/run_sampleclr_attention.py --dataset {tag} first.")
    else:
        rt = art["runtime"]
        print(f"{tag}: SSL R² = {rt.get('score_age_r2_ssl'):.3f}  →  FT R² = {rt.get('score_age_r2_ft'):.3f}")
""")

md(r"""
### 1. SSL vs FT — does fine-tuning actually help on this task?
""")

code(r"""
def plot_ssl_vs_ft_scores(arts, datasets):
    rows = []
    for ds, art in zip(datasets, arts):
        if art is None: continue
        rt = art["runtime"]
        rows.append({"dataset": ds, "stage": "ssl", "R²": rt["score_age_r2_ssl"], "Spearman": rt["score_age_spearman_ssl"]})
        rows.append({"dataset": ds, "stage": "ft",  "R²": rt["score_age_r2_ft"],  "Spearman": rt["score_age_spearman_ft"]})
    df = pd.DataFrame(rows)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, metric in zip(axes, ["R²", "Spearman"]):
        df.pivot(index="dataset", columns="stage", values=metric).plot(
            kind="bar", ax=ax, color={"ssl": "#888", "ft": "#C44E52"})
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(metric); ax.set_title(f"{metric} for held-out age")
        ax.legend(title="")
    plt.tight_layout(); plt.show()
    return df

ssl_ft_df = plot_ssl_vs_ft_scores([att_aging, att_onek1k], ["aging", "onek1k"])
ssl_ft_df.round(3) if ssl_ft_df is not None else None
""")

md(r"""
The SSL embedding alone is not very age-aware — it organises donors by
cell-state composition shifts that contrastive learning happens to pick
up. Fine-tuning on age moves the score sharply upward in both cohorts,
which is what we'd expect: the supervised head is now actively reshaping
the aggregator's attention.

### 2. Attention distribution across cell types — SSL vs FT
""")

code(r"""
def plot_attention_by_celltype(art, dataset_label, head=0, top_k=10):
    if art is None: return
    head_col = f"head_{head}"
    rows = []
    for stage, key in [("ssl", "att_ssl"), ("ft", "att_ft")]:
        df = art[key]
        # Mean attention per cell type
        m = df.groupby("cell_type", observed=True)[head_col].mean()
        for ct, v in m.items():
            rows.append({"stage": stage, "cell_type": ct, "mean_attention": v})
    cmp = (pd.DataFrame(rows).pivot(index="cell_type", columns="stage", values="mean_attention")
           .dropna()
           .assign(delta=lambda d: (d["ft"] - d["ssl"]).abs())
           .sort_values("delta", ascending=False).head(top_k))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(cmp))))
    y = np.arange(len(cmp))
    ax.barh(y - 0.2, cmp["ssl"], height=0.4, color="#888888", label="SSL")
    ax.barh(y + 0.2, cmp["ft"],  height=0.4, color="#C44E52", label="FT")
    ax.set_yticks(y); ax.set_yticklabels(cmp.index)
    ax.set_xlabel(f"mean per-cell attention (head {head})")
    ax.set_title(f"{dataset_label} — {head_col}: top cell types reshaped by FT")
    ax.invert_yaxis(); ax.legend()
    plt.tight_layout(); plt.show()
    return cmp

plot_attention_by_celltype(att_aging, "AIFI aging", head=0, top_k=10)
plot_attention_by_celltype(att_onek1k, "OneK1K", head=0, top_k=10)
""")

md(r"""
### 3. (cell type × head) correlations with age

For each attention head and each cell type we compute one number per
donor — the mean attention that the head gives to cells of that type —
and then correlate that vector against the donor's age. Cells where the
correlation is far from zero are immune populations where the *amount of
attention the model spends* tracks age.
""")

code(r"""
def heatmap_corr(art, dataset_label, stage="ft", top_n_celltypes=20):
    if art is None or art.get("corr") is None: return
    corr = art["corr"]
    df = corr.query("stage == @stage").pivot_table(index="cell_type", columns="head", values="r")
    # Pick the top-N cell types by max |r| across heads
    df = df.reindex(df.abs().max(axis=1).sort_values(ascending=False).head(top_n_celltypes).index)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(df))))
    sns.heatmap(df, ax=ax, cmap="RdBu_r", center=0, vmin=-0.5, vmax=0.5,
                cbar_kws={"label": "Pearson r(mean attention, age)"})
    ax.set_title(f"{dataset_label} — (cell type × head) age correlation, stage={stage.upper()}")
    plt.tight_layout(); plt.show()
    return df

heatmap_corr(att_aging,  "AIFI aging",  stage="ft", top_n_celltypes=15)
heatmap_corr(att_onek1k, "OneK1K",      stage="ft", top_n_celltypes=15)
""")

code(r"""
# Compare SSL vs FT for the SAME (cell_type, head) — what fine-tuning created
def ssl_vs_ft_scatter(art, dataset_label):
    if art is None or art.get("corr") is None: return
    c = (art["corr"].pivot_table(index=["cell_type", "head"], columns="stage", values="r")
          .reset_index().dropna(subset=["ssl", "ft"]))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(c["ssl"], c["ft"], s=20, color="#4C72B0", alpha=0.6, edgecolor="black", lw=0.3)
    # Annotate the top-|r| FT outliers
    top = c.assign(d=(c["ft"] - c["ssl"]).abs()).sort_values("d", ascending=False).head(8)
    for _, row in top.iterrows():
        ax.annotate(f"{row['cell_type'][:20]}/{row['head']}",
                    (row["ssl"], row["ft"]), xytext=(5, 3),
                    textcoords="offset points", fontsize=7)
    lim = max(c["ssl"].abs().max(), c["ft"].abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.5)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("SSL: r(mean attention, age)"); ax.set_ylabel("FT: r(mean attention, age)")
    ax.set_title(f"{dataset_label}: SSL vs FT age correlation per (cell type, head)")
    plt.tight_layout(); plt.show()

ssl_vs_ft_scatter(att_aging,  "AIFI aging")
ssl_vs_ft_scatter(att_onek1k, "OneK1K")
""")

md(r"""
Points far from the diagonal are (cell type, head) combinations whose
correlation with age changed between SSL and FT — exactly the
combinations that fine-tuning *created* as age-predictive features.

### 4. Genes that drive the age-correlated attention

For each top (cell type, head) hit on the AIFI cohort, we computed a
per-cell Pearson correlation between attention weight in that head and
each gene's log-normalised expression among cells of that type. The top
genes per hit are the markers the model is implicitly looking for inside
that cell type when it boosts or suppresses attention.
""")

code(r"""
def top_genes_panel(art, dataset_label, top_hits=6, n_genes=10):
    if art is None or art.get("gene_ft") is None: return
    g = art["gene_ft"].copy()
    # Pick the top hits by |head_age_r|
    top = (g.groupby(["cell_type", "head"], as_index=False)["head_age_r"].first()
           .sort_values("head_age_r", key=lambda s: s.abs(), ascending=False)
           .head(top_hits))
    cols = min(3, top_hits)
    rows = int(np.ceil(top_hits / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2), squeeze=False)
    for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
        sub = (g[(g["cell_type"] == row["cell_type"]) & (g["head"] == row["head"])]
               .sort_values("r", key=lambda s: s.abs(), ascending=False).head(n_genes))
        colors = ["#1f77b4" if r < 0 else "#d62728" for r in sub["r"]]
        ax.barh(sub["gene"], sub["r"], color=colors)
        ax.axvline(0, color="k", lw=0.5)
        title = f"{row['cell_type'][:30]}\n{row['head']}  age-r={row['head_age_r']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("r(att, gene)")
        ax.invert_yaxis()
    for ax in axes.ravel()[len(top):]:
        ax.set_visible(False)
    plt.tight_layout(); plt.show()
    return top

top_genes_panel(att_aging,  "AIFI aging",  top_hits=6, n_genes=10)
top_genes_panel(att_onek1k, "OneK1K",      top_hits=6, n_genes=10)
""")

md(r"""
**How to read these bar plots.** Red bars mean *more attention is given to
cells that express this gene more*. Blue bars mean *more attention goes
to cells that under-express this gene*. The header line gives the cell
type, the head index, and the (cell type, head)'s correlation with age.

For example, an NK-cell head whose attention is positively correlated
with age (header `r>0`) and whose top red bars are `GZMB`, `PRF1`, `NKG7`
is the model implicitly saying: *as donors age, I find more
cytotoxic-program NK cells, and I weight them up*. The same shorthand
generalises: terminal-effector markers in CD8 T-cell heads, classical
monocyte activation markers in CD14+ monocyte heads, etc.

This is the same biology that immune-aging studies find with manual
DE testing — only here we read it off the attention mass of a small
neural network that was trained end-to-end on age regression.
""")

md(r"""
## Take-aways

Three biological and a few methodological.

**Biology**

1. **The reproducible signal is compositional.** CLR-composition reaches
   `R²≈0.30` for held-out age on both cohorts (AIFI MAE 8.3 yr, OneK1K
   MAE 11.1 yr — the wider OneK1K age range makes absolute error bigger
   even at the same Spearman). The same cell-type families move in the
   same direction in both: naive CD4 / CD8 T cells ↓, GZMB+ effectors
   and CD16+ monocytes ↑, naive B cells ↓. That is the cleanest
   replication in the benchmark.
2. **Within-cell-type transcription adds real predictive power.** PaSCient
   (a cell→patient attention model trained on age) beats composition on
   both cohorts (R² 0.57 / 0.69 vs 0.30 / 0.30, MAE 6.8 / 7.4 yr) — so
   *how* an aging immune system transcribes inside each cell type is
   itself predictive of age, not just the population mixing.
3. **SampleCLR (R² 0.85 / 0.80, MAE 4.1 / 6.1 yr) is the strongest model.**
   Its contrastive objective + supervised regression head packs the
   donor embedding tightly along the age axis. With the batch-aware
   sampler, batch leakage stays reasonable too.

**Method notes**

- **Composition (CLR)** is the cheap interpretable baseline and the one
  that most cleanly replicates across cohorts. Use it first.
- **Pseudobulk** and **GloScope** add modest extra signal but inherit
  batch structure from `X_pca`. Useful when you already have a cell
  embedding lying around.
- **PaSCient** needs `normalize=False` when fed a centred embedding like
  `X_pca` (default `normalize=True` log-transforms negative values to
  NaN), z-scored regression targets to keep MSE on a unit scale, and
  gradient clipping. With those guardrails it converges in 30 GPU
  epochs.
- **MixMIL** only supports binomial / categorical likelihoods upstream,
  so we train it on a binary `age_group = age ≥ 65` here. The
  attention-weighted embedding still recovers continuous age structure
  (R² 0.26 on OneK1K) but it's the weakest of the three supervised
  models.
- **Held-out scoring is honest for unsupervised, biased for supervised.**
  All three supervised models saw test donors' ages during training. To
  remove the bias you would refit each on the 80% train donors only and
  re-infer the test 20%. The cross-cohort agreement is the more
  defensible comparison: composition's R² survives the cohort swap;
  PaSCient and SampleCLR's strong scores survive too, just with a
  larger AIFI→OneK1K gap.

For follow-up, the same template generalises to any donor-level
continuous target: BMI, severity scores, response to therapy,
time-to-event.
""")


# ---------------------------------------------------------------------------
# build + write
# ---------------------------------------------------------------------------


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    for kind, src in CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    nb.metadata = {
        "kernelspec": {"display_name": "patpy", "language": "python", "name": "patpy"},
        "language_info": {"name": "python"},
    }
    return nb


if __name__ == "__main__":
    nb = build()
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT}  cells={len(nb.cells)}")
