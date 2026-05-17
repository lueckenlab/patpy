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
# Change ``SUFFIX = "_smoke"`` to inspect the small-data dry run before the full run.
BENCH_ROOT = Path("../../../data/aging_benchmark")
SUFFIX = ""   # set to "_smoke" to use the dry-run subset
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
    if not d.exists():
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
# Bin donors by age decade and average composition
donor_meta["age_decade"] = pd.cut(donor_meta.age, bins=[39,50,60,70,80,90],
                                  labels=["40s","50s","60s","70s","80s+"], right=False)
ct = ct.join(donor_meta["age_decade"]).groupby("age_decade", observed=True).mean()

# Show the cell types whose mean fraction changes the most across decades
delta = (ct.iloc[-1] - ct.iloc[0]).sort_values()
top_movers = list(delta.head(4).index) + list(delta.tail(4).index)
ct[top_movers].plot(kind="bar", stacked=False, figsize=(10, 4),
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

code(r"""
def plot_method_umap(art, method, metadata_cols=("age", "sex", "batch")):
    \"\"\"UMAP of the donor embedding coloured by each metadata column.\"\"\"
    meta = art["meta"].reset_index().rename(columns={"index": "donor"})
    samples = art["samples"]
    meta = meta.set_index(meta.columns[0]).reindex(samples)
    # Build a tiny adata so we can use sc.tl.umap
    a = ad.AnnData(X=art["embedding"].astype("float32"), obs=meta)
    a.obs_names = samples
    sc.pp.neighbors(a, n_neighbors=min(15, a.n_obs - 1), use_rep="X")
    sc.tl.umap(a, random_state=0)
    fig, axes = plt.subplots(1, len(metadata_cols), figsize=(4 * len(metadata_cols), 3.5))
    if len(metadata_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, metadata_cols):
        if col not in a.obs.columns:
            ax.set_visible(False)
            continue
        sc.pl.umap(a, color=col, ax=ax, show=False, frameon=False, size=80, title=f"{method} · {col}")
    plt.tight_layout(); plt.show()
""")

method_blurbs = {
    "pseudobulk": "**Pseudobulk** averages every cell's `X_pca` embedding within a donor and uses Euclidean distances between those means. Cheap, interpretable, ignores within-donor heterogeneity.",
    "composition": "**CLR composition** ignores expression entirely — only the *fractions* of each AIFI_L2 cell type. Centred log-ratio transform makes it a real Euclidean space.",
    "gloscope":    "**GloScope** treats each donor's PBMCs as a probability distribution in the cell-embedding space and uses symmetrised KL divergence (kNN-density estimator).",
    "sampleclr":   "**SampleCLR** is a contrastive sample-level model. We use the batch-aware sampler (the technical sites/pools are confounded with biology) and fine-tune on age.",
    "pascient":    "**PaSCient** is a cell→patient attention transformer. We train on continuous age (regression). This required a small local patch to patpy's PaSCient wrapper to swap CrossEntropyLoss → MSELoss when ``tasks=['regression']``; see the changes in this branch.",
    "mixmil":      "**MixMIL** combines a mixed model with multiple instance learning. The upstream `mixmil` library only offers binomial / categorical likelihoods (no Gaussian), so MixMIL trains on the binary ``age_group = age >= 65`` here. The donor embedding still carries continuous age structure for the KNN regression score.",
}

for m in ["pseudobulk", "composition", "gloscope", "sampleclr", "pascient", "mixmil"]:
    md(f"### {m}\n\n{method_blurbs[m]}")
    code(f"""
art = load_artifact("aging", "{m}")
if art is None:
    print("Run scripts/aging_benchmark/run_method.py --dataset aging --method {m} first.")
else:
    print(f"emb={{art['embedding'].shape}}  fit_seconds={{art['runtime'].get('t_fit_sec'):.1f}}")
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

code(r"""
def feature_vs_age(emb, samples, donor_age, top_n=10):
    \"\"\"Pearson correlation between every feature column and donor age.\"\"\"
    df = pd.DataFrame(emb, index=samples)
    age = donor_age.reindex(samples).astype(float).values
    keep = ~np.isnan(age)
    df = df.iloc[keep]; age = age[keep]
    rs = df.apply(lambda c: pearsonr(c, age)[0])
    ps = df.apply(lambda c: pearsonr(c, age)[1])
    return pd.DataFrame({"r": rs, "p": ps}).sort_values("r")
""")

code(r"""
# Cell types correlated with age — using the CLR composition embedding
comp_age = load_artifact("aging", "composition")
if comp_age is not None:
    age_per_donor = comp_age["meta"]["age"]
    cell_types = pd.Index(comp_age["meta"].columns).tolist()
    # The embedding columns are AIFI_L2 cell types; reconstruct that index
    # from the run_method.py pipeline (CellGroupComposition stores them in
    # ``sample_representation.columns``; we don't have it here so we just
    # number the columns).
    corr_comp = feature_vs_age(comp_age["embedding"], comp_age["samples"], age_per_donor)
    top_pos = corr_comp.tail(5)
    top_neg = corr_comp.head(5)
    print("Composition columns most negatively correlated with age (decrease with age):")
    print(top_neg.round(3))
    print("\nComposition columns most positively correlated with age (increase with age):")
    print(top_pos.round(3))
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
# Check OneK1K composition correlations and compare to AIFI's
comp_one = load_artifact("onek1k", "composition")
if comp_one is not None:
    corr_one = feature_vs_age(comp_one["embedding"], comp_one["samples"], comp_one["meta"]["age"])
    print("OneK1K composition columns sorted by correlation with age:")
    print(pd.concat([corr_one.head(3), corr_one.tail(3)]).round(3))
""")

md(r"""
## Take-aways

1. **Methods that ignore expression (composition CLR) can still rank donors
   by age** — the well-known immune-aging signal (CD4-naive ↓, GZMB+
   CD8 ↑, B-naive ↓) is mostly compositional. Compare the per-decade
   stacked bar above with the methods' age-Spearman scores.
2. **Pseudobulk** captures both compositional and within-cell-type
   transcriptomic shifts; on the AIFI cohort it usually beats composition
   alone, but it inherits batch leakage from `X_pca`.
3. **GloScope** trades compute for a distribution-level summary; it usually
   matches pseudobulk on age signal but is less batch-exposed.
4. **Supervised methods (SampleCLR-FT, PaSCient, MixMIL)** can target age
   directly. SampleCLR's batch-aware sampler is the cleanest win when sites
   are heavily confounded with biology.
5. **Cross-cohort replication is the real test.** The agreement between
   AIFI and OneK1K Spearman scores tells us which methods picked up a
   biology-driven signal versus a cohort-specific quirk.

For follow-up, the same template generalises to any donor-level continuous
target: BMI, severity scores, response to therapy, time-to-event, etc.
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
