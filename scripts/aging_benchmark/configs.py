"""Per-dataset / per-method config for the age-prediction benchmark.

We benchmark six methods on two cohorts:

- ``aging`` — AIFI Immunobiology of Aging cohort. 234 donors, ages 40–89,
  17 well-balanced batches, AIFI_L2 (28 immune subsets). Loaded from the
  local preprocessed file (no ``patpy.datasets`` loader yet).
- ``onek1k`` — OneK1K, 981 donors, age column already integer. Loaded
  via ``patpy.datasets.onek1k``.

For each ``(dataset, method)`` cell we save:

    data/aging_benchmark/<dataset>/<method>/
        embedding.npy           sample x latent matrix
        distance.npy            sample x sample distance matrix
        samples.npy             order of samples (donor ids)
        meta.parquet            sample-level metadata (incl. ``age``)
        knn_scores.csv          ranking score for age + technical covariates
        runtime.json            wall time, peak RSS, status, etc.

That is the single artifact format the notebook consumes (see ``aggregate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path("/ictstr01/groups/luckylab/workspace/vladimir.shitov/patpy-aging-tutorial")
OUT_ROOT = REPO_ROOT / "data" / "aging_benchmark"

AGING_H5AD = Path(
    "/ictstr01/groups/luckylab/workspace/vladimir.shitov/aifi_data/imm_of_aging/imm-of-aging_pp.h5ad"
)


@dataclass
class DatasetConfig:
    name: str
    sample_key: str
    cell_type_key: str
    layer: str
    batch_key: str
    age_col: str               # raw obs column for age; cleaned into ``age`` by data_loader
    sample_cols: list[str]     # extra obs columns kept as donor metadata
    schema: dict               # {"relevant": {col: task}, "technical": {col: task}}
    loader_kind: str           # "patpy" or "local_h5ad"
    loader_arg: str            # function name on patpy.datasets, or path


CONFIGS: dict[str, DatasetConfig] = {
    "aging": DatasetConfig(
        name="aging",
        sample_key="subject.subjectGuid",
        cell_type_key="AIFI_L2",
        layer="X_pca",
        batch_key="batch_id",
        age_col="sample.subjectAgeAtDraw",
        sample_cols=[
            "subject.biologicalSex",
            "subject.cmv",
            "subject.race",
            "batch_id",
            "pool_id",
            "chip_id",
        ],
        schema={
            "relevant": {
                "age": "regression",
                "subject.biologicalSex": "classification",
                "subject.cmv": "classification",
            },
            "technical": {
                "batch_id": "classification",
                "pool_id": "classification",
                "chip_id": "classification",
            },
        },
        loader_kind="local_h5ad",
        loader_arg=str(AGING_H5AD),
    ),
    "onek1k": DatasetConfig(
        name="onek1k",
        sample_key="donor_id",
        cell_type_key="cell_type",
        layer="X_pca",
        batch_key="pool_number",
        age_col="age",
        sample_cols=["sex", "pool_number"],
        schema={
            "relevant": {
                "age": "regression",
                "sex": "classification",
            },
            "technical": {
                "pool_number": "classification",
            },
        },
        loader_kind="patpy",
        loader_arg="onek1k",
    ),
}


# Methods we benchmark. "supervised" indicates the method fine-tunes on ``age``
# and therefore should NOT be evaluated by predicting age back on itself in a
# leakage-free way without held-out donors — we apply a 5-fold donor split.
@dataclass
class MethodConfig:
    name: str
    kind: str                  # "unsupervised" or "supervised"
    cap_cells_per_donor: int | None  # None → use all cells
    needs_gpu: bool = False
    # batch_aware: relevant only for SampleCLR
    batch_aware: bool = False


METHODS: dict[str, MethodConfig] = {
    "pseudobulk": MethodConfig("pseudobulk", "unsupervised", cap_cells_per_donor=None),
    "composition": MethodConfig("composition", "unsupervised", cap_cells_per_donor=None),
    "gloscope": MethodConfig("gloscope", "unsupervised", cap_cells_per_donor=500),
    "pascient": MethodConfig("pascient", "supervised", cap_cells_per_donor=500, needs_gpu=False),
    "mixmil": MethodConfig("mixmil", "supervised", cap_cells_per_donor=500, needs_gpu=False),
    "sampleclr": MethodConfig("sampleclr", "supervised", cap_cells_per_donor=500, needs_gpu=False, batch_aware=True),
}
