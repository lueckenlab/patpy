import scanpy as sc

from patient_representation.pp import subsample
from patient_representation.tl import SCITD

ADATA_PATH = "/lustre/groups/ml01/workspace/moghareh.dehkordi/thesis/data/combat_processed.h5ad"
adata = sc.read_h5ad(ADATA_PATH)
sample_id_col = "scRNASeq_sample_ID"
cell_type_key = "Annotation_major_subset"
samples_metadata_cols = ["Source", "Outcome", "Death28", "Institute", "Pool_ID"]

adata = subsample(adata, obs_category_col=sample_id_col, min_samples_per_category=500, n_obs=500).copy()

scitd_method = SCITD(
    sample_key=sample_id_col,
    cell_group_key=cell_type_key,
    seed=67,
    n_factors=5,
    n_gene_sets=10,
    tucker_type="regular",
    rotation_type="hybrid",
    var_mask=None,
    norm_method="trim",
    scale_factor=1e4,
    var_scale_power=2,
    scale_var=True,
    threads=16,
)

scitd_method.prepare_anndata(adata)

distances = scitd_method.calculate_distance_matrix()

print("Distance matrix:\n", distances.shape)
print(
    "Outcome Ranking Evaluation:",
    scitd_method.evaluate_representation(target="Outcome", method="knn", n_neighbors=5, task="ranking"),
)
print(
    "Pool_ID Classification Evaluation:",
    scitd_method.evaluate_representation(target="Pool_ID", method="knn", n_neighbors=5, task="classification"),
)
print(
    "Outcome Classification Evaluation:",
    scitd_method.evaluate_representation(target="Outcome", method="knn", n_neighbors=5, task="classification"),
)
