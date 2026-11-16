# Using patpy with DonorData

A [DonorData](https://donordata.readthedocs.io) holds the cells of a dataset and one table row per sample. Methods of
`patpy.tl` accept it in place of an AnnData and store their results next to the samples. Install it with
`pip install 'patpy[donordata]'`.

## From an AnnData to a DonorData

`patpy.pp.as_donordata` moves the columns that hold one value per sample into the sample table. The arguments
follow `DatasetInfo`, so `info.sample_key` and `info.sample_metadata_columns` of a patpy dataset can be passed on.

```python
import patpy

ddata = patpy.pp.as_donordata(adata, sample_key="sample", sample_metadata_columns=["disease", "age"])
ddata
```

```
DonorData(n_D_obs=12, n_D_vars=0, n_C_obs=600, n_C_vars=30)
```

`ddata.C` holds the cells as an AnnData and `ddata.obs` the sample table:

```python
ddata.obs.head(3)
```

```
        disease  age
sample
s0      healthy   29
s1        covid   57
s10     healthy   48
```

## Running a method

The method reads the cells from `ddata.C` and the sample metadata from the sample table. `to_donordata` stores the
results with the samples:

```python
method = patpy.tl.CellGroupComposition(sample_key="sample", cell_group_key="cell_type")
method.prepare_anndata(ddata)
distances = method.calculate_distance_matrix()
method.embed(method="MDS")
method.to_donordata("composition")

list(ddata.obsm), list(ddata.obsp)
```

```
(['composition', 'X_mds_composition'], ['composition_distances'])
```

Selecting samples keeps their results aligned, and the object is saved with them:

```python
covid = ddata.filter_donors("disease == 'covid'")
covid.obsp["composition_distances"].shape, covid.n_cells
```

```
((6, 6), 300)
```

```python
ddata.write_h5_dd("cohort.dd.h5")
```

## Evaluating representations

The evaluation functions take the samples as an AnnData, which `ddata.levels["donor"]` provides, with the stored
distances in `obsp`:

```python
schema = {"relevant": {"disease": "classification"}, "technical": {"age": "regression"}}
patpy.tl.knn_prediction_score(ddata.levels["donor"], schema, representations=["composition"], n_neighbors=3)
```

## Donors and samples

When a DonorData has a sample level, `sample_key` chooses the units of a method: its `sample_id` gives one
representation per sample, its `donor_id` pools the samples of every donor. Any other `sample_key` raises an error.
To evaluate per sample with the donor covariates as columns, pass `ddata.flatten().levels["donor"]`, which repeats the
donor table for every sample of a donor.
