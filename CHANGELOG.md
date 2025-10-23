# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## 0.11.2

### Added

- GloScope tutorial

### Changed

- Update the rpy2 interface for R implementation of GloScope

## 0.11.1

### Fixed

- Fix bug in `tl/sample_representation/GloScope_py` with always accessing layer in obsm instead of a general slot

## 0.11.0

### Added
- Function `tl/evaluation/trajectory_correlation` to compute a corresponding SPARE metric
- Function `tl/evaluation/knn_prediction_score` to compute a corresponding SPARE metric
- Function `tl/evaluation/replicate_robustness` to compute a corresponding SPARE metric
- Utils function `tl/evaluation/_get_col_from_adata`
- Utils funciton `tl/evaluation/_identity_up_to_suffix`

## 0.10.0

### Added

- `GloScope_py` sample representation method (reimplementation of the original GloScope in Python for CPU and GPU)

### Changed

- `GloScope.calculate_distance_matrix` now returns a NumPy array instead of a pandas DataFrame

## 0.9.3

### Changed

-   Update rpy2 conversion in `Gloscope.prepare_anndata()`

## 0.9.2

### Changed

-   Update readme with an overview and pypi link

## 0.9.1

### Changed

-   Install PILOT and DiffusionEMD from PyPI, not GitHub
-   Fix actions and update documentation

## 0.9.0

### Changed

-   GitHub actions files to match an updated scverse cookiecutter template
-   Breaking! Rename wherever possible: `patient_representation` -> `patpy`
-   Breaking! Rename `tl.basic.py` to `tl.sample_representation`

## 0.8.0

### Added

-   `persistence_evaluation` method in `patient_representation.tl.evaluation`
-   Persistent homology file `src/patient_representation/tl/persistence.py`

## 0.7.2

### Changed

-   Fix typo: `patient_representations` -> `sample_representation` in correlation functions

## 0.7.1

### Changed

-   Fixed typo in `GloScope` causing empty distance matrix

## 0.7.0

### Added

-   `GloScope` sample representation method (interface to R package via `rpy2`)
-   conda environment for `gloscope`

### Changed

-   `GloScope` R script now accepts `n_workers` argument

## 0.6.1

### Changed

-   Use `layers` instead of `obsm` to store layer data in `_move_layer_to_X` method

## 0.6.0

### Changed

-   Use `cell_group_key` instead of `cell_type_key` in `MOFA` and `_get_pseudobulk`
-   Use `sample_representation` instead of `patient_representation` in `MOFA`

## 0.5.0

### Deleted

-   Remove mandatory filtering of cell types in and small samples in `prepare_anndata` method of `SampleRepresentationMethod` descendants

### Changed

-   Rerun example notebook with updated API
-   Add minor comments to the example notebook

## 0.4.0 – Synthetic data generation

### Added

-   Functions to generate synthetic data simulating disease severity in `src/datasets/synthetic.py`
-   Synthetic data generation example notebook: `docs/notebooks/synthetic_data_generation.ipynb`
-   `plot_embedding` method for sample representations now accepts custom axes

## 0.3.0

### Sample representation refactoring:

-   "cell type" is renamed to "cell group" everywhere to be more general
-   Some representation methods are renamed accordingly:
-   -   `CellTypesComposition` -> `CellGroupComposition`
-   -   `CellTypePseudobulk` -> `GroupedPseudobulk`
-   -   `TotalPseudobulk` -> `Pseudobulk`
-   `patient_representation` argument is renamed to `sample_representation`
-   "Patient representation" is now renamed to "Sample representation" eveywhere
-   The base class is now called `SampleRepresentationMethod` instead of `PatientRepresentationMethod`. This is important only for developers, users shouldn't use it anyway

### Deleted

-   Not used `SCellBow` class
-   Example notebook in the documentation

## 0.2.0

### Added

-   Warning about ongoing development in README
-   Function `correlate_composition` to the tools
-   Function `correlate_cell_type_expression` to the tools
-   Function `correlation_volcano` to the plotting
-   Patients trajectory example notebook

### Changed

-   Rename `patient_representation` to `patpy`
