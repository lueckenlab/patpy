# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

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
