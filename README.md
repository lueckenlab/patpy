# patient_representation

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/VladimirShitov/patient_representation/test.yaml?branch=main
[link-tests]: https://github.com/lueckenlab/patient_representation/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/patient_representation

Representing patients or samples from single-cell data

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install patient_representation:

<!--
1) Install the latest release of `patient_representation` from `PyPI <https://pypi.org/project/patient_representation/>`_:

```bash
pip install patient_representation
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/lueckenlab/patient_representation.git@main
```

To install specific dependencies for some sample representation tools, use the following command:

```bash
pip install git+https://github.com/lueckenlab/patient_representation.git@main[pilot]
```

All the available dependency groups: `diffusionemd`, `mrvi`, `pilot`, `scpoli`, `wassersteintsne`.

### Using R dependencies

To use some of the sample representation methods, which are implemented in R (for example, GloScope), you need to install additional dependencies. The easiest way to do it is via `conda`. To not break things, it is highly recommended to do it in a separate environment. You can find environments for R methods in the [environments](environments) directory. Here is an example of installing the gloscope environment:

```bash
conda env create -f environments/gloscope.yaml
conda activate gloscope
pip install git+https://github.com/lueckenlab/patient_representation.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Building docs

1. Install [sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html)

You may need add path to `sphinx-doc` to the `$PATH`

2. Install other `doc` section dependencies from the [pyproject.toml](https://github.com/lueckenlab/patient_representation/blob/main/pyproject.toml)

3. Build the documentation pages:

```bash
cd docs
make html
```

4. Open `docs/_build/html/index.html`

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/lueckenlab/patient_representation/issues
[changelog]: https://patient_representation.readthedocs.io/latest/changelog.html
[link-docs]: https://patient_representation.readthedocs.io
[link-api]: https://patient_representation.readthedocs.io/latest/api.html
