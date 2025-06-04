```{highlight} shell

```

# Installation

## Stable release

### PyPI

To install patpy, run this command in your terminal:

```console
pip install patpy
```

This is the preferred method to install patpy, as it will always install the most recent stable release.
If you don't have [pip] installed, this [Python installation guide] can guide you through the process.

### Development version
You can install the most recent version from GitHub:
```console
pip install git+https://github.com/lueckenlab/patpy.git@main
```

### Additional dependency groups

patpy comes with several optional dependency groups that can be installed based on your needs. Besides groups for development or docs, there are sample representation methods with tricky dependencies. To work with them, install dependencies with the examples below.

#### Development tools
```console
pip install "patpy[dev]"
```

#### Documentation
```console
pip install "patpy[doc]"
```

#### Testing dependencies
```console
pip install "patpy[test]"
```

#### MRVI sample representation method
```console
pip install "patpy[mrvi]"
```

#### PILOT sample representation method
```console
pip install "patpy[pilot]"
```

#### scPoli sample representation method
```console
pip install "patpy[scpoli]"
```

#### DiffusionEMD sample representation method

```console
pip install "patpy[diffusionemd]"
```

You can also install multiple dependency groups at once:
```console
pip install "patpy[dev,doc,test]"
```

[github repo]: https://github.com/lueckenlab/patpy
[pip]: https://pip.pypa.io
[python installation guide]: http://docs.python-guide.org/en/latest/starting/installation/
