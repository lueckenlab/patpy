# patpy

patpy is a toolbox for single-cell data analysis on sample level.

![overview](./_static/patpy_pipeline.png)

It provides:
- 👨‍⚕️ Interface to sample representation methods (otherwise known as patient representation)
- 📈 Analysis functions to get the most of your data
- 📊 Metrics for sample representation evaluation

```{eval-rst}
.. card:: Installation :octicon:`plug;1em;`
    :link: installation
    :link-type: doc

    New to *patpy*? Check out the installation guide.
```

```{eval-rst}
.. card:: Tutorials :octicon:`play;1em;`
    :link: tutorials/index
    :link-type: doc

    Follow tutorials for a quickstart and examples of patpy applications.
```

```{eval-rst}
.. card:: API reference :octicon:`book;1em;`
    :link: api/index
    :link-type: doc

    The API reference contains a detailed description of the patpy API.
```

```{eval-rst}
.. card:: Discussion :octicon:`megaphone;1em;`
    :link: https://github.com/lueckenlab/patpy/issues

    Need help? Found a bug? Interested in contributing to the code? Open an issue on GitHub!
```

```{toctree}
:caption: General
:hidden:
:maxdepth: 2

installation
api/index
contributing
changelog
references
```

```{toctree}
:caption: Gallery
:hidden:
:maxdepth: 2

tutorials/index
```

```{toctree}
:caption: About
:hidden:
:maxdepth: 2

about/background
about/cite
GitHub <https://github.com/lueckenlab/patpy>
```

## Citation

For now, patpy can be cited as a repository:
```bibtex
@misc{shitov_patpy_2024,
  author = {Shitov, Vladimir},
  title = {patpy – sample-level analysis framework for single-cell data},
  year = {2024},
  url = {https://github.com/lueckenlab/patpy/},
  note = {Version 0.9.2}
}
```

If you use it for benchmarking sample representation methods, check out our paper at LMRL workshop at ICLR:
```bibtex
@inproceedings{
    shitov2025benchmarking,
    title={Benchmarking Sample Representations from Single-Cell Data: Metrics for Biologically Meaningful Embeddings},
    author={Vladimir Shitov and Mohammad Moghareh Dehkordi and Malte D Luecken},
    booktitle={Learning Meaningful Representations of Life (LMRL) Workshop at ICLR 2025},
    year={2025},
    url={https://openreview.net/forum?id=IoRv5afWtb}
}
```
