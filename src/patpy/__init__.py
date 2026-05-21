from importlib.metadata import version

from . import datasets, pl, pp, tl
from ._settings import settings

__all__ = ["datasets", "pl", "pp", "tl", "settings"]

__version__ = version("patpy")
