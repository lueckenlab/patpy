"""Tool registry for patpy-mcp.

Each ``_<toolname>.py`` module defines exactly one tool decorated with
``@mcp.tool``; importing the module registers the tool on the
:data:`patpy_mcp.mcp.mcp` singleton. The :mod:`patpy_mcp.main` module
star-imports this package to perform that side-effect at CLI startup.
"""

from ._cellxgene_download_dataset import cellxgene_download_dataset
from ._cellxgene_get_collection import cellxgene_get_collection
from ._cellxgene_get_dataset import cellxgene_get_dataset
from ._cellxgene_list_collections import cellxgene_list_collections
from ._cellxgene_list_disease_terms import cellxgene_list_disease_terms
from ._cellxgene_list_tissue_terms import cellxgene_list_tissue_terms
from ._cellxgene_search_datasets import cellxgene_search_datasets
from ._describe_source import describe_source
from ._list_sources import list_sources

__all__ = [
    "cellxgene_download_dataset",
    "cellxgene_get_collection",
    "cellxgene_get_dataset",
    "cellxgene_list_collections",
    "cellxgene_list_disease_terms",
    "cellxgene_list_tissue_terms",
    "cellxgene_search_datasets",
    "describe_source",
    "list_sources",
]
