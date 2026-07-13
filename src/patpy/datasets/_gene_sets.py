"""Download immune-relevant gene sets for scoring single-cell data.

Every collection is returned as a ``{gene_set_name: [gene_symbols]}`` dictionary,
ready to be passed to :func:`patpy.pp.score_gene_sets`, :func:`scanpy.tl.score_genes`,
decoupler or gseapy.

Collections
-----------
``btm``
    Blood Transcription Modules (Li et al., Nat Immunol 2014), fetched from GitHub. Designed
    for whole blood / PBMC, and easier to interpret than pathway databases.
``hallmark``
    All 50 MSigDB Hallmark gene sets.
``hallmark_immune``
    The immune / activation / metabolism / proliferation subset of Hallmark
    (see :data:`HALLMARK_IMMUNE`) — a good default for a PBMC cohort.
``reactome``, ``kegg``, ``immunesigdb``
    MSigDB C2 CP:Reactome, C2 CP:KEGG (legacy) and C7 ImmuneSigDB. ImmuneSigDB
    holds ~5000 gene sets; use it deliberately.

The gene sets are downloaded from public URLs (no login, no extra dependency) and
cached as JSON, because the sources are occasionally flaky.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from patpy._settings import settings

# 346 modules with gene symbols, from the reference BTM repository
BTM_URL = "https://raw.githubusercontent.com/shuzhao-li/BTM/master/BTM/datasets/BTM_for_GSEA_20131008.gmt"

MSIGDB_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{version}/{category}.v{version}.symbols.gmt"

# Collection name -> MSigDB category. "hallmark_immune" is the full Hallmark
# collection filtered down to HALLMARK_IMMUNE afterwards.
MSIGDB_CATEGORIES = {
    "hallmark": "h.all",
    "hallmark_immune": "h.all",
    "reactome": "c2.cp.reactome",
    "kegg": "c2.cp.kegg_legacy",
    "immunesigdb": "c7.immunesigdb",
}

# Immune / activation / metabolic / proliferation subset of Hallmark, tuned for
# PBMC cohorts spanning viral infection, autoimmunity, sepsis and cancer
HALLMARK_IMMUNE = frozenset(
    {
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_IL2_STAT5_SIGNALING",
        "HALLMARK_COMPLEMENT",
        "HALLMARK_COAGULATION",
        "HALLMARK_ALLOGRAFT_REJECTION",
        "HALLMARK_TGF_BETA_SIGNALING",
        "HALLMARK_APOPTOSIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
        "HALLMARK_GLYCOLYSIS",
        "HALLMARK_HYPOXIA",
        "HALLMARK_MTORC1_SIGNALING",
        "HALLMARK_G2M_CHECKPOINT",
        "HALLMARK_E2F_TARGETS",
    }
)

# Prefixes keeping provenance when collections are flattened into one dictionary
_COLLECTION_PREFIXES = {
    "btm": "BTM",
    "hallmark": "H",
    "hallmark_immune": "H",
    "reactome": "REACTOME",
    "kegg": "KEGG",
    "immunesigdb": "C7",
}


def _parse_gmt(text: str) -> dict[str, list[str]]:
    """Parse GMT text (name, description, genes...) into `{gene_set: [genes]}`."""
    gene_sets = {}

    for line in text.strip().splitlines():
        name, _description, *genes = line.rstrip("\n").split("\t")
        genes = [gene.strip() for gene in genes if gene.strip()]

        if genes:
            gene_sets[name] = genes

    return gene_sets


def _download_gmt(url: str, timeout: int = 60) -> dict[str, list[str]]:
    """Download a GMT file and parse it."""
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return _parse_gmt(response.read().decode("utf-8"))


def _fetch_collection(collection: str, msigdb_version: str, drop_btm_unannotated: bool) -> dict[str, list[str]]:
    """Download a single collection from its public source."""
    if collection == "btm":
        gene_sets = _download_gmt(BTM_URL)

        if drop_btm_unannotated:
            # 87 BTM modules are named "TBA" (to be annotated) and carry no interpretation
            gene_sets = {name: genes for name, genes in gene_sets.items() if "TBA" not in name}

        return gene_sets

    if collection not in MSIGDB_CATEGORIES:
        raise ValueError(
            f"Unknown collection: {collection!r}. Choose from {['btm', *MSIGDB_CATEGORIES]}."
        )

    url = MSIGDB_URL.format(version=msigdb_version, category=MSIGDB_CATEGORIES[collection])
    gene_sets = _download_gmt(url)

    if collection == "hallmark_immune":
        gene_sets = {name: genes for name, genes in gene_sets.items() if name in HALLMARK_IMMUNE}

    return gene_sets


def download_gene_sets(
    collections: Iterable[str] = ("hallmark_immune",),
    cache_dir: str | Path | None = None,
    msigdb_version: str = "2024.1.Hs",
    drop_btm_unannotated: bool = True,
    flatten: bool = False,
    force: bool = False,
) -> dict[str, dict[str, list[str]]] | dict[str, list[str]]:
    """Download gene-set collections, caching each one as JSON.

    Parameters
    ----------
    collections : Iterable[str] = ("hallmark_immune",)
        Any of `"btm"`, `"hallmark"`, `"hallmark_immune"`, `"reactome"`, `"kegg"`,
        `"immunesigdb"`.
    cache_dir : str | Path | None = None
        Directory the collections are written to as `{collection}.json` and read back
        from on later calls. Defaults to `patpy.settings.datasetdir / "gene_sets"`.
    msigdb_version : str = "2024.1.Hs"
        MSigDB release tag used to build the download URL.
    drop_btm_unannotated : bool = True
        Drop the uninterpretable `"TBA"` modules from BTM.
    flatten : bool = False
        If `True`, merge every collection into a single `{gene_set: [genes]}` dictionary,
        prefixing gene-set names with their collection (e.g. `"BTM__..."`) to keep
        provenance and avoid name collisions.
    force : bool = False
        Ignore the cache and re-download.

    Returns
    -------
    gene_sets : dict[str, dict[str, list[str]]] | dict[str, list[str]]
        `{collection: {gene_set: [genes]}}`, or a single `{gene_set: [genes]}`
        dictionary when `flatten=True`.

    Examples
    --------
    >>> gene_sets = download_gene_sets(["hallmark_immune"])  # doctest: +SKIP
    >>> sorted(gene_sets["hallmark_immune"])[:2]  # doctest: +SKIP
    ['HALLMARK_ALLOGRAFT_REJECTION', 'HALLMARK_APOPTOSIS']
    """
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(settings.datasetdir) / "gene_sets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    gene_sets = {}

    for collection in collections:
        cache_file = cache_dir / f"{collection}.json"

        if cache_file.exists() and not force:
            gene_sets[collection] = json.loads(cache_file.read_text())
        else:
            gene_sets[collection] = _fetch_collection(collection, msigdb_version, drop_btm_unannotated)
            cache_file.write_text(json.dumps(gene_sets[collection]))

    if flatten:
        return {
            f"{_COLLECTION_PREFIXES.get(collection, collection)}__{name}": genes
            for collection, collection_sets in gene_sets.items()
            for name, genes in collection_sets.items()
        }

    return gene_sets
