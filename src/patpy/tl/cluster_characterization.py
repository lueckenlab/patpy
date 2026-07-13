"""Describe what distinguishes clusters of samples.

Clustering a sample representation gives groups of patients, but not a reason for them. The
functions here answer "what is this cluster made of?" for the two kinds of annotation a
sample-level AnnData usually carries:

* numeric features -- cell-type proportions, pathway activities, age, cell counts -- with
  :func:`characterize_clusters`, a one-vs-rest test per (cluster, feature) pair;
* categorical covariates -- disease, study, sex, chemistry -- with
  :func:`cluster_covariate_enrichment`, a one-vs-rest Fisher test per (cluster, covariate, level)
  pair plus a global association strength per covariate.

Both return tidy DataFrames with effect sizes and FDR-corrected p-values, so a cluster can be read
off as "enriched for these cell types, these pathways, this disease".
"""

from collections.abc import Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

_NUMERIC_TESTS = {"mannwhitneyu", "ttest"}


def _correct_p_values(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction that tolerates NaNs (returned as NaN)."""
    p_adjusted = np.full(len(p_values), np.nan)
    is_tested = ~np.isnan(p_values)
    if is_tested.any():
        p_adjusted[is_tested] = stats.false_discovery_control(p_values[is_tested], method="bh")
    return p_adjusted


def _cohens_d(in_cluster: np.ndarray, out_cluster: np.ndarray) -> float:
    """Standardised mean difference between the two groups, pooling their variances."""
    n_in, n_out = len(in_cluster), len(out_cluster)
    if n_in < 2 or n_out < 2:
        return np.nan

    pooled_variance = ((n_in - 1) * in_cluster.var(ddof=1) + (n_out - 1) * out_cluster.var(ddof=1)) / (n_in + n_out - 2)
    if pooled_variance <= 0:
        return 0.0

    return float((in_cluster.mean() - out_cluster.mean()) / np.sqrt(pooled_variance))


def _collect_numeric_features(
    adata: ad.AnnData,
    obs_keys: Sequence[str] | None,
    obsm_keys: Sequence[str] | None,
) -> pd.DataFrame:
    """Gather the requested numeric features from `.obs` and `.obsm` into one samples x features frame.

    Columns are named ``"<obsm_key>:<column>"`` for features taken from ``.obsm`` so that features with
    the same name in different ``.obsm`` entries (e.g. the same pathway scored in several cell types)
    stay distinguishable.
    """
    frames = []

    if obs_keys is None:
        obs_keys = [key for key in adata.obs.columns if pd.api.types.is_numeric_dtype(adata.obs[key])]
    missing = set(obs_keys) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Columns not found in adata.obs: {sorted(missing)}")
    if obs_keys:
        obs_features = adata.obs[list(obs_keys)].apply(pd.to_numeric, errors="coerce")
        obs_features.columns = pd.MultiIndex.from_product([["obs"], obs_features.columns])
        frames.append(obs_features)

    for obsm_key in obsm_keys or []:
        if obsm_key not in adata.obsm:
            raise ValueError(f"'{obsm_key}' not found in adata.obsm.")

        values = adata.obsm[obsm_key]
        obsm_features = (
            values.copy()
            if isinstance(values, pd.DataFrame)
            else pd.DataFrame(
                np.asarray(values),
                index=adata.obs_names,
                columns=[f"{i}" for i in range(np.asarray(values).shape[1])],
            )
        )
        obsm_features = obsm_features.apply(pd.to_numeric, errors="coerce")
        obsm_features.columns = pd.MultiIndex.from_product([[obsm_key], obsm_features.columns])
        frames.append(obsm_features)

    if not frames:
        raise ValueError("No numeric features selected: pass `obs_keys` and/or `obsm_keys`.")

    features = pd.concat(frames, axis=1)
    features.index = adata.obs_names
    return features


def characterize_clusters(
    adata: ad.AnnData,
    cluster_key: str,
    *,
    obs_keys: Sequence[str] | None = None,
    obsm_keys: Sequence[str] | None = None,
    test: str = "mannwhitneyu",
    min_cluster_size: int = 3,
) -> pd.DataFrame:
    """Test every numeric feature for enrichment in every cluster, one cluster versus all others.

    For each (cluster, feature) pair the samples in the cluster are compared against all samples
    outside it. The result is a tidy table that says, for example, "cluster L3 has a higher
    interferon-response score in monocytes than the rest of the cohort (Cohen's d = 1.4, FDR = 1e-8)".

    Parameters
    ----------
    adata : AnnData
        Sample-level object: one observation per sample.
    cluster_key : str
        Column in ``adata.obs`` holding the cluster labels.
    obs_keys : sequence of str, optional
        Numeric columns of ``adata.obs`` to test. Defaults to every numeric column.
        Pass an empty sequence to test nothing from ``.obs``.
    obsm_keys : sequence of str, optional
        Keys of ``adata.obsm`` whose columns are tested as features, e.g. a
        ``samples x cell_types`` composition matrix or a ``samples x pathways`` score matrix.
        Non-DataFrame entries get positional column names.
    test : {"mannwhitneyu", "ttest"}, default ``"mannwhitneyu"``
        Two-sided test comparing in-cluster against out-of-cluster values. The rank-based default
        makes no distributional assumption, which suits proportions and scores.
    min_cluster_size : int, default ``3``
        Clusters with fewer samples than this are skipped.

    Returns
    -------
    pd.DataFrame
        One row per (cluster, feature) pair, sorted by FDR, with columns:
        ``cluster``, ``feature``, ``feature_set`` (``"obs"`` or the ``.obsm`` key), ``n_in``,
        ``n_out``, ``mean_in``, ``mean_out``, ``median_in``, ``median_out``, ``cohens_d``,
        ``auroc`` (probability that a random in-cluster sample exceeds a random out-of-cluster one,
        0.5 = no separation), ``statistic``, ``p_value`` and ``p_adjusted`` (Benjamini-Hochberg,
        across all rows).

    Raises
    ------
    ValueError
        If ``cluster_key`` is missing from ``adata.obs``, a requested key is absent, no numeric
        feature is selected, or ``test`` is not recognised.

    Examples
    --------
    >>> import patpy
    >>> summary = patpy.tl.characterize_clusters(
    ...     sample_adata,
    ...     cluster_key="leiden",
    ...     obs_keys=["age", "n_cells"],
    ...     obsm_keys=["cell_type_composition"],
    ... )
    >>> summary.query("cluster == 'L3' and p_adjusted < 0.05").head()
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"'{cluster_key}' not found in adata.obs.")
    if test not in _NUMERIC_TESTS:
        raise ValueError(f"test must be one of {sorted(_NUMERIC_TESTS)}, got '{test}'.")

    features = _collect_numeric_features(adata, obs_keys, obsm_keys)
    clusters = adata.obs[cluster_key].astype(str).values

    rows = []
    for cluster in pd.unique(clusters):
        is_in_cluster = clusters == cluster
        if is_in_cluster.sum() < min_cluster_size:
            continue

        for (feature_set, feature), values in features.items():
            values = values.values.astype(float)
            in_cluster = values[is_in_cluster & ~np.isnan(values)]
            out_cluster = values[~is_in_cluster & ~np.isnan(values)]

            row = {
                "cluster": cluster,
                "feature": feature,
                "feature_set": feature_set,
                "n_in": len(in_cluster),
                "n_out": len(out_cluster),
                "mean_in": in_cluster.mean() if len(in_cluster) else np.nan,
                "mean_out": out_cluster.mean() if len(out_cluster) else np.nan,
                "median_in": np.median(in_cluster) if len(in_cluster) else np.nan,
                "median_out": np.median(out_cluster) if len(out_cluster) else np.nan,
                "cohens_d": np.nan,
                "auroc": np.nan,
                "statistic": np.nan,
                "p_value": np.nan,
            }

            # A test needs both groups populated and some variation to work with
            if len(in_cluster) >= 2 and len(out_cluster) >= 2 and np.ptp(np.concatenate([in_cluster, out_cluster])) > 0:
                if test == "mannwhitneyu":
                    statistic, p_value = stats.mannwhitneyu(in_cluster, out_cluster, alternative="two-sided")
                    row["auroc"] = statistic / (len(in_cluster) * len(out_cluster))
                else:
                    statistic, p_value = stats.ttest_ind(in_cluster, out_cluster, equal_var=False)
                row["statistic"] = float(statistic)
                row["p_value"] = float(p_value)
                row["cohens_d"] = _cohens_d(in_cluster, out_cluster)

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary["p_adjusted"] = _correct_p_values(summary["p_value"].values)

    return summary.sort_values(["p_adjusted", "p_value"], na_position="last").reset_index(drop=True)


def _cramers_v(contingency: pd.DataFrame) -> tuple[float, float]:
    """Bias-corrected Cramer's V and the chi-square p-value for a contingency table."""
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan, np.nan

    chi2, p_value = stats.chi2_contingency(contingency.values)[:2]
    n_samples = contingency.values.sum()
    n_rows, n_columns = contingency.shape

    # Bergsma's bias correction: with many levels and few samples, the raw V is inflated
    phi2 = max(0.0, chi2 / n_samples - (n_rows - 1) * (n_columns - 1) / (n_samples - 1))
    corrected_rows = n_rows - (n_rows - 1) ** 2 / (n_samples - 1)
    corrected_columns = n_columns - (n_columns - 1) ** 2 / (n_samples - 1)
    denominator = min(corrected_rows - 1, corrected_columns - 1)
    if denominator <= 0:
        return np.nan, float(p_value)

    return float(np.sqrt(phi2 / denominator)), float(p_value)


def cluster_covariate_enrichment(
    adata: ad.AnnData,
    cluster_key: str,
    covariates: Sequence[str],
    *,
    min_cluster_size: int = 3,
    dropna: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test which levels of each categorical covariate are over-represented in each cluster.

    Answers both "is this covariate related to the clustering at all?" (a global association strength
    per covariate) and "which cluster is enriched for which level?" (a one-vs-rest Fisher exact test
    per cluster and level). Use it to tell a clustering that tracks disease from one that tracks the
    study a sample came from.

    Parameters
    ----------
    adata : AnnData
        Sample-level object: one observation per sample.
    cluster_key : str
        Column in ``adata.obs`` holding the cluster labels.
    covariates : sequence of str
        Categorical columns in ``adata.obs`` to test.
    min_cluster_size : int, default ``3``
        Clusters with fewer samples than this are skipped.
    dropna : bool, default ``True``
        Drop samples with a missing value when testing a covariate. When ``False``, missingness
        becomes its own level, which is worth looking at when a covariate is only recorded for
        some studies.

    Returns
    -------
    enrichment : pd.DataFrame
        One row per (covariate, cluster, level), sorted by FDR, with columns: ``covariate``,
        ``cluster``, ``level``, ``n_in_cluster_with_level``, ``n_in_cluster``, ``frac_in_cluster``,
        ``frac_outside``, ``log2_odds_ratio`` (positive = enriched, clipped at +-10 when a cell is
        empty), ``p_value`` and ``p_adjusted`` (Benjamini-Hochberg, across all rows).
    association : pd.DataFrame
        One row per covariate, with ``cramers_v`` (0 = independent, 1 = cluster determines the
        covariate), the chi-square ``p_value``, ``n_samples`` and ``n_levels``. Sorted by
        ``cramers_v``, so the covariates the clustering is really tracking come first.

    Raises
    ------
    ValueError
        If ``cluster_key`` or any covariate is missing from ``adata.obs``.

    Examples
    --------
    >>> import patpy
    >>> enrichment, association = patpy.tl.cluster_covariate_enrichment(
    ...     sample_adata, cluster_key="leiden", covariates=["disease", "studyID", "sex"]
    ... )
    >>> association  # is the clustering tracking disease, or the study?
    """
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"'{cluster_key}' not found in adata.obs.")
    missing = set(covariates) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Covariates not found in adata.obs: {sorted(missing)}")

    cluster_labels = adata.obs[cluster_key].astype(str)
    cluster_sizes = cluster_labels.value_counts()
    kept_clusters = set(cluster_sizes[cluster_sizes >= min_cluster_size].index)

    enrichment_rows = []
    association_rows = []

    for covariate in covariates:
        values = adata.obs[covariate].astype(object)
        values = values.where(values.notna(), other=np.nan if dropna else "missing")

        is_tested = values.notna() if dropna else pd.Series(True, index=values.index)
        is_tested &= cluster_labels.isin(kept_clusters)
        if is_tested.sum() == 0:
            continue

        tested_values = values[is_tested].astype(str)
        tested_clusters = cluster_labels[is_tested]

        contingency = pd.crosstab(tested_clusters, tested_values)
        cramers_v, chi2_p_value = _cramers_v(contingency)
        association_rows.append(
            {
                "covariate": covariate,
                "cramers_v": cramers_v,
                "p_value": chi2_p_value,
                "n_samples": int(is_tested.sum()),
                "n_levels": int(tested_values.nunique()),
            }
        )

        for cluster in contingency.index:
            for level in contingency.columns:
                in_with = int(contingency.loc[cluster, level])
                in_without = int(contingency.loc[cluster].sum() - in_with)
                out_with = int(contingency[level].sum() - in_with)
                out_without = int(contingency.values.sum() - in_with - in_without - out_with)

                odds_ratio, p_value = stats.fisher_exact([[in_with, in_without], [out_with, out_without]])
                with np.errstate(divide="ignore"):
                    log2_odds_ratio = float(np.clip(np.log2(odds_ratio), -10, 10)) if odds_ratio > 0 else -10.0

                enrichment_rows.append(
                    {
                        "covariate": covariate,
                        "cluster": cluster,
                        "level": level,
                        "n_in_cluster_with_level": in_with,
                        "n_in_cluster": in_with + in_without,
                        "frac_in_cluster": in_with / (in_with + in_without) if in_with + in_without else np.nan,
                        "frac_outside": out_with / (out_with + out_without) if out_with + out_without else np.nan,
                        "log2_odds_ratio": log2_odds_ratio,
                        "p_value": float(p_value),
                    }
                )

    enrichment = pd.DataFrame(enrichment_rows)
    if not enrichment.empty:
        enrichment["p_adjusted"] = _correct_p_values(enrichment["p_value"].values)
        enrichment = enrichment.sort_values(["p_adjusted", "p_value"], na_position="last").reset_index(drop=True)

    association = pd.DataFrame(association_rows)
    if not association.empty:
        association = association.sort_values("cramers_v", ascending=False, na_position="last").reset_index(drop=True)

    return enrichment, association
