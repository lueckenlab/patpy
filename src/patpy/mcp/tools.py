from __future__ import annotations

import io
import json
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

import patpy
from patpy import datasets as patpy_datasets
from patpy import pl, pp, tl


@dataclass(frozen=True)
class RepresentationMethodSpec:
    """Registry entry for one patpy sample-representation method."""

    class_name: str
    supports_layer: bool = True
    requires_count_data: bool = False


@dataclass(frozen=True)
class SupervisedModelSpec:
    """Registry entry for one patpy supervised donor-level model."""

    class_name: str


REPRESENTATION_METHOD_SPECS: dict[str, RepresentationMethodSpec] = {
    "pseudobulk": RepresentationMethodSpec("Pseudobulk"),
    "grouped_pseudobulk": RepresentationMethodSpec("GroupedPseudobulk"),
    "cell_group_composition": RepresentationMethodSpec("CellGroupComposition"),
    "random_vector": RepresentationMethodSpec("RandomVector", supports_layer=False),
    "mrvi": RepresentationMethodSpec("MrVI", requires_count_data=True),
    "scpoli": RepresentationMethodSpec("SCPoli", requires_count_data=True),
    "pilot": RepresentationMethodSpec("PILOT"),
    "wasserstein_tsne": RepresentationMethodSpec("WassersteinTSNE"),
    "gloscope": RepresentationMethodSpec("GloScope"),
    "gloscope_py": RepresentationMethodSpec("GloScope_py"),
    "diffusion_earth_mover_distance": RepresentationMethodSpec("DiffusionEarthMoverDistance"),
    "mofa": RepresentationMethodSpec("MOFA"),
}

SUPERVISED_MODEL_SPECS: dict[str, SupervisedModelSpec] = {
    "mixmil": SupervisedModelSpec("MixMIL"),
    "pascient": SupervisedModelSpec("PaSCient"),
    "pulsar": SupervisedModelSpec("PULSAR"),
}

PLOT_TYPES = {"correlation_volcano", "embedding_covariate_heatmap"}


def _json_default(value: object):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_adata(path: str | Path) -> ad.AnnData:
    return ad.read_h5ad(Path(path).expanduser().resolve())


def _select_matrix(adata: ad.AnnData, layer: str | None):
    if layer in (None, "X"):
        return adata.X
    if layer in adata.obsm:
        return adata.obsm[layer]
    if layer in adata.layers:
        return adata.layers[layer]
    raise ValueError(f"layer='{layer}' not found in adata.obsm or adata.layers.")


def _capture(callable_):
    stdout = io.StringIO()
    with warnings.catch_warnings(record=True) as caught, redirect_stdout(stdout):
        warnings.simplefilter("always")
        result = callable_()

    messages = [str(item.message) for item in caught]
    log_lines = [line.strip() for line in stdout.getvalue().splitlines() if line.strip()]
    messages.extend(log_lines)
    return result, messages


def _base_provenance(tool_name: str, **payload: Any) -> dict[str, Any]:
    return {
        "tool": tool_name,
        "patpy_version": patpy.__version__,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **payload,
    }


def _finalize_result(
    tool_name: str,
    summary: dict[str, Any],
    artifacts: dict[str, str] | None = None,
    warnings_list: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    result = {
        "summary": summary,
        "artifacts": artifacts or {},
        "warnings": warnings_list or [],
        "provenance": provenance or _base_provenance(tool_name),
    }

    if output_dir is not None:
        manifest_path = output_dir / f"{tool_name}_manifest.json"
        _write_json(manifest_path, result)
        result["artifacts"]["manifest"] = str(manifest_path)

    return result


def _series_to_csv(series: pd.Series, path: Path, index_label: str) -> None:
    series.to_frame(name=series.name or "value").to_csv(path, index_label=index_label)


def _frame_to_csv(frame: pd.DataFrame, path: Path, index_label: str) -> None:
    frame.to_csv(path, index_label=index_label)


def _load_representation_inputs(
    representation_manifest_path: str | None,
    distance_matrix_path: str | None,
    samples_path: str | None,
) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    manifest = None
    if representation_manifest_path is not None:
        manifest = json.loads(Path(representation_manifest_path).expanduser().resolve().read_text(encoding="utf-8"))
        distance_matrix_path = manifest["artifacts"]["distances"]
        samples_path = manifest["artifacts"]["samples"]

    if distance_matrix_path is None or samples_path is None:
        raise ValueError("Either representation_manifest_path or both distance_matrix_path and samples_path are required.")

    distances = np.load(Path(distance_matrix_path).expanduser().resolve())
    samples_frame = pd.read_csv(Path(samples_path).expanduser().resolve())
    sample_column = "sample_id" if "sample_id" in samples_frame.columns else samples_frame.columns[0]
    samples = samples_frame[sample_column].astype(str).tolist()
    return distances, samples, manifest


def _metadata_frame(
    samples: list[str],
    *,
    adata_path: str | None,
    sample_key: str | None,
    metadata_csv_path: str | None,
    metadata_sample_column: str,
    required_columns: list[str],
) -> pd.DataFrame:
    if adata_path is not None:
        if sample_key is None:
            raise ValueError("sample_key is required when loading metadata from adata_path.")
        adata = _load_adata(adata_path)
        return pp.extract_metadata(adata, sample_key=sample_key, columns=required_columns, samples=samples)

    if metadata_csv_path is not None:
        metadata = pd.read_csv(Path(metadata_csv_path).expanduser().resolve())
        sample_column = metadata_sample_column if metadata_sample_column in metadata.columns else metadata.columns[0]
        metadata = metadata.set_index(sample_column)
        return metadata.loc[samples, required_columns]

    raise ValueError("Provide either adata_path/sample_key or metadata_csv_path to resolve target metadata.")


def _maybe_check_count_data(adata: ad.AnnData, layer: str | None, *, method_name: str) -> None:
    if not pp.is_count_data(_select_matrix(adata, layer)):
        raise ValueError(
            f"`{method_name}` requires count data with integer values in layer '{layer or 'X'}'."
        )


def _prepare_simulation_input(adata: ad.AnnData, layer: str | None) -> ad.AnnData:
    """Convert simulation inputs to the sparse layout expected by `patpy.datasets`."""

    prepared = adata.copy()
    effective_layer = "X" if layer is None else layer

    if effective_layer == "X":
        if not issparse(prepared.X):
            prepared.X = csr_matrix(prepared.X)
            warnings.warn(
                "Converted adata.X to CSR format for patpy.datasets.simulate_data compatibility.",
                stacklevel=2,
            )
        return prepared

    if effective_layer in prepared.layers and not issparse(prepared.layers[effective_layer]):
        prepared.layers[effective_layer] = csr_matrix(prepared.layers[effective_layer])
        warnings.warn(
            f"Converted adata.layers['{effective_layer}'] to CSR format for patpy.datasets.simulate_data compatibility.",
            stacklevel=2,
        )

    return prepared


def dataset_summary(
    adata_path: str,
    sample_key: str | None = None,
    cell_group_key: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Summarize a local `.h5ad` file for patpy workflows."""

    adata = _load_adata(adata_path)
    artifacts: dict[str, str] = {}
    output = _ensure_output_dir(output_dir) if output_dir is not None else None

    summary = {
        "adata_path": str(Path(adata_path).expanduser().resolve()),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obs_columns": list(map(str, adata.obs.columns)),
        "var_columns": list(map(str, adata.var.columns)),
        "layers": list(map(str, adata.layers.keys())),
        "obsm_keys": list(map(str, adata.obsm.keys())),
        "obsp_keys": list(map(str, adata.obsp.keys())),
        "uns_keys": list(map(str, adata.uns.keys())),
    }

    if sample_key is not None:
        if sample_key not in adata.obs.columns:
            raise ValueError(f"sample_key='{sample_key}' not found in adata.obs.")
        sample_counts = adata.obs[sample_key].value_counts().sort_index()
        summary["sample_key"] = sample_key
        summary["n_samples"] = int(sample_counts.shape[0])
        summary["sample_sizes"] = {str(idx): int(value) for idx, value in sample_counts.items()}
        if output is not None:
            sample_counts_path = output / "sample_counts.csv"
            _series_to_csv(sample_counts.rename("n_cells"), sample_counts_path, index_label=sample_key)
            artifacts["sample_counts"] = str(sample_counts_path)

    if cell_group_key is not None:
        if cell_group_key not in adata.obs.columns:
            raise ValueError(f"cell_group_key='{cell_group_key}' not found in adata.obs.")
        cell_group_counts = adata.obs[cell_group_key].value_counts().sort_index()
        summary["cell_group_key"] = cell_group_key
        summary["n_cell_groups"] = int(cell_group_counts.shape[0])
        summary["cell_group_sizes"] = {str(idx): int(value) for idx, value in cell_group_counts.items()}
        if output is not None:
            cell_group_counts_path = output / "cell_group_counts.csv"
            _series_to_csv(cell_group_counts.rename("n_cells"), cell_group_counts_path, index_label=cell_group_key)
            artifacts["cell_group_counts"] = str(cell_group_counts_path)

    return _finalize_result(
        "dataset_summary",
        summary=summary,
        artifacts=artifacts,
        provenance=_base_provenance("dataset_summary", adata_path=str(Path(adata_path).expanduser().resolve())),
        output_dir=output,
    )


def preprocess_dataset(
    adata_path: str,
    output_dir: str,
    sample_key: str,
    cell_group_key: str | None = None,
    sample_size_threshold: int | None = None,
    cluster_size_threshold: int | None = None,
    metadata_columns: list[str] | None = None,
    composition_keys: list[str] | None = None,
    cell_qc_vars: list[str] | None = None,
) -> dict[str, Any]:
    """Filter a local `.h5ad` and materialize patpy-ready summary tables."""

    adata = _load_adata(adata_path)
    output = _ensure_output_dir(output_dir)

    if sample_key not in adata.obs.columns:
        raise ValueError(f"sample_key='{sample_key}' not found in adata.obs.")
    if cluster_size_threshold is not None and cell_group_key is None:
        raise ValueError("cell_group_key is required when cluster_size_threshold is set.")

    initial_samples = adata.obs[sample_key].astype(str).unique().tolist()

    def _run():
        filtered = adata
        if sample_size_threshold is not None:
            filtered = pp.filter_small_samples(filtered, sample_key=sample_key, sample_size_threshold=sample_size_threshold)
        if cluster_size_threshold is not None:
            filtered = pp.filter_small_cell_groups(
                filtered,
                sample_key=sample_key,
                cell_group_key=cell_group_key,
                cluster_size_threshold=cluster_size_threshold,
            )
        return filtered

    filtered_adata, warning_messages = _capture(_run)

    filtered_path = output / "filtered_adata.h5ad"
    filtered_adata.write_h5ad(filtered_path)

    artifacts: dict[str, str] = {"filtered_adata": str(filtered_path)}

    sample_counts = pp.calculate_n_cells_per_sample(filtered_adata, sample_key=sample_key)
    sample_counts_path = output / "sample_counts.csv"
    _frame_to_csv(sample_counts, sample_counts_path, index_label=sample_key)
    artifacts["sample_counts"] = str(sample_counts_path)

    if metadata_columns:
        metadata = pp.extract_metadata(filtered_adata, sample_key=sample_key, columns=metadata_columns)
        metadata_path = output / "metadata.csv"
        _frame_to_csv(metadata, metadata_path, index_label=sample_key)
        artifacts["metadata"] = str(metadata_path)

    if composition_keys:
        composition = pp.calculate_compositional_metrics(
            filtered_adata,
            sample_key=sample_key,
            composition_keys=composition_keys,
        )
        composition_path = output / "compositional_metrics.csv"
        _frame_to_csv(composition, composition_path, index_label=sample_key)
        artifacts["compositional_metrics"] = str(composition_path)

    if cell_qc_vars:
        qc_metrics = pp.calculate_cell_qc_metrics(filtered_adata, sample_key=sample_key, cell_qc_vars=cell_qc_vars)
        qc_metrics_path = output / "qc_metrics.csv"
        _frame_to_csv(qc_metrics, qc_metrics_path, index_label=sample_key)
        artifacts["qc_metrics"] = str(qc_metrics_path)

    final_samples = filtered_adata.obs[sample_key].astype(str).unique().tolist()
    removed_samples = sorted(set(initial_samples) - set(final_samples))

    summary = {
        "adata_path": str(Path(adata_path).expanduser().resolve()),
        "sample_key": sample_key,
        "cell_group_key": cell_group_key,
        "n_obs_before": int(adata.n_obs),
        "n_obs_after": int(filtered_adata.n_obs),
        "n_samples_before": int(len(initial_samples)),
        "n_samples_after": int(len(final_samples)),
        "removed_samples": removed_samples,
        "sample_size_threshold": sample_size_threshold,
        "cluster_size_threshold": cluster_size_threshold,
    }

    return _finalize_result(
        "preprocess_dataset",
        summary=summary,
        artifacts=artifacts,
        warnings_list=warning_messages,
        provenance=_base_provenance(
            "preprocess_dataset",
            adata_path=str(Path(adata_path).expanduser().resolve()),
            output_dir=str(output),
        ),
        output_dir=output,
    )


def build_representation(
    adata_path: str,
    output_dir: str,
    sample_key: str,
    method: str = "pseudobulk",
    cell_group_key: str | None = None,
    layer: str | None = "X_pca",
    seed: int = 67,
    method_init: dict[str, Any] | None = None,
    distance_parameters: dict[str, Any] | None = None,
    fill_nan_distances: bool = False,
    metadata_columns: list[str] | None = None,
    write_sample_adata: bool = True,
) -> dict[str, Any]:
    """Build a patpy donor-level distance matrix from a local `.h5ad`."""

    if method not in REPRESENTATION_METHOD_SPECS:
        raise ValueError(f"Unsupported method '{method}'. Supported values: {sorted(REPRESENTATION_METHOD_SPECS)}")

    adata = _load_adata(adata_path)
    output = _ensure_output_dir(output_dir)
    spec = REPRESENTATION_METHOD_SPECS[method]
    init_kwargs = dict(method_init or {})
    run_kwargs = dict(distance_parameters or {})

    if spec.requires_count_data:
        _maybe_check_count_data(adata, layer, method_name=method)

    model_class = getattr(tl, spec.class_name)
    constructor_kwargs: dict[str, Any] = {
        "sample_key": sample_key,
        "cell_group_key": cell_group_key,
        "seed": seed,
        **init_kwargs,
    }
    if spec.supports_layer:
        constructor_kwargs["layer"] = layer

    np.random.seed(seed)
    model = model_class(**constructor_kwargs)

    def _run():
        model.prepare_anndata(adata)
        return np.asarray(model.calculate_distance_matrix(**run_kwargs))

    distances, warning_messages = _capture(_run)
    nan_count = int(np.isnan(distances).sum())
    if fill_nan_distances and nan_count:
        distances = pp.fill_nan_distances(distances)

    distances_path = output / "distances.npy"
    np.save(distances_path, distances)

    samples_path = output / "samples.csv"
    pd.DataFrame({"sample_id": model.samples}).to_csv(samples_path, index=False)

    artifacts: dict[str, str] = {
        "distances": str(distances_path),
        "samples": str(samples_path),
    }

    sample_representation = getattr(model, "sample_representation", None)
    if (
        write_sample_adata
        and sample_representation is not None
        and np.ndim(sample_representation) == 2
        and sample_representation.shape[0] == len(model.samples)
    ):
        metadata = None
        if metadata_columns:
            metadata = pp.extract_metadata(adata, sample_key=sample_key, columns=metadata_columns, samples=model.samples)
        sample_adata = model.to_adata(metadata=metadata)
        sample_adata_path = output / "sample_adata.h5ad"
        sample_adata.write_h5ad(sample_adata_path)
        artifacts["sample_adata"] = str(sample_adata_path)
    elif write_sample_adata:
        warning_messages.append(
            "Sample-level AnnData was not written because the method does not expose a 2-D sample representation."
        )

    summary = {
        "adata_path": str(Path(adata_path).expanduser().resolve()),
        "method": method,
        "sample_key": sample_key,
        "cell_group_key": cell_group_key,
        "layer": layer,
        "seed": seed,
        "distance_shape": list(distances.shape),
        "n_samples": int(len(model.samples)),
        "n_nan_distances": nan_count,
        "filled_nan_distances": bool(fill_nan_distances and nan_count),
    }

    return _finalize_result(
        "build_representation",
        summary=summary,
        artifacts=artifacts,
        warnings_list=warning_messages,
        provenance=_base_provenance(
            "build_representation",
            adata_path=str(Path(adata_path).expanduser().resolve()),
            output_dir=str(output),
            method=method,
        ),
        output_dir=output,
    )


def evaluate_representation(
    output_dir: str,
    target_column: str,
    method: str = "knn",
    representation_manifest_path: str | None = None,
    distance_matrix_path: str | None = None,
    samples_path: str | None = None,
    adata_path: str | None = None,
    sample_key: str | None = None,
    metadata_csv_path: str | None = None,
    metadata_sample_column: str = "sample_id",
    task: str = "classification",
    n_neighbors: int = 3,
    control_level: str | None = None,
    normalization_type: str = "total",
    groups_column: str | None = None,
    max_feature_difference: int | float | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a donor-level distance matrix against sample metadata."""

    output = _ensure_output_dir(output_dir)
    distances, samples, source_manifest = _load_representation_inputs(
        representation_manifest_path=representation_manifest_path,
        distance_matrix_path=distance_matrix_path,
        samples_path=samples_path,
    )
    metadata_columns = [target_column]
    if groups_column is not None and groups_column not in metadata_columns:
        metadata_columns.append(groups_column)
    metadata = _metadata_frame(
        samples,
        adata_path=adata_path,
        sample_key=sample_key,
        metadata_csv_path=metadata_csv_path,
        metadata_sample_column=metadata_sample_column,
        required_columns=metadata_columns,
    )

    evaluation_parameters = dict(parameters or {})
    if method == "knn":
        evaluation_parameters.setdefault("task", task)
        evaluation_parameters.setdefault("n_neighbors", n_neighbors)
    elif method == "distances":
        if control_level is None:
            raise ValueError("control_level is required for method='distances'.")
        evaluation_parameters.setdefault("control_level", control_level)
        evaluation_parameters.setdefault("normalization_type", normalization_type)
    elif method == "proportions":
        if groups_column is None:
            raise ValueError("groups_column is required for method='proportions'.")
        evaluation_parameters["groups"] = metadata[groups_column]
    elif method == "persistence":
        if max_feature_difference is None:
            raise ValueError("max_feature_difference is required for method='persistence'.")
        evaluation_parameters.setdefault("max_feature_difference", max_feature_difference)
        evaluation_parameters.setdefault("n_neighbors", n_neighbors)

    def _run():
        return tl.evaluate_representation(
            np.array(distances, copy=True),
            metadata[target_column],
            method=method,
            **evaluation_parameters,
        )

    result, warning_messages = _capture(_run)
    report_path = output / "evaluation_report.json"
    _write_json(report_path, result)

    target_values_path = output / "target_values.csv"
    _series_to_csv(metadata[target_column], target_values_path, index_label="sample_id")

    artifacts = {
        "report": str(report_path),
        "target_values": str(target_values_path),
    }
    if source_manifest is not None:
        artifacts["representation_manifest_source"] = str(Path(representation_manifest_path).expanduser().resolve())

    summary = {
        "target_column": target_column,
        "method": method,
        "n_samples": int(len(samples)),
        "evaluation_result": result,
    }

    return _finalize_result(
        "evaluate_representation",
        summary=summary,
        artifacts=artifacts,
        warnings_list=warning_messages,
        provenance=_base_provenance(
            "evaluate_representation",
            output_dir=str(output),
            target_column=target_column,
        ),
        output_dir=output,
    )


def run_supervised_prediction(
    adata_path: str,
    output_dir: str,
    sample_key: str,
    label_keys: list[str] | str,
    tasks: list[str] | str,
    model: str = "mixmil",
    cell_group_key: str | None = None,
    layer: str = "X_pca",
    seed: int = 67,
    model_parameters: dict[str, Any] | None = None,
    write_sample_importance: bool = True,
    write_cell_importance: bool = False,
    evaluate_predictions: bool = True,
) -> dict[str, Any]:
    """Train a supervised patpy model and materialize donor-level predictions."""

    if model not in SUPERVISED_MODEL_SPECS:
        raise ValueError(f"Unsupported model '{model}'. Supported values: {sorted(SUPERVISED_MODEL_SPECS)}")

    labels = [label_keys] if isinstance(label_keys, str) else list(label_keys)
    task_list = [tasks] if isinstance(tasks, str) else list(tasks)
    if len(labels) != len(task_list):
        raise ValueError("label_keys and tasks must have the same length.")

    adata = _load_adata(adata_path)
    output = _ensure_output_dir(output_dir)
    spec = SUPERVISED_MODEL_SPECS[model]
    model_class = getattr(tl, spec.class_name)
    model_parameters = dict(model_parameters or {})
    np.random.seed(seed)

    learner = model_class(
        sample_key=sample_key,
        label_keys=labels,
        tasks=task_list,
        cell_group_key=cell_group_key,
        layer=layer,
        seed=seed,
        **model_parameters,
    )

    def _run():
        learner.prepare_anndata(adata)
        predictions: dict[str, pd.Series | pd.DataFrame] = {}
        for label in labels:
            predictions[label] = learner.predict(label)
        importance = learner.get_sample_importance() if write_sample_importance else None
        cell_importance = learner.get_cell_importance() if write_cell_importance else None
        return predictions, importance, cell_importance

    (predictions, sample_importance, cell_importance), warning_messages = _capture(_run)

    artifacts: dict[str, str] = {}
    prediction_files: dict[str, str] = {}
    evaluations: dict[str, dict[str, Any]] = {}

    truth = pp.extract_metadata(adata, sample_key=sample_key, columns=labels, samples=learner.samples)
    for label, task in zip(labels, task_list, strict=True):
        prediction = predictions[label]
        prediction_path = output / f"prediction_{label}.csv"
        if isinstance(prediction, pd.Series):
            _series_to_csv(prediction, prediction_path, index_label=sample_key)
            predicted_values = prediction.loc[truth.index]
        else:
            _frame_to_csv(prediction, prediction_path, index_label=sample_key)
            pred_column = f"{label}_pred" if f"{label}_pred" in prediction.columns else prediction.columns[-1]
            predicted_values = prediction[pred_column].loc[truth.index]
        prediction_files[label] = str(prediction_path)
        artifacts[f"prediction_{label}"] = str(prediction_path)

        if evaluate_predictions:
            evaluations[label] = tl.evaluate_prediction(truth[label], predicted_values, task)

    if sample_importance is not None:
        sample_importance_path = output / "sample_importance.csv"
        _frame_to_csv(sample_importance, sample_importance_path, index_label=sample_key)
        artifacts["sample_importance"] = str(sample_importance_path)

    if cell_importance is not None:
        cell_importance_path = output / "cell_importance.csv"
        _frame_to_csv(cell_importance, cell_importance_path, index_label="cell_id")
        artifacts["cell_importance"] = str(cell_importance_path)

    if evaluations:
        evaluation_path = output / "prediction_evaluation.json"
        _write_json(evaluation_path, evaluations)
        artifacts["evaluation"] = str(evaluation_path)

    summary = {
        "adata_path": str(Path(adata_path).expanduser().resolve()),
        "model": model,
        "label_keys": labels,
        "tasks": task_list,
        "prediction_files": prediction_files,
        "evaluations": evaluations,
    }

    return _finalize_result(
        "run_supervised_prediction",
        summary=summary,
        artifacts=artifacts,
        warnings_list=warning_messages,
        provenance=_base_provenance(
            "run_supervised_prediction",
            adata_path=str(Path(adata_path).expanduser().resolve()),
            output_dir=str(output),
            model=model,
        ),
        output_dir=output,
    )


def generate_plot(
    table_path: str,
    output_dir: str,
    plot_type: str,
    plot_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render one of the packaged patpy plotting helpers from a CSV table."""

    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unsupported plot_type '{plot_type}'. Supported values: {sorted(PLOT_TYPES)}")

    output = _ensure_output_dir(output_dir)
    table = pd.read_csv(Path(table_path).expanduser().resolve())
    parameters = dict(plot_parameters or {})

    def _run():
        if plot_type == "correlation_volcano":
            figure, _ = pl.correlation_volcano(table, **parameters)
            return figure
        return pl.embedding_covariate_heatmap(table, return_fig=True, **parameters)

    figure, warning_messages = _capture(_run)

    svg_path = output / f"{plot_type}.svg"
    png_path = output / f"{plot_type}.png"
    figure.savefig(svg_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=200, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(figure)

    artifacts = {
        "svg": str(svg_path),
        "png": str(png_path),
    }
    summary = {
        "plot_type": plot_type,
        "table_path": str(Path(table_path).expanduser().resolve()),
        "n_rows": int(table.shape[0]),
        "n_columns": int(table.shape[1]),
    }

    return _finalize_result(
        "generate_plot",
        summary=summary,
        artifacts=artifacts,
        warnings_list=warning_messages,
        provenance=_base_provenance(
            "generate_plot",
            table_path=str(Path(table_path).expanduser().resolve()),
            output_dir=str(output),
            plot_type=plot_type,
        ),
        output_dir=output,
    )


def simulate_dataset(
    adata_path: str,
    output_dir: str,
    cell_group_key: str,
    layer: str | None = None,
    abundance_perturbation: dict[str, float] | None = None,
    gene_perturbation: dict[str, dict[str, float]] | None = None,
    perturbation_strength: float = 1.0,
    expression_noise_scale: float = 0.05,
    dropout_rate: float = 0.7,
    seed: int = 67,
    process_output: bool = False,
) -> dict[str, Any]:
    """Simulate a perturbed single-cell dataset from a local `.h5ad`."""

    source_adata = _load_adata(adata_path)
    if "distances" not in source_adata.obsp:
        raise ValueError("simulate_dataset requires `adata.obsp['distances']` in the source AnnData.")

    output = _ensure_output_dir(output_dir)
    np.random.seed(seed)

    def _run():
        adata = _prepare_simulation_input(source_adata, layer)
        simulated = patpy_datasets.simulate_data(
            adata,
            cell_type_key=cell_group_key,
            layer=layer,
            abundance_perturbation=abundance_perturbation,
            gene_perturbation=gene_perturbation,
            perturbation_strength=perturbation_strength,
            expression_noise_scale=expression_noise_scale,
            dropout_rate=dropout_rate,
        )
        if process_output:
            simulated = patpy_datasets.process_adata(simulated)
        return simulated

    simulated_adata, warning_messages = _capture(_run)

    simulated_path = output / "simulated_adata.h5ad"
    simulated_adata.write_h5ad(simulated_path)

    summary = {
        "adata_path": str(Path(adata_path).expanduser().resolve()),
        "cell_group_key": cell_group_key,
        "layer": layer,
        "seed": seed,
        "n_obs": int(simulated_adata.n_obs),
        "n_vars": int(simulated_adata.n_vars),
        "processed_output": bool(process_output),
    }
    artifacts = {"simulated_adata": str(simulated_path)}

    return _finalize_result(
        "simulate_dataset",
        summary=summary,
        artifacts=artifacts,
        warnings_list=warning_messages,
        provenance=_base_provenance(
            "simulate_dataset",
            adata_path=str(Path(adata_path).expanduser().resolve()),
            output_dir=str(output),
        ),
        output_dir=output,
    )
