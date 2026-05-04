---
name: patpy.tl evaluation
description: Score a sample-level distance matrix or prediction against per-sample metadata — kNN, silhouette, distance significance test, persistence, chi-square.
---

# patpy.tl — evaluation

All evaluation entry points consume **either** a distance matrix + target, **or** a prediction + ground truth.

## Symbols

```python
patpy.tl.evaluate_representation(distances, target, method, ...) -> dict
patpy.tl.predict_knn(distances, y_true, n_neighbors=3, task="classification") -> y_pred
patpy.tl.evaluate_prediction(y_true, y_pred, task, ...) -> dict
patpy.tl.test_distances_significance(distances, conditions, control_level, normalization_type, ...)
patpy.tl.test_proportions(target, groups) -> dict
patpy.tl.persistence_evaluation(distances, conditions, max_feature_difference, ...) -> dict
patpy.tl.associate_embedding_with_covariates(...)
```

## When to use which

| Question | Function | `method=` |
|---|---|---|
| Does the representation cluster by `disease`? | `evaluate_representation` | `"knn"` (default) |
| Are case–control distances significantly inflated? | `evaluate_representation` | `"distances"` (Petukhov et al.) |
| Does cluster membership track `disease`? | `evaluate_representation` | `"proportions"` |
| Geometric quality of a labelled clustering | `evaluate_representation` | `"silhouette"` |
| Topological connectivity along a continuous covariate (e.g. severity) | `evaluate_representation` | `"persistence"` |
| Score predictions from a supervised model | `evaluate_prediction` | n/a |

## Score conventions

- **`evaluate_prediction(task="classification")`** returns calibrated F1 in `[0, 1]`: 0 = random, 1 = perfect. Negative pre-clip values warn and are clipped to 0.
- **`evaluate_prediction(task="regression"/"ranking")`** returns Spearman ρ, clipped to `[0, 1]`. Negative correlations are clipped (the rationale is that anti-correlation isn't meaningful for a representation-quality metric).
- **`test_distances_significance`** returns `(normalized_distances, real_statistic, p_value)`. `p_value` is the bootstrap p-value under the null that case–control distances follow the same distribution as control–control.

## Minimal examples

**kNN evaluation of a distance matrix**

```python
import patpy

D = method.calculate_distance_matrix()  # from a SampleRepresentationMethod
target = patpy.pp.extract_metadata(adata, "donor_id", ["disease"])["disease"]

result = patpy.tl.evaluate_representation(
    D, target, method="knn", n_neighbors=3, task="classification"
)
# result == {"score": 0.42, "metric": "f1_macro_calibrated", "n_unique": 2, "n_observations": 24, "method": "knn"}
```

**Distance significance test**

```python
result = patpy.tl.evaluate_representation(
    D, target, method="distances",
    control_level="healthy", normalization_type="total",
    n_bootstraps=1000, trimmed_fraction=0.2, compare_by_difference=True,
)
# result["score"] is the trimmed-mean test statistic; result["p_value"] is bootstrap p.
```

**Score a supervised model's predictions**

```python
result = patpy.tl.evaluate_prediction(y_true, y_pred, task="classification")
```

## Common pitfalls

- **Diagonal leakage in `predict_knn`.** The function fills the diagonal with `distances.max()` *in place* before fitting kNN — it mutates your input matrix. Pass a copy if you need to keep the original.
- **NaN in `target`.** `evaluate_representation` filters samples where `target` is missing. The reported `n_observations` will be smaller than the matrix dimension. This is intentional; check it if you expected all samples scored.
- **`method="distances"` requires `control_level`.** It's a kwarg, not positional. Forgetting it raises a confusing error from inside the bootstrap.
- **`method="proportions"` requires `groups=` in `parameters`.** Pass cluster labels: `evaluate_representation(D, target, method="proportions", groups=clusters)`.
- **Subset arguments are mutually exclusive** with full evaluation — `num_donors_subset` and `proportion_donors_subset` cannot both be set.
- **Calibration of F1.** A score of 0 means *as good as random for the class balance*, not "always wrong". Don't interpret 0 as a chance prediction error; it's the expected value of `(F1 − 1/n_classes) / (1 − 1/n_classes)` for a uniform random predictor.

## Related skills

- Build the `distances` input → [../sample_representation/SKILL.md](../sample_representation/SKILL.md).
- Build the `y_true` / `y_pred` input → [../supervised_methods/SKILL.md](../supervised_methods/SKILL.md).
- Plot the result → [../plotting/SKILL.md](../plotting/SKILL.md).
