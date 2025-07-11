from typing import Literal

_NORMALIZATION_TYPES = Literal["total", "shift", "var"]
_PREDICTION_TASKS = Literal["classification", "regression", "ranking"]
_EVALUATION_METHODS = Literal["knn", "distances", "proportions", "silhouette", "persistence"]
