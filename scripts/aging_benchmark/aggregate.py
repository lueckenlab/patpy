"""Collect per-method per-dataset artifacts into one tidy frame.

Loads:

    data/aging_benchmark/<dataset>/<method>/knn_scores.csv
    data/aging_benchmark/<dataset>/<method>/runtime.json

Writes:

    data/aging_benchmark/all_scores.csv
    data/aging_benchmark/all_runtime.csv

The notebook reads these two flat tables instead of crawling directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from configs import CONFIGS, METHODS, OUT_ROOT  # noqa: E402


def collect(suffix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[pd.DataFrame] = []
    rt_rows: list[dict] = []
    for ds in CONFIGS:
        ds_dir = OUT_ROOT / f"{ds}{suffix}"
        for m in METHODS:
            d = ds_dir / m
            if not d.exists():
                continue
            rt_path = d / "runtime.json"
            if rt_path.exists():
                rt = json.loads(rt_path.read_text())
                rt["dataset"], rt["method"] = ds, m
                rt_rows.append(rt)
            kp = d / "knn_scores.csv"
            if kp.exists():
                df = pd.read_csv(kp)
                df["dataset"], df["method"] = ds, m
                score_rows.append(df)
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    runtimes = pd.DataFrame(rt_rows) if rt_rows else pd.DataFrame()
    return scores, runtimes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help="e.g. '_smoke' to aggregate smoke outputs.")
    args = parser.parse_args()

    scores, runtimes = collect(args.suffix)
    out_dir = OUT_ROOT
    suffix = args.suffix
    scores_path = out_dir / f"all_scores{suffix}.csv"
    runtime_path = out_dir / f"all_runtime{suffix}.csv"
    scores.to_csv(scores_path, index=False)
    runtimes.to_csv(runtime_path, index=False)
    print(f"wrote {scores_path}  shape={scores.shape}")
    print(f"wrote {runtime_path}  shape={runtimes.shape}")
    if not scores.empty:
        print("\nAge held-out by dataset x method:")
        age = scores.query("covariate == 'age'").pivot_table(
            index="method", columns="dataset", values="r2", aggfunc="first"
        )
        print(age.round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())
