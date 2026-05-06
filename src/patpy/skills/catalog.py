from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class SkillSpec:
    """Export metadata for one packaged patpy skill."""

    source_dir: str
    export_name: str
    display_name: str
    short_description: str
    default_prompt: str


SKILL_SPECS: tuple[SkillSpec, ...] = (
    SkillSpec(
        source_dir="",
        export_name="patpy",
        display_name="patpy",
        short_description="Route patpy sample-level analysis tasks",
        default_prompt="Use $patpy to pick the right patpy workflow for this sample-level single-cell analysis task.",
    ),
    SkillSpec(
        source_dir="datasets",
        export_name="patpy-datasets",
        display_name="patpy Datasets",
        short_description="Simulate perturbed single-cell datasets",
        default_prompt="Use $patpy-datasets to simulate perturbed single-cell data or inspect patpy hallmark helpers.",
    ),
    SkillSpec(
        source_dir="preprocessing",
        export_name="patpy-preprocessing",
        display_name="patpy Preprocessing",
        short_description="QC and filter AnnData before patpy runs",
        default_prompt="Use $patpy-preprocessing to QC, filter, and summarize an AnnData before running patpy workflows.",
    ),
    SkillSpec(
        source_dir="sample_representation",
        export_name="patpy-sample-representation",
        display_name="patpy Sample Representation",
        short_description="Build donor-level patpy distance matrices",
        default_prompt="Use $patpy-sample-representation to build a donor-level distance matrix from a cell-level AnnData.",
    ),
    SkillSpec(
        source_dir="supervised_methods",
        export_name="patpy-supervised-methods",
        display_name="patpy Supervised Methods",
        short_description="Predict donor labels with patpy models",
        default_prompt="Use $patpy-supervised-methods to predict donor-level labels from per-cell features with patpy.",
    ),
    SkillSpec(
        source_dir="evaluation",
        export_name="patpy-evaluation",
        display_name="patpy Evaluation",
        short_description="Score patpy outputs against metadata",
        default_prompt="Use $patpy-evaluation to score a patpy representation or prediction against donor metadata.",
    ),
    SkillSpec(
        source_dir="plotting",
        export_name="patpy-plotting",
        display_name="patpy Plotting",
        short_description="Plot patpy correlation and association results",
        default_prompt="Use $patpy-plotting to visualize patpy correlation results or embedding-covariate associations.",
    ),
)


def iter_skill_specs() -> Iterator[SkillSpec]:
    """Yield the packaged patpy skill specs in export order."""

    yield from SKILL_SPECS
