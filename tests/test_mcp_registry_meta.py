"""Validate the BioContextAI Registry meta.yaml against the upstream schema.

The schema is committed as a fixture at
``tests/fixtures/biocontext_registry_schema.json`` so this test runs offline.
Re-fetch it with::

    curl -sSL https://raw.githubusercontent.com/biocontext-ai/registry/main/schema.json \\
      -o tests/fixtures/biocontext_registry_schema.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
META_PATH = REPO_ROOT / "mcp" / "meta.yaml"
SCHEMA_FIXTURE = Path(__file__).parent / "fixtures" / "biocontext_registry_schema.json"


@pytest.fixture(scope="module")
def meta() -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML is required for the registry meta test.")
    if not META_PATH.is_file():
        pytest.skip(f"{META_PATH} is missing.")
    return yaml.safe_load(META_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def schema() -> dict:
    if not SCHEMA_FIXTURE.is_file():
        pytest.skip(f"{SCHEMA_FIXTURE} is missing; refresh it from the upstream registry.")
    return json.loads(SCHEMA_FIXTURE.read_text(encoding="utf-8"))


def test_meta_yaml_validates_against_registry_schema(meta: dict, schema: dict) -> None:
    """mcp/meta.yaml must satisfy biocontext-ai/registry/schema.json."""
    jsonschema = pytest.importorskip(
        "jsonschema", reason="jsonschema is required for registry meta validation."
    )
    jsonschema.validate(instance=meta, schema=schema)


def test_meta_yaml_carries_expected_identity_fields(meta: dict) -> None:
    """Sanity-check the fields the BioContextAI submission checklist reviews manually."""
    assert meta["identifier"] == "lueckenlab/patpy"
    assert meta["@id"].startswith("https://github.com/lueckenlab/patpy")
    assert meta["codeRepository"].startswith("https://github.com/lueckenlab/patpy")
    assert meta["license"].startswith("https://spdx.org/licenses/")
    assert meta["applicationCategory"] in {
        "HealthApplication",
        "EducationApplication",
        "ReferenceApplication",
        "DeveloperApplication",
        "UtilitiesApplication",
    }
    assert "Python" in meta["programmingLanguage"]
    assert meta["maintainer"], "At least one maintainer must be declared."
    assert all("name" in m and "@type" in m for m in meta["maintainer"])
