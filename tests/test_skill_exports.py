import json
from pathlib import Path

from patpy.skills import export_skill_bundle


def test_export_claude_code_bundle(tmp_path):
    manifest = export_skill_bundle(tmp_path, "claude-code")

    skill_path = tmp_path / ".claude" / "skills" / "patpy-preprocessing" / "SKILL.md"
    assert skill_path.is_file()
    assert skill_path.read_text(encoding="utf-8").startswith("---\nname: patpy-preprocessing\n")
    assert manifest["target"] == "claude-code"
    assert len(manifest["skills"]) == 7


def test_export_codex_bundle_writes_openai_metadata(tmp_path):
    export_skill_bundle(tmp_path, "codex")

    metadata_path = tmp_path / ".codex" / "skills" / "patpy-evaluation" / "agents" / "openai.yaml"
    metadata = metadata_path.read_text(encoding="utf-8")
    assert 'display_name: "patpy Evaluation"' in metadata
    assert "$patpy-evaluation" in metadata


def test_export_biocontext_bundle_writes_manifest(tmp_path):
    export_skill_bundle(tmp_path, "biocontext")

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["target"] == "biocontext"
    assert {entry["id"] for entry in manifest["skills"]} == {
        "patpy",
        "patpy-datasets",
        "patpy-evaluation",
        "patpy-plotting",
        "patpy-preprocessing",
        "patpy-sample-representation",
        "patpy-supervised-methods",
    }


def test_example_mcp_configs_are_valid_json():
    repo_root = Path(__file__).resolve().parents[1]
    example_paths = [
        "examples/mcp/project.mcp.json",
        "examples/mcp/claude_desktop_config.json",
        "examples/mcp/claude_desktop_with_biocontext.json",
    ]

    for relative_path in example_paths:
        with (repo_root / relative_path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert "mcpServers" in payload
