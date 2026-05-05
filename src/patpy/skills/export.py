from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Literal

from .catalog import SkillSpec, iter_skill_specs

ExportTarget = Literal["claude-code", "codex", "biocontext"]


@dataclass(frozen=True)
class ParsedSkill:
    """Canonical representation of one packaged skill document."""

    source_name: str
    description: str
    body: str


def _skills_root():
    return files("patpy.skills")


def _resource_for_skill(spec: SkillSpec):
    root = _skills_root()
    resource_dir = root if not spec.source_dir else root.joinpath(spec.source_dir)
    return resource_dir.joinpath("SKILL.md")


def _split_frontmatter(skill_text: str) -> tuple[dict[str, str], str]:
    lines = skill_text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("Skill file is missing YAML frontmatter.")

    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("Skill file has unterminated YAML frontmatter.")

    frontmatter: dict[str, str] = {}
    for raw_line in lines[1:end_idx]:
        if ":" not in raw_line:
            continue
        key, value = raw_line.split(":", 1)
        frontmatter[key.strip()] = value.strip().strip('"')

    body = "\n".join(lines[end_idx + 1 :]).lstrip()
    return frontmatter, body


def _load_skill(spec: SkillSpec) -> ParsedSkill:
    text = _resource_for_skill(spec).read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    return ParsedSkill(
        source_name=frontmatter["name"],
        description=frontmatter["description"],
        body=body,
    )


def render_exported_skill(spec: SkillSpec) -> str:
    """Render one packaged patpy skill with export-safe frontmatter."""

    parsed = _load_skill(spec)
    return (
        "---\n"
        f"name: {spec.export_name}\n"
        f'description: "{parsed.description}"\n'
        "---\n\n"
        f"{parsed.body.rstrip()}\n"
    )


def _quote_yaml(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def render_openai_metadata(spec: SkillSpec) -> str:
    """Render Codex/OpenAI-facing metadata for one exported skill."""

    return "\n".join(
        [
            "interface:",
            f"  display_name: {_quote_yaml(spec.display_name)}",
            f"  short_description: {_quote_yaml(spec.short_description)}",
            f"  default_prompt: {_quote_yaml(spec.default_prompt)}",
            "policy:",
            "  allow_implicit_invocation: true",
            "",
        ]
    )


def _target_root(destination: Path, target: ExportTarget) -> Path:
    if target == "claude-code":
        return destination / ".claude" / "skills"
    if target == "codex":
        return destination / ".codex" / "skills"
    return destination / "skills"


def export_skill_bundle(destination: str | Path, target: ExportTarget) -> dict[str, object]:
    """Export packaged patpy skills to Claude Code, Codex, or BioContext layouts."""

    dest = Path(destination).expanduser().resolve()
    root = _target_root(dest, target)
    root.mkdir(parents=True, exist_ok=True)

    skills_manifest: list[dict[str, str]] = []
    for spec in iter_skill_specs():
        skill_dir = root / spec.export_name
        skill_dir.mkdir(parents=True, exist_ok=True)

        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(render_exported_skill(spec), encoding="utf-8")

        metadata_path = None
        if target == "codex":
            agents_dir = skill_dir / "agents"
            agents_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = agents_dir / "openai.yaml"
            metadata_path.write_text(render_openai_metadata(spec), encoding="utf-8")

        parsed = _load_skill(spec)
        entry = {
            "id": spec.export_name,
            "display_name": spec.display_name,
            "description": parsed.description,
            "skill_path": str(skill_path),
        }
        if metadata_path is not None:
            entry["openai_metadata_path"] = str(metadata_path)
        skills_manifest.append(entry)

    manifest = {
        "target": target,
        "root": str(root),
        "skills": skills_manifest,
    }

    manifest_name = "manifest.json" if target == "biocontext" else f"{target}-manifest.json"
    (dest / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
