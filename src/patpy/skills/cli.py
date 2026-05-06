from __future__ import annotations

import argparse
import json
from typing import Sequence

from .export import ExportTarget, export_skill_bundle


def _build_parser(default_target: ExportTarget | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export patpy skills for Claude Code, Codex, or BioContext.")
    parser.add_argument("--dest", required=True, help="Destination directory that will receive the exported bundle.")
    if default_target is None:
        parser.add_argument(
            "--target",
            required=True,
            choices=("claude-code", "codex", "biocontext"),
            help="Export layout to generate.",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest = export_skill_bundle(destination=args.dest, target=args.target)
    print(json.dumps(manifest, indent=2))
    return 0


def export_biocontext_main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser(default_target="biocontext")
    args = parser.parse_args(argv)
    manifest = export_skill_bundle(destination=args.dest, target="biocontext")
    print(json.dumps(manifest, indent=2))
    return 0
