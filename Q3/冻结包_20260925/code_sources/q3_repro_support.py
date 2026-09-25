"""Resolve the installed math-modeling figure tools without a user-specific path."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_math_modeling_skill_root() -> Path:
    """Find the skill root, allowing an explicit override for non-default installs."""
    candidates: list[Path] = []
    override = os.environ.get("MATH_MODELING_SKILL_ROOT")
    if override:
        candidates.append(Path(override).expanduser())

    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        candidates.append(Path(codex_home).expanduser() / "skills" / "math-modeling")
    candidates.append(Path.home() / ".codex" / "skills" / "math-modeling")

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "tools" / "figure" / "scripts").is_dir():
            return resolved

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Cannot locate math-modeling figure tools. Set MATH_MODELING_SKILL_ROOT "
        f"to the installed skill root. Searched: {searched}"
    )


def figure_scripts_dir() -> Path:
    return resolve_math_modeling_skill_root() / "tools" / "figure" / "scripts"

