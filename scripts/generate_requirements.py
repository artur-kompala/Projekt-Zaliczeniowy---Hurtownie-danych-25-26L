"""Compile pinned requirements files for project subpackages.

This script uses pip-tools to generate requirements.txt files from the
requirements.in files located under src/api, src/prefect, and src/dashboard.
Run it from the project root or let it resolve the root from its own location.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGETS = (
    Path("src/api"),
    Path("src/prefect"),
    Path("src/dashboard"),
)


def compile_requirements(project_root: Path, target_dir: Path) -> None:
    requirements_in = project_root / target_dir / "requirements.in"
    requirements_txt = project_root / target_dir / "requirements.txt"

    if not requirements_in.exists():
        raise FileNotFoundError(f"Missing input file: {requirements_in}")

    command = [
        sys.executable,
        "-m",
        "piptools",
        "compile",
        "--output-file",
        str(requirements_txt),
        str(requirements_in),
    ]

    subprocess.run(command, cwd=project_root, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    for target_dir in TARGETS:
        compile_requirements(project_root, target_dir)


if __name__ == "__main__":
    main()
