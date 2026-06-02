from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PHASE0_REQUIRED_DIRS = [
    "data/raw",
    "data/processed",
    "pipelines/notebooks",
    "services/api",
    "services/ui",
    "docs/architecture",
    "docs/governance",
    "tests",
]

PHASE0_REQUIRED_FILES = [
    "README.md",
    "readme_phase_0.md",
    ".env.example",
    "scripts/start_api.ps1",
    "scripts/start_ui.ps1",
]


@dataclass
class BootstrapCheckResult:
    root: Path
    missing_dirs: List[str]
    missing_files: List[str]

    @property
    def is_valid(self) -> bool:
        return not self.missing_dirs and not self.missing_files


def resolve_data_dir(project_root: Path, env_raw_data_dir: Optional[str] = None) -> Path:
    """Resolve raw data directory with deterministic fallback order."""
    if env_raw_data_dir and env_raw_data_dir.strip():
        candidate = Path(env_raw_data_dir).expanduser().resolve()
        if candidate.exists():
            return candidate

    candidates = [
        project_root / "data" / "raw",
        project_root / "raw_kaggle_data",
        project_root.parent / "MAIN_RULE_EXTRACTION" / "raw_kaggle_data",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def validate_phase0_workspace(project_root: Path) -> BootstrapCheckResult:
    missing_dirs: List[str] = []
    missing_files: List[str] = []

    for rel in PHASE0_REQUIRED_DIRS:
        if not (project_root / rel).exists():
            missing_dirs.append(rel)

    for rel in PHASE0_REQUIRED_FILES:
        if not (project_root / rel).exists():
            missing_files.append(rel)

    return BootstrapCheckResult(
        root=project_root,
        missing_dirs=missing_dirs,
        missing_files=missing_files,
    )


def build_phase0_manifest(project_root: Path) -> Dict[str, object]:
    result = validate_phase0_workspace(project_root)
    return {
        "project_root": str(project_root.resolve()),
        "required_dirs": PHASE0_REQUIRED_DIRS,
        "required_files": PHASE0_REQUIRED_FILES,
        "missing_dirs": result.missing_dirs,
        "missing_files": result.missing_files,
        "is_valid": result.is_valid,
    }
