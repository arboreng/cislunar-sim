"""Shared path resolution for external validation data files."""

from __future__ import annotations

import os
from pathlib import Path

VALIDATION_DATA_ENV_VAR = "CISLUNAR_VALIDATION_DATA_DIR"


def _package_validation_dir() -> Path:
    return Path(__file__).resolve().parent


def _legacy_validation_data_dir() -> Path:
    return _package_validation_dir() / "data"


def _repo_root() -> Path | None:
    for parent in _package_validation_dir().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def default_validation_data_dir() -> str:
    """Return the canonical external-data directory for this checkout."""
    override = os.environ.get(VALIDATION_DATA_ENV_VAR)
    if override:
        return os.fspath(Path(override).expanduser())

    repo_root = _repo_root()
    if repo_root is not None:
        return os.fspath(repo_root / "validation" / "data")

    return os.fspath(_legacy_validation_data_dir())


def validation_data_file(filename: str) -> str:
    """
    Return the preferred path for an external validation file.

    During the migration to the top-level ``validation/data`` directory, we
    still honour the legacy package-local location if the requested file exists
    there and has not yet been moved.
    """
    preferred = Path(default_validation_data_dir()) / filename
    legacy = _legacy_validation_data_dir() / filename

    if preferred == legacy:
        return os.fspath(preferred)
    if preferred.exists() or not legacy.exists():
        return os.fspath(preferred)
    return os.fspath(legacy)


DEFAULT_VALIDATION_DATA_DIR = default_validation_data_dir()
DEFAULT_LIGHTSAIL2_ARCHIVE = validation_data_file("lightsail2_gp_history.json")
DEFAULT_BEACON_FILE = validation_data_file("lightsail2_beacons_p1.json")
