from __future__ import annotations

from pathlib import Path

from cislunar.validation import data_paths


def test_default_validation_data_dir_uses_repo_root_checkout():
    repo_root = Path(__file__).resolve().parent.parent
    expected = repo_root / "validation" / "data"
    assert Path(data_paths.DEFAULT_VALIDATION_DATA_DIR) == expected


def test_default_validation_file_paths_live_under_repo_validation_dir():
    root = Path(data_paths.DEFAULT_VALIDATION_DATA_DIR)
    assert Path(data_paths.DEFAULT_LIGHTSAIL2_ARCHIVE) == root / "lightsail2_gp_history.json"
    assert Path(data_paths.DEFAULT_BEACON_FILE) == root / "lightsail2_beacons_p1.json"


def test_validation_data_env_override_takes_precedence(monkeypatch):
    custom = Path("/tmp/cislunar-validation")
    monkeypatch.setenv(data_paths.VALIDATION_DATA_ENV_VAR, str(custom))

    assert Path(data_paths.default_validation_data_dir()) == custom
    assert Path(data_paths.validation_data_file("example.json")) == custom / "example.json"
