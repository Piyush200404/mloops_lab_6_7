"""Tiny helper to load configs/config.yaml."""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=None) -> dict:
    """Read the YAML configuration file and return it as a dict."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a config-relative path against the project root."""
    return PROJECT_ROOT / relative
