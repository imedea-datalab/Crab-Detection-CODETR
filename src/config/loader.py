import os
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv, dotenv_values


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    """Load base config and merge with optional overrides.

    Precedence (lowest to highest):
      0) .env file (non-APP prefixed, non-conflicting variables for local development)
      1) config.default.yaml
      2) config.override.yaml (ignored by VCS)
      3) Environment variables with prefix APP__ (from system or .env, double underscore for nesting)
    """
    project_root = Path(__file__).resolve().parents[2]
    base_path = project_root / "config.default.yaml"
    local_path = project_root / "config.override.yaml"

    cfg = _load_yaml(base_path)
    cfg = _deep_merge(cfg, _load_yaml(local_path))

    # Load variables from .env file.
    # We do this after loading YAMLs to handle non-conflicting, non-APP__ vars.
    dotenv_path = project_root / ".env"
    if dotenv_path.is_file():
        # Add variables from .env that are not APP__ prefixed and do not conflict
        # with keys from YAML files. These have the lowest precedence.
        dotenv_vars = dotenv_values(dotenv_path)
        for key, value in dotenv_vars.items():
            if not key.startswith("APP__") and key not in cfg:
                cfg[key] = value

        # Load .env into os.environ to process APP__ prefixed variables.
        # This will not override existing environment variables.
        load_dotenv(dotenv_path)

    # Overlay environment variables using APP__KEY__SUBKEY format.
    # This gives them the highest precedence.
    prefix = "APP__"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path_keys = env_key[len(prefix) :].split("__")
        cursor: Dict[str, Any] = cfg
        for part in path_keys[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]  # type: ignore[assignment]
        cursor[path_keys[-1]] = env_value

    return cfg

