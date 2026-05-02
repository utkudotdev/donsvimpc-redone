import json
from pathlib import Path


def discover_env_name(dir_path: Path | None, default: str = "basic") -> str:
    """Read the environment name from a sibling metadata.json.

    Supports both dataset metadata (env stored at top-level "env") and
    checkpoint metadata (env stored under "args"."env"). Returns `default`
    if `dir_path` is None or no metadata is found.
    """
    if dir_path is None:
        return default
    metadata_path = dir_path / "metadata.json"
    if not metadata_path.exists():
        return default
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    env = metadata.get("env") or metadata.get("args", {}).get("env")
    if env is None:
        return default
    print(f"Using env '{env}' from {metadata_path}")
    return env
