import os
from pathlib import Path

def is_running_in_docker() -> bool:
    """Check if we are currently running inside a Docker container."""
    return os.path.exists("/.dockerenv")

def resolve_volume_path(host_path: str, container_path: str) -> str:
    """Resolve local path based on environment."""
    if is_running_in_docker():
        return container_path
    return str(Path(host_path).resolve())
