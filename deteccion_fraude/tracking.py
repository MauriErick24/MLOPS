"""Lineage tracking para deteccion de fraude."""

import subprocess


def get_lineage_metadata() -> dict[str, str]:
    """Extrae hash de Git y estado de DVC para trazabilidad de auditoria."""
    metadata: dict[str, str] = {}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata["git_commit"] = result.stdout.strip() or "standalone"
    except OSError:
        metadata["git_commit"] = "git_unavailable"

    try:
        result = subprocess.run(
            ["dvc", "status"],
            capture_output=True,
            text=True,
            check=False,
        )
        metadata["dvc_status"] = "synced" if not result.stdout.strip() else "uncommitted"
    except OSError:
        metadata["dvc_status"] = "dvc_unavailable"

    return metadata
