"""Locate the project root so code works from scripts, notebooks and Google Colab."""

from __future__ import annotations

import os
from pathlib import Path

_MARKER = Path("configs") / "data.yaml"


def project_root(start: str | Path | None = None) -> Path:
    """Return the project root: $GENAI_PROJECT_ROOT, or the nearest parent containing configs/data.yaml."""
    env = os.environ.get("GENAI_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / _MARKER).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {_MARKER} above {here}. Set GENAI_PROJECT_ROOT to the project folder."
    )


def use_project_matplotlib_cache(root: Path) -> None:
    """Keep Matplotlib's font cache inside the project (call before importing matplotlib)."""
    cache = root / ".cache" / "matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
