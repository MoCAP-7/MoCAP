"""Import-path bootstrap for the out-of-tree Cap-X checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPX_REPO = Path("/home/yor/codefield/cap-x")


def configure_import_paths() -> tuple[Path, Path]:
    """Make the production YOR package and the upstream Cap-X package importable.

    ``CAPX_REPO`` may point to another Cap-X checkout.  The environment flag is
    understood by the pinned Cap-X checkout and avoids importing unrelated
    simulator/Franka integrations on the Jetson deployment machine.
    """

    capx_repo = Path(os.environ.get("CAPX_REPO", str(DEFAULT_CAPX_REPO))).expanduser()
    yor_src = REPO_ROOT / "yor_agent" / "src"
    for path in (capx_repo, yor_src, REPO_ROOT):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)
    os.environ.setdefault("CAPX_MINIMAL_YOR", "1")
    return capx_repo, yor_src

