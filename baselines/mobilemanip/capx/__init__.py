"""Cap-X + YOR navigation baseline.

This package intentionally depends on, but does not modify, the production
``yor_agent`` and ``nav_planner`` implementations.
"""

from .bootstrap import configure_import_paths

configure_import_paths()

__all__ = ["configure_import_paths"]

