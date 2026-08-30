"""Narrow pkg_resources compatibility for dependencies that only scan plugins."""

from __future__ import annotations

import sys
from importlib.metadata import entry_points
from types import ModuleType


def install_pkg_resources_compatibility() -> None:
    """Supply only iter_entry_points when legacy setuptools is unavailable."""

    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError:
        compatibility = ModuleType("pkg_resources")
        compatibility.iter_entry_points = lambda group: entry_points().select(
            group=group
        )
        sys.modules["pkg_resources"] = compatibility
