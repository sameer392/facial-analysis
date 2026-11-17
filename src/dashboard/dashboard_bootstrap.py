"""Reload the project's own modules at the start of every dashboard run.

Streamlit re-runs the page script on each interaction, but Python keeps imported
modules in memory, so edits to helper files were not picked up until the server was
restarted. Calling reload_project_modules() before the imports fixes that.

This file itself is not reloaded. Edits to it still need a server restart.
"""

import importlib
import sys
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "analysis"
DASHBOARD_DIR = Path(__file__).resolve().parent

# Dependency order: each module is reloaded after the modules it imports.
RELOAD_ORDER = [
    "project_paths",
    "motion_analyzer",
    "facial_3d_analyzer",
    "facial_analyzer",
    "visualization",
    "common",
]


def _refresh(name: str) -> None:
    """Reload a module, or import it fresh if it is missing or was dropped during a reload.

    importlib.reload raises ImportError ("not in sys.modules") when Streamlit's file watcher
    has already removed the module from sys.modules. In that case a normal import is the
    correct recovery.
    """
    module = sys.modules.get(name)
    if module is None:
        importlib.import_module(name)
        return
    try:
        importlib.reload(module)
    except ImportError:
        sys.modules.pop(name, None)
        importlib.import_module(name)


def reload_project_modules() -> None:
    for folder in (ANALYSIS_DIR, DASHBOARD_DIR):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    for name in RELOAD_ORDER:
        _refresh(name)
