"""Streamlit component wrapper for the React Three Fiber terrain canvas."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

import streamlit.components.v1 as components

_COMPONENT_NAME = "terrain_canvas"
_DEV_URL_ENV = "TERRAIN_CANVAS_DEV_URL"

# DEVELOPMENT MODE:
# Set this environment variable to use Vite dev server instead of built assets
# For example, before running streamlit:
#   $env:TERRAIN_CANVAS_DEV_URL = "http://0.0.0.0:5175"
#   streamlit run app.py
# Then in another terminal, run:
#   cd terrain_component/frontend
#   npm run dev -- --host 0.0.0.0 --port 5175


def _declare_component() -> Any:
    """
    Declare the Streamlit component.
    
    Priority:
    1. If TERRAIN_CANVAS_DEV_URL environment variable is set, use Vite dev server (development)
    2. Otherwise, use pre-built assets from ./dist/ (production)
    """
    dev_url = os.environ.get(_DEV_URL_ENV)
    if dev_url:
        try:
            print(f"🔧 Terrain component: Using dev server at {dev_url}")
            return components.declare_component(_COMPONENT_NAME, url=dev_url)
        except Exception as e:
            print(f"⚠️  Failed to connect to dev server at {dev_url}: {e}")
            print("Falling back to built assets...")

    build_dir = Path(__file__).parent / "frontend" / "dist"
    if not build_dir.exists():
        raise FileNotFoundError(
            "Terrain component build not found.\n"
            "Run the following commands to build:\n"
            "  cd terrain_component/frontend\n"
            "  npm install && npm run build\n"
            "  cd ../.."
        )
    print(f"✅ Terrain component: Using built assets from {build_dir}")
    return components.declare_component(_COMPONENT_NAME, path=str(build_dir))


_COMPONENT_FUNC = _declare_component()


def terrain_canvas(
    *,
    data: Optional[Dict[str, Any]] = None,
    height: int = 640,
    key: Optional[str] = None,
) -> Any:
    """Render the bundled React terrain canvas inside Streamlit."""
    payload = data or {}
    return _COMPONENT_FUNC(data=payload, height=height, key=key, default={})
