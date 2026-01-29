"""Streamlit component wrapper for the React Three Fiber terrain canvas."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import os

import streamlit.components.v1 as components

_COMPONENT_NAME = "terrain_canvas"
_DEV_URL_ENV = "TERRAIN_CANVAS_DEV_URL"


def _declare_component() -> Any:
    """Declare the Streamlit component, preferring the bundled build."""
    dev_url = os.environ.get(_DEV_URL_ENV)
    if dev_url:
        try:
            return components.declare_component(_COMPONENT_NAME, url=dev_url)
        except Exception:
            pass

    build_dir = Path(__file__).parent / "frontend" / "dist"
    if not build_dir.exists():
        raise FileNotFoundError(
            "Terrain component build not found. Run `npm install && npm run build` "
            "inside terrain_component/frontend before using terrain_canvas()."
        )
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
