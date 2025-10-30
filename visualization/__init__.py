"""Visualization package for layout optimization"""

from .layout_viz import (
    create_3d_layout_visualization,
    create_2d_layout_visualization,
    ELEMENT_COLORS
)

try:
    from .threejs_viz import (
        create_threejs_scene,
        display_threejs_scene,
        export_threejs_html,
        THREEJS_AVAILABLE
    )
except ImportError:
    THREEJS_AVAILABLE = False
    print("⚠️ PyThreeJS visualization not available")

__all__ = [
    'create_3d_layout_visualization',
    'create_2d_layout_visualization',
    'ELEMENT_COLORS',
    'create_threejs_scene',
    'display_threejs_scene',
    'export_threejs_html',
    'THREEJS_AVAILABLE'
]
