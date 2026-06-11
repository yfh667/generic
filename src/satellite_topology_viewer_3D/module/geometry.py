from __future__ import annotations

from typing import Iterable

import numpy as np
import pyvista as pv
from PyQt5 import QtGui

from src.config.viewer_config import ViewerConfig


EARTH_R_KM = 6371.0


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    color = QtGui.QColor(str(value))
    if not color.isValid():
        color = QtGui.QColor("#3b82f6")
    return color.red(), color.green(), color.blue()


def segments_to_polydata(points: np.ndarray, edges: Iterable[tuple[int, int]]) -> pv.PolyData | None:
    """Build independent line cells for edge-value coloring."""

    edges = list(edges)
    if not edges:
        return None

    seg_pts = np.empty((2 * len(edges), 3), dtype=np.float32)
    line_cells = np.empty((len(edges), 3), dtype=np.int64)
    for idx, (src, dst) in enumerate(edges):
        seg_pts[2 * idx] = points[int(src)]
        seg_pts[2 * idx + 1] = points[int(dst)]
        line_cells[idx] = [2, 2 * idx, 2 * idx + 1]

    mesh = pv.PolyData()
    mesh.points = seg_pts
    mesh.lines = line_cells.ravel()
    return mesh


def build_orbit_edges(config: ViewerConfig) -> list[tuple[int, int]]:
    """Connect satellites within each orbit plane as closed orbit rings."""

    edges: list[tuple[int, int]] = []
    for p in range(int(config.P)):
        base = p * int(config.N)
        for y in range(int(config.N)):
            edges.append((base + y, base + ((y + 1) % int(config.N))))
    return edges
