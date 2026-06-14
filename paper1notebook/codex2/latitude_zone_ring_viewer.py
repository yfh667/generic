from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer


@dataclass(frozen=True)
class LatitudeZone:
    name: str
    min_abs_deg: float
    max_abs_deg: float
    color: str


DEFAULT_LATITUDE_ZONES = (
    LatitudeZone("0-30 deg", 0.0, 30.0, "#1B9E77"),
    LatitudeZone("30-60 deg", 30.0, 60.0, "#2F6FED"),
    LatitudeZone("60-90 deg", 60.0, 90.000001, "#E3B100"),
)


def latitude_deg_from_xyz_km(positions_km: np.ndarray) -> np.ndarray:
    """Return geocentric latitude in degrees for shape (..., 3) positions."""

    positions = np.asarray(positions_km, dtype=np.float64)
    radius = np.linalg.norm(positions, axis=-1)
    z = positions[..., 2]
    ratio = np.divide(z, radius, out=np.zeros_like(z, dtype=np.float64), where=radius > 0)
    return np.degrees(np.arcsin(np.clip(ratio, -1.0, 1.0)))


def classify_latitude_zones(
    latitude_deg: np.ndarray,
    zones: Sequence[LatitudeZone] = DEFAULT_LATITUDE_ZONES,
) -> np.ndarray:
    abs_lat = np.abs(np.asarray(latitude_deg, dtype=np.float64))
    zone_ids = np.full(abs_lat.shape, -1, dtype=np.int16)
    for idx, zone in enumerate(zones):
        mask = (abs_lat >= float(zone.min_abs_deg)) & (abs_lat < float(zone.max_abs_deg))
        zone_ids[mask] = int(idx)
    return zone_ids


def load_latitude_zone_data(
    *,
    position_cache_dir: str | Path,
    steps: Sequence[int],
    zones: Sequence[LatitudeZone] = DEFAULT_LATITUDE_ZONES,
    chunk_rows: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Load latitude and zone ids for requested steps from a position cache."""

    cache_dir = Path(position_cache_dir)
    positions = np.load(cache_dir / "positions_km.npy", mmap_mode="r")
    times = np.load(cache_dir / "times_s.npy", mmap_mode="r")
    requested = np.asarray([int(x) for x in steps], dtype=np.int64)
    row_ids = np.searchsorted(np.asarray(times, dtype=np.int64), requested)
    if np.any(row_ids >= len(times)) or np.any(np.asarray(times)[row_ids] != requested):
        missing = requested[(row_ids >= len(times)) | (np.asarray(times)[np.minimum(row_ids, len(times) - 1)] != requested)]
        raise ValueError(f"position cache does not contain requested steps: {missing[:10].tolist()}")

    latitude = np.empty((len(row_ids), int(positions.shape[1])), dtype=np.float32)
    zone_ids = np.empty(latitude.shape, dtype=np.int16)
    chunk_rows = max(1, int(chunk_rows))
    for out_start in range(0, len(row_ids), chunk_rows):
        out_end = min(len(row_ids), out_start + chunk_rows)
        ids = row_ids[out_start:out_end]
        if ids.size and int(ids[-1]) - int(ids[0]) == ids.size - 1:
            pos_chunk = positions[int(ids[0]) : int(ids[-1]) + 1]
        else:
            pos_chunk = positions[ids]
        lat_chunk = latitude_deg_from_xyz_km(pos_chunk).astype(np.float32)
        latitude[out_start:out_end] = lat_chunk
        zone_ids[out_start:out_end] = classify_latitude_zones(lat_chunk, zones=zones)
    return latitude, zone_ids


class LatitudeZoneRingViewer(SatelliteTopology2DViewer):
    """Experimental 2D viewer overlay: node rings encode current latitude zone."""

    def __init__(
        self,
        *args,
        latitude_deg: np.ndarray,
        latitude_zone_ids: np.ndarray,
        latitude_zones: Sequence[LatitudeZone] = DEFAULT_LATITUDE_ZONES,
        ring_width: float = 0.075,
        ring_radius: float = 0.235,
        **kwargs,
    ):
        self.latitude_deg = np.asarray(latitude_deg, dtype=np.float32)
        self.latitude_zone_ids = np.asarray(latitude_zone_ids, dtype=np.int16)
        self.latitude_zones = tuple(latitude_zones)
        self.latitude_ring_width = float(ring_width)
        self.latitude_ring_radius = float(ring_radius)
        self.latitude_ring_items: list[QtWidgets.QGraphicsEllipseItem] = []
        super().__init__(*args, **kwargs)

        expected_shape = (len(self.steps), int(self.config.total_sats))
        if tuple(self.latitude_zone_ids.shape) != expected_shape:
            raise ValueError(f"latitude_zone_ids shape {self.latitude_zone_ids.shape} != {expected_shape}")
        if tuple(self.latitude_deg.shape) != expected_shape:
            raise ValueError(f"latitude_deg shape {self.latitude_deg.shape} != {expected_shape}")

    def _build_ui(self):
        super()._build_ui()
        legend = []
        for zone in self.latitude_zones:
            legend.append(
                f'<span style="color:{zone.color}; font-weight:700;">&#9711;</span> '
                f'{zone.name}'
            )
        self.latitude_legend_label = QtWidgets.QLabel("Latitude rings: " + "  ".join(legend))
        self.latitude_legend_label.setObjectName("statusLabel")
        self.latitude_legend_label.setWordWrap(True)
        self.controls_panel.layout().addWidget(self.latitude_legend_label)

    def _draw_nodes(self):
        self.latitude_ring_items = []
        for node in range(self.config.total_sats):
            p, y = self.node_grid_pos(node)
            item = QtWidgets.QGraphicsEllipseItem(
                float(p) - self.latitude_ring_radius,
                float(y) - self.latitude_ring_radius,
                2 * self.latitude_ring_radius,
                2 * self.latitude_ring_radius,
            )
            item.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 0)))
            item.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 0), self.latitude_ring_width))
            item.setZValue(18)
            self.scene.addItem(item)
            self.latitude_ring_items.append(item)
        super()._draw_nodes()

    def update_step(self, row: int, *, sync_slider: bool = True):
        super().update_step(row, sync_slider=sync_slider)
        self.update_latitude_rings(int(self.current_row))

    def update_latitude_rings(self, row: int):
        if not self.latitude_ring_items:
            return
        row = int(max(0, min(int(row), self.latitude_zone_ids.shape[0] - 1)))
        counts = [0 for _ in self.latitude_zones]
        for node, item in enumerate(self.latitude_ring_items):
            zone_id = int(self.latitude_zone_ids[row, int(node)])
            if 0 <= zone_id < len(self.latitude_zones):
                zone = self.latitude_zones[zone_id]
                color = QtGui.QColor(zone.color)
                color.setAlpha(235)
                pen = QtGui.QPen(color)
                pen.setWidthF(self.latitude_ring_width)
                pen.setCapStyle(QtCore.Qt.RoundCap)
                item.setPen(pen)
                item.setVisible(True)
                counts[zone_id] += 1
            else:
                item.setVisible(False)

        if hasattr(self, "latitude_legend_label"):
            parts = []
            for idx, zone in enumerate(self.latitude_zones):
                parts.append(
                    f'<span style="color:{zone.color}; font-weight:700;">&#9711;</span> '
                    f'{zone.name}: {counts[idx]}'
                )
            self.latitude_legend_label.setText("Latitude rings: " + "  ".join(parts))

    def node_latitude_text(self, node: int, row: int | None = None) -> str:
        row = self.current_row if row is None else int(row)
        node = int(node)
        if not (0 <= row < self.latitude_deg.shape[0] and 0 <= node < self.latitude_deg.shape[1]):
            return "latitude=unknown"
        zone_id = int(self.latitude_zone_ids[row, node])
        zone_name = self.latitude_zones[zone_id].name if 0 <= zone_id < len(self.latitude_zones) else "outside"
        return f"lat={float(self.latitude_deg[row, node]):.2f} deg; zone={zone_name}"

    def update_cursor_grid_label(self, x: float, y: float):
        p = int(round(float(x)))
        scene_y = int(round(float(y)))
        if not (0 <= p < int(self.config.P) and 0 <= scene_y < int(self.config.N)):
            self.coord_label.setText("Cursor grid: outside")
            return
        dist = math.hypot(float(x) - p, float(y) - scene_y)
        if dist > 0.55:
            axis_y = self.scene_y_to_axis_y(float(y))
            self.coord_label.setText(f"Cursor grid: display approx=({float(x):.2f}, {axis_y:.2f})")
            return
        raw_node = self.grid_pos_to_node(p, scene_y)
        raw_p, raw_y = divmod(int(raw_node), int(self.config.N))
        self.coord_label.setText(
            f"Cursor grid: x={p}, y={raw_y}; node={raw_node}; raw x={raw_p}, y={raw_y}; "
            f"{self.node_latitude_text(raw_node)}; groups={self.node_group_text(raw_node)}"
        )
