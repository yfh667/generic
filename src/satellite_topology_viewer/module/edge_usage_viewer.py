from __future__ import annotations

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer


class EdgeUsageTopology2DViewer(SatelliteTopology2DViewer):
    """2D topology viewer preset for shortest-path edge usage / betweenness."""

    def __init__(
        self,
        config: ViewerConfig,
        *,
        steps: list[int],
        edge_table,
        edge_usage_values,
        window_title: str,
        group_data: dict | None = None,
        show_groups: bool = True,
        value_max: float | None = None,
        edge_active_mask=None,
        edge_building_mask=None,
        **viewer_kwargs,
    ):
        edge_usage_values = np.asarray(edge_usage_values, dtype=np.float32)
        inferred_max = float(np.nanmax(edge_usage_values)) if edge_usage_values.size else 0.0
        defaults = {
            "edge_value_label": "edge_usage",
            "scale_edge_width_by_value": True,
            "value_width_min": 0.006,
            "value_width_max": 0.085,
            "value_color_mode": "red_alpha",
            "value_solid_color": "#C1121F",
            "value_alpha_min": 28,
            "value_alpha_max": 235,
            "zero_value_edges_visible": False,
            "zero_value_threshold": 0.0,
            "show_topology_under_edge_values": True,
            "topology_edge_color": "#000000",
            "topology_edge_alpha": 150,
            "topology_edge_width": 0.014,
            "show_grid_lines": False,
        }
        defaults.update(viewer_kwargs)
        super().__init__(
            config,
            steps=steps,
            edge_table=edge_table,
            edge_active_mask=edge_active_mask,
            edge_building_mask=edge_building_mask,
            edge_values=edge_usage_values,
            value_min=0.0,
            value_max=inferred_max if value_max is None else float(value_max),
            window_title=window_title,
            group_data=group_data or {},
            show_groups=show_groups,
            **defaults,
        )

    def format_edge_value(self, edge_idx: int, row: int, value: float) -> str:
        return f"edge_usage={float(value):.3f} shortest-path demand"

    def edge_extra_description(self, edge_idx: int, row: int, value: float) -> str:
        if float(value) <= 0.0:
            return "not used by current region-pair shortest paths"
        return "red overlay means this edge is used by current region-pair shortest paths"


class LazyEdgeUsageTopology2DViewer(EdgeUsageTopology2DViewer):
    """Compute edge usage only for the currently displayed row."""

    def __init__(
        self,
        config: ViewerConfig,
        *,
        steps: list[int],
        edge_table,
        edge_usage_provider,
        window_title: str,
        group_data: dict | None = None,
        show_groups: bool = True,
        initial_value_max: float = 1.0,
        edge_active_mask=None,
        edge_building_mask=None,
    ):
        self.edge_usage_provider = edge_usage_provider
        self.edge_usage_cache: dict[int, tuple[np.ndarray, str]] = {}
        self.edge_usage_status_text = ""
        super().__init__(
            config,
            steps=steps,
            edge_table=edge_table,
            edge_usage_values=np.zeros((1, int(edge_table.num_edges)), dtype=np.float32),
            value_max=float(max(1.0, initial_value_max)),
            window_title=window_title,
            group_data=group_data or {},
            show_groups=show_groups,
            edge_active_mask=edge_active_mask,
            edge_building_mask=edge_building_mask,
        )

    def values_for_row(self, row: int) -> tuple[np.ndarray, str]:
        row = int(row)
        cached = self.edge_usage_cache.get(row)
        if cached is not None:
            return cached
        step = int(self.steps[row])
        values, status_text = self.edge_usage_provider(row, step)
        values = np.asarray(values, dtype=np.float32)
        if int(values.size) != int(self.edge_table.num_edges):
            raise ValueError(
                f"edge_usage_provider returned {values.size} values, expected {self.edge_table.num_edges}"
            )
        payload = (values, str(status_text))
        self.edge_usage_cache[row] = payload
        return payload

    def update_step(self, row: int, *, sync_slider: bool = True):
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        values, status_text = self.values_for_row(row)
        self.edge_values[0, :] = values
        self.edge_usage_status_text = status_text
        super().update_step(row, sync_slider=sync_slider)
        self.step_label.setText(self.step_label.text() + f" | {self.edge_usage_status_text}")
