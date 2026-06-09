from __future__ import annotations

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer


class EdgeDelayTopologyViewer(SatelliteTopology2DViewer):
    """Satellite topology viewer where edge values are propagation delay in milliseconds."""

    def __init__(
        self,
        config: ViewerConfig,
        *,
        delay_ms,
        delay_min_ms: float | None = None,
        delay_max_ms: float | None = None,
        **kwargs,
    ):
        super().__init__(
            config,
            edge_values=delay_ms,
            value_min=delay_min_ms,
            value_max=delay_max_ms,
            edge_value_label="delay_ms",
            **kwargs,
        )

    def format_edge_value(self, edge_idx: int, row: int, value: float) -> str:
        return f"delay={float(value):.4f} ms"

    def edge_extra_description(self, edge_idx: int, row: int, value: float) -> str:
        distance_km = float(value) / 1000.0 * LIGHT_SPEED_KM_S
        return f"distance={distance_km:.3f} km"
