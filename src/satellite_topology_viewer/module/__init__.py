"""Parameter-transparent modules for extensible 2D satellite topology visualization."""

from .app import run_edge_delay_viewer, run_link_switch_viewer, run_viewer_widget
from .base_viewer import (
    SatelliteTopology2DViewer,
    build_fake_edge_value_matrix,
    color_from_value,
)
from .edge_delay_data import EdgeDelayViewerData, load_edge_delay_data_for_viewer
from .edge_delay_viewer import EdgeDelayTopologyViewer
from .edge_usage_viewer import EdgeUsageTopology2DViewer, LazyEdgeUsageTopology2DViewer
from .link_switch_viewer import (
    LinkSwitchTopologyViewer,
    LinkSwitchViewerData,
    NodeWorkingInspectorEdgeUsageViewer,
    RightLinkStateStore,
    build_right_link_state_store_from_link_switch_data,
    build_viewer_edge_payload,
    load_right_link_state_store,
    make_link_switch_viewer_data,
    row_indices_for_window,
)
from .multi_viewer import Topology2DPanel, UnifiedControlTopology2DViewer
from .region_groups import load_or_build_group_data
from .topology_edges import build_full_option_plus_intra_edges

__all__ = [
    "EdgeDelayTopologyViewer",
    "EdgeDelayViewerData",
    "EdgeUsageTopology2DViewer",
    "LazyEdgeUsageTopology2DViewer",
    "LinkSwitchTopologyViewer",
    "LinkSwitchViewerData",
    "NodeWorkingInspectorEdgeUsageViewer",
    "RightLinkStateStore",
    "SatelliteTopology2DViewer",
    "Topology2DPanel",
    "UnifiedControlTopology2DViewer",
    "build_fake_edge_value_matrix",
    "build_full_option_plus_intra_edges",
    "build_right_link_state_store_from_link_switch_data",
    "build_viewer_edge_payload",
    "color_from_value",
    "load_edge_delay_data_for_viewer",
    "load_right_link_state_store",
    "load_or_build_group_data",
    "make_link_switch_viewer_data",
    "row_indices_for_window",
    "run_edge_delay_viewer",
    "run_link_switch_viewer",
    "run_viewer_widget",
]
