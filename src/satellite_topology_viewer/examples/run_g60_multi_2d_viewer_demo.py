from __future__ import annotations

import os
import sys
from pathlib import Path


GENERIC_ROOT = Path(__file__).resolve().parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

import numpy as np
from PyQt5 import QtWidgets

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import build_full_option_edges
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer, build_fake_edge_value_matrix
from src.satellite_topology_viewer.module.multi_viewer import Topology2DPanel, UnifiedControlTopology2DViewer
from src.satellite_topology_viewer.module.topology_edges import build_full_option_plus_intra_edges


def main() -> int:
    steps = list(range(0, 101))
    full_option_edges = build_full_option_edges(G60_CONFIG, options=(0, 1, 2, 4))
    plus_intra_edges = build_full_option_plus_intra_edges(G60_CONFIG, inter_options=(0, 1, 2, 4))
    full_option_values = build_fake_edge_value_matrix(full_option_edges, steps)
    plus_intra_values = build_fake_edge_value_matrix(plus_intra_edges, steps)
    plus_intra_values = np.asarray(plus_intra_values, dtype=np.float32)

    viewer_a = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=full_option_edges,
        edge_values=full_option_values,
        value_min=float(np.nanmin(full_option_values)),
        value_max=float(np.nanmax(full_option_values)),
        window_title="G60 full-option fake values",
        show_groups=False,
    )
    viewer_b = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=plus_intra_edges,
        edge_values=plus_intra_values,
        value_min=float(np.nanmin(plus_intra_values)),
        value_max=float(np.nanmax(plus_intra_values)),
        window_title="G60 full-option plus intra fake values",
        show_groups=False,
    )
    window = UnifiedControlTopology2DViewer(
        title="G60 multi 2D viewer demo",
        panels=[
            Topology2DPanel("full option edges", viewer_a),
            Topology2DPanel("full option + intra edges", viewer_b),
        ],
        shared_value_scale=True,
    )
    return run_viewer_widget(window, width=1600, height=900)


if __name__ == "__main__":
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    raise SystemExit(main())
