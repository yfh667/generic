from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np


GENERIC_ROOT = Path(r"E:\paper11\generic")
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from PyQt5 import QtWidgets  # noqa: E402

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import (  # noqa: E402
    Topology2DPanel,
    UnifiedControlTopology2DViewer,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module import build_motif_text_edge_table, build_single_motif_edge_table  # noqa: E402


GROUP_XML = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml"
)
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
FULL_GROUP_CACHE_START = 0
FULL_GROUP_CACHE_END = 86164
FULL_GROUP_CACHE_STRIDE = 1
DEFAULT_SCREENSHOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\custom_w3h3_CDC_vs_motif000056\topology_2d"
    r"\custom_w3h3_CDC_vs_motif000056_2d_t0_86163_stride1.png"
)


CUSTOM_MOTIF = {
    "w": 3,
    "h": 3,
    "support": [
        [0, 0, "C"],
        [0, 1, "D"],
        [1, 1, "C"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show the custom w3h3 CDC motif as a G60 2D topology.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86163)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--single", action="store_true", help="Only show the custom motif, not the 000056 comparison.")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def compact_static_no_value_viewer(viewer: SatelliteTopology2DViewer) -> SatelliteTopology2DViewer:
    """Keep a long time axis without duplicating static edge arrays per step."""

    edge_count = int(viewer.edge_table.num_edges)
    viewer.edge_values = np.zeros((1, edge_count), dtype=np.float32)
    viewer.edge_active_mask = np.ones((1, edge_count), dtype=bool)
    viewer.edge_building_mask = np.zeros((1, edge_count), dtype=bool)
    gc.collect()
    viewer.update_step(0)
    return viewer


def load_group_data_for_steps(*, steps: list[int], stride: int, force: bool) -> dict:
    start = int(steps[0])
    end = int(steps[-1])
    stride = int(stride)
    if (
        stride == FULL_GROUP_CACHE_STRIDE
        and start >= FULL_GROUP_CACHE_START
        and end <= FULL_GROUP_CACHE_END
    ):
        full_steps = list(range(FULL_GROUP_CACHE_START, FULL_GROUP_CACHE_END + 1, FULL_GROUP_CACHE_STRIDE))
        full_group_data = load_or_build_group_data(
            xml_file=GROUP_XML,
            group_cache_dir=GROUP_CACHE_DIR,
            steps=full_steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=FULL_GROUP_CACHE_STRIDE,
            enabled=True,
            force=bool(force),
        )
        wanted = set(int(step) for step in steps)
        return {int(step): data for step, data in full_group_data.items() if int(step) in wanted}

    return load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=stride,
        enabled=True,
        force=bool(force),
    )


def make_custom_viewer(steps: list[int], group_data: dict | None) -> SatelliteTopology2DViewer:
    edge_table = build_single_motif_edge_table(
        topology_raw={
            "kind": "single_motif",
            "motif": CUSTOM_MOTIF,
            "tiling": {
                "allow_vertical_overlap": True,
                "allow_clipped_right": True,
            },
            "add_intra_ring": True,
            "wrap_planes": False,
        },
        config=G60_CONFIG,
    )
    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        window_title="custom_w3h3_CDC",
        group_data=group_data or {},
        show_groups=bool(group_data),
        show_grid_lines=False,
        topology_edge_color="#000000",
        topology_edge_alpha=170,
        topology_edge_width=0.025,
        hide_y_wrap_edges=True,
    )
    viewer.edge_width = 0.034
    viewer.edge_alpha = 215
    viewer.node_radius = 0.125
    return compact_static_no_value_viewer(viewer)


def make_motif56_viewer(steps: list[int], group_data: dict | None) -> SatelliteTopology2DViewer:
    edge_table = build_motif_text_edge_table(
        motif_text="DBD | --B",
        config=G60_CONFIG,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
        wrap_planes=False,
        add_intra_ring=True,
    )
    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        window_title="motif_000056_DBD_xxB",
        group_data=group_data or {},
        show_groups=bool(group_data),
        show_grid_lines=False,
        topology_edge_color="#000000",
        topology_edge_alpha=170,
        topology_edge_width=0.025,
        hide_y_wrap_edges=True,
    )
    viewer.edge_width = 0.034
    viewer.edge_alpha = 215
    viewer.node_radius = 0.125
    return compact_static_no_value_viewer(viewer)


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    print(
        f"[custom-w3h3-cdc-2d] steps={len(steps)} start={steps[0]} end={steps[-1]} stride={args.stride}",
        flush=True,
    )
    group_data = None
    if not bool(args.no_groups):
        group_data = load_group_data_for_steps(
            steps=steps,
            stride=int(args.stride),
            force=bool(args.force_group_cache),
        )
        print(f"[custom-w3h3-cdc-2d] group_steps={len(group_data)}", flush=True)

    if args.single:
        viewer = make_custom_viewer(steps, group_data)
        title = f"G60 custom_w3h3_CDC 2D topology | {len(steps)} steps"
        width = 1200
    else:
        viewer = UnifiedControlTopology2DViewer(
            title=f"G60 2D topology: custom_w3h3_CDC vs motif_000056_DBD_xxB | {len(steps)} steps",
            panels=[
                Topology2DPanel("custom_w3h3_CDC: C(0,0), D(0,1), C(1,1)", make_custom_viewer(steps, group_data)),
                Topology2DPanel("motif_000056_DBD_xxB: DBD | --B", make_motif56_viewer(steps, group_data)),
            ],
            shared_value_scale=False,
        )
        title = f"G60 custom motif topology comparison | {len(steps)} steps"
        width = 1700
    viewer.setWindowTitle(title)
    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = DEFAULT_SCREENSHOT
    return run_viewer_widget(
        viewer,
        width=width,
        height=900,
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
