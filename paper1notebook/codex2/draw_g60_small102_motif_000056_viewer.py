from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_selected_motifs_shortest_delay_parallel import edge_table_for_motif
from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer


MOTIF_ID = 56
MOTIF_TEXT = "DBD | --B"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_small102_motif_000056_viewer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open G60 small102 motif_000056 with SatelliteTopology2DViewer.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width", type=float, default=0.032)
    parser.add_argument("--edge-alpha", type=int, default=180)
    parser.add_argument("--offscreen", action="store_true", help="Run headless and save a screenshot.")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = args.screenshot
    if bool(args.offscreen) and screenshot is None:
        screenshot = out_dir / "motif_000056_DBD_xxB_viewer_1s.png"

    edge_table = edge_table_for_motif(MOTIF_TEXT)
    write_edges_csv(edge_table, out_dir / "viewer_edges.csv")

    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=[int(args.step)],
        edge_table=edge_table,
        window_title=f"G60 small102 motif_{MOTIF_ID:06d}: {MOTIF_TEXT}",
        group_data={},
        show_groups=False,
        hide_y_wrap_edges=True,
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(50, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(190, int(args.edge_alpha)))))
    viewer.update_step(0)

    print(f"[draw-motif-000056] motif={MOTIF_TEXT}", flush=True)
    print(f"[draw-motif-000056] edges={edge_table.num_edges}", flush=True)
    print(f"[draw-motif-000056] out_dir={out_dir}", flush=True)
    print(f"[draw-motif-000056] mode={'offscreen screenshot' if args.offscreen else 'interactive viewer'}", flush=True)
    if screenshot is not None:
        print(f"[draw-motif-000056] screenshot={screenshot}", flush=True)
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=bool(args.check_only),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
