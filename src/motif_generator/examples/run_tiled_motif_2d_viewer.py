from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.edge_options import write_edges_csv
from src.motif_generator.module.config_io import load_yaml_dict
from src.motif_generator.module.exact_box import motif_from_columns, pretty_motif
from src.motif_generator.module.support import (
    motif_support_from_dict,
    motif_support_label,
    motif_support_to_edge_records,
)
from src.motif_generator.module.tiling import (
    tile_edge_records_on_grid,
    tile_motif_on_grid,
    write_tiled_motif_outputs,
)
from src.motif_generator.module.viewer_adapter import make_viewer_config, tiled_result_to_edge_table
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer


DEFAULT_OUT_DIR = THIS_DIR / "outputs" / "tiled_motif_2d_viewer"
DEFAULT_P = 18
DEFAULT_N = 36
DEFAULT_STEP = 1
DEFAULT_WIDTH = 1400
DEFAULT_HEIGHT = 860
DEFAULT_EDGE_WIDTH = 0.045
DEFAULT_EDGE_ALPHA = 190


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile a motif and show it with SatelliteTopology2DViewer.")
    parser.add_argument("--config", type=Path, default=None, help="YAML file using motif.w/h/support format.")
    parser.add_argument("--p", type=int, default=None, help="Number of plane/x columns.")
    parser.add_argument("--n", type=int, default=None, help="Number of y rows.")
    parser.add_argument("--motif-columns", nargs="+", default=["DAD", "C--"])
    parser.add_argument("--horizontal-step", type=int, default=None)
    parser.add_argument("--no-vertical-overlap", action="store_true")
    parser.add_argument("--no-clipped-right", action="store_true")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--edge-width", type=float, default=None)
    parser.add_argument("--edge-alpha", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def value_from_args_config_default(value, raw: dict, key: str, default):
    if value is not None:
        return value
    return raw.get(key, default)


def main() -> int:
    args = parse_args()
    viewer_raw: dict = {}

    if args.config is not None:
        raw_config = load_yaml_dict(args.config)
        grid_raw = raw_config.get("grid", {})
        tiling_raw = raw_config.get("tiling", {})
        viewer_raw = raw_config.get("viewer", {})
        if not isinstance(grid_raw, dict) or not isinstance(tiling_raw, dict) or not isinstance(viewer_raw, dict):
            raise ValueError("YAML grid, tiling, and viewer sections must be mappings when present")

        motif_support = motif_support_from_dict(raw_config)
        p = int(value_from_args_config_default(args.p, grid_raw, "p", DEFAULT_P))
        n = int(value_from_args_config_default(args.n, grid_raw, "n", DEFAULT_N))
        horizontal_step = value_from_args_config_default(args.horizontal_step, tiling_raw, "horizontal_step", None)
        allow_vertical_overlap = (
            False
            if args.no_vertical_overlap
            else bool(tiling_raw.get("allow_vertical_overlap", True))
        )
        allow_clipped_right = (
            False
            if args.no_clipped_right
            else bool(tiling_raw.get("allow_clipped_right", True))
        )
        result = tile_edge_records_on_grid(
            p=p,
            n=n,
            motif_width=motif_support.w,
            motif_height=motif_support.h,
            local_edges=motif_support_to_edge_records(motif_support),
            horizontal_step=None if horizontal_step is None else int(horizontal_step),
            allow_vertical_overlap=allow_vertical_overlap,
            allow_clipped_right=allow_clipped_right,
        )
        motif_label = motif_support.name or motif_support_label(motif_support)
        config_label = str(args.config)
    else:
        motif = motif_from_columns(tuple(args.motif_columns))
        result = tile_motif_on_grid(
            p=int(args.p if args.p is not None else DEFAULT_P),
            n=int(args.n if args.n is not None else DEFAULT_N),
            motif=motif,
            horizontal_step=args.horizontal_step,
            allow_vertical_overlap=not bool(args.no_vertical_overlap),
            allow_clipped_right=not bool(args.no_clipped_right),
        )
        motif_label = pretty_motif(motif)
        config_label = None

    out_dir = Path(args.out_dir)
    write_tiled_motif_outputs(result, out_dir)
    edge_table = tiled_result_to_edge_table(result)
    write_edges_csv(edge_table, out_dir / "viewer_edges.csv")

    print(f"[motif-2d-viewer] motif={motif_label}", flush=True)
    if config_label is not None:
        print(f"[motif-2d-viewer] config={config_label}", flush=True)
    print(
        f"[motif-2d-viewer] p={result.p}, n={result.n}, "
        f"accepted={result.accepted_count}, rejected={result.rejected_count}, "
        f"edges={edge_table.num_edges}, out_dir={out_dir}",
        flush=True,
    )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    step = int(value_from_args_config_default(args.step, viewer_raw, "step", DEFAULT_STEP))
    width = int(value_from_args_config_default(args.width, viewer_raw, "width", DEFAULT_WIDTH))
    height = int(value_from_args_config_default(args.height, viewer_raw, "height", DEFAULT_HEIGHT))
    edge_width = float(value_from_args_config_default(args.edge_width, viewer_raw, "edge_width", DEFAULT_EDGE_WIDTH))
    edge_alpha = int(value_from_args_config_default(args.edge_alpha, viewer_raw, "edge_alpha", DEFAULT_EDGE_ALPHA))

    config = make_viewer_config(p=result.p, n=result.n, name=f"motif_{result.p}x{result.n}")
    viewer = SatelliteTopology2DViewer(
        config,
        steps=[step],
        edge_table=edge_table,
        window_title=f"motif topology {motif_label} at {step}s",
        group_data={},
        show_groups=False,
    )
    viewer.edge_width = edge_width
    viewer.edge_alpha = edge_alpha
    viewer.width_slider.setValue(int(max(8, min(50, round(edge_width * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(190, edge_alpha))))
    viewer.update_step(0)

    return run_viewer_widget(
        viewer,
        width=width,
        height=height,
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
