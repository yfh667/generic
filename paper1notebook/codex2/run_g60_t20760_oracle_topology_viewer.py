from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer


DEFAULT_ORACLE_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_oracle_one_right_topology_t20760_china_europe"
DEFAULT_EDGE_CSV = DEFAULT_ORACLE_DIR / "oracle_one_right_with_intra_edges.csv"
DEFAULT_INTER_CSV = DEFAULT_ORACLE_DIR / "oracle_one_right_inter_edges.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_t20760_oracle_topology_viewer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the fixed 20760s oracle topology in SatelliteTopology2DViewer.")
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--inter-csv", type=Path, default=DEFAULT_INTER_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--step", type=int, default=20760)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--plain",
        action="store_true",
        help="Show only the topology edges, without red full-link path-use overlay.",
    )
    parser.add_argument("--offscreen", action="store_true", help="Run headless and save a screenshot.")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def normalized_pair(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def read_edge_table(path: Path) -> EdgeTable:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty edge csv: {path}")

    src = np.asarray([int(row["src_node"]) for row in rows], dtype=np.int32)
    dst = np.asarray([int(row["dst_node"]) for row in rows], dtype=np.int32)
    option = np.asarray([int(row["option"]) for row in rows], dtype=np.int16)
    sat_ids = [str(i + 1) for i in range(int(G60_CONFIG.total_sats))]
    for row in rows:
        src_node = int(row["src_node"])
        dst_node = int(row["dst_node"])
        if 0 <= src_node < len(sat_ids):
            sat_ids[src_node] = str(row.get("src_sat_id", sat_ids[src_node]))
        if 0 <= dst_node < len(sat_ids):
            sat_ids[dst_node] = str(row.get("dst_sat_id", sat_ids[dst_node]))

    n_count = int(G60_CONFIG.N)
    return EdgeTable(
        src=src,
        dst=dst,
        option=option,
        src_plane=(src // n_count).astype(np.int16),
        src_y=(src % n_count).astype(np.int16),
        dst_plane=(dst // n_count).astype(np.int16),
        dst_y=(dst % n_count).astype(np.int16),
        sat_ids=sat_ids,
    )


def read_full_path_use_counts(path: Path) -> dict[tuple[int, int], int]:
    if not Path(path).exists():
        return {}
    out: dict[tuple[int, int], int] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src = int(row["src"])
            dst = int(row["dst"])
            out[normalized_pair(src, dst)] = int(row.get("full_path_use_count", 0))
    return out


def edge_values_from_counts(edge_table: EdgeTable, counts: dict[tuple[int, int], int]) -> np.ndarray:
    values = np.zeros((1, int(edge_table.num_edges)), dtype=np.float32)
    for idx, (src, dst) in enumerate(zip(edge_table.src, edge_table.dst)):
        values[0, idx] = float(counts.get(normalized_pair(int(src), int(dst)), 0))
    return values


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshot = args.screenshot
    if args.offscreen and screenshot is None:
        screenshot = out_dir / "g60_t20760_oracle_topology_viewer.png"

    if bool(args.offscreen):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    edge_table = read_edge_table(Path(args.edge_csv))
    counts = read_full_path_use_counts(Path(args.inter_csv))
    edge_values = None if args.plain else edge_values_from_counts(edge_table, counts)

    if edge_values is None:
        viewer = SatelliteTopology2DViewer(
            G60_CONFIG,
            steps=[int(args.step)],
            edge_table=edge_table,
            window_title=f"G60 20760s oracle topology plain at {int(args.step)}s",
            group_data={},
            show_groups=False,
            hide_y_wrap_edges=True,
        )
        viewer.edge_width = 0.026
        viewer.edge_alpha = 175
        viewer.width_slider.setValue(26)
        viewer.alpha_slider.setValue(175)
        viewer.update_step(0)
    else:
        viewer = SatelliteTopology2DViewer(
            G60_CONFIG,
            steps=[int(args.step)],
            edge_table=edge_table,
            edge_values=edge_values,
            value_min=0.0,
            value_max=float(np.nanmax(edge_values)) if edge_values.size else 0.0,
            edge_value_label="full_link_path_use_count_at_20760s",
            scale_edge_width_by_value=True,
            value_width_min=0.008,
            value_width_max=0.08,
            value_color_mode="red_alpha",
            value_solid_color="#C1121F",
            value_alpha_min=35,
            value_alpha_max=235,
            zero_value_edges_visible=False,
            zero_value_threshold=0.0,
            show_topology_under_edge_values=True,
            topology_edge_color="#000000",
            topology_edge_alpha=150,
            topology_edge_width=0.014,
            hide_y_wrap_edges=True,
            window_title=f"G60 20760s oracle topology with full-link-use overlay",
            group_data={},
            show_groups=False,
        )

    inter_edges = int(np.sum(np.asarray(edge_table.option) != -1))
    intra_edges = int(np.sum(np.asarray(edge_table.option) == -1))
    nonzero_used = int(np.count_nonzero(edge_values > 0)) if edge_values is not None else 0
    print(f"[oracle-viewer] edge_csv={Path(args.edge_csv)}", flush=True)
    print(f"[oracle-viewer] inter_edges={inter_edges} intra_edges={intra_edges}", flush=True)
    if edge_values is not None:
        print(f"[oracle-viewer] full_link_used_oracle_edges={nonzero_used}", flush=True)
        print(f"[oracle-viewer] max_use_count={float(np.nanmax(edge_values)):.0f}", flush=True)
    print(f"[oracle-viewer] mode={'plain topology' if args.plain else 'red use-count overlay'}", flush=True)
    if screenshot is not None:
        print(f"[oracle-viewer] screenshot={screenshot}", flush=True)

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
