from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.edge_usage_viewer import EdgeUsageTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.multi_viewer import Topology2DPanel, UnifiedControlTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.edge_betweenness_store import compute_edge_betweenness_store  # noqa: E402
from src.topology_metrics.module.stores import MetricStoreLayout  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import TopologySpec  # noqa: E402
from src.topology_workflow.module.edge_tables import build_motif_text_edge_table  # noqa: E402


MOTIF_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\dual_edge_usage_056_061_china_europe"
)

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3
DEFAULT_MOTIF_IDS = (56, 61)


@dataclass(frozen=True)
class PreparedPanel:
    spec: TopologySpec
    values: np.ndarray
    value_max: float
    cache_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show motif000056 and motif000061 side-by-side with China-Europe edge betweenness."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--initial-step", type=int, default=43020)
    parser.add_argument("--motifs", nargs="+", type=int, default=list(DEFAULT_MOTIF_IDS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--force-metric-cache", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    return parser.parse_args()


def read_motif_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["motif_id"])] = dict(row)
    return rows


def build_topology_spec(motif_id: int, row: dict[str, str]) -> TopologySpec:
    motif_text = str(row["motif"]).strip()
    edge_table = build_motif_text_edge_table(
        motif_text=motif_text,
        config=G60_CONFIG,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
        wrap_planes=False,
        add_intra_ring=True,
    )
    return TopologySpec(
        name=f"combined_motif_{int(motif_id):06d}",
        edge_table=edge_table,
        library="combined_motif",
        motif_id=int(motif_id),
        motif=motif_text,
        source_w=int(row.get("source_w") or 0) or None,
        source_h=int(row.get("source_h") or 0) or None,
        edge_count_local=int(row.get("edge_count") or 0) or None,
        support=str(row.get("support", "")),
        baseline=False,
        meta={k: v for k, v in row.items() if k != "motif"},
    )


def expanded_edge_usage_values(cache_dir: Path) -> np.ndarray:
    layout = MetricStoreLayout(cache_dir)
    unique_values = np.load(layout.unique_state_values_npy, mmap_mode="r")
    state_ids = np.load(layout.state_ids_npy, mmap_mode="r")
    return np.asarray(unique_values[np.asarray(state_ids, dtype=np.int64), :], dtype=np.float32)


def cache_dir_for(spec: TopologySpec, *, start: int, end: int, stride: int) -> Path:
    return OUT_DIR / "edge_betweenness_cache" / f"t{int(start)}_{int(end)}_stride{int(stride)}" / PAIR_KEY / spec.name


def prepare_panel(
    *,
    spec: TopologySpec,
    group_data: dict,
    steps: list[int],
    start: int,
    end: int,
    stride: int,
    workers: int,
    progress_every: int,
    force: bool,
) -> PreparedPanel:
    cache_dir = cache_dir_for(spec, start=start, end=end, stride=stride)
    compute_edge_betweenness_store(
        topology_name=spec.name,
        edge_table=spec.edge_table,
        total_nodes=G60_CONFIG.total_sats,
        group_data=group_data,
        steps=steps,
        source_group_id=SOURCE_GROUP_ID,
        target_group_id=TARGET_GROUP_ID,
        out_dir=cache_dir,
        workers=int(workers),
        progress_every=int(progress_every),
        force=bool(force),
        expand_full_matrix=False,
        sample_path_limit=0,
        extra_meta={
            "pair_key": PAIR_KEY,
            "pair_label": "China-Europe",
            "motif": spec.motif,
            "support": spec.support,
            "runner": str(THIS_FILE),
        },
    )
    values = expanded_edge_usage_values(cache_dir)
    return PreparedPanel(
        spec=spec,
        values=values,
        value_max=float(np.nanmax(values)) if values.size else 1.0,
        cache_dir=cache_dir,
    )


def motif_label(spec: TopologySpec) -> str:
    return f"motif {int(spec.motif_id or 0):06d} | {spec.motif}"


def main() -> int:
    args = parse_args()
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if int(args.initial_step) not in set(steps):
        raise ValueError(f"--initial-step {args.initial_step} is not on the requested time axis")

    motif_rows = read_motif_rows(MOTIF_LIBRARY_CSV)
    specs = []
    for motif_id in [int(x) for x in args.motifs]:
        if motif_id not in motif_rows:
            raise KeyError(f"motif id not found in library: {motif_id}")
        specs.append(build_topology_spec(motif_id, motif_rows[motif_id]))

    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    wanted_steps = set(int(step) for step in steps)
    group_data = {int(step): data for step, data in group_data.items() if int(step) in wanted_steps}

    panels = [
        prepare_panel(
            spec=spec,
            group_data=group_data,
            steps=steps,
            start=int(args.start),
            end=int(args.end),
            stride=int(args.stride),
            workers=int(args.workers),
            progress_every=int(args.progress_every),
            force=bool(args.force_metric_cache),
        )
        for spec in specs
    ]
    shared_max = max((panel.value_max for panel in panels), default=1.0)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewers: list[EdgeUsageTopology2DViewer] = []
    for panel in panels:
        viewer = EdgeUsageTopology2DViewer(
            G60_CONFIG,
            steps=steps,
            edge_table=panel.spec.edge_table,
            edge_usage_values=panel.values,
            value_max=shared_max,
            window_title=f"G60 {motif_label(panel.spec)} China-Europe edge betweenness",
            group_data=group_data,
            show_groups=not bool(args.no_groups),
            topology_edge_color="#000000",
            topology_edge_alpha=165,
            topology_edge_width=0.018,
            value_width_min=0.010,
            value_width_max=0.105,
            value_alpha_min=34,
            value_alpha_max=245,
            hide_y_wrap_edges=True,
            show_grid_lines=False,
        )
        viewer.edge_width = 0.040
        viewer.edge_alpha = 210
        viewer.node_radius = 0.125
        viewer.update_step(steps.index(int(args.initial_step)))
        viewers.append(viewer)

    dual = UnifiedControlTopology2DViewer(
        title="G60 China-Europe edge betweenness: motif000056 vs motif000061",
        panels=[
            Topology2DPanel(title=motif_label(panel.spec), viewer=viewer)
            for panel, viewer in zip(panels, viewers)
        ],
        shared_value_scale=True,
        orientation=QtCore.Qt.Horizontal,
    )
    dual._sync_from_master(steps.index(int(args.initial_step)), refresh_master=True)

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = OUT_DIR / f"motif0056_0061_edge_betweenness_dual_t{int(args.initial_step)}.png"

    print(
        "[motif0056-0061-edge-usage-dual] "
        f"steps={len(steps)} start={steps[0]} end={steps[-1]} stride={args.stride} "
        f"initial_step={args.initial_step} shared_max={shared_max:.3f} out_dir={OUT_DIR}",
        flush=True,
    )
    for panel in panels:
        print(
            "[motif0056-0061-edge-usage-dual] "
            f"{panel.spec.name}: motif={panel.spec.motif} edges={panel.spec.edge_table.num_edges} "
            f"cache={panel.cache_dir}",
            flush=True,
        )

    return run_viewer_widget(
        dual,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
