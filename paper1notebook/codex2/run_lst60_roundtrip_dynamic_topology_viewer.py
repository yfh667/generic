from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_lst60_roundtrip_setup_vs_static_0056_0061 as roundtrip_metrics  # noqa: E402
import compare_lst_dynamic_setup_vs_static_0056_0061 as one_way_metrics  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.app import run_viewer_widget  # noqa: E402
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402


FORWARD_LST_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_056_061_china_europe_1s_any_direct"
    r"\t0_86160_stride1\switch36000_54000\releaseguard1\lst060"
)
REVERSE_LST_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_061_056_china_europe_1s_any_direct"
    r"\t0_86160_stride1\switch54000_86160\releaseguard1\lst060"
)
DEFAULT_SCREENSHOT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\lst60_metric_compare_056_061_056\viewer_screenshots"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show G60 LST=60 dynamic topology 000056 -> 000061 -> 000056 in the 2D viewer."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--initial-step", type=int, default=35940)
    parser.add_argument("--forward-lst-dir", type=Path, default=FORWARD_LST_DIR)
    parser.add_argument("--reverse-lst-dir", type=Path, default=REVERSE_LST_DIR)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    return parser.parse_args()


def build_roundtrip_masks(
    *,
    steps: list[int],
    edge_table,
    motif56: one_way_metrics.TopologyLite,
    motif61: one_way_metrics.TopologyLite,
    forward_events: dict[int, tuple[int, int]],
    reverse_events: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    key_to_idx = one_way_metrics.edge_key_index(edge_table)
    intra_keys = one_way_metrics.all_intra_keys(edge_table)
    owners = sorted(set(motif56.right_by_owner) | set(motif61.right_by_owner))
    active_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)
    building_mask = np.zeros((len(steps), edge_table.num_edges), dtype=bool)

    for row_idx, step in enumerate(steps):
        active_keys = set(intra_keys)
        building_keys: set[tuple[int, int]] = set()
        for owner in owners:
            link56 = motif56.right_by_owner.get(owner)
            link61 = motif61.right_by_owner.get(owner)
            forward = forward_events.get(owner)

            if forward is None:
                # Forward-exempted owners never really left 000056.
                link = link56 or link61
                if link is not None:
                    active_keys.add(link.edge_key)
                continue

            f_start, f_end = forward
            if int(step) < int(f_start):
                if link56 is not None:
                    active_keys.add(link56.edge_key)
                continue
            if int(f_start) <= int(step) < int(f_end):
                if link61 is not None:
                    building_keys.add(link61.edge_key)
                continue

            reverse = reverse_events.get(owner)
            if reverse is None:
                if link61 is not None:
                    active_keys.add(link61.edge_key)
                continue

            r_start, r_end = reverse
            if int(step) < int(r_start):
                if link61 is not None:
                    active_keys.add(link61.edge_key)
            elif int(r_start) <= int(step) < int(r_end):
                if link56 is not None:
                    building_keys.add(link56.edge_key)
            else:
                if link56 is not None:
                    active_keys.add(link56.edge_key)

        for key in active_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                active_mask[row_idx, idx] = True
        for key in building_keys:
            idx = key_to_idx.get(key)
            if idx is not None:
                building_mask[row_idx, idx] = True
    return active_mask, building_mask


def main() -> int:
    args = parse_args()
    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time axis")
    if int(args.initial_step) not in set(steps):
        raise ValueError("--initial-step must lie on the requested time axis")

    motif56 = one_way_metrics.load_topology_lite(56)
    motif61 = one_way_metrics.load_topology_lite(61)
    edge_table = one_way_metrics.union_edge_table([motif56, motif61])
    forward_events = one_way_metrics.read_schedule_events(Path(args.forward_lst_dir) / "schedule_1s.csv")
    reverse_events = one_way_metrics.read_schedule_events(Path(args.reverse_lst_dir) / "schedule_1s.csv")
    active_mask, building_mask = build_roundtrip_masks(
        steps=steps,
        edge_table=edge_table,
        motif56=motif56,
        motif61=motif61,
        forward_events=forward_events,
        reverse_events=reverse_events,
    )

    group_data = {}
    if not bool(args.no_groups):
        raw_group_data = load_or_build_group_data(
            xml_file=one_way_metrics.GROUP_XML,
            group_cache_dir=one_way_metrics.GROUP_CACHE_DIR,
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(args.stride),
            enabled=True,
            force=bool(args.force_group_cache),
        )
        wanted = set(steps)
        group_data = {int(step): data for step, data in raw_group_data.items() if int(step) in wanted}

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        edge_active_mask=active_mask,
        edge_building_mask=building_mask,
        window_title="G60 dynamic topology LST=60: motif000056 -> motif000061 -> motif000056",
        group_data=group_data,
        show_groups=not bool(args.no_groups),
        topology_edge_color="#000000",
        topology_edge_alpha=178,
        topology_edge_width=0.030,
        building_edge_color="#2563EB",
        building_edge_alpha=238,
        building_edge_width=0.100,
        hide_y_wrap_edges=True,
        show_grid_lines=False,
    )
    viewer.node_radius = 0.125
    viewer.update_step(steps.index(int(args.initial_step)))

    screenshot = args.screenshot
    if screenshot is None and bool(args.offscreen):
        screenshot = DEFAULT_SCREENSHOT_DIR / f"lst60_roundtrip_dynamic_topology_t{int(args.initial_step)}.png"

    print(
        "[lst60-roundtrip-viewer] "
        f"steps={len(steps)} initial_step={int(args.initial_step)} edges={edge_table.num_edges} "
        f"forward_jobs={len(forward_events)} reverse_jobs={len(reverse_events)} "
        f"active_edges_at_initial={int(np.count_nonzero(active_mask[steps.index(int(args.initial_step))]))} "
        f"building_max={int(np.max(np.sum(building_mask, axis=1)))}",
        flush=True,
    )
    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        offscreen=bool(args.offscreen),
        screenshot=screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
