from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_learning.module.config_io import load_yaml_dict
from src.topology_learning.module.full_link_gap_selector import _viewer_config_from_raw, read_edge_weights_by_key
from src.topology_learning.module.link_setup_smoothing import (
    build_deadline_aware_setup_masks,
    build_staggered_setup_masks,
)
from src.topology_learning.module.lst_schedule_eval import (
    DynamicScheduleSeries,
    _edge_key,
    evaluate_dynamic_schedule_after_lst,
    load_schedule_series_from_by_step_csv,
)


def parse_pair(text: str) -> tuple[int, int, str]:
    parts = [part.strip() for part in str(text).split(":")]
    if len(parts) == 2:
        return int(parts[0]), int(parts[1]), f"group{parts[0]}_group{parts[1]}"
    if len(parts) == 3:
        return int(parts[0]), int(parts[1]), str(parts[2])
    raise argparse.ArgumentTypeError("pair must be source:target or source:target:key")


def edge_priority_from_usage(
    *,
    series: DynamicScheduleSeries,
    critical_edges_csv: Path | None,
    critical_weights_npy: Path | None,
    missing_edge_weight: float,
) -> np.ndarray:
    if critical_edges_csv is None and critical_weights_npy is None:
        return np.zeros(int(series.edge_table.num_edges), dtype=np.float32)
    if critical_edges_csv is None or critical_weights_npy is None:
        raise ValueError("critical priority requires both --critical-edges-csv and --critical-weights-npy")
    weights_by_key = read_edge_weights_by_key(
        edges_csv=critical_edges_csv,
        weights_npy=critical_weights_npy,
        normalize_weights="max",
        weight_power=1.0,
        weight_scale=1.0,
        base_new_edge_cost=0.0,
    )
    return np.asarray(
        [
            float(weights_by_key.get(_edge_key(int(src), int(dst)), float(missing_edge_weight)))
            for src, dst in zip(series.edge_table.src, series.edge_table.dst, strict=True)
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a by-step topology schedule with edge-level staggered "
            "link setup. New edges are spread over later rows; old inter-plane "
            "edges keep working until a ready replacement claims the same port."
        )
    )
    parser.add_argument("--by-step-csv", type=Path, required=True)
    parser.add_argument("--selector-dir", type=Path, required=True)
    parser.add_argument("--schedule-name", type=str, default=None)
    parser.add_argument("--setup-time", type=float, required=True)
    parser.add_argument(
        "--setup-policy",
        choices=("staggered", "deadline"),
        default="deadline",
        help=(
            "staggered starts spreading after a target change is observed; "
            "deadline schedules future setup commands just in time before each target-active run."
        ),
    )
    parser.add_argument(
        "--deadline-placement",
        choices=("balanced", "latest", "latest_safe"),
        default="balanced",
        help=(
            "Only used by --setup-policy deadline. balanced prefers low-load rows; "
            "latest keeps each setup command as close to its deadline as possible; "
            "latest_safe first finds a balanced feasible schedule, then shifts commands later."
        ),
    )
    parser.add_argument("--max-commands-per-step", type=int, required=True)
    parser.add_argument(
        "--spread-rows",
        type=int,
        required=True,
        help=(
            "For staggered: rows after a newly requested edge where setup may start. "
            "For deadline: rows before the latest feasible command row used as a smoothing window."
        ),
    )
    parser.add_argument("--delay-store-dir", type=Path, required=True)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--group-xml", type=Path, required=True)
    parser.add_argument("--group-cache-dir", type=Path, required=True)
    parser.add_argument("--pair", type=parse_pair, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--critical-edges-csv", type=Path, default=None)
    parser.add_argument("--critical-weights-npy", type=Path, default=None)
    parser.add_argument("--critical-missing-edge-weight", type=float, default=0.0)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source, target, key = args.pair
    schedule_name = str(args.schedule_name or f"{args.by_step_csv.parent.name}_{key}_staggered")

    raw_config = load_yaml_dict(Path(args.selector_dir) / "selector_config.yaml")
    config = _viewer_config_from_raw(raw_config)
    if not config.station_groups and config.name == "G60" and int(config.P) == 18 and int(config.N) == 36:
        from src.config.viewer_config import G60_CONFIG

        config = G60_CONFIG

    base_series = load_schedule_series_from_by_step_csv(
        by_step_csv=args.by_step_csv,
        selector_dir=args.selector_dir,
        setup_time_seconds=float(args.setup_time),
        setup_timing="reactive",
    )
    priority = edge_priority_from_usage(
        series=base_series,
        critical_edges_csv=args.critical_edges_csv,
        critical_weights_npy=args.critical_weights_npy,
        missing_edge_weight=float(args.critical_missing_edge_weight),
    )
    if str(args.setup_policy) == "deadline":
        smoothed = build_deadline_aware_setup_masks(
            steps=base_series.steps,
            target_mask=base_series.target_mask,
            edge_table=base_series.edge_table,
            setup_time_seconds=float(args.setup_time),
            max_commands_per_step=int(args.max_commands_per_step),
            schedule_window_rows=int(args.spread_rows),
            edge_priority=priority,
            warm_start=True,
            placement_mode=str(args.deadline_placement),
        )
    else:
        smoothed = build_staggered_setup_masks(
            steps=base_series.steps,
            target_mask=base_series.target_mask,
            edge_table=base_series.edge_table,
            setup_time_seconds=float(args.setup_time),
            max_commands_per_step=int(args.max_commands_per_step),
            spread_rows=int(args.spread_rows),
            edge_priority=priority,
            warm_start=True,
        )
    series = DynamicScheduleSeries(
        steps=base_series.steps,
        topology_names=base_series.topology_names,
        selected_action=base_series.selected_action,
        selected_topology_names=base_series.selected_topology_names,
        edge_table=base_series.edge_table,
        target_mask=base_series.target_mask,
        setup_command_mask=smoothed.setup_command_mask,
        active_mask=smoothed.active_mask,
        building_mask=smoothed.building_mask,
    )

    group_data = load_or_build_group_data(
        xml_file=Path(args.group_xml),
        group_cache_dir=Path(args.group_cache_dir),
        steps=[int(x) for x in series.steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(series.steps[1] - series.steps[0]) if series.steps.size > 1 else 1,
        enabled=True,
        force=bool(args.force_group_cache),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = evaluate_dynamic_schedule_after_lst(
        series=series,
        config=config,
        group_data=group_data,
        source_group_id=int(source),
        target_group_id=int(target),
        delay_store_dir=args.delay_store_dir,
        position_cache_dir=args.position_cache_dir,
        out_dir=out_dir,
        schedule_name=schedule_name,
        setup_time_seconds=float(args.setup_time),
        setup_mode=f"{args.setup_policy}_make_before_break",
        setup_timing=f"{args.setup_policy}_spread{int(args.spread_rows)}_cap{int(args.max_commands_per_step)}",
    )
    payload = {
        "out_dir": str(out_dir),
        "by_step_csv": str(args.by_step_csv),
        "setup_policy": str(args.setup_policy),
        "deadline_placement": str(args.deadline_placement),
        "max_commands_per_step": int(args.max_commands_per_step),
        "spread_rows": int(args.spread_rows),
        "critical_edges_csv": None if args.critical_edges_csv is None else str(args.critical_edges_csv),
        "critical_weights_npy": None if args.critical_weights_npy is None else str(args.critical_weights_npy),
        **meta,
    }
    (out_dir / "staggered_lst_eval_meta.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
