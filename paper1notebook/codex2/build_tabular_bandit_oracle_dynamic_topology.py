from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_m56_local_patch_hybrid_topology import build_hybrid_edge_table, degree_stats  # noqa: E402
from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config  # noqa: E402
from run_paper1_region_internal_grid_metrics import (  # noqa: E402
    _apply_constraint_context_fast,
    _build_pair_step_contexts,
)
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402
from src.topology_workflow.module.edge_tables import make_edge_table_from_records  # noqa: E402


DEFAULT_SCHEDULE = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_tabular_bandit_oracle_schedules"
    r"\row_mask_tabular_bandit_oracle_lambda0.50.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_tabular_bandit_oracle_dynamic_topology_lambda050"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a tabular row-mask oracle schedule for motif000040/motif000056 "
            "into viewer-ready union_edges.csv + edge_active_mask.npy, with the "
            "paper1 region-internal +grid constraint applied at every time step."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--schedule-csv", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-motif", type=int, default=56)
    parser.add_argument("--patch-motif", type=int, default=40)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--start", type=int, default=None, help="Optional inclusive step filter.")
    parser.add_argument("--end", type=int, default=None, help="Optional inclusive step filter.")
    parser.add_argument("--stride", type=int, default=None, help="Group-cache stride metadata; schedule rows are not resampled.")
    parser.add_argument("--patch-mode", choices=("c", "cb", "all"), default="all")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def edge_key(src: int, dst: int) -> tuple[int, int]:
    a, b = (int(src), int(dst)) if int(src) <= int(dst) else (int(dst), int(src))
    return a, b


def edge_record_from_table(edge_table: EdgeTable, idx: int) -> tuple[int, int, int, int, int]:
    return (
        int(edge_table.src_plane[idx]),
        int(edge_table.src_y[idx]),
        int(edge_table.dst_plane[idx]),
        int(edge_table.dst_y[idx]),
        int(edge_table.option[idx]),
    )


def edge_key_from_table(edge_table: EdgeTable, idx: int) -> tuple[int, int]:
    return edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))


def y_mask_from_schedule_row(row: dict[str, str], n: int) -> tuple[int, ...]:
    values: list[int] = []
    for y in range(int(n)):
        key = f"y{y:02d}"
        if key not in row:
            raise ValueError(f"schedule row is missing {key}")
        if int(float(row[key])) != 0:
            values.append(y)
    return tuple(values)


def read_schedule(path: Path, *, n: int, start: int | None, end: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            step = int(float(raw["step"]))
            if start is not None and step < int(start):
                continue
            if end is not None and step > int(end):
                continue
            rows.append(
                {
                    "step": step,
                    "policy": raw.get("policy", ""),
                    "action": raw.get("action", ""),
                    "action_index": int(float(raw.get("action_index", -1))),
                    "mask": raw.get("mask", ""),
                    "rows": y_mask_from_schedule_row(raw, int(n)),
                    "mean_hops": float(raw.get("mean_hops", "nan")),
                    "mean_delay_ms": float(raw.get("mean_delay_ms", "nan")),
                }
            )
    if not rows:
        raise ValueError(f"empty filtered schedule: {path}")
    return rows


def topology_specs_by_id(*, motif_csv: Path, config, name_prefix: str, wrap_planes: bool) -> dict[int, Any]:
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=name_prefix,
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes),
    )
    by_id = {int(spec.motif_id): spec for spec in specs if spec.motif_id is not None}
    return by_id


def active_indices_for_table(edge_table: EdgeTable, key_to_idx: dict[tuple[int, int], int]) -> list[int]:
    return [int(key_to_idx[edge_key_from_table(edge_table, idx)]) for idx in range(edge_table.num_edges)]


def build_union_from_records(records_by_key: dict[tuple[int, int], tuple[int, int, int, int, int]], *, p: int, n: int) -> EdgeTable:
    return make_edge_table_from_records(p=int(p), n=int(n), records=list(records_by_key.values()))


def validate_active_table(edge_table: EdgeTable, *, total_sats: int) -> dict[str, Any]:
    return degree_stats(edge_table, total_sats=int(total_sats))


def write_csv_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mask_token(rows: Iterable[int]) -> str:
    values = tuple(sorted(set(int(x) for x in rows)))
    return "{" + ",".join(f"{x:02d}" for x in values) + "}"


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    paths = raw.get("paths", {})
    motif_library = raw.get("motif_library", {})
    motif_csv = Path(str(motif_library.get("csv_name", "motif0040_0056.csv")))
    if not motif_csv.is_absolute():
        motif_csv = path_from(paths, "motif_library_dir") / motif_csv
    name_prefix = str(motif_library.get("name_prefix", "selected"))
    wrap_planes = bool(wrap_planes_from_config(raw))
    stride = int(args.stride if args.stride is not None else raw.get("time", {}).get("stride", 60))
    forced_option = int(raw.get("region_internal_constraint", {}).get("forced_option", 0))
    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]

    schedule_rows = read_schedule(
        Path(args.schedule_csv),
        n=int(config.N),
        start=args.start,
        end=args.end,
    )
    steps = [int(row["step"]) for row in schedule_rows]
    if len(set(steps)) != len(steps):
        raise ValueError("schedule contains duplicate steps after filtering")

    specs_by_id = topology_specs_by_id(
        motif_csv=motif_csv,
        config=config,
        name_prefix=name_prefix,
        wrap_planes=wrap_planes,
    )
    base_spec = specs_by_id[int(args.base_motif)]
    patch_spec = specs_by_id[int(args.patch_motif)]

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=True,
        force=bool(args.force_group_cache),
    )
    contexts = _build_pair_step_contexts(
        config=config,
        group_data=group_data,
        steps=steps,
        pair_specs=[pair_spec],
        forced_option=forced_option,
        wrap_planes=wrap_planes,
    )[str(pair_spec.key)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_table_cache: dict[tuple[int, ...], EdgeTable] = {}
    constrained_tables: list[EdgeTable] = []
    records_by_key: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    step_stats: list[dict[str, Any]] = []

    for row_idx, (sched, context) in enumerate(zip(schedule_rows, contexts)):
        rows = tuple(int(x) for x in sched["rows"])
        if rows not in raw_table_cache:
            raw_table, _added, _removed, _degree = build_hybrid_edge_table(
                base_spec=base_spec,
                patch_spec=patch_spec,
                p=int(config.P),
                n=int(config.N),
                band=rows,
                patch_mode=str(args.patch_mode),
            )
            raw_table_cache[rows] = raw_table
        constrained, forced_edges, dropped_edges = _apply_constraint_context_fast(
            base_edge_table=raw_table_cache[rows],
            context=context,
            total_nodes=int(config.total_sats),
            p=int(config.P),
            n=int(config.N),
            forced_option=forced_option,
            wrap_planes=wrap_planes,
        )
        constrained_tables.append(constrained)
        for edge_idx in range(constrained.num_edges):
            key = edge_key_from_table(constrained, edge_idx)
            records_by_key.setdefault(key, edge_record_from_table(constrained, edge_idx))
        degree = validate_active_table(constrained, total_sats=int(config.total_sats))
        step_stats.append(
            {
                "step": int(sched["step"]),
                "action": sched["action"],
                "action_index": int(sched["action_index"]),
                "mask": mask_token(rows),
                "row_count": len(rows),
                "active_edges": int(constrained.num_edges),
                "inter_edges": int(degree["inter_edges"]),
                "intra_edges": int(degree["intra_edges"]),
                "forced_internal_option_edges": int(forced_edges),
                "dropped_edges": int(dropped_edges),
                "max_out_degree": int(degree["max_out_degree"]),
                "max_in_degree": int(degree["max_in_degree"]),
                "out_degree_gt1_nodes": int(degree["out_degree_gt1_nodes"]),
                "in_degree_gt1_nodes": int(degree["in_degree_gt1_nodes"]),
                "schedule_mean_hops": float(sched["mean_hops"]),
                "schedule_mean_delay_ms": float(sched["mean_delay_ms"]),
            }
        )
        if (row_idx + 1) % int(args.progress_every) == 0 or row_idx + 1 == len(schedule_rows):
            print(
                f"[tabular-oracle-topology] constrained {row_idx + 1}/{len(schedule_rows)} "
                f"step={sched['step']} active_edges={constrained.num_edges}",
                flush=True,
            )

    union_edge_table = build_union_from_records(records_by_key, p=int(config.P), n=int(config.N))
    union_key_to_idx = {edge_key_from_table(union_edge_table, idx): int(idx) for idx in range(union_edge_table.num_edges)}
    active_mask = np.zeros((len(constrained_tables), int(union_edge_table.num_edges)), dtype=bool)
    for row_idx, table in enumerate(constrained_tables):
        active_mask[row_idx, active_indices_for_table(table, union_key_to_idx)] = True

    switch_rows: list[dict[str, Any]] = []
    previous: np.ndarray | None = None
    previous_action: str | None = None
    for row_idx, sched in enumerate(schedule_rows):
        current = active_mask[row_idx]
        if previous is None:
            added = int(np.count_nonzero(current))
            removed = 0
            changed = 0
            action_changed = 0
        else:
            added = int(np.count_nonzero(current & ~previous))
            removed = int(np.count_nonzero(previous & ~current))
            changed = int(added + removed)
            action_changed = int(str(sched["action"]) != str(previous_action))
        switch_rows.append(
            {
                "step": int(sched["step"]),
                "action": sched["action"],
                "action_index": int(sched["action_index"]),
                "action_changed": action_changed,
                "added_edges": added,
                "removed_edges": removed,
                "changed_edges": changed,
                "active_edges": int(np.count_nonzero(current)),
            }
        )
        previous = current.copy()
        previous_action = str(sched["action"])

    schedule_out = [
        {
            "step": int(row["step"]),
            "policy": row["policy"],
            "action": row["action"],
            "action_index": int(row["action_index"]),
            "mask": mask_token(row["rows"]),
            "row_count": len(row["rows"]),
            "mean_hops": float(row["mean_hops"]),
            "mean_delay_ms": float(row["mean_delay_ms"]),
            **{f"y{y:02d}": int(y in set(row["rows"])) for y in range(int(config.N))},
        }
        for row in schedule_rows
    ]

    write_edges_csv(union_edge_table, out_dir / "union_edges.csv")
    np.save(out_dir / "edge_active_mask.npy", active_mask)
    np.save(out_dir / "steps.npy", np.asarray(steps, dtype=np.int64))
    write_csv_rows(out_dir / "schedule_used.csv", schedule_out)
    write_csv_rows(out_dir / "dynamic_schedule.csv", schedule_out)
    write_csv_rows(out_dir / "step_topology_stats.csv", step_stats)
    write_csv_rows(out_dir / "switch_stats.csv", switch_rows)

    action_changes = int(sum(row["action_changed"] for row in switch_rows))
    nonzero_edge_switches = int(sum(1 for row in switch_rows[1:] if int(row["changed_edges"]) > 0))
    summary = {
        "config": str(Path(args.config)),
        "schedule_csv": str(Path(args.schedule_csv)),
        "out_dir": str(out_dir),
        "base_motif": int(args.base_motif),
        "patch_motif": int(args.patch_motif),
        "patch_mode": str(args.patch_mode),
        "target_pair": str(args.target_pair),
        "region_internal_forced_option": forced_option,
        "wrap_planes": wrap_planes,
        "steps": len(steps),
        "step_start": int(min(steps)),
        "step_end": int(max(steps)),
        "stride_metadata": stride,
        "unique_actions": int(len(set(str(row["action"]) for row in schedule_rows))),
        "unique_row_masks": int(len(raw_table_cache)),
        "union_edges": int(union_edge_table.num_edges),
        "active_edges_min": int(active_mask.sum(axis=1).min()),
        "active_edges_mean": float(active_mask.sum(axis=1).mean()),
        "active_edges_max": int(active_mask.sum(axis=1).max()),
        "action_changes": action_changes,
        "nonzero_edge_switch_steps": nonzero_edge_switches,
        "total_added_edges_after_step0": int(sum(int(row["added_edges"]) for row in switch_rows[1:])),
        "total_removed_edges_after_step0": int(sum(int(row["removed_edges"]) for row in switch_rows[1:])),
        "max_out_degree": int(max(row["max_out_degree"] for row in step_stats)),
        "max_in_degree": int(max(row["max_in_degree"] for row in step_stats)),
        "out_degree_gt1_steps": int(sum(1 for row in step_stats if int(row["out_degree_gt1_nodes"]) > 0)),
        "in_degree_gt1_steps": int(sum(1 for row in step_stats if int(row["in_degree_gt1_nodes"]) > 0)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "outputs": {
            "union_edges_csv": str(out_dir / "union_edges.csv"),
            "edge_active_mask_npy": str(out_dir / "edge_active_mask.npy"),
            "steps_npy": str(out_dir / "steps.npy"),
            "schedule_used_csv": str(out_dir / "schedule_used.csv"),
            "dynamic_schedule_csv": str(out_dir / "dynamic_schedule.csv"),
            "step_topology_stats_csv": str(out_dir / "step_topology_stats.csv"),
            "switch_stats_csv": str(out_dir / "switch_stats.csv"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
