from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_paper1_motif_shortest_hops import load_yaml  # noqa: E402
from search_motif0040_0056_hybrid_region_grid import DEFAULT_CONFIG  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402
from src.topology_workflow.module.edge_tables import build_full_option_plus_intra_edge_table  # noqa: E402


DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\plus_grid_option0_fixed_topology_t0_86160_s60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed +grid/option-0 topology in LST-compatible format.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    return parser.parse_args()


def write_union_edges(path: Path, edge_table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["edge_idx", "src_node", "dst_node", "option", "src_plane", "src_y", "dst_plane", "dst_y"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx in range(edge_table.num_edges):
            writer.writerow(
                {
                    "edge_idx": int(idx),
                    "src_node": int(edge_table.src[idx]),
                    "dst_node": int(edge_table.dst[idx]),
                    "option": int(edge_table.option[idx]),
                    "src_plane": int(edge_table.src_plane[idx]),
                    "src_y": int(edge_table.src_y[idx]),
                    "dst_plane": int(edge_table.dst_plane[idx]),
                    "dst_y": int(edge_table.dst_y[idx]),
                }
            )


def write_step_stats(path: Path, *, steps: np.ndarray, active_edges: int) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = [
            "step",
            "target_active_edges",
            "active_edges",
            "building_edges",
            "active_dropped_by_building",
            "building_dropped_by_conflict",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps.tolist():
            writer.writerow(
                {
                    "step": int(step),
                    "target_active_edges": int(active_edges),
                    "active_edges": int(active_edges),
                    "building_edges": 0,
                    "active_dropped_by_building": 0,
                    "building_dropped_by_conflict": 0,
                }
            )


def write_dynamic_schedule(path: Path, *, steps: np.ndarray) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["step", "candidate_idx", "name", "new_edges_from_prev"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row, step in enumerate(steps.tolist()):
            writer.writerow(
                {
                    "step": int(step),
                    "candidate_idx": 0,
                    "name": "plus_grid_option0",
                    "new_edges_from_prev": 0 if row == 0 else 0,
                }
            )


def main() -> int:
    args = parse_args()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    out_dir = Path(args.out_dir)
    lst_dir = out_dir / "lst120_fixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    lst_dir.mkdir(parents=True, exist_ok=True)

    edge_table = build_full_option_plus_intra_edge_table(
        config=config,
        options=(0,),
        add_intra_ring=True,
        wrap_planes=False,
    )
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    active = np.ones((steps.size, int(edge_table.num_edges)), dtype=bool)
    building = np.zeros_like(active, dtype=bool)

    for target_dir in (out_dir, lst_dir):
        write_union_edges(target_dir / "union_edges.csv", edge_table)
        np.save(target_dir / "steps.npy", steps)
        np.save(target_dir / "edge_active_mask.npy", active)
    np.save(lst_dir / "target_edge_active_mask.npy", active)
    np.save(lst_dir / "edge_building_mask.npy", building)
    write_step_stats(lst_dir / "lst_step_stats.csv", steps=steps, active_edges=edge_table.num_edges)
    write_step_stats(out_dir / "step_topology_stats.csv", steps=steps, active_edges=edge_table.num_edges)
    write_dynamic_schedule(out_dir / "dynamic_schedule.csv", steps=steps)

    meta = {
        "topology": "plus_grid_option0_plus_intra_y_ring",
        "out_dir": str(out_dir),
        "lst_dir": str(lst_dir),
        "steps": int(steps.size),
        "step_start": int(steps[0]),
        "step_end": int(steps[-1]),
        "edge_count": int(edge_table.num_edges),
        "inter_edges_option0": int(np.count_nonzero(np.asarray(edge_table.option) == 0)),
        "intra_edges": int(np.count_nonzero(np.asarray(edge_table.option) == -1)),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (lst_dir / "lst_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
