from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.topology_workflow.module.inter_edge_ports import (  # noqa: E402
    check_inter_edge_port_constraints,
    dataclass_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether an inter-edge set satisfies one-left/one-right port constraints. "
            "For G60-like topologies, each node may have at most one left neighbor and one right neighbor; "
            "plane 0 may not have a left neighbor, and the last plane may not have a right neighbor."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edges-csv", type=Path, help="CSV containing an edge set, e.g. union_edges.csv.")
    source.add_argument("--plan-dir", type=Path, help="Directory containing union_edges.csv, steps.npy, edge_active_mask.npy.")
    parser.add_argument("--step", type=int, default=None, help="When --plan-dir is used, check active edges at this step.")
    parser.add_argument("--start", type=int, default=None, help="When --plan-dir is used without --step, scan this start step.")
    parser.add_argument("--end", type=int, default=None, help="When --plan-dir is used without --step, scan this end step.")
    parser.add_argument("--stride", type=int, default=60, help="When scanning --plan-dir, check one row every stride seconds.")
    parser.add_argument("--p", type=int, default=int(G60_CONFIG.P))
    parser.add_argument("--n", type=int, default=int(G60_CONFIG.N))
    parser.add_argument(
        "--allowed-plane-deltas",
        type=int,
        nargs="*",
        default=[1, 2],
        help="Allowed inter-plane deltas. Empty means no delta restriction.",
    )
    parser.add_argument("--keep-duplicates", action="store_true", help="Count duplicate input edge rows instead of deduping.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fail-on-conflict", action="store_true")
    return parser.parse_args()


def parse_edge_key(text: str) -> tuple[int, int]:
    left, right = str(text).strip().split("-", 1)
    return int(left), int(right)


def read_edges_csv(path: Path) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        for row in reader:
            if {"src_node", "dst_node"}.issubset(fieldnames):
                edges.append((int(row["src_node"]), int(row["dst_node"])))
            elif {"src", "dst"}.issubset(fieldnames):
                edges.append((int(row["src"]), int(row["dst"])))
            elif {"u", "v"}.issubset(fieldnames):
                edges.append((int(row["u"]), int(row["v"])))
            elif {"source", "target"}.issubset(fieldnames):
                edges.append((int(row["source"]), int(row["target"])))
            elif "edge_key" in fieldnames:
                edges.append(parse_edge_key(row["edge_key"]))
            else:
                raise KeyError(
                    f"{path} must contain src_node/dst_node, src/dst, u/v, source/target, or edge_key columns"
                )
    return edges


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def row_for_step(steps: np.ndarray, step: int) -> int:
    pos = int(np.searchsorted(steps, int(step)))
    if pos >= int(steps.size) or int(steps[pos]) != int(step):
        raise KeyError(f"step={step} is not on this time axis")
    return pos


def selected_rows(steps: np.ndarray, *, start: int, end: int, stride: int) -> np.ndarray:
    mask = (steps >= int(start)) & (steps <= int(end)) & (((steps - int(start)) % int(stride)) == 0)
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError(f"no rows selected for start={start}, end={end}, stride={stride}")
    return rows.astype(np.int64)


def check_one(
    *,
    edges: list[tuple[int, int]],
    p: int,
    n: int,
    allowed_plane_deltas: tuple[int, ...] | None,
    deduplicate: bool,
):
    return check_inter_edge_port_constraints(
        edges,
        p=int(p),
        n=int(n),
        allowed_plane_deltas=allowed_plane_deltas,
        deduplicate=bool(deduplicate),
    )


def main() -> int:
    args = parse_args()
    allowed = tuple(int(x) for x in args.allowed_plane_deltas)
    allowed_plane_deltas = None if len(allowed) == 0 else allowed
    deduplicate = not bool(args.keep_duplicates)

    if args.edges_csv is not None:
        source_path = Path(args.edges_csv)
        out_dir = Path(args.out_dir) if args.out_dir is not None else source_path.parent / "inter_edge_port_check"
        result = check_one(
            edges=read_edges_csv(source_path),
            p=int(args.p),
            n=int(args.n),
            allowed_plane_deltas=allowed_plane_deltas,
            deduplicate=deduplicate,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(result.summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(out_dir / "violations.csv", dataclass_rows(result.violations))
        write_csv(out_dir / "invalid_edges.csv", dataclass_rows(result.invalid_edges))
        print(json.dumps(result.summary, ensure_ascii=False), flush=True)
        print(f"out_dir={out_dir}", flush=True)
        if bool(args.fail_on_conflict) and not result.ok:
            return 2
        return 0

    plan_dir = Path(args.plan_dir)
    all_edges = read_edges_csv(plan_dir / "union_edges.csv")
    steps = np.asarray(np.load(plan_dir / "steps.npy", mmap_mode="r"), dtype=np.int64)
    active_mask = np.load(plan_dir / "edge_active_mask.npy", mmap_mode="r")
    if active_mask.shape[1] != len(all_edges):
        raise ValueError(f"active mask edge count {active_mask.shape[1]} != union_edges rows {len(all_edges)}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else plan_dir / "inter_edge_port_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.step is not None:
        row = row_for_step(steps, int(args.step))
        active_edges = [edge for edge, enabled in zip(all_edges, np.asarray(active_mask[row], dtype=bool)) if bool(enabled)]
        result = check_one(
            edges=active_edges,
            p=int(args.p),
            n=int(args.n),
            allowed_plane_deltas=allowed_plane_deltas,
            deduplicate=deduplicate,
        )
        payload = dict(result.summary)
        payload["step"] = int(args.step)
        (out_dir / f"summary_t{int(args.step)}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_csv(out_dir / f"violations_t{int(args.step)}.csv", dataclass_rows(result.violations))
        write_csv(out_dir / f"invalid_edges_t{int(args.step)}.csv", dataclass_rows(result.invalid_edges))
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        print(f"out_dir={out_dir}", flush=True)
        if bool(args.fail_on_conflict) and not result.ok:
            return 2
        return 0

    start = int(steps[0]) if args.start is None else int(args.start)
    end = int(steps[-1]) if args.end is None else int(args.end)
    rows = selected_rows(steps, start=start, end=end, stride=int(args.stride))
    summary_rows: list[dict] = []
    first_bad: dict | None = None
    for row in rows:
        step = int(steps[int(row)])
        active_edges = [edge for edge, enabled in zip(all_edges, np.asarray(active_mask[int(row)], dtype=bool)) if bool(enabled)]
        result = check_one(
            edges=active_edges,
            p=int(args.p),
            n=int(args.n),
            allowed_plane_deltas=allowed_plane_deltas,
            deduplicate=deduplicate,
        )
        payload = dict(result.summary)
        payload["step"] = step
        summary_rows.append(payload)
        if first_bad is None and not result.ok:
            first_bad = payload
            write_csv(out_dir / f"violations_t{step}.csv", dataclass_rows(result.violations))
            write_csv(out_dir / f"invalid_edges_t{step}.csv", dataclass_rows(result.invalid_edges))

    write_csv(out_dir / "step_summary.csv", summary_rows)
    aggregate = {
        "ok": first_bad is None,
        "checked_steps": int(len(summary_rows)),
        "start": start,
        "end": end,
        "stride": int(args.stride),
        "first_bad": first_bad,
        "max_violation_nodes": int(max(row["violation_nodes"] for row in summary_rows)) if summary_rows else 0,
        "max_invalid_edges": int(max(row["invalid_edges"] for row in summary_rows)) if summary_rows else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, ensure_ascii=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)
    if bool(args.fail_on_conflict) and not aggregate["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
