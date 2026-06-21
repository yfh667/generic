from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import plan_motif0056_to_0061_link_setup as one_way  # noqa: E402
from compare_setup_plan_vs_static_metrics import GROUP_CACHE_DIR, GROUP_XML  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import open_position_cache_for_interval  # noqa: E402
from src.link_delay.module.query import open_delay_store_for_interval  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module import (  # noqa: E402
    build_weighted_adjacency,
    dijkstra_targets_with_prev_edge,
    reconstruct_path_and_edges,
)
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


DEFAULT_DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_switch_056_061_056_china_europe_1s_delay"
)


@dataclass(frozen=True)
class StaticTopology:
    motif_id: int
    data: one_way.TopologyData


_WORKER_EDGE_TABLE = None
_WORKER_ADJACENCY = None
_WORKER_TRACKED_BY_EDGE = None
_WORKER_DELAY_STORE = None
_WORKER_POSITION_STORE = None
_WORKER_DELAY_ROWS_BY_STEP: dict[int, int] = {}
_WORKER_POSITION_ROWS_BY_STEP: dict[int, int] = {}
_WORKER_LOOKUP = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 1s sampled shortest-delay usage only for right-link edges changed by "
            "motif000056 <-> motif000061, then write a usage-driven switch plan."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--switch-step", type=int, default=36000)
    parser.add_argument("--return-switch-step", type=int, default=54000)
    parser.add_argument("--source-motif-id", type=int, default=56)
    parser.add_argument("--middle-motif-id", type=int, default=61)
    parser.add_argument("--setup-duration", type=int, default=600)
    parser.add_argument("--workers", type=int, default=max(1, min(8, (mp.cpu_count() or 2) - 1)))
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--progress-every-chunks", type=int, default=10)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def build_static_topology(motif_id: int) -> StaticTopology:
    rows = one_way.read_motif_rows(one_way.MOTIF_LIBRARY_CSV)
    spec = one_way.build_topology_spec(int(motif_id), rows[int(motif_id)])
    right_by_owner, left_by_right = one_way.build_right_links(spec.edge_table)
    data = one_way.TopologyData(
        spec=spec,
        values=np.empty((0, int(spec.edge_table.num_edges)), dtype=np.float32),
        right_by_owner=right_by_owner,
        left_by_right=left_by_right,
        edge_key_to_idx=one_way.build_edge_key_index(spec.edge_table),
    )
    return StaticTopology(motif_id=int(motif_id), data=data)


def changed_rows(source: one_way.TopologyData, target: one_way.TopologyData) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    owners = sorted(set(source.right_by_owner) | set(target.right_by_owner))
    for owner in owners:
        old = source.right_by_owner.get(owner)
        new = target.right_by_owner.get(owner)
        if (None if old is None else old.edge_key) == (None if new is None else new.edge_key):
            continue
        rows.append(
            {
                "owner": int(owner),
                "old": old,
                "new": new,
            }
        )
    return rows


def tracked_edge_keys_for_topology(
    topology: one_way.TopologyData,
    *,
    first_changes: list[dict[str, object]],
    second_changes: list[dict[str, object]],
    role: str,
) -> list[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for change in first_changes:
        link = change["old"] if role == "source_old_first" else change["new"]
        if link is not None and link.edge_key in topology.edge_key_to_idx:
            keys.add(link.edge_key)
    for change in second_changes:
        link = change["old"] if role == "source_old_second" else change["new"]
        if link is not None and link.edge_key in topology.edge_key_to_idx:
            keys.add(link.edge_key)
    return sorted(keys)


def write_tracked_edges(path: Path, topology: one_way.TopologyData, keys: Sequence[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tracked_col", "edge_idx", "edge_key", "src", "dst", "option"])
        for col, key in enumerate(keys):
            edge_idx = int(topology.edge_key_to_idx[key])
            writer.writerow(
                [
                    int(col),
                    edge_idx,
                    f"{int(key[0])}-{int(key[1])}",
                    int(topology.spec.edge_table.src[edge_idx]),
                    int(topology.spec.edge_table.dst[edge_idx]),
                    int(topology.spec.edge_table.option[edge_idx]),
                ]
            )


def _init_worker(
    *,
    edge_table,
    tracked_edge_indices: Sequence[int],
    delay_store_dir: str,
    position_cache_dir: str,
    start: int,
    end: int,
    stride: int,
) -> None:
    global _WORKER_EDGE_TABLE, _WORKER_ADJACENCY, _WORKER_TRACKED_BY_EDGE
    global _WORKER_DELAY_STORE, _WORKER_POSITION_STORE, _WORKER_DELAY_ROWS_BY_STEP
    global _WORKER_POSITION_ROWS_BY_STEP, _WORKER_LOOKUP

    _WORKER_EDGE_TABLE = edge_table
    _WORKER_ADJACENCY = build_weighted_adjacency(edge_table, int(G60_CONFIG.total_sats))
    tracked = np.full(int(edge_table.num_edges), -1, dtype=np.int32)
    for col, edge_idx in enumerate(tracked_edge_indices):
        tracked[int(edge_idx)] = int(col)
    _WORKER_TRACKED_BY_EDGE = tracked
    _WORKER_DELAY_STORE = open_delay_store_for_interval(
        int(start),
        int(end),
        stride=int(stride),
        store_dir=Path(delay_store_dir),
        constellation_name=G60_CONFIG.name,
    )
    _WORKER_POSITION_STORE = open_position_cache_for_interval(
        int(start),
        int(end),
        stride=int(stride),
        cache_dir=Path(position_cache_dir),
    )
    steps = list(range(int(start), int(end) + 1, int(stride)))
    delay_rows = _WORKER_DELAY_STORE.rows_for_interval(int(start), int(end), int(stride))
    position_rows = _WORKER_POSITION_STORE.rows_for_interval(int(start), int(end), int(stride))
    _WORKER_DELAY_ROWS_BY_STEP = {int(step): int(delay_rows[row]) for row, step in enumerate(steps)}
    _WORKER_POSITION_ROWS_BY_STEP = {int(step): int(position_rows[row]) for row, step in enumerate(steps)}
    _WORKER_LOOKUP = build_weight_lookup(
        edge_table,
        _WORKER_DELAY_STORE,
        config=G60_CONFIG,
        allow_intra_fallback=True,
    )


def _init_worker_from_args(
    edge_table,
    tracked_edge_indices: Sequence[int],
    delay_store_dir: str,
    position_cache_dir: str,
    start: int,
    end: int,
    stride: int,
) -> None:
    _init_worker(
        edge_table=edge_table,
        tracked_edge_indices=tracked_edge_indices,
        delay_store_dir=delay_store_dir,
        position_cache_dir=position_cache_dir,
        start=start,
        end=end,
        stride=stride,
    )


def _compute_chunk(task: tuple[list[int], list[tuple[int, ...]], list[tuple[int, ...]]]) -> tuple[int, np.ndarray]:
    steps, sources_by_step, targets_by_step = task
    assert _WORKER_EDGE_TABLE is not None
    assert _WORKER_ADJACENCY is not None
    assert _WORKER_TRACKED_BY_EDGE is not None
    counts = np.zeros((len(steps), int(np.max(_WORKER_TRACKED_BY_EDGE)) + 1), dtype=np.uint16)
    for row, step in enumerate(steps):
        weights = edge_weights_for_step(
            edge_table=_WORKER_EDGE_TABLE,
            lookup=_WORKER_LOOKUP,
            delay_store=_WORKER_DELAY_STORE,
            position_store=_WORKER_POSITION_STORE,
            delay_row=int(_WORKER_DELAY_ROWS_BY_STEP[int(step)]),
            position_row=int(_WORKER_POSITION_ROWS_BY_STEP[int(step)]),
        )
        targets = tuple(int(x) for x in targets_by_step[row])
        for source in sources_by_step[row]:
            dist, prev_node, prev_edge = dijkstra_targets_with_prev_edge(
                adjacency=_WORKER_ADJACENCY,
                weights=weights,
                source=int(source),
                targets=targets,
                total_nodes=int(G60_CONFIG.total_sats),
            )
            for target in targets:
                if int(target) == int(source) or not math.isfinite(float(dist[int(target)])):
                    continue
                _, edge_indices = reconstruct_path_and_edges(
                    source=int(source),
                    target=int(target),
                    prev_node=prev_node,
                    prev_edge=prev_edge,
                )
                for edge_idx in edge_indices:
                    col = int(_WORKER_TRACKED_BY_EDGE[int(edge_idx)])
                    if col >= 0 and counts[row, col] < np.iinfo(np.uint16).max:
                        counts[row, col] += 1
    return int(steps[0]), counts


def chunked(items: list[int], size: int) -> list[list[int]]:
    return [items[pos:pos + int(size)] for pos in range(0, len(items), int(size))]


def group_nodes_by_step(group_data, steps: Sequence[int], group_id: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []
    for step in steps:
        payload = group_data[int(step)]
        groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
        values = groups.get(str(int(group_id)), groups.get(int(group_id), []))
        out.append(tuple(sorted(int(x) for x in values)))
    return out


def compute_tracked_usage_store(
    *,
    topology: one_way.TopologyData,
    tracked_keys: list[tuple[int, int]],
    group_data,
    steps: list[int],
    out_dir: Path,
    delay_store_dir: Path,
    position_cache_dir: Path,
    workers: int,
    chunk_size: int,
    progress_every_chunks: int,
    force: bool,
) -> tuple[np.ndarray, dict[tuple[int, int], int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tracked_edges(out_dir / "tracked_edges.csv", topology, tracked_keys)
    np.save(out_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))
    counts_path = out_dir / "tracked_edge_usage_counts.npy"
    if counts_path.exists() and not bool(force):
        arr = np.load(counts_path, mmap_mode="r")
        if arr.shape == (len(steps), len(tracked_keys)):
            print(f"[usage-1s] reuse {counts_path}", flush=True)
            return arr, {key: idx for idx, key in enumerate(tracked_keys)}

    counts = np.lib.format.open_memmap(
        counts_path,
        mode="w+",
        dtype=np.uint16,
        shape=(len(steps), len(tracked_keys)),
    )
    tracked_edge_indices = [int(topology.edge_key_to_idx[key]) for key in tracked_keys]
    sources_by_step = group_nodes_by_step(group_data, steps, 2)
    targets_by_step = group_nodes_by_step(group_data, steps, 3)
    step_to_row = {int(step): row for row, step in enumerate(steps)}
    tasks = []
    for step_chunk in chunked(steps, int(chunk_size)):
        rows = [step_to_row[int(step)] for step in step_chunk]
        tasks.append(
            (
                step_chunk,
                [sources_by_step[row] for row in rows],
                [targets_by_step[row] for row in rows],
            )
        )

    started = time.perf_counter()
    done = 0
    if int(workers) <= 1:
        _init_worker(
            edge_table=topology.spec.edge_table,
            tracked_edge_indices=tracked_edge_indices,
            delay_store_dir=str(delay_store_dir),
            position_cache_dir=str(position_cache_dir),
            start=int(steps[0]),
            end=int(steps[-1]),
            stride=int(steps[1] - steps[0] if len(steps) > 1 else 1),
        )
        iterator = map(_compute_chunk, tasks)
    else:
        pool = mp.Pool(
            processes=int(workers),
            initializer=_init_worker_from_args,
            initargs=(
                topology.spec.edge_table,
                tracked_edge_indices,
                str(delay_store_dir),
                str(position_cache_dir),
                int(steps[0]),
                int(steps[-1]),
                int(steps[1] - steps[0] if len(steps) > 1 else 1),
            ),
        )
        iterator = pool.imap_unordered(_compute_chunk, tasks)

    try:
        for first_step, chunk_counts in iterator:
            first_row = step_to_row[int(first_step)]
            counts[first_row:first_row + chunk_counts.shape[0], :] = chunk_counts
            done += 1
            if done == len(tasks) or (int(progress_every_chunks) > 0 and done % int(progress_every_chunks) == 0):
                elapsed = time.perf_counter() - started
                print(
                    f"[usage-1s] {topology.spec.name} chunks {done}/{len(tasks)} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
    finally:
        if int(workers) > 1:
            pool.close()
            pool.join()
    counts.flush()
    return counts, {key: idx for idx, key in enumerate(tracked_keys)}


def first_work(steps: np.ndarray, counts: np.ndarray, col: int, start: int, end: int) -> tuple[int | None, int]:
    rows = np.flatnonzero((steps >= int(start)) & (steps < int(end)))
    if rows.size == 0:
        return None, 0
    used = np.flatnonzero(np.asarray(counts[rows, int(col)]) > 0)
    if used.size == 0:
        return None, int(np.max(counts[rows, int(col)])) if rows.size else 0
    row = int(rows[int(used[0])])
    return int(steps[row]), int(counts[row, int(col)])


def last_work_before(steps: np.ndarray, counts: np.ndarray, col: int, start: int, deadline: int) -> tuple[int | None, int]:
    rows = np.flatnonzero((steps >= int(start)) & (steps < int(deadline)))
    if rows.size == 0:
        return None, 0
    used = np.flatnonzero(np.asarray(counts[rows, int(col)]) > 0)
    if used.size == 0:
        return None, int(np.max(counts[rows, int(col)])) if rows.size else 0
    row = int(rows[int(used[-1])])
    return int(steps[row]), int(counts[row, int(col)])


def link_text(link) -> str:
    return "" if link is None else str(link.edge_key_text)


def symbol_text(link) -> str:
    return "" if link is None else str(link.symbol)


def plan_transition(
    *,
    label: str,
    source_counts: np.ndarray,
    source_key_to_col: dict[tuple[int, int], int],
    target_counts: np.ndarray,
    target_key_to_col: dict[tuple[int, int], int],
    changes: list[dict[str, object]],
    steps_np: np.ndarray,
    segment_start: int,
    target_segment_start: int,
    target_segment_end: int,
    setup_duration: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for change in changes:
        owner = int(change["owner"])
        old = change["old"]
        new = change["new"]
        if new is None:
            rows.append(
                {
                    "transition": label,
                    "owner": owner,
                    "owner_p": int(owner // G60_CONFIG.N),
                    "owner_y": int(owner % G60_CONFIG.N),
                    "old_edge": link_text(old),
                    "old_symbol": symbol_text(old),
                    "new_edge": "",
                    "new_symbol": "",
                    "target_first_work_step": "",
                    "target_first_work_hour": "",
                    "target_first_work_value": 0,
                    "old_last_work_step_before_deadline": "",
                    "old_last_work_hour_before_deadline": "",
                    "old_last_work_value": 0,
                    "plan_start": "",
                    "plan_end": "",
                    "feasible": True,
                    "required_by_working": False,
                    "reason": "target_has_no_right_link",
                }
            )
            continue

        new_col = target_key_to_col.get(new.edge_key)
        if new_col is None:
            raise KeyError(f"missing target tracked edge {new.edge_key_text}")
        first_step, first_value = first_work(
            steps_np,
            target_counts,
            int(new_col),
            int(target_segment_start),
            int(target_segment_end),
        )
        if first_step is None:
            rows.append(
                {
                    "transition": label,
                    "owner": owner,
                    "owner_p": int(owner // G60_CONFIG.N),
                    "owner_y": int(owner % G60_CONFIG.N),
                    "old_edge": link_text(old),
                    "old_symbol": symbol_text(old),
                    "new_edge": link_text(new),
                    "new_symbol": symbol_text(new),
                    "target_first_work_step": "",
                    "target_first_work_hour": "",
                    "target_first_work_value": int(first_value),
                    "old_last_work_step_before_deadline": "",
                    "old_last_work_hour_before_deadline": "",
                    "old_last_work_value": 0,
                    "plan_start": "",
                    "plan_end": "",
                    "feasible": True,
                    "required_by_working": False,
                    "reason": "target_edge_never_working_in_segment",
                }
            )
            continue

        old_last_step = None
        old_last_value = 0
        if old is not None:
            old_col = source_key_to_col.get(old.edge_key)
            if old_col is not None:
                old_last_step, old_last_value = last_work_before(
                    steps_np,
                    source_counts,
                    int(old_col),
                    int(segment_start),
                    int(first_step),
                )
        old_release_after = int(segment_start) if old_last_step is None else int(old_last_step) + 1
        latest_start = int(first_step) - int(setup_duration)
        feasible = latest_start >= old_release_after and latest_start >= int(steps_np[0])
        rows.append(
            {
                "transition": label,
                "owner": owner,
                "owner_p": int(owner // G60_CONFIG.N),
                "owner_y": int(owner % G60_CONFIG.N),
                "old_edge": link_text(old),
                "old_symbol": symbol_text(old),
                "new_edge": link_text(new),
                "new_symbol": symbol_text(new),
                "target_first_work_step": int(first_step),
                "target_first_work_hour": float(first_step) / 3600.0,
                "target_first_work_value": int(first_value),
                "old_last_work_step_before_deadline": "" if old_last_step is None else int(old_last_step),
                "old_last_work_hour_before_deadline": "" if old_last_step is None else float(old_last_step) / 3600.0,
                "old_last_work_value": int(old_last_value),
                "plan_start": int(latest_start),
                "plan_end": int(first_step),
                "feasible": bool(feasible),
                "required_by_working": True,
                "reason": "switch_before_target_first_work" if feasible else "conflict_old_edge_working_inside_setup_window",
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_plan(path: Path, rows: list[dict[str, object]], *, switch_step: int, return_switch_step: int) -> None:
    required = [row for row in rows if row.get("required_by_working")]
    starts = np.asarray([int(row["plan_start"]) for row in required], dtype=np.int64) if required else np.asarray([], dtype=np.int64)
    owners = np.asarray([int(row["owner"]) for row in required], dtype=np.int32) if required else np.asarray([], dtype=np.int32)
    feasible = np.asarray([bool(row["feasible"]) for row in required], dtype=bool) if required else np.asarray([], dtype=bool)
    fig, ax = plt.subplots(figsize=(14.0, 5.4), dpi=170)
    if required:
        ax.scatter(starts / 3600.0, owners, s=10, c=np.where(feasible, "#2563eb", "#dc2626"), alpha=0.68)
    ax.axvline(float(switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    ax.axvline(float(return_switch_step) / 3600.0, color="#111827", linestyle="--", linewidth=0.9)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("owner node")
    ax.set_title("1s delay-betweenness usage-driven switch deadlines")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if int(args.stride) != 1:
        raise ValueError("this direct 1s script expects --stride 1")
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT
        / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
        / f"switch{int(args.switch_step)}_{int(args.return_switch_step)}"
        / "setup600_delay"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    source = build_static_topology(int(args.source_motif_id)).data
    middle = build_static_topology(int(args.middle_motif_id)).data
    first_changes = changed_rows(source, middle)
    second_changes = changed_rows(middle, source)
    source_keys = tracked_edge_keys_for_topology(
        source,
        first_changes=first_changes,
        second_changes=second_changes,
        role="source_old_first",
    )
    source_keys = sorted(set(source_keys) | set(
        tracked_edge_keys_for_topology(source, first_changes=first_changes, second_changes=second_changes, role="target_new_second")
    ))
    middle_keys = tracked_edge_keys_for_topology(
        middle,
        first_changes=first_changes,
        second_changes=second_changes,
        role="target_new_first",
    )
    middle_keys = sorted(set(middle_keys) | set(
        tracked_edge_keys_for_topology(middle, first_changes=first_changes, second_changes=second_changes, role="source_old_second")
    ))

    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )

    usage_root = out_dir / "tracked_usage"
    source_counts, source_key_to_col = compute_tracked_usage_store(
        topology=source,
        tracked_keys=source_keys,
        group_data=group_data,
        steps=steps,
        out_dir=usage_root / source.spec.name,
        delay_store_dir=Path(args.delay_store_dir),
        position_cache_dir=Path(args.position_cache_dir),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        progress_every_chunks=int(args.progress_every_chunks),
        force=bool(args.force),
    )
    middle_counts, middle_key_to_col = compute_tracked_usage_store(
        topology=middle,
        tracked_keys=middle_keys,
        group_data=group_data,
        steps=steps,
        out_dir=usage_root / middle.spec.name,
        delay_store_dir=Path(args.delay_store_dir),
        position_cache_dir=Path(args.position_cache_dir),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
        progress_every_chunks=int(args.progress_every_chunks),
        force=bool(args.force),
    )

    steps_np = np.asarray(steps, dtype=np.int64)
    rows = plan_transition(
        label="switch_056_to_061",
        source_counts=source_counts,
        source_key_to_col=source_key_to_col,
        target_counts=middle_counts,
        target_key_to_col=middle_key_to_col,
        changes=first_changes,
        steps_np=steps_np,
        segment_start=int(args.start),
        target_segment_start=int(args.switch_step),
        target_segment_end=int(args.return_switch_step),
        setup_duration=int(args.setup_duration),
    ) + plan_transition(
        label="switch_061_to_056",
        source_counts=middle_counts,
        source_key_to_col=middle_key_to_col,
        target_counts=source_counts,
        target_key_to_col=source_key_to_col,
        changes=second_changes,
        steps_np=steps_np,
        segment_start=int(args.switch_step),
        target_segment_start=int(args.return_switch_step),
        target_segment_end=int(args.end) + 1,
        setup_duration=int(args.setup_duration),
    )

    write_rows(out_dir / "usage_driven_right_link_switch_plan.csv", rows)
    required = [row for row in rows if row.get("required_by_working")]
    conflicts = [row for row in required if not row.get("feasible")]
    write_rows(out_dir / "usage_driven_right_link_switch_plan_required.csv", required)
    write_rows(out_dir / "usage_driven_right_link_switch_conflicts.csv", conflicts)
    plot_plan(out_dir / "usage_driven_right_link_switch_plan.png", rows, switch_step=int(args.switch_step), return_switch_step=int(args.return_switch_step))
    summary = {
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "switch_step": int(args.switch_step),
        "return_switch_step": int(args.return_switch_step),
        "setup_duration": int(args.setup_duration),
        "working_mode": "delay_1s_direct",
        "total_changed_right_ports": int(len(rows)),
        "required_by_working": int(len(required)),
        "not_required_by_working": int(len(rows) - len(required)),
        "conflicts": int(len(conflicts)),
        "feasible_required": int(len(required) - len(conflicts)),
        "source_tracked_edges": int(len(source_keys)),
        "middle_tracked_edges": int(len(middle_keys)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"out_dir={out_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
