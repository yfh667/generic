from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.query import open_delay_store_for_interval
from src.motif_generator.module.exact_box import EdgeRecord
from src.motif_generator.module.tiling import tile_edge_records_on_grid

from build_g60_motif_gridplus_shortest_delay_timeseries_parallel import (
    DEFAULT_XML,
    compute_chunk,
    group_nodes_for_step,
    init_worker,
    load_steps_and_rows,
    make_worker_spec,
    write_topology_outputs,
    write_worker_input_arrays,
)


DEFAULT_SELECTED_CSV = (
    PROJECT_ROOT / "data" / "linshi" / "motif_w4_h3_exact_box" / "exact_box_w4_h3_primitive_selected_100.csv"
)
DEFAULT_DELAY_STORE = PROJECT_ROOT / "data" / "linshi" / "G60_full_options_plus_intra_t0_86164_stride1"
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_w4h3_selected100_shortest_delay_t0_86164_stride60"
DEFAULT_GRIDPLUS_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_motif_gridplus_shortest_delay_parallel_t0_86164" / "gridplus"
DEFAULT_FULL_LINK_DIR = PROJECT_ROOT / "data" / "linshi" / "g60_full_link_shortest_delay_t0_86164" / "full_link"
INTRA_OPTION = -1
SYMBOL_DELTAS = {
    "A": (1, 0),
    "B": (1, -1),
    "C": (1, 1),
    "D": (2, 0),
}
OPTION_FROM_DELTA = {
    (1, 0): 0,
    (1, -1): 1,
    (2, 0): 2,
    (1, 1): 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shortest-delay simulation for selected exact-box motifs.")
    parser.add_argument("--selected-motif-csv", type=Path, default=DEFAULT_SELECTED_CSV)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--gridplus-dir", type=Path, default=DEFAULT_GRIDPLUS_DIR)
    parser.add_argument("--full-link-dir", type=Path, default=DEFAULT_FULL_LINK_DIR)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86164)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--source-group", type=int, default=2)
    parser.add_argument("--target-group", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=60)
    parser.add_argument("--sample-steps", type=int, default=0)
    parser.add_argument("--sample-pairs-per-step", type=int, default=0)
    parser.add_argument("--progress-every-futures", type=int, default=100)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def motif_text_to_edges(motif_text: str) -> tuple[int, int, list[EdgeRecord]]:
    columns = [part.strip() for part in str(motif_text).split("|")]
    if not columns:
        raise ValueError(f"empty motif text: {motif_text!r}")
    height = len(columns[0])
    if any(len(col) != height for col in columns):
        raise ValueError(f"inconsistent motif column heights: {motif_text!r}")

    edges: list[EdgeRecord] = []
    for col_idx, col in enumerate(columns):
        for row_idx, symbol in enumerate(col):
            if symbol == "-":
                continue
            if symbol not in SYMBOL_DELTAS:
                raise ValueError(f"unsupported motif symbol {symbol!r} in {motif_text!r}")
            dx, dy = SYMBOL_DELTAS[symbol]
            edges.append(
                EdgeRecord(
                    src_col=int(col_idx),
                    src_row=int(row_idx),
                    dst_col=int(col_idx + dx),
                    dst_row=int(row_idx + dy),
                    symbol=str(symbol),
                )
            )
    return len(columns) + 1, height, edges


def add_intra_ring_records(records: list[tuple[int, int, int, int, int]], *, p: int, n: int) -> None:
    for plane in range(int(p)):
        for y in range(int(n)):
            records.append((plane, y, plane, (y + 1) % int(n), INTRA_OPTION))


def make_edge_table_from_records(records: list[tuple[int, int, int, int, int]], *, p: int, n: int) -> EdgeTable:
    unique: dict[tuple[int, int], int] = {}
    for src_plane, src_y, dst_plane, dst_y, option in records:
        src = int(src_plane) * int(n) + int(src_y)
        dst = int(dst_plane) * int(n) + int(dst_y)
        if src == dst:
            continue
        key = (src, dst) if src < dst else (dst, src)
        unique.setdefault(key, int(option))

    src_values = []
    dst_values = []
    option_values = []
    for (src, dst), option in sorted(unique.items()):
        src_values.append(src)
        dst_values.append(dst)
        option_values.append(option)

    src_arr = np.asarray(src_values, dtype=np.int32)
    dst_arr = np.asarray(dst_values, dtype=np.int32)
    total_sats = int(p) * int(n)
    return EdgeTable(
        src=src_arr,
        dst=dst_arr,
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=(src_arr // int(n)).astype(np.int16),
        src_y=(src_arr % int(n)).astype(np.int16),
        dst_plane=(dst_arr // int(n)).astype(np.int16),
        dst_y=(dst_arr % int(n)).astype(np.int16),
        sat_ids=[str(i + 1) for i in range(total_sats)],
    )


def edge_table_for_motif(motif_text: str) -> EdgeTable:
    motif_width, motif_height, local_edges = motif_text_to_edges(motif_text)
    result = tile_edge_records_on_grid(
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
        motif_width=int(motif_width),
        motif_height=int(motif_height),
        local_edges=local_edges,
        horizontal_step=None,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
    )
    records: list[tuple[int, int, int, int, int]] = []
    for edge in result.placed_edges:
        dx = int(edge.dst_col) - int(edge.src_col)
        dy = int(edge.dst_row) - int(edge.src_row)
        option = OPTION_FROM_DELTA[(dx, dy)]
        records.append((int(edge.src_col), int(edge.src_row), int(edge.dst_col), int(edge.dst_row), int(option)))
    add_intra_ring_records(records, p=int(G60_CONFIG.P), n=int(G60_CONFIG.N))
    return make_edge_table_from_records(records, p=int(G60_CONFIG.P), n=int(G60_CONFIG.N))


def read_selected_motifs(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def safe_topology_name(motif_id: int) -> str:
    return f"motif_{int(motif_id):06d}"


def load_baseline_series(path: Path, steps: list[int]) -> np.ndarray:
    path = Path(path)
    values = np.load(path / "mean_shortest_delay_ms.npy")
    requested = np.asarray(steps, dtype=np.int64)

    for step_name in ("time_indices.npy", "steps.npy"):
        step_path = path / step_name
        if not step_path.exists():
            continue
        baseline_steps = np.load(step_path).astype(np.int64, copy=False)
        if baseline_steps.shape[0] != values.shape[0]:
            raise ValueError(
                f"{path} has {step_name} length {baseline_steps.shape[0]} "
                f"but mean_shortest_delay_ms.npy length {values.shape[0]}"
            )
        index_by_step = {int(step): idx for idx, step in enumerate(baseline_steps)}
        missing = [int(step) for step in requested if int(step) not in index_by_step]
        if missing:
            preview = ", ".join(str(step) for step in missing[:10])
            raise KeyError(f"{path} baseline is missing requested steps: {preview}")
        return np.asarray([values[index_by_step[int(step)]] for step in requested], dtype=np.float32)

    if requested.size and int(np.max(requested)) < values.shape[0]:
        return np.asarray(values[requested], dtype=np.float32)
    if values.shape[0] == requested.shape[0]:
        return np.asarray(values, dtype=np.float32)
    raise ValueError(
        f"{path} baseline cannot be aligned: values={values.shape[0]} "
        f"requested_steps={requested.shape[0]} max_step={int(np.max(requested)) if requested.size else None}"
    )


def write_selected_copy(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("selected motif rows are empty")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_combined_outputs(
    *,
    out_dir: Path,
    rows: list[dict[str, str]],
    steps: list[int],
    means_by_topology: dict[str, np.ndarray],
    gridplus: np.ndarray,
    full_link: np.ndarray,
    elapsed_s: float,
    args: argparse.Namespace,
) -> None:
    compare_path = out_dir / "compare_100motifs_gridplus_full_link.csv"
    fieldnames = ["step"] + [safe_topology_name(int(row["motif_id"])) for row in rows] + ["gridplus", "full_link"]
    with compare_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            item: dict[str, float | int] = {"step": int(step)}
            for row in rows:
                name = safe_topology_name(int(row["motif_id"]))
                item[name] = float(means_by_topology[name][idx])
            item["gridplus"] = float(gridplus[idx])
            item["full_link"] = float(full_link[idx])
            writer.writerow(item)

    summary_rows = []
    for row in rows:
        name = safe_topology_name(int(row["motif_id"]))
        values = means_by_topology[name]
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "motif_id": int(row["motif_id"]),
                "topology": name,
                "motif": row["motif"],
                "edges": row.get("edges", ""),
                "mean_delay_ms": float(np.mean(finite)),
                "min_delay_ms": float(np.min(finite)),
                "max_delay_ms": float(np.max(finite)),
                "output_dir": str(out_dir / name),
            }
        )
    summary_rows.sort(key=lambda item: item["mean_delay_ms"])
    with (out_dir / "motif_summary_sorted.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "motif_id",
                "topology",
                "motif",
                "edges",
                "mean_delay_ms",
                "min_delay_ms",
                "max_delay_ms",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(13, 5.2))
        x = np.asarray(steps, dtype=np.int64)
        for row in rows:
            name = safe_topology_name(int(row["motif_id"]))
            ax.plot(x, means_by_topology[name], color="#9aa0a6", alpha=0.25, linewidth=0.7)
        best_name = summary_rows[0]["topology"]
        ax.plot(x, means_by_topology[str(best_name)], color="#1f77b4", linewidth=1.3, label=f"best selected {best_name}")
        ax.plot(x, gridplus, color="#ff7f0e", linewidth=1.5, label="gridplus")
        ax.plot(x, full_link, color="#2ca02c", linewidth=1.5, label="full_link")
        ax.set_xlabel("time step (s)")
        ax.set_ylabel("China-Europe mean shortest delay (ms)")
        ax.grid(alpha=0.28, linestyle="--")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "mean_shortest_delay_100motifs_gridplus_full_link.png", dpi=180)
        plt.close(fig)
    except Exception as exc:
        print(f"[selected-motifs] plot skipped: {exc}", flush=True)

    meta = {
        "selected_motif_csv": str(Path(args.selected_motif_csv)),
        "selected_count": int(len(rows)),
        "selection_rule": "external selected CSV",
        "start": int(args.start),
        "end": int(args.end),
        "actual_start_step": int(steps[0]) if steps else None,
        "actual_end_step": int(steps[-1]) if steps else None,
        "stride": int(args.stride),
        "num_steps": int(len(steps)),
        "source_group": int(args.source_group),
        "target_group": int(args.target_group),
        "delay_store_dir": str(Path(args.delay_store_dir)),
        "gridplus_dir": str(Path(args.gridplus_dir)),
        "full_link_dir": str(Path(args.full_link_dir)),
        "elapsed_s": float(elapsed_s),
        "best_selected_motif": summary_rows[0],
        "gridplus_mean_delay_ms": float(np.mean(gridplus[np.isfinite(gridplus)])),
        "full_link_mean_delay_ms": float(np.mean(full_link[np.isfinite(full_link)])),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[selected-motifs] wrote {compare_path}")


def main() -> int:
    from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data

    args = parse_args()
    started_at = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = read_selected_motifs(Path(args.selected_motif_csv))
    write_selected_copy(selected_rows, out_dir / "selected_motifs_original_rows.csv")

    delay_store, steps, delay_rows = load_steps_and_rows(
        Path(args.delay_store_dir),
        int(args.start),
        int(args.end),
        int(args.stride),
    )
    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    source_nodes_by_index = [group_nodes_for_step(group_data, int(step), int(args.source_group)) for step in steps]
    target_nodes_by_index = [group_nodes_for_step(group_data, int(step), int(args.target_group)) for step in steps]
    worker_array_paths = write_worker_input_arrays(
        out_dir=out_dir,
        steps=steps,
        delay_rows=delay_rows,
        source_nodes_by_index=source_nodes_by_index,
        target_nodes_by_index=target_nodes_by_index,
    )

    edge_tables: dict[str, EdgeTable] = {}
    for row in selected_rows:
        name = safe_topology_name(int(row["motif_id"]))
        edge_tables[name] = edge_table_for_motif(str(row["motif"]))

    worker_specs = {name: make_worker_spec(edge_table, delay_store, name) for name, edge_table in edge_tables.items()}
    topology_names = list(edge_tables)
    args.max_workers_resolved = int(args.max_workers)
    args.chunk_size_resolved = int(args.chunk_size)

    tasks: list[tuple[str, int, int, int, int]] = []
    for name in topology_names:
        for start_idx in range(0, len(steps), int(args.chunk_size)):
            end_idx = min(len(steps), start_idx + int(args.chunk_size))
            tasks.append((name, start_idx, end_idx, int(args.sample_steps), int(args.sample_pairs_per_step)))

    print(
        f"[selected-motifs] motifs={len(topology_names)} steps={len(steps)} "
        f"range={steps[0]}..{steps[-1]} stride={args.stride} workers={args.max_workers} tasks={len(tasks)}",
        flush=True,
    )

    summaries_by_topology: dict[str, list[dict[str, Any] | None]] = {name: [None] * len(steps) for name in topology_names}
    means_by_topology: dict[str, np.ndarray] = {
        name: np.full(len(steps), np.nan, dtype=np.float32) for name in topology_names
    }
    samples_by_topology: dict[str, list[dict[str, Any]]] = {name: [] for name in topology_names}

    worker_init_kwargs = {
        "delay_store_dir": str(delay_store.store_dir),
        "topology_specs": worker_specs,
    }
    worker_init_kwargs.update(worker_array_paths)

    completed = 0
    with ProcessPoolExecutor(
        max_workers=int(args.max_workers),
        mp_context=get_context("spawn"),
        initializer=init_worker,
        initargs=(worker_init_kwargs,),
    ) as executor:
        futures = [executor.submit(compute_chunk, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            name = str(result["topology"])
            start_idx = int(result["start_idx"])
            end_idx = int(result["end_idx"])
            summaries_by_topology[name][start_idx:end_idx] = result["summaries"]
            means_by_topology[name][start_idx:end_idx] = np.asarray(result["means"], dtype=np.float32)
            samples_by_topology[name].extend(result["samples"])
            completed += 1
            if int(args.progress_every_futures) > 0 and (
                completed == len(tasks) or completed % int(args.progress_every_futures) == 0
            ):
                print(
                    f"[selected-motifs] completed {completed}/{len(tasks)} futures "
                    f"elapsed={time.time() - started_at:.1f}s",
                    flush=True,
                )

    elapsed_s = time.time() - started_at
    for name in topology_names:
        summaries = summaries_by_topology[name]
        if any(row is None for row in summaries):
            missing = [idx for idx, row in enumerate(summaries) if row is None][:10]
            raise RuntimeError(f"{name} has missing chunk results at {missing}")
        write_topology_outputs(
            out_dir=out_dir,
            topology_name=name,
            edge_table=edge_tables[name],
            steps=steps,
            summaries=[row for row in summaries if row is not None],
            samples=samples_by_topology[name],
            means=means_by_topology[name],
            worker_spec=worker_specs[name],
            args=args,
            elapsed_s=elapsed_s,
        )

    gridplus = load_baseline_series(Path(args.gridplus_dir), steps)
    full_link = load_baseline_series(Path(args.full_link_dir), steps)
    write_combined_outputs(
        out_dir=out_dir,
        rows=selected_rows,
        steps=steps,
        means_by_topology=means_by_topology,
        gridplus=gridplus,
        full_link=full_link,
        elapsed_s=elapsed_s,
        args=args,
    )
    print(f"[selected-motifs] done elapsed={elapsed_s:.1f}s out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
