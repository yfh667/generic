from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra, shortest_path  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import compare_setup_plan_vs_static_metrics as base  # noqa: E402
import plan_motif0056_to_0061_link_setup as motif_plan  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import load_position_cache  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.full_link_node_usage_viewer import read_edge_table_csv  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\static_z_chain_056_061_056_china_europe"
    r"\t0_86160_stride1\seq0056_0061_0056_tau36000_54000_lst060_latest"
)

SERIES = {
    "static_z_chain_56_61_56": ("static-z chain 56->61->56", "#d97706", 2.2, 1.0),
    "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.2, 0.86),
    "motif000061": ("motif000061 DCD | C--", "#64748b", 1.0, 0.64),
}

_WORKER: dict[str, object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute and plot 1s China-Europe metrics for current static-z chain.")
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    parser.add_argument("--chunk-size", type=int, default=1000)
    return parser.parse_args()


def selected_rows(steps_all: np.ndarray, *, start: int, end: int, stride: int) -> np.ndarray:
    rows = np.flatnonzero(
        (steps_all >= int(start))
        & (steps_all <= int(end))
        & (((steps_all - int(start)) % int(stride)) == 0)
    )
    if rows.size == 0:
        raise ValueError(f"no rows for start={start}, end={end}, stride={stride}")
    return rows


def edge_key(src: int, dst: int) -> tuple[int, int]:
    src = int(src)
    dst = int(dst)
    return (src, dst) if src < dst else (dst, src)


def edge_key_index(edge_table) -> dict[tuple[int, int], int]:
    return {
        edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(int(edge_table.num_edges))
    }


def static_motif_mask(*, motif_id: int, union_edge_table) -> np.ndarray:
    rows = motif_plan.read_motif_rows(motif_plan.MOTIF_LIBRARY_CSV)
    spec = motif_plan.build_topology_spec(int(motif_id), rows[int(motif_id)])
    key_to_idx = edge_key_index(union_edge_table)
    mask = np.zeros(int(union_edge_table.num_edges), dtype=bool)
    missing: list[tuple[int, int]] = []
    for idx in range(int(spec.edge_table.num_edges)):
        key = edge_key(int(spec.edge_table.src[idx]), int(spec.edge_table.dst[idx]))
        col = key_to_idx.get(key)
        if col is None:
            missing.append(key)
        else:
            mask[int(col)] = True
    if missing:
        raise ValueError(f"motif {motif_id} has {len(missing)} edges missing from union edge table")
    return mask


def metric_mean(values: np.ndarray) -> tuple[float, int, float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), 0, float("nan"), float("nan")
    return float(np.mean(finite)), int(finite.size), float(np.min(finite)), float(np.max(finite))


def compute_one_mask(
    *,
    mask: np.ndarray,
    weights: np.ndarray,
    src_all: np.ndarray,
    dst_all: np.ndarray,
    sources: list[int],
    targets: list[int],
    total_nodes: int,
) -> tuple[float, int, float, float, float, int, float, float, int]:
    active_src = src_all[mask]
    active_dst = dst_all[mask]
    if active_src.size == 0 or not sources or not targets:
        return (
            float("nan"),
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            0,
            float("nan"),
            float("nan"),
            0,
        )

    src_idx = np.asarray(sources, dtype=np.int32)
    dst_idx = np.asarray(targets, dtype=np.int32)

    hop_graph = csr_matrix(
        (
            np.ones(active_src.size * 2, dtype=np.float32),
            (np.concatenate([active_src, active_dst]), np.concatenate([active_dst, active_src])),
        ),
        shape=(int(total_nodes), int(total_nodes)),
    )
    hop_dist = shortest_path(hop_graph, directed=False, unweighted=True, indices=src_idx)
    hop_dist = np.atleast_2d(hop_dist)[:, dst_idx]
    hop_mean, hop_count, hop_min, hop_max = metric_mean(hop_dist)

    active_weights = np.asarray(weights[mask], dtype=np.float32)
    delay_graph = csr_matrix(
        (
            np.concatenate([active_weights, active_weights]),
            (np.concatenate([active_src, active_dst]), np.concatenate([active_dst, active_src])),
        ),
        shape=(int(total_nodes), int(total_nodes)),
    )
    delay_dist = dijkstra(delay_graph, directed=False, indices=src_idx)
    delay_dist = np.atleast_2d(delay_dist)[:, dst_idx]
    delay_mean, delay_count, delay_min, delay_max = metric_mean(delay_dist)
    return (
        hop_mean,
        hop_count,
        hop_min,
        hop_max,
        delay_mean,
        delay_count,
        delay_min,
        delay_max,
        int(active_src.size),
    )


def plot_metric(df: pd.DataFrame, out_path: Path, *, metric: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    x = df["hour"].to_numpy(dtype=np.float64)
    for name, (label, color, width, alpha) in SERIES.items():
        values = df[f"{name}_{metric}"].to_numpy(dtype=np.float64)
        ax.plot(x, values, color=color, linewidth=float(width), alpha=float(alpha), label=f"{label} | mean={np.nanmean(values):.3f}")
    for step in (36000, 54000):
        ax.axvline(float(step) / 3600.0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axvspan(10.0, 15.0, color="#0f766e", alpha=0.055, linewidth=0)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe 1s metrics: {ylabel}")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=8.6)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, (label, _color, _width, _alpha) in SERIES.items():
        rows.append(
            {
                "topology": label,
                "mean_hops": float(np.nanmean(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=np.float64))),
                "mean_shortest_delay_ms": float(
                    np.nanmean(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=np.float64))
                ),
                "min_hops": float(np.nanmin(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=np.float64))),
                "max_hops": float(np.nanmax(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=np.float64))),
                "min_delay_ms": float(np.nanmin(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=np.float64))),
                "max_delay_ms": float(np.nanmax(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=np.float64))),
                "steps": int(len(df)),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "summary_1s.csv", index=False, encoding="utf-8-sig")
    return out


def prepare_group_node_arrays(*, steps: list[int], out_dir: Path, stride: int) -> None:
    group_dir = out_dir / "group_nodes"
    group_dir.mkdir(parents=True, exist_ok=True)
    steps_path = group_dir / "steps.npy"
    source_nodes_path = group_dir / "source_nodes.npy"
    source_counts_path = group_dir / "source_counts.npy"
    target_nodes_path = group_dir / "target_nodes.npy"
    target_counts_path = group_dir / "target_counts.npy"

    if (
        steps_path.exists()
        and source_nodes_path.exists()
        and source_counts_path.exists()
        and target_nodes_path.exists()
        and target_counts_path.exists()
    ):
        cached_steps = np.load(steps_path, mmap_mode="r")
        if cached_steps.shape == (len(steps),) and bool(np.array_equal(cached_steps, np.asarray(steps, dtype=np.int64))):
            return

    cache_file = (
        base.GROUP_CACHE_DIR
        / f"station_visible_satellites_20250106_G60_t{int(steps[0])}_{int(steps[-1])}_stride{int(stride)}.json"
    )
    if not cache_file.exists():
        load_or_build_group_data(
            xml_file=base.GROUP_XML,
            group_cache_dir=base.GROUP_CACHE_DIR,
            steps=steps,
            station_groups=G60_CONFIG.station_groups,
            total_sats=G60_CONFIG.total_sats,
            constellation_name=G60_CONFIG.name,
            stride=int(stride),
            enabled=True,
            force=False,
        )
    with cache_file.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    groups_by_step = raw["groups_by_step"]

    source_lists = [
        [int(x) for x in groups_by_step[str(int(step))].get(str(base.SOURCE_GROUP_ID), [])]
        for step in steps
    ]
    target_lists = [
        [int(x) for x in groups_by_step[str(int(step))].get(str(base.TARGET_GROUP_ID), [])]
        for step in steps
    ]
    max_source = max((len(nodes) for nodes in source_lists), default=0)
    max_target = max((len(nodes) for nodes in target_lists), default=0)
    source_nodes = np.full((len(steps), max_source), -1, dtype=np.int32)
    target_nodes = np.full((len(steps), max_target), -1, dtype=np.int32)
    source_counts = np.zeros(len(steps), dtype=np.int16)
    target_counts = np.zeros(len(steps), dtype=np.int16)
    for idx, nodes in enumerate(source_lists):
        source_counts[idx] = int(len(nodes))
        if nodes:
            source_nodes[idx, : len(nodes)] = np.asarray(nodes, dtype=np.int32)
    for idx, nodes in enumerate(target_lists):
        target_counts[idx] = int(len(nodes))
        if nodes:
            target_nodes[idx, : len(nodes)] = np.asarray(nodes, dtype=np.int32)

    np.save(steps_path, np.asarray(steps, dtype=np.int64))
    np.save(source_nodes_path, source_nodes)
    np.save(source_counts_path, source_counts)
    np.save(target_nodes_path, target_nodes)
    np.save(target_counts_path, target_counts)


def _worker_init(plan_dir_str: str, out_dir_str: str, start: int, end: int, stride: int) -> None:
    plan_dir = Path(plan_dir_str)
    steps_all = np.asarray(np.load(plan_dir / "steps.npy", mmap_mode="r"), dtype=np.int64)
    rows = selected_rows(steps_all, start=int(start), end=int(end), stride=int(stride))
    steps = [int(x) for x in steps_all[rows]]
    edge_table = read_edge_table_csv(plan_dir / "union_edges.csv", total_nodes=int(G60_CONFIG.total_sats))
    current_active = np.load(plan_dir / "edge_active_mask.npy", mmap_mode="r")
    mask56 = static_motif_mask(motif_id=56, union_edge_table=edge_table)
    mask61 = static_motif_mask(motif_id=61, union_edge_table=edge_table)
    group_dir = Path(out_dir_str) / "group_nodes"
    source_nodes = np.load(group_dir / "source_nodes.npy", mmap_mode="r")
    source_counts = np.load(group_dir / "source_counts.npy", mmap_mode="r")
    target_nodes = np.load(group_dir / "target_nodes.npy", mmap_mode="r")
    target_counts = np.load(group_dir / "target_counts.npy", mmap_mode="r")

    delay_store = FullLinkDelayStore(base.DELAY_STORE_DIR)
    position_store = load_position_cache(base.POSITION_CACHE_DIR)
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    lookup = build_weight_lookup(edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)

    _WORKER.clear()
    _WORKER.update(
        {
            "plan_dir": plan_dir,
            "out_dir": Path(out_dir_str),
            "steps_all": steps_all,
            "rows": rows,
            "steps": steps,
            "edge_table": edge_table,
            "current_active": current_active,
            "mask56": mask56,
            "mask61": mask61,
            "source_nodes": source_nodes,
            "source_counts": source_counts,
            "target_nodes": target_nodes,
            "target_counts": target_counts,
            "delay_store": delay_store,
            "position_store": position_store,
            "delay_rows": delay_rows,
            "lookup": lookup,
            "src_all": np.asarray(edge_table.src, dtype=np.int32),
            "dst_all": np.asarray(edge_table.dst, dtype=np.int32),
            "total_nodes": int(G60_CONFIG.total_sats),
        }
    )


def _compute_chunk(task: tuple[int, int, int]) -> tuple[int, str, int, float]:
    chunk_id, start_pos, end_pos = task
    started = time.time()
    rows = _WORKER["rows"]  # type: ignore[assignment]
    steps = _WORKER["steps"]  # type: ignore[assignment]
    current_active = _WORKER["current_active"]  # type: ignore[assignment]
    source_nodes = _WORKER["source_nodes"]  # type: ignore[assignment]
    source_counts = _WORKER["source_counts"]  # type: ignore[assignment]
    target_nodes = _WORKER["target_nodes"]  # type: ignore[assignment]
    target_counts = _WORKER["target_counts"]  # type: ignore[assignment]
    delay_rows = _WORKER["delay_rows"]  # type: ignore[assignment]
    edge_table = _WORKER["edge_table"]
    src_all = _WORKER["src_all"]
    dst_all = _WORKER["dst_all"]
    total_nodes = int(_WORKER["total_nodes"])
    lookup = _WORKER["lookup"]
    delay_store = _WORKER["delay_store"]
    position_store = _WORKER["position_store"]
    mask56 = _WORKER["mask56"]
    mask61 = _WORKER["mask61"]
    out_dir = Path(_WORKER["out_dir"]) / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, float | int]] = []
    for local_idx in range(int(start_pos), int(end_pos)):
        step = int(steps[local_idx])
        step = int(step)
        current_mask = np.asarray(current_active[int(rows[local_idx]), :], dtype=bool)
        source_count = int(source_counts[local_idx])
        target_count = int(target_counts[local_idx])
        sources = [int(x) for x in np.asarray(source_nodes[local_idx, :source_count], dtype=np.int32)]
        targets = [int(x) for x in np.asarray(target_nodes[local_idx, :target_count], dtype=np.int32)]
        weights = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[local_idx]),
            position_row=int(delay_rows[local_idx]),
        )

        row: dict[str, float | int] = {
            "step": step,
            "hour": float(step) / 3600.0,
            "source_nodes": int(len(sources)),
            "target_nodes": int(len(targets)),
        }
        computed: list[tuple[np.ndarray, tuple[float, int, float, float, float, int, float, float, int]]] = []
        masks = {"static_z_chain_56_61_56": current_mask, "motif000056": mask56, "motif000061": mask61}
        for name, mask in masks.items():
            reused = None
            for prev_mask, prev_result in computed:
                if np.array_equal(mask, prev_mask):
                    reused = prev_result
                    break
            if reused is None:
                result = compute_one_mask(
                    mask=mask,
                    weights=weights,
                    src_all=src_all,
                    dst_all=dst_all,
                    sources=sources,
                    targets=targets,
                    total_nodes=total_nodes,
                )
                computed.append((mask.copy(), result))
            else:
                result = reused

            hop_mean, hop_count, hop_min, hop_max, delay_mean, delay_count, delay_min, delay_max, active_edges = result
            row[f"{name}_active_edges"] = int(active_edges)
            row[f"{name}_reachable_pairs_hops"] = int(hop_count)
            row[f"{name}_mean_shortest_hops"] = float(hop_mean)
            row[f"{name}_min_shortest_hops"] = float(hop_min)
            row[f"{name}_max_shortest_hops"] = float(hop_max)
            row[f"{name}_reachable_pairs_delay"] = int(delay_count)
            row[f"{name}_mean_shortest_delay_ms"] = float(delay_mean)
            row[f"{name}_min_shortest_delay_ms"] = float(delay_min)
            row[f"{name}_max_shortest_delay_ms"] = float(delay_max)
        output_rows.append(row)

    chunk_path = out_dir / f"chunk_{int(chunk_id):04d}.csv"
    pd.DataFrame(output_rows).to_csv(chunk_path, index=False, encoding="utf-8-sig")
    return int(chunk_id), str(chunk_path), int(end_pos - start_pos), float(time.time() - started)


def main() -> int:
    args = parse_args()
    plan_dir = Path(args.plan_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else plan_dir / f"metric_compare_t{args.start}_{args.end}_stride{args.stride}_1s_full"
    out_dir.mkdir(parents=True, exist_ok=True)

    steps_all = np.asarray(np.load(plan_dir / "steps.npy", mmap_mode="r"), dtype=np.int64)
    rows = selected_rows(steps_all, start=int(args.start), end=int(args.end), stride=int(args.stride))
    total = int(rows.size)
    chunk_size = max(1, int(args.chunk_size))
    tasks = [
        (chunk_id, start_pos, min(total, start_pos + chunk_size))
        for chunk_id, start_pos in enumerate(range(0, total, chunk_size))
    ]

    print(
        f"[1s-metrics] total_steps={total} chunks={len(tasks)} chunk_size={chunk_size} workers={int(args.workers)}",
        flush=True,
    )
    print("[1s-metrics] preparing compact group-node arrays", flush=True)
    prepare_group_node_arrays(
        steps=[int(x) for x in steps_all[rows]],
        out_dir=out_dir,
        stride=int(args.stride),
    )

    started = time.time()
    completed_rows = 0
    chunk_paths: dict[int, str] = {}
    with ProcessPoolExecutor(
        max_workers=max(1, int(args.workers)),
        initializer=_worker_init,
        initargs=(str(plan_dir), str(out_dir), int(args.start), int(args.end), int(args.stride)),
    ) as pool:
        futures = [pool.submit(_compute_chunk, task) for task in tasks]
        for future in as_completed(futures):
            chunk_id, path, count, elapsed = future.result()
            chunk_paths[int(chunk_id)] = str(path)
            completed_rows += int(count)
            if int(args.progress_every) > 0:
                print(
                    f"[1s-metrics] done {completed_rows}/{total} "
                    f"chunk={chunk_id} rows={count} chunk_elapsed={elapsed:.1f}s "
                    f"total_elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    frames = [pd.read_csv(chunk_paths[idx]) for idx in sorted(chunk_paths)]
    df = pd.concat(frames, ignore_index=True).sort_values("step").reset_index(drop=True)
    timeseries = out_dir / "timeseries_1s.csv"
    df.to_csv(timeseries, index=False, encoding="utf-8-sig")
    summary_df = summarize(df, out_dir)
    hops_plot = out_dir / "mean_shortest_hops_1s.png"
    delay_plot = out_dir / "mean_shortest_delay_ms_1s.png"
    plot_metric(df, hops_plot, metric="mean_shortest_hops", ylabel="mean shortest hops")
    plot_metric(df, delay_plot, metric="mean_shortest_delay_ms", ylabel="mean shortest delay (ms)")
    meta = {
        "plan_dir": str(plan_dir),
        "out_dir": str(out_dir),
        "start": int(args.start),
        "end": int(args.end),
        "stride": int(args.stride),
        "steps": int(total),
        "definition": "active edges only; building edges are not usable",
        "timeseries": str(timeseries),
        "summary": str(out_dir / "summary_1s.csv"),
        "hops_plot": str(hops_plot),
        "delay_plot": str(delay_plot),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)
    print(f"timeseries={timeseries}", flush=True)
    print(f"hops_plot={hops_plot}", flush=True)
    print(f"delay_plot={delay_plot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
