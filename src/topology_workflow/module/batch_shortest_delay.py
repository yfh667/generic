from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval

from .batch_shortest_hops import RegionPairSpec, TopologySpec, auto_worker_count, finite_mean
from .config import group_name
from .shortest_delay import compute_shortest_delay_timeseries


_DELAY_WORKER_CONTEXT: dict[str, Any] = {}


def read_shortest_delay_series(result_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read ``time_indices.npy`` and ``mean_shortest_delay_ms.npy`` from one result folder."""

    result_dir = Path(result_dir)
    return (
        np.load(result_dir / "time_indices.npy").astype(np.int64, copy=False),
        np.load(result_dir / "mean_shortest_delay_ms.npy").astype(np.float32, copy=False),
    )


def _aligned_steps(series_by_name: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    if not series_by_name:
        raise ValueError("series_by_name is empty")
    first_name = next(iter(series_by_name))
    steps = np.asarray(series_by_name[first_name][0], dtype=np.int64)
    for name, (other_steps, values) in series_by_name.items():
        other_steps = np.asarray(other_steps, dtype=np.int64)
        if not np.array_equal(steps, other_steps):
            raise ValueError(f"time steps are not aligned for {name!r}")
        if np.asarray(values).shape[0] != steps.shape[0]:
            raise ValueError(f"value length does not match steps for {name!r}")
    return steps


def _topology_pair_delay_complete(topology_dir: Path, steps: Sequence[int]) -> bool:
    time_path = topology_dir / "time_indices.npy"
    delay_path = topology_dir / "mean_shortest_delay_ms.npy"
    summary_path = topology_dir / "step_summary.csv"
    if not (time_path.exists() and delay_path.exists() and summary_path.exists()):
        return False
    try:
        saved_steps = np.load(time_path, mmap_mode="r")
        if not np.array_equal(np.asarray(saved_steps, dtype=np.int64), np.asarray(steps, dtype=np.int64)):
            return False
        delay = np.load(delay_path, mmap_mode="r")
        return delay.shape == (len(steps),)
    except Exception:
        return False


def _load_topology_pair_delay(topology_dir: Path) -> np.ndarray:
    return np.asarray(np.load(topology_dir / "mean_shortest_delay_ms.npy"), dtype=np.float32)


def write_shortest_delay_comparison(
    *,
    out_dir: str | Path,
    series_by_name: Mapping[str, tuple[np.ndarray, np.ndarray]],
    metric_name: str = "mean_shortest_delay_ms",
    plot_title: str | None = None,
    plot_y_label: str = "mean shortest delay (ms)",
) -> dict[str, Any]:
    """Write a compare CSV, summary CSV, metadata, and a line plot for aligned delay series."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = _aligned_steps(series_by_name)
    names = list(series_by_name.keys())

    compare_path = out_dir / f"compare_{metric_name}.csv"
    with compare_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *names])
        writer.writeheader()
        for idx, step in enumerate(steps):
            row: dict[str, Any] = {"step": int(step)}
            for name in names:
                row[name] = float(series_by_name[name][1][idx])
            writer.writerow(row)

    summary_rows: list[dict[str, Any]] = []
    for name in names:
        values = np.asarray(series_by_name[name][1], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary_rows.append(
            {
                "topology": name,
                "mean": finite_mean(values),
                "min": float(np.min(finite)) if finite.size else None,
                "max": float(np.max(finite)) if finite.size else None,
                "finite_steps": int(finite.size),
            }
        )
    summary_path = out_dir / f"summary_{metric_name}.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["topology", "mean", "min", "max", "finite_steps"])
        writer.writeheader()
        writer.writerows(summary_rows)

    plot_path = out_dir / f"compare_{metric_name}.png"
    plot_shortest_delay_series(
        plot_path,
        series_by_name=series_by_name,
        title=plot_title or f"Comparison of {metric_name}",
        y_label=plot_y_label,
    )

    meta = {
        "metric_name": str(metric_name),
        "num_steps": int(steps.size),
        "topologies": names,
        "compare_csv": str(compare_path),
        "summary_csv": str(summary_path),
        "plot": str(plot_path),
    }
    (out_dir / f"meta_{metric_name}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def plot_shortest_delay_series(
    path: str | Path,
    *,
    series_by_name: Mapping[str, tuple[np.ndarray, np.ndarray]],
    title: str,
    y_label: str = "mean shortest delay (ms)",
) -> None:
    """Plot aligned shortest-delay series to a static PNG."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = _aligned_steps(series_by_name)
    x_hours = steps.astype(np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(15, 7), dpi=180)
    for name, (_steps, values) in series_by_name.items():
        ax.plot(x_hours, values, linewidth=1.0, label=str(name))
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(str(y_label))
    ax.set_title(str(title))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _init_delay_worker(payload: dict[str, Any]) -> None:
    delay_store = open_delay_store_for_interval(
        int(payload["start"]),
        int(payload["end"]),
        stride=int(payload["stride"]),
        store_dir=payload.get("delay_store_dir"),
        output_base=payload.get("delay_output_base"),
        constellation_name=payload["config"].name,
    )
    position_store = None
    position_rows = None
    if payload.get("position_cache_dir") is not None or payload.get("position_cache_root") is not None:
        position_store = open_position_cache_for_interval(
            int(payload["start"]),
            int(payload["end"]),
            stride=int(payload["stride"]),
            cache_dir=payload.get("position_cache_dir"),
            cache_root=payload.get("position_cache_root"),
        )
        position_rows = position_store.rows_for_interval(
            int(payload["start"]),
            int(payload["end"]),
            int(payload["stride"]),
        )

    _DELAY_WORKER_CONTEXT.clear()
    _DELAY_WORKER_CONTEXT.update(
        {
            **payload,
            "delay_store": delay_store,
            "delay_rows": delay_store.rows_for_interval(
                int(payload["start"]),
                int(payload["end"]),
                int(payload["stride"]),
            ),
            "position_store": position_store,
            "position_rows": position_rows,
        }
    )


def _compute_delay_topology_task(spec: TopologySpec) -> dict[str, Any]:
    if not _DELAY_WORKER_CONTEXT:
        raise RuntimeError("delay worker context is not initialized")
    ctx = _DELAY_WORKER_CONTEXT
    steps = [int(x) for x in ctx["steps"]]
    out_dir = Path(ctx["out_dir"])
    result_by_pair: dict[str, np.ndarray] = {}
    for pair in ctx["pair_specs"]:
        pair_dir = out_dir / str(pair.key)
        topology_dir = pair_dir / str(spec.name)
        if not bool(ctx.get("force", False)) and _topology_pair_delay_complete(topology_dir, steps):
            result_by_pair[str(pair.key)] = _load_topology_pair_delay(topology_dir)
            continue
        values = compute_shortest_delay_timeseries(
            topology_name=str(spec.name),
            edge_table=spec.edge_table,
            config=ctx["config"],
            group_data=ctx["group_data"],
            steps=steps,
            delay_rows=ctx["delay_rows"],
            position_rows=ctx["position_rows"],
            delay_store=ctx["delay_store"],
            position_store=ctx["position_store"],
            source_group_id=int(pair.source_group_id),
            target_group_id=int(pair.target_group_id),
            out_dir=topology_dir,
            engine=str(ctx.get("engine", "auto")),
            sample_steps=int(ctx.get("sample_steps", 0)),
            sample_pairs_per_step=int(ctx.get("sample_pairs_per_step", 0)),
            progress_every=int(ctx.get("progress_every", 0)),
        )
        result_by_pair[str(pair.key)] = np.asarray(values, dtype=np.float32)
    return {"topology": str(spec.name), "pairs": result_by_pair}


def compute_shortest_delay_batch(
    *,
    topology_specs: Sequence[TopologySpec],
    pair_specs: Sequence[RegionPairSpec],
    config: ViewerConfig,
    group_data: Mapping,
    start: int,
    end: int,
    stride: int,
    out_dir: str | Path,
    delay_store_dir: str | Path | None = None,
    delay_output_base: str | Path | None = None,
    position_cache_dir: str | Path | None = None,
    position_cache_root: str | Path | None = None,
    engine: str = "auto",
    sample_steps: int = 0,
    sample_pairs_per_step: int = 0,
    progress_every: int = 200,
    max_workers: int = 1,
    force: bool = False,
) -> dict[str, Any]:
    """Run shortest-delay time series for many topologies and region pairs.

    The topology, group data, stores, and time interval are all supplied by the
    caller.  This function only orchestrates repeated calls and writes aligned
    comparison outputs.
    """

    if not topology_specs:
        raise ValueError("topology_specs is empty")
    if not pair_specs:
        raise ValueError("pair_specs is empty")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    delay_store = open_delay_store_for_interval(
        int(start),
        int(end),
        stride=int(stride),
        store_dir=delay_store_dir,
        output_base=delay_output_base,
        constellation_name=config.name,
    )
    steps = [int(x) for x in np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)]
    delay_rows = delay_store.rows_for_interval(int(start), int(end), int(stride))

    position_store = None
    position_rows = None
    if position_cache_dir is not None or position_cache_root is not None:
        position_store = open_position_cache_for_interval(
            int(start),
            int(end),
            stride=int(stride),
            cache_dir=position_cache_dir,
            cache_root=position_cache_root,
        )
        position_rows = position_store.rows_for_interval(int(start), int(end), int(stride))

    pair_series: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        str(pair.key): {} for pair in pair_specs
    }
    workers = auto_worker_count(int(max_workers), max_cap=32)
    if workers <= 1:
        for pair in pair_specs:
            pair_dir = out_dir / str(pair.key)
            pair_dir.mkdir(parents=True, exist_ok=True)
            for spec in topology_specs:
                topology_dir = pair_dir / str(spec.name)
                if not bool(force) and _topology_pair_delay_complete(topology_dir, steps):
                    means = _load_topology_pair_delay(topology_dir)
                else:
                    means = compute_shortest_delay_timeseries(
                        topology_name=str(spec.name),
                        edge_table=spec.edge_table,
                        config=config,
                        group_data=dict(group_data),
                        steps=steps,
                        delay_rows=delay_rows,
                        position_rows=position_rows,
                        delay_store=delay_store,
                        position_store=position_store,
                        source_group_id=int(pair.source_group_id),
                        target_group_id=int(pair.target_group_id),
                        out_dir=topology_dir,
                        engine=engine,
                        sample_steps=int(sample_steps),
                        sample_pairs_per_step=int(sample_pairs_per_step),
                        progress_every=int(progress_every),
                    )
                pair_series[str(pair.key)][str(spec.name)] = (
                    np.asarray(steps, dtype=np.int64),
                    np.asarray(means, dtype=np.float32),
                )
    else:
        print(
            f"[topology-workflow] shortest_delay_batch parallel topologies={len(topology_specs)} "
            f"pairs={len(pair_specs)} steps={len(steps)} workers={workers}",
            flush=True,
        )
        payload = {
            "config": config,
            "group_data": dict(group_data),
            "steps": steps,
            "start": int(start),
            "end": int(end),
            "stride": int(stride),
            "out_dir": str(out_dir),
            "delay_store_dir": str(delay_store.store_dir),
            "delay_output_base": str(delay_output_base) if delay_output_base is not None else None,
            "position_cache_dir": str(position_store.cache_dir) if position_store is not None else None,
            "position_cache_root": str(position_cache_root) if position_cache_root is not None else None,
            "pair_specs": list(pair_specs),
            "engine": str(engine),
            "sample_steps": int(sample_steps),
            "sample_pairs_per_step": int(sample_pairs_per_step),
            "progress_every": int(progress_every),
            "force": bool(force),
        }
        completed = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_delay_worker, initargs=(payload,)) as executor:
            futures = [executor.submit(_compute_delay_topology_task, spec) for spec in topology_specs]
            for future in as_completed(futures):
                result = future.result()
                topology = str(result["topology"])
                for pair_key, values in result["pairs"].items():
                    pair_series[str(pair_key)][topology] = (
                        np.asarray(steps, dtype=np.int64),
                        np.asarray(values, dtype=np.float32),
                    )
                completed += 1
                if int(progress_every) > 0 and (completed == len(futures) or completed % int(progress_every) == 0):
                    print(
                        f"[topology-workflow] shortest_delay_batch completed "
                        f"{completed}/{len(futures)} last={topology}",
                        flush=True,
                    )

    all_pair_meta: dict[str, Any] = {}
    for pair in pair_specs:
        pair_dir = out_dir / str(pair.key)
        pair_dir.mkdir(parents=True, exist_ok=True)
        series_by_name = {
            str(spec.name): pair_series[str(pair.key)][str(spec.name)]
            for spec in topology_specs
        }
        pair_label = f"{group_name(config, pair.source_group_id)}-{group_name(config, pair.target_group_id)}"
        all_pair_meta[str(pair.key)] = write_shortest_delay_comparison(
            out_dir=pair_dir,
            series_by_name=series_by_name,
            metric_name="mean_shortest_delay_ms",
            plot_title=f"{config.name} {pair_label} shortest delay",
            plot_y_label=f"{pair_label} mean shortest delay (ms)",
        )

    topology_library_path = out_dir / "topology_library.csv"
    with topology_library_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "name",
            "library",
            "motif_id",
            "source_w",
            "source_h",
            "edge_count",
            "baseline",
            "motif",
            "support",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for spec in topology_specs:
            writer.writerow(
                {
                    "name": spec.name,
                    "library": spec.library,
                    "motif_id": spec.motif_id,
                    "source_w": spec.source_w,
                    "source_h": spec.source_h,
                    "edge_count": spec.edge_table.num_edges,
                    "baseline": bool(spec.baseline),
                    "motif": spec.motif,
                    "support": spec.support,
                }
            )

    meta = {
        "constellation": config.name,
        "start": int(start),
        "end": int(end),
        "stride": int(stride),
        "num_steps": len(steps),
        "num_topologies": len(topology_specs),
        "num_pairs": len(pair_specs),
        "delay_store_dir": str(delay_store.store_dir),
        "position_cache_dir": str(position_store.cache_dir) if position_store is not None else None,
        "topology_library_csv": str(topology_library_path),
        "pairs": all_pair_meta,
    }
    (out_dir / "batch_shortest_delay_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta
