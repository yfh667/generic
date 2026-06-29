from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.topology_metrics.module import (
    build_group_pair_node_arrays,
    compute_group_pair_shortest_timeseries,
    plot_group_pair_shortest_timeseries,
    result_summary,
    write_group_pair_shortest_timeseries,
)

from .io_utils import finite_or_none, write_csv, write_json
from .shortest_hops import GroupPairSpec


def compute_shortest_delay_for_pair(
    *,
    edge_table: EdgeTable,
    total_nodes: int,
    group_data: Mapping,
    steps: Sequence[int] | np.ndarray,
    pair: GroupPairSpec,
    edge_delay_ms: np.ndarray | Sequence,
    edge_active_mask: np.ndarray | Sequence | None = None,
    workers: int = 1,
    chunk_size: int = 100,
):
    steps_array = np.asarray(steps, dtype=np.int64)
    active = (
        np.ones(int(edge_table.num_edges), dtype=bool)
        if edge_active_mask is None
        else np.asarray(edge_active_mask, dtype=bool)
    )
    group_nodes = build_group_pair_node_arrays(
        group_data=group_data,
        steps=steps_array,
        source_group_id=int(pair.source_group_id),
        target_group_id=int(pair.target_group_id),
    )
    return compute_group_pair_shortest_timeseries(
        edge_table=edge_table,
        total_nodes=int(total_nodes),
        steps=steps_array,
        edge_active_mask=active,
        edge_weights_ms=np.asarray(edge_delay_ms, dtype=np.float32),
        group_nodes=group_nodes,
        workers=int(workers),
        chunk_size=int(chunk_size),
        progress_every_chunks=10,
    )


def write_shortest_delay_outputs(
    *,
    out_dir: str | Path,
    edge_table: EdgeTable,
    steps: Sequence[int] | np.ndarray,
    pair_results: Mapping[GroupPairSpec, Any],
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    inputs_dir = out_dir / "_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, inputs_dir / "edges.csv")
    np.save(inputs_dir / "time_indices.npy", np.asarray(steps, dtype=np.int64))

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for pair, result in pair_results.items():
        pair_dir = out_dir / "pairs" / pair.key
        files = write_group_pair_shortest_timeseries(
            result,
            pair_dir,
            meta={"metric": "mean_shortest_delay_ms", "pair_key": pair.key, **dict(meta or {})},
        )
        delay_plot = plot_group_pair_shortest_timeseries(
            {pair.label: result},
            out_dir / "figures" / pair.key,
            metric="mean_shortest_delay_ms",
            ylabel="mean shortest propagation delay (ms)",
            title=f"{pair.label} mean shortest propagation delay",
            filename=f"{pair.key}_mean_shortest_delay_ms.png",
        )
        summary = result_summary(result)
        summary_rows.append(
            {
                "pair_key": pair.key,
                "pair_label": pair.label,
                "source_group_id": int(pair.source_group_id),
                "target_group_id": int(pair.target_group_id),
                "steps": int(summary["steps"]),
                "start": int(summary["start"]),
                "end": int(summary["end"]),
                "mean_shortest_delay_ms": finite_or_none(float(summary["mean_shortest_delay_ms"])),
                "min_mean_shortest_delay_ms": finite_or_none(float(summary["min_delay_ms"])),
                "max_mean_shortest_delay_ms": finite_or_none(float(summary["max_delay_ms"])),
                "timeseries_csv": files["timeseries"],
                "delay_plot": delay_plot,
            }
        )
        for idx, step in enumerate(result.steps):
            all_rows.append(
                {
                    "pair_key": pair.key,
                    "pair_label": pair.label,
                    "step": int(step),
                    "hour": float(int(step) / 3600.0),
                    "source_nodes": int(result.source_counts[idx]),
                    "target_nodes": int(result.target_counts[idx]),
                    "active_edges": int(result.active_edges[idx]),
                    "delay_reachable_pairs": int(result.delay_reachable_pairs[idx]),
                    "mean_shortest_delay_ms": finite_or_none(float(result.mean_shortest_delay_ms[idx])),
                    "min_shortest_delay_ms": finite_or_none(float(result.min_shortest_delay_ms[idx])),
                    "max_shortest_delay_ms": finite_or_none(float(result.max_shortest_delay_ms[idx])),
                }
            )

    files = {
        "all_pairs_timeseries": write_csv(out_dir / "all_pairs_delay_timeseries.csv", all_rows),
        "pair_summary": write_csv(out_dir / "pair_delay_summary.csv", summary_rows),
        "run_meta": write_json(out_dir / "delay_run_meta.json", dict(meta or {})),
    }
    return files

