from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g60_paper2_config import (
    DATA_ROOT,
    DEFAULT_EPHEM_DIR,
    DEFAULT_GROUP_CACHE_DIR,
    DEFAULT_MAPPING_DIR,
    DEFAULT_POSITION_CACHE_ROOT,
    DEFAULT_XML,
    build_paper2_g60_config,
    write_station_group_mapping,
)
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.link_delay.module.position_cache import (
    completed_position_cache,
    default_cache_dir,
    ensure_position_cache,
    open_position_cache_for_interval,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.satellite_topology_viewer.module.topology_edges import build_full_option_plus_intra_edges
from src.topology_metrics.module import (
    build_group_pair_node_arrays,
    compute_group_pair_shortest_timeseries,
    plot_group_pair_shortest_timeseries,
    result_summary,
    write_group_pair_shortest_timeseries,
)


DEFAULT_OUT_BASE = DATA_ROOT / "cache" / "paper2_g60_shortest_metrics"
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures" / "paper2_g60_shortest_metrics"


def normalize_topology_name(value: str) -> str:
    value = str(value).strip().lower().replace("-", "_")
    aliases = {
        "+grid": "plus_grid",
        "grid+": "plus_grid",
        "grid_plus": "plus_grid",
        "plusgrid": "plus_grid",
        "fulloption": "full_option_plus_intra",
        "full_option": "full_option_plus_intra",
        "full_options": "full_option_plus_intra",
    }
    return aliases.get(value, value)


def topology_plot_label(topology: str) -> str:
    topology = normalize_topology_name(topology)
    if topology == "plus_grid":
        return "+grid"
    if topology == "full_option_plus_intra":
        return "full-option+intra"
    return topology


def group_key(config, group_id: int) -> str:
    info = config.station_groups[int(group_id)]
    raw = str(info.get("key") or info.get("name") or f"group_{group_id}")
    return raw.strip().lower().replace(" ", "_").replace("-", "_")


def group_label(config, group_id: int) -> str:
    info = config.station_groups[int(group_id)]
    return str(info.get("name") or info.get("key") or f"Group {group_id}")


def pair_key(config, source_group_id: int, target_group_id: int) -> str:
    return f"{group_key(config, int(source_group_id))}_{group_key(config, int(target_group_id))}"


def pair_label(config, source_group_id: int, target_group_id: int) -> str:
    return f"{group_label(config, int(source_group_id))}-{group_label(config, int(target_group_id))}"


def build_topology_edge_table(config, topology: str) -> EdgeTable:
    topology = normalize_topology_name(topology)
    if topology == "plus_grid":
        return build_full_option_plus_intra_edges(config, inter_options=(0,), include_intra=True)
    if topology == "full_option_plus_intra":
        return build_full_option_plus_intra_edges(config)
    raise ValueError(f"Unsupported topology={topology!r}; use plus_grid or full_option_plus_intra")


def json_read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def edge_table_signature(edge_table: EdgeTable) -> dict[str, Any]:
    return {
        "num_edges": int(edge_table.num_edges),
        "src_sum": int(np.sum(np.asarray(edge_table.src, dtype=np.int64))),
        "dst_sum": int(np.sum(np.asarray(edge_table.dst, dtype=np.int64))),
        "option_sum": int(np.sum(np.asarray(edge_table.option, dtype=np.int64))),
    }


def delay_weight_signature(
    *,
    position_cache_dir: Path,
    edge_table: EdgeTable,
    steps: np.ndarray,
) -> dict[str, Any]:
    positions_path = position_cache_dir / "positions_km.npy"
    times_path = position_cache_dir / "times_s.npy"
    return {
        "script_version": 1,
        "position_cache_dir": str(position_cache_dir.resolve()),
        "positions_size": int(positions_path.stat().st_size),
        "positions_mtime_ns": int(positions_path.stat().st_mtime_ns),
        "times_size": int(times_path.stat().st_size),
        "times_mtime_ns": int(times_path.stat().st_mtime_ns),
        "start": int(steps[0]),
        "end": int(steps[-1]),
        "stride": int(steps[1] - steps[0]) if len(steps) > 1 else 1,
        "num_steps": int(len(steps)),
        "edge_table": edge_table_signature(edge_table),
        "light_speed_km_s": float(LIGHT_SPEED_KM_S),
    }


def build_or_load_delay_weights(
    *,
    out_dir: Path,
    position_cache_dir: Path,
    edge_table: EdgeTable,
    steps: np.ndarray,
    stride: int,
    chunk_steps: int,
    force: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    delay_path = out_dir / "edge_delay_ms.npy"
    meta_path = out_dir / "edge_delay_meta.json"

    store = open_position_cache_for_interval(
        int(steps[0]),
        int(steps[-1]),
        stride=int(stride),
        cache_dir=position_cache_dir,
        cache_root=position_cache_dir.parent,
    )
    signature = delay_weight_signature(position_cache_dir=store.cache_dir, edge_table=edge_table, steps=steps)
    old_meta = json_read(meta_path)
    if (
        not force
        and delay_path.exists()
        and isinstance(old_meta, dict)
        and old_meta.get("signature") == signature
    ):
        print(f"[paper2-shortest] Reusing edge delay weights: {delay_path}", flush=True)
        return delay_path

    rows = store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    delay_ms = open_memmap(
        delay_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(len(steps)), int(edge_table.num_edges)),
    )
    delay_min = math.inf
    delay_max = -math.inf
    delay_sum = 0.0
    delay_count = 0
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)

    print(
        f"[paper2-shortest] Building edge delay weights: steps={len(steps)} edges={edge_table.num_edges}",
        flush=True,
    )
    for local_start in range(0, int(len(steps)), int(chunk_steps)):
        local_end = min(local_start + int(chunk_steps), int(len(steps)))
        chunk_rows = rows[local_start:local_end]
        pos_chunk = np.asarray(store.positions_km[chunk_rows], dtype=np.float32)
        diff = pos_chunk[:, src, :] - pos_chunk[:, dst, :]
        dist_km = np.sqrt(np.sum(diff * diff, axis=2), dtype=np.float32)
        chunk_delay = (dist_km / np.float32(LIGHT_SPEED_KM_S) * np.float32(1000.0)).astype(np.float32)
        delay_ms[local_start:local_end, :] = chunk_delay

        delay_min = min(delay_min, float(np.nanmin(chunk_delay)))
        delay_max = max(delay_max, float(np.nanmax(chunk_delay)))
        delay_sum += float(np.nansum(chunk_delay, dtype=np.float64))
        delay_count += int(np.isfinite(chunk_delay).sum())
        if local_end == int(len(steps)) or (local_start // int(chunk_steps)) % 10 == 0:
            print(f"[paper2-shortest] delay weights {local_end}/{len(steps)}", flush=True)

    delay_ms.flush()
    json_write(
        meta_path,
        {
            "signature": signature,
            "delay_file": delay_path.name,
            "delay_unit": "ms",
            "delay_min_ms": float(delay_min),
            "delay_max_ms": float(delay_max),
            "delay_mean_ms": float(delay_sum / delay_count) if delay_count else None,
        },
    )
    print(f"[paper2-shortest] Wrote {delay_path}", flush=True)
    return delay_path


def ensure_paper2_position_cache(
    *,
    ephem_dir: Path,
    cache_root: Path,
    start: int,
    end: int,
    total_sats: int,
    workers: int,
    force: bool,
) -> Path:
    cache_dir = default_cache_dir(cache_root, int(start), int(end), 1)
    if completed_position_cache(cache_dir) and not force:
        print(f"[paper2-shortest] Reusing position cache: {cache_dir}", flush=True)
        return cache_dir

    try:
        store = open_position_cache_for_interval(start, end, stride=1, cache_root=cache_root)
        print(f"[paper2-shortest] Reusing covering position cache: {store.cache_dir}", flush=True)
        return store.cache_dir
    except FileNotFoundError:
        pass

    return ensure_position_cache(
        ephem_dir=ephem_dir,
        cache_root=cache_root,
        start=int(start),
        end=int(end),
        step=1,
        total_sats=int(total_sats),
        workers=int(workers),
        progress_every=32,
        mode="memmap",
        force=bool(force),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Paper2 G60 China-Europe mean shortest hops and propagation delay."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86400)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--source-group", type=int, default=0, help="Paper2 group 0 is China.")
    parser.add_argument("--target-group", type=int, default=1, help="Paper2 group 1 is Europe.")
    parser.add_argument(
        "--topology",
        type=str,
        default="plus_grid",
        help="Topology to evaluate. Default is plus_grid: A/right-neighbor inter links plus intra y-ring.",
    )
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--ephem-dir", type=Path, default=DEFAULT_EPHEM_DIR)
    parser.add_argument("--position-cache-root", type=Path, default=DEFAULT_POSITION_CACHE_ROOT)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING_DIR)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--metric-workers", type=int, default=1)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--metric-chunk-size", type=int, default=100)
    parser.add_argument("--force-position-cache", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--force-delay-weights", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")

    config = build_paper2_g60_config()
    write_station_group_mapping(out_dir=args.mapping_dir, config=config)
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    topology = normalize_topology_name(args.topology)
    plot_label = topology_plot_label(topology)
    current_pair_key = pair_key(config, int(args.source_group), int(args.target_group))
    current_pair_label = pair_label(config, int(args.source_group), int(args.target_group))
    out_dir = (
        Path(args.out_base)
        / topology
        / f"{current_pair_key}_t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    )
    inputs_dir = out_dir / "inputs"
    figures_dir = (
        Path(args.figure_dir)
        / topology
        / f"{current_pair_key}_t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    )

    group_data = load_or_build_group_data(
        xml_file=args.xml_file,
        group_cache_dir=args.group_cache_dir,
        steps=[int(x) for x in steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    edge_table = build_topology_edge_table(config, topology)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    edges_csv = inputs_dir / "edges.csv"
    write_edges_csv(edge_table, edges_csv)
    np.save(inputs_dir / "time_indices.npy", steps.astype(np.int64))
    active_mask = np.ones(int(edge_table.num_edges), dtype=bool)
    np.save(inputs_dir / "active_mask.npy", active_mask)

    position_cache_dir = ensure_paper2_position_cache(
        ephem_dir=args.ephem_dir,
        cache_root=args.position_cache_root,
        start=int(args.start),
        end=int(args.end),
        total_sats=config.total_sats,
        workers=int(args.workers),
        force=bool(args.force_position_cache),
    )
    delay_path = build_or_load_delay_weights(
        out_dir=inputs_dir,
        position_cache_dir=position_cache_dir,
        edge_table=edge_table,
        steps=steps,
        stride=int(args.stride),
        chunk_steps=int(args.chunk_steps),
        force=bool(args.force_delay_weights),
    )
    edge_delay_ms = np.load(delay_path, mmap_mode="r")

    group_nodes = build_group_pair_node_arrays(
        group_data=group_data,
        steps=steps,
        source_group_id=int(args.source_group),
        target_group_id=int(args.target_group),
    )
    result = compute_group_pair_shortest_timeseries(
        edge_table=edge_table,
        total_nodes=config.total_sats,
        steps=steps,
        edge_active_mask=active_mask,
        edge_weights_ms=edge_delay_ms,
        group_nodes=group_nodes,
        workers=int(args.metric_workers),
        chunk_size=int(args.metric_chunk_size),
        progress_every_chunks=10,
    )

    files = write_group_pair_shortest_timeseries(
        result,
        out_dir,
        meta={
            "constellation": config.name,
            "topology": topology,
            "source_group": group_label(config, int(args.source_group)),
            "target_group": group_label(config, int(args.target_group)),
            "source_group_id": int(args.source_group),
            "target_group_id": int(args.target_group),
            "xml_file": str(Path(args.xml_file)),
            "position_cache_dir": str(position_cache_dir),
            "edge_delay_ms": str(delay_path),
            "edges_csv": str(edges_csv),
            "active_mask": str(inputs_dir / "active_mask.npy"),
        },
    )
    hop_plot = plot_group_pair_shortest_timeseries(
        {plot_label: result},
        figures_dir,
        metric="mean_shortest_hops",
        ylabel="mean shortest hops",
        title=f"Paper2 G60 {current_pair_label} mean shortest hops ({plot_label})",
        filename=f"{current_pair_key}_mean_shortest_hops.png",
    )
    delay_plot = plot_group_pair_shortest_timeseries(
        {plot_label: result},
        figures_dir,
        metric="mean_shortest_delay_ms",
        ylabel="mean shortest propagation delay (ms)",
        title=f"Paper2 G60 {current_pair_label} mean shortest propagation delay ({plot_label})",
        filename=f"{current_pair_key}_mean_shortest_delay_ms.png",
    )
    summary = result_summary(result)
    print(
        f"[paper2-shortest] steps={len(steps)} edges={edge_table.num_edges} "
        f"mean_hops={summary['mean_hops']:.6f} "
        f"mean_delay_ms={summary['mean_shortest_delay_ms']:.6f}",
        flush=True,
    )
    print(f"[paper2-shortest] csv={files['timeseries']}", flush=True)
    print(f"[paper2-shortest] hop_plot={hop_plot}", flush=True)
    print(f"[paper2-shortest] delay_plot={delay_plot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
