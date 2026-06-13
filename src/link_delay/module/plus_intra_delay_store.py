from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from src.config.viewer_config import ViewerConfig

from .delay_store import LIGHT_SPEED_KM_S
from .edge_options import EdgeTable, write_edges_csv
from .position_cache import PositionCacheStore
from .query import FullLinkDelayStore


INTRA_OPTION = -1


def source_store_signature(store: FullLinkDelayStore) -> dict:
    return {
        "store_dir": str(store.store_dir.resolve()),
        "delay_size": int(store.delay_path.stat().st_size),
        "delay_mtime_ns": int(store.delay_path.stat().st_mtime_ns),
        "edges_size": int(store.edges_path.stat().st_size),
        "edges_mtime_ns": int(store.edges_path.stat().st_mtime_ns),
        "time_indices_size": int(store.time_indices_path.stat().st_size),
        "time_indices_mtime_ns": int(store.time_indices_path.stat().st_mtime_ns),
        "num_steps": int(store.num_steps),
        "num_edges": int(store.num_edges),
        "time_start": int(store.time_start),
        "time_end": int(store.time_end),
    }


def position_cache_signature(cache: PositionCacheStore) -> dict:
    return {
        "cache_dir": str(cache.cache_dir.resolve()),
        "positions_size": int(cache.positions_path.stat().st_size),
        "positions_mtime_ns": int(cache.positions_path.stat().st_mtime_ns),
        "times_size": int(cache.times_path.stat().st_size),
        "times_mtime_ns": int(cache.times_path.stat().st_mtime_ns),
        "positions_shape": [int(x) for x in cache.positions_km.shape],
        "positions_dtype": str(cache.positions_km.dtype),
        "time_start": int(cache.time_start),
        "time_end": int(cache.time_end),
    }


def build_signature(
    *,
    config: ViewerConfig,
    inter_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    steps: np.ndarray,
    stride: int,
    intra_option: int = INTRA_OPTION,
) -> dict:
    return {
        "module": "src.link_delay.module.plus_intra_delay_store",
        "script_version": 2,
        "constellation": str(config.name),
        "P": int(config.P),
        "N": int(config.N),
        "total_sats": int(config.total_sats),
        "source_inter_store": source_store_signature(inter_store),
        "source_position_cache": position_cache_signature(position_store),
        "time_start_index": int(steps[0]),
        "time_end_index": int(steps[-1]),
        "num_steps": int(steps.size),
        "stride": int(stride),
        "intra_option": int(intra_option),
        "intra_rule": "(p,y) -- (p,(y+1) mod N) for every p,y",
        "light_speed_km_s": float(LIGHT_SPEED_KM_S),
    }


def load_meta(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def existing_cache_matches(out_dir: str | Path, signature: dict) -> bool:
    out_dir = Path(out_dir)
    meta = load_meta(out_dir / "delay_meta.json")
    if not meta or meta.get("signature") != signature:
        return False
    required = [
        out_dir / "edge_delay_ms.npy",
        out_dir / "edges.csv",
        out_dir / "time_indices.npy",
        out_dir / "times_s.npy",
        out_dir / "edge_index_matrix.npy",
        out_dir / "delay_store_query_meta.json",
    ]
    return all(path.exists() for path in required)


def read_inter_edge_rows(store: FullLinkDelayStore) -> list[dict]:
    rows: list[dict] = []
    with store.edges_path.open("r", encoding="utf-8", newline="") as f:
        rows.extend(csv.DictReader(f))
    return rows


def build_plus_intra_edge_table(
    *,
    config: ViewerConfig,
    inter_store: FullLinkDelayStore,
    sat_ids: list[str],
    intra_option: int = INTRA_OPTION,
) -> EdgeTable:
    src_values: list[int] = []
    dst_values: list[int] = []
    option_values: list[int] = []
    src_plane_values: list[int] = []
    src_y_values: list[int] = []
    dst_plane_values: list[int] = []
    dst_y_values: list[int] = []
    seen: set[tuple[int, int]] = set()

    for row in read_inter_edge_rows(inter_store):
        src = int(row["src_node"])
        dst = int(row["dst_node"])
        key = (min(src, dst), max(src, dst))
        if key in seen:
            continue
        seen.add(key)
        src_values.append(src)
        dst_values.append(dst)
        option_values.append(int(row["option"]))
        src_plane_values.append(int(row["src_plane"]))
        src_y_values.append(int(row["src_y"]))
        dst_plane_values.append(int(row["dst_plane"]))
        dst_y_values.append(int(row["dst_y"]))

    for plane in range(int(config.P)):
        for y in range(int(config.N)):
            src = plane * int(config.N) + y
            dst = plane * int(config.N) + ((y + 1) % int(config.N))
            key = (min(src, dst), max(src, dst))
            if key in seen:
                continue
            seen.add(key)
            src_values.append(src)
            dst_values.append(dst)
            option_values.append(int(intra_option))
            src_plane_values.append(plane)
            src_y_values.append(y)
            dst_plane_values.append(plane)
            dst_y_values.append((y + 1) % int(config.N))

    return EdgeTable(
        src=np.asarray(src_values, dtype=np.int32),
        dst=np.asarray(dst_values, dtype=np.int32),
        option=np.asarray(option_values, dtype=np.int16),
        src_plane=np.asarray(src_plane_values, dtype=np.int16),
        src_y=np.asarray(src_y_values, dtype=np.int16),
        dst_plane=np.asarray(dst_plane_values, dtype=np.int16),
        dst_y=np.asarray(dst_y_values, dtype=np.int16),
        sat_ids=sat_ids,
    )


def write_edge_index_matrix(edge_table: EdgeTable, path: str | Path, total_sats: int) -> None:
    matrix = np.full((int(total_sats), int(total_sats)), -1, dtype=np.int32)
    for edge_idx in range(edge_table.num_edges):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        matrix[src, dst] = int(edge_idx)
        matrix[dst, src] = int(edge_idx)
    np.save(Path(path), matrix)


def contiguous_take(array: np.ndarray, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return array[rows]
    if int(rows[-1]) - int(rows[0]) + 1 == int(rows.size):
        return array[int(rows[0]) : int(rows[-1]) + 1]
    return array[rows]


def write_query_meta(
    *,
    out_dir: str | Path,
    edge_table: EdgeTable,
    time_indices: np.ndarray,
    intra_option: int = INTRA_OPTION,
) -> None:
    out_dir = Path(out_dir)
    payload = {
        "store_dir": str(out_dir.resolve()),
        "delay_file": "edge_delay_ms.npy",
        "edges_file": "edges.csv",
        "edge_index_matrix_file": "edge_index_matrix.npy",
        "time_indices_file": "time_indices.npy",
        "times_file": "times_s.npy",
        "query_rule": "edge_idx = edge_index_matrix[src_node, dst_node]; delay_ms = edge_delay_ms[row, edge_idx]",
        "undirected_edge_lookup": True,
        "options": sorted({int(x) for x in np.asarray(edge_table.option).tolist()}),
        "intra_option": int(intra_option),
        "time_start": int(time_indices[0]),
        "time_end": int(time_indices[-1]),
        "num_steps": int(time_indices.size),
        "num_edges": int(edge_table.num_edges),
    }
    with (out_dir / "delay_store_query_meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_plus_intra_meta(
    *,
    out_dir: str | Path,
    signature: dict,
    delay_min: float,
    delay_max: float,
    delay_sum: float,
    delay_count: int,
    intra_option: int = INTRA_OPTION,
) -> None:
    out_dir = Path(out_dir)
    payload = {
        "signature": signature,
        "delay_file": "edge_delay_ms.npy",
        "edges_file": "edges.csv",
        "edge_index_matrix_file": "edge_index_matrix.npy",
        "time_indices_file": "time_indices.npy",
        "times_file": "times_s.npy",
        "delay_unit": "ms",
        "distance_unit": "km",
        "distance_formula": "delay_ms / 1000 * light_speed_km_s",
        "edge_components": {
            "inter_from_existing_full_option_store": True,
            "intra_from_position_cache": True,
            "intra_option": int(intra_option),
        },
        "delay_min_ms": float(delay_min),
        "delay_max_ms": float(delay_max),
        "delay_mean_ms": float(delay_sum / delay_count) if delay_count else None,
        "built_at_unix_s": time.time(),
    }
    with (out_dir / "delay_meta.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def default_plus_intra_store_dir(
    output_base: str | Path,
    *,
    constellation_name: str,
    start: int,
    end: int,
    stride: int,
) -> Path:
    return (
        Path(output_base)
        / f"{str(constellation_name)}_full_options_plus_intra_t{int(start)}_{int(end)}_stride{int(stride)}"
    )


def build_plus_intra_delay_store(
    *,
    config: ViewerConfig,
    inter_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    out_dir: str | Path,
    start: int,
    end: int,
    stride: int = 1,
    chunk_steps: int = 256,
    force: bool = False,
    intra_option: int = INTRA_OPTION,
) -> Path:
    if int(position_store.num_sats) != int(config.total_sats):
        raise ValueError(
            f"Position cache has {position_store.num_sats} satellites, "
            f"but {config.name} expects {config.total_sats}"
        )

    inter_rows = inter_store.rows_for_interval(start, end, stride)
    position_rows = position_store.rows_for_interval(start, end, stride)
    time_indices = np.asarray(inter_store.time_indices[inter_rows], dtype=np.int64)
    position_times = np.asarray(position_store.times_s[position_rows], dtype=np.int64)
    if not np.array_equal(time_indices, position_times):
        raise ValueError("inter delay store and position cache are not aligned on time indices")

    signature = build_signature(
        config=config,
        inter_store=inter_store,
        position_store=position_store,
        steps=time_indices,
        stride=int(stride),
        intra_option=int(intra_option),
    )
    out_dir = Path(out_dir)
    if not bool(force) and existing_cache_matches(out_dir, signature):
        print(f"[plus-intra-delay] Reusing existing cache: {out_dir}", flush=True)
        return out_dir
    if int(chunk_steps) <= 0:
        raise ValueError("chunk_steps must be positive")

    out_dir.mkdir(parents=True, exist_ok=True)
    edge_table = build_plus_intra_edge_table(
        config=config,
        inter_store=inter_store,
        sat_ids=position_store.sat_ids,
        intra_option=int(intra_option),
    )
    inter_count = int(inter_store.num_edges)
    intra_count = int(edge_table.num_edges - inter_count)
    if intra_count != int(config.total_sats):
        raise ValueError(f"Expected {config.total_sats} intra edges, got {intra_count}")

    print(
        f"[plus-intra-delay] Building {out_dir} | steps={time_indices.size} "
        f"inter={inter_count} intra={intra_count} total_edges={edge_table.num_edges}",
        flush=True,
    )
    write_edges_csv(edge_table, out_dir / "edges.csv")
    np.save(out_dir / "time_indices.npy", time_indices)
    np.save(out_dir / "times_s.npy", np.asarray(inter_store.times_s[inter_rows], dtype=np.float64))
    write_edge_index_matrix(edge_table, out_dir / "edge_index_matrix.npy", int(config.total_sats))
    write_query_meta(out_dir=out_dir, edge_table=edge_table, time_indices=time_indices, intra_option=int(intra_option))

    delay_ms = open_memmap(
        out_dir / "edge_delay_ms.npy",
        mode="w+",
        dtype=np.float32,
        shape=(int(time_indices.size), int(edge_table.num_edges)),
    )
    intra_src = np.asarray(edge_table.src[inter_count:], dtype=np.int64)
    intra_dst = np.asarray(edge_table.dst[inter_count:], dtype=np.int64)
    delay_min = math.inf
    delay_max = -math.inf
    delay_sum = 0.0
    delay_count = 0
    started_at = time.perf_counter()

    for local_start in range(0, int(time_indices.size), int(chunk_steps)):
        local_end = min(local_start + int(chunk_steps), int(time_indices.size))
        inter_chunk_rows = inter_rows[local_start:local_end]
        position_chunk_rows = position_rows[local_start:local_end]

        inter_delay = np.asarray(contiguous_take(inter_store.delay_ms_array, inter_chunk_rows), dtype=np.float32)
        positions = np.asarray(contiguous_take(position_store.positions_km, position_chunk_rows), dtype=np.float32)
        diff = positions[:, intra_src, :] - positions[:, intra_dst, :]
        dist_km = np.sqrt(np.sum(diff * diff, axis=2), dtype=np.float32)
        intra_delay = (dist_km / np.float32(LIGHT_SPEED_KM_S) * np.float32(1000.0)).astype(np.float32)

        delay_ms[local_start:local_end, :inter_count] = inter_delay
        delay_ms[local_start:local_end, inter_count:] = intra_delay

        chunk_all_min = min(float(np.nanmin(inter_delay)), float(np.nanmin(intra_delay)))
        chunk_all_max = max(float(np.nanmax(inter_delay)), float(np.nanmax(intra_delay)))
        delay_min = min(delay_min, chunk_all_min)
        delay_max = max(delay_max, chunk_all_max)
        delay_sum += float(np.nansum(inter_delay, dtype=np.float64))
        delay_sum += float(np.nansum(intra_delay, dtype=np.float64))
        delay_count += int(np.isfinite(inter_delay).sum()) + int(np.isfinite(intra_delay).sum())

        chunk_no = local_start // int(chunk_steps)
        if chunk_no % 10 == 0 or local_end == int(time_indices.size):
            elapsed = time.perf_counter() - started_at
            pct = 100.0 * local_end / max(1, int(time_indices.size))
            print(
                f"[plus-intra-delay] {local_end}/{time_indices.size} steps ({pct:.1f}%), elapsed={elapsed:.1f}s",
                flush=True,
            )

    delay_ms.flush()
    write_plus_intra_meta(
        out_dir=out_dir,
        signature=signature,
        delay_min=delay_min,
        delay_max=delay_max,
        delay_sum=delay_sum,
        delay_count=delay_count,
        intra_option=int(intra_option),
    )
    print(f"[plus-intra-delay] done | delay={out_dir / 'edge_delay_ms.npy'}", flush=True)
    return out_dir
