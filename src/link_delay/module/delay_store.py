from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable, build_full_option_edges, write_edges_csv
from src.link_delay.module.position_cache import PositionCache, load_position_cache


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent

LIGHT_SPEED_KM_S = 299_792.458


@dataclass(frozen=True)
class DelayArtifacts:
    cache_dir: Path
    out_dir: Path
    delay_path: Path
    edges_csv_path: Path
    meta_path: Path
    time_indices_path: Path
    times_path: Path
    edge_table: EdgeTable
    time_indices: np.ndarray
    times_s: np.ndarray
    meta: dict


def resolve_time_indices(num_cache_steps: int, start: int, end: int | None, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be positive")
    if start < 0:
        raise ValueError("start must be >= 0")
    if end is None:
        end = num_cache_steps - 1
    if end < start:
        raise ValueError(f"end {end} is smaller than start {start}")
    if end >= num_cache_steps:
        raise ValueError(f"end {end} is outside cache range 0..{num_cache_steps - 1}")
    return np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)


def default_output_dir(output_base: str | Path, cache_dir: str | Path, start: int, end: int | None, stride: int) -> Path:
    end_label = "all" if end is None else str(end)
    return Path(output_base) / f"{Path(cache_dir).name}_t{start}_{end_label}_stride{stride}"


def cache_signature(
    *,
    cache: PositionCache,
    config: ViewerConfig,
    edge_table: EdgeTable,
    time_indices: np.ndarray,
    options: Iterable[int],
    stride: int,
    wrap_planes: bool = False,
) -> dict:
    positions_path = cache.cache_dir / "positions_km.npy"
    times_path = cache.cache_dir / "times_s.npy"
    return {
        "script_version": 1,
        "cache_dir": str(cache.cache_dir.resolve()),
        "positions_size": int(positions_path.stat().st_size),
        "positions_mtime_ns": int(positions_path.stat().st_mtime_ns),
        "times_size": int(times_path.stat().st_size),
        "times_mtime_ns": int(times_path.stat().st_mtime_ns),
        "positions_shape": [int(x) for x in cache.positions_km.shape],
        "positions_dtype": str(cache.positions_km.dtype),
        "config_name": str(config.name),
        "P": int(config.P),
        "N": int(config.N),
        "total_sats": int(config.total_sats),
        "options": [int(x) for x in options],
        "wrap_planes": bool(wrap_planes),
        "num_edges": int(edge_table.num_edges),
        "time_start_index": int(time_indices[0]),
        "time_end_index": int(time_indices[-1]),
        "num_steps": int(time_indices.size),
        "stride": int(stride),
        "light_speed_km_s": float(LIGHT_SPEED_KM_S),
    }


def load_meta(meta_path: str | Path) -> dict | None:
    meta_path = Path(meta_path)
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def existing_artifacts_match(out_dir: str | Path, signature: dict) -> bool:
    out_dir = Path(out_dir)
    meta_path = out_dir / "delay_meta.json"
    delay_path = out_dir / "edge_delay_ms.npy"
    edges_csv_path = out_dir / "edges.csv"
    time_indices_path = out_dir / "time_indices.npy"
    times_path = out_dir / "times_s.npy"
    meta = load_meta(meta_path)
    if meta is None:
        return False
    if meta.get("signature") != signature:
        return False
    return all(p.exists() for p in (delay_path, edges_csv_path, time_indices_path, times_path))


def compute_delay_cache(
    *,
    cache: PositionCache,
    edge_table: EdgeTable,
    out_dir: str | Path,
    signature: dict,
    time_indices: np.ndarray,
    force: bool,
    chunk_steps: int,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delay_path = out_dir / "edge_delay_ms.npy"
    edges_csv_path = out_dir / "edges.csv"
    time_indices_path = out_dir / "time_indices.npy"
    times_path = out_dir / "times_s.npy"
    meta_path = out_dir / "delay_meta.json"

    if not force and existing_artifacts_match(out_dir, signature):
        meta = load_meta(meta_path) or {}
        print(f"[delay] Reusing existing delay cache: {delay_path}", flush=True)
        return meta

    if chunk_steps <= 0:
        raise ValueError("chunk_steps must be positive")

    print(f"[delay] Building delay cache in {out_dir}", flush=True)
    print(f"[delay] steps={time_indices.size}, edges={edge_table.num_edges}, dtype=float32", flush=True)
    write_edges_csv(edge_table, edges_csv_path)
    np.save(time_indices_path, time_indices)
    np.save(times_path, np.asarray(cache.times_s[time_indices], dtype=np.float64))

    delay_ms = open_memmap(
        delay_path,
        mode="w+",
        dtype=np.float32,
        shape=(int(time_indices.size), int(edge_table.num_edges)),
    )

    src_idx = edge_table.src
    dst_idx = edge_table.dst
    delay_min = math.inf
    delay_max = -math.inf
    delay_sum = 0.0
    delay_count = 0
    started = time.perf_counter()

    for local_start in range(0, int(time_indices.size), int(chunk_steps)):
        local_end = min(local_start + int(chunk_steps), int(time_indices.size))
        idx = time_indices[local_start:local_end]

        if int(idx[-1]) - int(idx[0]) + 1 == int(idx.size):
            pos_chunk = cache.positions_km[int(idx[0]) : int(idx[-1]) + 1]
        else:
            pos_chunk = cache.positions_km[idx]

        src_xyz = pos_chunk[:, src_idx, :]
        dst_xyz = pos_chunk[:, dst_idx, :]
        diff = src_xyz - dst_xyz
        dist_km = np.sqrt(np.sum(diff * diff, axis=2), dtype=np.float32)
        chunk_delay = (dist_km / np.float32(LIGHT_SPEED_KM_S) * np.float32(1000.0)).astype(np.float32)

        delay_ms[local_start:local_end, :] = chunk_delay
        delay_min = min(delay_min, float(np.nanmin(chunk_delay)))
        delay_max = max(delay_max, float(np.nanmax(chunk_delay)))
        delay_sum += float(np.nansum(chunk_delay, dtype=np.float64))
        delay_count += int(np.isfinite(chunk_delay).sum())

        chunk_no = local_start // int(chunk_steps)
        if chunk_no % 10 == 0 or local_end == int(time_indices.size):
            elapsed = time.perf_counter() - started
            pct = 100.0 * local_end / max(1, int(time_indices.size))
            print(f"[delay] {local_end}/{time_indices.size} steps ({pct:.1f}%), elapsed={elapsed:.1f}s", flush=True)

    delay_ms.flush()
    meta = {
        "signature": signature,
        "delay_file": delay_path.name,
        "edges_file": edges_csv_path.name,
        "time_indices_file": time_indices_path.name,
        "times_file": times_path.name,
        "delay_unit": "ms",
        "distance_unit": "km",
        "distance_formula": "delay_ms / 1000 * light_speed_km_s",
        "delay_min_ms": float(delay_min),
        "delay_max_ms": float(delay_max),
        "delay_mean_ms": float(delay_sum / delay_count) if delay_count else None,
        "built_at_unix_s": time.time(),
    }
    write_json(meta_path, meta)
    print(f"[delay] Wrote {delay_path}", flush=True)
    print(f"[delay] Wrote {edges_csv_path}", flush=True)
    print(f"[delay] Wrote {meta_path}", flush=True)
    return meta


def build_or_load_artifacts(
    *,
    cache_dir: str | Path,
    out_dir: str | Path,
    config: ViewerConfig,
    start: int,
    end: int | None,
    stride: int,
    force: bool,
    chunk_steps: int,
    allow_incomplete_cache: bool,
    options: Iterable[int] = (0, 1, 2, 4),
    wrap_planes: bool = False,
) -> DelayArtifacts:
    cache = load_position_cache(cache_dir, allow_incomplete_cache=allow_incomplete_cache)
    if int(cache.positions_km.shape[1]) != int(config.total_sats):
        raise ValueError(
            f"Cache has {cache.positions_km.shape[1]} satellites, but {config.name} config expects {config.total_sats}"
        )

    edge_table = build_full_option_edges(
        config,
        options=options,
        sat_ids=cache.sat_ids,
        wrap_planes=bool(wrap_planes),
    )
    time_indices = resolve_time_indices(cache.positions_km.shape[0], start, end, stride)
    signature = cache_signature(
        cache=cache,
        config=config,
        edge_table=edge_table,
        time_indices=time_indices,
        options=options,
        stride=stride,
        wrap_planes=bool(wrap_planes),
    )
    meta = compute_delay_cache(
        cache=cache,
        edge_table=edge_table,
        out_dir=out_dir,
        signature=signature,
        time_indices=time_indices,
        force=force,
        chunk_steps=chunk_steps,
    )

    out_dir = Path(out_dir)
    return DelayArtifacts(
        cache_dir=Path(cache_dir),
        out_dir=out_dir,
        delay_path=out_dir / "edge_delay_ms.npy",
        edges_csv_path=out_dir / "edges.csv",
        meta_path=out_dir / "delay_meta.json",
        time_indices_path=out_dir / "time_indices.npy",
        times_path=out_dir / "times_s.npy",
        edge_table=edge_table,
        time_indices=np.load(out_dir / "time_indices.npy", mmap_mode="r"),
        times_s=np.load(out_dir / "times_s.npy", mmap_mode="r"),
        meta=meta,
    )


def write_edge_index_matrix(
    artifacts: DelayArtifacts,
    *,
    total_sats: int,
    directed_storage: bool = False,
) -> Path:
    matrix = np.full((int(total_sats), int(total_sats)), -1, dtype=np.int32)
    edge_table = artifacts.edge_table
    for edge_idx in range(edge_table.num_edges):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        matrix[src, dst] = int(edge_idx)
        if not directed_storage:
            matrix[dst, src] = int(edge_idx)

    path = Path(artifacts.out_dir) / "edge_index_matrix.npy"
    np.save(path, matrix)
    return path


def write_query_meta(
    artifacts: DelayArtifacts,
    edge_index_path: str | Path,
    *,
    options: Iterable[int],
    directed_storage: bool = False,
    wrap_planes: bool = False,
) -> Path:
    path = Path(artifacts.out_dir) / "delay_store_query_meta.json"
    payload = {
        "store_dir": str(Path(artifacts.out_dir).resolve()),
        "delay_file": str(Path(artifacts.delay_path).name),
        "edges_file": str(Path(artifacts.edges_csv_path).name),
        "edge_index_matrix_file": str(Path(edge_index_path).name),
        "time_indices_file": str(Path(artifacts.time_indices_path).name),
        "times_file": str(Path(artifacts.times_path).name),
        "query_rule": "edge_idx = edge_index_matrix[src_node, dst_node]; delay_ms = edge_delay_ms[row, edge_idx]",
        "undirected_edge_lookup": not bool(directed_storage),
        "options": [int(x) for x in options],
        "wrap_planes": bool(wrap_planes),
        "time_start": int(artifacts.time_indices[0]),
        "time_end": int(artifacts.time_indices[-1]),
        "num_steps": int(len(artifacts.time_indices)),
        "num_edges": int(artifacts.edge_table.num_edges),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def export_step_csv(artifacts: DelayArtifacts, cache_step: int) -> Path:
    matches = np.where(np.asarray(artifacts.time_indices) == int(cache_step))[0]
    if matches.size == 0:
        raise ValueError(
            f"cache step {cache_step} is not in this delay cache "
            f"({int(artifacts.time_indices[0])}..{int(artifacts.time_indices[-1])})"
        )

    local_row = int(matches[0])
    delay = np.load(artifacts.delay_path, mmap_mode="r")
    values = np.asarray(delay[local_row], dtype=np.float64)
    edge_table = artifacts.edge_table
    out_path = Path(artifacts.out_dir) / f"edge_delay_step_{cache_step}.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "edge_idx",
                "cache_step",
                "time_s",
                "src_node",
                "dst_node",
                "src_sat_id",
                "dst_sat_id",
                "src_plane",
                "src_y",
                "dst_plane",
                "dst_y",
                "option",
                "delay_ms",
                "distance_km",
            ]
        )
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            delay_ms = float(values[idx])
            writer.writerow(
                [
                    idx,
                    int(cache_step),
                    float(artifacts.times_s[local_row]),
                    src,
                    dst,
                    edge_table.sat_ids[src],
                    edge_table.sat_ids[dst],
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                    f"{delay_ms:.8f}",
                    f"{delay_ms / 1000.0 * LIGHT_SPEED_KM_S:.6f}",
                ]
            )
    print(f"[delay] Wrote step CSV: {out_path}", flush=True)
    return out_path
