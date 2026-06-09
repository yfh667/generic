from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
CODEX_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from src.config.viewer_config import G60_CONFIG
from full_link_delay_viewer import build_or_load_artifacts


DEFAULT_EPHEM_DIR = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "satellitesposition"
    / "satellite_pos"
)
DEFAULT_POSITION_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "postprocess"
    / "full_option_edge_delay"
    / "_position_cache"
)
DEFAULT_DELAY_OUTPUT_BASE = PROJECT_ROOT / "data" / "postprocess" / "full_option_edge_delay"


def cache_dir_for(root: Path, start: int, end: int, step: int) -> Path:
    return Path(root) / f"cache_{int(start)}_{int(end)}_{int(step)}s"


def default_delay_out_dir(start: int, end: int, stride: int) -> Path:
    return DEFAULT_DELAY_OUTPUT_BASE / f"G60_full_options_t{int(start)}_{int(end)}_stride{int(stride)}"


def read_json(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def completed_position_cache(cache_dir: Path) -> bool:
    report = read_json(Path(cache_dir) / "build_report.json")
    if not isinstance(report, dict):
        return False
    if report.get("status") != "completed" or not bool(report.get("build_succeeded", False)):
        return False
    required = ("positions_km.npy", "times_s.npy", "sat_ids.json", "cache_meta.json")
    return all((Path(cache_dir) / name).exists() for name in required)


def import_position_cache_builder():
    paper3py_dir = GENERIC_ROOT / "paper3py"
    if str(paper3py_dir) not in sys.path:
        sys.path.insert(0, str(paper3py_dir))
    return importlib.import_module("build_cache_parallel")


def ensure_position_cache(
    *,
    ephem_dir: Path,
    cache_root: Path,
    start: int,
    end: int,
    step: int,
    workers: int,
    progress_every: int,
    mode: str,
    force: bool,
) -> Path:
    cache_root = Path(cache_root)
    cache_dir = cache_dir_for(cache_root, start, end, step)
    if completed_position_cache(cache_dir) and not force:
        print(f"[prepare] Reusing completed position cache: {cache_dir}", flush=True)
        return cache_dir

    builder = import_position_cache_builder()
    builder.SIM_OUTPUT_DIR = cache_root
    builder.CACHE_DIR = cache_dir

    print(f"[prepare] Building helper position cache: {cache_dir}", flush=True)
    print(f"[prepare] ephem_dir={ephem_dir}", flush=True)
    builder.build_cache_parallel(
        ephem_dir,
        start,
        end,
        step,
        workers=workers,
        progress_every=progress_every,
        mode=mode,
        flush_every=64,
    )
    return cache_dir


def validate_position_cache(cache_dir: Path, expected_steps: int, expected_sats: int) -> None:
    positions = np.load(Path(cache_dir) / "positions_km.npy", mmap_mode="r")
    if positions.shape != (int(expected_steps), int(expected_sats), 3):
        raise ValueError(
            f"position cache shape mismatch: got {positions.shape}, "
            f"expected=({expected_steps}, {expected_sats}, 3)"
        )

    max_zero_rows = 0
    bad_step = None
    for start in range(0, int(expected_steps), 512):
        end = min(start + 512, int(expected_steps))
        chunk = positions[start:end]
        zero_counts = np.all(chunk == 0.0, axis=2).sum(axis=1)
        local_max = int(zero_counts.max())
        if local_max > max_zero_rows:
            max_zero_rows = local_max
            bad_step = int(start + int(zero_counts.argmax()))
    if max_zero_rows:
        raise ValueError(
            f"position cache still has all-zero satellite rows: "
            f"max_zero_rows={max_zero_rows} at local step {bad_step}"
        )
    print(f"[prepare] Position cache validated: {cache_dir}", flush=True)


def write_edge_index_matrix(artifacts) -> Path:
    total_sats = int(G60_CONFIG.total_sats)
    matrix = np.full((total_sats, total_sats), -1, dtype=np.int32)
    edge_table = artifacts.edge_table
    for edge_idx in range(edge_table.num_edges):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        matrix[src, dst] = int(edge_idx)
        matrix[dst, src] = int(edge_idx)

    path = Path(artifacts.out_dir) / "edge_index_matrix.npy"
    np.save(path, matrix)
    return path


def write_query_meta(artifacts, edge_index_path: Path) -> Path:
    path = Path(artifacts.out_dir) / "delay_store_query_meta.json"
    payload = {
        "store_dir": str(Path(artifacts.out_dir).resolve()),
        "delay_file": str(Path(artifacts.delay_path).name),
        "edges_file": str(Path(artifacts.edges_csv_path).name),
        "edge_index_matrix_file": str(Path(edge_index_path).name),
        "time_indices_file": str(Path(artifacts.time_indices_path).name),
        "times_file": str(Path(artifacts.times_path).name),
        "query_rule": "edge_idx = edge_index_matrix[src_node, dst_node]; delay_ms = edge_delay_ms[row, edge_idx]",
        "undirected_edge_lookup": True,
        "time_start": int(artifacts.time_indices[0]),
        "time_end": int(artifacts.time_indices[-1]),
        "num_steps": int(len(artifacts.time_indices)),
        "num_edges": int(artifacts.edge_table.num_edges),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable 0..10000s full-option ISL edge-delay store."
    )
    parser.add_argument("--ephem-dir", type=Path, default=DEFAULT_EPHEM_DIR)
    parser.add_argument("--position-cache-root", type=Path, default=DEFAULT_POSITION_CACHE_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10000)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=16)
    parser.add_argument("--mode", choices=("memory", "memmap"), default="memory")
    parser.add_argument("--chunk-steps", type=int, default=512)
    parser.add_argument("--force-position-cache", action="store_true")
    parser.add_argument("--force-delay-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.step != 1:
        raise ValueError("This helper currently expects --step 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    expected_steps = int((args.end - args.start) / args.step) + 1
    position_cache_dir = ensure_position_cache(
        ephem_dir=args.ephem_dir,
        cache_root=args.position_cache_root,
        start=args.start,
        end=args.end,
        step=args.step,
        workers=args.workers,
        progress_every=args.progress_every,
        mode=args.mode,
        force=args.force_position_cache,
    )
    validate_position_cache(position_cache_dir, expected_steps, int(G60_CONFIG.total_sats))

    out_dir = args.out_dir or default_delay_out_dir(args.start, args.end, args.stride)
    artifacts = build_or_load_artifacts(
        cache_dir=position_cache_dir,
        out_dir=out_dir,
        config=G60_CONFIG,
        start=args.start,
        end=args.end,
        stride=args.stride,
        force=args.force_delay_cache,
        chunk_steps=args.chunk_steps,
        allow_incomplete_cache=False,
        options=(0, 1, 2, 4),
    )

    edge_index_path = write_edge_index_matrix(artifacts)
    query_meta_path = write_query_meta(artifacts, edge_index_path)
    print(f"[prepare] Delay store ready: {artifacts.out_dir}", flush=True)
    print(f"[prepare] delay={artifacts.delay_path}", flush=True)
    print(f"[prepare] edges={artifacts.edges_csv_path}", flush=True)
    print(f"[prepare] edge_index={edge_index_path}", flush=True)
    print(f"[prepare] query_meta={query_meta_path}", flush=True)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
