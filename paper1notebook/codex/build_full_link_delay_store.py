from __future__ import annotations

import argparse
import copy
import importlib
import json
import multiprocessing as mp
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


GENERIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
CODEX_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CODEX_DIR / "configs" / "full_link_delay_store.yaml"
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from src.config.viewer_config import G60_CONFIG
from full_link_delay_viewer import build_or_load_artifacts


@dataclass(frozen=True)
class ConstellationSpec:
    name: str
    P: int
    N: int
    total_sats: int


@dataclass(frozen=True)
class EdgeSpec:
    options: tuple[int, ...]
    directed_storage: bool


@dataclass(frozen=True)
class TimeSpec:
    start: int
    end: int
    step: int
    stride: int


@dataclass(frozen=True)
class PathSpec:
    ephem_dir: Path
    position_cache_root: Path
    delay_output_base: Path
    out_dir: Path | None


@dataclass(frozen=True)
class RuntimeSpec:
    workers: int
    progress_every: int
    mode: str
    chunk_steps: int


@dataclass(frozen=True)
class BuildSpec:
    force_position_cache: bool
    force_delay_cache: bool
    validate_position_cache: bool
    reject_zero_position_rows: bool


@dataclass(frozen=True)
class DelayStoreBuildConfig:
    schema_version: int
    constellation: ConstellationSpec
    edges: EdgeSpec
    time: TimeSpec
    paths: PathSpec
    runtime: RuntimeSpec
    build: BuildSpec


def read_json(path: Path) -> dict | None:
    if not Path(path).exists():
        return None
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_yaml(path: Path, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_dict(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def raw_config_to_dataclass(raw: dict) -> DelayStoreBuildConfig:
    c = raw.get("constellation", {})
    e = raw.get("edges", {})
    t = raw.get("time", {})
    p = raw.get("paths", {})
    r = raw.get("runtime", {})
    b = raw.get("build", {})
    return DelayStoreBuildConfig(
        schema_version=int(raw.get("schema_version", 1)),
        constellation=ConstellationSpec(
            name=str(c.get("name", "G60")),
            P=int(c.get("P", 18)),
            N=int(c.get("N", 36)),
            total_sats=int(c.get("total_sats", 648)),
        ),
        edges=EdgeSpec(
            options=tuple(int(x) for x in e.get("options", [0, 1, 2, 4])),
            directed_storage=bool(e.get("directed_storage", False)),
        ),
        time=TimeSpec(
            start=int(t.get("start", 0)),
            end=int(t.get("end", 86164)),
            step=int(t.get("step", 1)),
            stride=int(t.get("stride", 1)),
        ),
        paths=PathSpec(
            ephem_dir=resolve_path(p.get("ephem_dir")) or (
                PROJECT_ROOT / "data" / "basic_file" / "satellitesposition" / "satellite_pos"
            ),
            position_cache_root=resolve_path(p.get("position_cache_root")) or (
                PROJECT_ROOT / "data" / "postprocess" / "full_option_edge_delay" / "_position_cache"
            ),
            delay_output_base=resolve_path(p.get("delay_output_base")) or (
                PROJECT_ROOT / "data" / "postprocess" / "full_option_edge_delay"
            ),
            out_dir=resolve_path(p.get("out_dir")),
        ),
        runtime=RuntimeSpec(
            workers=int(r.get("workers", 8)),
            progress_every=int(r.get("progress_every", 32)),
            mode=str(r.get("mode", "memory")),
            chunk_steps=int(r.get("chunk_steps", 512)),
        ),
        build=BuildSpec(
            force_position_cache=bool(b.get("force_position_cache", False)),
            force_delay_cache=bool(b.get("force_delay_cache", False)),
            validate_position_cache=bool(b.get("validate_position_cache", True)),
            reject_zero_position_rows=bool(b.get("reject_zero_position_rows", True)),
        ),
    )


def config_to_plain_dict(cfg: DelayStoreBuildConfig) -> dict:
    plain = asdict(cfg)
    for key in ("ephem_dir", "position_cache_root", "delay_output_base", "out_dir"):
        value = plain["paths"][key]
        plain["paths"][key] = None if value is None else str(Path(value))
    plain["edges"]["options"] = list(cfg.edges.options)
    return plain


def cache_dir_for(root: Path, start: int, end: int, step: int) -> Path:
    return Path(root) / f"cache_{int(start)}_{int(end)}_{int(step)}s"


def default_delay_out_dir(cfg: DelayStoreBuildConfig) -> Path:
    return (
        Path(cfg.paths.delay_output_base)
        / f"{cfg.constellation.name}_full_options_t{cfg.time.start}_{cfg.time.end}_stride{cfg.time.stride}"
    )


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


def ensure_position_cache(cfg: DelayStoreBuildConfig) -> Path:
    cache_dir = cache_dir_for(
        cfg.paths.position_cache_root,
        cfg.time.start,
        cfg.time.end,
        cfg.time.step,
    )
    if completed_position_cache(cache_dir) and not cfg.build.force_position_cache:
        print(f"[delay-build] Reusing completed position cache: {cache_dir}", flush=True)
        return cache_dir

    builder = import_position_cache_builder()
    builder.SIM_OUTPUT_DIR = Path(cfg.paths.position_cache_root)
    builder.CACHE_DIR = cache_dir

    print(f"[delay-build] Building position cache: {cache_dir}", flush=True)
    print(f"[delay-build] ephem_dir={cfg.paths.ephem_dir}", flush=True)
    builder.build_cache_parallel(
        cfg.paths.ephem_dir,
        cfg.time.start,
        cfg.time.end,
        cfg.time.step,
        workers=cfg.runtime.workers,
        progress_every=cfg.runtime.progress_every,
        mode=cfg.runtime.mode,
        flush_every=64,
    )
    return cache_dir


def validate_position_cache(cfg: DelayStoreBuildConfig, cache_dir: Path) -> None:
    positions = np.load(Path(cache_dir) / "positions_km.npy", mmap_mode="r")
    expected_steps = int((cfg.time.end - cfg.time.start) / cfg.time.step) + 1
    expected_shape = (expected_steps, int(cfg.constellation.total_sats), 3)
    if positions.shape != expected_shape:
        raise ValueError(f"position cache shape mismatch: got {positions.shape}, expected={expected_shape}")

    if cfg.build.reject_zero_position_rows:
        max_zero_rows = 0
        bad_step = None
        for local_start in range(0, expected_steps, 512):
            local_end = min(local_start + 512, expected_steps)
            chunk = positions[local_start:local_end]
            zero_counts = np.all(chunk == 0.0, axis=2).sum(axis=1)
            local_max = int(zero_counts.max())
            if local_max > max_zero_rows:
                max_zero_rows = local_max
                bad_step = int(local_start + int(zero_counts.argmax()))
        if max_zero_rows:
            raise ValueError(
                f"position cache has all-zero satellite rows: max_zero_rows={max_zero_rows} "
                f"at local step {bad_step}"
            )
    print(f"[delay-build] Position cache validated: {cache_dir}", flush=True)


def validate_constellation(cfg: DelayStoreBuildConfig) -> None:
    expected = {
        "name": str(G60_CONFIG.name),
        "P": int(G60_CONFIG.P),
        "N": int(G60_CONFIG.N),
        "total_sats": int(G60_CONFIG.total_sats),
    }
    actual = {
        "name": cfg.constellation.name,
        "P": cfg.constellation.P,
        "N": cfg.constellation.N,
        "total_sats": cfg.constellation.total_sats,
    }
    if actual != expected:
        raise ValueError(f"Only {expected} is wired currently, got {actual}")


def write_edge_index_matrix(artifacts, cfg: DelayStoreBuildConfig) -> Path:
    total_sats = int(cfg.constellation.total_sats)
    matrix = np.full((total_sats, total_sats), -1, dtype=np.int32)
    edge_table = artifacts.edge_table
    for edge_idx in range(edge_table.num_edges):
        src = int(edge_table.src[edge_idx])
        dst = int(edge_table.dst[edge_idx])
        matrix[src, dst] = int(edge_idx)
        if not cfg.edges.directed_storage:
            matrix[dst, src] = int(edge_idx)

    path = Path(artifacts.out_dir) / "edge_index_matrix.npy"
    np.save(path, matrix)
    return path


def write_query_meta(artifacts, edge_index_path: Path, cfg: DelayStoreBuildConfig) -> Path:
    path = Path(artifacts.out_dir) / "delay_store_query_meta.json"
    payload = {
        "store_dir": str(Path(artifacts.out_dir).resolve()),
        "delay_file": str(Path(artifacts.delay_path).name),
        "edges_file": str(Path(artifacts.edges_csv_path).name),
        "edge_index_matrix_file": str(Path(edge_index_path).name),
        "time_indices_file": str(Path(artifacts.time_indices_path).name),
        "times_file": str(Path(artifacts.times_path).name),
        "query_rule": "edge_idx = edge_index_matrix[src_node, dst_node]; delay_ms = edge_delay_ms[row, edge_idx]",
        "undirected_edge_lookup": not bool(cfg.edges.directed_storage),
        "options": list(cfg.edges.options),
        "time_start": int(artifacts.time_indices[0]),
        "time_end": int(artifacts.time_indices[-1]),
        "num_steps": int(len(artifacts.time_indices)),
        "num_edges": int(artifacts.edge_table.num_edges),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def build_delay_store(cfg: DelayStoreBuildConfig) -> Any:
    validate_constellation(cfg)
    if cfg.schema_version != 1:
        raise ValueError(f"Unsupported schema_version={cfg.schema_version}")
    if cfg.time.step != 1:
        raise ValueError("This builder currently expects time.step=1")
    if cfg.time.stride < 1:
        raise ValueError("time.stride must be >= 1")
    if cfg.time.end < cfg.time.start:
        raise ValueError("time.end must be >= time.start")
    if cfg.runtime.mode not in {"memory", "memmap"}:
        raise ValueError("runtime.mode must be 'memory' or 'memmap'")

    position_cache_dir = ensure_position_cache(cfg)
    if cfg.build.validate_position_cache:
        validate_position_cache(cfg, position_cache_dir)

    out_dir = cfg.paths.out_dir or default_delay_out_dir(cfg)
    artifacts = build_or_load_artifacts(
        cache_dir=position_cache_dir,
        out_dir=out_dir,
        config=G60_CONFIG,
        start=cfg.time.start,
        end=cfg.time.end,
        stride=cfg.time.stride,
        force=cfg.build.force_delay_cache,
        chunk_steps=cfg.runtime.chunk_steps,
        allow_incomplete_cache=False,
        options=cfg.edges.options,
    )

    edge_index_path = write_edge_index_matrix(artifacts, cfg)
    query_meta_path = write_query_meta(artifacts, edge_index_path, cfg)
    run_config_path = Path(artifacts.out_dir) / "run_config.yaml"
    write_yaml(run_config_path, config_to_plain_dict(cfg))

    print(f"[delay-build] Delay store ready: {artifacts.out_dir}", flush=True)
    print(f"[delay-build] delay={artifacts.delay_path}", flush=True)
    print(f"[delay-build] edges={artifacts.edges_csv_path}", flush=True)
    print(f"[delay-build] edge_index={edge_index_path}", flush=True)
    print(f"[delay-build] query_meta={query_meta_path}", flush=True)
    print(f"[delay-build] run_config={run_config_path}", flush=True)
    return artifacts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a reusable full-option ISL edge-delay store from YAML.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-steps", type=int, default=None)
    parser.add_argument("--force-position-cache", action="store_true")
    parser.add_argument("--force-delay-cache", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    return parser.parse_args(argv)


def load_config(args: argparse.Namespace) -> DelayStoreBuildConfig:
    raw = load_yaml_dict(args.config)
    overrides: dict[str, dict[str, Any]] = {}
    if args.start is not None:
        overrides.setdefault("time", {})["start"] = int(args.start)
    if args.end is not None:
        overrides.setdefault("time", {})["end"] = int(args.end)
    if args.stride is not None:
        overrides.setdefault("time", {})["stride"] = int(args.stride)
    if args.out_dir is not None:
        overrides.setdefault("paths", {})["out_dir"] = str(args.out_dir)
    if args.workers is not None:
        overrides.setdefault("runtime", {})["workers"] = int(args.workers)
    if args.chunk_steps is not None:
        overrides.setdefault("runtime", {})["chunk_steps"] = int(args.chunk_steps)
    if args.force_position_cache:
        overrides.setdefault("build", {})["force_position_cache"] = True
    if args.force_delay_cache:
        overrides.setdefault("build", {})["force_delay_cache"] = True

    merged = deep_update(raw, overrides)
    return raw_config_to_dataclass(merged)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args)
    if args.print_config:
        print(yaml.safe_dump(config_to_plain_dict(cfg), allow_unicode=True, sort_keys=False))
    build_delay_store(cfg)
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
