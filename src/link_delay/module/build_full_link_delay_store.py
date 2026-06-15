from __future__ import annotations

import argparse
import copy
import multiprocessing as mp
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.delay_store import build_or_load_artifacts, write_edge_index_matrix, write_query_meta
from src.link_delay.module.position_cache import ensure_position_cache, validate_position_cache


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
    wrap_planes: bool


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


def write_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def deep_update(base: dict, updates: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
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

    missing = [key for key in ("name", "P", "N") if key not in c]
    if missing:
        raise ValueError(f"constellation.{', constellation.'.join(missing)} is required")
    name = str(c["name"])
    P = int(c["P"])
    N = int(c["N"])
    total_sats = int(c.get("total_sats", P * N))

    return DelayStoreBuildConfig(
        schema_version=int(raw.get("schema_version", 1)),
        constellation=ConstellationSpec(
            name=name,
            P=P,
            N=N,
            total_sats=total_sats,
        ),
        edges=EdgeSpec(
            options=tuple(int(x) for x in e.get("options", [0, 1, 2, 4])),
            directed_storage=bool(e.get("directed_storage", False)),
            wrap_planes=bool(e.get("wrap_planes", False)),
        ),
        time=TimeSpec(
            start=int(t.get("start", 0)),
            end=int(t.get("end", 86164)),
            step=int(t.get("step", 1)),
            stride=int(t.get("stride", 1)),
        ),
        paths=PathSpec(
            ephem_dir=resolve_path(p.get("ephem_dir"))
            or (PROJECT_ROOT / "data" / "basic_file" / name / "satellitesposition" / "satellite_pos"),
            position_cache_root=resolve_path(p.get("position_cache_root"))
            or (PROJECT_ROOT / "data" / "basic_file" / name / "satellitesposition" / "_position_cache"),
            delay_output_base=resolve_path(p.get("delay_output_base"))
            or (PROJECT_ROOT / "data" / "basic_file" / name / "satellitesposition" / "full_option_edge_delay"),
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


def default_delay_out_dir(cfg: DelayStoreBuildConfig) -> Path:
    wrap_label = "_wrap" if bool(cfg.edges.wrap_planes) else ""
    return (
        Path(cfg.paths.delay_output_base)
        / f"{cfg.constellation.name}_full_options{wrap_label}_t{cfg.time.start}_{cfg.time.end}_stride{cfg.time.stride}"
    )


def validate_constellation(cfg: DelayStoreBuildConfig) -> None:
    if cfg.constellation.P <= 0 or cfg.constellation.N <= 0:
        raise ValueError("constellation.P and constellation.N must be positive")
    expected_total = int(cfg.constellation.P) * int(cfg.constellation.N)
    if int(cfg.constellation.total_sats) != expected_total:
        raise ValueError(
            f"constellation.total_sats must equal P*N: got {cfg.constellation.total_sats}, "
            f"expected={expected_total}"
        )


def viewer_config_from_build_config(cfg: DelayStoreBuildConfig) -> ViewerConfig:
    return ViewerConfig(
        name=cfg.constellation.name,
        N=int(cfg.constellation.N),
        P=int(cfg.constellation.P),
        station_groups={},
        group_colors=[],
    )


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

    viewer_config = viewer_config_from_build_config(cfg)
    position_cache_dir = ensure_position_cache(
        ephem_dir=cfg.paths.ephem_dir,
        cache_root=cfg.paths.position_cache_root,
        start=cfg.time.start,
        end=cfg.time.end,
        step=cfg.time.step,
        total_sats=cfg.constellation.total_sats,
        workers=cfg.runtime.workers,
        progress_every=cfg.runtime.progress_every,
        mode=cfg.runtime.mode,
        force=cfg.build.force_position_cache,
    )
    if cfg.build.validate_position_cache:
        validate_position_cache(
            position_cache_dir,
            total_sats=cfg.constellation.total_sats,
            start=cfg.time.start,
            end=cfg.time.end,
            step=cfg.time.step,
            reject_zero_position_rows=cfg.build.reject_zero_position_rows,
        )

    out_dir = cfg.paths.out_dir or default_delay_out_dir(cfg)
    artifacts = build_or_load_artifacts(
        cache_dir=position_cache_dir,
        out_dir=out_dir,
        config=viewer_config,
        start=cfg.time.start,
        end=cfg.time.end,
        stride=cfg.time.stride,
        force=cfg.build.force_delay_cache,
        chunk_steps=cfg.runtime.chunk_steps,
        allow_incomplete_cache=False,
        options=cfg.edges.options,
        wrap_planes=bool(cfg.edges.wrap_planes),
    )

    edge_index_path = write_edge_index_matrix(
        artifacts,
        total_sats=cfg.constellation.total_sats,
        directed_storage=cfg.edges.directed_storage,
    )
    query_meta_path = write_query_meta(
        artifacts,
        edge_index_path,
        options=cfg.edges.options,
        directed_storage=cfg.edges.directed_storage,
        wrap_planes=bool(cfg.edges.wrap_planes),
    )
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
    parser.add_argument("--config", type=Path, required=True)
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
