from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = GENERIC_ROOT.parent
PAPER3PY_DIR = GENERIC_ROOT / "paper3py"

CACHE_DIR_RE = re.compile(r"^cache_(?P<start>\d+)_(?P<end>\d+)_(?P<step>\d+)s$")


@dataclass(frozen=True)
class PositionCache:
    cache_dir: Path
    positions_km: np.ndarray
    times_s: np.ndarray
    sat_ids: list[str]
    meta: dict
    build_report: dict | None


def read_json(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_build_report(cache_dir: str | Path) -> dict | None:
    return read_json(Path(cache_dir) / "build_report.json")


def cache_dir_for(root: str | Path, start: int, end: int, step: int = 1) -> Path:
    return Path(root) / f"cache_{int(start)}_{int(end)}_{int(step)}s"


def completed_position_cache(cache_dir: str | Path) -> bool:
    cache_dir = Path(cache_dir)
    report = load_build_report(cache_dir)
    if not isinstance(report, dict):
        return False
    if report.get("status") != "completed" or not bool(report.get("build_succeeded", False)):
        return False
    required = ("positions_km.npy", "times_s.npy", "sat_ids.json", "cache_meta.json")
    return all((cache_dir / name).exists() for name in required)


def import_position_cache_builder():
    if str(PAPER3PY_DIR) not in sys.path:
        sys.path.insert(0, str(PAPER3PY_DIR))
    return importlib.import_module("build_cache_parallel")


def ensure_position_cache(
    *,
    ephem_dir: str | Path,
    cache_root: str | Path,
    start: int,
    end: int,
    step: int = 1,
    total_sats: int | None = None,
    workers: int = 8,
    progress_every: int = 32,
    mode: str = "memory",
    force: bool = False,
) -> Path:
    cache_dir = cache_dir_for(cache_root, start, end, step)
    if completed_position_cache(cache_dir) and not force:
        print(f"[delay-build] Reusing completed position cache: {cache_dir}", flush=True)
        return cache_dir

    builder = import_position_cache_builder()
    builder.SIM_OUTPUT_DIR = Path(cache_root)
    builder.CACHE_DIR = cache_dir

    print(f"[delay-build] Building position cache: {cache_dir}", flush=True)
    print(f"[delay-build] ephem_dir={Path(ephem_dir)}", flush=True)
    builder.build_cache_parallel(
        Path(ephem_dir),
        int(start),
        int(end),
        int(step),
        workers=int(workers),
        progress_every=int(progress_every),
        mode=str(mode),
        flush_every=64,
        expected_sat_count=None if total_sats is None else int(total_sats),
    )
    return cache_dir


def load_position_cache(cache_dir: str | Path, *, allow_incomplete_cache: bool = False) -> PositionCache:
    cache_dir = Path(cache_dir)
    positions_path = cache_dir / "positions_km.npy"
    times_path = cache_dir / "times_s.npy"
    sat_ids_path = cache_dir / "sat_ids.json"
    meta_path = cache_dir / "cache_meta.json"

    missing = [p for p in (positions_path, times_path, sat_ids_path, meta_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing cache files: " + ", ".join(str(p) for p in missing))

    positions_km = np.load(positions_path, mmap_mode="r")
    times_s = np.load(times_path, mmap_mode="r")
    with sat_ids_path.open("r", encoding="utf-8") as f:
        sat_ids = [str(x) for x in json.load(f)]
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    build_report = load_build_report(cache_dir)

    if build_report is not None:
        status = str(build_report.get("status", "")).lower()
        if status and status != "completed" and not allow_incomplete_cache:
            raise RuntimeError(
                f"Cache {cache_dir} is marked status={status!r}, not 'completed'. "
                "This usually means some satellite rows may still be zero. "
                "Rebuild the cache or pass allow_incomplete_cache=True only for inspection."
            )

    if positions_km.ndim != 3 or positions_km.shape[-1] != 3:
        raise ValueError(f"Expected positions shape (time, sat, xyz), got {positions_km.shape}")
    if positions_km.shape[0] != times_s.shape[0]:
        raise ValueError(
            f"positions time dimension {positions_km.shape[0]} != times_s length {times_s.shape[0]}"
        )
    if positions_km.shape[1] != len(sat_ids):
        raise ValueError(f"positions sat dimension {positions_km.shape[1]} != sat_ids length {len(sat_ids)}")

    return PositionCache(
        cache_dir=cache_dir,
        positions_km=positions_km,
        times_s=times_s,
        sat_ids=sat_ids,
        meta=meta,
        build_report=build_report,
    )


def validate_position_cache(
    cache_dir: str | Path,
    *,
    total_sats: int,
    start: int,
    end: int,
    step: int = 1,
    reject_zero_position_rows: bool = True,
) -> None:
    positions = np.load(Path(cache_dir) / "positions_km.npy", mmap_mode="r")
    expected_steps = int((int(end) - int(start)) / int(step)) + 1
    expected_shape = (expected_steps, int(total_sats), 3)
    if positions.shape != expected_shape:
        raise ValueError(f"position cache shape mismatch: got {positions.shape}, expected={expected_shape}")

    if reject_zero_position_rows:
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
    print(f"[delay-build] Position cache validated: {Path(cache_dir)}", flush=True)


class PositionCacheStore:
    """Read-only mmap access to a satellite position cache."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.positions_path = self.cache_dir / "positions_km.npy"
        self.times_path = self.cache_dir / "times_s.npy"
        self.sat_ids_path = self.cache_dir / "sat_ids.json"
        self.meta_path = self.cache_dir / "cache_meta.json"
        missing = [
            p
            for p in (self.positions_path, self.times_path, self.sat_ids_path, self.meta_path)
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing position cache files: " + ", ".join(str(p) for p in missing))

        self.positions_km = np.load(self.positions_path, mmap_mode="r")
        self.times_s = np.load(self.times_path, mmap_mode="r")
        with self.sat_ids_path.open("r", encoding="utf-8") as f:
            self.sat_ids = [str(x) for x in json.load(f)]
        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)

    @property
    def time_start(self) -> int:
        return int(self.times_s[0])

    @property
    def time_end(self) -> int:
        return int(self.times_s[-1])

    @property
    def num_steps(self) -> int:
        return int(self.positions_km.shape[0])

    @property
    def num_sats(self) -> int:
        return int(self.positions_km.shape[1])

    def row_for_time_step(self, time_step: int) -> int:
        time_step = int(time_step)
        pos = int(np.searchsorted(self.times_s, float(time_step)))
        if pos >= len(self.times_s) or int(self.times_s[pos]) != time_step:
            raise KeyError(
                f"time_step={time_step} is not in this position cache "
                f"({self.time_start}..{self.time_end})"
            )
        return pos

    def rows_for_interval(self, start: int, end: int, stride: int = 1) -> np.ndarray:
        if stride <= 0:
            raise ValueError("stride must be positive")
        wanted = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
        if wanted.size == 0:
            raise KeyError(f"empty interval: {start}..{end}")
        first = self.row_for_time_step(int(wanted[0]))
        last = self.row_for_time_step(int(wanted[-1]))
        rows = np.arange(first, last + 1, int(stride), dtype=np.int64)
        if rows.size != wanted.size:
            raise KeyError(f"interval is not contiguous in this position cache: {start}..{end}")
        if not np.array_equal(np.asarray(self.times_s[rows], dtype=np.int64), wanted):
            raise KeyError(f"interval is not available in this position cache: {start}..{end}, stride={stride}")
        return rows

    def position_km(self, time_step: int, node: int) -> np.ndarray:
        return np.asarray(self.positions_km[self.row_for_time_step(time_step), int(node), :], dtype=np.float32)

    def positions_for_interval(
        self,
        start: int,
        end: int,
        *,
        stride: int = 1,
        nodes: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        rows = self.rows_for_interval(start, end, stride)
        if nodes is None:
            return np.asarray(self.positions_km[rows, :, :], dtype=np.float32)
        node_idx = np.asarray(nodes, dtype=np.int64)
        return np.asarray(self.positions_km[np.ix_(rows, node_idx, np.arange(3))], dtype=np.float32)


def default_cache_dir(cache_root: str | Path, start: int, end: int, step: int = 1) -> Path:
    return Path(cache_root) / f"cache_{int(start)}_{int(end)}_{int(step)}s"


def discover_cache_dirs(cache_root: str | Path) -> list[Path]:
    cache_root = Path(cache_root)
    if not cache_root.exists():
        return []
    records: list[tuple[int, int, int, Path]] = []
    for path in cache_root.iterdir():
        if not path.is_dir():
            continue
        match = CACHE_DIR_RE.match(path.name)
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        step = int(match.group("step"))
        records.append((end - start, step, start, path))
    return [path for *_rest, path in sorted(records)]


def open_position_cache_for_interval(
    start: int,
    end: int,
    *,
    stride: int = 1,
    cache_dir: str | Path | None = None,
    cache_root: str | Path | None = None,
    full_cache_dir: str | Path | None = None,
) -> PositionCacheStore:
    candidates = []
    if cache_dir is not None:
        candidates.append(Path(cache_dir))
    if cache_root is not None:
        candidates.append(default_cache_dir(cache_root, start, end, 1))
        candidates.extend(discover_cache_dirs(cache_root))
    if full_cache_dir is not None:
        candidates.append(Path(full_cache_dir))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        store = PositionCacheStore(candidate)
        try:
            store.rows_for_interval(start, end, stride)
            return store
        except KeyError:
            continue

    raise FileNotFoundError(
        f"No position cache covers interval {start}..{end} stride={stride}. "
        f"Tried: {', '.join(str(p) for p in candidates) if candidates else 'no candidates; pass cache_dir or cache_root'}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a satellite position cache.")
    parser.add_argument("cache_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--full-cache-dir", type=Path, default=None)
    parser.add_argument("--time-step", type=int, required=True)
    parser.add_argument("--node", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_position_cache_for_interval(
        args.time_step,
        args.time_step,
        cache_dir=args.cache_dir,
        cache_root=args.cache_root,
        full_cache_dir=args.full_cache_dir,
    )
    pos = store.position_km(args.time_step, args.node)
    payload = {
        "cache_dir": str(store.cache_dir),
        "time_step": int(args.time_step),
        "node": int(args.node),
        "sat_id": store.sat_ids[int(args.node)],
        "position_km": [float(x) for x in pos],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
