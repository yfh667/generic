from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSITION_CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "postprocess"
    / "full_option_edge_delay"
    / "_position_cache"
)
DEFAULT_FULL_POSITION_CACHE_DIR = DEFAULT_POSITION_CACHE_ROOT / "cache_0_86164_1s"
CACHE_DIR_RE = re.compile(r"^cache_(?P<start>\d+)_(?P<end>\d+)_(?P<step>\d+)s$")


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


def default_cache_dir(start: int, end: int, step: int = 1) -> Path:
    return DEFAULT_POSITION_CACHE_ROOT / f"cache_{int(start)}_{int(end)}_{int(step)}s"


def discover_cache_dirs() -> list[Path]:
    if not DEFAULT_POSITION_CACHE_ROOT.exists():
        return []
    records: list[tuple[int, int, int, Path]] = []
    for path in DEFAULT_POSITION_CACHE_ROOT.iterdir():
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
) -> PositionCacheStore:
    candidates = []
    if cache_dir is not None:
        candidates.append(Path(cache_dir))
    candidates.append(default_cache_dir(start, end, stride))
    candidates.append(DEFAULT_FULL_POSITION_CACHE_DIR)
    candidates.extend(discover_cache_dirs())

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
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a satellite position cache.")
    parser.add_argument("cache_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--time-step", type=int, required=True)
    parser.add_argument("--node", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_position_cache_for_interval(args.time_step, args.time_step, cache_dir=args.cache_dir)
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
