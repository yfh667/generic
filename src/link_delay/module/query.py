from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


GENERIC_ROOT = Path(__file__).resolve().parents[3]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S


STORE_DIR_RE = re.compile(
    r"^(?P<constellation>.+)_full_options_t(?P<start>\d+)_(?P<end>\d+)_stride(?P<stride>\d+)$"
)


@dataclass(frozen=True)
class EdgeRecord:
    edge_idx: int
    src_node: int
    dst_node: int
    option: int
    src_sat_id: str
    dst_sat_id: str


class FullLinkDelayStore:
    """Read-only access to a full-option ISL delay cache."""

    def __init__(self, store_dir: str | Path):
        self.store_dir = Path(store_dir)
        self.delay_path = self.store_dir / "edge_delay_ms.npy"
        self.edges_path = self.store_dir / "edges.csv"
        self.time_indices_path = self.store_dir / "time_indices.npy"
        self.times_path = self.store_dir / "times_s.npy"
        self.meta_path = self.store_dir / "delay_meta.json"
        self.edge_index_matrix_path = self.store_dir / "edge_index_matrix.npy"

        missing = [
            path
            for path in (
                self.delay_path,
                self.edges_path,
                self.time_indices_path,
                self.times_path,
                self.meta_path,
                self.edge_index_matrix_path,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing delay store files: " + ", ".join(str(p) for p in missing))

        with self.meta_path.open("r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.delay_ms_array = np.load(self.delay_path, mmap_mode="r")
        self.time_indices = np.load(self.time_indices_path, mmap_mode="r")
        self.times_s = np.load(self.times_path, mmap_mode="r")
        self.edge_index_matrix = np.load(self.edge_index_matrix_path, mmap_mode="r")
        self.edges = self._load_edges()

    def _load_edges(self) -> list[EdgeRecord]:
        rows: list[EdgeRecord] = []
        with self.edges_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    EdgeRecord(
                        edge_idx=int(row["edge_idx"]),
                        src_node=int(row["src_node"]),
                        dst_node=int(row["dst_node"]),
                        option=int(row["option"]),
                        src_sat_id=str(row["src_sat_id"]),
                        dst_sat_id=str(row["dst_sat_id"]),
                    )
                )
        return rows

    @property
    def num_steps(self) -> int:
        return int(self.delay_ms_array.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.delay_ms_array.shape[1])

    @property
    def time_start(self) -> int:
        return int(self.time_indices[0])

    @property
    def time_end(self) -> int:
        return int(self.time_indices[-1])

    def contains_interval(self, start: int, end: int, stride: int = 1) -> bool:
        try:
            self.rows_for_interval(start, end, stride)
            return True
        except KeyError:
            return False

    def row_for_time_step(self, time_step: int) -> int:
        time_step = int(time_step)
        pos = int(np.searchsorted(self.time_indices, time_step))
        if pos >= len(self.time_indices) or int(self.time_indices[pos]) != time_step:
            raise KeyError(
                f"time_step={time_step} is not in this delay store "
                f"({int(self.time_indices[0])}..{int(self.time_indices[-1])})"
            )
        return pos

    def edge_idx(self, src_node: int, dst_node: int) -> int:
        src_node = int(src_node)
        dst_node = int(dst_node)
        if src_node < 0 or dst_node < 0:
            raise KeyError(f"node ids must be non-negative, got {src_node}, {dst_node}")
        if src_node >= self.edge_index_matrix.shape[0] or dst_node >= self.edge_index_matrix.shape[1]:
            raise KeyError(f"node pair is outside store range: {src_node}, {dst_node}")

        idx = int(self.edge_index_matrix[src_node, dst_node])
        if idx < 0:
            raise KeyError(f"node pair has no full-option edge: {src_node}, {dst_node}")
        return idx

    def delay_ms(self, time_step: int, src_node: int, dst_node: int) -> float:
        row = self.row_for_time_step(time_step)
        edge_idx = self.edge_idx(src_node, dst_node)
        return float(self.delay_ms_array[row, edge_idx])

    def distance_km(self, time_step: int, src_node: int, dst_node: int) -> float:
        return self.delay_ms(time_step, src_node, dst_node) / 1000.0 * LIGHT_SPEED_KM_S

    def delay_vector_for_edges(self, time_step: int, edges: list[tuple[int, int]]) -> np.ndarray:
        row = self.row_for_time_step(time_step)
        edge_indices = np.asarray([self.edge_idx(a, b) for a, b in edges], dtype=np.int64)
        return np.asarray(self.delay_ms_array[row, edge_indices], dtype=np.float32)

    def rows_for_interval(self, start: int, end: int, stride: int = 1) -> np.ndarray:
        if stride <= 0:
            raise ValueError("stride must be positive")
        start = int(start)
        end = int(end)
        wanted = np.arange(start, end + 1, int(stride), dtype=np.int64)
        if wanted.size == 0:
            raise KeyError(f"empty interval: {start}..{end}")

        first = self.row_for_time_step(int(wanted[0]))
        last = self.row_for_time_step(int(wanted[-1]))
        rows = np.arange(first, last + 1, int(stride), dtype=np.int64)
        if rows.size != wanted.size:
            raise KeyError(f"interval is not contiguous in this delay store: {start}..{end}, stride={stride}")
        if not np.array_equal(np.asarray(self.time_indices[rows], dtype=np.int64), wanted):
            raise KeyError(f"interval is not available in this delay store: {start}..{end}, stride={stride}")
        return rows

    def edge_indices_for_edges(self, edges: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray([self.edge_idx(a, b) for a, b in edges], dtype=np.int64)

    def delay_matrix_for_edges(
        self,
        start: int,
        end: int,
        edges: list[tuple[int, int]],
        *,
        stride: int = 1,
    ) -> np.ndarray:
        rows = self.rows_for_interval(start, end, stride)
        edge_indices = self.edge_indices_for_edges(edges)
        return np.asarray(self.delay_ms_array[np.ix_(rows, edge_indices)], dtype=np.float32)

    def edge_record(self, src_node: int, dst_node: int) -> EdgeRecord:
        return self.edges[self.edge_idx(src_node, dst_node)]


def default_store_dir(
    output_base: str | Path,
    constellation_name: str,
    start: int,
    end: int,
    stride: int = 1,
) -> Path:
    return (
        Path(output_base)
        / f"{str(constellation_name)}_full_options_t{int(start)}_{int(end)}_stride{int(stride)}"
    )


def discover_store_dirs(
    output_base: str | Path,
    *,
    constellation_name: str | None = None,
) -> list[Path]:
    output_base = Path(output_base)
    if not output_base.exists():
        return []
    records: list[tuple[int, int, int, Path]] = []
    for path in output_base.iterdir():
        if not path.is_dir():
            continue
        match = STORE_DIR_RE.match(path.name)
        if not match:
            continue
        if constellation_name is not None and match.group("constellation") != str(constellation_name):
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        stride = int(match.group("stride"))
        records.append((end - start, stride, start, path))
    return [path for *_rest, path in sorted(records)]


def open_delay_store_for_interval(
    start: int,
    end: int,
    *,
    stride: int = 1,
    store_dir: str | Path | None = None,
    output_base: str | Path | None = None,
    constellation_name: str | None = None,
) -> FullLinkDelayStore:
    candidates = []
    if store_dir is not None:
        candidates.append(Path(store_dir))
    if output_base is not None and constellation_name is not None:
        candidates.append(default_store_dir(output_base, constellation_name, start, end, stride))
    if output_base is not None:
        candidates.extend(discover_store_dirs(output_base, constellation_name=constellation_name))

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = Path(candidate)
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        store = FullLinkDelayStore(candidate)
        if store.contains_interval(start, end, stride):
            return store

    raise FileNotFoundError(
        f"No delay store covers interval {start}..{end} stride={stride}. "
        f"Tried: {', '.join(str(p) for p in candidates) if candidates else 'no candidates; pass store_dir or output_base'}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a full-option ISL delay store.")
    parser.add_argument("store_dir", type=Path, nargs="?", default=None)
    parser.add_argument("--output-base", type=Path, default=None)
    parser.add_argument("--constellation", type=str, default=None)
    parser.add_argument("--time-step", type=int, required=True)
    parser.add_argument("--src", type=int, required=True)
    parser.add_argument("--dst", type=int, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = open_delay_store_for_interval(
        args.time_step,
        args.time_step,
        store_dir=args.store_dir,
        output_base=args.output_base,
        constellation_name=args.constellation,
    )
    record = store.edge_record(args.src, args.dst)
    payload = {
        "store_dir": str(store.store_dir),
        "time_step": int(args.time_step),
        "src_node": int(args.src),
        "dst_node": int(args.dst),
        "edge_idx": int(record.edge_idx),
        "option": int(record.option),
        "delay_ms": store.delay_ms(args.time_step, args.src, args.dst),
        "distance_km": store.distance_km(args.time_step, args.src, args.dst),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
