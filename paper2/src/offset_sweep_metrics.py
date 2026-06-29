from __future__ import annotations

import csv
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path

from build_g60_voc_timeseries import build_station_specs
from plot_g60_china_europe_shortest_metrics import build_topology_edge_table
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S


PERIOD_SECONDS = 86164
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563


@dataclass(frozen=True)
class OffsetInput:
    offset_deg: float
    label: str
    data_dir: Path
    visibility_intervals_mat: Path
    position_delta_mat: Path


@dataclass(frozen=True)
class PairMetricColumns:
    hops: str = "mean_shortest_hops"
    delay_ms: str = "mean_shortest_delay_ms_with_ground"


def normalize_group(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def pair_key(source_group: str, target_group: str) -> str:
    return f"{normalize_group(source_group)}_{normalize_group(target_group)}"


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def station_ecef_km(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    e2 = WGS84_F * (2.0 - WGS84_F)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A_KM / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (n + alt_km) * cos_lat * math.cos(lon)
    y = (n + alt_km) * cos_lat * math.sin(lon)
    z = (n * (1.0 - e2) + alt_km) * sin_lat
    return np.asarray([x, y, z], dtype=np.float32)


def parse_offset_deg_from_dir(path: str | Path) -> float:
    name = Path(path).name
    match = re.search(r"raan[_-]?([0-9]+(?:\.[0-9]+)?)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse offset degree from directory name: {name!r}")
    return float(match.group(1))


def discover_offset_inputs(
    *,
    root: str | Path,
    pattern: str = "raan_*",
    visibility_name: str = "visibility_intervals.mat",
    position_name: str = "position_delta_m.mat",
) -> list[OffsetInput]:
    root = Path(root)
    items: list[OffsetInput] = []
    for data_dir in sorted(root.glob(pattern)):
        if not data_dir.is_dir():
            continue
        visibility = data_dir / visibility_name
        position = data_dir / position_name
        if not visibility.exists() or not position.exists():
            continue
        offset = parse_offset_deg_from_dir(data_dir)
        items.append(
            OffsetInput(
                offset_deg=offset,
                label=f"raan_{offset:g}",
                data_dir=data_dir,
                visibility_intervals_mat=visibility,
                position_delta_mat=position,
            )
        )
    return sorted(items, key=lambda item: item.offset_deg)


def load_metric_csv_for_steps(
    path: str | Path,
    *,
    steps: Sequence[int] | np.ndarray,
    metric_column: str,
) -> np.ndarray:
    steps = np.asarray(steps, dtype=np.int64)
    wanted = {int(step): idx for idx, step in enumerate(steps)}
    values = np.full(steps.size, np.nan, dtype=np.float32)
    for row in read_csv_rows(path):
        step = int(float(row["step"]))
        idx = wanted.get(step)
        if idx is not None:
            raw = row.get(metric_column, "")
            values[idx] = float(raw) if raw not in ("", "None", "nan", "NaN") else np.nan
    missing = int(np.count_nonzero(~np.isfinite(values)))
    if missing:
        raise ValueError(f"{path} misses {missing} requested finite values in column {metric_column!r}")
    return values


def load_visibility_intervals_mat(path: str | Path) -> np.ndarray:
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    if "intervals" not in mat:
        raise ValueError(f"Missing variable 'intervals' in {path}")
    intervals = np.asarray(mat["intervals"], dtype=np.int64)
    if intervals.ndim != 2 or intervals.shape[1] < 4:
        raise ValueError(f"intervals must be an Nx4 array in {path}")
    return intervals[:, :4]


def group_station_specs(group_name: str):
    group = normalize_group(group_name)
    specs = [spec for spec in build_station_specs() if normalize_group(spec.group) == group]
    if not specs:
        raise ValueError(f"No station specs for group={group_name!r}")
    return specs


def build_station_visibility_from_intervals(
    *,
    intervals: np.ndarray,
    station_ids: Sequence[int],
    total_sats: int,
    steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    station_ids = [int(x) for x in station_ids]
    station_to_col = {sid: idx for idx, sid in enumerate(station_ids)}
    step_to_row = {int(step): idx for idx, step in enumerate(steps)}
    station_sets: list[list[set[int]]] = [
        [set() for _ in station_ids]
        for _ in range(int(steps.size))
    ]

    if steps.size == 0:
        return np.empty((0, len(station_ids), 0), dtype=np.int32), np.empty((0, len(station_ids)), dtype=np.int16)
    start = int(steps[0])
    end = int(steps[-1])
    stride = int(np.median(np.diff(steps))) if steps.size >= 2 else 1
    station_id_set = set(station_ids)

    for station_id, sat_id, t_start, t_stop in intervals:
        station_id = int(station_id)
        if station_id not in station_id_set:
            continue
        sat_id = int(sat_id)
        if sat_id < 0 or sat_id >= int(total_sats):
            continue
        first = max(start, int(math.ceil(float(t_start) / stride) * stride))
        last = min(end, int(math.floor(float(t_stop) / stride) * stride))
        if first > last:
            continue
        col = station_to_col[station_id]
        for step in range(first, last + 1, stride):
            row = step_to_row.get(step)
            if row is not None:
                station_sets[row][col].add(sat_id)

    max_visible = 0
    for row_sets in station_sets:
        for sats in row_sets:
            max_visible = max(max_visible, len(sats))

    visible_sats = np.full((steps.size, len(station_ids), max_visible), -1, dtype=np.int32)
    visible_counts = np.zeros((steps.size, len(station_ids)), dtype=np.int16)
    for row, row_sets in enumerate(station_sets):
        for col, sats in enumerate(row_sets):
            values = np.asarray(sorted(sats), dtype=np.int32)
            visible_counts[row, col] = int(values.size)
            if values.size:
                visible_sats[row, col, : values.size] = values
    return visible_sats, visible_counts


def visible_nodes(visible_sats: np.ndarray, visible_counts: np.ndarray, step_idx: int, station_idx: int) -> np.ndarray:
    count = int(visible_counts[step_idx, station_idx])
    if count <= 0:
        return np.asarray([], dtype=np.int32)
    return np.asarray(visible_sats[step_idx, station_idx, :count], dtype=np.int32)


def region_visible_union(visible_sats: np.ndarray, visible_counts: np.ndarray, step_idx: int) -> np.ndarray:
    parts = [
        visible_nodes(visible_sats, visible_counts, step_idx, station_idx)
        for station_idx in range(visible_counts.shape[1])
    ]
    parts = [part for part in parts if part.size]
    if not parts:
        return np.asarray([], dtype=np.int32)
    return np.unique(np.concatenate(parts)).astype(np.int32)


def build_hop_distance(edge_src: np.ndarray, edge_dst: np.ndarray, total_nodes: int) -> np.ndarray:
    rows = np.concatenate([edge_src, edge_dst])
    cols = np.concatenate([edge_dst, edge_src])
    data = np.ones(rows.size, dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))
    dist = shortest_path(graph, directed=False, unweighted=True)
    if not np.all(np.isfinite(dist)):
        raise ValueError("Topology is disconnected; hop distance contains infinities.")
    return np.asarray(dist, dtype=np.float32)


class CompactPositionDeltaMat:
    def __init__(self, path: str | Path):
        path = Path(path)
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        required = ("p0", "dX", "dY", "dZ")
        missing = [name for name in required if name not in mat]
        if missing:
            raise ValueError(f"{path} misses variables: {missing}")
        self.path = path
        self.p0 = np.asarray(mat["p0"], dtype=np.float32)
        if self.p0.shape[0] != 3:
            self.p0 = self.p0.reshape(3, -1)
        self.deltas = [
            np.asarray(mat["dX"]),
            np.asarray(mat["dY"]),
            np.asarray(mat["dZ"]),
        ]
        self.scale = 1.0
        meta = mat.get("meta")
        if meta is not None and hasattr(meta, "scaleMetersPerCount"):
            self.scale = float(getattr(meta, "scaleMetersPerCount"))
        self._cursor_step = 0
        self._cum = np.zeros((3, self.p0.shape[1]), dtype=np.float32)

    @property
    def total_sats(self) -> int:
        return int(self.p0.shape[1])

    def position_km_at_step(self, step: int) -> np.ndarray:
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        if step < self._cursor_step:
            self._cursor_step = 0
            self._cum.fill(0.0)
        if step > self._cursor_step:
            start = self._cursor_step
            end = step
            for axis, delta in enumerate(self.deltas):
                self._cum[axis] += np.sum(delta[start:end, :], axis=0, dtype=np.float32)
            self._cursor_step = step
        pos_m = self.p0 + self._cum * np.float32(self.scale)
        return (pos_m.T / np.float32(1000.0)).astype(np.float32, copy=False)


def ground_delay_for_nodes(
    *,
    sat_positions_km: np.ndarray,
    station_position_km: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    if nodes.size == 0:
        return np.asarray([], dtype=np.float32)
    diff = np.asarray(sat_positions_km[nodes], dtype=np.float32) - station_position_km.reshape(1, 3)
    dist_km = np.sqrt(np.sum(diff * diff, axis=1), dtype=np.float32)
    return (dist_km / np.float32(LIGHT_SPEED_KM_S) * np.float32(1000.0)).astype(np.float32)


def compute_offset_pair_metrics(
    *,
    offset_input: OffsetInput,
    config,
    source_group: str,
    target_group: str,
    start: int,
    end: int,
    stride: int,
    topology: str = "plus_grid",
    progress_every: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    steps = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    source_specs = group_station_specs(source_group)
    target_specs = group_station_specs(target_group)
    source_ids = [int(spec.xml_station_id) for spec in source_specs]
    target_ids = [int(spec.xml_station_id) for spec in target_specs]
    source_positions = np.asarray([station_ecef_km(spec.lat, spec.lon) for spec in source_specs], dtype=np.float32)
    target_positions = np.asarray([station_ecef_km(spec.lat, spec.lon) for spec in target_specs], dtype=np.float32)

    intervals = load_visibility_intervals_mat(offset_input.visibility_intervals_mat)
    source_visible_sats, source_visible_counts = build_station_visibility_from_intervals(
        intervals=intervals,
        station_ids=source_ids,
        total_sats=int(config.total_sats),
        steps=steps,
    )
    target_visible_sats, target_visible_counts = build_station_visibility_from_intervals(
        intervals=intervals,
        station_ids=target_ids,
        total_sats=int(config.total_sats),
        steps=steps,
    )

    edge_table = build_topology_edge_table(config, topology)
    edge_src = np.asarray(edge_table.src, dtype=np.int32)
    edge_dst = np.asarray(edge_table.dst, dtype=np.int32)
    hop_dist = build_hop_distance(edge_src, edge_dst, int(config.total_sats))
    graph_row = np.concatenate([edge_src, edge_dst])
    graph_col = np.concatenate([edge_dst, edge_src])
    position_reader = CompactPositionDeltaMat(offset_input.position_delta_mat)
    if position_reader.total_sats != int(config.total_sats):
        raise ValueError(
            f"{offset_input.position_delta_mat} total_sats={position_reader.total_sats}, "
            f"config.total_sats={config.total_sats}"
        )

    hops = np.full(steps.size, np.nan, dtype=np.float32)
    delay_ms = np.full(steps.size, np.nan, dtype=np.float32)
    source_counts = np.zeros(steps.size, dtype=np.int16)
    target_counts = np.zeros(steps.size, dtype=np.int16)

    t0 = time.time()
    for idx, step in enumerate(steps):
        source_union = region_visible_union(source_visible_sats, source_visible_counts, idx)
        target_union = region_visible_union(target_visible_sats, target_visible_counts, idx)
        source_counts[idx] = int(source_union.size)
        target_counts[idx] = int(target_union.size)
        if source_union.size and target_union.size:
            hops[idx] = float(np.mean(hop_dist[np.ix_(source_union, target_union)], dtype=np.float64))

        sat_positions = position_reader.position_km_at_step(int(step))
        edge_dist_km = np.sqrt(
            np.sum((sat_positions[edge_src] - sat_positions[edge_dst]) ** 2, axis=1),
            dtype=np.float32,
        )
        edge_delay_ms = (edge_dist_km / np.float32(LIGHT_SPEED_KM_S) * np.float32(1000.0)).astype(np.float32)

        delay_values: list[float] = []
        source_unique_parts = [
            visible_nodes(source_visible_sats, source_visible_counts, idx, station_idx)
            for station_idx in range(len(source_specs))
        ]
        source_unique = (
            np.unique(np.concatenate([part for part in source_unique_parts if part.size]))
            if any(part.size for part in source_unique_parts)
            else np.asarray([], dtype=np.int32)
        )

        if source_unique.size:
            delay_graph = csr_matrix(
                (np.concatenate([edge_delay_ms, edge_delay_ms]), (graph_row, graph_col)),
                shape=(int(config.total_sats), int(config.total_sats)),
            )
            delay_from_source_unique = np.asarray(
                dijkstra(delay_graph, directed=False, indices=source_unique),
                dtype=np.float32,
            )
            source_lookup = {int(node): row for row, node in enumerate(source_unique)}
        else:
            delay_from_source_unique = np.empty((0, int(config.total_sats)), dtype=np.float32)
            source_lookup = {}

        for src_station_idx in range(len(source_specs)):
            src_nodes = visible_nodes(source_visible_sats, source_visible_counts, idx, src_station_idx)
            if src_nodes.size == 0:
                continue
            src_ground = ground_delay_for_nodes(
                sat_positions_km=sat_positions,
                station_position_km=source_positions[src_station_idx],
                nodes=src_nodes,
            )
            src_rows = np.asarray([source_lookup[int(node)] for node in src_nodes], dtype=np.int32)
            for dst_station_idx in range(len(target_specs)):
                dst_nodes = visible_nodes(target_visible_sats, target_visible_counts, idx, dst_station_idx)
                if dst_nodes.size == 0:
                    continue
                dst_ground = ground_delay_for_nodes(
                    sat_positions_km=sat_positions,
                    station_position_km=target_positions[dst_station_idx],
                    nodes=dst_nodes,
                )
                sat_delay = delay_from_source_unique[np.ix_(src_rows, dst_nodes)]
                total_delay = sat_delay + src_ground.reshape(-1, 1) + dst_ground.reshape(1, -1)
                best = float(np.nanmin(total_delay))
                if math.isfinite(best):
                    delay_values.append(best)
        if delay_values:
            delay_ms[idx] = float(np.mean(delay_values, dtype=np.float64))

        if progress_every and ((idx + 1) % int(progress_every) == 0 or idx + 1 == steps.size):
            print(
                f"[offset-sweep] {offset_input.label} {idx + 1}/{steps.size} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    return steps, hops, delay_ms, source_counts, target_counts


def write_offset_timeseries(
    *,
    path: str | Path,
    offset_input: OffsetInput,
    pair: str,
    steps: np.ndarray,
    hops: np.ndarray,
    delay_ms: np.ndarray,
    source_counts: np.ndarray,
    target_counts: np.ndarray,
) -> Path:
    rows = []
    for idx, step in enumerate(steps):
        rows.append(
            {
                "offset_deg": float(offset_input.offset_deg),
                "offset_label": offset_input.label,
                "pair_key": pair,
                "step": int(step),
                "hour": float(int(step) / 3600.0),
                "source_region_visible_sats": int(source_counts[idx]),
                "target_region_visible_sats": int(target_counts[idx]),
                "mean_shortest_hops": float(hops[idx]) if np.isfinite(hops[idx]) else None,
                "mean_shortest_delay_ms_with_ground": float(delay_ms[idx]) if np.isfinite(delay_ms[idx]) else None,
            }
        )
    return write_csv(path, rows)


def read_or_compute_offset_metrics(
    *,
    offset_input: OffsetInput,
    config,
    source_group: str,
    target_group: str,
    start: int,
    end: int,
    stride: int,
    out_dir: str | Path,
    force: bool = False,
    progress_every: int = 200,
) -> Path:
    out_dir = Path(out_dir)
    pkey = pair_key(source_group, target_group)
    tag = f"raan_{offset_input.offset_deg:g}".replace(".", "p")
    csv_path = out_dir / "gw_offsets" / tag / pkey / "timeseries.csv"
    if csv_path.exists() and not force:
        print(f"[offset-sweep] Reusing {csv_path}", flush=True)
        return csv_path
    steps, hops, delay_ms, source_counts, target_counts = compute_offset_pair_metrics(
        offset_input=offset_input,
        config=config,
        source_group=source_group,
        target_group=target_group,
        start=start,
        end=end,
        stride=stride,
        progress_every=progress_every,
    )
    return write_offset_timeseries(
        path=csv_path,
        offset_input=offset_input,
        pair=pkey,
        steps=steps,
        hops=hops,
        delay_ms=delay_ms,
        source_counts=source_counts,
        target_counts=target_counts,
    )


def summarize_offset_sweep(
    *,
    offset_inputs: Sequence[OffsetInput],
    offset_metric_csvs: Sequence[Path],
    g60_hops_csv: str | Path,
    g60_delay_csv: str | Path,
    source_group: str,
    target_group: str,
    start: int,
    end: int,
    stride: int,
    out_dir: str | Path,
    metric_columns: PairMetricColumns = PairMetricColumns(),
    write_timeseries: bool = True,
) -> tuple[Path, Path | None]:
    out_dir = Path(out_dir)
    steps = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    g60_hops = load_metric_csv_for_steps(g60_hops_csv, steps=steps, metric_column=metric_columns.hops)
    g60_delay = load_metric_csv_for_steps(g60_delay_csv, steps=steps, metric_column=metric_columns.delay_ms)

    summary_rows: list[dict[str, Any]] = []
    envelope_rows: list[dict[str, Any]] = []
    pkey = pair_key(source_group, target_group)

    for item, csv_path in zip(offset_inputs, offset_metric_csvs):
        gw_hops = load_metric_csv_for_steps(csv_path, steps=steps, metric_column=metric_columns.hops)
        gw_delay = load_metric_csv_for_steps(csv_path, steps=steps, metric_column=metric_columns.delay_ms)
        dual_hops = np.minimum(g60_hops, gw_hops)
        dual_delay = np.minimum(g60_delay, gw_delay)
        summary_rows.append(
            {
                "pair_key": pkey,
                "source_group": normalize_group(source_group),
                "target_group": normalize_group(target_group),
                "offset_deg": float(item.offset_deg),
                "offset_label": item.label,
                "samples": int(steps.size),
                "g60_mean_hops": float(np.nanmean(g60_hops)),
                "gw_offset_mean_hops": float(np.nanmean(gw_hops)),
                "dual_min_mean_hops": float(np.nanmean(dual_hops)),
                "g60_mean_delay_ms_with_ground": float(np.nanmean(g60_delay)),
                "gw_offset_mean_delay_ms_with_ground": float(np.nanmean(gw_delay)),
                "dual_min_mean_delay_ms_with_ground": float(np.nanmean(dual_delay)),
                "gw_metric_csv": str(csv_path),
            }
        )
        if write_timeseries:
            for idx, step in enumerate(steps):
                envelope_rows.append(
                    {
                        "pair_key": pkey,
                        "offset_deg": float(item.offset_deg),
                        "offset_label": item.label,
                        "step": int(step),
                        "hour": float(int(step) / 3600.0),
                        "g60_hops": float(g60_hops[idx]),
                        "gw_offset_hops": float(gw_hops[idx]),
                        "dual_min_hops": float(dual_hops[idx]),
                        "g60_delay_ms_with_ground": float(g60_delay[idx]),
                        "gw_offset_delay_ms_with_ground": float(gw_delay[idx]),
                        "dual_min_delay_ms_with_ground": float(dual_delay[idx]),
                    }
                )

    summary_path = write_csv(out_dir / "offset_sweep_summary.csv", summary_rows)
    timeseries_path = write_csv(out_dir / "offset_sweep_envelope_timeseries.csv", envelope_rows) if write_timeseries else None
    return summary_path, timeseries_path
