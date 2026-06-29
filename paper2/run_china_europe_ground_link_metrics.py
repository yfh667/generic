from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path


GENERIC_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_voc_timeseries import build_station_specs
from g60_paper2_config import (
    DATA_ROOT,
    DEFAULT_GROUP_CACHE_DIR,
    DEFAULT_GW_GROUP_CACHE_DIR,
    DEFAULT_GW_XML,
    DEFAULT_POSITION_CACHE_ROOT,
    DEFAULT_XML,
    build_paper2_g60_config,
    build_paper2_gw_config,
)
from plot_g60_china_europe_shortest_metrics import build_topology_edge_table
from src.link_delay.module.delay_store import LIGHT_SPEED_KM_S
from src.link_delay.module.position_cache import open_position_cache_for_interval


PERIOD_SECONDS = 86164
WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalize_group(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


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


def json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def last_completed_step(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    last_step: int | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw = row.get("step")
            if raw not in (None, ""):
                last_step = int(float(raw))
    return last_step


def visibility_cache_path(
    *,
    xml_file: Path,
    cache_dir: Path,
    constellation_name: str,
    start: int,
    end: int,
    station_ids: list[int],
) -> Path:
    station_tag = "_".join(str(x) for x in station_ids)
    return cache_dir / f"{xml_file.stem}_{constellation_name}_station_visibility_t{start}_{end}_stations_{station_tag}.npz"


def load_or_build_station_visibility(
    *,
    xml_file: Path,
    cache_dir: Path,
    constellation_name: str,
    station_ids: list[int],
    total_sats: int,
    start: int,
    end: int,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    station_ids = [int(x) for x in station_ids]
    cache_path = visibility_cache_path(
        xml_file=xml_file,
        cache_dir=cache_dir,
        constellation_name=constellation_name,
        start=start,
        end=end,
        station_ids=station_ids,
    )
    meta_path = cache_path.with_suffix(".json")
    if cache_path.exists() and meta_path.exists() and not force:
        print(f"[paper2-ground] Reusing station visibility cache: {cache_path}", flush=True)
        data = np.load(cache_path)
        return data["steps"], data["visible_sats"], data["visible_counts"]

    target_set = set(station_ids)
    station_col = {sid: idx for idx, sid in enumerate(station_ids)}
    steps: list[int] = []
    rows: list[list[list[int]]] = []
    max_visible = 0
    t0 = time.time()
    print(f"[paper2-ground] Parsing station visibility from {xml_file} for {start}..{end}", flush=True)

    for _event, elem in ET.iterparse(str(xml_file), events=("end",)):
        if local_name(elem.tag) != "time":
            continue
        step_attr = elem.get("step")
        if step_attr is None:
            elem.clear()
            continue
        step = int(float(step_attr))
        if step < int(start):
            elem.clear()
            continue
        if step > int(end):
            elem.clear()
            break

        station_lists: list[list[int]] = [[] for _ in station_ids]
        stations_elem = None
        for child in elem:
            if local_name(child.tag) == "stations":
                stations_elem = child
                break
        if stations_elem is not None:
            for station_elem in stations_elem:
                if local_name(station_elem.tag) != "station":
                    continue
                sid_attr = station_elem.get("id")
                if sid_attr is None:
                    continue
                sid = int(float(sid_attr))
                if sid not in target_set:
                    continue
                sats = station_lists[station_col[sid]]
                for sat_elem in station_elem:
                    if local_name(sat_elem.tag) != "satellite":
                        continue
                    sat_attr = sat_elem.get("id")
                    if sat_attr is None:
                        continue
                    sat_id = int(float(sat_attr))
                    if 0 <= sat_id < int(total_sats):
                        sats.append(sat_id)
                max_visible = max(max_visible, len(sats))

        steps.append(step)
        rows.append(station_lists)
        elem.clear()
        if len(steps) % 10000 == 0:
            print(
                f"[paper2-ground] visibility {len(steps)}/{end - start + 1} "
                f"elapsed={time.time() - t0:.1f}s max_visible={max_visible}",
                flush=True,
            )

    if len(steps) != int(end - start + 1):
        raise ValueError(f"Expected {end - start + 1} steps, parsed {len(steps)} from {xml_file}")

    visible_sats = np.full((len(steps), len(station_ids), max_visible), -1, dtype=np.int32)
    visible_counts = np.zeros((len(steps), len(station_ids)), dtype=np.int16)
    for t_idx, station_lists in enumerate(rows):
        for s_idx, sats in enumerate(station_lists):
            visible_counts[t_idx, s_idx] = len(sats)
            if sats:
                visible_sats[t_idx, s_idx, : len(sats)] = np.asarray(sats, dtype=np.int32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        steps=np.asarray(steps, dtype=np.int64),
        station_ids=np.asarray(station_ids, dtype=np.int32),
        visible_sats=visible_sats,
        visible_counts=visible_counts,
    )
    json_write(
        meta_path,
        {
            "xml_file": str(xml_file),
            "constellation_name": constellation_name,
            "start": int(start),
            "end": int(end),
            "station_ids": station_ids,
            "total_sats": int(total_sats),
            "max_visible": int(max_visible),
            "cache_file": str(cache_path),
        },
    )
    print(f"[paper2-ground] Wrote station visibility cache: {cache_path}", flush=True)
    return np.asarray(steps, dtype=np.int64), visible_sats, visible_counts


def build_hop_distance(edge_src: np.ndarray, edge_dst: np.ndarray, total_nodes: int) -> np.ndarray:
    rows = np.concatenate([edge_src, edge_dst])
    cols = np.concatenate([edge_dst, edge_src])
    data = np.ones(rows.size, dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))
    dist = shortest_path(graph, directed=False, unweighted=True)
    if not np.all(np.isfinite(dist)):
        raise ValueError("Topology is disconnected; hop distance contains infinities.")
    return np.asarray(dist, dtype=np.float32)


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "reachable_station_pairs": 0,
            "mean": None,
            "min": None,
            "p90": None,
            "max": None,
        }
    return {
        "reachable_station_pairs": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def visible_nodes(visible_sats: np.ndarray, visible_counts: np.ndarray, step_idx: int, station_idx: int) -> np.ndarray:
    count = int(visible_counts[step_idx, station_idx])
    if count <= 0:
        return np.asarray([], dtype=np.int32)
    return np.asarray(visible_sats[step_idx, station_idx, :count], dtype=np.int32)


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


def compute_ground_link_timeseries(
    *,
    constellation_name: str,
    config,
    xml_file: Path,
    station_visibility_cache_dir: Path,
    position_cache_dir: Path,
    edge_delay_path: Path,
    out_dir: Path,
    start: int,
    end: int,
    source_group: str,
    target_group: str,
    force_visibility: bool = False,
    progress_every: int = 5000,
) -> Path:
    specs = build_station_specs()
    source_specs = [spec for spec in specs if normalize_group(spec.group) == normalize_group(source_group)]
    target_specs = [spec for spec in specs if normalize_group(spec.group) == normalize_group(target_group)]
    if not source_specs or not target_specs:
        raise ValueError(f"Missing station specs for {source_group}:{target_group}")
    station_specs = source_specs + target_specs
    station_ids = [int(spec.xml_station_id) for spec in station_specs]
    source_rows = np.arange(0, len(source_specs), dtype=np.int32)
    target_rows = np.arange(len(source_specs), len(station_specs), dtype=np.int32)
    station_positions = np.asarray(
        [station_ecef_km(spec.lat, spec.lon) for spec in station_specs],
        dtype=np.float32,
    )

    steps, visible_sats, visible_counts = load_or_build_station_visibility(
        xml_file=xml_file,
        cache_dir=station_visibility_cache_dir,
        constellation_name=constellation_name,
        station_ids=station_ids,
        total_sats=config.total_sats,
        start=start,
        end=end,
        force=force_visibility,
    )

    edge_table = build_topology_edge_table(config, "plus_grid")
    edge_src = np.asarray(edge_table.src, dtype=np.int32)
    edge_dst = np.asarray(edge_table.dst, dtype=np.int32)
    hop_dist = build_hop_distance(edge_src, edge_dst, int(config.total_sats))
    row_index = np.concatenate([edge_src, edge_dst])
    col_index = np.concatenate([edge_dst, edge_src])
    edge_delay_ms = np.load(edge_delay_path, mmap_mode="r")
    if edge_delay_ms.shape[0] < len(steps):
        raise ValueError(f"{edge_delay_path} has {edge_delay_ms.shape[0]} rows, need {len(steps)}")

    position_store = open_position_cache_for_interval(
        int(start),
        int(end),
        stride=1,
        cache_dir=position_cache_dir,
        cache_root=position_cache_dir.parent,
    )
    position_rows = position_store.rows_for_interval(int(start), int(end), stride=1)
    if position_rows.size != len(steps):
        raise ValueError(f"position rows {position_rows.size} != steps {len(steps)}")

    pair_key = f"{normalize_group(source_group)}_{normalize_group(target_group)}"
    pair_label = f"{source_group.title()}-{target_group.title()}"
    pair_dir = out_dir / constellation_name / "plus_grid" / f"t{start}_{end}_stride1" / "pairs" / pair_key
    final_csv = pair_dir / "timeseries.csv"
    partial_csv = pair_dir / "timeseries.partial.csv"
    if final_csv.exists() and not force_visibility:
        print(f"[paper2-ground] Reusing completed ground-link timeseries: {final_csv}", flush=True)
        return final_csv

    resume_after = last_completed_step(partial_csv)
    if resume_after is not None:
        print(f"[paper2-ground] Resuming {constellation_name} {pair_key} after step {resume_after}", flush=True)

    batch_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for t_idx, step in enumerate(steps):
        if resume_after is not None and int(step) <= int(resume_after):
            continue
        hop_values: list[float] = []
        delay_values: list[float] = []

        source_unique_parts = [
            visible_nodes(visible_sats, visible_counts, t_idx, int(row))
            for row in source_rows
        ]
        source_unique = (
            np.unique(np.concatenate([part for part in source_unique_parts if part.size]))
            if any(part.size for part in source_unique_parts)
            else np.asarray([], dtype=np.int32)
        )

        sat_positions = np.asarray(position_store.positions_km[position_rows[t_idx]], dtype=np.float32)
        source_delay_by_row = {
            int(row): ground_delay_for_nodes(
                sat_positions_km=sat_positions,
                station_position_km=station_positions[int(row)],
                nodes=visible_nodes(visible_sats, visible_counts, t_idx, int(row)),
            )
            for row in source_rows
        }
        target_delay_by_row = {
            int(row): ground_delay_for_nodes(
                sat_positions_km=sat_positions,
                station_position_km=station_positions[int(row)],
                nodes=visible_nodes(visible_sats, visible_counts, t_idx, int(row)),
            )
            for row in target_rows
        }

        if source_unique.size:
            weights = np.asarray(edge_delay_ms[t_idx, :], dtype=np.float32)
            delay_graph = csr_matrix(
                (np.concatenate([weights, weights]), (row_index, col_index)),
                shape=(int(config.total_sats), int(config.total_sats)),
            )
            delay_from_source_unique = np.asarray(
                dijkstra(delay_graph, directed=False, indices=source_unique),
                dtype=np.float32,
            )
            source_unique_lookup = {int(node): idx for idx, node in enumerate(source_unique)}
        else:
            delay_from_source_unique = np.empty((0, int(config.total_sats)), dtype=np.float32)
            source_unique_lookup = {}

        for src_row in source_rows:
            src_nodes = visible_nodes(visible_sats, visible_counts, t_idx, int(src_row))
            if src_nodes.size == 0:
                continue
            src_ground_delay = source_delay_by_row[int(src_row)]
            src_local_rows = np.asarray([source_unique_lookup[int(node)] for node in src_nodes], dtype=np.int32)

            for dst_row in target_rows:
                dst_nodes = visible_nodes(visible_sats, visible_counts, t_idx, int(dst_row))
                if dst_nodes.size == 0:
                    continue
                dst_ground_delay = target_delay_by_row[int(dst_row)]

                hop_matrix = hop_dist[np.ix_(src_nodes, dst_nodes)] + np.float32(2.0)
                hop_values.append(float(np.nanmin(hop_matrix)))

                sat_delay = delay_from_source_unique[np.ix_(src_local_rows, dst_nodes)]
                total_delay = (
                    sat_delay
                    + src_ground_delay.reshape(-1, 1)
                    + dst_ground_delay.reshape(1, -1)
                )
                delay_values.append(float(np.nanmin(total_delay)))

        hop_summary = finite_summary(hop_values)
        delay_summary = finite_summary(delay_values)
        batch_rows.append(
            {
                "constellation": constellation_name,
                "pair_key": pair_key,
                "pair_label": pair_label,
                "step": int(step),
                "hour": float(int(step) / 3600.0),
                "source_station_count": int(len(source_rows)),
                "target_station_count": int(len(target_rows)),
                "station_pair_count": int(len(source_rows) * len(target_rows)),
                "hop_reachable_station_pairs": hop_summary["reachable_station_pairs"],
                "mean_shortest_hops_with_ground": hop_summary["mean"],
                "min_shortest_hops_with_ground": hop_summary["min"],
                "p90_shortest_hops_with_ground": hop_summary["p90"],
                "max_shortest_hops_with_ground": hop_summary["max"],
                "delay_reachable_station_pairs": delay_summary["reachable_station_pairs"],
                "mean_shortest_delay_ms_with_ground": delay_summary["mean"],
                "min_shortest_delay_ms_with_ground": delay_summary["min"],
                "p90_shortest_delay_ms_with_ground": delay_summary["p90"],
                "max_shortest_delay_ms_with_ground": delay_summary["max"],
            }
        )
        if len(batch_rows) >= int(progress_every) or t_idx + 1 == len(steps):
            append_csv_rows(partial_csv, batch_rows)
            batch_rows.clear()
        if (t_idx + 1) % int(progress_every) == 0 or t_idx + 1 == len(steps):
            print(
                f"[paper2-ground] {constellation_name} {pair_key} {t_idx + 1}/{len(steps)} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    if batch_rows:
        append_csv_rows(partial_csv, batch_rows)
        batch_rows.clear()
    if last_completed_step(partial_csv) != int(end):
        raise RuntimeError(f"{partial_csv} is incomplete after computation.")
    partial_csv.replace(final_csv)
    json_write(
        pair_dir / "meta.json",
        {
            "constellation": constellation_name,
            "pair_key": pair_key,
            "source_group": source_group,
            "target_group": target_group,
            "start": int(start),
            "end": int(end),
            "topology": "plus_grid",
            "metric_definition": (
                "For each ground-station pair, minimize over currently visible source/target satellites. "
                "Hops add two ground-satellite hops. Delay adds source ground-to-satellite and "
                "target satellite-to-ground propagation delays."
            ),
            "station_specs": [asdict(spec) for spec in station_specs],
            "xml_file": str(xml_file),
            "position_cache_dir": str(position_store.cache_dir),
            "edge_delay_path": str(edge_delay_path),
        },
    )
    return final_csv


def load_metric_series(path: Path, column: str, *, period: int) -> np.ndarray:
    steps: list[int] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            step = int(float(row["step"]))
            if 0 <= step < period:
                steps.append(step)
                raw = row[column]
                values.append(float(raw) if raw not in {"", "None", "nan", "NaN"} else float("nan"))
    if len(values) != period:
        raise ValueError(f"{path}: expected {period} rows, got {len(values)}")
    order = np.argsort(np.asarray(steps, dtype=np.int32))
    return np.asarray(values, dtype=np.float32)[order]


def eval_all_shifts(g60: np.ndarray, gw: np.ndarray, *, chunk: int = 128) -> np.ndarray:
    shifts = np.arange(g60.size, dtype=np.int32)
    base = np.arange(g60.size, dtype=np.int32)
    out = np.empty(g60.size, dtype=np.float64)
    for start in range(0, g60.size, int(chunk)):
        ss = shifts[start : start + int(chunk)]
        idx = (base[None, :] + ss[:, None]) % g60.size
        out[start : start + int(chunk)] = np.nanmean(np.minimum(g60[None, :], gw[idx]), axis=1)
    return out


def plot_series(
    *,
    out_path: Path,
    title: str,
    ylabel: str,
    g60: np.ndarray,
    gw: np.ndarray,
    gw_shift: np.ndarray | None = None,
    shift_label: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = np.arange(g60.size, dtype=np.float32) / 3600.0
    fig, ax = plt.subplots(figsize=(15.5, 5.8), dpi=180)
    ax.plot(hours, g60, linewidth=0.8, color="#1E88E5", label=f"G60 mean={np.nanmean(g60):.3f}")
    ax.plot(hours, gw, linewidth=0.8, color="#555555", label=f"GW raw mean={np.nanmean(gw):.3f}")
    if gw_shift is not None:
        ax.plot(hours, gw_shift, linewidth=0.8, color="#D81B60", label=shift_label or "GW shifted")
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def run_phase_optimization(*, g60_csv: Path, gw_csv: Path, out_dir: Path, period: int) -> None:
    metrics = [
        ("hops", "mean_shortest_hops_with_ground", "mean shortest hops with ground links"),
        ("delay_ms", "mean_shortest_delay_ms_with_ground", "mean shortest delay with ground links (ms)"),
    ]
    rows: list[dict[str, Any]] = []
    for metric_key, column, label in metrics:
        g60 = load_metric_series(g60_csv, column, period=period)
        gw = load_metric_series(gw_csv, column, period=period)
        scan = eval_all_shifts(g60, gw)
        best_shift = int(np.nanargmin(scan))
        best_mean = float(scan[best_shift])
        shift0 = float(scan[0])
        np.savez_compressed(
            out_dir / f"china_europe_ground_link_{metric_key}_exact1s_scan.npz",
            shifts=np.arange(period, dtype=np.int32),
            envelope_mean=scan,
        )
        rows.append(
            {
                "metric": metric_key,
                "column": column,
                "best_shift_seconds": best_shift,
                "best_shift_hours": best_shift / 3600.0,
                "best_shift_degrees": best_shift / period * 360.0,
                "best_envelope_mean": best_mean,
                "shift0_envelope_mean": shift0,
                "delta_best_minus_shift0": best_mean - shift0,
                "delta_percent_vs_shift0": (best_mean - shift0) / shift0 * 100.0,
                "single_g60_mean": float(np.nanmean(g60)),
                "single_gw_mean": float(np.nanmean(gw)),
            }
        )
        plot_series(
            out_path=out_dir / "figures" / f"china_europe_ground_link_{metric_key}_raw.png",
            title=f"China-Europe {label} | raw G60 vs raw GW",
            ylabel=label,
            g60=g60,
            gw=gw,
        )
        plot_series(
            out_path=out_dir / "figures" / f"china_europe_ground_link_{metric_key}_best_shift.png",
            title=(
                f"China-Europe {label} | GW shifted by {best_shift}s "
                f"({best_shift / period * 360.0:.3f} deg)"
            ),
            ylabel=label,
            g60=g60,
            gw=gw,
            gw_shift=np.roll(gw, -best_shift),
            shift_label=f"GW shifted mean={np.nanmean(gw):.3f}",
        )

    write_csv(out_dir / "china_europe_ground_link_phase_optimization_summary.csv", rows)
    json_write(
        out_dir / "method.json",
        {
            "period_seconds": int(period),
            "definition": "G60 fixed, GW shifted. Envelope is min(G60(t), GW((t+shift) mod period)).",
            "metrics": [row["column"] for row in rows],
            "g60_csv": str(g60_csv),
            "gw_csv": str(gw_csv),
        },
    )
    print(f"[paper2-ground] phase summary={out_dir / 'china_europe_ground_link_phase_optimization_summary.csv'}")
    for row in rows:
        print(
            f"[paper2-ground] {row['metric']}: shift={row['best_shift_seconds']}s "
            f"deg={row['best_shift_degrees']:.6f} best={row['best_envelope_mean']:.6f} "
            f"shift0={row['shift0_envelope_mean']:.6f}",
            flush=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Paper2 China-Europe metrics with ground links.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=PERIOD_SECONDS - 1)
    parser.add_argument("--source-group", type=str, default="china")
    parser.add_argument("--target-group", type=str, default="europe")
    parser.add_argument("--out-root", type=Path, default=DATA_ROOT / "outputs" / "paper2_ground_link_metrics")
    parser.add_argument(
        "--visibility-cache-root",
        type=Path,
        default=DATA_ROOT / "cache" / "paper2_ground_link_metrics" / "station_visibility",
    )
    parser.add_argument("--force-visibility", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = int(args.start)
    end = int(args.end)
    if end < start:
        raise ValueError("--end must be >= --start")
    period = end - start + 1

    configs = {
        "G60": {
            "config": build_paper2_g60_config(),
            "xml_file": DEFAULT_XML,
            "visibility_cache_dir": Path(args.visibility_cache_root) / "G60",
            "position_cache_dir": DEFAULT_POSITION_CACHE_ROOT / "cache_0_86400_1s",
            "edge_delay_path": DATA_ROOT
            / "cache"
            / "paper2_shortest_metrics"
            / "edge_delay"
            / "G60"
            / "plus_grid"
            / "t0_86164_stride1"
            / "edge_delay_ms.npy",
        },
        "GW": {
            "config": build_paper2_gw_config(),
            "xml_file": DEFAULT_GW_XML,
            "visibility_cache_dir": Path(args.visibility_cache_root) / "GW",
            "position_cache_dir": DATA_ROOT / "cache" / "position_cache" / "GW_baseRaan_0" / "cache_0_86400_1s",
            "edge_delay_path": DATA_ROOT
            / "cache"
            / "paper2_shortest_metrics"
            / "edge_delay"
            / "GW"
            / "plus_grid"
            / "t0_86400_stride1"
            / "edge_delay_ms.npy",
        },
    }

    csvs: dict[str, Path] = {}
    for name, payload in configs.items():
        csvs[name] = compute_ground_link_timeseries(
            constellation_name=name,
            config=payload["config"],
            xml_file=Path(payload["xml_file"]),
            station_visibility_cache_dir=Path(payload["visibility_cache_dir"]),
            position_cache_dir=Path(payload["position_cache_dir"]),
            edge_delay_path=Path(payload["edge_delay_path"]),
            out_dir=Path(args.out_root),
            start=start,
            end=end,
            source_group=args.source_group,
            target_group=args.target_group,
            force_visibility=bool(args.force_visibility),
        )

    phase_dir = (
        Path(args.out_root)
        / "phase_envelope"
        / f"G60_fixed_GW_plus_grid_{normalize_group(args.source_group)}_{normalize_group(args.target_group)}_ground_link_t{start}_{end}_stride1"
    )
    phase_dir.mkdir(parents=True, exist_ok=True)
    run_phase_optimization(g60_csv=csvs["G60"], gw_csv=csvs["GW"], out_dir=phase_dir, period=period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
