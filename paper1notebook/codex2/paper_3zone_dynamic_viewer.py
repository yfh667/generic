from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.link_delay.module.edge_options import EdgeTable

from latitude_zone_ring_viewer import LatitudeZone, LatitudeZoneRingViewer, latitude_deg_from_xyz_km


PAPER_ZONE_COLORS = ("#2F6FED", "#1B9E77", "#E3B100")


@dataclass(frozen=True)
class PaperZoneMotif:
    lat_bottom: float
    lat_top: float
    stretch: float
    hop: float
    metric: float
    sat_1_orb_offset: int
    sat_1_sat_offset: int
    sat_2_orb_offset: int
    sat_2_sat_offset: int

    @property
    def offsets(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (
            (int(self.sat_1_orb_offset), int(self.sat_1_sat_offset)),
            (int(self.sat_2_orb_offset), int(self.sat_2_sat_offset)),
        )

    @property
    def label(self) -> str:
        return (
            f"{self.lat_bottom:g}-{self.lat_top:g} deg: "
            f"({self.sat_1_orb_offset},{self.sat_1_sat_offset}),"
            f"({self.sat_2_orb_offset},{self.sat_2_sat_offset})"
        )


@dataclass(frozen=True)
class Paper3ZoneTopologySeries:
    edge_table: EdgeTable
    edge_active_mask: np.ndarray
    edge_zone_ids: np.ndarray
    latitude_deg: np.ndarray
    latitude_zone_ids: np.ndarray
    latitude_zones: tuple[LatitudeZone, ...]
    steps: tuple[int, ...]
    motifs: tuple[PaperZoneMotif, ...]


def read_level_wise_best_motif(path: str | Path) -> tuple[PaperZoneMotif, ...]:
    motifs: list[PaperZoneMotif] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if len(row) < 9:
                raise ValueError(f"Expected 9 columns in {path}, got {len(row)}: {row}")
            motifs.append(
                PaperZoneMotif(
                    lat_bottom=float(row[0]),
                    lat_top=float(row[1]),
                    stretch=float(row[2]),
                    hop=float(row[3]),
                    metric=float(row[4]),
                    sat_1_orb_offset=int(row[5]),
                    sat_1_sat_offset=int(row[6]),
                    sat_2_orb_offset=int(row[7]),
                    sat_2_sat_offset=int(row[8]),
                )
            )
    if not motifs:
        raise ValueError(f"No motifs found in {path}")
    return tuple(motifs)


def paper_motif_signature(motifs: Sequence[PaperZoneMotif]) -> list[dict[str, float | int]]:
    return [
        {
            "lat_bottom": float(m.lat_bottom),
            "lat_top": float(m.lat_top),
            "stretch": float(m.stretch),
            "hop": float(m.hop),
            "metric": float(m.metric),
            "sat_1_orb_offset": int(m.sat_1_orb_offset),
            "sat_1_sat_offset": int(m.sat_1_sat_offset),
            "sat_2_orb_offset": int(m.sat_2_orb_offset),
            "sat_2_sat_offset": int(m.sat_2_sat_offset),
        }
        for m in motifs
    ]


def _step_stride(steps: Sequence[int]) -> int:
    if len(steps) <= 1:
        return 1
    return int(steps[1]) - int(steps[0])


def _cache_meta(
    *,
    config: ViewerConfig,
    steps: Sequence[int],
    position_cache_dir: str | Path,
    motifs: Sequence[PaperZoneMotif],
    max_isl_km: float | None,
    wrap_planes: bool,
    author_motif_file: str | Path | None = None,
) -> dict:
    return {
        "schema": "paper_3zone_topology_series_v1",
        "constellation": str(config.name),
        "P": int(config.P),
        "N": int(config.N),
        "total_sats": int(config.total_sats),
        "position_cache_dir": str(Path(position_cache_dir).resolve()),
        "author_motif_file": str(Path(author_motif_file).resolve()) if author_motif_file is not None else None,
        "steps_start": int(steps[0]),
        "steps_end": int(steps[-1]),
        "steps_stride": _step_stride(steps),
        "num_steps": len(steps),
        "max_isl_km": None if max_isl_km is None else float(max_isl_km),
        "wrap_planes": bool(wrap_planes),
        "motifs": paper_motif_signature(motifs),
    }


def _save_edge_table(path: str | Path, edge_table: EdgeTable) -> None:
    np.savez_compressed(
        Path(path),
        src=edge_table.src,
        dst=edge_table.dst,
        option=edge_table.option,
        src_plane=edge_table.src_plane,
        src_y=edge_table.src_y,
        dst_plane=edge_table.dst_plane,
        dst_y=edge_table.dst_y,
    )


def _load_edge_table(path: str | Path, config: ViewerConfig) -> EdgeTable:
    with np.load(Path(path), allow_pickle=False) as data:
        return EdgeTable(
            src=np.asarray(data["src"], dtype=np.int32),
            dst=np.asarray(data["dst"], dtype=np.int32),
            option=np.asarray(data["option"], dtype=np.int16),
            src_plane=np.asarray(data["src_plane"], dtype=np.int16),
            src_y=np.asarray(data["src_y"], dtype=np.int16),
            dst_plane=np.asarray(data["dst_plane"], dtype=np.int16),
            dst_y=np.asarray(data["dst_y"], dtype=np.int16),
            sat_ids=[str(i + 1) for i in range(int(config.total_sats))],
        )


def latitude_zones_from_motifs(motifs: Sequence[PaperZoneMotif]) -> tuple[LatitudeZone, ...]:
    zones: list[LatitudeZone] = []
    for idx, motif in enumerate(motifs):
        color = PAPER_ZONE_COLORS[idx % len(PAPER_ZONE_COLORS)]
        zones.append(
            LatitudeZone(
                name=f"{motif.lat_bottom:g}-{motif.lat_top:g} deg",
                min_abs_deg=float(motif.lat_bottom),
                max_abs_deg=float(motif.lat_top) + 1e-6,
                color=color,
            )
        )
    return tuple(zones)


def load_latitude_for_steps(
    *,
    position_cache_dir: str | Path,
    steps: Sequence[int],
    chunk_rows: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir = Path(position_cache_dir)
    positions = np.load(cache_dir / "positions_km.npy", mmap_mode="r")
    times = np.load(cache_dir / "times_s.npy", mmap_mode="r")
    requested = np.asarray([int(x) for x in steps], dtype=np.int64)
    row_ids = np.searchsorted(np.asarray(times, dtype=np.int64), requested)
    if np.any(row_ids >= len(times)) or np.any(np.asarray(times)[row_ids] != requested):
        raise ValueError("position cache does not contain all requested steps")

    latitude = np.empty((len(row_ids), int(positions.shape[1])), dtype=np.float32)
    chunk_rows = max(1, int(chunk_rows))
    for out_start in range(0, len(row_ids), chunk_rows):
        out_end = min(len(row_ids), out_start + chunk_rows)
        ids = row_ids[out_start:out_end]
        if ids.size and int(ids[-1]) - int(ids[0]) == ids.size - 1:
            pos_chunk = positions[int(ids[0]) : int(ids[-1]) + 1]
        else:
            pos_chunk = positions[ids]
        latitude[out_start:out_end] = latitude_deg_from_xyz_km(pos_chunk).astype(np.float32)
    return latitude, row_ids


def classify_paper_zones(latitude_deg: np.ndarray, motifs: Sequence[PaperZoneMotif]) -> np.ndarray:
    abs_lat = np.abs(np.asarray(latitude_deg, dtype=np.float32))
    zone_ids = np.full(abs_lat.shape, -1, dtype=np.int8)
    for idx, motif in enumerate(motifs):
        zone_ids[(abs_lat >= float(motif.lat_bottom)) & (abs_lat < float(motif.lat_top) + 1e-6)] = int(idx)
    return zone_ids


def _edge_key(u: int, v: int) -> tuple[int, int]:
    return (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))


def _edge_record(config: ViewerConfig, u: int, v: int, option: int = 10) -> tuple[int, int, int, int, int, int, int]:
    src, dst = _edge_key(int(u), int(v))
    src_plane, src_y = divmod(src, int(config.N))
    dst_plane, dst_y = divmod(dst, int(config.N))
    return src, dst, int(option), int(src_plane), int(src_y), int(dst_plane), int(dst_y)


def _edge_in_paper_zone(abs_a: float, abs_b: float, bottom: float, top: float) -> bool:
    # Same logic as satnetwork.github.io/scripts/find_multi_motifs.py:
    # reject if both endpoints are above the upper zone, or both below the lower zone;
    # then require both endpoints to be above the lower bound before adding.
    if abs_a > abs(top) and abs_b > abs(top):
        return False
    if abs_a < abs(bottom) and abs_b < abs(bottom):
        return False
    return abs_a > abs(bottom) and abs_b > abs(bottom)


def build_paper_3zone_topology_series(
    *,
    config: ViewerConfig,
    steps: Sequence[int],
    position_cache_dir: str | Path,
    motifs: Sequence[PaperZoneMotif],
    max_isl_km: float | None = None,
    wrap_planes: bool = True,
) -> Paper3ZoneTopologySeries:
    steps_tuple = tuple(int(x) for x in steps)
    latitude_deg, cache_rows = load_latitude_for_steps(position_cache_dir=position_cache_dir, steps=steps_tuple)
    latitude_zone_ids = classify_paper_zones(latitude_deg, motifs)
    latitude_zones = latitude_zones_from_motifs(motifs)

    positions = np.load(Path(position_cache_dir) / "positions_km.npy", mmap_mode="r")
    total_nodes = int(config.total_sats)
    p_count = int(config.P)
    n_count = int(config.N)
    row_edges: list[dict[tuple[int, int], int]] = []
    records_by_key: dict[tuple[int, int], tuple[int, int, int, int, int, int, int]] = {}

    for row_idx, cache_row in enumerate(cache_rows):
        abs_lat = np.abs(np.asarray(latitude_deg[row_idx], dtype=np.float32))
        pos_row = None
        degree = np.zeros(total_nodes, dtype=np.int16)
        active: dict[tuple[int, int], int] = {}
        for zone_id, motif in enumerate(motifs):
            for u in range(total_nodes):
                src_plane, src_y = divmod(int(u), n_count)
                for d_plane, d_y in motif.offsets:
                    dst_plane_raw = src_plane + int(d_plane)
                    if bool(wrap_planes):
                        dst_plane = dst_plane_raw % p_count
                    elif not (0 <= dst_plane_raw < p_count):
                        continue
                    else:
                        dst_plane = dst_plane_raw
                    dst_y = (src_y + int(d_y)) % n_count
                    v = int(dst_plane) * n_count + int(dst_y)
                    if v == u:
                        continue
                    key = _edge_key(u, v)
                    if key in active:
                        continue
                    if not _edge_in_paper_zone(
                        float(abs_lat[u]),
                        float(abs_lat[v]),
                        float(motif.lat_bottom),
                        float(motif.lat_top),
                    ):
                        continue
                    if max_isl_km is not None:
                        if pos_row is None:
                            pos_row = np.asarray(positions[int(cache_row)], dtype=np.float32)
                        dist = float(np.linalg.norm(pos_row[int(u)] - pos_row[int(v)]))
                        if dist > float(max_isl_km):
                            continue
                    if int(degree[u]) > 3 or int(degree[v]) > 3:
                        continue
                    active[key] = int(zone_id)
                    degree[u] += 1
                    degree[v] += 1
                    records_by_key.setdefault(key, _edge_record(config, key[0], key[1], option=10))
        row_edges.append(active)

    sorted_keys = sorted(records_by_key)
    key_to_col = {key: idx for idx, key in enumerate(sorted_keys)}
    edge_active_mask = np.zeros((len(steps_tuple), len(sorted_keys)), dtype=bool)
    edge_zone_ids = np.full((len(steps_tuple), len(sorted_keys)), -1, dtype=np.int8)
    for row_idx, active in enumerate(row_edges):
        for key, zone_id in active.items():
            col = key_to_col[key]
            edge_active_mask[row_idx, col] = True
            edge_zone_ids[row_idx, col] = int(zone_id)

    records = [records_by_key[key] for key in sorted_keys]
    sat_ids = [str(i + 1) for i in range(total_nodes)]
    edge_table = EdgeTable(
        src=np.asarray([x[0] for x in records], dtype=np.int32),
        dst=np.asarray([x[1] for x in records], dtype=np.int32),
        option=np.asarray([x[2] for x in records], dtype=np.int16),
        src_plane=np.asarray([x[3] for x in records], dtype=np.int16),
        src_y=np.asarray([x[4] for x in records], dtype=np.int16),
        dst_plane=np.asarray([x[5] for x in records], dtype=np.int16),
        dst_y=np.asarray([x[6] for x in records], dtype=np.int16),
        sat_ids=sat_ids,
    )
    return Paper3ZoneTopologySeries(
        edge_table=edge_table,
        edge_active_mask=edge_active_mask,
        edge_zone_ids=edge_zone_ids,
        latitude_deg=latitude_deg,
        latitude_zone_ids=latitude_zone_ids,
        latitude_zones=latitude_zones,
        steps=steps_tuple,
        motifs=tuple(motifs),
    )


def load_or_build_paper_3zone_topology_cache(
    *,
    config: ViewerConfig,
    steps: Sequence[int],
    position_cache_dir: str | Path,
    motifs: Sequence[PaperZoneMotif],
    cache_dir: str | Path | None,
    force: bool = False,
    max_isl_km: float | None = None,
    wrap_planes: bool = True,
    author_motif_file: str | Path | None = None,
) -> Paper3ZoneTopologySeries:
    if cache_dir is None:
        return build_paper_3zone_topology_series(
            config=config,
            steps=steps,
            position_cache_dir=position_cache_dir,
            motifs=motifs,
            max_isl_km=max_isl_km,
            wrap_planes=wrap_planes,
        )

    cache_dir = Path(cache_dir)
    steps_tuple = tuple(int(x) for x in steps)
    expected = _cache_meta(
        config=config,
        steps=steps_tuple,
        position_cache_dir=position_cache_dir,
        motifs=motifs,
        max_isl_km=max_isl_km,
        wrap_planes=wrap_planes,
        author_motif_file=author_motif_file,
    )
    meta_path = cache_dir / "meta.json"
    edge_table_path = cache_dir / "edge_table.npz"
    active_path = cache_dir / "edge_active_mask.npy"
    edge_zone_path = cache_dir / "edge_zone_ids.npy"
    latitude_path = cache_dir / "latitude_deg.npy"
    latitude_zone_path = cache_dir / "latitude_zone_ids.npy"
    steps_path = cache_dir / "steps.npy"
    required = (
        meta_path,
        edge_table_path,
        active_path,
        edge_zone_path,
        latitude_path,
        latitude_zone_path,
        steps_path,
    )

    if not force and all(path.exists() for path in required):
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta == expected:
            print(f"[paper-3zone] Reusing topology cache: {cache_dir}", flush=True)
            return Paper3ZoneTopologySeries(
                edge_table=_load_edge_table(edge_table_path, config),
                edge_active_mask=np.load(active_path, mmap_mode="r"),
                edge_zone_ids=np.load(edge_zone_path, mmap_mode="r"),
                latitude_deg=np.load(latitude_path, mmap_mode="r"),
                latitude_zone_ids=np.load(latitude_zone_path, mmap_mode="r"),
                latitude_zones=latitude_zones_from_motifs(motifs),
                steps=tuple(int(x) for x in np.load(steps_path, mmap_mode="r")),
                motifs=tuple(motifs),
            )

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[paper-3zone] Building topology cache: {cache_dir}", flush=True)
    series = build_paper_3zone_topology_series(
        config=config,
        steps=steps_tuple,
        position_cache_dir=position_cache_dir,
        motifs=motifs,
        max_isl_km=max_isl_km,
        wrap_planes=wrap_planes,
    )
    _save_edge_table(edge_table_path, series.edge_table)
    np.save(active_path, series.edge_active_mask)
    np.save(edge_zone_path, series.edge_zone_ids.astype(np.int8, copy=False))
    np.save(latitude_path, series.latitude_deg.astype(np.float32, copy=False))
    np.save(latitude_zone_path, series.latitude_zone_ids.astype(np.int8, copy=False))
    np.save(steps_path, np.asarray(steps_tuple, dtype=np.int32))
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected, f, ensure_ascii=False, indent=2)
    return Paper3ZoneTopologySeries(
        edge_table=_load_edge_table(edge_table_path, config),
        edge_active_mask=np.load(active_path, mmap_mode="r"),
        edge_zone_ids=np.load(edge_zone_path, mmap_mode="r"),
        latitude_deg=np.load(latitude_path, mmap_mode="r"),
        latitude_zone_ids=np.load(latitude_zone_path, mmap_mode="r"),
        latitude_zones=latitude_zones_from_motifs(motifs),
        steps=steps_tuple,
        motifs=tuple(motifs),
    )


class Paper3ZoneDynamicViewer(LatitudeZoneRingViewer):
    def __init__(self, *args, edge_zone_ids: np.ndarray, zone_motifs: Sequence[PaperZoneMotif], **kwargs):
        self.edge_zone_ids = np.asarray(edge_zone_ids)
        self.zone_motifs = tuple(zone_motifs)
        super().__init__(*args, **kwargs)

    def _build_ui(self):
        super()._build_ui()
        parts = []
        for idx, motif in enumerate(self.zone_motifs):
            color = self.latitude_zones[idx].color if idx < len(self.latitude_zones) else "#444444"
            parts.append(f'<span style="color:{color}; font-weight:700;">━</span> {motif.label}')
        parts = []
        for idx, motif in enumerate(self.zone_motifs):
            color = self.latitude_zones[idx].color if idx < len(self.latitude_zones) else "#444444"
            parts.append(f'<span style="color:{color}; font-weight:700;">line</span> {motif.label}')
        self.edge_zone_legend_label = QtWidgets.QLabel("3-zone links: " + "  ".join(parts))
        self.edge_zone_legend_label.setObjectName("statusLabel")
        self.edge_zone_legend_label.setWordWrap(True)
        self.controls_panel.layout().addWidget(self.edge_zone_legend_label)

    def is_hidden_visual_edge(self, idx: int) -> bool:
        if super().is_hidden_visual_edge(idx):
            return True
        raw_delta_y = abs(int(self.edge_table.src_y[idx]) - int(self.edge_table.dst_y[idx]))
        return raw_delta_y > int(self.config.N) / 2

    def update_step(self, row: int, *, sync_slider: bool = True):
        super().update_step(row, sync_slider=sync_slider)
        self.update_edge_zone_colors(int(self.current_row))

    def update_edge_zone_colors(self, row: int):
        if not self.edge_items:
            return
        row = int(max(0, min(int(row), self.edge_zone_ids.shape[0] - 1)))
        active_row = 0 if int(self.edge_active_mask.shape[0]) == 1 else row
        for idx, item in enumerate(self.edge_items):
            if not item.isVisible() or not bool(self.edge_active_mask[active_row, idx]):
                continue
            zone_id = int(self.edge_zone_ids[row, idx])
            if 0 <= zone_id < len(self.latitude_zones):
                color = QtGui.QColor(self.latitude_zones[zone_id].color)
            else:
                color = QtGui.QColor("#111111")
            color.setAlpha(185)
            pen = QtGui.QPen(color)
            pen.setWidthF(max(0.026, float(self.edge_width) * 1.55))
            pen.setCapStyle(QtCore.Qt.RoundCap)
            item.setPen(pen)
            item.setZValue(8 + max(zone_id, 0))
