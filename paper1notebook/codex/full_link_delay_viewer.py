from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.lib.format import open_memmap


GENERIC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG, ViewerConfig


LIGHT_SPEED_KM_S = 299_792.458
DEFAULT_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "satellitesposition"
    / "satellite_pos"
    / "_cache"
    / "cache_86164s_1s"
)
FALLBACK_COMPLETED_CACHE_DIR = DEFAULT_CACHE_DIR.parent / "cache_0_1000_1s"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "data" / "postprocess" / "full_option_edge_delay"

# Same option semantics as the previous full-option topology work.
OPTION_DELTAS = {
    0: (1, 0),
    1: (1, -1),
    2: (2, 0),
    4: (1, 1),
}


@dataclass(frozen=True)
class PositionCache:
    cache_dir: Path
    positions_km: np.ndarray
    times_s: np.ndarray
    sat_ids: list[str]
    meta: dict
    build_report: dict | None


@dataclass(frozen=True)
class EdgeTable:
    src: np.ndarray
    dst: np.ndarray
    option: np.ndarray
    src_plane: np.ndarray
    src_y: np.ndarray
    dst_plane: np.ndarray
    dst_y: np.ndarray
    sat_ids: list[str]

    @property
    def num_edges(self) -> int:
        return int(self.src.size)


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


def load_build_report(cache_dir: Path) -> dict | None:
    path = Path(cache_dir) / "build_report.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_completed_cache(cache_dir: Path) -> bool:
    report = load_build_report(cache_dir)
    if report is None:
        return False
    return str(report.get("status", "")).lower() == "completed"


def choose_default_cache_dir() -> Path:
    if is_completed_cache(DEFAULT_CACHE_DIR):
        return DEFAULT_CACHE_DIR
    if is_completed_cache(FALLBACK_COMPLETED_CACHE_DIR):
        print(
            "[delay] Full 86164s cache is not marked completed; "
            f"using completed preview cache: {FALLBACK_COMPLETED_CACHE_DIR}"
        )
        return FALLBACK_COMPLETED_CACHE_DIR
    return DEFAULT_CACHE_DIR


def load_position_cache(cache_dir: Path, *, allow_incomplete_cache: bool = False) -> PositionCache:
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
                "Use --cache-dir data\\basic_file\\satellitesposition\\satellite_pos\\_cache\\cache_0_1000_1s "
                "for the completed preview cache, rebuild the full cache, or pass --allow-incomplete-cache "
                "if you intentionally want to inspect it."
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


def build_full_option_edges(
    config: ViewerConfig,
    *,
    options: Iterable[int] = (0, 1, 2, 4),
    sat_ids: list[str] | None = None,
) -> EdgeTable:
    options = tuple(int(x) for x in options)
    bad = [x for x in options if x not in OPTION_DELTAS]
    if bad:
        raise ValueError(f"Unsupported options: {bad}; supported={sorted(OPTION_DELTAS)}")

    src: list[int] = []
    dst: list[int] = []
    opt_values: list[int] = []
    src_plane: list[int] = []
    src_y: list[int] = []
    dst_plane: list[int] = []
    dst_y: list[int] = []

    for p in range(int(config.P)):
        for y in range(int(config.N)):
            u = p * int(config.N) + y
            for option in options:
                dp, dy = OPTION_DELTAS[option]
                q = p + dp
                if not (0 <= q < int(config.P)):
                    continue
                yy = (y + dy) % int(config.N)
                v = q * int(config.N) + yy
                src.append(u)
                dst.append(v)
                opt_values.append(option)
                src_plane.append(p)
                src_y.append(y)
                dst_plane.append(q)
                dst_y.append(yy)

    if sat_ids is None:
        sat_ids = [str(i + 1) for i in range(int(config.total_sats))]
    if len(sat_ids) != int(config.total_sats):
        raise ValueError(f"sat_ids length {len(sat_ids)} != total_sats {config.total_sats}")

    return EdgeTable(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        option=np.asarray(opt_values, dtype=np.int16),
        src_plane=np.asarray(src_plane, dtype=np.int16),
        src_y=np.asarray(src_y, dtype=np.int16),
        dst_plane=np.asarray(dst_plane, dtype=np.int16),
        dst_y=np.asarray(dst_y, dtype=np.int16),
        sat_ids=sat_ids,
    )


def resolve_time_indices(num_cache_steps: int, start: int, end: int | None, stride: int) -> np.ndarray:
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start < 0:
        raise ValueError("--start must be >= 0")
    if end is None:
        end = num_cache_steps - 1
    if end < start:
        raise ValueError(f"--end {end} is smaller than --start {start}")
    if end >= num_cache_steps:
        raise ValueError(f"--end {end} is outside cache range 0..{num_cache_steps - 1}")
    return np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)


def default_output_dir(cache_dir: Path, start: int, end: int | None, stride: int) -> Path:
    end_label = "all" if end is None else str(end)
    return DEFAULT_OUTPUT_BASE / f"{Path(cache_dir).name}_t{start}_{end_label}_stride{stride}"


def cache_signature(
    *,
    cache: PositionCache,
    config: ViewerConfig,
    edge_table: EdgeTable,
    time_indices: np.ndarray,
    options: Iterable[int],
    stride: int,
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
        "num_edges": int(edge_table.num_edges),
        "time_start_index": int(time_indices[0]),
        "time_end_index": int(time_indices[-1]),
        "num_steps": int(time_indices.size),
        "stride": int(stride),
        "light_speed_km_s": float(LIGHT_SPEED_KM_S),
    }


def load_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_edges_csv(edge_table: EdgeTable, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "edge_idx",
                "src_node",
                "dst_node",
                "src_sat_id",
                "dst_sat_id",
                "src_plane",
                "src_y",
                "dst_plane",
                "dst_y",
                "option",
            ]
        )
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            writer.writerow(
                [
                    idx,
                    src,
                    dst,
                    edge_table.sat_ids[src],
                    edge_table.sat_ids[dst],
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                ]
            )


def existing_artifacts_match(out_dir: Path, signature: dict) -> bool:
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
    out_dir: Path,
    signature: dict,
    time_indices: np.ndarray,
    force: bool,
    chunk_steps: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    delay_path = out_dir / "edge_delay_ms.npy"
    edges_csv_path = out_dir / "edges.csv"
    time_indices_path = out_dir / "time_indices.npy"
    times_path = out_dir / "times_s.npy"
    meta_path = out_dir / "delay_meta.json"

    if not force and existing_artifacts_match(out_dir, signature):
        meta = load_meta(meta_path) or {}
        print(f"[delay] Reusing existing delay cache: {delay_path}")
        return meta

    if chunk_steps <= 0:
        raise ValueError("--chunk-steps must be positive")

    print(f"[delay] Building delay cache in {out_dir}")
    print(f"[delay] steps={time_indices.size}, edges={edge_table.num_edges}, dtype=float32")
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
            print(f"[delay] {local_end}/{time_indices.size} steps ({pct:.1f}%), elapsed={elapsed:.1f}s")

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
    print(f"[delay] Wrote {delay_path}")
    print(f"[delay] Wrote {edges_csv_path}")
    print(f"[delay] Wrote {meta_path}")
    return meta


def build_or_load_artifacts(
    *,
    cache_dir: Path,
    out_dir: Path,
    config: ViewerConfig,
    start: int,
    end: int | None,
    stride: int,
    force: bool,
    chunk_steps: int,
    allow_incomplete_cache: bool,
    options: Iterable[int] = (0, 1, 2, 4),
) -> DelayArtifacts:
    cache = load_position_cache(cache_dir, allow_incomplete_cache=allow_incomplete_cache)
    if int(cache.positions_km.shape[1]) != int(config.total_sats):
        raise ValueError(
            f"Cache has {cache.positions_km.shape[1]} satellites, but {config.name} config expects {config.total_sats}"
        )

    edge_table = build_full_option_edges(config, options=options, sat_ids=cache.sat_ids)
    time_indices = resolve_time_indices(cache.positions_km.shape[0], start, end, stride)
    signature = cache_signature(
        cache=cache,
        config=config,
        edge_table=edge_table,
        time_indices=time_indices,
        options=options,
        stride=stride,
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

    return DelayArtifacts(
        cache_dir=Path(cache_dir),
        out_dir=Path(out_dir),
        delay_path=Path(out_dir) / "edge_delay_ms.npy",
        edges_csv_path=Path(out_dir) / "edges.csv",
        meta_path=Path(out_dir) / "delay_meta.json",
        time_indices_path=Path(out_dir) / "time_indices.npy",
        times_path=Path(out_dir) / "times_s.npy",
        edge_table=edge_table,
        time_indices=np.load(Path(out_dir) / "time_indices.npy", mmap_mode="r"),
        times_s=np.load(Path(out_dir) / "times_s.npy", mmap_mode="r"),
        meta=meta,
    )


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
    out_path = artifacts.out_dir / f"edge_delay_step_{cache_step}.csv"
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
    print(f"[delay] Wrote step CSV: {out_path}")
    return out_path


def load_qt_stack():
    pyqt_error: Exception | None = None
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg

        return "PyQt5", QtCore, QtGui, QtWidgets, pg
    except Exception as exc:
        pyqt_error = exc

    try:
        from PySide6 import QtCore, QtGui, QtWidgets
        import pyqtgraph as pg

        return "PySide6", QtCore, QtGui, QtWidgets, pg
    except Exception as pyside_error:
        raise RuntimeError(
            "Cannot start GUI because neither PyQt5 nor PySide6 with pyqtgraph is importable. "
            "Run this script in the same Python environment that can run SatelliteViewer, "
            "or install PyQt5/PySide6 and pyqtgraph there. "
            f"PyQt5 error: {pyqt_error}; PySide6 error: {pyside_error}"
        ) from pyside_error


COLOR_STOPS = (
    (0.00, (49, 54, 149)),
    (0.25, (69, 117, 180)),
    (0.50, (224, 243, 248)),
    (0.70, (254, 224, 144)),
    (0.85, (244, 109, 67)),
    (1.00, (165, 0, 38)),
)


def normalized_delay(value: float, vmin: float, vmax: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if vmax <= vmin:
        return 0.0
    return max(0.0, min(1.0, (float(value) - float(vmin)) / (float(vmax) - float(vmin))))


def rgb_for_norm(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    for idx in range(len(COLOR_STOPS) - 1):
        left_t, left_rgb = COLOR_STOPS[idx]
        right_t, right_rgb = COLOR_STOPS[idx + 1]
        if left_t <= t <= right_t:
            ratio = 0.0 if right_t == left_t else (t - left_t) / (right_t - left_t)
            return tuple(
                int(round(left_rgb[channel] + ratio * (right_rgb[channel] - left_rgb[channel])))
                for channel in range(3)
            )
    return COLOR_STOPS[-1][1]


def make_viewer_class(QtCore, QtGui, QtWidgets, pg):
    class ClickableEdgeItem(QtWidgets.QGraphicsPathItem):
        def __init__(self, path, viewer, edge_idx: int):
            super().__init__(path)
            self.viewer = viewer
            self.edge_idx = int(edge_idx)
            self.setAcceptHoverEvents(True)
            self.setZValue(3)

        def mousePressEvent(self, event):
            self.viewer.select_edge(self.edge_idx)
            event.accept()

        def hoverEnterEvent(self, event):
            self.setZValue(25)
            super().hoverEnterEvent(event)

        def hoverLeaveEvent(self, event):
            if self.viewer.selected_edge_idx != self.edge_idx:
                self.setZValue(3)
            super().hoverLeaveEvent(event)

        def shape(self):
            stroker = QtGui.QPainterPathStroker()
            stroker.setWidth(0.36)
            return stroker.createStroke(self.path())

    class FullLinkDelayViewer(QtWidgets.QWidget):
        def __init__(
            self,
            *,
            config: ViewerConfig,
            edge_table: EdgeTable,
            delay_ms: np.ndarray,
            time_indices: np.ndarray,
            times_s: np.ndarray,
            vmin_ms: float,
            vmax_ms: float,
            title: str,
        ):
            pg.setConfigOption("background", "w")
            pg.setConfigOption("foreground", "#222")
            super().__init__()
            self.config = config
            self.edge_table = edge_table
            self.delay_ms = delay_ms
            self.time_indices = time_indices
            self.times_s = times_s
            self.vmin_ms = float(vmin_ms)
            self.vmax_ms = float(vmax_ms)
            self.current_row = 0
            self.selected_edge_idx: int | None = None
            self.edge_items: list[ClickableEdgeItem] = []
            self.setWindowTitle(title)
            self._build_ui()
            self._draw_nodes()
            self._draw_edges_once()
            self.update_step(0)

        def _build_ui(self):
            self.main_layout = QtWidgets.QVBoxLayout(self)
            body = QtWidgets.QHBoxLayout()
            self.main_layout.addLayout(body)

            self.plot_widget = pg.PlotWidget()
            body.addWidget(self.plot_widget, stretch=1)
            self.plot_widget.setRange(
                xRange=[-0.7, int(self.config.P) - 0.3],
                yRange=[-0.8, int(self.config.N) - 0.2],
                padding=0.02,
            )
            self.plot_widget.setLabel("bottom", "Orbit Plane Index (P)")
            self.plot_widget.setLabel("left", "Satellite Index in Plane (N)")
            self.plot_widget.showGrid(x=True, y=True, alpha=0.18)

            side = QtWidgets.QWidget()
            side.setMinimumWidth(250)
            side.setMaximumWidth(310)
            side_layout = QtWidgets.QVBoxLayout(side)
            body.addWidget(side)

            self.step_label = QtWidgets.QLabel("")
            self.step_label.setWordWrap(True)
            side_layout.addWidget(self.step_label)

            self.stats_label = QtWidgets.QLabel("")
            self.stats_label.setWordWrap(True)
            side_layout.addWidget(self.stats_label)

            side_layout.addSpacing(8)
            self.colorbar_label = QtWidgets.QLabel()
            self.colorbar_label.setPixmap(self._make_colorbar_pixmap(32, 220))
            self.colorbar_label.setAlignment(QtCore.Qt.AlignHCenter)
            side_layout.addWidget(QtWidgets.QLabel(f"{self.vmax_ms:.2f} ms"))
            side_layout.addWidget(self.colorbar_label, alignment=QtCore.Qt.AlignHCenter)
            side_layout.addWidget(QtWidgets.QLabel(f"{self.vmin_ms:.2f} ms"))

            side_layout.addSpacing(8)
            self.selected_label = QtWidgets.QLabel("")
            self.selected_label.setWordWrap(True)
            side_layout.addWidget(self.selected_label)
            side_layout.addStretch(1)

            controls = QtWidgets.QHBoxLayout()
            self.main_layout.addLayout(controls)
            self.prev_btn = QtWidgets.QPushButton("<")
            self.next_btn = QtWidgets.QPushButton(">")
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, int(self.delay_ms.shape[0]) - 1))
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self.update_step)
            self.jump_input = QtWidgets.QLineEdit()
            self.jump_input.setMaximumWidth(100)
            self.jump_input.setPlaceholderText("cache step")
            self.jump_btn = QtWidgets.QPushButton("Jump")
            self.prev_btn.clicked.connect(self.step_prev)
            self.next_btn.clicked.connect(self.step_next)
            self.jump_btn.clicked.connect(self.jump_to_cache_step)
            controls.addWidget(self.prev_btn)
            controls.addWidget(self.slider, stretch=1)
            controls.addWidget(self.next_btn)
            controls.addWidget(self.jump_input)
            controls.addWidget(self.jump_btn)

        def _draw_nodes(self):
            sat_ids = np.arange(int(self.config.total_sats), dtype=np.int32)
            cols = sat_ids // int(self.config.N)
            rows = sat_ids % int(self.config.N)
            self._all_cols = cols
            self._all_rows = rows
            spots = [
                {
                    "pos": (float(cols[i]), float(rows[i])),
                    "brush": pg.mkBrush(255, 255, 255),
                    "pen": pg.mkPen("#555", width=0.7),
                }
                for i in range(int(self.config.total_sats))
            ]
            self.scatter = pg.ScatterPlotItem(size=12)
            self.scatter.setData(spots)
            self.scatter.setZValue(12)
            self.plot_widget.addItem(self.scatter)

        def _draw_edges_once(self):
            for idx in range(self.edge_table.num_edges):
                path = self._path_for_edge(idx)
                item = ClickableEdgeItem(path, self, idx)
                self.plot_widget.addItem(item)
                self.edge_items.append(item)

        def _path_for_edge(self, idx: int):
            x0 = float(self.edge_table.src_plane[idx])
            y0 = float(self.edge_table.src_y[idx])
            x1 = float(self.edge_table.dst_plane[idx])
            y1 = float(self.edge_table.dst_y[idx])
            path = QtGui.QPainterPath()
            path.moveTo(x0, y0)
            if int(self.edge_table.option[idx]) == 2:
                ctrl_x = (x0 + x1) / 2.0
                ctrl_y = (y0 + y1) / 2.0 + 0.5 * abs(x1 - x0)
                path.quadTo(ctrl_x, ctrl_y, x1, y1)
            else:
                path.lineTo(x1, y1)
            return path

        def _qcolor_for_delay(self, value_ms: float, alpha: int = 205):
            rgb = rgb_for_norm(normalized_delay(value_ms, self.vmin_ms, self.vmax_ms))
            color = QtGui.QColor(*rgb)
            color.setAlpha(int(alpha))
            return color

        def _pen_for_delay(self, value_ms: float, *, selected: bool = False):
            if selected:
                pen = QtGui.QPen(QtGui.QColor(20, 20, 20))
                pen.setWidthF(3.8)
            else:
                pen = QtGui.QPen(self._qcolor_for_delay(value_ms))
                width = 0.75 + 1.35 * normalized_delay(value_ms, self.vmin_ms, self.vmax_ms)
                pen.setWidthF(float(width))
            pen.setCosmetic(False)
            return pen

        def _make_colorbar_pixmap(self, width: int, height: int):
            pixmap = QtGui.QPixmap(int(width), int(height))
            painter = QtGui.QPainter(pixmap)
            for y in range(int(height)):
                t = 1.0 - y / max(1, int(height) - 1)
                rgb = rgb_for_norm(t)
                painter.setPen(QtGui.QColor(*rgb))
                painter.drawLine(0, y, int(width), y)
            painter.end()
            return pixmap

        def update_step(self, row: int):
            row = int(row)
            if row < 0 or row >= int(self.delay_ms.shape[0]):
                return
            self.current_row = row
            if self.slider.value() != row:
                self.slider.blockSignals(True)
                self.slider.setValue(row)
                self.slider.blockSignals(False)

            values = np.asarray(self.delay_ms[row], dtype=np.float32)
            for idx, item in enumerate(self.edge_items):
                selected = self.selected_edge_idx == idx
                item.setPen(self._pen_for_delay(float(values[idx]), selected=selected))
                item.setZValue(30 if selected else 3)

            cache_step = int(self.time_indices[row])
            time_s = float(self.times_s[row])
            self.step_label.setText(
                f"Cache step: {cache_step}\n"
                f"Time: {time_s:.0f} s\n"
                f"Row: {row + 1}/{int(self.delay_ms.shape[0])}"
            )
            self.stats_label.setText(
                f"Edges: {self.edge_table.num_edges}\n"
                f"Delay min: {float(np.nanmin(values)):.4f} ms\n"
                f"Delay mean: {float(np.nanmean(values)):.4f} ms\n"
                f"Delay max: {float(np.nanmax(values)):.4f} ms"
            )
            self._update_selected_label()

        def select_edge(self, edge_idx: int):
            self.selected_edge_idx = int(edge_idx)
            self.update_step(self.current_row)

        def _update_selected_label(self):
            idx = self.selected_edge_idx
            if idx is None:
                self.selected_label.setText("Selected edge: none")
                return
            delay_ms = float(self.delay_ms[self.current_row, idx])
            distance_km = delay_ms / 1000.0 * LIGHT_SPEED_KM_S
            src = int(self.edge_table.src[idx])
            dst = int(self.edge_table.dst[idx])
            self.selected_label.setText(
                f"Selected edge: {idx}\n"
                f"Option: {int(self.edge_table.option[idx])}\n"
                f"{src} ({int(self.edge_table.src_plane[idx])}, {int(self.edge_table.src_y[idx])})"
                f" -> {dst} ({int(self.edge_table.dst_plane[idx])}, {int(self.edge_table.dst_y[idx])})\n"
                f"Sat IDs: {self.edge_table.sat_ids[src]} -> {self.edge_table.sat_ids[dst]}\n"
                f"Delay: {delay_ms:.6f} ms\n"
                f"Distance: {distance_km:.3f} km"
            )

        def step_prev(self):
            self.update_step(max(0, self.current_row - 1))

        def step_next(self):
            self.update_step(min(int(self.delay_ms.shape[0]) - 1, self.current_row + 1))

        def jump_to_cache_step(self):
            text = self.jump_input.text().strip()
            if not text:
                return
            try:
                cache_step = int(text)
            except ValueError:
                return
            matches = np.where(np.asarray(self.time_indices) == cache_step)[0]
            if matches.size:
                self.update_step(int(matches[0]))

    return FullLinkDelayViewer


def run_gui(
    *,
    artifacts: DelayArtifacts,
    config: ViewerConfig,
    vmin_ms: float,
    vmax_ms: float,
    width: int,
    height: int,
) -> int:
    qt_name, QtCore, QtGui, QtWidgets, pg = load_qt_stack()
    delay = np.load(artifacts.delay_path, mmap_mode="r")
    title = f"{config.name} full-option ISL delay ({qt_name})"
    Viewer = make_viewer_class(QtCore, QtGui, QtWidgets, pg)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    viewer = Viewer(
        config=config,
        edge_table=artifacts.edge_table,
        delay_ms=delay,
        time_indices=artifacts.time_indices,
        times_s=artifacts.times_s,
        vmin_ms=vmin_ms,
        vmax_ms=vmax_ms,
        title=title,
    )
    viewer.resize(int(width), int(height))
    viewer.show()
    exec_fn = getattr(app, "exec", None) or getattr(app, "exec_", None)
    return int(exec_fn())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-option inter-satellite edge propagation delays from the satellite "
            "position cache, then open an interactive delay viewer."
        )
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--chunk-steps", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-incomplete-cache", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--export-step-csv", type=int, default=None)
    parser.add_argument("--vmin-ms", type=float, default=0.0)
    parser.add_argument("--vmax-ms", type=float, default=None)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=760)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = args.cache_dir or choose_default_cache_dir()
    out_dir = args.out_dir or default_output_dir(cache_dir, args.start, args.end, args.stride)
    artifacts = build_or_load_artifacts(
        cache_dir=cache_dir,
        out_dir=out_dir,
        config=G60_CONFIG,
        start=args.start,
        end=args.end,
        stride=args.stride,
        force=args.force,
        chunk_steps=args.chunk_steps,
        allow_incomplete_cache=args.allow_incomplete_cache,
        options=(0, 1, 2, 4),
    )

    if args.export_step_csv is not None:
        export_step_csv(artifacts, args.export_step_csv)

    if args.no_gui:
        print("[delay] --no-gui set; cache build/check complete.")
        return 0

    vmax_ms = float(args.vmax_ms) if args.vmax_ms is not None else float(artifacts.meta.get("delay_max_ms") or 10.5)

    return run_gui(
        artifacts=artifacts,
        config=G60_CONFIG,
        vmin_ms=args.vmin_ms,
        vmax_ms=vmax_ms,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    raise SystemExit(main())
