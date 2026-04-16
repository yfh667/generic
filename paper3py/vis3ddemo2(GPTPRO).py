from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re

import geopandas as gpd
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from shapely.ops import unary_union

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
)


EARTH_R_KM = 6371.0
EARTH_TEXTURE_PATH = None

BG_COLOR = "#ffffff"
EARTH_COLOR = "#eef2f5"
SAT_COLOR = "#3b82f6"
STATION_COLOR = "#7c8794"

UI_BG = "#ffffff"
UI_PANEL = "#f8fafc"
UI_BORDER = "#d9e1e8"
TEXT_COLOR = "#5f6b78"

# ---------- 数据配置 ----------
LOAD_MODE = "all"          # "all" 或 "preview"
PREVIEW_FILE = "1.e"       # LOAD_MODE="preview" 时使用
EPHEM_GLOB = "*.e"         # LOAD_MODE="all" 时使用
EPHEM_MAX_SATS = None       # 调试时可设成 50 / 100；正式运行可保留 None
TIME_STRIDE = 10            # 每隔多少个采样点取 1 个；1=不降采样
PLAY_TIMER_MS = 50


# ---------- 路径配置 ----------
def first_existing_path(*paths: str) -> Path:
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
    return Path(paths[0])


COUNTRY_SHP_PATH = first_existing_path(
    r"C:\user\data\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp",
    r"D:\paper3\data\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp",
)

EPHEM_DIR = first_existing_path(
    r"C:\user\data\satellitesposition\satellite_pos",
    r"D:\paper3\data\satellitesposition\satellite_pos",
)

PREVIEW_START_S = 0
PREVIEW_END_S = 100


# ---------- 基础工具 ----------
def ll_to_xyz(lat_deg: float, lon_deg: float, r: float) -> np.ndarray:
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=float)


_digit_re = re.compile(r"(\d+)")


def natural_sort_key(path: Path):
    parts = _digit_re.split(path.stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


# ---------- STK ephemeris 读取 ----------
def parse_stk_ephemeris(path: Path, start_s: float | None = None, end_s: float | None = None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Ephemeris file not found: {path}")

    if start_s is not None:
        start_s = float(start_s)
    if end_s is not None:
        end_s = float(end_s)

    if start_s is not None and end_s is not None and start_s > end_s:
        raise ValueError(f"start_s ({start_s}) cannot be greater than end_s ({end_s})")

    requested_start = 0.0 if start_s is None else start_s
    requested_end = np.inf if end_s is None else end_s

    meta: dict[str, object] = {"path": str(path)}
    fmt = None
    in_block = False
    dist_scale_km = None
    rows = []
    file_time_start_s = None
    file_time_end_s = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line == "BEGIN Ephemeris":
                in_block = True
                continue

            if line == "END Ephemeris":
                break

            if not in_block:
                continue

            if fmt is None:
                if line == "EphemerisTimePosVel":
                    fmt = line
                    meta["EphemerisFormat"] = line
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    meta[parts[0]] = " ".join(parts[1:])
                continue

            if dist_scale_km is None:
                coord_system = str(meta.get("CoordinateSystem", ""))
                if coord_system != "Fixed":
                    raise ValueError(
                        f"Only CoordinateSystem=Fixed is supported, got: {coord_system}"
                    )

                unit = str(meta.get("DistanceUnit", "Meters")).strip().lower()
                if unit.startswith("meter"):
                    dist_scale_km = 1.0 / 1000.0
                elif unit.startswith("kilometer"):
                    dist_scale_km = 1.0
                else:
                    raise ValueError(f"Unsupported DistanceUnit: {meta.get('DistanceUnit')}")

            parts = line.split()
            if len(parts) < 4:
                continue

            t_s = float(parts[0])
            x_km = float(parts[1]) * dist_scale_km
            y_km = float(parts[2]) * dist_scale_km
            z_km = float(parts[3]) * dist_scale_km

            if file_time_start_s is None:
                file_time_start_s = t_s
            file_time_end_s = t_s

            if t_s < requested_start:
                continue
            if t_s > requested_end:
                break

            rows.append((t_s, x_km, y_km, z_km))

    if fmt != "EphemerisTimePosVel":
        raise ValueError(f"Only EphemerisTimePosVel is supported, got: {fmt}")

    if not rows:
        raise ValueError(
            f"No ephemeris samples in requested range {requested_start}..{requested_end}s; "
            f"file range is {file_time_start_s}..{file_time_end_s}s"
        )

    arr = np.asarray(rows, dtype=float)
    times_s = arr[:, 0]
    positions_km = arr[:, 1:4].astype(np.float32)

    meta["file_time_start_s"] = float(file_time_start_s)
    meta["file_time_end_s"] = float(file_time_end_s)
    meta["actual_start_s"] = float(times_s[0])
    meta["actual_end_s"] = float(times_s[-1])
    meta["requested_start_s"] = None if start_s is None else float(start_s)
    meta["requested_end_s"] = None if end_s is None else float(end_s)

    return meta, times_s, positions_km



def downsample_timeseries(times_s: np.ndarray, positions_km: np.ndarray, stride: int):
    stride = max(1, int(stride))
    if stride == 1 or len(times_s) <= 2:
        return times_s, positions_km

    idx = np.arange(0, len(times_s), stride, dtype=int)
    if idx[-1] != len(times_s) - 1:
        idx = np.append(idx, len(times_s) - 1)
    return times_s[idx], positions_km[idx]



def load_ephemeris_collection(
    ephem_dir: Path,
    mode: str = "all",
    preview_file: str = "1.e",
    start_s: float | None = None,
    end_s: float | None = None,
    pattern: str = "*.e",
    max_sats: int | None = None,
    stride: int = 1,
):
    ephem_dir = Path(ephem_dir)
    if not ephem_dir.exists():
        raise FileNotFoundError(f"Ephemeris directory not found: {ephem_dir}")

    if mode not in {"all", "preview"}:
        raise ValueError("mode must be 'all' or 'preview'")

    if mode == "preview":
        files = [ephem_dir / preview_file]
    else:
        files = sorted(ephem_dir.glob(pattern), key=natural_sort_key)
        if max_sats is not None:
            files = files[: int(max_sats)]

    if not files:
        raise FileNotFoundError(f"No ephemeris files found in {ephem_dir}")

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing ephemeris files:\n" + "\n".join(missing))

    cache_dir = ephem_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    source_signature = [
        {
            "file": str(p.resolve()),
            "size": int(p.stat().st_size),
            "mtime_ns": int(p.stat().st_mtime_ns),
        }
        for p in files
    ]
    signature_payload = {
        "cache_version": 2,
        "mode": mode,
        "start_s": None if start_s is None else float(start_s),
        "end_s": None if end_s is None else float(end_s),
        "stride": int(stride),
        "sources": source_signature,
    }
    signature_json = json.dumps(signature_payload, ensure_ascii=False, sort_keys=True)
    signature_key = hashlib.sha1(signature_json.encode("utf-8")).hexdigest()[:16]

    times_path = cache_dir / f"collection_{signature_key}_times_s.npy"
    positions_path = cache_dir / f"collection_{signature_key}_positions_km.npy"
    sat_ids_path = cache_dir / f"collection_{signature_key}_sat_ids.json"
    meta_path = cache_dir / f"collection_{signature_key}_meta.json"

    if times_path.exists() and positions_path.exists() and sat_ids_path.exists() and meta_path.exists():
        times_s = np.load(times_path)
        positions_km = np.load(positions_path)
        sat_ids = json.loads(sat_ids_path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta, sat_ids, times_s, positions_km

    sat_ids: list[str] = []
    pos_list: list[np.ndarray] = []
    times_ref = None
    first_meta = None

    for i, path in enumerate(files, start=1):
        meta, times_s, positions_km = parse_stk_ephemeris(path, start_s=start_s, end_s=end_s)
        times_s, positions_km = downsample_timeseries(times_s, positions_km, stride=stride)

        if times_ref is None:
            times_ref = times_s
            first_meta = meta
        else:
            if len(times_s) != len(times_ref) or not np.allclose(times_s, times_ref):
                raise ValueError(
                    f"Time grid mismatch in {path.name}. "
                    f"Expected {len(times_ref)} samples, got {len(times_s)}"
                )

        sat_ids.append(path.stem)
        pos_list.append(positions_km.astype(np.float32, copy=False))

        if i % 50 == 0:
            print(f"Loaded {i}/{len(files)} ephemeris files...")

    assert times_ref is not None
    assert first_meta is not None

    positions_all = np.stack(pos_list, axis=1).astype(np.float32, copy=False)  # [T, S, 3]

    collection_meta = {
        **signature_payload,
        "num_sats": int(len(sat_ids)),
        "num_points": int(len(times_ref)),
        "actual_start_s": float(times_ref[0]),
        "actual_end_s": float(times_ref[-1]),
        "scenario_epoch": first_meta.get("ScenarioEpoch"),
        "coordinate_system": first_meta.get("CoordinateSystem"),
        "distance_unit": first_meta.get("DistanceUnit", "Meters"),
        "ephemeris_format": first_meta.get("EphemerisFormat"),
    }

    np.save(times_path, times_ref)
    np.save(positions_path, positions_all)
    sat_ids_path.write_text(json.dumps(sat_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(collection_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return collection_meta, sat_ids, times_ref, positions_all


class GlobeSatDemo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Globe + Satellite IDs")
        self.resize(1480, 920)

        self.collection_meta, self.sat_ids, self.ephem_times_s, self.ephem_positions = load_ephemeris_collection(
            EPHEM_DIR,
            mode=LOAD_MODE,
            preview_file=PREVIEW_FILE,
            start_s=PREVIEW_START_S,
            end_s=PREVIEW_END_S,
            pattern=EPHEM_GLOB,
            max_sats=EPHEM_MAX_SATS,
            stride=TIME_STRIDE,
        )

        self.total = len(self.sat_ids)
        self.selected_sat_idx: int | None = None
        self.step = 0
        self.playing = True

        self.range_text = (
            f"range={int(self.collection_meta['actual_start_s'])}"
            f"..{int(self.collection_meta['actual_end_s'])}s"
        )

        self._build_ui()
        self._build_scene()
        self._build_static_layers()
        self._update_frame(0)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(PLAY_TIMER_MS)

    def _status_text(self, step: int) -> str:
        t_s = self.ephem_times_s[step]
        text = (
            f"step={step}  t={t_s:.0f}s  sats={self.total}  "
            f"stride={TIME_STRIDE}  {self.range_text}"
        )
        if self.selected_sat_idx is not None:
            text += f"  sat={self.sat_ids[self.selected_sat_idx]}"
        return text

    def _build_ui(self):
        self.setStyleSheet(
            f"""
            QWidget {{
                background: {UI_BG};
                color: {TEXT_COLOR};
                font-size: 13px;
            }}
            QPushButton {{
                background: {UI_PANEL};
                border: 1px solid {UI_BORDER};
                border-radius: 5px;
                padding: 6px 12px;
                min-height: 28px;
            }}
            QPushButton:hover {{
                background: #f1f5f9;
            }}
            QLabel {{
                color: {TEXT_COLOR};
            }}
            QSlider::groove:horizontal {{
                border: 0;
                height: 4px;
                background: #dce3ea;
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: #8c98a5;
                border: 0;
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.btn = QPushButton("Pause")
        self.btn.clicked.connect(self._toggle_play)

        self.lbl = QLabel(self._status_text(0))
        self.lbl.setMinimumWidth(420)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(self.ephem_times_s) - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider)

        bar.addWidget(self.btn)
        bar.addWidget(self.lbl)
        bar.addWidget(self.slider)

        self.plotter = QtInteractor(self)

        root.addLayout(bar)
        root.addWidget(self.plotter.interactor)

    def _build_scene(self):
        self.plotter.set_background(BG_COLOR)
        self.plotter.enable_anti_aliasing()

        earth = pv.Sphere(radius=EARTH_R_KM, theta_resolution=220, phi_resolution=220)
        if EARTH_TEXTURE_PATH:
            tex = pv.read_texture(EARTH_TEXTURE_PATH)
            self.plotter.add_mesh(
                earth,
                texture=tex,
                smooth_shading=True,
                ambient=0.28,
                diffuse=0.72,
                specular=0.0,
                pickable=False,
            )
        else:
            self.plotter.add_mesh(
                earth,
                color=EARTH_COLOR,
                smooth_shading=True,
                ambient=0.35,
                diffuse=0.65,
                specular=0.0,
                pickable=False,
            )

        if COUNTRY_SHP_PATH.exists():
            self.add_country_outlines_from_polygons(COUNTRY_SHP_PATH)

        self.sat_poly = pv.PolyData(self.ephem_positions[0].astype(float, copy=True))
        self.sat_actor = self.plotter.add_mesh(
            self.sat_poly,
            render_points_as_spheres=True,
            point_size=8,
            color=SAT_COLOR,
            ambient=0.5,
            pickable=True,
            name="satellites",
        )

        self.plotter.enable_point_picking(
            callback=self._on_pick_satellite,
            picker="point",
            left_clicking=True,
            show_point=False,
            show_message="Left click a satellite to view its ID",
            use_picker=True,
            clear_on_no_selection=True,
        )

        self.plotter.camera_position = [
            (18000, -15000, 11000),
            (0, 0, 0),
            (0, 0, 1),
        ]

    def _build_static_layers(self):
        stations_ll = [
            (-15.7939, -47.8828),
            (12.1140, -86.2362),
            (30.0444, 31.2357),
            (39.9042, 116.4074),
            (-33.8688, 151.2093),
        ]
        station_xyz = np.array(
            [ll_to_xyz(lat, lon, EARTH_R_KM * 1.0005) for lat, lon in stations_ll],
            dtype=float,
        )
        st_poly = pv.PolyData(station_xyz)
        self.plotter.add_mesh(
            st_poly,
            render_points_as_spheres=True,
            point_size=11,
            color=STATION_COLOR,
            ambient=0.25,
            pickable=False,
        )

    def _iter_lines(self, geom):
        if geom is None or geom.is_empty:
            return

        gt = geom.geom_type
        if gt == "LineString":
            yield geom
        elif gt == "MultiLineString":
            for g in geom.geoms:
                yield from self._iter_lines(g)
        elif gt == "GeometryCollection":
            for g in geom.geoms:
                yield from self._iter_lines(g)

    def _segments_to_polydata(self, segments: list[np.ndarray]) -> pv.PolyData:
        pts = np.vstack(segments)
        cells = []
        offset = 0
        for seg in segments:
            cells.extend([len(seg), *range(offset, offset + len(seg))])
            offset += len(seg)

        mesh = pv.PolyData(pts)
        mesh.lines = np.array(cells, dtype=np.int64)
        return mesh

    def add_country_outlines_from_polygons(self, shp_path: Path):
        gdf = gpd.read_file(shp_path)
        boundary = unary_union(gdf.geometry.boundary)

        r = EARTH_R_KM * 1.004
        segments = []
        for line in self._iter_lines(boundary):
            xyz = np.array([ll_to_xyz(lat, lon, r) for lon, lat in line.coords], dtype=float)
            if len(xyz) >= 2:
                segments.append(xyz)

        if not segments:
            return

        mesh = self._segments_to_polydata(segments)

        self.plotter.add_mesh(
            mesh.copy(),
            color="#ffffff",
            line_width=3.0,
            opacity=0.95,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )
        self.plotter.add_mesh(
            mesh,
            color="#d7dde4",
            line_width=1.2,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )

    def _clear_selected_sat_label(self):
        try:
            self.plotter.remove_actor("sat_pick_label", reset_camera=False, render=False)
        except Exception:
            pass

    def _refresh_selected_sat_label(self):
        self._clear_selected_sat_label()
        if self.selected_sat_idx is None:
            return

        sat_id = self.sat_ids[self.selected_sat_idx]
        point = np.asarray([self.sat_poly.points[self.selected_sat_idx]])

        self.plotter.add_point_labels(
            point,
            [f"SAT {sat_id}"],
            name="sat_pick_label",
            always_visible=True,
            show_points=True,
            point_size=12,
            render_points_as_spheres=True,
            point_color="#1d4ed8",
            text_color="#1d4ed8",
            shape_color="#eff6ff",
            shape_opacity=0.95,
            font_size=14,
            pickable=False,
            reset_camera=False,
            render=False,
        )

    def _clear_selection(self):
        self.selected_sat_idx = None
        self._clear_selected_sat_label()
        self.lbl.setText(self._status_text(self.step))
        self.plotter.render()

    def _on_pick_satellite(self, picked_point, picker):
        if picker is None:
            self._clear_selection()
            return

        point_id = int(picker.GetPointId()) if hasattr(picker, "GetPointId") else -1
        if point_id < 0 or point_id >= self.total:
            self._clear_selection()
            return

        self.selected_sat_idx = point_id
        self._refresh_selected_sat_label()
        self.lbl.setText(self._status_text(self.step))
        self.plotter.render()

    def _update_frame(self, step: int):
        self.step = max(0, min(int(step), len(self.ephem_times_s) - 1))
        pts = self.ephem_positions[self.step]

        self.sat_poly.points = pts
        self.sat_poly.Modified()

        if self.selected_sat_idx is not None and 0 <= self.selected_sat_idx < self.total:
            self._refresh_selected_sat_label()
        else:
            self._clear_selected_sat_label()

        self.lbl.setText(self._status_text(self.step))
        self.plotter.render()

    def _tick(self):
        if not self.playing:
            return

        nxt = (self.step + 1) % (self.slider.maximum() + 1)
        self.slider.blockSignals(True)
        self.slider.setValue(nxt)
        self.slider.blockSignals(False)
        self._update_frame(nxt)

    def _on_slider(self, v: int):
        self._update_frame(int(v))

    def _toggle_play(self):
        self.playing = not self.playing
        self.btn.setText("Pause" if self.playing else "Play")


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = GlobeSatDemo()
    w.show()
    sys.exit(app.exec_())
