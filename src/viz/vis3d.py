from pathlib import Path
import json
import time

import geopandas as gpd
import numpy as np
import pyvista as pv
import vtk
from numpy.lib.format import open_memmap
from pyvistaqt import QtInteractor
from shapely.ops import unary_union

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QLabel, QLineEdit, QFrame
)


EARTH_R_KM = 6371.0
SAT_ALT_KM = 550.0
INC_DEG = 53.0

EARTH_TEXTURE_PATH = None

BG_COLOR = "#ffffff"
EARTH_COLOR = "#eef2f5"
COAST_COLOR = "#c9d1d9"
ORBIT_COLOR = "#d7dee5"
SAT_COLOR = "#3b82f6"
LINK_COLOR = "#b8c3ce"
PATH_COLOR = "#4f8fc4"
STATION_COLOR = "#7c8794"

UI_BG = "#ffffff"
UI_PANEL = "#f8fafc"
UI_BORDER = "#d9e1e8"
TEXT_COLOR = "#5f6b78"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent








PREVIEW_START_S = 0
PREVIEW_END_S = 86164
EPHEM_STEP_S = 1
EXPECTED_SAT_COUNT = 648
CACHE_VERSION = 2
CACHE_PROGRESS_EVERY = 8  # 想每颗都打印就改成 1


def first_existing_path(*paths):
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
    return Path(paths[0])
def first_ephemeris_dir(*paths):
    """
    Select the first ephemeris directory that actually contains .e files.

    first_existing_path() is not enough here because an old empty directory may
    still exist and would be selected before the real migrated data directory.
    """
    checked = [Path(raw) for raw in paths]

    for p in checked:
        if p.exists() and p.is_dir() and any(p.glob("*.e")):
            return p

    for p in checked:
        if p.exists() and p.is_dir():
            return p

    return checked[0]


COUNTRY_SHP_PATH = first_existing_path(
    # New data layout.
    PROJECT_DIR / "data" / "basic_file" / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp",
    r"D:\paper3\data\basic_file\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp",
    r"C:\user\data\basic_file\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp",

    # Old data layout fallback.
    PROJECT_DIR / "data" / "ne_50m_admin_0_countries" / "ne_50m_admin_0_countries.shp",
    r"D:\paper3\data\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp",
    r"C:\user\data\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp",
)


# EPHEM_DIR = first_existing_path(
#     PROJECT_DIR / "data" / "satellitesposition" / "satellite_pos",
#     r"D:\paper3\data\satellitesposition\satellite_pos",
#     r"C:\user\data\satellitesposition\satellite_pos",
# )
EPHEM_DIR = first_ephemeris_dir(
    # New data layout.
    PROJECT_DIR / "data" / "basic_file" / "satellitesposition" / "satellite_pos",
    r"D:\paper3\data\basic_file\satellitesposition\satellite_pos",
    r"C:\user\data\basic_file\satellitesposition\satellite_pos",

    # Old data layout fallback.
    PROJECT_DIR / "data" / "satellitesposition" / "satellite_pos",
    r"D:\paper3\data\satellitesposition\satellite_pos",
    r"C:\user\data\satellitesposition\satellite_pos",
)


EXTERNAL_CACHE_ROOT = EPHEM_DIR / "_cache"
CACHE_DIR = EXTERNAL_CACHE_ROOT / "cache_86164s_1s"


# PREVIEW_FILE = "1.e"
# PREVIEW_START_S = 0
# PREVIEW_END_S = 86164
# PREVIEW_START_S = 0
# PREVIEW_END_S = 100
# EXPECTED_SAT_COUNT = 648





# pip install geopandas pyogrio shapely
import geopandas as gpd

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def ll_to_xyz(lat_deg, lon_deg, r):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z], dtype=float)

def parse_stk_ephemeris(path, start_s=None, end_s=None):
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

    meta = {"path": str(path)}
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
                coord_system = meta.get("CoordinateSystem", "")
                if coord_system != "Fixed":
                    raise ValueError(
                        f"Only CoordinateSystem=Fixed is supported, got: {coord_system}"
                    )

                unit = meta.get("DistanceUnit", "Meters").strip().lower()
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


# def load_preview_ephemeris(ephem_dir, preview_file, start_s, end_s):
#     ephem_dir = Path(ephem_dir)
#     ephem_path = ephem_dir / preview_file
#     if not ephem_path.exists():
#         raise FileNotFoundError(f"Preview ephemeris not found: {ephem_path}")
#
#     cache_dir = ephem_dir / "_cache"
#     cache_dir.mkdir(parents=True, exist_ok=True)
#
#     stem = Path(preview_file).stem
#     times_path = cache_dir / f"preview_{stem}_times_s.npy"
#     positions_path = cache_dir / f"preview_{stem}_positions_km.npy"
#     meta_path = cache_dir / f"preview_{stem}_meta.json"
#
#     src_stat = ephem_path.stat()
#     expected = {
#         "cache_version": 1,
#         "source_file": str(ephem_path.resolve()),
#         "source_size": int(src_stat.st_size),
#         "source_mtime_ns": int(src_stat.st_mtime_ns),
#         "start_s": None if start_s is None else float(start_s),
#         "end_s": None if end_s is None else float(end_s),
#     }
#
#     if times_path.exists() and positions_path.exists() and meta_path.exists():
#         try:
#             cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
#         except json.JSONDecodeError:
#             cached_meta = None
#
#         if cached_meta and all(cached_meta.get(k) == v for k, v in expected.items()):
#             times_s = np.load(times_path)
#             positions_km = np.load(positions_path)
#             return cached_meta, times_s, positions_km
#
#     meta, times_s, positions_km = parse_stk_ephemeris(
#         ephem_path,
#         start_s=start_s,
#         end_s=end_s,
#     )
#
#     cache_meta = dict(expected)
#     cache_meta.update({
#         "ScenarioEpoch": meta.get("ScenarioEpoch"),
#         "CoordinateSystem": meta.get("CoordinateSystem"),
#         "DistanceUnit": meta.get("DistanceUnit", "Meters"),
#         "EphemerisFormat": meta.get("EphemerisFormat"),
#         "actual_start_s": meta["actual_start_s"],
#         "actual_end_s": meta["actual_end_s"],
#         "num_points": int(len(times_s)),
#     })
#
#     np.save(times_path, times_s)
#     np.save(positions_path, positions_km.astype(np.float32))
#     meta_path.write_text(
#         json.dumps(cache_meta, ensure_ascii=False, indent=2),
#         encoding="utf-8",
#     )
#
#     return cache_meta, times_s, positions_km
def list_ephemeris_files(ephem_dir):
    ephem_dir = Path(ephem_dir)
    files = [p for p in ephem_dir.glob("*.e") if p.stem.isdigit()]
    files = sorted(files, key=lambda p: int(p.stem))

    if not files:
        raise FileNotFoundError(f"No .e files found in: {ephem_dir}")

    if EXPECTED_SAT_COUNT is not None and len(files) != EXPECTED_SAT_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAT_COUNT} ephemeris files, got {len(files)}"
        )

    return files


def get_cache_dir(start_s, end_s, step_s=EPHEM_STEP_S, cache_root=None):
    start_i = int(0 if start_s is None else start_s)
    end_i = int(end_s)
    step_i = int(step_s)

    root = Path(cache_root) if cache_root is not None else EXTERNAL_CACHE_ROOT

    if start_i == 0 and end_i == 86164 and step_i == 1:
        return root / "cache_86164s_1s"

    return root / f"cache_{start_i}_{end_i}_{step_i}s"


def get_cache_paths(start_s, end_s, step_s=EPHEM_STEP_S, cache_root=None):
    cache_dir = get_cache_dir(start_s, end_s, step_s, cache_root=cache_root)
    return {
        "dir": cache_dir,
        "times": cache_dir / "times_s.npy",
        "positions": cache_dir / "positions_km.npy",
        "sat_ids": cache_dir / "sat_ids.json",
        "meta": cache_dir / "cache_meta.json",
        "report": cache_dir / "build_report.json",
        "progress": cache_dir / "parse_progress.jsonl",
    }



def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def append_jsonl(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_source_file_info(files):
    return [
        {
            "name": p.name,
            "size": int(p.stat().st_size),
            "mtime_ns": int(p.stat().st_mtime_ns),
        }
        for p in files
    ]


def build_expected_meta(ephem_dir, files, start_s, end_s, step_s):
    return {
        "cache_version": CACHE_VERSION,
        "start_s": None if start_s is None else float(start_s),
        "end_s": None if end_s is None else float(end_s),
        "step_s": float(step_s),
        "num_sats": len(files),
        "source_dir": str(Path(ephem_dir).resolve()),
        "source_files": build_source_file_info(files),
    }


def cache_meta_matches(cached_meta, expected_meta, *, ignore_source_dir=False):
    if not isinstance(cached_meta, dict):
        return False

    for key, value in expected_meta.items():
        if ignore_source_dir and key == "source_dir":
            continue
        if cached_meta.get(key) != value:
            return False

    return True


def open_cached_ephemerides(ephem_dir, start_s, end_s, step_s=EPHEM_STEP_S, cache_root=None):
    paths = get_cache_paths(start_s, end_s, step_s, cache_root=cache_root)


   # required = ("times", "positions", "sat_ids", "meta", "report", "progress")
    required = ("times", "positions", "sat_ids", "meta")

    missing = [key for key in required if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(f"Missing cache files: {missing} in {paths['dir']}")

    cache_meta = read_json(paths["meta"])
    sat_ids = read_json(paths["sat_ids"])
    times_s = np.load(paths["times"], mmap_mode="r")
    positions_km = np.load(paths["positions"], mmap_mode="r")

    return cache_meta, times_s, positions_km, sat_ids


def build_full_day_cache(ephem_dir, start_s, end_s, step_s=EPHEM_STEP_S, cache_root=None):
    ephem_dir = Path(ephem_dir)
    files = list_ephemeris_files(ephem_dir)
    paths = get_cache_paths(start_s, end_s, step_s, cache_root=cache_root)



    expected_meta = build_expected_meta(ephem_dir, files, start_s, end_s, step_s)

    paths["dir"].mkdir(parents=True, exist_ok=True)
    paths["progress"].write_text("", encoding="utf-8")

    build_started = time.perf_counter()
    report = {
        "status": "building",
        "requested_start_s": None if start_s is None else float(start_s),
        "requested_end_s": None if end_s is None else float(end_s),
        "step_s": float(step_s),
        "num_sats": len(files),
        "source_dir": str(ephem_dir.resolve()),
        "source_files": expected_meta["source_files"],
        "time_axis_check": "pending",
        "satellite_records": [],
        "failures": [],
    }
    write_json(paths["report"], report)

    ref_times_s = None
    sat_ids = []
    positions_mm = None

 #   print(f"[cache] building cache in {paths['dir']}")
    print(f"[cache] building cache in {paths['dir']}", flush=True)

    try:
        for sat_idx, path in enumerate(files):
            sat_started = time.perf_counter()

            try:
                meta, times_s, positions_km = parse_stk_ephemeris(
                    path,
                    start_s=start_s,
                    end_s=end_s,
                )

                times_s = np.asarray(times_s, dtype=np.float64)
                positions_km = np.asarray(positions_km, dtype=np.float32)

                if ref_times_s is None:
                    ref_times_s = times_s

                    if len(ref_times_s) < 2:
                        raise ValueError("Reference time axis has fewer than 2 samples")

                    dt = np.diff(ref_times_s)
                    if not np.allclose(dt, float(step_s)):
                        raise ValueError(
                            f"Reference time axis is not uniform {step_s}s; "
                            f"first deltas: {dt[:5].tolist()}"
                        )

                    times_mm = open_memmap(
                        paths["times"],
                        mode="w+",
                        dtype=np.float64,
                        shape=ref_times_s.shape,
                    )
                    times_mm[:] = ref_times_s
                    times_mm.flush()
                    del times_mm

                    positions_mm = open_memmap(
                        paths["positions"],
                        mode="w+",
                        dtype=np.float32,
                        shape=(len(ref_times_s), len(files), 3),
                    )

                    report["time_axis_check"] = "uniform"

                else:
                    if len(times_s) != len(ref_times_s) or not np.allclose(times_s, ref_times_s):
                        raise ValueError(f"Time axis mismatch in file: {path.name}")

                positions_mm[:, sat_idx, :] = positions_km
                # positions_mm.flush()
                positions_mm[:, sat_idx, :] = positions_km

                if (sat_idx + 1) % 16 == 0 or sat_idx + 1 == len(files):
                    positions_mm.flush()

                sat_ids.append(path.stem)

                sat_record = {
                    "sat_id": path.stem,
                    "sat_index": sat_idx,
                    "num_points": int(len(times_s)),
                    "elapsed_s": round(time.perf_counter() - sat_started, 3),
                    "status": "ok",
                    "actual_start_s": float(times_s[0]),
                    "actual_end_s": float(times_s[-1]),
                }
                report["satellite_records"].append(sat_record)
                append_jsonl(paths["progress"], sat_record)

                if (sat_idx + 1) % CACHE_PROGRESS_EVERY == 0 or sat_idx + 1 == len(files):
                    elapsed_total = time.perf_counter() - build_started
                    done = sat_idx + 1
                    avg_per_sat = elapsed_total / done
                    eta_s = avg_per_sat * (len(files) - done)

                    print(
                        f"[cache] parsed {done}/{len(files)} satellites | "
                        f"last={path.stem} | "
                        f"sat_elapsed={sat_record['elapsed_s']:.3f}s | "
                        f"total_elapsed={elapsed_total:.1f}s | "
                        f"eta={eta_s:.1f}s",
                        flush=True,
                    )


            #    if (sat_idx + 1) % 32 == 0 or sat_idx + 1 == len(files):
             #       print(f"[cache] parsed {sat_idx + 1}/{len(files)} satellites")

            except Exception as exc:
                failure = {
                    "sat_id": path.stem,
                    "sat_index": sat_idx,
                    "status": "failed",
                    "elapsed_s": round(time.perf_counter() - sat_started, 3),
                    "error": str(exc),
                }
                report["failures"].append(failure)
                append_jsonl(paths["progress"], failure)
                raise

        if positions_mm is None or ref_times_s is None:
            raise ValueError("No ephemeris data was parsed")

        positions_mm.flush()
        del positions_mm

        cache_meta = dict(expected_meta)
        cache_meta.update({
            "actual_start_s": float(ref_times_s[0]),
            "actual_end_s": float(ref_times_s[-1]),
            "num_steps": int(len(ref_times_s)),
        })

        write_json(paths["sat_ids"], sat_ids)
        write_json(paths["meta"], cache_meta)

        report["status"] = "completed"
        report["build_succeeded"] = True
        report["elapsed_s"] = round(time.perf_counter() - build_started, 3)
        report["actual_start_s"] = float(ref_times_s[0])
        report["actual_end_s"] = float(ref_times_s[-1])
        report["num_steps"] = int(len(ref_times_s))
        write_json(paths["report"], report)

    except Exception as exc:
        if positions_mm is not None:
            positions_mm.flush()

        report["status"] = "failed"
        report["build_succeeded"] = False
        report["elapsed_s"] = round(time.perf_counter() - build_started, 3)
        report["error"] = str(exc)
        write_json(paths["report"], report)
        raise
    return open_cached_ephemerides(ephem_dir, start_s, end_s, step_s, cache_root=cache_root)

    #return open_cached_ephemerides(ephem_dir, start_s, end_s, step_s)


def load_all_ephemerides(
    ephem_dir,
    start_s,
    end_s,
    step_s=EPHEM_STEP_S,
    cache_root=None,
    ignore_cache_source_dir=False,
):
    ephem_dir = Path(ephem_dir)
    files = list_ephemeris_files(ephem_dir)
    paths = get_cache_paths(start_s, end_s, step_s, cache_root=cache_root)
    expected_meta = build_expected_meta(ephem_dir, files, start_s, end_s, step_s)


    #required = ("times", "positions", "sat_ids", "meta", "report", "progress"
    required = ("times", "positions", "sat_ids", "meta")

    cache_ready = all(paths[key].exists() for key in required)
    cached_meta = read_json(paths["meta"]) if cache_ready else None

    if cache_ready and cache_meta_matches(
            cached_meta,
            expected_meta,
            ignore_source_dir=ignore_cache_source_dir,
    ):
        print(f"[cache] reusing cache from {paths['dir']}")
        return open_cached_ephemerides(
            ephem_dir,
            start_s,
            end_s,
            step_s,
            cache_root=cache_root,
        )
    return build_full_day_cache(
        ephem_dir,
        start_s,
        end_s,
        step_s,
        cache_root=cache_root,
    )

   # return build_full_day_cache(ephem_dir, start_s, end_s, step_s)


class GlobeSatDemo(QWidget):
    def __init__(
        self,
        parent=None,
        P=18,
        N=36,
        start_s=PREVIEW_START_S,
        end_s=PREVIEW_END_S,
        step_s=EPHEM_STEP_S,
        auto_play=False,
        timer_interval_ms=200,
        initial_step=0,
        ephem_dir=None,
        cache_root=None,
        ignore_cache_source_dir=False,
    ):


        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.resize(1400, 900)

        self.P = P
        self.N = N
        self.start_s = int(start_s)
        self.end_s = int(end_s)
        self.step_s = int(step_s)
        self.timer_interval_ms = max(1, int(timer_interval_ms))

        # self.preview_meta, self.ephem_times_s, self.ephem_positions, self.sat_ids = load_all_ephemerides(
        #     EPHEM_DIR,
        #     self.start_s,
        #     self.end_s,
        #     step_s=self.step_s,
        # )
    #    self.ephem_dir = Path(ephem_dir) if ephem_dir is not None else EPHEM_DIR

        self.ephem_dir = Path(ephem_dir) if ephem_dir is not None else EPHEM_DIR
        self.cache_root = Path(cache_root) if cache_root is not None else EXTERNAL_CACHE_ROOT
        self.ignore_cache_source_dir = bool(ignore_cache_source_dir)

        self.preview_meta, self.ephem_times_s, self.ephem_positions, self.sat_ids = load_all_ephemerides(
            self.ephem_dir,
            self.start_s,
            self.end_s,
            step_s=self.step_s,
            cache_root=self.cache_root,
            ignore_cache_source_dir=self.ignore_cache_source_dir,
        )

        self.selected_sat_idx = None
        self.selected_sat_idxs = set()
        
        # 外部注入的真实路径：key 是时间，value 是 [sat_id0, sat_id1, ...]
        self.path_by_time = {}
        self.show_demo_path = False


        self.total = int(self.ephem_positions.shape[1])

        self.setWindowTitle(
            f"3D Globe + {self.total} Satellites "
            f"({int(self.ephem_times_s[0])}..{int(self.ephem_times_s[-1])}s)"
        )

        if self.total != self.P * self.N:
            raise ValueError(
                f"Loaded {self.total} satellites, but P*N = {self.P * self.N}"
            )

        self.range_text = (
            f"range={int(self.ephem_times_s[0])}"
            f"..{int(self.ephem_times_s[-1])}s"
        )

        self.step = 0
        self.playing = bool(auto_play)

        self._build_ui()
        self._build_scene()
        self._build_static_layers()
        self._update_frame(initial_step)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.timer_interval_ms)

        self.btn.setText("Pause" if self.playing else "Play")


    # def __init__(self, parent=None, P=18, N=36):
    #     super().__init__(parent)
    #
    #     self.resize(1400, 900)
    #
    #     self.P = P
    #     self.N = N
    #
    #
    #     #  self.setWindowTitle(f"3D Globe + Satellite Preview ({PREVIEW_FILE})")
    #     # self.preview_meta, preview_times_s, preview_positions_km = load_preview_ephemeris(
    #     #     EPHEM_DIR,
    #     #     PREVIEW_FILE,
    #     #     PREVIEW_START_S,
    #     #     PREVIEW_END_S,
    #     # )
    #     #
    #     # self.ephem_times_s = preview_times_s
    #     # self.ephem_positions = preview_positions_km[:, np.newaxis, :]
    #     #
    #     # self.sat_ids = [Path(PREVIEW_FILE).stem]  # 现在是 ['1']，以后可扩成 648 颗
    #     #
    #
    #     self.preview_meta, self.ephem_times_s, self.ephem_positions, self.sat_ids = load_all_ephemerides(
    #         EPHEM_DIR,
    #         PREVIEW_START_S,
    #         PREVIEW_END_S,
    #     )
    #
    #     self.selected_sat_idx = None
    #     self.total = self.ephem_positions.shape[1]
    #     self.setWindowTitle(f"3D Globe + {self.total} Satellites")
    #
    #     if self.total != self.P * self.N:
    #         raise ValueError(
    #             f"Loaded {self.total} satellites, but P*N = {self.P * self.N}"
    #         )
    #
    #     self.selected_sat_idx = None
    #
    #
    #     self.total = self.ephem_positions.shape[1]
    #
    #     # self.range_text = (
    #     #     f"range={int(self.preview_meta['actual_start_s'])}"
    #     #     f"..{int(self.preview_meta['actual_end_s'])}s"
    #     # )
    #     self.range_text = (
    #         f"range={int(self.ephem_times_s[0])}"
    #         f"..{int(self.ephem_times_s[-1])}s"
    #     )
    #
    #     self.step = 0
    #     self.playing = True
    #
    #     self._build_ui()
    #     self._build_scene()
    #     self._build_static_layers()
    #     self._update_frame(0)
    #
    #     self.timer = QTimer(self)
    #     self.timer.timeout.connect(self._tick)
    #     self.timer.start(50)

    def _remove_sat_label(self):
        try:
            self.plotter.remove_actor("sat_pick_labels", render=False)
        except Exception:
            pass

    def _update_pick_status(self):
        count = len(self.selected_sat_idxs)

        if count == 0:
            self.pick_lbl.setText("selected=none")
            return

        if self.selected_sat_idx is None:
            self.pick_lbl.setText(f"labels={count}")
            return

        # self.pick_lbl.setText(
        #     f"selected=SAT {self.sat_ids[self.selected_sat_idx]}  labels={count}"
        # )
        self.pick_lbl.setText(
            f"selected={self._sat_display_name(self.selected_sat_idx)}  labels={count}"
        )

    def _is_sat_visible_from_camera(self, point):
        camera_pos = np.asarray(self.plotter.camera_position[0], dtype=float)
        target = np.asarray(point, dtype=float)

        ray = target - camera_pos
        ray_len2 = float(np.dot(ray, ray))
        if ray_len2 < 1e-12:
            return True

        # 计算相机到卫星这条线段，距离地心最近的点
        t = -float(np.dot(camera_pos, ray)) / ray_len2
        t = np.clip(t, 0.0, 1.0)
        closest = camera_pos + t * ray

        # 最近点如果落进地球半径内，说明视线被地球挡住
        return np.linalg.norm(closest) > EARTH_R_KM * 1.001

    def sid(self, p, s):
        return p * self.N + s

    def _time_to_index(self, target_s):
        target_s = float(target_s)
        times = self.ephem_times_s

        idx = int(np.searchsorted(times, target_s, side="left"))
        if idx <= 0:
            return 0
        if idx >= len(times):
            return len(times) - 1

        prev_t = float(times[idx - 1])
        next_t = float(times[idx])
        if abs(target_s - prev_t) <= abs(next_t - target_s):
            return idx - 1
        return idx

    def _set_current_step(self, step):
        step = max(0, min(int(step), len(self.ephem_times_s) - 1))
        self.slider.blockSignals(True)
        self.slider.setValue(step)
        self.slider.blockSignals(False)
        self._update_frame(step)

    def _step_prev(self):
        self.playing = False
        self.btn.setText("Play")
        self._set_current_step(self.step - 1)

    def _step_next(self):
        self.playing = False
        self.btn.setText("Play")
        self._set_current_step(self.step + 1)

    def _on_jump(self):
        raw = self.jump_input.text().strip()
        if not raw:
            return

        try:
            target_s = float(raw)
        except ValueError:
            self.jump_input.selectAll()
            return

        self.playing = False
        self.btn.setText("Play")
        self._set_current_step(self._time_to_index(target_s))

    def _sync_timeline_labels(self):
        t0 = float(self.ephem_times_s[0])
        t1 = float(self.ephem_times_s[-1])
        tc = float(self.ephem_times_s[self.step])

        self.axis_min_lbl.setText(f"{t0:.0f}s")
        self.axis_cur_lbl.setText(f"step {self.step}   t={tc:.0f}s")
        self.axis_max_lbl.setText(f"{t1:.0f}s")
    def jump_to_time(self, target_s):
        self._set_current_step(self._time_to_index(target_s))
        return self.snapshot_state()

    def jump_to_step(self, step):
        self._set_current_step(step)
        return self.snapshot_state()

    def set_playing(self, playing):
        self.playing = bool(playing)

        if hasattr(self, "btn"):
            self.btn.setText("Pause" if self.playing else "Play")

        if hasattr(self, "timer"):
            if self.playing:
                self.timer.start(self.timer_interval_ms)
            else:
                self.timer.stop()

        return self.playing

    def set_timer_interval(self, interval_ms):
        self.timer_interval_ms = max(1, int(interval_ms))

        if hasattr(self, "timer"):
            self.timer.setInterval(self.timer_interval_ms)
            if self.playing and not self.timer.isActive():
                self.timer.start(self.timer_interval_ms)

        return self.timer_interval_ms

    def clear_labels(self):
        self.selected_sat_idx = None
        self.selected_sat_idxs = set()

        try:
            self._remove_sat_label()
        except Exception:
            pass

        if hasattr(self, "_update_pick_status"):
            try:
                self._update_pick_status()
            except Exception:
                pass
        elif hasattr(self, "pick_lbl"):
            self.pick_lbl.setText("selected=none")

        if hasattr(self, "lbl"):
            self.lbl.setText(self._status_text(self.step))

        if hasattr(self, "plotter"):
            self.plotter.render()

        return self.snapshot_state()

    def reload_data(
        self,
        start_s=None,
        end_s=None,
        step_s=None,
        initial_step=0,
        keep_selection=False,
        auto_play=None,
    ):
        new_start = self.start_s if start_s is None else int(start_s)
        new_end = self.end_s if end_s is None else int(end_s)
        new_step = self.step_s if step_s is None else int(step_s)

        preview_meta, ephem_times_s, ephem_positions, sat_ids = load_all_ephemerides(
            self.ephem_dir,
            new_start,
            new_end,
            step_s=new_step,
            cache_root=self.cache_root,
            ignore_cache_source_dir=self.ignore_cache_source_dir,
        )

        new_total = int(ephem_positions.shape[1])
        if new_total != self.total:
            raise ValueError(
                f"Reloaded {new_total} satellites, but current scene expects {self.total}"
            )

        self.start_s = new_start
        self.end_s = new_end
        self.step_s = new_step

        self.preview_meta = preview_meta
        self.ephem_times_s = ephem_times_s
        self.ephem_positions = ephem_positions
        self.sat_ids = sat_ids

        self.range_text = (
            f"range={int(self.ephem_times_s[0])}"
            f"..{int(self.ephem_times_s[-1])}s"
        )

        self.setWindowTitle(
            f"3D Globe + {self.total} Satellites "
            f"({int(self.ephem_times_s[0])}..{int(self.ephem_times_s[-1])}s)"
        )

        if hasattr(self, "slider"):
            self.slider.blockSignals(True)
            self.slider.setRange(0, len(self.ephem_times_s) - 1)
            self.slider.setTickInterval(max(1, len(self.ephem_times_s) // 10))
            self.slider.setPageStep(max(1, len(self.ephem_times_s) // 50))
            self.slider.blockSignals(False)

        if hasattr(self, "jump_input"):
            self.jump_input.setPlaceholderText(
                f"跳转到时间(s)，范围 {int(self.ephem_times_s[0])}-{int(self.ephem_times_s[-1])}"
            )

        if not keep_selection:
            self.selected_sat_idx = None
            self.selected_sat_idxs = set()

            try:
                self._remove_sat_label()
            except Exception:
                pass

            if hasattr(self, "_update_pick_status"):
                try:
                    self._update_pick_status()
                except Exception:
                    pass
            elif hasattr(self, "pick_lbl"):
                self.pick_lbl.setText("selected=none")

        if auto_play is not None:
            self.set_playing(auto_play)

        self.jump_to_step(initial_step)
        return self.snapshot_state()

    def snapshot_state(self):
        return {
            "step": int(self.step),
            "time_s": float(self.ephem_times_s[self.step]),
            "playing": bool(self.playing),
            "timer_interval_ms": int(self.timer_interval_ms),
            "selected_sat_idx": None if self.selected_sat_idx is None else int(self.selected_sat_idx),
            "selected_sat_count": len(self.selected_sat_idxs),
            "range_start_s": float(self.ephem_times_s[0]),
            "range_end_s": float(self.ephem_times_s[-1]),
            "num_steps": int(len(self.ephem_times_s)),
            "num_sats": int(self.total),
        }

    # def _status_text(self, step):
    #     t_s = self.ephem_times_s[step]
    #     text = f"step={step}  t={t_s:.0f}s  {self.range_text}"
    #
    #     if self.selected_sat_idx is not None:
    #         text += f"  sat={self.sat_ids[self.selected_sat_idx]}"
    #
    #     if self.selected_sat_idxs:
    #         text += f"  labels={len(self.selected_sat_idxs)}"
    #
    #     return text
    def _sat_display_name(self, sat_idx):
        return f"SAT {int(sat_idx)}"

    def _status_text(self, step):
        t_s = self.ephem_times_s[step]
        text = f"step={step}  t={t_s:.0f}s  {self.range_text}"

        # if self.selected_sat_idx is not None:
        #     text += f"  sat={self.sat_ids[self.selected_sat_idx]}"
        if self.selected_sat_idx is not None:
            text += f"  sat={self._sat_display_name(self.selected_sat_idx)}"

        if self.selected_sat_idxs:
            text += f"  labels={len(self.selected_sat_idxs)}"

        nodes = self._get_path_nodes_for_step(step)
        if nodes:
            text += f"  route_hops={len(nodes) - 1}"

        return text

    # def _status_text(self, step):
    #     t_s = self.ephem_times_s[step]
    #     text = f"step={step}  t={t_s:.0f}s  {self.range_text}"
    #     if self.selected_sat_idx is not None:
    #         text += f"  sat={self.sat_ids[self.selected_sat_idx]}"
    #     return text

    def sid(self, p, s):
        return p * self.N + s
    def _build_ui(self):
        self.setStyleSheet(f"""
        QWidget {{
            background: {UI_BG};
            color: {TEXT_COLOR};
            font-size: 13px;
        }}
        QFrame#timelinePanel {{
            background: #f8fafc;
            border: 1px solid {UI_BORDER};
            border-radius: 12px;
        }}
        QPushButton {{
            background: {UI_PANEL};
            border: 1px solid {UI_BORDER};
            border-radius: 7px;
            padding: 6px 12px;
            min-height: 30px;
        }}
        QPushButton:hover {{
            background: #eef3f8;
        }}
        QPushButton#navBtn {{
            min-width: 34px;
            max-width: 34px;
            padding: 4px 0;
            font-size: 16px;
            font-weight: 600;
        }}
        QPushButton#jumpBtn {{
            min-width: 72px;
        }}
        QLabel {{
            color: {TEXT_COLOR};
        }}
        QLabel#axisMeta {{
            color: #7b8794;
            font-size: 12px;
        }}
        QLabel#axisCurrent {{
            color: #334155;
            font-size: 12px;
            font-weight: 600;
        }}
        QLineEdit#jumpInput {{
            background: #ffffff;
            border: 1px solid #cfd8e3;
            border-radius: 7px;
            padding: 6px 10px;
            min-width: 140px;
        }}
        QSlider::groove:horizontal {{
            height: 8px;
            border-radius: 4px;
            background: #dde5ec;
        }}
        QSlider::sub-page:horizontal {{
            background: #4f8fc4;
            border-radius: 4px;
        }}
        QSlider::add-page:horizontal {{
            background: #dde5ec;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: #ffffff;
            border: 2px solid #4f8fc4;
            width: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }}
        QSlider::handle:horizontal:hover {{
            border-color: #2f74ad;
        }}
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        self.btn = QPushButton("Pause")
        self.btn.clicked.connect(self._toggle_play)

        self.lbl = QLabel(self._status_text(0))
        self.lbl.setMinimumWidth(360)

        self.pick_lbl = QLabel("selected=none")
        self.pick_lbl.setMinimumWidth(140)

        top_bar.addWidget(self.btn)
        top_bar.addWidget(self.lbl, 1)
        top_bar.addWidget(self.pick_lbl)

        timeline_panel = QFrame()
        timeline_panel.setObjectName("timelinePanel")
        timeline_layout = QVBoxLayout(timeline_panel)
        timeline_layout.setContentsMargins(12, 10, 12, 10)
        timeline_layout.setSpacing(8)

        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)

        self.prev_btn = QPushButton("<")
        self.prev_btn.setObjectName("navBtn")
        self.prev_btn.clicked.connect(self._step_prev)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, len(self.ephem_times_s) - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(max(1, len(self.ephem_times_s) // 10))
        self.slider.setPageStep(max(1, len(self.ephem_times_s) // 50))
        self.slider.valueChanged.connect(self._on_slider)

        self.next_btn = QPushButton(">")
        self.next_btn.setObjectName("navBtn")
        self.next_btn.clicked.connect(self._step_next)

        slider_row.addWidget(self.prev_btn)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(self.next_btn)

        axis_row = QHBoxLayout()
        axis_row.setSpacing(8)

        self.axis_min_lbl = QLabel("")
        self.axis_min_lbl.setObjectName("axisMeta")

        self.axis_cur_lbl = QLabel("")
        self.axis_cur_lbl.setObjectName("axisCurrent")
        self.axis_cur_lbl.setAlignment(Qt.AlignCenter)

        self.axis_max_lbl = QLabel("")
        self.axis_max_lbl.setObjectName("axisMeta")
        self.axis_max_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        axis_row.addWidget(self.axis_min_lbl)
        axis_row.addWidget(self.axis_cur_lbl, 1)
        axis_row.addWidget(self.axis_max_lbl)

        jump_row = QHBoxLayout()
        jump_row.setSpacing(8)

        self.jump_input = QLineEdit()
        self.jump_input.setObjectName("jumpInput")
        self.jump_input.setPlaceholderText(
            f"跳转到时间(s)，范围 {int(self.ephem_times_s[0])}-{int(self.ephem_times_s[-1])}"
        )
        self.jump_input.returnPressed.connect(self._on_jump)

        self.jump_button = QPushButton("跳转")
        self.jump_button.setObjectName("jumpBtn")
        self.jump_button.clicked.connect(self._on_jump)

        jump_row.addWidget(self.jump_input)
        jump_row.addWidget(self.jump_button)
        jump_row.addStretch(1)

        timeline_layout.addLayout(slider_row)
        timeline_layout.addLayout(axis_row)
        timeline_layout.addLayout(jump_row)

       # self.plotter = QtInteractor(self)
        self.plotter = QtInteractor(
            self,
            auto_update=False,
            multi_samples=0,
        )

        root.addLayout(top_bar)
        root.addWidget(timeline_panel)
        root.addWidget(self.plotter.interactor)

        self._sync_timeline_labels()

    # def _build_ui(self):
    #     self.setStyleSheet(f"""
    #     QWidget {{
    #         background: {UI_BG};
    #         color: {TEXT_COLOR};
    #         font-size: 13px;
    #     }}
    #     QPushButton {{
    #         background: {UI_PANEL};
    #         border: 1px solid {UI_BORDER};
    #         border-radius: 5px;
    #         padding: 6px 12px;
    #         min-height: 28px;
    #     }}
    #     QPushButton:hover {{
    #         background: #f1f5f9;
    #     }}
    #     QLabel {{
    #         color: {TEXT_COLOR};
    #     }}
    #     QSlider::groove:horizontal {{
    #         border: 0;
    #         height: 4px;
    #         background: #dce3ea;
    #         border-radius: 2px;
    #     }}
    #     QSlider::handle:horizontal {{
    #         background: #8c98a5;
    #         border: 0;
    #         width: 14px;
    #         margin: -5px 0;
    #         border-radius: 7px;
    #     }}
    #     """)
    #
    #     root = QVBoxLayout(self)
    #     root.setContentsMargins(10, 10, 10, 10)
    #     root.setSpacing(8)
    #
    #     bar = QHBoxLayout()
    #     bar.setSpacing(8)
    #
    #     self.btn = QPushButton("Pause")
    #     self.btn.clicked.connect(self._toggle_play)
    #
    #     self.lbl = QLabel(self._status_text(0))
    #     self.lbl.setMinimumWidth(320)
    #
    #     self.pick_lbl = QLabel("selected=none")
    #     self.pick_lbl.setMinimumWidth(120)
    #
    #     self.slider = QSlider(Qt.Horizontal)
    #     self.slider.setRange(0, len(self.ephem_times_s) - 1)
    #     self.slider.setValue(0)
    #     self.slider.valueChanged.connect(self._on_slider)
    #
    #     bar.addWidget(self.btn)
    #     bar.addWidget(self.lbl)
    #     bar.addWidget(self.pick_lbl)
    #     bar.addWidget(self.slider)
    #
    #     self.plotter = QtInteractor(self)
    #
    #     root.addLayout(bar)
    #     root.addWidget(self.plotter.interactor)

    # def _build_scene(self):
    #     # 白底，和 Cesium 示例一致
    #     self.plotter.set_background(BG_COLOR)
    #     self.plotter.enable_anti_aliasing()
    #
    #     earth = pv.Sphere(radius=EARTH_R_KM, theta_resolution=220, phi_resolution=220)
    #
    #     if EARTH_TEXTURE_PATH:
    #         tex = pv.read_texture(EARTH_TEXTURE_PATH)
    #         self.plotter.add_mesh(
    #             earth,
    #             texture=tex,
    #             smooth_shading=True,
    #             ambient=0.28,
    #             diffuse=0.72,
    #             specular=0.0,
    #             pickable=False,
    #         )
    #     else:
    #         # 没有浅色纹理时，用浅灰白球体 + 大陆轮廓，整体气质接近 Cesium Positron
    #         self.plotter.add_mesh(
    #             earth,
    #             color=EARTH_COLOR,
    #             smooth_shading=True,
    #             ambient=0.35,
    #             diffuse=0.65,
    #             specular=0.0,
    #             pickable=False,
    #         )
    #         #self._add_continent_outlines()
    #
    #         self.add_country_outlines_from_polygons(r"D:\paper3\data\ne_50m_admin_0_countries\ne_50m_admin_0_countries.shp")
    #
    #     # 卫星点云（动态）
    #     self.sat_centers = pv.PolyData(np.zeros((self.total, 3), dtype=float))
    #     self.sat_geom = pv.Sphere(radius=120.0, theta_resolution=18, phi_resolution=18)
    #     self.sat_glyph_mesh = self.sat_centers.glyph(
    #         scale=False,
    #         orient=False,
    #         geom=self.sat_geom,
    #     )
    #
    #     self.sat_actor = self.plotter.add_mesh(
    #         self.sat_glyph_mesh,
    #         color=SAT_COLOR,
    #         smooth_shading=True,
    #         ambient=0.30,
    #         diffuse=0.75,
    #         specular=0.15,
    #         pickable=True,
    #     )
    #     self.plotter.enable_surface_point_picking(
    #         callback=self._on_pick_satellite,
    #         left_clicking=True,
    #         show_point=False,
    #         show_message="Left click the visible satellite to view its ID",
    #         picker="cell",
    #         use_picker=False,
    #     )
    #
    #     self.link_actor = None
    #     self.path_actor = None
    #
    #     # 改成更接近 Cesium 的明亮背景下观察角度
    #     self.plotter.camera_position = [
    #         (18000, -15000, 11000),
    #         (0, 0, 0),
    #         (0, 0, 1),
    #     ]
    def _build_scene(self):
        self.plotter.set_background(BG_COLOR)
    #    self.plotter.enable_anti_aliasing()

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
            self.add_country_outlines_from_polygons(COUNTRY_SHP_PATH)

        self.sat_centers = pv.PolyData(np.zeros((self.total, 3), dtype=np.float32))
        self.sat_actor = self.plotter.add_mesh(
            self.sat_centers,
            render_points_as_spheres=True,
            point_size=11,
            color=SAT_COLOR,
            ambient=0.30,
            opacity=1.0,
            pickable=True,
        )

        self.plotter.enable_point_picking(
            callback=self._on_pick_satellite,
            left_clicking=True,
            show_point=False,
            show_message="Left click the visible satellite to view its ID",
            picker="point",
            tolerance=0.006,
            pickable_window=False,
            clear_on_no_selection=False,
            use_picker=False,
        )

        self.link_actor = None
        self.path_actor = None

        self.plotter.camera_position = [
            (18000, -15000, 11000),
            (0, 0, 0),
            (0, 0, 1),
        ]
        try:
            if hasattr(self.plotter, "iren") and self.plotter.iren is not None:
                self.plotter.iren.add_observer("EndInteractionEvent", self._on_camera_interaction_end)
        except Exception:
            pass

    # def _add_continent_outlines(self):
    #     """
    #     用 VTK 自带的地球大陆轮廓，模拟 Cesium 里的浅色无标注底图。
    #     这样即使没有纹理，也不会只剩一个纯色球。
    #     """
    #     earth_src = vtk.vtkEarthSource()
    #     earth_src.OutlineOn()
    #
    #     if hasattr(earth_src, "SetOnRatio"):
    #         earth_src.SetOnRatio(1)
    #
    #     if hasattr(earth_src, "SetRadius"):
    #         earth_src.SetRadius(EARTH_R_KM * 1.0015)
    #
    #     earth_src.Update()
    #     coast = pv.wrap(earth_src.GetOutput())
    #
    #     # 老版本 vtkEarthSource 可能没有 SetRadius
    #     if not hasattr(earth_src, "SetRadius"):
    #         coast.points = coast.points * (EARTH_R_KM * 1.0015)
    #
    #     self.plotter.add_mesh(
    #         coast,
    #         color=COAST_COLOR,
    #         line_width=1.0,
    #         opacity=0.95,
    #         lighting=False,
    #     )
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

    def add_country_outlines_from_polygons(self, shp_path):
        import geopandas as gpd
        from shapely.ops import unary_union

        gdf = gpd.read_file(shp_path)

        # 关键：合并“边界线”，不是合并“国家面”
        boundary = unary_union(gdf.geometry.boundary)

        r = EARTH_R_KM * 1.004
        segments = []

        for line in self._iter_lines(boundary):
            xyz = np.array(
                [ll_to_xyz(lat, lon, r) for lon, lat in line.coords],
                dtype=float
            )
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

    def _segments_to_polydata(self, segments):
        pts = np.vstack(segments)
        cells = []
        offset = 0

        for seg in segments:
            cells.extend([len(seg), *range(offset, offset + len(seg))])
            offset += len(seg)

        mesh = pv.PolyData(pts)
        mesh.lines = np.array(cells, dtype=np.int64)
        return mesh

    def add_country_borders(self, shp_path):
        gdf = gpd.read_file(shp_path)
        r = EARTH_R_KM * 1.004
        segments = []

        for geom in gdf.geometry:
            if geom is None:
                continue
            geoms = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
            for line in geoms:
                xyz = np.array(
                    [ll_to_xyz(lat, lon, r) for lon, lat in line.coords],
                    dtype=float
                )
                if len(xyz) >= 2:
                    segments.append(xyz)

        border_mesh = self._segments_to_polydata(segments)


        self.plotter.add_mesh(
            border_mesh.copy(),
            color="#ffffff",
            line_width=3.0,
            opacity=0.9,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )
        self.plotter.add_mesh(
            border_mesh,
            color="#6f7c89",
            line_width=1.2,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )

    def _add_continent_outlines(self):
        earth_src = vtk.vtkEarthSource()
        earth_src.OutlineOn()
        earth_src.SetOnRatio(1)

        outline_r = EARTH_R_KM * 1.005
        if hasattr(earth_src, "SetRadius"):
            earth_src.SetRadius(outline_r)

        earth_src.Update()
        coast = pv.wrap(earth_src.GetOutput())

        if not hasattr(earth_src, "SetRadius"):
            coast.points = coast.points * outline_r

        # 白色底描边，制造“发亮边”
        self.plotter.add_mesh(
            coast.copy(),
            color="#ffffff",
            line_width=3.6,
            opacity=0.95,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )

        # 主轮廓线
        self.plotter.add_mesh(
            coast,
            color="#7f8a96",
            line_width=1.6,
            opacity=1.0,
            lighting=False,
            render_lines_as_tubes=True,
            pickable=False,
        )

    def _build_static_layers(self):
        stations_ll = [
            (-15.7939, -47.8828),  # Brazil
            (12.1140, -86.2362),  # Nicaragua
            (30.0444, 31.2357),  # Egypt
            (39.9042, 116.4074),  # Beijing
            (-33.8688, 151.2093),  # Sydney
        ]
        station_xyz = np.array(
            [ll_to_xyz(lat, lon, EARTH_R_KM) for lat, lon in stations_ll],
            dtype=float
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

    def _sat_positions(self, step):
        # 简单轨道动力学演示：按均匀角速度推进
        t = step * 0.015
        w = 0.9
        inc = np.deg2rad(INC_DEG)
        r = EARTH_R_KM + SAT_ALT_KM

        pts = np.zeros((self.total, 3), dtype=float)
        for p in range(self.P):
            raan = 2 * np.pi * p / self.P
            R = rot_z(raan) @ rot_x(inc)
            for s in range(self.N):
                u = 2 * np.pi * s / self.N + w * t
                vec_orb = np.array([r * np.cos(u), r * np.sin(u), 0.0], dtype=float)
                pts[self.sid(p, s)] = R @ vec_orb

        # 地球坐标系有一点自转视觉
        earth_spin = -0.08 * t
        pts = (rot_z(earth_spin) @ pts.T).T
        return pts

    @staticmethod
    def _edge_mesh(points, edges):
        if not edges:
            return None

        seg_pts = np.empty((2 * len(edges), 3), dtype=float)
        line_cells = np.empty((len(edges), 3), dtype=np.int64)
        for k, (i, j) in enumerate(edges):
            seg_pts[2 * k] = points[i]
            seg_pts[2 * k + 1] = points[j]
            line_cells[k] = [2, 2 * k, 2 * k + 1]

        mesh = pv.PolyData(seg_pts)
        mesh.lines = line_cells.ravel()
        return mesh

    def _build_edges(self):
        edges = []

        # 同轨
        for p in range(self.P):
            for s in range(self.N):
                i = self.sid(p, s)
                j = self.sid(p, (s + 1) % self.N)
                edges.append((i, j))

        # 跨轨（稀疏一些，避免太乱）
        for p in range(self.P):
            pn = (p + 1) % self.P
            for s in range(0, self.N, 3):
                i = self.sid(p, s)
                j = self.sid(pn, s)
                edges.append((i, j))

        return edges

    def _build_path_nodes(self, step):
        # 演示路径：随时间平移
        p0 = (step // 30) % self.P
        s = 2
        nodes = []
        for k in range(12):
            pp = (p0 + k) % self.P
            ss = (s + k) % self.N
            nodes.append(self.sid(pp, ss))
        return nodes
    def _remove_path_overlay(self):
        for actor_name in (
            "route_path_actor",
            "route_nodes_actor",
            "route_start_actor",
            "route_end_actor",
        ):
            try:
                self.plotter.remove_actor(actor_name, render=False)
            except Exception:
                pass


    def set_paths(self, path_by_time, render=True):
        cleaned = {}

        for key, value in (path_by_time or {}).items():
            if value is None:
                continue

            if isinstance(value, str):
                nodes = [int(x) for x in value.split("->") if str(x).strip()]
            else:
                nodes = [int(x) for x in value]

            if len(nodes) >= 2:
                cleaned[int(key)] = nodes

        self.path_by_time = cleaned

        if render:
            self._refresh_path_overlay()
            self.plotter.render()

        return self.path_by_time


    def set_paths_from_dataframe(self, df, time_col="time", path_col="path", render=True):
        path_by_time = {}

        for row in df[[time_col, path_col]].itertuples(index=False):
            t = getattr(row, time_col)
            p = getattr(row, path_col)

            if isinstance(p, str) and p.strip():
                path_by_time[int(t)] = [int(x) for x in p.split("->")]

        return self.set_paths(path_by_time, render=render)


    def _get_path_nodes_for_step(self, step):
        # 优先按真实时间取路径
        t_key = int(round(float(self.ephem_times_s[step])))

        nodes = self.path_by_time.get(t_key)
        if nodes is None:
            # 兼容有人直接用 step 当 key 的情况
            nodes = self.path_by_time.get(int(step))

        if nodes is not None:
            nodes = [int(x) for x in nodes if 0 <= int(x) < self.total]
            return nodes if len(nodes) >= 2 else None

        # 如果没注入真实路径，但你想保留原来的演示效果
        if self.show_demo_path:
            nodes = self._build_path_nodes(step)
            if nodes and len(nodes) >= 2:
                return nodes

        return None


    def _refresh_path_overlay(self):
        self._remove_path_overlay()

        nodes = self._get_path_nodes_for_step(self.step)
        if not nodes or len(nodes) < 2:
            return

        pts = np.asarray(self.sat_centers.points, dtype=float)
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

        path_mesh = self._edge_mesh(pts, edges)
        if path_mesh is not None:
            self.plotter.add_mesh(
                path_mesh,
                name="route_path_actor",
                color="#ff7a00",
                line_width=6.0,
                opacity=1.0,
                lighting=False,
                render_lines_as_tubes=True,
                pickable=False,
                reset_camera=False,
                render=False,
            )

        route_nodes = pv.PolyData(pts[nodes])
        self.plotter.add_mesh(
            route_nodes,
            name="route_nodes_actor",
            render_points_as_spheres=True,
            point_size=14,
            color="#ffd166",
            ambient=0.35,
            pickable=False,
            reset_camera=False,
            render=False,
        )

        start_node = pv.PolyData(pts[[nodes[0]]])
        self.plotter.add_mesh(
            start_node,
            name="route_start_actor",
            render_points_as_spheres=True,
            point_size=18,
            color="#22c55e",
            ambient=0.35,
            pickable=False,
            reset_camera=False,
            render=False,
        )

        end_node = pv.PolyData(pts[[nodes[-1]]])
        self.plotter.add_mesh(
            end_node,
            name="route_end_actor",
            render_points_as_spheres=True,
            point_size=18,
            color="#ef4444",
            ambient=0.35,
            pickable=False,
            reset_camera=False,
            render=False,
        )

    # def _on_pick_satellite(self, picked_point):
    #     if picked_point is None:
    #         return
    #
    #     sat_idx = int(self.sat_centers.find_closest_point(picked_point))
    #     self.selected_sat_idx = sat_idx
    #     self._refresh_selected_sat_label()
    #     self.lbl.setText(self._status_text(self.step))
    #     self.plotter.render()

    # def _on_pick_satellite(self, picked_point):
    #     if picked_point is None:
    #         return
    #
    #     sat_idx = int(self.sat_centers.find_closest_point(picked_point))
    #     self.selected_sat_idx = sat_idx
    #
    #     sat_id = self.sat_ids[sat_idx]
    #     self.pick_lbl.setText(f"selected=SAT {sat_id}")
    #
    #     self._refresh_selected_sat_label()
    #     self.lbl.setText(self._status_text(self.step))
    #     self.plotter.render()
    # def _on_pick_satellite(self, picked_point):
    #     if picked_point is None:
    #         return
    #
    #     sat_idx = int(self.sat_centers.find_closest_point(picked_point))
    #
    #     # 再点同一颗卫星：取消标签
    #     if self.selected_sat_idx == sat_idx:
    #         self.selected_sat_idx = None
    #         self.selected_sat_idxs = set()
    #
    #         self._remove_sat_label()
    #         self.pick_lbl.setText("selected=none")
    #         self.lbl.setText(self._status_text(self.step))
    #         self.plotter.render()
    #         return
    #
    #     # 点到新的卫星：切换到新的标签
    #     self.selected_sat_idx = sat_idx
    #     sat_id = self.sat_ids[sat_idx]
    #     self.pick_lbl.setText(f"selected=SAT {sat_id}")
    #
    #     self._refresh_selected_sat_label()
    #     self.lbl.setText(self._status_text(self.step))
    #     self.plotter.render()
    def _on_pick_satellite(self, picked_point):
        if picked_point is None:
            return

        sat_idx = int(self.sat_centers.find_closest_point(picked_point))

        # 再点同一颗：只取消这一颗
        if sat_idx in self.selected_sat_idxs:
            self.selected_sat_idxs.remove(sat_idx)


            if self.selected_sat_idx == sat_idx:
                if self.selected_sat_idxs:
                    self.selected_sat_idx = sorted(self.selected_sat_idxs)[-1]
                else:
                    self.selected_sat_idx = None
        else:
            # 点到新卫星：追加标签，不删旧标签
            self.selected_sat_idxs.add(sat_idx)
            self.selected_sat_idx = sat_idx

        self._refresh_selected_sat_label()
        self._update_pick_status()
        self.lbl.setText(self._status_text(self.step))
        self.plotter.render()

    # def _refresh_selected_sat_label(self):
    #     self._remove_sat_label()
    #
    #     if self.selected_sat_idx is None:
    #         return
    #
    #     sat_id = self.sat_ids[self.selected_sat_idx]
    #     center = np.asarray(self.sat_centers.points[self.selected_sat_idx], dtype=float)
    #
    #     if not self._is_sat_visible_from_camera(center):
    #         return
    #
    #     camera_pos = np.asarray(self.plotter.camera_position[0], dtype=float)
    #     cam_vec = camera_pos - center
    #     cam_norm = np.linalg.norm(cam_vec)
    #
    #     if cam_norm < 1e-9:
    #         cam_dir = np.array([0.0, 0.0, 1.0], dtype=float)
    #     else:
    #         cam_dir = cam_vec / cam_norm
    #
    #     label_point = np.asarray([center + cam_dir * 220.0])
    #
    #     self.plotter.add_point_labels(
    #         label_point,
    #         [f"SAT {sat_id}"],
    #         name="sat_pick_label",
    #         always_visible=True,
    #         show_points=False,
    #         text_color="#1d4ed8",
    #         shape_color="#eff6ff",
    #         shape_opacity=0.95,
    #         font_size=14,
    #         pickable=False,
    #         reset_camera=False,
    #         render=False,
    #     )
    def _refresh_selected_sat_label(self):
        self._remove_sat_label()

        if not self.selected_sat_idxs:
            return

        camera_pos = np.asarray(self.plotter.camera_position[0], dtype=float)

        label_points = []
        label_texts = []

        for sat_idx in sorted(self.selected_sat_idxs):
            center = np.asarray(self.sat_centers.points[sat_idx], dtype=float)

            # 背到地球后面时，不显示标签，但保留选中状态
            if not self._is_sat_visible_from_camera(center):
                continue

            cam_vec = camera_pos - center
            cam_norm = np.linalg.norm(cam_vec)

            if cam_norm < 1e-9:
                cam_dir = np.array([0.0, 0.0, 1.0], dtype=float)
            else:
                cam_dir = cam_vec / cam_norm

            label_point = center + cam_dir * 220.0

            label_points.append(label_point)
            #label_texts.append(f"SAT {self.sat_ids[sat_idx]}")
            label_texts.append(self._sat_display_name(sat_idx))

        if not label_points:
            return

        self.plotter.add_point_labels(
            np.asarray(label_points, dtype=float),
            label_texts,
            name="sat_pick_labels",
            always_visible=True,
            show_points=False,
            text_color="#1d4ed8",
            shape_color="#eff6ff",
            shape_opacity=0.95,
            font_size=14,
            pickable=False,
            reset_camera=False,
            render=False,
        )

    def _on_camera_interaction_end(self, *args):
        if self.selected_sat_idxs:
            self._refresh_selected_sat_label()
        else:
            self._remove_sat_label()

        self.plotter.render()

    # def _update_frame(self, step):
    #     self.step = max(0, min(int(step), len(self.ephem_times_s) - 1))
    #     pts = self.ephem_positions[self.step]
    #
    #     self.sat_centers.points = pts
    #     self.sat_centers.Modified()
    #
    #     new_glyph = self.sat_centers.glyph(
    #         scale=False,
    #         orient=False,
    #         geom=self.sat_geom,
    #     )
    #     self.sat_glyph_mesh.copy_from(new_glyph)
    #
    #     if self.selected_sat_idx is not None:
    #         self._refresh_selected_sat_label()
    #
    #     self.lbl.setText(self._status_text(self.step))
    #     self.plotter.render()
    # def _update_frame(self, step):
    #     self.step = max(0, min(int(step), len(self.ephem_times_s) - 1))
    #     pts = np.array(self.ephem_positions[self.step], dtype=np.float32, copy=True)
    #
    #     self.sat_centers.points = pts
    #     self.sat_centers.Modified()
    #
    #     if self.selected_sat_idx is not None:
    #         self._refresh_selected_sat_label()
    #
    #     self.lbl.setText(self._status_text(self.step))
    #     self.plotter.render()


    # def _update_frame(self, step):
    #     self.step = max(0, min(int(step), len(self.ephem_times_s) - 1))
    #     pts = np.array(self.ephem_positions[self.step], dtype=np.float32, copy=True)
    #
    #     self.sat_centers.points = pts
    #     self.sat_centers.Modified()
    #     if self.selected_sat_idxs:
    #         self._refresh_selected_sat_label()
    #     else:
    #         self._remove_sat_label()
    #
    #     # if self.selected_sat_idx is not None:
    #     #     self._refresh_selected_sat_label()
    #
    #     self.lbl.setText(self._status_text(self.step))
    #     self._sync_timeline_labels()
    #     self.plotter.render()
    def _update_frame(self, step):
        self.step = max(0, min(int(step), len(self.ephem_times_s) - 1))
        pts = np.array(self.ephem_positions[self.step], dtype=np.float32, copy=True)

        self.sat_centers.points = pts
        self.sat_centers.Modified()

        # 路径跟着当前时刻一起刷新
        self._refresh_path_overlay()

        if self.selected_sat_idxs:
            self._refresh_selected_sat_label()
        else:
            self._remove_sat_label()

        self.lbl.setText(self._status_text(self.step))
        self._sync_timeline_labels()
        self.plotter.render()

    # def _tick(self):
    #     if not self.playing:
    #         return
    #
    #     nxt = (self.step + 1) % (self.slider.maximum() + 1)
    #     self.slider.blockSignals(True)
    #     self.slider.setValue(nxt)
    #     self.slider.blockSignals(False)
    #     self._update_frame(nxt)
    def _tick(self):
        if not self.playing:
            return

        nxt = (self.step + 1) % (self.slider.maximum() + 1)
        self.slider.blockSignals(True)
        self.slider.setValue(nxt)
        self.slider.blockSignals(False)
        self._update_frame(nxt)

    def _on_slider(self, v):
        self._update_frame(int(v))

    def _toggle_play(self):
        self.set_playing(not self.playing)
    def shutdown(self):
        self.playing = False

        try:
            if hasattr(self, "timer") and self.timer is not None:
                self.timer.stop()
        except Exception:
            pass

        for fn_name in ("_remove_sat_label", "_remove_path_overlay"):
            try:
                fn = getattr(self, fn_name, None)
                if fn is not None:
                    fn()
            except Exception:
                pass

        try:
            if hasattr(self, "plotter") and self.plotter is not None:
                # 先尽量关掉交互/渲染
                try:
                    self.plotter.close()
                except Exception:
                    pass

                # 再尝试 finalize VTK render window
                ren_win = getattr(self.plotter, "ren_win", None)
                if ren_win is None:
                    ren_win = getattr(self.plotter, "render_window", None)

                if ren_win is not None:
                    try:
                        ren_win.Finalize()
                    except Exception:
                        pass

                # 如果 interactor 是单独对象，也一并关掉
                interactor = getattr(self.plotter, "interactor", None)
                if interactor is not None and interactor is not self.plotter:
                    try:
                        interactor.close()
                    except Exception:
                        pass
                    try:
                        interactor.deleteLater()
                    except Exception:
                        pass

                try:
                    self.plotter.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

def get_or_create_qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def create_globe_demo(
    P=18,
    N=36,
    start_s=PREVIEW_START_S,
    end_s=PREVIEW_END_S,
    step_s=EPHEM_STEP_S,
    auto_play=False,
    timer_interval_ms=200,
    initial_step=0,
    ephem_dir=None,
    cache_root=None,
    ignore_cache_source_dir=False,
):
    app = get_or_create_qapp()
    window = GlobeSatDemo(
        P=P,
        N=N,
        start_s=start_s,
        end_s=end_s,
        step_s=step_s,
        auto_play=auto_play,
        timer_interval_ms=timer_interval_ms,
        initial_step=initial_step,
        ephem_dir=ephem_dir,
        cache_root=cache_root,
        ignore_cache_source_dir=ignore_cache_source_dir,
    )

    return app, window



def show_globe_demo(
    P=18,
    N=36,
    start_s=PREVIEW_START_S,
    end_s=PREVIEW_END_S,
    step_s=EPHEM_STEP_S,
    auto_play=False,
    timer_interval_ms=200,
    initial_step=0,
    ephem_dir=None,
    cache_root=None,
    ignore_cache_source_dir=False,
):
    app, window = create_globe_demo(
        P=P,
        N=N,
        start_s=start_s,
        end_s=end_s,
        step_s=step_s,
        auto_play=auto_play,
        timer_interval_ms=timer_interval_ms,
        initial_step=initial_step,
        ephem_dir=ephem_dir,
        cache_root=cache_root,
        ignore_cache_source_dir=ignore_cache_source_dir,
    )

    window.show()

    try:
        window.raise_()
        window.activateWindow()
    except Exception:
        pass

    return app, window



def close_globe_demo(window=None):
    app = QApplication.instance()

    if window is not None:
        try:
            window.shutdown()
        except Exception:
            pass

        try:
            window.close()
        except Exception:
            pass

        try:
            window.deleteLater()
        except Exception:
            pass

        if app is not None:
            try:
                app.processEvents()
            except Exception:
                pass
        return

    if app is None:
        return

    for w in list(app.topLevelWidgets()):
        if isinstance(w, GlobeSatDemo):
            try:
                w.shutdown()
            except Exception:
                pass
            try:
                w.close()
            except Exception:
                pass
            try:
                w.deleteLater()
            except Exception:
                pass

    try:
        app.processEvents()
    except Exception:
        pass


def run_globe_demo(
    P=18,
    N=36,
    start_s=PREVIEW_START_S,
    end_s=PREVIEW_END_S,
    step_s=EPHEM_STEP_S,
    auto_play=True,
    timer_interval_ms=50,
    initial_step=0,
    ephem_dir=None,
    cache_root=None,
    ignore_cache_source_dir=False,
):
    app, window = show_globe_demo(
        P=P,
        N=N,
        start_s=start_s,
        end_s=end_s,
        step_s=step_s,
        auto_play=auto_play,
        timer_interval_ms=timer_interval_ms,
        initial_step=initial_step,
        ephem_dir=ephem_dir,
        cache_root=cache_root,
        ignore_cache_source_dir=ignore_cache_source_dir,
    )
    return app.exec_()



if __name__ == "__main__":
    run_globe_demo(P=18, N=36)
