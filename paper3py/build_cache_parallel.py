from pathlib import Path
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from numpy.lib.format import open_memmap


CACHE_VERSION = 2
EXPECTED_SAT_COUNT = 648
DEFAULT_START_S = 0
DEFAULT_END_S = 86164
DEFAULT_STEP_S = 1
DEFAULT_WORKERS = min(8, max(2, (os.cpu_count() or 4) // 2))
DEFAULT_PROGRESS_EVERY = 8
DEFAULT_MODE = "memory"   # "memory" or "memmap"
DEFAULT_FLUSH_EVERY = 64  # only used in memmap mode


def first_existing_path(*paths):
    for raw in paths:
        p = Path(raw)
        if p.exists():
            return p
    return Path(paths[0])


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
SIM_OUTPUT_DIR = SCRIPT_DIR / "sim_output"
CACHE_DIR = SIM_OUTPUT_DIR / "cache_86164s_1s"

EPHEM_DIR = first_existing_path(
    PROJECT_DIR / "data" / "satellitesposition" / "satellite_pos",
    r"D:\paper3\data\satellitesposition\satellite_pos",
    r"C:\user\data\satellitesposition\satellite_pos",
)


def _tmp_path(path: Path, suffix: str = ".tmp") -> Path:
    return path.with_name(path.name + suffix)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path, ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_npy_atomic(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path, ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    os.replace(tmp, path)


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


def get_cache_dir(start_s, end_s, step_s=DEFAULT_STEP_S):
    start_i = int(0 if start_s is None else start_s)
    end_i = int(end_s)
    step_i = int(step_s)

    if start_i == 0 and end_i == 86164 and step_i == 1:
        return CACHE_DIR

    return SIM_OUTPUT_DIR / f"cache_{start_i}_{end_i}_{step_i}s"


def get_cache_paths(start_s, end_s, step_s=DEFAULT_STEP_S):
    cache_dir = get_cache_dir(start_s, end_s, step_s)
    return {
        "dir": cache_dir,
        "times": cache_dir / "times_s.npy",
        "positions": cache_dir / "positions_km.npy",
        "sat_ids": cache_dir / "sat_ids.json",
        "meta": cache_dir / "cache_meta.json",
        "report": cache_dir / "build_report.json",
        "progress": cache_dir / "parse_progress.jsonl",
    }


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


def cache_meta_matches(cached_meta, expected_meta):
    if not isinstance(cached_meta, dict):
        return False

    for key, value in expected_meta.items():
        if cached_meta.get(key) != value:
            return False

    return True


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

            vals = np.fromstring(line, sep=" ")
            if vals.size < 4:
                continue

            t_s = float(vals[0])
            x_km = float(vals[1]) * dist_scale_km
            y_km = float(vals[2]) * dist_scale_km
            z_km = float(vals[3]) * dist_scale_km

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

    arr = np.asarray(rows, dtype=np.float64)
    times_s = arr[:, 0]
    positions_km = arr[:, 1:4].astype(np.float32)

    meta["file_time_start_s"] = float(file_time_start_s)
    meta["file_time_end_s"] = float(file_time_end_s)
    meta["actual_start_s"] = float(times_s[0])
    meta["actual_end_s"] = float(times_s[-1])

    return meta, times_s, positions_km


def validate_time_axis(times_s, step_s, label):
    times_s = np.asarray(times_s, dtype=np.float64)
    if times_s.ndim != 1:
        raise ValueError(f"{label}: time axis must be 1D")

    if len(times_s) < 2:
        raise ValueError(f"{label}: time axis has fewer than 2 samples")

    expected_step = float(step_s)
    dt = np.diff(times_s)
    if not np.allclose(dt, expected_step, rtol=0.0, atol=1e-9):
        bad = np.flatnonzero(np.abs(dt - expected_step) > 1e-9)
        idx = int(bad[0]) if bad.size else 0
        raise ValueError(
            f"{label}: non-uniform time axis at index {idx}; "
            f"got delta={dt[idx]:.12g}, expected={expected_step}"
        )

    expected_len = int(round((times_s[-1] - times_s[0]) / expected_step)) + 1
    if expected_len != len(times_s):
        raise ValueError(
            f"{label}: inconsistent time axis length; "
            f"expected {expected_len}, got {len(times_s)}"
        )

    return times_s


def parse_reference_satellite(path, start_s, end_s, step_s):
    path = Path(path)
    t0 = time.perf_counter()

    meta, times_s, positions_km = parse_stk_ephemeris(
        path,
        start_s=start_s,
        end_s=end_s,
    )
    times_s = validate_time_axis(times_s, step_s, path.name)

    return {
        "sat_index": 0,
        "sat_id": path.stem,
        "path": str(path),
        "times_s": np.asarray(times_s, dtype=np.float64),
        "positions_km": np.asarray(positions_km, dtype=np.float32),
        "actual_start_s": float(meta["actual_start_s"]),
        "actual_end_s": float(meta["actual_end_s"]),
        "num_points": int(len(times_s)),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def parse_one_satellite(task):
    sat_idx, path_str, start_s, end_s, step_s = task
    path = Path(path_str)
    t0 = time.perf_counter()

    meta, times_s, positions_km = parse_stk_ephemeris(
        path,
        start_s=start_s,
        end_s=end_s,
    )
    times_s = validate_time_axis(times_s, step_s, path.name)

    return {
        "sat_index": sat_idx,
        "sat_id": path.stem,
        "path": str(path),
        "positions_km": np.asarray(positions_km, dtype=np.float32),
        "actual_start_s": float(meta["actual_start_s"]),
        "actual_end_s": float(meta["actual_end_s"]),
        "num_points": int(len(times_s)),
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


def maybe_reuse_cache(ephem_dir, start_s, end_s, step_s):
    files = list_ephemeris_files(ephem_dir)
    paths = get_cache_paths(start_s, end_s, step_s)
    expected_meta = build_expected_meta(ephem_dir, files, start_s, end_s, step_s)

    required = ("times", "positions", "sat_ids", "meta", "report", "progress")
    if not all(paths[key].exists() for key in required):
        return False

    cached_meta = read_json(paths["meta"])
    if not cache_meta_matches(cached_meta, expected_meta):
        return False

    report = read_json(paths["report"])
    if not isinstance(report, dict):
        return False
    if report.get("status") != "completed" or not bool(report.get("build_succeeded", False)):
        return False

    print(f"[cache] reusing cache from {paths['dir']}", flush=True)
    return True


def positions_cache_mib(num_steps, num_sats):
    nbytes = int(num_steps) * int(num_sats) * 3 * np.dtype(np.float32).itemsize
    return nbytes / (1024 * 1024)


def make_sat_record(result):
    return {
        "sat_id": result["sat_id"],
        "sat_index": int(result["sat_index"]),
        "num_points": int(result["num_points"]),
        "elapsed_s": float(result["elapsed_s"]),
        "status": "ok",
        "actual_start_s": float(result["actual_start_s"]),
        "actual_end_s": float(result["actual_end_s"]),
    }


def make_failure_record(task, error_text):
    sat_idx, path_str, *_ = task
    return {
        "sat_id": Path(path_str).stem,
        "sat_index": int(sat_idx),
        "path": str(path_str),
        "status": "failed",
        "error": str(error_text),
    }


def build_cache_parallel(
    ephem_dir,
    start_s,
    end_s,
    step_s,
    workers,
    progress_every,
    mode=DEFAULT_MODE,
    flush_every=DEFAULT_FLUSH_EVERY,
):
    ephem_dir = Path(ephem_dir)
    files = list_ephemeris_files(ephem_dir)
    paths = get_cache_paths(start_s, end_s, step_s)
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
        "workers": int(workers),
        "source_dir": str(ephem_dir.resolve()),
        "source_files": expected_meta["source_files"],
        "time_axis_check": "pending",
        "storage_mode": str(mode),
        "flush_every": int(flush_every) if mode == "memmap" else None,
        "satellite_records": [],
        "failures": [],
    }
    write_json(paths["report"], report)

    print(f"[cache] building cache in {paths['dir']}", flush=True)
    print(
        f"[cache] workers={workers}, progress_every={progress_every}, mode={mode}",
        flush=True,
    )

    positions_store = None
    temp_positions_path = None
    ref_times_s = None
    sat_ids = [None] * len(files)
    done = 0
    first_parse_elapsed = None
    parse_fill_elapsed = None
    save_elapsed = None

    try:
        first_result = parse_reference_satellite(files[0], start_s, end_s, step_s)
        first_parse_elapsed = float(first_result["elapsed_s"])
        ref_times_s = first_result["times_s"]

        est_mib = positions_cache_mib(len(ref_times_s), len(files))
        print(
            f"[cache] positions shape=({len(ref_times_s)}, {len(files)}, 3) | "
            f"estimated_size={est_mib:.1f} MiB",
            flush=True,
        )

        if mode == "memory":
            positions_store = np.empty(
                (len(ref_times_s), len(files), 3),
                dtype=np.float32,
            )
        elif mode == "memmap":
            temp_positions_path = _tmp_path(paths["positions"], ".building")
            if temp_positions_path.exists():
                temp_positions_path.unlink()
            positions_store = open_memmap(
                temp_positions_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(ref_times_s), len(files), 3),
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        positions_store[:, 0, :] = first_result["positions_km"]
        sat_ids[0] = first_result["sat_id"]
        done = 1

        first_record = make_sat_record(first_result)
        report["satellite_records"].append(first_record)
        report["time_axis_check"] = "uniform"
        append_jsonl(paths["progress"], first_record)

        print(
            f"[cache] parsed {done}/{len(files)} satellites | "
            f"last={first_result['sat_id']} | "
            f"sat_elapsed={first_result['elapsed_s']:.3f}s | "
            f"total_elapsed={time.perf_counter() - build_started:.1f}s",
            flush=True,
        )

        tasks = [
            (sat_idx, str(path), start_s, end_s, step_s)
            for sat_idx, path in enumerate(files[1:], start=1)
        ]

        parse_fill_started = time.perf_counter()

        with cf.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(parse_one_satellite, task): task for task in tasks}

            for future in cf.as_completed(futures):
                task = futures[future]

                try:
                    result = future.result()
                except Exception as exc:
                    failure_record = make_failure_record(task, exc)
                    report["failures"].append(failure_record)
                    append_jsonl(paths["progress"], failure_record)
                    write_json(paths["report"], report)
                    raise

                if result["num_points"] != len(ref_times_s):
                    raise ValueError(
                        f"Time axis length mismatch in file: {result['sat_id']}.e | "
                        f"expected={len(ref_times_s)}, got={result['num_points']}"
                    )
                if not np.isclose(result["actual_start_s"], float(ref_times_s[0]), rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Time axis start mismatch in file: {result['sat_id']}.e | "
                        f"expected={float(ref_times_s[0])}, got={result['actual_start_s']}"
                    )
                if not np.isclose(result["actual_end_s"], float(ref_times_s[-1]), rtol=0.0, atol=1e-9):
                    raise ValueError(
                        f"Time axis end mismatch in file: {result['sat_id']}.e | "
                        f"expected={float(ref_times_s[-1])}, got={result['actual_end_s']}"
                    )

                sat_idx = result["sat_index"]
                positions_store[:, sat_idx, :] = result["positions_km"]

                if mode == "memmap" and (
                    done % max(1, int(flush_every)) == 0 or done + 1 == len(files)
                ):
                    positions_store.flush()

                sat_ids[sat_idx] = result["sat_id"]
                done += 1

                sat_record = make_sat_record(result)
                report["satellite_records"].append(sat_record)
                append_jsonl(paths["progress"], sat_record)

                if done % progress_every == 0 or done == len(files):
                    elapsed_total = time.perf_counter() - build_started
                    avg_per_sat = elapsed_total / done
                    eta_s = avg_per_sat * (len(files) - done)

                    print(
                        f"[cache] parsed {done}/{len(files)} satellites | "
                        f"last={result['sat_id']} | "
                        f"sat_elapsed={result['elapsed_s']:.3f}s | "
                        f"total_elapsed={elapsed_total:.1f}s | "
                        f"eta={eta_s:.1f}s",
                        flush=True,
                    )

        parse_fill_elapsed = time.perf_counter() - parse_fill_started

        if any(sid is None for sid in sat_ids):
            raise ValueError("Some satellite slots were not written into cache")

        print("[cache] parse/fill completed, saving cache files...", flush=True)
        save_started = time.perf_counter()

        save_npy_atomic(paths["times"], ref_times_s)

        if mode == "memory":
            save_npy_atomic(paths["positions"], positions_store)
        else:
            positions_store.flush()
            del positions_store
            positions_store = None
            os.replace(temp_positions_path, paths["positions"])

        save_elapsed = time.perf_counter() - save_started

        cache_meta = dict(expected_meta)
        cache_meta.update({
            "actual_start_s": float(ref_times_s[0]),
            "actual_end_s": float(ref_times_s[-1]),
            "num_steps": int(len(ref_times_s)),
        })

        write_json(paths["sat_ids"], sat_ids)
        write_json(paths["meta"], cache_meta)

        report["satellite_records"].sort(key=lambda x: x["sat_index"])
        report["status"] = "completed"
        report["build_succeeded"] = True
        report["elapsed_s"] = round(time.perf_counter() - build_started, 3)
        report["actual_start_s"] = float(ref_times_s[0])
        report["actual_end_s"] = float(ref_times_s[-1])
        report["num_steps"] = int(len(ref_times_s))
        report["phase_timings"] = {
            "reference_parse_s": round(float(first_parse_elapsed or 0.0), 3),
            "parallel_parse_fill_s": round(float(parse_fill_elapsed or 0.0), 3),
            "save_s": round(float(save_elapsed or 0.0), 3),
        }
        report["positions_shape"] = [int(len(ref_times_s)), int(len(files)), 3]
        report["positions_dtype"] = "float32"
        report["estimated_positions_mib"] = round(est_mib, 3)
        write_json(paths["report"], report)

        print(
            f"[cache] completed | elapsed={report['elapsed_s']:.1f}s | "
            f"parse_fill={report['phase_timings']['parallel_parse_fill_s']:.1f}s | "
            f"save={report['phase_timings']['save_s']:.1f}s | "
            f"cache_dir={paths['dir']}",
            flush=True,
        )
        return 0

    except Exception as exc:
        try:
            if positions_store is not None and mode == "memmap":
                positions_store.flush()
        except Exception:
            pass

        report["status"] = "failed"
        report["build_succeeded"] = False
        report["elapsed_s"] = round(time.perf_counter() - build_started, 3)
        report["error"] = str(exc)
        if ref_times_s is not None and len(ref_times_s) > 0:
            report["actual_start_s"] = float(ref_times_s[0])
            report["actual_end_s"] = float(ref_times_s[-1])
            report["num_steps"] = int(len(ref_times_s))
        write_json(paths["report"], report)

        print(f"[cache] failed: {exc}", file=sys.stderr, flush=True)
        raise

    finally:
        if mode == "memmap":
            try:
                if positions_store is not None:
                    del positions_store
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Build satellite cache in parallel.")
    parser.add_argument("--start", type=int, default=DEFAULT_START_S)
    parser.add_argument("--end", type=int, default=DEFAULT_END_S)
    parser.add_argument("--step", type=int, default=DEFAULT_STEP_S)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--progress-every", type=int, default=DEFAULT_PROGRESS_EVERY)
    parser.add_argument("--mode", choices=("memory", "memmap"), default=DEFAULT_MODE)
    parser.add_argument("--flush-every", type=int, default=DEFAULT_FLUSH_EVERY)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be >= 1")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be >= 1")
    if args.step != 1:
        raise ValueError("This builder currently expects --step 1")

    if not args.force and maybe_reuse_cache(EPHEM_DIR, args.start, args.end, args.step):
        return 0

    return build_cache_parallel(
        EPHEM_DIR,
        args.start,
        args.end,
        args.step,
        workers=args.workers,
        progress_every=args.progress_every,
        mode=args.mode,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
