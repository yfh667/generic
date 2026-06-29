from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path(r"E:\paper11\data_paper2\newdata")
DEFAULT_XML = DATA_ROOT / "xml" / "G60_baseraan_0.xml"
DEFAULT_CACHE_DIR = DATA_ROOT / "cache" / "g60_voc"
DEFAULT_FIGURE_DIR = Path(__file__).resolve().parent / "figures"


GROUP_POINTS: list[tuple[str, list[tuple[float, float]]]] = [
    (
        "china",
        [
            (30, 84),
            (19, 110),
            (22, 102),
            (27, 98),
            (25, 118),
            (34, 119),
            (41, 126),
            (48, 126),
            (36, 114),
            (36, 102),
            (36, 90),
            (36, 78),
        ],
    ),
    (
        "europe",
        [
            (43, 3.75),
            (52, 12),
            (46, 12),
            (46, 4.4),
            (47, -0.4),
            (41.8, 13.6),
            (39.6, -5),
            (59.8, 13.4),
            (54, -1.6),
        ],
    ),
    (
        "north_america",
        [
            (40.71, -74.01),
            (34.05, -118.24),
            (41.88, -87.63),
            (43.65, -79.38),
            (32.78, -96.80),
            (37.77, -122.42),
            (19.43, -99.13),
            (25.76, -80.19),
            (47.61, -122.33),
            (38.91, -77.04),
        ],
    ),
    (
        "africa",
        [
            (6.52, 3.38),
            (-26.20, 28.05),
            (-1.29, 36.82),
            (9.03, 38.74),
        ],
    ),
    (
        "south_america",
        [
            (-23.55, -46.63),
            (-34.60, -58.38),
            (4.71, -74.07),
            (-12.05, -77.04),
        ],
    ),
]


GROUP_LABELS = {
    "china": "China",
    "europe": "Europe",
    "north_america": "North America",
    "africa": "Africa",
    "south_america": "South America",
}


@dataclass(frozen=True)
class StationSpec:
    station_id: int
    xml_station_id: int
    group: str
    group_index: int
    lat: float
    lon: float


def build_station_specs() -> list[StationSpec]:
    specs: list[StationSpec] = []
    station_id = 1
    for group, points in GROUP_POINTS:
        for group_index, (lat, lon) in enumerate(points, start=1):
            specs.append(
                StationSpec(
                    station_id=station_id,
                    xml_station_id=station_id - 1,
                    group=group,
                    group_index=group_index,
                    lat=float(lat),
                    lon=float(lon),
                )
            )
            station_id += 1
    return specs


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_voc_matrix(
    xml_file: Path,
    station_specs: list[StationSpec],
    max_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    target_ids = {spec.xml_station_id for spec in station_specs}
    station_count = len(station_specs)
    steps: list[int] = []
    rows: list[list[int]] = []
    seen_station_ids: set[int] = set()

    for _, elem in ET.iterparse(xml_file, events=("end",)):
        if local_name(elem.tag) != "time":
            continue

        step_text = elem.attrib.get("step", str(len(steps)))
        step = int(float(step_text))
        counts = [0] * station_count

        stations_elem = None
        for child in elem:
            if local_name(child.tag) == "stations":
                stations_elem = child
                break

        if stations_elem is not None:
            for station_elem in stations_elem:
                if local_name(station_elem.tag) != "station":
                    continue
                xml_station_id = int(station_elem.attrib["id"])
                seen_station_ids.add(xml_station_id)
                if xml_station_id not in target_ids:
                    continue
                counts[xml_station_id] = sum(
                    1 for child in station_elem if local_name(child.tag) == "satellite"
                )

        steps.append(step)
        rows.append(counts)
        elem.clear()

        if max_steps is not None and len(steps) >= max_steps:
            break

    matrix = np.asarray(rows, dtype=np.int16)
    step_array = np.asarray(steps, dtype=np.int64)
    expected_ids = set(range(station_count))
    meta = {
        "xml_file": str(xml_file),
        "time_steps": int(matrix.shape[0]),
        "station_count": int(station_count),
        "min_time_step": int(step_array.min()) if len(step_array) else None,
        "max_time_step": int(step_array.max()) if len(step_array) else None,
        "missing_xml_station_ids": sorted(expected_ids - seen_station_ids),
        "extra_xml_station_ids": sorted(seen_station_ids - expected_ids),
        "max_steps_limit": max_steps,
    }
    return step_array, matrix, meta


def write_station_cache(cache_dir: Path, station_specs: list[StationSpec]) -> None:
    rows = [asdict(spec) for spec in station_specs]
    json_path = cache_dir / "station_groups.json"
    csv_path = cache_dir / "station_groups.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["station_id", "xml_station_id", "group", "group_index", "lat", "lon"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_matrix_csv(cache_dir: Path, steps: np.ndarray, matrix: np.ndarray) -> None:
    path = cache_dir / "g60_voc_matrix.csv"
    header = ["time_step"] + [f"station_{i:02d}" for i in range(1, matrix.shape[1] + 1)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step, row in zip(steps, matrix):
            writer.writerow([int(step), *[int(value) for value in row]])


def write_group_summary_csv(
    cache_dir: Path,
    steps: np.ndarray,
    matrix: np.ndarray,
    station_specs: list[StationSpec],
) -> None:
    path = cache_dir / "g60_voc_group_summary.csv"
    group_to_cols: dict[str, list[int]] = {}
    for col, spec in enumerate(station_specs):
        group_to_cols.setdefault(spec.group, []).append(col)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_step", "group", "mean_voc", "min_voc", "max_voc"])
        for step_index, step in enumerate(steps):
            for group, cols in group_to_cols.items():
                values = matrix[step_index, cols]
                writer.writerow(
                    [
                        int(step),
                        group,
                        float(values.mean()),
                        int(values.min()),
                        int(values.max()),
                    ]
                )


def save_cache(
    cache_dir: Path,
    steps: np.ndarray,
    matrix: np.ndarray,
    station_specs: list[StationSpec],
    meta: dict[str, object],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_station_cache(cache_dir, station_specs)
    write_matrix_csv(cache_dir, steps, matrix)
    write_group_summary_csv(cache_dir, steps, matrix, station_specs)
    np.savez_compressed(cache_dir / "g60_voc_matrix.npz", steps=steps, voc=matrix)
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_cached_matrix(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz_path = cache_dir / "g60_voc_matrix.npz"
    data = np.load(npz_path)
    return data["steps"], data["voc"]


def decimate_series(
    steps: np.ndarray,
    values: np.ndarray,
    max_points: int = 2400,
) -> tuple[np.ndarray, np.ndarray]:
    if len(steps) <= max_points:
        return steps, values
    bin_size = int(np.ceil(len(steps) / max_points))
    usable = (len(steps) // bin_size) * bin_size
    step_bins = steps[:usable].reshape(-1, bin_size)
    value_bins = values[:usable].reshape(-1, bin_size)
    return step_bins[:, 0], value_bins.mean(axis=1)


def group_boundaries(station_specs: list[StationSpec]) -> list[tuple[str, int, int]]:
    boundaries: list[tuple[str, int, int]] = []
    start = 0
    current_group = station_specs[0].group
    for idx, spec in enumerate(station_specs):
        if spec.group != current_group:
            boundaries.append((current_group, start, idx))
            start = idx
            current_group = spec.group
    boundaries.append((current_group, start, len(station_specs)))
    return boundaries


def plot_voc_heatmap(
    figure_dir: Path,
    steps: np.ndarray,
    matrix: np.ndarray,
    station_specs: list[StationSpec],
) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "g60_voc_station_time_heatmap.png"

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    extent = [int(steps[0]), int(steps[-1]), 0.5, matrix.shape[1] + 0.5]
    im = ax.imshow(
        matrix.T,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=extent,
        cmap="viridis",
    )

    ax.set_title("G60 station VOC over time")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Station ID")
    ax.set_yticks(np.arange(1, matrix.shape[1] + 1, 2))

    for group, start, end in group_boundaries(station_specs):
        if start > 0:
            ax.axhline(start + 0.5, color="white", linewidth=0.8, alpha=0.85)
        midpoint = (start + end + 1) / 2
        ax.text(
            steps[0],
            midpoint,
            f" {GROUP_LABELS.get(group, group)}",
            va="center",
            ha="left",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("VOC (# visible satellites)")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_group_mean_timeseries(
    figure_dir: Path,
    steps: np.ndarray,
    matrix: np.ndarray,
    station_specs: list[StationSpec],
) -> Path:
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "g60_voc_group_mean_timeseries.png"

    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    for group, start, end in group_boundaries(station_specs):
        values = matrix[:, start:end].mean(axis=1)
        plot_steps, plot_values = decimate_series(steps, values)
        ax.plot(plot_steps, plot_values, linewidth=1.4, label=GROUP_LABELS.get(group, group))

    ax.set_title("G60 mean station VOC by region")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean VOC (# visible satellites)")
    ax.grid(True, linewidth=0.45, alpha=0.35)
    ax.legend(ncol=5, fontsize=9, frameon=False)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def read_or_build(
    xml_file: Path,
    cache_dir: Path,
    station_specs: list[StationSpec],
    force: bool,
    max_steps: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    npz_path = cache_dir / "g60_voc_matrix.npz"
    metadata_path = cache_dir / "metadata.json"
    if npz_path.exists() and metadata_path.exists() and not force and max_steps is None:
        steps, matrix = load_cached_matrix(cache_dir)
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["loaded_from_cache"] = True
        return steps, matrix, meta

    steps, matrix, meta = parse_voc_matrix(xml_file, station_specs, max_steps=max_steps)
    meta["loaded_from_cache"] = False
    save_cache(cache_dir, steps, matrix, station_specs, meta)
    return steps, matrix, meta


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build G60 station VOC time-series cache and plots for paper2."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="G60 visibility XML file.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Cache output dir.")
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Figure output dir.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild cache even if it exists.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional quick-test limit. Omit it for the full XML.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    station_specs = build_station_specs()

    steps, matrix, meta = read_or_build(
        xml_file=args.xml,
        cache_dir=args.cache_dir,
        station_specs=station_specs,
        force=args.force,
        max_steps=args.max_steps,
    )
    heatmap_path = plot_voc_heatmap(args.figure_dir, steps, matrix, station_specs)
    group_path = plot_group_mean_timeseries(args.figure_dir, steps, matrix, station_specs)

    print(f"XML: {args.xml}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Figure dir: {args.figure_dir}")
    print(f"Loaded from cache: {meta.get('loaded_from_cache')}")
    print(f"Time steps: {matrix.shape[0]}")
    print(f"Stations: {matrix.shape[1]}")
    print(f"VOC min/mean/max: {matrix.min()} / {matrix.mean():.3f} / {matrix.max()}")
    print(f"Missing XML station ids: {meta.get('missing_xml_station_ids')}")
    print(f"Extra XML station ids: {meta.get('extra_xml_station_ids')}")
    print(f"Heatmap: {heatmap_path}")
    print(f"Group mean plot: {group_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
