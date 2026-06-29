from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


GENERIC_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from build_g60_voc_timeseries import build_station_specs
from g60_paper2_config import DATA_ROOT, build_paper2_gw_config
from plot_g60_china_europe_shortest_metrics import build_topology_edge_table


DEFAULT_RAA_DIR = DATA_ROOT / "raan_001"
DEFAULT_G60_GROUND_LINK_60S = (
    DATA_ROOT
    / "outputs"
    / "paper2_ground_link_metrics"
    / "ce_hop_opt_shift60_all_pairs_baseline_compare"
    / "timeseries"
    / "G60"
)
DEFAULT_OUT_DIR = DATA_ROOT / "outputs" / "paper2_raan_sweep" / "raan_001_hops_stride60"

PAIRS = [
    ("china_europe", "China-Europe", "china", "europe"),
    ("china_north_america", "China-North America", "china", "north_america"),
    ("china_africa", "China-Africa", "china", "africa"),
    ("europe_north_america", "Europe-North America", "europe", "north_america"),
    ("europe_africa", "Europe-Africa", "europe", "africa"),
]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_hop_distance(total_nodes: int) -> np.ndarray:
    config = build_paper2_gw_config()
    edge_table = build_topology_edge_table(config, "plus_grid")
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.size, dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))
    dist = shortest_path(graph, directed=False, unweighted=True)
    if not np.all(np.isfinite(dist)):
        raise RuntimeError("GW +grid graph is disconnected.")
    return np.asarray(dist, dtype=np.float32)


def load_visibility_intervals(mat_path: Path) -> tuple[np.ndarray, dict]:
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    intervals = np.asarray(mat["intervals"], dtype=np.int64)
    meta = mat.get("meta")
    meta_out = {}
    if meta is not None and hasattr(meta, "_fieldnames"):
        for field in meta._fieldnames:
            value = getattr(meta, field)
            if isinstance(value, np.ndarray):
                meta_out[field] = value.tolist()
            elif hasattr(value, "item"):
                try:
                    meta_out[field] = value.item()
                except Exception:
                    meta_out[field] = str(value)
            else:
                meta_out[field] = value if isinstance(value, (str, int, float)) else str(value)
    return intervals, meta_out


def build_visible_sets_from_intervals(
    *,
    intervals: np.ndarray,
    num_stations: int,
    total_nodes: int,
    start: int,
    end: int,
    stride: int,
) -> tuple[np.ndarray, list[list[list[int]]]]:
    steps = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    step_to_row = {int(step): idx for idx, step in enumerate(steps)}
    visible: list[list[list[int]]] = [[[] for _ in range(int(num_stations))] for _ in range(int(steps.size))]

    for station_id, sat_id, t_start, t_stop in intervals:
        station_id = int(station_id)
        sat_id = int(sat_id)
        if station_id < 0 or station_id >= int(num_stations):
            continue
        if sat_id < 0 or sat_id >= int(total_nodes):
            continue
        first = max(int(start), int(np.ceil(int(t_start) / int(stride)) * int(stride)))
        last = min(int(end), int(np.floor(int(t_stop) / int(stride)) * int(stride)))
        if first > last:
            continue
        for step in range(first, last + 1, int(stride)):
            visible[step_to_row[int(step)]][station_id].append(sat_id)

    for row in visible:
        for station_id, sats in enumerate(row):
            if len(sats) > 1:
                row[station_id] = sorted(set(int(x) for x in sats))
    return steps, visible


def station_rows_by_group() -> dict[str, np.ndarray]:
    rows: dict[str, list[int]] = {}
    specs = build_station_specs()
    for spec in specs:
        rows.setdefault(str(spec.group), []).append(int(spec.xml_station_id))
    return {key: np.asarray(value, dtype=np.int32) for key, value in rows.items()}


def station_pair_satellite_hops(
    *,
    hop_dist: np.ndarray,
    visible_step: list[list[int]],
    source_rows: np.ndarray,
    target_rows: np.ndarray,
) -> float:
    values: list[float] = []
    for source_station in source_rows:
        source_nodes = np.asarray(visible_step[int(source_station)], dtype=np.int32)
        if source_nodes.size == 0:
            continue
        for target_station in target_rows:
            target_nodes = np.asarray(visible_step[int(target_station)], dtype=np.int32)
            if target_nodes.size == 0:
                continue
            values.append(float(np.nanmin(hop_dist[np.ix_(source_nodes, target_nodes)])))
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def compute_gw_raan_hops(
    *,
    intervals: np.ndarray,
    start: int,
    end: int,
    stride: int,
    out_dir: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    config = build_paper2_gw_config()
    group_rows = station_rows_by_group()
    steps, visible = build_visible_sets_from_intervals(
        intervals=intervals,
        num_stations=len(build_station_specs()),
        total_nodes=config.total_sats,
        start=start,
        end=end,
        stride=stride,
    )
    hop_dist = build_hop_distance(config.total_sats)
    series: dict[str, np.ndarray] = {}
    all_rows: list[dict] = []
    for pair_key, pair_label, source_group, target_group in PAIRS:
        values = np.empty(steps.size, dtype=np.float32)
        for idx, _step in enumerate(steps):
            values[idx] = station_pair_satellite_hops(
                hop_dist=hop_dist,
                visible_step=visible[idx],
                source_rows=group_rows[source_group],
                target_rows=group_rows[target_group],
            )
        series[pair_key] = values
        rows = [
            {
                "pair_key": pair_key,
                "pair_label": pair_label,
                "step": int(step),
                "hour": float(int(step) / 3600.0),
                "gw_raan001_mean_satellite_hops": float(values[idx]),
            }
            for idx, step in enumerate(steps)
        ]
        write_csv(out_dir / "pairs" / pair_key / "gw_raan001_hops_stride60.csv", rows)
        all_rows.extend(rows)
    write_csv(out_dir / "all_pairs_gw_raan001_hops_stride60.csv", all_rows)
    return steps, series


def load_g60_station_pair_satellite_hops(
    *,
    base_dir: Path,
    steps: np.ndarray,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    wanted_steps = {int(step): idx for idx, step in enumerate(steps)}
    for pair_key, _pair_label, _source, _target in PAIRS:
        path = base_dir / f"{pair_key}_ground_link_stride60.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        values = np.full(steps.size, np.nan, dtype=np.float32)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(float(row["step"]))
                idx = wanted_steps.get(step)
                if idx is None:
                    continue
                # Existing ground-link hops include two ground-satellite access hops.
                values[idx] = float(row["mean_shortest_hops_with_ground"]) - 2.0
        out[pair_key] = values
    return out


def summarize_and_plot(
    *,
    steps: np.ndarray,
    g60: dict[str, np.ndarray],
    gw: dict[str, np.ndarray],
    out_dir: Path,
    raan_label: str,
) -> None:
    rows: list[dict] = []
    for pair_key, pair_label, _source, _target in PAIRS:
        g60_values = g60[pair_key]
        gw_values = gw[pair_key]
        rows.append(
            {
                "pair_key": pair_key,
                "pair_label": pair_label,
                "time_stride_seconds": int(steps[1] - steps[0]) if steps.size > 1 else 0,
                "samples": int(steps.size),
                "g60_mean_satellite_hops": float(np.nanmean(g60_values)),
                "gw_raan001_mean_satellite_hops": float(np.nanmean(gw_values)),
                "gw_minus_g60_mean_satellite_hops": float(np.nanmean(gw_values) - np.nanmean(g60_values)),
            }
        )
    write_csv(out_dir / "g60_vs_gw_raan001_hops_stride60_summary.csv", rows)

    labels = [row["pair_label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.34
    g60_means = np.asarray([row["g60_mean_satellite_hops"] for row in rows], dtype=np.float64)
    gw_means = np.asarray([row["gw_raan001_mean_satellite_hops"] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(13.8, 6.3), dpi=180)
    b1 = ax.bar(x - width / 2, g60_means, width, color="#1E88E5", label="G60")
    b2 = ax.bar(x + width / 2, gw_means, width, color="#D81B60", label=raan_label)
    ax.set_ylabel("mean shortest satellite hops")
    ax.set_title(f"G60 vs {raan_label} station-pair satellite-hop metric, 60s sampled")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.grid(True, axis="y", alpha=0.24, linestyle="--")
    ax.legend()
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "g60_vs_gw_raan001_mean_hops_bar.png")
    plt.close(fig)

    hours = steps.astype(np.float32) / 3600.0
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(16.2, 13.0), dpi=170, sharex=True)
    for ax, (pair_key, pair_label, _source, _target) in zip(axes, PAIRS):
        ax.plot(hours, g60[pair_key], color="#1E88E5", linewidth=0.85, label="G60")
        ax.plot(hours, gw[pair_key], color="#D81B60", linewidth=0.85, label=raan_label)
        ax.set_ylabel(pair_label)
        ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    axes[0].set_title(f"G60 vs {raan_label}: mean shortest satellite hops, 60s sampled")
    axes[-1].set_xlabel("time (hour)")
    axes[0].legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "g60_vs_gw_raan001_hops_timeseries.png")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute 60s satellite-hop metrics for GW raan_001 interval data.")
    parser.add_argument("--raan-dir", type=Path, default=DEFAULT_RAA_DIR)
    parser.add_argument("--g60-ground-link-60s-dir", type=Path, default=DEFAULT_G60_GROUND_LINK_60S)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(parents=True, exist_ok=True)

    intervals, interval_meta = load_visibility_intervals(args.raan_dir / "visibility_intervals.mat")
    steps, gw = compute_gw_raan_hops(
        intervals=intervals,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        out_dir=args.out_dir,
    )
    g60 = load_g60_station_pair_satellite_hops(base_dir=args.g60_ground_link_60s_dir, steps=steps)
    summarize_and_plot(steps=steps, g60=g60, gw=gw, out_dir=args.out_dir, raan_label="GW raan_001")
    (args.out_dir / "method.json").write_text(
        json.dumps(
            {
                "definition": (
                    "For each ground-station pair, choose the visible source/target satellites that minimize "
                    "satellite-network hops. Ground-satellite hops are not counted."
                ),
                "time_stride_seconds": int(args.stride),
                "start": int(args.start),
                "end": int(args.end),
                "gw_raan_dir": str(args.raan_dir),
                "visibility_interval_meta": interval_meta,
                "g60_source_dir": str(args.g60_ground_link_60s_dir),
                "topology": "+grid",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.out_dir / "g60_vs_gw_raan001_hops_stride60_summary.csv")
    print(args.out_dir / "figures" / "g60_vs_gw_raan001_hops_timeseries.png")
    print(args.out_dir / "figures" / "g60_vs_gw_raan001_mean_hops_bar.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
