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
DEFAULT_G60_REGION_HOP_CSV = (
    DATA_ROOT
    / "outputs"
    / "paper2_static_hop_metrics"
    / "G60"
    / "plus_grid"
    / "t0_86164_stride1"
    / "pairs"
    / "china_europe"
    / "timeseries.csv"
)
DEFAULT_OUT_DIR = DATA_ROOT / "outputs" / "paper2_raan_sweep" / "raan_001_region_avg_hops_stride60"


PAIR_KEY = "china_europe"
PAIR_LABEL = "China-Europe"
SOURCE_GROUP = "china"
TARGET_GROUP = "europe"


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
    meta_out: dict = {}
    if meta is not None and hasattr(meta, "_fieldnames"):
        for field in meta._fieldnames:
            value = getattr(meta, field)
            if isinstance(value, np.ndarray):
                meta_out[field] = value.tolist()
            elif isinstance(value, (str, int, float)):
                meta_out[field] = value
            else:
                meta_out[field] = str(value)
    return intervals, meta_out


def group_station_ids(group_name: str) -> np.ndarray:
    ids = [int(spec.xml_station_id) for spec in build_station_specs() if str(spec.group) == str(group_name)]
    if not ids:
        raise ValueError(f"No station ids for group={group_name!r}")
    return np.asarray(ids, dtype=np.int32)


def build_group_visible_sets_from_intervals(
    *,
    intervals: np.ndarray,
    station_ids: np.ndarray,
    total_nodes: int,
    start: int,
    end: int,
    stride: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    steps = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    step_to_row = {int(step): idx for idx, step in enumerate(steps)}
    station_set = set(int(x) for x in station_ids)
    visible_sets = [set() for _ in range(int(steps.size))]

    for station_id, sat_id, t_start, t_stop in intervals:
        station_id = int(station_id)
        if station_id not in station_set:
            continue
        sat_id = int(sat_id)
        if sat_id < 0 or sat_id >= int(total_nodes):
            continue

        first = max(int(start), int(np.ceil(int(t_start) / int(stride)) * int(stride)))
        last = min(int(end), int(np.floor(int(t_stop) / int(stride)) * int(stride)))
        if first > last:
            continue
        for step in range(first, last + 1, int(stride)):
            visible_sets[step_to_row[int(step)]].add(sat_id)

    visible_arrays = [np.asarray(sorted(values), dtype=np.int32) for values in visible_sets]
    return steps, visible_arrays


def region_average_hops(hop_dist: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("nan")
    values = hop_dist[np.ix_(left, right)]
    return float(np.mean(values, dtype=np.float64))


def compute_gw_raan001_region_avg_hops(
    *,
    intervals: np.ndarray,
    start: int,
    end: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = build_paper2_gw_config()
    china_ids = group_station_ids(SOURCE_GROUP)
    europe_ids = group_station_ids(TARGET_GROUP)
    steps, china_visible = build_group_visible_sets_from_intervals(
        intervals=intervals,
        station_ids=china_ids,
        total_nodes=config.total_sats,
        start=start,
        end=end,
        stride=stride,
    )
    steps2, europe_visible = build_group_visible_sets_from_intervals(
        intervals=intervals,
        station_ids=europe_ids,
        total_nodes=config.total_sats,
        start=start,
        end=end,
        stride=stride,
    )
    if not np.array_equal(steps, steps2):
        raise RuntimeError("source/target step axes do not match")

    hop_dist = build_hop_distance(config.total_sats)
    values = np.full(steps.size, np.nan, dtype=np.float32)
    source_counts = np.zeros(steps.size, dtype=np.int16)
    target_counts = np.zeros(steps.size, dtype=np.int16)
    for idx, (left, right) in enumerate(zip(china_visible, europe_visible)):
        source_counts[idx] = int(left.size)
        target_counts[idx] = int(right.size)
        values[idx] = region_average_hops(hop_dist, left, right)
    return steps, values, source_counts, target_counts


def load_g60_region_avg_hops(path: Path, steps: np.ndarray) -> np.ndarray:
    wanted = {int(step): idx for idx, step in enumerate(steps)}
    values = np.full(steps.size, np.nan, dtype=np.float32)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            step = int(float(row["step"]))
            idx = wanted.get(step)
            if idx is not None:
                values[idx] = float(row["mean_shortest_hops"])
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute region-average China-Europe hops for GW raan_001, 60s.")
    parser.add_argument("--raan-dir", type=Path, default=DEFAULT_RAA_DIR)
    parser.add_argument("--g60-region-hop-csv", type=Path, default=DEFAULT_G60_REGION_HOP_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    intervals, interval_meta = load_visibility_intervals(args.raan_dir / "visibility_intervals.mat")
    steps, gw_values, source_counts, target_counts = compute_gw_raan001_region_avg_hops(
        intervals=intervals,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    g60_values = load_g60_region_avg_hops(args.g60_region_hop_csv, steps)

    rows = []
    for idx, step in enumerate(steps):
        rows.append(
            {
                "pair_key": PAIR_KEY,
                "pair_label": PAIR_LABEL,
                "step": int(step),
                "hour": float(int(step) / 3600.0),
                "g60_region_avg_hops": float(g60_values[idx]),
                "gw_raan001_region_avg_hops": float(gw_values[idx]),
                "gw_source_region_visible_sats": int(source_counts[idx]),
                "gw_target_region_visible_sats": int(target_counts[idx]),
            }
        )
    write_csv(out_dir / "china_europe_region_avg_hops_stride60_timeseries.csv", rows)

    summary = [
        {
            "pair_key": PAIR_KEY,
            "pair_label": PAIR_LABEL,
            "time_stride_seconds": int(args.stride),
            "samples": int(steps.size),
            "g60_mean_region_avg_hops": float(np.nanmean(g60_values)),
            "gw_raan001_mean_region_avg_hops": float(np.nanmean(gw_values)),
            "gw_minus_g60_mean_region_avg_hops": float(np.nanmean(gw_values) - np.nanmean(g60_values)),
            "g60_min_region_avg_hops": float(np.nanmin(g60_values)),
            "gw_raan001_min_region_avg_hops": float(np.nanmin(gw_values)),
            "g60_max_region_avg_hops": float(np.nanmax(g60_values)),
            "gw_raan001_max_region_avg_hops": float(np.nanmax(gw_values)),
        }
    ]
    write_csv(out_dir / "china_europe_region_avg_hops_stride60_summary.csv", summary)

    hours = steps.astype(np.float32) / 3600.0
    fig, ax = plt.subplots(figsize=(15.4, 5.8), dpi=180)
    ax.plot(hours, g60_values, color="#1E88E5", linewidth=0.9, label=f"G60 mean={np.nanmean(g60_values):.3f}")
    ax.plot(hours, gw_values, color="#D81B60", linewidth=0.9, label=f"GW raan_001 mean={np.nanmean(gw_values):.3f}")
    ax.set_title("China-Europe region-average shortest hops, 60s sampled")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel("region-average shortest hops")
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_dir / "china_europe_region_avg_hops_timeseries.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=180)
    bars = ax.bar(
        ["G60", "GW raan_001"],
        [float(np.nanmean(g60_values)), float(np.nanmean(gw_values))],
        color=["#1E88E5", "#D81B60"],
        width=0.48,
    )
    ax.set_ylabel("mean region-average shortest hops")
    ax.set_title("China-Europe mean region-average hops")
    ax.grid(True, axis="y", alpha=0.24, linestyle="--")
    for rect in bars:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h, f"{h:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(fig_dir / "china_europe_region_avg_hops_mean_bar.png")
    plt.close(fig)

    (out_dir / "method.json").write_text(
        json.dumps(
            {
                "definition": (
                    "For each time t, form V_in^S(region,t) as the union of satellites visible "
                    "from all stations in the region. Then compute the average of h_t(m,n) over "
                    "all m in source-region visible set and n in target-region visible set."
                ),
                "formula": "J = 1/(|V_i||V_j|) sum_{m in V_i} sum_{n in V_j} h_t(m,n)",
                "pair": PAIR_KEY,
                "time_stride_seconds": int(args.stride),
                "start": int(args.start),
                "end": int(args.end),
                "topology": "+grid",
                "gw_raan_dir": str(args.raan_dir),
                "g60_region_hop_csv": str(args.g60_region_hop_csv),
                "visibility_interval_meta": interval_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out_dir / "china_europe_region_avg_hops_stride60_summary.csv")
    print(fig_dir / "china_europe_region_avg_hops_timeseries.png")
    print(fig_dir / "china_europe_region_avg_hops_mean_bar.png")
    print(
        f"G60 mean={float(np.nanmean(g60_values)):.6f}, "
        f"GW raan_001 mean={float(np.nanmean(gw_values)):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
