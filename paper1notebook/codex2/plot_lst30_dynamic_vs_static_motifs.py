from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_m56_local_patch_hybrid_topology import DEFAULT_CONFIG, normalize_motif_name, path_from, region_pairs_from_workflow  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import load_workflow_yaml, viewer_config_from_workflow  # noqa: E402


DEFAULT_STATS_DIR = Path(r"E:\paper11\data\satnet_experiments\runs\paper1\G60\lst30_dp_c_pen001_stats")
DEFAULT_DELAY_STORE = Path(r"E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LST=30 dynamic topology against static motifs 000056 and 000040.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
    parser.add_argument("--delay-store", type=Path, default=DEFAULT_DELAY_STORE)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--base-motif", default="56")
    parser.add_argument("--patch-motif", default="40")
    return parser.parse_args()


def read_dynamic_hops(path: Path) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            out[row["pair"]][int(row["step"])] = float(row["mean_hops"])
    return out


def read_dynamic_delay(path: Path) -> tuple[dict[str, dict[int, float]], dict[int, str]]:
    out: dict[str, dict[int, float]] = defaultdict(dict)
    reasons: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            step = int(row["step"])
            out[row["pair"]][step] = float(row["mean_delay_ms"])
            reasons[step] = row.get("sample_reason", "")
    return out, reasons


def mean_between_groups(dist: np.ndarray, sources: tuple[int, ...], targets: tuple[int, ...]) -> float:
    if not sources or not targets:
        return float("nan")
    block = dist[np.ix_(np.asarray(sources, dtype=np.int32), np.asarray(targets, dtype=np.int32))]
    finite = np.isfinite(block) & (block > 0)
    if not np.any(finite):
        return float("nan")
    return float(np.mean(block[finite]))


def edge_key_arrays(edge_table: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(edge_table.src, dtype=np.int32), np.asarray(edge_table.dst, dtype=np.int32)


def build_weighted_graph(*, total_nodes: int, src: np.ndarray, dst: np.ndarray, weights: np.ndarray) -> csr_matrix:
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.concatenate([weights, weights]).astype(np.float32, copy=False)
    return csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))


def compute_static_hops(
    *,
    steps: list[int],
    specs_by_label: dict[str, Any],
    config: Any,
    group_data: dict,
    pairs: dict[str, Any],
) -> dict[str, dict[str, dict[int, float]]]:
    out: dict[str, dict[str, dict[int, float]]] = {}
    for label, spec in specs_by_label.items():
        src, dst = edge_key_arrays(spec.edge_table)
        graph = csr_matrix(
            (
                np.ones(src.size * 2, dtype=np.float32),
                (np.concatenate([src, dst]), np.concatenate([dst, src])),
            ),
            shape=(int(config.total_sats), int(config.total_sats)),
        )
        dist = np.asarray(shortest_path(graph, directed=False, unweighted=True), dtype=np.float32)
        out[label] = defaultdict(dict)
        for step in steps:
            for pair_key, pair in pairs.items():
                sources = group_nodes_for_step(group_data, int(step), int(pair.source_group_id))
                targets = group_nodes_for_step(group_data, int(step), int(pair.target_group_id))
                out[label][pair_key][int(step)] = mean_between_groups(dist, sources, targets)
    return out


def compute_static_delay_samples(
    *,
    sample_steps: list[int],
    specs_by_label: dict[str, Any],
    config: Any,
    group_data: dict,
    pairs: dict[str, Any],
    delay_store: Path,
) -> dict[str, dict[str, dict[int, float]]]:
    delay_ms = np.load(delay_store / "edge_delay_ms.npy", mmap_mode="r")
    edge_index = np.load(delay_store / "edge_index_matrix.npy", mmap_mode="r")
    out: dict[str, dict[str, dict[int, float]]] = {}

    for label, spec in specs_by_label.items():
        src, dst = edge_key_arrays(spec.edge_table)
        delay_cols = np.asarray(edge_index[src, dst], dtype=np.int32)
        if np.any(delay_cols < 0):
            bad = int(np.flatnonzero(delay_cols < 0)[0])
            raise ValueError(f"{label} edge is missing from delay store: ({src[bad]}, {dst[bad]})")

        out[label] = defaultdict(dict)
        for pos, step in enumerate(sample_steps, 1):
            weights = np.asarray(delay_ms[int(step), delay_cols], dtype=np.float32)
            graph = build_weighted_graph(total_nodes=int(config.total_sats), src=src, dst=dst, weights=weights)
            china_sources = group_nodes_for_step(group_data, int(step), int(pairs["china_america"].source_group_id))
            if china_sources:
                dist_from_china = np.asarray(
                    shortest_path(
                        graph,
                        directed=False,
                        unweighted=False,
                        indices=np.asarray(china_sources, dtype=np.int32),
                    ),
                    dtype=np.float64,
                )
            else:
                dist_from_china = np.empty((0, int(config.total_sats)), dtype=np.float64)

            for pair_key, pair in pairs.items():
                sources = group_nodes_for_step(group_data, int(step), int(pair.source_group_id))
                targets = group_nodes_for_step(group_data, int(step), int(pair.target_group_id))
                if tuple(sources) != tuple(china_sources):
                    dist = np.asarray(shortest_path(graph, directed=False, unweighted=False), dtype=np.float64)
                    out[label][pair_key][int(step)] = mean_between_groups(dist, sources, targets)
                    continue
                if not targets or dist_from_china.size == 0:
                    out[label][pair_key][int(step)] = float("nan")
                    continue
                block = dist_from_china[:, np.asarray(targets, dtype=np.int32)]
                finite = np.isfinite(block) & (block > 0)
                out[label][pair_key][int(step)] = float(np.mean(block[finite])) if np.any(finite) else float("nan")

            if pos % 500 == 0 or pos == len(sample_steps):
                print(f"[compare-static] {label} delay samples {pos}/{len(sample_steps)} step={step}", flush=True)
    return out


def write_hops_csv(path: Path, *, steps: list[int], pairs: list[str], dynamic: dict, static: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "pair", "dynamic_lst30", "motif_000056", "motif_000040"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in steps:
            for pair in pairs:
                writer.writerow(
                    {
                        "step": int(step),
                        "pair": pair,
                        "dynamic_lst30": dynamic[pair][int(step)],
                        "motif_000056": static["motif_000056"][pair][int(step)],
                        "motif_000040": static["motif_000040"][pair][int(step)],
                    }
                )


def write_delay_csv(path: Path, *, sample_steps: list[int], reasons: dict[int, str], pairs: list[str], dynamic: dict, static: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "sample_reason", "pair", "dynamic_lst30", "motif_000056", "motif_000040"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in sample_steps:
            for pair in pairs:
                writer.writerow(
                    {
                        "step": int(step),
                        "sample_reason": reasons.get(int(step), ""),
                        "pair": pair,
                        "dynamic_lst30": dynamic[pair][int(step)],
                        "motif_000056": static["motif_000056"][pair][int(step)],
                        "motif_000040": static["motif_000040"][pair][int(step)],
                    }
                )


def plot_comparison(*, out_dir: Path, hops_csv: Path, delay_csv: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = ["china_america", "china_europe", "china_africa"]
    pair_labels = {
        "china_america": "China-America",
        "china_europe": "China-Europe",
        "china_africa": "China-Africa",
    }
    series = [
        ("dynamic_lst30", "dynamic LST=30", "#111827", "-"),
        ("motif_000056", "motif 000056", "#2563eb", "--"),
        ("motif_000040", "motif 000040", "#dc2626", ":"),
    ]

    hops_rows: dict[str, dict[str, list[float]]] = {
        pair: defaultdict(list)
        for pair in pairs
    }
    with hops_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pair = row["pair"]
            hops_rows[pair]["step"].append(int(row["step"]))
            for key, *_ in series:
                hops_rows[pair][key].append(float(row[key]))

    delay_rows: dict[str, dict[str, list[float]]] = {
        pair: defaultdict(list)
        for pair in pairs
    }
    with delay_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            pair = row["pair"]
            delay_rows[pair]["step"].append(int(row["step"]))
            for key, *_ in series:
                delay_rows[pair][key].append(float(row[key]))

    fig, axes = plt.subplots(3, 2, figsize=(16.5, 9.5), dpi=170, sharex="col")
    for row_idx, pair in enumerate(pairs):
        hx = np.asarray(hops_rows[pair]["step"], dtype=np.float64) / 3600.0
        dx = np.asarray(delay_rows[pair]["step"], dtype=np.float64) / 3600.0
        for key, label, color, linestyle in series:
            hy = np.asarray(hops_rows[pair][key], dtype=np.float64)
            dy = np.asarray(delay_rows[pair][key], dtype=np.float64)
            axes[row_idx, 0].plot(hx, hy, color=color, linestyle=linestyle, linewidth=1.05, label=f"{label} mean={np.nanmean(hy):.3f}")
            axes[row_idx, 1].plot(dx, dy, color=color, linestyle=linestyle, linewidth=1.0, label=f"{label} mean={np.nanmean(dy):.3f} ms")
        axes[row_idx, 0].set_ylabel(pair_labels[pair])
        axes[row_idx, 0].grid(alpha=0.25, linestyle="--", linewidth=0.55)
        axes[row_idx, 1].grid(alpha=0.25, linestyle="--", linewidth=0.55)
        axes[row_idx, 0].legend(fontsize=7, loc="upper right")
        axes[row_idx, 1].legend(fontsize=7, loc="upper right")
    axes[0, 0].set_title("mean shortest hops")
    axes[0, 1].set_title("mean shortest delay (ms, sampled)")
    axes[-1, 0].set_xlabel("time (hour)")
    axes[-1, 1].set_xlabel("time (hour)")
    fig.suptitle("Dynamic LST=30 vs static motif 000056 / 000040", y=0.995)
    fig.tight_layout()
    fig.savefig(out_dir / "lst30_dynamic_vs_static_000056_000040_panels.png")
    plt.close(fig)

    for metric_name, rows_by_pair, y_label, title, out_name in (
        ("hops", hops_rows, "mean shortest hops", "Mean shortest hops", "lst30_dynamic_vs_static_hops.png"),
        ("delay", delay_rows, "mean shortest delay (ms)", "Mean shortest delay (sampled)", "lst30_dynamic_vs_static_delay_ms.png"),
    ):
        fig, axes = plt.subplots(3, 1, figsize=(15.5, 9), dpi=170, sharex=True)
        for row_idx, pair in enumerate(pairs):
            x = np.asarray(rows_by_pair[pair]["step"], dtype=np.float64) / 3600.0
            for key, label, color, linestyle in series:
                y = np.asarray(rows_by_pair[pair][key], dtype=np.float64)
                axes[row_idx].plot(x, y, color=color, linestyle=linestyle, linewidth=1.05, label=f"{label} mean={np.nanmean(y):.3f}")
            axes[row_idx].set_ylabel(pair_labels[pair])
            axes[row_idx].grid(alpha=0.25, linestyle="--", linewidth=0.55)
            axes[row_idx].legend(fontsize=8, loc="upper right")
        axes[0].set_title(title)
        axes[-1].set_xlabel("time (hour)")
        fig.supylabel(y_label, x=0.01)
        fig.tight_layout()
        fig.savefig(out_dir / out_name)
        plt.close(fig)


def write_summary(path: Path, *, hops_csv: Path, delay_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    for metric, csv_path, fields in (
        ("hops", hops_csv, ("dynamic_lst30", "motif_000056", "motif_000040")),
        ("delay_ms_sampled", delay_csv, ("dynamic_lst30", "motif_000056", "motif_000040")),
    ):
        by_pair: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for field in fields:
                    by_pair[row["pair"]][field].append(float(row[field]))
        for pair, values_by_field in by_pair.items():
            row: dict[str, Any] = {"metric": metric, "pair": pair}
            for field in fields:
                arr = np.asarray(values_by_field[field], dtype=np.float64)
                row[f"{field}_mean"] = float(np.nanmean(arr))
                row[f"{field}_min"] = float(np.nanmin(arr))
                row[f"{field}_max"] = float(np.nanmax(arr))
                row[f"{field}_p95"] = float(np.nanpercentile(arr, 95))
            row["dynamic_minus_motif_000056_mean"] = row["dynamic_lst30_mean"] - row["motif_000056_mean"]
            row["dynamic_minus_motif_000040_mean"] = row["dynamic_lst30_mean"] - row["motif_000040_mean"]
            rows.append(row)

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.stats_dir) / "compare_static_motifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    dynamic_hops = read_dynamic_hops(Path(args.stats_dir) / "lst_active_hops_timeseries.csv")
    dynamic_delay, sample_reasons = read_dynamic_delay(Path(args.stats_dir) / "lst_active_delay_sample_timeseries.csv")
    steps = sorted(next(iter(dynamic_hops.values())).keys())
    sample_steps = sorted(sample_reasons.keys())

    paths_raw = workflow.get("paths", {})
    motif_raw = workflow.get("motif_library", {})
    name_prefix = str(motif_raw.get("name_prefix", "combined"))
    motif_csv = path_from(paths_raw, "motif_library_dir") / str(motif_raw.get("csv_name"))
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=name_prefix,
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=False,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    base_name = normalize_motif_name(args.base_motif, name_prefix=name_prefix)
    patch_name = normalize_motif_name(args.patch_motif, name_prefix=name_prefix)
    static_specs = {
        "motif_000056": specs_by_name[base_name],
        "motif_000040": specs_by_name[patch_name],
    }

    pair_by_key = region_pairs_from_workflow(workflow)
    pair_keys = ["china_america", "china_europe", "china_africa"]
    pairs = {key: pair_by_key[key] for key in pair_keys}
    group_data = load_or_build_group_data(
        xml_file=path_from(paths_raw, "group_xml"),
        group_cache_dir=path_from(paths_raw, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=1,
        enabled=True,
        force=False,
    )

    print("[compare-static] computing static hops", flush=True)
    static_hops = compute_static_hops(
        steps=steps,
        specs_by_label=static_specs,
        config=config,
        group_data=group_data,
        pairs=pairs,
    )
    print("[compare-static] computing static sampled delays", flush=True)
    static_delay = compute_static_delay_samples(
        sample_steps=sample_steps,
        specs_by_label=static_specs,
        config=config,
        group_data=group_data,
        pairs=pairs,
        delay_store=Path(args.delay_store),
    )

    hops_csv = out_dir / "lst30_dynamic_vs_static_hops_timeseries.csv"
    delay_csv = out_dir / "lst30_dynamic_vs_static_delay_sample_timeseries.csv"
    write_hops_csv(hops_csv, steps=steps, pairs=pair_keys, dynamic=dynamic_hops, static=static_hops)
    write_delay_csv(delay_csv, sample_steps=sample_steps, reasons=sample_reasons, pairs=pair_keys, dynamic=dynamic_delay, static=static_delay)
    plot_comparison(out_dir=out_dir, hops_csv=hops_csv, delay_csv=delay_csv)
    write_summary(out_dir / "lst30_dynamic_vs_static_summary.csv", hops_csv=hops_csv, delay_csv=delay_csv)

    for path in (
        out_dir / "lst30_dynamic_vs_static_000056_000040_panels.png",
        out_dir / "lst30_dynamic_vs_static_hops.png",
        out_dir / "lst30_dynamic_vs_static_delay_ms.png",
        out_dir / "lst30_dynamic_vs_static_summary.csv",
        hops_csv,
        delay_csv,
    ):
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
