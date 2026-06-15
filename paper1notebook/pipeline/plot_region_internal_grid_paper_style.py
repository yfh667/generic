from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config
from run_paper1_region_internal_grid_metrics import (
    DEFAULT_CONFIG,
    _compute_topology_all_pairs_task,
    _init_worker_context,
)
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module import TopologySpec, build_full_option_plus_intra_edge_table
from src.topology_workflow.module.config import time_axis_from_config, viewer_config_from_workflow


DEFAULT_RUN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3\region_internal_grid_metrics_t0_86160_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-style plots for region-internal +grid motif metrics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--delay-store-dir", type=Path, default=None)
    parser.add_argument("--position-cache-dir", type=Path, default=None)
    parser.add_argument("--no-position-cache", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--force-baselines", action="store_true")
    parser.add_argument("--baseline-delay-engine", choices=("auto", "scipy", "heapq"), default="auto")
    parser.add_argument("--baseline-step-heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--pairs", nargs="*", default=None)
    return parser.parse_args()


def read_compare_csv(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    names = [str(x) for x in header[1:]]
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.float64, encoding="utf-8-sig")
    data = np.atleast_2d(data)
    return data[:, 0].astype(np.int64), names, data[:, 1:].astype(np.float64)


def read_complete_reachability_mask(pair_dir: str | Path, topology_names: list[str], steps: np.ndarray) -> np.ndarray:
    pair_dir = Path(pair_dir)
    step_to_idx = {int(step): idx for idx, step in enumerate(np.asarray(steps, dtype=np.int64).tolist())}
    mask = np.zeros((len(step_to_idx), len(topology_names)), dtype=bool)
    for col_idx, name in enumerate(topology_names):
        metrics_path = pair_dir / "topologies" / str(name) / "step_metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                step = int(row["step"])
                row_idx = step_to_idx.get(step)
                if row_idx is None:
                    continue
                reachable = int(float(row["reachable_pairs"]))
                expected = int(float(row["expected_pairs"]))
                mask[row_idx, col_idx] = expected > 0 and reachable == expected
    return mask


def write_dynamic_csv(
    path: str | Path,
    *,
    steps: np.ndarray,
    dynamic_names: np.ndarray,
    dynamic_values: np.ndarray,
    best_static_name: str,
    best_static_values: np.ndarray,
    full_link_values: np.ndarray,
    gridplus_values: np.ndarray,
    dynamic_complete: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "step",
            "dynamic_best_topology",
            "dynamic_best_delay_ms",
            "best_static_topology",
            "best_static_delay_ms",
            "full_link_delay_ms",
            "gridplus_delay_ms",
            "dynamic_complete_reachability",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            writer.writerow(
                {
                    "step": int(step),
                    "dynamic_best_topology": str(dynamic_names[idx]),
                    "dynamic_best_delay_ms": float(dynamic_values[idx]),
                    "best_static_topology": str(best_static_name),
                    "best_static_delay_ms": float(best_static_values[idx]),
                    "full_link_delay_ms": float(full_link_values[idx]),
                    "gridplus_delay_ms": float(gridplus_values[idx]),
                    "dynamic_complete_reachability": bool(dynamic_complete[idx]),
                }
            )


def finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def ensure_baseline_outputs(
    *,
    run_dir: Path,
    raw: dict[str, Any],
    config,
    pair_specs,
    force: bool,
    delay_store_dir: Path,
    position_cache_dir: Path | None,
    delay_engine: str,
    step_heartbeat_seconds: float,
) -> None:
    wrap_planes = wrap_planes_from_config(raw)
    meta_path = run_dir / "region_internal_grid_metrics_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    start = int(meta["start"])
    end = int(meta["end"])
    stride = int(meta["stride"])
    steps = list(range(start, end + 1, stride))
    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=True,
        force=False,
    )
    baseline_specs = [
        TopologySpec(
            name="full_link",
            edge_table=build_full_option_plus_intra_edge_table(
                config=config,
                options=(0, 1, 2, 4),
                add_intra_ring=True,
                wrap_planes=wrap_planes,
            ),
            library="baseline",
            motif="full_options_plus_intra_region_internal_grid",
            baseline=True,
        ),
        TopologySpec(
            name="gridplus",
            edge_table=build_full_option_plus_intra_edge_table(
                config=config,
                options=(0,),
                add_intra_ring=True,
                wrap_planes=wrap_planes,
            ),
            library="baseline",
            motif="option0_plus_intra",
            baseline=True,
        ),
    ]
    _init_worker_context(
        {
            "config": config,
            "group_data": group_data,
            "pair_specs": pair_specs,
            "steps": steps,
            "stride": stride,
            "delay_store_dir": str(delay_store_dir),
            "position_cache_dir": str(position_cache_dir) if position_cache_dir is not None else None,
            "out_dir": str(run_dir),
            "forced_option": 0,
            "force": bool(force),
            "wrap_planes": bool(wrap_planes),
            "compute_hops": False,
            "delay_engine": str(delay_engine),
            "step_heartbeat_seconds": float(step_heartbeat_seconds),
        }
    )
    for spec in baseline_specs:
        _compute_topology_all_pairs_task((spec, pair_specs))


def plot_pair_delay(
    *,
    pair_key: str,
    pair_label: str,
    pair_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, motif_names, matrix = read_compare_csv(pair_dir / "compare_mean_shortest_delay_ms.csv")
    complete_mask = read_complete_reachability_mask(pair_dir, motif_names, steps)
    valid_matrix = np.where(complete_mask, matrix, np.nan)

    all_step_complete = np.all(complete_mask, axis=0)
    if np.any(all_step_complete):
        means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
        means[all_step_complete] = np.nanmean(valid_matrix[:, all_step_complete], axis=0)
        best_static_pool = "all_steps_complete"
    else:
        means = np.full(matrix.shape[1], np.nan, dtype=np.float64)
        has_any_valid = np.any(np.isfinite(valid_matrix), axis=0)
        if np.any(has_any_valid):
            means[has_any_valid] = np.nanmean(valid_matrix[:, has_any_valid], axis=0)
            best_static_pool = "partial_complete_fallback"
        else:
            valid_matrix = np.asarray(matrix, dtype=np.float64)
            complete_mask = np.isfinite(valid_matrix)
            has_any_valid = np.any(np.isfinite(valid_matrix), axis=0)
            means[has_any_valid] = np.nanmean(valid_matrix[:, has_any_valid], axis=0)
            best_static_pool = "reachable_mean_fallback"
    if not np.any(np.isfinite(means)):
        raise ValueError(f"No finite motif delay values are available for {pair_key}")
    best_static_idx = int(np.nanargmin(means))
    best_static_name = motif_names[best_static_idx]
    best_static_values = valid_matrix[:, best_static_idx]

    finite_valid = np.isfinite(valid_matrix)
    has_valid = np.any(finite_valid, axis=1)
    dynamic_idx = np.full(matrix.shape[0], -1, dtype=np.int32)
    dynamic_values = np.full(matrix.shape[0], np.nan, dtype=np.float64)
    if np.any(has_valid):
        safe = np.where(finite_valid, valid_matrix, np.inf)
        dynamic_idx[has_valid] = np.argmin(safe[has_valid], axis=1)
        dynamic_values[has_valid] = safe[np.where(has_valid)[0], dynamic_idx[has_valid]]
    dynamic_names = np.asarray(
        [motif_names[int(idx)] if int(idx) >= 0 else "" for idx in dynamic_idx],
        dtype=object,
    )

    full_link_values = np.asarray(
        np.load(pair_dir / "topologies" / "full_link" / "mean_shortest_delay_ms.npy"),
        dtype=np.float64,
    )
    gridplus_values = np.asarray(
        np.load(pair_dir / "topologies" / "gridplus" / "mean_shortest_delay_ms.npy"),
        dtype=np.float64,
    )

    dynamic_csv = pair_dir / "paper_style_dynamic_best_mean_shortest_delay_ms.csv"
    write_dynamic_csv(
        dynamic_csv,
        steps=steps,
        dynamic_names=dynamic_names,
        dynamic_values=dynamic_values,
        best_static_name=best_static_name,
        best_static_values=best_static_values,
        full_link_values=full_link_values,
        gridplus_values=gridplus_values,
        dynamic_complete=has_valid,
    )

    violation_mask = np.isfinite(dynamic_values) & np.isfinite(full_link_values) & (dynamic_values < full_link_values - 1e-9)
    summary = {
        "pair_key": pair_key,
        "pair_label": pair_label,
        "num_steps": int(steps.size),
        "num_motifs": int(len(motif_names)),
        "num_all_steps_complete_motifs": int(np.count_nonzero(all_step_complete)),
        "num_incomplete_motifs": int(len(motif_names) - np.count_nonzero(all_step_complete)),
        "best_static_pool": best_static_pool,
        "best_static_topology": str(best_static_name),
        "best_static_mean_delay_ms": finite_mean(best_static_values),
        "dynamic_best_mean_delay_ms": finite_mean(dynamic_values),
        "dynamic_unique_topologies": int(len(set(str(x) for x in dynamic_names))),
        "dynamic_steps_without_complete_candidate": int(np.count_nonzero(~has_valid)),
        "dynamic_less_than_full_link_steps": int(np.count_nonzero(violation_mask)),
        "dynamic_less_than_full_link_max_gap_ms": (
            float(np.nanmax(full_link_values[violation_mask] - dynamic_values[violation_mask]))
            if np.any(violation_mask)
            else 0.0
        ),
        "full_link_mean_delay_ms": finite_mean(full_link_values),
        "gridplus_mean_delay_ms": finite_mean(gridplus_values),
        "dynamic_csv": str(dynamic_csv),
    }
    (pair_dir / "paper_style_delay_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    x_hours = steps.astype(np.float64) / 3600.0
    fig, ax = plt.subplots(figsize=(16, 7.3), dpi=180)
    for col_idx in range(valid_matrix.shape[1]):
        ax.plot(x_hours, valid_matrix[:, col_idx], color="#64748b", alpha=0.035, linewidth=0.45)
    ax.plot(
        x_hours,
        best_static_values,
        color="#2563eb",
        linewidth=2.1,
        label=f"best static motif: {best_static_name}",
    )
    ax.plot(
        x_hours,
        dynamic_values,
        color="#111827",
        linestyle="--",
        linewidth=2.0,
        label="dynamic best motif",
    )
    ax.plot(x_hours, full_link_values, color="#dc2626", linewidth=1.8, label="full_link")
    ax.plot(x_hours, gridplus_values, color="#16a34a", linewidth=1.8, label="+grid")
    ax.set_title(f"{pair_label} mean shortest delay with region-internal +grid constraint")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(f"{pair_label} mean shortest delay (ms)")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    plot_path = pair_dir / "paper_style_mean_shortest_delay_ms.png"
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main() -> int:
    args = parse_args()
    raw = load_yaml(args.config)
    config = viewer_config_from_workflow(raw)
    pairs = region_pair_specs(raw, subset=args.pairs)
    run_dir = Path(args.run_dir) if args.run_dir is not None else path_from(raw.get("paths", {}), "out_dir")
    meta = json.loads((run_dir / "region_internal_grid_metrics_meta.json").read_text(encoding="utf-8"))
    delay_store_dir = Path(args.delay_store_dir or meta.get("delay_store_dir") or path_from(raw["paths"], "delay_store_dir"))
    if args.no_position_cache:
        position_cache_dir = None
    elif args.position_cache_dir is not None:
        position_cache_dir = args.position_cache_dir
    elif meta.get("position_cache_dir"):
        position_cache_dir = Path(meta["position_cache_dir"])
    else:
        position_cache_dir = None

    if not args.skip_baselines:
        ensure_baseline_outputs(
            run_dir=run_dir,
            raw=raw,
            config=config,
            pair_specs=pairs,
            force=bool(args.force_baselines),
            delay_store_dir=delay_store_dir,
            position_cache_dir=position_cache_dir,
            delay_engine=str(args.baseline_delay_engine),
            step_heartbeat_seconds=float(args.baseline_step_heartbeat_seconds),
        )

    outputs: list[str] = []
    for pair in pairs:
        pair_dir = run_dir / str(pair.key)
        outputs.append(str(plot_pair_delay(pair_key=str(pair.key), pair_label=str(pair.label), pair_dir=pair_dir)))
    print(json.dumps({"plots": outputs}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
