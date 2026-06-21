from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402
from scipy.sparse.csgraph import dijkstra, shortest_path  # noqa: E402


THIS_FILE = Path(__file__).resolve()
GENERIC_ROOT = THIS_FILE.parents[2]
CODEX2_DIR = THIS_FILE.parent
for path in (GENERIC_ROOT, CODEX2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402

from plan_guarded_t2_switch_0056_0061 import (  # noqa: E402
    build_static_topology,
    edge_key_index,
    union_edge_table,
)
from plan_motif0056_to_0061_link_setup import edge_key  # noqa: E402


GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\full_option_edge_delay\G60_full_options_t0_86164_stride1"
)
POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition"
    r"\_position_cache\cache_0_86164_1s"
)
DEFAULT_PLAN_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\nominal_epoch_smooth_switch_056_061\t0_86160_stride1"
    r"\tau36000_lst060_delta000_rho_eta"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\nominal_epoch_smooth_switch_056_061_metric_compare"
)

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3
SOURCE_MOTIF_ID = 56
TARGET_MOTIF_ID = 61


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute China-Europe shortest-hop and shortest-delay metrics for the rho/eta "
            "dynamic motif000056->motif000061 topology, compared with static 000056 and 000061."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--plan-dir", type=Path, default=DEFAULT_PLAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dynamic-key", type=str, default="dynamic_rho_eta")
    parser.add_argument("--dynamic-label", type=str, default="dynamic rho/eta LST=60")
    parser.add_argument("--file-prefix", type=str, default="rho_eta_dynamic_vs_static_0056_0061")
    parser.add_argument("--title-detail", type=str, default="rho/eta dynamic vs static motifs")
    return parser.parse_args()


def edge_key_set(edge_table) -> set[tuple[int, int]]:
    return {
        edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(int(edge_table.num_edges))
    }


def static_mask(union_table, topology) -> np.ndarray:
    key_to_idx = edge_key_index(union_table)
    mask = np.zeros(int(union_table.num_edges), dtype=bool)
    for key in edge_key_set(topology.spec.edge_table):
        idx = key_to_idx.get(key)
        if idx is not None:
            mask[int(idx)] = True
    return mask


def selected_plan_rows(plan_steps: np.ndarray, *, start: int, end: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    wanted = np.arange(int(start), int(end) + 1, int(stride), dtype=np.int64)
    rows = np.searchsorted(plan_steps, wanted)
    if rows.size == 0:
        raise ValueError("empty selected step range")
    bad = (rows >= plan_steps.size) | (np.asarray(plan_steps[rows], dtype=np.int64) != wanted)
    if bool(np.any(bad)):
        missing = wanted[np.flatnonzero(bad)[:10]]
        raise KeyError(f"plan does not contain requested steps, examples={missing.tolist()}")
    return wanted, rows.astype(np.int64)


def metric_mean(values: np.ndarray) -> tuple[float, int, float, float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), 0, float("nan"), float("nan")
    return float(np.mean(finite)), int(finite.size), float(np.min(finite)), float(np.max(finite))


def hop_summary_for_mask(
    *,
    mask: np.ndarray,
    src_all: np.ndarray,
    dst_all: np.ndarray,
    total_nodes: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[float, int, float, float]:
    if not sources or not targets or not bool(np.any(mask)):
        return float("nan"), 0, float("nan"), float("nan")
    active_src = src_all[mask]
    active_dst = dst_all[mask]
    graph = csr_matrix(
        (
            np.ones(active_src.size * 2, dtype=np.float32),
            (np.concatenate([active_src, active_dst]), np.concatenate([active_dst, active_src])),
        ),
        shape=(int(total_nodes), int(total_nodes)),
    )
    dist = shortest_path(
        graph,
        directed=False,
        unweighted=True,
        indices=np.asarray(sources, dtype=np.int32),
    )
    dist = np.atleast_2d(dist)[:, np.asarray(targets, dtype=np.int32)]
    return metric_mean(dist)


def delay_summary_for_mask(
    *,
    mask: np.ndarray,
    weights: np.ndarray,
    src_all: np.ndarray,
    dst_all: np.ndarray,
    total_nodes: int,
    sources: tuple[int, ...],
    targets: tuple[int, ...],
) -> tuple[float, int, float, float]:
    if not sources or not targets or not bool(np.any(mask)):
        return float("nan"), 0, float("nan"), float("nan")
    active_src = src_all[mask]
    active_dst = dst_all[mask]
    active_weights = np.asarray(weights[mask], dtype=np.float32)
    graph = csr_matrix(
        (
            np.concatenate([active_weights, active_weights]),
            (np.concatenate([active_src, active_dst]), np.concatenate([active_dst, active_src])),
        ),
        shape=(int(total_nodes), int(total_nodes)),
    )
    dist = dijkstra(
        graph,
        directed=False,
        indices=np.asarray(sources, dtype=np.int32),
    )
    dist = np.atleast_2d(dist)[:, np.asarray(targets, dtype=np.int32)]
    return metric_mean(dist)


def compute_metrics(
    *,
    steps: np.ndarray,
    active_masks: np.ndarray,
    union_table,
    source_topology,
    target_topology,
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    progress_every: int,
    dynamic_key: str,
) -> pd.DataFrame:
    if active_masks.shape[0] != steps.size:
        raise ValueError(f"active mask rows {active_masks.shape[0]} != steps {steps.size}")
    if active_masks.shape[1] != int(union_table.num_edges):
        raise ValueError(f"active mask columns {active_masks.shape[1]} != union edges {union_table.num_edges}")

    masks_static = {
        "motif000056": static_mask(union_table, source_topology),
        "motif000061": static_mask(union_table, target_topology),
    }
    src_all = np.asarray(union_table.src, dtype=np.int32)
    dst_all = np.asarray(union_table.dst, dtype=np.int32)
    total_nodes = int(G60_CONFIG.total_sats)
    stride = int(steps[1] - steps[0]) if steps.size > 1 else 1
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(stride))
    position_rows = delay_rows
    lookup = build_weight_lookup(
        union_table,
        delay_store,
        config=G60_CONFIG,
        allow_intra_fallback=True,
    )

    rows: list[dict[str, float | int]] = []
    started = time.time()
    for row_idx, step in enumerate(steps):
        masks = {
            str(dynamic_key): np.asarray(active_masks[row_idx], dtype=bool),
            **masks_static,
        }
        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        weights = edge_weights_for_step(
            edge_table=union_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=int(delay_rows[row_idx]),
            position_row=int(position_rows[row_idx]),
        )

        out: dict[str, float | int] = {
            "step": int(step),
            "hour": float(step) / 3600.0,
            "source_nodes": int(len(sources)),
            "target_nodes": int(len(targets)),
        }
        for name, mask in masks.items():
            hop_mean, hop_pairs, hop_min, hop_max = hop_summary_for_mask(
                mask=mask,
                src_all=src_all,
                dst_all=dst_all,
                total_nodes=total_nodes,
                sources=sources,
                targets=targets,
            )
            delay_mean, delay_pairs, delay_min, delay_max = delay_summary_for_mask(
                mask=mask,
                weights=weights,
                src_all=src_all,
                dst_all=dst_all,
                total_nodes=total_nodes,
                sources=sources,
                targets=targets,
            )
            out[f"{name}_active_edges"] = int(np.count_nonzero(mask))
            out[f"{name}_reachable_pairs_hops"] = int(hop_pairs)
            out[f"{name}_mean_shortest_hops"] = float(hop_mean)
            out[f"{name}_min_shortest_hops"] = float(hop_min)
            out[f"{name}_max_shortest_hops"] = float(hop_max)
            out[f"{name}_reachable_pairs_delay"] = int(delay_pairs)
            out[f"{name}_mean_shortest_delay_ms"] = float(delay_mean)
            out[f"{name}_min_shortest_delay_ms"] = float(delay_min)
            out[f"{name}_max_shortest_delay_ms"] = float(delay_max)
        rows.append(out)

        if int(progress_every) > 0 and ((row_idx + 1) % int(progress_every) == 0 or row_idx + 1 == steps.size):
            print(
                f"[dynamic-metrics] {row_idx + 1}/{steps.size} step={int(step)} "
                f"dynamic_hops={out[f'{dynamic_key}_mean_shortest_hops']:.3f} "
                f"dynamic_delay={out[f'{dynamic_key}_mean_shortest_delay_ms']:.3f}ms "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, out_dir: Path, *, dynamic_key: str, file_prefix: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name in (str(dynamic_key), "motif000056", "motif000061"):
        hops = df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float)
        delay = df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float)
        rows.append(
            {
                "series": name,
                "mean_hops_over_time": float(np.nanmean(hops)),
                "min_hops_over_time": float(np.nanmin(hops)),
                "max_hops_over_time": float(np.nanmax(hops)),
                "p95_hops_over_time": float(np.nanpercentile(hops, 95)),
                "mean_delay_ms_over_time": float(np.nanmean(delay)),
                "min_delay_ms_over_time": float(np.nanmin(delay)),
                "max_delay_ms_over_time": float(np.nanmax(delay)),
                "p95_delay_ms_over_time": float(np.nanpercentile(delay, 95)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / f"summary_{file_prefix}.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_series(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    metric_suffix: str,
    ylabel: str,
    filename: str,
    dynamic_key: str,
    dynamic_label: str,
    title_detail: str,
) -> None:
    styles = {
        str(dynamic_key): (str(dynamic_label), "#d97706", 2.25, 0.96),
        "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.35, 0.82),
        "motif000061": ("motif000061 DCD | C--", "#C1121F", 1.35, 0.78),
    }
    fig, ax = plt.subplots(figsize=(14.8, 6.3), dpi=170)
    hours = df["hour"].to_numpy(dtype=float)
    for name, (label, color, linewidth, alpha) in styles.items():
        values = df[f"{name}_{metric_suffix}"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        mean_text = f"{float(np.mean(finite)):.3f}" if finite.size else "nan"
        ax.plot(
            hours,
            values,
            label=f"{label} | mean={mean_text}",
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.axvline(10.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.text(10.03, ax.get_ylim()[1], "nominal switch @10h", fontsize=9, color="#111827", va="top")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe {ylabel}: {title_detail}")
    ax.grid(True, alpha=0.24, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    plan_dir = Path(args.plan_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else (
        DEFAULT_OUT_ROOT / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_steps = np.load(plan_dir / "steps.npy", mmap_mode="r")
    steps, plan_rows = selected_plan_rows(
        np.asarray(plan_steps, dtype=np.int64),
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
    )
    active_masks = np.asarray(np.load(plan_dir / "edge_active_mask.npy", mmap_mode="r")[plan_rows, :], dtype=bool)

    source_topology = build_static_topology(SOURCE_MOTIF_ID)
    target_topology = build_static_topology(TARGET_MOTIF_ID)
    union_table = union_edge_table(source_topology, target_topology)

    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=[int(x) for x in steps],
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )
    delay_store = FullLinkDelayStore(DELAY_STORE_DIR)
    position_store = PositionCacheStore(POSITION_CACHE_DIR)

    df = compute_metrics(
        steps=steps,
        active_masks=active_masks,
        union_table=union_table,
        source_topology=source_topology,
        target_topology=target_topology,
        group_data=group_data,
        delay_store=delay_store,
        position_store=position_store,
        progress_every=int(args.progress_every),
        dynamic_key=str(args.dynamic_key),
    )
    file_prefix = str(args.file_prefix)
    df.to_csv(out_dir / f"timeseries_{file_prefix}.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "steps.npy", steps.astype(np.int64))
    for name in (str(args.dynamic_key), "motif000056", "motif000061"):
        np.save(out_dir / f"{name}_mean_shortest_hops.npy", df[f"{name}_mean_shortest_hops"].to_numpy(dtype=np.float32))
        np.save(
            out_dir / f"{name}_mean_shortest_delay_ms.npy",
            df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=np.float32),
        )

    summary = write_summary(df, out_dir, dynamic_key=str(args.dynamic_key), file_prefix=file_prefix)
    plot_series(
        df,
        out_dir,
        metric_suffix="mean_shortest_hops",
        ylabel="mean shortest hops",
        filename=f"{file_prefix}_hops.png",
        dynamic_key=str(args.dynamic_key),
        dynamic_label=str(args.dynamic_label),
        title_detail=str(args.title_detail),
    )
    plot_series(
        df,
        out_dir,
        metric_suffix="mean_shortest_delay_ms",
        ylabel="mean shortest delay (ms)",
        filename=f"{file_prefix}_delay_ms.png",
        dynamic_key=str(args.dynamic_key),
        dynamic_label=str(args.dynamic_label),
        title_detail=str(args.title_detail),
    )

    meta = {
        "pair": PAIR_KEY,
        "source_group_id": SOURCE_GROUP_ID,
        "target_group_id": TARGET_GROUP_ID,
        "source_motif_id": SOURCE_MOTIF_ID,
        "target_motif_id": TARGET_MOTIF_ID,
        "plan_dir": str(plan_dir),
        "steps": {"start": int(steps[0]), "end": int(steps[-1]), "stride": int(args.stride), "count": int(steps.size)},
        "edge_table_edges": int(union_table.num_edges),
        "dynamic_active_mask_source": "edge_active_mask.npy",
        "dynamic_key": str(args.dynamic_key),
        "dynamic_label": str(args.dynamic_label),
        "building_links_are_routable": False,
        "output_files": [
            f"timeseries_{file_prefix}.csv",
            f"summary_{file_prefix}.csv",
            f"{file_prefix}_hops.png",
            f"{file_prefix}_delay_ms.png",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(summary.to_string(index=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
