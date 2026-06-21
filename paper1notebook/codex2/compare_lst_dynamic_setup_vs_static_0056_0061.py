from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
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

import plan_motif0056_to_0061_link_setup as plan  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION, make_edge_table_from_records  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


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
DEFAULT_LST_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\usage_driven_iterative_exemption_lst_sweep_056_061_china_europe_1s_any_direct"
    r"\t0_86160_stride1\switch36000_54000\releaseguard1\lst060"
)
DEFAULT_OUT_ROOT = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\lst60_metric_compare_056_061"
)

PAIR_KEY = "china_europe"
SOURCE_GROUP_ID = 2
TARGET_GROUP_ID = 3
SOURCE_MOTIF_ID = 56
TARGET_MOTIF_ID = 61


@dataclass(frozen=True)
class TopologyLite:
    name: str
    motif_id: int
    motif_text: str
    edge_table: object
    right_by_owner: dict[int, plan.RightLink]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute China-Europe shortest-hop and shortest-delay metrics for an LST-constrained "
            "dynamic 000056->000061 setup sequence, compared with static 000056 and static 000061."
        )
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--lst-dir", type=Path, default=DEFAULT_LST_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def load_topology_lite(motif_id: int) -> TopologyLite:
    rows = plan.read_motif_rows(plan.MOTIF_LIBRARY_CSV)
    row = rows[int(motif_id)]
    spec = plan.build_topology_spec(int(motif_id), row)
    right_by_owner, _left_by_right = plan.build_right_links(spec.edge_table)
    return TopologyLite(
        name=str(spec.name),
        motif_id=int(motif_id),
        motif_text=str(spec.motif),
        edge_table=spec.edge_table,
        right_by_owner=right_by_owner,
    )


def edge_key_index(edge_table) -> dict[tuple[int, int], int]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx])): int(idx)
        for idx in range(edge_table.num_edges)
    }


def edge_key_set(edge_table) -> set[tuple[int, int]]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(edge_table.num_edges)
    }


def all_intra_keys(edge_table) -> set[tuple[int, int]]:
    return {
        plan.edge_key(int(edge_table.src[idx]), int(edge_table.dst[idx]))
        for idx in range(edge_table.num_edges)
        if int(edge_table.option[idx]) == INTRA_OPTION
    }


def union_edge_table(topologies: list[TopologyLite]):
    records: list[tuple[int, int, int, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for topology in topologies:
        edge_table = topology.edge_table
        for idx in range(edge_table.num_edges):
            src = int(edge_table.src[idx])
            dst = int(edge_table.dst[idx])
            key = plan.edge_key(src, dst)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                (
                    int(edge_table.src_plane[idx]),
                    int(edge_table.src_y[idx]),
                    int(edge_table.dst_plane[idx]),
                    int(edge_table.dst_y[idx]),
                    int(edge_table.option[idx]),
                )
            )
    return make_edge_table_from_records(p=int(G60_CONFIG.P), n=int(G60_CONFIG.N), records=records)


def static_mask(edge_table, topology: TopologyLite) -> np.ndarray:
    key_to_idx = edge_key_index(edge_table)
    mask = np.zeros(edge_table.num_edges, dtype=bool)
    for key in edge_key_set(topology.edge_table):
        idx = key_to_idx.get(key)
        if idx is not None:
            mask[idx] = True
    return mask


def read_schedule_events(path: Path) -> dict[int, tuple[int, int]]:
    events: dict[int, tuple[int, int]] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            owner = int(row["owner"])
            plan_start = int(row["plan_start"])
            plan_end = int(row["plan_end"])
            previous = events.get(owner)
            if previous is not None:
                raise ValueError(f"owner has multiple schedule rows, owner={owner}: {previous}, {(plan_start, plan_end)}")
            events[owner] = (plan_start, plan_end)
    return events


def dynamic_mask_for_step(
    *,
    step: int,
    edge_table,
    source: TopologyLite,
    target: TopologyLite,
    events_by_owner: dict[int, tuple[int, int]],
    key_to_idx: dict[tuple[int, int], int],
    intra_keys: set[tuple[int, int]],
) -> np.ndarray:
    active_keys = set(intra_keys)
    all_owners = sorted(set(source.right_by_owner) | set(target.right_by_owner))
    for owner in all_owners:
        old_link = source.right_by_owner.get(owner)
        new_link = target.right_by_owner.get(owner)
        event = events_by_owner.get(owner)
        if event is None:
            link = old_link or new_link
            if link is not None:
                active_keys.add(link.edge_key)
            continue

        setup_start, setup_end = event
        if int(step) < int(setup_start):
            if old_link is not None:
                active_keys.add(old_link.edge_key)
        elif int(setup_start) <= int(step) < int(setup_end):
            # Building state is not routable. It should be visible in GUI, but excluded from shortest paths.
            continue
        else:
            if new_link is not None:
                active_keys.add(new_link.edge_key)

    mask = np.zeros(edge_table.num_edges, dtype=bool)
    for key in active_keys:
        idx = key_to_idx.get(key)
        if idx is not None:
            mask[idx] = True
    return mask


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
    steps: list[int],
    edge_table,
    source: TopologyLite,
    target: TopologyLite,
    events_by_owner: dict[int, tuple[int, int]],
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    progress_every: int,
) -> pd.DataFrame:
    if len(steps) < 2:
        raise ValueError("need at least two steps")

    key_to_idx = edge_key_index(edge_table)
    intra_keys = all_intra_keys(edge_table)
    static_masks = {
        "motif000056": static_mask(edge_table, source),
        "motif000061": static_mask(edge_table, target),
    }
    src_all = np.asarray(edge_table.src, dtype=np.int32)
    dst_all = np.asarray(edge_table.dst, dtype=np.int32)
    total_nodes = int(G60_CONFIG.total_sats)
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), int(steps[1] - steps[0]))
    position_rows = delay_rows
    lookup = build_weight_lookup(
        edge_table,
        delay_store,
        config=G60_CONFIG,
        allow_intra_fallback=True,
    )

    rows: list[dict[str, float | int]] = []
    started = time.time()
    for row_idx, step in enumerate(steps):
        dynamic_mask = dynamic_mask_for_step(
            step=int(step),
            edge_table=edge_table,
            source=source,
            target=target,
            events_by_owner=events_by_owner,
            key_to_idx=key_to_idx,
            intra_keys=intra_keys,
        )
        masks = {"dynamic_lst60": dynamic_mask, **static_masks}
        sources = group_nodes_for_step(group_data, int(step), SOURCE_GROUP_ID)
        targets = group_nodes_for_step(group_data, int(step), TARGET_GROUP_ID)
        weights = edge_weights_for_step(
            edge_table=edge_table,
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

        if int(progress_every) > 0 and ((row_idx + 1) % int(progress_every) == 0 or row_idx + 1 == len(steps)):
            print(
                f"[lst-metrics] {row_idx + 1}/{len(steps)} step={step} "
                f"dynamic_hops={out['dynamic_lst60_mean_shortest_hops']:.3f} "
                f"dynamic_delay={out['dynamic_lst60_mean_shortest_delay_ms']:.3f}ms "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return pd.DataFrame(rows)


def write_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for name in ("dynamic_lst60", "motif000056", "motif000061"):
        rows.append(
            {
                "series": name,
                "mean_hops_over_time": float(np.nanmean(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float))),
                "min_hops_over_time": float(np.nanmin(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float))),
                "max_hops_over_time": float(np.nanmax(df[f"{name}_mean_shortest_hops"].to_numpy(dtype=float))),
                "mean_delay_ms_over_time": float(np.nanmean(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))),
                "min_delay_ms_over_time": float(np.nanmin(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))),
                "max_delay_ms_over_time": float(np.nanmax(df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=float))),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_dynamic_lst60_vs_static_0056_0061.csv", index=False, encoding="utf-8-sig")
    return summary


def plot_series(df: pd.DataFrame, out_dir: Path, *, metric_suffix: str, ylabel: str, filename: str) -> None:
    styles = {
        "dynamic_lst60": ("dynamic LST=60 setup", "#d97706", 2.2, 0.95),
        "motif000056": ("motif000056 DBD | --B", "#1B4F9C", 1.4, 0.82),
        "motif000061": ("motif000061 DCD | C--", "#C1121F", 1.35, 0.78),
    }
    fig, ax = plt.subplots(figsize=(14.5, 6.2), dpi=150)
    hours = df["hour"].to_numpy(dtype=float)
    for name, (label, color, linewidth, alpha) in styles.items():
        ax.plot(
            hours,
            df[f"{name}_{metric_suffix}"].to_numpy(dtype=float),
            label=label,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
        )
    ax.axvline(10.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.45)
    ax.text(10.03, ax.get_ylim()[1], "10h switch target", fontsize=9, color="#111827", va="top")
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"G60 China-Europe {ylabel}: dynamic LST=60 vs static motifs")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / filename)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if args.out_dir is None:
        out_dir = DEFAULT_OUT_ROOT / f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source = load_topology_lite(SOURCE_MOTIF_ID)
    target = load_topology_lite(TARGET_MOTIF_ID)
    edge_table = union_edge_table([source, target])
    events_by_owner = read_schedule_events(Path(args.lst_dir) / "schedule_1s.csv")
    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
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
        edge_table=edge_table,
        source=source,
        target=target,
        events_by_owner=events_by_owner,
        group_data=group_data,
        delay_store=delay_store,
        position_store=position_store,
        progress_every=int(args.progress_every),
    )
    df.to_csv(out_dir / "timeseries_dynamic_lst60_vs_static_0056_0061.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "steps.npy", np.asarray(steps, dtype=np.int64))
    for name in ("dynamic_lst60", "motif000056", "motif000061"):
        np.save(out_dir / f"{name}_mean_shortest_hops.npy", df[f"{name}_mean_shortest_hops"].to_numpy(dtype=np.float32))
        np.save(
            out_dir / f"{name}_mean_shortest_delay_ms.npy",
            df[f"{name}_mean_shortest_delay_ms"].to_numpy(dtype=np.float32),
        )
    summary = write_summary(df, out_dir)
    plot_series(
        df,
        out_dir,
        metric_suffix="mean_shortest_hops",
        ylabel="mean shortest hops",
        filename="dynamic_lst60_vs_static_0056_0061_hops.png",
    )
    plot_series(
        df,
        out_dir,
        metric_suffix="mean_shortest_delay_ms",
        ylabel="mean shortest delay (ms)",
        filename="dynamic_lst60_vs_static_0056_0061_delay_ms.png",
    )
    meta = {
        "pair": PAIR_KEY,
        "source_group_id": SOURCE_GROUP_ID,
        "target_group_id": TARGET_GROUP_ID,
        "source_motif": {"id": source.motif_id, "name": source.name, "motif": source.motif_text},
        "target_motif": {"id": target.motif_id, "name": target.name, "motif": target.motif_text},
        "lst_dir": str(args.lst_dir),
        "schedule_rows": len(events_by_owner),
        "steps": {"start": int(args.start), "end": int(args.end), "stride": int(args.stride), "count": len(steps)},
        "edge_table_edges": int(edge_table.num_edges),
        "output_files": [
            "timeseries_dynamic_lst60_vs_static_0056_0061.csv",
            "summary_dynamic_lst60_vs_static_0056_0061.csv",
            "dynamic_lst60_vs_static_0056_0061_hops.png",
            "dynamic_lst60_vs_static_0056_0061_delay_ms.png",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
