from __future__ import annotations

import argparse
import json
import sys
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

import compare_lst_dynamic_setup_vs_static_0056_0061 as cmp  # noqa: E402
from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.link_delay.module.position_cache import PositionCacheStore  # noqa: E402
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_metrics.module.group_states import group_nodes_for_step  # noqa: E402
from src.topology_workflow.module.shortest_delay import build_weight_lookup, edge_weights_for_step  # noqa: E402


DEFAULT_TIMESERIES = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\switch_setup"
    r"\lst60_metric_compare_056_061_056\t0_86160_stride60"
    r"\timeseries_roundtrip_lst60_vs_static_0056_0061.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_0056_0061_common_spike_analysis"
    r"\china_europe_t0_86160_stride60"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze common sudden spikes shared by motif000056 and motif000061 China-Europe metrics."
    )
    parser.add_argument("--timeseries", type=Path, default=DEFAULT_TIMESERIES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=12)
    return parser.parse_args()


def node_xy(node: int) -> str:
    return f"{int(node)}({int(node) // int(G60_CONFIG.N)},{int(node) % int(G60_CONFIG.N)})"


def nodes_text(nodes: list[int]) -> str:
    return ";".join(node_xy(node) for node in nodes)


def graph_from_edges(edge_table, weights: np.ndarray | None = None):
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    if weights is None:
        data = np.ones(src.size * 2, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        data = np.concatenate([weights, weights])
    return csr_matrix(
        (data, (np.concatenate([src, dst]), np.concatenate([dst, src]))),
        shape=(int(G60_CONFIG.total_sats), int(G60_CONFIG.total_sats)),
    )


def mean_matrix_for(
    *,
    motif_id: int,
    metric: str,
    step: int,
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    delay_row_by_step: dict[int, int],
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    topology = cmp.load_topology_lite(int(motif_id))
    edge_table = topology.edge_table
    sources = group_nodes_for_step(group_data, int(step), cmp.SOURCE_GROUP_ID)
    targets = group_nodes_for_step(group_data, int(step), cmp.TARGET_GROUP_ID)
    if metric == "hops":
        matrix = shortest_path(
            graph_from_edges(edge_table),
            directed=False,
            unweighted=True,
            indices=np.asarray(sources, dtype=np.int32),
        )
    elif metric == "delay_ms":
        lookup = build_weight_lookup(edge_table, delay_store, config=G60_CONFIG, allow_intra_fallback=True)
        delay_row = int(delay_row_by_step[int(step)])
        weights = edge_weights_for_step(
            edge_table=edge_table,
            lookup=lookup,
            delay_store=delay_store,
            position_store=position_store,
            delay_row=delay_row,
            position_row=delay_row,
        )
        matrix = dijkstra(
            graph_from_edges(edge_table, weights),
            directed=False,
            indices=np.asarray(sources, dtype=np.int32),
        )
    else:
        raise ValueError(metric)
    matrix = np.atleast_2d(matrix)[:, np.asarray(targets, dtype=np.int32)]
    return sources, targets, np.asarray(matrix, dtype=np.float64)


def changed_node_rows(
    *,
    motif_id: int,
    metric: str,
    step_prev: int,
    step_now: int,
    group_data: dict,
    delay_store: FullLinkDelayStore,
    position_store: PositionCacheStore,
    delay_row_by_step: dict[int, int],
) -> list[dict[str, object]]:
    src_prev, tgt_prev, mat_prev = mean_matrix_for(
        motif_id=motif_id,
        metric=metric,
        step=step_prev,
        group_data=group_data,
        delay_store=delay_store,
        position_store=position_store,
        delay_row_by_step=delay_row_by_step,
    )
    src_now, tgt_now, mat_now = mean_matrix_for(
        motif_id=motif_id,
        metric=metric,
        step=step_now,
        group_data=group_data,
        delay_store=delay_store,
        position_store=position_store,
        delay_row_by_step=delay_row_by_step,
    )
    src_prev_index = {int(node): idx for idx, node in enumerate(src_prev)}
    src_now_index = {int(node): idx for idx, node in enumerate(src_now)}
    tgt_prev_index = {int(node): idx for idx, node in enumerate(tgt_prev)}
    tgt_now_index = {int(node): idx for idx, node in enumerate(tgt_now)}
    rows: list[dict[str, object]] = []
    for side, state, nodes, index, matrix in (
        ("source", "added", sorted(set(src_now) - set(src_prev)), src_now_index, mat_now),
        ("source", "removed", sorted(set(src_prev) - set(src_now)), src_prev_index, mat_prev),
        ("target", "added", sorted(set(tgt_now) - set(tgt_prev)), tgt_now_index, mat_now.T),
        ("target", "removed", sorted(set(tgt_prev) - set(tgt_now)), tgt_prev_index, mat_prev.T),
    ):
        for node in nodes:
            arr = np.asarray(matrix[int(index[int(node)])], dtype=np.float64)
            finite = arr[np.isfinite(arr)]
            rows.append(
                {
                    "step": int(step_now),
                    "step_prev": int(step_prev),
                    "hour": float(step_now) / 3600.0,
                    "motif_id": int(motif_id),
                    "metric": str(metric),
                    "side": side,
                    "change": state,
                    "node": int(node),
                    "node_xy": node_xy(int(node)),
                    "mean_to_opposite_group": float(np.mean(finite)) if finite.size else float("nan"),
                    "min_to_opposite_group": float(np.min(finite)) if finite.size else float("nan"),
                    "max_to_opposite_group": float(np.max(finite)) if finite.size else float("nan"),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.timeseries)
    steps = [int(x) for x in df["step"].tolist()]
    stride = int(steps[1] - steps[0])

    group_data = load_or_build_group_data(
        xml_file=cmp.GROUP_XML,
        group_cache_dir=cmp.GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=stride,
        enabled=True,
        force=False,
    )
    delay_store = FullLinkDelayStore(cmp.DELAY_STORE_DIR)
    position_store = PositionCacheStore(cmp.POSITION_CACHE_DIR)
    delay_rows = delay_store.rows_for_interval(int(steps[0]), int(steps[-1]), stride)
    delay_row_by_step = {int(step): int(delay_rows[idx]) for idx, step in enumerate(steps)}

    source_group_changes = [0]
    target_group_changes = [0]
    total_group_churn = [0]
    for prev, now in zip(steps[:-1], steps[1:]):
        src_prev = set(group_nodes_for_step(group_data, prev, cmp.SOURCE_GROUP_ID))
        src_now = set(group_nodes_for_step(group_data, now, cmp.SOURCE_GROUP_ID))
        tgt_prev = set(group_nodes_for_step(group_data, prev, cmp.TARGET_GROUP_ID))
        tgt_now = set(group_nodes_for_step(group_data, now, cmp.TARGET_GROUP_ID))
        src_churn = len(src_prev ^ src_now)
        tgt_churn = len(tgt_prev ^ tgt_now)
        source_group_changes.append(src_churn)
        target_group_changes.append(tgt_churn)
        total_group_churn.append(src_churn + tgt_churn)
    df["source_group_churn"] = source_group_changes
    df["target_group_churn"] = target_group_changes
    df["group_churn"] = total_group_churn

    for metric_col in ("mean_shortest_hops", "mean_shortest_delay_ms"):
        d56 = df[f"motif000056_{metric_col}"].diff().clip(lower=0.0)
        d61 = df[f"motif000061_{metric_col}"].diff().clip(lower=0.0)
        df[f"common_positive_jump_{metric_col}"] = d56 * d61
        df[f"motif000056_positive_jump_{metric_col}"] = d56
        df[f"motif000061_positive_jump_{metric_col}"] = d61

    top = (
        df.sort_values("common_positive_jump_mean_shortest_delay_ms", ascending=False)
        .head(int(args.top_n))
        .copy()
    )
    summary_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    for row in top.itertuples(index=False):
        step_now = int(row.step)
        step_prev = int(step_now - stride)
        if step_prev < steps[0]:
            continue
        src_prev = set(group_nodes_for_step(group_data, step_prev, cmp.SOURCE_GROUP_ID))
        src_now = set(group_nodes_for_step(group_data, step_now, cmp.SOURCE_GROUP_ID))
        tgt_prev = set(group_nodes_for_step(group_data, step_prev, cmp.TARGET_GROUP_ID))
        tgt_now = set(group_nodes_for_step(group_data, step_now, cmp.TARGET_GROUP_ID))
        summary_rows.append(
            {
                "step": step_now,
                "hour": float(step_now) / 3600.0,
                "source_group_churn": int(len(src_prev ^ src_now)),
                "target_group_churn": int(len(tgt_prev ^ tgt_now)),
                "group_churn": int(len(src_prev ^ src_now) + len(tgt_prev ^ tgt_now)),
                "source_added": nodes_text(sorted(src_now - src_prev)),
                "source_removed": nodes_text(sorted(src_prev - src_now)),
                "target_added": nodes_text(sorted(tgt_now - tgt_prev)),
                "target_removed": nodes_text(sorted(tgt_prev - tgt_now)),
                "motif000056_delay_jump_ms": float(
                    row.motif000056_positive_jump_mean_shortest_delay_ms
                ),
                "motif000061_delay_jump_ms": float(
                    row.motif000061_positive_jump_mean_shortest_delay_ms
                ),
                "common_delay_jump_score": float(row.common_positive_jump_mean_shortest_delay_ms),
                "motif000056_hops_jump": float(row.motif000056_positive_jump_mean_shortest_hops),
                "motif000061_hops_jump": float(row.motif000061_positive_jump_mean_shortest_hops),
            }
        )
        for motif_id in (56, 61):
            for metric in ("hops", "delay_ms"):
                node_rows.extend(
                    changed_node_rows(
                        motif_id=motif_id,
                        metric=metric,
                        step_prev=step_prev,
                        step_now=step_now,
                        group_data=group_data,
                        delay_store=delay_store,
                        position_store=position_store,
                        delay_row_by_step=delay_row_by_step,
                    )
                )

    summary = pd.DataFrame(summary_rows)
    node_detail = pd.DataFrame(node_rows)
    df.to_csv(out_dir / "timeseries_with_common_jump_and_group_churn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "top_common_delay_spikes_summary.csv", index=False, encoding="utf-8-sig")
    node_detail.to_csv(out_dir / "top_common_delay_spikes_changed_node_detail.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(3, 1, figsize=(14.5, 9.2), dpi=150, sharex=True)
    hours = df["hour"].to_numpy(dtype=float)
    axes[0].plot(hours, df["motif000056_mean_shortest_delay_ms"], color="#1B4F9C", linewidth=1.1, label="motif000056")
    axes[0].plot(hours, df["motif000061_mean_shortest_delay_ms"], color="#C1121F", linewidth=1.1, label="motif000061")
    axes[0].set_ylabel("mean delay (ms)")
    axes[0].legend(loc="best")
    axes[1].bar(hours, df["group_churn"], width=float(stride) / 3600.0, color="#64748b", alpha=0.7)
    axes[1].set_ylabel("group churn")
    axes[2].plot(
        hours,
        df["common_positive_jump_mean_shortest_delay_ms"],
        color="#d97706",
        linewidth=1.0,
        label="positive jump product",
    )
    axes[2].scatter(
        summary["hour"],
        summary["common_delay_jump_score"],
        color="#111827",
        s=22,
        label=f"top {len(summary)} common spikes",
        zorder=4,
    )
    axes[2].set_ylabel("common jump score")
    axes[2].set_xlabel("time (hour)")
    axes[2].legend(loc="best")
    for ax in axes:
        ax.grid(True, alpha=0.24, linewidth=0.6)
    fig.suptitle("G60 China-Europe motif000056/000061 common spikes vs group endpoint churn")
    fig.tight_layout()
    fig.savefig(out_dir / "common_spikes_vs_group_churn.png")
    plt.close(fig)

    meta = {
        "timeseries": str(args.timeseries),
        "out_dir": str(out_dir),
        "top_n": int(args.top_n),
        "main_interpretation": (
            "Common spikes are dominated by ground-region endpoint-set changes. "
            "The important factor is not just churn count, but whether added/removed satellites "
            "are favorable or unfavorable endpoints under both static motifs."
        ),
        "outputs": [
            "timeseries_with_common_jump_and_group_churn.csv",
            "top_common_delay_spikes_summary.csv",
            "top_common_delay_spikes_changed_node_detail.csv",
            "common_spikes_vs_group_churn.png",
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(int(args.top_n)).round(4).to_string(index=False))
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
