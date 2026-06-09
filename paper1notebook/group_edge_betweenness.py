from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG, ViewerConfig
from src.io import read_snap_xml
from src.model.static_hop_table import precompute_hop_and_next_hop, reconstruct_path


OPTION_DELTAS = {
    0: (1, 0),
    1: (1, -1),
    2: (2, 0),
    4: (1, 1),
}


@dataclass(frozen=True)
class FullOptionGraph:
    adj: dict[int, set[int]]
    edge_to_idx: dict[tuple[int, int], int]
    idx_to_edge: list[tuple[int, int]]
    edge_options: dict[tuple[int, int], str]
    graph: nx.Graph


def _edge_key(u: int, v: int) -> tuple[int, int]:
    u = int(u)
    v = int(v)
    return (u, v) if u < v else (v, u)


def _add_directed(adj: dict[int, set[int]], u: int, v: int) -> None:
    adj.setdefault(int(u), set()).add(int(v))


def _add_undirected(adj: dict[int, set[int]], u: int, v: int) -> None:
    _add_directed(adj, u, v)
    _add_directed(adj, v, u)


def edge_option_label(u: int, v: int, n: int) -> str:
    a, b = _edge_key(u, v)
    pa, ya = divmod(a, int(n))
    pb, yb = divmod(b, int(n))

    if pa == pb:
        dy = (yb - ya) % int(n)
        if dy in {1, int(n) - 1}:
            return "intra"
        return "same_plane"

    dx = pb - pa
    dy = (yb - ya) % int(n)
    for option, (op_dx, op_dy) in OPTION_DELTAS.items():
        if dx == op_dx and dy == (op_dy % int(n)):
            return f"option{option}"
    return "unknown"


def build_full_option_adj(
    p_count: int,
    n_per_plane: int,
    *,
    options: Iterable[int] = (0, 1, 2, 4),
    include_intra: bool = True,
    skip_first_last_source: bool = False,
    bidirectional: bool = True,
) -> dict[int, set[int]]:
    p_count = int(p_count)
    n_per_plane = int(n_per_plane)
    total_sats = p_count * n_per_plane
    adj: dict[int, set[int]] = {node: set() for node in range(total_sats)}

    if include_intra:
        for p in range(p_count):
            for y in range(n_per_plane):
                u = p * n_per_plane + y
                v = p * n_per_plane + ((y + 1) % n_per_plane)
                _add_undirected(adj, u, v)

    for p in range(p_count):
        if skip_first_last_source and p in {0, p_count - 1}:
            continue

        for y in range(n_per_plane):
            src = p * n_per_plane + y
            for option in options:
                dp, dy = OPTION_DELTAS[int(option)]
                q = p + dp
                if not (0 <= q < p_count):
                    continue
                yy = (y + dy) % n_per_plane
                dst = q * n_per_plane + yy
                if bidirectional:
                    _add_undirected(adj, src, dst)
                else:
                    _add_directed(adj, src, dst)

    return adj


def build_full_option_graph(
    config: ViewerConfig = G60_CONFIG,
    *,
    options: Iterable[int] = (0, 1, 2, 4),
    include_intra: bool = True,
    skip_first_last_source: bool = False,
) -> FullOptionGraph:
    adj = build_full_option_adj(
        config.P,
        config.N,
        options=options,
        include_intra=include_intra,
        skip_first_last_source=skip_first_last_source,
        bidirectional=True,
    )

    graph = nx.Graph()
    graph.add_nodes_from(range(config.total_sats))
    for u, dsts in adj.items():
        for v in dsts:
            if int(u) != int(v):
                graph.add_edge(int(u), int(v))

    idx_to_edge = sorted(_edge_key(u, v) for u, v in graph.edges())
    edge_to_idx = {edge: idx for idx, edge in enumerate(idx_to_edge)}
    edge_options = {edge: edge_option_label(edge[0], edge[1], config.N) for edge in idx_to_edge}
    return FullOptionGraph(
        adj=adj,
        edge_to_idx=edge_to_idx,
        idx_to_edge=idx_to_edge,
        edge_options=edge_options,
        graph=graph,
    )


def repeat_static_adj_for_viewer(static_adj: dict[int, set[int]], group_data: dict) -> dict[int, dict[int, set[int]]]:
    return {int(step): static_adj for step in sorted(group_data)}


def _valid_group_nodes(group_data: dict, step: int, group_id: int, total_sats: int) -> list[int]:
    raw = group_data.get(int(step), {}).get("groups", {}).get(int(group_id), set()) or set()
    return sorted({int(x) for x in raw if 0 <= int(x) < int(total_sats)})


def count_step_edge_betweenness(
    *,
    step: int,
    group_data: dict,
    group_a: int,
    group_b: int,
    next_hop: np.ndarray,
    edge_to_idx: dict[tuple[int, int], int],
    total_sats: int,
    inspect: bool = False,
) -> tuple[np.ndarray, dict, list[dict]]:
    nodes_a = _valid_group_nodes(group_data, step, group_a, total_sats)
    nodes_b = _valid_group_nodes(group_data, step, group_b, total_sats)

    counts = np.zeros(len(edge_to_idx), dtype=np.uint32)
    rows: list[dict] = []
    reachable = 0
    hop_sum = 0
    missing = 0

    for src in nodes_a:
        for dst in nodes_b:
            path = reconstruct_path(next_hop, src, dst)
            if not path:
                missing += 1
                if inspect:
                    rows.append(
                        {
                            "step": int(step),
                            "src": int(src),
                            "dst": int(dst),
                            "hops": "",
                            "path": "",
                            "path_indexed": "",
                        }
                    )
                continue

            hops = max(0, len(path) - 1)
            reachable += 1
            hop_sum += hops

            for u, v in zip(path[:-1], path[1:]):
                idx = edge_to_idx.get(_edge_key(u, v))
                if idx is not None:
                    counts[idx] += 1

            if inspect:
                rows.append(
                    {
                        "step": int(step),
                        "src": int(src),
                        "dst": int(dst),
                        "hops": int(hops),
                        "path": "->".join(str(x) for x in path),
                        "path_indexed": ",".join(
                            f"{idx + 1}:{node}" for idx, node in enumerate(path)
                        ),
                    }
                )

    summary = {
        "step": int(step),
        "group_a": int(group_a),
        "group_b": int(group_b),
        "group_a_node_count": int(len(nodes_a)),
        "group_b_node_count": int(len(nodes_b)),
        "pair_count": int(len(nodes_a) * len(nodes_b)),
        "reachable_pair_count": int(reachable),
        "missing_pair_count": int(missing),
        "mean_hops": float(hop_sum / reachable) if reachable else np.nan,
        "edge_traversal_total": int(counts.sum()),
        "max_edge_count": int(counts.max()) if counts.size else 0,
    }
    return counts, summary, rows


def edge_counts_to_frame(
    counts: np.ndarray,
    idx_to_edge: list[tuple[int, int]],
    edge_options: dict[tuple[int, int], str],
    config: ViewerConfig,
    *,
    count_col: str = "count",
) -> pd.DataFrame:
    rows = []
    for idx, (u, v) in enumerate(idx_to_edge):
        pu, yu = divmod(int(u), config.N)
        pv, yv = divmod(int(v), config.N)
        rows.append(
            {
                "edge_idx": int(idx),
                "u": int(u),
                "v": int(v),
                "u_p": int(pu),
                "u_y": int(yu),
                "v_p": int(pv),
                "v_y": int(yv),
                "edge_type": edge_options.get((int(u), int(v)), "unknown"),
                count_col: int(counts[idx]),
            }
        )
    return pd.DataFrame(rows)


def write_inspect_text(
    path: Path,
    *,
    step: int,
    group_a: int,
    group_b: int,
    nodes_a: list[int],
    nodes_b: list[int],
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f"step = {int(step)}\n")
        f.write(f"group_a = {int(group_a)} nodes = {nodes_a}\n")
        f.write(f"group_b = {int(group_b)} nodes = {nodes_b}\n")
        f.write(f"pair_paths = {len(rows)}\n")
        f.write("\n")
        f.write("src,dst,hops,path,path_indexed\n")
        for row in rows:
            f.write(
                f"{row['src']},{row['dst']},{row['hops']},"
                f"{row['path']},{row['path_indexed']}\n"
            )


def plot_edge_betweenness(
    counts: np.ndarray,
    idx_to_edge: list[tuple[int, int]],
    *,
    config: ViewerConfig,
    out_png: Path,
    title: str,
    group_a_nodes: Iterable[int] = (),
    group_b_nodes: Iterable[int] = (),
    top_edges: int = 0,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(counts)

    if top_edges and top_edges > 0 and top_edges < len(counts):
        keep = np.argsort(counts)[-int(top_edges):]
        edge_indices = [int(i) for i in keep if counts[int(i)] > 0]
    else:
        edge_indices = [idx for idx, c in enumerate(counts) if int(c) > 0]

    max_count = max((int(counts[idx]) for idx in edge_indices), default=1)

    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(-1, config.P)
    ax.set_ylim(-1, config.N)
    ax.invert_yaxis()
    ax.set_xlabel("orbit plane p")
    ax.set_ylabel("satellite index y")

    for idx in edge_indices:
        count = int(counts[idx])
        if count <= 0:
            continue
        u, v = idx_to_edge[idx]
        pu, yu = divmod(int(u), config.N)
        pv, yv = divmod(int(v), config.N)
        width = 0.25 + 5.5 * ((count / max_count) ** 0.65)
        alpha = 0.15 + 0.75 * ((count / max_count) ** 0.5)
        ax.plot([pu, pv], [yu, yv], color="#d62728", linewidth=width, alpha=alpha, zorder=2)

    sat_ids = np.arange(config.total_sats)
    xs = sat_ids // config.N
    ys = sat_ids % config.N
    ax.scatter(xs, ys, s=14, c="#e8e8e8", edgecolors="#666", linewidths=0.25, zorder=3)

    group_a_nodes = sorted(int(x) for x in group_a_nodes)
    group_b_nodes = sorted(int(x) for x in group_b_nodes)
    if group_a_nodes:
        ax.scatter(
            [x // config.N for x in group_a_nodes],
            [x % config.N for x in group_a_nodes],
            s=42,
            c="#1f77b4",
            label="group A visible sats",
            zorder=4,
        )
    if group_b_nodes:
        ax.scatter(
            [x // config.N for x in group_b_nodes],
            [x % config.N for x in group_b_nodes],
            s=42,
            c="#ff7f0e",
            label="group B visible sats",
            zorder=4,
        )

    if group_a_nodes or group_b_nodes:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def run_group_edge_betweenness(
    *,
    xml_file: Path,
    out_dir: Path,
    config: ViewerConfig = G60_CONFIG,
    group_a: int = 2,
    group_b: int = 3,
    start: int = 0,
    end: int = 86164,
    inspect_step: int = 0,
    include_intra: bool = True,
    skip_first_last_source: bool = False,
    topk_per_step: int = 25,
    plot_top_edges: int = 600,
) -> dict[str, Path]:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_graph = build_full_option_graph(
        config,
        include_intra=include_intra,
        skip_first_last_source=skip_first_last_source,
    )

    print(
        f"[setup] nodes={config.total_sats} edges={len(full_graph.idx_to_edge)} "
        f"include_intra={include_intra}",
        flush=True,
    )
    dist, next_hop = precompute_hop_and_next_hop(full_graph.graph, config.total_sats)
    print(f"[setup] shortest path table ready in {time.time() - t0:.2f}s", flush=True)

    group_data = read_snap_xml.parse_xml_group_data(xml_file, config, int(start), int(end))
    steps = sorted(int(s) for s in group_data)
    if not steps:
        raise ValueError(f"No group_data parsed from {xml_file} in [{start}, {end}]")
    print(f"[data] parsed steps={len(steps)} range={steps[0]}..{steps[-1]}", flush=True)

    total_counts = np.zeros(len(full_graph.idx_to_edge), dtype=np.uint64)
    summary_rows = []
    top_rows = []
    inspect_rows: list[dict] = []
    inspect_counts = None

    for pos, step in enumerate(steps, start=1):
        do_inspect = int(step) == int(inspect_step)
        counts, summary, rows = count_step_edge_betweenness(
            step=step,
            group_data=group_data,
            group_a=group_a,
            group_b=group_b,
            next_hop=next_hop,
            edge_to_idx=full_graph.edge_to_idx,
            total_sats=config.total_sats,
            inspect=do_inspect,
        )
        total_counts += counts.astype(np.uint64)
        summary_rows.append(summary)

        if do_inspect:
            inspect_rows = rows
            inspect_counts = counts.copy()

        if topk_per_step and topk_per_step > 0 and counts.size:
            nonzero = np.flatnonzero(counts)
            if nonzero.size:
                top = nonzero[np.argsort(counts[nonzero])[-int(topk_per_step):]][::-1]
                for idx in top:
                    u, v = full_graph.idx_to_edge[int(idx)]
                    top_rows.append(
                        {
                            "step": int(step),
                            "edge_idx": int(idx),
                            "u": int(u),
                            "v": int(v),
                            "edge_type": full_graph.edge_options.get((int(u), int(v)), "unknown"),
                            "count": int(counts[int(idx)]),
                        }
                    )

        if pos % 1000 == 0 or pos == len(steps):
            print(
                f"[run] {pos}/{len(steps)} step={step} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    summary_csv = out_dir / "step_summary_group2_group3.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    total_csv = out_dir / "edge_betweenness_total_group2_group3.csv"
    total_df = edge_counts_to_frame(
        total_counts,
        full_graph.idx_to_edge,
        full_graph.edge_options,
        config,
        count_col="total_count",
    ).sort_values("total_count", ascending=False)
    total_df.to_csv(total_csv, index=False, encoding="utf-8-sig")

    top_csv = out_dir / "edge_betweenness_step_topk_group2_group3.csv"
    pd.DataFrame(top_rows).to_csv(top_csv, index=False, encoding="utf-8-sig")

    inspect_step_actual = int(inspect_step)
    if inspect_rows:
        nodes_a = _valid_group_nodes(group_data, inspect_step_actual, group_a, config.total_sats)
        nodes_b = _valid_group_nodes(group_data, inspect_step_actual, group_b, config.total_sats)
        inspect_txt = out_dir / f"inspect_paths_step_{inspect_step_actual}_group{group_a}_group{group_b}.txt"
        write_inspect_text(
            inspect_txt,
            step=inspect_step_actual,
            group_a=group_a,
            group_b=group_b,
            nodes_a=nodes_a,
            nodes_b=nodes_b,
            rows=inspect_rows,
        )
        inspect_csv = out_dir / f"inspect_paths_step_{inspect_step_actual}_group{group_a}_group{group_b}.csv"
        pd.DataFrame(inspect_rows).to_csv(inspect_csv, index=False, encoding="utf-8-sig")
    else:
        inspect_txt = out_dir / f"inspect_paths_step_{inspect_step_actual}_group{group_a}_group{group_b}.txt"
        inspect_txt.write_text(
            f"No inspect rows for step {inspect_step_actual}. "
            f"Available range: {steps[0]}..{steps[-1]}\n",
            encoding="utf-8",
        )
        inspect_csv = out_dir / f"inspect_paths_step_{inspect_step_actual}_group{group_a}_group{group_b}.csv"
        pd.DataFrame().to_csv(inspect_csv, index=False, encoding="utf-8-sig")

    total_png = out_dir / "edge_betweenness_total_group2_group3.png"
    plot_edge_betweenness(
        total_counts,
        full_graph.idx_to_edge,
        config=config,
        out_png=total_png,
        title=f"Total edge path count: group {group_a} -> group {group_b}",
        top_edges=plot_top_edges,
    )

    step_png = out_dir / f"edge_betweenness_step_{inspect_step_actual}_group{group_a}_group{group_b}.png"
    if inspect_counts is None:
        inspect_counts = np.zeros(len(full_graph.idx_to_edge), dtype=np.uint32)
    plot_edge_betweenness(
        inspect_counts,
        full_graph.idx_to_edge,
        config=config,
        out_png=step_png,
        title=f"Step {inspect_step_actual}: group {group_a} -> group {group_b}",
        group_a_nodes=_valid_group_nodes(group_data, inspect_step_actual, group_a, config.total_sats),
        group_b_nodes=_valid_group_nodes(group_data, inspect_step_actual, group_b, config.total_sats),
        top_edges=plot_top_edges,
    )

    print(f"[done] output={out_dir} elapsed={time.time() - t0:.1f}s", flush=True)
    return {
        "summary_csv": summary_csv,
        "total_csv": total_csv,
        "top_csv": top_csv,
        "inspect_txt": inspect_txt,
        "inspect_csv": inspect_csv,
        "total_png": total_png,
        "step_png": step_png,
    }


def parse_args() -> argparse.Namespace:
    default_xml = (
        PROJECT_ROOT
        / "data"
        / "basic_file"
        / "satellitesposition"
        / "station_visible_satellites_20250106.xml"
    )
    default_out = PROJECT_ROOT / "data" / "postprocess" / "group2_group3_edge_betweenness"

    p = argparse.ArgumentParser(
        description="Compute group-to-group shortest-path edge counts on the full option topology."
    )
    p.add_argument("--xml-file", type=Path, default=default_xml)
    p.add_argument("--out-dir", type=Path, default=default_out)
    p.add_argument("--group-a", type=int, default=2)
    p.add_argument("--group-b", type=int, default=3)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=86164, help="Inclusive end step.")
    p.add_argument("--inspect-step", type=int, default=0)
    p.add_argument("--no-intra", action="store_true", help="Do not include same-plane ring links.")
    p.add_argument("--skip-first-last-source", action="store_true")
    p.add_argument("--topk-per-step", type=int, default=25)
    p.add_argument("--plot-top-edges", type=int, default=600)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_group_edge_betweenness(
        xml_file=args.xml_file,
        out_dir=args.out_dir,
        group_a=args.group_a,
        group_b=args.group_b,
        start=args.start,
        end=args.end,
        inspect_step=args.inspect_step,
        include_intra=not args.no_intra,
        skip_first_last_source=args.skip_first_last_source,
        topk_per_step=args.topk_per_step,
        plot_top_edges=args.plot_top_edges,
    )


if __name__ == "__main__":
    main()
