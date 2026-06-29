from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


GENERIC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
THIS_DIR = Path(__file__).resolve().parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from g60_paper2_config import (
    DATA_ROOT,
    DEFAULT_GROUP_CACHE_DIR,
    DEFAULT_GW_GROUP_CACHE_DIR,
    DEFAULT_GW_MAPPING_DIR,
    DEFAULT_GW_XML,
    DEFAULT_MAPPING_DIR,
    DEFAULT_XML,
    build_paper2_g60_config,
    build_paper2_gw_config,
    write_station_group_mapping,
)
from plot_g60_china_europe_shortest_metrics import group_key, group_label, normalize_topology_name, topology_plot_label
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.satellite_topology_viewer.module.topology_edges import build_full_option_plus_intra_edges


DEFAULT_OUT_ROOT = DATA_ROOT / "outputs" / "paper2_static_hop_metrics"


@dataclass(frozen=True)
class ConstellationBundle:
    name: str
    xml_file: Path
    group_cache_dir: Path
    mapping_dir: Path
    config: Any


@dataclass(frozen=True)
class PairSpec:
    key: str
    label: str
    source_group_id: int
    target_group_id: int
    source_group_key: str
    target_group_key: str


def get_constellation_bundle(name: str) -> ConstellationBundle:
    normalized = str(name).strip().lower()
    if normalized in {"g60", "g60_paper2"}:
        return ConstellationBundle(
            name="G60",
            xml_file=DEFAULT_XML,
            group_cache_dir=DEFAULT_GROUP_CACHE_DIR,
            mapping_dir=DEFAULT_MAPPING_DIR,
            config=build_paper2_g60_config(),
        )
    if normalized in {"gw", "gw_paper2"}:
        return ConstellationBundle(
            name="GW",
            xml_file=DEFAULT_GW_XML,
            group_cache_dir=DEFAULT_GW_GROUP_CACHE_DIR,
            mapping_dir=DEFAULT_GW_MAPPING_DIR,
            config=build_paper2_gw_config(),
        )
    raise ValueError(f"Unsupported constellation={name!r}. Configured: G60, GW")


def normalize_group_token(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def group_lookup(config) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for gid, info in config.station_groups.items():
        gid = int(gid)
        for token in (
            str(gid),
            normalize_group_token(str(info.get("key", ""))),
            normalize_group_token(str(info.get("name", ""))),
            normalize_group_token(f"group_{gid}"),
        ):
            if token:
                lookup[token] = gid
    return lookup


def split_pair_token(token: str) -> tuple[str, str]:
    for sep in (":", ",", "->"):
        if sep in token:
            left, right = token.split(sep, 1)
            return left.strip(), right.strip()
    raise ValueError(f"Pair {token!r} must look like china:europe or 0:1")


def parse_pair(token: str, config) -> PairSpec:
    lookup = group_lookup(config)
    left_raw, right_raw = split_pair_token(token)
    left = normalize_group_token(left_raw)
    right = normalize_group_token(right_raw)
    if left not in lookup:
        raise ValueError(f"Unknown source group {left_raw!r}. Available: {sorted(lookup)}")
    if right not in lookup:
        raise ValueError(f"Unknown target group {right_raw!r}. Available: {sorted(lookup)}")
    source_group_id = lookup[left]
    target_group_id = lookup[right]
    src_key = group_key(config, source_group_id)
    dst_key = group_key(config, target_group_id)
    return PairSpec(
        key=f"{src_key}_{dst_key}",
        label=f"{group_label(config, source_group_id)}-{group_label(config, target_group_id)}",
        source_group_id=source_group_id,
        target_group_id=target_group_id,
        source_group_key=src_key,
        target_group_key=dst_key,
    )


def build_topology_edge_table(config, topology: str) -> EdgeTable:
    topology = normalize_topology_name(topology)
    if topology == "plus_grid":
        return build_full_option_plus_intra_edges(config, inter_options=(0,), include_intra=True)
    if topology == "full_option_plus_intra":
        return build_full_option_plus_intra_edges(config)
    raise ValueError(f"Unsupported topology={topology!r}; use plus_grid or full_option_plus_intra")


def compute_all_pairs_hop_distance(edge_table: EdgeTable, total_nodes: int) -> np.ndarray:
    src = np.asarray(edge_table.src, dtype=np.int32)
    dst = np.asarray(edge_table.dst, dtype=np.int32)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.size, dtype=np.float32)
    graph = csr_matrix((data, (rows, cols)), shape=(int(total_nodes), int(total_nodes)))
    dist = shortest_path(graph, directed=False, unweighted=True)
    if not np.all(np.isfinite(dist)):
        raise ValueError("Topology is disconnected; some hop distances are infinite.")
    return np.asarray(dist, dtype=np.float32)


def group_nodes(group_data: dict, step: int, group_id: int) -> tuple[int, ...]:
    payload = group_data.get(int(step), {}) if group_data else {}
    groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
    nodes = groups.get(int(group_id), set()) or set()
    return tuple(sorted(int(x) for x in nodes))


def finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def summarize_values(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
    if finite.size == 0:
        return {
            "reachable_pairs": 0,
            "mean_shortest_hops": None,
            "min_shortest_hops": None,
            "p90_shortest_hops": None,
            "max_shortest_hops": None,
        }
    return {
        "reachable_pairs": int(finite.size),
        "mean_shortest_hops": float(np.mean(finite)),
        "min_shortest_hops": float(np.min(finite)),
        "p90_shortest_hops": float(np.percentile(finite, 90.0)),
        "max_shortest_hops": float(np.max(finite)),
    }


def compute_pair_timeseries(
    *,
    pair: PairSpec,
    steps: np.ndarray,
    group_data: dict,
    hop_dist: np.ndarray,
    active_edges: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        source_nodes = group_nodes(group_data, int(step), pair.source_group_id)
        target_nodes = group_nodes(group_data, int(step), pair.target_group_id)
        if source_nodes and target_nodes:
            values = hop_dist[np.ix_(np.asarray(source_nodes, dtype=np.int32), np.asarray(target_nodes, dtype=np.int32))]
            summary = summarize_values(values.reshape(-1))
        else:
            summary = summarize_values(np.asarray([], dtype=np.float32))
        rows.append(
            {
                "pair_key": pair.key,
                "pair_label": pair.label,
                "source_group_id": int(pair.source_group_id),
                "target_group_id": int(pair.target_group_id),
                "source_group_key": pair.source_group_key,
                "target_group_key": pair.target_group_key,
                "step": int(step),
                "hour": float(int(step) / 3600.0),
                "source_nodes": int(len(source_nodes)),
                "target_nodes": int(len(target_nodes)),
                "active_edges": int(active_edges),
                **summary,
            }
        )
        if (idx + 1) % 10000 == 0 or idx + 1 == len(steps):
            print(f"[paper2-static-hop] {pair.key} {idx + 1}/{len(steps)}", flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_metric(*, pair: PairSpec, rows: list[dict[str, Any]], metric: str, ylabel: str, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([float(row["hour"]) for row in rows], dtype=np.float32)
    y = np.asarray(
        [np.nan if row.get(metric) is None else float(row[metric]) for row in rows],
        dtype=np.float32,
    )
    finite = y[np.isfinite(y)]
    mean = float(np.mean(finite)) if finite.size else float("nan")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16.2, 6.8), dpi=180)
    ax.plot(x, y, linewidth=1.2, label=f"{pair.label} | mean={mean:.3f}")
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def summarize_pair_rows(pair: PairSpec, rows: list[dict[str, Any]], *, mean_plot: Path, p90_plot: Path, csv_path: Path) -> dict[str, Any]:
    def col_stats(name: str) -> tuple[float | None, float | None, float | None]:
        values = np.asarray(
            [np.nan if row.get(name) is None else float(row[name]) for row in rows],
            dtype=np.float64,
        )
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None, None, None
        return float(np.mean(finite)), float(np.min(finite)), float(np.max(finite))

    mean_hops, min_mean_hops, max_mean_hops = col_stats("mean_shortest_hops")
    mean_p90, min_p90, max_p90 = col_stats("p90_shortest_hops")
    return {
        "pair_key": pair.key,
        "pair_label": pair.label,
        "source_group_id": pair.source_group_id,
        "target_group_id": pair.target_group_id,
        "source_group_key": pair.source_group_key,
        "target_group_key": pair.target_group_key,
        "steps": len(rows),
        "start": int(rows[0]["step"]) if rows else None,
        "end": int(rows[-1]["step"]) if rows else None,
        "mean_hops": mean_hops,
        "min_mean_hops": min_mean_hops,
        "max_mean_hops": max_mean_hops,
        "mean_p90_hops": mean_p90,
        "min_p90_hops": min_p90,
        "max_p90_hops": max_p90,
        "timeseries_csv": str(csv_path),
        "mean_hop_plot": str(mean_plot),
        "p90_hop_plot": str(p90_plot),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute static-topology group-pair hop metrics for Paper2.")
    parser.add_argument("--constellation", type=str, default="GW")
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86400)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--topology", type=str, default="plus_grid")
    parser.add_argument("--xml-file", type=Path, default=None)
    parser.add_argument("--group-cache-dir", type=Path, default=None)
    parser.add_argument("--mapping-dir", type=Path, default=None)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.stride) <= 0:
        raise ValueError("--stride must be positive")
    if int(args.end) < int(args.start):
        raise ValueError("--end must be >= --start")

    bundle = get_constellation_bundle(args.constellation)
    config = bundle.config
    xml_file = Path(args.xml_file) if args.xml_file is not None else bundle.xml_file
    group_cache_dir = Path(args.group_cache_dir) if args.group_cache_dir is not None else bundle.group_cache_dir
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir is not None else bundle.mapping_dir
    topology = normalize_topology_name(args.topology)
    topology_label = topology_plot_label(topology)
    pairs = [parse_pair(token, config) for token in args.pairs]
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    run_key = f"t{int(args.start)}_{int(args.end)}_stride{int(args.stride)}"
    run_dir = Path(args.out_root) / bundle.name / topology / run_key
    inputs_dir = run_dir / "_inputs"

    write_station_group_mapping(out_dir=mapping_dir, config=config)
    group_data = load_or_build_group_data(
        xml_file=xml_file,
        group_cache_dir=group_cache_dir,
        steps=[int(x) for x in steps],
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    edge_table = build_topology_edge_table(config, topology)
    inputs_dir.mkdir(parents=True, exist_ok=True)
    write_edges_csv(edge_table, inputs_dir / "edges.csv")
    np.save(inputs_dir / "time_indices.npy", steps.astype(np.int64))
    print(
        f"[paper2-static-hop] constellation={bundle.name} topology={topology_label} "
        f"nodes={config.total_sats} edges={edge_table.num_edges} steps={len(steps)}",
        flush=True,
    )
    hop_dist = compute_all_pairs_hop_distance(edge_table, config.total_sats)
    np.save(inputs_dir / "all_pairs_hop_distance.npy", hop_dist.astype(np.float32))

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for pair in pairs:
        print(f"[paper2-static-hop] pair={pair.key} label={pair.label}", flush=True)
        rows = compute_pair_timeseries(
            pair=pair,
            steps=steps,
            group_data=group_data,
            hop_dist=hop_dist,
            active_edges=edge_table.num_edges,
        )
        pair_dir = run_dir / "pairs" / pair.key
        pair_csv = pair_dir / "timeseries.csv"
        write_csv(pair_csv, rows)
        mean_plot = run_dir / "figures" / pair.key / f"{pair.key}_mean_shortest_hops.png"
        p90_plot = run_dir / "figures" / pair.key / f"{pair.key}_p90_shortest_hops.png"
        plot_metric(
            pair=pair,
            rows=rows,
            metric="mean_shortest_hops",
            ylabel="mean shortest hops",
            title=f"Paper2 {bundle.name} {pair.label} mean shortest hops ({topology_label})",
            out_path=mean_plot,
        )
        plot_metric(
            pair=pair,
            rows=rows,
            metric="p90_shortest_hops",
            ylabel="p90 shortest hops",
            title=f"Paper2 {bundle.name} {pair.label} p90 shortest hops ({topology_label})",
            out_path=p90_plot,
        )
        write_json(
            pair_dir / "summary.json",
            {"summary": summarize_pair_rows(pair, rows, mean_plot=mean_plot, p90_plot=p90_plot, csv_path=pair_csv)},
        )
        summary_rows.append(summarize_pair_rows(pair, rows, mean_plot=mean_plot, p90_plot=p90_plot, csv_path=pair_csv))
        all_rows.extend(rows)

    write_csv(run_dir / "all_pairs_timeseries.csv", all_rows)
    write_csv(run_dir / "pair_summary.csv", summary_rows)
    write_json(
        run_dir / "run_meta.json",
        {
            "constellation": bundle.name,
            "topology": topology,
            "topology_label": topology_label,
            "P": int(config.P),
            "N": int(config.N),
            "total_sats": int(config.total_sats),
            "start": int(args.start),
            "end": int(args.end),
            "stride": int(args.stride),
            "pairs": [pair.__dict__ for pair in pairs],
            "xml_file": str(xml_file),
            "group_cache_dir": str(group_cache_dir),
            "edges_csv": str(inputs_dir / "edges.csv"),
            "all_pairs_hop_distance": str(inputs_dir / "all_pairs_hop_distance.npy"),
            "all_pairs_timeseries_csv": str(run_dir / "all_pairs_timeseries.csv"),
            "pair_summary_csv": str(run_dir / "pair_summary.csv"),
            "note": "Hop-only metrics. Delay requires constellation-specific satellite position files.",
        },
    )
    print(f"[paper2-static-hop] run_dir={run_dir}", flush=True)
    print(f"[paper2-static-hop] all_pairs_csv={run_dir / 'all_pairs_timeseries.csv'}", flush=True)
    print(f"[paper2-static-hop] summary_csv={run_dir / 'pair_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
