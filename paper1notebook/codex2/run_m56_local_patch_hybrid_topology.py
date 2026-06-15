from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PyQt5 import QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from weighted_base_viewer import EdgeUsageTopology2DViewer

from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module.edge_betweenness import (
    build_undirected_adjacency,
    edge_betweenness_between_node_sets,
)
from src.topology_metrics.module.group_states import group_nodes_for_step
from src.topology_workflow.module.batch_shortest_hops import TopologySpec, topology_specs_from_motif_csv
from src.topology_workflow.module.config import load_workflow_yaml, time_axis_from_config, viewer_config_from_workflow
from src.topology_workflow.module.edge_tables import make_edge_table_from_records


DEFAULT_CONFIG = GENERIC_ROOT / "paper1notebook" / "pipeline" / "configs" / "g60_w_le4_h_le3_shortest_hops.yaml"


@dataclass(frozen=True)
class RegionPair:
    key: str
    label: str
    source_group_id: int
    target_group_id: int


@dataclass(frozen=True)
class EdgeRecord:
    src_plane: int
    src_y: int
    dst_plane: int
    dst_y: int
    option: int

    @property
    def tuple(self) -> tuple[int, int, int, int, int]:
        return (self.src_plane, self.src_y, self.dst_plane, self.dst_y, self.option)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local hybrid topology: base motif 000056 plus a local y-band patch "
            "from motif 000040, with out-degree/in-degree conflict resolution."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-motif", default="56")
    parser.add_argument("--patch-motif", default="40")
    parser.add_argument("--band-start", type=int, default=12)
    parser.add_argument("--band-end", type=int, default=17)
    parser.add_argument(
        "--patch-mode",
        choices=("c", "cb", "all"),
        default="c",
        help=(
            "c: use only option C from patch motif; "
            "cb: use option C and B from patch motif; "
            "all: use all patch motif inter links in the y-band."
        ),
    )
    parser.add_argument("--start", type=int, default=29340)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--metric-pairs",
        nargs="+",
        default=("china_europe", "china_america", "china_africa"),
        help="Region pairs to compare in metrics.csv.",
    )
    parser.add_argument(
        "--usage-pairs",
        nargs="+",
        default=("china_america",),
        help="Region pairs whose edge usage is drawn as red overlay in the viewer.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=190)
    parser.add_argument("--show-panel-controls", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def path_from(raw: dict[str, Any], key: str) -> Path:
    value = raw.get(key)
    if value in (None, ""):
        raise ValueError(f"missing paths.{key} in workflow config")
    return Path(str(value))


def normalize_motif_name(raw: str, *, name_prefix: str) -> str:
    token = str(raw).strip()
    if token.isdigit():
        return f"{name_prefix}_motif_{int(token):06d}"
    if token.startswith("motif_") and token.removeprefix("motif_").isdigit():
        return f"{name_prefix}_{token}"
    if token.startswith(f"{name_prefix}_motif_"):
        return token
    if token.startswith("combined_motif_"):
        return token
    raise ValueError(f"unsupported motif token {raw!r}")


def region_pairs_from_workflow(raw: dict[str, Any]) -> dict[str, RegionPair]:
    out: dict[str, RegionPair] = {}
    for item in raw.get("region_pairs", []) or []:
        pair = RegionPair(
            key=str(item["key"]),
            label=str(item.get("label", item["key"])),
            source_group_id=int(item["source_group_id"]),
            target_group_id=int(item["target_group_id"]),
        )
        out[pair.key] = pair
    return out


def cyclic_band(start: int, end: int, *, n: int) -> tuple[int, ...]:
    start = int(start) % int(n)
    end = int(end) % int(n)
    if start <= end:
        return tuple(range(start, end + 1))
    return tuple(list(range(start, int(n))) + list(range(0, end + 1)))


def node_id(plane: int, y: int, *, n: int) -> int:
    return int(plane) * int(n) + int(y)


def edge_records_from_table(edge_table: EdgeTable) -> list[EdgeRecord]:
    return [
        EdgeRecord(
            src_plane=int(edge_table.src_plane[idx]),
            src_y=int(edge_table.src_y[idx]),
            dst_plane=int(edge_table.dst_plane[idx]),
            dst_y=int(edge_table.dst_y[idx]),
            option=int(edge_table.option[idx]),
        )
        for idx in range(edge_table.num_edges)
    ]


def patch_option_allowed(option: int, mode: str) -> bool:
    option = int(option)
    if mode == "c":
        return option == 4
    if mode == "cb":
        return option in (4, 1)
    if mode == "all":
        return option != -1
    raise ValueError(f"unknown patch mode: {mode}")


def build_hybrid_edge_table(
    *,
    base_spec: TopologySpec,
    patch_spec: TopologySpec,
    p: int,
    n: int,
    band: Iterable[int],
    patch_mode: str,
) -> tuple[EdgeTable, list[EdgeRecord], list[EdgeRecord], dict[str, Any]]:
    band_set = {int(y) % int(n) for y in band}
    base_records = edge_records_from_table(base_spec.edge_table)
    patch_records_all = edge_records_from_table(patch_spec.edge_table)
    patch_records = [
        record
        for record in patch_records_all
        if record.option != -1 and record.src_y in band_set and patch_option_allowed(record.option, patch_mode)
    ]
    patch_sources = {node_id(r.src_plane, r.src_y, n=n) for r in patch_records}
    patch_targets = {node_id(r.dst_plane, r.dst_y, n=n) for r in patch_records}

    kept: list[EdgeRecord] = []
    removed: list[EdgeRecord] = []
    for record in base_records:
        if record.option == -1:
            kept.append(record)
            continue
        src = node_id(record.src_plane, record.src_y, n=n)
        dst = node_id(record.dst_plane, record.dst_y, n=n)
        if src in patch_sources or dst in patch_targets:
            removed.append(record)
            continue
        kept.append(record)

    hybrid_records = kept + patch_records
    edge_table = make_edge_table_from_records(
        p=int(p),
        n=int(n),
        records=[record.tuple for record in hybrid_records],
    )
    stats = degree_stats(edge_table, total_sats=int(p) * int(n))
    return edge_table, patch_records, removed, stats


def degree_stats(edge_table: EdgeTable, *, total_sats: int) -> dict[str, Any]:
    out_degree: Counter[int] = Counter()
    in_degree: Counter[int] = Counter()
    for idx in range(edge_table.num_edges):
        if int(edge_table.option[idx]) == -1:
            continue
        out_degree[int(edge_table.src[idx])] += 1
        in_degree[int(edge_table.dst[idx])] += 1
    return {
        "total_edges": int(edge_table.num_edges),
        "inter_edges": int(np.count_nonzero(np.asarray(edge_table.option) != -1)),
        "intra_edges": int(np.count_nonzero(np.asarray(edge_table.option) == -1)),
        "max_out_degree": max(out_degree.values(), default=0),
        "max_in_degree": max(in_degree.values(), default=0),
        "out_degree_gt1_nodes": int(sum(1 for value in out_degree.values() if value > 1)),
        "in_degree_gt1_nodes": int(sum(1 for value in in_degree.values() if value > 1)),
        "zero_out_degree_nodes": int(total_sats - len(out_degree)),
        "zero_in_degree_nodes": int(total_sats - len(in_degree)),
    }


def records_to_dicts(records: list[EdgeRecord]) -> list[dict[str, int]]:
    return [
        {
            "src_plane": r.src_plane,
            "src_y": r.src_y,
            "dst_plane": r.dst_plane,
            "dst_y": r.dst_y,
            "option": r.option,
        }
        for r in records
    ]


def compute_pair_metric(
    *,
    edge_table: EdgeTable,
    config,
    group_data: dict,
    step: int,
    pair: RegionPair,
    adjacency: list[list[tuple[int, int]]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    values, summary, _samples = edge_betweenness_between_node_sets(
        edge_table,
        total_nodes=int(config.total_sats),
        source_nodes=group_nodes_for_step(group_data, int(step), int(pair.source_group_id)),
        target_nodes=group_nodes_for_step(group_data, int(step), int(pair.target_group_id)),
        adjacency=adjacency,
        sample_path_limit=0,
    )
    return values, {
        "source_nodes": int(summary.source_nodes),
        "target_nodes": int(summary.target_nodes),
        "reachable_pairs": int(summary.reachable_pairs),
        "mean_hops": float(summary.mean_shortest_distance_hops),
        "total_hops": float(summary.total_shortest_distance_hops),
        "max_edge_usage": float(summary.max_edge_betweenness),
        "nonzero_edges": int(summary.nonzero_edges),
        "edge_usage_sum": float(summary.edge_value_sum),
    }


def write_records_csv(path: Path, records: list[EdgeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src_plane", "src_y", "dst_plane", "dst_y", "option"])
        writer.writeheader()
        for row in records_to_dicts(records):
            writer.writerow(row)


def default_out_dir(
    *,
    run_out_dir: Path,
    base_name: str,
    patch_name: str,
    patch_mode: str,
    band_start: int,
    band_end: int,
    start: int,
    end: int,
    stride: int,
) -> Path:
    return (
        run_out_dir
        / "hybrid_topologies"
        / f"{base_name}_plus_{patch_name}_{patch_mode}_y{band_start}_{band_end}_t{start}_{end}_stride{stride}"
    )


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    cfg_start, _cfg_end, cfg_stride = time_axis_from_config(workflow)
    start = int(args.start if args.start is not None else cfg_start)
    end = int(args.end if args.end is not None else start)
    stride = int(args.stride if args.stride is not None else cfg_stride)
    steps = list(range(start, end + 1, stride))
    if not steps:
        raise ValueError("empty steps")

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
    if base_name not in specs_by_name or patch_name not in specs_by_name:
        raise ValueError(f"missing base or patch motif: {base_name}, {patch_name}")
    base_spec = specs_by_name[base_name]
    patch_spec = specs_by_name[patch_name]

    band = cyclic_band(args.band_start, args.band_end, n=int(config.N))
    hybrid_table, added_records, removed_records, hybrid_degree = build_hybrid_edge_table(
        base_spec=base_spec,
        patch_spec=patch_spec,
        p=int(config.P),
        n=int(config.N),
        band=band,
        patch_mode=str(args.patch_mode),
    )
    if hybrid_degree["max_out_degree"] > 1 or hybrid_degree["max_in_degree"] > 1:
        raise RuntimeError(f"hybrid topology violates degree constraints: {hybrid_degree}")

    out_dir = args.out_dir or default_out_dir(
        run_out_dir=path_from(paths_raw, "out_dir"),
        base_name=base_name,
        patch_name=patch_name,
        patch_mode=str(args.patch_mode),
        band_start=int(args.band_start),
        band_end=int(args.band_end),
        start=start,
        end=end,
        stride=stride,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    group_data = load_or_build_group_data(
        xml_file=path_from(paths_raw, "group_xml"),
        group_cache_dir=path_from(paths_raw, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=stride,
        enabled=True,
        force=bool(args.force_group_cache),
    )
    pair_by_key = region_pairs_from_workflow(workflow)
    metric_pairs = [pair_by_key[key] for key in args.metric_pairs]
    usage_pairs = [pair_by_key[key] for key in args.usage_pairs]

    write_edges_csv(hybrid_table, out_dir / "hybrid_edges.csv")
    write_records_csv(out_dir / "added_patch_records.csv", added_records)
    write_records_csv(out_dir / "removed_base_records.csv", removed_records)

    base_degree = degree_stats(base_spec.edge_table, total_sats=int(config.total_sats))
    patch_degree = degree_stats(patch_spec.edge_table, total_sats=int(config.total_sats))
    meta = {
        "base_topology": base_name,
        "base_motif": base_spec.motif,
        "patch_topology": patch_name,
        "patch_motif": patch_spec.motif,
        "patch_mode": str(args.patch_mode),
        "band_start": int(args.band_start),
        "band_end": int(args.band_end),
        "band_y": list(int(y) for y in band),
        "start": start,
        "end": end,
        "stride": stride,
        "steps": steps,
        "added_patch_records": len(added_records),
        "removed_base_records": len(removed_records),
        "base_degree": base_degree,
        "patch_degree": patch_degree,
        "hybrid_degree": hybrid_degree,
        "outputs": {
            "hybrid_edges": str(out_dir / "hybrid_edges.csv"),
            "added_patch_records": str(out_dir / "added_patch_records.csv"),
            "removed_base_records": str(out_dir / "removed_base_records.csv"),
            "metrics": str(out_dir / "metrics.csv"),
        },
    }
    (out_dir / "hybrid_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    topologies: list[tuple[str, str, EdgeTable]] = [
        ("base", base_name, base_spec.edge_table),
        ("patch", patch_name, patch_spec.edge_table),
        ("hybrid", f"{base_name}+{patch_name}:{args.patch_mode}:y{args.band_start}_{args.band_end}", hybrid_table),
    ]
    adjacency_by_topology = {
        label: build_undirected_adjacency(edge_table, int(config.total_sats))
        for label, _name, edge_table in topologies
    }

    usage_values = np.zeros((len(steps), int(hybrid_table.num_edges)), dtype=np.float32)
    metric_rows: list[dict[str, Any]] = []
    for row_idx, step in enumerate(steps):
        for topo_label, topo_name, edge_table in topologies:
            adjacency = adjacency_by_topology[topo_label]
            for pair in metric_pairs:
                values, summary = compute_pair_metric(
                    edge_table=edge_table,
                    config=config,
                    group_data=group_data,
                    step=int(step),
                    pair=pair,
                    adjacency=adjacency,
                )
                metric_rows.append(
                    {
                        "step": int(step),
                        "topology_label": topo_label,
                        "topology": topo_name,
                        "pair": pair.key,
                        "pair_label": pair.label,
                        **summary,
                    }
                )
                if topo_label == "hybrid" and pair.key in {p.key for p in usage_pairs}:
                    usage_values[row_idx, :] += values

    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "topology_label",
            "topology",
            "pair",
            "pair_label",
            "source_nodes",
            "target_nodes",
            "reachable_pairs",
            "mean_hops",
            "total_hops",
            "max_edge_usage",
            "nonzero_edges",
            "edge_usage_sum",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metric_rows:
            writer.writerow(row)
    np.save(out_dir / "hybrid_usage_values.npy", usage_values)

    print(
        f"[hybrid-topology] base={base_name} patch={patch_name} mode={args.patch_mode} "
        f"band={list(band)} steps={len(steps)} out_dir={out_dir}",
        flush=True,
    )
    print(
        f"[hybrid-topology] added={len(added_records)} removed={len(removed_records)} "
        f"degree={hybrid_degree}",
        flush=True,
    )
    for row in metric_rows:
        if int(row["step"]) != steps[0]:
            continue
        print(
            f"[hybrid-topology] step={row['step']} {row['topology_label']} {row['pair']}: "
            f"mean_hops={float(row['mean_hops']):.6f}",
            flush=True,
        )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    usage_label = "+".join(pair.key for pair in usage_pairs)
    viewer = EdgeUsageTopology2DViewer(
        config,
        steps=steps,
        edge_table=hybrid_table,
        edge_usage_values=usage_values,
        value_max=float(np.nanmax(usage_values)) if usage_values.size else 1.0,
        window_title=(
            f"Hybrid {base_name} + {patch_name} {args.patch_mode} "
            f"y{args.band_start}..{args.band_end} | usage={usage_label}"
        ),
        group_data=group_data,
        show_groups=not bool(args.no_groups),
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(85, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(220, int(args.edge_alpha)))))
    if not bool(args.show_panel_controls):
        viewer.controls_scroll.hide()
        viewer.main_splitter.setSizes([1000, 0])
    viewer.update_step(0)

    return run_viewer_widget(
        viewer,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
