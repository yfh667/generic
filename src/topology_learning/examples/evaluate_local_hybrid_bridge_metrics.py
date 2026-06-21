from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.position_cache import open_position_cache_for_interval
from src.link_delay.module.query import open_delay_store_for_interval
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module.batch_shortest_hops import RegionPairSpec, TopologySpec
from src.topology_workflow.module.edge_tables import build_motif_text_edge_table
from src.topology_workflow.module.hybrid_edges import build_y_band_hybrid_edge_table
from src.topology_workflow.module.shortest_delay import compute_shortest_delay_timeseries
from src.topology_workflow.module.shortest_hops import compute_shortest_hops_timeseries


DEFAULT_CANDIDATE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\local_hybrid_bridge_candidates_466_to_621_b489_quick"
)
DEFAULT_MOTIF_CSV = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w4_h3"
    r"\shortest_delay_t0_86160_stride60\topology_library.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\topology_learning"
    r"\local_hybrid_bridge_metrics_466_to_621_b489_quick"
)
DEFAULT_GROUP_XML = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE_DIR = Path(
    r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache"
)
DEFAULT_DELAY_STORE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\full_option_edge_delay"
    r"\G60_full_options_t0_86164_stride1"
)
DEFAULT_POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s"
)


PAIR_SPECS = (
    RegionPairSpec("china_europe", "China-Europe", 2, 3),
    RegionPairSpec("china_africa", "China-Africa", 2, 1),
    RegionPairSpec("china_america", "China-America", 2, 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local hybrid bridge candidates on hop and delay metrics.")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--candidate-csv", type=Path, default=None)
    parser.add_argument("--motif-csv", type=Path, default=DEFAULT_MOTIF_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-step", type=int, default=57540)
    parser.add_argument("--end-step", type=int, default=61440)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-base", default=None)
    parser.add_argument("--candidate-patch", default=None)
    parser.add_argument("--candidate-max-total-setup", type=int, default=None)
    parser.add_argument("--candidate-max-entry-setup", type=int, default=None)
    parser.add_argument("--candidate-max-exit-setup", type=int, default=None)
    parser.add_argument("--candidate-min-overlap-with-base", type=float, default=None)
    parser.add_argument("--include-topologies", nargs="*", default=("w4h3_motif_000558", "w4h3_motif_000680", "w4h3_motif_000622"))
    parser.add_argument("--group-xml", type=Path, default=DEFAULT_GROUP_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE_DIR)
    parser.add_argument("--delay-store-dir", type=Path, default=DEFAULT_DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=DEFAULT_POSITION_CACHE_DIR)
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def _safe_token(value: str) -> str:
    out = []
    for ch in str(value):
        out.append(ch if ch.isalnum() or ch in ("_", "-") else "_")
    return "".join(out).strip("_")


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else math.nan


def _read_motif_rows(path: Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return {str(row["name"]): dict(row) for row in csv.DictReader(f) if row.get("name")}


def _build_static_specs(names: Iterable[str], motif_rows: dict[str, dict[str, str]]) -> dict[str, TopologySpec]:
    out: dict[str, TopologySpec] = {}
    for name in names:
        name = str(name)
        if name in out:
            continue
        row = motif_rows[name]
        out[name] = TopologySpec(
            name=name,
            edge_table=build_motif_text_edge_table(
                motif_text=str(row["motif"]),
                config=G60_CONFIG,
                add_intra_ring=True,
                wrap_planes=False,
            ),
            library=str(row.get("library", "motif")),
            motif_id=int(row.get("motif_id") or 0),
            motif=str(row["motif"]),
            source_w=int(row["source_w"]) if row.get("source_w") else None,
            source_h=int(row["source_h"]) if row.get("source_h") else None,
            edge_count_local=int(row["edge_count"]) if row.get("edge_count") else None,
            support=str(row.get("support", "")),
            baseline=False,
            meta={k: v for k, v in row.items() if k != "motif"},
        )
    return out


def _candidate_specs(
    *,
    candidate_dir: Path,
    candidate_csv: Path | None,
    motif_rows: dict[str, dict[str, str]],
    top_k: int,
    include_topologies: Iterable[str],
    candidate_base: str | None,
    candidate_patch: str | None,
    candidate_max_total_setup: int | None,
    candidate_max_entry_setup: int | None,
    candidate_max_exit_setup: int | None,
    candidate_min_overlap_with_base: float | None,
) -> list[TopologySpec]:
    if candidate_csv is None:
        top = json.loads((Path(candidate_dir) / "top_candidates.json").read_text(encoding="utf-8"))
        candidates = list(top)
    else:
        with Path(candidate_csv).open("r", encoding="utf-8-sig", newline="") as f:
            candidates = list(csv.DictReader(f))
        for item in candidates:
            for key in ("band_start", "band_end", "band_width", "entry_setup", "exit_setup", "total_bridge_setup"):
                if key in item and item[key] not in ("", None):
                    item[key] = int(item[key])
            for key in ("overlap_with_base", "overlap_with_patch", "overlap_with_previous", "overlap_with_next"):
                if key in item and item[key] not in ("", None):
                    item[key] = float(item[key])
    if candidate_base is not None:
        candidates = [item for item in candidates if str(item.get("base")) == str(candidate_base)]
    if candidate_patch is not None:
        candidates = [item for item in candidates if str(item.get("patch")) == str(candidate_patch)]
    if candidate_max_total_setup is not None:
        candidates = [item for item in candidates if int(item.get("total_bridge_setup", 10**9)) <= int(candidate_max_total_setup)]
    if candidate_max_entry_setup is not None:
        candidates = [item for item in candidates if int(item.get("entry_setup", 10**9)) <= int(candidate_max_entry_setup)]
    if candidate_max_exit_setup is not None:
        candidates = [item for item in candidates if int(item.get("exit_setup", 10**9)) <= int(candidate_max_exit_setup)]
    if candidate_min_overlap_with_base is not None:
        candidates = [
            item
            for item in candidates
            if float(item.get("overlap_with_base", 0.0)) >= float(candidate_min_overlap_with_base)
        ]
    candidates.sort(
        key=lambda item: (
            int(item.get("total_bridge_setup", 10**9)),
            -float(item.get("overlap_with_base", 0.0)),
            str(item.get("candidate", "")),
        )
    )
    chosen = list(candidates[: int(top_k)])
    needed = {str(name) for name in include_topologies}
    for item in chosen:
        needed.add(str(item["base"]))
        needed.add(str(item["patch"]))
    static_specs = _build_static_specs(sorted(needed), motif_rows)
    specs: list[TopologySpec] = [static_specs[str(name)] for name in include_topologies if str(name) in static_specs]
    for rank, item in enumerate(chosen, start=1):
        base = static_specs[str(item["base"])]
        patch = static_specs[str(item["patch"])]
        rows = [int(x) for x in str(item["band_rows"]).split()]
        result = build_y_band_hybrid_edge_table(
            base_edge_table=base.edge_table,
            patch_edge_table=patch.edge_table,
            p=int(G60_CONFIG.P),
            n=int(G60_CONFIG.N),
            rows=rows,
            patch_options=None,
            fail_on_degree_violation=True,
        )
        specs.append(
            TopologySpec(
                name=f"hybrid_top{rank:02d}_{_safe_token(str(item['candidate']))}",
                edge_table=result.edge_table,
                library="hybrid_y_band",
                motif=f"{base.name}+patch:{patch.name}",
                baseline=False,
                meta=dict(item),
            )
        )
    return specs


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start_step), int(args.end_step) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty steps")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motif_rows = _read_motif_rows(args.motif_csv)
    specs = _candidate_specs(
        candidate_dir=args.candidate_dir,
        candidate_csv=args.candidate_csv,
        motif_rows=motif_rows,
        top_k=int(args.top_k),
        include_topologies=args.include_topologies,
        candidate_base=args.candidate_base,
        candidate_patch=args.candidate_patch,
        candidate_max_total_setup=args.candidate_max_total_setup,
        candidate_max_entry_setup=args.candidate_max_entry_setup,
        candidate_max_exit_setup=args.candidate_max_exit_setup,
        candidate_min_overlap_with_base=args.candidate_min_overlap_with_base,
    )
    group_data = load_or_build_group_data(
        xml_file=args.group_xml,
        group_cache_dir=args.group_cache_dir,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=False,
    )
    delay_store = open_delay_store_for_interval(
        int(args.start_step),
        int(args.end_step),
        stride=int(args.stride),
        store_dir=args.delay_store_dir,
        constellation_name=G60_CONFIG.name,
    )
    delay_rows = delay_store.rows_for_interval(int(args.start_step), int(args.end_step), int(args.stride))
    position_store = open_position_cache_for_interval(
        int(args.start_step),
        int(args.end_step),
        stride=int(args.stride),
        cache_dir=args.position_cache_dir,
    )
    position_rows = position_store.rows_for_interval(int(args.start_step), int(args.end_step), int(args.stride))

    summary_rows: list[dict] = []
    step_rows: list[dict] = []
    for spec in specs:
        print(f"[hybrid-metrics] topology={spec.name} edges={spec.edge_table.num_edges}", flush=True)
        for pair in PAIR_SPECS:
            pair_dir = out_dir / str(pair.key) / _safe_token(spec.name)
            hops_path = pair_dir / "hops" / "mean_shortest_hops.npy"
            delay_path = pair_dir / "delay" / "mean_shortest_delay_ms.npy"
            if bool(args.force) or not hops_path.exists():
                hops = compute_shortest_hops_timeseries(
                    topology_name=spec.name,
                    edge_table=spec.edge_table,
                    config=G60_CONFIG,
                    group_data=group_data,
                    steps=steps,
                    source_group_id=int(pair.source_group_id),
                    target_group_id=int(pair.target_group_id),
                    out_dir=pair_dir / "hops",
                    sample_steps=0,
                    sample_pairs_per_step=0,
                )
            else:
                hops = np.load(hops_path)
            if bool(args.force) or not delay_path.exists():
                delay = compute_shortest_delay_timeseries(
                    topology_name=spec.name,
                    edge_table=spec.edge_table,
                    config=G60_CONFIG,
                    group_data=group_data,
                    steps=steps,
                    delay_rows=delay_rows,
                    position_rows=position_rows,
                    delay_store=delay_store,
                    position_store=position_store,
                    source_group_id=int(pair.source_group_id),
                    target_group_id=int(pair.target_group_id),
                    out_dir=pair_dir / "delay",
                    engine=str(args.engine),
                    sample_steps=0,
                    sample_pairs_per_step=0,
                    progress_every=int(args.progress_every),
                )
            else:
                delay = np.load(delay_path)

            summary_rows.append(
                {
                    "topology": spec.name,
                    "pair": pair.key,
                    "mean_hops": _finite_mean(hops),
                    "mean_delay_ms": _finite_mean(delay),
                    "window_start_step": int(steps[0]),
                    "window_end_step": int(steps[-1]),
                    "num_steps": int(len(steps)),
                    "candidate_meta": json.dumps(spec.meta, ensure_ascii=False),
                }
            )
            for idx, step in enumerate(steps):
                step_rows.append(
                    {
                        "step": int(step),
                        "topology": spec.name,
                        "pair": pair.key,
                        "mean_hops": float(hops[idx]),
                        "mean_delay_ms": float(delay[idx]),
                    }
                )

    with (out_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (out_dir / "by_step.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
        writer.writeheader()
        writer.writerows(step_rows)

    meta = {
        "candidate_dir": str(args.candidate_dir),
        "candidate_csv": None if args.candidate_csv is None else str(args.candidate_csv),
        "motif_csv": str(args.motif_csv),
        "start_step": int(args.start_step),
        "end_step": int(args.end_step),
        "stride": int(args.stride),
        "topologies": [spec.name for spec in specs],
        "pairs": [pair.key for pair in PAIR_SPECS],
        "outputs": {
            "summary": str(out_dir / "summary.csv"),
            "by_step": str(out_dir / "by_step.csv"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
