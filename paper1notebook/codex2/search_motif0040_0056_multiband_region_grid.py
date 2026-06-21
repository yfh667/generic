from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PIPELINE_DIR = GENERIC_ROOT / "paper1notebook" / "pipeline"
for path in (GENERIC_ROOT, THIS_DIR, PIPELINE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_m56_local_patch_hybrid_topology import build_hybrid_edge_table, cyclic_band, degree_stats  # noqa: E402
from run_paper1_motif_shortest_hops import load_yaml, path_from, region_pair_specs, wrap_planes_from_config  # noqa: E402
from search_motif0040_0056_hybrid_region_grid import (  # noqa: E402
    CandidateSpec,
    DEFAULT_CONFIG,
    PURE_METRIC_DIR,
    edge_key_set,
    evaluate_candidates,
    evaluate_candidates_with_reuse,
    plot_full,
    read_pure_envelopes,
    score_candidates,
    write_compare_csv,
    write_selected_indices,
    write_summary_csv,
)
from src.link_delay.module.query import FullLinkDelayStore  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module.batch_shortest_hops import TopologySpec, topology_specs_from_motif_csv  # noqa: E402
from src.topology_workflow.module.config import viewer_config_from_workflow  # noqa: E402


DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif0040_0056_region_internal_plus_grid_multiband_search"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search multi-band 000040 patches on 000056 under China-Europe region-internal +grid constraints."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--target-pair", default="china_europe")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--screen-sample-stride", type=int, default=1800)
    parser.add_argument(
        "--primitive-bands",
        nargs="+",
        default=("8-13", "14-19", "15-20", "20-25", "21-26", "32-1", "33-2"),
        help="Candidate primitive y-bands, inclusive, with wrap allowed, e.g. 32-1.",
    )
    parser.add_argument("--max-subset-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--selected-indices",
        nargs="+",
        type=int,
        default=None,
        help="Skip screen evaluation and run full metrics for these candidate indices. Pure candidates are added automatically.",
    )
    parser.add_argument(
        "--reuse-full-dir",
        type=Path,
        default=None,
        help="Reuse candidate columns from an existing full_t*/ directory and only evaluate missing selected candidates.",
    )
    parser.add_argument("--hop-weight", type=float, default=1.0)
    parser.add_argument("--delay-weight", type=float, default=1.0)
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def parse_band(raw: str) -> tuple[int, int]:
    left, right = str(raw).split("-", 1)
    return int(left), int(right)


def band_token(start: int, end: int) -> str:
    return f"y{int(start):02d}_{int(end):02d}"


def build_multiband_candidates(
    *,
    config: Any,
    motif_csv: Path,
    name_prefix: str,
    primitive_bands: list[tuple[int, int]],
    max_subset_size: int,
    wrap_planes: bool,
) -> list[CandidateSpec]:
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library="selected_motif",
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=bool(wrap_planes),
    )
    by_id = {int(spec.motif_id): spec for spec in specs if spec.motif_id is not None}
    base = by_id[56]
    patch = by_id[40]

    candidates: list[CandidateSpec] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for spec, label in ((base, "pure_000056"), (patch, "pure_000040")):
        key = edge_key_set(spec)
        seen.add(key)
        candidates.append(
            CandidateSpec(
                name=label,
                topology=TopologySpec(
                    name=label,
                    edge_table=spec.edge_table,
                    library="selected_motif",
                    motif_id=spec.motif_id,
                    motif=spec.motif,
                    source_w=spec.source_w,
                    source_h=spec.source_h,
                    edge_count_local=spec.edge_count_local,
                    support=spec.support,
                    baseline=False,
                    meta={"kind": "pure"},
                ),
                mode="pure",
                band_start=None,
                band_end=None,
                band_len=0,
                added_edges=0,
                removed_edges=0,
                degree=degree_stats(spec.edge_table, total_sats=int(config.total_sats)),
            )
        )

    band_nodes: list[tuple[tuple[int, int], tuple[int, ...]]] = [
        (band, cyclic_band(band[0], band[1], n=int(config.N))) for band in primitive_bands
    ]
    for size in range(1, int(max_subset_size) + 1):
        for subset in itertools.combinations(range(len(band_nodes)), size):
            bands = [band_nodes[idx][0] for idx in subset]
            union_y = sorted({y for idx in subset for y in band_nodes[idx][1]})
            edge_table, added, removed, degree = build_hybrid_edge_table(
                base_spec=base,
                patch_spec=patch,
                p=int(config.P),
                n=int(config.N),
                band=union_y,
                patch_mode="all",
            )
            if int(degree["max_out_degree"]) > 1 or int(degree["max_in_degree"]) > 1:
                continue
            name = "hybrid_m56_m40_multiband_" + "__".join(band_token(a, b) for a, b in bands)
            topology = TopologySpec(
                name=name,
                edge_table=edge_table,
                library="hybrid_multiband_000056_000040",
                motif_id=None,
                motif="base=000056 patch=000040 bands=" + ",".join(f"{a}-{b}" for a, b in bands),
                baseline=False,
                meta={"kind": "multiband", "bands": bands, "patch_mode": "all"},
            )
            key = edge_key_set(topology)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                CandidateSpec(
                    name=name,
                    topology=topology,
                    mode="multiband",
                    band_start=None,
                    band_end=None,
                    band_len=len(union_y),
                    added_edges=int(len(added)),
                    removed_edges=int(len(removed)),
                    degree=degree,
                )
            )
    return candidates


def main() -> int:
    args = parse_args()
    raw = load_yaml(Path(args.config))
    config = viewer_config_from_workflow(raw)
    paths = raw.get("paths", {})
    motif_library = raw.get("motif_library", {})
    motif_csv = Path(str(motif_library.get("csv_name", "motif0040_0056.csv")))
    if not motif_csv.is_absolute():
        motif_csv = path_from(paths, "motif_library_dir") / motif_csv
    wrap_planes = wrap_planes_from_config(raw)
    forced_option = int(raw.get("region_internal_constraint", {}).get("forced_option", 0))
    pair_spec = region_pair_specs(raw, subset=[str(args.target_pair)])[0]
    steps = np.arange(int(args.start), int(args.end) + 1, int(args.stride), dtype=np.int64)
    sample_steps = np.arange(int(args.start), int(args.end) + 1, int(args.screen_sample_stride), dtype=np.int64)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    primitive_bands = [parse_band(text) for text in args.primitive_bands]

    candidates = build_multiband_candidates(
        config=config,
        motif_csv=motif_csv,
        name_prefix=str(motif_library.get("name_prefix", "selected")),
        primitive_bands=primitive_bands,
        max_subset_size=int(args.max_subset_size),
        wrap_planes=bool(wrap_planes),
    )
    print(f"[multiband-search] candidates={len(candidates)} sample_steps={len(sample_steps)} full_steps={len(steps)}")

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps.tolist(),
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )
    delay_store = FullLinkDelayStore(path_from(paths, "delay_store_dir"))

    screen_dir = out_dir / "screen"
    if args.selected_indices:
        selected_indices = sorted(set([0, 1] + [int(x) for x in args.selected_indices]))
        invalid = [idx for idx in selected_indices if idx < 0 or idx >= len(candidates)]
        if invalid:
            raise ValueError(f"selected candidate indices out of range: {invalid}; candidates={len(candidates)}")
    else:
        sample_hop, sample_delay = evaluate_candidates(
            candidates=candidates,
            config=config,
            group_data=group_data,
            pair_spec=pair_spec,
            steps=sample_steps.tolist(),
            delay_store=delay_store,
            forced_option=forced_option,
            wrap_planes=bool(wrap_planes),
            progress_label="screen",
        )
        sample_score = score_candidates(
            sample_hop,
            sample_delay,
            hop_weight=float(args.hop_weight),
            delay_weight=float(args.delay_weight),
        )
        write_summary_csv(screen_dir / "candidate_screen_summary.csv", candidates=candidates, hop=sample_hop, delay=sample_delay, score=sample_score)
        selected_indices = sorted(set([0, 1] + [int(x) for x in np.argsort(sample_score)[: int(args.top_k)].tolist()]))
    selected_candidates = [candidates[idx] for idx in selected_indices]
    write_selected_indices(screen_dir / "selected_candidate_indices.json", selected_indices=selected_indices, selected_candidates=selected_candidates)
    print(f"[multiband-search] selected={selected_indices}")
    if bool(args.screen_only):
        return 0

    full_hop, full_delay, reuse_meta = evaluate_candidates_with_reuse(
        candidates=selected_candidates,
        config=config,
        group_data=group_data,
        pair_spec=pair_spec,
        steps=steps.tolist(),
        delay_store=delay_store,
        forced_option=forced_option,
        wrap_planes=bool(wrap_planes),
        reuse_full_dir=args.reuse_full_dir,
    )
    full_score = score_candidates(full_hop, full_delay, hop_weight=float(args.hop_weight), delay_weight=float(args.delay_weight))
    oracle_idx = np.nanargmin(
        ((full_hop - np.nanmin(full_hop)) / max(1e-9, float(np.nanmax(full_hop) - np.nanmin(full_hop))))
        + ((full_delay - np.nanmin(full_delay)) / max(1e-9, float(np.nanmax(full_delay) - np.nanmin(full_delay)))),
        axis=1,
    )
    selected_counts = np.bincount(oracle_idx, minlength=len(selected_candidates))
    full_dir = out_dir / f"full_t{int(args.start)}_{int(args.end)}_s{int(args.stride)}"
    write_summary_csv(
        full_dir / "candidate_full_summary.csv",
        candidates=selected_candidates,
        hop=full_hop,
        delay=full_delay,
        score=full_score,
        selected_steps=selected_counts,
    )
    write_compare_csv(full_dir / "compare_mean_shortest_hops.csv", steps=steps.tolist(), candidates=selected_candidates, values=full_hop)
    write_compare_csv(full_dir / "compare_mean_shortest_delay_ms.csv", steps=steps.tolist(), candidates=selected_candidates, values=full_delay)
    pure_hop_env, pure_delay_env = read_pure_envelopes(PURE_METRIC_DIR, steps=steps.tolist())
    hybrid_hop_env = np.nanmin(full_hop, axis=1)
    hybrid_delay_env = np.nanmin(full_delay, axis=1)
    meta = {
        "config": str(Path(args.config)),
        "out_dir": str(out_dir),
        "candidate_count": len(candidates),
        "selected_indices": selected_indices,
        "selected_names": [c.name for c in selected_candidates],
        "primitive_bands": args.primitive_bands,
        "max_subset_size": int(args.max_subset_size),
        "mean_pure_hop_envelope": float(np.mean(pure_hop_env)),
        "mean_multiband_hop_envelope": float(np.mean(hybrid_hop_env)),
        "mean_pure_delay_envelope_ms": float(np.mean(pure_delay_env)),
        "mean_multiband_delay_envelope_ms": float(np.mean(hybrid_delay_env)),
        "multiband_hop_better_than_pure_env_steps": int(np.count_nonzero(hybrid_hop_env < pure_hop_env - 1e-9)),
        "multiband_delay_better_than_pure_env_steps": int(np.count_nonzero(hybrid_delay_env < pure_delay_env - 1e-9)),
        "scalar_selected_counts": {
            selected_candidates[idx].name: int(count)
            for idx, count in enumerate(selected_counts.tolist())
            if int(count) > 0
        },
        "reuse_meta": reuse_meta,
    }
    (full_dir / "multiband_search_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_full(
        out_dir=full_dir,
        steps=steps,
        candidates=selected_candidates,
        hop=full_hop,
        delay=full_delay,
        pure_hop_env=pure_hop_env,
        pure_delay_env=pure_delay_env,
        score=full_score,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
