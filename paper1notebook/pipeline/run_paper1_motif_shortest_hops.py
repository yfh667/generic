from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from paper1_edge_tables import build_legacy_gridplus_edge_table
from src.motif_generator.module.library import write_combined_canonical_motif_library_csv
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_workflow.module.batch_shortest_hops import (
    RegionPairSpec,
    TopologySpec,
    compute_shortest_hops_batch,
    full_link_topology_spec,
    topology_specs_from_motif_csv,
)
from src.topology_workflow.module.config import time_axis_from_config, viewer_config_from_workflow


DEFAULT_CONFIG = THIS_DIR / "configs" / "g60_w_le4_h_le3_shortest_hops.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper1 motif-library shortest-hop pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit-motifs", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=None)
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--regenerate-library", action="store_true")
    parser.add_argument("--write-edges", action="store_true")
    parser.add_argument("--skip-gridplus", action="store_true")
    parser.add_argument("--pairs", nargs="*", default=None, help="Optional subset of region pair keys.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def path_from(raw: dict[str, Any], key: str) -> Path:
    value = raw.get(key)
    if value in (None, ""):
        raise ValueError(f"missing paths.{key}")
    return Path(str(value))


def motif_library_csv(raw: dict[str, Any]) -> Path:
    paths = raw.get("paths", {})
    library = raw.get("motif_library", {})
    if not isinstance(paths, dict) or not isinstance(library, dict):
        raise ValueError("paths and motif_library must both be mappings")
    library_dir = path_from(paths, "motif_library_dir")
    csv_name = str(library.get("csv_name", "combined_motif_library.csv"))
    csv_path = Path(csv_name)
    return csv_path if csv_path.is_absolute() else library_dir / csv_name


def ensure_motif_library(raw: dict[str, Any], *, force: bool) -> Path:
    library = raw.get("motif_library", {})
    if not isinstance(library, dict):
        raise ValueError("motif_library must be a mapping")
    csv_path = motif_library_csv(raw)
    regenerate = bool(force or library.get("regenerate", False) or not csv_path.exists())
    if regenerate:
        print(f"[paper1-pipeline] generating motif library: {csv_path}", flush=True)
        meta = write_combined_canonical_motif_library_csv(
            csv_path,
            min_w=int(library.get("min_w", 2)),
            min_h=int(library.get("min_h", 1)),
            max_w=int(library["max_w"]),
            max_h=int(library["max_h"]),
            include_max_size=bool(library.get("include_max_size", True)),
            row_pitch=int(raw.get("constellation", {}).get("n", 36)),
            phase_count=raw.get("constellation", {}).get("n"),
        )
        print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    else:
        print(f"[paper1-pipeline] reusing motif library: {csv_path}", flush=True)
    return csv_path


def region_pair_specs(raw: dict[str, Any], subset: list[str] | None = None) -> list[RegionPairSpec]:
    wanted = set(str(x) for x in subset) if subset else None
    out: list[RegionPairSpec] = []
    for item in raw.get("region_pairs", []) or []:
        key = str(item["key"])
        if wanted is not None and key not in wanted:
            continue
        out.append(
            RegionPairSpec(
                key=key,
                label=str(item.get("label", key)),
                source_group_id=int(item["source_group_id"]),
                target_group_id=int(item["target_group_id"]),
            )
        )
    if not out:
        raise ValueError("no region pairs selected")
    return out


def build_baselines(raw: dict[str, Any], *, config, skip_gridplus: bool = False) -> list[TopologySpec]:
    baselines = raw.get("baselines", {})
    paths = raw.get("paths", {})
    out: list[TopologySpec] = []
    full_raw = baselines.get("full_link", {}) if isinstance(baselines, dict) else {}
    if bool(full_raw.get("enabled", False)):
        out.append(
            full_link_topology_spec(
                config=config,
                name="full_link",
                options=tuple(int(x) for x in full_raw.get("options", [0, 1, 2, 4])),
                add_intra_ring=bool(full_raw.get("add_intra_ring", True)),
            )
        )

    grid_raw = baselines.get("gridplus", {}) if isinstance(baselines, dict) else {}
    if bool(grid_raw.get("enabled", False)) and not bool(skip_gridplus):
        gridplus_config = path_from(paths, "gridplus_config")
        out.append(
            TopologySpec(
                name="gridplus",
                edge_table=build_legacy_gridplus_edge_table(
                    motif_json=gridplus_config,
                    add_intra_ring=bool(grid_raw.get("add_intra_ring", True)),
                ),
                library="baseline",
                motif="legacy_gridplus",
                baseline=True,
                meta={"gridplus_config": str(gridplus_config)},
            )
        )
    return out


def main() -> int:
    args = parse_args()
    raw = load_yaml(args.config)
    config = viewer_config_from_workflow(raw)

    start, end, stride = time_axis_from_config(raw)
    if args.start is not None:
        start = int(args.start)
    if args.end is not None:
        end = int(args.end)
    if args.stride is not None:
        stride = int(args.stride)
    if end < start:
        raise ValueError("end must be >= start")
    if stride <= 0:
        raise ValueError("stride must be positive")
    steps = list(range(int(start), int(end) + 1, int(stride)))

    run_raw = raw.get("run", {}) if isinstance(raw.get("run", {}), dict) else {}
    paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
    out_dir = Path(args.out_dir) if args.out_dir is not None else path_from(paths, "out_dir")
    limit_motifs = int(args.limit_motifs if args.limit_motifs is not None else run_raw.get("limit_motifs", 0))
    max_workers = int(args.max_workers if args.max_workers is not None else run_raw.get("max_workers", 0))
    progress_every = int(args.progress_every if args.progress_every is not None else run_raw.get("progress_every", 25))
    write_edges = bool(args.write_edges or run_raw.get("write_edges", False))
    force_group_cache = bool(args.force_group_cache or run_raw.get("force_group_cache", False))
    run_label = str(run_raw.get("label", "paper1_motif_shortest_hops"))

    csv_path = ensure_motif_library(raw, force=bool(args.regenerate_library))
    library_raw = raw.get("motif_library", {}) if isinstance(raw.get("motif_library", {}), dict) else {}
    motif_specs = topology_specs_from_motif_csv(
        csv_path,
        config=config,
        library="combined_motif",
        name_prefix=str(library_raw.get("name_prefix", "combined")),
        limit=limit_motifs,
        add_intra_ring=True,
    )
    baseline_specs = build_baselines(raw, config=config, skip_gridplus=bool(args.skip_gridplus))
    topology_specs = motif_specs + baseline_specs

    print(
        f"[paper1-pipeline] constellation={config.name} P={config.P} N={config.N} "
        f"motifs={len(motif_specs)} baselines={len(baseline_specs)} steps={len(steps)}",
        flush=True,
    )

    group_data = load_or_build_group_data(
        xml_file=path_from(paths, "group_xml"),
        group_cache_dir=path_from(paths, "group_cache_dir"),
        steps=steps,
        station_groups=config.station_groups,
        total_sats=config.total_sats,
        constellation_name=config.name,
        stride=int(stride),
        enabled=True,
        force=force_group_cache,
    )
    pair_specs = region_pair_specs(raw, subset=args.pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    effective = {
        "config_file": str(args.config),
        "start": int(start),
        "end": int(end),
        "stride": int(stride),
        "out_dir": str(out_dir),
        "motif_library_csv": str(csv_path),
        "limit_motifs": int(limit_motifs),
        "max_workers": int(max_workers),
        "pairs": [pair.key for pair in pair_specs],
        "run_label": run_label,
        "skip_gridplus": bool(args.skip_gridplus),
    }
    (out_dir / "effective_run_config.json").write_text(
        json.dumps(effective, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = compute_shortest_hops_batch(
        topology_specs=topology_specs,
        pair_specs=pair_specs,
        group_data=group_data,
        steps=steps,
        total_nodes=int(config.total_sats),
        out_dir=out_dir,
        run_label=run_label,
        max_workers=max_workers,
        progress_every=progress_every,
        write_edges=write_edges,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
