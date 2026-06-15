from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


GENERIC_ROOT = Path(r"E:\paper11\generic")
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data  # noqa: E402
from src.topology_workflow.module import (  # noqa: E402
    RegionPairSpec,
    TopologySpec,
    build_motif_text_edge_table,
    build_single_motif_edge_table,
    compute_shortest_delay_batch,
    read_shortest_delay_series,
)


GROUP_XML = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml"
)
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DELAY_STORE_DIR = Path(r"E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1")
POSITION_CACHE_DIR = Path(
    r"E:\paper11\data\basic_file\G60\satellitesposition\_position_cache\cache_0_86164_1s"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\custom_w3h3_CDC_vs_motif000056\shortest_delay_t0_86160_stride60"
)

CUSTOM_TOPOLOGY_NAME = "custom_w3h3_CDC"
BASELINE_TOPOLOGY_NAME = "motif_000056_DBD_xxB"


PAIR_SPECS = (
    RegionPairSpec(key="china_europe", label="China-Europe", source_group_id=2, target_group_id=3),
    RegionPairSpec(key="china_america", label="China-America", source_group_id=2, target_group_id=0),
    RegionPairSpec(key="china_africa", label="China-Africa", source_group_id=2, target_group_id=1),
)


CUSTOM_MOTIF = {
    "w": 3,
    "h": 3,
    "support": [
        [0, 0, "C"],
        [0, 1, "D"],
        [1, 1, "C"],
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a manual w3h3 CDC motif with motif_000056_DBD_xxB on G60 shortest delay."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=86160)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--delay-store-dir", type=Path, default=DELAY_STORE_DIR)
    parser.add_argument("--position-cache-dir", type=Path, default=POSITION_CACHE_DIR)
    parser.add_argument("--engine", choices=("auto", "scipy", "heapq"), default="auto")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    return parser.parse_args()


def make_topology_specs() -> list[TopologySpec]:
    custom_edge_table = build_single_motif_edge_table(
        topology_raw={
            "kind": "single_motif",
            "motif": CUSTOM_MOTIF,
            "tiling": {
                "allow_vertical_overlap": True,
                "allow_clipped_right": True,
            },
            "add_intra_ring": True,
            "wrap_planes": False,
        },
        config=G60_CONFIG,
    )
    motif56_edge_table = build_motif_text_edge_table(
        motif_text="DBD | --B",
        config=G60_CONFIG,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
        wrap_planes=False,
        add_intra_ring=True,
    )
    return [
        TopologySpec(
            name=CUSTOM_TOPOLOGY_NAME,
            edge_table=custom_edge_table,
            library="manual",
            motif="support",
            source_w=3,
            source_h=3,
            edge_count_local=3,
            support=str([(0, 0, "C"), (0, 1, "D"), (1, 1, "C")]),
            meta={"motif": CUSTOM_MOTIF},
        ),
        TopologySpec(
            name=BASELINE_TOPOLOGY_NAME,
            edge_table=motif56_edge_table,
            library="combined_motif",
            motif_id=56,
            motif="DBD | --B",
            source_w=3,
            source_h=3,
            edge_count_local=4,
            support=str([(0, 0, "D"), (0, 1, "B"), (0, 2, "D"), (1, 2, "B")]),
            meta={"note": "This is the user-facing motif_000056_DBD_xxB label."},
        ),
    ]


def write_three_pair_plot(out_dir: Path, pair_specs: tuple[RegionPairSpec, ...]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_path = fig_dir / "custom_w3h3_CDC_vs_motif000056_three_pairs_delay.png"

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), dpi=180, sharex=True)
    colors = {
        CUSTOM_TOPOLOGY_NAME: "#d95f02",
        BASELINE_TOPOLOGY_NAME: "#1b4f9c",
    }
    labels = {
        CUSTOM_TOPOLOGY_NAME: "custom w3h3 CDC",
        BASELINE_TOPOLOGY_NAME: "motif 000056 DBD | --B",
    }
    summary_rows: list[dict[str, Any]] = []

    for ax, pair in zip(axes, pair_specs):
        series = {}
        for name in (CUSTOM_TOPOLOGY_NAME, BASELINE_TOPOLOGY_NAME):
            steps, values = read_shortest_delay_series(out_dir / pair.key / name)
            series[name] = (steps, values)
            finite = values[np.isfinite(values)]
            summary_rows.append(
                {
                    "pair": pair.key,
                    "label": pair.label,
                    "topology": name,
                    "mean_ms": float(np.mean(finite)) if finite.size else None,
                    "min_ms": float(np.min(finite)) if finite.size else None,
                    "max_ms": float(np.max(finite)) if finite.size else None,
                    "finite_steps": int(finite.size),
                }
            )

        x_hours = series[CUSTOM_TOPOLOGY_NAME][0].astype(np.float64) / 3600.0
        for name, (_steps, values) in series.items():
            ax.plot(
                x_hours,
                values,
                linewidth=1.0,
                color=colors[name],
                label=labels[name],
            )
        ax.set_title(pair.label)
        ax.set_ylabel("mean shortest delay (ms)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=9)

    axes[-1].set_xlabel("time (hour)")
    fig.suptitle("G60 shortest-delay comparison: custom w3h3 CDC vs motif 000056", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(plot_path)
    plt.close(fig)

    summary_path = fig_dir / "custom_w3h3_CDC_vs_motif000056_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pair", "label", "topology", "mean_ms", "min_ms", "max_ms", "finite_steps"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)
    return plot_path


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty time axis")

    topology_specs = make_topology_specs()
    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=True,
        force=bool(args.force_group_cache),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manual_motif_spec.json").write_text(
        json.dumps(
            {
                "custom_topology_name": CUSTOM_TOPOLOGY_NAME,
                "baseline_topology_name": BASELINE_TOPOLOGY_NAME,
                "custom_motif": CUSTOM_MOTIF,
                "baseline_motif_text": "DBD | --B",
                "start": int(args.start),
                "end": int(args.end),
                "stride": int(args.stride),
                "delay_store_dir": str(args.delay_store_dir),
                "position_cache_dir": str(args.position_cache_dir),
                "pairs": [pair.key for pair in PAIR_SPECS],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = compute_shortest_delay_batch(
        topology_specs=topology_specs,
        pair_specs=PAIR_SPECS,
        config=G60_CONFIG,
        group_data=group_data,
        start=int(args.start),
        end=int(args.end),
        stride=int(args.stride),
        out_dir=args.out_dir,
        delay_store_dir=args.delay_store_dir,
        position_cache_dir=args.position_cache_dir,
        engine=str(args.engine),
        sample_steps=0,
        sample_pairs_per_step=0,
        progress_every=1,
        max_workers=int(args.max_workers),
        force=bool(args.force),
    )
    plot_path = write_three_pair_plot(args.out_dir, PAIR_SPECS)
    meta["three_pair_plot"] = str(plot_path)
    (args.out_dir / "custom_compare_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
