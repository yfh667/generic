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
    compute_shortest_delay_batch,
    read_shortest_delay_series,
)


START = 0
END = 86160
STRIDE = 60

GROUP_XML = Path(r"E:\paper11\data\basic_file\G60\satellitesposition\station_visible_satellites_20250106.xml")
GROUP_CACHE_DIR = Path(r"E:\paper11\data\satnet_experiments\caches\G60\group_data_cache")
DELAY_STORE_DIR = Path(r"E:\paper11\data\linshi\G60_full_options_plus_intra_t0_86164_stride1")
HOPS_WIDE_CSV = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\motif_w_le4_h_le3\shortest_hops_t0_86160_stride60"
    r"\china_europe\compare_g60_w_le4_h_le3_808_china_europe_strict_reachable.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60"
    r"\static_motif0040_0056_china_europe_original"
)

CHINA_EUROPE = RegionPairSpec(
    key="china_europe",
    label="China-Europe",
    source_group_id=2,
    target_group_id=3,
)

MOTIFS = {
    "motif_000040": {
        "label": "motif 000040: CBD | CB-",
        "motif_text": "CBD | CB-",
        "csv_column": "combined_motif_000040",
        "support": "[(0,0,C), (0,1,B), (0,2,D), (1,0,C), (1,1,B)]",
        "color": "#C1121F",
    },
    "motif_000056": {
        "label": "motif 000056: DBD | --B",
        "motif_text": "DBD | --B",
        "csv_column": "combined_motif_000056",
        "support": "[(0,0,D), (0,1,B), (0,2,D), (1,2,B)]",
        "color": "#1B4F9C",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot original static motif 000040/000056 China-Europe shortest-delay and shortest-hop series."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force-delay", action="store_true")
    parser.add_argument("--engine", choices=("auto", "scipy", "heapq"), default="auto")
    parser.add_argument("--max-workers", type=int, default=2)
    return parser.parse_args()


def topology_specs() -> list[TopologySpec]:
    specs: list[TopologySpec] = []
    for name, info in MOTIFS.items():
        edge_table = build_motif_text_edge_table(
            motif_text=str(info["motif_text"]),
            config=G60_CONFIG,
            allow_vertical_overlap=True,
            allow_clipped_right=True,
            wrap_planes=False,
            add_intra_ring=True,
        )
        specs.append(
            TopologySpec(
                name=name,
                edge_table=edge_table,
                library="combined_motif",
                motif_id=int(name.rsplit("_", 1)[1]),
                motif=str(info["motif_text"]),
                source_w=3,
                source_h=3,
                edge_count_local=str(info["support"]).count("("),
                support=str(info["support"]),
                baseline=False,
                meta={"label": info["label"]},
            )
        )
    return specs


def write_hops_timeseries(out_dir: Path) -> Path:
    if not HOPS_WIDE_CSV.exists():
        raise FileNotFoundError(HOPS_WIDE_CSV)
    out_path = out_dir / "china_europe_static_motif0040_0056_mean_shortest_hops.csv"
    wanted = {info["csv_column"]: name for name, info in MOTIFS.items()}
    with HOPS_WIDE_CSV.open("r", encoding="utf-8-sig", newline="") as f_in, out_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f_out:
        reader = csv.DictReader(f_in)
        missing = sorted(set(wanted) - set(reader.fieldnames or []))
        if missing:
            raise KeyError(f"missing columns in hops CSV: {missing}")
        writer = csv.DictWriter(f_out, fieldnames=["step", *MOTIFS.keys()])
        writer.writeheader()
        for row in reader:
            out_row: dict[str, Any] = {"step": int(row["step"])}
            for column, name in wanted.items():
                value = row.get(column, "")
                out_row[name] = None if value in ("", None) else float(value)
            writer.writerow(out_row)
    return out_path


def write_delay_timeseries(out_dir: Path) -> Path:
    out_path = out_dir / "china_europe_static_motif0040_0056_mean_shortest_delay_ms.csv"
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in MOTIFS:
        steps, values = read_shortest_delay_series(out_dir / "delay" / "china_europe" / name)
        series[name] = (steps, values)
    first_steps = next(iter(series.values()))[0]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *MOTIFS.keys()])
        writer.writeheader()
        for idx, step in enumerate(first_steps):
            row: dict[str, Any] = {"step": int(step)}
            for name, (_steps, values) in series.items():
                row[name] = float(values[idx])
            writer.writerow(row)
    return out_path


def read_metric_csv(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    steps = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    values = {
        name: np.asarray(
            [float(row[name]) if row.get(name) not in ("", None) else np.nan for row in rows],
            dtype=np.float64,
        )
        for name in MOTIFS
    }
    return steps, values


def plot_one_metric(*, csv_path: Path, out_path: Path, title: str, y_label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, values_by_name = read_metric_csv(csv_path)
    x = steps.astype(np.float64) / 3600.0

    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=180)
    for name, values in values_by_name.items():
        finite = values[np.isfinite(values)]
        mean_text = f"{float(np.mean(finite)):.3f}" if finite.size else "nan"
        info = MOTIFS[name]
        ax.plot(
            x,
            values,
            linewidth=1.15,
            color=str(info["color"]),
            label=f"{info['label']} | mean={mean_text}",
        )
    ax.set_title(title)
    ax.set_xlabel("time (hour)")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.55)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_summary(*, out_dir: Path, hops_csv: Path, delay_csv: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for metric, path in (("hops", hops_csv), ("delay_ms", delay_csv)):
        _steps, values = read_metric_csv(path)
        for name, arr in values.items():
            finite = arr[np.isfinite(arr)]
            rows.append(
                {
                    "metric": metric,
                    "pair": "china_europe",
                    "topology": name,
                    "motif_text": MOTIFS[name]["motif_text"],
                    "support": MOTIFS[name]["support"],
                    "mean": float(np.mean(finite)) if finite.size else None,
                    "min": float(np.min(finite)) if finite.size else None,
                    "max": float(np.max(finite)) if finite.size else None,
                    "p95": float(np.percentile(finite, 95)) if finite.size else None,
                    "finite_steps": int(finite.size),
                }
            )
    out_path = out_dir / "china_europe_static_motif0040_0056_summary.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "pair",
                "topology",
                "motif_text",
                "support",
                "mean",
                "min",
                "max",
                "p95",
                "finite_steps",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = list(range(START, END + 1, STRIDE))

    group_data = load_or_build_group_data(
        xml_file=GROUP_XML,
        group_cache_dir=GROUP_CACHE_DIR,
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=STRIDE,
        enabled=True,
        force=False,
    )

    delay_meta = compute_shortest_delay_batch(
        topology_specs=topology_specs(),
        pair_specs=[CHINA_EUROPE],
        config=G60_CONFIG,
        group_data=group_data,
        start=START,
        end=END,
        stride=STRIDE,
        out_dir=out_dir / "delay",
        delay_store_dir=DELAY_STORE_DIR,
        engine=str(args.engine),
        progress_every=1,
        max_workers=int(args.max_workers),
        force=bool(args.force_delay),
    )

    hops_csv = write_hops_timeseries(out_dir)
    delay_csv = write_delay_timeseries(out_dir)

    hops_png = out_dir / "china_europe_static_motif0040_0056_mean_shortest_hops.png"
    delay_png = out_dir / "china_europe_static_motif0040_0056_mean_shortest_delay_ms.png"
    plot_one_metric(
        csv_path=hops_csv,
        out_path=hops_png,
        title="G60 China-Europe original static motif comparison: mean shortest hops",
        y_label="mean shortest hops",
    )
    plot_one_metric(
        csv_path=delay_csv,
        out_path=delay_png,
        title="G60 China-Europe original static motif comparison: mean shortest delay",
        y_label="mean shortest delay (ms)",
    )
    summary_csv = write_summary(out_dir=out_dir, hops_csv=hops_csv, delay_csv=delay_csv)
    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "start": START,
                "end": END,
                "stride": STRIDE,
                "delay_meta": delay_meta,
                "hops_csv": str(hops_csv),
                "delay_csv": str(delay_csv),
                "hops_png": str(hops_png),
                "delay_png": str(delay_png),
                "summary_csv": str(summary_csv),
                "motifs": MOTIFS,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in (hops_png, delay_png, summary_csv, hops_csv, delay_csv):
        print(path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
