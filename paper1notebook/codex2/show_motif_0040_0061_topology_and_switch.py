from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PyQt5 import QtWidgets


GENERIC_ROOT = Path(__file__).resolve().parents[2]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG  # noqa: E402
from src.satellite_topology_viewer.module.base_viewer import SatelliteTopology2DViewer  # noqa: E402
from src.topology_workflow.module.edge_tables import INTRA_OPTION, build_motif_text_edge_table  # noqa: E402


DEFAULT_LIBRARY_CSV = Path(
    r"E:\paper11\data\satnet_experiments\libraries\motif\exact_box"
    r"\w_le_4_h_le_3\combined_w_le4_h_le3_808.csv"
)
DEFAULT_OUT_DIR = Path(
    r"E:\paper11\data\satnet_experiments\runs\paper1\G60\motif_w_le4_h_le3"
    r"\motif_0040_0061_topology_switch_review"
)
BASE_MOTIF_ID = 56
TARGET_MOTIF_IDS = (40, 61)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw G60 motif 000040/000061 topologies and switch counts against motif 000056."
    )
    parser.add_argument("--library-csv", type=Path, default=DEFAULT_LIBRARY_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-motif-id", type=int, default=BASE_MOTIF_ID)
    parser.add_argument("--target-motif-id", type=int, nargs="*", default=list(TARGET_MOTIF_IDS))
    parser.add_argument("--width", type=int, default=1300)
    parser.add_argument("--height", type=int, default=860)
    return parser.parse_args()


def read_motif_library(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"motif library not found: {path}")
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            motif_id = int(row["motif_id"])
            rows[motif_id] = dict(row)
    return rows


def edge_key(edge_table, edge_idx: int) -> tuple[int, int]:
    src = int(edge_table.src[edge_idx])
    dst = int(edge_table.dst[edge_idx])
    return (src, dst) if src < dst else (dst, src)


def inter_edge_set(edge_table) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for edge_idx in range(int(edge_table.num_edges)):
        if int(edge_table.option[edge_idx]) == INTRA_OPTION:
            continue
        edges.add(edge_key(edge_table, edge_idx))
    return edges


def build_table(motif_text: str):
    return build_motif_text_edge_table(
        motif_text=str(motif_text),
        config=G60_CONFIG,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
        wrap_planes=False,
        add_intra_ring=True,
    )


def switch_stats(*, base_id: int, base_text: str, target_id: int, target_text: str) -> dict[str, object]:
    base_table = build_table(base_text)
    target_table = build_table(target_text)
    base_edges = inter_edge_set(base_table)
    target_edges = inter_edge_set(target_table)
    common = base_edges & target_edges
    removed = base_edges - target_edges
    added = target_edges - base_edges
    union = base_edges | target_edges
    changed = removed | added

    return {
        "base_motif_id": int(base_id),
        "base_motif": str(base_text),
        "target_motif_id": int(target_id),
        "target_motif": str(target_text),
        "base_inter_edges": len(base_edges),
        "target_inter_edges": len(target_edges),
        "common_inter_edges": len(common),
        "removed_from_base": len(removed),
        "added_by_target": len(added),
        "changed_symmetric_diff": len(changed),
        "union_inter_edges": len(union),
        "kept_base_percent": 100.0 * len(common) / len(base_edges) if base_edges else 0.0,
        "removed_base_percent": 100.0 * len(removed) / len(base_edges) if base_edges else 0.0,
        "added_target_percent": 100.0 * len(added) / len(target_edges) if target_edges else 0.0,
        "changed_union_percent": 100.0 * len(changed) / len(union) if union else 0.0,
    }


def save_stats_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "base_motif_id",
        "base_motif",
        "target_motif_id",
        "target_motif",
        "base_inter_edges",
        "target_inter_edges",
        "common_inter_edges",
        "removed_from_base",
        "added_by_target",
        "changed_symmetric_diff",
        "union_inter_edges",
        "kept_base_percent",
        "removed_base_percent",
        "added_target_percent",
        "changed_union_percent",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_single_change_plot(row: dict[str, object], path: Path) -> None:
    target = int(row["target_motif_id"])
    motif_text = str(row["target_motif"])
    values = [
        int(row["common_inter_edges"]),
        int(row["removed_from_base"]),
        int(row["added_by_target"]),
    ]
    labels = ["common", "removed from 000056", "added by target"]
    colors = ["#2563eb", "#dc2626", "#16a34a"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=160)
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("inter-link count")
    ax.set_title(
        f"motif {target:06d} vs motif 000056\n"
        f"{motif_text} | changed={int(row['changed_symmetric_diff'])}, "
        f"changed/union={float(row['changed_union_percent']):.2f}%"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.22 if values else 1)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_summary_change_plot(rows: list[dict[str, object]], path: Path) -> None:
    names = [f"{int(row['target_motif_id']):06d}\n{row['target_motif']}" for row in rows]
    common = [int(row["common_inter_edges"]) for row in rows]
    removed = [int(row["removed_from_base"]) for row in rows]
    added = [int(row["added_by_target"]) for row in rows]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    ax.bar(x, common, label="common", color="#2563eb")
    ax.bar(x, removed, bottom=common, label="removed from 000056", color="#dc2626")
    bottom = [a + b for a, b in zip(common, removed)]
    ax.bar(x, added, bottom=bottom, label="added by target", color="#16a34a")
    for idx, row in enumerate(rows):
        total = common[idx] + removed[idx] + added[idx]
        ax.text(
            idx,
            total + 12,
            f"changed {int(row['changed_symmetric_diff'])}\n{float(row['changed_union_percent']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(list(x), names)
    ax.set_ylabel("inter-link count")
    ax.set_title("Topology change counts against motif 000056")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(common[idx] + removed[idx] + added[idx] for idx in x) * 1.22)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_topology_screenshot(*, app: QtWidgets.QApplication, motif_id: int, motif_text: str, out_path: Path, width: int, height: int) -> None:
    table = build_table(motif_text)
    viewer = SatelliteTopology2DViewer(
        G60_CONFIG,
        steps=[0],
        edge_table=table,
        window_title=f"G60 motif {motif_id:06d}: {motif_text}",
        group_data={},
        show_groups=False,
        show_grid_lines=False,
        topology_edge_color="#000000",
        topology_edge_alpha=210,
        topology_edge_width=0.032,
        hide_y_wrap_edges=True,
    )
    viewer.edge_width = 0.034
    viewer.edge_alpha = 220
    viewer.node_radius = 0.125
    viewer.resize(int(width), int(height))
    viewer.show()
    for _ in range(16):
        app.processEvents()
        time.sleep(0.04)
    ok = viewer.grab().save(str(out_path))
    print(f"[motif-topology-switch] screenshot={out_path} ok={ok}", flush=True)
    viewer.close()
    app.processEvents()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    motif_rows = read_motif_library(args.library_csv)
    base_id = int(args.base_motif_id)
    target_ids = [int(x) for x in args.target_motif_id]
    missing = [mid for mid in [base_id, *target_ids] if mid not in motif_rows]
    if missing:
        raise KeyError(f"motif id(s) not found in library: {missing}")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    base_text = motif_rows[base_id]["motif"].strip()
    stats_rows: list[dict[str, object]] = []
    for target_id in target_ids:
        target_text = motif_rows[target_id]["motif"].strip()
        save_topology_screenshot(
            app=app,
            motif_id=target_id,
            motif_text=target_text,
            out_path=out_dir / f"motif_{target_id:06d}_topology_2d.png",
            width=int(args.width),
            height=int(args.height),
        )
        row = switch_stats(
            base_id=base_id,
            base_text=base_text,
            target_id=target_id,
            target_text=target_text,
        )
        stats_rows.append(row)
        save_single_change_plot(row, out_dir / f"motif_{target_id:06d}_switch_counts_vs_000056.png")

    save_stats_csv(stats_rows, out_dir / "motif_0040_0061_switch_counts_vs_000056.csv")
    save_summary_change_plot(stats_rows, out_dir / "motif_0040_0061_switch_counts_vs_000056_summary.png")

    for row in stats_rows:
        print(
            "[motif-topology-switch] "
            f"000056 -> {int(row['target_motif_id']):06d} | "
            f"target={row['target_motif']} | "
            f"base={row['base_inter_edges']} target_edges={row['target_inter_edges']} "
            f"common={row['common_inter_edges']} removed={row['removed_from_base']} "
            f"added={row['added_by_target']} changed={row['changed_symmetric_diff']} "
            f"changed/union={float(row['changed_union_percent']):.2f}% "
            f"kept/base={float(row['kept_base_percent']):.2f}%",
            flush=True,
        )
    print(f"[motif-topology-switch] out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
