from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
PROJECT_ROOT = GENERIC_ROOT.parent
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))

from src.config.viewer_config import G60_CONFIG
from src.link_delay.module.edge_options import EdgeTable, write_edges_csv
from src.motif_generator.module.exact_box import motif_from_columns
from src.motif_generator.module.tiling import tile_motif_on_grid, write_tiled_motif_outputs
from src.motif_generator.module.viewer_adapter import tiled_result_to_edge_table
from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.base_viewer import (
    SatelliteTopology2DViewer,
    distance_point_to_polyline,
)
from src.satellite_topology_viewer.module.layout_transforms import build_rev_group_offsets
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data


DEFAULT_XML = (
    PROJECT_ROOT
    / "data"
    / "basic_file"
    / "G60"
    / "satellitesposition"
    / "station_visible_satellites_20250106.xml"
)
DEFAULT_GROUP_CACHE = PROJECT_ROOT / "data" / "linshi" / "cache" / "group_data_cache"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "linshi" / "motif_000116_dynamic_2d_viewer"
MOTIF_ID = 116
MOTIF_COLUMNS = ("AAA", "DBD", "--B")

OPTION_TO_SYMBOL = {
    -1: "I",
    0: "A",
    1: "B",
    2: "D",
    4: "C",
}
OPTION_COLORS = {
    -1: "#737373",
    0: "#1f77b4",
    1: "#7b2cbf",
    2: "#2a9d8f",
    4: "#f77f00",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a dynamic 2D viewer for G60 motif_000116 with edge-order labels."
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=100, help="Inclusive end step.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--initial-step", type=int, default=0)
    parser.add_argument("--xml-file", type=Path, default=DEFAULT_XML)
    parser.add_argument("--group-cache-dir", type=Path, default=DEFAULT_GROUP_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1500)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=180)
    parser.add_argument("--label-pixel-size", type=int, default=14)
    parser.add_argument("--label-scale", type=float, default=0.018)
    parser.add_argument(
        "--edge-label-mode",
        choices=("symbol", "option", "symbol-option"),
        default="symbol-option",
        help="Text drawn above each inter edge.",
    )
    parser.add_argument("--include-intra", action="store_true", help="Also draw y-ring intra links.")
    parser.add_argument("--label-intra", action="store_true", help="Also label intra links as I/-1.")
    parser.add_argument("--hide-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument(
        "--no-rev-group",
        action="store_true",
        help="Do not shift the displayed topology by the base group. Edges are static in raw coordinates.",
    )
    parser.add_argument("--rev-group-base-id", type=int, default=0)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--offscreen", action="store_true")
    parser.add_argument("--screenshot", type=Path, default=None)
    return parser.parse_args()


def append_intra_ring(edge_table: EdgeTable, *, p_count: int, y_count: int) -> EdgeTable:
    src = list(int(x) for x in edge_table.src)
    dst = list(int(x) for x in edge_table.dst)
    option = list(int(x) for x in edge_table.option)
    src_plane = list(int(x) for x in edge_table.src_plane)
    src_y = list(int(x) for x in edge_table.src_y)
    dst_plane = list(int(x) for x in edge_table.dst_plane)
    dst_y = list(int(x) for x in edge_table.dst_y)

    seen = {tuple(sorted((int(u), int(v)))) for u, v in zip(src, dst)}
    for plane in range(int(p_count)):
        for y in range(int(y_count)):
            u = plane * int(y_count) + y
            v = plane * int(y_count) + ((y + 1) % int(y_count))
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            src.append(u)
            dst.append(v)
            option.append(-1)
            src_plane.append(plane)
            src_y.append(y)
            dst_plane.append(plane)
            dst_y.append((y + 1) % int(y_count))

    order = sorted(range(len(src)), key=lambda idx: (src[idx], dst[idx], option[idx]))
    sat_ids = [str(i + 1) for i in range(int(p_count) * int(y_count))]
    return EdgeTable(
        src=np.asarray([src[idx] for idx in order], dtype=np.int32),
        dst=np.asarray([dst[idx] for idx in order], dtype=np.int32),
        option=np.asarray([option[idx] for idx in order], dtype=np.int16),
        src_plane=np.asarray([src_plane[idx] for idx in order], dtype=np.int16),
        src_y=np.asarray([src_y[idx] for idx in order], dtype=np.int16),
        dst_plane=np.asarray([dst_plane[idx] for idx in order], dtype=np.int16),
        dst_y=np.asarray([dst_y[idx] for idx in order], dtype=np.int16),
        sat_ids=sat_ids,
    )


def build_motif_000116_edge_table(*, include_intra: bool) -> tuple[EdgeTable, object]:
    motif = motif_from_columns(MOTIF_COLUMNS)
    result = tile_motif_on_grid(
        p=int(G60_CONFIG.P),
        n=int(G60_CONFIG.N),
        motif=motif,
        horizontal_step=None,
        allow_vertical_overlap=True,
        allow_clipped_right=True,
    )
    edge_table = tiled_result_to_edge_table(result)
    if include_intra:
        edge_table = append_intra_ring(edge_table, p_count=int(G60_CONFIG.P), y_count=int(G60_CONFIG.N))
    return edge_table, result


def shift_node_by_offset(node: int, *, y_count: int, offset: int) -> int:
    plane, y = divmod(int(node), int(y_count))
    shifted_y = (int(y) - int(offset) + int(y_count) - 1) % int(y_count)
    return int(plane) * int(y_count) + int(shifted_y)


def shift_group_data(group_data: dict, offsets_by_step: dict[int, int], *, y_count: int) -> dict:
    shifted: dict[int, dict] = {}
    for step, step_data in group_data.items():
        offset = int(offsets_by_step.get(int(step), 0))
        groups = step_data.get("groups", {}) if isinstance(step_data, dict) else {}
        shifted_groups: dict[int, set[int]] = {}
        all_mentioned: set[int] = set()
        for gid, nodes in groups.items():
            target = shifted_groups.setdefault(int(gid), set())
            for node in nodes or []:
                shifted_node = shift_node_by_offset(int(node), y_count=int(y_count), offset=offset)
                target.add(shifted_node)
                all_mentioned.add(shifted_node)
        shifted[int(step)] = {"groups": shifted_groups, "all_mentioned": all_mentioned}
    return shifted


def edge_label_text(option: int, mode: str) -> str:
    symbol = OPTION_TO_SYMBOL.get(int(option), str(option))
    if mode == "symbol":
        return symbol
    if mode == "option":
        return str(int(option))
    return f"{symbol}{int(option)}"


class EdgeOrderDynamicTopologyViewer(SatelliteTopology2DViewer):
    """2D viewer variant that shifts edge endpoints by per-step offsets and labels edge order."""

    def __init__(
        self,
        *args,
        offsets_by_step: dict[int, int] | None = None,
        use_rev_offsets: bool = True,
        edge_label_mode: str = "symbol-option",
        label_intra: bool = False,
        label_pixel_size: int = 8,
        label_scale: float = 0.03,
        **kwargs,
    ):
        self.offsets_by_step = {int(k): int(v) for k, v in (offsets_by_step or {}).items()}
        self.use_rev_offsets = bool(use_rev_offsets)
        self.edge_label_mode = str(edge_label_mode)
        self.label_intra = bool(label_intra)
        self.label_pixel_size = int(label_pixel_size)
        self.label_scale = float(label_scale)
        self.edge_label_items: list[QtWidgets.QGraphicsPixmapItem] = []
        self.edge_label_texts: list[str] = []
        self.edge_label_pixmaps: dict[tuple[str, str], QtGui.QPixmap] = {}
        super().__init__(*args, **kwargs)

    def _build_ui(self):
        super()._build_ui()
        controls_layout = self.controls_panel.layout()
        label_row = QtWidgets.QHBoxLayout()
        self.show_edge_labels_checkbox = QtWidgets.QCheckBox("Show edge order labels")
        self.show_edge_labels_checkbox.setChecked(True)
        self.show_edge_labels_checkbox.stateChanged.connect(lambda _state: self.update_step(self.current_row))
        label_row.addWidget(self.show_edge_labels_checkbox)
        self.edge_label_legend = QtWidgets.QLabel("A0=(+1,0), B1=(+1,-1), C4=(+1,+1), D2=(+2,0), I-1=intra")
        self.edge_label_legend.setObjectName("statusLabel")
        label_row.addWidget(self.edge_label_legend, stretch=1)
        controls_layout.addLayout(label_row)

    def _edge_label_pixmap(self, text: str, color: str) -> QtGui.QPixmap:
        key = (str(text), str(color))
        cached = self.edge_label_pixmaps.get(key)
        if cached is not None:
            return cached

        from PIL import Image, ImageDraw, ImageFont

        font_path_candidates = [
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\consolab.ttf"),
            Path(r"C:\Windows\Fonts\consola.ttf"),
        ]
        font = None
        for font_path in font_path_candidates:
            if font_path.exists():
                font = ImageFont.truetype(str(font_path), size=28)
                break
        if font is None:
            font = ImageFont.load_default()

        dummy = Image.new("RGBA", (1, 1), (255, 255, 255, 0))
        draw = ImageDraw.Draw(dummy)
        box = draw.textbbox((0, 0), text, font=font)
        width = max(24, int(box[2] - box[0] + 10))
        height = max(18, int(box[3] - box[1] + 8))
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        rgb = QtGui.QColor(str(color))
        draw.text((5 - box[0], 4 - box[1]), text, fill=(rgb.red(), rgb.green(), rgb.blue(), 255), font=font)
        raw = image.tobytes("raw", "RGBA")
        qimage = QtGui.QImage(raw, image.width, image.height, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qimage.copy())
        self.edge_label_pixmaps[key] = pixmap
        return pixmap

    def _build_scene(self):
        super()._build_scene()
        self.edge_label_items = []
        self.edge_label_texts = []
        for idx in range(int(self.edge_table.num_edges)):
            text = edge_label_text(int(self.edge_table.option[idx]), self.edge_label_mode)
            color = OPTION_COLORS.get(int(self.edge_table.option[idx]), "#222222")
            item = QtWidgets.QGraphicsPixmapItem(self._edge_label_pixmap(text, color))
            item.setScale(float(self.label_scale))
            item.setZValue(70)
            item.setTransformationMode(QtCore.Qt.SmoothTransformation)
            self.scene.addItem(item)
            self.edge_label_items.append(item)
            self.edge_label_texts.append(text)

    def _offset_for_row(self, row: int | None = None) -> int:
        if not self.use_rev_offsets:
            return 0
        row = self.current_row if row is None else int(row)
        step = int(self.steps[int(max(0, min(row, len(self.steps) - 1)))])
        return int(self.offsets_by_step.get(step, 0))

    def _display_node(self, raw_node: int, row: int | None = None) -> int:
        if not self.use_rev_offsets:
            return int(raw_node)
        return shift_node_by_offset(
            int(raw_node),
            y_count=int(self.config.N),
            offset=self._offset_for_row(row),
        )

    def _display_edge_raw_y_pair(self, idx: int, row: int | None = None) -> tuple[int, int]:
        src_display = self._display_node(int(self.edge_table.src[idx]), row)
        dst_display = self._display_node(int(self.edge_table.dst[idx]), row)
        return int(src_display % int(self.config.N)), int(dst_display % int(self.config.N))

    def _edge_path_and_samples(self, idx: int, row: int | None = None):
        src_display = self._display_node(int(self.edge_table.src[idx]), row)
        dst_display = self._display_node(int(self.edge_table.dst[idx]), row)
        x0, y0 = self.node_grid_pos(src_display)
        x1, y1 = self.node_grid_pos(dst_display)
        path = QtGui.QPainterPath()
        path.moveTo(x0, y0)

        option = int(self.edge_table.option[idx])
        if option == -1:
            raw_y0, raw_y1 = self._display_edge_raw_y_pair(idx, row)
            if abs(raw_y0 - raw_y1) == int(self.config.N) - 1:
                side_x = float(x0) + 0.32
                top_y = min(float(y0), float(y1)) - 0.55
                bottom_y = max(float(y0), float(y1)) + 0.55
                path.lineTo(side_x, top_y)
                path.lineTo(side_x, bottom_y)
                path.lineTo(x1, y1)
                samples = [(float(x0), float(y0)), (side_x, top_y), (side_x, bottom_y), (float(x1), float(y1))]
            else:
                path.lineTo(x1, y1)
                samples = [(float(x0), float(y0)), (float(x1), float(y1))]
        elif option == 2:
            ctrl_x = (x0 + x1) / 2.0
            ctrl_y = (y0 + y1) / 2.0 + 0.5 * abs(x1 - x0)
            path.quadTo(ctrl_x, ctrl_y, x1, y1)
            samples = []
            for t in np.linspace(0.0, 1.0, 17):
                qx = (1 - t) * (1 - t) * x0 + 2 * (1 - t) * t * ctrl_x + t * t * x1
                qy = (1 - t) * (1 - t) * y0 + 2 * (1 - t) * t * ctrl_y + t * t * y1
                samples.append((float(qx), float(qy)))
        else:
            path.lineTo(x1, y1)
            samples = [(float(x0), float(y0)), (float(x1), float(y1))]
        return path, samples

    def _refresh_dynamic_edge_geometry(self, row: int) -> None:
        self.node_to_edge = {}
        for idx, item in enumerate(self.edge_items):
            path, samples = self._edge_path_and_samples(idx, row)
            item.setPath(path)
            self.edge_samples[idx] = samples
            src_display = self._display_node(int(self.edge_table.src[idx]), row)
            dst_display = self._display_node(int(self.edge_table.dst[idx]), row)
            self.node_to_edge[(src_display, dst_display)] = idx
            self.node_to_edge[(dst_display, src_display)] = idx

    def is_hidden_visual_edge(self, idx: int) -> bool:
        if not self.hide_y_wrap_edges:
            return False
        raw_y0, raw_y1 = self._display_edge_raw_y_pair(idx, self.current_row)
        return abs(int(raw_y0) - int(raw_y1)) > int(self.config.N) // 2

    def update_step(self, row: int, *, sync_slider: bool = True):
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        if self.edge_items:
            self._refresh_dynamic_edge_geometry(row)
        super().update_step(row, sync_slider=sync_slider)
        self._refresh_edge_order_labels(row)

    def _refresh_edge_order_labels(self, row: int) -> None:
        show_labels = bool(self.show_edge_labels_checkbox.isChecked()) if hasattr(self, "show_edge_labels_checkbox") else True
        for idx, item in enumerate(self.edge_label_items):
            option = int(self.edge_table.option[idx])
            visible = (
                show_labels
                and self.edge_items[idx].isVisible()
                and bool(self.visible_options.get(option, False))
                and (option != -1 or self.label_intra)
            )
            item.setVisible(bool(visible))
            if not visible:
                continue
            samples = self.edge_samples[idx]
            if not samples:
                item.setVisible(False)
                continue
            x, y = samples[len(samples) // 2]
            rect = item.boundingRect()
            scale = float(item.scale())
            item.setPos(
                float(x) - float(rect.center().x()) * scale,
                float(y) - float(rect.center().y()) * scale,
            )

    def find_nearest_edge(self, x: float, y: float) -> tuple[int | None, float]:
        best_idx: int | None = None
        best_dist = math.inf
        for idx, samples in enumerate(self.edge_samples):
            if idx < len(self.edge_items) and not self.edge_items[idx].isVisible():
                continue
            option = int(self.edge_table.option[idx])
            if not self.visible_options.get(option, False):
                continue
            dist = distance_point_to_polyline(x, y, samples)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None or best_dist > self.edge_hit_threshold:
            return None, best_dist
        return best_idx, best_dist

    def describe_edge(self, edge_idx: int, *, prefix: str) -> str:
        edge_idx = int(edge_idx)
        option = int(self.edge_table.option[edge_idx])
        order = edge_label_text(option, "symbol-option")
        src = int(self.edge_table.src[edge_idx])
        dst = int(self.edge_table.dst[edge_idx])
        src_display = self._display_node(src, self.current_row)
        dst_display = self._display_node(dst, self.current_row)
        sx, sy = divmod(src, int(self.config.N))
        dx, dy = divmod(dst, int(self.config.N))
        dsx, dsy = divmod(src_display, int(self.config.N))
        ddx, ddy = divmod(dst_display, int(self.config.N))
        return (
            f"{prefix}: idx={edge_idx}; order={order}; option={option}; "
            f"raw {src} ({sx},{sy}) -> {dst} ({dx},{dy}); "
            f"display {src_display} ({dsx},{dsy}) -> {dst_display} ({ddx},{ddy}); "
            f"offset={self._offset_for_row(self.current_row)}; step={self.steps[self.current_row]}"
        )


def write_offset_csv(path: Path, offsets_by_step: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "rev_group_offset"])
        for step in sorted(offsets_by_step):
            writer.writerow([int(step), int(offsets_by_step[step])])


def main() -> int:
    args = parse_args()
    steps = list(range(int(args.start), int(args.end) + 1, int(args.stride)))
    if not steps:
        raise ValueError("empty step range")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_table, tiled_result = build_motif_000116_edge_table(include_intra=bool(args.include_intra))
    write_tiled_motif_outputs(tiled_result, out_dir)
    write_edges_csv(edge_table, out_dir / "viewer_edges.csv")

    group_data = load_or_build_group_data(
        xml_file=Path(args.xml_file),
        group_cache_dir=Path(args.group_cache_dir),
        steps=steps,
        station_groups=G60_CONFIG.station_groups,
        total_sats=G60_CONFIG.total_sats,
        constellation_name=G60_CONFIG.name,
        stride=int(args.stride),
        enabled=not bool(args.hide_groups),
        force=bool(args.force_group_cache),
    )
    use_rev = not bool(args.no_rev_group) and bool(group_data)
    offsets_by_step = (
        build_rev_group_offsets(
            group_data,
            p_count=int(G60_CONFIG.P),
            y_count=int(G60_CONFIG.N),
            base_groupid=int(args.rev_group_base_id),
        )
        if use_rev
        else {int(step): 0 for step in steps}
    )
    display_group_data = (
        shift_group_data(group_data, offsets_by_step, y_count=int(G60_CONFIG.N))
        if use_rev
        else group_data
    )
    write_offset_csv(out_dir / "rev_group_offsets.csv", offsets_by_step)
    meta = {
        "motif_id": MOTIF_ID,
        "motif_columns": list(MOTIF_COLUMNS),
        "steps": {"start": int(steps[0]), "end": int(steps[-1]), "count": int(len(steps)), "stride": int(args.stride)},
        "edge_count": int(edge_table.num_edges),
        "inter_edge_count": int(tiled_result.edge_count),
        "include_intra": bool(args.include_intra),
        "use_rev_group": bool(use_rev),
        "rev_group_base_id": int(args.rev_group_base_id),
        "edge_label_mode": str(args.edge_label_mode),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[motif-000116-dynamic-viewer] steps={len(steps)} range={steps[0]}..{steps[-1]} "
        f"edges={edge_table.num_edges} inter_edges={tiled_result.edge_count} "
        f"rev_group={use_rev} group_steps={len(display_group_data)} out_dir={out_dir}",
        flush=True,
    )
    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = EdgeOrderDynamicTopologyViewer(
        G60_CONFIG,
        steps=steps,
        edge_table=edge_table,
        window_title=f"G60 motif_000116 dynamic 2D topology {steps[0]}..{steps[-1]}s",
        group_data=display_group_data,
        show_groups=not bool(args.hide_groups),
        offsets_by_step=offsets_by_step,
        use_rev_offsets=bool(use_rev),
        edge_label_mode=str(args.edge_label_mode),
        label_intra=bool(args.label_intra),
        label_pixel_size=int(args.label_pixel_size),
        label_scale=float(args.label_scale),
    )
    viewer.edge_width = float(args.edge_width)
    viewer.edge_alpha = int(args.edge_alpha)
    viewer.width_slider.setValue(int(max(8, min(50, round(float(args.edge_width) * 1000)))))
    viewer.alpha_slider.setValue(int(max(25, min(190, int(args.edge_alpha)))))

    if int(args.initial_step) in steps:
        row = steps.index(int(args.initial_step))
        viewer.slider.setValue(row)
        viewer.update_step(row)
    else:
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
