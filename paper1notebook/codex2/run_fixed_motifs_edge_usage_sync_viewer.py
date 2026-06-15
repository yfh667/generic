from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PyQt5 import QtCore, QtWidgets


THIS_DIR = Path(__file__).resolve().parent
GENERIC_ROOT = THIS_DIR.parents[1]
if str(GENERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERIC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from weighted_base_viewer import EdgeUsageTopology2DViewer

from src.satellite_topology_viewer.module.app import run_viewer_widget
from src.satellite_topology_viewer.module.region_groups import load_or_build_group_data
from src.topology_metrics.module.edge_betweenness_store import compute_edge_betweenness_store
from src.topology_metrics.module.stores import MetricStoreLayout
from src.topology_workflow.module.batch_shortest_hops import TopologySpec, topology_specs_from_motif_csv
from src.topology_workflow.module.config import load_workflow_yaml, time_axis_from_config, viewer_config_from_workflow


DEFAULT_CONFIG = GENERIC_ROOT / "paper1notebook" / "pipeline" / "configs" / "g60_w_le4_h_le3_shortest_hops.yaml"
DEFAULT_MOTIFS = ("56", "40", "61")
DEFAULT_PAIRS = ("china_europe",)


@dataclass(frozen=True)
class RegionPair:
    key: str
    label: str
    source_group_id: int
    target_group_id: int


@dataclass
class WeightedMotifPanelData:
    spec: TopologySpec
    values: np.ndarray
    max_value: float
    cache_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show fixed motif topologies with per-step region-pair shortest-path edge usage. "
            "This script uses the copied codex2 weighted viewer, not src/base_viewer.py."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--motifs", nargs="+", default=list(DEFAULT_MOTIFS))
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--force-metric-cache", action="store_true")
    parser.add_argument("--show-panel-controls", action="store_true")
    parser.add_argument("--no-groups", action="store_true")
    parser.add_argument("--force-group-cache", action="store_true")
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--height", type=int, default=980)
    parser.add_argument("--edge-width", type=float, default=0.045)
    parser.add_argument("--edge-alpha", type=int, default=190)
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
    if not token:
        raise ValueError("empty motif token")
    if token.isdigit():
        return f"{name_prefix}_motif_{int(token):06d}"
    if token.startswith("motif_") and token.removeprefix("motif_").isdigit():
        return f"{name_prefix}_{token}"
    if token.startswith(f"{name_prefix}_motif_"):
        return token
    if token.startswith("combined_motif_"):
        return token
    raise ValueError(
        f"unsupported motif token {raw!r}; use 56, motif_000056, or {name_prefix}_motif_000056"
    )


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


def motif_label(spec: TopologySpec) -> str:
    motif_id = "?" if spec.motif_id is None else f"{int(spec.motif_id):06d}"
    return f"motif {motif_id} | {spec.motif}"


def load_expanded_edge_usage(cache_dir: Path) -> np.ndarray:
    layout = MetricStoreLayout(Path(cache_dir))
    unique_values = np.load(layout.unique_state_values_npy, mmap_mode="r")
    state_ids = np.load(layout.state_ids_npy)
    return np.asarray(unique_values[np.asarray(state_ids, dtype=np.int64), :], dtype=np.float32)


def metric_cache_dir(
    *,
    out_dir: Path,
    pair_key: str,
    topology_name: str,
    start: int,
    end: int,
    stride: int,
) -> Path:
    return (
        out_dir
        / "edge_usage_viewer_cache"
        / f"t{int(start)}_{int(end)}_stride{int(stride)}"
        / str(pair_key)
        / str(topology_name)
    )


def prepare_weighted_panel_data(
    *,
    pair: RegionPair,
    specs: list[TopologySpec],
    config,
    group_data: dict,
    steps: list[int],
    run_out_dir: Path,
    start: int,
    end: int,
    stride: int,
    workers: int,
    progress_every: int,
    force_metric_cache: bool,
) -> list[WeightedMotifPanelData]:
    panels: list[WeightedMotifPanelData] = []
    for spec in specs:
        cache_dir = metric_cache_dir(
            out_dir=run_out_dir,
            pair_key=pair.key,
            topology_name=spec.name,
            start=start,
            end=end,
            stride=stride,
        )
        compute_edge_betweenness_store(
            topology_name=spec.name,
            edge_table=spec.edge_table,
            total_nodes=config.total_sats,
            group_data=group_data,
            steps=steps,
            source_group_id=pair.source_group_id,
            target_group_id=pair.target_group_id,
            out_dir=cache_dir,
            workers=int(workers),
            progress_every=int(progress_every),
            force=bool(force_metric_cache),
            expand_full_matrix=False,
            sample_path_limit=0,
            extra_meta={
                "pair_key": pair.key,
                "pair_label": pair.label,
                "motif": spec.motif,
                "support": spec.support,
                "runner": str(Path(__file__).resolve()),
            },
        )
        values = load_expanded_edge_usage(cache_dir)
        panels.append(
            WeightedMotifPanelData(
                spec=spec,
                values=values,
                max_value=float(np.nanmax(values)) if values.size else 0.0,
                cache_dir=cache_dir,
            )
        )
    global_max = max((panel.max_value for panel in panels), default=0.0)
    for panel in panels:
        panel.max_value = float(global_max)
    return panels


class PairTab(QtWidgets.QWidget):
    def __init__(
        self,
        *,
        pair: RegionPair,
        config,
        steps: list[int],
        panel_data: list[WeightedMotifPanelData],
        group_data: dict,
        show_groups: bool,
        edge_width: float,
        edge_alpha: int,
        show_panel_controls: bool,
    ):
        super().__init__()
        self.pair = pair
        self.steps = [int(x) for x in steps]
        self.viewers: list[EdgeUsageTopology2DViewer] = []
        self._syncing = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        self.prev_btn = QtWidgets.QPushButton("<")
        self.next_btn = QtWidgets.QPushButton(">")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.steps) - 1))
        self.slider.setValue(0)
        self.jump_input = QtWidgets.QLineEdit()
        self.jump_input.setPlaceholderText("step")
        self.jump_input.setFixedWidth(120)
        self.jump_btn = QtWidgets.QPushButton("jump")
        self.fit_btn = QtWidgets.QPushButton("fit all")
        self.group_checkbox = QtWidgets.QCheckBox("groups")
        self.group_checkbox.setChecked(bool(show_groups))
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setMinimumWidth(390)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.slider, stretch=1)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.jump_input)
        controls.addWidget(self.jump_btn)
        controls.addWidget(self.fit_btn)
        controls.addWidget(self.group_checkbox)
        controls.addWidget(self.status_label)

        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.splitter.setHandleWidth(8)
        layout.addWidget(self.splitter, stretch=1)

        value_max = max((panel.max_value for panel in panel_data), default=0.0)
        for panel in panel_data:
            frame = QtWidgets.QFrame()
            frame_layout = QtWidgets.QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.setSpacing(4)
            title = QtWidgets.QLabel(
                f"{motif_label(panel.spec)} | max edge usage in this pair={panel.max_value:.3f}"
            )
            title.setStyleSheet("font-size: 15px; font-weight: 700; color: #1f2933;")
            frame_layout.addWidget(title)
            viewer = EdgeUsageTopology2DViewer(
                config,
                steps=self.steps,
                edge_table=panel.spec.edge_table,
                edge_usage_values=panel.values,
                value_max=value_max,
                window_title=f"{pair.label} edge usage | {panel.spec.name}",
                group_data=group_data,
                show_groups=show_groups,
            )
            viewer.edge_width = float(edge_width)
            viewer.edge_alpha = int(edge_alpha)
            viewer.width_slider.setValue(int(max(8, min(85, round(float(edge_width) * 1000)))))
            viewer.alpha_slider.setValue(int(max(25, min(220, int(edge_alpha)))))
            if not bool(show_panel_controls):
                viewer.controls_scroll.hide()
                viewer.main_splitter.setSizes([1000, 0])
            viewer.update_step(0)
            self.viewers.append(viewer)
            frame_layout.addWidget(viewer, stretch=1)
            self.splitter.addWidget(frame)
        self.splitter.setSizes([1 for _ in panel_data])

        self.prev_btn.clicked.connect(lambda: self.set_row(self.current_row() - 1))
        self.next_btn.clicked.connect(lambda: self.set_row(self.current_row() + 1))
        self.slider.valueChanged.connect(self.set_row)
        self.jump_btn.clicked.connect(self.jump_to_step)
        self.jump_input.returnPressed.connect(self.jump_to_step)
        self.fit_btn.clicked.connect(self.fit_all)
        self.group_checkbox.stateChanged.connect(self.set_groups_visible)
        self.set_row(0)

    def current_row(self) -> int:
        return int(self.slider.value())

    def set_row(self, row: int):
        if not self.steps:
            return
        row = int(max(0, min(int(row), len(self.steps) - 1)))
        if self._syncing:
            return
        self._syncing = True
        try:
            if self.slider.value() != row:
                self.slider.blockSignals(True)
                self.slider.setValue(row)
                self.slider.blockSignals(False)
            for viewer in self.viewers:
                viewer.update_step(row)
            step = self.steps[row]
            self.status_label.setText(f"{self.pair.label} | step {step}s | row {row + 1}/{len(self.steps)}")
        finally:
            self._syncing = False

    def jump_to_step(self):
        text = self.jump_input.text().strip()
        if not text:
            return
        try:
            target = int(float(text))
        except ValueError:
            return
        best_row = min(range(len(self.steps)), key=lambda idx: abs(int(self.steps[idx]) - target))
        self.set_row(best_row)

    def fit_all(self):
        for viewer in self.viewers:
            viewer.reset_view()

    def set_groups_visible(self):
        checked = bool(self.group_checkbox.isChecked())
        for viewer in self.viewers:
            viewer.show_groups_checkbox.setChecked(checked)
            viewer.update_step(self.current_row())


class MultiPairWeightedWindow(QtWidgets.QWidget):
    def __init__(self, *, config_name: str, tabs: list[tuple[RegionPair, PairTab]]):
        super().__init__()
        self.setWindowTitle(f"{config_name} fixed motif edge-usage comparison")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)
        for pair, tab in tabs:
            self.tabs.addTab(tab, pair.label)


def main() -> int:
    args = parse_args()
    workflow = load_workflow_yaml(args.config)
    config = viewer_config_from_workflow(workflow)
    cfg_start, cfg_end, cfg_stride = time_axis_from_config(workflow)
    start = cfg_start if args.start is None else int(args.start)
    end = cfg_end if args.end is None else int(args.end)
    stride = cfg_stride if args.stride is None else int(args.stride)
    if end < start:
        raise ValueError("--end must be >= --start")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    steps = list(range(start, end + 1, stride))

    paths_raw = workflow.get("paths", {})
    motif_raw = workflow.get("motif_library", {})
    name_prefix = str(motif_raw.get("name_prefix", "combined"))
    motif_csv = path_from(paths_raw, "motif_library_dir") / str(motif_raw.get("csv_name"))
    run_out_dir = path_from(paths_raw, "out_dir")

    wanted_names = [normalize_motif_name(token, name_prefix=name_prefix) for token in args.motifs]
    specs = topology_specs_from_motif_csv(
        motif_csv,
        config=config,
        library=name_prefix,
        name_prefix=name_prefix,
        add_intra_ring=True,
        wrap_planes=False,
    )
    specs_by_name = {spec.name: spec for spec in specs}
    missing = [name for name in wanted_names if name not in specs_by_name]
    if missing:
        raise ValueError(f"motifs not found in library {motif_csv}: {missing}")
    selected_specs = [specs_by_name[name] for name in wanted_names]

    pair_by_key = region_pairs_from_workflow(workflow)
    missing_pairs = [key for key in args.pairs if key not in pair_by_key]
    if missing_pairs:
        raise ValueError(f"unknown pairs {missing_pairs}; available={sorted(pair_by_key)}")
    selected_pairs = [pair_by_key[key] for key in args.pairs]

    show_groups = not bool(args.no_groups)
    group_data = {}
    if show_groups or selected_pairs:
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

    print(
        f"[edge-usage-sync-viewer] steps={len(steps)}, range={start}..{end}, stride={stride}, "
        f"motifs={', '.join(wanted_names)}, pairs={', '.join(pair.key for pair in selected_pairs)}",
        flush=True,
    )

    tab_payloads: list[tuple[RegionPair, list[WeightedMotifPanelData]]] = []
    for pair in selected_pairs:
        print(f"[edge-usage-sync-viewer] preparing pair={pair.key} {pair.label}", flush=True)
        panel_data = prepare_weighted_panel_data(
            pair=pair,
            specs=selected_specs,
            config=config,
            group_data=group_data,
            steps=steps,
            run_out_dir=run_out_dir,
            start=start,
            end=end,
            stride=stride,
            workers=int(args.workers),
            progress_every=int(args.progress_every),
            force_metric_cache=bool(args.force_metric_cache),
        )
        tab_payloads.append((pair, panel_data))
        for panel in panel_data:
            print(
                f"[edge-usage-sync-viewer] {pair.key}/{panel.spec.name}: "
                f"edges={panel.spec.edge_table.num_edges}, max_usage={panel.max_value:.3f}, "
                f"cache={panel.cache_dir}",
                flush=True,
            )

    if args.check_only:
        return 0

    if args.offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    tabs: list[tuple[RegionPair, PairTab]] = []
    for pair, panel_data in tab_payloads:
        tabs.append(
            (
                pair,
                PairTab(
                    pair=pair,
                    config=config,
                    steps=steps,
                    panel_data=panel_data,
                    group_data=group_data,
                    show_groups=show_groups,
                    edge_width=float(args.edge_width),
                    edge_alpha=int(args.edge_alpha),
                    show_panel_controls=bool(args.show_panel_controls),
                ),
            )
        )

    window = MultiPairWeightedWindow(config_name=config.name, tabs=tabs)
    return run_viewer_widget(
        window,
        width=int(args.width),
        height=int(args.height),
        check_only=False,
        offscreen=bool(args.offscreen),
        screenshot=args.screenshot,
    )


if __name__ == "__main__":
    raise SystemExit(main())
