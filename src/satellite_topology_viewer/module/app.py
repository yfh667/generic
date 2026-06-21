from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module.edge_delay_data import EdgeDelayViewerData
from src.satellite_topology_viewer.module.edge_delay_viewer import EdgeDelayTopologyViewer
from src.satellite_topology_viewer.module.link_switch_viewer import LinkSwitchTopologyViewer, LinkSwitchViewerData


def run_viewer_widget(
    viewer: QtWidgets.QWidget,
    *,
    width: int = 1200,
    height: int = 760,
    check_only: bool = False,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
) -> int:
    if check_only:
        return 0

    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer.resize(int(width), int(height))
    viewer.show()

    if screenshot is not None:
        screenshot_path = Path(screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        def save_screenshot_and_quit():
            app.processEvents()
            ok = viewer.grab().save(str(screenshot_path))
            print(f"[sat-topology-viewer] screenshot={screenshot_path} ok={ok}", flush=True)
            app.quit()

        QtCore.QTimer.singleShot(800, save_screenshot_and_quit)

    return int(app.exec_())


def run_edge_delay_viewer(
    *,
    config: ViewerConfig,
    delay_data: EdgeDelayViewerData,
    group_data: dict[int, dict] | None = None,
    width: int = 1200,
    height: int = 760,
    window_title: str | None = None,
    show_groups: bool = True,
    check_only: bool = False,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
) -> int:
    print(
        f"[sat-topology-viewer] edge-delay view | steps={len(delay_data.steps)}, "
        f"edges_per_step={delay_data.edge_table.num_edges}, "
        f"delay_ms=({delay_data.delay_min_ms:.4f}, {delay_data.delay_max_ms:.4f}), "
        f"group_steps={len(group_data or {})}, store={delay_data.store_dir}",
        flush=True,
    )
    if check_only:
        return 0

    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    if window_title is None and delay_data.steps:
        window_title = (
            f"{config.name} full-option ISL propagation delay "
            f"{delay_data.steps[0]}..{delay_data.steps[-1]}s"
        )

    viewer = EdgeDelayTopologyViewer(
        config,
        steps=delay_data.steps,
        edge_table=delay_data.edge_table,
        delay_ms=delay_data.delay_ms,
        delay_min_ms=delay_data.delay_min_ms,
        delay_max_ms=delay_data.delay_max_ms,
        window_title=window_title or f"{config.name} edge-delay topology",
        group_data=group_data or {},
        show_groups=show_groups,
    )
    return run_viewer_widget(
        viewer,
        width=width,
        height=height,
        check_only=check_only,
        offscreen=offscreen,
        screenshot=screenshot,
    )


def run_link_switch_viewer(
    *,
    config: ViewerConfig,
    data: LinkSwitchViewerData,
    group_data: dict[int, dict] | None = None,
    value_mode: str = "secondary",
    working_mode: str = "any",
    inspector_mode: str = "window",
    topology_motif_id=None,
    width: int = 1500,
    height: int = 900,
    window_title: str | None = None,
    show_groups: bool = True,
    check_only: bool = False,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
    **viewer_kwargs,
) -> int:
    print(
        f"[sat-topology-viewer] link-switch view | steps={len(data.steps)}, "
        f"edges={data.edge_table.num_edges}, value_mode={value_mode}, "
        f"groups={len(group_data or {})}",
        flush=True,
    )
    if check_only:
        return 0

    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    viewer = LinkSwitchTopologyViewer(
        config,
        data=data,
        value_mode=str(value_mode),
        working_mode=str(working_mode),
        inspector_mode=str(inspector_mode),
        topology_motif_id=topology_motif_id,
        window_title=window_title or f"{config.name} link-switch topology",
        group_data=group_data or {},
        show_groups=show_groups,
        **viewer_kwargs,
    )
    return run_viewer_widget(
        viewer,
        width=width,
        height=height,
        check_only=check_only,
        offscreen=offscreen,
        screenshot=screenshot,
    )
