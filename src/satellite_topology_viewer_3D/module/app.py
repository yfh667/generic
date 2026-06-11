from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from src.config.viewer_config import ViewerConfig
from src.satellite_topology_viewer.module.edge_delay_data import EdgeDelayViewerData

from .position_data import PositionSeries
from .synced_2d3d_viewer import Synced2D3DTopologyWindow


def run_synced_2d3d_viewer(
    *,
    config: ViewerConfig,
    delay_data: EdgeDelayViewerData,
    position_series: PositionSeries,
    group_data: dict[int, dict] | None = None,
    width: int = 1600,
    height: int = 900,
    show_groups: bool = True,
    show_3d_links: bool = True,
    show_3d_orbits: bool = True,
    link_stride: int = 1,
    timer_interval_ms: int = 180,
    check_only: bool = False,
    offscreen: bool = False,
    screenshot: str | Path | None = None,
) -> int:
    if check_only:
        print(
            f"[synced-2d3d] check OK | steps={len(delay_data.steps)} "
            f"edges={delay_data.edge_table.num_edges} position_cache={position_series.cache_dir} "
            f"backend=pyvista",
            flush=True,
        )
        return 0

    if offscreen:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])

    window = Synced2D3DTopologyWindow(
        config=config,
        delay_data=delay_data,
        position_series=position_series,
        group_data=group_data,
        show_groups=show_groups,
        show_3d_links=show_3d_links,
        show_3d_orbits=show_3d_orbits,
        link_stride=link_stride,
        timer_interval_ms=timer_interval_ms,
    )
    window.resize(int(width), int(height))
    window.show()

    def settle_initial_layout() -> None:
        try:
            window.viewer2d.fit_scene()
        except Exception:
            pass
        try:
            window.viewer3d.set_row(window.current_row)
        except Exception:
            pass

    QtCore.QTimer.singleShot(120, settle_initial_layout)

    if screenshot is not None:
        screenshot_path = Path(screenshot)
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        def save_screenshot_and_quit() -> None:
            app.processEvents()
            ok = window.grab().save(str(screenshot_path))
            print(f"[synced-2d3d] screenshot={screenshot_path} ok={ok}", flush=True)
            app.quit()

        QtCore.QTimer.singleShot(2200, save_screenshot_and_quit)

    return int(app.exec_())
