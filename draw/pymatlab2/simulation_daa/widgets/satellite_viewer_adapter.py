# simulation_daa/widgets/satellite_viewer_adapter.py
from PyQt5 import QtWidgets
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer  # 你现有的类

def create_satellite_viewer(group_data):
    v = SatelliteViewer(group_data)
    v.setWindowTitle("Topology")
    return v

class SatelliteViewerWrapper(SatelliteViewer):
    def set_topology(self, group_data, edges_by_step, pending_by_step=None):
        self.group_data = group_data or {}
        self.steps = sorted(self.group_data.keys()) or sorted(edges_by_step.keys())
        self.full_steps = self.steps
        self.edges_by_step = edges_by_step or {}
        self.pending_links_by_step = pending_by_step or {}
        # 启用/刷新
        if self.steps:
            self._apply_steps_and_draw()

def create_satellite_viewer(group_data):
    return SatelliteViewerWrapper(group_data or {})
