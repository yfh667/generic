# simulation_daa/widgets/topology_adapter.py
from PyQt5 import QtWidgets
from draw.pyqt_draw.pyqt_main2 import SatelliteViewer

class SatelliteViewerWrapper(SatelliteViewer):
    def set_topology(self, group_data, edges_by_step, pending_by_step=None):
        self.group_data = group_data or {}
        self.steps = sorted(self.group_data.keys()) or sorted(edges_by_step.keys())
        self.full_steps = self.steps
        self.edges_by_step = edges_by_step or {}
        self.pending_links_by_step = pending_by_step or {}
        if self.steps:
            self._apply_steps_and_draw()

def create_topology_widget():
    return SatelliteViewerWrapper({})
