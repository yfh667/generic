# panels/satellite_viewer_panel.py
from PyQt5 import QtWidgets
from core.models import DataBundle
from core.registry import register_panel
from widgets.satellite_viewer import SatelliteViewer

@register_panel
class SatelliteViewerPanel(QtWidgets.QWidget):
    title = "Topology"

    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.viewer = SatelliteViewer(group_data=None)
        lay.addWidget(self.viewer)

    def set_data(self, bundle: DataBundle):
        # edges
        self.viewer.edges_by_step = bundle.edges_by_step
        self.viewer.pending_links_by_step = bundle.pending_by_step
        # group_data（可为空）
        if bundle.group_data:
            self.viewer.group_data = bundle.group_data
            self.viewer.steps = sorted(bundle.group_data.keys())
        else:
            steps = sorted(set(bundle.edges_by_step) | set(bundle.pending_by_step))
            self.viewer.group_data = {t: {"groups": {}, "all_mentioned": set()} for t in steps}
            self.viewer.steps = steps
        self.viewer.full_steps = self.viewer.steps
        # 刷新
        if hasattr(self.viewer, "_apply_steps_and_draw"):
            self.viewer._apply_steps_and_draw()
