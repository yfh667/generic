# simulation_daa/plugins/plugin_topology.py
from PyQt5 import QtWidgets
from .base import WindowPlugin
from widgets.topology_adapter import create_topology_widget

class TopologyPlugin(WindowPlugin):
    title = "Topology"
    placement = "tab"

    def build(self) -> QtWidgets.QWidget:
        self.widget = create_topology_widget()
        return self.widget

    def on_data_loaded(self, bundle: dict):
        self.widget.set_topology(
            group_data=bundle["group_data"],
            edges_by_step=bundle["all_edges"],
            pending_by_step=bundle["pending_edges"]
        )
