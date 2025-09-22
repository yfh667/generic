# simulation_daa/plugins/plugin_switch_counts.py
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from .base import WindowPlugin

class _Canvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(6, 3.5), tight_layout=True)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)

class SwitchCountsPlugin(WindowPlugin):
    title = "Switch counts"
    placement = "dock"  # 放到主窗口 Dock

    def build(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        self.canvas = _Canvas()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(self.canvas)
        return w

    def on_data_loaded(self, bundle: dict):
        steps = bundle["steps"]
        pend = bundle["pending_edges"]
        y = [sum(len(vs) for vs in pend.get(t, {}).values()) for t in steps]
        ax = self.canvas.ax
        ax.clear()
        ax.plot(steps, y, lw=1.8, color="tab:green")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Edges/sec")
        ax.grid(True, ls="--", alpha=.35)
        self.canvas.draw()
