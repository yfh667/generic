# simulation_daa/plugins/plugin_avg_latency.py
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import networkx as nx
import numpy as np
from .base import WindowPlugin

class _Canvas(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(figsize=(7.5, 5), tight_layout=True)
        super().__init__(fig)
        self.ax1 = fig.add_subplot(211)
        self.ax2 = fig.add_subplot(212, sharex=self.ax1)

class AvgLatencyPlugin(WindowPlugin):
    title = "Avg Hops vs Switches"
    placement = "tab"

    def build(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        self.canvas = _Canvas()
        lay = QtWidgets.QVBoxLayout(w)
        lay.addWidget(self.canvas)
        return w

    def on_data_loaded(self, bundle: dict):
        steps = bundle["steps"]
        all_edges = bundle["all_edges"]
        pending_edges = bundle["pending_edges"]
        group_data = bundle["group_data"]

        if not steps: return

        avg = []
        for t in steps:
            gd = group_data.get(t, {})
            gs = gd.get("groups", {})
            if 0 not in gs or 4 not in gs or not gs[0] or not gs[4]:
                avg.append(np.nan); continue
            G = nx.Graph()
            edges = [(u, v) for u, vs in all_edges.get(t, {}).items() for v in vs]
            if not edges: avg.append(np.nan); continue
            G.add_edges_from(edges)
            g0, g4 = set(gs[0]), set(gs[4])
            s, c = 0, 0
            for u in g0:
                if u not in G: continue
                lens = nx.single_source_shortest_path_length(G, u)
                for v in g4:
                    if v in lens: s += lens[v]; c += 1
            avg.append(s/c if c else np.nan)

        counts = [sum(len(vs) for vs in pending_edges.get(t, {}).values()) for t in steps]
        ax1, ax2 = self.canvas.ax1, self.canvas.ax2
        ax1.clear(); ax2.clear()
        ax1.plot(steps, avg, lw=1.8, color="tab:blue")
        ax1.set_ylabel("Avg hops (0↔4)")
        ax1.grid(True, ls="--", alpha=.3)

        ax2.plot(steps, counts, lw=1.8, color="tab:orange", drawstyle="steps-mid")
        ax2.set_ylabel("Edges/sec")
        ax2.set_xlabel("Time (s)")
        ax2.grid(True, ls="--", alpha=.3)
        self.canvas.draw()
