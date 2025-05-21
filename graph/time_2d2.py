# -*- coding: utf-8 -*-
"""
Dynamic Graph Visualizer
=======================

**Changelog**
-------------
* **2025‑05‑21 (c)** – *Qt‑first backend strategy*
  • Matplotlib now **prefers a Qt backend** (`QtAgg` / `Qt5Agg` / `Qt6Agg`).  This
    shares the same event‑loop family as **Vedo (VTK‑Qt)**, eliminating the
    GIL‑related crash when both the 2‑D slider window and the 3‑D Vedo window
    are open simultaneously.
  • Fallback sequence:
    1. Qt (PySide6/PySide2/PyQt6/PyQt5 detected)
    2. Tk (if available)
    3. Agg (headless‑PNG only)
  • No API changes – `vis.show()` is still non‑blocking by default.

Input data
~~~~~~~~~~
```
adjacency_list_array : List[List[Tuple[int,int]]]
region_groups_array  : List[List[List[int]]]
N, P : int
```
Arrays must have identical length.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import List, Tuple, Sequence

import matplotlib

# -------------------------------------------------------------
# Backend selection: Qt → Tk → Agg
# -------------------------------------------------------------
_BACKEND_SELECTED = ""

# helper: check importability
_def_mods = (
    ("PySide6", "QtAgg"),
    ("PyQt6", "QtAgg"),
    ("PySide2", "Qt5Agg"),
    ("PyQt5", "Qt5Agg"),
)
for mod_name, backend in _def_mods:
    if importlib.util.find_spec(mod_name) is not None:
        matplotlib.use(backend, force=True)
        _BACKEND_SELECTED = backend
        break
else:  # try Tk next
    try:
        import tkinter  # noqa: F401
        matplotlib.use("TkAgg", force=True)
        _BACKEND_SELECTED = "TkAgg"
    except ImportError:
        matplotlib.use("Agg", force=True)
        _BACKEND_SELECTED = "Agg"

print(f"[DynamicGraphVisualizer] matplotlib backend → {_BACKEND_SELECTED}")
_INTERACTIVE = _BACKEND_SELECTED != "Agg"

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx


class DynamicGraphVisualizer:
    """Time-series visualiser for satellite topologies with curved edge support."""

    _COLORS = [
        "lightcoral", "lightgreen", "plum", "sandybrown",
        "khaki", "teal", "orchid", "salmon",
    ]

    def __init__(
        self,
        adjacency_list_array: Sequence[Sequence[Tuple[int, int]]],
        region_groups_array: Sequence[Sequence[Sequence[int]]],
        N: int,
        P: int,
        default_rad: float = 0.3,
        tol: float = 1e-9,
    ) -> None:
        if len(adjacency_list_array) != len(region_groups_array):
            raise ValueError("adjacency_list_array and region_groups_array must be the same length")
        self._adj = adjacency_list_array
        self._regions = region_groups_array
        self.N, self.P = N, P
        self._rad, self._tol = default_rad, tol
        self._pos = {i: (i // N, i % N) for i in range(N * P)}

        self._fig, self._ax = plt.subplots(figsize=(max(6, P * 1.2), max(4, N * 1.2)))
        plt.subplots_adjust(bottom=0.25)

        if _INTERACTIVE:
            slider_ax = plt.axes([0.2, 0.1, 0.65, 0.03])
            self._slider = Slider(slider_ax, "Time-step", 0, len(self._adj) - 1, valinit=0, valstep=1)
            self._slider.on_changed(self._update)
        else:
            self._slider = None

        self._artists: List = []
        self._setup_axes()
        self._update(0)

    def show(self, block: bool = False):
        if _INTERACTIVE:
            plt.show(block=block)
        else:
            out = Path("graph.png")
            self._fig.savefig(out, dpi=300)
            print(f"[DynamicGraphVisualizer] Saved first frame to {out.resolve()}")

    def _setup_axes(self):
        self._ax.set_xlim(-0.5, self.P - 0.5)
        self._ax.set_ylim(-0.5, self.N - 0.5)
        self._ax.set_xticks(range(self.P))
        self._ax.set_yticks(range(self.N))
        self._ax.set_xlabel("Orbit Plane Index (x)")
        self._ax.set_ylabel("Satellite Index in Plane (y)")
        self._ax.grid(True, linestyle="--", alpha=0.4)
        self._ax.set_aspect("equal", adjustable="box")

    def _graph(self, step: int) -> nx.Graph:
        g = nx.Graph()
        g.add_nodes_from(range(self.N * self.P))
        g.add_edges_from(self._adj[step])
        return g

    def _edge_split(self, g: nx.Graph):
        straight, curved = [], []
        for u, v in g.edges():
            (x1, y1), (x2, y2) = self._pos[u], self._pos[v]
            if abs(y1 - y2) <= self._tol and abs(x1 - x2) > 1 + self._tol:
                curved.append((u, v))
            else:
                straight.append((u, v))
        return straight, curved

    def _colors(self, step: int):
        col = ["skyblue"] * (self.N * self.P)
        for idx, reg in enumerate(self._regions[step]):
            c = self._COLORS[idx % len(self._COLORS)]
            for n in reg:
                if 0 <= n < self.N * self.P:
                    col[n] = c
        return col

    def _clear(self):
        for art in self._artists:
            if isinstance(art, list):
                for subart in art:
                    if hasattr(subart, 'remove'):
                        subart.remove()
            elif hasattr(art, 'remove'):
                art.remove()
        self._artists.clear()

    def _update(self, val):
        step = int(val)
        self._clear()

        g = self._graph(step)
        straight, curved = self._edge_split(g)
        node_col = self._colors(step)

        if straight:
            art = nx.draw_networkx_edges(g, self._pos, ax=self._ax,
                                          edgelist=straight, edge_color="black", alpha=0.6)
            if art is not None:
                self._artists.append(art)

        if curved:
            connection_style = f"arc3,rad={self._rad}"
            art = nx.draw_networkx_edges(
                g, self._pos, ax=self._ax,
                edgelist=curved, edge_color="red",
                connectionstyle=connection_style, width=1.5,
                arrows=True, arrowstyle='-'
            )
            if art is not None:
                self._artists.append(art)

        nodes = nx.draw_networkx_nodes(
            g, self._pos, ax=self._ax,
            node_color=node_col, node_size=500
        )
        if nodes is not None:
            self._artists.append(nodes)

        labels = nx.draw_networkx_labels(
            g, self._pos, ax=self._ax,
            font_size=8, font_weight="bold"
        )
        self._artists.extend(label for label in labels.values() if hasattr(label, 'remove'))

        self._ax.set_title(f"Time-step {step}")
        self._fig.canvas.draw_idle()
