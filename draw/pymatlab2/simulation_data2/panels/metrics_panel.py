# panels/metrics_panel.py
from typing import Dict, Set, Tuple, List
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from core.models import DataBundle
from core.registry import register_panel

def _count_unique_undirected(m: Dict[int, Set[int]]) -> int:
    seen, c = set(), 0
    for u, dsts in m.items():
        for v in dsts:
            if u == v: continue
            a, b = (u, v) if u < v else (v, u)
            if (a, b) not in seen:
                seen.add((a, b)); c += 1
    return c

@register_panel
class MetricsPanel(QtWidgets.QWidget):
    title = "Metrics"

    def __init__(self):
        super().__init__()
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self._edges_by_step = {}
        self._pending_by_step = {}

        layout = QtWidgets.QVBoxLayout(self)
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)
        self.smooth = QtWidgets.QSpinBox(); self.smooth.setRange(1, 301); self.smooth.setValue(5)
        self.smooth.setSingleStep(2)
        ctrl.addWidget(QtWidgets.QLabel("Smooth:"))
        ctrl.addWidget(self.smooth); ctrl.addStretch(1)

        self.plot = pg.PlotWidget()
        layout.addWidget(self.plot)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.cur_all = self.plot.plot([], [], pen=pg.mkPen("#1f77b4", width=2), name="built")
        self.cur_pen = self.plot.plot([], [], pen=pg.mkPen("#ff7f0e", width=2, style=QtCore.Qt.DashLine), name="pending")
        self.plot.addLegend()
        self.smooth.valueChanged.connect(self.redraw)

    def set_data(self, bundle: DataBundle):
        self._edges_by_step = bundle.edges_by_step or {}
        self._pending_by_step = bundle.pending_by_step or {}
        self.redraw()

    def _counts(self, m: Dict[int, Dict[int, Set[int]]]) -> Tuple[List[int], List[int]]:
        steps = sorted(m.keys())
        y = [_count_unique_undirected(m[t]) for t in steps]
        return steps, y

    def _smooth(self, y: List[int], w: int) -> List[float]:
        if w <= 1 or len(y) < 3: return y
        import numpy as np
        y = np.asarray(y, float)
        pad = w // 2
        ypad = np.pad(y, (pad, pad), mode='edge')
        ker = np.ones(w) / w
        return np.convolve(ypad, ker, mode='valid').tolist()

    def redraw(self):
        if not self._edges_by_step:
            self.cur_all.setData([], [])
            self.cur_pen.setData([], [])
            return
        sa, ya = self._counts(self._edges_by_step)
        sp, yp = self._counts(self._pending_by_step) if self._pending_by_step else ([], [])
        w = self.smooth.value()
        ya = self._smooth(ya, w)
        yp = self._smooth(yp, w) if sp else []
        self.cur_all.setData(sa, ya)
        self.cur_pen.setData(sp, yp)
