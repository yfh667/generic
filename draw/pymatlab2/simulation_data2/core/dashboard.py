# simulation_daa/core/dashboard.py
from PyQt5 import QtWidgets, QtCore
from .loader import DataLoader
from .config_runtime import P, N
from widgets.satellite_viewer_adapter import create_satellite_viewer
from plugins.base import discover_plugins

class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topology Dashboard")

        # 中央：Tab（第一个是拓扑，其余是插件）
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # 日志 Dock
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        dock = QtWidgets.QDockWidget("日志", self)
        dock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

        # 先放一个空的拓扑视图（等数据完毕再填）
        self.viewer = create_satellite_viewer({})
        self.tabs.addTab(self.viewer, "Topology")

        # 加载数据（后台线程）
        self._start_loader()

    def _start_loader(self):
        self.thread = QtCore.QThread(self)
        self.worker = DataLoader()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self._log)
        self.worker.done.connect(self._on_data_ready)
        self.worker.failed.connect(self._on_failed)
        self.worker.done.connect(lambda _: self.thread.quit())
        self.worker.failed.connect(lambda _: self.thread.quit())
        self.thread.start()

    @QtCore.pyqtSlot(str)
    def _log(self, msg: str):
        self.log.append(msg)

    @QtCore.pyqtSlot(dict)
    def _on_data_ready(self, bundle: dict):
        self._log("渲染拓扑 …")
        # 更新拓扑视图
        self.viewer.set_topology(
            group_data=bundle["group_data"],
            edges_by_step=bundle["all_edges"],
            pending_by_step=bundle["pending_edges"]
        )

        # 自动发现插件并注入数据
        self._log("加载插件 …")
        for PluginCls in discover_plugins():
            try:
                widget = PluginCls(parent=self)
                title = getattr(widget, "title", PluginCls.__name__)
                self.tabs.addTab(widget, title)
                if hasattr(widget, "on_data_loaded"):
                    widget.on_data_loaded(bundle)
                self._log(f"✓ 插件: {title}")
            except Exception as e:
                self._log(f"✗ 插件加载失败 {PluginCls}: {e}")

    @QtCore.pyqtSlot(str)
    def _on_failed(self, err: str):
        self._log(err)
