# simulation_daa/core/window_manager.py
from PyQt5 import QtWidgets, QtCore
from .loader import DataLoader
from .context import AppContext
from plugins.base import discover_plugins

class DashboardWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Topology Dashboard")
        self.ctx = AppContext()

        # 中央区域：Tab 容器
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)

        # 日志 Dock
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        dock = QtWidgets.QDockWidget("日志", self)
        dock.setWidget(self.log)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

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
        self.ctx.set_bundle(bundle)
        self._log("数据已就绪，开始加载插件…")

        # 自动发现并实例化插件
        for PluginCls in discover_plugins():
            try:
                plugin = PluginCls(ctx=self.ctx, parent=self)
                place = getattr(plugin, "placement", "tab")  # tab/dock/window
                title = getattr(plugin, "title", PluginCls.__name__)

                widget = plugin.build()  # 返回 QWidget
                if place == "tab":
                    self.tabs.addTab(widget, title)
                elif place == "dock":
                    dock = QtWidgets.QDockWidget(title, self)
                    dock.setWidget(widget)
                    self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
                elif place == "window":
                    widget.setWindowTitle(title)
                    widget.show()
                else:
                    self.tabs.addTab(widget, title)

                if hasattr(plugin, "on_data_loaded"):
                    plugin.on_data_loaded(bundle)

                self._log(f"✓ 插件：{title} ({place})")
            except Exception as e:
                self._log(f"✗ 插件加载失败 {PluginCls}: {e}")

    @QtCore.pyqtSlot(str)
    def _on_failed(self, err: str):
        self._log(err)
