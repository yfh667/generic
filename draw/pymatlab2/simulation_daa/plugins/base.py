# simulation_daa/plugins/base.py
import inspect, pkgutil, importlib
from pathlib import Path
from PyQt5 import QtWidgets

class MetricPlugin(QtWidgets.QWidget):
    """所有度量插件的基类"""
    title = "Metric"

    def on_data_loaded(self, bundle: dict):
        """子类实现：bundle 含 group_data / inter_edges / all_edges / pending_edges / steps"""
        pass

def discover_plugins():
    """自动发现 plugins/ 下所有继承 MetricPlugin 的类"""
    from . import __path__ as PLUG_PATH
    classes = []
    for _, modname, _ in pkgutil.iter_modules(PLUG_PATH):
        if not modname.startswith("plugin_"):
            continue
        mod = importlib.import_module(f"{__package__}.{modname}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, MetricPlugin) and obj is not MetricPlugin:
                classes.append(obj)
    return classes
