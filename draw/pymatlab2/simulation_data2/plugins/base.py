# simulation_daa/plugins/base.py
import inspect, pkgutil, importlib
from PyQt5 import QtWidgets

class WindowPlugin:
    """窗口插件基类：最少实现 build()；可选 on_data_loaded(bundle)。"""
    title = "Window"
    placement = "tab"  # tab / dock / window

    def __init__(self, ctx, parent=None):
        self.ctx = ctx
        self.parent = parent

    def build(self) -> QtWidgets.QWidget:
        raise NotImplementedError

    def on_data_loaded(self, bundle: dict):
        pass

def discover_plugins():
    """自动发现 plugins/ 目录下 plugin_*.py 中的 WindowPlugin 子类"""
    from . import __path__ as PKG_PATH, __name__ as PKG_NAME
    classes = []
    for _, modname, _ in pkgutil.iter_modules(PKG_PATH):
        if not modname.startswith("plugin_"):
            continue
        mod = importlib.import_module(f"{PKG_NAME}.{modname}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, WindowPlugin) and cls is not WindowPlugin:
                classes.append(cls)
    return classes
