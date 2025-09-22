# core/registry.py
import importlib, pkgutil
from typing import List, Type
from PyQt5.QtWidgets import QWidget

_PANEL_REGISTRY: List[Type[QWidget]] = []

def register_panel(cls: Type[QWidget]):
    """装饰器：注册一个面板类。要求类属性 title:str 存在，并实现 set_data(bundle)。"""
    if cls not in _PANEL_REGISTRY:
        _PANEL_REGISTRY.append(cls)
    return cls

def get_registered_panels() -> List[Type[QWidget]]:
    return list(_PANEL_REGISTRY)

def load_plugins(package_name: str):
    """
    自动发现并 import `package_name` 包下的所有模块，
    模块里使用 @register_panel 的类会注册进来。
    """
    pkg = importlib.import_module(package_name)
    prefix = pkg.__name__ + "."
    for m in pkgutil.iter_modules(pkg.__path__, prefix):
        importlib.import_module(m.name)
