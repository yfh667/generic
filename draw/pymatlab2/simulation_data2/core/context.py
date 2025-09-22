# simulation_daa/core/context.py
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class AppContext:
    """共享数据/服务：插件通过 ctx 读数据，不直接耦合主窗体。"""
    bundle: Dict[str, Any] = field(default_factory=dict)

    def set_bundle(self, b: Dict[str, Any]):
        self.bundle = b or {}

    def get(self, key: str, default=None):
        return self.bundle.get(key, default)
