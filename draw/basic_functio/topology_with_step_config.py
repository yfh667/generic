from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]


@dataclass
class AdjustConfig:
    """
    用来保存 / 读取一次性窗口的配置：
    - start_ts_onestep
    - end_ts_onestep
    - ratio
    - adjust_flag
    """
    start_ts_onestep: int
    end_ts_onestep: int
    ratio: float = 0.3
    adjust_flag: int = 1

    # ---------- 序列化辅助 ----------

    def to_dict(self) -> Dict[str, Any]:
        """转成可以直接 json.dump 的 dict。"""
        return {
            "start_ts_onestep": int(self.start_ts_onestep),
            "end_ts_onestep": int(self.end_ts_onestep),
            "ratio": float(self.ratio),
            "adjust_flag": int(self.adjust_flag),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdjustConfig":
        """从 dict 还原配置对象。"""
        return cls(
            start_ts_onestep=int(data["start_ts_onestep"]),
            end_ts_onestep=int(data["end_ts_onestep"]),
            ratio=float(data.get("ratio", 0.3)),
            adjust_flag=int(data.get("adjust_flag", 1)),
        )

    # ---------- 文件读写接口 ----------

    def save(self, path: PathLike) -> None:
        """
        把当前配置写入到 json 文件里。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: PathLike) -> "AdjustConfig":
        """
        从 json 文件读取配置。
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


# 也可以给你一个函数式封装，方便直接调用
def save_adjust_config(
    path: PathLike,
    start_ts_onestep: int,
    end_ts_onestep: int,
    ratio: float = 0.3,
    adjust_flag: int = 1,
) -> AdjustConfig:
    cfg = AdjustConfig(
        start_ts_onestep=start_ts_onestep,
        end_ts_onestep=end_ts_onestep,
        ratio=ratio,
        adjust_flag=adjust_flag,
    )
    cfg.save(path)
    return cfg


def load_adjust_config(path: PathLike) -> AdjustConfig:
    return AdjustConfig.load(path)
