# mod_origin_min.py
# 轻量 Origin 适配：获取工作表 + 读为 pandas.DataFrame
# 依赖：originpro, pandas

from __future__ import annotations
from typing import Any
import pandas as pd
import originpro as op

__all__ = ["get_ws", "ws_to_df"]

def get_ws(book_name: str, sheet_name: str) -> Any:
    """
    在当前 Origin 工程里找到指定工作簿/工作表并返回 Worksheet 对象。
    - 仅查找，不创建；找不到将抛出 RuntimeError。
    """
    bk = op.find_book('w', book_name)
    if bk is None:
        raise RuntimeError(f"找不到工作簿 '{book_name}'")

    target = None
    for w in bk:
        # 优先匹配 Long Name，不存在则回退短名
        lname = getattr(w, "lname", None) or w.name
        if lname == sheet_name:
            target = w
            break

    if target is None:
        raise RuntimeError(f"在 '{book_name}' 中找不到工作表 '{sheet_name}'")

    return target

def ws_to_df(ws: Any) -> pd.DataFrame:
    """
    将 Origin 的 Worksheet 转为 pandas.DataFrame。
    - 首选 ws.to_df()（较新版本支持）
    - 若失败，抛出明确的异常以便调用方决定兜底策略
    """
    try:
        df = ws.to_df()
    except Exception as e:
        raise RuntimeError(
            "当前 originpro 不支持 ws.to_df()，或此方法调用失败。"
            "请升级 originpro，或在上层改用自定义兜底读取逻辑（例如逐列读取）。"
        ) from e

    # 清掉全空列（某些版本可能带出空列）
    return df.dropna(axis=1, how="all")
