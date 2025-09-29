# mod_gap.py
from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional, Sequence

_TIME_CANDIDATES = {"time", "t", "step", "timestamp", "ts"}


def guess_time_and_values(
    df: pd.DataFrame,
    time_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
) -> tuple[str, list[str]]:
    """猜测时间列与值列；也可显式传入。"""
    if time_col is None:
        for c in df.columns:
            if str(c).strip().lower() in _TIME_CANDIDATES:
                time_col = c
                break
        if time_col is None:
            time_col = df.columns[0]

    if value_cols is None:
        value_cols = [c for c in df.columns if c != time_col]
        if not value_cols:
            raise ValueError("需要至少 1 列作为数值列。")

    return time_col, list(value_cols)

def df_fill_time_gaps(
    df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
    value_cols: Optional[Sequence[str]] = None,
    time_start: Optional[int] = 0,
    time_end: Optional[int] = None,
    step: int = 1,
    fill_value: float | int = 0,
    dup_agg: str = "first",  # {'first','last','mean','sum','max','min'}
) -> pd.DataFrame:
    """
    按整数时间轴补齐缺失行；返回新的 DataFrame。
    - 只依赖 pandas，方便单测 & 复用。
    """
    if df.empty:
        raise ValueError("输入 df 为空。")

    tcol, vcols = guess_time_and_values(df, time_col, value_cols)
    out = df.copy()

    # 时间列转 int
    out[tcol] = pd.to_numeric(out[tcol], errors="coerce")
    out = out.dropna(subset=[tcol]).copy()
    out[tcol] = out[tcol].astype(int)

    # 重复时间的聚合
    if dup_agg != "first":
        if dup_agg == "mean":
            out = out.groupby(tcol, as_index=False).mean(numeric_only=True)
        elif dup_agg == "sum":
            out = out.groupby(tcol, as_index=False).sum(numeric_only=True)
        elif dup_agg == "last":
            out = out.sort_values(tcol).drop_duplicates(subset=[tcol], keep="last")
        elif dup_agg == "max":
            out = out.groupby(tcol, as_index=False).max(numeric_only=True)
        elif dup_agg == "min":
            out = out.groupby(tcol, as_index=False).min(numeric_only=True)
        else:
            raise ValueError(f"不支持的 dup_agg: {dup_agg}")
    else:
        out = out.sort_values(tcol).drop_duplicates(subset=[tcol], keep="first")

    if time_end is None:
        time_end = int(out[tcol].max())
    if time_start is None:
        time_start = int(out[tcol].min())

    full_index = pd.RangeIndex(start=int(time_start), stop=int(time_end) + 1, step=int(step))

    out = out.set_index(tcol)
    out = out[vcols]  # 只保留值列
    out_filled = out.reindex(full_index)

    # 数值列填 fill_value；非数值列填 None
    for c in out_filled.columns:
        if pd.api.types.is_numeric_dtype(out_filled[c]):
            out_filled[c] = out_filled[c].fillna(fill_value)
        else:
            out_filled[c] = out_filled[c].fillna(None)

    out_filled = out_filled.reset_index().rename(columns={"index": tcol})
    return out_filled[[tcol, *vcols]]


import  draw.pymatlab2.origin_function.readorigin as readorigin

def origin_fill_time_gaps(
    *,
    book: str,
    sheet: str,
    time_start: int = 0,
    time_end: int | None = None,
    step: int = 1,
    fill_value: float | int = 0,
    dup_agg: str = "first",
    dest_sheet: str | None = None,
    clear_dest: bool = True,
    time_col: str | None = None,
    value_cols: list[str] | None = None,
):
    """一行调用：读取 Origin 工作表 → 按时间补齐 → 写回（默认覆盖原表）。"""
    ws = readorigin.get_ws(book, sheet)
    df = readorigin.ws_to_df(ws)

    # 补齐
    df_filled = df_fill_time_gaps(
        df,
        time_col=time_col,
        value_cols=value_cols,
        time_start=time_start,
        time_end=time_end,
        step=step,
        fill_value=fill_value,
        dup_agg=dup_agg,
    )

    # 写回（默认覆盖原表）

    # csv2origin.write2origin()df_to_origin(df_filled, book=book, sheet=target_sheet, clear=clear_dest)


    #     ws = write2origin.ensure_sheet('Book1', 'Ration', clear_existing=True, activate=True)
    ws.from_df(df_filled)


    return df_filled


# filename = f"pending_edges_ttb{TIME_2_BUILD}"
# fillgap.origin_fill_time_gaps(
#     book="Book1",
#     sheet=filename,     # 原表名
#     time_start=0,
#     step=1,
#     fill_value=0,
#     dest_sheet=None,    # None=覆盖写回原表；也可以 "xxx_filled"
#     clear_dest=True
# )