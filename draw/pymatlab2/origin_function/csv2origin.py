# import_csv_to_origin.py
# 将 CSV 导入到 Origin 的 Book1/指定工作表
# 用法示例：
#   python import_csv_to_origin.py --csv "E:/data/a.csv" --sheet Ration
#   python import_csv_to_origin.py --folder "E:/data/csvs" --prefix Run_ --clear

import argparse
from pathlib import Path
import sys
import pandas as pd
import chardet
import csv
import originpro as op
import draw.pymatlab2.origin_function.write2origin as write2origin
# ========= 基础工具 =========

def _detect_encoding(fp: Path, nbytes: int = 200000) -> str:
    """简易编码探测，默认回退 utf-8-sig。"""
    try:
        raw = fp.read_bytes()[:nbytes]
        enc = chardet.detect(raw).get("encoding") or "utf-8-sig"
        return enc
    except Exception:
        return "utf-8-sig"

def _detect_sep(fp: Path, encoding: str) -> str:
    """用 csv.Sniffer 简单猜分隔符；失败则回退逗号."""
    try:
        sample = fp.open("r", encoding=encoding, errors="ignore").read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter or ","
    except Exception:
        return ","

def _read_csv_safely(fp: Path) -> pd.DataFrame:
    enc = _detect_encoding(fp)
    sep = _detect_sep(fp, enc)
    # 常见设置：首行列名；空值处理；自动日期识别可按需打开
    df = pd.read_csv(
        fp,
        encoding=enc,
        sep=sep,
        header=0,
        na_values=["", "NA", "NaN", "null", "NULL"],
        low_memory=False,
        # parse_dates=True,   # 如果你的 CSV 里有日期列，可打开
    )
    return df

# ========= 写入 Origin 的统一函数（用 ensure_sheet） =========

def _ws_get_cols(ws):
    c = getattr(ws, "cols", None)
    return c() if callable(c) else c

def _ws_set_cols(ws, n):
    try:
        if not callable(getattr(ws, "cols", None)):
            ws.cols = n
            return
    except Exception:
        pass
    try:
        ws.cols(n)
    except Exception:
        pass

def _write_df_to_sheet(book_name: str, sheet_name: str, df, clear: bool):
    """
    与你示例保持一致的写入方式：
      1) 用 ensure_sheet 获取/创建并可选清空目标表
      2) 直接 ws.from_df(df) 写入
    """
    ws = write2origin.ensure_sheet(
        book_name=book_name,
        sheet_name=sheet_name,
        clear_existing=clear,
        activate=True
    )

    # 直接写 DataFrame（与你示例一致）
    try:
        ws.from_df(df)
    except TypeError:
        # 极旧版 originpro 若对 from_df 参数签名挑剔，做一次安全兜底
        # 先把列名写入 Long Name，再逐列写数据（仅当上面失败时才执行）
        for i, name in enumerate(map(str, df.columns.tolist())):
            try:
                ws.from_list(i, [name], lname=True)
            except TypeError:
                ws.from_list(i, [name], lname=True, start=0)
        for i in range(df.shape[1]):
            col_data = df.iloc[:, i].tolist()
            try:
                ws.from_list(i, col_data)
            except TypeError:
                ws.from_list(i, col_data, start=0)

    return ws


# === 统一的写入接口（都走 ensure_sheet + ws.from_df） ===

def to_origin(df: pd.DataFrame, *, book: str = "Book1", sheet: str = "Ration", clear: bool = True):
    """把任意 DataFrame 写入 Origin 的指定工作表。"""
    ws = write2origin.ensure_sheet(book_name=book, sheet_name=sheet,
                                   clear_existing=clear, activate=True)
    # 直接写 DataFrame（与你的习惯保持一致）
    try:
        ws.from_df(df)
    except TypeError:
        # 极旧版兼容：列名写到 Long Name，数据逐列填充
        for i, name in enumerate(map(str, df.columns.tolist())):
            try:
                ws.from_list(i, [name], lname=True)
            except TypeError:
                ws.from_list(i, [name], lname=True, start=0)
        for i in range(df.shape[1]):
            col_data = df.iloc[:, i].tolist()
            try:
                ws.from_list(i, col_data)
            except TypeError:
                ws.from_list(i, col_data, start=0)
    return ws

def csv_to_origin(csv_path: Path, *, book: str = "Book1", sheet: str | None = None, clear: bool = True):
    """读取 CSV → DataFrame → 写入 Origin；sheet 为空则用文件名作为 sheet。"""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")
    df = _read_csv_safely(csv_path)
    if sheet is None:
        sheet = csv_path.stem
    return to_origin(df, book=book, sheet=sheet, clear=clear)

def import_csv_to_origin(csv_path: Path, sheet_name: str, book_name: str = "Book1", clear: bool = True):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 不存在：{csv_path}")
    df = _read_csv_safely(csv_path)
    ws = _write_df_to_sheet(book_name, sheet_name, df, clear)
    return ws



# filename = f"pending_edges_ttb{TIME_2_BUILD}"
# FIGURE_DIR = Path(FIGURE_DIR)           # 确保是 Path
# csv1_path = FIGURE_DIR / f"{filename}.csv"
#
# csv2origin.csv_to_origin(csv1_path, book="Book1", sheet=filename, clear=True)