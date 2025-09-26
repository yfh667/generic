# ===== 标准头：让 Origin 用到你 conda 包，并修复 originpro 的 pd 依赖 =====
import sys
import importlib
from pathlib import Path
py_ext_path = Path(r"C:\ProgramData\miniconda3\envs\graph_ga\Lib\site-packages")
if py_ext_path.exists() and str(py_ext_path) not in sys.path:
    sys.path.insert(0, str(py_ext_path))

import pandas as pd, numpy as np, originpro as op
opw = importlib.import_module("originpro.worksheet")
opw.pd = pd; opw.np = np


# 读取当前激活工作簿（Book）里所有 pending_edges_ttbXX 工作表
# 合并为一个 DataFrame，并回写成新的工作表 pending_edges_all

import re
 
 
 
 
# 读取当前激活的工作表
w = op.find_sheet()
if w is None:
    raise RuntimeError("没有找到激活的工作表，请先用鼠标点击你要处理的那张表。")

sheet_name = getattr(w, "lname", None) or w.name
print("当前表：", sheet_name)

df = w.to_df()          # 转成 pandas.DataFrame
print("形状：", df.shape)



print(df.head(5))       # 看前5行