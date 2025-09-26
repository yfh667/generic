# 1) 先把你的 Conda 包路径放到 sys.path
import sys, importlib
from pathlib import Path
py_ext_path = Path(r"C:\ProgramData\miniconda3\envs\graph_ga\Lib\site-packages")
if py_ext_path.exists() and str(py_ext_path) not in sys.path:
    sys.path.insert(0, str(py_ext_path))

# 2) 导入 pandas / numpy
import pandas as pd
import numpy as np

# 3) 导入 originpro，并把 pd/np 注入到 originpro.worksheet 模块
import originpro as op
opw = importlib.import_module("originpro.worksheet")
opw.pd = pd          # 关键：让 to_df() 能看到 pd
opw.np = np          # 有些函数也会用到 np

# 4) 现在就可以安全调用 to_df()
w = op.find_sheet()  # 当前激活工作表
print("Active sheet:", w.name)
df = w.to_df()
print("Rows x Cols =", df.shape)
