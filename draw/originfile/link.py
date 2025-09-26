# ===== 标准头：让 Origin 用到你 conda 包，并修复 originpro 的 pd 依赖 =====
import sys, importlib
from pathlib import Path
py_ext_path = Path(r"C:\ProgramData\miniconda3\envs\graph_ga\Lib\site-packages")
if py_ext_path.exists() and str(py_ext_path) not in sys.path:
    sys.path.insert(0, str(py_ext_path))

import pandas as pd, numpy as np, originpro as op
opw = importlib.import_module("originpro.worksheet")
opw.pd = pd; opw.np = np

# ===== 导入你自己的模块 =====
import genaric2.tegnode as tegnode
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes

# ===== 找到并读取 “Nodes” 表 =====
w = op.find_sheet()
if w is None:
    raise RuntimeError("没有找到激活的工作表，请先点击一次 'Nodes' 表。")
df = w.to_df()
print("Loaded DataFrame:", df.shape)

# ===== 工具函数：找列名（不区分大小写） =====
def pick(df, name):  # 必选
    for c in df.columns:
        if c.strip().lower() == name.lower():
            return c
    raise KeyError(f'Column "{name}" not found. Have: {list(df.columns)}')

def pick_opt(df, name):  # 可选
    for c in df.columns:
        if c.strip().lower() == name.lower():
            return c
    return None

# 必选列
col_step = pick(df, 'step')
col_i    = pick(df, 'i')
col_j    = pick(df, 'j')
col_rni  = pick(df, 'rn_i')
col_rnj  = pick(df, 'rn_j')
col_rns  = pick_opt(df, 'rn_step')   # 可能没有

# 可选列
col_lni     = pick_opt(df, 'ln_i')
col_lnj     = pick_opt(df, 'ln_j')
col_lns     = pick_opt(df, 'ln_step')
col_region  = pick_opt(df, 'asc_nodes_region_id')
col_lstate  = pick_opt(df, 'left_state')
col_rstate  = pick_opt(df, 'right_state')

has_ln  = (col_lni and col_lnj and col_lns)
has_reg = (col_region is not None)

# ---------- 工具：安全取整 & 解析邻居 ----------
def safe_int(x):
    """将 x 转为 int；遇到 None/NaN/''/'None' 都返回 None。"""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return None if np.isnan(x) else int(x)
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def parse_neighbor(row, col_i, col_j, col_s, fallback_step=None):
    """
    从三列解析邻居三元组；只要有一个分量无效就返回 None。
    col_s 允许缺失；若缺失并提供 fallback_step，则用 fallback_step 代替。
    """
    ni = safe_int(row[col_i]) if col_i else None
    nj = safe_int(row[col_j]) if col_j else None
    ns = safe_int(row[col_s]) if col_s else None
    if ns is None and fallback_step is not None:
        ns = int(fallback_step)
    if ni is None or nj is None or ns is None:
        return None
    return (ni, nj, ns)

    
# ===== 转为 totalnode: {(i,j,step) -> tegnode.tegnode_complete} =====
# ===== 转为 totalnode: {(i,j,step) -> tegnode.tegnode_complete} =====
def df_to_totalnode(df, col_step, col_i, col_j,
                    col_rni, col_rnj, col_rns,
                    col_lni=None, col_lnj=None, col_lns=None,
                    col_region=None, col_lstate=None, col_rstate=None):
    total = {}

    # 先把可能的空串清成 NA，然后做数值化（存在的列才处理）
    cols_to_clean = [c for c in [col_step,col_i,col_j,
                                 col_rni,col_rnj,col_rns,
                                 col_lni,col_lnj,col_lns,
                                 col_region,col_lstate,col_rstate] if c]
    df[cols_to_clean] = df[cols_to_clean].apply(
        lambda s: s.replace(r'^\s*$', pd.NA, regex=True) if s.dtype == object else s
    )
    for c in cols_to_clean:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    base = df.dropna(subset=[col_step, col_i, col_j]).copy()
    base[[col_step, col_i, col_j]] = base[[col_step, col_i, col_j]].astype(int)

    for _, row in base.iterrows():
        step = int(row[col_step]); i = int(row[col_i]); j = int(row[col_j])

        # 右邻：若 rn_step 列缺失则用当前 step 兜底；任一分量空则返回 None
        rightneighbor = parse_neighbor(row, col_rni, col_rnj, col_rns, fallback_step=step)

        # 左邻：三列齐且有效才生成，否则 None；若 ln_step 列缺失同样用 step 兜底
        leftneighbor  = parse_neighbor(row, col_lni, col_lnj, col_lns, fallback_step=step) \
                        if (col_lni and col_lnj) else None

        asc_id = safe_int(row[col_region]) if col_region else -1
        lstate = safe_int(row[col_lstate]) if col_lstate else -1
        rstate = safe_int(row[col_rstate]) if col_rstate else -1

        node = tegnode.tegnode_complete(
            asc_nodes_region_id = (-1 if asc_id is None else asc_id),
            rightneighbor = rightneighbor,
            leftneighbor  = leftneighbor,
            left_state    = (-1 if lstate is None else lstate),
            right_state   = (-1 if rstate is None else rstate),
        )
        total[(i, j, step)] = node

    return total


totalnode = df_to_totalnode(
    df,
    col_step=col_step, col_i=col_i, col_j=col_j,
    col_rni=col_rni, col_rnj=col_rnj, col_rns=col_rns,
    col_lni=col_lni, col_lnj=col_lnj, col_lns=col_lns,
    col_region=col_region, col_lstate=col_lstate, col_rstate=col_rstate
)
def densify_totalnode(totalnode, P, N, start_ts, end_ts, tegnode):
    missing = 0
    for t in range(int(start_ts), int(end_ts)):
        for i in range(int(P)):
            for j in range(int(N)):
                key = (i, j, t)
                if key not in totalnode:
                    totalnode[key] = tegnode.tegnode_complete(
                        asc_nodes_region_id=-1,
                        rightneighbor=None,
                        leftneighbor=None,
                        left_state=-1,
                        right_state=-1
                    )
                    missing += 1
    print(f"[FILL] inserted {missing} placeholder node(s)")
    return totalnode

print("totalnode size:", len(totalnode))

# ===== 调用你已有的函数 =====
P, N = 18, 36                # 请改成实际的 P/N
start_ts = int(df[col_step].min())
end_ts   = int(df[col_step].max()) + 1
time_2_build = 30           # 你的建链时长
    
totalnode = densify_totalnode(totalnode, P, N, start_ts, end_ts, tegnode)

all_inter_edge = inter_edge2nodes.trans_nodes2edges(totalnode, P, N)
pending_edges  = inter_edge2nodes.trans_nodes2_pendingedges2(totalnode, start_ts, end_ts, time_2_build, P, N)

# ===== 统计每秒 active/pending =====
def _norm_edge(e):
    if isinstance(e[0], tuple):
        a = (int(e[0][0]), int(e[0][1]))
        b = (int(e[1][0]), int(e[1][1]))
    else:
        a = (int(e[0]), int(e[1]))
        b = (int(e[2]), int(e[3]))
    return (a, b) if a <= b else (b, a)

def count_per_step(edge_dict, start_ts, end_ts):
    out = {}
    for t in range(start_ts, end_ts):
        es = edge_dict.get(t, ()) or ()
        out[t] = len({_norm_edge(e) for e in es})
    return out

active_per_t  = count_per_step(all_inter_edge, start_ts, end_ts)
pending_per_t = count_per_step(pending_edges,  start_ts, end_ts)

perstep = pd.DataFrame({
    "step": list(range(start_ts, end_ts)),
    "active": [active_per_t.get(t, 0) for t in range(start_ts, end_ts)],
    "pending": [pending_per_t.get(t, 0) for t in range(start_ts, end_ts)],
})

# ===== 写回 Origin 新表 =====
def write_sheet(name, dfx):
    try:
        op.find_sheet(name).destroy()
    except:
        pass
    ws = op.new_sheet('w', lname=name)
    ws.from_df(dfx)

write_sheet("PerStep_Active_Pending", perstep)
print("OK: wrote PerStep_Active_Pending. Rows =", len(perstep))
