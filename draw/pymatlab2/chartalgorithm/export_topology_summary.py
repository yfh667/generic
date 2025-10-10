from __future__ import annotations
from pathlib import Path
from typing import Dict, Set, Tuple, Iterable, Optional
import numpy as np
import pandas as pd

# 类型别名
Adj   = Dict[int, Set[int]]      # u -> {v1, v2, ...}
AllAdj= Dict[int, Adj]           # t -> Adj

# ---------- 基础：邻接 -> 边集 ----------
def _edges_from_adj(adj: Adj, directed: bool=False) -> Set[Tuple[int,int]]:
    edges: Set[Tuple[int,int]] = set()
    for u, nbrs in adj.items():
        if not isinstance(nbrs, Iterable):
            continue
        for v in nbrs:
            if u == v:
                continue
            if directed:
                edges.add((u, v))
            else:
                a, b = (u, v) if u < v else (v, u)
                edges.add((a, b))
    return edges

# ---------- 1) 逐时刻统计 ----------
def summarize_topology(all_inter_edge: AllAdj, directed: bool=False) -> pd.DataFrame:
    """
    返回列：
      t, link_count, changed(0/1), add_edges, del_edges
    """
    times = sorted(all_inter_edge.keys())
    rows  = []
    prev: Set[Tuple[int,int]] = set()

    for i, t in enumerate(times):
        e = _edges_from_adj(all_inter_edge[t], directed=directed)
        add = e - prev
        rem = prev - e
        rows.append(dict(
            t=int(t),
            link_count=len(e),
            changed=0 if i==0 else int(bool(add or rem)),
            add_edges=len(add) if i>0 else 0,
            del_edges=len(rem) if i>0 else 0,
        ))
        prev = e

    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)

# ---------- 2) 辅助列（便于画稳定段与竖线） ----------
def add_helper_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    新增：
      stable_line: changed==0 时取 link_count，否则 NaN
      change_line: changed==1 时取 link_count，否则 NaN
      change_mark: changed==1 时在底部给一个基准值（竖线起点）
    """
    out = df.copy()
    y   = out["link_count"].to_numpy(float)
    ch  = out["changed"].to_numpy(int)
    out["stable_line"] = np.where(ch==0, y, np.nan)
    out["change_line"] = np.where(ch==1, y, np.nan)
    ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
    base = ymin - 0.05*(ymax - ymin)
    out["change_mark"] = np.where(ch==1, base, np.nan)
    return out

# ---------- 3) 颜色交替辅助列（可选） ----------
def add_alternating_group(df: pd.DataFrame, col_name: str="color_group") -> pd.DataFrame:
    """
    新增 0/1 交替列：每遇到一次 changed==1 就切换，用于绘图分段上色。
    """
    out = df.copy()
    g = 0
    groups = []
    for c in out["changed"].astype(int).to_list():
        if c == 1:
            g ^= 1   # 0/1 翻转
        groups.append(g)
    out[col_name] = groups
    return out

# ---------- 4) 导出函数（风格与你的 pending 导出一致） ----------
def export_topology_summary(
    all_inter_edge: AllAdj,
    *,
    out_dir: Path | str,
    basename: str = "topology_change_summary",
    to: tuple[str, ...] = ("csv",),          # 可选: ("csv","xlsx","parquet")
    directed: bool = False,
    with_helpers: bool = True,
    with_alt_group: bool = True,             # 是否加 0/1 交替列
) -> dict[str, Path]:
    """
    统计并写出文件；返回 {格式: 路径}
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = summarize_topology(all_inter_edge, directed=directed)
    if with_helpers:
        df = add_helper_cols(df)
    if with_alt_group:
        df = add_alternating_group(df, col_name="color_group")

    written: dict[str, Path] = {}

    if "csv" in to:
        p = out_dir / f"{basename}.csv"
        df.to_csv(p, index=False)
        written["csv"] = p

    if "xlsx" in to:
        p = out_dir / f"{basename}.xlsx"
        try:
            with pd.ExcelWriter(p, engine="xlsxwriter") as xw:
                df.to_excel(xw, index=False, sheet_name="topology_change")
            written["xlsx"] = p
        except Exception as e:
            print(f"[warn] 写 Excel 失败（可忽略）：{e}")

    if "parquet" in to:
        p = out_dir / f"{basename}.parquet"
        try:
            import pyarrow as pa, pyarrow.parquet as pq
            pq.write_table(pa.Table.from_pandas(df, preserve_index=False), p, compression="zstd")
            written["parquet"] = p
        except Exception as e:
            print(f"[warn] 写 Parquet 失败（可忽略）：{e}")

    return written
