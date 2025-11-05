# -*- coding: utf-8 -*-
"""
两城（地面站）最短跳数（min shortest hops）随时间：
- compute_stationpair_min_hops_over_time
- plot_stationpair_min_hops_over_time
- export_stationpair_min_hops_to_origin

输入：
  all_edges: { step: { u: iterable(vs) } }
  paris:     { step: {sat_ids...} }   # 集合到集合（巴黎能接入的卫星集合）
  chongqin:  { step: {sat_ids...} }   # 集合到集合（重庆能接入的卫星集合）

说明：
  在每个 step 上，从 P=paris[step] 到 C=chongqin[step] 的所有可达对中，
  取最小的最短路径“跳数”（边数）。若 P 或 C 为空，或完全不可达，则记 NaN。
"""

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set, List
import networkx as nx
import pandas as pd
import numpy as np


def _to_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _get_set(container, t) -> Set[int]:
    """兼容 dict/list 两种索引，返回 int 化后的集合。"""
    try:
        v = container[t] if not hasattr(container, "get") else container.get(t, set())
    except Exception:
        v = set()
    return set(_to_int(x) for x in (v or []) if _to_int(x) is not None)


def compute_stationpair_min_hops_over_time(
    all_edges: Dict[int, Dict[Any, Iterable[Any]]],
    paris: Dict[int, Iterable[Any]],
    chongqin: Dict[int, Iterable[Any]],
    *,
    steps: Optional[Iterable[int]] = None,   # None=用 all_edges 的全部 step；(start,end)；或迭代器
    undirected: bool = True,
    return_pair: bool = True,                # 是否返回实现最小跳数的 (best_s, best_d)
    return_path: bool = False                # 是否返回对应路径（节点序列，用 '->' 拼）
) -> pd.DataFrame:
    """
    返回 DataFrame 列：
      time, min_shortest_path
      可选：best_s, best_d, path
    备注：min_shortest_path 为“边数”；不可达/集合为空 -> NaN
    """
    # ---- step 选择（保持你原模板风格：基于 all_edges 的键）----
    all_steps = sorted(s for s in (_to_int(k) for k in all_edges.keys()) if s is not None)
    if steps is None:
        use_steps = all_steps
    elif isinstance(steps, tuple) and len(steps) == 2:
        a, b = int(steps[0]), int(steps[1])
        use_steps = [t for t in all_steps if a <= t <= b]
    else:
        use_steps = sorted(set(_to_int(s) for s in steps if _to_int(s) is not None))

    records: List[Dict[str, Any]] = []

    for step in use_steps:
        rec = {"time": step, "min_shortest_path": np.nan}
        if return_pair:
            rec["best_s"] = ""
            rec["best_d"] = ""
        if return_path:
            rec["path"] = ""

        # --- 两集合 ---
        P = _get_set(paris, step)
        C = _get_set(chongqin, step)
        if not P or not C:
            records.append(rec)
            continue

        # --- 构图 ---
        G = nx.Graph() if undirected else nx.DiGraph()
        adj = all_edges.get(step, {}) or {}
        for u, vs in adj.items():
            uu = _to_int(u)
            if uu is None:
                continue
            if not vs:
                G.add_node(uu)
                continue
            for v in vs:
                vv = _to_int(v)
                if vv is None or uu == vv:
                    continue
                G.add_edge(uu, vv)

        # 确保集合内节点都在图里（即使孤点）
        G.add_nodes_from(P | C)

        # --- 多源 BFS（从 P 同时出发），记录父指针以便回溯路径 ---
        from collections import deque
        dist: Dict[int, int] = {}
        parent: Dict[int, int] = {}   # v -> u
        source_of: Dict[int, int] = {}  # 记录最初源（可选）

        q = deque()
        for s in P:
            if s in G:
                dist[s] = 0
                source_of[s] = s
                q.append(s)

        if not q:
            records.append(rec)
            continue

        while q:
            u = q.popleft()
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    source_of[v] = source_of.get(u, u)
                    q.append(v)

        # --- 在 C 中找最小的可达距离 ---
        best_len = None
        best_dst = None
        for d in C:
            if d in dist:
                if (best_len is None) or (dist[d] < best_len):
                    best_len = dist[d]
                    best_dst = d

        if best_len is None:
            records.append(rec)
            continue

        rec["min_shortest_path"] = float(best_len)

        if return_pair:
            # 对应的源：从 best_dst 回溯到起点
            bs = source_of.get(best_dst, None)
            rec["best_s"] = "" if bs is None else str(bs)
            rec["best_d"] = "" if best_dst is None else str(best_dst)

        if return_path:
            # 回溯路径节点序列
            path_nodes = []
            x = best_dst
            while x in parent:
                path_nodes.append(x)
                x = parent[x]
            # x 现在是源节点
            if x is not None and x in P:
                path_nodes.append(x)
                path_nodes.reverse()
                rec["path"] = "->".join(map(str, path_nodes))
            else:
                rec["path"] = ""

        records.append(rec)

    cols = ["time", "min_shortest_path"]
    if return_pair:
        cols += ["best_s", "best_d"]
    if return_path:
        cols += ["path"]
    return pd.DataFrame(records, columns=cols)


def plot_stationpair_min_hops_over_time(
    all_edges: Dict[int, Dict[Any, Iterable[Any]]],
    paris: Dict[int, Iterable[Any]],
    chongqin: Dict[int, Iterable[Any]],
    *,
    steps: Optional[Iterable[int]] = None,
    undirected: bool = True,
    figsize=(10, 4),
    line_width=1.8,
    marker='o',
    title=None,
    rc_override=None,
    show=True,
    save=None,                  # None/False 不保存；True=默认名；或 str/Path/列表
    save_dir="figs",
    basename="minhops_paris_chongqing",
    formats=("png", "pdf"),
    dpi=300,
    transparent=False,
    return_handles=True
):
    """绘制 min shortest hops 时序曲线（风格与 intergroup 版本一致）"""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    df = compute_stationpair_min_hops_over_time(
        all_edges, paris, chongqin,
        steps=steps, undirected=undirected,
        return_pair=False, return_path=False
    )

    base_rc = {
        "font.family": "Times New Roman",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
    }
    if rc_override:
        base_rc.update(rc_override)

    plt.ion()
    with mpl.rc_context(base_rc):
        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title("Paris–Chongqing Min Shortest Hops")
        except Exception:
            pass

        ax.plot(df["time"], df["min_shortest_path"], marker=marker, linewidth=line_width)
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Min shortest path (hops)")
        if title is None:
            title = "Paris–Chongqing minimal shortest hops vs. time"
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()

        if show:
            plt.show(block=False)
            try:
                plt.pause(0.01)
            except Exception:
                pass

    # 保存
    if save:
        targets = []
        if save is True:
            outdir = Path(save_dir); outdir.mkdir(parents=True, exist_ok=True)
            for ext in formats:
                targets.append(outdir / f"{basename}.{ext.lstrip('.')}")
        else:
            if isinstance(save, (str, Path)):
                targets = [save]
            else:
                targets = list(save)
        for p in targets:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent)

    return (fig, ax, df) if return_handles else None


def export_stationpair_min_hops_to_origin(
    all_edges: Dict[int, Dict[Any, Iterable[Any]]],
    paris: Dict[int, Iterable[Any]],
    chongqin: Dict[int, Iterable[Any]],
    *,
    out_dir: Path | str,
    basename: str = "minhops_paris_chongqing",
    steps: Optional[Iterable[int]] = None,
    undirected: bool = True,
    with_pair: bool = True,
    with_path: bool = False
) -> Path:
    """
    计算 + 写 CSV，返回 CSV 路径。
    CSV 列：time, min_shortest_path, (best_s, best_d), (path)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = compute_stationpair_min_hops_over_time(
        all_edges, paris, chongqin,
        steps=steps, undirected=undirected,
        return_pair=with_pair, return_path=with_path
    )
    csv_path = out_dir / f"{basename}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
