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

def _build_graph_from_adj(adj: Dict[Any, Iterable[Any]], undirected: bool):
    G = nx.Graph() if undirected else nx.DiGraph()
    for u, vs in (adj or {}).items():
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
    return G


def _best_pair_by_multisource_bfs(G, src_nodes: Set[int], dst_nodes: Set[int]):
    """
    返回:
      best_len, best_src, best_dst, parent
    含义:
      在 src_nodes × dst_nodes 中找到最短的那一对。
    """
    from collections import deque

    dist: Dict[int, int] = {}
    parent: Dict[int, int] = {}
    source_of: Dict[int, int] = {}

    q = deque()
    for s in src_nodes:
        if s in G:
            dist[s] = 0
            source_of[s] = s
            q.append(s)

    if not q:
        return None, None, None, parent

    while q:
        u = q.popleft()
        for v in G.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                parent[v] = u
                source_of[v] = source_of.get(u, u)
                q.append(v)

    best_len = None
    best_dst = None
    for d in dst_nodes:
        dd = dist.get(d)
        if dd is None:
            continue
        if best_len is None or dd < best_len:
            best_len = dd
            best_dst = d

    if best_len is None:
        return None, None, None, parent

    best_src = source_of.get(best_dst)
    return best_len, best_src, best_dst, parent


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
    steps: Optional[Iterable[int]] = None,
    undirected: bool = True,
    return_pair: bool = True,
    return_path: bool = False,
    # 新增: 静态拓扑加速选项
    static_topology: Optional[bool] = None,   # None=自动检测（按对象引用）
    static_step: Optional[int] = None,        # 静态图取哪个 step 的拓扑
    precompute_all_pairs: bool = False        # True=预计算所有源的最短距离
) -> pd.DataFrame:
    """
    返回 DataFrame 列：
      time, min_shortest_path
      可选：best_s, best_d, path

    语义不变：每个 step 上，取 P(step) 与 C(step) 间最短路径的最小值。
    """
    # ############## AI/HUMAN COMMENT: FUNCTION CONTRACT ##############
    # For each step t, compute:
    #   min_{s in P(t), d in C(t)} shortest_hops_t(s, d)
    # hops = number of edges on an unweighted shortest path.
    #
    # Output rules:
    # - P(t) empty or C(t) empty -> NaN
    # - P/C non-empty but unreachable -> NaN
    #
    # Non-goals:
    # - No latency/capacity/load weighting
    # - No traffic engineering objective
    # ###############################################################


    # ---- step 选择 ----
    all_steps = sorted(s for s in (_to_int(k) for k in all_edges.keys()) if s is not None)
    if steps is None:
        use_steps = all_steps
    elif isinstance(steps, tuple) and len(steps) == 2:
        a, b = int(steps[0]), int(steps[1])
        use_steps = [t for t in all_steps if a <= t <= b]
    else:
        use_steps = sorted(set(_to_int(s) for s in steps if _to_int(s) is not None))

    # ---- 自动检测静态拓扑（可被参数覆盖）----
    if static_topology is None and use_steps:
        ref_adj = all_edges.get(use_steps[0], {}) or {}
        static_topology = all((all_edges.get(t, {}) or {}) is ref_adj for t in use_steps[1:])

    static_G = None
    dist_cache = None

    if static_topology and use_steps:
        ref_step = use_steps[0] if static_step is None else int(static_step)
        static_G = _build_graph_from_adj(all_edges.get(ref_step, {}) or {}, undirected)

        # 确保所有可能接入卫星都在图内（即使是孤点）
        access_nodes = set()
        for t in use_steps:
            access_nodes.update(_get_set(paris, t))
            access_nodes.update(_get_set(chongqin, t))
        static_G.add_nodes_from(access_nodes)

        if precompute_all_pairs:
            dist_cache = {
                s: nx.single_source_shortest_path_length(static_G, s)
                for s in static_G.nodes
            }

    records: List[Dict[str, Any]] = []

    for step in use_steps:
        rec = {"time": step, "min_shortest_path": np.nan}
        if return_pair:
            rec["best_s"] = ""
            rec["best_d"] = ""
        if return_path:
            rec["path"] = ""

        P = _get_set(paris, step)
        C = _get_set(chongqin, step)
        if not P or not C:
            records.append(rec)
            continue

        best_len = None
        best_s = None
        best_d = None
        parent = {}

        # 分支1: 静态图 + 预计算距离
        if static_G is not None and dist_cache is not None:
            for s in P:
                ds = dist_cache.get(s)

                if ds is None:

                    continue
                for d in C:
                    hops = ds.get(d)
                    if hops is None:
                        continue
                    if best_len is None or hops < best_len:
                        best_len = hops
                        best_s = s
                        best_d = d

        else:
            # 分支2: 静态图不预计算 / 动态图逐step
            if static_G is not None:
                G = static_G
            else:
                G = _build_graph_from_adj(all_edges.get(step, {}) or {}, undirected)
                G.add_nodes_from(P | C)

            best_len, best_s, best_d, parent = _best_pair_by_multisource_bfs(G, P, C)

        if best_len is None:
            records.append(rec)
            continue

        rec["min_shortest_path"] = float(best_len)

        if return_pair:
            rec["best_s"] = "" if best_s is None else str(best_s)
            rec["best_d"] = "" if best_d is None else str(best_d)

        if return_path:
            if best_s is None or best_d is None:
                rec["path"] = ""
            elif static_G is not None and dist_cache is not None:
                # 仅对最优 pair 求一次路径，避免额外大开销
                try:
                    path_nodes = nx.shortest_path(static_G, best_s, best_d)
                    rec["path"] = "->".join(map(str, path_nodes))
                except Exception:
                    rec["path"] = ""
            else:
                # 复用 BFS parent 回溯
                path_nodes = []
                x = best_d
                while x in parent:
                    path_nodes.append(x)
                    x = parent[x]
                # 建议改为：
                if x is not None and x in P:  # ✅ 确保回溯到的终点确实是源集合中的节点
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
        static_topology: Optional[bool] = None,
        precompute_all_pairs: bool = False,
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

    # df = compute_stationpair_min_hops_over_time(
    #     all_edges, paris, chongqin,
    #     steps=steps, undirected=undirected,
    #     return_pair=False, return_path=False
    # )

    df = compute_stationpair_min_hops_over_time(
        all_edges, paris, chongqin,
        steps=steps, undirected=undirected,
        return_pair=False, return_path=False,          # ← 画图不需要，写死 False

        static_topology=static_topology,
        precompute_all_pairs=precompute_all_pairs,
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
        static_topology: Optional[bool] = None,
        precompute_all_pairs: bool = False,

        with_pair: bool = True,
    with_path: bool = False

) -> Path:
    """
    计算 + 写 CSV，返回 CSV 路径。
    CSV 列：time, min_shortest_path, (best_s, best_d), (path)
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # df = compute_stationpair_min_hops_over_time(
    #     all_edges, paris, chongqin,
    #     steps=steps, undirected=undirected,
    #     return_pair=with_pair, return_path=with_path
    # )

    df = compute_stationpair_min_hops_over_time(
        all_edges, paris, chongqin,
        steps=steps, undirected=undirected,
        return_pair=with_pair, return_path=with_path,
        static_topology=static_topology,              # ← 加
        precompute_all_pairs=precompute_all_pairs,    # ← 加
    )


    csv_path = out_dir / f"{basename}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
