from pathlib import Path
import networkx as nx

def plot_intergroup_avg_shortest_path(
    all_edges: dict,            # {step: {basicSa: iterable(dsts)}}
    group_data: dict,           # {step: {'groups': {group_id: set(nodes), ...}}}
    *,
    group_a=0,
    group_b=4,
    steps=None,                 # None=用 all_edges 的全部 step；或传 (basicSa,end) / 迭代器
    undirected=True,            # True=无向图；False=有向图
    # 绘图
    figsize=(10,4),
    line_width=1.8,
    marker='o',
    title=None,
    rc_override=None,           # dict: matplotlib rcParam 覆盖
    # 展示/保存
    show=True,                  # Jupyter 非阻塞显示
    save=None,                  # None/False 不保存；True 用默认规则；或传 str/Path/列表
    save_dir="figs",
    basename=None,              # 默认: f"avgspath_g{group_a}_{group_b}"
    formats=("png","pdf"),
    dpi=300,
    transparent=False,
    # 返回
    return_handles=True         # 返回 (fig, ax, df)
):
    """
    计算并绘制：在每个 step 上，group_a 中每个节点到 group_b 各节点的
    最短路径长度的“成对平均值”（只统计可达的 pair）。
    """
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # ---- step 选择 ----
    def _to_int(x):
        try: return int(x)
        except: return None
    all_steps = sorted(s for s in (_to_int(k) for k in all_edges.keys()) if s is not None)
    if steps is None:
        use_steps = all_steps
    elif isinstance(steps, tuple) and len(steps) == 2:
        a, b = int(steps[0]), int(steps[1])
        use_steps = [t for t in all_steps if a <= t <= b]
    else:
        use_steps = sorted(set(_to_int(s) for s in steps if _to_int(s) is not None))

    # ---- 逐 step 计算 ----
    avg_list, pair_cnt_list, g0_sz_list, g4_sz_list = [], [], [], []
    for step in use_steps:
        # 构图
        G = nx.Graph() if undirected else nx.DiGraph()
        adj = all_edges.get(step, {}) or {}
        for src, dsts in adj.items():
            if dsts is None:
                continue
            for dst in dsts:
                if src == dst:
                    continue
                G.add_edge(src, dst)

        # 取两组
        ok = (step in group_data and
              'groups' in group_data[step] and
              group_a in group_data[step]['groups'] and
              group_b in group_data[step]['groups'])
        if not ok:
            avg_list.append(np.nan); pair_cnt_list.append(0)
            g0_sz_list.append(0);    g4_sz_list.append(0)
            continue

        group0 = set(group_data[step]['groups'][group_a])
        group4 = set(group_data[step]['groups'][group_b])

        # 把组节点加入图（即便是孤立点，BFS 也只会给到自身=0；与另一组无边仍不可达）
        G.add_nodes_from(group0 | group4)

        # 成对平均：对 group0 的每个节点做一次 BFS
        path_lengths = []
        for n0 in group0:
            try:
                lengths = nx.single_source_shortest_path_length(G, n0)
            except Exception:
                continue
            for n4 in group4:
                if n4 in lengths:
                    path_lengths.append(lengths[n4])

        if path_lengths:
            avg_list.append(float(np.mean(path_lengths)))
            pair_cnt_list.append(int(len(path_lengths)))
        else:
            avg_list.append(np.nan)
            pair_cnt_list.append(0)

        g0_sz_list.append(len(group0))
        g4_sz_list.append(len(group4))

    # ---- 打包结果 ----
    import pandas as pd
    df = pd.DataFrame({
        "step": use_steps,
        "avg_shortest_path": avg_list,
        "pairs_counted": pair_cnt_list,
        f"|G{group_a}|": g0_sz_list,
        f"|G{group_b}|": g4_sz_list,
    })

    # ---- 绘图（Jupyter 非阻塞）----
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

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    plt.ion()  # 非阻塞
    with mpl.rc_context(base_rc):
        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.canvas.manager.set_window_title("Inter-Group Avg Shortest Path")
        except Exception:
            pass

        ax.plot(df["step"], df["avg_shortest_path"], marker=marker, linewidth=line_width)
        ax.set_xlabel("Time Step")
        ax.set_ylabel(f"Avg shortest path (G{group_a} ↔ G{group_b})")
        if title is None:
            title = f"Inter-region average shortest path vs. time (G{group_a}↔G{group_b})"
        ax.set_title(title)
        ax.grid(alpha=0.3, linestyle="--")
        plt.tight_layout()

        if show:
            plt.show(block=False)
            try: plt.pause(0.01)
            except: pass

    # ---- 保存（可选）----
    if save:
        targets = []
        if save is True:
            base = basename or f"avgspath_g{group_a}_{group_b}"
            outdir = Path(save_dir); outdir.mkdir(parents=True, exist_ok=True)
            for ext in formats:
                targets.append(outdir / f"{base}.{ext.lstrip('.')}")
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




from pathlib import Path
from typing import Any, Iterable, Optional
import pandas as pd
import networkx as nx

def compute_intergroup_avg_shortest_path(
    all_edges: dict,            # {step: {basicSa: iterable(dsts)}}
    group_data: dict,           # {step: {'groups': {group_id: set(nodes), ...}}}
    *,
    group_a: int = 0,
    group_b: int = 4,
    steps: Optional[Iterable[int]] = None,   # None=all steps; (basicSa,end); or iterable
    undirected: bool = True
) -> pd.DataFrame:
    """
    返回一个 DataFrame，列包含：
    step, avg_shortest_path, pairs_counted, |G{a}|, |G{b}|
    （只统计可达 pair，若该 step 无可达对则 avg 为 NaN）
    """
    # --- 规范 step 集 ---
    def _to_int(x):
        try: return int(x)
        except: return None
    all_steps = sorted(s for s in (_to_int(k) for k in all_edges.keys()) if s is not None)
    if steps is None:
        use_steps = all_steps
    elif isinstance(steps, tuple) and len(steps) == 2:
        a, b = int(steps[0]), int(steps[1])
        use_steps = [t for t in all_steps if a <= t <= b]
    else:
        use_steps = sorted(set(_to_int(s) for s in steps if _to_int(s) is not None))

    import numpy as np

    avg_list, pair_cnt_list, gA_sz_list, gB_sz_list = [], [], [], []

    for step in use_steps:
        # 构图
        G = nx.Graph() if undirected else nx.DiGraph()
        adj = all_edges.get(step, {}) or {}
        for src, dsts in adj.items():
            if not dsts:
                continue
            for dst in dsts:
                if src == dst:
                    continue
                G.add_edge(src, dst)

        # 两个组是否存在
        ok = (
            step in group_data and
            'groups' in group_data[step] and
            group_a in group_data[step]['groups'] and
            group_b in group_data[step]['groups']
        )
        if not ok:
            avg_list.append(np.nan); pair_cnt_list.append(0)
            gA_sz_list.append(0);    gB_sz_list.append(0)
            continue

        gA = set(group_data[step]['groups'][group_a])
        gB = set(group_data[step]['groups'][group_b])

        # 即便孤立点也加入图，便于 BFS 判断可达性
        G.add_nodes_from(gA | gB)

        # 成对平均（只计可达）
        path_lengths = []
        for nA in gA:
            try:
                lengths = nx.single_source_shortest_path_length(G, nA)
            except Exception:
                continue
            for nB in gB:
                if nB in lengths:
                    path_lengths.append(lengths[nB])

        if path_lengths:
            avg_list.append(float(np.mean(path_lengths)))
            pair_cnt_list.append(int(len(path_lengths)))
        else:
            avg_list.append(np.nan)
            pair_cnt_list.append(0)

        gA_sz_list.append(len(gA))
        gB_sz_list.append(len(gB))

    df = pd.DataFrame({
        "time": use_steps,
        "avg_shortest_path": avg_list,
        "pairs_counted": pair_cnt_list,
        f"|G{group_a}|": gA_sz_list,
        f"|G{group_b}|": gB_sz_list,
    })
    return df


def export_intergroup_avgspath_to_origin(
    all_edges: dict,
    group_data: dict,
    *,
    out_dir: Path | str,
    basename: str = "avgspath",
    group_a: int = 0,
    group_b: int = 4,
    steps: Optional[Iterable[int]] = None,
    undirected: bool = True,
) -> Path:
    """
    计算 + 写 CSV，返回 CSV 路径。
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = compute_intergroup_avg_shortest_path(
        all_edges, group_data,
        group_a=group_a, group_b=group_b,
        steps=steps, undirected=undirected
    )
    csv_path = out_dir / f"{basename}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path
