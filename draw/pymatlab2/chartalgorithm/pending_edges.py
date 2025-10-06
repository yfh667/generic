# edge_series_ops.py
# -*- coding: utf-8 -*-
"""
时间片级边集操作：
- intersect_edge_series(a, b): a 与 b 的交集（按共同时间片、共同节点，邻居集合取交）
- difference_edge_series(a, b): a - b 的差集（逐时间片、逐节点，邻居集合做差）

数据结构约定：
- EdgeAdj: {u: {v1, v2, ...}}          邻接表（集合为邻居）
- EdgeSerie:
    1) dict 形式：{t: EdgeAdj}         时间片 -> 邻接表
    2) list 形式：[EdgeAdj, ...]       下标为时间片

注意：函数内部会把输入的邻居容器转成 set 再运算，保证健壮性。
"""

from typing import Dict, List, Set, Union, Iterable, Tuple

EdgeAdj   = Dict[int, Set[int]]                      # u -> {v,...}
EdgeSerie = Union[Dict[int, EdgeAdj], List[EdgeAdj]] # t -> (u->{v})


def _iter_slices(series: EdgeSerie) -> Iterable[Tuple[int, EdgeAdj]]:
    """
    统一把 dict/list 的时间片序列迭代为 (t, EdgeAdj) 对。
    """
    return series.items() if isinstance(series, dict) else enumerate(series)


def intersect_edge_series(a: EdgeSerie, b: EdgeSerie, keep_empty: bool = False) -> Dict[int, EdgeAdj]:
    """
    时间片级交集：
      - 仅保留两者共同的时间片
      - 对每个时间片，仅保留两者都出现的节点
      - 节点的邻居集合取交集
      - keep_empty=False 时，若该时间片结果为空则不输出该时间片

    参数
    ----
    a, b : EdgeSerie
        两个时间片序列（dict 或 list）。
    keep_empty : bool
        若 True，即使该时间片结果为空也保留该时间片键。

    返回
    ----
    Dict[int, EdgeAdj]
        交集后的 {t: {u: {v,...}}}
    """
    ad = dict(_iter_slices(a))
    bd = dict(_iter_slices(b))
    out: Dict[int, EdgeAdj] = {}

    for t in (set(ad) & set(bd)):
        ia, ib = ad[t], bd[t]
        merged: EdgeAdj = {}
        # 只保留双方都出现的节点
        for u in (set(ia) & set(ib)):
            nbr = set(ia.get(u, set())) & set(ib.get(u, set()))
            if nbr:
                merged[u] = nbr
        if keep_empty or merged:
            out[t] = merged
    return out


def difference_edge_series(a: EdgeSerie, b: EdgeSerie, drop_empty: bool = True) -> Dict[int, EdgeAdj]:
    """
    a - b 的时间片差集：
      - 逐时间片：只在同一时间片 t 下做差
      - 对每个节点 u：邻居集合做 set 差 a[u] - b[u]
      - drop_empty=True: 结果为空的时间片不输出

    说明：若 b 在某个时间片 t 缺失，视为该 t 的 b 为空邻接表，因此 a[t] 将原样保留。

    参数
    ----
    a, b : EdgeSerie
        两个时间片序列（dict 或 list）。
    drop_empty : bool
        若 True，删除结果为空的时间片键。

    返回
    ----
    Dict[int, EdgeAdj]
        差集后的 {t: {u: {v,...}}}
    """
    ad = dict(_iter_slices(a))
    bd = dict(_iter_slices(b))
    out: Dict[int, EdgeAdj] = {}

    for t, ia in ad.items():
        ib = bd.get(t, {})
        merged: EdgeAdj = {}
        for u, nbrs_a in ia.items():
            rem = set(nbrs_a) - set(ib.get(u, set()))
            if rem:
                merged[u] = rem
        if not drop_empty or merged:
            out[t] = merged
    return out


__all__ = [
    "EdgeAdj",
    "EdgeSerie",
    "intersect_edge_series",
    "difference_edge_series",
]
