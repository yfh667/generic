from collections import defaultdict
from typing import Dict, Set, Mapping, Iterable

# 目标返回类型：{step: {u: {v,...}}}
EdgeDict = Dict[int, Dict[int, Set[int]]]

def merge_edges_by_step(
    *edge_dicts: Mapping[int, Mapping[int, Iterable[int]]]
) -> EdgeDict:
    """
    深度合并多个 {step: {u: neighbors}} 结构：
    - 同 step、同 u 的邻接集合取并集
    - 不修改传入的字典，返回全新对象
    """
    out: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for D in edge_dicts:
        if not D:
            continue
        for step, u2nbrs in D.items():
            dst_u2 = out[step]
            for u, nbrs in u2nbrs.items():
                dst_u2[u].update(nbrs)  # 只写 out，不触碰输入
    # 转为普通 dict + set
    return {s: {u: set(vs) for u, vs in u2.items()} for s, u2 in out.items()}
