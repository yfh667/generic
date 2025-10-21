
# topology_config_dynamic.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import ast

# 你现有模块
from draw.basic_functio import motif as motif_mod
import draw.read_snap_xml_archi as read_snap_xml


# ---------- 1) 安全表达式求值（只允许 + - * // % () 和变量名） ----------
class _SafeEval(ast.NodeVisitor):
    ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)

    def __init__(self, env: Dict[str, int]):
        self.env = env

    def visit_Module(self, node):
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValueError("Only a single expression is allowed.")
        return self.visit(node.body[0].value)

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Num(self, node):  # py<3.8
        return int(node.n)

    def visit_Constant(self, node):  # py>=3.8
        if isinstance(node.value, (int,)):
            return int(node.value)
        raise ValueError("Only integer constants are allowed.")

    def visit_Name(self, node):
        if node.id not in self.env:
            raise ValueError(f"Unknown symbol: {node.id}")
        v = self.env[node.id]
        if not isinstance(v, int):
            raise ValueError(f"Symbol '{node.id}' must be int, got {type(v)}")
        return v

    def visit_UnaryOp(self, node):
        if not isinstance(node.op, self.ALLOWED_UNARYOPS):
            raise ValueError("Unary operator not allowed.")
        val = self.visit(node.operand)
        return +val if isinstance(node.op, ast.UAdd) else -val

    def visit_BinOp(self, node):
        if not isinstance(node.op, self.ALLOWED_BINOPS):
            raise ValueError("Binary operator not allowed.")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):      return left + right
        if isinstance(node.op, ast.Sub):      return left - right
        if isinstance(node.op, ast.Mult):     return left * right
        if isinstance(node.op, ast.FloorDiv): return left // right
        if isinstance(node.op, ast.Mod):      return left % right
        raise ValueError("Unexpected operator.")

    # 禁止：函数调用、属性、下标、比较、布尔、if表达式、列表/字典等
    def generic_visit(self, node):
        disallowed = (
            ast.Call, ast.Attribute, ast.Subscript, ast.Compare, ast.BoolOp,
            ast.IfExp, ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp,
            ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Lambda, ast.Assign
        )
        if isinstance(node, disallowed):
            raise ValueError(f"Disallowed expression node: {type(node).__name__}")
        return super().generic_visit(node)


def eval_int_expr(expr_or_int: Optional[Union[int, str]],
                  env: Optional[Dict[str, int]]) -> Optional[int]:
    """None -> None；int -> int；str(表达式) -> 在 env 下安全求值为 int。"""
    if expr_or_int is None:
        return None
    if isinstance(expr_or_int, int):
        return expr_or_int
    if not isinstance(expr_or_int, str):
        raise ValueError("t_start/t_end must be int|str|None")
    if env is None:
        raise ValueError("Expression needs an eval env, but env=None.")
    tree = ast.parse(expr_or_int, mode="exec")
    return int(_SafeEval(env).visit(tree))


# ---------- 2) 数据模型：t_start/t_end 允许 int/str/None ----------
@dataclass
class MotifSpec:
    p_start: int
    p_end:   int
    y_start: int
    y_end:   int
    option:  int = 0
    # 时间窗口：左闭右开；可为 int / str 表达式 / None（全时段）
    t_start: Optional[Union[int, str]] = None
    t_end:   Optional[Union[int, str]] = None

@dataclass
class TopologyConfig:
    P: int
    N: int
    base_groupid: int
    motifs: List[MotifSpec]

    # 可选：把“符号列表”也保存进去（非必需，只是给人看/校验）
    symbols: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "P": int(self.P),
            "N": int(self.N),
            "base_groupid": int(self.base_groupid),
            "motifs": [asdict(m) for m in self.motifs],
        }
        if self.symbols is not None:
            d["symbols"] = list(self.symbols)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TopologyConfig":
        for k in ("P", "N", "base_groupid", "motifs"):
            if k not in d:
                raise ValueError(f"Missing required field: {k}")
        motifs = [MotifSpec(**m) for m in d["motifs"]]
        return TopologyConfig(
            P=int(d["P"]), N=int(d["N"]),
            base_groupid=int(d["base_groupid"]),
            motifs=motifs,
            symbols=d.get("symbols")
        )


def save_config(cfg: TopologyConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)


def load_config(path: str | Path) -> TopologyConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        d = json.load(f)
    return TopologyConfig.from_dict(d)


# ---------- 3) Recorder：兼容老API + 支持时间表达式 ----------
class TopologyRecorder:
    def __init__(self, P: int, N: int):
        self.P = int(P)
        self.N = int(N)
        self.base_groupid: Optional[int] = None
        self._motifs: List[MotifSpec] = []

    # 保持老用法：对齐基准组
    def modify_group_data(self, group_data: Dict, base_groupid: int):
        self.base_groupid = int(base_groupid)
        return read_snap_xml.modify_group_data(group_data, P=self.P, N=self.N, base_groupid=base_groupid)

    # 保持老签名；若需要时间窗口，在下面这个 with_time 版本里传
    def write_distinct_motif(self, p_start: int, p_end: int, y_start: int, y_end: int,
                             nodes: Dict, option: int = 0) -> None:
        self._motifs.append(MotifSpec(p_start, p_end, y_start, y_end, option))
        motif_mod.write_distinct_motif(p_start, p_end, y_start, y_end, self.P, self.N, nodes, option=option)

    # 新增：支持时间窗口（int 或 str 表达式）
    def write_distinct_motif_with_time(self, p_start: int, p_end: int, y_start: int, y_end: int,
                                       nodes: Dict, t_start: Optional[Union[int, str]],
                                       t_end: Optional[Union[int, str]], option: int = 0) -> None:
        self._motifs.append(MotifSpec(p_start, p_end, y_start, y_end, option, t_start, t_end))
        # 录制阶段不应用到 nodes（只“记账”）；真正渲染时按 t 决定

    def save(self, path: str | Path, symbols: Optional[List[str]] = None) -> None:
        if self.base_groupid is None:
            raise ValueError("Call recorder.modify_group_data(...) before save().")
        cfg = TopologyConfig(P=self.P, N=self.N, base_groupid=self.base_groupid,
                             motifs=self._motifs, symbols=symbols)
        save_config(cfg, path)

    # ===== 渲染：给定某个 t 与变量环境，生成该时刻的 nodes / 邻接 =====
    def _motifs_active_at(self, t: int, eval_env: Optional[Dict[str, int]]) -> List[MotifSpec]:
        act = []
        for m in self._motifs:
            ts = eval_int_expr(m.t_start, eval_env)
            te = eval_int_expr(m.t_end,   eval_env)
            # None 表示全时段；否则 [ts, te) 左闭右开
            if ts is None and te is None:
                act.append(m)
            else:
                if ts is None or te is None:
                    # 防止只给一端
                    raise ValueError("Both t_start and t_end must be set (or both None).")
                if ts <= t < te:
                    act.append(m)
        return act

    def render_nodes_at(self, t: int, eval_env: Optional[Dict[str, int]] = None) -> Dict:
        nodes: Dict = {}
        for m in self._motifs_active_at(t, eval_env):
            motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
                                           self.P, self.N, nodes, option=m.option)
        return nodes

    def render_adj_at(self, t: int, eval_env: Optional[Dict[str, int]] = None) -> Dict[int, set]:
        nodes = self.render_nodes_at(t, eval_env=eval_env)
        return motif_mod.transform_nodes_2_adjacent(nodes, self.P, self.N)

    def render_adj_range(self, t_start_incl: int, t_end_excl: int,
                         eval_env: Optional[Dict[str, int]] = None) -> Dict[int, Dict[int, set]]:
        out: Dict[int, Dict[int, set]] = {}
        for t in range(t_start_incl, t_end_excl):
            out[t] = self.render_adj_at(t, eval_env=eval_env)
        return out


# ---------- 4) 便捷：加载并渲染 ----------
def apply_topology_config(group_data: Dict, cfg: TopologyConfig,
                          t: Optional[int] = None,
                          eval_env: Optional[Dict[str, int]] = None):
    rec = TopologyRecorder(cfg.P, cfg.N)
    rec.base_groupid = cfg.base_groupid
    rec._motifs = cfg.motifs  # 直接复用

    rev_group_data, offset = read_snap_xml.modify_group_data(
        group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
    )

    if t is None:
        return rec, rev_group_data, offset
    else:
        adj_at_t = rec.render_adj_at(t, eval_env=eval_env)
        return rec, rev_group_data, offset, adj_at_t


#
#
# # topology_config_min.py
# from __future__ import annotations
# import json
# from dataclasses import dataclass, asdict, field
# from pathlib import Path
# from typing import List, Dict, Any, Tuple, Optional
#
# import draw.read_snap_xml as read_snap_xml
# from draw.basic_functio import motif as motif_mod
#
#
# # ---------- Data models ----------
# @dataclass
# class MotifSpec:
#     p_start: int
#     p_end: int
#     y_start: int
#     y_end: int
#     option: int = 0
#     t_start: Optional[int] = None  # None 表示“全时段”
#     t_end:   Optional[int] = None  # None 表示“全时段”
#
# @dataclass
# class TopologyConfig:
#     P: int
#     N: int
#     base_groupid: int
#     motifs: List[MotifSpec] = field(default_factory=list)
#
#     def to_dict(self) -> Dict[str, Any]:
#         # 手写 to_dict，保证 None 不会丢
#         return {
#             "P": int(self.P),
#             "N": int(self.N),
#             "base_groupid": int(self.base_groupid),
#             "motifs": [
#                 {
#                     "p_start": m.p_start,
#                     "p_end":   m.p_end,
#                     "y_start": m.y_start,
#                     "y_end":   m.y_end,
#                     "option":  m.option,
#                     "t_start": m.t_start,
#                     "t_end":   m.t_end,
#                 } for m in self.motifs
#             ],
#         }
#
#     @staticmethod
#     def from_dict(d: Dict[str, Any]) -> "TopologyConfig":
#         for k in ("P", "N", "base_groupid", "motifs"):
#             if k not in d:
#                 raise ValueError(f"Missing required field: {k}")
#         motifs = []
#         for m in d["motifs"]:
#             motifs.append(MotifSpec(
#                 p_start=int(m["p_start"]),
#                 p_end=int(m["p_end"]),
#                 y_start=int(m["y_start"]),
#                 y_end=int(m["y_end"]),
#                 option=int(m.get("option", 0)),
#                 t_start=m.get("t_start", None),
#                 t_end=m.get("t_end", None),
#             ))
#         return TopologyConfig(
#             P=int(d["P"]), N=int(d["N"]),
#             base_groupid=int(d["base_groupid"]),
#             motifs=motifs
#         )
#
#
# # ---------- Save / Load ----------
# def save_config(cfg: TopologyConfig, path: str | Path) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as f:
#         json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
#
# def load_config(path: str | Path) -> TopologyConfig:
#     with Path(path).open("r", encoding="utf-8") as f:
#         d = json.load(f)
#     return TopologyConfig.from_dict(d)
#
#
# # ---------- Recorder ----------
# class TopologyRecorder:
#     """
#     用法：
#       rec = TopologyRecorder(P, N)
#       rev_group_data, offset = rec.modify_group_data(group_data, base_groupid=4)
#
#       # 1) 固定写法（立即落到 nodes，且全时段记录）
#       rec.write_distinct_motif(0,17,0,8, nodes, option=1)
#       rec.write_distinct_motif(0,17,9,29, nodes, option=0)
#       ...
#
#       # 2) 动态写法（只记录时间窗，不立即落到 nodes；回放时应用）
#       rec.write_distinct_motif(9,17,31,32, nodes=None, option=2, t_start=1232, t_end=1232+time_2_build+1)
#
#       rec.save("xxx.json")
#     """
#     # def __init__(self, P: int, N: int):
#     #     self.P = int(P)
#     #     self.N = int(N)
#     #     self.base_groupid: Optional[int] = None
#     #     self._motifs: List[MotifSpec] = []
#     def __init__(self, P: int, N: int):
#         self.P = int(P)
#         self.N = int(N)
#         self.base_groupid = None
#         self._motifs: list[MotifSpec] = []
#
#     def modify_group_data(self, group_data: Dict, base_groupid: int):
#         self.base_groupid = int(base_groupid)
#         return read_snap_xml.modify_group_data(group_data, P=self.P, N=self.N, base_groupid=self.base_groupid)
#
#     def write_distinct_motif(self, p_start, p_end, y_start, y_end,
#                              nodes=None, option=0, t_start=None, t_end=None):
#         """nodes!=None 表示立刻写入（通常是全时段基底）；无论如何都会记录 MotifSpec。"""
#         self._motifs.append(MotifSpec(p_start, p_end, y_start, y_end, option, t_start, t_end))
#         if nodes is not None:
#             from draw.basic_functio import motif as motif_mod
#             motif_mod.write_distinct_motif(p_start, p_end, y_start, y_end,
#                                            self.P, self.N, nodes, option=option)
#
#
#     # 渲染某一时刻：全时段 + 在 t 内有效的动态
#     def render_nodes_at(self, t: int) -> Dict[Tuple[int,int,int], object]:
#         from draw.basic_functio import motif as motif_mod
#         nodes: Dict[Tuple[int,int,int], object] = {}
#
#         # 1) 全时段 motif（t_start/t_end 都是 None）
#         for m in self._motifs:
#             if m.t_start is None and m.t_end is None:
#                 motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
#                                                self.P, self.N, nodes, option=m.option)
#         # 2) 在 t 有效的动态 motif（t_start <= t < t_end）
#         for m in self._motifs:
#             if m.t_start is not None and m.t_end is not None and (m.t_start <= t < m.t_end):
#                 motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
#                                                self.P, self.N, nodes, option=m.option)
#         return nodes
#
#     def render_adj_at(self, t: int) -> Dict[int, set]:
#         from draw.basic_functio import motif as motif_mod
#         nodes = self.render_nodes_at(t)
#         return motif_mod.transform_nodes_2_adjacent(nodes, self.P, self.N)
#     def save(self, path: str | Path) -> None:
#         if self.base_groupid is None:
#             raise ValueError("Call recorder.modify_group_data(...) before save().")
#         cfg = TopologyConfig(P=self.P, N=self.N, base_groupid=self.base_groupid, motifs=self._motifs)
#         save_config(cfg, path)
#
#
# # ---------- Replayer / Applier ----------
# def _apply_motif_list_to_nodes(motifs: List[MotifSpec], P: int, N: int, nodes: Dict) -> None:
#     """
#     按记录顺序应用（“后写覆盖先写”），与你现在的“最后写 wins”一致。
#     """
#     for m in motifs:
#         motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end, P, N, nodes, option=m.option)
#
# def _split_motifs(cfg: TopologyConfig):
#     """
#     返回 (global_motifs, timed_motifs)
#     global_motifs: t_start/t_end 均为 None
#     timed_motifs:  有时间窗的
#     """
#     g, t = [], []
#     for m in cfg.motifs:
#         if m.t_start is None and m.t_end is None:
#             g.append(m)
#         else:
#             t.append(m)
#     return g, t
#
# def apply_at_time(cfg: TopologyConfig, t: int) -> Dict:
#     """
#     在某一时刻 t 应用：
#       1) 先应用全时段 motif
#       2) 再应用覆盖 t 的时间窗 motif（左闭右开）
#     返回 nodes（step=-1 的模板图），你再 transform 成邻接即可。
#     """
#     nodes: Dict = {}
#     global_motifs, timed_motifs = _split_motifs(cfg)
#
#     # 1) 全时段
#     _apply_motif_list_to_nodes(global_motifs, cfg.P, cfg.N, nodes)
#     # 2) 覆盖 t 的时间窗
#     if timed_motifs:
#         for m in timed_motifs:
#             if m.t_start is not None and m.t_end is not None and (m.t_start <= t < m.t_end):
#                 motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
#                                                cfg.P, cfg.N, nodes, option=m.option)
#     return nodes
#
# def apply_over_range(group_data: Dict, cfg: TopologyConfig, start_ts: int, end_ts: int):
#     """
#     回放整个时间段：
#       - 先做 modify_group_data（一次）
#       - 对每个 t：apply_at_time -> nodes -> transform_nodes_2_adjacent
#     返回：rev_group_data, offset, all_nodes_by_t, all_rev_inter_edge_by_t
#     """
#     rev_group_data, offset = read_snap_xml.modify_group_data(
#         group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
#     )
#
#     all_nodes_by_t: Dict[int, Dict] = {}
#     all_rev_inter_edge: Dict[int, Dict[int, set]] = {}
#
#     # 小优化：预分好 global 与 timed
#     global_motifs, timed_motifs = _split_motifs(cfg)
#
#     for t in range(start_ts, end_ts):
#         nodes: Dict = {}
#         # 1) 全时段
#         _apply_motif_list_to_nodes(global_motifs, cfg.P, cfg.N, nodes)
#         # 2) t 落在的时间窗
#         if timed_motifs:
#             for m in timed_motifs:
#                 if m.t_start is not None and m.t_end is not None and (m.t_start <= t < m.t_end):
#                     motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
#                                                    cfg.P, cfg.N, nodes, option=m.option)
#
#         all_nodes_by_t[t] = nodes
#         all_rev_inter_edge[t] = motif_mod.transform_nodes_2_adjacent(nodes, cfg.P, cfg.N)
#
#     return rev_group_data, offset, all_nodes_by_t, all_rev_inter_edge
#
#
# # 便捷函数：直接从文件加载并回放
# def load_and_apply_over_range(group_data: Dict, config_path: str | Path, start_ts: int, end_ts: int):
#     cfg = load_config(config_path)
#     return apply_over_range(group_data, cfg, start_ts, end_ts)
#


# version1
# topology_config_min.py
# from __future__ import annotations
# import json
# from dataclasses import dataclass, asdict
# from pathlib import Path
# from typing import List, Dict, Any, Tuple
#
# import draw.read_snap_xml as read_snap_xml
# from draw.basic_functio import motif as motif_mod
#
# # ---- Data models: only what you want ----
# @dataclass
# class MotifSpec:
#     p_start: int
#     p_end: int
#     y_start: int
#     y_end: int
#     option: int = 0
#
# @dataclass
# class TopologyConfig:
#     P: int
#     N: int
#     base_groupid: int
#     motifs: List[MotifSpec]
#
#     def to_dict(self) -> Dict[str, Any]:
#         return {
#             "P": int(self.P),
#             "N": int(self.N),
#             "base_groupid": int(self.base_groupid),
#             "motifs": [asdict(m) for m in self.motifs],
#         }
#
#     @staticmethod
#     def from_dict(d: Dict[str, Any]) -> "TopologyConfig":
#         # minimal validation
#         for k in ("P", "N", "base_groupid", "motifs"):
#             if k not in d:
#                 raise ValueError(f"Missing required field: {k}")
#         motifs = [MotifSpec(**m) for m in d["motifs"]]
#         return TopologyConfig(P=int(d["P"]), N=int(d["N"]),
#                               base_groupid=int(d["base_groupid"]),
#                               motifs=motifs)
#
# def save_config(cfg: TopologyConfig, path: str | Path) -> None:
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with path.open("w", encoding="utf-8") as f:
#         json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
#
# def load_config(path: str | Path) -> TopologyConfig:
#     with Path(path).open("r", encoding="utf-8") as f:
#         d = json.load(f)
#     return TopologyConfig.from_dict(d)
#
# # ---- Recorder: capture your manual operations (no extra fields) ----
# class TopologyRecorder:
#     def __init__(self, P: int, N: int):
#         self.P = int(P)
#         self.N = int(N)
#         self.base_groupid: int | None = None
#         self._motifs: List[MotifSpec] = []
#
#     def modify_group_data(self, group_data: Dict, base_groupid: int) -> Tuple[Dict, Any]:
#         self.base_groupid = int(base_groupid)
#         return read_snap_xml.modify_group_data(group_data, P=self.P, N=self.N, base_groupid=base_groupid)
#
#     def write_distinct_motif(self, p_start: int, p_end: int, y_start: int, y_end: int,
#                              nodes: Dict, option: int = 0) -> None:
#         self._motifs.append(MotifSpec(p_start, p_end, y_start, y_end, option))
#         motif_mod.write_distinct_motif(p_start, p_end, y_start, y_end, self.P, self.N, nodes, option=option)
#
#     def save(self, path: str | Path) -> None:
#         if self.base_groupid is None:
#             raise ValueError("Call recorder.modify_group_data(...) before save().")
#         cfg = TopologyConfig(P=self.P, N=self.N, base_groupid=self.base_groupid, motifs=self._motifs)
#         save_config(cfg, path)
#
# # ---- Replayer: apply an existing config ----
# def apply_topology_config(group_data: Dict, cfg: TopologyConfig):
#     """
#     Returns: rev_group_data, offset, nodes, rev_inter_edge
#     """
#     rev_group_data, offset = read_snap_xml.modify_group_data(
#         group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
#     )
#     nodes: Dict[int, set] = {}
#     for m in cfg.motifs:
#         motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
#                                        cfg.P, cfg.N, nodes, option=m.option)
#     rev_inter_edge = motif_mod.transform_nodes_2_adjacent(nodes, cfg.P, cfg.N)
#     return rev_group_data, offset, nodes, rev_inter_edge
#
# def load_and_apply(group_data: Dict, config_path: str | Path):
#     cfg = load_config(config_path)
#     return apply_topology_config(group_data, cfg)
