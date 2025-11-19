# -*- coding: utf-8 -*-
"""
Triple-window -> middle-window writer, batch for TIME_2_BUILD in [30..150].

For each TIME_2_BUILD (TTB):
  read A/B/C from INPUT_DIR/topology_{TTB}/raw
  merge -> transnodes -> get_no_conflict_link_nodes3 on [A.basicSa, C.end)
  filter to middle window B
  write to INPUT_DIR/topology_{TTB}/modify

并行策略（默认）:
  - 外层 TTB 顺序
  - 每个 TTB 内部，所有 triples 并行 (ProcessPool)
如需外层也并行，见 main() 里的 OUTER_PARALLEL=true 开关。
"""

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import os, sys, traceback
from typing import Iterable, Tuple

# ===== 你的模块 =====
import genaric2.tegnode as tegnode
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.transnodes as transnodes
import draw.basic_functio.conflict_link as conflict_link
# 可选：如需导出中段 edges / pending_edges 再引入
# import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import draw.basic_functio.topology_with_step_config as topology_with_step_config

# ===== 基础配置 =====
from config import INPUT_DIR, DATA_DIR

# 星座参数
P, N = 18, 36

# 滑动窗口：左闭右开
RANGES: list[Tuple[int, int]] = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
     (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]


TTB_VALUES = [  10,20,30,50,60, 70, 80,90, 100, 110, 120,130,140]

# TTB_VALUES = [  10,20,30,50 , 70, 80,90, 100, 110, 120,130,140]
# TTB_VALUES = [   40 ]
# 每个 TTB 内并行 worker 数（别把磁盘打爆；你 128C 可拉到 32/48 先压测）
WORKERS_PER_TTB = min(32, os.cpu_count() or 8, len(RANGES))
SIMULATION_EDITION = 'motif3'
CONFIG_DIR = Path(INPUT_DIR) / f"{SIMULATION_EDITION}" / "config"
# ---------------- helpers ----------------

def _triples(ranges: list[Tuple[int,int]]) -> Iterable[Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]]:
    """Yield consecutive triples (A,B,C)."""
    for i in range(len(ranges) - 2):
        yield (ranges[i], ranges[i+1], ranges[i+2])

# def _dirs_for_ttb(ttb: int) -> tuple[Path, Path]:
#     """按 TTB 生成读写目录：in=.../topology_{ttb}/raw, out=.../topology_{ttb}/modify"""
#     version = f"topology_{ttb}"
#     in_dir  = Path(INPUT_DIR) / version / "raw"
#     out_dir = Path(INPUT_DIR) / version / "modify"
#     out_dir.mkdir(parents=True, exist_ok=True)  # 没有就创建
#     return in_dir, out_dir



def _dirs_for_ttb(ttb: int):
    """
    返回 (modify_dir, figure_dir, xml_paths)
    modify_dir: INPUT_DIR/topology_{ttb}/modify
    figure_dir: INPUT_DIR/topology_{ttb}/figure
    xml_paths:  该 TTB 对应的 13 段 XML 完整路径列表
    """


    DEFAULT_VERSION = f"{SIMULATION_EDITION}/topology_{ttb}"
    # version = version_name(ttb)
    version = os.getenv("TOPOLOGY_VERSION", DEFAULT_VERSION)


    raw_dir = Path(INPUT_DIR) / version / "raw"
    modify_dir = Path(INPUT_DIR) / version / "modify"
    modify_dir.mkdir(parents=True, exist_ok=True)  # 没有就创建

    return raw_dir,  modify_dir


def _load_nodes_or_empty(path: Path):
    """读 XML->nodes；缺失报错；解析失败返回 {}."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    nodes = write2xml.xml_to_nodes_test(path, tegnode.tegnode_new)
    return nodes or {}




from pathlib import Path
from typing import Tuple

# def _write_nodes_any(writer_mod, nodes_dict, out_path: Path, *, overwrite: bool = True):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     if out_path.exists() and not overwrite:
#         return str(out_path)
#     try:
#         writer = getattr(writer_mod, "nodes_to_xml2_fast")
#         writer(nodes_dict, out_path)
#     except Exception:
#         try:
#             writer_mod.nodes_to_xml2(nodes_dict, out_path)
#         except Exception:
#             writer_mod.nodes_to_xml(nodes_dict, out_path)
#     return str(out_path)

def _one_triple_job(ttb: int,
                    triple: Tuple[Tuple[int,int],Tuple[int,int],Tuple[int,int]],
                    in_raw_dir: str,
                    out_dir: str,
                    *,
                    emit_A: bool = False,     # 新增：首三元组时导出 A 段
                    emit_C: bool = False,     # 新增：末三元组时导出 C 段
                    overwrite: bool = True) -> str:
    """
    子进程任务：处理一个 triple (A,B,C)，产出中段 B 的 XML；
    额外：若 emit_A/emit_C 为真，也分别导出 A/C 段。
    返回：中段 B 的输出路径（主返回值）；A/C 的写出在日志里提示。
    """
    # 防止数值库在进程内再开多线程
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    (start1, end1), (start2, end2), (start3, end3) = triple
    in_dir  = Path(in_raw_dir)
    out_dp  = Path(out_dir)
    start_ts_onestep = start1
    # end_ts   = 86399Q
    end_ts_onestep = end3

    # 1) load A/B/C nodes
    fp1 = in_dir / f"interplane_links_{start1}_{end1}.xml"
    fp2 = in_dir / f"interplane_links_{start2}_{end2}.xml"
    fp3 = in_dir / f"interplane_links_{start3}_{end3}.xml"

    nodes1 = _load_nodes_or_empty(fp1)
    nodes2 = _load_nodes_or_empty(fp2)
    nodes3 = _load_nodes_or_empty(fp3)

    # merge (later overwrites earlier)
    total_nodes = {}
    if nodes1: total_nodes.update(nodes1)
    if nodes2: total_nodes.update(nodes2)
    if nodes3: total_nodes.update(nodes3)

    # 2) 补齐左右邻接
    total_complete =total_nodes
    #
    # total_complete = transnodes.transnodes(total_nodes)

    # 3) 在 [A.basicSa, C.end) 上做冲突消解
    # _, _, modified_nodes = conflict_link.get_no_conflict_link_nodes3(
    #     total_complete, start1, end3, ttb, N, P
    # )

    cfg_path = CONFIG_DIR / f"adjust_{start2}_{end2}.json"
    adj_cfg = topology_with_step_config.load_adjust_config(cfg_path)

    # start_ts_onestep = adj_cfg.start_ts_onestep
    # end_ts_onestep   = adj_cfg.end_ts_onestep
    ratio = adj_cfg.ratio
    adjust_flag = adj_cfg.adjust_flag

    modify_raw_edges_by_step_onestep, modify_pending_edges_onstep, IG_link_onestep, by_y = conflict_link.get_no_conflict_link_nodes4(
        total_complete, start_ts_onestep, end_ts_onestep, ttb, N, P, ratio, end2, 1, adjustflag=adjust_flag)


    # 4) 只保留中间段 B
    # nodes_B = {k: v for k, v in modified_nodes.items() if start2 <= k[2] < end2}
    # out_B = out_dp / f"interplane_links_{start2}_{end2}.xml"
    # out_B_str = _write_nodes_any(write2xml, nodes_B, out_B, overwrite=overwrite)

    file_path_B = out_dp / f"interplane_links_{start2}_{end2}.xml"

    write2xml.nodes_to_xml_test(
        nodes2,
        file_path_B
    )


    # 5) （可选）在首三元组导出 A 段
    if emit_A:
        file_path = out_dp / f"interplane_links_{start1}_{end1}.xml"

        write2xml.nodes_to_xml_test(
            nodes1,
            file_path
        )
        #
        # nodes_A = {k: v for k, v in modified_nodes.items() if start1 <= k[2] < end1}
        # out_A = out_dp / f"interplane_links_{start1}_{end1}.xml"
        # _write_nodes_any(write2xml, nodes_A, out_A, overwrite=overwrite)

        print(f"[EMIT A] ({start1},{end1}) -> {file_path}")

    # 6) （可选）在末三元组导出 C 段
    if emit_C:
        file_path = out_dp / f"interplane_links_{start3}_{end3}.xml"

        write2xml.nodes_to_xml_test(
            nodes3,
            file_path
        )

        #
        # nodes_C = {k: v for k, v in modified_nodes.items() if start3 <= k[2] < end3}
        # out_C = out_dp / f"interplane_links_{start3}_{end3}.xml"
        # _write_nodes_any(write2xml, nodes_C, out_C, overwrite=overwrite)
        print(f"[EMIT C] ({start3},{end3}) -> {file_path}")



    print(f"[OK] (B) ({start2},{end2}) -> {file_path_B}")

    return file_path_B

def run_parallel_for_ttb(ttb: int,
                         ranges: list[Tuple[int,int]],
                         backend: str = "process",
                         workers: int | None = None) -> list[str]:
    """
    跑单个 TTB 的所有 triples；返回中段 B 的输出文件列表。
    额外：首三元组自动导出 A 段；末三元组自动导出 C 段。
    """
    in_dir, out_dir = _dirs_for_ttb(ttb)


    triples = list(_triples(ranges))
    if workers is None:
        workers = min(WORKERS_PER_TTB, len(triples))

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    results = []

    print(f"[TTB={ttb}] in={in_dir}  out={out_dir}  workers={workers}")

    with Executor(max_workers=workers) as ex:
        fut2t = {}
        for idx, t in enumerate(triples):
            emit_A = (idx == 0)                   # 只有第一个 triple 产出 A 段
            emit_C = (idx == len(triples) - 1)    # 只有最后一个 triple 产出 C 段
            fut = ex.submit(
                _one_triple_job, ttb, t, str(in_dir), str(out_dir),
                emit_A=emit_A, emit_C=emit_C, overwrite=True
            )
            fut2t[fut] = t

        for fut in as_completed(fut2t):
            t = fut2t[fut]
            try:
                outp = fut.result()
                print(f"[OK  ttb={ttb}] (B){t[1]} -> {outp}")
                results.append(outp)
            except Exception as e:
                print(f"[FAIL ttb={ttb}] {t[1]}: {e}", file=sys.stderr)
                traceback.print_exc()

    return results

# ---------------- main ----------------

def main():
    # Windows 安全
    try:
        if sys.platform.startswith("win"):
            mp.freeze_support()
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # 限制主进程里数值库的默认线程占用
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # 外层是否对不同 TTB 并行（I/O 压力大，默认 False）
    OUTER_PARALLEL = False

    if OUTER_PARALLEL:
        # 外层并行：按需调整 worker（比如 min(len(TTB_VALUES), 8)）
        outer_workers = min(8, len(TTB_VALUES))
        with ProcessPoolExecutor(max_workers=outer_workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(run_parallel_for_ttb, ttb, RANGES, "process", WORKERS_PER_TTB): ttb
                    for ttb in TTB_VALUES}
            for fut in as_completed(futs):
                ttb = futs[fut]
                try:
                    outs = fut.result()
                    print(f"[DONE ttb={ttb}] files={len(outs)}")
                except Exception as e:
                    print(f"[TTB={ttb} FAIL]: {e}", file=sys.stderr)
                    traceback.print_exc()
    else:
        # 外层顺序：最稳（推荐从这个跑起，再逐步加压测）
        for ttb in TTB_VALUES:
            outs = run_parallel_for_ttb(ttb, RANGES, backend="process", workers=WORKERS_PER_TTB)
            print(f"[DONE ttb={ttb}] files={len(outs)}")

    print("\nAll TTB done.")

if __name__ == "__main__":
    main()
