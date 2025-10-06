# batch_pending_export.py
# 并行处理多个 TTB：读取各自目录的 XML -> 计算 pending_edges -> 导出 CSV/XLSX

import os
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import multiprocessing as mp

# ==== 你的工程依赖 ====
from config import INPUT_DIR  # 只需要 INPUT_DIR
import genaric2.tegnode as tegnode
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import draw.pymatlab2.chartalgorithm.plot_switches_nonblocking as psn  # export_pending_series_to_origin
import draw.pymatlab2.chartalgorithm.pending_edges as pending_function
# ==== 星座参数 & 时间段（按你的项目固定） ====
P, N = 18, 36
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
    (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]
START_TS, END_TS = RANGES[0][0], RANGES[-1][1]  # [0, 22005)

# 默认要批量计算的建链时长（秒）
# DEFAULT_TTB_VALUES = [10,20,30,40, 50, 60,70,80, 90, 100, 110, 120, 130, 140]
DEFAULT_TTB_VALUES = [ 50 ]

def version_name(ttb: int) -> str:
    return f"topology_{ttb}"


# def dirs_and_xmls(ttb: int):
#     """
#     返回 (modify_dir, figure_dir, xml_paths)
#     modify_dir: INPUT_DIR/topology_{ttb}/modify
#     figure_dir: INPUT_DIR/topology_{ttb}/figure
#     xml_paths:  该 TTB 对应的 13 段 XML 完整路径列表
#     """
#     version = version_name(ttb)
#     modify_dir = Path(INPUT_DIR) / version / "modify"
#     raw_dir = Path(INPUT_DIR) / version / "raw"
#     figure_dir = Path(INPUT_DIR) / version / "figure"
#     figure_dir.mkdir(parents=True, exist_ok=True)
#
#     P1_DIR = Path(INPUT_DIR) / version / "p1_pending_edges"
#
#     P2_DIR = Path(INPUT_DIR) / version / "p2_pending_edges"
#
#
#
#
#     xml_paths = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
#     raw_xml_paths = [raw_dir / f"interplane_pending_links_{s}_{e}.xml" for (s, e) in RANGES]
#     return P1_DIR, P2_DIR,xml_paths,raw_xml_paths


def dirs_and_xmls(ttb: int):
    """
    返回 (P1_DIR, P2_DIR, xml_paths, raw_xml_paths)
    仅创建输出目录（figure/p1_pending_edges/p2_pending_edges），输入目录不创建。
    """
    version    = version_name(ttb)
    base_dir   = Path(INPUT_DIR) / version
    modify_dir = base_dir / "modify"
    raw_dir    = base_dir / "raw"
    figure_dir = base_dir / "figure"
    P1_DIR     = base_dir / "p1_pending_edges"
    P2_DIR     = base_dir / "p2_pending_edges"

    # 如不存在则创建（输出目录）
    for d in (figure_dir, P1_DIR, P2_DIR):
        d.mkdir(parents=True, exist_ok=True)

    xml_paths      = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
    raw_xml_paths  = [raw_dir    / f"interplane_pending_links_{s}_{e}.xml" for (s, e) in RANGES]

    return P1_DIR, P2_DIR, xml_paths, raw_xml_paths


def run_one_ttb(ttb: int) -> tuple[int, list[str]]:
    """
    单个 TTB 的完整流程：读 -> 算 -> 导出
    返回: (ttb, 导出文件路径列表)
    """
    # 限制子进程内部的并行库，以免过度竞争
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.time()
    P1_DIR, P2_DIR, xml_paths, raw_xml_paths = dirs_and_xmls(ttb)

    # 基本检查
    missing = [str(p) for p in xml_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[TTB={ttb}] 缺少 XML（{len(missing)} 个），例如：{missing[:2]} ..."
        )
    missing = [str(p) for p in raw_xml_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[TTB={ttb}] 缺少 XML（{len(missing)} 个），例如：{missing[:2]} ..."
        )



    print(f"[TTB={ttb}] 读取 XML（{len(xml_paths)} 个）…")
    # 顺序读取最稳（IO 型任务），也最节省内存



    totalnode = write2xml.load_all_nodes_sequential(xml_paths, tegnode.tegnode_complete)
    pending_nodes = write2xml.load_all_nodes_sequential(raw_xml_paths, tegnode.tegnode_complete)

    print(f"[TTB={ttb}] 计算 pending_edges（time_2_build={ttb}）…")
    pending_edges = inter_edge2nodes.trans_nodes2_pendingedges2(
        totalnode, START_TS, END_TS, ttb, P, N
    )

    raw_pending_edges = inter_edge2nodes.trans_nodes2edges(pending_nodes, P, N)

    p1_pending = pending_function.intersect_edge_series(pending_edges, raw_pending_edges)
    p2_pending = pending_function.difference_edge_series(pending_edges, p1_pending)

    p1_pending_nodes = inter_edge2nodes.trans_edge2node(p1_pending, P, N)
    # 推荐：用 raw string 防止反斜杠转义，并改成有意义的文件名
    # 这里，我们要把原始的边转为node进行存储

    p1_file_path = P1_DIR / f"interplane_P1_pending_links_{START_TS}_{END_TS}.xml"

    write2xml.nodes_to_xml(
        p1_pending_nodes,
        p1_file_path
    )


    p2_pending_nodes = inter_edge2nodes.trans_edge2node(p2_pending, P, N)

    p2_file_path = P2_DIR / f"interplane_P2_pending_links_{START_TS}_{END_TS}.xml"

    write2xml.nodes_to_xml(
        p2_pending_nodes,
        p2_file_path
    )

    print(f"[TTB={ttb}] 导出 CSV …")




    dt = time.time() - t0
    out_files = [p1_file_path, p2_file_path]

    print(f"[TTB={ttb}] 完成，用时 {dt:.1f}s -> {out_files}")
    return ttb, [str(p) for p in out_files]


def parse_args():
    ap = argparse.ArgumentParser(
        description="批量导出各 TTB 的 pending_edges（供 Origin 画图）"
    )
    ap.add_argument(
        "--ttb",
        type=str,
        default=",".join(str(x) for x in DEFAULT_TTB_VALUES),
        help=f"要计算的 TTB 列表，逗号分隔（默认: {DEFAULT_TTB_VALUES}）"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, mp.cpu_count() // 2)),
        help="并行进程数（默认：min(4, CPU/2)）"
    )
    return ap.parse_args()


def main():
    # Windows 需要 spawn 安全
    if sys.platform.startswith("win"):
        mp.freeze_support()
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

    args = parse_args()
    ttb_values = [int(x) for x in args.ttb.split(",") if x.strip()]
    workers = max(1, min(args.workers, len(ttb_values)))

    print(f"TTB 列表: {ttb_values}")
    print(f"并行进程: {workers}\n")

    results: dict[int, list[str]] = {}
    errors: dict[int, str] = {}

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(run_one_ttb, ttb): ttb for ttb in ttb_values}
        for fut in as_completed(futs):
            ttb = futs[fut]
            try:
                k, outs = fut.result()
                results[k] = outs
            except Exception as e:
                errors[ttb] = f"{type(e).__name__}: {e}"
                traceback.print_exc()

    print("\n=== 成功 ===")
    for ttb in sorted(results):
        print(f"TTB={ttb} -> {results[ttb]}")

    if errors:
        print("\n=== 失败 ===")
        for ttb in sorted(errors):
            print(f"TTB={ttb} -> {errors[ttb]}")


if __name__ == "__main__":
    main()
