# batch_pending_export.py
# 并行处理多个 TTB：读取各自目录的 XML -> 计算 pending_edges -> 导出 CSV/XLSX
import  draw.pymatlab2.chartalgorithm.plot_switches_nonblocking as plot_switches_nonblocking

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
import draw.basic_functio.motif as motif
# ==== 星座参数 & 时间段（按你的项目固定） ====
import draw.pymatlab2.chartalgorithm.plot_intergroup_avg_shortest_path as avgsp  # 你之前的模块（含 export_intergroup_avgspath_to_origin）
import draw.pymatlab2.chartalgorithm.export_topology_summary as export_topology_summary

# zhge shi 3ge yiqi zuo
# 1. p1 p2
# 2. hops
# 3. topology summary

P, N = 18, 36
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
    (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]
START_TS, END_TS = RANGES[0][0], RANGES[-1][1]  # [0, 22005)

# 默认要批量计算的建链时长（秒）
DEFAULT_TTB_VALUES = [10,20,30,40, 50, 60,70,80, 90, 100, 110, 120, 130, 140]
# DEFAULT_TTB_VALUES = [ 60 ]
SIMULATION_EDITION = 'motif1'

def version_name(ttb: int) -> str:
    return f"topology_{ttb}"


def load_group_data_cached() -> dict:
    """子进程调用：读取缓存。"""
    import pickle
    with open(group_cache_path(), "rb") as f:
        return pickle.load(f)
def group_cache_path() -> Path:
    return cache_dir() / f"group_data_{START_TS}_{END_TS}.pkl"
def cache_dir() -> Path:
    d = Path(INPUT_DIR) / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def dirs_and_xmls(ttb: int):
    """
    返回 (P1_DIR, P2_DIR, xml_paths, raw_xml_paths)
    仅创建输出目录（figure/p1_pending_edges/p2_pending_edges），输入目录不创建。
    """
    DEFAULT_VERSION = f"{SIMULATION_EDITION}/topology_{ttb}"
    version = os.getenv("TOPOLOGY_VERSION", DEFAULT_VERSION)
    base_dir   = Path(INPUT_DIR) / version
    modify_dir = base_dir / "modify"
    raw_dir    = base_dir / "raw"
    figure_dir = base_dir / "figure"
    P1_DIR     = base_dir / "p1_pending_edges"
    P2_DIR     = base_dir / "p2_pending_edges"

    # 如不存在则创建（输出目录）
    for d in (figure_dir, P1_DIR, P2_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # xml_paths      = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
    # raw_xml_paths  = [raw_dir    / f"interplane_pending_links_{s}_{e}.xml" for (s, e) in RANGES]

    return modify_dir, figure_dir

# ===== 工具 =====
def make_edges_bidirectional(edge_dict):
    """{basicSa: set(dsts)} -> 双向"""
    new_edges = {}
    for src, dsts in edge_dict.items():
        for dst in dsts:
            new_edges.setdefault(src, set()).add(dst)
            new_edges.setdefault(dst, set()).add(src)
    return new_edges
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
    modify_dir, figure_dir= dirs_and_xmls(ttb)
    paths = [modify_dir / f"interplane_links_{s}_{e}.xml" for s, e in RANGES]
    totalnode = write2xml.load_all_nodes_sequential_test(paths, tegnode.tegnode_new)
    all_inter_edge, pending_edge, iG_edge = motif.transform_nodes_2_rawedge_test(totalnode, P, N, START_TS, END_TS)





    print(f"[TTB={ttb}] 导出 CSV …")

    p1_file_path = psn.export_pending_series_to_origin(
        pending_edge,
        out_dir=figure_dir,
        basename=f"p1_pending_edges_ttb{ttb}",
        to=("csv",),  # 需要 Excel 同时导出可改为 ("csv","xlsx")
        fill_missing=True,
        max_t_seconds=None,
        smooth_window=None  # 想多导一列平滑曲线就填窗口大小（如 5）
    )

    p2_file_path = psn.export_pending_series_to_origin(
        iG_edge,
        out_dir=figure_dir,
        basename=f"p2_pending_edges_ttb{ttb}",
        to=("csv",),  # 需要 Excel 同时导出可改为 ("csv","xlsx")
        fill_missing=True,
        max_t_seconds=None,
        smooth_window=None  # 想多导一列平滑曲线就填窗口大小（如 5）
    )



    # 双向化 inter
    for step in list(all_inter_edge.keys()):
        all_inter_edge[step] = make_edges_bidirectional(all_inter_edge[step])

    # 读取公共 group_data（缓存）
    group_data = load_group_data_cached()

    # —— 构造 all_edges（含 intra+inter）供 avgsp.compute 使用 ——
    # 这里不生成巨大的 all_intra_edge；每步把环内边追加进去即可。
    all_edges = {}
    # 预计算 648 个节点的左右邻居
    base_neighbors = {
        i * N + j: (i * N + ((j + 1) % N), i * N + ((j - 1) % N))
        for i in range(P) for j in range(N)
    }
    for step in range(START_TS, END_TS):
        adj = {}
        # intra（左右邻居）
        for node, (r, l) in base_neighbors.items():
            adj.setdefault(node, set()).update((r, l))
        # inter
        inter = all_inter_edge.get(step, {})
        for src, dsts in inter.items():
            adj.setdefault(src, set()).update(dsts)
        all_edges[step] = adj

    # 计算并导出 CSV
    csv_path = avgsp.export_intergroup_avgspath_to_origin(
        all_edges, group_data,
        out_dir=figure_dir,
        basename=f"avgspath_{ttb}",
        group_a=0, group_b=4,
        steps=(START_TS, END_TS-1),
        undirected=True
    )
    csv_path2 = export_topology_summary.export_topology_summary(
        all_inter_edge,
        out_dir=figure_dir,
        basename=f"topology_snapshot_ttb{ttb}",
        to=("csv"),  # 想要什么格式就写什么
        directed=False,
        with_helpers=True,  # 生成 stable_line / change_line / change_mark
        with_alt_group=True  # 生成 color_group（0/1 交替，用于分段上色）
    )

    dt = time.time() - t0
    out_files = [p1_file_path, p2_file_path,csv_path,csv_path2]
  #  out_files = [ csv_path,csv_path2]


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
