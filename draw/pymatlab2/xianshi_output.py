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

# ==== 星座参数 & 时间段（按你的项目固定） ====
P, N = 18, 36
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
    (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]
START_TS, END_TS = RANGES[0][0], RANGES[-1][1]  # [0, 22005)

# 默认要批量计算的建链时长（秒）
# DEFAULT_TTB_VALUES = [40, 50, 70,80, 90, 100, 110, 120, 130, 140]

DEFAULT_TTB_VALUES = [ 80 ]
def version_name(ttb: int) -> str:
    return f"topology_{ttb}"


def dirs_and_xmls(ttb: int):
    """
    返回 (modify_dir, figure_dir, xml_paths)
    modify_dir: INPUT_DIR/topology_{ttb}/modify
    figure_dir: INPUT_DIR/topology_{ttb}/figure
    xml_paths:  该 TTB 对应的 13 段 XML 完整路径列表
    """
    version = version_name(ttb)
    modify_dir = Path(INPUT_DIR) / version / "modify"
    figure_dir = Path(INPUT_DIR) / version / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    xml_paths = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
    return modify_dir, figure_dir, xml_paths


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
    modify_dir, figure_dir, xml_paths = dirs_and_xmls(ttb)

    # 基本检查
    missing = [str(p) for p in xml_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[TTB={ttb}] 缺少 XML（{len(missing)} 个），例如：{missing[:2]} ..."
        )

    print(f"[TTB={ttb}] 读取 XML（{len(xml_paths)} 个）…")
    # 顺序读取最稳（IO 型任务），也最节省内存
    totalnode = write2xml.load_all_nodes_sequential(xml_paths, tegnode.tegnode_complete)

    print(f"[TTB={ttb}] 计算 pending_edges（time_2_build={ttb}）…")
    pending_edges = inter_edge2nodes.trans_nodes2_pendingedges2(
        totalnode, START_TS, END_TS, ttb, P, N
    )

    print(f"[TTB={ttb}] 导出 CSV …")
    out_paths = psn.export_pending_series_to_origin(
        pending_edges,
        out_dir=figure_dir,
        basename=f"pending_edges_ttb{ttb}",
        to=("csv",),          # 需要 Excel 同时导出可改为 ("csv","xlsx")
        fill_missing=True,
        max_t_seconds=None,
        smooth_window=None    # 想多导一列平滑曲线就填窗口大小（如 5）
    )

    dt = time.time() - t0
    print(f"[TTB={ttb}] 完成，用时 {dt:.1f}s -> {out_paths}")
    return ttb, [str(p) for p in out_paths]


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
