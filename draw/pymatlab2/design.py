# -*- coding: utf-8 -*-
"""
Parallel runner for independent (start_ts, end_ts) ranges.

For each range:
  - load config: INPUT_DIR/<VERSION>/config/{basicSa}_{end}.json
  - read group_data from raw XML for [basicSa, end)
  - modify_group_data -> motifs -> rev_inter_edge
  - revedge2rawedge -> get_no_conflict_link (time_2_build + envelopes)
  - edges -> nodes
  - write XML to: INPUT_DIR/<VERSION>/raw/interplane_links_{basicSa}_{end}.xml
"""

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os, sys, traceback

# ========= 基础参数 =========
from config import DATA_DIR, INPUT_DIR

# 星座参数（与配置一致时不会用到这里；保底放着）
N_DEFAULT = 36
P_DEFAULT = 18

import os
from pathlib import Path

TIME_2_BUILD = 80  # 你已有





# 默认用 "topology_{TIME_2_BUILD}"，也允许用环境变量 TOPOLOGY_VERSION 覆盖
VERSION = os.getenv("TOPOLOGY_VERSION", f"topology_{TIME_2_BUILD}")

CONFIG_DIR = Path(INPUT_DIR)  / "config"
RAW_DIR    = Path(INPUT_DIR) / VERSION / "raw"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 原始可见性 XML
XML_FILE = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"

# 时间窗口（左闭右开）
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,9355),
    (9355,11640),(11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]

# ========= 依赖模块 =========
import draw.read_snap_xml as read_snap_xml
import draw.basic_functio.motif as motif
import draw.basic_functio.topology_config as topology_config
import draw.basic_functio.revdata2rawdata as revdata2rawdata
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import draw.basic_functio.conflict_link as conflict_link
from draw.basic_functio.get_rectangular_size_interval import calc_envelope_for_group
import draw.basic_functio.write2xml as write2xml


# ========= 单区间处理 =========
def _process_one_range(start_ts: int, end_ts: int) -> str:
    """
    处理单个区间；返回输出文件路径（str）
    注意：在子进程中执行
    """
    # 1) 读配置（只含 P/N/base_groupid/motifs）
    cfg_path = CONFIG_DIR / f"{start_ts}_{end_ts}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {cfg_path}")
    cfg = topology_config.load_config(str(cfg_path))

    # 2) 解析该时间窗的可见性数据（子进程各自读取，避免主进程大对象跨进程拷贝）
    group_data = read_snap_xml.parse_xml_group_data(XML_FILE, start_ts, end_ts)

    # 3) 反向缝对齐
    rev_group_data, offset = read_snap_xml.modify_group_data(
        group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
    )

    # 4) motifs -> 同构邻接
    nodes = {}
    for m in cfg.motifs:
        motif.write_distinct_motif(
            m.p_start, m.p_end, m.y_start, m.y_end,
            cfg.P, cfg.N, nodes, option=getattr(m, "option", 0)
        )
    rev_inter_edge = motif.transform_nodes_2_adjacent(nodes, cfg.P, cfg.N)

    # 5) 将同构边扩展到整个区间（每步相同）
    all_rev_inter_edge = {t: rev_inter_edge for t in range(start_ts, end_ts)}

    # 6) 反变换为原始编号
    raw_inter_edge = revdata2rawdata.revedge2rawedge(all_rev_inter_edge, offset)

    # 7) 计算包络矩形（示例：组 0 与 base_groupid；出错容忍）
    rects = {}
    try:
        rect0 = calc_envelope_for_group(rev_group_data, [start_ts, end_ts], 0, cfg.P, cfg.N)
        rects[0] = rect0
    except Exception:
        pass
    try:
        rectb = calc_envelope_for_group(rev_group_data, [start_ts, end_ts], cfg.base_groupid, cfg.P, cfg.N)
        rects[cfg.base_groupid] = rectb
    except Exception:
        pass

    # 8) 建链时间 + 包络矩形约束
    raw_edges_by_step, pending_edges = conflict_link.get_no_conflict_link(
        raw_inter_edge, offset, rects, start_ts, end_ts, TIME_2_BUILD, cfg.N, cfg.P
    )

    # 9) 边 -> 节点（只存异轨链路）
    all_nodes = inter_edge2nodes.trans_edge2node(raw_edges_by_step, cfg.P, cfg.N)

    # 10) 写出 XML
    out_path = RAW_DIR / f"interplane_links_{start_ts}_{end_ts}.xml"
    # 如你已实现更快的流式 writer，可改成 nodes_to_xml2 / nodes_to_xml2_fast
    write2xml.nodes_to_xml(all_nodes, out_path)

    print(f"[OK] ({start_ts},{end_ts}) -> {out_path}")
    return str(out_path)


# ========= 并行调度器 =========
def run_parallel(ranges, workers: int | None = None) -> list[str]:
    # 防止底层数值库在每个进程内再开多线程导致过度竞争
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    if workers is None:
        # 128 核机器建议先 24~32；磁盘很强可以再上调
        workers = min(32, os.cpu_count() or 8, len(ranges))

    print(f"Spawn {workers} workers for {len(ranges)} ranges …")

    results = []
    # Windows 需 spawn，上层 main 里会设置
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(_process_one_range, s, e): (s, e) for (s, e) in ranges}
        for fut in as_completed(futs):
            s, e = futs[fut]
            try:
                outp = fut.result()
                results.append(outp)
            except Exception as err:
                print(f"[FAIL] ({s},{e}): {err}", file=sys.stderr)
                traceback.print_exc()
    return results


def main():
    outs = run_parallel(RANGES, workers=None)
    print("\nDone. Outputs:")
    for p in outs:
        print("  ", p)


if __name__ == "__main__":
    # Windows/跨平台安全
    try:
        if sys.platform.startswith("win"):
            mp.freeze_support()
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    main()
