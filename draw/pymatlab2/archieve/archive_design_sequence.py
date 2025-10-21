# -*- coding: utf-8 -*-
"""
按 RANGES 逐段运行：
1) 读取配置 (INPUT_DIR/<VERSION>/config/{basicSa}_{end}.json)
2) modify_group_data -> motifs -> 同构边
3) 反变换为原始边 + 建链时间约束 + 包络矩形约束
4) 边->节点 并写出 XML 到 RAW_DIR (INPUT_DIR/<VERSION>/raw/interplane_links_{basicSa}_{end}.xml)
"""

from __future__ import annotations
from pathlib import Path
import os
import sys
import traceback

# ========= 基础参数 =========
from config import DATA_DIR, INPUT_DIR

# 星座参数
N = 36
P = 18

# 建链时间（秒/步）
time_2_build = 70

# 版本目录（可改成环境变量或命令行参数）
VERSION = os.getenv("TOPOLOGY_VERSION", "version3")
CONFIG_DIR = (Path(INPUT_DIR) / VERSION / "config")
RAW_DIR    = (Path(INPUT_DIR) / VERSION / "raw")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

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

# ========= 工具函数 =========
def slice_group_data(raw_group_data: dict, start: int, end: int) -> dict:
    """裁剪时间区间 [basicSa, end)"""
    return {step: raw_group_data[step] for step in range(start, end) if step in raw_group_data}

def process_one_range(raw_group_data: dict, start_ts: int, end_ts: int) -> str:
    """
    对单个 (start_ts, end_ts) 区间执行全流程。
    返回输出文件路径（str）。
    """
    # 读取配置
    cfg_path = CONFIG_DIR / f"{start_ts}_{end_ts}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {cfg_path}")

    cfg = topology_config.load_config(str(cfg_path))  # 只含 P,N,base_groupid,motifs

    # 裁剪 group_data
    group_data = slice_group_data(raw_group_data, start_ts, end_ts)

    # 归一化（反向缝对齐）
    rev_group_data, offset = read_snap_xml.modify_group_data(
        group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
    )

    # 写 motifs => 同构邻接
    nodes = {}
    for m in cfg.motifs:
        motif.write_distinct_motif(
            m.p_start, m.p_end, m.y_start, m.y_end,
            cfg.P, cfg.N, nodes, option=getattr(m, "option", 0)
        )
    rev_inter_edge = motif.transform_nodes_2_adjacent(nodes, cfg.P, cfg.N)

    # 同构边扩展到整个区间（每个 step 相同的同构拓扑）
    all_rev_inter_edge = {t: rev_inter_edge for t in range(start_ts, end_ts)}

    # 反变换到原始编号
    raw_inter_edge = revdata2rawdata.revedge2rawedge(all_rev_inter_edge, offset)

    # 包络矩形（示例：对 group 0 与 base_groupid 计算）
    # 根据你之前习惯：0 和 4；这里用 cfg.base_groupid 动态替代 4
    try:
        t0 = calc_envelope_for_group(rev_group_data, [start_ts, end_ts], 0,               cfg.P, cfg.N)
    except Exception:
        t0 = None
    try:
        tb = calc_envelope_for_group(rev_group_data, [start_ts, end_ts], cfg.base_groupid, cfg.P, cfg.N)
    except Exception:
        tb = None

    rects = {}
    if t0 is not None:
        rects[0] = t0
    if tb is not None:
        rects[cfg.base_groupid] = tb

    # 建链时间 + 包络矩形约束 -> 生成 raw_edges_by_step / pending_edges
    raw_edges_by_step, pending_edges = conflict_link.get_no_conflict_link(
        raw_inter_edge, offset, rects, start_ts, end_ts, time_2_build, cfg.N, cfg.P
    )

    # 边 -> 节点（只存异轨链路）
    all_nodes = inter_edge2nodes.trans_edge2node(raw_edges_by_step, cfg.P, cfg.N)

    # 写 XML 到 RAW_DIR
    out_path = RAW_DIR / f"interplane_links_{start_ts}_{end_ts}.xml"
    write2xml.nodes_to_xml(all_nodes, out_path)

    print(f"[OK] ({start_ts},{end_ts}) -> {out_path}")
    return str(out_path)

def main():
    # 一次性解析可见性大文件（避免每段都重复读）
    xml_file = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"
    RAW_START, RAW_END = 0, 30152
    raw_group_data = read_snap_xml.parse_xml_group_data(xml_file, RAW_START, RAW_END)

    results = []
    for (start_ts, end_ts) in RANGES:
        try:
            outp = process_one_range(raw_group_data, start_ts, end_ts)
            results.append(outp)
        except Exception as e:
            print(f"[FAIL] ({start_ts},{end_ts}): {e}", file=sys.stderr)
            traceback.print_exc()

    print("\nDone. Outputs:")
    for p in results:
        print("  ", p)

if __name__ == "__main__":
    # 可选：限制底层数值库线程，防止多线程/多进程内卷
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    main()
