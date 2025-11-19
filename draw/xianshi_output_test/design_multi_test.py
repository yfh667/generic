# -*- coding: utf-8 -*-
"""
极简并行版：
- 主进程：一次性 parse XML -> 得到完整 group_data，然后把每个区间的切片各自 dump 成 .pkl；
- 子进程：每个任务只读自己的切片 .pkl + 自己的 config.json，跑完整流水线并写出 XML。

注意：
- CONFIG_DIR 我设为 INPUT_DIR/config（与 TTB 无关，所有 TTB 复用同一套 {basicSa}_{end}.json）
- 输出目录按 TTB 建：INPUT_DIR/topology_{TTB}/raw
"""

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pickle, os, sys, traceback
from datetime import datetime

# =============== 基础路径与常量 ===============
from config import DATA_DIR, INPUT_DIR

XML_FILE = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"

# 你的区间（左闭右开）
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
    (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]

# 要批量跑的建链时间
# TTB_VALUES = [10,20,30,40,50, 70,80, 90, 100, 110,  120,130,140]
TTB_VALUES = [10,20,30,40,50, 60,70,80, 90, 100, 110,  120,130,140]
# 每个 TTB 的并行进程数（别把磁盘打爆，32 已很猛）
WORKERS_PER_TTB = min(32, os.cpu_count() or 8, len(RANGES))
SIMULATION_EDITION = 'motif3'
CONFIG_DIR = Path(INPUT_DIR) / f"{SIMULATION_EDITION}" / "config"

# SIMULATION_EDITION = 'motif2'

# # 默认："{SIMULATION_EDITION}/topology_{TIME_2_BUILD}"；可用环境变量 TOPOLOGY_VERSION 覆盖
# DEFAULT_VERSION = f"{SIMULATION_EDITION}/topology_{TIME_2_BUILD}"
# VERSION = os.getenv("TOPOLOGY_VERSION", DEFAULT_VERSION)
#
# RAW_DIR    = Path(INPUT_DIR) / VERSION / "raw"
# CONFIG_DIR = Path(INPUT_DIR) / f"{SIMULATION_EDITION}/config"
#
# # 配置目录（与 TTB 无关）
# # CONFIG_DIR = Path(INPUT_DIR) / "config"
# CONFIG_DIR.mkdir(parents=True, exist_ok=True)
#
# # 切片缓存目录（一次 parse 后，按区间 dump）
# SLICE_DIR = Path(INPUT_DIR) / "cache_slices"
# SLICE_DIR.mkdir(parents=True, exist_ok=True)
# 切片缓存目录（一次 parse 后，按区间 dump）
SLICE_DIR = Path(INPUT_DIR) / f"{SIMULATION_EDITION}" /  "cache_slices"
SLICE_DIR.mkdir(parents=True, exist_ok=True)

# =============== 业务依赖 ===============
import draw.read_snap_xml  as read_snap_xml
import draw.basic_functio.motif as motif
import draw.basic_functio.topology_config as topology_config
import draw.basic_functio.revdata2rawdata as revdata2rawdata
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import draw.basic_functio.conflict_link as conflict_link

import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.get_rectangular_size_interval as get_rectangular_size_interval
import draw.basic_functio.conflict_link as conflict_link

def dirs_and_xmls(ttb: int):
    """
    返回 (modify_dir, figure_dir, xml_paths)
    modify_dir: INPUT_DIR/topology_{ttb}/modify
    figure_dir: INPUT_DIR/topology_{ttb}/figure
    xml_paths:  该 TTB 对应的 13 段 XML 完整路径列表
    """


    DEFAULT_VERSION = f"{SIMULATION_EDITION}/topology_{ttb}"
    # version = version_name(ttb)
    version = os.getenv("TOPOLOGY_VERSION", DEFAULT_VERSION)
    modify_dir = Path(INPUT_DIR) / version / "modify"
    figure_dir = Path(INPUT_DIR) / version / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)

    xml_paths = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
    return modify_dir, figure_dir, xml_paths

# =============== 工具函数 ===============
def _slice_group_data(raw_group_data: dict, start: int, end: int) -> dict:
    """从完整 group_data 中裁剪 [basicSa, end)（保持你原来的结构：{step: {'groups': {gid:set}, 'all_mentioned': set}}）"""
    return {
        step: raw_group_data[step]
        for step in range(start, end)
        if step in raw_group_data
    }

def _version_name(ttb: int) -> str:
    return f"topology_{ttb}"

def _ensure_dirs_for_ttb(ttb: int) -> Path:

    DEFAULT_VERSION = f"{SIMULATION_EDITION}/topology_{ttb}"
    # version = version_name(ttb)
    version = os.getenv("TOPOLOGY_VERSION", DEFAULT_VERSION)


    out_dir = Path(INPUT_DIR) / version/ "raw"

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def _dump_slice_to_file(slice_obj: dict, out_path: Path) -> None:
    # 直接 pickle（set/list 都可），优先速度；如需更省空间可换 gzip 压缩。
    with open(out_path, "wb") as f:
        pickle.dump(slice_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_slice_from_file(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def _manifest_write_line(ttb: int, start_ts: int, end_ts: int, out_path: Path, manifest_file: Path):
    # 简单记录一下产物；可按需调整字段
    line = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "ttb": ttb,
        "range": [start_ts, end_ts],
        "output": str(out_path),
    }
    with open(manifest_file, "a", encoding="utf-8") as f:
        f.write(str(line) + "\n")


# =============== 子进程任务 ===============
def _worker_one_range(ttb: int, start_ts: int, end_ts: int, pkl_path: str, out_dir: str) -> str:
    """子进程任务：加载自己的区间切片 + 配置，跑完整流程并写 XML。"""
    # 限制子进程里数值库的内部线程，避免过度竞争
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    # 1) 区间切片
    group_data = _load_slice_from_file(Path(pkl_path))
    if not group_data:
        raise RuntimeError(f"空切片: {pkl_path}")

    # 2) 配置（约定放在 CONFIG_DIR/{basicSa}_{end}.json）
    modify_dir, figure_dir, xml_paths = dirs_and_xmls(ttb)



    cfg_path = CONFIG_DIR / f"{start_ts}_{end_ts}.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"缺少配置文件: {cfg_path}")


    cfg = topology_config.load_config(CONFIG_DIR / f"{start_ts}_{end_ts}.json")
    rev_group_data, offset = read_snap_xml.modify_group_data(
        group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
    )
    # 渲染时提供变量环境（可以动态变更 time_2_build 等）
    env = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "time_2_build": ttb,
    }

    rec = topology_config.TopologyRecorder(cfg.P, cfg.N)
    rec.base_groupid = cfg.base_groupid
    rec._motifs = cfg.motifs

    # 渲染某一秒的邻接
    # adj_1232 = rec.render_adj_at(1232, eval_env=env)

    # 渲染整段并生成 all_rev_inter_edge（你的老变量名）
    all_rev_inter_edge = rec.render_adj_range(start_ts, end_ts, eval_env=env)



    # 6) 还原编号
    raw_inter_edge = revdata2rawdata.revedge2rawedge(all_rev_inter_edge, offset)
     # here 我们就得在这里进行一次简练切换




    # 7) 包络（可选，失败忽略）

    t1, t2 = get_rectangular_size_interval.calc_envelope_for_group(rev_group_data, [start_ts, end_ts], 0, cfg.P, cfg.N)
    t3, t4 = get_rectangular_size_interval.calc_envelope_for_group(rev_group_data, [start_ts, end_ts], 4, cfg.P, cfg.N)


    rects = {
        0: (t1, t2),
        4: (t3, t4),
    }

    # rects = {}
    # try:
    #     rects[0] = calc_envelope_for_group(rev_group_data, [start_ts, end_ts], 0, cfg.P, cfg.N)
    # except Exception:
    #     pass
    # try:
    #     rects[cfg.base_groupid] = calc_envelope_for_group(rev_group_data, [start_ts, end_ts], cfg.base_groupid, cfg.P, cfg.N)
    # except Exception:
    #     pass

    # 8) 建链时间约束
    # raw_edges_by_step, pending_edges = conflict_link.get_no_conflict_link(
    #     raw_inter_edge, offset, rects, start_ts, end_ts, ttb, cfg.N, cfg.P
    # )
    change_terminal, realnodes = conflict_link.get_no_conflict_link_test(raw_inter_edge, offset, rects, start_ts,
                                                                         end_ts, ttb,cfg.N, cfg.P)

    # 9) 边 -> 节点
    # all_nodes = inter_edge2nodes.trans_edge2node(raw_edges_by_step, cfg.P, cfg.N)

    # 10) 写 XML
    out_dir_p = Path(out_dir)
    out_path = out_dir_p / f"interplane_links_{start_ts}_{end_ts}.xml"
    # 你已有更快的 writer 可替换：write2xml.nodes_to_xml2(all_nodes, out_path)

    write2xml.nodes_to_xml_test(realnodes, out_path)

    # out_path2 = out_dir_p / f"interplane_pending_links_{start_ts}_{end_ts}.xml"
    # # 你已有更快的 writer 可替换：write2xml.nodes_to_xml2(all_nodes, out_path)
    # pending_nodes = inter_edge2nodes.trans_edge2node(pending_edges, cfg.P, cfg.N)
    #
    # write2xml.nodes_to_xml(pending_nodes, out_path2)


    print(f"[OK ttb={ttb}] ({start_ts},{end_ts}) -> {out_path}")
    return str(out_path)


# =============== 主流程 ===============
def main():
    # Windows/跨平台安全
    try:
        if sys.platform.startswith("win"):
            mp.freeze_support()
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # A. 主进程：一次性解析 XML -> 完整 group_data
    print("[STEP A] parse XML -> full group_data …")
    RAW_START, RAW_END = 0, 30152
    full = read_snap_xml.parse_xml_group_data(str(XML_FILE), RAW_START, RAW_END)
    print(f"[STEP A] done. steps={len(full)}")

    # B. 主进程：把每个区间的切片先落盘（避免把大对象传给子进程）
    print("[STEP B] dump slices for all ranges …")
    slice_paths: dict[tuple[int,int], Path] = {}
    for (s, e) in RANGES:
        p = SLICE_DIR / f"slice_{s}_{e}.pkl"
        if not p.exists():  # 已存在就直接复用
            sl = _slice_group_data(full, s, e)
            _dump_slice_to_file(sl, p)
        slice_paths[(s, e)] = p
    print(f"[STEP B] done. slices={len(slice_paths)} at {SLICE_DIR}")

    # C. 逐个 TTB 并行计算（每个 TTB 内对区间并行；外层顺序，避免磁盘极端争用）
    for ttb in TTB_VALUES:
        out_dir = _ensure_dirs_for_ttb(ttb)
        manifest = out_dir.parent / f"manifest_ttb_{ttb}.log"
        print(f"\n[STEP C] TTB={ttb}, outputs -> {out_dir}, workers={WORKERS_PER_TTB}")

        # 限制各进程内部的数值库线程
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        results = []
        with ProcessPoolExecutor(max_workers=WORKERS_PER_TTB,
                                 mp_context=mp.get_context("spawn")) as ex:
            futs = {}
            for (s, e), spath in slice_paths.items():
                fut = ex.submit(_worker_one_range, ttb, s, e, str(spath), str(out_dir))
                futs[fut] = (s, e)
            for fut in as_completed(futs):
                s, e = futs[fut]
                try:
                    outp = fut.result()
                    results.append(outp)
                    _manifest_write_line(ttb, s, e, Path(outp), manifest)
                except Exception as err:
                    print(f"[FAIL ttb={ttb}] ({s},{e}): {err}", file=sys.stderr)
                    traceback.print_exc()

        print(f"[STEP C] TTB={ttb} done. {len(results)} files written.")

    print("\nAll TTB done.")


if __name__ == "__main__":
    main()
