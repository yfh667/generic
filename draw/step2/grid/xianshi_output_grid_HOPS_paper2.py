# -*- coding: utf-8 -*-
# 并行导出各 TTB 的 “G0↔G4 平均最短路 vs 时间” 到 CSV

import os, sys, time, traceback, argparse, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
import  draw.pymatlab2.chartalgorithm.plot_intergroup_avg_shortest_path as plot_intergroup_avg_shortest_path

# ===== 项目依赖 =====
from config import DATA_DIR, INPUT_DIR
import genaric2.tegnode as tegnode
import draw.read_snap_xml  as read_snap_xml
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import draw.pymatlab2.chartalgorithm.plot_intergroup_avg_shortest_path as avgsp  # 你之前的模块（含 export_intergroup_avgspath_to_origin）
import draw.basic_functio.motif as motif
import draw.basic_functio.topology_config as topology_config
# 注意上述我们是在同构拓扑序列上进行的，因此，我们还要将同构拓扑序列进行还原，同时，我们还要考虑到建链时间约束
import draw.basic_functio.revdata2rawdata as revdata2rawdata
# ===== 星座 & 时间段 =====
P, N = 18, 36
# RANGES = [
#     (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
#     (11640,13057),(13057,14065),(14065,16604),(16604,18396),
#     (18396,19831),(19831,20814),(20814,22005)
# ]
START_TS, END_TS  =0,86400

DEFAULT_TTB_VALUES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# SIMULATION_EDITION = 'motif2'
# DEFAULT_TTB_VALUES = [60]
# ===== 路径 & 公共数据缓存 =====
# def version_name(ttb: int) -> str:
#     return f"topology_{ttb}"

def dirs_and_xmls(ttb: int):


    # DEFAULT_VERSION = f"{SIMULATION_EDITION}/topology_{ttb}"
    # # version = version_name(ttb)
    # version = os.getenv("TOPOLOGY_VERSION", DEFAULT_VERSION)
    # # version = version_name(ttb)
    # modify_dir = Path(INPUT_DIR) / version / "modify"
    # figure_dir = Path(INPUT_DIR) / version / "figure"
    # figure_dir.mkdir(parents=True, exist_ok=True)
    # xml_paths = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]

    DATA_DIR = Path(r"C:\usrspace\mywork\data_paper2")
    BASEDIR = DATA_DIR / "visibile_data"

    VERSION1 = 'paper1_G60'


    # 基准日期：2025-01-06
    base_date = datetime.strptime("20250106", "%Y%m%d").date()
    day_date = base_date + timedelta(days=ttb)

    version2 = f"day_{day_date.strftime('%Y%m%d')}"
    FIGURE_DIR = BASEDIR / VERSION1 / version2/"path"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)  # 不存在就创建（包含父目录）



    xml_file = BASEDIR / VERSION1 / version2 / f"station_visible_satellites_{day_date.strftime('%Y%m%d')}.xml"




    return xml_file,FIGURE_DIR

def cache_dir() -> Path:
    d = Path(INPUT_DIR) / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def group_cache_path() -> Path:
    return cache_dir() / f"group_data_{START_TS}_{END_TS}.pkl"

def make_edges_bidirectional(edge_dict):
    """{basicSa: set(dsts)} -> 双向"""
    new_edges = {}
    for src, dsts in edge_dict.items():
        for dst in dsts:
            new_edges.setdefault(src, set()).add(dst)
            new_edges.setdefault(dst, set()).add(src)
    return new_edges


def ensure_group_data_cached():
    """主进程调用：若无缓存则 parse 一次并写入磁盘。"""
    pkl = group_cache_path()
    if pkl.exists():
        print(f"[CACHE] use cached group_data: {pkl}")
        return pkl
    xml_file = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"
    if not xml_file.exists():
        raise FileNotFoundError(f"XML not found: {xml_file}")
    print(f"[CACHE] parse group XML once: {xml_file} range=[0,22006)")
    raw = read_snap_xml.parse_xml_group_data(xml_file, 0, 22006)
    # 只保留 [START_TS, END_TS)
    group_data = {t: raw[t] for t in range(START_TS, END_TS) if t in raw}
    import pickle
    with open(pkl, "wb") as f:
        pickle.dump(group_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[CACHE] dump -> {pkl}")
    return pkl

def load_group_data_cached() -> dict:
    """子进程调用：读取缓存。"""
    import pickle
    with open(group_cache_path(), "rb") as f:
        return pickle.load(f)


def slice_group_data(raw_group_data, start, end):
    """
    从 raw_group_data 中裁剪时间区间 [basicSa, end)
    """
    return {
        step: raw_group_data[step]
        for step in range(start, end)
        if step in raw_group_data
    }

def run_one_ttb(ttb: int) -> tuple[int, list[str]]:
    """单个 TTB：读 inter 边 -> 合成每步图 -> 计算均最短路 -> 导出 CSV"""
    # 限制子进程内部的并行库，以免过度竞争
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.time()

    xml_file,FIGURE_DIR  = dirs_and_xmls(ttb)




    # 读取公共 group_data（缓存）
    group_data = read_snap_xml.parse_xml_group_data(xml_file, START_TS, END_TS )
    base_groupid_now = 1

    rev_group_data, offset = read_snap_xml.modify_group_data(group_data, P, N, base_groupid=base_groupid_now)

    rec = topology_config.TopologyRecorder(P, N)
    rec.modify_group_data(group_data, base_groupid=base_groupid_now)

    nodes = {}

    rec.write_distinct_motif(0, 17, 0, 35, nodes, option=0)
    env = {
        "start_ts": START_TS,
        "end_ts": END_TS,

    }

    all_rev_inter_edge = rec.render_adj_range(START_TS, END_TS, eval_env=env)


    raw_inter_edge = revdata2rawdata.revedge2rawedge(all_rev_inter_edge, offset)

    # 双向化 inter
    for step in list(raw_inter_edge.keys()):
        raw_inter_edge[step] = make_edges_bidirectional(raw_inter_edge[step])

    base_neighbors = {
        i * N + j: (i * N + ((j + 1) % N), i * N + ((j - 1) % N))
        for i in range(P) for j in range(N)
    }
    intra_template = {node: {r, l} for node, (r, l) in base_neighbors.items()}

    #all_inter_edge = inter_edge2nodes.trans_nodes2edges(totalnode, P, N)
    # all_inter_edge, pending_edge, iG_edge = motif.transform_nodes_2_rawedge_test(totalnode, P, N, START_TS, END_TS)

    TOTAL_START = START_TS
    TOTAL_END = END_TS  # 测试总长度
    CHUNK_SIZE = 10000  # 切片大小
    # 你的输出路径

    # 确保文件夹存在
    if not FIGURE_DIR.exists():
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== 开始顺序测试: 范围 {TOTAL_START}-{TOTAL_END}, 分片大小 {CHUNK_SIZE} ===")

    # ================= 2. 顺序循环执行 =================
    for batch_start in range(TOTAL_START, TOTAL_END, CHUNK_SIZE):
        batch_end = min(batch_start + CHUNK_SIZE, TOTAL_END)
        print(f"\n>> 正在处理分片: {batch_start} 到 {batch_end} ...")

        # --- A. 只构造当前 chunk 的拓扑 ---
        sub_edges = {}
        for step in range(batch_start, batch_end):
            adj = {node: set(neis) for node, neis in intra_template.items()}  # 每步复制一份





            inter = raw_inter_edge.get(step, {})
            for src, dsts in inter.items():
                adj.setdefault(src, set()).update(dsts)

            sub_edges[step] = adj

        print(f"   [数据] sub_edges 包含 {len(sub_edges)} 个时刻。")

        # --- B. 切片 group_data ---
        sub_group_data = slice_group_data(group_data, batch_start, batch_end)

        # --- C. 导出 ---
        basename = f"avgspath_g0_4_baseline_{batch_start}_to_{batch_end}"
        csv_path = plot_intergroup_avg_shortest_path.export_intergroup_avgspath_to_origin(
            all_edges=sub_edges,
            group_data=sub_group_data,
            out_dir=FIGURE_DIR,
            basename=basename,
            group_a=0,
            group_b=1,
            steps=(batch_start, batch_end),
            undirected=True
        )
        print(f"   [成功] 文件已生成: {csv_path}")

    print("\n=== 测试运行结束 ===")

    # # 双向化 inter
    # for step in list(all_inter_edge.keys()):
    #     all_inter_edge[step] = make_edges_bidirectional(all_inter_edge[step])



    # —— 构造 all_edges（含 intra+inter）供 avgsp.compute 使用 ——
    # 这里不生成巨大的 all_intra_edge；每步把环内边追加进去即可。
    # all_edges = {}
    # # 预计算 648 个节点的左右邻居
    # base_neighbors = {
    #     i * N + j: (i * N + ((j + 1) % N), i * N + ((j - 1) % N))
    #     for i in range(P) for j in range(N)
    # }
    # for step in range(START_TS, END_TS):
    #     adj = {}
    #     # intra（左右邻居）
    #     for node, (r, l) in base_neighbors.items():
    #         adj.setdefault(node, set()).update((r, l))
    #     # inter
    #     inter = all_inter_edge.get(step, {})
    #     for src, dsts in inter.items():
    #         adj.setdefault(src, set()).update(dsts)
    #     all_edges[step] = adj
    #
    # # 计算并导出 CSV
    # csv_path = avgsp.export_intergroup_avgspath_to_origin(
    #     all_edges, group_data,
    #     out_dir=figure_dir,
    #     basename=f"avgspath_{ttb}",
    #     group_a=0, group_b=4,
    #     steps=(START_TS, END_TS-1),
    #     undirected=True
    # )
    # dt = time.time() - t0
    # print(f"[TTB={ttb}] 完成，用时 {dt:.1f}s -> {csv_path}")



    return ttb, [str(csv_path)]

# ===== CLI =====
def parse_args():
    ap = argparse.ArgumentParser(description="批量导出各 TTB 的 G0↔G4 平均最短路（供 Origin 画图）")
    ap.add_argument("--ttb", type=str,
                    default=",".join(str(x) for x in DEFAULT_TTB_VALUES),
                    help=f"要计算的 TTB 列表，逗号分隔（默认: {DEFAULT_TTB_VALUES})")
    ap.add_argument("--workers", type=int,
                    default=max(1, min(4, mp.cpu_count() // 2)),
                    help="并行进程数（默认：min(4, CPU/2)）")
    return ap.parse_args()

def main():
    # Windows 安全
    if sys.platform.startswith("win"):
        mp.freeze_support()
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass

    # —— 先在主进程准备公共缓存 ——
    # ensure_group_data_cached()

    args = parse_args()
    ttb_values = [int(x) for x in args.ttb.split(",") if x.strip()]
    workers = max(1, min(args.workers, len(ttb_values)))

    print(f"TTB 列表: {ttb_values}")
    print(f"并行进程: {workers}\n")

    results, errors = {}, {}

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
