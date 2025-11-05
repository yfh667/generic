# -*- coding: utf-8 -*-
# 并行导出各 TTB 的 “G0↔G4 平均最短路 vs 时间” 到 CSV

import os, sys, time, traceback, argparse, multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ===== 项目依赖 =====
from config import DATA_DIR, INPUT_DIR
import genaric2.tegnode as tegnode
import draw.read_snap_xml as read_snap_xml
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.inter_edge2nodes as inter_edge2nodes
import  draw.pymatlab2.chartalgorithm.plot_2city_shortest_path as plot_2city_shortest_path

# ===== 星座 & 时间段 =====
P, N = 18, 36
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
    (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]
START_TS, END_TS = RANGES[0][0], RANGES[-1][1]   # [0, 22005)

DEFAULT_TTB_VALUES = [10,20,30,40,50,60,70,80,90,100,110,120,130,140]

# ===== 路径 & 公共数据缓存 =====
def version_name(ttb: int) -> str:
    return f"topology_{ttb}"

def dirs_and_xmls(ttb: int):
    version = version_name(ttb)
    modify_dir = Path(INPUT_DIR) / version / "modify"
    figure_dir = Path(INPUT_DIR) / version / "figure"
    figure_dir.mkdir(parents=True, exist_ok=True)
    xml_paths = [modify_dir / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
    xml_file2 = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"
    return modify_dir, figure_dir, xml_paths,xml_file2

def cache_dir() -> Path:
    d = Path(INPUT_DIR) / "_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d

def group_cache_path() -> Path:
    return cache_dir() / f"group_data_{START_TS}_{END_TS}.pkl"

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
    """单个 TTB：读 inter 边 -> 合成每步图 -> 计算均最短路 -> 导出 CSV"""
    # 限制子进程内部的并行库，以免过度竞争
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    t0 = time.time()
    modify_dir, figure_dir, xml_paths,xml_file2 = dirs_and_xmls(ttb)

    # 检查 inter XML
    missing = [str(p) for p in xml_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"[TTB={ttb}] 缺少 XML（{len(missing)}），例如：{missing[:2]} ...")

    print(f"[TTB={ttb}] 读取 inter XML（{len(xml_paths)} 个）…")
    totalnode = write2xml.load_all_nodes_sequential(xml_paths, tegnode.tegnode_complete)
    series = read_snap_xml.parse_station_timeseries(xml_file2, [0, 1, 11, 13], START_TS, END_TS)

    all_inter_edge = inter_edge2nodes.trans_nodes2edges(totalnode, P, N)

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


    paris = series[3]
    chongqin = series[1]
    huasha = series[2]
    beijin = series[0]

    csv_path = plot_2city_shortest_path.export_stationpair_min_hops_to_origin(
        all_edges, huasha, beijin,
        out_dir=figure_dir,
        basename=f"minhops_huasha_beijin_ttb{ttb}",
        steps=(START_TS, END_TS),
        undirected=True,
        with_pair=True,
        with_path=False
    )


    dt = time.time() - t0
    print(f"[TTB={ttb}] 完成，用时 {dt:.1f}s -> {csv_path}")
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
    ensure_group_data_cached()

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
