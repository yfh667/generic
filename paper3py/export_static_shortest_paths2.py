# src/scripts/export_shortest_path.py
# ====================================================================
# 功能: 基于静态拓扑(× grid 或 + grid motif)，批量导出所有区域对
#       的最短路径 CSV，包含 path、intra/inter 跳数、端到端可靠性。
#
# 复用模块:
#   - src.model.static_hop_table  (Floyd 预计算 + 路径重构)
#   - src.io.read_snap_xml        (XML 解析接入卫星)
#   - src.config.viewer_config    (星座/分组配置)
#   - draw.basic_functio.topology_config (motif → 拓扑)
#   - draw.basic_functio.motif    (transform_nodes_2_adjacent)
#
# 对标 notebook: design_grid+new.ipynb 中
#   "static_hop_table" + "compute_region_pair_timeseries" 那一整段
# ====================================================================


import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from datetime import datetime
import src.model.basiclink as basiclink
# ---------- 项目内模块 ----------

from src.config.viewer_config import G60_CONFIG
from src.io import read_snap_xml
from src.io.operate_group_data import slice_group_data
import src.model.static_hop_table as static_hop_table
from draw.basic_functio.topology_config import TopologyRecorder
from draw.basic_functio import motif as motif_mod
from draw.basic_functio.topology_config import load_config
from src.paper3_postprocess.static_hop_table_fast_patch import export_pair_csvs_streaming
# ====================================================================
# 0) 日志
# ====================================================================
_t0 = time.time()
def log(msg):
    elapsed = time.time() - _t0
    print(f"[{datetime.now().strftime('%H:%M:%S')} | +{elapsed:8.2f}s] {msg}")



#0. 最初要改变的变量
Topology_Version = 'grid+'



# ====================================================================
# 1) 基础参数（对标 notebook 前几个 cell）
# ====================================================================
P = G60_CONFIG.P         # 18
N = G60_CONFIG.N         # 36
TOTAL_SATS = G60_CONFIG.total_sats  # 648




DATA_DIR = Path(r"D:\paper3")
BASEDIR =  DATA_DIR / "data"

# VERSION1 = 'satellitesposition'




BASEDIR = DATA_DIR / "data"
Topology_DIR = 'topology_design'
# VERSION1 = "satellitesposition"
# xml_file = BASEDIR / VERSION1 / "station_visible_satellites_20250106.xml"
xml_file = BASEDIR / 'satellitesposition' / "station_visible_satellites_20250106.xml"


RAW_START = 0
RAW_END   = 86164        # 一个恒星日

# 处理窗口
WIN_START = 0
WIN_END   = RAW_END           # 含端点，共 100 个 step

# 输出目录
FIGURE_DIR = BASEDIR / Topology_DIR / Topology_Version/"path"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# # 概率参数
# P_INTRA = 0.999
# P_INTER = 0.99

log("Step 1: 构建 × grid motif 静态拓扑")
# ====================================================================
# 2) 构建 × grid 静态拓扑（对标 notebook "Grid ×" + "motif设计" cell）
# ====================================================================


# rec = TopologyRecorder(P, N)
# nodes = {}

# × grid: 每对相邻行交叉连接
# for y in range(0, N, 2):
#     rec.write_distinct_motif(
#         p_start=0, p_end=P - 1,
#         y_start=y, y_end=y + 1,
#         nodes=nodes, option=4,     # 偶行斜上
#     )
#     rec.write_distinct_motif(
#         p_start=0, p_end=P - 1,
#         y_start=y, y_end=y + 1,
#         nodes=nodes, option=1,     # 奇行斜下
#     )
#



# 导入juptyer里已经弄好的motif configuration
cfg = load_config(BASEDIR / Topology_DIR / Topology_Version/"config" / "motif.json")
# 重建 recorder，把 motif 列表灌进去
rec = TopologyRecorder(cfg.P, cfg.N)
rec._motifs = cfg.motifs

# inter_adj = rec.render_adj_at(t=0, eval_env={"start_ts": 0, "end_ts": 1})

# render 一次静态 inter 拓扑
inter_once = rec.render_adj_at(
    t=WIN_START,
    eval_env={"start_ts": WIN_START, "end_ts": WIN_END},
)

# 双向化
# def make_edges_bidirectional(edge_dict):
#     new_edges = {}
#     for src, dsts in edge_dict.items():
#         for dst in dsts:
#             new_edges.setdefault(src, set()).add(dst)
#             new_edges.setdefault(dst, set()).add(src)
#     return new_edges

raw_inter_once = basiclink.make_edges_bidirectional(inter_once)

# intra ring 邻居 + inter motif → 完整静态图
base_neighbors = {
    i * N + j: ((i * N + (j + 1) % N), (i * N + (j - 1) % N))
    for i in range(P) for j in range(N)
}

static_edges = {node: {r, l} for node, (r, l) in base_neighbors.items()}
for src, dsts in raw_inter_once.items():
    static_edges.setdefault(src, set()).update(dsts)




# ====================================================================
# 3) Floyd 预计算（对标 notebook "precompute_hop_and_next_hop" cell）
# ====================================================================
log("Step 2: 构建 NetworkX 图 + Floyd 预计算")


G = static_hop_table.build_static_graph(
    {0: static_edges},   # 只需一个 step 的 adj，key 随便写
    TOTAL_SATS,
)

log(f"  G: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

dist, next_hop = static_hop_table.precompute_hop_and_next_hop(G, TOTAL_SATS)
log(f"  dist matrix shape={dist.shape}")


# ====================================================================
# 4) 读取接入卫星时间序列（对标 notebook "parse_all_station_timeseries" cell）
# ====================================================================
log("Step 3: 读取接入卫星时间序列")

# 区域定义（从 G60_CONFIG.station_groups 读取）
all_regions = {}
for gid, info in G60_CONFIG.station_groups.items():
    all_regions[gid] = info["stations"]

# 收集所有 station id
all_station_ids = sorted(set(
    sid for stations in all_regions.values() for sid in stations
))
log(f"  regions={len(all_regions)}, stations={len(all_station_ids)}")

series_list = read_snap_xml.parse_station_timeseries(
    xml_file, all_station_ids, WIN_START, WIN_END
)
series_by_station = {sid: ts for sid, ts in zip(all_station_ids, series_list)}
log(f"  series_by_station 加载完成")


# ====================================================================
# 5) 构造 station pairs（所有区域对 × 所有站对）
# ====================================================================
log("Step 4: 构造 station pairs")

region_ids = sorted(all_regions.keys())
pairs = []


for ra, rb in combinations(region_ids, 2):
    for sa in all_regions[ra]:
        for sb in all_regions[rb]:
            pairs.append((sa, sb))
log(f"  区域对={len(list(combinations(region_ids, 2)))}, 站对={len(pairs)}")

steps = sorted({int(t) for ts in series_by_station.values() for t in ts.keys()})
log(f"  actual_steps={len(steps)}")
log(f"  expected_rows={len(steps) * len(pairs):,}")
# ====================================================================
# 6) 批量查表（对标 notebook "compute_region_pair_timeseries" cell）
# ====================================================================
# log("Step 5: 批量查表计算最短路径")
#
# df_all = static_hop_table.compute_region_pair_timeseries(
#     dist=dist,
#     next_hop=next_hop,
#     series_by_station=series_by_station,
#     station_pairs=pairs,
#     steps=range(WIN_START, WIN_END + 1),
#     left="region1",
#     right="region2",
# )
# log(f"  结果行数={len(df_all)}")
#
#
#
# # ====================================================================
# # 8) 导出 CSV（对标 notebook "export_pair_csvs" cell）
# # ====================================================================
# log("Step 7: 导出 CSV")
#
# out_dir = FIGURE_DIR / f"region_pairs_{WIN_START}_{WIN_END}"
# csv_paths = static_hop_table.export_pair_csvs(
#     df_all, out_dir, left="region1", right="region2"
# )
# log(f"  导出 {len(csv_paths)} 个 CSV 到 {out_dir}")

log("Step 5: 流式导出最短路径 CSV")

out_dir = FIGURE_DIR / f"region_pairs_{steps[0]}_{steps[-1]}"
csv_paths = export_pair_csvs_streaming(
    dist=dist,
    next_hop=next_hop,
    series_by_station=series_by_station,
    station_pairs=pairs,
    steps=steps,
    out_dir=out_dir,
    left="region1",
    right="region2",
    include_path=True,
    include_path_indexed=False,
)

log(f"  导出 {len(csv_paths)} 个 CSV 到 {out_dir}")

# ====================================================================
# 9) 打印摘要
# ====================================================================
print("\n" + "=" * 60)
print("导出完成摘要")
print("=" * 60)
print(f"  拓扑类型   : × grid (option 4+1)")
print(f"  星座       : P={P}, N={N}, 共{TOTAL_SATS}颗")
print(f"  时间窗口   : [{WIN_START}, {WIN_END}]")
print(f"  区域数     : {len(all_regions)}")
print(f"  站点总数   : {len(all_station_ids)}")
print(f"  站对总数   : {len(pairs)}")
# print(f"  p_intra    : {P_INTRA}")
# print(f"  p_inter    : {P_INTER}")
print(f"  输出目录   : {out_dir}")
print(f"  CSV 文件数 : {len(csv_paths)}")
print("=" * 60)
