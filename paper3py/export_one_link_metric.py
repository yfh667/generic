#0. 最初要改变的变量

import src.model.get_intra_inter_link as get_intra_inter_link
Topology_Version = 'gridx'
P=18
N=36
from pathlib import Path
from src.viz.pyqt_main2 import SatelliteViewer
DATA_DIR = Path(r"D:\paper3")
BASEDIR =  DATA_DIR / "data"

Topology_DIR = 'topology_design'
Topology_Version = 'gridx'
TOPO_CFG_PATH=BASEDIR / Topology_DIR / Topology_Version/"config" / "motif.json"
WIN_START = 0
WIN_END = 86164
xml_file = BASEDIR / 'satellitesposition' / "station_visible_satellites_20250106.xml"

import src.model.basiclink as basiclink
from draw.basic_functio.topology_config import TopologyRecorder, load_config

cfg = load_config(BASEDIR / Topology_DIR / Topology_Version / "config" / "motif.json")
N = cfg.N
P = cfg.P

rec = TopologyRecorder(cfg.P, cfg.N)
rec._motifs = cfg.motifs

# 静态拓扑，只 render 一次
inter_adj = rec.render_adj_at(
    t=WIN_START,
    eval_env={"start_ts": WIN_START, "end_ts": WIN_END + 1}
)

raw_inter_once = inter_adj
raw_inter_once = basiclink.make_edges_bidirectional(raw_inter_once)

base_neighbors = {
    i * N + j: (i * N + ((j + 1) % N), i * N + ((j - 1) % N))
    for i in range(P) for j in range(N)
}
STATIC_EDGES = {node: {r, l} for node, (r, l) in base_neighbors.items()}
for src, dsts in raw_inter_once.items():
    STATIC_EDGES.setdefault(src, set()).update(dsts)


DATA_DIR = Path(r"D:\paper3")
BASEDIR =  DATA_DIR / "data"

Topology_DIR = 'topology_design'

FIGURE_DIR = BASEDIR / Topology_DIR / Topology_Version/"path"

data_DIR = Path(FIGURE_DIR) / "region_pairs_0_86164"   # 你导出的原始 csv 目录

import  src.paper3_postprocess.read_path_csv as read_path_csv
# data_DIR = Path(FIGURE_DIR) / "region1_to_region2_0_100"   # 你导出的原始 csv 目录

df = read_path_csv.read_pair_csv(data_DIR / "region1--station2-region2--station8.csv")

all_intra = []
all_inter = []

for idx, path_str in enumerate(df["path"]):
    intra, inter = get_intra_inter_link.parse_path_links(path_str, N=N)
    all_intra.append(intra)
    all_inter.append(inter)

# 写回 DataFrame
df["intra_links"] = all_intra    # 每行是 [(src,dst), ...] 的 list
df["inter_links"] = all_inter

# 同时统计跳数
df["intra_hops"] = df["intra_links"].apply(len)
df["inter_hops"] = df["inter_links"].apply(len)
df["total_hops"] = df["intra_hops"] + df["inter_hops"]

import  src.paper3_postprocess.route_reliable as route_reliable

rel_0999_099 = route_reliable.compute_route_reliability_series(
    df,
    p_intra=0.999,
    p_inter=0.99,
)
df = df.copy()
df["rel_0999_099"] = rel_0999_099   # rel_0999_099 是你已算好的 Series

import src.paper3_postprocess.route_statistic as route_statistic
global_stat = route_statistic.reliability_global_stats(df, rel_col="rel_0999_099")
win_5min = route_statistic.reliability_window_stats(df, rel_col="rel_0999_099", window_sec=300)
switch_summary, switch_segments = route_statistic.path_switch_stats(df, path_col="path")

print(global_stat)
print(win_5min.head())
print(switch_summary)