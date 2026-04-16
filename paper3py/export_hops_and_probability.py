#0. 最初要改变的变量
import src.paper3_postprocess.plot_hops_and_reliability_summary as plot_hops_and_reliability_summary
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

xml_file = BASEDIR / 'satellitesposition' / "station_visible_satellites_20250106.xml"
import draw.read_snap_xml  as read_snap_xml
from src.config.viewer_config import ViewerConfig, G60_CONFIG

# 当我们修改df的时候，实际上，下面的是无需去修改的
# df 就是你前面已经读好的 DataFrame



# 2) 时间窗直接用 df 的范围，不要整天都画
WIN_START = 0
WIN_END = 86164

# 区域定义（从 G60_CONFIG.station_groups 读取）
all_regions = {}
for gid, info in G60_CONFIG.station_groups.items():
    all_regions[gid] = info["stations"]

# 收集所有 station id
all_station_ids = sorted(set(
    sid for stations in all_regions.values() for sid in stations
))

# 如果你要严格用 1..11 和 12..21，请改成：
# region_a_stations = list(range(1, 12))
# region_b_stations = list(range(12, 22))



series_list = read_snap_xml.parse_station_timeseries(
    xml_file, all_station_ids, WIN_START, WIN_END - 1
)
series_by_station = {sid: ts for sid, ts in zip(all_station_ids, series_list)}
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



##  here we need get the all the file in D:\paper3\data\topology_design\gridx\path\region_pairs_0_86164

df = read_path_csv.read_pair_csv(data_DIR / "region1--station15-region2--station24.csv")


import src.model.get_intra_inter_link as get_intra_inter_link

df = df.copy()

all_intra = []
all_inter = []

for path_str in df["path"]:
    if isinstance(path_str, str) and path_str.strip():
        intra, inter = get_intra_inter_link.parse_path_links(path_str, N=N)
    else:
        intra, inter = [], []

    all_intra.append(intra)
    all_inter.append(inter)

df["intra_links"] = all_intra
df["inter_links"] = all_inter

df["intra_hops"] = df["intra_links"].apply(len)
df["inter_hops"] = df["inter_links"].apply(len)
df["total_hops"] = df["intra_hops"] + df["inter_hops"]

import  src.paper3_postprocess.route_reliable as route_reliable
rel_0999_099 = route_reliable.compute_route_reliability_series(
    df,
    p_intra=0.999,
    p_inter=0.99,
)

df[rel_0999_099.name] = rel_0999_099
df_s6_s8 = df
station_a = int(df_s6_s8["station_a"].iloc[0])
station_b = int(df_s6_s8["station_b"].iloc[0])
fig, axes, df_plot = plot_hops_and_reliability_summary.plot_hops_and_reliability_summary(
    df,
    reliability_col=rel_0999_099.name,
    title=f"Station {station_a} ↔ Station {station_b}: Hops and Reliability",
    hops_title="Intra / Inter / Total Hops",
    reliability_title="End-to-End Route Reliability",
    show=False,
    save=True,
    save_dir="figs/one_link_summary",
    basename=f"s{station_a}_s{station_b}_hops_reliability",
)
