# ============================================================
# export_one_link_metric.py
# 作用：
# 1) 读取某个 station-pair 的最短路径 CSV
# 2) 计算每时刻 intra/inter 链路与 hops
# 3) 计算端到端可靠性序列
# 4) 计算全局统计 / 窗口统计 / 路径切换统计
# ============================================================

from pathlib import Path
from datetime import datetime
import time

import src.model.get_intra_inter_link as get_intra_inter_link
import src.model.basiclink as basiclink
import src.paper3_postprocess.read_path_csv as read_path_csv
import src.paper3_postprocess.route_reliable as route_reliable
import src.paper3_postprocess.route_statistic as route_statistic
from draw.basic_functio.topology_config import TopologyRecorder, load_config


# ============================================================
# 0) 参数区（后续统一改这里）
# ============================================================
TOPOLOGY_VERSION = "gridx"

CSV_DIR = "region_pairs_0_86164"
PAIR_CSV_NAME = "region1--station2-region2--station8.csv"
P_INTRA = 0.999
P_INTER = 0.99

####

DATA_DIR = Path(r"D:\paper3")
BASEDIR = DATA_DIR / "data"
TOPOLOGY_DIR = "topology_design"

TOPO_CFG_PATH = BASEDIR / TOPOLOGY_DIR / TOPOLOGY_VERSION / "config" / "motif.json"
XML_FILE = BASEDIR / "satellitesposition" / "station_visible_satellites_20250106.xml"

WIN_START = 0
WIN_END = 86164

PATH_DIR = BASEDIR / TOPOLOGY_DIR / TOPOLOGY_VERSION / "path" / CSV_DIR

PAIR_CSV_PATH = PATH_DIR / PAIR_CSV_NAME


REL_COL = "rel_0999_099"
WINDOW_SEC = 300

# 可选：是否先重建静态拓扑（你后续要 viewer 时可直接开）
BUILD_STATIC_TOPOLOGY = False

# 可选：是否导出处理后的结果
EXPORT_RESULT = True
OUT_DIR = BASEDIR / TOPOLOGY_DIR / TOPOLOGY_VERSION / "analysis_link"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 工具函数
# ============================================================
_T0 = time.perf_counter()


def log(msg: str):
    now = datetime.now().strftime("%H:%M:%S")
    dt = time.perf_counter() - _T0
    print(f"[{now} | +{dt:8.2f}s] {msg}", flush=True)


def build_static_edges(cfg, start_step: int, end_step: int):
    """
    复用你 notebook 的逻辑：
    motif render 一次 + 双向化 + intra ring
    """
    n = cfg.N
    p = cfg.P

    rec = TopologyRecorder(cfg.P, cfg.N)
    rec._motifs = cfg.motifs

    inter_adj = rec.render_adj_at(
        t=start_step,
        eval_env={"start_ts": start_step, "end_ts": end_step + 1},
    )
    raw_inter_once = basiclink.make_edges_bidirectional(inter_adj)

    base_neighbors = {
        i * n + j: (i * n + ((j + 1) % n), i * n + ((j - 1) % n))
        for i in range(p) for j in range(n)
    }
    static_edges = {node: {r, l} for node, (r, l) in base_neighbors.items()}
    for src, dsts in raw_inter_once.items():
        static_edges.setdefault(src, set()).update(dsts)

    return static_edges


def append_intra_inter_metrics(df, n_per_orbit: int):
    """
    给 df 增加：
    intra_links / inter_links / intra_hops / inter_hops / total_hops
    """
    all_intra = []
    all_inter = []

    for path_str in df["path"]:
        intra, inter = get_intra_inter_link.parse_path_links(path_str, N=n_per_orbit)
        all_intra.append(intra)
        all_inter.append(inter)

    out = df.copy()
    out["intra_links"] = all_intra
    out["inter_links"] = all_inter
    out["intra_hops"] = out["intra_links"].apply(len)
    out["inter_hops"] = out["inter_links"].apply(len)
    out["total_hops"] = out["intra_hops"] + out["inter_hops"]
    return out




if __name__ == "__main__":
    log("Step 1/6: 加载 topology config")
    cfg = load_config(TOPO_CFG_PATH)
    n = cfg.N
    p = cfg.P
    log(f"config: P={p}, N={n}, motifs={len(cfg.motifs)}")

    if BUILD_STATIC_TOPOLOGY:
        log("Step 2/6: 重建 STATIC_EDGES（可选）")
        static_edges = build_static_edges(cfg, WIN_START, WIN_END)
        edge_cnt = sum(len(v) for v in static_edges.values())
        log(f"STATIC_EDGES done, nodes={len(static_edges)}, edge_refs={edge_cnt}")
    else:
        log("Step 2/6: 跳过 STATIC_EDGES 重建")

    log("Step 3/6: 读取 pair csv")
    df = read_path_csv.read_pair_csv(PAIR_CSV_PATH)
    if df.empty:
        raise ValueError(f"CSV 为空: {PAIR_CSV_PATH}")
    log(f"rows={len(df)}, cols={list(df.columns)}")

    log("Step 4/6: 计算 intra/inter hops")
    df = append_intra_inter_metrics(df, n_per_orbit=n)

    log("Step 5/6: 计算可靠性 + 统计")
    rel_series = route_reliable.compute_route_reliability_series(
        df,
        p_intra=P_INTRA,
        p_inter=P_INTER,
        intra_col="intra_hops",
        inter_col="inter_hops",
        name=REL_COL,
    )
    df[REL_COL] = rel_series

    global_stat = route_statistic.reliability_global_stats(df, rel_col=REL_COL)

    print(global_stat)
