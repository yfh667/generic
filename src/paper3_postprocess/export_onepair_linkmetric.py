from pathlib import Path

import src.model.get_intra_inter_link as get_intra_inter_link
import src.paper3_postprocess.read_path_csv as read_path_csv
import src.paper3_postprocess.route_reliable as route_reliable
import src.paper3_postprocess.route_statistic as route_statistic
from draw.basic_functio.topology_config import load_config


DATA_DIR = Path(r"D:\paper3")
BASEDIR = DATA_DIR / "data"
TOPOLOGY_DIR = "topology_design"


def compute_pair_global_stat(
    topology_version: str,
    csv_dir: str,
    pair_csv_name: str,
    p_intra: float,
    p_inter: float,
):
    """
    输入:
      topology_version, csv_dir, pair_csv_name, p_intra, p_inter

    输出:
      global_stat (pandas Series)
    """
    topo_cfg_path = BASEDIR / TOPOLOGY_DIR / topology_version / "config" / "motif.json"
    pair_csv_path = BASEDIR / TOPOLOGY_DIR / topology_version / "path" / csv_dir / pair_csv_name

    cfg = load_config(topo_cfg_path)
    n = cfg.N

    df = read_path_csv.read_pair_csv(pair_csv_path)
    if df.empty:
        raise ValueError(f"CSV 为空: {pair_csv_path}")

    all_intra = []
    all_inter = []

    for path_str in df["path"]:
        intra, inter = get_intra_inter_link.parse_path_links(path_str, N=n)
        all_intra.append(intra)
        all_inter.append(inter)

    df = df.copy()
    df["intra_hops"] = [len(x) for x in all_intra]
    df["inter_hops"] = [len(x) for x in all_inter]

    rel_col = "reliability"
    df[rel_col] = route_reliable.compute_route_reliability_series(
        df,
        p_intra=p_intra,
        p_inter=p_inter,
        intra_col="intra_hops",
        inter_col="inter_hops",
        name=rel_col,
    )

    global_stat = route_statistic.reliability_global_stats(df, rel_col=rel_col)
    return global_stat


if __name__ == "__main__":
    TOPOLOGY_VERSION = "gridx"
    CSV_DIR = "region_pairs_0_86164"
    PAIR_CSV_NAME = "region1--station2-region2--station8.csv"
    P_INTRA = 0.999
    P_INTER = 0.99

    global_stat = compute_pair_global_stat(
        TOPOLOGY_VERSION,
        CSV_DIR,
        PAIR_CSV_NAME,
        P_INTRA,
        P_INTER,
    )
    print(global_stat)
