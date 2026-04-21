from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
import src.model.route_policy_probability as route_policy_probability


import src.paper3_postprocess.read_path_csv as read_path_csv

import src.paper3_postprocess.route_statistic as route_statistic
from draw.basic_functio.topology_config import load_config


DATA_DIR = Path(r"D:\paper3")
BASEDIR = DATA_DIR / "data"
TOPOLOGY_DIR = "topology_design"
def _numeric_stats_with_prefix(series, prefix: str) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").dropna()

    if x.empty:
        return pd.Series({
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_p95": float("nan"),
            f"{prefix}_max": float("nan"),
        })

    return pd.Series({
        f"{prefix}_count": int(x.size),
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_std": float(x.std(ddof=1)),
        f"{prefix}_min": float(x.min()),
        f"{prefix}_p05": float(x.quantile(0.05)),
        f"{prefix}_p10": float(x.quantile(0.10)),
        f"{prefix}_p90": float(x.quantile(0.90)),
        f"{prefix}_p95": float(x.quantile(0.95)),
        f"{prefix}_max": float(x.max()),
    })


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


    policy_path = BASEDIR / TOPOLOGY_DIR / topology_version / "config" / "route_policy.json"
    policy = route_policy_probability.load_route_policy(
        policy_path,
        n=n,
        fallback_p_intra=p_intra,
        fallback_p_inter=p_inter,
    )

    p_intra_used = float(policy["p_intra"])
    default_p_inter_used = policy["default_p_inter"]
    option_p_inter = dict(policy["option_p_inter"])
    delta2option = dict(policy["delta2option"])

    df = read_path_csv.read_pair_csv(pair_csv_path)
    if df.empty:
        raise ValueError(f"CSV 为空: {pair_csv_path}")

    rel_list = []
    intra_hops_list = []
    inter_hops_list = []
    unknown_edges_list = []
    option_counter_total = Counter()

    for path_str in df["path"].astype(str):
        rel, intra_hops, inter_hops, unknown_edges, option_counter = (
            route_policy_probability.compute_path_reliability_and_hops(
                path_str,
                n=n,
                p_intra=p_intra_used,
                default_p_inter=default_p_inter_used,
                option_p_inter=option_p_inter,
                delta2option=delta2option,
                unknown_option_action="use_default",
            )
        )
        rel_list.append(rel)
        intra_hops_list.append(intra_hops)
        inter_hops_list.append(inter_hops)
        unknown_edges_list.append(unknown_edges)
        option_counter_total.update(option_counter)

    df = df.copy()
    df["intra_hops"] = intra_hops_list
    df["inter_hops"] = inter_hops_list
    df["unknown_option_edges"] = unknown_edges_list

    rel_col = "reliability"
    df[rel_col] = rel_list




  #  global_stat = route_statistic.reliability_global_stats(df, rel_col=rel_col)
    df["total_hops"] = df["intra_hops"] + df["inter_hops"]

    rel_stat = route_statistic.reliability_global_stats(df, rel_col=rel_col)
    hop_stat = pd.concat(
        [
            _numeric_stats_with_prefix(df["total_hops"], "hop"),
            _numeric_stats_with_prefix(df["intra_hops"], "intra_hop"),
            _numeric_stats_with_prefix(df["inter_hops"], "inter_hop"),
        ],
        axis=0,
    )


    policy_stat = {
        "policy_name": policy["policy_name"],
        "p_intra_used": p_intra_used,
        "default_p_inter_used": float(default_p_inter_used),
        "unknown_option_edge_total": int(df["unknown_option_edges"].sum()),
    }
    for op in sorted(option_p_inter.keys()):
        policy_stat[f"option{op}_p_inter"] = float(option_p_inter[op])
        policy_stat[f"option{op}_edge_count"] = int(option_counter_total.get(op, 0))
    policy_stat = pd.Series(policy_stat)

    global_stat = pd.concat([rel_stat, hop_stat, policy_stat], axis=0)
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
