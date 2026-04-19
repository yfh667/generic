from pathlib import Path
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import src.paper3_postprocess.export_onepair_linkmetric as export_onepair_linkmetric


TOPOLOGY_VERSION = "grid_four"
CSV_DIR = "region_pairs_0_86164"
P_INTRA = 0.999
P_INTER = 0.99

DATA_DIR = Path(r"D:\paper3")
BASEDIR = DATA_DIR / "data"
TOPOLOGY_DIR = "topology_design"

PAIR_CSV_DIR = BASEDIR / TOPOLOGY_DIR / TOPOLOGY_VERSION / "path" / CSV_DIR
OUT_CSV_PATH = BASEDIR / TOPOLOGY_DIR / TOPOLOGY_VERSION / "analysis_link" / "all_pair_global_stat_parallel.csv"
OUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

# 并行进程数，别开太满，通常 4~8 就够
MAX_WORKERS = min(8, os.cpu_count() or 1)


def _worker(pair_csv_name: str):
    global_stat = export_onepair_linkmetric.compute_pair_global_stat(
        TOPOLOGY_VERSION,
        CSV_DIR,
        pair_csv_name,
        P_INTRA,
        P_INTER,
    )
    return {
        "PAIR_CSV_NAME": pair_csv_name,
        **global_stat.to_dict(),
    }


if __name__ == "__main__":
    pair_csv_name_list = sorted([p.name for p in PAIR_CSV_DIR.glob("*.csv")])

    rows = []
    total = len(pair_csv_name_list)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_worker, pair_csv_name): pair_csv_name
            for pair_csv_name in pair_csv_name_list
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            pair_csv_name = future_map[future]
            try:
                row = future.result()
                rows.append(row)
                print(f"[{idx}/{total}] done: {pair_csv_name}")
            except Exception as e:
                print(f"[{idx}/{total}] failed: {pair_csv_name} -> {e}")

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values("PAIR_CSV_NAME").reset_index(drop=True)
    df_out.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print(df_out.head())
    print(f"已导出: {OUT_CSV_PATH}")
