from pathlib import Path
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

import src.paper3_postprocess.export_onepair_linkmetric as export_onepair_linkmetric


# ============================================================
# 0) 公共配置
# 这些参数本轮批量跑是统一的
# ============================================================
DATA_DIR = Path(r"D:\paper3")
BASEDIR = DATA_DIR / "data"
TOPOLOGY_DIR = "topology_design"

CSV_DIR = "region_pairs_0_86164"
P_INTRA = 0.999
P_INTER = 0.99

TOPOLOGY_VERSIONS = [
    "grid_four",
    # "gridx",
    # "grid_plane_alternating",
    # "grid_x_sparse",
    # "grid+",
]

MAX_WORKERS = min(8, os.cpu_count() or 1)


def build_stat_filename():
    return f"pairs_stat_intra_{P_INTRA}_inter_{P_INTER}.csv"


def build_failed_filename():
    return f"pairs_stat_intra_{P_INTRA}_inter_{P_INTER}_failed.csv"


EXPORT_COMBINED_CSV = True
COMBINED_OUT_CSV = BASEDIR / TOPOLOGY_DIR / build_stat_filename()



COMBINED_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1) 路径工具
# ============================================================
def get_pair_csv_dir(topology_version: str) -> Path:
    return BASEDIR / TOPOLOGY_DIR / topology_version / "path" / CSV_DIR


def get_out_csv_path(topology_version: str) -> Path:
    # out_csv = (
    #     BASEDIR
    #     / TOPOLOGY_DIR
    #     / topology_version
    #     / "analysis_link"
    #     / f"all_pair_global_stat_parallel__{CSV_DIR}__pi{P_INTRA}_pe{P_INTER}.csv"
    # )
    out_csv = (
            BASEDIR
            / TOPOLOGY_DIR
            / topology_version
            / "analysis_link"
            / build_stat_filename()
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_csv


# ============================================================
# 2) 构造任务
# 每个 task 只带 topology_version + pair_csv_name
# 其他参数全走公共配置
# ============================================================
def build_tasks_for_topology(topology_version: str):
    pair_csv_dir = get_pair_csv_dir(topology_version)
    pair_csv_name_list = sorted([p.name for p in pair_csv_dir.glob("*.csv")])

    tasks = []
    for pair_csv_name in pair_csv_name_list:
        tasks.append(
            {
                "topology_version": topology_version,
                "pair_csv_name": pair_csv_name,
            }
        )

    return tasks


# ============================================================
# 3) worker
# ============================================================
def _worker(task: dict):
    topology_version = task["topology_version"]
    pair_csv_name = task["pair_csv_name"]

    global_stat = export_onepair_linkmetric.compute_pair_global_stat(
        topology_version,
        CSV_DIR,
        pair_csv_name,
        P_INTRA,
        P_INTER,
    )

    return {
        "TOPOLOGY_VERSION": topology_version,
        "PAIR_CSV_NAME": pair_csv_name,
        **global_stat.to_dict(),
    }

# ============================================================
# 4) 单拓扑导出
# ============================================================
def export_per_topology_csv(df_all: pd.DataFrame):
    if df_all.empty:
        return

    for topology_version, sub in df_all.groupby("TOPOLOGY_VERSION"):
        out_csv_path = get_out_csv_path(topology_version)
        sub = sub.sort_values("PAIR_CSV_NAME").reset_index(drop=True)
        sub.to_csv(out_csv_path, index=False, encoding="utf-8-sig")
        debug_cols = [
            c for c in sub.columns
            if c in {"policy_name", "p_intra_used", "default_p_inter_used", "unknown_option_edge_total"}
            or c.startswith("option")
        ]
        if debug_cols:
            debug_csv = out_csv_path.with_name(out_csv_path.stem + "_option_debug.csv")
            sub[["PAIR_CSV_NAME"] + debug_cols].to_csv(debug_csv, index=False, encoding="utf-8-sig")
            print(f"[export] {topology_version} option-debug -> {debug_csv}")

        print(f"[export] {topology_version} -> {out_csv_path}")

def run_one_topology(topology_version: str):
    tasks = build_tasks_for_topology(topology_version)
    total = len(tasks)

    if total == 0:
        print(f"[skip] {topology_version}: 没有找到任何 pair csv")
        return pd.DataFrame(), pd.DataFrame()

    print("=" * 80)
    print(f"[topology] {topology_version}")
    print(f"[info] csv_dir = {CSV_DIR}")
    print(f"[info] p_intra = {P_INTRA}")
    print(f"[info] p_inter = {P_INTER}")
    print(f"[info] total tasks = {total}")
    print(f"[info] max_workers = {MAX_WORKERS}")

    rows = []
    failed_rows = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(_worker, task): task
            for task in tasks
        }

        for idx, future in enumerate(as_completed(future_map), start=1):
            task = future_map[future]
            pair_csv_name = task["pair_csv_name"]

            try:
                row = future.result()
                rows.append(row)
                print(f"[{topology_version}] [{idx}/{total}] done: {pair_csv_name}")
            except Exception as e:
                failed_rows.append(
                    {
                        "TOPOLOGY_VERSION": topology_version,
                        "PAIR_CSV_NAME": pair_csv_name,
                        "ERROR": str(e),
                    }
                )
                print(f"[{topology_version}] [{idx}/{total}] failed: {pair_csv_name} -> {e}")

    df_ok = pd.DataFrame(rows)
    df_failed = pd.DataFrame(failed_rows)

    if not df_ok.empty:
        df_ok = df_ok.sort_values("PAIR_CSV_NAME").reset_index(drop=True)
        export_per_topology_csv(df_ok)

    if not df_failed.empty:
        failed_csv_path = get_out_csv_path(topology_version).with_name(build_failed_filename())
        df_failed.to_csv(failed_csv_path, index=False, encoding="utf-8-sig")
        print(f"[export] {topology_version} failed -> {failed_csv_path}")

    return df_ok, df_failed

# ============================================================
# 5) 主程序
# ============================================================
def main():
    print(f"[info] topology_versions = {TOPOLOGY_VERSIONS}")
    print(f"[info] csv_dir = {CSV_DIR}")
    print(f"[info] p_intra = {P_INTRA}")
    print(f"[info] p_inter = {P_INTER}")
    print(f"[info] max_workers = {MAX_WORKERS}")

    all_ok = []
    all_failed = []

    for topology_version in TOPOLOGY_VERSIONS:
        df_ok, df_failed = run_one_topology(topology_version)

        if not df_ok.empty:
            all_ok.append(df_ok)
        if not df_failed.empty:
            all_failed.append(df_failed)

    if not all_ok:
        raise RuntimeError("所有 topology 都失败了，没有生成任何统计结果。")

    df_all = pd.concat(all_ok, ignore_index=True)
    df_all = df_all.sort_values(["TOPOLOGY_VERSION", "PAIR_CSV_NAME"]).reset_index(drop=True)

    if EXPORT_COMBINED_CSV:
        df_all.to_csv(COMBINED_OUT_CSV, index=False, encoding="utf-8-sig")
        print(f"[export] combined -> {COMBINED_OUT_CSV}")

    if all_failed:
        df_failed_all = pd.concat(all_failed, ignore_index=True)
        failed_csv = COMBINED_OUT_CSV.with_name(build_failed_filename())
        df_failed_all.to_csv(failed_csv, index=False, encoding="utf-8-sig")
        print(f"[export] combined failed -> {failed_csv}")

    print(df_all.head())


if __name__ == "__main__":
    main()
