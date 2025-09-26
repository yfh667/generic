# xml2origin_tables.py
from pathlib import Path
import re, xml.etree.ElementTree as ET
import pandas as pd
import os
# --------- 你可以改这里 ----------
# XML 文件列表
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,11640),
     (11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]

from config import DATA_DIR,INPUT_DIR
time_2_build = 30
TIME_2_BUILD = time_2_build
from pathlib import Path
# 默认用 "topology_{TIME_2_BUILD}"，也允许用环境变量 TOPOLOGY_VERSION 覆盖
VERSION = os.getenv("TOPOLOGY_VERSION", f"topology_{TIME_2_BUILD}")

RAW_DIR    = Path(INPUT_DIR) / VERSION / "raw"
CONFIG_DIR = Path(INPUT_DIR) /"config"


MODIFY_DIR = Path(INPUT_DIR) / VERSION / "modify"

xml_paths = [MODIFY_DIR / f"interplane_links_{s}_{e}.xml" for s, e in RANGES]

OUT_DIR = MODIFY_DIR / "origin_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUT_DIR / "nodes_all.xlsx"      # 也可写 CSV/Parquet，见文末
# -----------------------------------

def parse_triplet(s, paren=False):
    """'i,j,step' 或 '(i, j, step)' / 'None' -> (i,j,step) 或 (None,None,None)"""
    if not s or s == "None":
        return (None, None, None)
    s2 = s.strip()
    if paren and s2.startswith("(") and s2.endswith(")"):
        s2 = s2[1:-1]
    parts = [p.strip() for p in s2.split(",")]
    if len(parts) != 3:
        return (None, None, None)
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (None, None, None)

def scenario_from_name(p: Path):
    """
    interplane_links_1204_3669.xml -> ("1204_3669", 1204, 3669)
    """
    m = re.search(r"interplane_links_(\d+)_(\d+)\.xml$", p.name)
    if not m:
        return (p.stem, None, None)
    s, e = int(m.group(1)), int(m.group(2))
    return (f"{s}_{e}", s, e)

def xml_to_rows(xml_path: Path):
    sid, s0, e0 = scenario_from_name(xml_path)
    # 用 iterparse 节省内存
    rows = []
    for event, elem in ET.iterparse(str(xml_path), events=("end",)):
        if elem.tag != "Node":
            continue
        coord = elem.attrib.get("coordination")
        rn    = elem.attrib.get("rightneighbor")
        ln    = elem.attrib.get("leftneighbor")
        ascid = elem.attrib.get("asc_nodes_region_id") or elem.attrib.get("asc_nodes_flag")  # 两种命名都兼容
        lstate = elem.attrib.get("left_state")
        rstate = elem.attrib.get("right_state")
        i,j,step             = parse_triplet(coord)
        rn_i,rn_j,rn_step    = parse_triplet(rn, paren=True)
        ln_i,ln_j,ln_step    = parse_triplet(ln, paren=True)

        rows.append({
            "scenario_id": sid, "range_start": s0, "range_end": e0,
            "step": step, "i": i, "j": j,
            "rn_i": rn_i, "rn_j": rn_j, "rn_step": rn_step,
            "ln_i": ln_i, "ln_j": ln_j, "ln_step": ln_step,
            "asc_nodes_region_id": None if ascid in (None, "None") else int(ascid),
            "left_state":  None if lstate in (None, "None") else int(lstate),
            "right_state": None if rstate in (None, "None") else int(rstate),
        })
        elem.clear()
    return rows

def main():
    all_rows = []
    for p in xml_paths:
        print("Parsing:", p)
        all_rows.extend(xml_to_rows(p))

    df = pd.DataFrame(all_rows)
    # 排序更好用
    df.sort_values(["scenario_id", "step", "i", "j"], inplace=True)
    # 输出 Excel（Origin 直接打开）
    with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as xw:
        df.to_excel(xw, index=False, sheet_name="Nodes")

    # 也可同时输出 CSV/Parquet（Origin 也能导）
    df.to_csv(OUT_DIR / "nodes_all.csv", index=False)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, OUT_DIR / "nodes_all.parquet", compression="zstd")
    except Exception:
        pass

    print("Done ->", OUT_XLSX)

if __name__ == "__main__":
    main()
