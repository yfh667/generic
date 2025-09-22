# simulation_daa/core/config_runtime.py
import os, re
from pathlib import Path

# 尝试优先从包相对导入你的根 config（更稳）
try:
    from ...config import INPUT_DIR, DATA_DIR   # 如果包层级不同，调整点数
except Exception:
    from config import INPUT_DIR, DATA_DIR      # 退路

TIME_2_BUILD = int(os.getenv("TIME_2_BUILD", "30"))
VERSION = os.getenv("TOPOLOGY_VERSION", f"topology_{TIME_2_BUILD}")

P, N = 18, 36
RAW_RANGE = (0, 22006)
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,9355),
    (9355,11640),(11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]

MODIFY_DIR = Path(INPUT_DIR) / VERSION / "modify"
XML_FILE   = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"

MODIFY_DIR.mkdir(parents=True, exist_ok=True)

def discover_paths():
    """
    自动发现/兜底构建需要加载的 interplane_links_*.xml 列表（按时间排序）。
    """
    files = sorted(MODIFY_DIR.glob("interplane_links_*.xml"))
    if files:
        # 尝试解析 {start}_{end}
        def key_of(p: Path):
            m = re.search(r"(\d+)_(\d+)\.xml$", p.name)
            return (int(m.group(1)), int(m.group(2))) if m else (10**12, 10**12)
        files = sorted(files, key=key_of)
        return files

    # 兜底：用 RANGES 构造（只加存在的）
    out = []
    for s, e in RANGES:
        p = MODIFY_DIR / f"interplane_links_{s}_{e}.xml"
        if p.exists():
            out.append(p)
    return out
