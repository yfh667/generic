# app_settings.py
from pathlib import Path
import os

from config import INPUT_DIR, DATA_DIR




N_DEFAULT = 36
P_DEFAULT = 18

time_2_build = int(os.getenv("TIME_2_BUILD", "80"))
TIME_2_BUILD = time_2_build
VERSION = os.getenv("TOPOLOGY_VERSION", f"topology_{TIME_2_BUILD}")

MODIFY_DIR = INPUT_DIR / VERSION / "modify"
MODIFY_DIR.mkdir(parents=True, exist_ok=True)

# 一次性预读可见性 XML 的范围
RAW_START, RAW_END = 0, 22006
XML_FILE = DATA_DIR / "station_visible_satellites_648_1d_real.xml"

# 启动即自动加载
AUTOLOAD_ON_START = True
PRELOAD_GROUP_DATA = True  # True=读取 [RAW_START,RAW_END)；False=只读所选步范围

RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,9355),
    (9355,11640),(11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]

def expected_modify_paths():
    return [MODIFY_DIR / f"interplane_links_{s}_{e}.xml" for (s, e) in RANGES]
