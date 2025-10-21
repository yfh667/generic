# -*- coding: utf-8 -*-
"""
Batch triple-window processing with parallelism (PyCharm script version)

For each consecutive triple (A,B,C) in RANGES:
  1) Read interplane_links_{A}.xml, {B}.xml, {C}.xml
  2) Merge -> transnodes -> get_no_conflict_link_nodes3 on [A.basicSa, C.end)
  3) Filter nodes for middle window B only
  4) Write to INPUT_DIR/modify/interplane_links_{B}.xml

Backends:
  - process (default): faster for CPU-heavy, requires __main__ guard (Windows-safe)
  - thread: alternative if you prefer threads
"""

from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import sys
import traceback

# ==== your own modules ====
import genaric2.tegnode as tegnode
import draw.basic_functio.write2xml as write2xml
import draw.basic_functio.transnodes as transnodes
import draw.basic_functio.conflict_link as conflict_link
# 可选：如果你还需要导出中段的 edges/pending_edges，再引入：
# import draw.basic_functio.inter_edge2nodes as inter_edge2nodes

# ==== config (按你的环境就地引用) ====
from config import INPUT_DIR, DATA_DIR

# 星座与建链窗口
P, N = 18, 36


# 输入大 XML（只用于你别处用的 raw_group_data；本脚本不依赖）
XML_FILE = Path(DATA_DIR) / "station_visible_satellites_648_1d_real.xml"

# 滑动窗口定义
RANGES = [
    (0,1204),(1204,3669),(3669,4094),(4094,6814),(6814,8485),(8485,9355),
    (9355,11640),(11640,13057),(13057,14065),(14065,16604),(16604,18396),
    (18396,19831),(19831,20814),(20814,22005)
]


TIME_2_BUILD = 30
VERSION = os.getenv("TOPOLOGY_VERSION", f"topology_{TIME_2_BUILD}")


INPUT_DIR_RAW    = Path(INPUT_DIR) / VERSION / "raw"
from pathlib import Path

OUT_DIR = Path(INPUT_DIR) / VERSION / "modify"
OUT_DIR.mkdir(parents=True, exist_ok=True)  # 没有就创建，已存在不报错

print(f"[OUT_DIR] {OUT_DIR}")





# 版本目录：可用环境变量覆盖





# ---------------- helpers ----------------

def _triples(ranges):
    """Yield consecutive triples (A,B,C)."""
    for i in range(len(ranges) - 2):
        yield (ranges[i], ranges[i+1], ranges[i+2])

def _load_nodes_or_empty(path: Path):
    """Robust loader: return {} on failure instead of None."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    nodes = write2xml.xml_to_nodes(path, tegnode.tegnode)
    return nodes or {}

def _process_one_triple(triple) -> str:
    """
    Process one triple (A,B,C) and write middle window B results.
    Returns output file path (str).
    """
    (start1, end1), (start2, end2), (start3, end3) = triple

    # 1) load A/B/C nodes
    fp1 = INPUT_DIR_RAW / f"interplane_links_{start1}_{end1}.xml"
    fp2 = INPUT_DIR_RAW / f"interplane_links_{start2}_{end2}.xml"
    fp3 = INPUT_DIR_RAW / f"interplane_links_{start3}_{end3}.xml"

    nodes1 = _load_nodes_or_empty(fp1)
    nodes2 = _load_nodes_or_empty(fp2)
    nodes3 = _load_nodes_or_empty(fp3)

    # merge (later overwrites earlier)
    total_nodes = {}
    if nodes1: total_nodes.update(nodes1)
    if nodes2: total_nodes.update(nodes2)
    if nodes3: total_nodes.update(nodes3)

    # 2) fill left/right neighbors
    total_complete = transnodes.transnodes(total_nodes)

    # 3) resolve conflicts on the wide window [A.basicSa, C.end)
    _, _, modified_nodes = conflict_link.get_no_conflict_link_nodes3(
        total_complete, start1, end3, TIME_2_BUILD, N, P
    )

    # 4) filter only middle window B
    #   —— 比三层 for-循环快很多
    filternodes = {k: v for k, v in modified_nodes.items() if start2 <= k[2] < end2}

    # （如需中段 edges/pending_edges，可打开下面注释）
    # edges_mid = inter_edge2nodes.trans_nodes2edges(filternodes, P, N)
    # pending_edges_mid = inter_edge2nodes.trans_nodes2_pendingedges(
    #     filternodes, start1, end3, TIME_2_BUILD, P, N
    # )

    # 5) write middle window B to OUT_DIR
    out_path = OUT_DIR / f"interplane_links_{start2}_{end2}.xml"

    # 若你已实现更快的写法 nodes_to_xml2_fast，可优先用它
    try:
        writer = getattr(write2xml, "nodes_to_xml2_fast")
        writer(filternodes, out_path)
    except Exception:
        # 回退到原版（带 pretty 的）
        write2xml.nodes_to_xml2(filternodes, out_path)

    return str(out_path)

def run_parallel(ranges, backend: str = "process", workers: int | None = None):
    """
    Run all triples in parallel and write outputs into OUT_DIR.
    backend: 'process' (default, faster in scripts) or 'thread'
    """
    triples = list(_triples(ranges))
    if workers is None:
        workers = min(8, os.cpu_count() or 4)

    Executor = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    results = []

    # 进程模式：在 Windows 上需要 __main__ 保护；PyCharm 运行本脚本时没问题
    with Executor(max_workers=workers) as ex:
        fut2trip = {ex.submit(_process_one_triple, t): t for t in triples}
        for fut in as_completed(fut2trip):
            t = fut2trip[fut]
            try:
                out_path = fut.result()
                print(f"[OK] {t[1]} -> {out_path}")
                results.append(out_path)
            except Exception as e:
                print(f"[FAIL] {t[1]}: {e}")
                traceback.print_exc()

    return results

# ---------------- main ----------------

if __name__ == "__main__":
    # Windows 安全：确保 spawn/freeze_support
    try:
        import multiprocessing as mp
        if sys.platform.startswith("win"):
            mp.freeze_support()
            mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    # backend='process' 更快；如遇到环境问题可改为 'thread'
    outs = run_parallel(RANGES, backend="process", workers=min(8, (os.cpu_count() or 4)))
    print("\nDone. Outputs:")
    for p in outs:
        print("  ", p)
