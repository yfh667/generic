# topology_config_min.py
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import draw.read_snap_xml as read_snap_xml
from draw.basic_functio import motif as motif_mod

# ---- Data models: only what you want ----
@dataclass
class MotifSpec:
    p_start: int
    p_end: int
    y_start: int
    y_end: int
    option: int = 0

@dataclass
class TopologyConfig:
    P: int
    N: int
    base_groupid: int
    motifs: List[MotifSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "P": int(self.P),
            "N": int(self.N),
            "base_groupid": int(self.base_groupid),
            "motifs": [asdict(m) for m in self.motifs],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TopologyConfig":
        # minimal validation
        for k in ("P", "N", "base_groupid", "motifs"):
            if k not in d:
                raise ValueError(f"Missing required field: {k}")
        motifs = [MotifSpec(**m) for m in d["motifs"]]
        return TopologyConfig(P=int(d["P"]), N=int(d["N"]),
                              base_groupid=int(d["base_groupid"]),
                              motifs=motifs)

def save_config(cfg: TopologyConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)

def load_config(path: str | Path) -> TopologyConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        d = json.load(f)
    return TopologyConfig.from_dict(d)

# ---- Recorder: capture your manual operations (no extra fields) ----
class TopologyRecorder:
    def __init__(self, P: int, N: int):
        self.P = int(P)
        self.N = int(N)
        self.base_groupid: int | None = None
        self._motifs: List[MotifSpec] = []

    def modify_group_data(self, group_data: Dict, base_groupid: int) -> Tuple[Dict, Any]:
        self.base_groupid = int(base_groupid)
        return read_snap_xml.modify_group_data(group_data, P=self.P, N=self.N, base_groupid=base_groupid)

    def write_distinct_motif(self, p_start: int, p_end: int, y_start: int, y_end: int,
                             nodes: Dict, option: int = 0) -> None:
        self._motifs.append(MotifSpec(p_start, p_end, y_start, y_end, option))
        motif_mod.write_distinct_motif(p_start, p_end, y_start, y_end, self.P, self.N, nodes, option=option)

    def save(self, path: str | Path) -> None:
        if self.base_groupid is None:
            raise ValueError("Call recorder.modify_group_data(...) before save().")
        cfg = TopologyConfig(P=self.P, N=self.N, base_groupid=self.base_groupid, motifs=self._motifs)
        save_config(cfg, path)

# ---- Replayer: apply an existing config ----
def apply_topology_config(group_data: Dict, cfg: TopologyConfig):
    """
    Returns: rev_group_data, offset, nodes, rev_inter_edge
    """
    rev_group_data, offset = read_snap_xml.modify_group_data(
        group_data, P=cfg.P, N=cfg.N, base_groupid=cfg.base_groupid
    )
    nodes: Dict[int, set] = {}
    for m in cfg.motifs:
        motif_mod.write_distinct_motif(m.p_start, m.p_end, m.y_start, m.y_end,
                                       cfg.P, cfg.N, nodes, option=m.option)
    rev_inter_edge = motif_mod.transform_nodes_2_adjacent(nodes, cfg.P, cfg.N)
    return rev_group_data, offset, nodes, rev_inter_edge

def load_and_apply(group_data: Dict, config_path: str | Path):
    cfg = load_config(config_path)
    return apply_topology_config(group_data, cfg)
