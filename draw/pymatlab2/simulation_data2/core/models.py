# core/models.py
from dataclasses import dataclass
from typing import Dict, Set, List, Optional

@dataclass
class DataBundle:
    edges_by_step: Dict[int, Dict[int, Set[int]]]
    pending_by_step: Dict[int, Dict[int, Set[int]]]
    group_data: Optional[dict]   # 可见性分组；可为 None
    step_min: int
    step_max: int
    N: int
    P: int
    time_2_build: int
    used_files: List[str]
