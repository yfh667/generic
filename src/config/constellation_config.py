from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class ConstellationConfig:
    name: str
    N: 36
    P: 18
    station_groups: Dict[int, dict]
    group_colors: List[str]

    @property
    def total_sats(self) -> int:
        return self.N * self.P