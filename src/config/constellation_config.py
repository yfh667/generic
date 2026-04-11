from dataclasses import dataclass
from typing import Dict, List

@dataclass(frozen=True)
class ConstellationConfig:
    name: str
    N: int
    P: int


    @property
    def total_sats(self) -> int:
        return self.N * self.P