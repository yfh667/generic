from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ViewerConfig:
    name: str
    N: int
    P: int
    station_groups: Dict[int, dict]
    group_colors: List[str]

    @property
    def total_sats(self) -> int:
        return self.N * self.P


GW_CONFIG = ViewerConfig(
    name="GW",
    N=48,
    P=18,
    station_groups={
        0: {"name": "Group 0", "stations": list(range(0, 4))},
        1: {"name": "Group 1", "stations": list(range(5, 10))},
        2: {"name": "Group 2", "stations": list(range(11, 22))},
        3: {"name": "Group 3", "stations": list(range(23, 30))},

    },
    group_colors=[
        '#FF0000',
        '#00FF00',
        '#0000FF',
        '#FFA500',

    ]
)


G60_CONFIG = ViewerConfig(
    name="G60",
    N=36,
    P=18,
    # 这里就要根据实际进行修改
    station_groups={
        0: {"name": "Group 0", "stations": list(range(0, 5))},
        1: {"name": "Group 1", "stations": list(range(5, 11))},
        2: {"name": "Group 2", "stations": list(range(11, 23))},
        3: {"name": "Group 3", "stations": list(range(23, 31))},

    },
    group_colors=[
        '#FF0000',
        '#00FF00',
        '#0000FF',
        '#FFA500',

    ]



)

# G60_CONFIG = ViewerConfig(
#     name="G60",
#     N=36,
#     P=18,
#     # 这里就要根据实际进行修改
#     station_groups={
#             0: {"name": "Group 0", "stations": list(range(0, 12))},
#             1: {"name": "Group 1", "stations": list(range(12, 21))},
#
#
#     },
#     group_colors=[
#         '#FF0000',
#         '#00FF00',
#
#
#     ]
#
#
#
# )