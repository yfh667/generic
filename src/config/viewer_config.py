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


# G60_CONFIG = ViewerConfig(
#     name="G60",
#     N=36,
#     P=18,
#     # 这里就要根据实际进行修改
#     station_groups={
#         0: {"name": "Group 0", "stations": list(range(0, 4))},
#         1: {"name": "Group 1", "stations": list(range(4, 9))},
#         2: {"name": "Group 2", "stations": [9]},
#         3: {"name": "Group 3", "stations": [10]},
#         4: {"name": "Group 4", "stations": list(range(11, 15))},
#         5: {"name": "Group 5", "stations": list(range(15, 17))},
#         6: {"name": "Group 6", "stations": list(range(17, 20))},
#     },
#     group_colors=[
#         '#FF0000',
#         '#00FF00',
#         '#0000FF',
#         '#FFA500',
#         '#800080',
#         '#00FFFF',
#         '#FFFF00',
#     ]
#
#
#
# )

G60_CONFIG = ViewerConfig(
    name="G60",
    N=36,
    P=18,
    # 这里就要根据实际进行修改
    station_groups={
            0: {"name": "Group 0", "stations": list(range(0, 12))},
            1: {"name": "Group 1", "stations": list(range(12, 21))},


    },
    group_colors=[
        '#FF0000',
        '#00FF00',


    ]



)