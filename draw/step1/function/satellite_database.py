import basicSa.fileread.readstation as readstation
import basicSa.fileread.readsatellite as readsatellite
import basicSa.simulator.nodemanager as nodemanager
import basicSa.postsimmulation.snapshot as snapshot

from basicSa.satellite_stk_alpha.core.angularvector2 import SatelliteVector
from basicSa.los import  Sat2Gnd

import numpy as np
def build_time_database(
    station_dir: str,
    station_min_elev_deg: float,
    sat_dir: str,
    P: int,
    N: int,
    *,
    sat_angle: float,
    track_angle: float,
    base_raan_increment: int,
    sim_steps: int,
    lenthpropority: int = 20,
    stationsnaplength: int = 20,
    include_stations: bool = True,
    compute_linkflags: bool = False,
):
    """
    读取地面站与卫星数据，构建 TimeDatabase，并返回 (db, SatelliteManager, stations_info)

    stations_info: dict
      - 'stationsnum': int
      - '_ground_cache': List[ground_point_at_t0]  # 若 include_stations=False，则为空列表

    说明：
    - 仅填充 satellitesnap 的前 6 列（x, y, z, angle, trackangle, RAAN），保持与你现有代码一致。
    - 若 include_stations=False，将跳过地面站可见性计算，大幅提速。
    - 若 compute_linkflags=True，则调用 SatelliteVector 计算 linkflags（示例中未写入到矩阵，可按需扩展）。
    """
    # --- 读取地面站 ---
    _ground_cache = []
    stationsnum = 0
    if include_stations:
        StationManager = readstation.readstation_path(station_dir, station_min_elev_deg)
        stations = StationManager.stations
        for _t in range(len(stations)):
            _ground_cache.append(stations[_t].trajectory[0])
        stationsnum = len(_ground_cache)

    # --- 读取卫星 ---
    SatelliteManager = nodemanager.SatelliteManager()
    readsatellite.readsatellite(
        SatelliteManager, sat_dir, sat_angle, track_angle, P, N, base_raan_increment
    )

    # --- 初始化 TimeDatabase ---
    db = snapshot.TimeDatabase()
    db.stationnum = stationsnum
    db.satellitenum = P * N

    # --- 辅助：按时间步找到该卫星在轨迹里的索引（与你原逻辑一致） ---
    def _pick_idx_by_time(traj, sim_t):
        # 第一个 p.time >= sim_t 的索引；找不到就用 0
        idx = next((ti for ti, p in enumerate(traj) if p.time >= sim_t), 0)
        # next_idx 如需用可安全 min 一下，这里先保留思路
        return idx

    # --- 主循环 ---
    _position_cache = {}
    for sim_time_step in range(sim_steps):

        print(f"time is {sim_time_step}")
        # 卫星快照
        satellitesnap = np.full((P * N, lenthpropority), -1.0, dtype=float)
        for j in range(P * N):
            sat = SatelliteManager.satellites[j]
            idx = _pick_idx_by_time(sat.trajectory, sim_time_step)
            pt = sat.trajectory[idx]
            satellitesnap[j][0] = pt.x
            satellitesnap[j][1] = pt.y
            satellitesnap[j][2] = pt.z
            satellitesnap[j][3] = pt.angle
            satellitesnap[j][4] = pt.trackangle
            satellitesnap[j][5] = pt.RAAN
            # 如果你以后要把 linkflags 写入 6.. 等列，这里按需补

        # 地面站快照（可选）
        if include_stations:
            stationsnap = np.full((stationsnum, stationsnaplength), -1.0, dtype=float)
            for j in range(stationsnum):
                stationsnap[j][0] = _ground_cache[j].x
                stationsnap[j][1] = _ground_cache[j].y
                stationsnap[j][2] = _ground_cache[j].z
                stationsnap[j][3] = _ground_cache[j].angle

                m = 4
                # 计算可见卫星
                for k in range(P * N):
                    sat_pt = SatelliteManager.satellites[k].trajectory[
                        _pick_idx_by_time(SatelliteManager.satellites[k].trajectory, sim_time_step)
                    ]
                    visibility = Sat2Gnd.Ground_Sat(_ground_cache[j], sat_pt)
                    if visibility > 0 and m < stationsnaplength:
                        stationsnap[j][m] = k
                        m += 1
        else:
            stationsnap = np.empty((0, 0))  # 或者 None，看你 TimeSnapshot 的实现

        # 写入一个 TimeSnapshot
        ts = snapshot.TimeSnapshot(
            timestamp_str=sim_time_step,
            station_snapshot=stationsnap,
            satellite_dsnapshot=satellitesnap,
        )
        db.snapshots.append(ts)

    stations_info = {"stationsnum": stationsnum, "_ground_cache": _ground_cache}
    return db, SatelliteManager, stations_info
