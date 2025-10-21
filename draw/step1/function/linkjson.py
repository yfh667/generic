# --- 常量（WGS-84） ---
import json, math
from pathlib import Path

_WGS84_A = 6378137.0           # 半长轴 a (m)
_WGS84_E2 = 6.69437999014e-3   # 第一偏心率平方 e^2
_WGS84_B = _WGS84_A * math.sqrt(1 - _WGS84_E2)

def _ecef_to_lla(x, y, z):
    """
    将 ECEF (m) 转为 (lon, lat, alt) = (经度°, 纬度°, 海拔m)，WGS-84。
    采用常用的 Bowring 近似，精度满足可视化需求。
    """
    a = _WGS84_A
    e2 = _WGS84_E2
    b = _WGS84_B

    # 经度
    lon = math.degrees(math.atan2(y, x))

    # 中间量
    p = math.hypot(x, y)
    if p < 1e-9:  # 极点附近的稳健性
        lat = 90.0 if z >= 0 else -90.0
        N = a / math.sqrt(1 - e2 * math.sin(math.radians(lat))**2)
        alt = abs(z) - (a*(1 - e2))  # 近似
        return lon, lat, alt

    # Bowring 公式
    theta = math.atan2(z * a, p * b)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    lat = math.atan2(z + (e2 * b) * (sin_t**3),
                     p - (e2 * a) * (cos_t**3))
    sin_lat = math.sin(lat)
    N = a / math.sqrt(1 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - N

    lat = math.degrees(lat)
    return lon, lat, alt

def export_satellites_snapshot_to_json(SatelliteManager, time_idx, out_path, *,
                                       ecef_in_km=False, alt_unit='km', links=None):
    """
    从 SatelliteManager 在指定时间步导出所有卫星的 (lng, lat, alt) 到 JSON。
    新增:
      - links: list[{"source": id, "target": id}]，不传则默认为 []
    """
    sats = []
    scale = 1000.0 if ecef_in_km else 1.0

    for sid, sat in SatelliteManager.satellites.items():
        if time_idx >= len(sat.trajectory):
            raise IndexError(f"satellite {sid} has trajectory length {len(sat.trajectory)}, "
                             f"but time_idx={time_idx}")
        p = sat.trajectory[time_idx]
        x, y, z = p.x * scale, p.y * scale, p.z * scale
        lon, lat, alt_m = _ecef_to_lla(x, y, z)
        alt_out = alt_m / 1000.0 if alt_unit == 'km' else alt_m
        sats.append({
            "id": str(sid),
            "lng": float(lon),
            "lat": float(lat),
            "alt": float(alt_out)
        })

    payload = { "satellites": sats, "links": (links or []) }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(out_path)