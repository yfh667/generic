import pandas as pd
import numpy as np
from datetime import datetime

def estimate_orbital_parameters(times, positions):
    """
    从轨道位置拟合出：
    - 轨道倾角 inclination（deg）
    - 平均轨道高度 altitude（km）
    - 平均运动 rev/day
    """
    center = np.mean(positions, axis=0)
    positions_centered = positions - center

    # SVD 求主平面法向量
    _, _, vh = np.linalg.svd(positions_centered)
    normal_vector = vh[-1]
    inclination_rad = np.arccos(abs(normal_vector[2]))  # 与 Z 轴夹角
    inclination_deg = np.degrees(inclination_rad)

    # 平均轨道半径和高度
    earth_radius = 6371e3  # m
    radius = np.linalg.norm(positions, axis=1)
    altitudes = radius - earth_radius
    mean_altitude_km = np.mean(altitudes) / 1000.0

    # 平均运动：Kepler 3rd Law
    semi_major_axis = earth_radius + mean_altitude_km * 1e3
    mu = 3.986004418e14
    T = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu)  # orbital period in sec
    rev_per_day = 86400 / T

    return inclination_deg, mean_altitude_km, rev_per_day

def generate_tle(inclination, rev_per_day, epoch=None):
    """
    构建 TLE 字符串（简化模型，其他元素为默认值）
    """
    if epoch is None:
        epoch = datetime.utcnow()
    day_of_year = epoch.timetuple().tm_yday
    frac_of_day = (epoch.hour * 3600 + epoch.minute * 60 + epoch.second) / 86400
    epoch_str = f"{epoch.year % 100:02d}{day_of_year:03d}.{frac_of_day:.8f}".replace("0.", "")

    tle_line1 = f"1 99999U 25001A   {epoch_str}  .00000000  00000-0  00000-0 0 00001"
    tle_line2 = f"2 99999 {inclination:8.4f}  0.0000 0001000  0.0000  0.0000 {rev_per_day:11.8f}    01"
    return tle_line1, tle_line2

def fit_tle_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    positions = df[["x", "y", "z"]].values
    times = df["timestamp"]

    inclination, mean_alt_km, rev_per_day = estimate_orbital_parameters(times, positions)
    tle1, tle2 = generate_tle(inclination, rev_per_day, epoch=times.iloc[0])
    return tle1, tle2

if __name__ == "__main__":
    # 替换为你的 CSV 路径
    tle1, tle2 = fit_tle_from_csv("E:\\研究生\\研究进展\\待整理\\千帆数据\\1.csv")
    print(tle1)
    print(tle2)

    # 可选写入文件
    with open("fitted_output.tle", "w") as f:
        f.write(tle1 + "\n")
        f.write(tle2 + "\n")
