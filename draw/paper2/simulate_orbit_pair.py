
# -*- coding: utf-8 -*-
# """
# 局部轨道间通信仿真 - 修正版
# 修复了跨 0 度边界时的区间判定逻辑错误
# """
from __future__ import annotations

from pathlib import Path


import numpy as np


def calculate_dy_torus(j1, j2, N):
    """
    计算环向轨道上的最短距离
    由于每个轨道是环向的，需要考虑两个方向
    """
    dy_direct = abs(j2 - j1)
    dy_wrap = abs(j2 - j1 + N) if j2 < j1 else abs(j2 - j1 - N)
    return min(dy_direct, dy_wrap)
def simulate_orbit_pair(alpha_1_deg, alpha_2_deg, Delta_x=10, theta_diff_deg=10,
                        N=36, f=4.5, T_orbit_s=6360, output_file=None):
    """
    仿真单对轨道间的平均跳数随时间变化

    参数:
        alpha_1_deg: EC1 的覆盖地心角（度）
        alpha_2_deg: EC2 的覆盖地心角（度）
        Delta_x: 轨道编号差
        theta_diff_deg: 纬度差（度，theta2 - theta1）
        N: 每轨道卫星数
        f: 相位因子 (度)
        T_orbit_s: 轨道周期（秒），默认 106 分钟
        output_file: 输出文件路径（可选）
    """

    # --- 1. 参数初始化 ---
    alpha_1 = float(alpha_1_deg)
    alpha_2 = float(alpha_2_deg)

    # 时间设置
    T_topo = T_orbit_s / N
    dt = 1.0
    time_steps = np.arange(0, T_topo, dt)

    # 卫星分布参数
    sat_spacing = 360.0 / N

    # 打印配置信息
    print("=" * 70)
    print(f"仿真配置:")
    print(f"  α₁ = {alpha_1:.2f}°, α₂ = {alpha_2:.2f}°")
    print(f"  Δx = {Delta_x}, Δθ = {theta_diff_deg:.2f}°")
    print(f"  拓扑周期 = {T_topo:.2f} s, 步数 = {len(time_steps)}")
    print("=" * 70)

    # --- 2. 关键修复：计算 EC2 的角度窗口 ---
    # 根据坐标系定义：
    # EC1 范围: [0, 2*alpha1] (假设已归一化到起始点)
    # EC2 中心相对于 EC1 中心的偏移量为: theta_diff_deg
    # EC2 的绝对范围: [alpha1 + theta_diff - alpha2, alpha1 + theta_diff + alpha2]

    # 注意：这里沿用你代码中的逻辑公式
    l_raw = theta_diff_deg - (alpha_2 - alpha_1)
    r_raw = theta_diff_deg + (alpha_2 + alpha_1)

    # 归一化到 [0, 360)
    l_norm = np.mod(l_raw, 360.0)
    r_norm = np.mod(r_raw, 360.0)

    print(f"调试信息 - EC2 窗口: Raw[{l_raw:.2f}, {r_raw:.2f}] -> Norm[{l_norm:.2f}, {r_norm:.2f}]")

    # 确定是否跨零 (Wrap-around)
    # 如果 l <= r，说明区间连续 (如 10 到 50) -> 使用 AND
    # 如果 l > r，说明区间跨越 0 度 (如 350 到 10) -> 使用 OR
    is_wrap_around = l_norm > r_norm

    # --- 3. 开始时间循环 ---
    hop_counts = []
    sat_counts_1 = []
    sat_counts_2 = []

    # 轨道编号初始化
    i_1 = 0

    # 预先生成卫星索引 [0, 1, ..., N-1]
    j_indices = np.arange(N)

    for t in time_steps:
        # A. 计算首星相位
        # 轨道 1 首星
        theta_prime_i1_0 = -360.0 * (t / T_orbit_s) - f * i_1
        # 轨道 2 首星 (仅考虑轨道相位差，窗口偏移已在 l_norm/r_norm 体现)
        theta_prime_i2_0 = theta_prime_i1_0 - f * Delta_x

        # B. 生成所有卫星位置
        # pos = theta_0 - j * spacing
        positions_i1 = theta_prime_i1_0 - j_indices * sat_spacing
        positions_i2 = theta_prime_i2_0 - j_indices * sat_spacing

        # C. 归一化到 [0, 360)
        pos_i1_norm = np.mod(positions_i1, 360.0)
        pos_i2_norm = np.mod(positions_i2, 360.0)

        # D. 筛选 EC 内卫星

        # EC1 筛选 (标准区间 [0, 2*alpha1])
        # 注意：如果 2*alpha1 > 360 (虽然不可能)，也需要逻辑判断，但这里通常 alpha 很小
        mask_i1 = (pos_i1_norm >= 0) & (pos_i1_norm <= 2 * alpha_1)
        L_i1 = j_indices[mask_i1]

        # EC2 筛选 (关键修复点!!!)
        if not is_wrap_around:
            # 正常区间：大于左边界 且 小于右边界
            mask_i2 = (pos_i2_norm >= l_norm) & (pos_i2_norm <= r_norm)
        else:
            # 跨零区间：大于左边界(靠近360) 或 小于右边界(靠近0)
            mask_i2 = (pos_i2_norm >= l_norm) | (pos_i2_norm <= r_norm)

        L_i2 = j_indices[mask_i2]

        # 记录卫星数量
        sat_counts_1.append(len(L_i1))
        sat_counts_2.append(len(L_i2))

        # E. 计算跳数
        if len(L_i1) == 0 or len(L_i2) == 0:
            hop_counts.append(np.nan)
        else:
            total_hops = 0
            count = 0
            # 这里的双重循环对于 N=36 来说非常快，无需向量化
            for j1 in L_i1:
                for j2 in L_i2:
                    dy = calculate_dy_torus(j1, j2, N)
                    hop = abs(Delta_x) + dy
                    total_hops += hop
                    count += 1

            avg_hop = total_hops / count if count > 0 else np.nan
            hop_counts.append(avg_hop)

    # --- 4. 结果整理 ---
    hop_counts = np.array(hop_counts)
    sat_counts_1 = np.array(sat_counts_1)
    sat_counts_2 = np.array(sat_counts_2)

    # 统计输出
    valid_hops = hop_counts[~np.isnan(hop_counts)]
    if len(valid_hops) > 0:
        print(f"统计结果: 均值={np.mean(valid_hops):.4f}, 范围=[{np.min(valid_hops):.4f}, {np.max(valid_hops):.4f}]")
    else:
        print("警告: 无有效连接路径")

    # --- 5. 文件保存 ---
    if output_file:
        out_path = Path(output_file)
        # 确保父目录存在
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(f"# 仿真参数: alpha1={alpha_1:.3f}, alpha2={alpha_2:.3f}, Dx={Delta_x}\n")
                f.write("Time_s,Hop_Count,EC1_Sats,EC2_Sats\n")
                for i in range(len(time_steps)):
                    h_val = f"{hop_counts[i]:.4f}" if not np.isnan(hop_counts[i]) else "nan"
                    f.write(f"{time_steps[i]:.1f},{h_val},{sat_counts_1[i]},{sat_counts_2[i]}\n")
            print(f"数据已保存至: {out_path}")
        except Exception as e:
            print(f"保存文件失败: {e}")

    return time_steps, hop_counts, sat_counts_1, sat_counts_2



# ========== 主程序示例 ==========
# if __name__ == "__main__":
#     # 示例配置
#     alpha_1 = 10.325
#     alpha_2 = 9.85
#     dx = 11
#     d_theta = 16.78
#
#     # 路径配置
#     base_dir = Path(r"C:\usrspace\mywork\data_paper2\visibile_data\test\orbit")
#     # 自动创建目录防止报错
#     base_dir.mkdir(parents=True, exist_ok=True)
#
#     output_csv = base_dir / f'orbit_pair_a{alpha_1:.0f}_a{alpha_2:.0f}.csv'
#
#     time, hops, sats1, sats2 = simulate_orbit_pair(
#         alpha_1_deg=alpha_1,
#         alpha_2_deg=alpha_2,
#         Delta_x=dx,
#         theta_diff_deg=d_theta,
#         output_file=output_csv
#     )
#
#     print(f"\n仿真完成！数据点数: {len(time)}")