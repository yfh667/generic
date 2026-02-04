import numpy as np


def solve_theta_for_orbit_intersection(theta0_deg, phi0_deg, i, delta_omega_deg, alpha_deg):
    """
    求解方程: n_x*sin(theta)*cos(phi) + n_y*sin(theta)*sin(phi) + n_z*cos(theta) = cos(alpha)
    其中 phi 已知为 i * delta_omega

    参数:
    theta0_deg, phi0_deg: EC极点坐标 (度)
    i: 轨道编号 (整数)
    delta_omega_deg: 轨道间距 (度)
    alpha_deg: 覆盖半角 (度)

    返回:
    theta_solutions: 一个列表，包含所有满足条件的 theta 解 (度)。通常有0个、1个或2个解。
    """

    # 1. 角度转弧度
    deg2rad = np.pi / 180.0
    rad2deg = 180.0 / np.pi

    theta0 = theta0_deg * deg2rad
    phi0 = phi0_deg * deg2rad
    alpha = alpha_deg * deg2rad
    delta_omega = delta_omega_deg * deg2rad

    # 2. 计算 n 向量 (EC 极点方向)
    n_x = np.sin(theta0) * np.cos(phi0)
    n_y = np.sin(theta0) * np.sin(phi0)
    n_z = np.cos(theta0)

    # 3. 确定当前轨道的 phi
    phi_target = i * delta_omega

    # 4. 构造形式 A*sin(theta) + B*cos(theta) = C
    # 原式: sin(theta) * [n_x*cos(phi) + n_y*sin(phi)] + cos(theta) * [n_z] = cos(alpha)

    A = n_x * np.cos(phi_target) + n_y * np.sin(phi_target)
    B = n_z
    C = np.cos(alpha)

    # 5. 求解 A*sin(x) + B*cos(x) = C
    # 令 sqrt(A^2 + B^2) * sin(x + beta) = C
    # sin(x + beta) = C / sqrt(A^2 + B^2)

    norm = np.sqrt(A ** 2 + B ** 2)

    # 如果 C 的绝对值大于 norm，说明 |sin(x+beta)| > 1，无解
    # (这意味着该轨道完全在覆盖圆锥之外，没有交点)
    if abs(C) > norm:
        return []

        # 计算基础角度 arcsin(C / norm)
    base_angle = np.arcsin(C / norm)

    # 计算辅助角 beta (注意 atan2 的参数顺序是 y, x)
    # 我们将方程看作 sin(x)*A + cos(x)*B， 对应 R*sin(x+beta) 展开是 R(sin x cos beta + cos x sin beta)
    # 所以 R cos beta = A, R sin beta = B -> tan beta = B/A
    beta = np.arctan2(B, A)

    # 两个可能的解 (在 0 到 2pi 周期内):
    # 解1: x + beta = base_angle
    theta1 = base_angle - beta

    # 解2: x + beta = pi - base_angle
    theta2 = (np.pi - base_angle) - beta

    # 6. 归一化结果到 [0, 180] (因为 theta 通常定义为 [0, pi])
    # 注意：如果不限制在 [0, pi]，则需要根据球坐标系的定义来处理
    solutions_deg = []

    for th in [theta1, theta2]:
        # 将角度规范化到 [0, 360)
        th_deg = (th * rad2deg) % 360

        # 物理约束检查: 球坐标 theta 范围通常是 [0, 180]
        # 如果解在 (180, 360)，对应的是背面的交点，或者需要调整 phi
        # 但在这个特定几何问题中，通常只取 [0, 180]
        if 0 <= th_deg <= 180:
            solutions_deg.append(th_deg)

    return sorted(solutions_deg)


# --- 测试用例 ---
if __name__ == "__main__":
    # 示例参数
    theta0 = 65 # EC极点fan 纬度 (90-lat)
    phi0 =118 # EC极点经度
    i = 12  # 第2号轨道


    d_omega = 10.2  # 轨道间距
    alpha = 10.63  # 覆盖半径

    sols = solve_theta_for_orbit_intersection(theta0, phi0, i, d_omega, alpha)
    print(f"对于第 {i} 号轨道 (phi={i * d_omega}°):")
    if sols:
        print(f"相交点的 Theta 值: {sols}")
        print(f"即纬度可能是: {[90 - x for x in sols]}")
    else:
        print("该轨道与覆盖区不相交")