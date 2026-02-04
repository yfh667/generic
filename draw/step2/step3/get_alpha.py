import numpy as np


def calculate_effective_earth_central_angle(gamma_deg, beta_deg, h_km, R_E_km=6371):
    """
    计算受限于地面站仰角和卫星半锥角的实际地心角 alpha。

    参数:
    gamma_deg : float - 地面站最小仰角 (degrees)
    beta_deg  : float - 卫星半锥角/视场半角 (degrees)
    h_km      : float - 卫星轨道高度 (km)
    R_E_km    : float - 地球半径 (km, 默认 6371)

    返回:
    alpha_rad : float - 实际地心角 (radians)
    limit_type: str   - 限制因素 ('Elevation' or 'Cone')
    """

    # 1. 角度转弧度
    gamma = np.deg2rad(gamma_deg)
    beta = np.deg2rad(beta_deg)
    R_sat = R_E_km + h_km

    # --- 计算情形 1: 受仰角 gamma 限制的地心角 ---
    # 公式: pi/2 - gamma - arcsin( (R_E * cos(gamma)) / R_sat )
    # 这一项对应的是纳底角 eta
    sin_eta_ele = (R_E_km * np.cos(gamma)) / R_sat
    # 数值保护: 防止浮点误差导致 > 1
    sin_eta_ele = np.clip(sin_eta_ele, -1.0, 1.0)

    # 计算对应地心角 alpha1
    alpha_elevation_limit = (np.pi / 2) - gamma - np.arcsin(sin_eta_ele)

    # --- 计算情形 2: 受卫星半锥角 beta 限制的地心角 ---
    # 检查: 如果 beta 太大，超过了地球切线角，那么视场就不受 beta 限制，而是受地球遮挡限制
    # 地球可视最大半锥角 sin(beta_max) = R_E / R_sat
    sin_beta_max = R_E_km / R_sat

    if np.sin(beta) > sin_beta_max:
        # 如果半锥角比地球轮廓还大，那么限制实际上就是地平线（仰角=0的情况）
        # 但通常函数应按公式计算，这里做个标注或处理
        alpha_cone_limit = np.arccos(R_E_km / R_sat)  # 也就是几何视界
    else:
        # 使用你提供的公式: pi/2 - beta - arccos( (R_sat/R_E) * sin(beta) )
        # 注意：这里各项是导致地面上的夹角。
        # 让我们复用你的公式逻辑:
        term_cone = (R_sat / R_E_km) * np.sin(beta)
        term_cone = np.clip(term_cone, -1.0, 1.0)  # 数值保护

        alpha_cone_limit = (np.pi / 2) - beta - np.arccos(term_cone)

        # 注：若计算结果为负，说明几何构型不成立（卫星太低或角度太大无法形成闭合三角）
        if alpha_cone_limit < 0:
            alpha_cone_limit = 0

    # --- 取交集 (最小值) ---
    if alpha_elevation_limit < alpha_cone_limit:
        return alpha_elevation_limit, "Elevation_Limited"
    else:
        return alpha_cone_limit, "Cone_Limited"


# ================= 示例测试 =================
h = 1066  # Starlink 高度
Re = 6371
gamma = 20  # 地面站仰角 25度
beta = 45  # 卫星半锥角 40度 (假设值)

alpha, reason = calculate_effective_earth_central_angle(gamma, beta, h, Re)

print(f"限制因素: {reason}")
print(f"地心角 (rad): {alpha:.4f}")
print(f"地心角 (deg): {np.rad2deg(alpha):.2f}°")
print(f"覆盖半径 (km): {alpha * Re:.2f} km")  # 地面覆盖半径