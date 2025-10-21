
from datetime import timedelta
import basicSa.satellite_stk_alpha.config.settings as settings
import basicSa.calculate_angular.gettheta as gettheta
import numpy as np


def get_vectors(  RAAN, inclination,vectorbase, delta_current):
    cosR = np.cos(RAAN)  # RAAN余弦
    sinR = np.sin(RAAN)  # RAAN正弦
    cosB = np.cos(inclination)  # 轨道倾角余弦
    sinB = np.sin(vectorbase)  # 轨道倾角正弦

    # 基准向量模长
    base_norm = np.linalg.norm(vectorbase, axis=1, keepdims=True)

    # =============================
    # 2. Φ角相关参数计算
    # =============================
    # 防止除零错误（当sinB接近0时自动屏蔽后续无效计算）
    safe_sinB = np.where(sinB < 1e-6, np.nan, sinB)

    cos_phi = vectorbase[:, 0] / (base_norm[:, 0] + 1e-8)  # x分量占比
    sin_phi = vectorbase[:, 2] / (base_norm[:, 0] * safe_sinB + 1e-8)  # 经倾角校正后的z分量占比

    def apply_rotation(dx, dy, dz):
        """应用复合旋转的向量化计算"""
        # X轴旋转分量
        x_rot = (-cosR * sin_phi + sinR * cos_phi * cosB) * dx \
                + (sinR * sin_phi + cosB * cos_phi * cosR) * dy \
                + sinB * cos_phi * dz

        # Y轴旋转分量
        y_rot = (sinR * sinB) * dx \
                + (cosR * sinB) * dy \
                - cosB * dz

        # Z轴旋转分量
        z_rot = (-cosR * cos_phi - sinR * sin_phi * cosB) * dx \
                + (sinR * cos_phi - cosR * sin_phi * cosB) * dy \
                - sin_phi * sinB * dz

        return np.column_stack((x_rot, y_rot, z_rot))

    rotated = apply_rotation(*delta_current.T)

    return rotated


def calculate_angular_velocity( RAAN,inclination, time,vecs_base,vecs_next,vecs_deleta_current,vecs_deleta_next):
    """计算卫星间相对角速度（严格三维公式实现）

    返回值：
        np.ndarray: 各卫星的角速度模值（度/秒）
    """
   # vecs = self.get_calculation_vectors()
    # the base  delta_current  delta_next all is the ecef  ,we need trans the ecef into eci

    time2 = settings.SatelliteConfig.basetime + timedelta(seconds=time)
    theta = gettheta.gettheta(time2)

    # 构造旋转矩阵（Z 轴逆时针旋转 theta）
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])  # shape (3, 3)

    ecef_coords = vecs_base  # shape (N, 3)
    # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
    vecs_base_eci = ecef_coords @ R.T  # 注意：R.T 是转置！

    # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
   # vecs['base_eci'] = eci_coords

    ecef_coords =vecs_next # shape (N, 3)
    # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
    vecs_next_eci = ecef_coords @ R.T  # 注意：R.T 是转置！

    # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
    #vecs['next_eci'] = eci_coords

    ecef_coords = vecs_deleta_current  # shape (N, 3)
    # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
    vecs_deleta_current_eci = ecef_coords @ R.T  # 注意：R.T 是转置！

    # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
  #  vecs['delta_current_eci'] = eci_coords

    ecef_coords = vecs_deleta_next # shape (N, 3)
    # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
    vecs_deleta_next_eci = ecef_coords @ R.T  # 注意：R.T 是转置！

    # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
   # vecs['delta_next_eci'] = eci_coords
    #

    current_rotated =  get_vectors(RAAN, inclination, vecs_base_eci, vecs_deleta_current_eci)
    # 下一时刻旋转

    next_rotated =  get_vectors(RAAN, inclination, vecs_next_eci, vecs_deleta_next_eci)

    # =============================
    # 4. 角速度计算
    # =============================
    # 速度向量计算（有限差分）
    velocity = next_rotated - current_rotated

    # 三维叉乘计算
    cross_product = np.cross(current_rotated, velocity)

    # 模长平方安全计算
    norm_sq = np.sum(current_rotated ** 2, axis=1, keepdims=True) + 1e-8

    # 角速度向量（弧度/秒）
    angular_velocity_rad = cross_product / norm_sq

    # az = angular_velocity_rad[2]
    angular_velocity_z = np.abs(angular_velocity_rad[:, 2])  # 取绝对值表示大小
    # 转换为角度制模长
    return angular_velocity_z * 180 / np.pi

