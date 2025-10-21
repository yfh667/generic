import math

import numpy as np


def Get_angular_velocity(params):
    # 初始化参数（所有角度输入为度数）
   #params = init_values()
    xa1, ya1, za1, xa2, ya2, za2, xb1, yb1, zb1, xb2, yb2, zb2, beta_deg, timestep, RAAN_deg = params

    # 计算两个时刻的本体坐标系矢量（传入度数）
    Vector_body = get_vector_body(xa1, ya1, za1, xb1, yb1, zb1, beta_deg, RAAN_deg)
    Vector_body2 = get_vector_body(xa2, ya2, za2, xb2, yb2, zb2, beta_deg, RAAN_deg)

    # 计算导数
    Vector_body_d = (Vector_body2 - Vector_body) / timestep

    # 计算角速度
    w = calculate_angular_velocity(Vector_body, Vector_body_d)
    w = w*180/math.pi
    # 输出结果
    np.set_printoptions(precision=16, suppress=True)
    # print("数值结果 Vector_body =")
    # print(Vector_body)
    # print("\n数值结果 Vector_body2 =")
    # print(Vector_body2)
    # print("\n数值结果 Vector_body_d =")
    # print(Vector_body_d)
    # print("\n数值结果 w =")
    # print(w)

    return   w


def calculate_geometry(xa, ya, za, beta_deg):
    """计算几何参数（beta_deg 为度数）"""
    beta = np.deg2rad(beta_deg)  # 角度转弧度
    ra = np.linalg.norm([xa, ya, za])
    cos_phi = xa / ra
    sin_phi = za / (ra * np.sin(beta))
    return cos_phi, sin_phi


def build_A_matrix(cos_phi, sin_phi, beta_deg, RAAN_deg):
    """构建旋转矩阵（beta_deg 和 RAAN_deg 均为度数）"""
    beta = np.deg2rad(beta_deg)  # beta 角度转弧度
    RAAN = np.deg2rad(RAAN_deg)  # RAAN 角度转弧度

    # 构建底层矩阵
    A_low = np.array([
        [-sin_phi, np.cos(beta) * cos_phi, np.sin(beta) * cos_phi],
        [0, np.sin(beta), -np.cos(beta)],
        [-cos_phi, -np.cos(beta) * sin_phi, -np.sin(beta) * sin_phi]
    ])

    # 构建RAAN旋转矩阵
    R_RAAN = np.array([
        [np.cos(RAAN), np.sin(RAAN), 0],
        [-np.sin(RAAN), np.cos(RAAN), 0],
        [0, 0, 1]
    ])

    return A_low @ R_RAAN


def calculate_angular_velocity(r, r_dot):
    """计算角速度"""
    return np.cross(r, r_dot) / (np.linalg.norm(r) ** 2 + np.finfo(float).eps)


def get_vector_body(xa, ya, za, xb, yb, zb, beta_deg, RAAN_deg):
    """计算本体坐标系矢量（beta_deg 和 RAAN_deg 均为度数）"""
    cos_phi, sin_phi = calculate_geometry(xa, ya, za, beta_deg)
    A = build_A_matrix(cos_phi, sin_phi, beta_deg, RAAN_deg)
    u = np.array([xb - xa, yb - ya, zb - za]).reshape(-1, 1)
    return (A @ u).flatten()





