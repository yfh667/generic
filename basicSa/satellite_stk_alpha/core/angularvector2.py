
from datetime import timedelta
from datetime import datetime
import numpy as np
import basicSa.calculate_angular.gettheta as gettheta
import numpy as np
#from config.settings import SatelliteConfig
import basicSa.satellite_stk_alpha.config.settings as settings
class SatelliteVector:
    def __init__(self, P, N):
        """
        P: 轨道面数量
        N: 每个轨道面的卫星数
        """
        self.P = P
        self.N = N
        self.M = P * N  # 需要计算的卫星对

        # 定义数据类型 (可扩展)
        self.dtype = np.dtype([
            ('current_pos', '3f8'),  # xyz当前时刻
            ('next_pos', '3f8'),  # xyz下一时刻
            ('ref_current', '3f8'),  # 参考星当前时刻 (右邻居)
            ('ref_next', '3f8'),  # 参考星下一时刻
            ('RAAN', 'f8'),  # 升交点赤经
            ('inclination', 'f8')  # 轨道倾角
        ])

        # 初始化存储 (内存预分配)
        self.data = np.zeros(self.M, dtype=self.dtype)


    def load_from_3dview(self, satellite_positions_3d):
        """将原始3D视图数据加载到向量结构"""
        # 预计算轨道参数


        for i in range(self.M):
            sat = satellite_positions_3d[i]

            # 基本信息注入
            self.data[i]['current_pos'] = [sat[0].x, sat[0].y,sat[0].z]
            self.data[i]['next_pos'] = [sat[1].x, sat[1].y, sat[1].z]
            self.data[i]['RAAN'] =sat[0].RAAN
            self.data[i]['inclination'] = sat[0].trackangle

            # 计算参考星（右邻居）位置
            ref_id = self._get_right_neighbor(i)
            if ref_id is not None:
                ref_sat = satellite_positions_3d[ref_id]
                self.data[i]['ref_current'] = [ref_sat[0].x, ref_sat[0].y, ref_sat[0].z]
                self.data[i]['ref_next'] = [ref_sat[1].x, ref_sat[1].y, ref_sat[1].z]


    def _get_right_neighbor(self, sat_id):
        """获取右邻居ID"""
        orbit = sat_id // self.N+1
        plane = sat_id % self.N
        if orbit < self.P :
            return (orbit ) * self.N + plane
        return None

    def _get_right_neighbor2(self, sat_id,i):
        #i=1 to 6
        """获取右邻居ID"""
        if i==1:
            orbit = sat_id // self.N+1
            plane = (sat_id+1) % self.N
            if orbit < self.P :
                return orbit,plane,(orbit ) * self.N + plane
        elif i==2:
            orbit = sat_id // self.N+1
            plane = (sat_id) % self.N
            if orbit < self.P :
                return orbit,plane,(orbit ) * self.N + plane
        elif i==3:
            orbit = sat_id // self.N+1
            plane = (sat_id+self.N-1) % self.N
            if orbit < self.P :
                return orbit,plane,(orbit ) * self.N + plane
        elif i==4:
            orbit = sat_id // self.N+1
            plane = (sat_id+self.N-2) % self.N
            if orbit < self.P :
                return orbit,plane,(orbit ) * self.N + plane
        elif i==5:
            orbit = sat_id // self.N+2
            plane = (sat_id) % self.N
            if orbit < self.P :
                return orbit,plane,(orbit ) * self.N + plane

        # elif i==6:
        #
        #     orbit = sat_id // self.N+2
        #     plane = (sat_id-1+ self.N) % self.N
        #     if orbit < self.P :
        #         return orbit,plane,(orbit ) * self.N + plane

        else:
            orbit = sat_id // self.N + 2
            plane = (sat_id - 2 + self.N) % self.N
            if orbit < self.P:
                return orbit,plane,(orbit ) * self.N + plane
     #   print(1)
        return None


    def get_calculation_vectors(self):
        """返回用于计算的向量化数据视图"""
        return {
            'base':self.data['current_pos'],
            'next': self.data['next_pos'],
            # 当前时刻相对向量 (目标星-本星)
            'delta_current': self.data['ref_current'] - self.data['current_pos'],

            # 下一时刻相对向量
            'delta_next': self.data['ref_next'] - self.data['next_pos'],

            # 轨道参数,attention
            'RAAN': np.deg2rad(self.data['RAAN']),
            'inclination': np.deg2rad(self.data['inclination'])
        }
    def get_vectors(self,vecs,vectorbase,delta_current ):

        cosR = np.cos(vecs['RAAN'])  # RAAN余弦
        sinR = np.sin(vecs['RAAN'])  # RAAN正弦
        cosB = np.cos(vecs['inclination'])  # 轨道倾角余弦
        sinB = np.sin(vecs['inclination'])  # 轨道倾角正弦

        # 基准向量模长
        base_norm = np.linalg.norm(vectorbase, axis=1, keepdims=True)

        # =============================
        # 2. Φ角相关参数计算
        # =============================
        # 防止除零错误（当sinB接近0时自动屏蔽后续无效计算）
        safe_sinB = np.where(sinB < 1e-6, np.nan, sinB)


        cos_phi =  vectorbase[:, 0] / (base_norm[:, 0] + 1e-8)  # x分量占比
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

    def get_vectors2(self,RAAN,inclination,length,vectorbase,delta_current ):

        cosR = np.cos(RAAN)[:length]  # RAAN余弦
        sinR = np.sin(RAAN)[:length]    # RAAN正弦
        cosB = np.cos(inclination)[:length]   # 轨道倾角余弦
        sinB = np.sin(inclination)[:length]    # 轨道倾角正弦

        # 基准向量模长
        base_norm = np.linalg.norm(vectorbase, axis=1, keepdims=True)

        # =============================
        # 2. Φ角相关参数计算
        # =============================
        # 防止除零错误（当sinB接近0时自动屏蔽后续无效计算）
        safe_sinB = np.where(sinB < 1e-6, np.nan, sinB)


        cos_phi =  vectorbase[:, 0] / (base_norm[:, 0] + 1e-8)  # x分量占比
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


    # use the ecef vector
    def calculate_angular_velocity(self, time):
        """计算卫星间相对角速度（严格三维公式实现）

        返回值：
            np.ndarray: 各卫星的角速度模值（度/秒）
        """
        vecs = self.get_calculation_vectors()
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


        ecef_coords = vecs['base']  # shape (N, 3)
        # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
        eci_coords = ecef_coords @ R.T  # 注意：R.T 是转置！

        # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
        vecs['base_eci'] = eci_coords

        ecef_coords = vecs['next']  # shape (N, 3)
        # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
        eci_coords = ecef_coords @ R.T  # 注意：R.T 是转置！

        # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
        vecs['next_eci'] = eci_coords


        ecef_coords = vecs['delta_current']  # shape (N, 3)
        # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
        eci_coords = ecef_coords @ R.T  # 注意：R.T 是转置！

        # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
        vecs['delta_current_eci'] = eci_coords

        ecef_coords = vecs['delta_next']  # shape (N, 3)
        # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
        eci_coords = ecef_coords @ R.T  # 注意：R.T 是转置！

        # 现在 eci_coords 是 shape (N, 3)，即转换后的 ECI 坐标
        vecs['delta_next_eci'] = eci_coords
        #




        current_rotated = self.get_vectors( vecs,vecs['base_eci'],vecs['delta_current_eci']  )
        # 下一时刻旋转


        next_rotated = self.get_vectors( vecs,vecs['next_eci'],vecs['delta_next_eci']  )

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

    def calculate_angular_velocity_7_all(self, time):
        vecs = self.get_calculation_vectors()
       # baseposition = vecs['base']
       # nextposition = vecs['next']

        RAAN= vecs['RAAN']
        inclination  = vecs[ 'inclination']
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

        ecef_coords = vecs['base']  # shape (N, 3)
        # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
        baseposition = ecef_coords @ R.T  # 注意：R.T 是转置！

        ecef_coords = vecs['next'] # shape (N, 3)
        # 批量旋转：每一行右乘旋转矩阵，等价于 (ecef @ R.T)
        nextposition = ecef_coords @ R.T  # 注意：R.T 是转置！

        # HERE the baseposition and the nextposition is all the exef for the all noedes
        allangulars = []

        linkflags  = []


        for i in range(6):

            B_baseposition = []
            B_nextposition = []
            if i <=3:
                Plane = self.P-1

            else:
                Plane = self.P-2
            allnodeslen = Plane * self.N
            linkflag  =  np.zeros(allnodeslen)

            for j in range(allnodeslen):

                _,_,ref_id = self._get_right_neighbor2(j,i+1)
                B_baseposition.append(baseposition[ref_id])
                B_nextposition.append(nextposition[ref_id])



            angular = self.calculate_angular_velocity_7_base(  RAAN,inclination,allnodeslen,baseposition[:allnodeslen],nextposition[:allnodeslen],B_baseposition,B_nextposition)

           # angular =  np.ones(allnodeslen)


            for j in range(allnodeslen):
                if angular[j]<settings.SatelliteConfig.ANGULAR_VELOCITY:
                    linkflag[j] = 1

            linkflags.append(linkflag)


        return linkflags


    def calculate_angular_velocity_7_base(self, RAAN,inclination,length,A_baseposition,A_nextposition,B_baseposition,B_nextposition):

        delta_current  = B_baseposition-A_baseposition

        current_rotated = self.get_vectors2( RAAN,inclination,length,A_baseposition,delta_current  )
        # 下一时刻旋转

        delta_next = B_nextposition - A_nextposition
        next_rotated = self.get_vectors2( RAAN,inclination,length,A_nextposition,delta_next  )

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
        pass
