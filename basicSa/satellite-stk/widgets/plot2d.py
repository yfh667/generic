
# widgets/plot2d.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtGui import QFont, QColor
import pyqtgraph as pg
import numpy as np
import basicSa.calculate_angular.calculate_angular as calculate_angular
import basicSa.los.satllite2satellite as sat2sat
import numpy as np
from config.settings import SatelliteConfig  # 新增导入

from collections import defaultdict
class SatelliteVector:
    def __init__(self, P, N):
        """
        P: 轨道面数量
        N: 每个轨道面的卫星数
        """
        self.P = P
        self.N = N
        self.M = (P-1) * N  # 需要计算的卫星对

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

        print('8')

    def _get_right_neighbor(self, sat_id):
        """获取右邻居ID"""
        orbit = sat_id // self.N+1
        plane = sat_id % self.N
        if orbit < self.P :
            return (orbit ) * self.N + plane
        return None




    def get_calculation_vectors(self):
        """返回用于计算的向量化数据视图"""
        return {
            'base':self.data['current_pos'],
            # 当前时刻相对向量 (目标星-本星)
            'delta_current': self.data['ref_current'] - self.data['current_pos'],

            # 下一时刻相对向量
            'delta_next': self.data['ref_next'] - self.data['next_pos'],

            # 轨道参数
            'RAAN': np.deg2rad(self.data['RAAN']),
            'inclination': np.deg2rad(self.data['inclination'])
        }

    def calculate_angular_velocity(self):
        """计算卫星间相对角速度（严格三维公式实现）

        返回值：
            np.ndarray: 各卫星的角速度模值（度/秒）
        """
        vecs = self.get_calculation_vectors()

        # =============================
        # 1. 基础参数计算
        # =============================
        # 轨道参数三角函数
        cosR = np.cos(vecs['RAAN'])  # RAAN余弦
        sinR = np.sin(vecs['RAAN'])  # RAAN正弦
        cosB = np.cos(vecs['inclination'])  # 轨道倾角余弦
        sinB = np.sin(vecs['inclination'])  # 轨道倾角正弦

        # 基准向量模长
        base_norm = np.linalg.norm(vecs['base'], axis=1, keepdims=True)

        # =============================
        # 2. Φ角相关参数计算
        # =============================
        # 防止除零错误（当sinB接近0时自动屏蔽后续无效计算）
        safe_sinB = np.where(sinB < 1e-6, np.nan, sinB)

        cos_phi = vecs['base'][:, 0] / (base_norm[:, 0] + 1e-8)  # x分量占比
        sin_phi = vecs['base'][:, 2] / (base_norm[:, 0] * safe_sinB + 1e-8)  # 经倾角校正后的z分量占比

        # =============================
        # 3. 复合旋转矩阵展开计算
        # =============================
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

        # 当前时刻旋转
        current_rotated = apply_rotation(*vecs['delta_current'].T)

        # 下一时刻旋转
        next_rotated = apply_rotation(*vecs['delta_next'].T)

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

        #az = angular_velocity_rad[2]
        angular_velocity_z = np.abs(angular_velocity_rad[:, 2])  # 取绝对值表示大小
        # 转换为角度制模长
        return angular_velocity_z * 180 / np.pi


class Satellite2DView(QWidget):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.current_params = (0, 0)
        self.active_sats = set()
        self.is_animating = False


        # 存储图形对象
        self.nodes = {}
        self.lines = []
        self.labels = []
        self.scatter = None


        # self.base_font_size = 8
        # self.min_font_size = 6
        # self.max_font_size = 24
        self.base_font = QFont("SimHei", 8)  # 初始化基础字体
        self.label_style = {'color': '#2c3e80', 'font-size': '8pt'}  # 使用CSS样式

        self._init_ui()
        self._init_view_scaling()


    def _init_view_scaling(self):
        """初始化视图缩放处理"""
        # 连接视图范围变化信号
        self.plot.sigRangeChanged.connect(self._update_font_sizes)
        # 记录初始视图范围
        self.original_view = self.plot.viewRect()

    def _update_font_sizes(self):
        """修正字体更新逻辑"""
        current_view = self.plot.viewRect()
        width_ratio = current_view.width() / self.original_view.width()

        # 计算新字号
        new_size = min(24, max(6, int(self.base_font.pointSize() / width_ratio)))

        # 更新所有标签
        for label in self.labels:
            # 通过HTML更新字体大小
            text = label.textItem.toPlainText()
            label.setHtml(
                f'<span style="font-family: SimHei; font-size: {new_size}pt; '
                f'color: {self.label_style["color"]};">{text}</span>'
            )
    def _init_ui(self):
        # 配置PyQtGraph
        pg.setConfigOptions(antialias=True, foreground='k', background='w')

        # 创建主布局
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 创建绘图部件
        self.plot = pg.PlotWidget()
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=False)
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.enableAutoRange()

        #隐藏坐标周
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')


        # 配置中文字体
        font = QFont()
        font.setFamily("SimHei")
        font.setPointSize(10)
        self.plot.getAxis("left").setStyle(tickFont=font)
        self.plot.getAxis("bottom").setStyle(tickFont=font)

        layout.addWidget(self.plot)

        # 初始化散点图
        self._init_scatter()

    def _init_scatter(self):
        """初始化散点图对象"""
        if self.scatter is None:
            self.scatter = pg.ScatterPlotItem(
                size=18,
                pen=pg.mkPen(QColor(30, 144, 255), width=2),
                brush=pg.mkBrush(QColor(135, 206, 250)),
                symbol='d',
                hoverable=True
            )
            self.plot.addItem(self.scatter)
        else:
            self.scatter.clear()

    def update_topology(self, N, P, adjacency):
        """更新网络拓扑"""
     #   self.current_params = (N, P)
        self._clear_drawing()
        pos = self._generate_positions(N, P)
        self._draw_nodes(pos)
        self._draw_connections(pos, adjacency)
        self._draw_labels(pos, N)
        self._set_view_range(N, P)

    def _generate_positions(self, N, P):
        """生成节点位置"""
        pos = {}

        for i in range(P):
            for j in range(N + 1):

                pos[(i, j)] = (i, -j)
        return pos

    def _draw_nodes(self, pos):
        """绘制节点"""
        points = []
        for (x, y) in pos.values():
            points.append({'pos': (x, y), 'data': 1})
        self.scatter.setData(points)

    def _draw_connections(self, pos, adjacency):
        """绘制连接线"""
        for node, neighbors in adjacency.items():
            if node not in pos:
                continue
            x1, y1 = pos[node]
            for neighbor in neighbors:
                if neighbor in pos:
                    x2, y2 = pos[neighbor]
                    line = pg.PlotCurveItem(
                        x=[x1, x2], y=[y1, y2],
                        pen=pg.mkPen(QColor(231, 76, 60), width=1.5)
                    )
                    self.plot.addItem(line)
                    self.lines.append(line)

    def _draw_labels(self, pos, N):
        """修正后的标签绘制方法"""
        for (i, j), (x, y) in pos.items():
            text = f"({i},{j})" if j < N else f"({i},0_COPY)"

            # 使用HTML样式设置字体
            label = pg.TextItem(
                html=f'<span style="font-family: SimHei; font-size: {self.base_font.pointSize()}pt; color: {self.label_style["color"]};">{text}</span>',
                anchor=(0.5, 1)
            )
            label.setPos(x, y - 0.1)
            self.plot.addItem(label)
            self.labels.append(label)

    def _set_view_range(self, N, P):
        """设置视图范围"""
        self.plot.setXRange(-1, P)
        self.plot.setYRange(-N - 1, 1)

    def _clear_drawing(self):
        """清除图形元素"""
        # 清除线条
        for line in self.lines:
            self.plot.removeItem(line)
        self.lines.clear()

        # 清除标签
        for label in self.labels:
            self.plot.removeItem(label)
        self.labels.clear()

        # 清空散点数据（使用兼容方式）
        if self.scatter is not None:
            self.scatter.setData([])  # 替换clear()方法
        else:
            self._init_scatter()  # 确保scatter存在

    def highlight_nodes(self, active_nodes):
        """高亮指定节点"""
        colors = []
        points = self.scatter.data

        for p in points:
            pos = (round(p['pos'][0]), round(-p['pos'][1]))
            if pos in active_nodes:
                colors.append((231, 76, 60, 255))  # 红色
            else:
                colors.append((52, 152, 219, 255))  # 蓝色

        self.scatter.setBrush(colors)




    def calculatelink(self,id,adj_list,satellite_positions_3d):
        P,N = self.current_params
        orbitid = (id-1) //N
        planeid_start =orbitid * N+1
        planeid_id = id-planeid_start
        # if(id==30):
        #     print("1")
        # # # right
        if orbitid != P-1:
            #RIGHTE
            rightnode = (orbitid+1)* N+planeid_id+1

            # sat1 = satellite_positions_3d[id  ]
            # x1,y1,z1 =  sat1.x,sat1.y,sat1.z
            # sat2 = satellite_positions_3d[rightnode]
            #
            # x2,y2,z2 = sat2.x,sat2.y,sat2.z
            #
            # #here is the los code
            #
            # #this use the simplest los based on the light of sight
            # los = sat2sat.Sat_Sat2(x1, y1, z1, x2, y2, z2)

            sat1_1 = satellite_positions_3d[id][0]
            x1_1,y1_1,z1_1 = sat1_1.x,sat1_1.y,sat1_1.z
            sat1_2 = satellite_positions_3d[id][1]
            x1_2,y1_2,z1_2 = sat1_2.x,sat1_2.y,sat1_2.z


            sat2_1 = satellite_positions_3d[rightnode][0]
            x2_1,y2_1,z2_1 = sat2_1.x,sat2_1.y,sat2_1.z
            sat2_2 = satellite_positions_3d[rightnode][1]
            x2_2,y2_2,z2_2 = sat2_2.x,sat2_2.y,sat2_2.z


            timestep =1
            params = ( x1_1,y1_1,z1_1 ,   x1_2,y1_2,z1_2,
                      x2_1,y2_1,z2_1,   x2_2,y2_2,z2_2 ,
                      sat1_1.trackangle, timestep,  sat1_1.RAAN)

            angular_velocity = calculate_angular.Get_angular_velocity(params)

            if abs(angular_velocity[2])< SatelliteConfig.ANGULAR_VELOCITY:
                los=1
            else:
                los=0

            if los:
                adj_list[orbitid, planeid_id].append((orbitid+1, planeid_id))



    def update_satellite_positions(self, satellite_positions_3d):
        """
        3D坐标更新入口
        这里可以添加从3D坐标到拓扑结构的转换逻辑
        """
        # 示例：生成虚拟邻接关系

        print("we have get")
        #print(satellite_positions_3d)
        print(f"en(satellite_positions_3d)")


        # adjacency = self._process_3d_positions(satellite_positions_3d)
        # self.update_topology(adjacency)

        P,N = self.current_params


        # 初始化邻接表





# it is simplest accomplishment
#         numberofsatellites= len(satellite_positions_3d)
#
#         nodes = [(i, j) for i in range(0, P) for j in range(0, N + 1)]
#         adj_list = {node: [] for node in nodes}
#         for id in range(1,(P-1)*N+1):
#             self.calculatelink( id, adj_list, satellite_positions_3d)


# we need
        sv = SatelliteVector(P, N)

        # 从原始数据加载
        sv.load_from_3dview(satellite_positions_3d)

        # 批量计算所有链路
        angular_vel = sv.calculate_angular_velocity()
        los_mask =angular_vel < SatelliteConfig.ANGULAR_VELOCITY # 可见性判断

        # 生成邻接表
        adj_list = defaultdict(list)
        for i in range(len(los_mask)):
            if los_mask[i] and sv._get_right_neighbor(i) is not None:
                orbit = i // N
                plane = i % N
                adj_list[(orbit, plane)].append((orbit + 1, plane))
#




        print(adj_list)
        self.update_topology( N,P,adj_list)

        # adjacency_example = {
        #     (1, 1): [(1, 2), (2, 1)],
        #     (1, 2): [(1, 1)],
        #     (2, 1): [(1, 1)],
        # }
        # N =3
        # P =4
        # self.update_topology( N,P,adjacency_example)



    def _process_3d_positions(self, positions):
        """处理3D坐标（示例）"""
        # 这里添加实际处理逻辑


        # HERE IS THE CODE TO DO
        adjacency = None

        return adjacency




    def update_layout(self, num_planes, sats_per_plane):
        """更新为点状布局"""
        self.current_params = (num_planes, sats_per_plane)




      #  self.satellite_positions_2d = satellite_positions_2d
      #  self.update_layout(*self.current_params)  # 重新更新布局，传递当前参数


    def set_animation_state(self, state):
        """设置动画状态"""
        self.is_animating = state
        if not state:
            self.active_sats = set()
            self.update_layout(*self.current_params)

    def highlight_active_sats(self, active_ids):
        """高亮活动卫星"""
        self.active_sats = set(active_ids)
        self.update_layout(*self.current_params)
