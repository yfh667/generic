import basicSa.Dataprocessing.readns3file as readns3file
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')  # or 'TkAgg', 'GTK3Agg', 'WXAgg' depending on what you have installed
import matplotlib.pyplot as plt
# 常量定义
STATION_NUM = 8
ORBIT_NUM = 18
SAT_PER_ORBIT = 36


# 链路类型枚举
class LinkType:
    EAST_UP_1 = 0
    SAME_ORBIT = 1
    EAST_DOWN_1 = 2
    EAST_DOWN_2 = 3
    JUMP_ORBIT_SAME = 4
    JUMP_ORBIT_DOWN_2 = 5

    WEST_UP_2 = 6
    WEST_UP_1 = 7
    WAST_SAME_ORBIT = 8

    WEST_DOWN_1 = 9
    JUMP_WEST_ORBIT_UP_2 = 10
    JUMP_WEST_ORBIT_SAME = 11

    UP_ORBIT = 12
    DOWN_ORBIT = 13



def calculate_link_type(N, first_plane, first_orbit, next_plane, next_orbit):
    """计算两个节点间的链路类型"""
    plane_diff = next_plane - first_plane
    orbit_diff = (next_orbit - first_orbit+N) % N

    if plane_diff == 1:
        if orbit_diff == 0:
            return LinkType.SAME_ORBIT
        elif orbit_diff == 1:
            return LinkType.EAST_UP_1
        elif orbit_diff == N - 1:
            return LinkType.EAST_DOWN_1
        elif orbit_diff == N - 2:
            return LinkType.EAST_DOWN_2
    elif plane_diff == 2:
        if orbit_diff == 0:
            return LinkType.JUMP_ORBIT_SAME
        else:
            return LinkType.JUMP_ORBIT_DOWN_2
    elif plane_diff == -1:
        if orbit_diff == 0:
            return LinkType.WAST_SAME_ORBIT
        elif orbit_diff == 1:
            return LinkType.WEST_UP_1
        elif orbit_diff == 2:
            return LinkType.WEST_UP_2
        elif orbit_diff == N - 1:
            return LinkType.WEST_DOWN_1
    elif    plane_diff == -2:
        if orbit_diff == 0:
            return LinkType.JUMP_WEST_ORBIT_SAME
        else:
            return LinkType.JUMP_WEST_ORBIT_UP_2
    elif plane_diff == 0:
        if orbit_diff == 1:
            return LinkType.UP_ORBIT
        else:
            return LinkType.DOWN_ORBIT

class PathSignatureMapper:
    def __init__(self):
        self.signature_map = {}  # 存储签名到数值的映射
        self.counter = 1  # 起始编号为1

    def get_mapped_value(self, signature):
        """获取签名的唯一数值标识"""
        if not signature:  # 空路径处理
            return 0

        if signature not in self.signature_map:
            self.signature_map[signature] = self.counter
            self.counter += 1
        return self.signature_map[signature]


def calculate_path_signature(path):
    """计算路径特征签名"""
    signature = []

    for i in range(1,len(path) - 2):
        first_node = path[i] - STATION_NUM
        next_node = path[i + 1] - STATION_NUM

        first_plane = first_node // SAT_PER_ORBIT
        first_orbit = first_node % SAT_PER_ORBIT
        next_plane = next_node // SAT_PER_ORBIT
        next_orbit = next_node % SAT_PER_ORBIT

        link_type = calculate_link_type(SAT_PER_ORBIT, first_plane, first_orbit,
                                        next_plane, next_orbit)
        # 使用字母编码链路类型
        signature.append(chr(65 + link_type))  # A-L 对应 0-11
    return ''.join(signature) if signature else 'N/A'


# def plot_signature_series(signatures, timestamps):
#     """修复版时间序列绘图"""
#     plt.figure(figsize=(14, 7))
#
#     # 主曲线
#     plt.plot(timestamps, signatures, 'b-', alpha=0.7,
#              label='Path ID', linewidth=1.5)
#
#     # 转换数组类型
#     signatures_arr = np.array(signatures)
#     timestamps_arr = np.array(timestamps)
#
#     # 计算变化点 (修复索引错误)
#     if len(signatures_arr) > 0:
#         # 使用prepend=signatures_arr[0]替代np.nan
#         diff_mask = np.insert(np.diff(signatures_arr) != 0, 0, False)
#         diff_indices = np.where(diff_mask)[0].astype(int)
#     else:
#         diff_indices = np.array([], dtype=int)
#
#     # 安全绘制变化点
#     if len(diff_indices) > 0:
#         plt.scatter(timestamps_arr[diff_indices],
#                     signatures_arr[diff_indices],
#                     c='red', s=40, zorder=3,
#                     label='Routing Change')
#
#     # 坐标轴优化
#     if signature_mapper.signature_map:
#         max_id = max(signature_mapper.signature_map.values())
#         plt.ylim(0.5, max_id + 0.5)
#         plt.yticks(
#             sorted(signature_mapper.signature_map.values()),
#             [k for k, _ in sorted(signature_mapper.signature_map.items(),
#                                   key=lambda x: x[1])],
#             fontsize=8
#         )
#
#     # 辅助元素
#     plt.title('Routing Path Transition Timeline')
#     plt.xlabel('Simulation Time (s)')
#     plt.ylabel('Path Signature')
#     plt.legend()
#     plt.grid(axis='y', alpha=0.3)
#
#     # 智能标注（修复索引错误）
#     if len(diff_indices) > 0:
#         last_change = None
#         # 使用逆序时注意索引范围
#         for idx in reversed(diff_indices):
#             if idx < len(signatures_arr):
#                 current_sig = signatures_arr[idx]
#                 if current_sig != last_change:
#                     plt.annotate(f'ID:{current_sig}',
#                                  (timestamps_arr[idx], current_sig),
#                                  textcoords="offset points",
#                                  xytext=(0, 10),
#                                  ha='center',
#                                  arrowprops=dict(arrowstyle="->",
#                                                  color='black',
#                                                  alpha=0.6))
#                     last_change = current_sig
#
#     plt.tight_layout()
#     plt.show()

def plot_signature_series(mapped_values, time_points):
    """Simplified line plot showing path transitions"""
    plt.figure(figsize=(14, 7))

    # Basic line plot
    plt.plot(time_points, mapped_values, 'b-', linewidth=1.5)

    # Basic formatting
    plt.title('Routing Path Transition Timeline')
    plt.xlabel('Simulation Time (s)')
    plt.ylabel('Path ID')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# def plot_signature_series(mapped_values, time_points):
#     plt.figure(figsize=(14, 7))
#
#     # 自定义 y 轴顺序：将 12,14,13 等手动排序
#     custom_order = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 15, 16, 17, 18]
#     plt.yticks(custom_order, [f"ID={i}" for i in custom_order])
#
#     plt.plot(time_points, mapped_values, 'b-', linewidth=1.5)
#     plt.title('Routing Path Transition Timeline (Custom Order)')
#     plt.grid(alpha=0.3)
#     plt.show()


def plot_path_analysis(end_node, db):
    """主分析函数"""
    time_points = []
    path_signatures = []
    all_signatures = []

    for snapshot in db.snapshots:
        paths = snapshot.active_paths
        if end_node in paths and paths[end_node]:
            path = paths[end_node][0]['path']

            signature = calculate_path_signature(path)
            all_signatures.append(signature)
            time_points.append(snapshot.timestamp)



    # 绘制图表
    # 批量映射（保证统一编号）

    mapped_values = [signature_mapper.get_mapped_value(s) for s in all_signatures]

    # 打印统计信息
    print(f"Total unique paths: {signature_mapper.counter - 1}")
    print("Signature mapping:", signature_mapper.signature_map)

    # 绘图
    plot_signature_series(mapped_values, time_points)

def main():
    # 文件读取
    try:
        db = readns3file.readxml('/home/yfh/Desktop/Data/basicSa.xml')
        print(f"Loaded {len(db.snapshots)} snapshots")
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return

    # 分析指定节点
    TARGET_NODE = 1
    try:
        plot_path_analysis(TARGET_NODE, db)
    except KeyError:
        print(f"Node {TARGET_NODE} not found in path data")

if __name__ == "__main__":
    # 初始化全局映射器
    signature_mapper = PathSignatureMapper()
    main()

