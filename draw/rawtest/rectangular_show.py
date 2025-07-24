# -*- coding: utf-8 -*-
import matplotlib
from matplotlib.patches import Rectangle   # ← 在文件顶部再加这一行

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors
import sys
import matplotlib.patches as patches  # <--- 新增：导入 patches 模块

import draw.read_snap_xml as read_snap_xml
# --- 参数配置 ---
N = 36  # 每轨道卫星数
P = 18  # 轨道平面数


# --- 地面站分组 ---
# 注意：这里的地面站ID需要与XML文件中station元素的id属性一致
STATION_GROUPS = {
    0: {"name": "Group 0", "stations": list(range(0, 4))},

    1: {"name": "Group 1", "stations": list(range(4, 9))},
    2: {"name": "Group 2", "stations": [9]},
    3: {"name": "Group 3", "stations": [10]},
    4: {"name": "Group 4", "stations": list(range(11, 15))},
    5: {"name": "Group 5", "stations": list(range(15, 17))},
6: {"name": "Group 6", "stations": list(range(17, 20))},
}

# 高对比度颜色（每组唯一）
# 颜色索引与STATION_GROUPS的键对应
GROUP_COLORS = [
    '#FF0000',  # 红 (Group 0)
    '#00FF00',  # 绿 (Group 1)
    '#0000FF',  # 蓝 (Group 2)
    '#FFA500',  # 橙 (Group 3)
    '#800080',  # 紫 (Group 4)
    '#00FFFF',  # 青 (Group 5)
'#FFFF00',  # 黄   (Group 6)
]


#
# # # --- 绘制所有卫星（动态颜色更新） ---
# def plot_grouped_satellites(group_data):
#     total_sats = N * P  # 卫星总数
#     sat_ids = np.arange(total_sats)
#     # 计算每个卫星在网格上的位置 (平面索引, 卫星索引)
#     all_cols = sat_ids // N  # X轴：轨道平面索引 (0 to P-1)
#     all_rows = sat_ids % N  # Y轴：卫星在平面内的索引 (0 to N-1)
#
#     # fig, ax = plt.subplots(figsize=(14, 9))
#     # plt.subplots_adjust(bottom=0.25)
#     fig, ax = plt.subplots(figsize=(14, 9))
#   # ① 把右侧也留出来给图例
#     fig.subplots_adjust(bottom=0.25, right=0.75)
#     # 初始化所有卫星为默认状态（未被任何组可见）
#     # 默认颜色为白色填充，浅灰色边框
#     scat = ax.scatter(
#         all_cols,
#         all_rows,
#         s=80,
#         facecolors='white',        # 默认填充白色
#         edgecolors='#CCCCCC',      # 默认浅灰边框
#         linewidths=0.8,
#         alpha=0.9
#     )
#
#     # 配置坐标轴
#     ax.set_xlabel('Orbit Plane Index (P)')
#     ax.set_ylabel('Satellite Index in Plane (N)')
#     ax.set_title('Grouped Satellite Visibility')
#     # 设置轴范围和刻度，确保每个卫星位置居中显示
#     ax.set_xlim(-0.5, P - 0.5)
#     ax.set_ylim(-0.5, N - 0.5)
#     ax.set_xticks(np.arange(P))
#     ax.set_yticks(np.arange(N))
#     ax.grid(True, linestyle='--', alpha=0.5) # 添加网格线
#
#     # 创建图例
#     legend_handles = [
#         # 为每个组创建一个图例项，显示组的颜色
#         plt.Line2D([0], [0], marker='o', color='w', markersize=10,
#                    markerfacecolor=GROUP_COLORS[gid], markeredgecolor='k',
#                    label=STATION_GROUPS[gid]["name"])
#         for gid in STATION_GROUPS
#     ]
#     # 添加一个图例项表示未被任何组可见的卫星（默认颜色）
#     legend_handles.append(
#          plt.Line2D([0], [0], marker='o', color='w', markersize=10,
#                    markerfacecolor='white', markeredgecolor='#CCCCCC',
#                    label='Not Visible to Any Group')
#     )
#    # ax.legend(handles=legend_handles, loc='upper right', title="Visible to Group")
#     ax.legend(
#                 handles = legend_handles,
#             loc = 'center left',  # 图例框左侧对齐锚点
#             bbox_to_anchor = (1.02, 0.5),  # x>1 放到轴外；y=0.5 垂直居中
#             borderaxespad = 0.0,
#             title = "Visible to Group"
#                           )
#
#     # 更新时间函数：根据当前时间步的数据更新卫星颜色
#     def update(step):
#         # 获取当前时间步的数据，如果步数不存在，则使用空数据
#         current_data = group_data.get(int(step), {"groups": {}, "all_mentioned": set()})
#         total_sats = N * P # 确保total_sats在函数内部可用
#
#         # 初始化所有卫星的颜色和边框为默认状态
#         facecolors = np.array([(1.0, 1.0, 1.0, 1.0)] * total_sats)  # 全白，RGBA格式
#         edgecolors = np.array([(0.8, 0.8, 0.8, 1.0)] * total_sats)  # 浅灰边框，RGBA格式
#
#         # 遍历所有可能的卫星ID
#         for sat_id in range(total_sats):
#             # 跳过卫星ID超出范围的情况 (虽然range(total_sats)已经确保了范围，这里留着作为安全检查)
#             if sat_id < 0 or sat_id >= total_sats:
#                 continue
#
#             # 如果卫星ID在当前时间步的'all_mentioned'集合中，说明它至少被某个地面站提及过
#             if sat_id in current_data["all_mentioned"]:
#                 # 找到当前时间步看到这颗卫星的所有组
#                 seeing_groups = []
#                 for gid in STATION_GROUPS:
#                     # 检查卫星是否在该组可见的卫星集合中
#                     if sat_id in current_data["groups"].get(gid, set()):
#                         seeing_groups.append(gid)
#
#                 # 如果卫星被至少一个组看到
#                 if len(seeing_groups) > 0:
#                     # --- 上色逻辑 ---
#                     # 如果卫星被多个组看到，我们选择组ID最小的那个组的颜色作为代表色
#                     # 您可以根据需要修改这里的逻辑（例如，使用一个特殊的颜色表示多组可见）
#                     assigned_gid = min(seeing_groups)
#                     facecolors[sat_id] = mcolors.to_rgba(GROUP_COLORS[assigned_gid])
#                     # 将边框颜色改为黑色，表示该卫星当前可见（被至少一个组可见）
#                     edgecolors[sat_id] = (0.0, 0.0, 0.0, 1.0)
#                 # Note: 如果一个卫星在all_mentioned中，但不在任何一个组的集合中
#                 # (这可能发生在XML中存在一个地面站，但该站的ID不在STATION_GROUPS里)，
#                 # 那么len(seeing_groups)会是0。根据上面的if判断，它会保持默认颜色。
#                 # 然而，根据parse_xml_group_data的逻辑，如果一个站不在STATION_GROUPS，
#                 # 它的可见卫星就不会被加入到任何组的集合，也不会加入all_mentioned。
#                 # 所以，如果sat_id在all_mentioned中，它肯定会被至少一个组看到。
#
#         # 更新散点图的颜色和边框
#         scat.set_facecolors(facecolors)
#         scat.set_edgecolors(edgecolors)
#
#         # 更新图标题
#         ax.set_title(f'Grouped Satellite Visibility (Step {int(step)})')
#
#         # 刷新画布
#         fig.canvas.draw_idle()
#
#     # 时间滑块
#     ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
#     steps = sorted(group_data.keys())
#
#     if not steps:
#          print("Error: No time steps found in XML data after parsing.")
#          sys.exit(1) # 退出程序如果没有任何时间步数据
#
#     time_slider = Slider(
#         ax=ax_time,
#         label='Time Step',
#         valmin=min(steps),          # 滑块最小值
#         valmax=max(steps),          # 滑块最大值
#         valinit=min(steps),         # 初始值
#         valstep=1                   # 步长为1
#     )
#
#     # 当滑块值改变时调用update函数
#     time_slider.on_changed(update)
#
#     # 初始化显示第一个时间步的数据
#     update(min(steps))
#     plt.show()
from matplotlib.patches import Rectangle


# --- 绘制所有卫星（动态颜色更新） ---
def plot_grouped_satellites(group_data):
    total_sats = N * P  # 卫星总数
    sat_ids = np.arange(total_sats)
    # 计算每个卫星在网格上的位置 (平面索引, 卫星索引)
    all_cols = sat_ids // N  # X轴：轨道平面索引 (0 to P-1)
    all_rows = sat_ids % N  # Y轴：卫星在平面内的索引 (0 to N-1)

    fig, ax = plt.subplots(figsize=(14, 9))
    # ① 把右侧也留出来给图例
    fig.subplots_adjust(bottom=0.25, right=0.75)
    # 初始化所有卫星为默认状态（未被任何组可见）
    # 默认颜色为白色填充，浅灰色边框
    scat = ax.scatter(
        all_cols,
        all_rows,
        s=80,
        facecolors='white',  # 默认填充白色
        edgecolors='#CCCCCC',  # 默认浅灰边框
        linewidths=0.8,
        alpha=0.9
    )

    # 配置坐标轴
    ax.set_xlabel('Orbit Plane Index (P)')
    ax.set_ylabel('Satellite Index in Plane (N)')
    ax.set_title('Grouped Satellite Visibility')
    # 设置轴范围和刻度，确保每个卫星位置居中显示
    ax.set_xlim(-0.5, P - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_xticks(np.arange(P))
    ax.set_yticks(np.arange(N))
    ax.grid(True, linestyle='--', alpha=0.5)

    # ===================================================================
    # START: 新增代码 - 创建包络矩形
    # ===================================================================
    # 创建一个矩形对象，初始时不可见。我们将在 update 函数中更新它的位置和可见性。
    # 我们用一个显眼的颜色（如深粉色）和虚线来表示这个矩形。
    envelope_rect = patches.Rectangle(
        (0, 0), 0, 0,  # 初始坐标(x,y)和尺寸(width,height)，将被动态更新
        linewidth=2,
        edgecolor='deeppink',
        facecolor='none',  # 不填充颜色
        linestyle='--',
        visible=False,  # 初始时隐藏
        zorder=10  # 设置 zorder 以确保它绘制在网格之上
    )
    ax.add_patch(envelope_rect)  # 将矩形添加到图表中



    envelope_rect2 = patches.Rectangle(
        (0, 0), 0, 0,  # 初始坐标(x,y)和尺寸(width,height)，将被动态更新
        linewidth=2,
        edgecolor='deeppink',
        facecolor='none',  # 不填充颜色
        linestyle='--',
        visible=False,  # 初始时隐藏
        zorder=10  # 设置 zorder 以确保它绘制在网格之上
    )
    ax.add_patch(envelope_rect2)  # 将矩形添加到图表中
    # ===================================================================
    # END: 新增代码
    # ===================================================================

    # 创建图例
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                   markerfacecolor=GROUP_COLORS[gid], markeredgecolor='k',
                   label=STATION_GROUPS[gid]["name"])
        for gid in STATION_GROUPS
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker='o', color='w', markersize=10,
                   markerfacecolor='white', markeredgecolor='#CCCCCC',
                   label='Not Visible to Any Group')
    )
    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        title="Visible to Group"
    )
    old_x_min_group4 = [-1]  # 用list封装，list是可变对象
    old_x_min_group0 = [-1]  # 用list封装，list是可变对象
    # 更新时间函数：根据当前时间步的数据更新卫星颜色
    def update(step):
        current_data = group_data.get(int(step), {"groups": {}, "all_mentioned": set()})
        total_sats = N * P

        facecolors = np.array([(1.0, 1.0, 1.0, 1.0)] * total_sats)
        edgecolors = np.array([(0.8, 0.8, 0.8, 1.0)] * total_sats)

        for sat_id in range(total_sats):
            if sat_id in current_data["all_mentioned"]:
                seeing_groups = []
                for gid in STATION_GROUPS:
                    if sat_id in current_data["groups"].get(gid, set()):
                        seeing_groups.append(gid)

                if len(seeing_groups) > 0:
                    assigned_gid = min(seeing_groups)
                    facecolors[sat_id] = mcolors.to_rgba(GROUP_COLORS[assigned_gid])
                    edgecolors[sat_id] = (0.0, 0.0, 0.0, 1.0)

        # ===================================================================
        # START: 新增代码 - 更新包络矩形
        # ===================================================================
        # 获取 group 4 在当前时间步的所有卫星ID
        group4_sats = current_data["groups"].get(4, set())

        if group4_sats:  # 如果 group 4 在此时间步有可见卫星
            # 1. 计算所有 group 4 卫星的 x 坐标 (轨道平面索引)
            x_coords = [sid // N for sid in group4_sats]
            x_min = min(x_coords)  # 找到 x 坐标的最小值
            x_max = max(x_coords)  # 找到 x 坐标的最大值
            # 2. 定义矩形参数
            # 为了美观，我们将矩形向外扩展0.5个单位，使其完全包围卫星点
            rect_x = x_min - 0.5
            # 你的变换逻辑将 group 4 的最高点移至 y=35。
            # 矩形高度为 4，所以它应覆盖 y=32, 33, 34, 35 的行。
            # 矩形的左下角 y 坐标为 35.5 - 4 = 31.5。
            rect_y = 35.5 - 4
            rect_width = 7  # 固定的宽度
            rect_height = 4  # 固定的高度
            if old_x_min_group4[0] == -1:
                old_x_min_group4[0] = x_min
            elif old_x_min_group4[0] + rect_width > x_max:
                rect_x = old_x_min_group4[0] - 0.5
            else:
                old_x_min_group4[0] = x_min
            # 3. 更新矩形的位置、尺寸并使其可见
            envelope_rect.set_xy((rect_x, rect_y))
            envelope_rect.set_width(rect_width)
            envelope_rect.set_height(rect_height)
            envelope_rect.set_visible(True)
        else:
            # 如果 group 4 在此时间步没有可见卫星，则隐藏矩形
            envelope_rect.set_visible(False)



        # 获取 group 4 在当前时间步的所有卫星ID
        group0_sats = current_data["groups"].get(0, set())

        if group0_sats:  # 如果 group 4 在此时间步有可见卫星
            # 1. 计算所有 group 4 卫星的 x 坐标 (轨道平面索引)
            x_coords = [sid // N for sid in group0_sats]
            x_min = min(x_coords)  # 找到 x 坐标的最小值
            x_max = max(x_coords)  # 找到 x 坐标的最大值
            # 2. 定义矩形参数
            # 为了美观，我们将矩形向外扩展0.5个单位，使其完全包围卫星点
            rect_x = x_min - 0.5
            # 你的变换逻辑将 group 4 的最高点移至 y=35。
            # 矩形高度为 4，所以它应覆盖 y=32, 33, 34, 35 的行。
            # 矩形的左下角 y 坐标为 35.5 - 4 = 31.5。
            rect_y = 8.5
            rect_width = 7  # 固定的宽度
            rect_height = 5  # 固定的高度
            if old_x_min_group0[0] == -1:
                old_x_min_group0[0] = x_min
            elif old_x_min_group0[0] + rect_width > x_max:
                rect_x = old_x_min_group0[0] - 0.5
            else:
                old_x_min_group0[0] = x_min
            # 3. 更新矩形的位置、尺寸并使其可见
            envelope_rect2.set_xy((rect_x, rect_y))
            envelope_rect2.set_width(rect_width)
            envelope_rect2.set_height(rect_height)
            envelope_rect2.set_visible(True)
        else:
            # 如果 group 4 在此时间步没有可见卫星，则隐藏矩形
            envelope_rect2.set_visible(False)
        # ===================================================================
        # END: 新增代码
        # ===================================================================

        scat.set_facecolors(facecolors)
        scat.set_edgecolors(edgecolors)
        ax.set_title(f'Grouped Satellite Visibility (Step {int(step)})')
        fig.canvas.draw_idle()

    # 时间滑块
    ax_time = plt.axes([0.2, 0.1, 0.65, 0.03])
    steps = sorted(group_data.keys())

    if not steps:
        print("Error: No time steps found in XML data after parsing.")
        sys.exit(1)

    time_slider = Slider(
        ax=ax_time,
        label='Time Step',
        valmin=min(steps),
        valmax=max(steps),
        valinit=min(steps),
        valstep=1
    )

    time_slider.on_changed(update)
    update(min(steps))
    plt.show()
def modify_group_data(group_data, N=36):
    new_group_data = {}

    for step in range(len(group_data)):
        raw_groups = group_data[step]['groups']
        new_group_data[step] = {'groups': {}, 'all_mentioned': set()}

        # 1. 获取 group 4 中最大 y（决定顶部对齐用）
        group4_sats = raw_groups.get(4, set())
        y_up = max([sid % N for sid in group4_sats]) if group4_sats else 0
        offset = N - y_up - 1  # 将 group 4 提到最顶格

        # 2. 修改 group 4 的卫星位置：直接向上提
        # new_group_data[step]['groups'][4] = set()
        # for t in group4_sats:
        #     x = t // N
        #     y = t % N + offset
        #     new_sid = x * N + y
        #     new_group_data[step]['groups'][4].add(new_sid)
        #     new_group_data[step]['all_mentioned'].add(new_sid)

        # 3. 修改其他 group 的卫星（向上移动或保持不变）
        for gid, sats in raw_groups.items():
            # if gid == 4:
            #     continue  # 已处理

            if gid not in new_group_data[step]['groups']:
                new_group_data[step]['groups'][gid] = set()

            for sid in sats:
                y = sid % N
                x = sid // N

                # 如果在 group 4 上方，直接平移下去
                if y > y_up:
                    y_new = y - y_up-1
                else:
                    y_new = y+ offset

                new_sid = x * N + y_new
                new_group_data[step]['groups'][gid].add(new_sid)
                new_group_data[step]['all_mentioned'].add(new_sid)

    return new_group_data

# --- 主程序 ---
if __name__ == "__main__":
   # xml_file =  "E:\Data\station_visible_satellites_648.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>
    xml_file =  "E:\Data\station_visible_satellites_648_8_h.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>

   # dummy_file_name =
   # dummy_file_name =
    start_ts = 1202
  #  end_ts = 21431
    end_ts = 3320


    try:
        # 解析XML数据
        group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
        group_data = read_snap_xml.modify_group_data(group_data, N=36,groupid=4)
        if not group_data:
            print(f"Error: No valid group visibility data parsed from {xml_file}.")
            print("Please check if the XML file exists and contains 'time' elements with 'stations' and 'satellite' data.")
            sys.exit(1) # 退出程序如果解析失败或没有数据

        # 绘制可视化图
        plot_grouped_satellites(group_data)

    except FileNotFoundError:
        print(f"Error: XML file not found at {xml_file}")
        sys.exit(1) # 退出程序如果文件未找到
    except ET.ParseError:
        print(f"Error: Could not parse XML file {xml_file}.")
        print("Please check the XML file format for syntax errors.")
        sys.exit(1) # 退出程序如果XML解析错误
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1) # 退出程序如果发生其他错误