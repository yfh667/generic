
# import matplotlib
#
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import xml.etree.ElementTree as ET
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Set, List, Tuple



# --- 地面站分组 ---
# 注意：这里的地面站ID需要与XML文件中station元素的id属性一致
# STATION_GROUPS = {
#     0: {"name": "Group 0", "stations": list(range(0, 4))},
#     1: {"name": "Group 1", "stations": list(range(4, 9))},
#     2: {"name": "Group 2", "stations": [9]},
#     3: {"name": "Group 3", "stations": [10]},
#     4: {"name": "Group 4", "stations": list(range(11, 15))},
#     5: {"name": "Group 5", "stations": list(range(15, 17))},
#     6: {"name": "Group 6", "stations": list(range(17, 20))},  # 修正这里的缩进
# }

# STATION_GROUPS = {
#     0: {"name": "Group 0", "stations": list(range(0, 12))},
#     1: {"name": "Group 1", "stations": list(range(12, 21))},
#     # 2: {"name": "Group 2", "stations": [9]},
#     # 3: {"name": "Group 3", "stations": [10]},
#     # 4: {"name": "Group 4", "stations": list(range(11, 15))},
#     # 5: {"name": "Group 5", "stations": list(range(15, 17))},
#     # 6: {"name": "Group 6", "stations": list(range(17, 20))},  # 修正这里的缩进
# }

# STATION_GROUPS = {
#     0: {"name": "Group 0", "stations": list(range(0, 1))},
#     1: {"name": "Group 1", "stations": list(range(1, 2))},
#     # 2: {"name": "Group 2", "stations": [9]},
#     # 3: {"name": "Group 3", "stations": [10]},
#     # 4: {"name": "Group 4", "stations": list(range(11, 15))},
#     # 5: {"name": "Group 5", "stations": list(range(15, 17))},
#     # 6: {"name": "Group 6", "stations": list(range(17, 20))},  # 修正这里的缩进
# }

# STATION_GROUPS = {
#     0: {"name": "Group 0", "stations": list(range(0, 4))},
#     1: {"name": "Group 1", "stations": list(range(4, 8))},
#     # 2: {"name": "Group 2", "stations": [9]},
#     # 3: {"name": "Group 3", "stations": [10]},
#     # 4: {"name": "Group 4", "stations": list(range(11, 15))},
#     # 5: {"name": "Group 5", "stations": list(range(15, 17))},
#     # 6: {"name": "Group 6", "stations": list(range(17, 20))},  # 修正这里的缩进
# }




# 高对比度颜色（每组唯一）
# 颜色索引与STATION_GROUPS的键对应
# GROUP_COLORS = [
#     '#FF0000',  # 红 (Group 0)
#     '#00FF00',  # 绿 (Group 1)
#     '#0000FF',  # 蓝 (Group 2)
#     '#FFA500',  # 橙 (Group 3)
#     '#800080',  # 紫 (Group 4)
#     '#00FFFF',  # 青 (Group 5)
# '#FFFF00',  # 黄   (Group 6)
# ]
#




from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Dict, Set

# 假设已有 STATION_GROUPS: Dict[int, Dict[str, Set[int]]]
# 例如: STATION_GROUPS = {0: {"stations": {1,2}}, 4: {"stations": {7,8}}}

def parse_xml_group_data(
    xml_file: str | Path,
config,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,   # 含 end_step
) -> Dict[int, Dict[str, Dict[int, Set[int]] | Set[int]]]:
    """
    解析 XML 并按时间窗口过滤（含 end_step）。
    返回结构:
      { step: { "groups": {gid: set(sat_id)}, "all_mentioned": set(sat_id) } }
    """
    xml_file = str(xml_file)

    # -------- 预处理：station_id -> group_id 映射（O(总站数)）--------
    # 避免对每个 station 都在 STATION_GROUPS 里线性搜索
    station_to_gid: Dict[int, int] = {}
    #for gid, info in STATION_GROUPS.items():
    for gid, info in config.station_groups.items():
        for sid in info["stations"]:
            station_to_gid[sid] = gid

    # 结果字典
    group_data: Dict[int, Dict[str, Dict[int, Set[int]] | Set[int]]] = {}

    # 为了速度做局部绑定
    station_to_gid_get = station_to_gid.get
   # groups_template = {gid: set() for gid in STATION_GROUPS}
    groups_template = {gid: set() for gid in config.station_groups}

    # iterparse 流式解析，内存友好
    # 仅在遇到 <time> 结束标签时处理这个 time 块
    context = ET.iterparse(xml_file, events=("end",))
    for event, elem in context:
        if elem.tag != "time":
            continue

        step_attr = elem.get("step")
        if step_attr is None:
            elem.clear()
            continue

        try:
            step = int(step_attr)
        except ValueError:
            elem.clear()
            continue

        # 时间窗口（不含 end_step）
        if start_step is not None and step < start_step:
            elem.clear()
            continue
        if end_step is not None and step > end_step:
            elem.clear()
            continue

        # 初始化当前 step 的容器（深拷贝 groups_template 的“空壳”）
        # 用字典推导比 deepcopy 更轻量
        groups = {gid: set() for gid in groups_template}
        all_mentioned: Set[int] = set()

        # stations 元素
        stations_elem = elem.find("stations")
        if stations_elem is not None:
            # 局部绑定以减少属性查找开销
            station_findall = stations_elem.findall
            for station_elem in station_findall("station"):
                sid_attr = station_elem.get("id")
                if sid_attr is None:
                    continue
                try:
                    sid = int(sid_attr)
                except ValueError:
                    continue

                gid = station_to_gid_get(sid)
                if gid is None:
                    # 未分组的地面站直接跳过
                    continue

                sats = groups[gid]
                # 直接遍历子元素，比多次 findall+临时集合快
                for sat_elem in station_elem:
                    if sat_elem.tag != "satellite":
                        continue
                    sat_id_attr = sat_elem.get("id")
                    if not sat_id_attr:
                        continue
                    # 一些 XML 会把 id 写成 "123" 或 "123.0"；先尝试 int，退回 float->int
                    try:
                        sat_id = int(sat_id_attr)
                    except ValueError:
                        try:
                            sat_id = int(float(sat_id_attr))
                        except ValueError:
                            continue

                    sats.add(sat_id)
                    all_mentioned.add(sat_id)

        group_data[step] = {"groups": groups, "all_mentioned": all_mentioned}

        # 释放已处理节点，降低内存
        elem.clear()

    return group_data

def parse_all_station_timeseries_array(
    xml_file: str | Path,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,   # 含 end_step
):
    """
    返回数组，每个元素对应一个地面站：
    [
      {
        "station_id": int,
        "series": { step:int -> set(sat_id:int) },
        "all_sats": [sat_id, ...]   # 该站在窗口内出现过的卫星并集（可选辅助）
      },
      ...
    ]
    station_id 按升序排列。
    """
    xml_file = str(xml_file)

    # 1) 先扫描窗口内出现过的全部 station id
    station_ids: Set[int] = set()
    context = ET.iterparse(xml_file, events=("end",))
    for _, elem in context:
        if elem.tag != "time":
            continue

        step_attr = elem.get("step")
        if step_attr is None:
            elem.clear()
            continue
        try:
            step = int(step_attr)
        except ValueError:
            elem.clear()
            continue

        if start_step is not None and step < start_step:
            elem.clear()
            continue
        if end_step is not None and step > end_step:
            elem.clear()
            continue

        stations_elem = elem.find("stations")
        if stations_elem is not None:
            for st in stations_elem.findall("station"):
                sid_attr = st.get("id")
                if sid_attr is None:
                    continue
                try:
                    sid = int(sid_attr)
                except ValueError:
                    try:
                        sid = int(float(sid_attr))
                    except ValueError:
                        continue
                station_ids.add(sid)

        elem.clear()

    ordered_station_ids = sorted(station_ids)

    # 2) 复用现有函数，拿每个站的时序
    series_list = parse_station_timeseries(
        xml_file=xml_file,
        station_ids=ordered_station_ids,
        start_step=start_step,
        end_step=end_step,
    )

    # 3) 组装成你要的“数组”
    out = []
    for sid, ts in zip(ordered_station_ids, series_list):
        all_sats = sorted(set().union(*ts.values())) if ts else []
        out.append({
            "station_id": sid,
            "series": ts,
            "all_sats": all_sats,
        })
    return out


#
#
# def parse_xml_group_data(xml_file, start_step=None, end_step=None):
#     """
#     解析 XML 并（可选）按时间窗口过滤。
#
#     参数
#     ----
#     xml_file   : str | PathLike
#     start_step : int | None   # 起始时间步（含），None 表示从头开始
#     end_step   : int | None   # 结束时间步（含），None 表示读到文件末尾
#
#     返回
#     ----
#     {
#         step: {
#             "groups": {gid: {sat_id, ...}},
#             "all_mentioned": {sat_id, ...}
#         },
#         ...
#     }
#     """
#     tree = ET.parse(xml_file)
#     root = tree.getroot()
#
#     group_data = {}
#     for time_elem in root.findall('time'):
#         step = int(time_elem.get('step'))
#
#         # -------- 时间窗口过滤 --------
#         if start_step is not None and step < start_step:
#             continue
#         if end_step is not None and step >= end_step:
#             continue
#         # --------------------------------
#
#         group_data[step] = {
#             "groups": {gid: set() for gid in STATION_GROUPS},
#             "all_mentioned": set()
#         }
#
#         stations_elem = time_elem.find('stations')
#         if stations_elem is None:
#             continue
#
#         for station_elem in stations_elem.findall('station'):
#             station_id = int(station_elem.get('id'))
#
#             # 判断地面站属于哪一组
#             assigned_gid = next(
#                 (gid for gid, info in STATION_GROUPS.items()
#                  if station_id in info["stations"]),
#                 None
#             )
#
#             if assigned_gid is None:
#                 continue  # 该地面站未划分到任何组
#
#             # 读取并转换卫星 ID
#             valid_ids = {
#                 int(float(sat.get('id')))
#                 for sat in station_elem.findall('basicSa')
#                 if sat.get('id') is not None
#             }
#
#             group_data[step]["groups"][assigned_gid].update(valid_ids)
#             group_data[step]["all_mentioned"].update(valid_ids)
#
#     return group_data
#

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Set

def parse_station_timeseries(
    xml_file: str | Path,
    station_ids: List[int],
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,   # 含 end_step
) -> List[Dict[int, Set[int]]]:
    """
    返回：list[ len(station_ids) ]，其中每个元素是 { step:int -> set(satellite_ids:int) }
         索引 i 对应 station_ids[i]。
    例如：
        series = parse_station_timeseries(xml, [0,4], 0, 100)
        series[0][1]  # station 0 在 step=1 的接入卫星集合
        series[1][1]  # station 4 在 step=1 的接入卫星集合
    若某一步该站未出现，仍会记录为空集合 set()，便于直接索引。
    """
    xml_file = str(xml_file)

    # 位置索引映射，保证按传入顺序组织结果
    id_to_idx: Dict[int, int] = {sid: i for i, sid in enumerate(station_ids)}
    want_set: Set[int] = set(station_ids)

    # 输出容器：每个站一个字典（step -> set(sats)）
    series: List[Dict[int, Set[int]]] = [dict() for _ in station_ids]

    context = ET.iterparse(xml_file, events=("end",))
    for event, elem in context:
        if elem.tag != "time":
            continue

        step_attr = elem.get("step")
        if step_attr is None:
            elem.clear(); continue
        try:
            step = int(step_attr)
        except ValueError:
            elem.clear(); continue

        # 时间窗口：含 end_step
        if start_step is not None and step < start_step:
            elem.clear(); continue
        if end_step is not None and step > end_step:
            elem.clear(); continue

        # 本 step 已填充的站（避免重复写入）
        filled_idx: Set[int] = set()

        stations_elem = elem.find("stations")
        if stations_elem is not None:
            for st in stations_elem.findall("station"):
                sid_attr = st.get("id")
                if sid_attr is None:
                    continue
                try:
                    sid = int(sid_attr)
                except ValueError:
                    continue

                if sid not in want_set:
                    continue

                idx = id_to_idx[sid]
                sats: Set[int] = set()

                for sat_elem in st:
                    if sat_elem.tag != "satellite":
                        continue
                    sat_id_attr = sat_elem.get("id")
                    if not sat_id_attr:
                        continue
                    # 兼容 "123" / "123.0"
                    try:
                        sat_id = int(sat_id_attr)
                    except ValueError:
                        try:
                            sat_id = int(float(sat_id_attr))
                        except ValueError:
                            continue
                    sats.add(sat_id)

                series[idx][step] = sats
                filled_idx.add(idx)

        # 对于本 step 未出现的站，补空集合，便于直接索引 series[i][step]
        if len(filled_idx) < len(series):
            for i in range(len(series)):
                if i not in filled_idx:
                    if step not in series[i]:
                        series[i][step] = set()

        elem.clear()

    return series



#
# def plot_grouped_satellites(group_data):
#     total_sats = N * P  # 卫星总数
#     sat_ids = np.arange(total_sats)
#     # 计算每个卫星在网格上的位置 (平面索引, 卫星索引)
#     all_cols = sat_ids // N  # X轴：轨道平面索引 (0 to P-1)
#     all_rows = sat_ids % N  # Y轴：卫星在平面内的索引 (0 to N-1)
#
#     fig, ax = plt.subplots(figsize=(14, 9))
#     plt.subplots_adjust(bottom=0.25)
#
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
#   #  ax.legend(handles=legend_handles, loc='upper right', title="Visible to Group")
#     # 调整图像布局
#     fig.subplots_adjust(right=0.80)  # 右边留出空间放图例
#
#     # 把图例放在图外右侧
#     ax.legend(
#         handles=legend_handles,
#         loc='center left',
#         bbox_to_anchor=(1.01, 0.5),
#         borderaxespad=0.0,
#         title="Visible to Group"
#     )
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
#          print("Error: No time steps found in XML paper_dataresult after parsing.")
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
#
# import math
#



