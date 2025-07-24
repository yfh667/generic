# -*- coding: utf-8 -*-
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
import xml.etree.ElementTree as ET
import matplotlib.colors as mcolors
import sys
import  math
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

def modify_group_data(group_data, N=36):
    new_group_data = {}

    for step, raw_step_dict in group_data.items():   # ← 改这里
        raw_groups = raw_step_dict['groups']
        new_group_data[step] = {'groups': {}, 'all_mentioned': set()}

        # 1. group 4 中最大 y
        group4_sats = raw_groups.get(4, set())
        y_up = max((sid % N for sid in group4_sats), default=0)
        y_down = min((sid % N for sid in group4_sats), default=0)
        if step==6004:
            print(1)
        if y_down!=0:
            offset = N - y_up - 1

            for gid, sats in raw_groups.items():
                tgt_set = new_group_data[step]['groups'].setdefault(gid, set())
                for sid in sats:
                    y = sid % N
                    x = sid // N
                    y_new = y - y_up - 1 if y > y_up else y + offset
                    new_sid = x * N + y_new
                    tgt_set.add(new_sid)
                    new_group_data[step]['all_mentioned'].add(new_sid)
        else:# here we need sove the down's hights
            max1 = -math.inf

            offset=0
            for sid in group4_sats:
                y = sid % N
                x = sid // N
                here= y-5
                if here<=0:
                    if here>max1:
                        max1 = here
            y_up=max1+5


            offset = N - y_up - 1
            for gid, sats in raw_groups.items():
                tgt_set = new_group_data[step]['groups'].setdefault(gid, set())
                for sid in sats:
                    y = sid % N
                    x = sid // N
                    y_new = y - y_up - 1 if y > y_up else y + offset
                    new_sid = x * N + y_new
                    tgt_set.add(new_sid)
                    new_group_data[step]['all_mentioned'].add(new_sid)



    return new_group_data





# --- 主程序 ---
if __name__ == "__main__":
   # xml_file =  "E:\Data\station_visible_satellites_648.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>
    xml_file =  "E:\Data\station_visible_satellites_648_1d_real.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>

   # dummy_file_name =
   # dummy_file_name =
    start_ts = 1
    end_ts = 6733
    try:
        # 解析XML数据
        group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
        group_data = modify_group_data(group_data)
        if not group_data:
            print(f"Error: No valid group visibility data parsed from {xml_file}.")
            print("Please check if the XML file exists and contains 'time' elements with 'stations' and 'satellite' data.")
            sys.exit(1) # 退出程序如果解析失败或没有数据

        # 绘制可视化图

        # 假设 group_data 已经准备好，键是时间戳，值是包含 'groups' 子字典
        times = sorted(group_data.keys())

        widths = []
        heights = []
        groupid = 0
        for t in times:
            # 拿到这一步的 Group 4 所有卫星 ID
            sats4 = group_data[t]['groups'][groupid]
            # 转成 x,y 坐标
            xs = [sid // N for sid in sats4]
            ys = [sid % N for sid in sats4]
            # 计算宽、高
            w = max(xs) - min(xs) + 1
            h = max(ys) - min(ys) + 1
            widths.append(w)
            heights.append(h)

        # 绘图
        plt.figure(figsize=(8, 4))
        plt.plot(times, widths, label='Width (Δx)')
        plt.plot(times, heights, label='Height (Δy)')
        plt.xlabel('Time Step')
        plt.ylabel('Size (grid units)')
        plt.title('Group 4 Bounding Box Size Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
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