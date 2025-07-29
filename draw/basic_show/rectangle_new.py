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

def cyclic_width(ys, N):
    if not ys:
        return 0
    ys = sorted(set(ys))
    gaps = []
    for i in range(1, len(ys)):
        gaps.append(ys[i] - ys[i-1])
    # 跨越环首尾
    gaps.append((ys[0] + N) - ys[-1])
    maxgap = max(gaps)
    return N - maxgap + 1

def fanxiangfeng(sats, P, N):
    xs = [sid // N for sid in sats]
    ys = [sid % N for sid in sats]

    left = min(xs)
    right = max(xs)
    block_info = [(None, None), (None, None)]

    # 判断是否两块
    if left == 0 and right == P - 1:
        # --------左侧块--------
        leftgroup = []
        x = left
        while x in xs:
            leftgroup.append(x)
            x += 1
        # 统计左块包络
        leftys = [sid % N for sid in sats if (sid // N) in leftgroup]
        left_width = cyclic_width(leftys, N)
        left_length = leftgroup[-1] - leftgroup[0] + 1 if leftgroup else 0
        block_info[0] = (left_length, left_width)

        # --------右侧块--------
        rightgroup = []
        x = right
        while x in xs:
            rightgroup.append(x)
            x -= 1
        rightys = [sid % N for sid in sats if (sid // N) in rightgroup]
        right_width = cyclic_width(rightys, N)
        right_length = rightgroup[0] - rightgroup[-1] + 1 if rightgroup else 0
        block_info[1] = (right_length, right_width)

    else:
        # 只有一块，取所有点
        w = max(xs) - min(xs) + 1
        h = cyclic_width(ys, N)
        block_info[0] = (w, h)
        block_info[1] = (None, None)

    return block_info  # [(块1长, 宽), (块2长, 宽)]



# --- 主程序 ---
if __name__ == "__main__":
   # xml_file =  "E:\Data\station_visible_satellites_648.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>
    xml_file =  "E:\Data\station_visible_satellites_648_1d_real.xml"  # <<<<<<< 请替换为你的XML文件路径 >>>>>>>

   # dummy_file_name =
   # dummy_file_name =

    start_ts = 1
    # #  end_ts = 21431
    end_ts = 86399
    # start_ts = 1202
    # end_ts = 3320
    # start_ts = 53232
    # end_ts = 55574
    # start_ts = 1
    # end_ts = 6733
    try:
        # 解析XML数据
        group_data = read_snap_xml.parse_xml_group_data(xml_file, start_ts, end_ts)
       # group_data = modify_group_data(group_data)
        if not group_data:
            print(f"Error: No valid group visibility data parsed from {xml_file}.")
            print("Please check if the XML file exists and contains 'time' elements with 'stations' and 'satellite' data.")
            sys.exit(1) # 退出程序如果解析失败或没有数据

        # 绘制可视化图

        # 假设 group_data 已经准备好，键是时间戳，值是包含 'groups' 子字典
        times = sorted(group_data.keys())


        groupid = 4
        # --- 主程序修改 ---
        widths1, heights1, widths2, heights2 = [], [], [], []
        for t in times:
            sats4 = group_data[t]['groups'][groupid]
            (l1, w1), (l2, w2) = fanxiangfeng(sats4,   P,N)
            widths1.append(l1)
            heights1.append(w1)
            widths2.append(l2)
            heights2.append(w2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        # 上面是Block1
        ax1.plot(times, widths1, label='Block1 Length', color='tab:blue')
        ax1.plot(times, heights1, label='Block1 Width', color='tab:orange')
        ax1.set_ylabel('Block1 Size')
        ax1.set_title('Block1（一般为主包络）长宽随时间变化')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 下面是Block2
        ax2.plot(times, widths2, label='Block2 Length', color='tab:green')
        ax2.plot(times, heights2, label='Block2 Width', color='tab:red')
        ax2.set_ylabel('Block2 Size')
        ax2.set_title('Block2（有时不存在）长宽随时间变化')
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.xlabel('Time Step')
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