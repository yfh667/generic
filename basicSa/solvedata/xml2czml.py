import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json
#这个模块主要是将ns-3产生的satellite.xml文件转为czml文件。
#用于后续的cesiun显示
#
#
#
#

def parse_ns3_data(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 初始化数据结构
    ip_to_node = {}
    node_positions = {}
    link_intervals = {}
    node_types = {}  # 记录节点类型：'ground' 或 'basicSa'
    satellite_id_mapping = {}  # 原始卫星节点 ID 到调整后 ID 的映射

    # 解析 IPAddresses，建立 IP 到节点 ID 的映射，并识别地面节点和卫星节点
    ip_addresses = root.find('IPAddresses')
    satellite_counter = 0  # 用于给卫星编号，从 0 开始
    for node in ip_addresses.findall('Node'):
        node_id = node.get('id')
        ips = [ip.text.strip() for ip in node.findall('IP')]
        ip_to_node.update({ip: node_id for ip in ips})

        # 判断节点类型
        if len(ips) == 1:
            node_types[node_id] = 'ground'
        else:
            node_types[node_id] = 'basicSa'
            satellite_id_mapping[node_id] = satellite_counter
            satellite_counter += 1

    # 解析 TimeData，收集节点位置和链路信息
    for time_data in root.findall('TimeData'):
        time_str = time_data.get('time').strip()
        # 假设时间以秒为单位，从某个基准时间开始
        time_seconds = float(time_str.split()[0])
        # 将时间转换为 ISO 8601 格式
        if time_seconds ==73:
            print("yes")

        time_iso = (datetime(2024, 11, 12) + timedelta(seconds=time_seconds)).isoformat() + 'Z'
        # if(time_str =='389 s'):
        #     print("ere")
        for node in time_data.findall('Node'):
            node_id = node.get('id')
            node_type = node_types.get(node_id, 'unknown')
            if node_id=='63':
                print('yes')
            # 获取节点位置
            position_elem = node.find('Position')
            x = float(position_elem.get('x'))
            y = float(position_elem.get('y'))
            z = float(position_elem.get('z'))

            # 初始化节点的位置列表
            if node_id not in node_positions:
                node_positions[node_id] = {'type': node_type, 'positions': []}

            # 添加当前位置
            node_positions[node_id]['positions'].append({'time': time_iso, 'cartesian': [x, y, z]})
            # if(node_id=='2'):
            #     print("here")



            # 获取链路信息
            links_elem = node.find('Links')
            if links_elem is not None:
                for link in links_elem.findall('Link'):
                    target_ip = link.text.strip()
                    if target_ip != '0':
                        # 获取目标节点 ID
                        target_node_id = ip_to_node.get(target_ip)
                        target_node_type = node_types.get(target_node_id, 'unknown')


                        if target_node_id is not None:

                            # node_pair = tuple(sorted([str(node_id), str(target_node_id)]))
                            # if node_pair not in link_intervals:
                            #     link_intervals[node_pair] = []

                            if (target_node_type == 'ground' and node_type == 'basicSa') or \
                                (target_node_type == 'basicSa' and node_type == 'ground'):
                                # 处理星地链路（地面站 <-> 卫星）
                                ground_id = str(target_node_id) if target_node_type == 'ground' else str(node_id)
                                sat_id = str(node_id) if target_node_type == 'ground' else str(target_node_id)
                                node_pair = (ground_id, sat_id)  # 保持有序以区分方向

                            elif target_node_type == 'basicSa' and node_type == 'basicSa':
                                # 处理星间链路（卫星 <-> 卫星）
                                node_pair = tuple(sorted([str(node_id), str(target_node_id)]))
                            else:
                                # 地面站之间的连接（如果存在）
                                node_pair = tuple(sorted([str(node_id), str(target_node_id)]))

                            if node_pair not in link_intervals:
                                link_intervals[node_pair] = []
                            link_intervals[node_pair].append(time_iso)


                        else:
                            print(f"Warning: IP address {target_ip} not found in mapping.")
                    # 如果 target_ip 为 '0'，表示没有链接，不需要处理

    return node_positions, link_intervals, node_types, satellite_id_mapping

def generate_czml(node_positions, link_intervals, node_types, satellite_id_mapping):
    # 定义 CZML 头部信息，包括时间范围
    all_times = set()
    for node_data in node_positions.values():
        for pos in node_data['positions']:
            all_times.add(pos['time'])
    sorted_times = sorted(all_times)
    start_time = sorted_times[0]
    end_time = sorted_times[-1]

    czml = [
        {
            "id": "document",
            "version": "1.0",
            "clock": {
                "interval": f"{start_time}/{end_time}",
                "currentTime": start_time,
                "multiplier": 1,
                "range": "LOOP_STOP",
                "step": "SYSTEM_CLOCK_MULTIPLIER"
            }
        }
    ]

    # 添加节点实体（包括地面节点和卫星）
    for node_id_str, node_data in node_positions.items():
        node_id = int(node_id_str)  # 将节点 ID 转换为整数
        node_type = node_data['type']
        positions = node_data['positions']

        # 构建 position 的 cartesian 数据
        if node_type == 'ground':
            # 地面节点，位置固定
            pos = positions[0]
            x, y, z = pos['cartesian']
            ground_packet = {
                "id": f"ground-{node_id}",
                "name": f"Ground Station {node_id}",
                "description": f"Ground Station {node_id}",
                "availability": f"{start_time}/{end_time}",
                "position": {
                    "cartesian": [x, y, z],
                    "referenceFrame": "FIXED"  # 使用地固参考系
                },
                "point": {  # 使用简单的点来表示地面节点
                    "pixelSize": 10,
                    "color": {"rgba": [0, 255, 0, 255]},
                    "outlineWidth": 1,
                    "outlineColor": {"rgba": [0, 0, 0, 255]}
                },



                "label": {
                    "text": f"Ground {node_id}",
                    "font": "10pt Lucida Console",
                    "fillColor": {
                        "rgba": [0, 0, 0, 255]
                    },
                    "outlineColor": {
                        "rgba": [0, 0, 0, 255]
                    },
                    "outlineWidth": 2,
                    "style": "FILL_AND_OUTLINE",
                    "horizontalOrigin": "LEFT",
                    "pixelOffset": {
                        "cartesian2": [12, 0]
                    },
                    "show": True
                }
            }
            czml.append(ground_packet)
        elif node_type == 'basicSa':
            # 卫星节点，位置随时间变化
            epoch = positions[0]['time']
            cartesian = []
            for pos in positions:
                time_offset = (datetime.fromisoformat(pos['time'].rstrip('Z')) - datetime.fromisoformat(
                    epoch.rstrip('Z'))).total_seconds()
                cartesian.extend([time_offset, *pos['cartesian']])

            # 获取调整后的卫星 ID
            adjusted_id = satellite_id_mapping[node_id_str]

            satellite_packet = {
                "id": f"basicSa-{adjusted_id}",
                "description": f"Orbit of Satellite {adjusted_id}",
                "availability": f"{positions[0]['time']}/{positions[-1]['time']}",
                "point": {
                             "color": {
                                 "rgba": [0, 0, 0, 255]
                             },
                             "pixelSize": 10,
                "outlineColor": {
                    "rgba": [255, 255, 255, 255]
                },
                "outlineWidth": 1
                },

                "label": {
                    "horizontalOrigin": "LEFT",
                    "show": True,
                    "pixelOffset": {
                        "cartesian2": [12, 0]
                    },
                    "font": "10px Lucida Console",
                    "text": f"Satellite {adjusted_id}",  # 确保标签文本为纯文本
                    "fillColor": {
                        "rgba": [0, 0, 0, 255]  # 标签字体颜色为黑色
                    },
                    "outlineWidth": 0  # 确保没有轮廓
                },


                "position": {
                    "epoch": epoch,
                    "cartesian": cartesian,
                    "interpolationAlgorithm": "LAGRANGE",
                    "interpolationDegree": 1,
                    "referenceFrame": "FIXED"  # 使用地固参考系
                }
            }
            czml.append(satellite_packet)
        else:
            print(f"Unknown node type for node {node_id}")

    # 添加链路实体
    for node_pair, times in link_intervals.items():
        from_node_str, to_node_str = node_pair
        from_node_type = node_types.get(from_node_str, 'unknown')
        to_node_type = node_types.get(to_node_str, 'unknown')

        # 确定实体的 id 前缀和节点 ID
        if from_node_type == 'basicSa':
            from_prefix = 'basicSa'
            from_node_id = satellite_id_mapping[from_node_str]
        else:
            from_prefix = 'ground'
            from_node_id = int(from_node_str)

        if to_node_type == 'basicSa':
            to_prefix = 'basicSa'
            to_node_id = satellite_id_mapping[to_node_str]
        else:
            to_prefix = 'ground'
            to_node_id = int(to_node_str)

        link_id = f"link-{from_prefix}-{from_node_id}-{to_prefix}-{to_node_id}"

        # 根据时间列表，生成时间间隔
        times = sorted(times)
        intervals = []
        start = times[0]
        end = times[0]
        for time in times[1:]:
            previous_time = datetime.fromisoformat(end.rstrip('Z'))
            current_time = datetime.fromisoformat(time.rstrip('Z'))
            # 如果时间连续，更新结束时间
            if (current_time - previous_time).total_seconds() <= 1:
                end = time
            else:
                # 添加之前的时间间隔
                intervals.append({"basicSa": start, "end": end})
                start = time
                end = time
        intervals.append({"basicSa": start, "end": end})
        availability_interval  = []
        # 构建 show 属性的时间间隔
        show_intervals = []
        for interval in intervals:
            show_intervals.append({
                "interval": f"{interval['basicSa']}/{interval['end']}",
                "boolean": True
            })
            availability_interval.append(f"{interval['basicSa']}/{interval['end']}")


        link_packet = {
            "id": link_id,
            "availability": ", ".join(availability_interval),  # 将多个区间合并成字符串
            "polyline": {
                "positions": {
                    "references": [
                        f"{from_prefix}-{from_node_id}#position",
                        f"{to_prefix}-{to_node_id}#position"
                    ]
                },
                "material": {
                    "solidColor": {
                        "color": {"rgba": [255, 0, 0, 255]}
                    }
                },
                "width": 2,
                "arcType": "NONE",  # 确保链路为直线
                "show": show_intervals
            }
        }
        czml.append(link_packet)

    # 将 CZML 数据转换为 JSON 格式
    czml_json = json.dumps(czml, indent=2)
    return czml_json

def save_to_czml_file(czml_content, filename="output.czml"):
    with open(filename, "w") as file:
        file.write(czml_content)

if __name__ == "__main__":
    # 读取并解析数据
    xml_file = '/home/yfh/Desktop/Data/basicSa.xml'  # 您的 XML 数据文件路径
    node_positions, link_intervals, node_types, satellite_id_mapping = parse_ns3_data(xml_file)

    # 生成 CZML 内容
    czml_content = generate_czml(node_positions, link_intervals, node_types, satellite_id_mapping)

    # 保存到文件
    output_file = '/home/yfh/Desktop/NS3/manswn/Cesium_module/Cesium/satellite_trajectory.czml'  # 输出文件路径
    save_to_czml_file(czml_content, output_file)

    print(f"CZML 文件已生成并保存到 {output_file}")
