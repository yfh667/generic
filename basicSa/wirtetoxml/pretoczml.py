# 预处理阶段，python的卫星轨道拓扑cesium可视化


import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json

def parse_ns3_data(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 获取地面站和卫星的数量
    num_stations = int(root.get('stations', '0'))
    num_satellites = int(root.get('satellites', '0'))

    # 初始化数据结构
    node_positions = {}
    link_intervals = {}
    node_types = {}  # 记录节点类型：'ground' 或 'basicSa'
    satellite_id_mapping = {}  # 原始卫星节点 ID 到调整后 ID 的映射

    # 根据节点 ID 分配节点类型
    for node_id in range(num_stations + num_satellites):
        node_id_str = str(node_id)
        if node_id < num_stations:
            node_types[node_id_str] = 'ground'
        else:
            node_types[node_id_str] = 'basicSa'
            satellite_id_mapping[node_id_str] = node_id - num_stations  # 调整卫星编号，使其从 0 开始

    # 解析时间数据
    for time_data in root.findall('Time'):
        time_value = time_data.get('value').strip()
        time_seconds = float(time_value)
        # 将时间转换为 ISO 8601 格式
        time_iso = (datetime(2024, 11, 12) + timedelta(seconds=time_seconds)).isoformat() + 'Z'

        for node in time_data.findall('Node'):
            node_id = node.get('id')
            node_type = node_types.get(node_id, 'unknown')

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

            # 获取链路信息
            links_elem = node.find('LinkArray')
            if links_elem is not None:
                link_indices = [link.text.strip() for link in links_elem.findall('Link')]
                for target_node_id in link_indices:
                    if target_node_id != '-1':
                        # 确保链接的节点 ID 是字符串
                        node_pair = tuple(sorted([str(node_id), str(target_node_id)]))
                        if node_pair not in link_intervals:
                            link_intervals[node_pair] = []
                        # 添加链路的激活时间
                        link_intervals[node_pair].append(time_iso)
        # print("")

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
        node_type = node_data['type']
        positions = node_data['positions']

        # 构建 position 的 cartesian 数据
        if node_type == 'ground':
            # 地面节点，位置固定
            pos = positions[0]
            x, y, z = pos['cartesian']
            ground_packet = {
                "id": f"ground-{node_id_str}",
                "name": f"Ground Station {node_id_str}",
                "description": f"Ground Station {node_id_str}",
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
                    "text": f"Ground {node_id_str}",
                    "font": "10pt Lucida Console",
                    "fillColor": {
                        "rgba": [255, 255, 255, 255]
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
                "model": {
                    "gltf": "simple_satellite_low_poly_free.glb",  # 您的卫星模型文件
                    "scale": 5e-23,  # 您的缩放比例
                    "minimumPixelSize": 20
                },
                "label": {
                    "horizontalOrigin": "LEFT",
                    "show": True,
                    "pixelOffset": {
                        "cartesian2": [12, 0]
                    },
                    "outlineWidth": 2,
                    "font": "10px Lucida Console",
                    "text": f"Satellite {adjusted_id}",  # 确保卫星编号从 0 开始
                    "outlineColor": {
                        "rgba": [0, 0, 0, 255]
                    },
                    "fillColor": {
                        "rgba": [213, 255, 0, 255]
                    }
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
            print(f"Unknown node type for node {node_id_str}")

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
            from_node_id = from_node_str

        if to_node_type == 'basicSa':
            to_prefix = 'basicSa'
            to_node_id = satellite_id_mapping[to_node_str]
        else:
            to_prefix = 'ground'
            to_node_id = to_node_str

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

        availability_interval = []
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

