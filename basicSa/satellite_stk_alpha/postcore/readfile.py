import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json
from typing import List, Dict
import basicSa.utilis.Node as Node
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict

# @dataclass
class TimeSnapshot:
    """单个时间点的网络快照（完全按照需求定义）"""
    def __init__(self, timestamp_str, nodes_dict, active_paths_dict):
        self.timestamp = timestamp_str  # ISO 8601时间字符串
        self.nodes = nodes_dict         # 节点ID到节点对象的映射
        self.active_paths = active_paths_dict  # 路径字典

class TimeDatabase:
    """时间序列数据库"""
    def __init__(self):
        self.snapshots = []  # 存储TimeSnapshot对象的列表
        self.stationnum = 0
        self.satellitenum = 0
    # Implement __getitem__ to allow indexing
    def __getitem__(self, index):
        return self.snapshots[index]


def build_complete_paths(routegnd, route_dict):
    """基于地面站路由表和全局路由表构建完整路径"""
    complete_paths = {}

    # 创建路由表字典以提高查找速度 (basicSa -> {dest -> nexthop})
   # route_dict = {}
    # for route in routetables:
    #     if route['basicSa'] not in route_dict:
    #         route_dict[route['basicSa']] = {}
    #     route_dict[route['basicSa']][route['dest']] = route['nexthop']

    # 处理每个地面站发起的路由
    for ground_route in routegnd:
        src = ground_route['basicSa']
        dest = ground_route['dest']
        current_hop = ground_route['nexthop']

        # 初始化路径
        path = [src, current_hop]
        visited = set([src, current_hop])

        # 递归查找后续路径
        while current_hop != dest:
            # 查找当前节点的路由条目
            if current_hop in route_dict and dest in route_dict[current_hop]:
                next_hop = route_dict[current_hop][dest]

                # 环路检测
                if next_hop in visited:
                    print(f"路径环路检测: {path} -> {next_hop}")
                    break

                path.append(next_hop)
                visited.add(next_hop)
                current_hop = next_hop
            else:
                # 如果没有找到路由条目，则认为没有路径
                print(f"无法找到路径从 {current_hop} 到 {dest}")
                break

        # 存储最终路径
        if dest not in complete_paths:
            complete_paths[dest] = []
        complete_paths[dest].append({

            'path': path + ([dest] if path[-1] != dest else [])
        })

    return complete_paths





def readxml(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 初始化数据库
    db = TimeDatabase()
    # 初始化数据结构
    ip_to_node = {}

    node_types = {}  # 记录节点类型：'ground' 或 'basicSa'
    satellite_id_mapping = {}  # 原始卫星节点 ID 到调整后 ID 的映射

    # 解析 IPAddresses，建立 IP 到节点 ID 的映射，并识别地面节点和卫星节点
    ip_addresses = root.find('IPAddresses')

    stationnum = 0
    Nodesnum = 0

    for node in ip_addresses.findall('Node'):
        node_id = node.get('id')
        ips = [ip.text.strip() for ip in node.findall('IP')]
        ip_to_node.update({ip: node_id for ip in ips})
        Nodesnum+=1
        # 判断节点类型
        if len(ips) == 1:
            node_types[node_id] = 'ground'
            stationnum+=1
         #   station_counter+=1
        else:
            node_types[node_id] = 'basicSa'


    # 解析 TimeData，收集节点位置和链路信息
    for time_data in root.findall('TimeData'):

        nodes = [Node.Node for _ in range(Nodesnum)]

        time_str = time_data.get('time').strip()
        # 假设时间以秒为单位，从某个基准时间开始
        time_seconds = float(time_str.split()[0])
        # if(time_seconds>400):
        #     print("1")
        routetables = {}
        routegnd = []

        for node in time_data.findall('Node'):
            node_id = node.get('id')
            node_type = node_types.get(node_id, 'unknown')

            # 获取节点位置
            position_elem = node.find('Position')
            x = float(position_elem.get('x'))
            y = float(position_elem.get('y'))
            z = float(position_elem.get('z'))

# we set the node structure
            nodes[int(node_id)] = Node.Node(x=x, y=y, z=z)

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

                            if (target_node_type == 'ground' and node_type == 'basicSa') or \
                                (target_node_type == 'basicSa' and node_type == 'ground'):
                                # 处理星地链路（地面站 <-> 卫星）
                               ground_id = str(target_node_id) if target_node_type == 'ground' else str(node_id)
                               sat_id = str(node_id) if target_node_type == 'ground' else str(target_node_id)

                              # we absolutely need write the linktable
                               nodes[int(ground_id)].linked_array[0] = int(sat_id)


                            else:
                                #处理星间链路（卫星 <-> 卫星）

                                # we absolutely need write the linktable
                                for i in range(1,5):
                                    if nodes[int(node_id)].linked_array[i] ==-1:
                                        nodes[int(node_id)].linked_array[i] = int(target_node_id)
                                        break

                        else:
                            print(f"Warning: IP address {target_ip} not found in mapping.")
                    # 如果 target_ip 为 '0'，表示没有链接，不需要处理

            route_elem = node.find('Routes')
            if route_elem is not None:
                for route in route_elem.findall('Route'):
                    route_text = route.text.strip()

                    # Split the route text by the '->' separator
                    dest_ip, next_ip = route_text.split('->')
                    dest_id = ip_to_node.get(dest_ip)


                    next_id = ip_to_node.get(next_ip)



                    if node_type == 'ground':


                        routetable = { "dest": int(dest_id),"basicSa": int(node_id),"nexthop": int(next_id)}
                        routegnd.append(routetable)
                    else:

                        if int(node_id) not in routetables:
                            routetables[int(node_id) ] = {}

                        routetables[int(node_id)][int(dest_id)] = int(next_id)

        # 在原有代码的最后部分替换为：
        complete_paths = build_complete_paths(routegnd, routetables)



                    # 创建时间快照
        snapshot = TimeSnapshot(
            timestamp_str=time_seconds,
            nodes_dict=nodes,
            active_paths_dict=complete_paths
        )

        # 添加到数据库
        db.snapshots.append(snapshot)
        if not db.stationnum :
            db.stationnum  =stationnum
            db.satellitenum = Nodesnum-stationnum
    return db




#
#
# if __name__ == "__main__":
#     # 读取并解析数据
#     xml_file = '/home/yfh/Desktop/Data/basicSa.xml'  # 您的 XML 数据文件路径
#     db = parse_ns3_data(xml_file)
#
#     # 生成 CZML 内容
#    # czml_content = generate_czml(node_positions, link_intervals, node_types, satellite_id_mapping)
#
#     # 保存到文件
#     output_file = '/home/yfh/Desktop/NS3/manswn/Cesium_module/Cesium/satellite_trajectory.czml'  # 输出文件路径
#   #  save_to_czml_file(czml_content, output_file)
#
#     print(f"CZML 文件已生成并保存到 {output_file}")
