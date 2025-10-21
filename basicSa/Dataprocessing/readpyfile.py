import xml.etree.ElementTree as ET
from collections import defaultdict

import basicSa.utilis.Node as Node


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
        self.simulation_time = 0
    # Implement __getitem__ to allow indexing
    def __getitem__(self, index):
        return self.snapshots[index]





def readxml(xml_file):
    # 解析 XML 文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # 初始化数据库
    db = TimeDatabase()


    # Parse metadata
    metadata = root.find('metadata')
    if metadata is not None:
        stationnum_elem = metadata.find('stationnum')
        if stationnum_elem is not None:
            db.stationnum = int(stationnum_elem.text)

        satellitenum_elem = metadata.find('satellite_count')

        if satellitenum_elem is not None:
            db.satellitenum = int(satellitenum_elem.text)

        sim_time_elem = metadata.find('simulation_time')
        if sim_time_elem is not None:
            db.simulation_time = int(sim_time_elem.text)


    Nodesnum =  db.stationnum+db.satellitenum

    # 解析 TimeData，收集节点位置和链路信息
    for time_data in root.findall('TimeData'):



        time_seconds = int(time_data.get('value'))


        # 假设时间以秒为单位，从某个基准时间开始
      #  time_seconds = float(time_str.split()[0])

        nodes = [Node.Node for _ in range(Nodesnum)]
        paths = defaultdict(list)  # 现在用 list 存储，格式如 { end_node: [{'path': path_list}, ...] }

        # Parse nodes
        nodes_elem = time_data.find('nodes')
        if nodes_elem is not None:
            for node_elem in nodes_elem.findall('node'):
                node_id = int(node_elem.get('id'))
                pos_elem = node_elem.find('Position')

                x = float(pos_elem.get('x'))
                y = float(pos_elem.get('y'))
                z = float(pos_elem.get('z'))

                # Create node object (adjust according to your Node class)
                node = Node.Node(x=x, y=y, z=z)

                # Ensure nodes list is large enough
                # while len(nodes) <= node_id:
                #     nodes.append(None)
                nodes[node_id] = node

        # Parse paths
        paths_elem = time_data.find('paths')

        if paths_elem is not None:
            for path_elem in paths_elem.findall('path'):
                path_id = int(path_elem.get('id'))
                path_str = path_elem.text.strip()

                # Convert "[0,1,2,3]" to [0, 1, 2, 3]
                path = list(map(int, path_str[1:-1].split(',')))
                end_node = path[-1]  # 关键点：用 end_node 作为 key

                paths[end_node].append({"path": path})  # 格式转换

                    # 创建时间快照
        snapshot = TimeSnapshot(
            timestamp_str=time_seconds,
            nodes_dict=nodes,
            active_paths_dict=paths
        )

        # 添加到数据库
        db.snapshots.append(snapshot)
        print(f"reading time {time_seconds}")
        # if not db.stationnum :
        #     db.stationnum  =stationnum
        #     db.satellitenum = Nodesnum-stationnum
    return db




#
#
if __name__ == "__main__":
    # 读取并解析数据
    xml_file = '/home/yfh/Desktop/Data/simulation_paths.xml'  # 您的 XML 数据文件路径
    db = readxml(xml_file)

    # 生成 CZML 内容
   # czml_content = generate_czml(node_positions, link_intervals, node_types, satellite_id_mapping)

    # 保存到文件
    output_file = '/home/yfh/Desktop/NS3/manswn/Cesium_module/Cesium/satellite_trajectory.czml'  # 输出文件路径
  #  save_to_czml_file(czml_content, output_file)

    print(f"CZML 文件已生成并保存到 {output_file}")
