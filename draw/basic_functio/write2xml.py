import xml.etree.ElementTree as ET
import re
from xml.dom import minidom
import os
import ast

import genaric2.tegnode as tegnode
def nodes_to_xml(nodes, filename):
    """
    将 nodes 字典保存为 XML 文件
    nodes: dict[(x, y, step)] -> tegnode
    filename: 保存的xml路径
    """
    root = ET.Element("Nodes")
    for coord, node in nodes.items():
        node_elem = ET.SubElement(root, "Node")
        node_elem.set("coordination", f"{coord[0]},{coord[1]},{coord[2]}")
        node_elem.set("asc_nodes_flag", str(node.asc_nodes_flag))
        node_elem.set("rightneighbor", str(node.rightneighbor) if node.rightneighbor is not None else "None")
        node_elem.set("leftneighbor", str(node.leftneighbor) if node.leftneighbor is not None else "None")
        node_elem.set("state", str(node.state))
        node_elem.set("importance", str(node.importance))

    # 格式化输出
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def xml_to_nodes(filename, tegnode_cls):
    """
    从 XML 文件读取 nodes 字典
    Args:
        filename: XML 文件路径
        tegnode_cls: 你的 tegnode 类
    Returns:
        nodes: dict[(x, y, step)] -> tegnode
    """
    nodes = {}
    try:
        tree = ET.parse(filename)
        root = tree.getroot()
        for node_elem in root.findall('Node'):
            # 1. 解析坐标
            coord_str = node_elem.get('coordination')
            coords = tuple(map(int, coord_str.split(',')))
            # 2. 解析右邻居
            right_str = node_elem.get('rightneighbor')
            rightneighbor = None
            if right_str != 'None':
                try:
                    rightneighbor = ast.literal_eval(right_str)
                except Exception:
                    # 没括号时直接按逗号切分
                    rightneighbor = tuple(map(int, right_str.split(',')))
            # 3. 解析左邻居
            left_str = node_elem.get('leftneighbor')
            leftneighbor = None
            if left_str != 'None':
                try:
                    leftneighbor = ast.literal_eval(left_str)
                except Exception:
                    leftneighbor = tuple(map(int, left_str.split(',')))
            # 4. 其他属性
            asc_nodes_flag = node_elem.get('asc_nodes_flag') == 'True'
            state = int(node_elem.get('state', -1))
            importance = int(node_elem.get('importance', 0))
            # 5. 构造节点对象
            node = tegnode_cls(
                asc_nodes_flag=asc_nodes_flag,
                rightneighbor=rightneighbor,
                leftneighbor=leftneighbor,
                state=state,
                importance=importance
            )
            nodes[coords] = node
    except Exception as e:
        print(f"解析XML时出错: {e}")
        return None
    return nodes
if __name__ == '__main__':

    # 保存
    #nodes_to_xml(nodes, "test_nodes.xml")

    # 读取    dummy_file_name = "E:\\code\\data\\station_visible_satellites_100_test.xml"
    nodes_loaded = xml_to_nodes("test_nodes.xml", tegnode.tegnode)

     # print(nodes)
