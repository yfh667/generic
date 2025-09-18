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


# def _parse_tuple_int(s: str):
#     """把 '1,2,3' 或 '(1, 2, 3)' 转成 tuple[int,...]；'None'/空 返回 None。"""
#     if not s or s == 'None':
#         return None
#     s = s.strip()
#     if s and s[0] == '(' and s[-1] == ')':
#         s = s[1:-1]
#     # 更轻量，避免 ast.literal_eval
#     return tuple(int(p.strip()) for p in s.split(','))

def _parse_tuple_int(s: str):
    """把 '1,2,3' 或 '(1, 2, 3)' 转成 tuple[int,...]；'None'/空 -> None（比 ast.literal_eval 更快）。"""
    if not s or s == 'None':
        return None
    s = s.strip()
    if s and s[0] == '(' and s[-1] == ')':
        s = s[1:-1]
    # 允许有空格
    parts = s.split(',')
    # 过滤空片段，防止 '1,2,' 之类
    return tuple(int(p.strip()) for p in parts if p.strip())

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



def nodes_to_xml2(nodes, filename):
    """
    将 nodes 字典保存为 XML 文件
    nodes: dict[(x, y, step)] -> tegnode
    filename: 保存的xml路径
    """
    root = ET.Element("Nodes")
    for coord, node in nodes.items():
        node_elem = ET.SubElement(root, "Node")
        node_elem.set("coordination", f"{coord[0]},{coord[1]},{coord[2]}")
        node_elem.set("asc_nodes_region_id", str(node.asc_nodes_region_id))
        node_elem.set("rightneighbor", str(node.rightneighbor) if node.rightneighbor is not None else "None")
        node_elem.set("leftneighbor", str(node.leftneighbor) if node.leftneighbor is not None else "None")
        node_elem.set("left_state", str(node.left_state))
        node_elem.set("right_state", str(node.right_state))

    # 格式化输出
    xml_str = ET.tostring(root, encoding='utf-8')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)



def xml_to_nodes2(filename, tegnode_cls):
    """
    更快的 XML -> nodes(dict) 读取：
      - ET.iterparse 流式解析（边读边清理，省内存、快）
      - 用 _parse_tuple_int() 替代 ast.literal_eval()
    返回: dict[(x, y, step)] -> tegnode
    """
    nodes = {}
    try:
        # 只监听元素结束事件；到 </Node> 时处理该节点
        for event, elem in ET.iterparse(filename, events=('end',)):
            if elem.tag != 'Node':
                continue

            at = elem.attrib  # 局部绑定，少一次查找
            coord_str = at.get('coordination')
            if not coord_str:   # 缺关键字段就跳过
                elem.clear()
                continue

            # 坐标
            coords = tuple(int(x.strip()) for x in coord_str.split(','))

            # 邻居（更轻量）
            rightneighbor = _parse_tuple_int(at.get('rightneighbor'))
            leftneighbor  = _parse_tuple_int(at.get('leftneighbor'))

            # 其他属性（沿用你原来的字段映射/默认值）
            asc_nodes_region_id = (at.get('asc_nodes_region_id') == 'True')
            left_state  = int(at.get('left_state',  '-1'))
            right_state = int(at.get('right_state', '0'))

            # 构造节点对象（保持你原来的入参名）
            node = tegnode_cls(
                asc_nodes_flag=asc_nodes_region_id,
                rightneighbor=rightneighbor,
                leftneighbor=leftneighbor,
                state=left_state,
                importance=right_state,
            )
            nodes[coords] = node

            # 关键：清理已处理元素，释放内存（大文件明显加速）
            elem.clear()

    except Exception as e:
        print(f"解析XML时出错: {e}")
        return None

    return nodes
# def xml_to_nodes2(filename, tegnode_cls):
#     """
#     从 XML 文件读取 nodes 字典
#     Args:
#         filename: XML 文件路径
#         tegnode_cls: 你的 tegnode 类
#     Returns:
#         nodes: dict[(x, y, step)] -> tegnode
#     """
#     nodes = {}
#     try:
#         tree = ET.parse(filename)
#         root = tree.getroot()
#         for node_elem in root.findall('Node'):
#             # 1. 解析坐标
#             coord_str = node_elem.get('coordination')
#             coords = tuple(map(int, coord_str.split(',')))
#             # 2. 解析右邻居
#             right_str = node_elem.get('rightneighbor')
#             rightneighbor = None
#             if right_str != 'None':
#                 try:
#                     rightneighbor = ast.literal_eval(right_str)
#                 except Exception:
#                     # 没括号时直接按逗号切分
#                     rightneighbor = tuple(map(int, right_str.split(',')))
#             # 3. 解析左邻居
#             left_str = node_elem.get('leftneighbor')
#             leftneighbor = None
#             if left_str != 'None':
#                 try:
#                     leftneighbor = ast.literal_eval(left_str)
#                 except Exception:
#                     leftneighbor = tuple(map(int, left_str.split(',')))
#             # 4. 其他属性
#             asc_nodes_region_id = node_elem.get('asc_nodes_region_id') == 'True'
#             left_state = int(node_elem.get('left_state', -1))
#             right_state = int(node_elem.get('right_state', 0))
#             # 5. 构造节点对象
#             node = tegnode_cls(
#                 asc_nodes_flag=asc_nodes_region_id,
#                 rightneighbor=rightneighbor,
#                 leftneighbor=leftneighbor,
#                 state=left_state,
#                 importance=right_state
#             )
#             nodes[coords] = node
#     except Exception as e:
#         print(f"解析XML时出错: {e}")
#         return None
#     return nodes

if __name__ == '__main__':

    # 保存
    #nodes_to_xml(nodes, "test_nodes.xml")

    # 读取    dummy_file_name = "E:\\code\\dataresult\\station_visible_satellites_100_test.xml"
    nodes_loaded = xml_to_nodes("test_nodes.xml", tegnode.tegnode)

     # print(nodes)
