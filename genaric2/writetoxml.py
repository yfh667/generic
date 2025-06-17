import xml.etree.ElementTree as ET
from xml.dom import minidom
import genaric2.tegnode as tegnode

def nodes_to_xml(nodes_data, filename):
    """
    将节点数据保存为XML文件

    参数:
        nodes_data: 节点数据列表，格式如示例
        filename: 要保存的XML文件名
    """
    # 创建根元素
    root = ET.Element("Nodes")

    # 添加每个节点
    for coords, node in nodes_data.items():
        # print(f"{attr}: {value}")
   # for coords, node in nodes_data:
        node_elem = ET.SubElement(root, "Node")

        # 添加坐标属性
        node_elem.set("cordination", coords)


        # 添加节点属性
        node_elem.set("asc_nodes_flag", str(node.asc_nodes_flag))
        node_elem.set("rightneighbor", str(node.rightneighbor))
        node_elem.set("leftneighbor", str(node.leftneighbor))
        node_elem.set("state", str(node.state))

    # 生成XML字符串
    xml_str = ET.tostring(root, encoding='utf-8')

    # 格式化XML
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


import xml.etree.ElementTree as ET
from collections import defaultdict


def xml_to_nodes(filename):
    """
    从XML文件读取节点数据，返回nodes字典

    参数:
        filename: XML文件名

    返回:
        nodes: 字典格式为 {(x,y,z): tegnode对象}
    """
    # 初始化nodes字典
    nodes = {}

    try:
        # 解析XML文件
        tree = ET.parse(filename)
        root = tree.getroot()

        # 遍历所有Node元素
        for node_elem in root.findall('Node'):
            # 获取坐标字符串并转换为元组
            coord_str = node_elem.get('cordination')
            # if coord_str=='(0, 1, 16)':
            #     print(1)
            try:
                # 处理不同格式的坐标字符串
                if coord_str.startswith('(') and coord_str.endswith(')'):
                    # 格式为"(x, y, z)"
                    coords = tuple(map(int, coord_str[1:-1].split(',')))
                else:
                    # 格式为"x,y,z"
                    coords = tuple(map(int, coord_str.split(',')))
            except Exception as e:
                print(f"坐标解析错误: {coord_str}, 错误: {e}")
                continue

            # 创建tegnode对象
            node = tegnode.tegnode(
                asc_nodes_flag=bool(node_elem.get('asc_nodes_flag') == "1"),
                rightneighbor=eval(node_elem.get('rightneighbor')) if node_elem.get(
                    'rightneighbor') != 'None' else None,
                leftneighbor=eval(node_elem.get('leftneighbor')) if node_elem.get(
                    'leftneighbor') != 'None' else None,

              #  leftneighbor=None,  # 原始保存时没有这个属性
                state=int(node_elem.get('state', -1))
            )

            # 添加到字典
            nodes[coords] = node

    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None

    return nodes


# # 使用示例
# if __name__ == "__main__":
#     # 示例数据 (需要替换为您的实际数据)
#     nodes_data = [
#         ((9, 9, 13), type('', (), {'asc_nodes_flag': False, 'rightneighbor': None, 'state': -1})()),
#         ((9, 9, 14), type('', (), {'asc_nodes_flag': False, 'rightneighbor': None, 'state': -1})()),
#         # 添加更多节点数据...
#     ]
#
#     save_nodes_to_xml(nodes_data, "nodes.xml")
