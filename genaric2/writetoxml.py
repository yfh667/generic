import xml.etree.ElementTree as ET
from xml.dom import minidom


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
        node_elem.set("state", str(node.state))

    # 生成XML字符串
    xml_str = ET.tostring(root, encoding='utf-8')

    # 格式化XML
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


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
