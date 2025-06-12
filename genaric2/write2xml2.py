# genaric2/xml_utils.py
import xml.etree.ElementTree as ET
from xml.dom import minidom




def manager_to_xml(manager, filename):
    """
    将ChromosomeManager保存为XML文件

    参数:
        manager: ChromosomeManager实例
        filename: 要保存的XML文件名
    """
    # 创建根元素
    root = ET.Element("Nodes")

    # 添加每个节点
    for coord, node in manager.items():
        node_elem = ET.SubElement(root, "Node")

        # 添加坐标属性
        node_elem.set("cordination", f"{coord[0]},{coord[1]},{coord[2]}")

        # 添加节点属性
        node_elem.set("asc_nodes_flag", str(node.asc_nodes_flag))
        node_elem.set("rightneighbor", str(node.rightneighbor) if node.rightneighbor else "None")
        node_elem.set("leftneighbor", str(node.leftneighbor) if node.leftneighbor else "None")
        node_elem.set("state", str(node.state))

    # 生成XML字符串
    xml_str = ET.tostring(root, encoding='utf-8')

    # 格式化XML
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(pretty_xml)


def xml_to_manager(manager,filename):
    """
    从XML文件读取数据，返回ChromosomeManager实例

    参数:
        filename: XML文件名

    返回:
        manager: ChromosomeManager实例
    """
   # manager = chromosome_manager.ChromosomeManager()

    try:
        # 解析XML文件
        tree = ET.parse(filename)
        root = tree.getroot()

        # 第一遍：创建所有节点
        for node_elem in root.findall('Node'):
            # 获取坐标字符串并转换为元组
            coord_str = node_elem.get('cordination')
            try:
                # 处理坐标字符串
                coords = tuple(map(int, coord_str.split(',')))
            except Exception as e:
                print(f"坐标解析错误: {coord_str}, 错误: {e}")
                continue

            # 创建tegnode对象
            node = tegnode.tegnode(
                asc_nodes_flag=node_elem.get('asc_nodes_flag') == 'True',
                state=int(node_elem.get('state', -1)))

            # 添加到管理器
            manager.add_node(coords, node)

            # 第二遍：设置邻居关系
            for node_elem in root.findall('Node'):
                coord_str = node_elem.get('cordination')
            coords = tuple(map(int, coord_str.split(',')))

            # 设置右邻居
            right_str = node_elem.get('rightneighbor')
            if right_str != 'None':
                try:
                    # 尝试两种格式的坐标
                    if '(' in right_str:
                        right_coord = eval(right_str)
                    else:
                        right_coord = tuple(map(int, right_str.split(',')))

                    # 确保邻居节点存在
                    if right_coord in manager:
                        manager.chromosome[coords]._set_rightneighbor(right_coord)
                except:
                    print(f"右邻居解析错误: {right_str}")

            # 设置左邻居
            left_str = node_elem.get('leftneighbor')
            if left_str != 'None':
                try:
                    # 尝试两种格式的坐标
                    if '(' in left_str:
                        left_coord = eval(left_str)
                    else:
                        left_coord = tuple(map(int, left_str.split(',')))

                    # 确保邻居节点存在
                    if left_coord in manager:
                        manager.chromosome[coords]._set_leftneighbor(left_coord)
                except:
                    print(f"左邻居解析错误: {left_str}")

    except ET.ParseError as e:
        print(f"XML解析错误: {e}")
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None

    return manager

if __name__ == "__main__":
    import genaric2.tegnode2 as tegnode



    # 创建染色体管理器
    manager = tegnode.Chromosome()

    # 添加节点
    coord1 = (1, 0, 1)
    node1 = tegnode.tegnode(False, -1)
    manager.add_node(coord1, node1)

    coord2 = (2, 0, 1)
    node2 = tegnode.tegnode(False, -1)
    manager.add_node(coord2, node2)

    # 设置节点1的右邻居为节点2
    manager.set_right_neighbor(coord1, coord2)

    # 将节点1状态改为0（建链）
    manager.set_node_state(coord1, 0)

    # 保存到XML
    manager_to_xml(manager, "nodes.xml")

    # 从XML加载
    loaded_manager = xml_to_manager(manager,"nodes.xml")

    # 验证加载结果
    print(f"节点1: {loaded_manager[coord1]}")
    print(f"节点2: {loaded_manager[coord2]}")
