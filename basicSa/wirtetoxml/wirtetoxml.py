import xml.etree.ElementTree as ET
import os
from xml.dom.minidom import parseString

class WriteToXml:
    def __init__(self, file_name, stations_count, satellites_count):
        """
        初始化 WriteToXml 类。
        如果文件不存在，创建一个空的 XML 文件结构。
        如果文件已存在，加载现有内容用于追加写入。
        初始化 stations 和 satellites 的数量。
        """
        self.file_name = file_name
        self.stations_count = stations_count
        self.satellites_count = satellites_count

        # 检查文件是否存在
        if not os.path.exists(self.file_name):
            # 创建新的 XML 文件结构
            root = ET.Element('Data')

            # 初始化 stations 和 satellites 数量
            root.set('stations', str(self.stations_count))
            root.set('satellites', str(self.satellites_count))

            # 保存到文件
            tree = ET.ElementTree(root)
            tree.write(self.file_name, encoding='utf-8', xml_declaration=True)
        else:
            # 加载现有的 XML 文件
            self.tree = ET.parse(self.file_name)
            self.root = self.tree.getroot()

            # 从现有 XML 文件中读取 stations 和 satellites 数量
            self.stations_count = int(self.root.get('stations', 0))
            self.satellites_count = int(self.root.get('satellites', 0))

    def writeToXml(self, time, nodes):
        """
        向 XML 文件追加写入新的时间片数据，包括节点信息。
        """
        # 加载根节点
        tree = ET.parse(self.file_name)
        root = tree.getroot()

        # 添加时间节点
        time_element = ET.SubElement(root, 'Time')
        time_element.set('value', str(time))

        # 遍历节点并写入到 XML
        for idx, node in enumerate(nodes):
            node_element = ET.SubElement(time_element, 'Node')
            node_element.set('id', str(idx))  # 使用索引作为节点 ID

            # 添加位置子标签
            position_element = ET.SubElement(node_element, 'Position')
            position_element.set('x', f"{node.x:.6f}")
            position_element.set('y', f"{node.y:.6f}")
            position_element.set('z', f"{node.z:.6f}")

            # 添加链路数组结构化子标签
            link_array_element = ET.SubElement(node_element, 'LinkArray')
            for link_value in node.linked_array:
                link_element = ET.SubElement(link_array_element, 'Link')
                link_element.text = str(link_value)

        # 格式化并保存更新到文件
        self._pretty_save(tree)

    def _pretty_save(self, tree):
        """
        使用自定义逻辑格式化 XML 并保存到文件。
        """
        # 将 ElementTree 转换为字符串
        rough_string = ET.tostring(tree.getroot(), encoding='utf-8')
        # 使用 minidom 格式化 XML
        parsed = parseString(rough_string)
        # 手动调整换行符，删除多余的空白行
        pretty_string = parsed.toprettyxml(indent="    ").strip()
        # 写入文件
        with open(self.file_name, 'w', encoding='utf-8') as f:
            for line in pretty_string.splitlines():
                if line.strip():  # 去掉多余的空行
                    f.write(line + '\n')
