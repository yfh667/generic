import basicSa.wirtetoxml.pretoczml as pretoczml

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import json
import xml.etree.ElementTree as ET
if __name__ == "__main__":
    # 读取并解析数据
    xml_file = '/home/yfh/Desktop/Data/prefigure.xml'  # 您的 XML 数据文件路径
    node_positions, link_intervals, node_types, satellite_id_mapping = pretoczml.parse_ns3_data(xml_file)

    # 生成 CZML 内容
    czml_content = pretoczml.generate_czml(node_positions, link_intervals, node_types, satellite_id_mapping)

    # 保存到文件
    output_file = '/home/yfh/Desktop/Data/satellite_trajectory.czml'  # 输出文件路径
    pretoczml.save_to_czml_file(czml_content, output_file)

    print(f"CZML 文件已生成并保存到 {output_file}")
