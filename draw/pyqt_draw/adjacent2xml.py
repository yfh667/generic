import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def write_steps_to_xml(modify_edges_by_step, filename):
    """
    写入美化（带缩进、换行）的 XML
    """
    root = ET.Element("graphs")
    for timestamp, edges in modify_edges_by_step.items():
        step_elem = ET.SubElement(root, "step", timestamp=str(timestamp))
        for node, neighbors in edges.items():
            node_elem = ET.SubElement(step_elem, "node", id=str(node))
            for neighbor in neighbors:
                neighbor_elem = ET.SubElement(node_elem, "neighbor")
                neighbor_elem.text = str(neighbor)
    # 转字符串
    xml_str = ET.tostring(root, encoding="utf-8")
    # 用minidom美化
    dom = minidom.parseString(xml_str)
    pretty_xml_as_str = dom.toprettyxml(indent="  ")
    # 写文件
    with open(filename, "w", encoding="utf-8") as f:
        f.write(pretty_xml_as_str)




def read_steps_from_xml(filename):
    """
    filename: str, 输入xml文件名
    返回 dict, {timestamp: {node: set(neighbors)}}
    """
    tree = ET.parse(filename)
    root = tree.getroot()
    steps = {}
    for step_elem in root.findall("step"):
        timestamp = int(step_elem.get("timestamp"))
        edges = {}
        for node_elem in step_elem.findall("node"):
            node_id = int(node_elem.get("id"))
            neighbors = set(int(neighbor_elem.text) for neighbor_elem in node_elem.findall("neighbor"))
            edges[node_id] = neighbors
        steps[timestamp] = edges
    return steps

if __name__ == "__main__":
    modify_edges_by_step = {
        1202: {14: {13, 15, 52},},
        1300: {16: {13, 15, 59},},
        1301: {16: {13, 15, 52}, },

    }
    xml_file = "E:\\Data\\test.xml"

    write_steps_to_xml(modify_edges_by_step, xml_file)
