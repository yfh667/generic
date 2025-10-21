import basicSa.fileread.readstation as readstation
import basicSa.fileread.readsatellite as readsatellite
import basicSa.simulator.nodemanager as nodemanager
import basicSa.simulator.linkengine as linkengine
import basicSa.simulator.websocket as websocket
import time
import logging
import basicSa.simulator.sendtocesium as sendtocesium2
import basicSa.simulator.cesiumswitch.heightlight as heightlight
import xml.etree.ElementTree as ET
from xml.dom import minidom

class PathWriter:
    def __init__(self, filename="simulation_paths.xml", stationnum=0, **metadata):
        self.filename = filename
        self.root = ET.Element("simulation")
        self.tree = ET.ElementTree(self.root)
        # Add simulation metadata
        metadata_elem = ET.SubElement(self.root, "metadata")
        ET.SubElement(metadata_elem, "stationnum").text = str(stationnum)

        # Add any additional metadata
        for key, value in metadata.items():
            ET.SubElement(metadata_elem, key).text = str(value)

        # Clear the file at initialization
        with open(self.filename, "w") as f:
            f.write("")

    def add_time_step(self, Nodes,current_time, min_paths):
        """Add a time step with paths to the XML structure in memory"""
        # Create time step element
        time_step = ET.SubElement(self.root, "TimeData")
        time_step.set("value", str(current_time))
        # First add all nodes with their positions
        nodes_elem = ET.SubElement(time_step, "nodes")
        nodeid = 0
        for node in Nodes:
            node_elem = ET.SubElement(nodes_elem, "node")
            node_elem.set("id", str(nodeid))

            pos_elem = ET.SubElement(node_elem, "Position")
            pos_elem.set("x", str(node.x))
            pos_elem.set("y", str(node.y))
            pos_elem.set("z", str(node.z))
            nodeid += 1

        # Add all paths
        # 添加所有路径（紧凑格式）
        paths_container = ET.SubElement(time_step, "paths")

        # Then add individual paths to the container
        for path_idx, path in enumerate(min_paths):
            path_elem = ET.SubElement(paths_container, "path")  # Add to container
            path_elem.set("id", str(path_idx))
            path_elem.text = str(path).replace(' ', '')# 去除空格保持紧凑

    def write_to_file(self):
        """Write the complete XML structure to file"""
        # Beautify XML format
        xml_str = ET.tostring(self.root, encoding="utf-8")
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Write to file
        with open(self.filename, "w", encoding="utf-8") as f:
            f.write(pretty_xml)
