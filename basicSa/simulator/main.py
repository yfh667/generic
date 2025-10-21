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


def main():
    dirpath  = '/home/yfh/Desktop/Data/station_qianfan_tongjin'
    stationangle = 20
    StationManager = readstation.readstation_path(dirpath, stationangle)
    stations =  StationManager.stations
    _ground_cache = []
    for i in range(len(stations)):
         _ground_cache.append(stations[i].trajectory[0])
    stationsnum = len(_ground_cache)
    # ground_cache
    #sendtocesium flag
    cesium_flag = 0
    if cesium_flag == 1:
        ws_manager = websocket.WebSocketManager()
        ws_manager.start_server()
    SatelliteManager = nodemanager.SatelliteManager()

    sat_dir_path = '/home/yfh/Desktop/Data/648qianfan'
    satangle = 45
    track_angle = 89
    P = 18
    N = 36
    BaseRAAN_INCREMENT = 10.2
    readsatellite.readsatellite(SatelliteManager,sat_dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT)

    oldpath = {}
    oldpath['[0, 1]'] = 'JDJDJDJDDD'
    simulatationtime =1000
    _position_cache = {}

    #path_writer = PathWriter("/home/yfh/Desktop/Data/simulation_paths.xml")
    path_writer = PathWriter(
        "/home/yfh/Desktop/Data/simulation_paths.xml",
        stationnum=stationsnum,
        # You can add other metadata here:
        simulation_time=simulatationtime,
        satellite_count=len(SatelliteManager.satellites)
    )

    # 清空旧文件
    for i in range(simulatationtime):
        current_time = i
        _position_cache.clear()

        for sat in SatelliteManager.satellites.values():
            idx = next((i for i, p in enumerate(sat.trajectory)
                        if p.time >=  current_time), 0)
            _position_cache[sat.id] = (
                sat.trajectory[idx],  # 当前时刻
                sat.trajectory[idx + 1]  # 下一时刻（用于计算）
            )
        #here we get the gndnodes and the basicSa nodes
        nodes,adj_list,min_paths = linkengine.linkengine(current_time, P, N,_position_cache, _ground_cache, oldpath)

        print(min_paths)

        Nodes = heightlight.highlight_selected_path([0, 1], nodes)
        if cesium_flag==1:

            sendtocesium2.CesiumSender.send_to_cesium(Nodes,  stationsnum,  ws_manager)
        path_writer.add_time_step(Nodes, current_time, min_paths)
        print(f"time is {current_time}")
     #   writetofile(time, min_paths)
        #writetofile(current_time=i, min_paths=min_paths)
     #   print("Hello, World!")

    path_writer.write_to_file()
    print("finish")
if __name__ == "__main__":
    main()
