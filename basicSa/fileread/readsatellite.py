import basicSa.utilis.readdata as readdata
import basicSa.simulator.nodemanager as nodemanager
import  os


def readsatellite(manager,dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT):
    satsnodes = readdata.readsats_multi(dir_path, satangle)

    total_files = 0
    success_count = 0
    error_files = []

   # manager = nodemanager.SatelliteManager()  # 创建新的管理器

    # 处理每个节点
    for satsnode in satsnodes:
        try:
            node_id = satsnode[0].nodeid - 1
            i_index = node_id // N
            RAAN = i_index * BaseRAAN_INCREMENT

            manager.add_trajectory(node_id, satsnode)
            readdata.add_number_RAAN(satsnode, RAAN, track_angle)
            success_count += 1
        except Exception as e:
            error_files.append(f"Node {node_id} error: {str(e)}")

    # 精确复制原文件统计逻辑
    for filename in os.listdir(dir_path):
        if filename.lower().endswith('.txt'):
            total_files += 1

    return   total_files, success_count, error_files
