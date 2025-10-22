import basicSa.utilis.readdata as readdata
import basicSa.simulator.nodemanager as nodemanager
import  os


def readsatellite_raw(manager,dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT,t_start=None, t_end=None):
    satsnodes = readdata.readsats_multi(dir_path, satangle,t_start=None, t_end=None)

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


from lxml import etree
import os
import re
import  math
import basicSa.utilis.Node as Node
def natural_sort_key(s):
    """
    Returns a key that will sort strings containing numbers in a natural way.
    例如，'file1' 会排在 'file10' 前面，'file2' 排在 'file10' 后面
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]


# def readsatellite(manager, dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT, t_start=None, t_end=None):
#     # 获取并读取所有 XML 文件
#     xml_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".xml")],
#                        key=natural_sort_key)
#
#     total_files = 0
#     success_count = 0
#     error_files = []
#
#     for xml_file in xml_files:
#         file_path = os.path.join(dir_path, xml_file)
#
#         # 获取该文件的卫星数据
#         try:
#             satsnode = read_xml_data(file_path, satangle, t_start, t_end)  # 调用自定义的解析方法
#             if satsnode:
#                 total_files += 1
#                 node_id = satsnode[0].nodeid - 1  # 假设节点id从1开始
#                 i_index = node_id // N
#                 RAAN = i_index * BaseRAAN_INCREMENT
#
#                 # 存储轨迹到管理器
#                 manager.add_trajectory(node_id, satsnode)
#                 readdata.add_number_RAAN(satsnode, RAAN, track_angle)
#                 success_count += 1
#         except Exception as e:
#             error_files.append(f"Error processing {xml_file}: {str(e)}")
#
#     return total_files, success_count, error_files
import os
import math
from lxml import etree
from concurrent.futures import ThreadPoolExecutor, as_completed

def readsatellite(manager, dir_path, satangle, track_angle, P, N, BaseRAAN_INCREMENT, t_start=None, t_end=None):
    # 获取并读取所有 XML 文件
    xml_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".xml")],
                       key=natural_sort_key)

    total_files = 0
    success_count = 0
    error_files = []

    # 定义并行处理的函数
    def process_xml_file(xml_file):
        file_path = os.path.join(dir_path, xml_file)

        # 获取该文件的卫星数据
        try:
            satsnode = read_xml_data(file_path, satangle, t_start, t_end)  # 调用自定义的解析方法
            if satsnode:
                node_id = satsnode[0].nodeid - 1  # 假设节点id从1开始
                i_index = node_id // N
                RAAN = i_index * BaseRAAN_INCREMENT

                # 存储轨迹到管理器
                manager.add_trajectory(node_id, satsnode)
                readdata.add_number_RAAN(satsnode, RAAN, track_angle)
                return 1  # 成功处理一个文件
            else:
                return 0  # 如果该文件没有数据
        except Exception as e:
            error_files.append(f"Error processing {xml_file}: {str(e)}")
            return 0  # 出错的文件返回 0

    # 使用多线程池处理所有文件
    with ThreadPoolExecutor(max_workers=os.cpu_count() * 2) as executor:
        futures = {executor.submit(process_xml_file, xml_file): xml_file for xml_file in xml_files}

        for future in as_completed(futures):
            result = future.result()
            if result == 1:
                success_count += 1
            total_files += 1

    return total_files, success_count, error_files

def read_xml_data(xml_file_path, satangle, t_start, t_end):
    """
    读取单个 XML 文件的数据，并返回满足 [t_start, t_end) 时间段的节点列表
    """
    satnode = []
    try:
        tree = etree.parse(xml_file_path)

        # 解析卫星的所有 p 元素
        for p in tree.xpath("//p"):
            t = float(p.get("t"))
            if t_start is not None and t < t_start:
                continue
            if t_end is not None and t >= t_end:
                break

            x = float(p.get("x"))
            y = float(p.get("y"))
            z = float(p.get("z"))

            # 创建 Node 并加入 satnode 列表
            node = Node.Node(
                time=t,
                x=x,
                y=y,
                z=z,
                angle=math.radians(satangle),  # 使用角度转换为弧度
                nodeid=int(os.path.splitext(os.path.basename(xml_file_path))[0])  # 用文件名作为 nodeid
            )
            satnode.append(node)
    except Exception as e:
        print(f"Error reading {xml_file_path}: {str(e)}")

    return satnode
