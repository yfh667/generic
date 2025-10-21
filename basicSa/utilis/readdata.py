import math
from basicSa.utilis import Node, db
import concurrent.futures
import os
import re
from basicSa.los import visibilitydb
import  math
import basicSa.utilis.Node as Node

def add_number_RAAN(Node:list[Node.Node],RAAN,trackangle):
    for node in Node:
        node.trackangle = trackangle
        node.RAAN =RAAN
        #node.nodeid = id







def natural_sort_key(s):
    # 提取数字以进行自然排序
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def readstation(path,stationagnle):
    nodes = []

    files = os.listdir(path)
    # 使用自定义排序函数进行自然排序
    files.sort(key=natural_sort_key)

    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():  # 忽略空行
                            data = list(map(float, line.split()))
                            node = Node.Node(x=data[0], y=data[1], z=data[2], angle=math.radians(stationagnle))
                            nodes.append(node)
                            # print(f"内容来自 {file}:")
                            # print(f"Node: x={node.x}, y={node.y}, z={node.z}, angle={node.angle}")
                            # print("-" * 40)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")

    return nodes


# # 示例调用
# path = "/home/yfh/Desktop/Data/stations"
# readstation(path,20)
#

def readsats(path,satangle):
    nodes = []
    files = os.listdir(path)
    # 使用自定义排序函数进行自然排序
    files.sort(key=natural_sort_key)



    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(path, file)
            print(f"file path: {file}")
            try:
                file_number = int(os.path.splitext(file)[0])
            except ValueError:
                print(f"文件名格式错误: {file}，无法提取数字")
                continue  # 跳过无效文件

           # print(f"正在处理文件: {file} → 提取编号: {file_number}")

            satnode = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():  # 忽略空行
                            data = list(map(float, line.split()))
                            node = Node.Node(x=data[1], y=data[2], z=data[3], angle=math.radians(satangle),nodeid=int(file_number))
                            satnode.append(node)
                         #   print(f"内容来自 {file}:")
                         #   print(f"Node: x={node.x}, y={node.y}, z={node.z}, angle={node.angle}")
                         #   print("-" * 40)
                    nodes.append(satnode)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    return nodes


def readsats_multi(path, satangle):
    # 获取并自然排序文件列表
    files = sorted([f for f in os.listdir(path) if f.endswith(".txt")],
                   key=natural_sort_key)

    # 计算最佳线程数（文件数 vs CPU核心数）
    max_workers = min(len(files), os.cpu_count() * 2)  # I/O密集型可超线程
    chunk_size = max(1, len(files) // max_workers)  # 动态分块

    def process_batch(file_batch):
        """处理文件批次的线程函数"""

        batch_nodes = []
        for file in file_batch:
            print(f"file path: {file}")
            file_path = os.path.join(path, file)

            # 提取文件名中的数字
            try:
                file_number = int(os.path.splitext(file)[0])
            except ValueError:
                print(f"文件名格式错误: {file}，无法提取数字")
                continue  # 跳过无效文件

         #   print(f"正在处理文件: {file} → 提取编号: {file_number}")

            satnode = []
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    lines = content.split('\n')
                    for line in lines:
                        if line.strip():
                            data = list(map(float, line.split()))
                            node = Node.Node(
                                time = data[0],
                                x=data[1],
                                y=data[2],
                                z=data[3],
                                angle=math.radians(satangle),
                                # 可选：将文件编号存入节点
                                nodeid=file_number  # 假设Node类有file_id属性
                            )
                            satnode.append(node)
                    batch_nodes.append(satnode)

            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                batch_nodes.append([])
        return batch_nodes

    # 划分文件批次（保持顺序）
    batches = [files[i:i + chunk_size]
               for i in range(0, len(files), chunk_size)]

    # 并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        results = []
        for future in concurrent.futures.as_completed(futures):
            results.extend(future.result())  # 按批次完成顺序合并

    # 按原始文件顺序重组结果
    ordered_results = []
    for batch in batches:
        for file in batch:
            idx = files.index(file)
            ordered_results.append(results[idx])

    return [res for res in ordered_results if res]


import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]  # 添加 r 前缀改为原始字符串
