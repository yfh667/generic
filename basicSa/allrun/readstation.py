import basicSa.utilis.readdata as readdata
import os
import basicSa.satellite_stk_alpha.config.settings as settings  # 新增导入
import math
import basicSa.utilis.Node as Node
def load_station(dir_path):
    """加载文件夹下所有txt文件"""

    if not dir_path:
        return
    stations = []
    sta_id =0
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_path, filename)

            with open(file_path, "r") as file:
                lines = file.readlines()

                # 假设文件每行包含一个坐标和角度数据
                for line in lines:
                    try:
                        # 将每行数据拆分并转换为浮动数值
                        data = line.split()
                        x = float(data[0])
                        y = float(data[1])
                        z = float(data[2])
                        # angle = float(data[3])

                        # 将角度转换为弧度
                        angle = math.radians(settings.SatelliteConfig.stationangle)
                        stations.append(Node.Node(x=x, y=y, z=z, angle=angle, nodeid=sta_id))
                        sta_id += 1

                        # 发射信号
                 #       self.station_added.emit(Node.Node(x=x, y=y, z=z, angle=angle, nodeid=sta_id))
                    except ValueError as e:
                        print(f"Error parsing line in {filename}: {e}")
                        continue  # Skip to the next line if parsing fails
    return stations
