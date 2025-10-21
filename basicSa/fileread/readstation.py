import  os
import math
import basicSa.simulator.nodemanager as nodemanager


def readstation_path(dir_path,stationangle):
    # 获取所有txt文件并按数字顺序排序
    files = []
    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            try:
                # 提取文件名中的数字部分（假设文件名格式为数字.txt）
                num = int(os.path.splitext(filename)[0])
                files.append((num, filename))
            except ValueError:
                continue  # 跳过不符合命名规范的文件

    # 按数字顺序排序
    files.sort(key=lambda x: x[0])
    #stationNodes = []
    station_manager = nodemanager.StationManager()  # 创建单例实例

    # 按顺序处理文件
    for num, filename in files:
        file_path = os.path.join(dir_path, filename)
        with open(file_path, "r") as file:
            lines = file.readlines()

            for line in lines:
                try:
                    data = line.split()
                    x = float(data[0])
                    y = float(data[1])
                    z = float(data[2])
                    angle = math.radians(stationangle)

                    station_manager.add_ground_station(x, y, z, angle)


                    # self.station_added.emit(Node.Node(x=x, y=y, z=z, angle=angle, nodeid=sta_id))
                  #  stationNodes = stationNodes.append(Node.Node(x=x, y=y, z=z, angle=angle, nodeid=sta_id))

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line in {filename}: {e}")
                    continue

    return station_manager
