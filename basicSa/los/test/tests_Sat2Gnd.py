import math
import basicSa.utilis.Node as Node
import basicSa.los.Sat2Gnd as Sat2Gnd

##usage
# 注意normalize函数，只是将单位变化一下，如果输入的就是m，那么就不需要变化
station = Node.Node(-3330.416038 * 1000, 5432.320718 * 1000, 279.864892 * 1000, math.radians(10))
satellite = Node.Node(-1343.875841 * 1000, 6837.318990 * 1000, 0.000000 * 1000, math.radians(30))
if Sat2Gnd.Ground_Sat(station, satellite):
    print(f" visible")
else:
    print(f" not visible")


def read_satellite_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        satellite_data = []
        for line in lines:
            parts = line.strip().split()
            time = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])

            satellite_data.append((time, x, y, z))
        return satellite_data


##usage


# test
if __name__ == "__main__":
    station = Node.Node(-1216906.217000, 6260439.768000, 82928.652000,
                        math.radians(30))  # Ground station, with an elevation angle of 30 degrees
    pwd = '/home/yfh/Desktop/Data/sats/'
    satellite_data = read_satellite_data(pwd + '9.txt')

    for time, x, y, z in satellite_data:
        satellite = Node.Node(x, y, z,
                              math.radians(45))  # Assume the basicSa also has an elevation angle of 30 degrees
        #   print(f"Node {x}:{y}:{z}")

        if Sat2Gnd.Ground_Sat(station, satellite):
            print(f"Time {time}: visible")
