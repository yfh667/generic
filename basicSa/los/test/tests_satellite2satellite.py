LEO_PROP_EARTH_RAD = 6.37101e6  # Earth's radius in meters
import basicSa.los.satllite2satellite as satllite2satellite


def read_satellite_data(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()
            # Assume timestamp is the first four parts and coordinates basicSa from the fifth part
           # timestamp = ' '.join(parts[:1])  # This combines date and time into one string
            timestamp = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            data[timestamp] = NodeAndGnd.Node(x, y, z)
    return data

# node1 = NodeAndGnd.Node(-1096.152685, -5053.095180, -4535.796180).normalize()
# node2 = NodeAndGnd.Node(3828.032424, -3475.843979, -4535.795959).normalize()
# # //注意normalize函数，只是将单位变化一下，如果输入的就是m，那么就不需要变化
# # 判断两个节点是否连通
# if Sat_Sat(node1, node2):
#     print("两个节点可以连通")
# else:
#     print("两个节点不可以连通")
#



# test
if __name__ == "__main__":
    pwd = '/home/yfh/Desktop/Data/test/'
    satellite1_data = read_satellite_data(pwd+'1.txt')
    print(satellite1_data)
    satellite2_data = read_satellite_data(pwd+'2.txt')

    with open(pwd+'output.txt', 'w') as f:
        if isinstance(satellite1_data, dict) and isinstance(satellite2_data, dict):
            for time in satellite1_data:
                if time in satellite2_data:
                    node1 = satellite1_data[time]
                    node2 = satellite2_data[time]


                    if satllite2satellite.Sat_Sat(node1, node2):
                       # print(f"time is {time}")

                        f.write(f"At {time}, \n")
        else:
            f.write("Data is not in the expected format.\n")
