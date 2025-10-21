import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt


# 读取 XML 文件
tree = ET.parse("/home/yfh/Desktop/Data/bound.xml")  # 替换成你的 XML 文件名
root = tree.getroot()

# 初始化数据列表
times = []
inbounds = []
outbounds = []

# 遍历每一个 <TimeData>
for time_data in root.findall("TimeData"):
    # 提取时间，去掉单位 "s"
    time = float(time_data.attrib["time"].replace(" s", ""))

    # 提取入站和出站流量，去掉单位 "byte"
    inbound = time_data.find("m_packet_inbound").attrib["value"]
    outbound = time_data.find("m_packet_ountbound").attrib["value"]

    inbound_val = int(inbound.replace(" byte", ""))
    outbound_val = int(outbound.replace(" byte", ""))

    # 存入列表
    times.append(time)
    inbounds.append(inbound_val)
    outbounds.append(outbound_val)

# 绘制图像
plt.plot(times, inbounds, label="Inbound Traffic", marker='o')
plt.plot(times, outbounds, label="Outbound Traffic", marker='s')
plt.xlabel("Time (s)")
plt.ylabel("Traffic (bytes)")
plt.title("Inbound / Outbound Traffic Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
