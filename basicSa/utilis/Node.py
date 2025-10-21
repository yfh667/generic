import math

LEO_PROP_EARTH_RAD = 6.37101e6  # Earth's radius in meters
import numpy as np


class Node:
    # angle 是弧度值，因此输入角度的时候需要转换下
    def __init__(self, x=0, y=0, z=0, angle=0, utilization=0,trackangle = 0,RAAN = 0,nodeid =0,time = 0):
        self.x = x
        self.y = y
        self.z = z
        self.angle = angle
        self.utilization = utilization
        self.trackangle = trackangle
        self.nodeid = nodeid
        self.RAAN = RAAN
        self.time = time

        #这个代表建链表
        self.linked_array = [-1] * 5  # 初始化大小为5的数组,注意这个实际上是用来辅助的，也就是在python中其作用的。
        #这个代表与ns-3沟通的数据结构，其实就是建链表+路由表
        self.matrix = [[-1, -1] for _ in range(15)]  # 初始化一个 15x2 的矩阵
    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        """Python解释器/列表的字符串表示"""
        return self.__str__()  # 直接复用__str__的格式
    def get_position(self):
        return (self.x, self.y, self.z)

    def get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self):
        self.x *= 1000
        self.y *= 1000
        self.z *= 1000
        return self

    # def linkarrary2matrix(self):
    #     for i in range(5):
    #         self.set_value( i, (value))

    def set_value(self, row, value):
        # 设置矩阵中的值，包含越界检查，直接设置一行的值
        if 0 <= row < 15 and len(value) == 2:
            self.matrix[row] = value
        else:
            raise IndexError("Index out of range or value size is incorrect for setting value.")

    def get_value(self, row):
        if 0 <= row < len(self.matrix):
            return self.matrix[row]
        else:
            raise IndexError("Row index out of range")


    def __str__(self):
        return f"Node(x={self.x}, y={self.y}, z={self.z}, angle={self.angle}, utilization={self.utilization},trackangle={self.trackangle},RAAN={self.RAAN},nodeid={self.nodeid},      time={self.time},   link_array={self.linked_array})"


class Vector2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def get_length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)


