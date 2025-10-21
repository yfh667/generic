import numpy as np
import math
from basicSa.utilis import Node
LEO_PROP_EARTH_RAD = 6.37101e6  # Earth's radius in meters
import basicSa.utilis.Node as Node
class Vector3D:
    def __init__(self, x, y, z):
        self.coords = np.array([x, y, z])

    def __repr__(self):
        return f"Vector3D({self.coords[0]}, {self.coords[1]}, {self.coords[2]})"

    def __neg__(self):
        return Vector3D(-self.coords[0], -self.coords[1], -self.coords[2])

    def __sub__(self, other):
        return Vector3D(self.coords[0] - other.coords[0], self.coords[1] - other.coords[1], self.coords[2] - other.coords[2])

    def __mul__(self, scalar):
        return Vector3D(self.coords[0] * scalar, self.coords[1] * scalar, self.coords[2] * scalar)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def dot_product(self, other):
        return np.dot(self.coords, other.coords)

    def length(self):
        return np.linalg.norm(self.coords)

def critical_cos_alpha(satellite1) :
    r = LEO_PROP_EARTH_RAD
    hs = satellite1.length()
  #  print(r)
  #  print(hs)
    return math.sqrt(1-(r/hs)**2)

def real_cos_alpha(satellite1,satellite2):
    os1 = satellite1
    os2 = satellite2
   ##   print(f"os2:{os2}")
    s1s2 = os2-os1

    return -os1.dot_product(s1s2)/(os1.length()*s1s2.length())




def Sat_Sat(satellite1, satellite2,i,j):
    p = Vector3D(*satellite1.get_position())
    q = Vector3D(*satellite2.get_position())
    if real_cos_alpha(p, q) < critical_cos_alpha(p):
        return 1
    return 0
def Sat_Sat2(x1,y1,z1, x2,y2,z2):
    satellite1 = Node.Node(x1,y1,z1)
    satellite2 = Node.Node(x2,y2,z2)

    p = Vector3D(*satellite1.get_position())
    q = Vector3D(*satellite2.get_position())
    if real_cos_alpha(p, q) < critical_cos_alpha(p):
        return 1
    return 0


##useage

