import math

LEO_PROP_EARTH_RAD = 6.37101e6  # Earth's radius in meters
import numpy as np



def custom_print(*args, **kwargs):
    print("Sat2Gnd.py:", *args, **kwargs)

def get_cutoff_distance(sat, elevation_angle):
    hs = sat.get_length()
    a = 1
    b = 2 * LEO_PROP_EARTH_RAD * math.sin(elevation_angle)
    c = LEO_PROP_EARTH_RAD * LEO_PROP_EARTH_RAD - hs * hs
    delta = b * b - 4 * a * c


    return (-b + math.sqrt(delta)) / (2 * a)


def get_sat_angle_distance(sat, sat_elevation_angle):
    satangle = sat_elevation_angle
    hs = sat.get_length()
    theta_max = math.asin(LEO_PROP_EARTH_RAD / hs)

    if theta_max <= satangle:
        d = math.sqrt(hs * hs - LEO_PROP_EARTH_RAD * LEO_PROP_EARTH_RAD)
        return d

    hs = hs - LEO_PROP_EARTH_RAD
    b = -2.0 * math.cos(satangle) * (hs + LEO_PROP_EARTH_RAD)
    c = hs * hs + 2 * hs * LEO_PROP_EARTH_RAD
    disc = b * b - 4 * c
    d = (-b - math.sqrt(disc)) / 2.0

    return d


def GetDistance(satellite, station):
    sat_position = satellite.get_position()
    station_position = station.get_position()

    dx = sat_position[0] - station_position[0]
    dy = sat_position[1] - station_position[1]
    dz = sat_position[2] - station_position[2]

    return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


# 测试星地是否可见的代码,这包括，卫星与地面站均可见。
# station 和 basicSa 是单一的，就是一个node
def Ground_Sat(station, satellite):
    distance = GetDistance(station, satellite)
    cutoff_distance = get_cutoff_distance(satellite, station.angle)
    #ground see the sat
    if (distance > cutoff_distance):
        return 0
    cutoff_distance = get_sat_angle_distance(satellite, satellite.angle)
    # sat see the ground
    if (distance > cutoff_distance):
        return 0



   # return 1
    visibilty_ratio  = distance/satellite.get_length()
    return visibilty_ratio
