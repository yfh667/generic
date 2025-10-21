import numpy as np
from datetime import datetime, timezone

def datetime_to_julian_date(dt):
    """Convert datetime to Julian Date."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    year = dt.year
    month = dt.month
    day = dt.day + dt.hour / 24 + dt.minute / 1440 + dt.second / 86400 + dt.microsecond / 8.64e10

    if month <= 2:
        year -= 1
        month += 12

    A = np.floor(year / 100)
    B = 2 - A + np.floor(A / 4)

    JD = np.floor(365.25 * (year + 4716)) + np.floor(30.6001 * (month + 1)) + day + B - 1524.5
    return JD

def gmst_from_julian(JD):
    """Compute Greenwich Mean Sidereal Time in radians."""
    T = (JD - 2451545.0) / 36525
    GMST_deg = 280.46061837 + 360.98564736629 * (JD - 2451545) \
               + 0.000387933 * T**2 - T**3 / 38710000
    GMST_deg = GMST_deg % 360
    return np.deg2rad(GMST_deg)

def gettheta(utc_datetime):

    JD = datetime_to_julian_date(utc_datetime)

    theta = gmst_from_julian(JD)
    return theta



def ecef_to_eci(r_ecef, utc_datetime):
    """Convert ECEF (3,) vector to ECI (3,) using UTC datetime."""
    # JD = datetime_to_julian_date(utc_datetime)
    #
    # theta = gmst_from_julian(JD)
    theta = gettheta(utc_datetime)
    # Rotation matrix about Z axis (inverse of ECI->ECEF)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    r_eci = R @ r_ecef
    return r_eci
