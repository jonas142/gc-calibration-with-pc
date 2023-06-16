import numpy as np
from .parametersBC import maxR

def radialDispersion0(distance):    # linear function dependend of the distance a boundary object is encountered
    return (distance + 8) * 0.08

# returns a list of 16 elements which contains the different scaled distances up to 16
def radialScaling():
    iterator = np.arange(1, 17, 1)  # array from 1 to 16 range doesnt work here since first distance unit is 1
    rf_inc = 0.1 * np.array([x for x in iterator])   # 0.1 - 1.6
    # p.maxR
    polar_distance = np.zeros(16)     # 16 x 0
    for rr in iterator:
        if rr == 1:
            polar_distance[rr - 1] = rr
        else:
            polar_distance[rr - 1] = polar_distance[rr - 2] + rf_inc[rr - 1]

    polar_distance = (polar_distance / (max(polar_distance)) * (maxR - 0.5))    # some scaling
    return polar_distance