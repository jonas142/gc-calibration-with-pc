import math
import numpy as np

import system.bvc_model.parametersBC as p
import system.bvc_model.subroutines as subroutines



def boundaryCellActivityTraining(boundary_location):
    '''
    This method calculates egocentric and allocentric boundary cell activity during the training phase.
    It is called from MakeWeights.py

    until line 76: mathematical setup to get all points in between start and end points that constitute the boundary

    :param boundary_location: start and endpoints of a certain boundary given as function parameters.
    :return: activity vector which contains each cell's activity
    '''
    # get linearly growing distances in relation to the number of distance units chosen
    polar_distances = subroutines.radialScaling()
    polar_angles = np.linspace(0, 2 * math.pi, 51)

    p_dist, p_ang = np.meshgrid(polar_distances, polar_angles)
    grid_distance = np.reshape(np.transpose(p_dist), (np.prod(np.size(p_dist)), 1),
                               order="F")  # vector of distances from each neuron to the origin
    grid_angle = np.reshape(np.transpose(p_ang), (np.prod(np.size(p_ang)), 1), order="F")  # corresponding angles
    transform_matrix = np.where(grid_angle > math.pi, 1, 0)
    grid_angle = grid_angle - 2 * math.pi * transform_matrix

    # cartesian grid points that cover the region covered by the neurons
    min_x = -p.maxR
    max_x = p.maxR
    min_y = -p.maxR
    max_y = p.maxR
    nr_x = round((max_x - min_x) / p.resolution)
    nr_y = round((max_y - min_y) / p.resolution)
    x_points = np.arange(min_x + p.resolution / 2, min_x + (nr_x - 0.5) * p.resolution, p.resolution)
    y_points = np.arange(min_y + p.resolution / 2, min_y + (nr_y - 0.5) * p.resolution, p.resolution)
    x, y = np.meshgrid(x_points, y_points)

    activity_vector = np.zeros((816, 1))

    # boundary start and endpoints
    x_start = boundary_location[0]
    y_start = boundary_location[1]
    x_end = boundary_location[2]
    y_end = boundary_location[3]

    boundary_length = math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)
    nr_x = (x_end - x_start) / boundary_length
    nr_y = (y_end - y_start) / boundary_length

    # perpendicular displacements
    perpDispFromGrdPts_x = -(x - x_start) * (1 - math.pow(nr_x, 2)) + (y - y_start) * nr_y * nr_x
    perpDispFromGrdPts_y = -(y - y_start) * (1 - math.pow(nr_y, 2)) + (x - x_start) * nr_x * nr_y


    if x_end != x_start:
        t = (x + perpDispFromGrdPts_x - x_start) / (x_end - x_start)
    else:
        t = (y + perpDispFromGrdPts_y - y_start) / (y_end - y_start)

    t = np.where((t >= 0) & (t <= 1), 0.5, 0)

    perpDispFromGrdPts_x = np.where(
        (perpDispFromGrdPts_x >= (-p.resolution / 2)) & (perpDispFromGrdPts_x < (p.resolution / 2)), 0.5, 0)
    perpDispFromGrdPts_y = np.where(
        (perpDispFromGrdPts_y >= (-p.resolution / 2)) & (perpDispFromGrdPts_y < (p.resolution / 2)), 0.5, 0)

    result = t + perpDispFromGrdPts_x + perpDispFromGrdPts_y
    result = np.where(result == 1.5, result, 0)

    xBndryPts = x[np.where(result == 1.5)]
    yBndryPts = y[np.where(result == 1.5)]

    # transform from cartesian to polar coordinates
    thetaBndryPts = np.arctan2(yBndryPts, xBndryPts)
    rBndryPts = np.sqrt(xBndryPts ** 2 + yBndryPts ** 2)

    #for each boundary point the activity is calculated according to the activity function (eq. 4.1 of Thesis)
    for index in range(thetaBndryPts.size):
        angDiff1 = abs(grid_angle - thetaBndryPts[index])
        angDiff2 = 2 * math.pi - abs(-grid_angle + thetaBndryPts[index])
        angDiff = np.multiply(np.where(angDiff1 < math.pi, 1, 0), angDiff1) + np.multiply(
            np.where(angDiff1 > math.pi, 1, 0), angDiff2)
        sigmaR = subroutines.radialDispersion0(rBndryPts[index])
        activity_vector = activity_vector + 1 / rBndryPts[index] * (
            np.multiply(np.exp(-np.power(np.divide(angDiff, p.angularDispersion), 2)),
                        np.exp(-np.power((np.divide((grid_distance - rBndryPts[index]), sigmaR)), 2))))

    maximum = max(activity_vector)
    if maximum > 0.0:
        activity_vector = activity_vector / maximum
    return activity_vector


def boundaryCellActivitySimulation(thetaBndryPts, rBndryPts):
    '''
    Calculates BC activity during simulation
    :param thetaBndryPts: polar coordinate of boundary points representing angle
    :param rBndryPts: polar coordinate of boundary points representing distance
    :return: activity vector containing each cell's activity
    '''
    polar_distances = subroutines.radialScaling()
    polar_angles = np.linspace(0, 2 * math.pi, 51) # - parametersBC.polarAngularResolution

    p_dist, p_ang = np.meshgrid(polar_distances, polar_angles)
    grid_distance = np.reshape(np.transpose(p_dist), (np.prod(np.size(p_dist)), 1),
                               order="F")  # vector of distances from each neuron to the origin
    grid_angle = np.reshape(np.transpose(p_ang), (np.prod(np.size(p_ang)), 1), order="F")  # corresponding angles
    transform_matrix = np.where(grid_angle > math.pi, 1, 0)
    grid_angle = grid_angle - 2 * math.pi * transform_matrix
    activity_vector = np.zeros((816, 1))

    for index in range(thetaBndryPts.size):
        angDiff1 = abs(grid_angle - thetaBndryPts[index])
        angDiff2 = 2 * math.pi - abs(-grid_angle + thetaBndryPts[index])
        angDiff = np.multiply(np.where(angDiff1 < math.pi, 1, 0), angDiff1) + np.multiply(
            np.where(angDiff1 > math.pi, 1, 0), angDiff2)
        sigmaR = subroutines.radialDispersion0(rBndryPts[index])
        activity_vector = activity_vector + 1 / rBndryPts[index] * (
            np.multiply(np.exp(-np.power(np.divide(angDiff, p.angularDispersion), 2)),
                        np.exp(-np.power((np.divide((grid_distance - rBndryPts[index]), sigmaR)), 2))))
    maximum = max(activity_vector)
    if maximum > 0.0:
        activity_vector = activity_vector / maximum
    return activity_vector
