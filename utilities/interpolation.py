import numpy as np
from math import atan2, sqrt
import math
MAX_VELOCITY = 0.3
TIME = 0.02

def dist_between_nodes(node1_val, node2_val):
    sum_v = 0
    for dim in range(node1_val.shape[0]):
        diff = (node1_val[dim] - node2_val[dim]) ** 2
        sum_v = sum_v + diff
    dist = np.sqrt(sum_v)
    return dist

def spline_interpolation(points, time_interval=None, total_time=20):
    # points (n x 3) array x, y, z points  #3 points
    # have at least 4 initial points
    print('spline interpolationl')
    desired_points = 4
    if time_interval is not None and time_interval.shape[0] == points.shape[0]:
        total_time = time_interval[-1]
    else:
        time_interval = np.linspace(0, total_time, points.shape[0])  # 3 times
    if points.shape[0] == 2:
        # print("shape is 2")
        new_points = np.array([[]])
        for i in range(desired_points):
            if i == 0:
                point = [points[0]]
                new_points = point
            elif i == desired_points:
                point = [points[1]]
                new_points = np.append(new_points, point, axis=0)
            else:
                if i == 1:
                    mid_point = calculate_midpoint(points[i - 1], points[i])
                else:
                    mid_point = calculate_midpoint(new_points[i - 1], points[1])
                new_points = np.append(new_points, [mid_point], axis=0)

        dist = np.linalg.norm((points[0] - points[1]), 2)
        velocity = float(dist) / total_time
        # print("velocity before: ", velocity)
        if velocity > MAX_VELOCITY:
            total_time = dist / MAX_VELOCITY
        velocity = float(dist) / total_time
        # print("velocity: ", velocity)
        points = new_points

    print("points: ", points)
    print("time interval: ", time_interval)
    degree = 5
    num_of_polys = points.shape[
                       0] - 1  # 3
    a_matrix = np.zeros(((degree + 1) * num_of_polys, (
                degree + 1) * num_of_polys))  # 18 x 18
    num_of_coeffs = int(float(a_matrix.shape[1]) / float(
        num_of_polys))  # 6
    b_vector = np.zeros(((degree + 1) * num_of_polys, 1))
    coefficients = np.array([])

    for dim in range(points.shape[1]):
        w = points[:, dim]
        t = time_interval
        i = 0
        for poly in range(num_of_polys):
            if poly == 0:
                poly_index_0 = poly * (degree + 1)
                # print("poly index 0: ", poly_index_0)
                a_matrix[poly_index_0, 0:num_of_coeffs] = np.array(
                    [t[i] ** 5, t[i] ** 4, t[i] ** 3, t[i] ** 2, t[i], 1])

                a_matrix[poly_index_0 + 1, 0:num_of_coeffs] = np.array(
                    [5 * t[i] ** 4, 4 * t[i] ** 3, 3 * t[i] ** 2, 2 * t[i], 1, 0])
                # first deriv
                a_matrix[poly_index_0 + 2, 0:num_of_coeffs] = np.array(
                    [5 * t[i + 1] ** 4, 4 * t[i + 1] ** 3, 3 * t[i + 1] ** 2, 2 * t[i + 1], 1, 0])
                a_matrix[poly_index_0 + 2, num_of_coeffs:2 * num_of_coeffs] = np.array(
                    [-5 * t[i + 1] ** 4, -4 * t[i + 1] ** 3, -3 * t[i + 1] ** 2, -2 * t[i + 1], -1, 0])
                # second deriv
                a_matrix[poly_index_0 + 3, 0:num_of_coeffs] = np.array(
                    [20 * t[i + 1] ** 3, 12 * t[i + 1] ** 2, 6 * t[i + 1], 2, 0, 0])
                a_matrix[poly_index_0 + 3, num_of_coeffs:2 * num_of_coeffs] = np.array(
                    [-20 * t[i + 1] ** 3, -12 * t[i + 1] ** 2, -6 * t[i + 1], -2, 0, 0])
                # third deriv
                a_matrix[poly_index_0 + 4, 0:num_of_coeffs] = np.array([60 * t[i + 1] ** 2, 24 * t[i + 1], 6, 0, 0, 0])
                a_matrix[poly_index_0 + 4, num_of_coeffs:2 * num_of_coeffs] = np.array(
                    [-60 * t[i + 1] ** 2, -24 * t[i + 1], -6, 0, 0, 0])
                # fourth deriv
                a_matrix[poly_index_0 + 5, 0:num_of_coeffs] = np.array([120 * t[i + 1], 24, 0, 0, 0, 0])
                a_matrix[poly_index_0 + 5, num_of_coeffs:2 * num_of_coeffs] = np.array(
                    [-120 * t[i + 1], -24, 0, 0, 0, 0])
                # fifth deriv condition
                a_matrix[poly_index_0 + 6, 0:num_of_coeffs] = np.array([120, 0, 0, 0, 0, 0])
                a_matrix[poly_index_0 + 6, num_of_coeffs:2 * num_of_coeffs] = np.array([-120, 0, 0, 0, 0, 0])

                a_matrix[poly_index_0 + 7, 0:num_of_coeffs] = np.array(
                    [t[i + 1] ** 5, t[i + 1] ** 4, t[i + 1] ** 3, t[i + 1] ** 2, t[i + 1], 1])

                b_vector[poly_index_0] = w[i]
                b_vector[poly_index_0 + 7] = w[i + 1]

            elif poly == (num_of_polys - 1):
                a_matrix[(poly + 1) * (degree + 1) - 4, (degree + 1) * (poly + 1) - num_of_coeffs:] = np.array(
                    [t[i] ** 5, t[i] ** 4, t[i] ** 3, t[i] ** 2, t[i], 1])

                a_matrix[(poly + 1) * (degree + 1) - 3, (degree + 1) * (poly + 1) - num_of_coeffs:] = np.array(
                    [20 * t[i + 1] ** 3, 12 * t[i + 1] ** 2, 6 * t[i + 1], 2, 0, 0])
                a_matrix[(poly + 1) * (degree + 1) - 2,
                (((degree + 1) * (poly + 1) - num_of_coeffs) - num_of_coeffs):(degree + 1) * (
                        poly + 1) - num_of_coeffs] = np.array([120, 0, 0, 0, 0, 0])
                a_matrix[(poly + 1) * (degree + 1) - 2, (degree + 1) * (poly + 1) - num_of_coeffs:] = np.array(
                    [-120, 0, 0, 0, 0, 0])

                a_matrix[(poly + 1) * (degree + 1) - 1, (degree + 1) * (poly + 1) - num_of_coeffs:] = np.array(
                    [t[i + 1] ** 5, t[i + 1] ** 4, t[i + 1] ** 3, t[i + 1] ** 2, t[i + 1], 1])
                b_vector[(poly + 1) * (degree + 1) - 4] = w[i]
                b_vector[(poly + 1) * (degree + 1) - 1] = w[i + 1]

            elif poly < (num_of_polys - 1):
                poly_index = poly * (degree + 1) + 2
                a_matrix[poly_index, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = np.array(
                    [t[i] ** 5, t[i] ** 4, t[i] ** 3, t[i] ** 2, t[i], 1])
                # first deriv
                a_matrix[poly_index + 1, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = np.array(
                    [5 * t[i + 1] ** 4, 4 * t[i + 1] ** 3, 3 * t[i + 1] ** 2, 2 * t[i + 1], 1, 0])
                a_matrix[poly_index + 1,
                (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = np.array(
                    [-5 * t[i + 1] ** 4, -4 * t[i + 1] ** 3, -3 * t[i + 1] ** 2, -2 * t[i + 1], -1, 0])
                # second deriv
                a_matrix[poly_index + 2, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = np.array(
                    [20 * t[i + 1] ** 3, 12 * t[i + 1] ** 2, 6 * t[i + 1], 2, 0, 0])
                a_matrix[poly_index + 2,
                (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = np.array(
                    [-20 * t[i + 1] ** 3, -12 * t[i + 1] ** 2, -6 * t[i + 1], -2, 0, 0])
                # third deriv
                a_matrix[poly_index + 3, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = np.array(
                    [60 * t[i + 1] ** 2, 24 * t[i + 1], 6, 0, 0, 0])
                a_matrix[poly_index + 3,
                (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = np.array(
                    [-60 * t[i + 1] ** 2, -24 * t[i + 1], -6, 0, 0, 0])
                # fourth deriv
                a_matrix[poly_index + 4, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = np.array(
                    [120 * t[i + 1], 24, 0, 0, 0, 0])
                a_matrix[poly_index + 4,
                (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = np.array(
                    [-120 * t[i + 1], -24, 0, 0, 0, 0])

                # 0th derivative
                a_matrix[poly_index + 5, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = np.array(
                    [t[i + 1] ** 5, t[i + 1] ** 4, t[i + 1] ** 3, t[i + 1] ** 2, t[i + 1], 1])

                b_vector[poly_index] = w[i]
                b_vector[poly_index + 5] = w[i + 1]
            i += 1
        coeffs = np.linalg.solve(a_matrix, b_vector)
        if coefficients.shape[0] == 0:
            coefficients = coeffs
        else:
            coefficients = np.append(coefficients, coeffs, axis=1)
    return coefficients, num_of_coeffs, time_interval


def extract_points(coefficients_info, dt=0.02):
    coefficients, num_of_coeffs, time_interval = coefficients_info
    t = 0
    index = 0
    point = np.zeros((4, 3))
    points = np.array([])
    for time in range(1, time_interval.shape[0]):
        curr_coeffs = coefficients[index * num_of_coeffs:num_of_coeffs * (index + 1)]
        # print("curr coeffs: ", curr_coeffs)
        # print("time interval: ", time_interval)
        while t < time_interval[time]:
            # print("t: ", t)
            # print("t now: ", time_interval[time])
            for dim in range(curr_coeffs.shape[1]):
                dim_coeffs = curr_coeffs[:, dim]
                # print("dim coeefs:", dim_coeffs)
                point[0, dim] = dim_coeffs[0] * t ** 5 + dim_coeffs[1] * t ** 4 + dim_coeffs[2] * t ** 3 + dim_coeffs[
                    3] * t ** 2 + dim_coeffs[4] * t + dim_coeffs[5]
                point[1, dim] = 5 * dim_coeffs[0] * t ** 4 + 4 * dim_coeffs[1] * t ** 3 + 3 * dim_coeffs[
                    2] * t ** 2 + 2 * dim_coeffs[3] * t + dim_coeffs[4]
                point[2, dim] = 20 * dim_coeffs[0] * t ** 3 + 12 * dim_coeffs[1] * t ** 2 + 6 * dim_coeffs[2] * t + 2 * \
                                dim_coeffs[3]
                point[3, dim] = 60 * dim_coeffs[0] * t ** 2 + 24 * dim_coeffs[1] * t + 6 * dim_coeffs[2]
                # point[4, dim] = 120 * dim_coeffs[0] * t + 24 * dim_coeffs[1]
            if points.shape[0] == 0:
                points = np.expand_dims(point, axis=0)
            else:
                points = np.append(points, np.expand_dims(point, axis=0), axis=0)
            t += dt
        index += 1
        # print("coefficients: ", coefficients)
        # input("extracting points")
    return points


def parameterize_time_waypoint_generator(phat, x, s, dt):
    ks = 10
    sd = math.exp(-ks * dist_between_nodes(phat[0, :], x[0,
                                                       :]) ** 2)  # paramed time dynamics
    pd = phat[1, :] * sd
    sdd = -ks * math.exp(-ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2) * 2 * np.dot(phat[0, :] - x[0, :],
                                                                                          pd - x[1, :])
    pdd = phat[2, :] * sd ** 2 + phat[1, :] * sdd
    sddd = ks * ks * math.exp(-ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2) * 4 * (
        np.dot(phat[0, :] - x[0, :], pd - x[1, :])) ** 2 - ks * 2 * math.exp(
        -ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2) * (np.dot(pd - x[1, :], pd - x[1, :]) +
                                                               np.dot(phat[0, :] - x[0, :],
                                                                      pdd - x[2, :]) * dist_between_nodes(pdd,
                                                                                                          x[2, :]) +
                                                               np.dot(phat[0, :] - x[0, :], pd - x[1, :]))
    pddd = phat[3, :] * sd ** 3 + 3 * phat[2, :] * sd * sdd + phat[1, :] * sddd
    phatnew = np.vstack([phat[0, :], pd, pdd, pddd])
    s = s + sd * dt
    return phatnew, s


def calculate_midpoint(point_0, point_1, flat=False):
    mid_x = float(point_0[0] + point_1[0]) / 2
    mid_y = float(point_0[1] + point_1[1]) / 2

    if flat is True:
        middle_point = np.array([mid_x, mid_y])
    else:
        mid_z = float(point_0[2] + point_1[2]) / 2
        middle_point = np.array([mid_x, mid_y, mid_z])

    return middle_point


def cost_of_path(path):
    tot_distance = 0
    # print ("path shape: ", path)
    for i in range(path.shape[0]):
        if i < (path.shape[0] - 1):
            # print ("point i: ", path[i, :, :][0])
            k = i + 1
            # print ("point K: ", path[k, :, :][0])
            # dist = dist_between_nodes(path[i, :, :][0], path[k, :, :][0])
            point_i = path[i]
            point_k = path[k]
            dist = dist_between_nodes(point_i, point_k)
            # print("dist: ", dist)
            tot_distance += dist
        else:
            break
            # print ("total distance", tot_distance)
    return tot_distance


def spline_return(points, dim=3, deg=5):
    array_of_points = np.zeros((5, points.shape[1]))
    coefficients = np.array([])
    array_of_points[2] = calculate_midpoint(points[0], points[1])
    array_of_points[4] = points[1]
    for i in range(array_of_points.shape[0]):
        if i == 1 or i == 3:
            array_of_points[i] = calculate_midpoint(array_of_points[i - 1], array_of_points[i + 1])
            continue
        if i == 2 or i == 4:
            continue
        array_of_points[i] = points[i]

    dist = cost_of_path(array_of_points)
    velocity = dist / TIME
    time = TIME
    if velocity > MAX_VELOCITY:
        time = dist / MAX_VELOCITY

    time_array = np.linspace(0, time, array_of_points.shape[0])
    # print("time : ", time_array)
    # print("points: ", array_of_points[:, 0])
    for i in range(dim):
        coeffs = np.expand_dims(np.polyfit(time_array, array_of_points[:, i], deg=deg), 1)
        # print("coeffs:", coeffs.shape)
        if coefficients.shape[0] == 0:
            coefficients = coeffs
        else:
            coefficients = np.append(coefficients, coeffs, axis=1)
            # print("coefficients: ", coefficients)
    points = extract_points((coefficients, deg + 1, time_array))
    return points