#!/usr/bin/env python
import numpy as np
import math
MAX_VELOCITY = 0.5
TIME = 1

''' name: Actuation File
    author: Christopher Banks & Yousef Emam
    date: 07/20/2020
    description: Contains files for generating motion in simulation/experiment.'''

def dist_between_nodes(node1_val, node2_val):
    sum_v = 0
    for dim in range(node1_val.shape[0]):
        diff = (node1_val[dim] - node2_val[dim]) ** 2
        sum_v = sum_v + diff
    dist = np.sqrt(sum_v)
    return dist

def spline_interpolation(points, time_interval=None, total_time=None):
    # points (n x 3) array x, y, z points  #3 points
    # have at least 4 initial points

    # Ensure that trajectory has at least 4 points, if not generate some using midpoints.
    if points.shape[0] == 2:
        new_points = np.zeros((4, 3))
        new_points[0] = points[0]
        new_points[3] = points[1]
        dist = np.linalg.norm(new_points[3] - new_points[0], 2) * 1/2 #TODO: this 1/2 should go but it keeps the same speed as before (i.e. sim is too slow if not)
        if dist == 0:
            raise Exception('Waypoint provided is the same as current pose.')
        new_points[1] = 1/2 * (new_points[3] - new_points[0]) + new_points[0]
        new_points[2] = 3/4 * (new_points[3] - new_points[0]) + new_points[0]
        points = new_points.copy()
    elif points.shape[0] == 3:
        raise Exception('Spline Interpolation for 3 points not supported.')
    else:
        dist = 0
        for i in range(0, points.shape[0]-1):
            dist += np.linalg.norm((points[i] - points[i+1]), 2)

    # Calculate the time intervals for the spline
    if time_interval is None:
        if total_time is None:
            velocity = np.min([float(dist) / TIME, MAX_VELOCITY])
            total_time = dist / velocity
        time_interval = np.linspace(0, total_time, points.shape[0])

    degree = 5
    num_of_polys = points.shape[0] - 1  # 3
    a_matrix = np.zeros(((degree + 1) * num_of_polys, (degree + 1) * num_of_polys))  # 18 x 18
    num_of_coeffs = int(float(a_matrix.shape[1]) / float(num_of_polys))  # 6
    b_vector = np.zeros(((degree + 1) * num_of_polys, 1))
    coefficients = np.array([])

    for dim in range(points.shape[1]):
        w = points[:, dim]
        t = time_interval
        i = 0
        for poly in range(num_of_polys):#TODO: Go over this, what are we doing? Are these if statements necessary?
            if poly == 0:
                poly_index_0 = poly * (degree + 1)
                a_matrix[poly_index_0, 0:num_of_coeffs] = get_standard_poly(t[i], ord=6, der=0)
                a_matrix[poly_index_0 + 1, 0:num_of_coeffs] = get_standard_poly(t[i], ord=6, der=1)


                # first deriv
                a_matrix[poly_index_0 + 2, 0:num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=1)
                # second deriv
                a_matrix[poly_index_0 + 3, 0:num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=2)
                # third deriv
                a_matrix[poly_index_0 + 4, 0:num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=3)
                # fourth deriv
                a_matrix[poly_index_0 + 5, 0:num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=4)
                # fifth deriv condition
                a_matrix[poly_index_0 + 6, 0:num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=5)  # [120, 0, 0..]

                # Insert negative of first to fifth derivatives
                a_matrix[poly_index_0 + 2: poly_index_0 + 7, num_of_coeffs:2*num_of_coeffs] = - a_matrix[poly_index_0 + 2: poly_index_0 + 7, :num_of_coeffs].copy()


                a_matrix[poly_index_0 + 7, 0:num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=0)

                b_vector[poly_index_0] = w[i]
                b_vector[poly_index_0 + 7] = w[i + 1]

            elif poly == (num_of_polys - 1):
                a_matrix[(poly + 1) * (degree + 1) - 4, (degree + 1) * (poly + 1) - num_of_coeffs:] = get_standard_poly(t[i], ord=6, der=0)
                a_matrix[(poly + 1) * (degree + 1) - 3, (degree + 1) * (poly + 1) - num_of_coeffs:] = get_standard_poly(t[i+1], ord=6, der=2)
                a_matrix[(poly + 1) * (degree + 1) - 2,
                (((degree + 1) * (poly + 1) - num_of_coeffs) - num_of_coeffs):(degree + 1) * (
                        poly + 1) - num_of_coeffs] = get_standard_poly(t[i], ord=6, der=5)
                a_matrix[(poly + 1) * (degree + 1) - 2, (degree + 1) * (poly + 1) - num_of_coeffs:] = - get_standard_poly(t[i], ord=6, der=5)
                a_matrix[(poly + 1) * (degree + 1) - 1, (degree + 1) * (poly + 1) - num_of_coeffs:] = get_standard_poly(t[i+1], ord=6, der=0)
                b_vector[(poly + 1) * (degree + 1) - 4] = w[i]
                b_vector[(poly + 1) * (degree + 1) - 1] = w[i + 1]

            elif poly < (num_of_polys - 1):
                poly_index = poly * (degree + 1) + 2
                a_matrix[poly_index, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_standard_poly(t[i], ord=6, der=0)
                # first deriv
                a_matrix[poly_index + 1, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=1)
                a_matrix[poly_index + 1,
                (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_standard_poly(t[i+1], ord=6, der=1)
                # second deriv
                a_matrix[poly_index + 2, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=2)
                a_matrix[poly_index + 2, (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_standard_poly(t[i+1], ord=6, der=2)
                # third deriv
                a_matrix[poly_index + 3, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=3)
                a_matrix[poly_index + 3, (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_standard_poly(t[i+1], ord=6, der=3)
                # fourth deriv
                a_matrix[poly_index + 4, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=4)
                a_matrix[poly_index + 4, (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_standard_poly(t[i+1], ord=6, der=4)
                # 0th derivative
                a_matrix[poly_index + 5, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_standard_poly(t[i+1], ord=6, der=0)

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
    points = np.zeros((0, 4, 3))
    # For each time interval
    for interval in range(0, time_interval.shape[0]-1):
        # Grab the corresponding coefficients
        curr_coeffs = coefficients[interval * num_of_coeffs:num_of_coeffs * (interval + 1)]
        # Compute the time stamps for the entire interval
        if interval == 0:
            t = np.arange(0, time_interval[interval+1], dt)
        else:
            t = np.arange(time_interval[interval], time_interval[interval+1], dt) #TODO: What should we use, t[-1] + dt or t_int[int]?
        points_ = np.zeros((t.shape[0], 4, 3))
        # Compute Points for the interval
        for dim in range(curr_coeffs.shape[1]):
            dim_coeffs = curr_coeffs[:, dim]
            points_[:, 0, dim] = dim_coeffs[0] * t ** 5 + dim_coeffs[1] * t ** 4 + dim_coeffs[2] * t ** 3 + dim_coeffs[
                3] * t ** 2 + dim_coeffs[4] * t + dim_coeffs[5]
            points_[:, 1, dim] = 5 * dim_coeffs[0] * t ** 4 + 4 * dim_coeffs[1] * t ** 3 + 3 * dim_coeffs[
                2] * t ** 2 + 2 * dim_coeffs[3] * t + dim_coeffs[4]
            points_[:, 2, dim] = 20 * dim_coeffs[0] * t ** 3 + 12 * dim_coeffs[1] * t ** 2 + 6 * dim_coeffs[2] * t + 2 * \
                            dim_coeffs[3]
            points_[:, 3, dim] = 60 * dim_coeffs[0] * t ** 2 + 24 * dim_coeffs[1] * t + 6 * dim_coeffs[2]

        points = np.append(points, points_, axis=0)

    return points

def calculate_midpoint(point_0, point_1, flat=False):

    #TODO: Check input dimensionalities to this function
    #return (point_0 + point_1) / 2
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
    for i in range(path.shape[0]):
        if i < (path.shape[0] - 1):
            k = i + 1
            point_i = path[i]
            point_k = path[k]
            dist = dist_between_nodes(point_i, point_k)
            tot_distance += dist
        else:
            break
    return tot_distance


# def spline_return(points, dim=3, deg=5):
#     array_of_points = np.zeros((5, points.shape[1]))
#     coefficients = np.array([])
#     array_of_points[2] = calculate_midpoint(points[0], points[1])
#     array_of_points[4] = points[1]
#     for i in range(array_of_points.shape[0]):
#         if i == 1 or i == 3:
#             array_of_points[i] = calculate_midpoint(array_of_points[i - 1], array_of_points[i + 1])
#             continue
#         if i == 2 or i == 4:
#             continue
#         array_of_points[i] = points[i]
#
#     dist = cost_of_path(array_of_points)
#     velocity = dist / TIME
#     time = TIME
#     if velocity > MAX_VELOCITY:
#         time = dist / MAX_VELOCITY
#
#     time_array = np.linspace(0, time, array_of_points.shape[0])
#     for i in range(dim):
#         coeffs = np.expand_dims(np.polyfit(time_array, array_of_points[:, i], deg=deg), 1)
#         if coefficients.shape[0] == 0:
#             coefficients = coeffs
#         else:
#             coefficients = np.append(coefficients, coeffs, axis=1)
#     points = extract_points((coefficients, deg + 1, time_array))
#     return points

# def parameterize_time_waypoint_generator(phat, x, s, dt):
#     ks = 10
#     sd = math.exp(-ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2)  # paramed time dynamics
#     pd = phat[1, :] * sd
#     sdd = -ks * math.exp(-ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2) * 2 * np.dot(phat[0, :] - x[0, :], pd - x[1, :])
#     pdd = phat[2, :] * sd ** 2 + phat[1, :] * sdd
#     sddd = ks * ks * math.exp(-ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2) * 4 * (
#         np.dot(phat[0, :] - x[0, :], pd - x[1, :])) ** 2 - ks * 2 * math.exp(
#         -ks * dist_between_nodes(phat[0, :], x[0, :]) ** 2) * (np.dot(pd - x[1, :], pd - x[1, :]) + np.dot(phat[0, :] - x[0, :],
#                                                                                                            pdd - x[2, :]) * dist_between_nodes(pdd, x[2, :]) + np.dot(phat[0, :] - x[0, :], pd - x[1, :]))
#     pddd = phat[3, :] * sd ** 3 + 3 * phat[2, :] * sd * sdd + phat[1, :] * sddd
#     phatnew = np.vstack([phat[0, :], pd, pdd, pddd])
#     s = s + sd * dt
#     return phatnew, s

def get_standard_poly(t, ord=6, der=0):
    """Generates polynomials of standard form with all coeffs = 1 if derivative (der) = 0.
    E.g. (ord=4, der=0):
        p =    t**3  + t**2 + t + 1
        (ord=3, der=1):
        p = 3*(t**2) + 2*t  + 1 + 0

    Args:
        t (float): The value the polynomial is built around
        ord (int): order of the polynomial
        der (int): derivative (1 is first derivative, etc...)

    Returns:
        p (np.array): 1-D array of size (ord,) containing the polynomial

    """

    if der > ord:
        return np.zeros(ord)

    # Initialize Exponents Ignoring derivative order for now [ord, ord-1, ... 0]
    exponents = np.flip(np.arange(0, ord))

    coeffs = np.ones(ord)
    for i in range(coeffs.shape[0]):
        coeffs[i] *= np.prod(exponents[i:i+der])

    # Bases are all identically = t
    bases = t * np.ones(ord)

    # Increment Exponents according to chain rule
    exponents = np.clip(exponents - der, 0, None)

    return coeffs * (bases ** exponents)
