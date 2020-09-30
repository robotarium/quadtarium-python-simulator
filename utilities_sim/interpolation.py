#!/usr/bin/env python
import numpy as np

MAX_VELOCITY = 0.5
TIME = 1

''' name: Interpolation File
    authors: Christopher Banks and Yousef Emam
    date: 09/28/2020
    description: Contains helper functions for generating the spline as per the Mellinger controller.'''


def spline_interpolation(points, time_interval=None, total_time=None):
    """Perform the spline interpolation to reach all the input points given the time intervals or the total time.

    Parameters
    ----------
    points : ndarray
           Array of xyz points to be reached. The array is 2 dimensional and has size (num_points, 3). Must have at
           least 4, if not, more points will be automatically generated.
    time_interval : ndarray, optional
                 Time interval to reach each of the specified points.
    total_time : float, optional
               Total time to reach all points.

    Returns
    -------
    coefficients  : ndarray
                  The coefficients of the spline.
    num_of_coeffs : int
                 Number of coefficients.
    time_interval : ndarray
                 The time intervals desired (same as input if specified).
    """

    # Ensure that trajectory has at least 4 points, if not generate some using midpoints.
    if points.shape[0] == 2:
        new_points = np.zeros((4, 3))
        new_points[0] = points[0]
        new_points[3] = points[1]
        dist = np.linalg.norm(new_points[3] - new_points[0], 2) * 1/2  #TODO: this 1/2 should go but it keeps the same speed as before (i.e. sim is too slow if not)
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

    degree = 5  # degree of the polynomial to be generated
    num_of_polys = points.shape[0] - 1  # number of polynomials (1 between each 2 desired points, typically 3 total)
    a_matrix = np.zeros(((degree + 1) * num_of_polys, (degree + 1) * num_of_polys))  # 18 x 18
    num_of_coeffs = int(float(a_matrix.shape[1]) / float(num_of_polys))  # 6
    b_vector = np.zeros(((degree + 1) * num_of_polys, 1))
    coefficients = np.array([])

    #TODO: Clean this up...
    for dim in range(points.shape[1]):
        w = points[:, dim]
        t = time_interval
        for poly in range(num_of_polys):
            # First Polynomial
            if poly == 0:
                a_matrix[0, 0:num_of_coeffs] = get_unit_poly(t[poly], ord=6, der=0)
                a_matrix[1, 0:num_of_coeffs] = get_unit_poly(t[poly], ord=6, der=1)
                # first deriv
                a_matrix[2, 0:num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=1)
                # second deriv
                a_matrix[3, 0:num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=2)
                # third deriv
                a_matrix[4, 0:num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=3)
                # fourth deriv
                a_matrix[5, 0:num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=4)
                # fifth deriv condition
                a_matrix[6, 0:num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=5)  # [120, 0, 0..]
                # Insert negative of first to fifth derivatives
                a_matrix[2:7, num_of_coeffs:2*num_of_coeffs] = -a_matrix[2:7, :num_of_coeffs].copy()
                a_matrix[7, 0:num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=0)
                b_vector[0] = w[poly]
                b_vector[7] = w[poly + 1]
            # Middle Polynomials
            elif poly == (num_of_polys - 1):
                poly_index = (poly + 1) * (degree + 1)
                a_matrix[poly_index - 4, poly_index - num_of_coeffs:] = get_unit_poly(t[poly], ord=6, der=0)
                a_matrix[poly_index - 3, poly_index - num_of_coeffs:] = get_unit_poly(t[poly + 1], ord=6, der=2)
                a_matrix[poly_index - 2, poly_index - 2*num_of_coeffs:poly_index - num_of_coeffs] = get_unit_poly(t[poly], ord=6, der=5)
                a_matrix[poly_index - 2, poly_index - num_of_coeffs:] = -get_unit_poly(t[poly], ord=6, der=5)
                a_matrix[poly_index - 1, poly_index - num_of_coeffs:] = get_unit_poly(t[poly + 1], ord=6, der=0)
                b_vector[poly_index - 4] = w[poly]
                b_vector[poly_index - 1] = w[poly + 1]
            # Last Polynomial
            elif poly < (num_of_polys - 1):
                poly_index = poly * (degree + 1) + 2
                a_matrix[poly_index, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_unit_poly(t[poly], ord=6, der=0)
                # first deriv
                a_matrix[poly_index + 1, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=1)
                a_matrix[poly_index + 1,
                (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_unit_poly(t[poly + 1], ord=6, der=1)
                # second deriv
                a_matrix[poly_index + 2, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=2)
                a_matrix[poly_index + 2, (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_unit_poly(t[poly + 1], ord=6, der=2)
                # third deriv
                a_matrix[poly_index + 3, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=3)
                a_matrix[poly_index + 3, (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_unit_poly(t[poly + 1], ord=6, der=3)
                # fourth deriv
                a_matrix[poly_index + 4, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=4)
                a_matrix[poly_index + 4, (poly + 1) * num_of_coeffs:(poly + 1) * num_of_coeffs + num_of_coeffs] = -get_unit_poly(t[poly + 1], ord=6, der=4)
                # 0th derivative
                a_matrix[poly_index + 5, poly * num_of_coeffs:poly * num_of_coeffs + num_of_coeffs] = get_unit_poly(t[poly + 1], ord=6, der=0)
                b_vector[poly_index] = w[poly]
                b_vector[poly_index + 5] = w[poly + 1]

        coeffs = np.linalg.solve(a_matrix, b_vector)

        if coefficients.shape[0] == 0:
            coefficients = coeffs
        else:
            coefficients = np.append(coefficients, coeffs, axis=1)

    return coefficients, num_of_coeffs, time_interval

def extract_points(coefficients_info, dt=0.02):
    """Extract the xyz poses with derivatives given the spline coefficients.

    Parameters
    ----------
    coefficients_info : tuple
                      (coefficients, num_of_coeffs, time_interval) as returned by spline_interpolation.
    dt : float, optional
       timestep

    Returns
    -------
    points : ndarray
           xyz poses to track including the derivatives. Three dimensional array of size (num_points, 4, 3).
    """

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

def get_unit_poly(t, ord=6, der=0):
    """Generates polynomials of standard form with all coeffs = 1 if derivative (der) = 0.
    E.g. (ord=4, der=0):
        p =    t**3  + t**2 + t + 1
        (ord=3, der=1):
        p = 3*(t**2) + 2*t  + 1 + 0

    Parameters
    ----------
        t : float
            The value the polynomial is built around
        ord : int, optional
            Order of the polynomial
        der : int, optional
            Derivative (1 is first derivative, etc...)

    Returns
    -------
        p : ndarray
            1-D array of size (ord,) containing the polynomial
    """

    if der > ord:
        return np.zeros(ord)

    # Initialize Exponents Ignoring derivative order for now [ord, ord-1, ... 0]
    exponents = np.arange(ord-1, -1, -1)

    coeffs = np.ones(ord)
    for i in range(coeffs.shape[0]):
        coeffs[i] *= np.prod(exponents[i:i+der])

    # Bases are all identically = t
    bases = t * np.ones(ord)

    # Increment Exponents according to chain rule
    exponents = np.clip(exponents - der, 0, None)

    return coeffs * (bases ** exponents)
