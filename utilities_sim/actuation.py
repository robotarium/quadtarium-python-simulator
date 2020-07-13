#!/usr/bin/env python

from math import atan2, sqrt
import numpy as np
from utilities_sim.interpolation import spline_interpolation, extract_points
''' name: Actuation File
    author: Christopher Banks
    date: 09/15/2019
    description: Contains files for generating motion in simulation/experiment.'''

def invert_diff_flat_output(x, thrust_hover=0):
    """Given the full state of the robot, find the control inputs necessary to control the quadcopter in the experiment
    using differential flatness.

    Args:
        x (ndarray): desired state of size (4,3)
        thrust_hover (float): thrust required for hovering (TODO: is this correct?)

    Returns:
        roll (float):
        pitch (float):
        yaw (float): Fixed to zero
        thrust (float):
    """

    m = 35.89 / 1000  # mass of the quad in kg
    g = 9.8  # gravity constant

    beta1 = - x[2, 0]  # acceleration in the x direction
    beta2 = - x[2, 2] + 9.8  # acceleration in the z direction
    beta3 = x[2, 1]  # acceleration in the y direction

    roll = atan2(beta3, sqrt(beta1 ** 2 + beta2 ** 2))
    pitch = atan2(beta1, beta2)
    yaw = 0.0  # NOTE: yaw is fixed!
    a_temp = np.linalg.norm(np.array([0, 0, g]) - x[2, :])
    # acc g correspond to 49000, at thrust_hover
    thrust = (a_temp / g) * thrust_hover  # changed from int val

    return roll, pitch, yaw, thrust

def gen_splines(p_now, p_future):
    """Finds a n-differentiable function from p_now to p_future which respects the Robotarium imposed constraints such
    as max velocities.

    Args:
        p_now (ndarray): current x,y,z position of size (3,)
        p_future (ndarray): desired x,y,z position of size (3,)

    Returns:
        traj (ndarray): planned trajectory of size (n_points, 4, 3)

    """
    points = np.stack((p_now, p_future), axis=0)
    traj_coeffs = spline_interpolation(points)
    traj = extract_points(traj_coeffs)
    return traj


# TODO: not working
def vel_back_step(x_state, vel_prev, vel_des, vel_des_prev, dt=0.02):
    v = x_state[1, :]
    dv_dt = (v - vel_prev) / dt
    dv_dt_2 = dv_dt*dv_dt

    d_ves_dt = (vel_des - vel_des_prev) / dt
    d_ves_dt_2 = d_ves_dt*d_ves_dt

    k_1 = np.array([6, 6, 6])
    u_3 = -k_1*(v-vel_des)*dv_dt_2 - k_1*(v-vel_des)*d_ves_dt_2
    return u_3