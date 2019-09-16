#!/usr/bin/env python

from math import atan2, sqrt
import numpy as np
from utilities_sim.interpolation import spline_interpolation, extract_points
''' name: Actuation File
    author: Christopher Banks
    date: 09/15/2019
    description: Contains files for generating motion in simulation/experiment.'''

# invert_diff_flat_output
# input: state, thrust
# output: roll, pitch, yawrate, thrust
# description: given the full state of the robot, find the control inputs necessary to control
# the quadcopter in the experiment using differential flatness.
def invert_diff_flat_output(x, thrust_hover=0):

    m = 35.89 / 1000
    g = 9.8

    beta1 = - x[2, 0]
    beta2 = - x[2, 2] + 9.8
    beta3 = x[2, 1]

    roll = atan2(beta3, sqrt(beta1 ** 2 + beta2 ** 2))
    pitch = atan2(beta1, beta2)
    yaw = 0.0  # NOTE: yaw is fixed!
    a_temp = np.linalg.norm(np.array([0, 0, g]) - x[2, :])
    # acc g correspond to 49000, at thrust_hover
    thrust = (a_temp / g) * thrust_hover  # changed from int val

    return roll, pitch, yaw, thrust

# gen splines
# input: current point, future point
# output: the returned full trajectory
# Finds a n-differentiable function from p_now to p_future under Robotarium imposed constraints (e.g. max velocities)
def gen_splines(p_now, p_future):
    points = np.stack((p_now, p_future), axis=0)
    traj_coeffs = spline_interpolation(points)
    traj = extract_points(traj_coeffs)
    return traj


# not working
def vel_back_step(x_state, vel_prev, vel_des, vel_des_prev, dt=0.02):
    v = x_state[1, :]
    dv_dt = (v - vel_prev) / dt
    dv_dt_2 = dv_dt*dv_dt

    d_ves_dt = (vel_des - vel_des_prev) / dt
    d_ves_dt_2 = d_ves_dt*d_ves_dt

    k_1 = np.array([6, 6, 6])
    u_3 = -k_1*(v-vel_des)*dv_dt_2 - k_1*(v-vel_des)*d_ves_dt_2
    return u_3