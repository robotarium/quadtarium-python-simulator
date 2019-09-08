#!/usr/bin/env python

from math import atan2, sqrt
import numpy as np
import math
from control import acker
from utilities_sim.interpolation import spline_interpolation, extract_points
MAX_VEL = 0.5
MAX_ACC = 0.05
MAX_J = 0.005
T = 0.02

z_2 = np.zeros((1, 3))
z_3 = np.zeros((1, 3))

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

def normalize(x):
    # nomalize a vector
    temp = sqrt(sum(x ** 2) * 1.0)
    if (temp == 0):
        return x
    else:
        return x / temp


def delta_func(q_goal, q):
    u = np.subtract(q_goal, q)
    return u


def gen_splines(p_now, p_future):
    print("shape p_now: ", p_now)
    print("shape p_future: ", p_future)
    points = np.stack((p_now, p_future), axis=0)
    traj_coeffs = spline_interpolation(points)
    traj = extract_points(traj_coeffs)
    return traj

def vel_back_step(x_state, vel_prev, vel_des, vel_des_prev, dt=0.02):
    v = x_state[1, :]
    dv_dt = (v - vel_prev) / dt
    dv_dt_2 = dv_dt*dv_dt

    d_ves_dt = (vel_des - vel_des_prev) / dt
    d_ves_dt_2 = d_ves_dt*d_ves_dt

    k_1 = np.array([6, 6, 6])
    print("dv_dt: ", dv_dt)
    print("dves_dt: ", d_ves_dt)
    u_3 = -k_1*(v-vel_des)*dv_dt_2 - k_1*(v-vel_des)*d_ves_dt_2

    print("u3: ", u_3)
    return u_3



# if __name__ == "__main__":
#    print("spline interpolation")
#    points = np.array([[1, 2, 4], [3, 4, 4], [5, 5, 5]])
#
#    coeffs = spline_interpolation(points, total_time=100)
#    print("coeefs: ", coeffs)