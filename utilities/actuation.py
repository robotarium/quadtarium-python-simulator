#!/usr/bin/env python

from math import atan2, sqrt
import numpy as np
import math

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


def projection_controller(p_now, p_future, max_vel=0.6, dt=0.02):
    p_des = np.zeros((4, 3))
    vel = delta_func(p_future, p_now[0, :])
    #print("vel: ", vel)
    vel_pre = np.linalg.norm(vel)
    print(" vel pre: ", vel_pre)
    if np.linalg.norm(vel_pre) >= max_vel:
        vel_magnitude = np.linalg.norm(vel)
        print("***MAXX VEL***")
        vel = max_vel*(np.divide(vel, vel_magnitude))
        print("new vel: ", vel)
        new_x = p_now[0, :] + vel * dt
        print("new goal:", new_x)
        p_des[0, :] = new_x
    else:
        p_des[0, :] = p_future
    delta_v = delta_func(vel, p_now[1, :])
    vel_new = p_now[1, :] + delta_v
    p_des[1, :] = vel_new
    print("vel now: ", np.linalg.norm(p_des[1, :]))
    delta_a = delta_func(delta_v, p_now[2, :])
    a_new = p_now[2, :] + delta_a
    p_des[2, :] = a_new
    delta_jerk = delta_func(delta_a, p_now[3, :])
    jerk_new = p_now[3, :] + delta_jerk
    p_des[3, :] = jerk_new
     # print("p_des: ", p_des)
    return p_des


# if __name__ == "__main__":
#    print("spline interpolation")
#    points = np.array([[1, 2, 4], [3, 4, 4], [5, 5, 5]])
#
#    coeffs = spline_interpolation(points, total_time=100)
#    print("coeefs: ", coeffs)