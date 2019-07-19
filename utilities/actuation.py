#!/usr/bin/env python

from math import atan2, sqrt
import numpy as np
import math

MAX_VEL = 0.5
MAX_ACC = 0.05
MAX_J = 0.005
T = 0.02

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


def projection_controller(p_now, p_future):
    # pose and vel are good
    #print("p before: ", p_now)
    #print("p future: ", p_future)
    p_des = np.zeros((4, 3))
    vel = delta_func(p_future, p_now[0, :])
    #print("vel: ", vel)
    vel_pre = np.linalg.norm(vel)
    #print("vel pre: ", vel_pre)
    if np.sum(vel_pre) > 0:
        if vel_pre >= MAX_VEL:
            #print("GREATER THAN MAX VEL")
            vel_max = MAX_VEL*np.divide(vel, vel_pre)
            p = p_now[0, :] + vel_max*T
            #print("p: ", p)
            #print("p now: ", p_now[0, :])
            #print("vel max: ", vel_max)
            v = vel_max
        else:
            p_des[0, :] = p_future
            p = p_now[0, :] + vel * T
            v = vel
        #print("v: ", v)
        delta_v = delta_func(v, p_now[1, :])
        len_del_v = np.linalg.norm(delta_v)
        a = MAX_ACC*np.divide(delta_v, len_del_v)
        #print("a: ", a)
        delta_a = delta_func(a, p_now[2, :])
        len_del_a = np.linalg.norm(delta_a)
        j = MAX_J*np.divide(delta_a, len_del_a)
    else:
        p = p_future
        v = np.zeros(3)
        a = np.zeros(3)
        j = np.zeros(3)
    #print("j: ", j)
    p_des[0, :] = p
    p_des[1, :] = v
    p_des[2, :] = a
    p_des[3, :] = j
    #print("p after: ", p_des)

    return p_des


# if __name__ == "__main__":
#    print("spline interpolation")
#    points = np.array([[1, 2, 4], [3, 4, 4], [5, 5, 5]])
#
#    coeffs = spline_interpolation(points, total_time=100)
#    print("coeefs: ", coeffs)