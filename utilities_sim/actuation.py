from math import atan2, sqrt
import numpy as np
from utilities_sim.interpolation import spline_interpolation, extract_points
from control import acker

''' name: Actuation File
    author: Christopher Banks and Yousef Emam
    date: 09/28/2020
    description: Contains files for generating motion in simulation/experiment.'''


def gen_chain_of_integrators():
    """Defines a chain of integrator for xyz: position, velocity, acceleration and jerk.
       Dynamics of the form: xd = Ax + bu

    Returns
    -------
    A : ndarray
    b : ndarray
    Kb : ndarray
    """

    # Define chain of integrator dynamics and controller gains.
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    b = np.array([[0], [0], [0], [1]])
    Kb = np.array(acker(A, b, [-12.2, -12.4, -12.6, -12.8]))  # Generate gains using pole placement.
    return A, b, Kb

def invert_diff_flat_output(x, thrust_hover=0):
    """Given the xyz state of the robot, find the control inputs necessary to control the quadcopter in the experiment
    using differential flatness. No longer used by the simulator.

    Parameters
    ----------
    x : ndarray
        desired xyz state of the quad. Two dimensional array of size (4,3).
    thrust_hover : float, optional
                amount of thrust needed to hover (#TODO:units?)
    Returns
    -------
    roll : float
    pitch : float
    yaw :  float
        As of now, yaw is fixed to 0.
    thrust : float
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

    Parameters
    ----------
        p_now (ndarray): current x,y,z position of size (3,)
        p_future (ndarray): desired x,y,z position of size (3,)

    Returns
    -------
        traj (ndarray): planned trajectory of size (n_points, 4, 3)

    """
    points = np.stack((p_now, p_future), axis=0)
    traj_coeffs = spline_interpolation(points)
    traj = extract_points(traj_coeffs)
    return traj


# TODO: not working
def vel_back_step(x_state, vel_prev, vel_des, vel_des_prev, dt=0.02):
    """#TODO: What is this for?

    Args:
        x_state (ndarray): current state of size (4,3)
        vel_prev (ndarray): previous velocity of size (3,)
        vel_des (ndarray): current desired velocity of size (3,)
        vel_des_prev (ndarray): previous desired velocity of size (3,)
        dt (float): time step

    Returns:
        u_3 (ndarray): #TODO: what is this?

    """

    v = x_state[1, :]
    dv_dt = (v - vel_prev) / dt
    dv_dt_2 = dv_dt*dv_dt

    d_ves_dt = (vel_des - vel_des_prev) / dt
    d_ves_dt_2 = d_ves_dt*d_ves_dt

    k_1 = np.array([6, 6, 6])
    u_3 = -k_1*(v-vel_des)*dv_dt_2 - k_1*(v-vel_des)*d_ves_dt_2

    return u_3
