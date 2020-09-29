from utilities_sim.robotarium_simulation_builder import RobotariumEnvironment
from utilities_sim.actuation import gen_chain_of_integrators
import numpy as np
from math import cos, sin

''' name: Quads Circling Demo
    authors: Christopher Banks and Yousef Emam
    date: 09/28/2020
    description: Quads circling demo. Four quadcopters circle in the xy-plane around the origin.
    quadcopters trajectory.'''


if __name__ == "__main__":

    # start the robotarium environment
    robotarium = RobotariumEnvironment(number_of_agents=4, barriers=False)
    time_total = 21.0  # Total Time of the experiment
    dt = 0.02  # size of time-step
    p_hat = dict()  # desired state of chain of integrators
    u_hat = dict()  # nominal input for chain of integrators
    x_state = dict()  # actual state of chain of integrators

    desired_poses = np.zeros((robotarium.number_of_agents, 3))
    init_flag = False

    # Define chain of integrator dynamics and controller gains.
    A, b, Kb = gen_chain_of_integrators()

    # Initial position of the quads
    initial_points = np.array([[-1.0, 1.0, 0.6],
                               [1.0, 1.0, 0.6],
                               [1.0, -1.0, 0.6],
                               [-1., -1.0, 0.6],
                               [0.0, 0.5, 0.8],
                               [0.0, -0.5, 0.8]])

    # set initial desired poses (optional)
    robotarium.initial_poses = np.copy(initial_points)

    # build the robotarium environment (must be called)
    robotarium.build()
    goal_x = dict()

    # Initialize single integrator states
    for i in range(robotarium.number_of_agents):
        x_state[i] = np.zeros((4, 3))
        x_state[i][0, :] = robotarium.poses[i]

    radii = 0.8  # circle radius
    for t in range(int(time_total/dt)):
        for i in range(robotarium.number_of_agents):
            om = 0.5 * np.pi
            th = t * dt * om + i * np.pi / 2  # Desired theta
            # Desired State
            p_hat[i] = np.array([[radii * cos(th), radii * sin(th), 0.6],
                                 [-radii * om * sin(th), radii * om * cos(th), 0],
                                 [-radii * om ** 2 * cos(th), -radii * om ** 2 * sin(th), 0],
                                 [radii * om ** 3 * sin(th), -radii * om ** 3 * cos(th), 0]])
            u_hat[i] = p_hat[i][3, :] - np.dot(Kb, x_state[i] - p_hat[i])
            # Bound input snap
            if np.linalg.norm(u_hat[i]) > 1e4:
                u_hat[i] = u_hat[i]/np.linalg.norm(u_hat[i])*1e4

        # Minimally alter snap to make trajectory safe using Barrier Functions
        u = robotarium.Safe_Barrier_3D(x_state, u_hat)
        # Generate desired xyz poses for quadcopters to reach
        for i in range(robotarium.number_of_agents):
            xd = np.dot(A, x_state[i]) + np.dot(b, u[i])
            x_state[i] = x_state[i] + xd*dt
            desired_poses[i] = x_state[i][0, :]
        # set desired poses (must call)
        robotarium.set_desired_poses(desired_poses)
        # update the poses in the robotarium (must call)
        robotarium.update_poses()

    # After we're done circling, we tell the quadcopters to return to their initial positions
    # initialize goal poses (which are the initial poses)
    for i in range(robotarium.number_of_agents):
        goal_x[i] = np.zeros((4, 3))
        goal_x[i][0, :] = initial_points[i]
    while np.any(np.linalg.norm(robotarium.poses - initial_points) > 0.05):
        # Generate control input
        for i in range(robotarium.number_of_agents):
            u_hat[i] = goal_x[i][3, :] - np.dot(Kb, x_state[i] - goal_x[i])
        u = robotarium.Safe_Barrier_3D(x_state, u_hat)
        for i in range(robotarium.number_of_agents):
            xd = np.dot(A, x_state[i]) + np.dot(b, u[i])
            x_state[i] = x_state[i] + xd*dt
            desired_poses[i] = x_state[i][0]

        robotarium.set_desired_poses(desired_poses)
        robotarium.update_poses()
