from utilities_sim.robotarium_simulation_builder import RobotariumEnvironment
from utilities_sim.interpolation import spline_interpolation, extract_points
from utilities_sim.actuation import gen_chain_of_integrators
import numpy as np
from math import cos, sin

''' name: Barriers Demo
    authors: Christopher Banks and Yousef Emam
    date: 09/28/2020
    description: Showcases the barrier function utility used in the Robotarium to ensure collision free trajectories.
    A total of five (5) quadcopters are shown in this simulation where four (4) are given commands to fly in a
    circle. The one (1) remaining quadcopter is instructed to fly in straight lines through the circle formation,
    facilitating colliding behavior. Barrier functions around each quadcopter ensure there is a region of space surrounding
    each quadcopter that remains collision free, maintaining safe flight throughout the experiment.'''


if __name__ == "__main__":

    # start the robotarium environment
    robotarium = RobotariumEnvironment(number_of_agents=5, barriers=True)

    radii = 0.8  # radius of the circle
    dt = 0.02  # size of time-step
    p_hat = dict()  # desired state of chain of integrators
    u_hat = dict()  # nominal input for chain of integrators
    x_state = dict()  # actual state of chain of integrators
    desired_poses = np.zeros((robotarium.number_of_agents, 3))  # desired poses

    # Obtain Chain of integrator Transition matrices and Controller Gains
    A, b, Kb = gen_chain_of_integrators()

    # Waypoints for the last quadrotor
    waypoints = np.array([[-1.1, -1.1, 0.8],
                          [0.9, 0.9, 0.8],
                          [-1, 0.9, 0.8],
                          [0.9, -0.9, 0.8],
                          [-0.9, -0.9, 0.8]])

    # Initial xyz poses
    initial_points = np.array([[1.0, 0.0, 0.8],
                               [0.5, 0.0, 0.8],
                               [0.0, 0.0, 0.8],
                               [-0.5, 0.0, 0.8],
                               [-1.0, 0.0, 0.8]])

    # Time intervals for each waypoint
    interval = np.array([5, 3, 5, 3, 5])
    time_total = np.sum(interval)  # total time of experiment
    wp_ind = 0  # current waypoint index for the last robot
    t_count = 0

    # set initial desired poses (optional)
    robotarium.initial_poses = np.copy(initial_points)
    # build the robotarium environment (must be called)
    robotarium.build()

    # Initialize chain of integrator states
    for i in range(robotarium.number_of_agents):
        x_state[i] = np.zeros((4, 3))
        x_state[i][0, :] = robotarium.poses[i]

    for t_real in range(int(time_total/dt)):

        for i in range(robotarium.number_of_agents):
            # The last quad is travelling across the quads that are circling
            if i == 4:
                if t_real == 0:
                    coefficient_info = spline_interpolation(np.array([robotarium.poses[i], waypoints[wp_ind]]), total_time=interval[wp_ind])
                    points = extract_points(coefficient_info, dt)
                    p_hat[i] = points[t_count]
                elif wp_ind > waypoints.shape[0]:
                    p_hat[i] = points[-1]
                elif np.linalg.norm((robotarium.poses[i] - waypoints[wp_ind])) < 0.11:
                    t_count = 0
                    coefficient_info = spline_interpolation(np.array([waypoints[wp_ind], waypoints[wp_ind + 1]]), total_time=interval[wp_ind])
                    points = extract_points(coefficient_info, dt)
                    p_hat[i] = points[t_count]
                    wp_ind += 1
                elif t_count < points.shape[0]:
                    p_hat[i] = points[t_count]
                else:
                    p_hat[i] = points[-1]
            # The first four quads are simply circling
            else:
                om = 0.5*np.pi
                th = t_real*dt*om + i*np.pi/2
                p_hat[i] = np.array([[radii*cos(th), radii*sin(th), 0.8],
                                     [-radii*om*sin(th), radii*om*cos(th), 0],
                                     [-radii*om**2*cos(th), -radii*om**2*sin(th), 0],
                                     [radii*om**3*sin(th), -radii*om**3*cos(th), 0]])
            # Obtain nominal control (snap) to control chain of integrators
            u_hat[i] = p_hat[i][3, :] - np.dot(Kb, x_state[i] - p_hat[i])
            # Bound the nominal input
            if np.linalg.norm(u_hat[i]) > 1e4:
                u_hat[i] = u_hat[i]/np.linalg.norm(u_hat[i])*1e4
        # Update state of single integrator for each quad
        for i in range(robotarium.number_of_agents):
            xd = np.dot(A, x_state[i]) + np.dot(b, u_hat[i])
            x_state[i] = x_state[i] + xd*dt
            desired_poses[i] = x_state[i][0]
        # set desired poses (must call)
        robotarium.set_desired_poses(desired_poses)
        # update the poses in the robotarium (must call)
        robotarium.update_poses()
        t_count += 1

    # After we're done circling, we tell the quadcopters to return to their initial positions
    # initialize goal poses (which are the initial poses)
    goal_x = dict()
    for i in range(robotarium.number_of_agents):
        goal_x[i] = np.zeros((4, 3))
        goal_x[i][0, :] = initial_points[i]

    while np.any(np.linalg.norm(robotarium.poses - initial_points) > 0.050):
        # Generate control input
        for i in range(robotarium.number_of_agents):
            u_hat[i] = goal_x[i][3, :] - np.dot(Kb, x_state[i] - goal_x[i])
            xd = np.dot(A, x_state[i]) + np.dot(b, u_hat[i])
            x_state[i] = x_state[i] + xd*dt
            desired_poses[i] = x_state[i][0]

        robotarium.set_desired_poses(desired_poses)
        robotarium.update_poses()

    # Save experiment data (time, position, orientation, input)
    robotarium.save_data()
