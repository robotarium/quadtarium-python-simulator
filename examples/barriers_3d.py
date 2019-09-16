# !/usr/bin/env python
from utilities_sim.robotarium_simulation_builder import RobotariumEnvironment
from utilities_sim.interpolation import spline_interpolation, extract_points, parameterize_time_waypoint_generator
import numpy as np
from math import cos, sin
from control import acker
''' name: Barriers Demo
    author: Christopher Banks
    date: 09/15/2019
    description: Showcases the barrier function utility used in the Robotarium to ensure collision free trajectories.
    A total of five (5) quadcopters are shown in this simulation where four (4) are given commands to fly in a
    circle. The one (1) remaining quadcopter is instructed to fly in straight lines through the circle formation, 
    facilitating colliding behavior. Barrier functions around each quadcopter ensure there is a region of space surrounding
    each quadcopter that remains collision free, maintaining safe flight throughout the experiment.'''


if __name__ == "__main__":
    # start the robotarium environment
    robotarium = RobotariumEnvironment(barriers=False, save_data=False)

    # declare the number of agents
    robotarium.number_of_agents = 5
    t_real = 0
    radii = 0.8
    dt = 0.02
    s = 0
    p_hat = dict()
    u_hat = dict()
    x_state = dict()
    xd = dict()


    desired_poses = np.zeros((robotarium.number_of_agents, 3))
    init_flag = False
    flag = False
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    b = np.array([[0], [0], [0], [1]])
    Kb = acker(A, b, [-12.2, -12.4, -12.6, -12.8])

    waypoints = np.array([[-1.1, -1.1, -0.8],
                          [0.9, 0.9, -0.8],
                          [-1, 0.9, -0.8],
                          [0.9, -0.9, -0.8],
                          [-0.9, -0.9, -0.8]])

    initial_points = np.array([[1.0, 0.0, -0.8],
                               [0.5, 0.0, -0.8],
                               [0.0, 0.0, -0.8],
                               [-0.5, 0.0, -0.8],
                               [-1.0, 0.0, -0.8]])

    interval = np.array([5, 3, 5, 3, 5])
    time_total = np.sum(interval)
    wp_ind = 0
    t_count = 0


    # set initial desired poses (optional)
    robotarium.initial_poses = np.copy(initial_points)

    # build the robotarium environment (must be called)
    robotarium.build()
    goal_x = dict()

    if t_real == 0:
        for i in range(robotarium.number_of_agents):
            x_state[i] = np.zeros((4, 3))
            x_state[i][0, :] = robotarium.poses[i]

    desired_dist = np.zeros(robotarium.number_of_agents)
    while not t_real*dt > int(time_total):
        try:
            for i in range(robotarium.number_of_agents):
                if i == 4:
                    if t_real == 0:
                        coefficient_info = spline_interpolation(np.array([robotarium.poses[i], waypoints[wp_ind]]), total_time=interval[wp_ind])
                        points = extract_points(coefficient_info, dt)
                        p_hat[i] = points[t_count]
                    elif wp_ind > waypoints.shape[0]:
                        p_hat[i] = points[-1]
                    elif np.linalg.norm((robotarium.poses[i] - waypoints[wp_ind])) < 0.1:
                        t_count = 0
                        coefficient_info = spline_interpolation(np.array([waypoints[wp_ind], waypoints[wp_ind + 1]]), total_time=interval[wp_ind])
                        points = extract_points(coefficient_info, dt)
                        p_hat[i] = points[t_count]
                        wp_ind += 1
                    elif t_count < points.shape[0]:
                        p_hat[i] = points[t_count]
                    else:
                        p_hat[i] = points[-1]
                else:
                    om = 0.5*np.pi
                    th = t_real*dt*om + i*np.pi/2
                    p_hat[i] = np.array([[radii*cos(th), radii*sin(th), -0.8],
                                         [-radii*om*sin(th), radii*om*cos(th), 0],
                                         [-radii*om**2*cos(th), -radii*om**2*sin(th), 0],
                                         [radii*om**3*sin(th), -radii*om**3*cos(th), 0]])

                u_hat[i] = p_hat[i][3, :] - np.dot(Kb, x_state[i] - p_hat[i])
                if np.linalg.norm(u_hat[i]) > 1e4:
                    u_hat[i] = u_hat[i]/np.linalg.norm(u_hat[i])*1e4
            u = robotarium.Safe_Barrier_3D(x_state, u_hat)
            for i in range(robotarium.number_of_agents):
                xd[i] = np.dot(A, x_state[i]) + np.dot(b, u[i])
                x_state[i] = x_state[i] + xd[i]*dt
                desired_poses[i] = x_state[i][0]

            # set desired poses (must call)
            robotarium.set_desired_poses(desired_poses)


            # update the poses in the robotarium (must call)
            robotarium.update_poses()
            t_real += 1
            t_count += 1

        except KeyboardInterrupt:
            print("Interrupt Occurred")
            break
    else:
        curr_dist = np.linalg.norm(robotarium.poses - initial_points)
        while np.linalg.norm(curr_dist - desired_dist) > 0.05:
            if flag is False:
                for i in range(robotarium.number_of_agents):
                    goal_x[i] = np.zeros((4, 3))
                    goal_x[i][0, :] = initial_points[i]
                flag = True

            for i in range(robotarium.number_of_agents):
                u_hat[i] = goal_x[i][3, :] - np.dot(Kb, x_state[i] - goal_x[i])
            u = robotarium.Safe_Barrier_3D(x_state, u_hat)

            for i in range(robotarium.number_of_agents):
                xd[i] = np.dot(A, x_state[i]) + np.dot(b, u[i])
                x_state[i] = x_state[i] + xd[i]*dt
                desired_poses[i] = x_state[i][0]
            robotarium.set_desired_poses(desired_poses)
            robotarium.update_poses()
            curr_dist = np.linalg.norm(robotarium.poses - initial_points)
        print("-----Experiment completed-----")
