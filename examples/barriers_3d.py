# !/usr/bin/env python
from utilities_sim.robotarium_simulation_builder import RobotariumEnvironment
from utilities_sim.interpolation import spline_interpolation, extract_points, parameterize_time_waypoint_generator
import numpy as np
from math import cos, sin
from control import acker

if __name__ == "__main__":
    robotarium = RobotariumEnvironment(barriers=False)
    robotarium.number_of_agents = 5
    t_real = 0
    # time_total = 30
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
    robotarium.initial_poses = np.copy(initial_points)

    robotarium.build()
    goal_x = dict()

    if t_real == 0:
        for i in range(robotarium.number_of_agents):
            x_state[i] = np.zeros((4, 3))
            x_state[i][0, :] = robotarium.poses[i]

    # move quads to desired start points
    desired_dist = np.zeros(robotarium.number_of_agents)
    # curr_dist = np.linalg.norm(robotarium.poses - initial_points)
    # while np.linalg.norm(curr_dist - desired_dist) > 0.05:
    #     curr_dist = np.linalg.norm(robotarium.poses - initial_points)
    #     robotarium.set_desired_poses(initial_points)
    #     robotarium.update_poses()

    while not t_real*dt > int(time_total):
        try:
            for i in range(robotarium.number_of_agents):
                if i == 4:
                    # print("dist to waypoint: ", np.linalg.norm((robotarium.poses[i] - waypoints[wp_ind])))
                    # print("next waypoint: ", waypoints[wp_ind])
                    #print("current pose: ", robotarium.poses[i])
                    #print("x state: ", x_state[i][0, :])
                    if t_real == 0:
                        coefficient_info = spline_interpolation(np.array([robotarium.poses[i], waypoints[wp_ind]]), total_time=interval[wp_ind])
                        points = extract_points(coefficient_info, dt)
                        p_hat[i] = points[t_count]
                    elif wp_ind > waypoints.shape[0]:
                        # print("waypoint index too great")
                        p_hat[i] = points[-1]
                    elif np.linalg.norm((robotarium.poses[i] - waypoints[wp_ind])) < 0.1:
                        # print("switch to next waypoint")
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

                # print("phat: ", p_hat[i])
                # p_hat[i], s = parameterize_time_waypoint_generator(p_hat[i], x_state[i],s, dt)
                u_hat[i] = p_hat[i][3, :] - np.dot(Kb, x_state[i] - p_hat[i])
                if np.linalg.norm(u_hat[i]) > 1e4:
                    u_hat[i] = u_hat[i]/np.linalg.norm(u_hat[i])*1e4
            u = robotarium.Safe_Barrier_3D(x_state, u_hat)
            for i in range(robotarium.number_of_agents):
                xd[i] = np.dot(A, x_state[i]) + np.dot(b, u[i])
                x_state[i] = x_state[i] + xd[i]*dt
                desired_poses[i] = x_state[i][0]

            #print("desired poses: ", desired_poses)
            # print("current poses: ", robotarium.poses)
            robotarium.set_desired_poses(desired_poses)
            robotarium.update_poses()
            t_real += 1
            t_count += 1

        except KeyboardInterrupt:
            print("Interrupt Occurred")
            break
    else:
        #print("poses: ", robotarium.poses)
        #print("initi: ", initial_points)
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
            # print("u : ", u)

            for i in range(robotarium.number_of_agents):
                xd[i] = np.dot(A, x_state[i]) + np.dot(b, u[i])
                x_state[i] = x_state[i] + xd[i]*dt
                desired_poses[i] = x_state[i][0]
            robotarium.set_desired_poses(desired_poses)
            robotarium.update_poses()
            curr_dist = np.linalg.norm(robotarium.poses - initial_points)
        print("-----Experiment completed-----")
