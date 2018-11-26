#!/usr/bin/env python
from utilities.robotarium_simulation_builder import RobotariumEnvironment
from utilities.interpolation import spline_interpolation, extract_points
import numpy as np
from math import cos, sin
from control import acker

if __name__ == "__main__":
    robotarium = RobotariumEnvironment(barriers=False)
    robotarium.number_of_agents = 5
    t_real = 0
    # time_total = 30
    radii = 0.5
    dt = 0.02
    p_hat = dict()
    u_hat = dict()
    x_state = dict()
    xd = dict()
    desired_poses = np.zeros((robotarium.number_of_agents, 3))
    flag = False
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1],
                  [0, 0, 0, 0]])
    b = np.array([[0], [0], [0], [1]])
    Kb = acker(A, b, [-12.8, -12.6, -12.4, -12.2])
    waypoints = np.array([[-0.9, -0.9, -0.8], [0.9, 0.7, -0.8],
                          [0.9, -0.7, -0.8], [-0.9, 0.9, -0.8],
                          [-0.7, -0.9, -0.8], [1.1, 0.4, -0.8]])

    end_points = np.array([[1.0, 0.0, -0.8],
                           [0.5, 0.0, -0.8],
                           [0.0, 0.0, -0.8],
                           [-0.5, 0.0, -0.8],
                           [-1.0, 0.0, -0.8]])

    interval = np.array([0, 5, 8, 13, 19, 24])
    time_total = interval[-1]
    coefficient_info = spline_interpolation(waypoints, time_interval=interval)
    points = extract_points(coefficient_info, dt)
    print("points:", points.shape)
    robotarium.build()
    while not t_real == int(time_total/dt):
        if t_real == 0:
            for i in range(robotarium.number_of_agents):
                x_state[i] = np.zeros((4, 3))
                x_state[i][0, :] = robotarium.poses[i]
        try:
            for i in range(robotarium.number_of_agents):
                if i == 4:
                    p_hat[i] = points[t_real]
                else:
                    om = 0.5*np.pi
                    th = t_real*dt*om + i*np.pi/2
                    p_hat[i] = np.array([[radii*cos(th), radii*sin(th), -0.8],
                                         [-radii*om*sin(th), radii*om*cos(th), 0],
                                         [-radii*om**2*cos(th), -radii*om**2*sin(th), 0],
                                         [radii*om**3*sin(th), -radii*om**3*cos(th), 0]])

                print("phat: ", p_hat[i])
                u_hat[i] = p_hat[i][3, :] - np.dot(Kb, x_state[i] - p_hat[i])
                if np.linalg.norm(u_hat[i]) > 1e4:
                    u_hat[i] = u_hat[i]/np.linalg.norm(u_hat[i])*1e4
            u = robotarium.Safe_Barrier_3D(x_state, u_hat)
            for i in range(robotarium.number_of_agents):
                xd[i] = np.dot(A, x_state[i]) + np.dot(b, u[i])
                x_state[i] = x_state[i] + xd[i]*dt
                desired_poses[i] = x_state[i][0]
            robotarium.set_desired_poses(desired_poses)
            robotarium.update_poses()
            t_real += 1
        except KeyboardInterrupt:
            print("Interrupt Occurred")
            break
    print("diff from end points: ", np.linalg.norm(robotarium.poses - end_points))
    goal_x = dict()
    while np.linalg.norm(robotarium.poses - end_points) > 0.05:
        print("drive system to end goal")
        if flag == False:
            for i in range(robotarium.number_of_agents):
                goal_x[i] = np.zeros((4, 3))
                goal_x[i][0, :] = end_points[i]
        print("goal: ", goal_x)
        print("u: ", u_hat)
        print("x: ", x_state)
        for i in range(robotarium.number_of_agents):
            u_hat[i] = goal_x[i][3, :] - np.dot(Kb, x_state[i] - goal_x[i])
        u = robotarium.Safe_Barrier_3D(x_state, u_hat)
        for i in range(robotarium.number_of_agents):
            xd[i] = np.dot(A, x_state[i]) + np.dot(b, u[i])
            x_state[i] = x_state[i] + xd[i]*dt
            desired_poses[i] = x_state[i][0]
        flag = True
        robotarium.set_desired_poses(desired_poses)
        robotarium.update_poses()
    print("-----Experiment completed-----")
