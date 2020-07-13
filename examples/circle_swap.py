# !/usr/bin/env python
# Examples Go-to-Point File
# Christopher Banks
# 10/14/18
# This file provides an example of how to send commands to the Robotarium for moving a number of quadcopters from one pose
# to another.

import numpy as np
from utilities_sim import robotarium_simulation_builder
'''name: Point-to-Point Demo author: Yousef Emam date: 07/15/2020 description: N quadrotors are initialized to 
position themselves around a circle. They then repeatedly attempt to traverse to the opposite side of the circle. '''



TIMEOUT = False

if __name__ == "__main__":

    # number of iterations the experiment will run for
    COUNT = 900
    # Number of agents
    N = 3
    # creates robotarium object, indicate if user would like to save data
    robotarium = robotarium_simulation_builder.RobotariumEnvironment(barriers=True, number_of_agents=N, save_data=False)

    # this code ensures that the agents are initially distributed around an ellipse
    xybound = np.array([-2.0, 2.0, -2.0, 2.0])
    p_theta = np.arange(1, 2 * N, 2) / (2 * N) * 2 * np.pi
    p_circ = np.array([[xybound[1] * np.cos(p_theta), xybound[1] * np.cos(p_theta + np.pi)],
                       [xybound[3] * np.sin(p_theta), xybound[3] * np.sin(p_theta + np.pi)]])
    p_circ = np.transpose(np.append(p_circ, -0.8*np.ones((1, 2, N)), axis=0), (2, 0, 1))

    # Initialize Goal
    x_goal = p_circ[:, :, 0]

    # Flag of task completion
    flag = 0

    # if specific initial poses are desired, set robotarium.initial_poses
    # robotarium.initial_poses = np.array([[0.8, 0.7, -0.8]])

    # instantiates Robotarium and initializes quadcopters at random poses if initial_poses is not set
    robotarium.build()

    t = 0

    while t < COUNT:

        # retrieve quadcopter poses (numpy array, n x 3) where n is the quadcopter index
        x = robotarium.get_quadcopter_poses()

        # Let's make sure we're close enough to the goals
        if np.linalg.norm(x_goal[:, :2] - x[:, :2], 1) < 0.03:
            flag = 1 - flag

        if flag == 0:
            x_goal = p_circ[:, :, 0]
        else:
            x_goal = p_circ[:, :, 1]


        # Set desired pose
        robotarium.set_desired_poses(x_goal)
        # send pose commands to robotarium
        robotarium.update_poses()

        t += 1


    #robotarium.save_data()





