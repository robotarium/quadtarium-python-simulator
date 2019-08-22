# !/usr/bin/env python
# Examples Go-to-Point File
# Christopher Banks
# 10/14/18
# This file provides an example of how to send commands to the Robotarium for moving a number of quadcopters from one pose
# to another.
import sys
import os
import numpy as np
from utilities_sim import robotarium_simulation_builder

TIMEOUT = False

if __name__ == "__main__":
    # creates robotarium object, indictate if user would like to save data
    robotarium = robotarium_simulation_builder.RobotariumEnvironment(save_data=True)

    # robotarium object sets a random number of agents to be created

    # sets the number of agents
    robotarium.number_of_agents = 1

    # iterate until time limit is reached, the max time our quadcopters can run is 5 minutes so experiments will be
    # limited to that time

    # Iteration method is arbitrary, we will provide a function to check if experiment will timeout however

    # if specific initial poses are desired, set robotarium.initial_poses
    # robotarium.initial_poses = np.array([[0.8, 0.7, -0.8]])
    # instantiates Robotarium and initializes quadcopters at random poses if initial_poses is not set
    robotarium.build()

    x_desired = np.array([[0.8, 1.0, -1.2]])

    i = 0
    COUNT = 1000
    while i < COUNT:
        # retrieve quadcopter poses (numpy array, n x m x 3) where n is the quadcopter index

        # Insert your code here!!

        # Set desired pose
        robotarium.set_desired_poses(x_desired)
        # send pose commands to robotarium
        robotarium.update_poses()
        i +=1
    robotarium.save_data()





