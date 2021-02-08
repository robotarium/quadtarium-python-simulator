import numpy as np
from utilities_sim import robotarium_simulation_builder

""" 
name: Point-to-Point Demo
author: Christopher Banks and Yousef Emam
date: 09/28/2020
description: A simple go to point example. Quadcopters are controlled by allowing users to specify points to the 
quadcopters in the form of a numpy array. A spline interpolation program finds splines that satisfy the constraints
of the quadcopter and generates an n-differentiable function.
"""

TIMEOUT = False

if __name__ == "__main__":

    # creates robotarium object, indicate if user would like to use barrier functions for safety
    robotarium = robotarium_simulation_builder.RobotariumEnvironment(number_of_agents=1, barriers=True)

    # if specific initial poses are desired, set robotarium.initial_poses
    robotarium.initial_poses = np.array([[0.8, 0.7, 0.8]])

    # instantiates Robotarium and initializes quadcopters at random poses if initial_poses is not set
    robotarium.build()

    x_desired = np.array([[1.0, -0.9, 0.7]])  # desired position to go to
    t = 0  # current iteration
    COUNT = 500  # Total number of desired iterations

    while t < COUNT:
        # retrieve quadcopter poses (numpy array, n x m x 3) where n is the quadcopter index

        # Insert your code here!!

        # Set desired pose
        robotarium.set_desired_poses(x_desired)

        # send pose commands to robotarium
        robotarium.update_poses()

        # Increment current iteration (t)
        t += 1

    # If the user wants to save the trajectory data
    robotarium.save_data()





