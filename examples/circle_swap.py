import numpy as np
from utilities_sim import robotarium_simulation_builder

''' name: Point-to-Point Demo 
    author: Yousef Emam 
    date: 09/28/2020
    description: N quadrotors are initialized to position themselves around a circle. 
    They then repeatedly attempt to traverse to the opposite side of the circle. '''

TIMEOUT = False

if __name__ == "__main__":

    # number of iterations the experiment will run for
    COUNT = 900
    # Number of agents
    N = 3
    # creates robotarium object, indicate if user would like to save data
    robotarium = robotarium_simulation_builder.RobotariumEnvironment(barriers=True, number_of_agents=N)
    # if specific initial poses are desired, set robotarium.initial_poses (np.array of size Nx3)
    # robotarium.initial_poses = ....
    # instantiates Robotarium and initializes quadcopters at random poses if initial_poses is not set
    robotarium.build()

    # this code ensures that the agents are initially distributed around an ellipse
    xybound = 1.0 * np.array([-1, 1, -1, 1])
    p_theta = np.arange(1, 2 * N, 2) / (2 * N) * 2 * np.pi
    p_circ = np.array([[xybound[1] * np.cos(p_theta), xybound[1] * np.cos(p_theta + np.pi)],
                       [xybound[3] * np.sin(p_theta), xybound[3] * np.sin(p_theta + np.pi)]])
    p_circ = np.transpose(np.append(p_circ, 0.8*np.ones((1, 2, N)), axis=0), (2, 0, 1))

    # Initialize Goal
    x_goal = p_circ[:, :, 0]

    # Plot targets
    robotarium.robotarium_simulator_plot.scatter3D(p_circ[:, 0, 0], p_circ[:, 1, 0], p_circ[:, 2, 0], cmap='Greens')
    robotarium.robotarium_simulator_plot.scatter3D(p_circ[:, 0, 1], p_circ[:, 1, 1], p_circ[:, 2, 1], cmap='Greens')

    # Flag of task completion
    flag = 0

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







