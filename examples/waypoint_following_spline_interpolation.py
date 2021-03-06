from utilities_sim.robotarium_simulation_builder import RobotariumEnvironment
from utilities_sim.interpolation import spline_interpolation, extract_points
import numpy as np

''' name: Spline Interpolation Demo
    authors: Christopher Banks and Yousef Emam
    date: 09/28/2020
    description: A simple go-to-spline example. Quadcopters are controlled by allowing users to specify points to the 
    quadcopters in the form of a numpy array. A spline interpolation program finds splines that satisfy the constraints
    of the quadcopter and generates an n-differentiable function. Users can use the spline interpolation function to generate
    splines from a sequence of waypoints and send these points to the Robotarium.'''

if __name__ == "__main__":

    # creates robotarium object, indicate if user would like to use barriers for safety
    robotarium = RobotariumEnvironment(number_of_agents=3, barriers=True)

    # if specific initial poses are desired, set robotarium.initial_poses
    # robotarium.initial_poses = np.array([[0.1, 0.1, -0.5], [1, 0.3, -1.3], [1, 1, -1.2]])

    # instantiates Robotarium and initializes quadcopters at random poses if initial_poses is not set
    x_desired = np.array([[[0.2, 0.2, 0.5], [0.5, -0.5, 0.8], [0.2, 0.2, 0.9], [1, 0.4, 1.3]],
                         [[0.4, 0.4, 0.7], [0.1, 0.1, 0.4], [-1.1, 0.2, .3], [-0.1, -0.3, 0.6]],
                         [[1.1, 1.2, 1.2], [0.2, 0.5, 0.8], [-0.1, -0.3, 0.6], [-0.7, 0.3, 0.7]]])

    SPLINE_COUNT = 300
    robotarium.initial_poses = np.array([])
    point_spline_for_quads = dict()
    index_update = dict()
    init_pose = np.zeros((robotarium.number_of_agents, 3))

    # Generate spline coefficients
    for i in range(robotarium.number_of_agents):
        points = np.stack((x_desired[i]))
        spline_coeffs_for_quad = spline_interpolation(points, total_time=5)
        spline_for_quads = extract_points(spline_coeffs_for_quad)
        point_spline_for_quads[i] = spline_for_quads
        init_pose[i] = point_spline_for_quads[i][0][0, :]
        index_update[i] = 0

    # specify initial poses and build the robotarium object
    robotarium.initial_poses = init_pose
    robotarium.build()

    point_update = np.zeros((robotarium.number_of_agents, 3))
    all_splines_satisfied = 0

    for i in range(SPLINE_COUNT):
        # retrieve quadcopter poses (numpy array, n x m x 3) where n is the quadcopter index
        for j in range(robotarium.number_of_agents):
            desired_point = point_spline_for_quads[j][index_update[j]]
            if index_update[j] < point_spline_for_quads[j].shape[0] - 1:
                index_update[j] += 1
            else:
                index_update[j] = point_spline_for_quads[j].shape[0] - 1
                all_splines_satisfied += 1
            point_update[j] = desired_point[0, :]
        # Set desired pose
        robotarium.set_desired_poses(point_update)
        # send pose commands to robotarium
        robotarium.update_poses()






