import numpy as np
from utilities_exp.robotarium_simulation_builder import RobotariumEnvironment
from utilities_exp.interpolation import spline_interpolation, extract_points

if __name__ == "__main__":
    print("leader follower")
    number_of_bots = 3
    time_to_takeff = 2.0
    waypoint_time = 3.0
    x_state = dict()
    x_desired = dict()
    x_d = dict()
    u = dict()
    p_hat = dict()
    x_d_update = dict()
    leader_points = np.array([])
    dt = 0.02
    time_total = 500
    gamma = 0.3
    kappa = 0.5
    alpha = 0.2
    formation_gain = 0.1

    initial_points = np.array([[1.0, 0.0, -0.8],
                               [0.5, 0.0, -0.8],
                               [0.0, 0.0, -0.8],
                               [-0.5, 0.0, -0.8],
                               [-1.0, 0.0, -0.8]])

    #goal points
    desired_poses = np.array([[1, 1, -1.1],
                              [-1, 1, -1],
                              [-1, -1, -0.8],
                              [1, -1, -0.8]])
    # simulation stuff

    robotarium = RobotariumEnvironment(barriers=False)
    robotarium.number_of_agents = number_of_bots
    robotarium.initial_poses = initial_points
    t_real = 0
    robotarium.build()

    for i in range(robotarium.number_of_agents):
        x_state[i] = np.zeros((1, 3))
        x_state[i] = robotarium.poses[i]
        x_desired[i] = robotarium.poses[i]
        x_d[i] = np.zeros((1, 3))

    print("state shape: ", np.expand_dims(x_state[0], 0).shape)
    print("desired poses: ", desired_poses.shape)
    leader_points = np.concatenate((np.expand_dims(x_state[0], 0), desired_poses))
    print("leader points: ", leader_points)
    leader_spline_co = spline_interpolation(leader_points)
    leader_spline = extract_points(leader_spline_co)
    print("spline: ", leader_spline)
    k = 0
    max_k = leader_spline.shape[0] - 1

    while not k*dt > time_total:
        # leader points
        print("t real : ", k*dt)
        # if k < leader_spline.shape[0] - 1: # and np.linalg.norm([x_state[0] - leader_spline[k][0]]) < 0.02:
        #     x_d[0] = kappa*(leader_spline[k][0] - x_state[0])
        #     k += 1
        #
        # elif k >= leader_spline.shape[0] - 1:
        #     x_d[0] = kappa*(leader_spline[max_k][0] - x_state[0])

        #x_d[i] = np.zeros((1, 3))
        for i in range(robotarium.number_of_agents):
            #x_i_rel = np.linalg.norm([x_state[i] - x_state[0]])
            for j in range(robotarium.number_of_agents):
                #x_j_rel = np.linalg.norm([x_state[j] - x_state[0]], axis=0)
                #print("x_j: ", x_j_rel)
                dist_i_j = x_state[i] - x_state[j]
                print("dist i j: ", dist_i_j)
                print("i: ", i)
                print("j: ", j)
                # x_d[i] += ((x_j_rel - x_i_rel)**2 - gamma)*(x_j_rel - x_i_rel)**2 + ((dist_i_j)**2 - alpha)*(dist_i_j)**2
                x_d[i] += ((dist_i_j)**2 - 0)*(dist_i_j)
            # x_d[i] += (x_i_rel**2 - alpha)*(x_i_rel)**2
            x_d[i] = formation_gain*x_d[i]
        for i in range(robotarium.number_of_agents):
            x_state[i] = x_state[i] + x_d[i] * dt
            print("robot: {0} \n x_state: {1} \n x_d: {2}".format(i, x_state[i], x_d[i]))
            desired_poses[i] = x_state[i]
        robotarium.set_desired_poses(desired_poses)
        robotarium.update_poses()
