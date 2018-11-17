import numpy as np
from utilities import quadcopter_plot
# plots the quadcopter

class RobotariumCommunication(object):
    def __init__(self, robotarium_sim_environment, index):
        self.name = 'crazyflie_{0}'.format(index)
        self.first_flag = True
        self.sim_env = robotarium_sim_environment
        self.quadcopter_communicate = None
        self.my_pose = None
        self.orientation = np.array([])
        self.thrust_hover = 34000 #arbitrary value

    def set_initial_random_pose(self):
        pose_x = (1.3 - (-1.3))*np.random.sample() + (-1.3)
        pose_y = (1.3 - (-1.3))*np.random.sample() + (-1.3)
        pose_z = 0
        pose = np.array([pose_x, pose_y, pose_z])
        return pose

    def get_init_pose(self):
        if self.first_flag is True:
            self.first_flag = False
            pose = self.set_initial_random_pose()
            self.quadcopter_communicate = quadcopter_plot.QuadPlotObject(self.sim_env, pose)
            self.orientation = np.zeros((1, 3))
            return pose, self.orientation

    def set_init_pose(self, initial_pose):
        if self.first_flag is True:
            self.first_flag = False
            self.quadcopter_communicate = quadcopter_plot.QuadPlotObject(self.sim_env, initial_pose)
            self.orientation = np.zeros((1, 3))
            return initial_pose, self.orientation

    def set_pose(self, pose, sim_env, roll=0, pitch=0, yaw=0, thrust=0):
        self.quadcopter_communicate.update(sim_env, pose, roll, pitch, yaw)
        self.my_pose = pose
        self.orientation = np.array([roll, pitch, yaw])


    def get_pose_and_orientation(self):
        return self.my_pose, self.orientation