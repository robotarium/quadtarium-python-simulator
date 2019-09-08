#!/usr/bin/env python
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as Dim3
from control import acker
from cvxopt import matrix, solvers
import time
import pickle
from utilities_sim.robotarium_communication_interface import RobotariumCommunication
from utilities_sim.actuation import invert_diff_flat_output, vel_back_step, gen_splines


TIMEOUT_FLAG = False
TIMEOUT_TIME = 30

class RobotariumEnvironment(object):
    def __init__(self, save_data=True, barriers=True):
        self.number_of_agents = rand.randint(3, 10)
        self.initial_poses = np.array([])
        self.poses = np.array([])
        self.desired_poses = np.zeros((self.number_of_agents, 3))
        self.desired_vels = np.zeros((self.number_of_agents, 3))
        self.crazyflie_objects = {}
        self.time = time.time()
        self.x_state = dict()
        self.vel_prev = dict()
        self.des_vel_prev = dict()
        self.xd = dict()
        self.u = dict()
        self.orientation_real = dict()
        self.pose_real = dict()
        self.dt = 0.03
        self.count = 0
        self.time_record = dict()
        self.x_record = dict()
        self.input_record = dict()
        self.orientation_record = dict()
        self.AA = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        self.bb = np.array([[0], [0], [0], [1]])
        self.Kb = np.asarray(acker(self.AA, self.bb, [-12.2, -12.4, -12.6, -12.8]))
        self.robotarium_simulator_plot = None
        self.barriers = barriers
        self.save_flag = save_data
        solvers.options['show_progress'] = False

    def get_quadcopter_poses(self):
        poses = np.zeros((self.number_of_agents, 3))
        for i in range(self.number_of_agents):
            pose = self.crazyflie_objects[i].position
            poses[i] = pose
        return poses

    def set_desired_poses(self, poses):
        # sets desired goal points for all quadcopters
        self.desired_poses = poses

    def set_desired_vels(self, vels):
        self.desired_vels = vels

    def build(self):
        # builds robotarium simulator
        # creates quadcopter objects for each quadcopter and plots each quadcopter
        try:
            assert self.number_of_agents > 0

        except AssertionError:
            print("number of agents must be greater than 0!!")
            self.number_of_agents = rand.randint(0, 10)

        self.robotarium_simulator_plot = self.plot_robotarium()
        if len(self.initial_poses) == 0:
            for i in range(self.number_of_agents):
                self.crazyflie_objects[i] = QuadcopterObject(self.robotarium_simulator_plot, index=i)
                self.poses = self.get_quadcopter_poses()
                self.hover_quads_at_initial_poses()
                self.vel_prev[i] = np.zeros((1, 3))
                self.des_vel_prev[i] = np.zeros((1, 3))



        else:
            for i in range(self.number_of_agents):
                self.crazyflie_objects[i] = QuadcopterObject(self.robotarium_simulator_plot, self.initial_poses[i], index=i)
                self.poses = self.initial_poses
                self.x_state[i] = np.zeros((4, 3))
                self.x_state[i][0] = self.poses[i]
                self.vel_prev[i] = np.zeros((1, 3))
                self.des_vel_prev[i] = np.zeros((1, 3))




    def hover_quads_at_initial_poses(self, takeoff_time=10.0):
        print("HOVER QUADS")
        reached_des_flag = np.zeros((self.number_of_agents))
        t0 = time.time()
        self.desired_poses = np.zeros((self.number_of_agents, 3))
        for i in range(self.number_of_agents):
            if len(self.initial_poses) > 0:
                self.desired_poses[i] = self.initial_poses[i]
            else:
                self.desired_poses[i] = np.array([self.poses[i][0], self.poses[i][1], -0.6])

            self.x_state[i] = np.zeros((4, 3))
            self.x_state[i][0] = self.poses[i]

        while np.sum(reached_des_flag) < self.number_of_agents:
            t = time.time()
            s = min((t - t0) / takeoff_time, 1.0)
            #print('HOVER')
            for i in range(self.number_of_agents):
                reached_des_flag[i], self.poses[i] = self.crazyflie_objects[i].hover_bot(self.desired_poses[i], s, self.robotarium_simulator_plot)
            plt.pause(0.001)

        for i in range(self.number_of_agents):
            self.x_state[i] = np.zeros((4, 3))
            self.x_state[i][0] = self.poses[i]



    def update_poses(self, velocities=False):
        # while loop until all quads are at desired new poses
        # iterates over quadcopter objects
        # send desired points to quadcopters
        # make interpolation points from desired point and make max velocity can fly at
        # quadcopter updates dynamics from current point based on interpolation point
        #do one iteration to get to that point, go to next quad
        # update until desired poses are acquired and return flag for each quad that you are done
        # print("update points")
        if velocities is True:
            for i in range(self.number_of_agents):
                #print("goal point: ", self.desired_poses[i])
                print("current: ", self.x_state[i])
                desired_point = vel_back_step(self.x_state[i], self.vel_prev[i], self.desired_vels[i], self.des_vel_prev[i])
                #print("desired point: {0} for robot: {1} \n".format(desired_point, i))
                #self.u[i] = desired_point[3, :] - np.dot(self.Kb, self.x_state[i] - desired_point)
                self.u[i] = desired_point
                print("u: ", self.u[i])
                # print("norm of u:", np.linalg.norm(self.u[i]))
                if np.linalg.norm(self.u[i]) > 1e4:
                    self.u[i] = (self.u[i] / np.linalg.norm(self.u[i])) * 1e4

            if self.barriers is True:
                self.u = self.Safe_Barrier_3D(self.x_state, self.u)

            for i in range(self.number_of_agents):
                #print("u shape: ", self.u[i].shape)
                #print("b: ", self.bb.shape)
                self.xd[i] = np.dot(self.AA, self.x_state[i]) + np.dot(self.bb, self.u[i])
                self.x_state[i] = self.x_state[i] + self.xd[i]*self.dt
                self.crazyflie_objects[i].go_to(self.x_state[i], self.robotarium_simulator_plot)
                self.poses[i] = self.x_state[i][0, :]
                self.pose_real[i], self.orientation_real[i] = self.crazyflie_objects[i].update_pose_and_orientation()
                self.vel_prev[i] = self.x_state[i][1, :]
                self.des_vel_prev[i] = self.desired_vels[i]
            plt.pause(0.02)
            self.time_record[self.count] = str(self.run_time())
            self.x_record[self.count] = self.pose_real
            self.orientation_record[self.count] = self.orientation_real
            self.input_record[self.count] = self.u.copy()
            self.count += 1
        else:
            desired_trajs = dict()
            index = dict()
            end_points = np.array([[]])
            #N = 0
            for i in range(self.number_of_agents):
                desired_trajs[i] = np.zeros((4, 3))
                #print("goal point: ", self.desired_poses[i])
                #print("current: ", self.x_state[i])
                if np.linalg.norm((self.x_state[i][0, :] - self.desired_poses[i])) == 0:
                    desired_trajs[i][0, :] = self.desired_poses[i]
                    desired_trajs[i] = np.stack((desired_trajs[i], desired_trajs[i]),axis=0)
                else:
                    desired_trajs[i] = gen_splines(self.x_state[i][0, :], self.desired_poses[i])
                #print("des points; ", desired_trajs[i])
                if i == 0:
                    end_points = np.array([desired_trajs[i][-1][0, :]])
                    #print("end: ", end_points)
                else:
                    #print("end points: ", end_points.shape)
                    end_points = np.append(end_points, np.array([desired_trajs[i][-1][0, :]]), axis=0)
                index[i] = 0
                #N += desired_trajs[i].shape[0]

                #print("des traj: ", desired_trajs[i][index[i]])
            #print("all end point: ", end_points)
            #print("N: ", N)
            index_sum = 0

            #while index_sum < N:
           # print("N: ", N)
            index_sum = 0
            #print("diff: ", np.linalg.norm((self.poses - end_points)))
            for i in range(self.number_of_agents):
                #if index[i] < desired_trajs[i].shape[0]:
                #print("input u: ", desired_trajs[i][index[i]][3, :])
                self.u[i] = desired_trajs[i][index[i]][3, :] - np.dot(self.Kb, self.x_state[i] - desired_trajs[i][index[i]])
                #else:
                #    self.u[i] = desired_trajs[i][-1][3, :] - np.dot(self.Kb, self.x_state[i] - desired_trajs[i][-1])
                #
                #print("u: ", self.u[i])
                # print("norm of u:", np.linalg.norm(self.u[i]))
                if np.linalg.norm(self.u[i]) > 1e4:
                    self.u[i] = (self.u[i] / np.linalg.norm(self.u[i])) * 1e4
                #index[i] += 1
                #index_sum += index[i]
                #print("sum :", index_sum)

            if self.barriers == True:
                self.u = self.Safe_Barrier_3D(self.x_state, self.u)

            for i in range(self.number_of_agents):
                #print("u shape: ", self.u[i].shape)
                #print("b: ", self.bb.shape)
                self.xd[i] = np.dot(self.AA, self.x_state[i]) + np.dot(self.bb, self.u[i])
                self.x_state[i] = self.x_state[i] + self.xd[i]*self.dt
                self.crazyflie_objects[i].go_to(self.x_state[i], self.robotarium_simulator_plot)
                self.poses[i] = self.x_state[i][0, :]
                self.pose_real[i], self.orientation_real[i] = self.crazyflie_objects[i].update_pose_and_orientation()
               # self.vel_prev[i] = self.x_state[i][1, :]
                #self.des_vel_prev[i] = self.desired_vels[i]
            plt.pause(0.02)
            self.time_record[self.count] = str(self.run_time())
            self.x_record[self.count] = self.pose_real
            self.orientation_record[self.count] = self.orientation_real
            self.input_record[self.count] = self.u.copy()
            self.count += 1




    def check_timeout(self):
        if self.run_time() > TIMEOUT_TIME:
            global TIMEOUT_FLAG
            TIMEOUT_FLAG = True
            return TIMEOUT_FLAG
        else:
            return TIMEOUT_FLAG

    def run_time(self):
        time_now = time.time() - self.time
        return time_now

    def save_data(self):
        time_stamp = time.strftime('%d_%B_%Y_%I:%M%p')
        #print("time:", time_stamp)
        file_n = 'quads_robotarium_'+ time_stamp +'.pckl'
        arrays = [self.time_record, self.x_record, self.orientation_real, self.input_record]
        with open(file_n, 'wb') as file:
            pickle.dump(arrays, file)

    def Safe_Barrier_3D(self, x, u=None, accept_u=True):
        '''Barrier function method: creates a ellipsoid norm around each quadcopter with a z=0.3 meters
        A QP-solver is used to solve the inequality Lgh*(ui-uj) < gamma*h + Lfh. '''
        Kb = self.Kb
        if accept_u is True:
            u = u.copy()
        else:
            u = self.u
        N = len(u)
        zscale = 3
        gamma = 5e-1
        Ds = 0.3
        H = 2 * np.eye(3 * N)
        # print("u :", u)
        f = -2 * np.reshape(np.hstack(u.values()), (3 * N, 1))
        A = np.empty((0, 3 * N))
        b = np.empty((0, 1))
        for i in range(N - 1):
            for j in range(i + 1, N):
                pr = np.multiply(x[i][0, :] - x[j][0, :], np.array([1, 1, 1.0 / zscale]))
                prd = np.multiply(x[i][1, :] - x[j][1, :], np.array([1, 1, 1.0 / zscale]))
                prdd = np.multiply(x[i][2, :] - x[j][2, :], np.array([1, 1, 1.0 / zscale]))
                prddd = np.multiply(x[i][3, :] - x[j][3, :], np.array([1, 1, 1.0 / zscale]))
                h = np.linalg.norm(pr, 4) ** 4 - Ds ** 4
                hd = sum(4 * pr ** 3 * prd)
                hdd = sum(12 * pr ** 2 * prd ** 2 + 4 * pr ** 3 * prdd)
                hddd  = sum(24*pr*prd**3 + 36*pr**2*prd*prdd + 4*pr**3*prddd)
                # Lfh = sum(24*pr*prd**3 + 36*pr ** 2*prd*prdd)
                Lfh = sum(24*pr*prd**3 + 36*pr**2*prd*prdd + 4*pr**3*prddd)
                Lgh = 4 * pr ** 3 * np.array([1, 1, 1.0 / zscale])
                Anew = np.zeros((3 * N,))
                Anew[3 * i:3 * i + 3] = - Lgh
                Anew[3 * j:3 * j + 3] = Lgh
                bnew = gamma * np.dot(Kb, [h, hd, hdd, hddd]) + Lfh
                A = np.vstack([A, Anew])
                b = np.vstack([b, bnew])

        G = np.vstack([A, -np.eye(3 * N), np.eye(3 * N)])
        amax = 1e4
        h = np.vstack([b, amax * np.ones((3 * N, 1)), amax * np.ones((3 * N, 1))])
        sol = solvers.qp(matrix(H), matrix(f), matrix(G), matrix(h))
        x = sol['x']
        for i in range(N):
            u[i] = np.reshape(x[3 * i:3 * i + 3], (1, 3))
        return u

    def plot_robotarium(self):
        fig = plt.figure()
        ax = Dim3.Axes3D(fig)
        ax.set_aspect('equal')
        ax.invert_zaxis()
        ax.set_xlim3d([-1.3, 1.3])
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylim3d([-1.3, 1.3])
        ax.set_ylabel('y', fontsize=10)
        ax.set_zlim3d([0, -1.8])
        ax.set_zlabel('z', fontsize=10)
        ax.tick_params(labelsize=10)
        return ax


class QuadcopterObject(RobotariumCommunication):
    def __init__(self, robotarium_simulator, initial_pose=None, index=0):
        self.position = np.array([])
        self.orientation = np.array([])
        self.velocity = np.array([])
        self.thrust_hover = 0

        super().__init__(robotarium_simulator, str(index))
        if initial_pose is None:
            self.my_pose, self.orientation = self.get_init_pose()
            self.thrust_hover = self.thrust_hover
        else:
            self.my_pose, self.orientation = self.set_init_pose(initial_pose)
            self.thrust_hover = self.thrust_hover
        self.update_pose_and_orientation()

    def hover_bot(self, hover_point, s, sim_env):
        next_pos = np.zeros((3))
        #print("pose" , self.position)
        # print("hove point: ", hover_point)
        dx = hover_point[0] - self.position[0]
        dy = hover_point[1] - self.position[1]
        dz = hover_point[2] - self.position[2]
        next_pos[0] = self.position[0] + s*dx
        next_pos[1] = self.position[1] + s*dy
        next_pos[2] = self.position[2] + s*dz
        self.set_pose(next_pos, sim_env)
        self.position = next_pos
        error = np.linalg.norm((hover_point - self.position))
        if error < 0.1:
            return 1, self.position
        else:
            return 0, self.position

    def go_to(self, desired_pose, sim_env):
        roll, pitch, yaw, thrust = self.set_diff_flat_term(desired_pose)
        self.set_pose(desired_pose[0, :], sim_env, roll, pitch, yaw)
        self.update_pose_and_orientation()


    def set_diff_flat_term(self, goal_pose):
        r, p, y, t = invert_diff_flat_output(goal_pose, thrust_hover=self.thrust_hover)
        return r, p, y, t

    def update_pose_and_orientation(self):
        self.position, self.orientation = self.get_pose_and_orientation()
        return self.position, self.orientation