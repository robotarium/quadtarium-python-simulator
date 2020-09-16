from utilities_sim.robotarium_communication_interface import RobotariumCommunication
import numpy as np
from utilities_sim.actuation import invert_diff_flat_output

class QuadcopterObject(RobotariumCommunication):
    """Quadcopter object created for each quadcopter in order to update dynamics, plot, etc...
    """

    def __init__(self, robotarium_simulator, initial_pose=None, index=0, dt=0.02):

        self.thrust_hover = 0

        # Moment of Inertia Matrix
        #self.I_moment = np.array([[3.11977, -4.94033, -4.6007], [-4.94033, 3.12815, -4.71309], [-4.6007, -4.71309, 7.00414]]) * (10**-5)
        self.I_moment = np.diag(np.array([3.11977, 3.12815, 7.00414]) * (10 ** -5))
        self.I_moment_inv = np.linalg.inv(self.I_moment)

        # Mass, Gravity and Time Step
        self.m = 35.89 / 1000  # mass of the quad in kg
        self.g = 9.81  # gravity constant
        self.dt = dt  # time step for simulation

        # XYZ Position PID
        self.kp = np.array([0.4, 0.4, 1.25]) * 0.23  # P np.array([0.4, 0.4, 1.25])
        self.ki = np.array([0.05, 0.05, 0.05]) * 1e-10  # I np.array([0.05, 0.05, 0.05])
        self.kd = np.array([0.2, 0.2, 0.4]) * 0.2  # D np.array([0.2, 0.2, 0.4])
        self.i_range = np.array([2.0, 2.0, 0.4])

        # Attitude Gains
        self.kR = np.array([.7, .7, .6]) * 1e-2  # P np.array([70000, 70000, 60000])
        self.kw = np.array([0.2, 0.2, 0.12]) * 3e-3  # D [np.array([20000, 20000, 12000])
        self.ki_m = np.array([0.0, 0.0, 500]) * 1e-10  # I
        self.i_range_m = np.array([1.0, 1.0, 1500])

        # Integral Error
        self.i_error = np.zeros(3)
        self.i_error_m = np.zeros(3)

        super().__init__(robotarium_simulator, str(index))

        if initial_pose is None:
            initial_pose = self.get_init_pose()

        self.set_init_pose(initial_pose)
        self.thrust_hover = self.thrust_hover

    def hover_bot(self, hover_point, s, sim_env):
        """

        Args:
            hover_point (ndarray): Hover points as x,y,z
            s (float): time step
            sim_env: simulation environment

        Returns:
            position (ndarray): next position to go as x,y,z

        """
        dx = hover_point - self.pose  # x,y,z diff
        next_pos = self.pose + s*dx
        self.set_pose(next_pos, sim_env)
        self.state[:3] = next_pos
        self.pose = next_pos
        error = np.linalg.norm((hover_point - self.pose))
        if error < 0.1:
            return 1, self.pose
        else:
            return 0, self.pose


    def set_diff_flat_term(self, goal_pose):
        """Obtain roll, pitch, yaw, thrust for going to a desired position.

        Args:
            goal_pose (ndarray): Desired pose of size (4, 3)

        Returns:
            r (float): Roll
            p (float): Pitch
            y (float): Yaw Rate (currently just 0 here)
            t (float): Thrust

        """
        r, p, y, t = invert_diff_flat_output(goal_pose, thrust_hover=self.thrust_hover)
        return r, p, y, t

    def go_to(self, desired_pose, desired_orientation=None):
        """Obtains u_1, u_2, u_3, u_4 for going to a desired pose.

        Args:
            goal_pose (ndarray): Desired pose of size (4, 3)

        Returns:
            state (ndrray): Returns the fulls state array
                            [x,y,z,roll,pitch,yaw,x_dot,y_dot,z_dot,roll_dot,pitch_dot,yaw_dot]
        """

        # Obtain Control Inputs needed to follow desired trajectory
        u = self.obtain_desired_inputs(desired_pose, desired_orientation=None)  # TODO: Account for desired Orientation

        return u

    def obtain_desired_inputs(self, desired_pose, desired_orientation=None):

        if desired_orientation is None:  # Just keep the current angles
            desired_orientation = np.zeros((4, 3))
            desired_orientation[0] = self.state[3:6]

        u = np.zeros(4)  # Control Input that we want to compute [Thrust, Mx, My, Mz]

        # XYZ Position and Velocity Errors
        e_p = self.state[0:3] - desired_pose[0]
        e_v = self.state[6:9] - desired_pose[1]
        self.i_error = np.clip(self.i_error + e_p * self.dt, -self.i_range, self.i_range)

        # Desired Thrust in world frame
        F_des = - self.kp * e_p - self.kd * e_v - self.ki * self.i_error + self.m * (desired_pose[2] + self.g * np.array([0, 0, 1]))

        R = self.get_Rwb()  # Rotation matrix

        # Current Body Frame Axis
        x_b = R[:, 0].squeeze()
        y_b = R[:, 1].squeeze()
        z_b = R[:, 2].squeeze()

        # Desired Thrust in body frame
        u[0] = np.dot(F_des, z_b)

        # x-axis of intermediate frame (only yaw rotation)
        x_c_des = np.array([np.cos(desired_orientation[0][2]), np.sin(desired_orientation[0][2]), 0])

        # Desired Body Frame Axes
        z_b_des = F_des / np.linalg.norm(F_des)
        y_b_des = np.cross(z_b_des, x_c_des) / np.linalg.norm(np.cross(z_b_des, x_c_des))
        x_b_des = np.cross(y_b_des, z_b_des)

        R_des = np.stack((x_b_des, y_b_des, z_b_des)).T  # Rwb_desired

        # Rotation Matrix Error (eR), Angular Velocity Error (ew)
        eR = 0.5 * self.vee_map((np.matmul(R_des.T, R) - np.matmul(R.T, R_des)))
        ew = desired_orientation[1] - self.state[9:]
        # Integral Error of eR
        self.i_error_m = np.clip(self.i_error_m + -eR * self.dt,  -self.i_range_m, self.i_range_m)

        # Compute Moments u[2,3,4] using eR and eOmega
        u[1] = -self.kR[0] * eR[0] + self.kw[0] * ew[0] + self.ki_m[0] * self.i_error_m[0]
        u[2] = -self.kR[1] * eR[1] + self.kw[1] * ew[1] + self.ki_m[1] * self.i_error_m[1]
        u[3] = -self.kR[2] * eR[2] + self.kw[2] * ew[2] + self.ki_m[2] * self.i_error_m[2]

        return u

    def forward_model(self, sim_env, u):

        alpha = self.state[3:6]  # Euler Angles of body (phi,theta,psi)
        vel = self.state[6:9]  # Linear Velocity of body (u,v,w)
        omega_b = self.state[9:]  # Angular Velocity of body (p,q,r)

        Rwb = self.get_Rwb()
        Twb = self.get_Twb()

        state_d = np.zeros(12)
        state_d[0:3] = vel
        state_d[3:6] = Twb @ omega_b

        # Linear Acceleration
        omega_bw = self.state[9:]
        z_w = np.array([0, 0, 1])
        z_b = Rwb[:, 2].squeeze()
        state_d[6:9] = (-self.m * self.g * z_w + u[0] * z_b) / self.m

        # Acceleration of roll pitch yaw
        state_d[9:12] = np.matmul(self.I_moment_inv, np.cross(-omega_bw, np.matmul(self.I_moment, omega_bw)) + u[1:])

        next_state = self.state + self.dt * state_d

        # Update Properties of Quadcopter (self) object
        self.set_state(next_state, sim_env)

        return next_state

    def get_Twb(self):
        """Angular velocity transformation matrix from BODY to WORLD frame.

        """

        phi, theta, _ = self.state[3:6]

        T = np.array([[1, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0,                 np.cos(phi),                -np.sin(phi)],
                      [0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]])

        return T

    def get_Rwb(self):
        """Get rotation matrix from BODY to WORLD frame.

        """

        theta = self.state[3:6]

        R = np.array([[np.cos(theta[1]) * np.cos(theta[2]),
                       np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2]) - np.sin(theta[2]) * np.cos(theta[0]),
                       np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2]) + np.sin(theta[0]) * np.sin(theta[2])],
                      [np.sin(theta[2]) * np.cos(theta[1]),
                       np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
                       np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0]) - np.sin(theta[0]) * np.cos(theta[2])],
                      [-np.sin(theta[1]), np.sin(theta[0]) * np.cos(theta[1]), np.cos(theta[0]) * np.cos(theta[1])]])

        return R

    @staticmethod
    def vee_map(x_hat):
        """
        Inverse of the hat map. Given a matrix of the form:
                 x_hat = np.array(
                         [[0, -x[2], x[1]],
                          [x[2], 0, -x[0]],
                          [-x[1], x[0], 0]])
        Extract and return x.
        """


        assert (len(x_hat.shape) == 2 and x_hat.shape[0] == 3 and x_hat.shape[1] == 3), 'The input to the vee map must be a matrix of size (3,3).'

        return np.array([x_hat[2, 1], -x_hat[2, 0], x_hat[1, 0]])

    @staticmethod
    def hat_map(x):

        assert (len(x.shape) == 1 and x.shape[0] == 3), 'The input to the hat map must be a vector of length 3.'

        x_hat = np.array(
            [[0, -x[2], x[1]],
             [x[2], 0, -x[0]],
             [-x[1], x[0], 0]]
        )

        return x_hat

    # def go_to_old(self, desired_pose, sim_env):
    #     """Go to goal.
    #
    #     Args:
    #         desired_pose (ndarray): Array of size 4 by 3 indicating the desired pose
    #         sim_env (object): simulation environment (robotarium object)
    #
    #     Returns:
    #
    #     """
    #     roll, pitch, yaw, thrust = self.set_diff_flat_term(desired_pose)
    #     self.set_pose(desired_pose[0, :], sim_env, roll, pitch, yaw)