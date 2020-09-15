import numpy as np
import matplotlib.patches as pch
import mpl_toolkits.mplot3d.art3d as art3d
from numpy.linalg import inv, norm


'''
name: Quadcopter Plot File
author: Christopher Banks
date: 10/14/18
description: This file contains the functions and classes used to simulate the quadcopters in the Robotarium to scale. Included
are rotation matrix files for rotating the quadcopter about given vectors and we plot x, y and z orientations for the 
quadcopters as red, green and blue respectively. This file is very sensitive to changes. DO NOT edit!
'''

def rotation_matrix(d):
    """
    Calculates a rotation matrix, M, given a vector, d:
    Direction of d corresponds to the rotation axis
    Length of d corresponds to the sine of the angle of rotation

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye  = np.eye(3)
    ddt  = np.outer(d, d)
    skew = np.array([[    0,  d[2], -d[1]],
                     [-d[2],     0,  d[0]],
                     [ d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z=0, normal=np.array([0, 0, 0])):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into the XY plane, rotated about the origin
    and finally translated by z.
    """

    normal /= norm(normal)  # Make sure the vector is normalised

    path = pathpatch.get_path()  # Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path)  # Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D  # Change the class
    pathpatch._code3d = path.codes  # Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor  # Get the face color

    verts = path.vertices  # Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1))  # Obtain the rotation vector
    M = rotation_matrix(d)  # Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


class QuadPlotObject(object):
    def __init__(self, ax, initpos, trans=1.0):
        self.traj = np.expand_dims(initpos, 0)
        ax.plot(self.traj[:, 0], self.traj[:, 1], self.traj[:, 2])
        self.radius0 = 0.1
        self.radius1 = 0.067 # 0.1/1.5
        R = self.RotMat()
        L = 0.15
        x_direction = np.array([[initpos[0] , initpos[0]+L*R[0,0]], [initpos[1] , initpos[1]+L*R[1,0]], [initpos[2] , initpos[2]+L*R[2,0]]])
        y_direction = np.array([[initpos[0] , initpos[0]+L*R[0,1]], [initpos[1] , initpos[1]+L*R[1,1]], [initpos[2] , initpos[2]+L*R[2,1]]])
        z_direction = np.array([[initpos[0] , initpos[0]+L*R[0,2]], [initpos[1] , initpos[1]+L*R[1,2]], [initpos[2] , initpos[2]+L*R[2,2]]])
        self.x_dir_visual = ax.plot(x_direction[0, :], x_direction[1, :], x_direction[2, :], 'r', alpha=min(0.6, trans))[0]
        self.y_dir_visual = ax.plot(y_direction[0, :], y_direction[1, :], y_direction[2, :], 'g', alpha=min(0.6, trans))[0]
        self.z_dir_visual = ax.plot(z_direction[0, :], z_direction[1, :], z_direction[2, :], 'b', alpha=min(0.6, trans))[0]

        d = np.cross((R[0,2], R[1,2], R[2,2]),(0, 0, 1)) # Obtain the rotation vector
        self.M = rotation_matrix(d) # Get the rotation matrix
        self.pos = initpos
        self.disk1 = pch.Circle((initpos[0] + self.radius0 / np.sqrt(2), initpos[1] + self.radius0 / np.sqrt(2)), self.radius1, alpha=trans)
        self.disk2 = pch.Circle((initpos[0] + self.radius0 / np.sqrt(2), initpos[1] - self.radius0 / np.sqrt(2)), self.radius1, alpha=trans)
        self.disk3 = pch.Circle((initpos[0] - self.radius0 / np.sqrt(2), initpos[1] + self.radius0 / np.sqrt(2)), self.radius1, alpha=trans)
        self.disk4 = pch.Circle((initpos[0] - self.radius0 / np.sqrt(2), initpos[1] - self.radius0 / np.sqrt(2)), self.radius1, alpha=trans)
        ax.add_patch(self.disk1)
        ax.add_patch(self.disk2)
        ax.add_patch(self.disk3)
        ax.add_patch(self.disk4)
        pathpatch_2d_to_3d(self.disk1, z=initpos[2], normal=np.array([R[0,2],R[1,2],R[2,2]]))
        pathpatch_2d_to_3d(self.disk2, z=initpos[2], normal=np.array([R[0,2],R[1,2],R[2,2]]))
        pathpatch_2d_to_3d(self.disk3, z=initpos[2], normal=np.array([R[0,2],R[1,2],R[2,2]]))
        pathpatch_2d_to_3d(self.disk4, z=initpos[2], normal=np.array([R[0,2],R[1,2],R[2,2]]))

    def RotMat(self, r=0.0,p=0.0,y=0.0):
        R = np.array([[np.cos(p)*np.cos(y), np.sin(r)*np.sin(p)*np.cos(y)-np.cos(r)*np.sin(y), np.cos(r)*np.sin(p)*np.cos(y)+np.sin(r)*np.sin(y)],
                      [np.cos(p)*np.sin(y), np.sin(r)*np.sin(p)*np.sin(y)+np.cos(r)*np.cos(y), np.cos(r)*np.sin(p)*np.sin(y)-np.sin(r)*np.cos(y)],
                      [-np.sin(p), np.sin(r)*np.cos(p), np.cos(r)*np.cos(p)]])
        return R

    def update(self, ax, pos, r, p, y=0, color='k--'):

        R = self.RotMat(r, p, y)
        L = 0.15
        x_direction = np.array([[pos[0] , pos[0]+L*R[0,0]], [pos[1] , pos[1]+L*R[1,0]], [pos[2] , pos[2]+L*R[2,0]]])
        y_direction = np.array([[pos[0] , pos[0]+L*R[0,1]], [pos[1] , pos[1]+L*R[1,1]], [pos[2] , pos[2]+L*R[2,1]]])
        z_direction = np.array([[pos[0] , pos[0]+L*R[0,2]], [pos[1] , pos[1]+L*R[1,2]], [pos[2] , pos[2]+L*R[2,2]]])
        self.x_dir_visual.set_data(x_direction[0, :], x_direction[1, :])
        self.x_dir_visual.set_3d_properties(x_direction[2, :])
        self.y_dir_visual.set_data(y_direction[0, :], y_direction[1, :])
        self.y_dir_visual.set_3d_properties(y_direction[2, :])
        self.z_dir_visual.set_data(z_direction[0, :], z_direction[1, :])
        self.z_dir_visual.set_3d_properties(z_direction[2, :])
        d = np.cross((R[0,2], R[1,2], R[2,2]),(0, 0, 1)) # Obtain the rotation vector
        newM = rotation_matrix(d) # Get the rotation matrix
        self.disk1._segment3d -= np.kron(np.ones((self.disk1._segment3d[:,0].size,1)), self.pos)
        self.disk2._segment3d -= np.kron(np.ones((self.disk2._segment3d[:,0].size,1)), self.pos)
        self.disk3._segment3d -= np.kron(np.ones((self.disk3._segment3d[:,0].size,1)), self.pos)
        self.disk4._segment3d -= np.kron(np.ones((self.disk4._segment3d[:,0].size,1)), self.pos)
        self.disk1._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk1._segment3d.T)).T
        self.disk2._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk2._segment3d.T)).T
        self.disk3._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk3._segment3d.T)).T
        self.disk4._segment3d = np.dot(newM, np.dot(inv(self.M), self.disk4._segment3d.T)).T
        self.disk1._segment3d += np.kron(np.ones((self.disk1._segment3d[:,0].size,1)), pos)
        self.disk2._segment3d += np.kron(np.ones((self.disk2._segment3d[:,0].size,1)), pos)
        self.disk3._segment3d += np.kron(np.ones((self.disk3._segment3d[:,0].size,1)), pos)
        self.disk4._segment3d += np.kron(np.ones((self.disk4._segment3d[:,0].size,1)), pos)
        self.pos = pos

        self.M = newM

        # update trajectory
        self.traj = np.vstack((self.traj, pos))
