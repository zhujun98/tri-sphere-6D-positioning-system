import numpy as np
from scipy.optimize import fsolve


def rotation_x(theta):
    """Matrix of rotation about x axis."""
    return np.matrix([[ 1,             0,              0],
                      [ 0, np.cos(theta), -np.sin(theta)],
                      [ 0, np.sin(theta),  np.cos(theta)]])


def rotation_y(theta):
    """Matrix of rotation about y axis."""
    return np.matrix([[ np.cos(theta), 0,  np.sin(theta)],
                      [             0, 1,              0],
                      [-np.sin(theta), 0,  np.cos(theta)]])


def rotation_z(theta):
    """Matrix of rotation about z axis."""
    return np.matrix([[ np.cos(theta), -np.sin(theta), 0],
                      [ np.sin(theta),  np.cos(theta), 0],
                      [             0,              0, 1]])


def rotation_fixed_angle(gamma, beta, alpha):
    """Matrix for X-Y-Z fixed angle rotation."""
    return rotation_z(alpha) * rotation_y(beta) * rotation_x(gamma)


def trisphere_forward_kinematics(j_p, j_np, c_np,
                                 c_p0=np.zeros(3),
                                 c_r0=np.zeros(3)):
    """Forward kinematics of the Tri-sphere 6D system.

    :param tuple: jacks' target positions (j1_p, j2_p, j3_p) relative to
        their respective null positions.
    :param tuple j_np: null positions of the jacks (j1_np, j2_np, j3_np)
        in the Tri-Sphere coordinate system.
    :param numpy.array c_np: null position of the control point in the
        Tri-Sphere coordinate system.
    :param numpy.array c_p0: current position of the control point in the
        Tri-Sphere coordinate system.
    :param numpy.array c_r0: current position of the control point in the
        Tri-Sphere coordinate system.

    :return: target of the position (x, y, z) and rotation
        (gamma, beta, alpha) of the control point in radians.
    """
    def obj_func(x):
        ret, _ = trisphere_inverse_kinematics(x[:3], x[3:], c_np, j_np)
        # z1, y1, x2, y2, x3, y3
        return np.array([ret[0][2] - j_p[0][2], ret[0][1] - j_p[0][1],
                         ret[1][0] - j_p[1][0], ret[1][1] - j_p[1][1],
                         ret[2][0] - j_p[2][0], ret[2][1] - j_p[2][1]])

    x0 = np.concatenate((c_p0, c_r0))
    ret = fsolve(obj_func, x0)
    # c_p and c_r
    return ret[:3], ret[3:]


def trisphere_inverse_kinematics(c_p, c_r, c_np, j_np):
    """Inverse kinematics of the Tri-sphere 6D system.

    :param numpy.array c_p: target position (x, y, z) of the control point
        relative to its null position.
    :param numpy.array c_r: target rotation (gamma, beta, alpha) of the
        control point in radians.
    :param numpy.array c_np: null position of the control point in the
        Tri-Sphere coordinate system.
    :param tuple j_np: null positions of the jacks (j1_np, j2_np, j3_np)
        in the Tri-Sphere coordinate system.

    :return: jacks' target positions (j1_p, j2_p, j3_p) relative to their
        respective null position and "transformed null positions"
        (j1_tnp, j2_tnp, j3_tnp). The latter is used for testing and
        debugging.
    """
    j1_np, j2_np, j3_np = j_np
    gamma, beta, alpha = c_r

    # calculate rotation matrix
    rt_matrix = rotation_fixed_angle(gamma, beta, alpha)

    # calculate transformed null positions
    j1_tnp = ((j1_np - c_np) * rt_matrix.T).getA()[0] + c_np + c_p
    j2_tnp = ((j2_np - c_np) * rt_matrix.T).getA()[0] + c_np + c_p
    j3_tnp = ((j3_np - c_np) * rt_matrix.T).getA()[0] + c_np + c_p

    # calculate jacks' new positions
    j1_p = j1_tnp - j1_np
    dx = j1_np[0] - j1_tnp[0]
    j1_p[0] = 0
    j1_p[1] += np.tan(alpha) * dx
    j1_p[2] += -np.tan(beta)/np.cos(alpha) * dx

    j2_p = j2_tnp - j2_np
    dz = j2_np[2] - j2_tnp[2]
    j2_p[0] += (np.tan(beta)*np.cos(alpha)
                + np.sin(alpha)*np.tan(gamma)/np.cos(beta)) * dz
    j2_p[1] += -(np.tan(gamma)/np.cos(beta)*np.cos(alpha)
                 - np.tan(beta)*np.sin(alpha)) * dz
    j2_p[2] = 0

    j3_p = j3_tnp - j3_np
    dz = j3_np[2] - j3_tnp[2]
    j3_p[0] += (np.tan(beta)*np.cos(alpha)
                + np.sin(alpha)*np.tan(gamma)/np.cos(beta)) * dz
    j3_p[1] += -(np.tan(gamma)/np.cos(beta)*np.cos(alpha)
                 - np.tan(beta)*np.sin(alpha)) * dz
    j3_p[2] = 0

    return (j1_p, j2_p, j3_p), (j1_tnp, j2_tnp, j3_tnp)
