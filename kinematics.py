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


def trisphere_forward_kinematics(j_dp, j_p0, c_p0):
    """Forward kinematics of the Tri-sphere 6D system.

    :param tuple: changes of the jacks' positions
        (j1_dp, j2_dp, j3_dp)
    :param tuple j_p0: current positions of the jacks
        (j1_p0, j2_p0, j3_p0) in the Tri-Sphere coordinate system.
    :param numpy.array c_p0: current position of the control point in the
        Tri-Sphere coordinate system.

    :return: changes of the position (dx, dy, dz) and orientation
        (gamma, beta, alpha) of the control point.
    """
    def obj_func(x):
        ret, _ = trisphere_inverse_kinematics(x[:3], x[3:], c_p0, j_p0)
        return np.array([ret[0][1] - j_dp[0][1], ret[0][2] - j_dp[0][2],
                         ret[1][0] - j_dp[1][0], ret[1][2] - j_dp[1][2],
                         ret[2][0] - j_dp[2][0], ret[2][2] - j_dp[2][2]])

    x0 = np.array([c_p0[0], c_p0[1], c_p0[2], 0., 0., 0.])
    ret = fsolve(obj_func, x0)
    ret[3:] = np.rad2deg(ret[3:])
    return ret


def trisphere_inverse_kinematics(c_dp, c_dr, c_p0, j_p0):
    """Inverse kinematics of the Tri-sphere 6D system.

    :param numpy.array c_dp: position change (dx, dy, dz) of the control
        point.
    :param numpy.array c_dr: orientation change (gamma, beta, alpha) of
        the control point.
    :param numpy.array c_p0: current position of the control point in the
        Tri-Sphere coordinate system.
    :param tuple j_p0: current positions of the jacks
        (j1_p0, j2_p0, j3_p0) in the Tri-Sphere coordinate system.

    :return: changes of the jacks' positions (j1_dp, j2_dp, j3_dp) and
        "transformed current positions" (j1_tp0, j2_tp0, j3_tp0). The
        latter is used for testing and debugging.
    """
    j1_p0, j2_p0, j3_p0 = j_p0
    gamma, beta, alpha = c_dr

    # calculate overall rotation matrix
    rt_matrix = rotation_fixed_angle(gamma, beta, alpha)

    # calculate transformed current positions
    j1_tp0 = ((j1_p0 - c_p0) * rt_matrix.T).getA()[0] + c_p0 + c_dp
    j2_tp0 = ((j2_p0 - c_p0) * rt_matrix.T).getA()[0] + c_p0 + c_dp
    j3_tp0 = ((j3_p0 - c_p0) * rt_matrix.T).getA()[0] + c_p0 + c_dp

    # calculate jacks' new positions
    j1_dp = j1_tp0 - j1_p0
    dx = j1_p0[0] - j1_tp0[0]
    j1_dp[0] = 0
    j1_dp[1] += np.tan(alpha) * dx
    j1_dp[2] += -np.tan(beta)/np.cos(alpha) * dx

    j2_dp = j2_tp0 - j2_p0
    dy = j2_p0[1] - j2_tp0[1]
    j2_dp[0] += -(np.tan(alpha) - np.sin(beta)*np.tan(gamma)) * dy
    j2_dp[1] = 0
    j2_dp[2] += (np.tan(gamma)/np.cos(alpha) - np.sin(beta)**2*np.sin(gamma))*dy

    j3_dp = j3_tp0 - j3_p0
    dy = j3_p0[1] - j3_tp0[1]
    j3_dp[0] += -(np.tan(alpha) - np.sin(beta)*np.tan(gamma)) * dy
    j3_dp[1] = 0
    j3_dp[2] += (np.tan(gamma)/np.cos(alpha) - np.sin(beta)**2*np.sin(gamma))*dy

    return (j1_dp, j2_dp, j3_dp), (j1_tp0, j2_tp0, j3_tp0)
