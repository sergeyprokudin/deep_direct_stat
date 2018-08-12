import numpy as np


def rad2bit(angles_rad):
    """ radians to biternion ([cos, sin])
    """
    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


def deg2bit(angles_deg):
    """ degrees to biternion ([cos, sin])
    """
    angles_rad = np.deg2rad(angles_deg)
    return np.array([np.cos(angles_rad), np.sin(angles_rad)]).T


def bit2deg(angles_bit):
    """ biternion ([cos, sin]) ->  degrees
    """
    return (np.rad2deg(np.arctan2(angles_bit[:, 1], angles_bit[:, 0])) + 360) % 360

def bit2rad(angles_bit):
    """ biternion ([cos, sin]) ->  radians
    """
    return np.deg2rad(bit2deg(angles_bit))

def bit2deg_multi(angles_bit):
    """ Convert biternion representation to degree for multiple samples

    Parameters
    ----------
    angles_bit: numpy array of shape [n_points, n_predictions, 2]
        multiple predictions

    Returns
    -------

    deg_angles: numpy array of shape [n_points, n_predictions]
        multiple predictions converted to degree representation
    """

    deg_angles = np.asarray([bit2deg(angles_bit[:, i, :]) for i in range(0, angles_bit.shape[1])]).T

    return deg_angles


def cart_to_spherical(xyz):

    r_phi_theta = np.zeros(xyz.shape)
    xy = xyz[:, 0]**2 + xyz[:, 1]**2
    r_phi_theta[:, 0] = np.sqrt(xy + xyz[:, 2]**2)
    r_phi_theta[:, 1] = np.arctan2( xyz[:, 1], xyz[:, 0]) # for elevation angle defined from XY-plane up
    r_phi_theta[:, 2] = np.arccos(xyz[:, 2]) #theta = arccos(z/r)

    return r_phi_theta