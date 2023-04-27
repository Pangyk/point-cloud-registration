import numpy as np
import scipy
import open3d as o3d


def rot_x(t):
    '''
    Generate 3D rotation matrix for rotation around x-axis (in z-y plane)

    Args:
    t     -   rotation angle (radians, float)

    Returns:
    3x3 rotation matrix
    '''
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t), np.cos(t)]])


def rot_y(t):
    '''
    Generate 3D rotation matrix for rotation around y-axis (in x-z plane)

    Args:
    t     -   rotation angle (radians, float)

    Returns:
    3x3 rotation matrix
    '''
    return np.array([[np.cos(t), 0, np.sin(t)],
                     [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])


def rot_z(t):
    '''
    Generate 3D rotation matrix for rotation around z-axis (in x-y plane)

    Args:
    t     -   rotation angle (radians, float)

    Returns:
    3x3 rotation matrix
    '''
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t), np.cos(t), 0],
                     [0, 0, 1]])


def deform(pcd, df):
    '''
    Apply deformation field to point cloud. Interpolates deformation for each
    point location via trilinear interpolation. Deformation field must have
    points at each pixel location from range [0,...,n].

    Args:
    pcd     -   PointCloud object (HxWxL)
    df      -   3D deformation field (HxWxLx3)

    Returns:
    PointCloud object
    '''
    points = pcd.points
    min_pt = np.amin(points, axis=0)
    result = []
    eps = 1e-4
    xmax, ymax, zmax = df.shape[0] - 1, df.shape[1] - 1, df.shape[2] - 1
    for pt in points:
        pt -= min_pt
        left = np.floor(pt).astype(int)
        right = np.ceil(pt + [eps, eps, eps]).astype(int)
        df_left = df[left[0], left[1], left[2]]
        df_right = df[min(xmax, right[0]), min(ymax, right[1]), min(zmax, right[2])]
        def_vec = (df_right - df_left) * (pt - left) / (right - left) + df_left
        new_pt = pt + def_vec + min_pt
        result.append(new_pt)

    # construct point cloud
    result = np.stack(result, axis=0)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(result)
    pc.estimate_normals()
    return pc


def generate_deformation_field(pcd, points, values):
    '''
    Generate deformation field by cublic splining between values at points.

    Args:
    pcd     -   PointCloud object
    points  -   nx3 array of points
    values  -   nx3 array of deformation vectors

    Returns:
    deformation field
    '''
    assert np.all(points.shape == values.shape)

    pcd_points = pcd.points
    xmin, ymin, zmin = (np.floor(np.amin(pcd_points, axis=0))).astype(int)
    xmax, ymax, zmax = (np.ceil(np.amax(pcd_points, axis=0))).astype(int)
    xi = []
    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            for z in range(zmin, zmax):
                xi.append([x, y, z])
    df = scipy.interpolate.NearestNDInterpolator(points, values)(xi)

    # DEBUG ONLY: visualize raw field
    # eps = 1e-4
    # ax = plt.figure().add_subplot(projection='3d')
    # xi = np.array(xi)
    # ax.quiver(points[:,0], points[:,1], points[:,2], values[:,0]+eps, values[:,1]+eps, values[:,2]+eps, length=1)
    # plt.show()
    # plt.clf()

    # DEBUG ONLY: visualize deformation field
    # ax = plt.figure().add_subplot(projection='3d')
    # xi = np.array(xi)
    # ax.quiver(xi[:,0], xi[:,1], xi[:,2], df[:,0]+eps, df[:,1]+eps, df[:,2]+eps, length=1)
    # plt.show()
    # END DEBUG

    df = df.reshape(xmax - xmin, ymax - ymin, zmax - zmin, -1)
    return df


def generate_centerline_transform(uval, spline):
    '''
    Compute transform from flat to curved centerline (4x4 homogenous).

    Args:
    uval        -   x-coordinate on flat centerline (scalar)
    spline      -   scipy.interpolate.splrep object

    Returns:
    4x4 homogenous transform matrix
    '''
    new_pos = scipy.interpolate.splev(uval, spline, der=0)
    t = np.array([*new_pos, 1]).reshape((4, 1))
    deriv = scipy.interpolate.splev(uval, spline, der=1)
    # convert derivative to rotation matrix
    theta = np.arctan(deriv)
    rot = np.matmul(np.matmul(rot_x(theta[0]),
                              rot_y(theta[1])),
                    rot_z(theta[2]))
    rot = np.concatenate([rot, np.array([[0, 0, 0]])], axis=0)
    assert rot.shape[0] == 4 and rot.shape[1] == 3
    result = np.concatenate([rot, t], axis=1)
    assert result.shape[0] == 4 and result.shape[1] == 4
    return result
