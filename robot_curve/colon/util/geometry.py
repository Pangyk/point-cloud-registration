import open3d as o3d
import numpy as np
import scipy.interpolate

import util.transform


def generate_cycle(**kwargs):
    '''
    Generate single ridge+pocket cycle with given parameters. Assumes straight
    centerline through cycle.

    Args:
    r_width     -   ridge max width (cm)
    r_height    -   ridge max height (cm)
    r_ang_cov   -   ridge angular coverage (radians arclength)
    r_ang_cen   -   ridge angular center (radians)
    r_h_falloff -   ridge height falloff rate (recommended=6)
    r_w_falloff -   ridge width falloff rate
    p_width     -   pocket width (cm)
    rad_y       -   cross-sectional y-radius (cm)
    rad_z       -   cross-sectional z-radius (cm)
    theta       -   cross-sectional angulation (radians offset from perpendicular to centerline)

    Returns:
    pc  -   PointCloud object
    '''
    dx = 0.05  # sample x-axis @ 0.5mm
    dt = 2 * np.pi / 180  # sample angles @ 2 degree
    n_subdivisions = int(round(np.pi * 2 / dt))
    points = []

    # generate ridge points
    for j, raw_x in enumerate(np.arange(0, kwargs['r_width'], dx)):
        x = np.full(n_subdivisions, raw_x)
        t = np.arange(0, np.pi * 2, dt)
        # expit defined [-6,6]=>[0,1]
        # expit to decrease ridge height across crest
        if raw_x < kwargs['r_width'] / 2:
            h = kwargs['r_height'] * scipy.special.expit(2 * x / kwargs['r_width'] * 12 - 6)
        else:
            h = kwargs['r_height'] * scipy.special.expit((2 - 2 * x / kwargs['r_width']) * 12 - 6)
        # expit to decrease ridge height along crest
        w = kwargs['r_ang_cen'] - kwargs['r_ang_cov'] / 2  # starting angle
        cross_ridge_mult = np.logical_or(np.logical_and(t > w, t < w + kwargs['r_ang_cov']), \
                                         np.logical_and(w + kwargs['r_ang_cov'] > np.pi * 2,
                                                        t < w + kwargs['r_ang_cov'] - np.pi * 2)).astype(np.int)
        cross_ridge_mult = cross_ridge_mult * np.where(((t - w) % (2 * np.pi)) < kwargs['r_ang_cov'] / 2, \
                                                       scipy.special.expit(
                                                           2 * ((t - w) % (2 * np.pi)) / kwargs['r_ang_cov'] * 12 - 6), \
                                                       scipy.special.expit((2 - 2 * ((t - w) % (2 * np.pi)) / kwargs[
                                                           'r_ang_cov']) * 12 - 6))
        h = h * cross_ridge_mult
        # compute final coordinates
        y = (kwargs['rad_y'] - h) * np.cos(t)
        z = (kwargs['rad_z'] - h) * np.sin(t)
        coords = np.stack([x, y, z], axis=-1)
        # rotation around y-axis
        com = np.mean(coords, axis=0)
        rot = util.transform.rot_z(kwargs['theta'])
        coords = np.matmul(rot, (coords - com).T).T + com

        points.extend(coords)

    # fill in between ridge and pocket
    x_offset = kwargs['r_width']
    pocket_offset = 2 * kwargs['rad_y'] * np.sin(kwargs['theta'])
    # for raw_x in np.arange(0,pocket_offset,dx):
    #     # TODO: compute excluded t range
    #     t_start = 0
    #     t_start = t_start % (2*np.pi)
    #     t_end = (-t_start)%(2*np.pi)
    #     print(t_start, t_end)
    #     # generate remaining points per usual
    #     t = np.arange(0,t_start,dt) + np.arange(t_end,np.pi*2,dt)
    #     x = np.full(t.shape,raw_x+x_offset)
    #     y = kwargs['rad_y'] * np.cos(t)
    #     z = kwargs['rad_z'] * np.sin(t)
    #     points.extend(np.stack([x,y,z],axis=-1))

    # generate pocket points
    x_offset += pocket_offset
    for raw_x in np.arange(0, kwargs['p_width'], dx):
        x = np.full(n_subdivisions, raw_x + x_offset)
        t = np.arange(0, np.pi * 2, dt)
        y = kwargs['rad_y'] * np.cos(t)
        z = kwargs['rad_z'] * np.sin(t)
        points.extend(np.stack([x, y, z], axis=-1))

    # convert to PointCloud object
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals()
    return pc


def join_cycles(cycles, relrot=None):
    '''
    Join cycles along centerline.

    Args:
    cycles  -   list of cycle point clouds
    relrot  -   list of relative rotation between cycles

    Returns:
    PointCloud
    '''
    unified_pc = o3d.geometry.PointCloud()
    if relrot is None:
        relrot = [np.eye(3) for _ in range(len(cycles) - 1)]
    translation = np.zeros((3, 1))
    prev_rot = np.eye(3)
    for i, pc in enumerate(cycles):
        max_x = pc.get_max_bound()[0]
        if i == 0:
            rot = np.eye(3)
        else:
            rot = np.matmul(relrot[i - 1], prev_rot)
        transform = np.concatenate([rot, translation], axis=-1)
        homogeneous = np.array([0, 0, 0, 1]).reshape((1, 4))
        transform = np.concatenate([transform, homogeneous], axis=0)
        pc.transform(transform)
        translation = np.matmul(transform, np.array([max_x, 0, 0, 1]).reshape((4, 1)))[:3]
        prev_rot = rot
        unified_pc += pc
    return unified_pc


def generate_centerline_curve(points, param):
    '''
    Generate centerline curve by computing 1D cubic spline through 3D points.
    At position u along flat centerline, fit curve points[u].

    Args:
    points      -   points for centerline to run through (3xn array)
    param       -   curve parameterization (n array)

    Returns:
    scipy.interpolate.splprep object
    '''
    spline, _ = scipy.interpolate.splprep(points, u=param, k=3)
    return spline


def build_mesh(pcd):
    '''
    Generate mesh from point cloud via Poisson method. Clean up by removing
    vertices with <0.05 density.

    Args:
    pcd     -   PointCloud object

    Returns:
    mesh    -   TriangleMesh object
    '''
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8)
        densities = np.asarray(densities)
        # remove low-density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.05)
        mesh.remove_vertices_by_mask(vertices_to_remove)
    return mesh
