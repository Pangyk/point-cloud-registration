import open3d as o3d
import numpy as np
import argparse
import os
import scipy.interpolate
import matplotlib.pyplot as plt
import pickle

import util
import pandas as pd
from pyntcloud import PyntCloud


# np.random.seed(123)

class rand_shape:
    def __init__(self):
        self.v = 0

    def gen_rand(self, is_same):
        if is_same:
            self.v = 4 * (np.random.randn(1)[0] * 0.5 + 1)
        return self.v


def generate_point_cloud(n_ridges, rad_y, rad_z, length, variation, cap, ridge_centers=[], cs_offset=None,
                         centerline=None):
    '''
    Generate point cloud from parameters. Elliptical cross sections in y-z plane.
    Centerline along x-axis. Independently random walk y and z radii along length.
    Haustral ridges randomly generated within interior 80% or via ridge_pos.

    Args:
    n_ridges        -   number haustral ridges (int, ignored if ridge_centers non-None)
    rad_y           -   starting cross-sectional y-axis radius (cm, float)
    rad_z           -   starting cross-sectional z-axis radius (cm, float)
    length          -   centerline length (cm, float)
    variation       -   random walk step size (cm, float)
    cap             -   toggle rounded cap on ends (bool)
    ridge_centers   -   x-coord positions for ridge centers (optional, float[])
    cs_offset       -   angular offset of cross section from centerline
    centerline      -   centerline curve (optional, scipy.interpolate.splprep object)

    Returns:
    PointCloud object
    '''
    d_lspacing = 0.1
    d_aspacing = 5 * np.pi / 180
    rad_hist = [[-length / 2 - d_lspacing, rad_y, rad_z]]
    points = []
    offset = 0

    # optional: cap left end
    if cap:
        n_angles = int(np.pi / 2 / d_aspacing)
        for i, phi in enumerate(np.arange(0, np.pi / 2, d_aspacing)):
            offset = (d_aspacing / 2) - offset
            for theta in np.arange(offset, 2 * np.pi, d_aspacing * (n_angles - i) / 2):
                x = -length / 2 - abs(np.sqrt(rad_y ** 2 + rad_z ** 2) * np.cos(phi))
                y = rad_y * np.sin(phi) * np.cos(theta)
                z = rad_z * np.sin(phi) * np.sin(theta)
                if centerline is not None:
                    transform = util.transform.generate_centerline_transform(-length / 2, centerline)
                    coords = np.array([x + length / 2, y, z, 1]).reshape((4, 1))
                    new_coords = np.matmul(transform, coords)
                    x, y, z = new_coords[0], new_coords[1], new_coords[2]
                points.append([x, y, z])

    # generate points
    for x in np.arange(-length / 2, length / 2, d_lspacing):
        # random walk radius
        rad_y = rad_y + np.random.choice([variation, -variation])
        rad_z = rad_z + np.random.choice([variation, -variation])
        rad_hist.append([x, rad_y, rad_z])
        offset = (d_aspacing / 2) - offset
        if centerline is not None:
            transform = util.transform.generate_centerline_transform(x, centerline)
        for angle in np.arange(offset, 2 * np.pi, d_aspacing):
            y = -rad_y * np.cos(angle)
            z = rad_z * np.sin(angle)
            if centerline is not None:
                coords = np.array([0, y, z, 1]).reshape((4, 1))
                new_coords = np.matmul(transform, coords)
                xprime, y, z = new_coords[0], new_coords[1], new_coords[2]
            else:
                xprime = x
            points.append([xprime, y, z])
    rad_hist.append([length / 2, rad_y, rad_z])

    # optional: cap right end
    if cap:
        for i, phi in enumerate(np.arange(np.pi / 2, 0, -d_aspacing)):
            offset = d_aspacing / 2 - offset
            for theta in np.arange(offset, 2 * np.pi, d_aspacing * (i + 1) / 2):
                x = length / 2 + abs(np.sqrt(rad_y ** 2 + rad_z ** 2) * np.cos(phi))
                y = rad_y * np.sin(phi) * np.cos(theta)
                z = rad_z * np.sin(phi) * np.sin(theta)
                if centerline is not None:
                    transform = util.transform.generate_centerline_transform(x, centerline)
                    coords = np.array([x - length / 2, y, z, 1]).reshape((4, 1))
                    new_coords = np.matmul(transform, coords)
                    x, y, z = new_coords[0], new_coords[1], new_coords[2]
                points.append([x, y, z])

    # construct point cloud
    points = np.stack(points, axis=0).reshape((-1, 3))
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.estimate_normals()
    print(pc)

    # add ridges by deforming mesh
    rad_hist = np.array(rad_hist)
    points = []
    values = []
    # centerline position of center
    subsection_length = 0.8 * length / n_ridges
    if ridge_centers is None:
        # ridge_centers = np.random.rand(n_ridges) * 0.8 * length - length/2
        subsection_centers = np.random.rand(n_ridges) * subsection_length
        ridge_centers = [(subsection_length * i + subsection_centers[i]) - 0.8 * length / 2 for i in range(n_ridges)]
    else:
        n_ridges = len(ridge_centers)
        assert np.amin(ridge_centers) > (-length / 2) and np.amax(ridge_centers) < (length / 2), \
            f"Defined ridge centers outside range [{-length / 2},{length / 2}]"
    # angular offset of cross section from centerline
    if cs_offset is None:
        cs_offset = np.random.rand(n_ridges) * np.pi / 6 - (np.pi / 6)
    else:
        cs_offset = np.array(cs_offset)
        assert len(cs_offset) >= n_ridges, f'Expected as least {n_ridges} cs_offset values, got {len(cs_offset)}'
    # angular position of center (-pi,pi)
    ridge_theta = np.random.rand(n_ridges) * 2 * np.pi - np.pi
    # arclength (rads)
    ridge_arclength = np.random.normal(loc=3 * np.pi / 2, scale=np.pi / 8, size=n_ridges)
    # ridge rotation in xy plane (rads)
    ridge_rot = np.random.normal(scale=np.pi / 16, size=n_ridges)
    # ridge width (cm)
    ridge_width = np.random.normal(loc=subsection_length / 2, scale=10, size=n_ridges)
    # ridge height (cm)
    avg_rad = np.abs(np.mean(rad_hist[:, 1:]))
    ridge_height = np.clip(np.random.normal(loc=avg_rad / 3, scale=avg_rad / 4, size=n_ridges), 0.1, avg_rad)
    # clip ridge width to at least height
    ridge_width = np.maximum(ridge_width, ridge_height)
    # compute y and z radii
    interp_yrad = scipy.interpolate.interp1d(rad_hist[:, 0], rad_hist[:, 1])
    interp_zrad = scipy.interpolate.interp1d(rad_hist[:, 0], rad_hist[:, 2])
    rad_y = interp_yrad(ridge_centers)
    rad_z = interp_zrad(ridge_centers)

    # display ridge config
    print('---ridge config (deg)---')
    print('centers:\t', ridge_centers)
    print('cs_offset:\t', cs_offset * 180 / np.pi)
    print('theta:\t\t', ridge_theta * 180 / np.pi)
    print('arclength:\t', ridge_arclength * 180 / np.pi)
    print('rot:\t\t', ridge_rot * 180 / np.pi)
    print('width:\t\t', ridge_width)
    print('height:\t\t', ridge_height)
    print('------------------------')

    # set 0 deformation past ridge region
    xmin = np.amin(ridge_centers - ridge_width)
    xmax = np.amax(ridge_centers + ridge_width)
    n_subdivisions = 100
    for i, x in enumerate(np.arange(-length / 2, xmin, d_lspacing)):
        ry = interp_yrad(x)
        rz = interp_zrad(x)
        for t in np.arange(0, 2 * np.pi, d_aspacing):
            pt = [x, ry * np.cos(t), rz * np.sin(t)]
            points.append(pt)
    print("finish 1")
    for i, x in enumerate(np.arange(xmax, length / 2, d_lspacing)):
        ry = interp_yrad(x)
        rz = interp_zrad(x)
        for t in np.arange(0, 2 * np.pi, d_aspacing):
            pt = [x, ry * np.cos(t), rz * np.sin(t)]
            points.append(pt)
    print("finish 2")
    # set endpoints to 0 deformation
    left_endpoints = np.stack(
        [ridge_centers + rad_z * np.cos(cs_offset), rad_y * np.cos(ridge_theta - ridge_arclength / 2),
         rad_z * np.sin(ridge_theta - ridge_arclength / 2)], axis=-1)
    right_endpoints = np.stack(
        [ridge_centers + rad_z * np.cos(cs_offset), rad_y * np.cos(ridge_theta + ridge_arclength / 2),
         rad_z * np.sin(ridge_theta + ridge_arclength / 2)], axis=-1)
    for i in range(n_ridges):
        points.append(left_endpoints[i])
        points.append(right_endpoints[i])
    # set both sides of center to 0 deformation
    left_endpoints = np.stack([ridge_centers - ridge_width / 2 + rad_z * np.cos(cs_offset), rad_y * np.cos(ridge_theta),
                               rad_z * np.sin(ridge_theta)], axis=-1)
    right_endpoints = np.stack(
        [ridge_centers + ridge_width / 2 + rad_z * np.cos(cs_offset), rad_y * np.cos(ridge_theta),
         rad_z * np.sin(ridge_theta)], axis=-1)
    points.extend(left_endpoints)
    points.extend(right_endpoints)
    values.extend(np.zeros((len(points), 3)))
    print("finish 3")
    # set centerpoint of ridge along subdivisions
    for i in range(n_subdivisions):
        frac = (2 * i) / n_subdivisions - 1
        rh_frac = scipy.special.expit((1 - np.abs(1 - 2 * i / n_subdivisions)) * 12 - 6)
        theta = ridge_theta + (ridge_arclength / 2 * frac)
        base = np.stack([ridge_centers, rad_y * np.cos(theta), rad_z * np.sin(theta)], axis=-1)
        peak = np.stack([ridge_centers, (rad_y - ridge_height * rh_frac) * np.cos(theta),
                         (rad_z - ridge_height * rh_frac) * np.sin(theta)], axis=-1)
        points.extend(base)
        values.extend(peak - base)
        # sigmoid decrease magnitude down sides
        for j in range(n_subdivisions):
            if j == n_subdivisions / 2: continue
            d_width = ridge_width / 2 * (2 * j / n_subdivisions - 1)
            rw_frac = rh_frac * scipy.special.expit((1 - np.abs(1 - 2 * j / n_subdivisions)) * 12 - 6)
            base = np.stack([ridge_centers + d_width, rad_y * np.cos(theta), rad_z * np.sin(theta)], axis=-1)
            peak = np.stack([ridge_centers + d_width, (rad_y - ridge_height * rw_frac) * np.cos(theta),
                             (rad_z - ridge_height * rw_frac) * np.sin(theta)], axis=-1)
            points.extend(base)
            values.extend(peak - base)
    print("finish 4")
    # generate deformation & apply
    # df = util.transform.generate_deformation_field(pc, np.stack(points, axis=0), np.stack(values, axis=0))
    print("finish 5")
    # pc = util.transform.deform(pc, df)

    return pc


def main():
    parser = argparse.ArgumentParser(description='Generate colon-shaped mesh')
    parser.add_argument('--filename', type=str, default='logs/down/pcd_1.obj', help='save location')
    parser.add_argument('--pcd_filename', type=str, default='logs/pcd_2.pcd',
                        help='save point cloud location (default "" no save)')
    parser.add_argument('--rad_y', type=float, default=30, help='average radius in y direction (cm, default 3cm)')
    parser.add_argument('--rad_z', type=float, default=35, help='average radius in z direction (cm, default 3.5cm)')
    parser.add_argument('--length', type=float, default=150, help='average length (cm, default 10cm)')
    parser.add_argument('--variation', type=float, default=100, help='random walk step distance (default 0.02cm)')
    parser.add_argument('--n_ridges', type=int, default=5, help='number of haustral ridges')
    parser.add_argument('--ridge_centers', nargs='*', help='[optional] ridge center x-coords')
    parser.add_argument('--cs_offset', nargs='*', help='[optional] angular cross section offset from centerline')
    parser.add_argument('--cap', action='store_true', help='toggle spherical cap on ends')
    parser.add_argument('-v', '--visualize', action='store_true', help='toggle visualization')
    args = parser.parse_args()
    print(args)

    # # define centerline curve
    # x_pts = np.array([-75, -73, -71, -70, -35,  0, 35, 70,  75,   5,   0])
    # z_pts = np.array([  0,   7,  14,  20,  22, 24, 22, 20,  -5,   5, -20])
    # x_pts = np.array([-75, 0, 40, 75])
    # z_pts = np.zeros_like(x_pts)
    # y_pts = np.zeros_like(x_pts)
    # points = np.stack([x_pts, y_pts, z_pts], axis=0)
    # # param = np.array([  0,  10,  20,  30,  40, 50, 60, 70, 100, 130, 150])
    # param = np.array([0, 75, 115, 150])
    # # normalize parameterization for scale, center at 0
    # param = (param * args.length / 150) - (args.length / 2)
    # curve = util.geometry.generate_centerline_curve(points, param)
    #
    # # generate point cloud
    # rc = None if args.ridge_centers is None else [float(i) for i in args.ridge_centers]
    # cso = None if args.cs_offset is None else [float(i) for i in args.cs_offset]
    # pcd = generate_point_cloud(args.n_ridges,
    #                            args.rad_y,
    #                            args.rad_z,
    #                            args.length,
    #                            args.variation,
    #                            args.cap,
    #                            ridge_centers=rc,
    #                            cs_offset=cso,
    #                            centerline=curve)
    theta_params = [0, np.pi / 12, np.pi / 9, np.pi / 6]
    relrots = [util.transform.rot_y(np.pi / 6),
               np.matmul(util.transform.rot_y(np.pi / 12),
                         util.transform.rot_z(np.pi / 12)),
               np.matmul(util.transform.rot_y(np.pi / 24),
                         util.transform.rot_z(np.pi / 12)),
               ]
    print(len(theta_params))
    cycles = [util.geometry.generate_cycle(r_width=4 * (np.random.randn(1)[0] * 0.5 + 1),
                                           r_height=5,
                                           r_ang_cen=6 * np.pi / 5,
                                           r_ang_cov=5 * np.pi / 2,
                                           p_width=3,
                                           rad_y=4 - t * 3,
                                           rad_z=4 - t * 3,
                                           theta=t,
                                           ) for t in theta_params]
    pcd = util.geometry.join_cycles(cycles, relrot=relrots)
    mesh = util.geometry.build_mesh(pcd)
    print(np.asarray(mesh.vertices).shape)
    # pcd = mesh.sample_points_uniformly(number_of_points=30000)
    alpha = 1.0
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    util.io.save_pcd(args.pcd_filename, pcd)
    print(np.asarray(mesh.vertices).shape)
    print(np.asarray(mesh.triangles).shape)
    # points = np.asarray(pcd.points, dtype=np.float32)
    # np.save("logs/pcd_2.npy", points)
    # cloud = PyntCloud.from_instance("open3d", pcd)
    # k_neighbors = cloud.get_neighbors(k=5)
    # ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    # cloud.add_scalar_field("curvature", ev=ev)

    # save result
    util.io.save_mesh(args.filename, mesh)
    # source_mesh = o3d.io.read_triangle_mesh(args.filename)
    # print(np.asarray(source_mesh.vertices).shape)
    # print(np.asarray(source_mesh.triangles).shape)
    pcd.paint_uniform_color([1, 0, 0])
    # util.io.save_pcd(args.pcd_filename, mesh.vertices)
    # if args.visualize:
    # print(np.asarray(cloud.points["curvature(6)"]))
    # converted_triangle_mesh = cloud.to_instance("open3d", mesh=False)
    o3d.visualization.draw_geometries([pcd], mesh_show_back_face=False)

    # DEBUG ONLY: exit before computing deformation
    exit()

    # # generate deformation field based on insufflation
    # avg_rad = (args.rad_y + args.rad_z) / 2
    # exp_fac = avg_rad / 8
    # d_sample = 10
    # n_df_pts = int(args.length / d_sample)
    # n_div = 36
    # expansion_value = np.random.randint(int(-exp_fac), int(exp_fac), size=n_df_pts)
    # x_coord = np.random.choice(np.arange(int(-args.length/2), int(args.length/2),2),size=n_df_pts,replace=False)
    # indices = x_coord.argsort()
    # spline = scipy.interpolate.splrep(x_coord[indices], expansion_value[indices], s=5)
    # sample_x = np.arange(-args.length/2, args.length/2, 1)
    # theta_range = np.arange(0, 2*np.pi, 2*np.pi/n_div)
    # y_coord = args.rad_y * np.cos(theta_range)
    # z_coord = args.rad_z * np.sin(theta_range)
    # points = np.stack([np.repeat(sample_x, n_div),
    #                    np.tile(y_coord, sample_x.shape[0]),
    #                    np.tile(z_coord, sample_x.shape[0])], axis=1)
    # values = []
    # ev = scipy.interpolate.splev(sample_x, spline, ext=3)
    # for i in range(ev.shape[0]):
    #     for t in theta_range:
    #         values.append([0, ev[i] * np.cos(t), ev[i] * np.sin(t)])
    # values = np.array(values)
    #
    # df = util.transform.generate_deformation_field(pcd, points, values)
    # d_pcd = util.transform.deform(pcd, df)
    # d_mesh = util.geometry.build_mesh(d_pcd)
    # if args.visualize:
    #     o3d.visualization.draw_geometries([d_pcd, d_mesh], mesh_show_back_face=True)
    #
    # # save deformation results
    # pcd_fn_comp = args.pcd_filename.split('.')
    # d_pcd_filename = '.'.join((*pcd_fn_comp[:-2], pcd_fn_comp[-2]+'_deformed', pcd_fn_comp[-1]))
    # util.io.save_pcd(d_pcd_filename, d_pcd)
    # fn_comp = args.filename.split('.')
    # d_filename = '.'.join((*fn_comp[:-2], fn_comp[-2]+'_deformed', fn_comp[-1]))
    # util.io.save_mesh(d_filename, d_mesh)
    # df_filename = '.'.join((*pcd_fn_comp[:-2], pcd_fn_comp[-2]+'_def_field.pkl'))
    # with open(df_filename, 'wb') as f:
    #     pickle.dump(df, f)


def tester():
    x_pts = np.array([-75, -73, -71, -70, -35, 0, 35, 70, 75, 5, 0])
    z_pts = np.array([0, 7, 14, 20, 22, 24, 22, 20, -5, 5, -20])
    y_pts = np.zeros_like(x_pts)
    points = np.stack([x_pts, y_pts, z_pts], axis=0)
    param = np.array([0, 10, 20, 30, 40, 50, 60, 70, 100, 130, 150])
    param = param - 75
    curve = util.geometry.generate_centerline_curve(points, param)

    # plot demo
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_pts, y_pts, z_pts, c='red')
    cm = plt.get_cmap("viridis")
    for u in np.arange(-74, 74, 5):
        i = (u + 75) / 150
        trans_mat = util.transform.generate_centerline_transform(u, curve)
        rad_y = 3
        rad_z = 4
        pts = [[0, 0, 0, 1]]
        for theta in range(0, 360, 10):
            rad_theta = theta * np.pi / 180
            pts.append([0, rad_y * np.cos(rad_theta), rad_z * np.sin(rad_theta), 1])
        pts = np.array(pts).T
        print(pts)
        new_pts = np.matmul(trans_mat, pts)
        print(new_pts)
        ax.scatter(new_pts[0, :], new_pts[1, :], new_pts[2, :], color=cm(i))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-20, 20)
    plt.show()


if __name__ == '__main__':
    main()
    # tester()
