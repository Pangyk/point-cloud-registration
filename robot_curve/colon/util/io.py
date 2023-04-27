import os
import open3d as o3d


def save_pcd(filename, pcd):
    '''
    Saves point cloud to filename and creates intermediate directories if needed.

    Args:
    filename    -   full save file name (str)
    pcd         -   PointCloud object

    Returns:
    None
    '''
    if filename == '':
        return
    dir = os.path.join(*os.path.split(filename)[:-1])
    if not os.path.isdir(dir):
        os.makedirs(dir)
    o3d.io.write_point_cloud(filename, pcd)


def save_mesh(filename, mesh):
    '''
    Saves mesh to filename and creates intermediate directories if needed.

    Args:
    filename    -   full save file name (str)
    mesh        -   TriangleMesh object

    Returns:
    None
    '''
    dir = os.path.join(*os.path.split(filename)[:-1])
    if not os.path.isdir(dir):
        os.makedirs(dir)
    o3d.io.write_triangle_mesh(filename, mesh)
