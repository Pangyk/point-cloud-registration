import copy
import numpy as np
use_cuda = True
if use_cuda:
    import cupy as cp
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x
import open3d as o3
from probreg import cpd
from probreg import callbacks
import utils
import time
from utils import estimate_normals

# load source and target point cloud
source_mesh = o3.io.read_triangle_mesh('data/down/pcd_1_rigid.obj')
target_mesh = o3.io.read_triangle_mesh('data/down/pcd_2.obj')
# transform target point cloud
th = np.deg2rad(30.0)
source = o3.geometry.PointCloud()
source.points = o3.utility.Vector3dVector(np.asarray(source_mesh.vertices))
target = o3.geometry.PointCloud()
target.points = o3.utility.Vector3dVector(np.asarray(target_mesh.vertices))

source_pt = cp.asarray(source.points, dtype=cp.float32)
target_pt = cp.asarray(target.points, dtype=cp.float32)

print(len(np.asarray(source.points)), len(np.asarray(target.points)))

print("start non-rigid")
acpd = cpd.NonRigidCPD(source_pt, use_cuda=use_cuda)
tf_param, _, _ = acpd.registration(target_pt)

result = tf_param.transform(source_pt)
pc = o3.geometry.PointCloud()
pc.points = o3.utility.Vector3dVector(to_cpu(result))

# draw result
o3.io.write_point_cloud('data/down/tar_pcd.pcd', target)
o3.io.write_point_cloud('data/down/reg_pcd.pcd', pc)
# o3.visualization.draw_geometries([source, target, result])
