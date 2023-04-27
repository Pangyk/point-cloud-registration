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
source_mesh = o3.io.read_triangle_mesh('data/down/pcd_1.obj')
target_mesh = o3.io.read_triangle_mesh('data/down/pcd_2.obj')
# transform target point cloud
th = np.deg2rad(30.0)
source = o3.geometry.PointCloud()
source.points = o3.utility.Vector3dVector(np.asarray(source_mesh.vertices, np.float32))
target = o3.geometry.PointCloud()
target.points = o3.utility.Vector3dVector(np.asarray(target_mesh.vertices, np.float32))
# transform target point cloud
th = np.deg2rad(30.0)

source_pt = cp.asarray(source.points, dtype=cp.float32)
target_pt = cp.asarray(target.points, dtype=cp.float32)

source2 = source.voxel_down_sample(voxel_size=0.4)
target2 = target.voxel_down_sample(voxel_size=0.4)

source_pt2 = cp.asarray(source2.points, dtype=cp.float32)
target_pt2 = cp.asarray(target2.points, dtype=cp.float32)
print((np.asarray(source_mesh.vertices).shape), len(np.asarray(target_mesh.vertices)))
print(len(np.asarray(source.points)), len(np.asarray(target.points)))
print(len(np.asarray(source2.points)), len(np.asarray(target2.points)))

print("start reg")

# compute cpd registration
acpd = cpd.AffineCPD(source_pt, use_cuda=use_cuda)
tf_param, _, _ = acpd.registration(target_pt)
result = tf_param.transform(source_pt)

# result = tf_param.transform(result)
mesh = o3.geometry.TriangleMesh()
np_vertices = to_cpu(result)
np_triangles = np.array(source_mesh.triangles).astype(np.int32)
mesh.vertices = o3.utility.Vector3dVector(np_vertices)
mesh.triangles = o3.utility.Vector3iVector(np_triangles)
o3.io.write_triangle_mesh('data/down/pcd_1_rigid.obj', mesh)
