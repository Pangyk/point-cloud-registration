import copy
import numpy as np

import open3d as o3
from probreg import cpd
from probreg import callbacks
from probreg import filterreg
from probreg import transformation
import utils
import time

# load source and target point cloud
source = o3.io.read_point_cloud('pcd_0.pcd')
source.remove_non_finite_points()
target = o3.io.read_point_cloud('pcd_2.pcd')
# transform target point cloud
th = np.deg2rad(30.0)
source2 = source.voxel_down_sample(voxel_size=0.18)
target2 = target.voxel_down_sample(voxel_size=0.18)

source_pt = source2
target_pt = target2

source_pt2 = source
target_pt2 = target

print(len(np.asarray(source2.points)), len(np.asarray(target2.points)))

print("start reg")

# compute cpd registration
objective_type = 'pt2pt'
tf_param, _, _ = filterreg.registration_filterreg(source_pt.points, target_pt.points,
                                                  objective_type=objective_type,
                                                  sigma2=None,
                                                  update_sigma2=True)
result = tf_param.transform(source_pt.points)

print("start non-rigid")
#cbs = [callbacks.Open3dVisualizerCallback(result, target)]
cv = lambda x: np.asarray(x.points if isinstance(x, o3.geometry.PointCloud) else x)
n_points = len(np.asarray(result))
ws = transformation.DeformableKinematicModel.SkinningWeight(n_points)
reg = filterreg.DeformableKinematicFilterReg(cv(result), ws, 0.18)
#reg.set_callbacks(cbs)
tf_param, _, _ = reg.registration(cv(target_pt))

result = tf_param.transform(result)
pc = o3.geometry.PointCloud()
pc.points = o3.utility.Vector3dVector(result)


# draw result
o3.io.write_point_cloud('tar_pcd.pcd', target2)
o3.io.write_point_cloud('reg_pcd.pcd', pc)
# o3.visualization.draw_geometries([source, target, result])
