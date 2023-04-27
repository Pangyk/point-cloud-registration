import open3d as o3
import numpy as np
import torch
from chamferdist import ChamferDistance
from geomloss import SamplesLoss

source = o3.io.read_point_cloud('data/down/new_flow_1.pcd')
# source = o3.io.read_triangle_mesh('data/down/pcd_1.obj')
target = o3.io.read_triangle_mesh('data/down/pcd_2.obj')

source_pt = torch.FloatTensor(np.asarray(source.points, dtype=np.float32))
target_pt = torch.FloatTensor(np.asarray(target.vertices, dtype=np.float32))

source_pt = source_pt.unsqueeze(0)
target_pt = target_pt.unsqueeze(0)

chamferDist = ChamferDistance()
dist_forward = chamferDist(source_pt, target_pt)
print(dist_forward.detach().item())
loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
L = loss(source_pt, target_pt)
print(L.detach().item())



