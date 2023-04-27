"""
this script provides toy examples on Robust optimal transpart/spline projection/LDDMM /LDDMM projection/ Discrete flow(point drift)
"""

import os, sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
import numpy as np
import torch
from robot.utils.module_parameters import ParameterDict
from robot.datasets.data_utils import get_file_name, generate_pair_name, get_obj
from robot.shape.shape_pair_utils import create_shape_pair
from robot.models_reg.multiscale_optimization import (
    build_single_scale_model_embedded_solver,
    build_multi_scale_solver,
)
from robot.global_variable import MODEL_POOL, Shape, shape_type
from robot.utils.utils import get_grid_wrap_points
from robot.utils.visualizer import (
    visualize_point_fea,
    visualize_point_pair,
    visualize_multi_point, visualize_source_flowed_target_overlap,
)
from robot.demos.demo_utils import *
from robot.utils.utils import timming
from robot.experiments.datasets.toy.visualizer import toy_plot
import open3d as o3d
# set shape_type = "pointcloud"  in global_variable.py
assert (
    shape_type == "pointcloud"
), "set shape_type = 'pointcloud'  in global_variable.py"
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
server_path = "../../data/new/"
base_path = "/shenlab/lab_stor/yunkuipa/data/new/"
source_path = server_path + "pcd_1_rigid.obj"
target_path = server_path + "pcd_2.obj"


####################  prepare data ###########################
pair_name = generate_pair_name([source_path, target_path])
reader_obj = "toy_dataset_utils.toy_reader()"
sampler_obj = "toy_dataset_utils.toy_sampler()"
normalizer_obj = "toy_dataset_utils.toy_normalizer()"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device)
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval, target_interval)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
shape_pair = create_shape_pair(source, target)
shape_pair.pair_name = "toy"


""" Experiment 1:  Robust optimal transport """
task_name = "gradient_flow"
solver_opt = ParameterDict()
record_path = base_path + "output/toy_reg/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt["record_path"] = record_path
model_name = "gradient_flow_opt"
model_opt = ParameterDict()
model_opt[
    "interpolator_obj"
] = "point_interpolator.nadwat_kernel_interpolator(scale=1, exp_order=1)"
model_opt[("sim_loss", {}, "settings for sim_loss_opt")]
model_opt["sim_loss"]["loss_list"] = ["geomloss"]
model_opt["sim_loss"][("geomloss", {}, "settings for geomloss")]
model_opt["sim_loss"]["geomloss"]["attr"] = "points"
blur = 0.005
reach = 0.1  # 0.1  # change the value to explore behavior of the OT
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur={}, scaling=0.1,debias=False,reach={})".format(
    blur, reach
)
model = MODEL_POOL[model_name](model_opt)
solver = build_single_scale_model_embedded_solver(solver_opt, model)
model.init_reg_param(shape_pair)
shape_pair = timming(solver)(shape_pair)
print("the registration complete")
# fea_to_map = shape_pair.source.weights[0]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(shape_pair.flowed.points.detach().cpu().numpy()[0][:, :3])
o3d.io.write_point_cloud(base_path + "flow_1.pcd", pcd)

pcdw = o3d.geometry.PointCloud()
pcdw.points = o3d.utility.Vector3dVector(shape_pair.target.points.detach().cpu().numpy()[0][:, :3])
o3d.io.write_point_cloud(base_path + "tar_pcd.pcd", pcdw)


print("start robot lddmm")
import time
time.sleep(5)

source_path = base_path + "flow_1.pcd"
target_path = base_path + "tar_pcd.pcd"


####################  prepare data ###########################
pair_name = generate_pair_name([source_path, target_path])
reader_obj = "toy_dataset_utils.toy_reader2()"
sampler_obj = "toy_dataset_utils.toy_sampler()"
normalizer_obj = "toy_dataset_utils.toy_normalizer()"
get_obj_func = get_obj(reader_obj, normalizer_obj, sampler_obj, device)
source_obj, source_interval = get_obj_func(source_path)
target_obj, target_interval = get_obj_func(target_path)
min_interval = min(source_interval, target_interval)
input_data = {"source": source_obj, "target": target_obj}
create_shape_pair_from_data_dict = obj_factory(
    "shape_pair_utils.create_source_and_target_shape()"
)
source, target = create_shape_pair_from_data_dict(input_data)
shape_pair = create_shape_pair(source, target)
shape_pair.pair_name = "toy"

task_name = "gradient_flow_guided_by_lddmm"
solver_opt = ParameterDict()
record_path = server_path + "output/toy_reg/{}".format(task_name)
os.makedirs(record_path, exist_ok=True)
solver_opt["record_path"] = record_path
solver_opt["point_grid_scales"] = [-1]
solver_opt["iter_per_scale"] = [150]
solver_opt["rel_ftol_per_scale"] = [1e-9,]
solver_opt["init_lr_per_scale"] = [8e-2]
solver_opt["save_3d_shape_every_n_iter"] = 10
solver_opt["shape_sampler_type"] = "point_grid"
solver_opt["stragtegy"] = "use_optimizer_defined_here"
solver_opt["optim"]["type"] = "sgd"  # lbgfs
solver_opt["scheduler"]["type"] = "step_lr"
solver_opt["scheduler"]["step_lr"]["gamma"] = 0.9
solver_opt["scheduler"]["step_lr"]["step_size"] = 100

model_name = "lddmm_opt"
model_opt = ParameterDict()
model_opt["running_result_visualize"] = True
model_opt["saving_running_result_visualize"] = False
model_opt["module"] = "hamiltonian"
model_opt["hamiltonian"][
    "kernel"
] = "keops_kernels.LazyKeopsKernel(kernel_type='multi_gauss', sigma_list=[0.05,0.1, 0.2],weight_list=[0.2,0.3, 0.5])"
model_opt["use_gradflow_guided"] = True
model_opt["gradflow_guided"]["mode"] = "ot_mapping"
model_opt["gradflow_guided"]["update_gradflow_every_n_step"] = 10
model_opt["gradflow_guided"]["gradflow_blur_init"] = 0.005  # 0.05
model_opt["gradflow_guided"]["update_gradflow_blur_by_raito"] = 0.5
model_opt["gradflow_guided"]["gradflow_blur_min"] = 0.005
model_opt["gradflow_guided"]["geomloss"][
    "attr"
] = "points"  # todo  the pointfea will be  more generalized choice
model_opt["gradflow_guided"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur=blurplaceholder, scaling=0.8,debias=False, backend='online')"

model_opt["sim_loss"]["loss_list"] = ["l2"]
model_opt["sim_loss"]["l2"]["attr"] = "points"
model_opt["sim_loss"]["geomloss"]["attr"] = "points"
model_opt["sim_loss"]["geomloss"][
    "geom_obj"
] = "geomloss.SamplesLoss(loss='sinkhorn',blur=blurplaceholder, scaling=0.8, debias=False, backend='online')"
# shape_pair.flowed.points = shape_pair.flowed.points
model = MODEL_POOL[model_name](model_opt)
model.init_reg_param(shape_pair, force=True)
solver = build_multi_scale_solver(solver_opt, model)
shape_pair = solver(shape_pair)
print("the registration complete")

pcd = o3d.geometry.PointCloud()
print(shape_pair.flowed.points.shape)
pcd.points = o3d.utility.Vector3dVector(shape_pair.flowed.points.detach().cpu().numpy()[0][:, :3])
o3d.io.write_point_cloud(base_path + "flow.pcd", pcd)
