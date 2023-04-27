"""
data reader for the toys
given a file path, the reader will return a dict
{"points":Nx3, "weights":Nx1, "faces":Nx3}
"""
import numpy as np
import open3d as o3
from pyntcloud import PyntCloud
from robot.datasets.vtk_utils import read_vtk
import igl
import scipy as sp


thresh = 0.05



def normalize(x):
    res = (x - np.min(x)) / (np.max(x) - np.min(x))
    # res = np.where(res < thresh, np.zeros_like(x), res)
    return res



def read_pcd(path):
    print(path)
    data = o3.io.read_point_cloud(path)
    data_dict = {}
    data_dict["points"] = np.asarray(data.points, np.float32)
    data_dict["faces"] = np.zeros([2000, 3])
    # for name in data.array_names:
    #     try:
    #         data_dict[name] = data[name]
    #     except:
    #         pass
    return data_dict
    
    
def read_pcd2(path):
    mesh = o3.io.read_triangle_mesh(path)
    data_dict = {}
    # cloud = PyntCloud.from_instance("open3d", data)
    # k_neighbors = cloud.get_neighbors(k=10)
    # ev = cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)
    # cloud.add_scalar_field("curvature", ev=ev)
    # pcd = np.asarray(data.points, dtype=np.float32)
    
    # base[:, :3] = np.asarray(mesh.vertices, dtype=np.float32)
    v, f = np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.triangles, dtype=np.int32)
    base = np.zeros([len(v), 5], dtype=np.float32)
    base[:, :3] = np.asarray(v, dtype=np.float32)
    k = igl.gaussian_curvature(v, f)
    l = igl.cotmatrix(v, f)
    m = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    minv = sp.sparse.diags(1 / m.diagonal())
    hn = -minv.dot(l.dot(v))
    h = np.linalg.norm(hn, axis=1)
    c = np.asarray(k, dtype=np.float32)
    cn = np.asarray(h, dtype=np.float32)
    base[:, 3] = normalize(np.nan_to_num(c))
    base[:, 4] = normalize(np.nan_to_num(cn))
    data_dict["points"] = base
    # data_dict["points"] = np.asarray(mesh.vertices, dtype=np.float32)
    data_dict["faces"] = np.zeros([2000, 3])
    # for name in data.array_names:
    #     try:
    #         data_dict[name] = data[name]
    #     except:
    #         pass
    return data_dict
    
    
def toy_reader2():
    """
    :return:
    """
    # reader = read_vtk
    reader = read_pcd

    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["faces"] = raw_data_dict["faces"]
        print(data_dict["points"].shape)
        return data_dict

    return read


def toy_reader():
    """
    :return:
    """
    # reader = read_vtk
    reader = read_pcd2

    def read(file_info):
        path = file_info["data_path"]
        raw_data_dict = reader(path)
        data_dict = {}
        data_dict["points"] = raw_data_dict["points"]
        data_dict["faces"] = raw_data_dict["faces"]
        print(data_dict["points"].shape)
        return data_dict

    return read


def toy_sampler():
    """
    :param args:
    :return:
    """

    def do_nothing(data_dict):
        return data_dict, None

    return do_nothing


def toy_normalizer(scale=1, add_random_noise_on_weight=False):
    """
    :return:
    """

    def scale_data(data_dict):
        data_dict["points"] = data_dict["points"] *scale
        return data_dict
    def randomized_weight(data_dict):
        weights = data_dict["weights"]
        min_weight = np.min(weights)
        npoints = len(weights)
        rand_noise =np.random.rand(npoints) * abs(min_weight)/10
        weights = weights + rand_noise
        data_dict["weights"] = weights/np.sum(weights)
        return data_dict
    return scale_data if not add_random_noise_on_weight else randomized_weight


if __name__ == "__main__":
    from robot.utils.obj_factory import obj_factory

    reader_obj = "toy_dataset_utils.toy_reader()"
    sampler_obj = "toy_dataset_utils.toy_sampler()"
    normalizer_obj = "toy_dataset_utils.toy_normalizer()"
    reader = obj_factory(reader_obj)
    normalizer = obj_factory(normalizer_obj)
    sampler = obj_factory(sampler_obj)
    file_path = "/playpen-raid1/zyshen/proj/robot/settings/datasets/toy/toy_synth/divide_3d_sphere_level1.vtk"
    file_info = {"name": file_path, "data_path": file_path}
    raw_data_dict = reader(file_info)
    normalized_data_dict = normalizer(raw_data_dict)
    sampled_data_dict = sampler(normalized_data_dict)
