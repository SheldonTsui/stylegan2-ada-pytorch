import os
import os.path as osp
from glob import glob
import numpy as np
import open3d as o3d

def param_parsing(param_txt_file):
    with open(param_file, 'r') as cur_file:
        param_content = cur_file.readlines()
    transformation_content = param_content[3: 7]
    transformation = []
    for line in transformation_content:
        transformation.append([float(item) for item in line[:-1].split()])
    transformation = np.stack(transformation)
    assert transformation.shape == (4,4)

    return transformation

base_path = '/home/xuxudong/3D/data/samples/THUman/results_gyc_20181010_wyl_1_M'
folders = sorted(glob(osp.join(base_path, '*')))
for folder in folders:
    param_file = osp.join(folder, 'smpl_params.txt')
    transformation = param_parsing(param_file)
    #print(transformation)
    mesh = o3d.io.read_triangle_mesh(osp.join(folder, 'smpl_neutral.obj'))
    mesh = mesh.transform(transformation)
    o3d.io.write_triangle_mesh(osp.join(folder, 'smpl_trans.ply'), mesh)