# smoothing: an add-on option for the post-processing for the displaced mesh
# check: check the vertices of the displaced mesh are aligned with original expanded smpl mesh

import os
import os.path as osp
import open3d as o3d
import numpy as np
from glob import glob

def check_align(folder):
    obj_file = osp.join(folder, 'smpl_disp.ply')
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.paint_uniform_color([1,1,1])
    
    # painting to check the alignment
    N = 6890
    color = np.repeat(np.array([1, 0.706, 0])[np.newaxis, :], N, axis=0)
    mesh.vertex_colors[:6890] = o3d.utility.Vector3dVector(color) 
    o3d.io.write_triangle_mesh(osp.join(folder, 'colored_trimesh.ply'), mesh)
    
def smoothness(obj_file, out_obj_file):
    mesh = o3d.io.read_triangle_mesh(obj_file)
    # smoothing
    mesh_out = mesh.filter_smooth_simple(number_of_iterations=1)
    #mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=2)
    #mesh_out = mesh.filter_smooth_taubin(number_of_iterations=3)
    o3d.io.write_triangle_mesh(out_obj_file, mesh_out)

if __name__ == '__main__':
    base_path = '/home/xuxudong/3D/data/samples/Multi-Garment_dataset'
    folders = glob(osp.join(base_path, '1*'))
    print("# folders:", len(folders))

    for folder in folders:
        check_align(folder)
    """
    folder = '/home/xuxudong/3D/data/samples/THUman/13010'
    obj_file = osp.join(folder, 'UV_sampled.obj')
    out_obj_file = osp.join(folder, 'UV_smooth.ply')
    smoothness(obj_file, out_obj_file)
    """
