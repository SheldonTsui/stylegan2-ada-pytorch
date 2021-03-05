import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import open3d as o3d

from laplacian import get_disp
# get_disp(obj_file, scan_file, out_file)

#base_path = "/home/xuxudong/3D/data/Multi-Garment/Multi-Garment_dataset"
#folders = sorted(glob(osp.join(base_path, '12*')))
base_path = "/home/xuxudong/3D/data/THUman/dataset"
folders = sorted(glob(osp.join(base_path, 'result*/*')))
print("# folders:", len(folders))

N_iters = 2
for folder in tqdm(folders):
    scan_file = osp.join(folder, 'mesh.obj')
    #print(folder.split('/')[-1])
    for n_iter in range(N_iters):
        if n_iter == 0:
            input_obj_file = osp.join(folder, 'smpl_expanded_align.ply')
        else:
            input_obj_file = osp.join(folder, 'smpl_disp.ply')

        out_disp_file = osp.join(folder, 'smpl_disp.ply')
        get_disp(input_obj_file, scan_file, out_disp_file)
