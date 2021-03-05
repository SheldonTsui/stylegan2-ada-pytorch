import os
import os.path as osp
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from get_smpl_mesh.get_smpl import SMPL_Obtainer
from expand_smpl import expand_smpl_model
from uv_generator import Index_UV_Generator
from objfile import read_obj, write_obj

smpl_obtainer = SMPL_Obtainer()

base_path = '/home/xuxudong/3D/data/THUman'
label_file = osp.join(base_path, 'THUman_params.npy')
labels = np.load(label_file)
disp_files = sorted(glob(osp.join(base_path, 'dataset', 'result*/*/disp_uv_map_256.npy')))
visualize_path = './out'

for idx, label in enumerate(labels[:5]):
    disp_file = disp_files[idx]
    disp_map = np.load(disp_file)
    tmp_mesh_out_file = osp.join(visualize_path, f'smpl_{idx:04d}.obj')
    tmp_mesh_expand_file = osp.join(visualize_path, f'smpl_{idx:04d}_expand.obj')
    smpl_obtainer.get_smpl_pose_shape(pose=label[10:], shape=label[:10], out_mesh_file=tmp_mesh_out_file)
    expand_smpl_model(obj_file=tmp_mesh_out_file, out_obj_file=tmp_mesh_expand_file)
    # 
    uv_map_size = disp_map.shape[1]
    data_dir = '/home/xuxudong/3D/DecoMR/data/uv_sampler'
    generator = Index_UV_Generator(
        UV_height = uv_map_size,
        UV_width = uv_map_size,
        uv_type = 'SMPL',
        data_dir = data_dir
    )
    disp_v = generator.resample(torch.from_numpy(disp_map))
    disp_v = disp_v.squeeze(0).numpy()
    vertices, faces = read_obj(tmp_mesh_expand_file)
    vertices = np.stack(vertices) + disp_v
    
    write_obj(osp.join(visualize_path, f'smpl_{idx:04d}_gen.obj'), vertices, faces)
    #os.remove(tmp_mesh_expand_file)
    #os.remove(tmp_mesh_out_file)