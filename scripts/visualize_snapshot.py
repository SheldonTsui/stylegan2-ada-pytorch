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

snapshot_path = '../training-runs/00000-THUman-cond-auto8'
npy_list = sorted(glob(osp.join(snapshot_path, 'reals.npy')))
start = 10
stride = 5
resolution = 256
visualize_path = './out'
EXPAND = 8
with_EXPAND = False 
for npy_file in tqdm(npy_list):
    disp_contents = np.load(npy_file)
    H, W, c = disp_contents.shape
    gh, gw = H // resolution, W // resolution
    disp_maps = disp_contents.reshape(gh, resolution, gw, resolution, c).transpose(0,2,4,1,3).reshape(-1, resolution, resolution, c)
    if with_EXPAND:
        disp_maps /= with_EXPAND
    labels = np.load(npy_file.replace('.npy', '_label.npy'))
    assert disp_maps.shape[0] == labels.shape[0]
    
    npy_stride = 30
    for idx, disp_map in enumerate(disp_maps[::npy_stride]):
        idx = idx * npy_stride
        label = labels[idx]
        npy_file_idx = 'real' # npy_file[-4-6:-4]
        tmp_mesh_out_file = osp.join(visualize_path, f'smpl_{npy_file_idx}_{idx:04d}.obj')
        tmp_mesh_expand_file = osp.join(visualize_path, f'smpl_{npy_file_idx}_{idx:04d}_expand.obj')
        smpl_obtainer.get_smpl_pose_shape(pose=label[10:], shape=label[:10], out_mesh_file=tmp_mesh_out_file)
        expand_smpl_model(obj_file=tmp_mesh_out_file, out_obj_file=tmp_mesh_expand_file)
        # 
        uv_map_size = disp_map.shape[0]
        data_dir = '/home/xuxudong/3D/DecoMR/data/uv_sampler'
        generator = Index_UV_Generator(
            UV_height = uv_map_size,
            UV_width = uv_map_size,
            uv_type = 'SMPL',
            data_dir = data_dir
        )
        disp_v = generator.resample(torch.from_numpy(disp_map[np.newaxis, :]))
        disp_v = disp_v.squeeze(0).numpy().clip(min=-1/EXPAND, max=1/EXPAND)
        vertices, faces = read_obj(tmp_mesh_expand_file)
        vertices = np.stack(vertices) + disp_v
        
        write_obj(osp.join(visualize_path, f'smpl_{npy_file_idx}_{idx:04d}_gen.obj'), vertices, faces)
        #os.remove(tmp_mesh_expand_file)
        #os.remove(tmp_mesh_out_file)
