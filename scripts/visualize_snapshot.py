import os
import os.path as osp
import torch
import numpy as np
import mmcv
from glob import glob
from tqdm import tqdm
import open3d as o3d

from get_smpl_mesh.get_smpl import SMPL_Obtainer
from expand_smpl import expand_smpl_model
from uv_generator import Index_UV_Generator
from objfile import read_obj, write_obj
from disp_modulation import disp_modulate, disp_demodulate

class Mesh_Handler(object):
    def __init__(self,
        resolution,
        data_dir,
        retrieval_SMPL_model=True,
        gender='neutral',
        orig_label_path='',
        expanded_mesh_dir=''
    ):
        self.resolution = resolution
        self.generator = Index_UV_Generator(
            UV_height = resolution,
            UV_width = resolution,
            uv_type = 'SMPL',
            data_dir = data_dir
        )

        self.retrieval_SMPL_model = retrieval_SMPL_model
        if not retrieval_SMPL_model:
            self.smpl_obtainer = SMPL_Obtainer(gender=gender)
        else:
            assert osp.isfile(orig_label_path)
            assert orig_label_path.split('.')[-1] in ['npy', 'npz']
            assert osp.isdir(expanded_mesh_dir)
            self.orig_labels = np.load(orig_label_path)
            self.expanded_mesh_files = sorted(
                glob(osp.join(expanded_mesh_dir, 'dataset', 'result*/*/smpl_expanded.obj'))
            )
            assert len(self.orig_labels) == len(self.expanded_mesh_files)

    def get_disped_mesh(self, disp_map, expanded_mesh_file, out_mesh_file):
        disp_v = self.generator.resample(torch.from_numpy(disp_map[np.newaxis, :]))
        disp_v = disp_v.squeeze(0).numpy()

        vertices, faces = read_obj(expanded_mesh_file)
        vertices = np.stack(vertices) + disp_v
            
        write_obj(out_mesh_file, vertices, faces)

    def get_expanded_mesh(self, label, out_label=''):
        if self.retrieval_SMPL_model:
            EPS = 1e-6
            dists = np.linalg.norm(label - self.orig_labels, axis=1)
            indices = np.where(dists < EPS)
            assert len(indices) == 1
            expanded_mesh_file = self.expanded_mesh_files[int(indices[0])]
        else:
            tmp_mesh_out_file = osp.join('out', f'smpl_{out_label}.obj')
            tmp_mesh_expand_file = osp.join('out', f'smpl_{out_label}_expand.obj')
            self.smpl_obtainer.get_smpl_pose_shape(
                pose=label[10:], shape=label[:10], out_mesh_file=tmp_mesh_out_file)
            expand_smpl_model(obj_file=tmp_mesh_out_file, out_obj_file=tmp_mesh_expand_file)
            expanded_mesh_file = tmp_mesh_expand_file

            os.remove(tmp_mesh_expand_file)
            os.remove(tmp_mesh_out_file)

        return expanded_mesh_file

    def save_normal_map(self, normals, out_img_file):
        assert type(normals) is np.ndarray
        normals = (normals - normals.min(axis=0, keepdims=True)) / (
            normals.max(axis=0, keepdims=True) - normals.min(axis=0, keepdims=True))
        uv_map = self.generator.get_UV_map(torch.from_numpy(normals))
        img = uv_map.squeeze(0).numpy()
        img /= img.max()
        img *= 255
        mmcv.imwrite(img, out_img_file)

    def get_normal_map(self, input_mesh_file, out_img_file):
        mesh = o3d.io.read_triangle_mesh(input_mesh_file)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        self.save_normal_map(normals, out_img_file)

    def get_disp_normal_map(self, input_mesh_file, expanded_mesh_file, out_img_file):
        mesh = o3d.io.read_triangle_mesh(input_mesh_file)
        expanded_mesh = o3d.io.read_triangle_mesh(expanded_mesh_file)
        disp_vertices = np.asarray(mesh.vertices) - np.asarray(expanded_mesh.vertices)
        mesh.vertices = o3d.utility.Vector3dVector(disp_vertices)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        self.save_normal_map(normals, out_img_file)

    def get_normal_map_local(self, input_mesh_file, expanded_mesh_file, out_img_file):
        mesh = o3d.io.read_triangle_mesh(input_mesh_file)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)

        expanded_mesh = o3d.io.read_triangle_mesh(expanded_mesh_file)
        expanded_mesh.compute_vertex_normals()
        expanded_normals = np.asarray(expanded_mesh.vertex_normals)

        normals_local = normals - expanded_normals
        self.save_normal_map(normals_local, out_img_file)

def stylegan2_visualize(mesh_handler, save_img=False):
    snapshot_path = '../training-runs/00002-THUman-cond-auto8'
    # npy_list = sorted(glob(osp.join(snapshot_path, 'fakes*[0-9].npy')))
    npy_list = sorted(glob(osp.join(snapshot_path, 'reals.npy')))
    start = 0
    stride = 1

    def process_npy_file(npy_file, resolution=256):
        disp_contents = np.load(npy_file)
        H, W, c = disp_contents.shape
        gh, gw = H // resolution, W // resolution
        disp_maps = disp_contents.reshape(gh, resolution, gw, resolution, c)
        disp_maps = disp_maps.transpose(0,2,1,3,4).reshape(-1, resolution, resolution, c)

        return disp_maps

    exp_mode = 'new_exp' # 'old_exp' or 'new_exp'
    # out path
    visualize_path = './out/stylegan2/exp2/real'
    if not osp.exists(visualize_path):
        os.mkdir(visualize_path)

    for npy_file in tqdm(npy_list[start::stride]):
        disp_maps = process_npy_file(npy_file, resolution=resolution)
        labels = np.load(npy_file.replace('.npy', '_label.npy'))
        assert disp_maps.shape[0] == labels.shape[0]

        npy_stride = 30
        for idx, disp_map in enumerate(disp_maps[::npy_stride]):
            idx = idx * npy_stride
            npy_file_idx = osp.basename(npy_file).split('.')[0]
            label = labels[idx]
            out_label = f'{npy_file_idx}_{idx:04d}'
            if exp_mode == 'new_exp':
                disp_img = (disp_map + 1) / 2 * 255
                disp_img = np.rint(disp_img).clip(0, 255).astype(np.uint8)
                disp_map = disp_demodulate(disp_img)
            elif exp_mode == 'old_exp':
                disp_img = disp_modulate(disp_map, max_value=255)
            else:
                raise ValueError("please input correct exp mode")

            if save_img:
                mmcv.imwrite(disp_img, osp.join(visualize_path, f'UV_{out_label}.jpg'))

            expanded_mesh_file = mesh_handler.get_expanded_mesh(label, out_label)
            out_mesh_file = osp.join(visualize_path, f'smpl_{out_label}_gen.obj')
            mesh_handler.get_disped_mesh(disp_map, expanded_mesh_file, out_mesh_file)

def GT_visualize(mesh_handler, out_json_file='', save_img=False):
    base_path = '/home/xuxudong/3D/data/THUman/dataset'
    folders = sorted(glob(osp.join(base_path, 'results_gyc_20181010_hsc_1_M', '*')))
    out_folder = 'out/GT'
    folders = folders[::5]
    expanded_mesh_dict = {}
    
    for idx, folder in enumerate(folders):
        disp_file = osp.join(folder, 'disp_uv_map_256.npy')

        disp_map = np.load(disp_file)[0]
        disp_img = disp_modulate(disp_map, max_value=255)
        if save_img:
            mmcv.imwrite(disp_img, osp.join(out_folder, f'UV_{idx:04d}.jpg'))
        
        disp_map = disp_demodulate(disp_img)
        expanded_mesh_file = osp.join(folder, 'smpl_expanded.obj')
        out_mesh_file = osp.join(out_folder, f'smpl_{idx:04d}_gen.obj')
        mesh_handler.get_disped_mesh(disp_map, expanded_mesh_file, out_mesh_file)
        expanded_mesh_dict[idx] = expanded_mesh_file

    if len(out_json_file) > 0:
        mmcv.dump(expanded_mesh_dict, osp.join(out_folder, out_json_file))

def CIPS_visualize(mesh_handler, out_json_file='', save_img=False, n_iter=50000):
    src_path = '/home/xuxudong/3D/CIPS/outputs/skip-THUman/samples'
    disp_imgs = np.load(osp.join(src_path, f'sample_imgs_{n_iter}.npy'))
    labels = np.load(osp.join(src_path, f'sample_c_{n_iter}.npy'))
    out_folder = f'out/CIPS/with_label/{n_iter}'
    if not osp.exists(out_folder):
        os.mkdir(out_folder)
    expanded_mesh_dict = {}

    idx = 0
    for label, disp_img in zip(labels, disp_imgs):
        disp_img = (disp_img + 1) / 2 * 255
        disp_img = np.rint(disp_img).clip(0, 255).astype(np.uint8)
        disp_img = disp_img.transpose(1, 2, 0) # to shape [H, W, C]
        if save_img:
            mmcv.imwrite(disp_img, osp.join(out_folder, f'UV_{idx:04d}.jpg'))
        
        disp_map = disp_demodulate(disp_img)
        expanded_mesh_file = mesh_handler.get_expanded_mesh(label)
        out_mesh_file = osp.join(out_folder, f'smpl_{idx:04d}_gen.obj')
        mesh_handler.get_disped_mesh(disp_map, expanded_mesh_file, out_mesh_file)
        expanded_mesh_dict[idx] = expanded_mesh_file

        idx += 1

    if len(out_json_file) > 0:
        mmcv.dump(expanded_mesh_dict, osp.join(out_folder, out_json_file)) 

if __name__ == '__main__':
    resolution = 256
    data_dir = '/home/xuxudong/3D/DecoMR/data/uv_sampler'
    expanded_mesh_dir = '/home/xuxudong/3D/data/THUman'
    orig_label_path = osp.join(expanded_mesh_dir, 'THUman_params.npy')
    mesh_handler = Mesh_Handler(
        resolution=resolution,
        data_dir=data_dir,
        retrieval_SMPL_model=True,
        orig_label_path=orig_label_path,
        expanded_mesh_dir=expanded_mesh_dir
    )

    stylegan2_visualize(mesh_handler, save_img=True)
    # GT_visualize(mesh_handler, out_json_file='expanded_info.json', save_img=True)
    # CIPS_visualize(mesh_handler, out_json_file='expanded_info.json', n_iter=50000)
    # CIPS_visualize(mesh_handler, out_json_file='expanded_info.json', n_iter=100000, save_img=True)
    # CIPS_visualize(mesh_handler, out_json_file='expanded_info.json', n_iter=150000)
    '''
    mesh_folder = 'out/generate/100000'
    mesh_list = sorted(glob(osp.join(mesh_folder, '*.obj')))
    expanded_mesh_info = mmcv.load(osp.join(mesh_folder, 'expanded_info.json'))
    for idx, mesh_file in enumerate(mesh_list):
        mesh_handler.get_normal_map(mesh_file, osp.join(mesh_folder, f'{idx:04}.jpg'))
        expanded_mesh_file = expanded_mesh_info[str(idx)]
        assert osp.isfile(expanded_mesh_file)
        #mesh_handler.get_disp_normal_map(mesh_file, expanded_mesh_file, osp.join(mesh_folder, f'disp_{idx:04}.jpg'))
        mesh_handler.get_normal_map_local(mesh_file, expanded_mesh_file, osp.join(mesh_folder, f'local_{idx:04}.jpg'))
    '''
