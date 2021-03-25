import numpy as np
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import torch
import smplx

class SMPL_Obtainer(object):
    def __init__(self, gender='neutral'):
        if gender == 'neutral':
            model_file = 'SMPL_NEUTRAL.pkl'
        elif gender == 'male':
            model_file = 'SMPL_MALE.pkl'
        elif gender == 'female':
            model_file = 'SMPL_FEMALE.pkl'
        else:
            raise ValueError('Please input correct gender')

        abs_model_file = osp.join(osp.dirname(osp.abspath(__file__)), model_file)
        self.smpl_model = smplx.create(abs_model_file, 'smpl', pose2rot=False)

    @staticmethod
    def save_mesh_to_obj(obj_path, verts, faces=None):
        assert isinstance(verts, np.ndarray)
        assert isinstance(faces, np.ndarray)

        with open(obj_path, 'w') as out_f:
            # write verts
            for v in verts:
                out_f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
            # write faces 
            if faces is not None:
                faces = faces.copy() + 1
                for f in faces:
                    out_f.write(f"f {f[0]} {f[1]} {f[2]}\n")

    @staticmethod
    def _param_txt_parsing(txt_fname):
        with open(txt_fname, 'r') as cur_file:
            param_content = cur_file.readlines()

        shape_line = param_content[1]
        pose_lines = param_content[8:]
        shape = np.array([float(item) for item in shape_line[:-1].split()])
        pose = np.array([float(line[:-1]) for line in pose_lines])

        return shape, pose

    @staticmethod
    def _param_pkl_parsing(pkl_fname):
        assert pkl_fname.endswith(".pkl")
        with open(pkl_fname, 'rb') as cur_file:
            try:
                data = pickle.load(cur_file)
            except UnicodeDecodeError:
                cur_file.seek(0)
                data = pickle.load(cur_file, encoding='latin1')

        assert isinstance(data, dict)
        
        return data['betas'], data['pose']

    def get_smpl_pose_shape(self, pose, shape, out_mesh_file):
        assert type(pose) is np.ndarray
        assert type(shape) is np.ndarray
        pose = torch.from_numpy(pose).view(1, 72).float()
        shape = torch.from_numpy(shape).view(1, 10).float()
        output = self.smpl_model(
            global_orient = pose[:, :3],
            body_pose = pose[:, 3:],
            betas = shape
        )
        verts = output.vertices.detach().cpu().numpy()[0]
        faces = self.smpl_model.faces

        self.save_mesh_to_obj(out_mesh_file, verts, faces)

def multi_garment(gender='neutral'):
    base_path = '/home/xuxudong/3D/data/Multi-Garment/Multi-Garment_dataset'
    smpl_obtainer = SMPL_Obtainer(gender=gender)

    pkl_files = sorted(glob(osp.join(base_path, '12*', 'registration.pkl')))
    print("# pkl files:", len(pkl_files))

    for pkl_file in tqdm(pkl_files):
        out_mesh_file = pkl_file.replace('registration.pkl', 'smpl.obj')
        shape, pose = smpl_obtainer._param_pkl_parsing(pkl_file)
        smpl_obtainer.get_smpl_pose_shape(pose=pose, shape=shape, out_mesh_file=out_mesh_file)

def thuman_npy_params(gender='neutral'):
    base_path = '/home/xuxudong/3D/data/THUman/dataset'
    smpl_obtainer = SMPL_Obtainer(gender=gender)

    params = np.load(osp.join(osp.dirname(base_path), 'THUman_params.npy'))
    folders = sorted(glob(osp.join(base_path, 'results*/*')))
    assert len(params) == len(folders)
    for idx, folder in tqdm(enumerate(folders)):
        out_file = osp.join(folder, 'smpl_neutral.obj')
        param = params[idx]
        smpl_obtainer.get_smpl_pose_shape(pose=param[10:], shape=param[:10], out_mesh_file=out_file)

def thuman(gender='neutral'):
    base_path = '/home/xuxudong/3D/data/THUman/dataset'
    smpl_obtainer = SMPL_Obtainer(gender=gender)

    txt_files = sorted(glob(osp.join(base_path, 'results*/*/smpl_params.txt')))
    print("# txt files:", len(txt_files))

    for txt_file in tqdm(txt_files):
        out_mesh_file = txt_file.replace('smpl_params.txt', f'smpl_{gender}.obj')
        shape, pose = smpl_obtainer._param_txt_parsing(txt_file)
        smpl_obtainer.get_smpl_pose_shape(pose=pose, shape=shape, out_mesh_file=out_mesh_file)

if __name__ == '__main__':
    gender = 'neutral'
    #multi_garment()
    thuman(gender)
