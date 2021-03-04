import numpy as np
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import torch
import smplx
from .ry_utils import load_pkl 

class SMPL_Obtainer(object):
    def __init__(self, model_file='SMPL_NEUTRAL.pkl'):
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

    def get_smpl_pkl_file(self, input_pkl_file, out_mesh_file):
        params = load_pkl(input_pkl_file)
        pose = params['pose']
        shape = params['betas']
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

def main():
    base_path = '/home/xuxudong/3D/data/Multi-Garment/Multi-Garment_dataset'
    smpl_obtainer = SMPL_Obtainer()

    pkl_files = sorted(glob(osp.join(base_path, '12*', 'registration.pkl')))
    print("# pkl files:", len(pkl_files))

    for pkl_file in tqdm(pkl_files):
        out_mesh_file = pkl_file.replace('registration.pkl', 'smpl.obj')
        smpl_obtainer.get_smpl_pkl_file(pkl_file, out_mesh_file) 

if __name__ == '__main__':
    main()
