import os
import os.path as osp
import numpy as np
import pickle
from glob import glob

def _param_txt_parsing(txt_fname):
    with open(txt_fname, 'r') as cur_file:
        param_content = cur_file.readlines()

    shape_line = param_content[1]
    pose_lines = param_content[8:]
    shape = np.array([float(item) for item in shape_line[:-1].split()])
    pose = np.array([float(line[:-1]) for line in pose_lines])

    return np.concatenate((shape, pose))

def _param_pkl_parsing(pkl_fname):
    assert pkl_fname.endswith(".pkl")
    with open(pkl_fname, 'rb') as cur_file:
        try:
            data = pickle.load(cur_file)
        except UnicodeDecodeError:
            cur_file.seek(0)
            data = pickle.load(cur_file, encoding='latin1')

    assert isinstance(data, dict)
    
    return np.concatenate((data['betas'], data['pose']))

if __name__ == '__main__':
    base_path = '/home/xuxudong/3D/data/Multi-Garment'
    param_files = sorted(glob(osp.join(base_path, 'Multi-Garment_dataset', '12*', 'registration.pkl')))
    param_list = []
    for param_file in param_files:
        param_list.append(_param_pkl_parsing(param_file))

    params = np.stack(param_list)
    np.save(osp.join(base_path, 'params.npy'), params)
