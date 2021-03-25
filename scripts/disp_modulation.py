import os
import os.path as osp
import numpy as np
from glob import glob
from tqdm import tqdm
import mmcv

def disp_modulate(disp_map, max_value=1):
    """ Transfer the value of disp maps to the [img] range -1 ~ 1
    """
    EPS = 1e-3
    Gamma = 0.3
    EXPAND = 10
    disp_map = (disp_map * EXPAND).astype(np.float32)

    zero_part1 = disp_map < EPS
    zero_part2 = disp_map > -EPS
    zero_part = zero_part1 * zero_part2 

    sign = np.sign(disp_map)
    disp_map = np.abs(disp_map)
    disp_img = np.power(disp_map, Gamma).clip(0, 1)
    disp_img = disp_img * sign
    disp_img[zero_part] = 0 # range: [-1, 1]

    if max_value == 255:
        disp_img = (disp_img + 1) / 2 * 255
        disp_img = np.rint(disp_img).clip(0, 255).astype(np.uint8)

    return disp_img

def disp_demodulate(disp_img):
    """ Transfer the values of visualized disp images
        from [0, 255] to the normal disp values
    """
    EXPAND = 10
    iGamma = 10 / 3
    assert disp_img.dtype == np.uint8
    zero_part1 = disp_img == 127
    zero_part2 = disp_img == 128
    zero_part = zero_part1 & zero_part2

    disp_img = disp_img / 127.5 - 1
    sign = np.sign(disp_img)

    disp_img[zero_part] = 0
    disp_img = np.abs(disp_img).clip(0, 1)
    disp_map = np.power(disp_img, iGamma) / EXPAND
    disp_map = disp_map * sign

    return disp_map

if __name__ == '__main__':
    base_path = '/home/xuxudong/3D/data/THUman/dataset'
    folders = sorted(glob(osp.join(base_path, 'results_gyc_20181010_hsc_1_M', '*')))
    
    for idx, folder in enumerate(folders):
        disp_file = osp.join(folder, 'disp_uv_map_256.npy')
        out_folder = 'outputs/UV_maps/real'

        disp_map = np.load(disp_file)[0]
        disp_img = disp_modulate(disp_map, max_value=255)
        mmcv.imwrite(disp_img, osp.join(out_folder, f'{idx}.jpg'))
