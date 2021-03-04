import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import numpy as np
import mmcv

import trimesh
from objfile import read_obj, read_obj_full, write_obj

def expand_smpl_model(obj_file, out_obj_file, out_mapping_file=None):
    cur_mesh = trimesh.load_mesh(obj_file, process=False)
    orig_v = np.asarray(cur_mesh.vertices)
    orig_f = np.asarray(cur_mesh.faces)
    edges = cur_mesh.edges_unique
    new_v_mapping = dict()

    new_vertices = [] 
    # add new vertices for original smpl mesh
    for idx, edge in enumerate(edges):
        v1, v2 = edge
        new_vertex = orig_v[v1] / 2 + orig_v[v2] / 2
        new_v_mapping[(v1, v2)] = idx + len(orig_v) # record edge-vertex mapping
        new_v_mapping[(v2, v1)] = idx + len(orig_v)
        
        new_vertices.append(new_vertex)
        
    new_vertices = np.stack(new_vertices)
    new_vertices = np.concatenate((orig_v, new_vertices), axis=0)
    #print("# new_vertices", len(new_vertices)) 

    new_faces = []
    for face in orig_f:
        v1, v2, v3 = face
        # MUST mind the vertex order, very important!
        cur_new_face1 = [v1, new_v_mapping[(v1, v2)], new_v_mapping[(v1, v3)]] 
        cur_new_face2 = [new_v_mapping[(v1, v2)], v2, new_v_mapping[(v2, v3)]] 
        cur_new_face3 = [new_v_mapping[(v1, v3)], new_v_mapping[(v2, v3)], v3] 
        cur_new_face4 = [new_v_mapping[(v1, v2)], new_v_mapping[(v2, v3)], new_v_mapping[(v1, v3)]] 
        
        new_faces.extend([cur_new_face1, cur_new_face2, cur_new_face3, cur_new_face4])

    new_faces = (np.stack(new_faces) + 1).astype(np.int32)
    #print("# new_faces", len(new_faces))

    write_obj(out_obj_file, new_vertices, new_faces)
    if out_mapping_file is not None:
        mmcv.dump(new_v_mapping, out_mapping_file)

if __name__ == '__main__':
    base_path = "/home/xuxudong/3D/data/Multi-Garment/Multi-Garment_dataset"
    #folders = sorted(glob(osp.join(base_path, 'results*/*')))
    folders = sorted(glob(osp.join(base_path, '12*')))
    print("# folders:", len(folders))

    # expand smpl model
    # vertices: 6890 -> 27554
    # faces: 13776 -> 55104
    for folder in folders:
        obj_file = osp.join(folder, 'smpl.obj')
        out_obj_file = osp.join(folder, 'smpl_expanded.obj')
        out_mapping_file = osp.join(folder, 'new_v_mapping.pkl')
        expand_smpl_model(obj_file, out_obj_file, out_mapping_file)
