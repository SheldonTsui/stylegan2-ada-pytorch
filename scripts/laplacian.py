import os
import os.path as osp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, lsmr
import trimesh
import open3d as o3d
import time

from objfile import write_obj

def get_disp(obj_file, scan_file, out_file):
    w = 5 

    # load aligned mesh
    mesh = trimesh.load_mesh(obj_file, process=False)
    vertices = np.asarray(mesh.vertices)

    # build graph Laplacian matrix
    degree = mesh.vertex_degree 
    N = len(degree)
    edges = np.asarray(mesh.edges_unique)
    Lap_edges_half = sparse.csr_matrix((np.ones(len(edges)), (edges[:,0], edges[:,1])), shape=(N, N))
    Laplacian = sparse.diags(degree) - Lap_edges_half - Lap_edges_half.T 

    # load scan mesh
    scan_mesh = o3d.io.read_triangle_mesh(scan_file)
    #scan_mesh = trimesh.load_mesh(scan_file, process=False)
    #print(scan_mesh)
    max_points = 40000 
    if len(scan_mesh.vertices) > max_points:
        scan_pcd = scan_mesh.sample_points_uniformly(number_of_points=max_points)
        scan_vertices = np.asarray(scan_pcd.points)
    else:
        scan_vertices = np.asarray(scan_mesh.vertices)
    Dis_query = trimesh.proximity.ProximityQuery(mesh)
    _, closest_index = Dis_query.vertex(scan_vertices)
    L = len(closest_index)
    index_matrix = sparse.csr_matrix((np.ones(L) * w, (list(range(L)), closest_index)), shape=(L, N)) # get matrix I

    delta = np.matmul(Laplacian.toarray(), vertices) # compute initial delta
    A = sparse.vstack([Laplacian, index_matrix])
    b = np.concatenate((delta, scan_vertices * w), axis=0)
    #print("A shape:", A.shape)
    #print("b shape", b.shape)
    # solve least-squares solution
    # lsqr only support vector format in the right hand side
    b1, b2, b3 = b.T
    eps = 1e-6 
    x1 = lsmr(A, b1, atol=eps, btol=eps, x0=vertices[:,0]) 
    x2 = lsmr(A, b2, atol=eps, btol=eps, x0=vertices[:,1]) 
    x3 = lsmr(A, b3, atol=eps, btol=eps, x0=vertices[:,2]) 
    x = np.stack([x1[0],x2[0],x3[0]], axis=1)
    #print(x.shape)

    mesh.vertices = x
    mesh.export(out_file)

if __name__ == '__main__':
    #folder = '/home/xuxudong/3D/data/samples/Multi-Garment_dataset/125611497894336' 
    folder = '/home/xuxudong/3D/data/samples/THUman/13010'
    obj_file = osp.join(folder, 'smpl_expanded.obj')
    scan_file = osp.join(folder, 'mesh.obj')
    out_file = osp.join(folder, 'smpl_disp.ply')
    start = time.time()
    get_disp(obj_file, scan_file, out_file)
    print(time.time() - start)
