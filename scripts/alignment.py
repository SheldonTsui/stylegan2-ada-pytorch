import os
import os.path as osp
from glob import glob
from tqdm import tqdm
import numpy as np

import open3d as o3d
from objfile import read_obj, read_obj_full, write_obj

voxel_size = 0.05  # means 5cm

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(pcd_down,
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                        target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), 4, 
            [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.registration.RANSACConvergenceCriteria(4000000, 500)
        )

    return result

def alignment(obj_file, scan_file, out_file, out_corres_file):
    expand_mesh = o3d.io.read_triangle_mesh(obj_file)
    scan_mesh = o3d.io.read_triangle_mesh(scan_file)
    
    # transfer mesh vertices to point cloud to do the registration
    expand_pcd = o3d.geometry.PointCloud()
    expand_pcd.points = expand_mesh.vertices 
    scan_pcd = o3d.geometry.PointCloud()
    scan_pcd.points = scan_mesh.vertices

    source = expand_pcd
    target = scan_pcd
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    # global registration
    global_result = execute_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh, voxel_size
        )
    
    distance_threshold = 0.05
    # icp registration
    icp_result = o3d.registration.registration_icp(
            source, target, distance_threshold, global_result.transformation,
            o3d.registration.TransformationEstimationPointToPoint()
        )
    #print(icp_result.transformation)
    # there are several properties in icp_result, like correspondence, transformation
    expand_mesh = expand_mesh.transform(icp_result.transformation)
    o3d.io.write_triangle_mesh(out_file, expand_mesh)
    #corres_set = np.asarray(icp_result.correspondence_set)
    #print(corres_set.shape)
    #np.save(out_corres_file, corres_set) 

if __name__ == '__main__':
    base_path = "/home/xuxudong/3D/data/Multi-Garment/Multi-Garment_dataset"
    folders = sorted(glob(osp.join(base_path, '12*')))
    print("# folders:", len(folders))

    for folder in folders:
        obj_file = osp.join(folder, 'smpl_expanded.obj')
        scan_file = osp.join(folder, 'smpl_registered.obj')
        out_file = osp.join(folder, 'smpl_expanded_align.ply')
        out_corres_file = osp.join(folder, 'smpl_corres.npy')
        alignment(obj_file, scan_file, out_file, out_corres_file)
