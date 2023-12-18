#
# Project:  3D object detection
# Describe: it is used inside mmdetection (voxel rcnn)
#           to take advantage of rgbd data
#           cut-and-paste data augmentation needs to be turned off
#           before use 
# Author: anpei
# Data: 2021.02.25
# Email: anpei@hust.edu.cn
#

import os
import time
import numpy as np
import cv2 as cv
import open3d as o3d
from skimage import io

from utilsfusion import GetImageCv, GetSegImageCv, GetCalib, ShowPcd, showpcd
from pointProcess import PointPcdProcessor, ProjectionProcessor
from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from depth_map_utils import fill_in_fast, fill_in_multiscale
from laserscan import LaserScan

dataPath = "/media/anpei/DiskA/voxelrcnn-perception-anpei/data/kitti/"
basePath = "/media/anpei/DiskA/voxelrcnn-perception-anpei/data/kitti/training/"
savePath = "/media/anpei/DiskA/weather-transfer-anpei/depth_comp/msg_chn_wacv20-master/kitti-dataset/"

dataPath_stf = "/media/anpei/DiskA/weather-transfer-anpei/data/seeingthroughfog/"
basePath_stf = "/media/anpei/DiskA/weather-transfer-anpei/data/seeingthroughfog/training/"

width = 1242
height = 375
img_shape = [height, width]
PointPcdProcessor = PointPcdProcessor(width, height)
ProjectionProcessor = ProjectionProcessor(width, height)

def get_fov_flag(pts_rect, img_shape, calib):
    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)
    return pts_valid_flag

def basic_op(img_path, pts_path, calib_path, is_kitti=True):
    image     = cv.imread(str(img_path))
    calib     = calibration_kitti.Calibration(str(calib_path))
    
    if is_kitti == True:
        rawPoints = np.fromfile(str(pts_path), dtype=np.float32).reshape(-1, 4)
        rawPoints = rawPoints[:,:4]
        rawPts = rawPoints*1.0
    else: # stf case
        rawPoints = np.fromfile(str(pts_path), dtype=np.float32).reshape(-1, 5)
        rawPoints = rawPoints[:,:4]
        rawPoints[:, 3] /= 256.0 # debug
        rawPts = rawPoints*1.0
        # remove strange points
        z = rawPoints[:, 2]
        valid_idx = (z >= -1.8)
        rawPts = rawPts[valid_idx, :]
        x = rawPoints[:, 0]
        valid_idx &= (x >= 1.5)
        rawPoints = rawPoints[valid_idx, :]

    image    = cv.resize(image,    (1242, 375))
    pts_rect = calib.lidar_to_rect(rawPoints[:, 0:3])
    fov_flag = get_fov_flag(pts_rect, img_shape, calib)
    points   = rawPoints[fov_flag]
    return points, image, calib, rawPts

def basic_op_pts(points, image, calib):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pixels, depths = calib.lidar_to_img(points[:, 0:3])
    pcd = PointPcdProcessor.normalComp(pcd)
    normals = np.array(pcd.normals)
    intens  = points[:,3]
    rgbs, depthImg, normalImg, intenImg = ProjectionProcessor.getDepthNormalIntensity(depths, normals, intens, pixels, image)
    rgbs = rgbs.astype(np.float32)/255.0 # [0,255]=>[0,1] for open3d visulization
    return pcd, depthImg, normalImg, intenImg

def basic_depth_comp(depthImg):
    # Fast Fill with bilateral blur, no extrapolation @87Hz (recommended)
    fill_type = 'fast'
    extrapolate = False
    blur_type = 'bilateral'

    # Multi-scale dilations with extra noise removal, no extrapolation @ 30Hz
    # fill_type = 'multiscale'
    # extrapolate = False
    # blur_type = 'bilateral'

    if fill_type == 'fast':
        final_depths = fill_in_fast(
            depthImg, extrapolate=extrapolate, blur_type=blur_type)
    elif fill_type == 'multiscale':
        final_depths, process_dict = fill_in_multiscale(
            depthImg, extrapolate=extrapolate, blur_type=blur_type,
            show_process=False)
    else:
        raise ValueError('Invalid fill_type {}'.format(fill_type))
    final_depths = np.expand_dims(final_depths, axis=2)
    return final_depths

def basic_back_proj(depths, calib):
    idx = (depths != 0)
    num = int(np.sum(idx))
    us = np.zeros((num, 1))
    vs = np.zeros((num, 1))
    ds = np.zeros((num, 1))
    ct = 0

    for u in range(depths.shape[0]):
        for v in range(depths.shape[1]):
            d = depths[u,v,0]
            if d != 0:
                us[ct] = v
                vs[ct] = u
                ds[ct] = d
                ct += 1

    # print("depths: ", depths.shape)
    pts_rect  = calib.img_to_rect(us, vs, ds)
    pts_lidar = calib.rect_to_lidar(pts_rect) 
    return pts_lidar
        
def subprocess_stf2kit(
    imagePath, pointPath, calibPath, current_name, current_name_stf,
    imagePath_stf, pointPath_stf, calibPath_stf):

    # == pre-process == #
    points, image, calib, rawPoints = basic_op(imagePath, pointPath, calibPath, is_kitti=True)
    points_stf, image_stf, calib_stf, rawPoints_stf = basic_op(imagePath_stf, pointPath_stf, calibPath_stf, is_kitti=False)
    
    is_kitti = True
    # is_kitti = False
    if is_kitti == False:
        # to remove the ring in the rawPoints
        raw_x = rawPoints_stf[:,0]
        raw_y = rawPoints_stf[:,1]
        invalid_idx = (raw_x <= 2.6) & (raw_x >= -2.6) \
            & (raw_y <= 2.6) & (raw_y >= -2.6)
        valid_idx = ~invalid_idx
        rawPoints_stf_pro = rawPoints_stf[valid_idx]
    else:
        rawPoints_stf = rawPoints
        raw_z = rawPoints_stf[:,2]
        invalid_idx = (raw_z <= -2.2)
        valid_idx = ~invalid_idx
        rawPoints_stf_pro = rawPoints[valid_idx]
        rawPoints_stf = rawPoints[valid_idx]

    # (optional) == depth completin in range view == #
    laser_scan_cp = LaserScan(H=64*2, W=2048, fov_up=3.0, fov_down=-25.0)
    laser_scan_cp.reset()
    laser_scan_cp.set_points(rawPoints_stf_pro[:, 0:3], rawPoints_stf_pro[:, 3])
    laser_scan_cp.do_range_projection(rgbs=rawPoints_stf_pro[:, 0:3], is_rgb=False)
    range_view = np.copy(laser_scan_cp.proj_range)
    range_inte = np.copy(laser_scan_cp.proj_remission)
    invlid = (range_view == -1) 
    range_view[invlid] = 0
    range_view = np.expand_dims(range_view,-1)
    range_inte[invlid] = 0
    range_inte = np.expand_dims(range_inte,-1)

    final_range_view = basic_depth_comp(range_view*1.0)
    final_range_inte = basic_depth_comp(range_inte*1.0)
    
    final_pts = laser_scan_cp.do_range_back_proj(
        final_range_view, final_range_inte)
    final_pts[:,3] = 0
    final_pts = np.concatenate((rawPoints_stf, final_pts))
    final_rgb = np.ones_like(final_pts)[:,:3]
    num_raw = rawPoints_stf.shape[0]
    final_rgb[num_raw:-1,:2] = 0

    raw_pcd   = o3d.geometry.PointCloud()
    debug_pcd = o3d.geometry.PointCloud()
    raw_pcd.points   = o3d.utility.Vector3dVector(rawPoints_stf[:, 0:3])
    debug_pcd.points = o3d.utility.Vector3dVector(final_pts[:, 0:3])
    # debug_pcd.colors = o3d.utility.Vector3dVector(final_rgb[:, 0:3])

    is_save = False
    # is_save = True
    if is_save:
        save_path = basePath_stf + "velodyne_kit_style/" \
            + current_name_stf + ".bin"
        print("save_path:", save_path)
        nums_pts  = final_pts.shape[0]
        pts_saved = np.zeros((nums_pts, 5))
        pts_saved[:,:4] = final_pts[:, :4]

        valid_idx = (pts_saved[:,2] >= -1.8)
        pts_saved = pts_saved[valid_idx, :]

        # debug_pcd.points = o3d.utility.Vector3dVector(pts_saved[:, 0:3])
        pts_saved.tofile(save_path)

    # debug
    DebugMode = False
    DebugMode = True
    if DebugMode == True:
        print("image path: ", imagePath)
        depthImgVisR = ProjectionProcessor.getDepthVis(range_view)
        depthImgVis  = ProjectionProcessor.getDepthVis(final_range_view)

        showpcd(raw_pcd)
        showpcd(debug_pcd)

        cv.imshow("depthImgVis", depthImgVis)
        cv.imshow("depthImgVisR", depthImgVisR)
        cv.waitKey(0)
        cv.destroyAllWindows()

def offlineProcess():
    # => kitti files
    imageIdx = 3
    current_name = str("%06d" % imageIdx)
    imagePath = basePath + "image_2/"     + str("%06d" % imageIdx) + ".png"
    pointPath = basePath + "velodyne/"    + str("%06d" % imageIdx) + ".bin"
    calibPath = basePath + "calib/"       + str("%06d" % imageIdx) + ".txt"
    
    # => stf files
    file_name_stf = []
    all_txt_stf   = dataPath_stf + "ImageSets/all.txt"
    all_txt_stf   = dataPath_stf + "ImageSets/train.txt"
    all_txt_stf   = dataPath_stf + "ImageSets/val.txt"
    # all_txt_stf   = dataPath_stf + "ImageSets/light_fog_day.txt"
    # all_txt_stf   = dataPath_stf + "ImageSets/dense_fog_day.txt"
    file      = open(all_txt_stf, 'r')
    file_data = file.readlines()
    for row in file_data:
        tmp = row[0:-1]
        str_a = tmp[0:-6]
        str_b = tmp[-5:]
        tmp = str_a + "_" + str_b
        file_name_stf.append(tmp)

    imageIdx_stf = 5
    imageIdx_stf = 8
    imageIdx_stf = 99
    imageIdx_stf = 101 + 45
    current_name_stf = file_name_stf[imageIdx_stf]
    imagePath_stf = basePath_stf + "image_2/"  + current_name_stf + ".png"
    pointPath_stf = basePath_stf + "velodyne/" + current_name_stf + ".bin"
    calibPath_stf = basePath_stf + "calib/"    + "kitti_stereo_velodynehdl_calib.txt"
    calibPath_stf = basePath_stf + "calib/"    + "000010.txt"

    # debug
    timeBegin = time.time()
    pointsFea = subprocess_stf2kit(
        imagePath, pointPath, calibPath, current_name, current_name_stf,
        imagePath_stf, pointPath_stf, calibPath_stf)
    timeEnd = time.time()
    print("pointsFea generateion times: ", timeEnd-timeBegin)

    # process in batch
    # is_need_batch = False
    # is_need_batch = True
    # from tqdm import tqdm
    # if is_need_batch:
    #     for imageIdx_stf in tqdm(range(len(file_name_stf))):
    #         current_name_stf = file_name_stf[imageIdx_stf]
    #         imagePath_stf = basePath_stf + "image_2/"  + current_name_stf + ".png"
    #         pointPath_stf = basePath_stf + "velodyne/" + current_name_stf + ".bin"
    #         calibPath_stf = basePath_stf + "calib/"    + "kitti_stereo_velodynehdl_calib.txt"
    #         calibPath_stf = basePath_stf + "calib/"    + "000010.txt"
    #         timeBegin = time.time()
    #         pointsFea = subprocess_stf2kit(
    #             imagePath, pointPath, calibPath, current_name, current_name_stf,
    #             imagePath_stf, pointPath_stf, calibPath_stf)
    #         timeEnd = time.time()
    #         print("pointsFea generateion times: ", timeEnd-timeBegin)

if __name__ == '__main__':
    isNeedRefresh = False
    isNeedRefresh = True
    if isNeedRefresh:
        offlineProcess()
        # offlineProcessBatch()
    else:
        print("check whether it is done before")