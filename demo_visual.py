#
# Project:  LiDAR point cloud upsampling
# Describe: upsample 16/32/64-beam lidar to 128/256-beam lidar
# Author: anpei
# Data: 2022.12.08
# Email: anpei@wit.edu.cn
#

from email.mime import image
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

from baseline import datapreparation, baseline
from propose  import propose

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

def basic_op_normal_pts(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, 0:3])
    pcd = PointPcdProcessor.normalComp(pcd)
    normals = np.array(pcd.normals)
    return normals

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
        
def subprocess_matrixcomp(
    imagePath, pointPath, calibPath, current_name, current_name_stf,
    imagePath_stf, pointPath_stf, calibPath_stf):

    # == pre-process == #
    points_kit, image_kit, calib_kit, rawPoints_kit = basic_op(imagePath, pointPath, calibPath, is_kitti=True)
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
        rawPoints_ = rawPoints_stf[valid_idx]
    else:
        raw_z = rawPoints_kit[:,2]
        raw_x = rawPoints_kit[:,0]
        raw_y = rawPoints_kit[:,1]
        # invalid_idx = (raw_z <= -2.2)
        invalid_idx = (raw_z <= -2.2) | (raw_z >= 2.0) \
            | ((raw_x <= 2.6) & (raw_x >= -2.6) \
            & (raw_y <= 2.6) & (raw_y >= -2.6))
        valid_idx = ~invalid_idx
        rawPoints_ = rawPoints_kit[valid_idx]
    normals = basic_op_normal_pts(rawPoints_)

    # == lidar point cloud preparation (start) == #    
    datacvtgt = datapreparation(beam_num = 64)
    gt_pts, gt_range_view, gt_range_inte, gt_range_norm = datacvtgt.convert(
        rawPoints_, normals)

    raw_range_view = np.zeros_like(gt_range_view)
    raw_range_inte = np.zeros_like(gt_range_inte)
    raw_range_norm = np.zeros_like(gt_range_norm)
    # raw_beam_num = 64
    raw_beam_num = 32
    # raw_beam_num = 16
    range_stride = 64 // raw_beam_num
    for i in range(64):
        if i % range_stride == 0:
            raw_range_view[i,:,:] = gt_range_view[i,:,:]
            raw_range_inte[i,:,:] = gt_range_inte[i,:,:]
            raw_range_norm[i,:,:] = gt_range_norm[i,:,:]
    
    laser_scan = LaserScan(H=64, W=2048, fov_up=3.0, fov_down=-25.0)
    sub_pts = laser_scan.do_range_back_proj(raw_range_view,raw_range_view)
    # == lidar point cloud preparation (end) == #

    # == lidar point cloud upsample (propose-start) == #
    upsample_solve_2 = propose(up_beam_num = 64)
    res_range_norm, res_range_dept = \
        upsample_solve_2.upsample(
            raw_range_view, raw_range_inte, raw_range_norm)
    final_pts = upsample_solve_2.final_pts
    
    # upsample_solve_1 = baseline(up_beam_num = 64)
    # final_pts, final_range_view = \
    #     upsample_solve_1.upsample(raw_range_view, raw_range_inte)

    raw_x = final_pts[:,0]
    raw_y = final_pts[:,1]
    raw_z = final_pts[:,2]
    invalid_idx = (raw_z <= -2.2) | (raw_z >= 1.6)
    valid_idx = ~invalid_idx
    final_pts = final_pts[valid_idx]

    raw_pcd   = o3d.geometry.PointCloud()
    debug_pcd = o3d.geometry.PointCloud()
    raw_pcd.points   = o3d.utility.Vector3dVector(sub_pts[:, 0:3])
    # raw_pcd.colors   = o3d.utility.Vector3dVector(np.ones_like(sub_pts[:, 0:3]))
    debug_pcd.points = o3d.utility.Vector3dVector(final_pts[:, 0:3])
    # debug_pcd.colors   = o3d.utility.Vector3dVector(np.ones_like(final_pts[:, 0:3]))
    # == lidar point cloud upsample (baseline-end) == #

    # debug
    DebugMode = False
    DebugMode = True
    if DebugMode == True:
        print("image path: ", imagePath)
        depthImg_input  = ProjectionProcessor.getDepthLogVis(raw_range_view[:,:,0])
        normalImg_input = ProjectionProcessor.getNormalVis(raw_range_norm)
        depthImg_output = ProjectionProcessor.getDepthLogVis(res_range_dept[:,:,0])
        depthImg_gt  = ProjectionProcessor.getDepthLogVis(gt_range_view[:,:,0])
        normalImg_gt = ProjectionProcessor.getNormalVis(gt_range_norm)
        normalImg_pd = ProjectionProcessor.getNormalVis(res_range_norm)

        # normalImg_pd = cv.medianBlur(normalImg_pd, 3)

        showpcd(raw_pcd)
        showpcd(debug_pcd)

        # cv.imwrite("./vis/depthImg_input.png", depthImg_input)
        # cv.imwrite("./vis/depthImg_output.png", depthImg_output)
        # cv.imwrite("./vis/depthImg_gt.png", depthImg_gt)
        # cv.imwrite("./vis/normalImg_input.png", normalImg_input)
        # cv.imwrite("./vis/normalImg_pd.png", normalImg_pd)
        # cv.imwrite("./vis/normalImg_gt.png", normalImg_gt)

        # cv.imshow("depthImg_input", depthImg_input)
        # cv.imshow("normalImg_input", normalImg_input)
        # cv.imshow("normalImg_pd", normalImg_pd)
        # cv.imshow("depthImg_output", depthImg_output)
        # # cv.imshow("depthImg_gt", depthImg_gt)
        # # # cv.imshow("normalImg_gt", normalImg_gt)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

def offlineProcess():
    # => kitti files
    imageIdx = 3
    imageIdx = 10
    # imageIdx = 99
    # imageIdx = 666
    imageIdx = 66+20
    imageIdx = 100+20
    imageIdx = 100+11
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
    # imageIdx_stf = 101 + 45
    current_name_stf = file_name_stf[imageIdx_stf]
    imagePath_stf = basePath_stf + "image_2/"  + current_name_stf + ".png"
    pointPath_stf = basePath_stf + "velodyne/" + current_name_stf + ".bin"
    calibPath_stf = basePath_stf + "calib/"    + "kitti_stereo_velodynehdl_calib.txt"
    calibPath_stf = basePath_stf + "calib/"    + "000010.txt"

    # debug
    pointsFea = subprocess_matrixcomp(
        imagePath, pointPath, calibPath, current_name, current_name_stf,
        imagePath_stf, pointPath_stf, calibPath_stf)

if __name__ == '__main__':
    offlineProcess()
