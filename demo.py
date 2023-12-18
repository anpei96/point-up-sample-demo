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
    normals = basic_op_normal_pts(rawPoints_stf)

    # == lidar point cloud preparation (start) == #    
    datacvtgt = datapreparation(beam_num = 64)
    gt_pts, gt_range_view, gt_range_inte, gt_range_norm = datacvtgt.convert(
        rawPoints_stf, normals)

    raw_range_view = np.zeros_like(gt_range_view)
    raw_range_inte = np.zeros_like(gt_range_inte)
    raw_range_norm = np.zeros_like(gt_range_norm)
    raw_beam_num = 32
    # raw_beam_num = 16
    # raw_beam_num =  8
    range_stride = 64 // raw_beam_num
    for i in range(64):
        if i % range_stride == 0:
            raw_range_view[i,:,:] = gt_range_view[i,:,:]
            raw_range_inte[i,:,:] = gt_range_inte[i,:,:]
            raw_range_norm[i,:,:] = gt_range_norm[i,:,:]
    # == lidar point cloud preparation (end) == #

    # == lidar point cloud upsample (baseline-start) == #
    upsample_solve_1 = baseline(up_beam_num = 64)
    final_pts, final_range_view = \
        upsample_solve_1.upsample(raw_range_view, raw_range_inte)

    final_range_view_inte = \
        upsample_solve_1.upsample_interploate(raw_range_view, range_stride)

    # raw_pcd   = o3d.geometry.PointCloud()
    # debug_pcd = o3d.geometry.PointCloud()
    # raw_pcd.points   = o3d.utility.Vector3dVector(rawPoints_stf[:, 0:3])
    # debug_pcd.points = o3d.utility.Vector3dVector(final_pts[:, 0:3])
    # == lidar point cloud upsample (baseline-end) == #

    # == lidar point cloud upsample (propose-start) == #
    upsample_solve_2 = propose(up_beam_num = 64)

    res_range_dept_cp, res_range_dept_knn, res_range_dept_svd = \
        upsample_solve_2.upsample_matrix_cp(raw_range_view)

    # final_pts_our, final_range_view_our = \
    res_range_norm, res_range_dept = \
        upsample_solve_2.upsample(
            raw_range_view, raw_range_inte, raw_range_norm)
    
    # == lidar point cloud upsample (baseline-end) == #

    # == upsample evaluation (start) == #
    rmse, mae, irmse, imae = datacvtgt.evaluate(final_range_view, gt_range_view)
    print("==evaluate==")
    print("baseline rmse: ", rmse)
    print("baseline mae: ", mae)
    print("baseline irmse: ", irmse)
    print("baseline imae: ", imae)

    rmse, mae, irmse, imae = datacvtgt.evaluate(final_range_view_inte, gt_range_view)
    print("==evaluate==")
    print("mat interpolate rmse: ", rmse)
    print("mat interpolate mae: ", mae)
    print("mat interpolate irmse: ", irmse)
    print("mat interpolate imae: ", imae)

    rmse, mae, irmse, imae = datacvtgt.evaluate(res_range_dept_svd, gt_range_view)
    print("==evaluate==")
    print("iterative svd-cp rmse: ", rmse)
    print("iterative svd-cp mae: ", mae)
    print("iterative svd-cp irmse: ", irmse)
    print("iterative svd-cp imae: ", imae)

    rmse, mae, irmse, imae = datacvtgt.evaluate(res_range_dept_knn, gt_range_view)
    print("==evaluate==")
    print("knn rmse: ", rmse)
    print("knn mae: ", mae)
    print("knn irmse: ", irmse)
    print("knn imae: ", imae)

    rmse, mae, irmse, imae = datacvtgt.evaluate(res_range_dept_cp, gt_range_view)
    print("==evaluate==")
    print("matrix completion rmse: ", rmse)
    print("matrix completion mae: ", mae)
    print("matrix completion irmse: ", irmse)
    print("matrix completion imae: ", imae)

    rmse, mae, irmse, imae = datacvtgt.evaluate(res_range_dept, gt_range_view)
    print("==evaluate==")
    print("our method rmse: ", rmse)
    print("our method mae: ", mae)
    print("our method irmse: ", irmse)
    print("our method imae: ", imae)
    # == upsample evaluation (end) == #

    # debug
    DebugMode = False
    # DebugMode = True
    if DebugMode == True:
        print("image path: ", imagePath)
        depthImg_input  = ProjectionProcessor.getDepthVis(raw_range_view)
        normalImg_input = ProjectionProcessor.getNormalVis(raw_range_norm)
        depthImg_output = ProjectionProcessor.getDepthVis(res_range_dept)
        depthImg_gt  = ProjectionProcessor.getDepthVis(gt_range_view)
        normalImg_gt = ProjectionProcessor.getNormalVis(gt_range_norm)
        normalImg_pd = ProjectionProcessor.getNormalVis(res_range_norm)

        # normalImg_pd = cv.medianBlur(normalImg_pd, 3)

        # showpcd(raw_pcd)
        # showpcd(debug_pcd)

        cv.imshow("depthImg_input", depthImg_input)
        # cv.imshow("normalImg_input", normalImg_input)
        # cv.imshow("normalImg_pd", normalImg_pd)
        cv.imshow("depthImg_output", depthImg_output)
        cv.imshow("depthImg_gt", depthImg_gt)
        # cv.imshow("normalImg_gt", normalImg_gt)
        cv.waitKey(0)
        cv.destroyAllWindows()

def offlineProcess():
    # => kitti files
    imageIdx = 3
    imageIdx = 10
    # imageIdx = 99
    # imageIdx = 666
    imageIdx = 66+20
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
