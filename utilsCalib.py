#
# Project: LiDAR-camera target-less calibration
# Describe: attempt to calibrate the extrinsic parameters 
# Author: anpei
# Data: 2021.03.14
# Email: anpei@hust.edu.cn
#

import os
import time
import numpy as np
import cv2 as cv
import open3d as o3d
from skimage import io

def loadImageAndPoints(basePath, imageIdx):
    imagePath = basePath + "image_2/"   + str("%06d" % imageIdx) + ".png"
    pointPath = basePath + "velodyne/"  + str("%06d" % imageIdx) + ".bin"
    image     = cv.imread(str(imagePath))
    points    = np.fromfile(str(pointPath), dtype=np.float32).reshape(-1, 4)
    return image, points

def loadCameraIntrinsicParam():
    # intrinsic parameters in kitti dataset
    fx = 7.070493000000e+02
    fy = 7.070493000000e+02
    cx = 6.040814000000e+02
    cy = 1.805066000000e+02

    kk      = np.zeros((3,3), dtype=np.double)
    kk[0,0] = fx
    kk[1,1] = fy
    kk[0,2] = cx
    kk[1,2] = cy
    kk[2,2] = 1.0

    return kk

def loadExtrinsicParam():
    rr  = np.eye(3, dtype=np.double)
    tt  = np.zeros((3,1), dtype=np.double)

    thx = np.pi/180 * 90
    rx  = np.mat([[1, 0, 0],
                 [0, np.cos(thx), -np.sin(thx)],
                 [0, np.sin(thx), np.cos(thx)]])

    thy = np.pi/180 * (-90-0.5)
    ry  = np.mat([[np.cos(thy),  0, np.sin(thy)],
                 [0, 1, 0],
                 [-np.sin(thy), 0, np.cos(thy)]])

    thz = np.pi/180 * 0
    rz  = np.mat([[np.cos(thz), -np.sin(thz), 0],
                 [np.sin(thz),  np.cos(thz), 0],
                 [0, 0, 1]])

    rr  = np.matmul(ry, rx)
    rr  = np.matmul(rz, rr)

    tt  = np.mat([[0.5],
                 [0],
                 [0]])

    return rr, tt

def projection(points, kk, rr, tt):
    pts = np.transpose(points[:, :3])  # [3,N]
    pts = np.matmul(rr, pts) + tt
    tmp = np.matmul(kk, pts)           # [3,N]
    pixels = tmp/(tmp[2:3, :]+1e-5)
    depths = tmp[2:3, :]
    pixels = np.transpose(pixels)
    depths = np.transpose(depths)
    pixels = pixels[:, :2]
    print("pixels: ")
    print(pixels)
    print("depths: ")
    print(depths)
    return pixels, depths

def showPcd(pcd, show_normal=False, show_single_color=False):
    if show_single_color:
        pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd], point_show_normal=show_normal)

class ProjectionProcessor():
    """
    It is a tool class to obtain depth and normal images from 3D points
    Image size is [H, W, x] tensor
    """
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)

    def getDepth(self, depths, pixels):
        depthImg  = np.zeros((self.h, self.w, 1))
        num = pixels.shape[0]
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) | (wIdx < 0):
                continue
            if (hIdx >= self.h) | (hIdx < 0):
                continue
            d = depths[i]
            if d <= 0:
                continue
            # print("depths: ", depths.shape)
            # print("pixels: ", pixels.shape)
            # print("wIdx, hIdx: ", wIdx, hIdx)
            depthImg[hIdx, wIdx, 0]  = d
        return depthImg

    def getDepthVis(self, depthImg):
        maxDepth = np.max(depthImg)
        depthImgVis = ((depthImg/maxDepth) * 255).astype(np.uint8)
        depthImgVis = cv.applyColorMap(depthImgVis, cv.COLORMAP_JET)
        emptyIdx = (depthImg[:,:,0] == 0)
        depthImgVis[emptyIdx, 0] = 0
        return depthImgVis