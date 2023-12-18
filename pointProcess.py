#
# Project: multi-sensor fusion based 3D object detector
# Describe: it is used inside mmdetection (voxel rcnn)
#           to take advantage of rgbd data
#           cut-and-paste data augmentation needs to be turned off
#           before use 
# Author: anpei
# Data: 2021.02.25
# Email: anpei@hust.edu.cn
#

import numpy as np
import cv2 as cv
import open3d as o3d

class ProjectionProcessor():
    """
    It is a tool class to obtain depth and normal images from 3D points
    Image size is [H, W, x] tensor
    """
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)

    def getDepthNormal(self, depths, normals, pixels):
        depthImg  = np.zeros((self.h, self.w, 1))
        normalImg = np.zeros((self.h, self.w, 3))
        num = pixels.shape[0]
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) & (wIdx < 0):
                continue
            if (hIdx >= self.h) & (hIdx > 0):
                continue
            d    = depths[i]
            if d <= 0:
                continue
            nxyz = normals[i,:]
            depthImg[hIdx, wIdx, 0]  = d
            normalImg[hIdx, wIdx, :] = nxyz
        return depthImg, normalImg

    def getDepthNormalIntensity(self, depths, normals, intensitys, pixels, image):
        depthImg  = np.zeros((self.h, self.w, 1), dtype=np.float32)
        normalImg = np.zeros((self.h, self.w, 3), dtype=np.float32)
        intenImg  = np.zeros((self.h, self.w, 1), dtype=np.float32)
        num  = pixels.shape[0]
        rgbs = np.zeros((num, 3))  
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) | (wIdx < 0):
                continue
            if (hIdx >= self.h) | (hIdx < 0):
                continue
            d     = depths[i]
            if d <= 0:
                continue
            if d >= 60:
                continue
            inten = intensitys[i]
            nxyz  = normals[i,:]
            depthImg[hIdx, wIdx, 0]  = d
            normalImg[hIdx, wIdx, :] = nxyz
            intenImg[hIdx, wIdx, 0]  = inten
            rgbs[i, 0] = image[hIdx, wIdx, 2] # R
            rgbs[i, 1] = image[hIdx, wIdx, 1] # G
            rgbs[i, 2] = image[hIdx, wIdx, 0] # B
            # rgbs[i, 0] = 255
            # rgbs[i, 1] = 0
            # rgbs[i, 2] = 0
        return rgbs, depthImg, normalImg, intenImg

    def getMcrImage(self, mcr, pixels):
        mcrImg  = np.zeros((self.h, self.w, 1))
        num = mcr.shape[0]
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) | (wIdx < 0):
                continue
            if (hIdx >= self.h) | (hIdx < 0):
                continue
            mcrImg[hIdx, wIdx, 0]  = mcr[i]
        return mcrImg

    def getDepthLogVis(self, depthImg):
        valid_idx = (depthImg != 0)
        maxDepth = np.max(depthImg)
        invdepthImg = depthImg
        invdepthImg[~valid_idx] = 0
        invdepthImg[valid_idx] = np.log(1+invdepthImg[valid_idx])
        maxDepth = np.max(invdepthImg)

        depthImgVis = ((invdepthImg/maxDepth) * 255).astype(np.uint8)
        depthImgVis = cv.applyColorMap(depthImgVis, cv.COLORMAP_WINTER)
        depthImgVis[~valid_idx, 0] = 0
        return depthImgVis

    def getDepthVis(self, depthImg):
        maxDepth = np.max(depthImg)
        minDepth = np.min(depthImg)

        if minDepth < 0:
            depthImg += minDepth
            maxDepth += minDepth

        depthImgVis = ((depthImg/maxDepth) * 255).astype(np.uint8)
        depthImgVis = cv.applyColorMap(depthImgVis, cv.COLORMAP_JET)

        emptyIdx = (depthImg[:,:,0] == 0)
        depthImgVis[emptyIdx, 0] = 0

        return depthImgVis

    def getIntensityVis(self, intenImg):
        maxInten = np.max(intenImg)
        intenImgVis = ((intenImg/maxInten) * 255).astype(np.uint8)
        intenImgVis = cv.applyColorMap(intenImgVis, cv.COLORMAP_JET)

        emptyIdx = (intenImg[:,:,0] == 0)
        intenImgVis[emptyIdx, 0] = 0

        return intenImgVis

    def getNormalVis(self, normalImg):
        # temp = np.zeros((self.h, self.w, 3))
        temp = np.zeros_like(normalImg)
        temp[:,:,0] = normalImg[:,:,0] # B
        temp[:,:,1] = normalImg[:,:,2] # G
        temp[:,:,2] = normalImg[:,:,1] # R
        normalImgVis = (temp * 255).astype(np.uint8)
        return normalImgVis

class PointPcdProcessor():
    """
    It is a tool class to process point cloud with Open3D, such as
    object cluster, normal computation, and so on.
    """
    def __init__(self, w, h):
        self.w = int(w)
        self.h = int(h)
    
    def normalComp(self, pcd):
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        # normal adjust
        normals = np.asarray(pcd.normals)
        nxIdx = (normals[:, 0] <= 0)
        nyIdx = (normals[:, 1] <= 0)
        nzIdx = (normals[:, 2] <= 0)
        normals[nxIdx, 0] *= (-1)
        normals[nyIdx, 1] *= (-1)
        normals[nzIdx, 2] *= (-1)
        pcd.normals = o3d.utility.Vector3dVector(normals[:, 0:3])
        return pcd

    def materialFeatureExtractor(self, depths, intens, normals, rgbs):
        """
        Func: Extract material coefficient ratio features (maybe works)
        Args:
            depths  [N,1] distance of LiDAR points
            intens  [N,1] intensity of LiDAR points
            normals [N,3] normals of LiDAR points
            rgbs    [N,3] rgb value of LiDAR points
            scheme  0 or 1 choose the scheme to compute mcr
        Returns:
            mcr     [N,1] mcr features of LiDAR points
        """
        num = depths.shape[0]
        mcr = np.zeros((num, 1))

        """
        Supposed that the direction of LiDAR is (0,1,0)^T
        """
        k = 1e-2
        for i in range(num):
            d      = depths[i]
            rgb    = np.mean(rgbs[i])
            ny     = np.abs(normals[i,1])
            inten  = intens[i]
            mcr[i] = k*d*d*inten/rgb      
        return mcr
    
    def preSegmentation(self, points, pixels, segImage):
        """
        Func: Extract segmentation features from LiDAR points
        Args:
            points    [N,4] xyz+i LiDAR points
            pixels    [N,2] projected LiDAR points
            segImage  [W,H,3] semantic segmenation image from pre-traiend Deeplabv3
        Returns:
            pointSeg [N,1] segmentation results
        Note:
            please run preSegmentation.py to generate segmentation results before use
            for the background pixel, its BGR value is [84, 1, 68]
        """
        num = points.shape[0]
        pointSeg = np.zeros((num, 1))
        for i in range(num):
            wIdx = int(pixels[i,0])
            hIdx = int(pixels[i,1])
            if (wIdx >= self.w) | (wIdx < 0):
                continue
            if (hIdx >= self.h) | (hIdx < 0):
                continue
            B = segImage[hIdx, wIdx, 0]
            G = segImage[hIdx, wIdx, 1]
            R = segImage[hIdx, wIdx, 2]
            pointSeg[i,0] = 1
            if (B <= 85) & (B>= 83):
                if (G <= 2) & (G >= 0):
                    if (R <= 69) & (R >= 67):
                        pointSeg[i,0] = 0 # background
        return pointSeg