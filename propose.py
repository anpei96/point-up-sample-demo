#
# Project:  LiDAR point cloud upsampling
# Describe: upsample 16/32/64-beam lidar to 128/256-beam lidar
# Author: anpei
# Data: 2022.12.08
# Email: anpei@wit.edu.cn
#

import os
import time
import numpy as np
import cv2 as cv
import open3d as o3d
from skimage import io
import tqdm

import prox_tv as ptv
# from matrix_completion import *
from utilsMatrix import svt_solve, pmf_solve, biased_mf_solve

from fancyimpute import (
    BiScaler,
    KNN,
    NuclearNormMinimization,
    IterativeSVD,
    SoftImpute,
    SimpleFill
)

from guided_filter.cv.image import to32F
from guided_filter.core.filters import GuidedFilterGray, FastGuidedFilter, GuidedFilter, GuidedFilterColor

from depth_map_utils import fill_in_fast, fill_in_multiscale
from laserscan import LaserScan

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

def is_in_border(i,j,H,W):
    if ((i >= 0) & (i < H) & (j >= 0) & (j < W)):
        return True
    else:
        return False
    

def is_empty_normal(n):
    if ((n[0] == 0) & (n[1] == 0) & (n[2] == 0)):
        return True
    else:
        return False 

def getDepthLogVis(depthImg):
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

def getNormalVis(normalImg):
    # temp = np.zeros((self.h, self.w, 3))
    temp = np.zeros_like(normalImg)
    temp[:,:,0] = normalImg[:,:,0] # B
    temp[:,:,1] = normalImg[:,:,2] # G
    temp[:,:,2] = normalImg[:,:,1] # R
    normalImgVis = (temp * 255).astype(np.uint8)
    return normalImgVis

def getMapVis(depthImg):
    maxDepth = np.max(depthImg)
    minDepth = np.min(depthImg)
    # print("max-value: ", maxDepth)
    # print("min-value: ", minDepth)
    depthImgVis = ((depthImg/maxDepth) * 255).astype(np.uint8)
    depthImgVis = cv.applyColorMap(depthImgVis, cv.COLORMAP_JET)

    # emptyIdx = (depthImg[:,:,0] == 0)
    # depthImgVis[emptyIdx, 0] = 0
    return depthImgVis

class propose():
    '''
    a baseline upsample method that uses the basic image processing
    upsample 8/16/32-beam lidar point to 64/128-beam lidar point
    '''
    def __init__(self, up_beam_num=64):
        self.up_beam_num = up_beam_num
        self.laser_scan = LaserScan(H=up_beam_num, W=2048, fov_up=3.0, fov_down=-25.0)

    def weight_estimation(self, i, j, di, dj, raw_range_dept, raw_range_inte):
        distance_diff = np.abs(di) + np.abs(dj)
        # depth_diff = np.abs(raw_range_dept[i,j,:]-raw_range_dept[i+di,j+dj,:])
        # inten_diff = np.abs(raw_range_inte[i,j,:]-raw_range_inte[i+di,j+dj,:])
        
        a1 = 0.2
        a2 = 1.0
        a3 = 0.6
        # weight = np.exp(-a1*distance_diff-a2*depth_diff-a3*inten_diff)
        weight = np.exp(-a1*distance_diff)
        return weight

    def normal_comp(self, raw_range_norm, raw_range_dept, raw_range_inte):
        '''
            sub-step-1: rough normal estimation
            sub-step-2: normal smooth with total general variation (tgv)
        '''
        invalid_idx = ((raw_range_norm[:,:,0] == 0) & \
            (raw_range_norm[:,:,1] == 0) & (raw_range_norm[:,:,2] == 0))
        valid_idx = ~invalid_idx

        half_window_h = 3
        half_window_w = 5
        half_window_h = 5
        half_window_w = 5
        H = raw_range_norm.shape[0]
        W = raw_range_norm.shape[1]
        res_range_norm = raw_range_norm*(1.0)

        print("==> raw normal completion")
        for i in tqdm.trange(H):
            # i = H-i-1
            for j in range(W):
                n = raw_range_norm[i,j,:]
                # find the empty normal
                if is_empty_normal(n) == True:
                    # obtain its window
                    effect_num = 0
                    for di in range(-half_window_h-2, half_window_h+1-2):
                        for dj in range(-half_window_w, half_window_w+1):
                            if is_in_border(i+di,j+dj,H,W):
                                effect_num += 1
                    if effect_num == 0:
                        continue
                    weight = np.zeros((effect_num,1))
                    normal = np.zeros((effect_num,3))
                    effect_num = 0
                    for di in range(-half_window_h-2, half_window_h+1-2):
                        for dj in range(-half_window_w, half_window_w+1):
                            if is_in_border(i+di,j+dj,H,W):
                                # weight[effect_num] = 1.0
                                weight[effect_num] = self.weight_estimation(
                                    i,j,di,dj,\
                                    raw_range_dept, raw_range_inte
                                )
                                normal[effect_num,:] = raw_range_norm[i+di,j+dj,:]
                                effect_num += 1
                    weight /= np.sum(weight)
                    res_n = np.zeros((1,3))
                    for k in range(effect_num):
                        res_n += weight[k,:]*normal[k,:]
                    res_n /= np.linalg.norm(res_n)
                    res_range_norm[i,j] = res_n

        return res_range_norm

    def normal_comp_matrix_completion(self, raw_range_norm):
        nx_map = raw_range_norm[:,:,0]
        ny_map = raw_range_norm[:,:,1]
        nz_map = raw_range_norm[:,:,2]
        invalid_idx = ((nx_map == 0) & (ny_map == 0) & (nz_map == 0))
        valid_idx   = ~invalid_idx

        phis = np.arcsin(nz_map[valid_idx]/1.) 
        phi_map = np.zeros_like(nz_map)
        phi_map[valid_idx] = phis

        theta = np.arctan2(ny_map[valid_idx], nx_map[valid_idx])
        theta_map = np.zeros_like(nz_map)
        theta_map[valid_idx] = theta

        # pre-process with medianblur
        CROSS_KERNEL_3 = np.asarray(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.uint8)
        phi_map_d   = cv.dilate(phi_map, CROSS_KERNEL_3)
        theta_map_d = cv.dilate(theta_map, CROSS_KERNEL_3)
        # phi_map_d   = cv.dilate(phi_map_d, CROSS_KERNEL_3)
        # theta_map_d = cv.dilate(theta_map_d, CROSS_KERNEL_3)
        phi_mask   = (phi_map_d == 0)
        theta_mask = (theta_map_d == 0)

        # matrix completion on theta and phi maps
        softImpute = SoftImpute()
        # softImpute = KNN(k=5)
        # softImpute = IterativeSVD(rank=10)
        # softImpute=SimpleFill(fill_method="median") # mean median

        phi_map_incp = phi_map_d*1.0
        phi_map_incp[phi_mask] = np.nan
        phi_map_cp = softImpute.fit_transform(phi_map_incp)

        theta_map_incp = theta_map_d*1.0
        theta_map_incp[phi_mask] = np.nan
        theta_map_cp = softImpute.fit_transform(theta_map_incp)

        # lam = 10./255 * 0.5
        # phi_map_cp = ptv.tvgen(phi_map_cp,      np.array([lam, lam]),        [1, 2],                   np.array([1, 1]))
        # theta_map_cp = ptv.tvgen(theta_map_cp,      np.array([lam, lam]),        [1, 2],                   np.array([1, 1]))

        # generate dense normal map from theta and phi maps
        # # phi_map_cp = np.clip(phi_map_cp, 0, 1)
        nz_map_new = np.sin(phi_map_cp)
        # # theta_map_cp = np.clip(theta_map_cp, 0, 1)
        tmp_map = np.sqrt((np.ones_like(nz_map_new) - nz_map_new**2))

        nx_map_new = np.cos(theta_map_cp)*tmp_map
        ny_map_new = np.sin(theta_map_cp)*tmp_map

        res_range_norm = np.zeros_like(raw_range_norm)
        res_range_norm[:,:,0] = nx_map_new
        res_range_norm[:,:,1] = ny_map_new
        res_range_norm[:,:,2] = nz_map_new

        # debug
        # phi_map_vis   = getMapVis(phi_map)
        # theta_map_vis = getMapVis(theta_map)
        # phi_map_cp_vis   = getMapVis(phi_map_cp)
        # theta_map_cp_vis = getMapVis(theta_map_cp)

        # cv.imshow("phi_map_vis", phi_map_vis)
        # cv.imshow("phi_map_cp_vis", phi_map_cp_vis)
        # cv.imshow("theta_map_vis", theta_map_vis)
        # cv.imshow("theta_map_cp_vis", theta_map_cp_vis)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return res_range_norm
    
    # guided filter performance bad
    def guidedfilter(self, I, p, r, eps):
        # height, width = I.shape
        m_I = cv.boxFilter(I, -1, (r, r))
        m_p = cv.boxFilter(p, -1, (r, r))
        m_Ip = cv.boxFilter(I * p, -1, (r, r))
        
        # m_I = np.expand_dims(m_I,-1)
        cov_Ip = m_Ip - m_I * m_p

        m_II = cv.boxFilter(I * I, -1, (r, r))
        # m_II = np.expand_dims(m_II,-1)
        var_I = m_II - m_I * m_I

        a = cov_Ip / (var_I + eps)
        b = m_p - a * m_I

        m_a = cv.boxFilter(a, -1, (r, r))
        m_b = cv.boxFilter(b, -1, (r, r))
        return m_a * I + m_b

    # not good
    def normal_guided_depth_estimation(self, range_dept, range_norm):
        '''
            estimate depth with least-square optimization in the 
            spherical coordinate system

            however, it has bad performance =_=
        '''
        CROSS_KERNEL_3 = np.asarray(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.uint8)
        range_dept_d = cv.dilate(range_dept, CROSS_KERNEL_3)

        fov_up   =   3.0 / 180.0 * np.pi
        fov_down = -25.0 / 180.0 * np.pi
        fov = abs(fov_down) + abs(fov_up)
        H   = range_dept.shape[0]
        W   = range_dept.shape[1]
        
        yaw_mat   = np.zeros_like(range_dept) # [64, 2048]
        pitch_mat = np.zeros_like(range_dept)

        for id_w in range(W):
            proj_x = id_w/W
            yaw    = proj_x*2.0-1.0
            yaw    = yaw*np.pi*(-1)
            yaw_mat[:, id_w] = (yaw)

        for id_h in range(H):
            proj_y = id_h/H
            pitch  = (1.0-proj_y)*fov-abs(fov_down)
            pitch_mat[id_h, :] = (pitch)

        map_depth_normal_sphere = \
            range_norm[:,:,0]*range_dept_d*np.cos(pitch_mat)*np.cos(yaw_mat) + \
            range_norm[:,:,1]*range_dept_d*np.cos(pitch_mat)*np.sin(yaw_mat) + \
            range_norm[:,:,2]*range_dept_d*np.sin(pitch_mat)
        mask = (map_depth_normal_sphere != 0)

        map_normal_sphere = \
            range_norm[:,:,0]*np.cos(pitch_mat)*np.cos(yaw_mat) + \
            range_norm[:,:,1]*np.cos(pitch_mat)*np.sin(yaw_mat) + \
            range_norm[:,:,2]*np.sin(pitch_mat)

        softImpute = SoftImpute()
        map_depth_normal_sphere_incp = (map_depth_normal_sphere*1.0)
        map_depth_normal_sphere_incp[~mask] = np.nan
        map_depth_normal_sphere_cp = softImpute.fit_transform(map_depth_normal_sphere_incp)

        map_depth_normal_sphere_d = map_depth_normal_sphere
        # map_depth_normal_sphere_d = np.abs(map_normal_sphere)
        map_depth_normal_sphere_d = map_depth_normal_sphere_cp
        # return np.abs(map_depth_normal_sphere_cp)

        mask_real = (range_dept != 0)
        map_depth_normal_sphere_d[mask_real] = map_depth_normal_sphere[mask_real]

        depth_map = map_depth_normal_sphere_d/(map_normal_sphere)
        

        # removel strange value
        th_max = 100
        idx_strange = (depth_map >= th_max)
        depth_map[idx_strange] = 0
        th_min = 0
        idx_strange = (depth_map <= th_min)
        depth_map[idx_strange] = 0

        return depth_map

    def normal_depth_opt_vis(self, raw_depth, coarse_depth, normal):
        # attempt to use multi guided filter
        sigma = 1
        normal_32F = normal.astype(np.float32)
        coarse_depth_32F = coarse_depth.astype(np.float32)
        raw_depth_32F = raw_depth.astype(np.float32)
        valid_idx = (raw_depth_32F[:,:] != 0)

        normal_32F[:,:,0] = coarse_depth_32F/100.0
        normal_32F[:,:,1] = coarse_depth_32F/100.0
        normal_32F[:,:,2] = coarse_depth_32F/100.0

        guided_filter = GuidedFilterColor(
            normal_32F, radius=sigma, epsilon=0.0005)
        C_smooth = guided_filter.filter(coarse_depth_32F)
        C_smooth[valid_idx] = raw_depth_32F[valid_idx]

        # normal_32F[:,:,0] = C_smooth/100.0
        # normal_32F[:,:,1] = C_smooth/100.0
        # normal_32F[:,:,2] = C_smooth/100.0
        # guided_filter = GuidedFilterColor(
        #     normal_32F, radius=sigma, epsilon=0.0005)
        # C_smooth = guided_filter.filter(C_smooth)
        # C_smooth[valid_idx] = raw_depth_32F[valid_idx]

        return C_smooth

    def normal_depth_opt(self, raw_depth, coarse_depth, normal):
        '''
            how to smooth the coarse depth with the noised normal?
            it is an essential problem in this project.
        '''

        # attempt to use multi guided filter
        sigma = 1
        # sigma = 4
        normal_32F = normal.astype(np.float32)
        coarse_depth_32F = coarse_depth.astype(np.float32)
        raw_depth_32F = raw_depth.astype(np.float32)
        valid_idx = (raw_depth_32F[:,:] != 0)

        # normal_32F[:,:,0] = raw_depth_32F/100.0
        # normal_32F[:,:,1] = raw_depth_32F/100.0
        # normal_32F[:,:,2] = raw_depth_32F/100.0

        guided_filter = GuidedFilterColor(
            normal_32F, radius=sigma, epsilon=0.0005)
        C_smooth = guided_filter.filter(coarse_depth_32F)
        # C_smooth[valid_idx] = raw_depth_32F[valid_idx]

        # remove fly points
        ref_depth_map = self.ref_depth_map
        residual_map  = np.abs(C_smooth - ref_depth_map)/ref_depth_map
        fly_index     = (residual_map >= 0.15) \
            & (C_smooth <= 15.0) #& (ref_depth_map != 0)
        C_smooth[fly_index] = ref_depth_map[fly_index]
        C_smooth[valid_idx] = raw_depth_32F[valid_idx]

        normal_32F[:,:,0] = C_smooth/100.0
        normal_32F[:,:,1] = C_smooth/100.0
        normal_32F[:,:,2] = C_smooth/100.0
        guided_filter = GuidedFilterColor(
            normal_32F, radius=sigma, epsilon=0.0005)
        C_smooth = guided_filter.filter(C_smooth)
        # C_smooth[valid_idx] = raw_depth_32F[valid_idx]

        # remove fly points
        ref_depth_map = self.ref_depth_map
        residual_map  = np.abs(C_smooth - ref_depth_map)/ref_depth_map
        fly_index     = (residual_map >= 0.15) \
            & (C_smooth <= 15.0) #& (ref_depth_map != 0)
        C_smooth[fly_index] = ref_depth_map[fly_index]
        C_smooth[valid_idx] = raw_depth_32F[valid_idx]

        # for debug
        # raw_depth_vis = getDepthLogVis(raw_depth*1.0)
        # coarse_depth_vis = getDepthLogVis(coarse_depth*1.0)
        # normal_vis = getNormalVis(normal*1.0)
        # refine_depth_vis = getDepthLogVis(C_smooth*1.0)

        # cv.imshow("raw_depth_vis", raw_depth_vis)
        # cv.imshow("coarse_depth_vis", coarse_depth_vis)
        # cv.imshow("normal_vis", normal_vis)
        # cv.imshow("refine_depth_vis", refine_depth_vis)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # return coarse_depth
        return C_smooth

    def depth_comp_normal_guided_vis(self, raw_range_view, res_range_norm):
        valid_idx = (raw_range_view[:,:,0] != 0)
        coarse_depth_map = basic_depth_comp(raw_range_view*1.0)
        coarse_depth_map[valid_idx,0] = raw_range_view[valid_idx,0]

        depth_map_cp = self.normal_depth_opt_vis(
                raw_range_view[:,:,0], coarse_depth_map[:,:,0], res_range_norm
            )
        depth_map_cp = cv.medianBlur(depth_map_cp, 5)

        # depth_map = coarse_depth_map
        depth_map = np.expand_dims(depth_map_cp,-1)
        return depth_map

    def depth_comp_normal_guided(self, raw_range_view, res_range_norm):
        CROSS_KERNEL_3 = np.asarray(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.uint8)
        valid_idx = (raw_range_view[:,:,0] != 0)
        coarse_depth_map = cv.dilate(raw_range_view, CROSS_KERNEL_3)
        # coarse_depth_map = cv.dilate(coarse_depth_map, CROSS_KERNEL_3)
        # coarse_depth_map = cv.dilate(coarse_depth_map, CROSS_KERNEL_3)
        # coarse_depth_map = cv.medianBlur(coarse_depth_map, 3)
        coarse_depth_map[valid_idx] = raw_range_view[valid_idx,0]

        # a simple baseline
        softImpute = SoftImpute()
        depth_map_incp = coarse_depth_map*1.0
        dep_mask   = (coarse_depth_map == 0)
        depth_map_incp[dep_mask] = np.nan
        depth_map_cp = softImpute.fit_transform(depth_map_incp)

        # used to correct the fly points
        depth_map_cp[valid_idx] = raw_range_view[valid_idx,0]
        yy = basic_depth_comp(raw_range_view*1.0)
        self.ref_depth_map = depth_map_cp*1.0
        self.ref_depth_map = yy[:,:,0]

        # normal optimization (guided filter)
        ## however, it has the bad performance
        ## depth_map_cp = self.normal_guided_depth_estimation(raw_range_view[:,:,0], res_range_norm)
        depth_houbei = depth_map_cp*1.0
        depth_map_cp = self.normal_depth_opt(
                raw_range_view[:,:,0], depth_map_cp, res_range_norm
            )
        
        # remove holes
        holes_idx = (depth_map_cp <= 1.0)
        depth_map_cp[holes_idx] = depth_houbei[holes_idx]

        # to make depth more smooth and improve point cloud quality
        depth_map_cp = cv.bilateralFilter(depth_map_cp, 5, 1.5, 2.0)

        # remove fly points
        # ref_depth_map = self.ref_depth_map
        # residual_map  = np.abs(depth_map_cp - ref_depth_map)/ref_depth_map
        # fly_index     = (residual_map >= 0.2)
        # depth_map_cp[fly_index] = ref_depth_map[fly_index]

        # depth_map_cp = depth_map
        depth_map = np.expand_dims(depth_map_cp,-1)
        return depth_map

    def upsample(
        self, raw_range_view, raw_range_inte, raw_range_norm):

        # step-1: decompose normal map as theta and phi maps
        #         and complete these maps with matrix completion
        res_range_norm = self.normal_comp_matrix_completion(raw_range_norm)

        # step-2: complete the depth map with the normal map
        res_range_dept = self.depth_comp_normal_guided(raw_range_view, res_range_norm)
        # res_range_dept = self.depth_comp_normal_guided_vis(raw_range_view, res_range_norm)

        # step-3: back-projection to get dense point cloud
        # res_range_dept[:16] = 0
        self.final_pts = self.laser_scan.do_range_back_proj(
            res_range_dept, res_range_dept)

        return res_range_norm, res_range_dept

    def upsample_matrix_cp(self, raw_range_view):
        CROSS_KERNEL_3 = np.asarray(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ], dtype=np.uint8)
        valid_idx = (raw_range_view[:,:,0] != 0)
        
        depth_map = cv.dilate(raw_range_view, CROSS_KERNEL_3)
        # depth_map = cv.dilate(depth_map, CROSS_KERNEL_3)
        # depth_map = cv.dilate(depth_map, CROSS_KERNEL_3)
        # depth_map[valid_idx] = raw_range_view[valid_idx,0]

        # a simple baseline
        softImpute = SoftImpute()
        depth_map_incp = depth_map*1.0
        dep_mask   = (depth_map == 0)
        depth_map_incp[dep_mask] = np.nan
        depth_map_cp = softImpute.fit_transform(depth_map_incp)

        knnImpute = KNN(k=3)
        depth_map_knn = knnImpute.fit_transform(depth_map_incp)
        depth_map_knn = cv.medianBlur(depth_map_knn, 3)

        svdImpute = IterativeSVD(rank=10)
        depth_map_svd = svdImpute.fit_transform(depth_map_incp)

        depth_map = np.expand_dims(depth_map_cp,-1)
        depth_map_knn = np.expand_dims(depth_map_knn,-1)
        depth_map_svd = np.expand_dims(depth_map_svd,-1)

        return depth_map, depth_map_knn, depth_map_svd