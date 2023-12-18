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

class datapreparation():
    '''
    1) convert 64-beam lidar point cloud to 8/16/32-beam lidar point cloud
    2) evaluate the accuracy of up-sample operation
    '''
    def __init__(self, beam_num=32):
        self.beam_num = beam_num

    def preview(self, pts):
        pass

    def convert(self, pts):
        laser_scan = LaserScan(H=self.beam_num, W=2048, fov_up=3.0, fov_down=-25.0)
        laser_scan.reset()
        laser_scan.set_points(pts[:, 0:3], pts[:, 3])
        laser_scan.do_range_projection(rgbs=pts[:, 0:3], is_rgb=False)
        range_view = np.copy(laser_scan.proj_range)
        range_inte = np.copy(laser_scan.proj_remission)
        invlid = (range_view == -1) 
        range_view[invlid] = 0
        range_view = np.expand_dims(range_view,-1)
        range_inte[invlid] = 0
        range_inte = np.expand_dims(range_inte,-1)
        final_pts = laser_scan.do_range_back_proj(
            range_view, range_inte)
        return final_pts, range_view, range_inte

    def convert(self, pts, normals):
        laser_scan = LaserScan(H=self.beam_num, W=2048, fov_up=3.0, fov_down=-25.0)
        laser_scan.reset()
        laser_scan.set_points(pts[:, 0:3], pts[:, 3])
        laser_scan.do_range_projection(rgbs=normals[:, 0:3], is_rgb=True)
        range_view = np.copy(laser_scan.proj_range)
        range_inte = np.copy(laser_scan.proj_remission)
        range_normal = np.copy(laser_scan.proj_xyzrgb[:,:,3:6])
        invlid = (range_view == -1) 
        range_view[invlid] = 0
        range_view = np.expand_dims(range_view,-1)
        range_inte[invlid] = 0
        range_inte = np.expand_dims(range_inte,-1)
        range_normal[invlid, :] = 0
        final_pts = laser_scan.do_range_back_proj(
            range_view, range_inte)
        return final_pts, range_view, range_inte, range_normal

    def evaluate(self, pd_range_view, gt_range_view):
        '''
        provide rmse and mae metric of depth upsampling
        '''
        valid_idx = (gt_range_view > 0) 
        inv_valid_idx = (gt_range_view > 0) & (pd_range_view > 0)
        sumdata = np.sum(valid_idx)
        inv_sumdata = np.sum(inv_valid_idx)
        delta = gt_range_view[valid_idx]-pd_range_view[valid_idx]
        inv_delta = 1.0/gt_range_view[inv_valid_idx]-1.0/pd_range_view[inv_valid_idx]
        rmse = np.sqrt(np.sum(delta**2)/sumdata)
        mae  = np.sum(np.abs(delta))/sumdata
        irmse = np.sqrt(np.sum(inv_delta**2)/inv_sumdata)
        imae  = np.sum(np.abs(inv_delta))/inv_sumdata
        return rmse, mae, irmse, imae

class baseline():
    '''
    a baseline upsample method that uses the basic image processing
    upsample 8/16/32-beam lidar point to 64/128-beam lidar point
    '''
    def __init__(self, up_beam_num=64):
        self.up_beam_num = up_beam_num
        self.laser_scan = LaserScan(H=up_beam_num, W=2048, fov_up=3.0, fov_down=-25.0)

    def upsample(self, range_view, range_inte):
        final_range_view = basic_depth_comp(range_view*1.0)
        final_range_inte = basic_depth_comp(range_inte*1.0)
        # final_range_view = range_view*1.0
        # final_range_inte = range_inte*1.0
        
        final_pts = self.laser_scan.do_range_back_proj(
            final_range_view, final_range_inte)

        return final_pts, final_range_view

    def upsample_interploate(self, range_view, range_stride):
        h_raw = self.up_beam_num // range_stride
        raw_depth = np.zeros((h_raw, range_view.shape[1], 1))

        j = 0
        for i in range(self.up_beam_num):
            if i%range_stride == 0:
                raw_depth[j,:,:] = range_view[i,:,:]
                j += 1
        
        inte_depth = cv.resize(raw_depth, (range_view.shape[1], range_view.shape[0]))
        inte_depth = np.expand_dims(inte_depth,-1)

        return inte_depth
