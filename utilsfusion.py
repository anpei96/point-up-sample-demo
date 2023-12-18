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

import cv2 as cv
import numpy as np
import open3d as o3d
from skimage import io

from pcdet.utils import box_utils, calibration_kitti, common_utils, object3d_kitti

def GetRgbdFeature(root_split_path, idx):
    fea_file = root_split_path / 'velodyne_reduced_ptfea' / ('%s.npy' % idx)
    assert fea_file.exists()
    pointFea = np.load(str(fea_file))
    return pointFea

def GetImage(root_split_path, idx):
    img_file = root_split_path / 'image_2' / ('%s.png' % idx)
    assert img_file.exists()
    return np.array(io.imread(img_file), dtype=np.int32)

def GetImageCv(root_split_path, idx):
    img_file = root_split_path / 'image_2' / ('%s.png' % idx)
    assert img_file.exists()
    image = cv.imread(str(img_file))
    return image

def GetSegImageCv(root_split_path, idx):
    img_file = root_split_path / 'image_2_seg' / ('%s.png' % idx)
    assert img_file.exists()
    image = cv.imread(str(img_file))
    return image

def GetCalib(root_split_path, idx):
    calib_file = root_split_path / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

def ShowPcd(pcd, show_normal=False, show_single_color=False):
    if show_single_color:
        pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([pcd], point_show_normal=show_normal)

def showpcd(pcd, namewin="point cloud", is_black_bg=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window(namewin)
    render_options: o3d.visualization.RenderOption = vis.get_render_option()
    if is_black_bg:
        render_options.background_color = np.array([0,0,0])
    render_options.point_size = 2.0
    # render_options.point_size = 0.5
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.run() 

def cls_type_to_id(cls_type):
    type_to_id = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]

class baseObject3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.cls_id = cls_type_to_id(self.cls_type)
        self.truncation = float(label[1])
        self.occlusion = float(label[2])  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown
        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(label[6]), float(label[7])), dtype=np.float32)
        self.h = float(label[8])
        self.w = float(label[9])
        self.l = float(label[10])
        self.loc = np.array((float(label[11]), float(label[12]), float(label[13])), dtype=np.float32)
        self.dis_to_cam = np.linalg.norm(self.loc)
        self.ry = float(label[14])
        self.score = float(label[15]) if label.__len__() == 16 else -1.0
        self.level_str = None
        self.level = self.get_kitti_obj_level()

    def get_kitti_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1
        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

class fakeObject3d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.src = line
        self.cls_type = label[0]
        self.loc = np.array((float(label[1]), float(label[2]), float(label[3])), dtype=np.float32)
        self.l   = float(label[4])
        self.w   = float(label[5])
        self.h   = float(label[6])
        self.ry  = float(label[7])

def get_baseObjects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    baseObjects = [baseObject3d(line) for line in lines]
    return baseObjects

def get_fakeObjects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    fakeObjects = [fakeObject3d(line) for line in lines]
    return fakeObjects

baseFakePath  = "/media/anpei/anpei_free/project/voxel-perception-anpei/data/kitti/training/"
baseFakePath  = "/media/anpei/DiskA/voxelrcnn-perception-anpei/data/kitti/training/"
fakeLabelPath = baseFakePath + "label_2_fake_64p/"
fakeLabelPath = baseFakePath + "label_2_fake_4p/"
# fakeLabelPath = baseFakePath + "label_2_fake_8p/"
# fakeLabelPath = baseFakePath + "label_2_fake_16p/"
# fakeLabelPath = baseFakePath + "label_2_fake_32p/"
# fakeLabelPath = baseFakePath + "label_2_fake_64p/"

"""
    path setting for weak label training
"""

def pseudolabelGenerator(idx_data, infos):
    """
        only revise infos['annos']

        remember run script anpei_generate_psuedo_labels.sh to generate pseudo labels, 
        and copy these labels txt into label_2_fake
    """
    baseLabelFile = fakeLabelPath + idx_data + ".txt"
    fakeLabelFile = fakeLabelPath + "fake_"  + idx_data + ".txt"
    baseObjects   = get_baseObjects_from_label(baseLabelFile)
    fakeObjects   = get_fakeObjects_from_label(fakeLabelFile)
    
    objectNum = len(baseObjects) # len(baseObjects) = len(fakeObjects)
    fakeInfos = infos

    # return fakeInfos
    if objectNum == 0:
        return fakeInfos

    # name, truncated, occluded, alpha, bbox, dimensions, location
    # rotation_y, score, difficulty, index
    # gt_boxes_lidar, num_points_in_gt
    arrayName  = np.zeros((objectNum)).astype("U10")
    arrayTrunc = np.zeros((objectNum)).astype(np.float)
    arrayOcclu = np.zeros((objectNum)).astype(np.float)
    arrayAlpha = np.zeros((objectNum)).astype(np.float)

    arrayBbox  = np.zeros((objectNum, 4)).astype(np.float)
    arrayDimen = np.zeros((objectNum, 3)).astype(np.float)
    arrayLocal = np.zeros((objectNum, 3)).astype(np.float)

    arrayRy    = np.zeros((objectNum)).astype(np.float)
    arrayScore = np.zeros((objectNum)).astype(np.float)
    arrayDiff  = np.zeros((objectNum)).astype(np.int32)
    arrayIndex = np.zeros((objectNum)).astype(np.int32)

    arrayLidGt = np.zeros((objectNum, 7)).astype(np.float)
    arrayNumGt = np.zeros((objectNum)).astype(np.int32)

    for i in range(objectNum):
        arrayName[i]  = baseObjects[i].cls_type
        arrayTrunc[i] = baseObjects[i].truncation
        arrayOcclu[i] = baseObjects[i].occlusion
        arrayAlpha[i] = baseObjects[i].alpha

        arrayTrunc[i] = 0
        arrayOcclu[i] = 0

        arrayDimen[i][0] = baseObjects[i].l
        arrayDimen[i][1] = baseObjects[i].h # w
        arrayDimen[i][2] = baseObjects[i].w # h
        
        arrayLocal[i, :] = baseObjects[i].loc
        arrayRy[i]    = baseObjects[i].ry
        arrayScore[i] = baseObjects[i].score
        arrayDiff[i]  = baseObjects[i].level
        arrayIndex[i] = i

        arrayLidGt[i][0:3] = fakeObjects[i].loc
        arrayLidGt[i][3]   = fakeObjects[i].l
        arrayLidGt[i][4]   = fakeObjects[i].w
        arrayLidGt[i][5]   = fakeObjects[i].h
        arrayLidGt[i][6]   = fakeObjects[i].ry

        arrayNumGt[i] = 99
        
    fakeInfos['annos']['name'] = arrayName
    fakeInfos['annos']['truncated'] = arrayTrunc
    fakeInfos['annos']['occluded'] = arrayOcclu
    fakeInfos['annos']['alpha'] = arrayAlpha
    fakeInfos['annos']['bbox'] = arrayBbox
    fakeInfos['annos']['dimensions'] = arrayDimen
    fakeInfos['annos']['location'] = arrayLocal
    
    fakeInfos['annos']['rotation_y'] = arrayRy
    fakeInfos['annos']['score'] = arrayScore
    fakeInfos['annos']['difficulty'] = arrayDiff
    fakeInfos['annos']['index'] = arrayIndex

    fakeInfos['annos']['gt_boxes_lidar'] = arrayLidGt
    fakeInfos['annos']['num_points_in_gt'] = arrayNumGt
    
    # debug
    # print("=> ", idx_data)
    # print("objectNum: ", len(baseObjects), len(fakeObjects))
    # print("====================")
    # print("fakeInfos['annos']: ")
    # print(fakeInfos['annos'])
    # assert False

    return fakeInfos

def pseudolabelGeneratorFilter(idx_data, infos):
    """
        only revise infos['annos']
        add a filter to remove low quality bounding boxes

        remember run script anpei_generate_psuedo_labels.sh to generate pseudo labels, 
        and copy these labels txt into label_2_fake
    """
    baseLabelFile = fakeLabelPath + idx_data + ".txt"
    fakeLabelFile = fakeLabelPath + "fake_"  + idx_data + ".txt"
    baseObjects   = get_baseObjects_from_label(baseLabelFile)
    fakeObjects   = get_fakeObjects_from_label(fakeLabelFile)
    
    objectNum = len(baseObjects) # len(baseObjects) = len(fakeObjects)
    fakeInfos = infos

    # return fakeInfos
    if objectNum == 0:
        return fakeInfos

    # add a proposal filter here (adaptive threshold)
    objectNumReal = 0
    scoreTh = 0.4 
    for i in range(objectNum):
        score = baseObjects[i].score
        if score >= scoreTh:
            objectNumReal += 1

    # name, truncated, occluded, alpha, bbox, dimensions, location
    # rotation_y, score, difficulty, index
    # gt_boxes_lidar, num_points_in_gt
    arrayName  = np.zeros((objectNumReal)).astype("U10")
    arrayTrunc = np.zeros((objectNumReal)).astype(np.float)
    arrayOcclu = np.zeros((objectNumReal)).astype(np.float)
    arrayAlpha = np.zeros((objectNumReal)).astype(np.float)

    arrayBbox  = np.zeros((objectNumReal, 4)).astype(np.float)
    arrayDimen = np.zeros((objectNumReal, 3)).astype(np.float)
    arrayLocal = np.zeros((objectNumReal, 3)).astype(np.float)

    arrayRy    = np.zeros((objectNumReal)).astype(np.float)
    arrayScore = np.zeros((objectNumReal)).astype(np.float)
    arrayDiff  = np.zeros((objectNumReal)).astype(np.int32)
    arrayIndex = np.zeros((objectNumReal)).astype(np.int32)

    arrayLidGt = np.zeros((objectNumReal, 7)).astype(np.float)
    arrayNumGt = np.zeros((objectNumReal)).astype(np.int32)

    idx = 0
    for i in range(objectNum):
        # pass low-confidence proposals
        if baseObjects[i].score < scoreTh:
            continue

        arrayName[idx]  = baseObjects[i].cls_type
        arrayTrunc[idx] = baseObjects[i].truncation
        arrayOcclu[idx] = baseObjects[i].occlusion
        arrayAlpha[idx] = baseObjects[i].alpha

        arrayTrunc[idx] = 0
        arrayOcclu[idx] = 0

        arrayDimen[idx][0] = baseObjects[i].l
        arrayDimen[idx][1] = baseObjects[i].h # w
        arrayDimen[idx][2] = baseObjects[i].w # h
        
        arrayLocal[idx, :] = baseObjects[i].loc
        arrayRy[idx]    = baseObjects[i].ry
        arrayScore[idx] = baseObjects[i].score
        arrayDiff[idx]  = baseObjects[i].level
        arrayIndex[idx] = idx

        arrayLidGt[idx][0:3] = fakeObjects[i].loc
        arrayLidGt[idx][3]   = fakeObjects[i].l
        arrayLidGt[idx][4]   = fakeObjects[i].w
        arrayLidGt[idx][5]   = fakeObjects[i].h
        arrayLidGt[idx][6]   = fakeObjects[i].ry

        arrayNumGt[idx] = 99
        idx += 1
        
    fakeInfos['annos']['name'] = arrayName
    fakeInfos['annos']['truncated'] = arrayTrunc
    fakeInfos['annos']['occluded'] = arrayOcclu
    fakeInfos['annos']['alpha'] = arrayAlpha
    fakeInfos['annos']['bbox'] = arrayBbox
    fakeInfos['annos']['dimensions'] = arrayDimen
    fakeInfos['annos']['location'] = arrayLocal
    
    fakeInfos['annos']['rotation_y'] = arrayRy
    fakeInfos['annos']['score'] = arrayScore
    fakeInfos['annos']['difficulty'] = arrayDiff
    fakeInfos['annos']['index'] = arrayIndex

    fakeInfos['annos']['gt_boxes_lidar'] = arrayLidGt
    fakeInfos['annos']['num_points_in_gt'] = arrayNumGt
    
    # debug
    # print("=> ", idx_data)
    # print("objectNum: ", len(baseObjects), len(fakeObjects))
    # print("====================")
    # print("fakeInfos['annos']: ")
    # print(fakeInfos['annos'])
    # assert False

    return fakeInfos

def pseudolabelGeneratorFilter_v2(idx_data, infos, fakeLabelPath):
    """
        only revise infos['annos']
        add a filter to remove low quality bounding boxes

        remember run script anpei_generate_psuedo_labels.sh to generate pseudo labels, 
        and copy these labels txt into label_2_fake
    """
    baseLabelFile = fakeLabelPath + idx_data + ".txt"
    fakeLabelFile = fakeLabelPath + "fake_"  + idx_data + ".txt"
    baseObjects   = get_baseObjects_from_label(baseLabelFile)
    fakeObjects   = get_fakeObjects_from_label(fakeLabelFile)
    
    objectNum = len(baseObjects) # len(baseObjects) = len(fakeObjects)
    fakeInfos = infos

    # return fakeInfos
    if objectNum == 0:
        return fakeInfos

    # add a proposal filter here (adaptive threshold)
    objectNumReal = 0
    scoreTh = 0.4 
    for i in range(objectNum):
        score = baseObjects[i].score
        if score >= scoreTh:
            objectNumReal += 1

    # name, truncated, occluded, alpha, bbox, dimensions, location
    # rotation_y, score, difficulty, index
    # gt_boxes_lidar, num_points_in_gt
    arrayName  = np.zeros((objectNumReal)).astype("U10")
    arrayTrunc = np.zeros((objectNumReal)).astype(np.float)
    arrayOcclu = np.zeros((objectNumReal)).astype(np.float)
    arrayAlpha = np.zeros((objectNumReal)).astype(np.float)

    arrayBbox  = np.zeros((objectNumReal, 4)).astype(np.float)
    arrayDimen = np.zeros((objectNumReal, 3)).astype(np.float)
    arrayLocal = np.zeros((objectNumReal, 3)).astype(np.float)

    arrayRy    = np.zeros((objectNumReal)).astype(np.float)
    arrayScore = np.zeros((objectNumReal)).astype(np.float)
    arrayDiff  = np.zeros((objectNumReal)).astype(np.int32)
    arrayIndex = np.zeros((objectNumReal)).astype(np.int32)

    arrayLidGt = np.zeros((objectNumReal, 7)).astype(np.float)
    arrayNumGt = np.zeros((objectNumReal)).astype(np.int32)

    idx = 0
    for i in range(objectNum):
        # pass low-confidence proposals
        if baseObjects[i].score < scoreTh:
            continue

        arrayName[idx]  = baseObjects[i].cls_type
        arrayTrunc[idx] = baseObjects[i].truncation
        arrayOcclu[idx] = baseObjects[i].occlusion
        arrayAlpha[idx] = baseObjects[i].alpha

        arrayTrunc[idx] = 0
        arrayOcclu[idx] = 0

        arrayDimen[idx][0] = baseObjects[i].l
        arrayDimen[idx][1] = baseObjects[i].h # w
        arrayDimen[idx][2] = baseObjects[i].w # h
        
        arrayLocal[idx, :] = fakeObjects[i].loc
        arrayRy[idx]    = baseObjects[i].ry
        arrayScore[idx] = baseObjects[i].score
        arrayDiff[idx]  = baseObjects[i].level
        arrayIndex[idx] = idx

        arrayLidGt[idx][0:3] = fakeObjects[i].loc
        arrayLidGt[idx][3]   = fakeObjects[i].l
        arrayLidGt[idx][4]   = fakeObjects[i].w
        arrayLidGt[idx][5]   = fakeObjects[i].h
        arrayLidGt[idx][6]   = fakeObjects[i].ry

        arrayNumGt[idx] = 99
        idx += 1
        
    fakeInfos['annos']['name'] = arrayName
    fakeInfos['annos']['truncated'] = arrayTrunc
    fakeInfos['annos']['occluded'] = arrayOcclu
    fakeInfos['annos']['alpha'] = arrayAlpha
    fakeInfos['annos']['bbox'] = arrayBbox
    fakeInfos['annos']['dimensions'] = arrayDimen
    fakeInfos['annos']['location'] = arrayLocal
    
    fakeInfos['annos']['rotation_y'] = arrayRy
    fakeInfos['annos']['score'] = arrayScore
    fakeInfos['annos']['difficulty'] = arrayDiff
    fakeInfos['annos']['index'] = arrayIndex

    fakeInfos['annos']['gt_boxes_lidar'] = arrayLidGt
    fakeInfos['annos']['num_points_in_gt'] = arrayNumGt
    
    # debug
    # print("=> ", idx_data)
    # print("objectNum: ", len(baseObjects), len(fakeObjects))
    # print("====================")
    # print("fakeInfos['annos']: ")
    # print(fakeInfos['annos'])
    # assert False

    return fakeInfos



