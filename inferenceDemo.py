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

import argparse
import glob
import numpy as np
import torch
import open3d as o3d

# if has mayavi
# import mayavi.mlab as mlab
# from utilsVisual import draw_scenes

from pathlib import Path
from easydict import EasyDict
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pointProcess import PointPcdProcessor
from post3dVisulizer import show3DdetectionFromModel

class DemoDataset(DatasetTemplate):
    def __init__(
        self, dataset_cfg, class_names, 
        training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list
        width = 1242
        height = 375
        self.pointProcessor = PointPcdProcessor(width, height)

    def __len__(self):
        return len(self.sample_file_list)

    def preProcessOpen3d(self, pcd):
        # sub-sampling, normal computation, add intensity value
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        downpcd = self.pointProcessor.normalComp(downpcd)
        
        points = np.array(downpcd.points) 
        normal = np.array(downpcd.normals)
        num    = points.shape[0]
        cha    = points.shape[1]
        
        if cha == 3:
            intens = np.ones((num,1), dtype=np.float32) * 0.05 
            rgbs   = np.ones((num,3), dtype=np.float32) * 0.3 
            mcr    = np.ones((num,1), dtype=np.float32) * 0.5 
            seg    = np.ones((num,1), dtype=np.float32) * 0.5
            points = np.concatenate((points, intens, rgbs, normal, mcr, seg), axis=1)
        return points

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
            points = points[:,:4]
            points[:, 3] /= 256.0 # debug

            z = points[:, 2]
            valid_idx = (z >= -1.8)
            points = points[valid_idx, :]

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            pcd = o3d.io.read_point_cloud(self.sample_file_list[index])
            points = self.preProcessOpen3d(pcd)
        elif self.ext == '.PCD':
            pcd = o3d.io.read_point_cloud(self.sample_file_list[index])  
            points = self.preProcessOpen3d(pcd)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def main():
    logger = common_utils.create_logger()
    base_path = "/media/anpei/DiskA/weather-transfer-anpei/"
    cfg_file_path = base_path + "tools/cfgs/voxel_rcnn/" + "inference_anpei.yaml"
    cfg = EasyDict()
    cfg_from_yaml_file(cfg_file_path, cfg)

    # data_path = base_path + "anpei_visulization/inference_data/" + "test_PCD/"
    data_path = base_path + "anpei_visulization/inference_data/" + "test_bin/"

    ckpt_path = base_path + "model_zoos/" + "semi_spl_64p_allin.pth"
    ckpt_path = base_path + "model_zoos/" + "semi_spl_32p_allin.pth"
    ckpt_path = base_path + "toy_models/" + "base_60.pth"
    data_format = ".bin"
    # data_format = ".pcd"
    # data_format = ".PCD"
    logger.info('==> data_path: %s' % data_path)

    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(data_path), ext=data_format, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=ckpt_path, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print("pred_dicts: ", pred_dicts)

            # visulization with open3d
            show3DdetectionFromModel(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'])

            # draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )
            # mlab.show(stop=True)

    logger.info('Demo done.') 

if __name__ == '__main__':
    main()