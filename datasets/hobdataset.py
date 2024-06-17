import time
from numpy import random
from numpy.core.numeric import full
import torch
import numpy as np
import os
import scipy.ndimage
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import random
from torch._C import dtype, set_flush_denormal
import utils.basic
import utils.improc
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset, BBoxDataset
from pathlib import Path
import utils.misc


class HOBDataset(BBoxDataset):
    def __init__(self,
                 dataset_location='/orion/group/hob',
                 S=32, fullseq=False, chunk=None,
                 strides=[1], # low fps < this is actually entangled with our discard strat
                 zooms=[1,1.25,1.5],
                 crop_size=(384,512),
                 use_augs=False,
                 is_training=True):
        print('loading hob dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training)

        clip_step = S//2
        if not is_training:
            # the dataset is only a couple videos
            strides = [1]
            clip_step = S
        
        self.root = Path(dataset_location)
        self.gt_path = self.root / 'HOB.json'
        self.gt = json.loads(open(self.gt_path, "r").read())
        self.videos = list(self.gt.keys())

        # we'll use the whole set as training

        # N = len(self.videos)//10
        # if is_training:
        #     self.videos = self.videos[N:]
        # else:
        #     self.videos = self.videos[:N]
        print('found {:d} {} videos in {}'.format(len(self.videos), ('train' if is_training else 'test'), self.dataset_location))

        # discard the frames where we don't have gt
        for video_name in self.videos:
            all_frames = self.gt[video_name]["img_names"]
            gt_frames = self.gt[video_name]["gt_frames"]
            gt_frames = [video_name + "\\" + i for i in gt_frames]
            gt_indices = [all_frames.index(i) for i in gt_frames]
            self.gt[video_name]["img_names"] = [all_frames[i] for i in gt_indices]
            self.gt[video_name]["gt_rect"] = [self.gt[video_name]["gt_rect"][i] for i in gt_indices]
            self.gt[video_name]["gt_occlusion"] = [self.gt[video_name]["gt_occlusion"][i] for i in gt_indices]
            #print(video_name, gt_indices)
            #exit(0)

        self.all_info = []
        for fn in self.videos:

            bboxes = self.gt[fn]["gt_rect"]
            occluder_bboxes = self.gt[fn]["gt_occlusion"]

            bboxes = np.asarray(bboxes)
            occluder_bboxes = np.asarray(occluder_bboxes)

            bboxes[..., 2:] += bboxes[..., :2]
            occluder_bboxes[..., 2:] += occluder_bboxes[..., :2]
            
            S_local = len(bboxes)
            print('S_local', S_local)
            inter_ratio = np.zeros((len(bboxes)))
            for i in range(len(bboxes)):
                xA = max(bboxes[i][0], occluder_bboxes[i][0])
                yA = max(bboxes[i][1], occluder_bboxes[i][1])
                xB = min(bboxes[i][2], occluder_bboxes[i][2])
                yB = min(bboxes[i][3], occluder_bboxes[i][3])

                inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
                bbox_area = (bboxes[i][2] - bboxes[i][0] + 1) * (bboxes[i][3] - bboxes[i][1] + 1)

                inter_ratio[i] = inter_area / float(bbox_area)
            visibs = 1.0 - inter_ratio

            for stride in strides:
                for ii in range(0, S_local, clip_step*stride):
                    # print('bbox_areas[%d]' % ii, bbox_areas[ii])
                    full_idx = ii + np.arange(self.S)*stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    if len(full_idx)==S and np.sum(visibs.round()) > 3: # fullseq
                        for zoom in zooms: 
                            self.all_info.append([fn, full_idx, zoom])
                        
        print('found {:d} {} samples in {}'.format(len(self.all_info), ('train' if is_training else 'test'), self.dataset_location))

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
        

    def __len__(self):
        return len(self.all_info)

    def getitem_helper(self, index):
        video_name, full_idx, zoom = self.all_info[index]
        img_paths = [self.root / i.replace("\\","/") for i in self.gt[video_name]["img_names"]]
        #print(img_paths[0])
        bboxes = self.gt[video_name]["gt_rect"]
        occluder_bboxes = self.gt[video_name]["gt_occlusion"]     
        
        if len(img_paths) > len(bboxes):
            # print(f'{gt_fn.parts[-2]}: {len(bboxes)} != {len(img_paths)}')
            img_paths = img_paths[:len(bboxes)]

        bboxes = np.asarray(bboxes)
        occluder_bboxes = np.asarray(occluder_bboxes)
        
        bboxes = bboxes[full_idx]
        occluder_bboxes = occluder_bboxes[full_idx]
        
        img_paths = [img_paths[ii] for ii in full_idx]
            
        bboxes[..., 2:] += bboxes[..., :2]
        occluder_bboxes[..., 2:] += occluder_bboxes[..., :2]
        inter_ratio = np.zeros((len(bboxes)))
        for i in range(len(bboxes)):
            xA = max(bboxes[i][0], occluder_bboxes[i][0])
            yA = max(bboxes[i][1], occluder_bboxes[i][1])
            xB = min(bboxes[i][2], occluder_bboxes[i][2])
            yB = min(bboxes[i][3], occluder_bboxes[i][3])

            inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            bbox_area = (bboxes[i][2] - bboxes[i][0] + 1) * (bboxes[i][3] - bboxes[i][1] + 1)

            inter_ratio[i] = inter_area / float(bbox_area)
        visibs = 1.0 - inter_ratio

        rgbs = [cv2.imread(str(path))[..., ::-1].copy() for path in img_paths]
        
        rgbs = np.stack(rgbs)
        bboxes = np.stack(bboxes)

        rgbs, bboxes = utils.misc.data_pad_if_necessary_b(rgbs, bboxes, visibs)
        S,H,W,C = rgbs.shape
        if zoom > 1:
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            bboxes, visibs, rgbs = utils.misc.data_zoom_bbox(zoom, bboxes, visibs, rgbs)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None
        
        sample = {
            'rgbs': rgbs,
            'visibs': visibs, # S
            'bboxes': bboxes,
        }
        return sample


