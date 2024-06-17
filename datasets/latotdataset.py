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
from utils.misc import mask2bbox
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import PointDataset, BBoxDataset, bbox2mask
from pathlib import Path
from icecream import ic


class LaTOTDataset(BBoxDataset):
    def __init__(self,
                 dataset_location='../LaTOT',
                 S=32, fullseq=False, chunk=None,
                 rand_frames=False,
                 crop_size=(384,512),
                 strides=[1, 2, 3, 4],
                 zooms=[1, 2],
                 use_augs=False,
                 is_training=True):
        print('loading LaTOT dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            fullseq=fullseq,
            rand_frames=rand_frames,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training
        )

        # test gt not provided
        self.root = Path(dataset_location) / 'LaTOT/TOT'
        self.seq_names = open(Path(dataset_location) / '{}.txt'.format('train' if is_training else 'test')).read().splitlines()
        print('found {:d} videos in {}'.format(len(self.seq_names), self.dataset_location))

        if not is_training:
            strides = [1]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S // 2
            
        self.all_info = []
        for seq_name in self.seq_names:
            self.process_video(seq_name, strides, zooms, clip_step)

            sys.stdout.write(".")
            sys.stdout.flush()
            
        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

    def process_video(self, seq_name, strides, zooms, clip_step):
        bboxes = self.load_bboxes(seq_name)
        if len(bboxes) < 4:
            return
        
        for stride in strides:
            self.extract_clips(seq_name, stride, zooms, bboxes, clip_step)
            
    def load_bboxes(self, seq_name):
        file_path = self.root / seq_name / f'{seq_name}.txt'
        for delimiter in [' ', ',', '\t']:
            try:
                return np.loadtxt(file_path, delimiter=delimiter).astype(np.float32)
            except:
                continue
        raise ValueError(f'Failed to load bounding boxes for sequence {seq_name}')

    def extract_clips(self, seq_name, stride, zooms, bboxes, clip_step):
        S_local = len(bboxes)
        for start_idx in range(0, S_local, clip_step*stride):
            full_idx = start_idx + np.arange(self.S) * stride
            full_idx = full_idx[full_idx < S_local]
            
            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if self.fullseq else 4):
                continue
            
            bboxes_here = bboxes[full_idx]
            dists = np.linalg.norm(bboxes_here[1:, :2] - bboxes_here[:-1, :2], axis=-1)
            if np.mean(dists) < 2.:
                continue

            # # print('np.max(dists)', np.max(dists))
            # if np.max(dists) > 64.:
            #     continue
            
            for zoom in zooms:
                self.all_info.append((seq_name, stride, full_idx, zoom))
                 
    def getitem_helper(self, index):
        seq_name, stride, full_idx, zoom = self.all_info[index]
        image_paths = []
        bboxes = self.load_bboxes(seq_name)
        found_paths = sorted(glob.glob(str(self.root / seq_name / 'img/*.jpg')))
        for i in range(len(bboxes)):
            image_paths.append(found_paths[i])
        
        image_paths = [image_paths[ii] for ii in full_idx]
        bboxes = bboxes[full_idx]
        # from xywh to xyxy
        bboxes[..., 2:] += bboxes[..., :2]
        
        rgb = cv2.imread(str(image_paths[0]))
        H, W = rgb.shape[:2]
            
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)

        rgbs = np.stack(rgbs)
        S = rgbs.shape[0]

        visibs = np.ones_like(bboxes[:,0])

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
            'visibs': visibs,  # S
            'bboxes': bboxes,
        }
        return sample


    def __len__(self):
        return len(self.all_info)
