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
from datasets.dataset import MaskDataset, BBoxDataset, mask2bbox
from icecream import ic
import utils.misc
from pathlib import Path
from datasets.dataset_utils import make_split

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

class YCBInEOATDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../YCBInEOAT',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384, 512),
                 strides=[1,2,3],
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True):
        print('loading YCBInEOAT dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            fullseq=fullseq,
            use_augs=use_augs,
            is_training=is_training
        )
        
        if not is_training:
            strides = [1]
            zooms = [1]
            clip_step = S
        else:
            clip_step = S // 2

        self.root = Path(dataset_location)
        # self.split = 'train' if is_training else 'val'
        self.seq_names = sorted([seq_dir.name for seq_dir in self.root.glob('*')])
        # self.seq_names = make_split(self.seq_names, is_training, shuffle=True)  # Define make_split or similar functionality

        print("found {:d} sequences in {}".format(len(self.seq_names), self.dataset_location))

        self.all_info = []
        for seq_name in self.seq_names:
            self.process_video(seq_name, strides, zooms, clip_step)
            sys.stdout.write('.')
            sys.stdout.flush()

        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))

        print('found {:d} samples in {}'.format(len(self.all_info), self.dataset_location))

    def process_video(self, seq_name, strides, zooms, clip_step):
        # NOTE YCBInEOAT has only one tracking id per video
        for stride in strides:
            self.extract_clips(seq_name, stride, zooms, clip_step)

    def extract_clips(self, seq_name, stride, zooms, clip_step):
        S_local = len(sorted(list((self.root / seq_name / 'rgb').glob('*.png'))))
        print('S_Local', S_local)
        for start_idx in range(0, S_local, clip_step * stride):
            full_idx = start_idx + np.arange(self.S) * stride
            full_idx = full_idx[full_idx < S_local]
            
            if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if self.fullseq else 4):
                    continue
            
            for zoom in zooms:
                self.all_info.append((seq_name, stride, full_idx, zoom))

    def getitem_helper(self, index):
        seq_name, stride, full_idx, zoom = self.all_info[index]
        
        image_paths = np.array(sorted((self.root / seq_name / 'rgb').glob('*.png')))[full_idx]
        rgbs = np.stack([cv2.imread(str(path))[..., ::-1] for path in image_paths])
        masks = np.stack([cv2.imread(str(path).replace('rgb', 'gt_mask'), cv2.IMREAD_GRAYSCALE) > 0 for path in image_paths]).astype(np.float32)
        S, H, W, C = rgbs.shape
        
        mask_cnts = np.stack([mask.sum() for mask in masks])
        if mask_cnts.max() <= 64:
            print('burst: max_cnts', mask_cnts.max())
            return None

        bboxes = [mask2bbox(mask) for mask in masks]
        bboxes = np.stack(bboxes, axis=0)

        for i in range(1, len(bboxes)):
            xy_prev = (bboxes[i - 1, :2] + bboxes[i - 1, 2:]) / 2
            xy = (bboxes[i, :2] + bboxes[i, 2:]) / 2
            dist = np.linalg.norm(xy - xy_prev)
            if np.sum(masks[i]) > 0 and np.sum(masks[i - 1]) > 0:
                if dist > 64:
                    print('large motion detected in {}'.format(image_paths[i]))
                    return None

        xys = bboxes[:,0:2]*0.5 + bboxes[:,2:4]*0.5
        travel = np.sum(np.linalg.norm(xys[1:]-xys[:-1], axis=-1))
        if travel < S: return None

        # padding and zooming
        mask_areas = (masks > 0).reshape(masks.shape[0],-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        if zoom > 1:
            xys, _, valids, fills = utils.misc.data_get_traj_from_masks(masks)
            xys = utils.misc.data_replace_with_nearest(xys, valids)
        
            bboxes = np.stack([mask2bbox(mask) for mask in masks])
            whs = bboxes[:,2:4] - bboxes[:,0:2]
            whs = whs[visibs > 0.5]
            if np.mean(whs[:,0])*zoom >= W and np.mean(whs[:,1])*zoom >= H:
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)
        visibs = 1.0 * (visibs>0.5)
        safe = np.concatenate(([0], visibs[:-2] * visibs[1:-1] * visibs[2:], [0]))
        if np.sum(safe) < 2:
            return None
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
