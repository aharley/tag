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
import glob
import json
import imageio
import cv2
import re
import sys
from torchvision.transforms import ColorJitter, GaussianBlur
from datasets.dataset import MaskDataset, mask2bbox
from icecream import ic
from pathlib import Path
from pycocotools import mask as mask_utils
import utils.misc

from datasets.dataset_utils import make_split

class BDD100KDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../BDD100K',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1,2,3],
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True,
    ):

        print('loading BDD100K dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        self.dataset_location = Path(dataset_location) / 'bdd100k/images/seg_track_20/{}'.format('train' if is_training else 'val')
        self.label_location = Path(dataset_location) / 'bdd100k/labels/seg_track_20/rles/{}'.format('train' if is_training else 'val')
        self.video_names = sorted(list(map(lambda fn: fn.stem, self.label_location.glob('*.json'))))
        # self.video_names = make_split(self.video_names, is_training=is_training)
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_names = chunkify(self.video_names,100)[chunk]
            print('filtered to %d video_names' % len(self.video_names))
            print('self.video_names', self.video_names)
        
        # self.all_video_names = []
        # self.all_tids = []
        # self.all_zooms = []
        # self.all_full_idx = []

        clip_step = S//2

        self.all_info = []

        for video_name in self.video_names[:]:
            annos = json.load(open(self.label_location / (video_name + '.json')))['frames']
            tids = [label['id'] for label in annos[0]['labels'] if ('crowd' not in label['attributes'] or label['attributes']['crowd'] == False)]
            
            for tid in tids:
                # find valid anno idx where contain this tid
                frame_ids = [i for i, anno in enumerate(annos) if len([label for label in anno['labels'] if label['id'] == tid]) > 0]
                sidx = min(frame_ids)
                eidx = max(frame_ids) + 1
                for stride in strides:
                    for ii in range(sidx, eidx, clip_step*stride):
                        full_idx = ii + np.arange(self.S)*stride
                        full_idx = sorted(list(set(frame_ids).intersection(full_idx)))

                        if len(full_idx)==S: # always fullseq in this dset
                            for zoom in zooms:
                                self.all_info.append([video_name, full_idx, tid, zoom])
                    sys.stdout.write('.')
                    sys.stdout.flush()
                            
        print('\nfound {:d} samples in {}'.format(len(self.all_info), self.dataset_location))
        
        # if chunk is not None:
        #     assert(len(self.all_info) > 100)
        #     def chunkify(lst,n):
        #         return [lst[i::n] for i in range(n)]
        #     self.all_info = chunkify(self.all_info,100)[chunk]
        #     print('filtered to %d' % len(self.all_info))
        
    def __len__(self):
        return len(self.all_info)
        
    def getitem_helper(self, index):
        video_name, full_idx, tid, zoom = self.all_info[index]

        # print('full_idx', full_idx)

        annos = json.load(open(self.label_location / (video_name + '.json')))['frames']
        image_paths = []
        for idx in full_idx:
            image_paths.append(self.dataset_location / (video_name + '/{}'.format(annos[idx]['name'])))

        rgb = cv2.imread(str(image_paths[0]))
        H, W = rgb.shape[:2]
        # print('S, H, W', len(image_paths), H, W)
        # sc = 1.0
        # if H > 512:
        #     sc = 512/H
        #     H_, W_ = int(H*sc), int(W*sc)
            
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            # if sc < 1.0:
            #     rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
            rgbs.append(rgb)
            
        masks = []
        for idx in full_idx:
            labels = annos[idx]['labels']
            # find the target tid's label index
            target_idx = [i for i, label in enumerate(labels) if label['id'] == tid]
            if len(target_idx) == 0:
                masks.append(np.zeros((H_, W_)))
                continue
            target_idx = target_idx[0]
            if ('crowd' in labels[target_idx]['attributes']) and (labels[target_idx]['attributes']['crowd']):
                masks.append(np.zeros((H_, W_)))
            else:
                rle = labels[target_idx]['rle']
                # use pycocotools to pass rle to mask
                mask = mask_utils.decode(rle)
                # if sc < 1.0:
                #     mask = cv2.resize(mask, (W_, H_), interpolation=cv2.INTER_NEAREST)
                masks.append(mask)

        # if np.mean(masks[0]) < 0.001:
        #     print('mask0_mean', np.mean(masks[0]))
        #     return None

        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        S,H,W,C = rgbs.shape
        # print('H, W', H, W)

        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
         
        rgbs, masks = utils.misc.data_pad_if_necessary(rgbs, masks)
        S,H,W,C = rgbs.shape

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
            print('safe', safe)
            return None

        sample = {
            "rgbs": rgbs,
            "masks": masks,
        }
        return sample

