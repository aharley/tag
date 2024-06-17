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
from datasets.dataset import MaskDataset, mask2bbox
from icecream import ic
from scipy.io import loadmat

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')

class VSB100Dataset(MaskDataset):
    def __init__(self,
                 dataset_location='../VSB100',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1], # already low fps
                 zooms=[1,1.5],
                 use_augs=False,
                 is_training=True):

        print('loading VSB100 dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )

        clip_step = S//2
        if not is_training:
            strides = [1]
            clip_step = S
        
        self.dataset_location = os.path.join(dataset_location, 'General_{}_fullres'.format('traindense' if is_training else 'test'))
        self.video_names = os.listdir(os.path.join(self.dataset_location, 'Groundtruth'))
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))

        self.all_info = []

        for video_name in self.video_names:
            # print(video_name)
            # sys.stdout.write(video_name)
            
            video_dir = os.path.join(self.dataset_location, 'Images', video_name)
            annotation_dir = os.path.join(self.dataset_location, 'Groundtruth', video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = len(frames)
            print('S_local', S_local)
        
            frame0 = frames[0]

            gt = loadmat(os.path.join(annotation_dir, frame0.replace('.jpg', '.mat')))['groundTruth']
            for anno_type in range(4):  # four annotators; https://lmb.informatik.uni-freiburg.de/resources/datasets/vsb
                if len(gt[0, anno_type][0]) == 0:
                    continue
                seg0 = gt[0, anno_type][0, 0]['Segmentation']
                valid_segs = np.array([v for v in np.unique(seg0.reshape(-1))])
                for stride in strides:
                    for ii in range(0, max(S_local-self.S*stride,1), clip_step*stride):
                        full_idx = ii + np.arange(self.S)*stride
                        full_idx = [ij for ij in full_idx if ij < S_local]
                        if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 4): 
                            continue
                        
                        for tid in valid_segs:
                            for zoom in zooms:
                                self.all_info.append([video_name, full_idx, tid, anno_type, zoom])
                            sys.stdout.write('.')
                            sys.stdout.flush()

        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))
        if chunk is not None:
            assert(len(self.all_info) > 100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.all_info = chunkify(self.all_info,100)[chunk]
            print('filtered to %d' % len(self.all_info))
                            
        
    def __len__(self):
        return len(self.all_info)
        
    def getitem_helper(self, index):
        video, full_idx, tid, anno_type, zoom = self.all_info[index]

        video_dir = os.path.join(self.dataset_location, 'Images', video)
        annotation_dir = os.path.join(self.dataset_location, 'Groundtruth', video)

        frames = sorted(os.listdir(video_dir))

        frames = [frames[idx] for idx in full_idx]

        img_paths = [os.path.join(video_dir, fn) for fn in frames]
        rgb = cv2.imread(str(img_paths[0]))
        H, W = rgb.shape[:2]
        rgbs = []
        for path in img_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        
        segs = []
        for fn in frames:
            gt = loadmat(os.path.join(annotation_dir, fn.replace('.jpg', '.mat')))['groundTruth']
            gt0 = gt[0, anno_type]
            if gt0.size==0:
                print('gt0.shape', gt0.shape)
                return None
            seg = gt0[0, 0]['Segmentation']
            segs.append(seg)

        masks = [(seg == tid).astype(np.float32) for seg in segs]

        rgbs = np.stack(rgbs)
        masks = np.stack(masks)

        S,H,W,C = rgbs.shape
        # print('H, W', H, W)

        print('rgbs', rgbs.shape)
        print('masks', masks.shape)

        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        S,H,W,C = rgbs.shape
        bboxes = np.stack([mask2bbox(mask) for mask in masks])
        whs = bboxes[:,2:4] - bboxes[:,0:2]
        # print('whs', whs, 'max whs', np.max(whs[:,0]), np.max(whs[:,1]))
        if np.max(whs[:,0]) > 3*W/4 or np.max(whs[:,1]) > 3*H/4:
            print('whs', whs)
            return None
        
        # _, _, _, fills = utils.misc.data_get_traj_from_masks(masks)
        # if np.sum(fills) < S//2:
        #     print('fills', fills)
        #     return None

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
            return None
        
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }
        
        return sample

