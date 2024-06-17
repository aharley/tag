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

# np.random.seed(125)
# torch.multiprocessing.set_sharing_strategy('file_system')
from datasets.dataset_utils import make_split

class YoutubeVOSDataset(MaskDataset):
    def __init__(self,
                 dataset_location='../youtube',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 strides=[1,2],
                 zooms=[1,2],
                 use_augs=False,
                 is_training=True,
    ):

        print('loading Youtube VOS dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        # note (oct21 by adam): actually the validation set in ytvos is basically empty. we need to split the trainset. 
        # self.dataset_location = os.path.join(dataset_location, 'train' if is_training else 'valid')
        self.dataset_location = os.path.join(dataset_location, 'train')
        
        self.video_names = sorted(os.listdir(os.path.join(self.dataset_location, 'JPEGImages')))

        # note (may3 by adam): i really want strong mask pred now, so let's actually include all of it
        # self.video_names = make_split(self.video_names, is_training, shuffle=True)
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_names = chunkify(self.video_names,100)[chunk]
            print('filtered to %d video_names' % len(self.video_names))
            print('self.video_names', self.video_names)
        
        
        self.all_video_names = []
        self.all_tids = []
        self.all_full_idx = []
        self.all_zooms = []

        clip_step = S//2

        for video_name in self.video_names:
            # print(video_name)
           # sys.stdout.write(video_name)
            
            video_dir = os.path.join(self.dataset_location, 'JPEGImages', video_name)
            annotation_dir = os.path.join(self.dataset_location, 'Annotations', video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = len(frames)

            frame0 = frames[0]
            frameE = frames[-1]
            seg0 = cv2.imread(os.path.join(annotation_dir, frame0.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
            segE = cv2.imread(os.path.join(annotation_dir, frameE.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)

            seg_ = np.concatenate([seg0.reshape(-1), segE.reshape(-1)])
            
            valid_segs = np.array([v for v in np.unique(seg_) if v > 0]) # NSeg
            # print('valid_segs', valid_segs)

            print('S_local', S_local)
            # ytvos videos are around 30 frames or less

            if S_local < (self.S if fullseq else 8): continue

            # full_idx = np.arange(0, S_local, self.S)

            # in this dataset it seems best to sample a single clip per video,
            # adapting the fps to the video length
            full_idx = np.linspace(0, S_local-1, min(self.S, S_local), dtype=np.int32)


            # # for stride in strides:
            # #     for ii in range(0, S_local, clip_step):
            # full_idx = ii + np.arange(self.S)*stride
            # full_idx = [ij for ij in full_idx if ij < S_local]

            # if len(full_idx) >= 8:
            for tid in valid_segs:
                for zoom in zooms:
                    self.all_video_names.append(video_name)
                    self.all_full_idx.append(full_idx)
                    self.all_tids.append(tid)
                    self.all_zooms.append(zoom)
                    sys.stdout.write('.')
                    sys.stdout.flush()
        
    def __len__(self):
        return len(self.all_video_names)
        
    def getitem_helper(self, index):
        video = self.all_video_names[index]
        full_idx = self.all_full_idx[index]
        tid = self.all_tids[index]
        zoom = self.all_zooms[index]

        video_dir = os.path.join(self.dataset_location, 'JPEGImages', video)
        annotation_dir = os.path.join(self.dataset_location, 'Annotations', video)

        frames = sorted(os.listdir(video_dir))
        S_local = len(frames)

        frames = [frames[idx] for idx in full_idx]
        
        # rgbs = [cv2.imread(os.path.join(video_dir, fn))[..., ::-1] for fn in frames]
        # segs = [cv2.imread(os.path.join(annotation_dir, fn.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE) for fn in frames]

        image_paths = [os.path.join(video_dir, fn) for fn in frames]
        seg_paths = [os.path.join(annotation_dir, fn.replace('.jpg', '.png')) for fn in frames]
        rgb = cv2.imread(str(image_paths[0]))
        H, W = rgb.shape[:2]
        # print('S, H, W', len(image_paths), H, W)
        sc = 1.0
        if H > 512:
            sc = 512/H
            H_, W_ = int(H*sc), int(W*sc)
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            if sc < 1.0:
                rgb = cv2.resize(rgb, (W_, H_), interpolation=cv2.INTER_AREA)
            rgbs.append(rgb)
            
        segs = []
        for path in seg_paths:
            seg = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if sc < 1.0:
                seg = cv2.resize(seg, (W_, H_), interpolation=cv2.INTER_NEAREST)
            segs.append(seg)
        masks = [(seg == tid).astype(np.float32) for seg in segs]

        #     # print('would reject')
        #     # big already
        #     return None
        # xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)
        
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        S,H,W,C = rgbs.shape

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
            # print('mean wh', np.mean(whs[:,0]), np.mean(whs[:,1]))
            if np.mean(whs[:,0]) >= W/2 and np.mean(whs[:,1]) >= H/2:
                # print('would reject')
                # big already
                return None
            xys, visibs, valids, rgbs, masks = utils.misc.data_zoom(zoom, xys, visibs, valids, rgbs, masks=masks)

        if np.sum(visibs>0.5) < 3:
            print('np.sum(visibs>0.5)', np.sum(visibs>0.5))
            return None
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }
        
        return sample

