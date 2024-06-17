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

class VISORDataset(MaskDataset):
    def __init__(self,
                 dataset_location='/orion/group/VISOR/VISOR_2022_Dense',
                 S=32, fullseq=False, chunk=None,
                 crop_size=(384,512), 
                 # strides=[2], # super high framerate already
                 zooms=[1,1.5,2],
                 use_augs=False,
                 is_training=True,
    ):
        print('loading VISOR dataset...')
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training
        )
        
        if not is_training:
            zooms = [1]

        self.is_dense = 'Dense' in dataset_location
        self.dataset_location = os.path.join(dataset_location)
        split = "train" if is_training else "val"
        self.video_names = open(os.path.join(self.dataset_location, f"ImageSets/2022/{split}.txt")).read().splitlines()
        print('found {:d} videos in {}'.format(len(self.video_names), self.dataset_location))
        self.all_info = []

        if chunk is not None:
            assert(len(self.video_names)>100)
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_names = chunkify(self.video_names,100)[chunk]
            print('filtered to %d video_names' % len(self.video_names))
            print('self.video_names', self.video_names)

        assert(self.is_dense) # bc i hardcoded the sampling strategy
        # clip_step = S//2

        for video_name in self.video_names:
            print(video_name)
            # sys.stdout.write(video_name)
            
            video_dir = os.path.join(self.dataset_location, 'JPEGImages', '480p', video_name)
            annotation_dir = os.path.join(self.dataset_location, 'Annotations', '480p', video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = len(frames)
            # print('S_local', S_local)

            frame0 = frames[0]
            seg0 = cv2.imread(os.path.join(annotation_dir, frame0.replace('.jpg', '.png')), cv2.IMREAD_GRAYSCALE)
            
            valid_segs = np.array([v for v in np.unique(seg0.reshape(-1)) if v > 0]) # NSeg
            
            # for stride in strides:
            # for ii in range(0, S_local, clip_step):
            if S_local >= S: # fullseq
                # full_idx = np.arange(0, S_local, self.S)

                # in this dataset it seems best to sample a single clip per video,
                # adapting the fps to the video length
                full_idx = np.linspace(0, S_local-1, min(self.S, S_local), dtype=np.int32)

                for tid in valid_segs:
                    for zoom in zooms:
                        self.all_info.append([video_name, full_idx, tid, zoom])
                    sys.stdout.write('.')
                    sys.stdout.flush()
        print("found {:d} samples in {}".format(len(self.all_info), self.dataset_location))

    def __len__(self):
        return len(self.all_info)
        
    def getitem_helper(self, index):
        video, full_idx, tid, zoom = self.all_info[index]
        # print('full_idx', full_idx, len(full_idx))

        video_dir = os.path.join(self.dataset_location, 'JPEGImages', '480p', video)
        annotation_dir = os.path.join(self.dataset_location, 'Annotations', '480p', video)

        frames = sorted(os.listdir(video_dir))

        frames = [frames[idx] for idx in full_idx]

        image_paths = [os.path.join(video_dir, fn) for fn in frames]
        seg_paths = [os.path.join(annotation_dir, fn.replace('.jpg', '.png')) for fn in frames]
        rgb = cv2.imread(str(image_paths[0]))
        rgbs = []
        for path in image_paths:
            rgb = cv2.imread(str(path))[..., ::-1].copy()
            rgbs.append(rgb)
        segs = []
        for path in seg_paths:
            seg = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if self.is_dense:  # dense mask is scaled
                seg = cv2.resize(seg[:214, :381], (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            segs.append(seg)
        masks = [(seg == tid).astype(np.float32) for seg in segs]
        
        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        S,H,W,C = rgbs.shape

        bboxes = np.stack([mask2bbox(mask) for mask in masks])
        whs = bboxes[:,2:4] - bboxes[:,0:2]
        if np.max(whs[:,0]) >= 3*W/4 or np.max(whs[:,1]) >= 3*H/4:
            return None

        
        if np.sum(masks) == 0:
            print('np.sum(masks)', np.sum(masks))
            return None

        mask_areas = (masks > 0).reshape(S,-1).sum(axis=1)
        mask_areas_norm = mask_areas / np.max(mask_areas)
        visibs = mask_areas_norm
        if self.is_training:
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
        if np.sum(safe) < 2: return None
        
        sample = {
            'rgbs': rgbs,
            'masks': masks,
        }
        
        return sample
