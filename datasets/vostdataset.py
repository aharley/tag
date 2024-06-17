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

from datasets.dataset_utils import make_split


class VOSTDataset(MaskDataset):
    def __init__(
            self,
            dataset_location="../VOST",
            S=32, fullseq=False, chunk=None,
            crop_size=(384, 512),
            strides=[1,2], # already low fps
            zooms=[1],
            use_augs=False,
            is_training=True,
    ):
        print("loading VOST dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            crop_size=crop_size,
            use_augs=use_augs,
            is_training=is_training,
        )
        self.dataset_location = dataset_location
        self.video_names = os.listdir(os.path.join(self.dataset_location, "JPEGImages"))
        self.video_names = make_split(self.video_names, is_training, shuffle=True)
        print(
            "found {:d} videos in {}".format(
                len(self.video_names), self.dataset_location
            )
        )

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

        for video_name in self.video_names:
            # print(video_name)
            # sys.stdout.write(video_name)

            video_dir = os.path.join(self.dataset_location, "JPEGImages", video_name)
            annotation_dir = os.path.join(self.dataset_location, "Annotations", video_name)
            frames = sorted(os.listdir(video_dir))
            S_local = len(frames)
            print('S_local', S_local)

            # frame = frames[S_local//2]
            frame = frames[0]
            seg = cv2.imread(os.path.join(annotation_dir, frame.replace(".jpg", ".png")))
            # print('seg', seg.shape)
            # valid_segs = np.array([v for v in np.unique(seg.reshape(-1,3)) if v > 0])
            valid_segs = np.unique(seg.reshape(-1,3), axis=0)
            # print('valid_segs', valid_segs)
            for stride in strides:
                # for ii in range(0, S_local, self.S//2):

                # in this data, objects begin complete and gradually break,
                # and once an object is broken, sometimes a piece that was "initialized" in an earlier window
                # becomes visible again,
                # which is unacceptable in our setup
                # so, we only take samples that begin at 0
                ii = 0
                full_idx = ii + np.arange(self.S) * stride
                full_idx = [ij for ij in full_idx if ij < S_local]

                if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 8):
                    continue
                # if len(full_idx) >= 8:
                for tid in valid_segs:
                    if np.sum(tid) > 0 and not np.all(tid==[192, 224, 224]): # bkg and ignore
                        for zoom in zooms:
                            self.all_video_names.append(video_name)
                            self.all_full_idx.append(full_idx)
                            self.all_tids.append(tid)
                            self.all_zooms.append(zoom)
                sys.stdout.write(".")
                sys.stdout.flush()

    def __len__(self):
        return len(self.all_video_names)

    def getitem_helper(self, index):
        video = self.all_video_names[index]
        full_idx = self.all_full_idx[index]
        tid = self.all_tids[index]
        zoom = self.all_zooms[index]

        video_dir = os.path.join(self.dataset_location, "JPEGImages", video)
        annotation_dir = os.path.join(self.dataset_location, "Annotations", video)

        frames = sorted(os.listdir(video_dir))
        num_frames = len(frames)

        frames = [frames[idx] for idx in full_idx]

        # print('tid', tid)
        rgbs = [cv2.imread(os.path.join(video_dir, fn))[..., ::-1] for fn in frames]
        segs = [cv2.imread(os.path.join(annotation_dir, fn.replace(".jpg", ".png"))) for fn in frames]
        masks = [(np.all(seg == tid, axis=-1)).astype(np.float32) for seg in segs]

        tid_ign = [192, 224, 224]
        masks_ign = [(np.all(seg == tid_ign, axis=-1)).astype(np.float32) for seg in segs]
        # print('masks[0]', masks[0].shape)

        rgbs = np.stack(rgbs)
        masks = np.stack(masks)
        masks_ign = np.stack(masks_ign)
        S,H,W,C = rgbs.shape

        masks = masks.astype(np.float32)
        masks_ign = masks_ign.astype(np.float32)
        masks[masks_ign > 0] = 0.5

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
