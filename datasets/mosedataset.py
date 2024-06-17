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


class MOSEDataset(MaskDataset):
    def __init__(
            self,
            dataset_location="../MOSE",
            S=32, fullseq=False, chunk=None,
            rand_frames=False,
            crop_size=(384, 512),
            strides=[1,2],
            zooms=[1,2],
            use_augs=False,
            is_training=True,
    ):
        print("loading MOSE dataset...")
        super().__init__(
            dataset_location=dataset_location,
            S=S,
            rand_frames=rand_frames,
            crop_size=crop_size, 
            use_augs=use_augs,
            is_training=is_training,
        )

        clip_step = S//2
        if not is_training:
            strides = [1]
            zooms = [1]
            clip_step = S
        
        # MOSE requires manually validation through their server, not providing labels for validation set, https://codalab.lisn.upsaclay.fr/competitions/10703
        self.dataset_location = os.path.join(dataset_location, "train")
        self.video_names = sorted(os.listdir(os.path.join(self.dataset_location, "JPEGImages")))

        assert(is_training)
        self.video_names = make_split(self.video_names, is_training, shuffle=True)

        print('found %d unique videos in %s' % (len(self.video_names), dataset_location))

        if chunk is not None:
            def chunkify(lst,n):
                return [lst[i::n] for i in range(n)]
            self.video_names = chunkify(self.video_names,100)[chunk]
            print('filtered to %d video_names' % len(self.video_names))
            print('self.video_names', self.video_names)

        self.all_info = []

        for video in self.video_names:
            video_dir = os.path.join(self.dataset_location, "JPEGImages", video)
            annotation_dir = os.path.join(self.dataset_location, "Annotations", video)
            frames = sorted(os.listdir(video_dir))

            S_local = len(frames)
            print('S_local', S_local)

            all_valid_ids = []
            for fn in frames:
                seg = cv2.imread(
                    os.path.join(annotation_dir, fn.replace(".jpg", ".png")),
                    cv2.IMREAD_GRAYSCALE,
                )
                valid_ids = np.array([v for v in np.unique(seg.reshape(-1)) if v > 0])

                # for oid in obj_ids:
                #     masks = [(seg == oid).astype(np.float32) for seg in segs]
                #     masks = np.stack(masks, axis=0)
                #     if np.sum(masks) < 8*S:
                #         continue
                #     bboxes = np.stack([mask2bbox(mask) for mask in masks])
                #     whs = bboxes[:,2:4] - bboxes[:,0:2]
                #     if np.max(whs[:,0]) > W/2 or np.max(whs[:,1]) > H/2:
                #         continue
                #     _, _, _, fills = utils.misc.data_get_traj_from_masks(masks)
                #     if np.sum(fills) < S//4:
                #         continue
                #     all_valid_ids.append(oid)
                # # print('found %d complete masks' % len(all_masks))
                
                all_valid_ids.append(valid_ids)

            for stride in strides:
                for ii in range(0, S_local, clip_step * stride):
                    full_idx = ii + np.arange(self.S) * stride
                    full_idx = [ij for ij in full_idx if ij < S_local]
                    # print('full_idx', full_idx)

                    if len(full_idx) < (((S_local - 1) // stride + 1 if self.S == -1 else self.S) if fullseq else 4):
                        continue
                    
                    valid_ids_here = [all_valid_ids[idx] for idx in full_idx]
                    valid_ids_here = np.unique(np.concatenate(valid_ids_here))
                    # print('valid_ids_here', valid_ids_here)
                
                    for tid in valid_ids_here:
                        for zoom in zooms:
                            self.all_info.append([video, tid, full_idx, zoom])

            sys.stdout.write(".")
            sys.stdout.flush()

        print(
            "found %d samples in %s"
            % (len(self.all_info), self.dataset_location)
        )

    def getitem_helper(self, index):
        video, tid, full_idx, zoom = self.all_info[index]

        video_dir = os.path.join(self.dataset_location, "JPEGImages", video)
        annotation_dir = os.path.join(self.dataset_location, "Annotations", video)
        frames = sorted(os.listdir(video_dir))
        S_local = len(frames)

        frames = [frames[idx] for idx in full_idx]
        S = len(frames)

        image_paths = [os.path.join(video_dir, fn) for fn in frames]
        rgbs = []
        segs = []
        for fn in frames:
            rgb = cv2.imread(os.path.join(video_dir, fn))[..., ::-1].copy()
            seg = cv2.imread(
                os.path.join(annotation_dir, fn.replace(".jpg", ".png")),
                cv2.IMREAD_GRAYSCALE,
            )
            rgbs.append(rgb)
            segs.append(seg)
        # print('rgbs[0]', rgbs[0].shape)

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

        S,H,W,C = rgbs.shape
        bboxes = np.stack([mask2bbox(mask) for mask in masks])
        whs = bboxes[:,2:4] - bboxes[:,0:2]
        # print('whs', whs, 'max whs', np.max(whs[:,0]), np.max(whs[:,1]))
        if np.max(whs[:,0]) > W-8 or np.max(whs[:,1]) > H-8:
            # print('whs', whs, 'max whs', np.max(whs[:,0]), np.max(whs[:,1]))
            print('max whs', np.max(whs[:,0]), np.max(whs[:,1]))
            # print('whs', whs)
            return None
        _, _, _, fills = utils.misc.data_get_traj_from_masks(masks)
        if np.sum(fills) < S//4:
            print('fills', fills)
            return None
        

        sample = {
            "rgbs": rgbs,
            "masks": masks,
        }
        return sample

    def __len__(self):
        return len(self.all_info)
